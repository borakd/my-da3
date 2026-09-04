#!/usr/bin/env python3
"""Batched RAY-CONDITIONED CUT3R inference + depth/pose evaluation over a shard
of test scenes.

Same harness as infer_and_eval_worker.py (one checkpoint loaded once per worker,
resumable, cooperative claim queue, OOM retry), but inference goes through
demo_ray.py's GT-pose ray-map conditioning instead of demo.py's image-only
prepare_input:
  1. Frames + GT poses/intrinsics are read from <scene>/dense/{rgb,cam} (paired
     by basename), pushed through the training-style crop/resize, and views 1..N-1
     get their own GT ray map (relative to frame 0) with ray_mask=True — exactly
     the feed_gt_ray_map training setup (--conditioning gt; prev_gt/prev_pred/none
     reproduce the other demo_ray variants).
  2. Save per-frame depth (.npy) + camera (.npz: pose=c2w, intrinsics) — the same
     depth/camera demo_ray.py's prepare_output writes (conf/color are skipped).
  3. Run eval_depth_poses.py (all default args) comparing pred vs <scene>/dense GT.

Under --conditioning prev_pred_gru a SECOND pose trajectory is scored with the
byte-for-byte same eval script and arguments: the GRU-refined pose res["gru_pose"]
(previous predicted pose + the GRU's residual, quaternion renormalized) — exactly
the tensor the decoder rendered into each view's conditioning ray map. It is
written to <pred_base>_gru/<scene>/camera (depth symlinked from the regular
prediction) and evaluated into <eval_base>_gru/<scene>/, so the regular eval
tree and everything watching it are untouched.

Resumable: a scene is skipped once its eval CSV exists (non-empty) — under
prev_pred_gru, once BOTH the regular and the GRU eval CSVs exist.
"""
import os
# Reduce CUDA fragmentation OOMs on very long sequences (read before torch init).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import contextlib
import gc
import glob
import subprocess
import sys
import time
import traceback

import numpy as np
import torch


def save_depth_camera(outputs, outdir, pose_encoding_to_camera, estimate_focal_knowing_depth,
                      gru_outdir=None):
    """Replicates the depth + camera saving of demo_ray.py:prepare_output (revisit=1),
    writing ONLY depth/ and camera/ (skips conf/ and color/).

    gru_outdir, when set, additionally writes the GRU-refined pose trajectory as a
    parallel prediction root (camera/ npz files + a depth/ symlink) so the SAME
    eval script can score it with unchanged code and arguments."""
    preds = outputs["pred"]

    pts3ds_self = torch.cat([p["pts3d_in_self_view"].cpu() for p in preds], 0)  # B,H,W,3

    pr_poses = [pose_encoding_to_camera(p["camera_pose"].clone()).cpu() for p in preds]
    cam2world = torch.cat(pr_poses)  # B,4,4

    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    depths = pts3ds_self[..., 2]  # B,H,W
    intrinsics = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    intrinsics[:, 0, 0] = focal.detach().cpu()
    intrinsics[:, 1, 1] = focal.detach().cpu()
    intrinsics[:, 0, 2] = pp[:, 0]
    intrinsics[:, 1, 2] = pp[:, 1]

    depth_dir = os.path.join(outdir, "depth")
    cam_dir = os.path.join(outdir, "camera")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(cam_dir, exist_ok=True)
    for i in range(B):
        np.save(os.path.join(depth_dir, f"{i:06d}.npy"), depths[i].cpu().numpy())
        np.savez(
            os.path.join(cam_dir, f"{i:06d}.npz"),
            pose=cam2world[i].cpu().numpy(),
            intrinsics=intrinsics[i].cpu().numpy(),
        )

    if gru_outdir is not None:
        # GRU-refined trajectory (A4 closed loop): for every view x >= 1 the pose
        # saved here is EXACTLY the tensor the decoder rendered into view x's
        # conditioning ray map — the previous predicted pose P(x-1) plus the
        # GRU's residual, quaternion renormalized (res["gru_pose"] stashed in
        # _forward_decoder_group_step right where prev_pose_enc is replaced).
        # View 0 has no GRU output (nothing precedes it); the head's own view-0
        # pose anchors the trajectory — the same anchor the closed loop itself
        # starts from, and identical to frame 0 of the regular eval.
        missing = [i for i, p in enumerate(preds[1:], start=1) if "gru_pose" not in p]
        assert not missing, (
            f"GRU eval requested but views {missing[:5]} carry no 'gru_pose' — "
            "is the PoseGRU active (conditioning=prev_pred_gru)?"
        )
        gru_cam2world = torch.cat(
            [cam2world[:1]]
            + [pose_encoding_to_camera(p["gru_pose"].clone().float()).cpu()
               for p in preds[1:]]
        )
        gru_cam_dir = os.path.join(gru_outdir, "camera")
        os.makedirs(gru_cam_dir, exist_ok=True)
        for i in range(B):
            np.savez(
                os.path.join(gru_cam_dir, f"{i:06d}.npz"),
                pose=gru_cam2world[i].cpu().numpy(),
                intrinsics=intrinsics[i].cpu().numpy(),
            )
        # The GRU changes no depth; a relative symlink to the regular depth dir
        # keeps the eval script's inputs shaped identically without duplicating
        # the per-scene depth predictions.
        gru_depth = os.path.join(gru_outdir, "depth")
        if not os.path.lexists(gru_depth):
            os.symlink(os.path.relpath(depth_dir, gru_outdir), gru_depth)
    return B


def _csv_done(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _release_claim(claim_path, eval_csvs):
    """Remove a cooperative claim so the scene can be retried by another worker,
    unless it actually completed (ALL required eval CSVs exist non-empty)."""
    if not claim_path:
        return
    if all(_csv_done(p) for p in eval_csvs):
        return
    try:
        os.rmdir(claim_path)
    except OSError:
        pass


def _run_eval(eval_script, pred_root, gt_root, out_csv):
    """Run the vendored eval script (default args — identical math for every
    trajectory it is pointed at) and raise unless it produced a non-empty CSV.

    The CSV is written to a per-pid temp name and atomically renamed into
    place: the skip/claim logic treats a non-empty CSV as 'done', so a
    half-written file (or two workers racing in the narrow backfill window)
    must never be observable at the final path."""
    tmp_csv = f"{out_csv}.tmp.{os.getpid()}"
    cmd = [
        sys.executable, eval_script,
        "--pred_root", pred_root,
        "--gt_root", gt_root,
        "--output_csv", tmp_csv,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not _csv_done(tmp_csv):
        try:
            os.remove(tmp_csv)
        except OSError:
            pass
        raise RuntimeError(f"eval failed rc={r.returncode}: {r.stderr[-500:]}")
    os.replace(tmp_csv, out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument(
        "--conditioning",
        default="gt",
        choices=["gt", "prev_gt", "prev_pred", "prev_pred_gru", "none"],
        help="Ray conditioning variant, forwarded to demo_ray's input builders. "
        "prev_pred_gru = closed loop with the PoseGRU refiner active (module and "
        "mode/hidden_dim are restored from the checkpoint automatically).",
    )
    ap.add_argument(
        "--oracle",
        default="off",
        choices=["off", "gt"],
        help="pose_gru_oracle run mode (DIAGNOSTIC, not an arm; mirrors "
        "demo_ray.py --oracle): 'gt' feeds the GRU the CURRENT view's GT pose "
        "(real GT camera_pose is loaded into the views) — requires "
        "--conditioning prev_pred_gru. Default 'off' = honest closed loop, "
        "even for oracle-TRAINED checkpoints (loudly flagged either way).",
    )
    ap.add_argument("--scenes_root", required=True, help=".../test/dl3dv_multi/wrist")
    ap.add_argument("--scene_list", required=True, help="txt of scene names (one per line)")
    ap.add_argument("--pred_base", required=True)
    ap.add_argument("--eval_base", required=True)
    ap.add_argument("--eval_script", required=True)
    ap.add_argument("--cut3r_dir", required=True)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="process at most N scenes (smoke test)")
    ap.add_argument(
        "--claim_dir",
        default="",
        help="If set, enable cooperative-queue mode: every worker walks ALL "
        "scenes and atomically claims each via mkdir under this dir, so any "
        "number of workers (across nodes / GPU types) cooperatively drain one "
        "shared queue with no double-processing and no stall if some workers "
        "never start. shard_id/num_shards then only set a starting offset.",
    )
    ap.add_argument(
        "--ignore_skip_sentinel",
        action="store_true",
        help="Run even if <OUT>/logs/SKIP_<label> exists. Default: respect the "
        "sentinel and exit immediately (used to disable a job's pass for a label "
        "that another job is handling).",
    )
    args = ap.parse_args()

    # Skip sentinel: lets us disable a specific (job, label) pass without editing
    # a running job's launch loop. OUT = parent-of-parent of eval_base.
    out_root = os.path.dirname(os.path.dirname(os.path.abspath(args.eval_base)))
    sentinel = os.path.join(out_root, "logs", f"SKIP_{args.label}")
    if (not args.ignore_skip_sentinel) and os.path.exists(sentinel):
        print(f"[{args.label} shard {args.shard_id}] SKIP sentinel present "
              f"({sentinel}); exiting without work.", flush=True)
        return

    sys.path.insert(0, args.cut3r_dir)
    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(args.ckpt)

    import demo_ray  # ray-conditioned input builders (no GPU/viser work on import)
    from src.dust3r.inference import inference
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    with open(args.scene_list) as f:
        all_scenes = [ln.strip() for ln in f if ln.strip()]
    if args.claim_dir:
        # Cooperative-queue mode: every worker considers ALL scenes and claims
        # each atomically (mkdir) below. shard_id/num_shards only pick a distinct
        # starting offset so workers begin in different parts of the list and
        # don't all contend for the same first scene.
        os.makedirs(args.claim_dir, exist_ok=True)
        n = len(all_scenes)
        off = (args.shard_id * (n // max(args.num_shards, 1))) % n if n else 0
        my_scenes = all_scenes[off:] + all_scenes[:off]
    else:
        my_scenes = [s for i, s in enumerate(all_scenes)
                     if i % args.num_shards == args.shard_id]
    if args.limit > 0:
        my_scenes = my_scenes[: args.limit]

    tag = f"[{args.label} shard {args.shard_id}/{args.num_shards}]"
    print(f"{tag} {len(my_scenes)} scenes (of {len(all_scenes)} total). "
          f"device={device} conditioning={args.conditioning}", flush=True)

    print(f"{tag} loading model from {args.ckpt} ...", flush=True)
    t0 = time.time()
    model = ARCroco3DStereo.from_pretrained(args.ckpt).to(device)
    if args.conditioning in ("prev_pred", "prev_pred_gru"):
        # Closed-loop conditioning happens inside the decoder step; same flag
        # demo_ray.py / the captain_ray_prev_pred training config set.
        model.feed_prev_pred = True
    if args.conditioning == "prev_pred_gru":
        # The GRU module + mode/hidden_dim are auto-restored from the ckpt by
        # load_model. A ckpt without pose_gru weights gets an identity-init
        # residual GRU so smoke tests still run — but that is NOT a trained
        # refiner, so say so loudly.
        if getattr(model, "pose_gru", None) is None:
            print(f"{tag} NOTE: ckpt has no pose_gru weights — enabling an "
                  "identity-init residual GRU (== plain prev_pred). Smoke-test "
                  "only; results are not a trained-GRU arm.", flush=True)
            model.enable_pose_gru()
            model.pose_gru.to(device)
        gru = model.pose_gru
        print(
            f"{tag} PoseGRU active (from ckpt): mode={gru.mode}, "
            f"input={gru.input_mode}, hidden_dim={gru.hidden_dim}, "
            f"img_feat={gru.img_feat}"
            + (
                # img_feat_dim is the APPENDED cell width: the projector dim
                # when proj=True, the full src*blocks width when proj=False.
                # src is the SOURCE sub-lever (pooled/resnet18/dinov2_vits14/
                # corr) — getattr'd so pickled pre-lever modules still print.
                f" (src={getattr(gru, 'img_feat_src', 'pooled')}, "
                f"appended={gru.img_feat_dim}, frames={gru.img_feat_frames}, "
                f"proj={gru.img_feat_proj})"
                if gru.img_feat == "input"
                else ""
            )
            + f", iters={gru.iters}.", flush=True)
    elif getattr(model, "pose_gru", None) is not None:
        # v2 runs the GRU whenever the module exists and feed_prev_pred is on.
        # For every non-GRU arm, strip it so a GRU checkpoint evaluated under
        # plain prev_pred really is the raw closed loop (GRU-less ablation).
        print(f"{tag} ckpt carries pose_gru but conditioning={args.conditioning} "
              "— disabling the GRU for this arm.", flush=True)
        model.pose_gru = None

    # Run-mode cross-checks against how the ckpt was TRAINED (stashed on the
    # net by load_model from ckpt['args']) — kept identical to demo_ray.py so
    # the demo and the batch worker never diverge. Mismatches are legal
    # ablation arms, but never silent ones.
    trained_cond = getattr(model, "trained_conditioning", "none")
    if trained_cond != args.conditioning:
        print(f"{tag} WARNING: ckpt was trained with conditioning "
              f"'{trained_cond}' but this arm runs '{args.conditioning}' — an "
              "out-of-distribution probe, not the ckpt's honest arm.", flush=True)
    trained_oracle = getattr(model, "trained_pose_gru_oracle", "off")
    if args.oracle == "gt":
        assert args.conditioning == "prev_pred_gru", (
            "--oracle gt refines the GRU input — it requires "
            "--conditioning prev_pred_gru"
        )
        model.pose_gru_oracle = "gt"
        print(f"{tag} *** pose_gru_oracle=gt — DIAGNOSTIC RUN, NOT an arm. GT "
              "pose is injected at the GRU input (views carry real GT "
              "camera_pose); outputs are GT-derived — never compare against "
              "honest arms. ***", flush=True)
    elif trained_oracle != "off":
        print(f"{tag} WARNING: ckpt was TRAINED with pose_gru_oracle="
              f"'{trained_oracle}' (GT-injection diagnostic arm) but this run "
              "keeps the oracle OFF: honest-loop metrics of an oracle-trained "
              "arm are an out-of-distribution probe (pass --oracle gt to "
              "reproduce the training-time wiring).", flush=True)
    model.eval()
    print(f"{tag} model loaded in {time.time()-t0:.1f}s", flush=True)

    fail_log = os.path.join(args.eval_base, f"_failures_shard{args.shard_id}.txt")
    os.makedirs(args.eval_base, exist_ok=True)

    # Second scored trajectory under the GRU arm: the pose fed into the ray map
    # (prev pred + GRU residual), evaluated with the byte-for-byte same script
    # and args into a parallel tree that leaves <label>/eval untouched.
    gru_mode = args.conditioning == "prev_pred_gru"
    gru_pred_base = args.pred_base.rstrip("/") + "_gru"
    gru_eval_base = args.eval_base.rstrip("/") + "_gru"
    if gru_mode:
        print(f"{tag} GRU-pose eval active: preds -> {gru_pred_base}, "
              f"evals -> {gru_eval_base}", flush=True)

    done = 0
    skipped = 0
    failed = 0
    t_start = time.time()
    for idx, scene in enumerate(my_scenes):
        claim_path = None
        eval_dir = os.path.join(args.eval_base, scene)
        eval_csv = os.path.join(eval_dir, "eval_depth_pose_metrics.csv")
        gru_pred_dir = os.path.join(gru_pred_base, scene) if gru_mode else None
        gru_eval_csv = (
            os.path.join(gru_eval_base, scene, "eval_depth_pose_metrics.csv")
            if gru_mode else None
        )
        req_csvs = [eval_csv] + ([gru_eval_csv] if gru_mode else [])
        reg_done = _csv_done(eval_csv)
        gru_done = (not gru_mode) or _csv_done(gru_eval_csv)
        if reg_done and gru_done:
            skipped += 1
            continue

        # Cooperative claim: atomically reserve this scene so concurrent workers
        # don't process it twice. mkdir is atomic on POSIX; FileExistsError means
        # another worker already owns it (in-progress or done).
        if args.claim_dir:
            # A completed scene keeps its claim forever, so a GRU backfill pass
            # over a label evaluated BEFORE the GRU eval existed would find
            # every scene claimed and silently do nothing. Backfill work
            # (regular CSV done, only the gru CSV missing) therefore claims
            # under a separate name — still one atomic mkdir per scene per
            # kind of work, and the full pass's claim never blocks it.
            claim_name = scene + ".gru" if reg_done else scene
            claim_path = os.path.join(args.claim_dir, claim_name)
            try:
                os.mkdir(claim_path)
            except FileExistsError:
                skipped += 1
                continue

        rgb_dir = os.path.join(args.scenes_root, scene, "dense", "rgb")
        cam_dir = os.path.join(args.scenes_root, scene, "dense", "cam")
        gt_dense = os.path.join(args.scenes_root, scene, "dense")
        pred_dir = os.path.join(args.pred_base, scene)
        img_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")) +
                           glob.glob(os.path.join(rgb_dir, "*.jpg")))
        if len(img_paths) < 2:
            skipped += 1
            _release_claim(claim_path, req_csvs)
            continue

        try:
            ts = time.time()
            # Backfill fast path: the regular eval already completed and the GRU
            # camera trajectory was fully written by the pass that produced it —
            # only the GRU eval subprocess is missing. Completeness is judged
            # against the regular camera dir written by the same save call;
            # anything short (or already cleaned up) re-runs inference.
            need_infer = not reg_done
            nfr = None
            if gru_mode and not gru_done and not need_infer:
                n_reg_cam = len(glob.glob(os.path.join(pred_dir, "camera", "*.npz")))
                n_gru_cam = len(glob.glob(os.path.join(gru_pred_dir, "camera", "*.npz")))
                if n_reg_cam >= 2 and n_gru_cam == n_reg_cam:
                    nfr = n_reg_cam
                else:
                    need_infer = True
            if need_infer:
                # Run inference with one OOM retry (after a cache clear) for very
                # long sequences; silence per-frame prints.
                last_exc = None
                for attempt in range(2):
                    outputs = state_args = views = None
                    images = ray_maps = intrinsics_list = gt_poses = None
                    try:
                        with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
                            if args.conditioning == "none":
                                views = demo_ray.prepare_input_none(img_paths, args.size)
                            else:
                                images, ray_maps, intrinsics_list, gt_poses = (
                                    demo_ray.load_frames_training_style(
                                        img_paths, cam_dir, args.size))
                                views = demo_ray.prepare_input(
                                    images=images,
                                    ray_maps=ray_maps,
                                    intrinsics_list=intrinsics_list,
                                    # Real GT camera_pose in the views: required by
                                    # --oracle gt, inert data for honest arms.
                                    gt_poses=gt_poses,
                                    # prev_pred_gru builds views exactly like
                                    # prev_pred (no data-side rays); the GRU acts
                                    # inside the decoder step, not in the inputs.
                                    conditioning=(
                                        "prev_pred"
                                        if args.conditioning == "prev_pred_gru"
                                        else args.conditioning
                                    ),
                                )
                            with torch.no_grad():
                                outputs, state_args = inference(views, model, device)
                            nfr = save_depth_camera(
                                outputs, pred_dir, pose_encoding_to_camera,
                                estimate_focal_knowing_depth,
                                gru_outdir=gru_pred_dir if gru_mode else None,
                            )
                        break
                    except torch.cuda.OutOfMemoryError as oom:
                        last_exc = oom
                        print(f"{tag} OOM on {scene} (frames={len(img_paths)}) "
                              f"attempt {attempt+1}/2", flush=True)
                    finally:
                        del outputs, state_args, views, images, ray_maps, intrinsics_list, gt_poses
                        gc.collect()
                        if device.startswith("cuda"):
                            torch.cuda.empty_cache()
                if nfr is None:
                    raise last_exc if last_exc is not None else RuntimeError("no output")
                # Inference just OVERWROTE both prediction trees with a fresh
                # rollout — any pre-existing CSV describes a different rollout
                # than its sibling. Remove stale CSVs NOW (not merely re-score
                # below): if a re-score fails or the worker dies before it, a
                # retry must see the scene as unscored, or its fast path would
                # pair the old CSV with the new trajectory forever.
                for stale in ([eval_csv] if reg_done else []) + (
                    [gru_eval_csv] if (gru_mode and gru_done) else []
                ):
                    try:
                        os.remove(stale)
                    except FileNotFoundError:
                        pass
                reg_done = False
                gru_done = not gru_mode

            os.makedirs(eval_dir, exist_ok=True)
            if not reg_done:
                _run_eval(args.eval_script, pred_dir, gt_dense, eval_csv)
            if gru_mode and not gru_done:
                # EXACT same eval invocation as the regular pass — only the
                # prediction root (GRU camera trajectory) and the output CSV
                # differ, so the pose math is identical by construction.
                os.makedirs(os.path.dirname(gru_eval_csv), exist_ok=True)
                _run_eval(args.eval_script, gru_pred_dir, gt_dense, gru_eval_csv)

            done += 1
            dt = time.time() - ts
            if done <= 3 or done % 25 == 0:
                rate = (time.time() - t_start) / max(done, 1)
                remaining = (len(my_scenes) - skipped - done - failed) * rate
                print(f"{tag} [{idx+1}/{len(my_scenes)}] {scene} frames={nfr} "
                      f"{dt:.1f}s | done={done} skip={skipped} fail={failed} "
                      f"ETA~{remaining/3600:.1f}h", flush=True)
        except Exception as e:
            failed += 1
            with open(fail_log, "a") as fh:
                fh.write(f"{scene}\t{repr(e)}\n")
            print(f"{tag} FAIL {scene}: {repr(e)[:200]}", flush=True)
            traceback.print_exc()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            # Release the claim so another worker (or a later sweep) can retry.
            _release_claim(claim_path, req_csvs)

    print(f"{tag} DONE done={done} skipped={skipped} failed={failed} "
          f"elapsed={(time.time()-t_start)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
