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

Resumable: a scene whose eval CSV already exists (non-empty) is skipped.
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


def save_depth_camera(outputs, outdir, pose_encoding_to_camera, estimate_focal_knowing_depth):
    """Replicates the depth + camera saving of demo_ray.py:prepare_output (revisit=1),
    writing ONLY depth/ and camera/ (skips conf/ and color/)."""
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
    return B


def _release_claim(claim_path, eval_csv):
    """Remove a cooperative claim so the scene can be retried by another worker,
    unless it actually completed (a non-empty eval CSV exists)."""
    if not claim_path:
        return
    if os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0:
        return
    try:
        os.rmdir(claim_path)
    except OSError:
        pass


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

    done = 0
    skipped = 0
    failed = 0
    t_start = time.time()
    for idx, scene in enumerate(my_scenes):
        claim_path = None
        eval_dir = os.path.join(args.eval_base, scene)
        eval_csv = os.path.join(eval_dir, "eval_depth_pose_metrics.csv")
        if os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0:
            skipped += 1
            continue

        # Cooperative claim: atomically reserve this scene so concurrent workers
        # don't process it twice. mkdir is atomic on POSIX; FileExistsError means
        # another worker already owns it (in-progress or done).
        if args.claim_dir:
            claim_path = os.path.join(args.claim_dir, scene)
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
            _release_claim(claim_path, eval_csv)
            continue

        try:
            ts = time.time()
            # Run inference with one OOM retry (after a cache clear) for very long
            # sequences; silence per-frame prints.
            nfr = None
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

            os.makedirs(eval_dir, exist_ok=True)
            cmd = [
                sys.executable, args.eval_script,
                "--pred_root", pred_dir,
                "--gt_root", gt_dense,
                "--output_csv", eval_csv,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not (os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0):
                raise RuntimeError(f"eval failed rc={r.returncode}: {r.stderr[-500:]}")

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
            _release_claim(claim_path, eval_csv)

    print(f"{tag} DONE done={done} skipped={skipped} failed={failed} "
          f"elapsed={(time.time()-t_start)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
