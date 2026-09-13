#!/usr/bin/env python3
"""Batched CUT3R inference + depth/pose evaluation over a shard of test scenes.

Loads ONE checkpoint once and processes a (round-robin) shard of scenes:
  1. CUT3R inference on all frames of <scene>/dense/rgb  (reuses demo.py's
     prepare_input + the model's inference(), so outputs match demo.py).
  2. Save per-frame depth (.npy) + camera (.npz: pose=c2w, intrinsics) -- the exact
     same depth/camera demo.py's prepare_output writes (conf/color are skipped).
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


def save_depth_camera(outputs, outdir, pose_encoding_to_camera, estimate_focal_knowing_depth,
                      frame_ids=None):
    """Replicates the depth + camera saving of demo.py:prepare_output (revisit=1),
    writing ONLY depth/ and camera/ (skips conf/ and color/).

    frame_ids: optional list of ORIGINAL frame indices, one per emitted view, used
    as the file numbers. Needed by --skip_mode drop, where the input sequence is a
    subset of the scene: the evaluator joins pred and GT on file index, so kept
    frames must keep their original numbers. Default: 0..B-1."""
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
    if frame_ids is None:
        frame_ids = list(range(B))
    assert len(frame_ids) == B, f"frame_ids has {len(frame_ids)} entries for {B} views"
    for i in range(B):
        fid = int(frame_ids[i])
        np.save(os.path.join(depth_dir, f"{fid:06d}.npy"), depths[i].cpu().numpy())
        np.savez(
            os.path.join(cam_dir, f"{fid:06d}.npz"),
            pose=cam2world[i].cpu().numpy(),
            intrinsics=intrinsics[i].cpu().numpy(),
        )
    return B


def build_kept_gt(gt_dense, kept_ids, dst):
    """Symlink the GT depth/cam files of kept_ids into dst/{depth,cam}.

    eval_depth_poses.py renumbers files POSITIONALLY per camera stream
    (_split_stream_by_camera), so a subset of predictions scored against the
    full GT is misaligned from the first gap on. Scoring a subset therefore
    needs a GT tree holding exactly the same frame numbers."""
    keep = set(int(i) for i in kept_ids)
    for sub, suf in (("depth", ".npy"), ("cam", ".npz")):
        src_dir = os.path.join(gt_dense, sub)
        if sub == "cam" and not os.path.isdir(src_dir):
            src_dir = os.path.join(gt_dense, "camera")
        dst_dir = os.path.join(dst, sub)
        os.makedirs(dst_dir, exist_ok=True)
        for old in os.listdir(dst_dir):
            os.remove(os.path.join(dst_dir, old))
        for fn in sorted(os.listdir(src_dir)):
            stem, ext = os.path.splitext(fn)
            if ext == suf and stem.isdigit() and int(stem) in keep:
                os.symlink(os.path.abspath(os.path.join(src_dir, fn)),
                           os.path.join(dst_dir, fn))
    return dst


def _plot_metrics(eval_csv, out_dir, scene, tag):
    """Write metrics_over_time*.png next to the eval CSV. Never fatal: a plot
    failure must not fail a scene that has already been scored."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from plot_metrics_over_time import plot_metrics_over_time
        plot_metrics_over_time(eval_csv, out_dir, title=scene)
    except Exception as e:  # matplotlib missing, malformed CSV, ...
        print(f"{tag} WARNING: metrics plot failed for {scene}: {e!r}", flush=True)


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
        "--skip_frames_file",
        default="",
        help="json {scene: [frame indices]} of frames to IGNORE (harmful-frame "
        "pilot, see HARMFUL_FRAME_SKIP_PLAN.md). Scenes absent from the json run "
        "normally. Frame 0 is never skipped.",
    )
    ap.add_argument(
        "--control_json",
        default="",
        help="Richer eval-time write control, json {scene: spec}. spec keys (all "
        "optional): block [frames] (update=False); alpha {frame: a} (soft commit "
        "state<-a*new+(1-a)*old); reset [frames] (state/mem reset to init AFTER "
        "that frame's commit); token_gate {frames: [..] | 'all', q, gmin} "
        "(per-state-token commit keyed on the frame's own confidence). Frame 0 "
        "is never blocked. Combines with --skip_frames_file.",
    )
    ap.add_argument(
        "--skip_mode",
        choices=["block", "drop"],
        default="block",
        help="block: listed frames still get a forward pass and a prediction but "
        "their state/memory write is suppressed (views[i]['update']=False). "
        "drop: listed frames are removed from the input sequence entirely; the "
        "kept frames' predictions are saved under their ORIGINAL indices.",
    )
    ap.add_argument(
        "--no_keep_preds",
        action="store_true",
        help="Delete the scene's predictions (depth/ + camera/) once its eval CSV "
        "is written, keeping only the eval outputs. For large sweeps.",
    )
    ap.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip the per-scene metrics-over-frames plots (metrics_over_time*.png "
        "next to the eval CSV, see eval_pipeline/plot_metrics_over_time.py).",
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

    import demo  # provides prepare_input (no GPU/viser side effects on import)
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
    print(f"{tag} {len(my_scenes)} scenes (of {len(all_scenes)} total). device={device}",
          flush=True)

    print(f"{tag} loading model from {args.ckpt} ...", flush=True)
    t0 = time.time()
    model = ARCroco3DStereo.from_pretrained(args.ckpt).to(device)
    # This worker is IMAGE-ONLY (demo.py-style views, no ray conditioning, no
    # feed_prev_pred). A checkpoint trained with conditioning still loads —
    # but evaluating it here is an out-of-distribution open-loop probe, so say
    # so loudly and strip any PoseGRU (it would be inert anyway: the GRU only
    # runs inside the feed_prev_pred loop). The conditioned arms belong to
    # infer_and_eval_worker_ray.py.
    trained_cond = getattr(model, "trained_conditioning", "none")
    if trained_cond != "none":
        print(f"{tag} WARNING: ckpt was trained with conditioning "
              f"'{trained_cond}' but this worker runs image-only open loop — "
              "an out-of-distribution probe, NOT the ckpt's honest arm. Use "
              "eval_pipeline/infer_and_eval_worker_ray.py --conditioning "
              f"{trained_cond} for the honest evaluation.", flush=True)
    if getattr(model, "pose_gru", None) is not None:
        print(f"{tag} ckpt carries pose_gru — disabling it for this image-only "
              "worker (it only ever runs inside the feed_prev_pred loop).",
              flush=True)
        model.pose_gru = None
    model.eval()
    print(f"{tag} model loaded in {time.time()-t0:.1f}s", flush=True)

    skip_frames = {}
    if args.skip_frames_file:
        import json
        with open(args.skip_frames_file) as f:
            skip_frames = {k: sorted(set(int(i) for i in v) - {0})
                           for k, v in json.load(f).items()}
        print(f"{tag} skip-frames file: {args.skip_frames_file} "
              f"({len(skip_frames)} scenes, mode={args.skip_mode})", flush=True)

    controls = {}
    if args.control_json:
        import json
        with open(args.control_json) as f:
            controls = json.load(f)
        print(f"{tag} control json: {args.control_json} ({len(controls)} scenes)", flush=True)

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
            # sequences; silence load_images/inference per-frame prints.
            nfr = None
            last_exc = None
            frame_ids = None
            for attempt in range(2):
                outputs = state_args = views = None
                try:
                    ctl = controls.get(scene) or controls.get("*", {})
                    bad = set(skip_frames.get(scene, [])) | set(int(b) for b in ctl.get("block", []))
                    bad = {b for b in bad if 0 < b < len(img_paths)}
                    frame_ids = None
                    run_paths = img_paths
                    if bad and args.skip_mode == "drop":
                        frame_ids = [i for i in range(len(img_paths)) if i not in bad]
                        run_paths = [img_paths[i] for i in frame_ids]
                    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
                        views = demo.prepare_input(
                            img_paths=run_paths,
                            img_mask=[True] * len(run_paths),
                            size=args.size,
                            revisit=1,
                            update=True,
                        )
                        if bad and args.skip_mode == "block":
                            # Native memory-write switch, honored by
                            # _forward_decoder_group_step (model.py ~1867):
                            # the frame is still decoded and predicted, but
                            # state_feat/mem keep their pre-frame values.
                            for vi_, v_ in enumerate(views):
                                if vi_ in bad:
                                    v_["update"] = torch.tensor(False).unsqueeze(0)
                        if ctl and args.skip_mode == "block":
                            if ctl.get("alpha_all") is not None:
                                for v_ in views:
                                    v_["update_alpha"] = torch.tensor(float(ctl["alpha_all"])).unsqueeze(0)
                            for fr_, a_ in (ctl.get("alpha") or {}).items():
                                fr_ = int(fr_)
                                if 0 <= fr_ < len(views):
                                    views[fr_]["update_alpha"] = torch.tensor(float(a_)).unsqueeze(0)
                            for fr_ in ctl.get("reset") or []:
                                fr_ = int(fr_)
                                if 0 <= fr_ < len(views):
                                    views[fr_]["reset"] = torch.tensor(True).unsqueeze(0)
                            trig = ctl.get("trigger")  # {"z": 1.0, "warmup": 8}: causal conf trigger
                            if trig:
                                for v_ in views:
                                    v_["conf_trigger_z"] = torch.tensor(float(trig.get("z", 1.0))).unsqueeze(0)
                                    v_["conf_trigger_warmup"] = torch.tensor(float(trig.get("warmup", 8))).unsqueeze(0)
                                    v_["conf_trigger_window"] = torch.tensor(float(trig.get("window", 0))).unsqueeze(0)
                            tg = ctl.get("token_gate")
                            if tg:
                                frs = range(len(views)) if tg.get("frames") == "all" else [int(x) for x in tg.get("frames", [])]
                                for fr_ in frs:
                                    if 0 <= fr_ < len(views):
                                        views[fr_]["token_gate_q"] = torch.tensor(float(tg["q"])).unsqueeze(0)
                                        views[fr_]["token_gate_gmin"] = torch.tensor(float(tg.get("gmin", 0.0))).unsqueeze(0)
                        with torch.no_grad():
                            outputs, state_args = inference(views, model, device)
                        nfr = save_depth_camera(
                            outputs, pred_dir, pose_encoding_to_camera,
                            estimate_focal_knowing_depth, frame_ids=frame_ids,
                        )
                        if ctl and ctl.get("trigger"):
                            import json as _json
                            trig_frames = sorted(int(t) for t in getattr(model, "_maks_triggered", []))
                            os.makedirs(eval_dir, exist_ok=True)
                            with open(os.path.join(eval_dir, "triggered_frames.json"), "w") as _f:
                                _json.dump({scene: trig_frames}, _f)
                            with open(os.path.join(eval_dir, "conf_trace.json"), "w") as _f:
                                _json.dump({"mean_log_conf_self": list(getattr(model, "_maks_conf_hist", [])),
                                            "z": list(getattr(model, "_maks_z_hist", []))}, _f)
                            model._maks_conf_hist = []
                            model._maks_triggered = []
                    if bad or ctl:
                        print(f"{tag} {scene}: {args.skip_mode} {len(bad)}/{len(img_paths)} frames"
                              + (f"; controls={sorted(ctl.keys())}" if ctl else ""), flush=True)
                    break
                except torch.cuda.OutOfMemoryError as oom:
                    last_exc = oom
                    print(f"{tag} OOM on {scene} (frames={len(img_paths)}) "
                          f"attempt {attempt+1}/2", flush=True)
                finally:
                    del outputs, state_args, views
                    gc.collect()
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
            if nfr is None:
                raise last_exc if last_exc is not None else RuntimeError("no output")

            os.makedirs(eval_dir, exist_ok=True)
            gt_for_eval = gt_dense
            if frame_ids is not None:
                # drop mode: score the kept frames against a GT tree with the
                # same frame numbers (positional join, see build_kept_gt).
                gt_for_eval = build_kept_gt(
                    gt_dense, frame_ids, os.path.join(pred_dir, "_gt_kept"))
            cmd = [
                sys.executable, args.eval_script,
                "--pred_root", pred_dir,
                "--gt_root", gt_for_eval,
                "--output_csv", eval_csv,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not (os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0):
                raise RuntimeError(f"eval failed rc={r.returncode}: {r.stderr[-500:]}")
            if not args.no_plots:
                _plot_metrics(eval_csv, eval_dir, scene, tag)
            if args.no_keep_preds:
                import shutil
                shutil.rmtree(pred_dir, ignore_errors=True)

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
