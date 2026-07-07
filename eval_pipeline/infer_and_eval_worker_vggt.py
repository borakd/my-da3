#!/usr/bin/env python3
"""Batched InfiniteVGGT (StreamVGGT) inference + depth/pose evaluation over a
shard of test scenes -- the SAME evaluation approach as the CUT3R workers, with
the inference engine swapped to InfiniteVGGT.

Per scene:
  1. Load all frames of <scene>/dense/rgb with InfiniteVGGT's official
     load_and_preprocess_images (mode="crop", width=518 -> 320x180 becomes
     518x294, the model's native inference resolution).
  2. Stream them through model.inference(); a custom frame_writer saves, per
     frame, depth (H,W) + camera (.npz: pose=c2w 4x4, intrinsics 3x3) -- the
     exact layout eval_depth_poses.py consumes. cache_results=False keeps GPU
     memory bounded (the streaming/rolling-memory design), so long sequences
     don't OOM by accumulating every frame's outputs.
  3. Run eval_depth_poses.py (all default args) vs <scene>/dense GT. The eval
     resizes the 518x294 prediction to the GT 320x180 (cv2 INTER_CUBIC).

Resumable + cooperative claim-queue identical to infer_and_eval_worker.py.
"""
import os
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


def _release_claim(claim_path, eval_csv):
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
    ap.add_argument("--ckpt", required=True, help="StreamVGGT .safetensors checkpoint")
    ap.add_argument("--label", required=True)
    ap.add_argument("--vggt_root", required=True, help="InfiniteVGGT repo root (contains src/)")
    ap.add_argument("--scenes_root", required=True, help=".../test/dl3dv_multi/wrist")
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--pred_base", required=True)
    ap.add_argument("--eval_base", required=True)
    ap.add_argument("--eval_script", required=True)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--claim_dir", default="")
    ap.add_argument("--ignore_skip_sentinel", action="store_true")
    args = ap.parse_args()

    out_root = os.path.dirname(os.path.dirname(os.path.abspath(args.eval_base)))
    sentinel = os.path.join(out_root, "logs", f"SKIP_{args.label}")
    if (not args.ignore_skip_sentinel) and os.path.exists(sentinel):
        print(f"[{args.label} shard {args.shard_id}] SKIP sentinel present; exiting.", flush=True)
        return

    # --- InfiniteVGGT imports ---
    src_root = os.path.join(args.vggt_root, "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from streamvggt.models.streamvggt import StreamVGGT
    from streamvggt.utils.load_fn import load_and_preprocess_images
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    from safetensors.torch import load_file as load_safetensors

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    with open(args.scene_list) as f:
        all_scenes = [ln.strip() for ln in f if ln.strip()]
    if args.claim_dir:
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
    print(f"{tag} {len(my_scenes)} scenes (of {len(all_scenes)}). device={device}", flush=True)

    print(f"{tag} loading StreamVGGT from {args.ckpt} ...", flush=True)
    t0 = time.time()
    model = StreamVGGT(total_budget=1200000)
    ckpt = load_safetensors(args.ckpt, device="cpu")
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device).eval()
    del ckpt
    amp_dtype = torch.bfloat16 if (device.startswith("cuda")
                                   and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(f"{tag} model loaded in {time.time()-t0:.1f}s (amp={amp_dtype})", flush=True)

    def save_scene(img_paths, pred_dir):
        """Stream a scene through InfiniteVGGT, saving per-frame depth + camera."""
        images = load_and_preprocess_images(img_paths).to(device)  # (N,3,H,W)
        frames = [{"img": images[i].unsqueeze(0)} for i in range(images.shape[0])]
        depth_dir = os.path.join(pred_dir, "depth")
        cam_dir = os.path.join(pred_dir, "camera")
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(cam_dir, exist_ok=True)
        counter = {"n": 0}

        def writer(frame_idx, frame, res_cpu):
            H, W = frame["img"].shape[-2:]
            pose_enc = res_cpu["camera_pose"]              # (1,9)
            extri, intri = pose_encoding_to_extri_intri(pose_enc.unsqueeze(1), (H, W))
            extri = extri[:, 0].float()                    # (1,3,4) world->cam
            extri_h = torch.eye(4).unsqueeze(0).repeat(extri.shape[0], 1, 1)
            extri_h[:, :3, :4] = extri
            c2w = torch.linalg.inv(extri_h)[0].numpy()     # (4,4) cam->world
            if intri is not None:
                K = intri[:, 0][0].float().numpy()         # (3,3)
            else:
                K = np.eye(3, dtype=np.float32)
            depth = res_cpu["depth"].squeeze(0).squeeze(-1).float().numpy()  # (H,W)
            np.save(os.path.join(depth_dir, f"{frame_idx:06d}.npy"), depth)
            np.savez(os.path.join(cam_dir, f"{frame_idx:06d}.npz"),
                     pose=c2w.astype(np.float32), intrinsics=K.astype(np.float32))
            counter["n"] += 1

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=amp_dtype) if device.startswith("cuda") \
                    else contextlib.nullcontext():
                model.inference(frames, frame_writer=writer, cache_results=False)
        del images, frames
        return counter["n"]

    fail_log = os.path.join(args.eval_base, f"_failures_shard{args.shard_id}.txt")
    os.makedirs(args.eval_base, exist_ok=True)

    done = skipped = failed = 0
    t_start = time.time()
    for idx, scene in enumerate(my_scenes):
        claim_path = None
        eval_dir = os.path.join(args.eval_base, scene)
        eval_csv = os.path.join(eval_dir, "eval_depth_pose_metrics.csv")
        if os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0:
            skipped += 1
            continue
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
            nfr = None
            last_exc = None
            for attempt in range(2):
                try:
                    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
                        nfr = save_scene(img_paths, pred_dir)
                    break
                except torch.cuda.OutOfMemoryError as oom:
                    last_exc = oom
                    print(f"{tag} OOM on {scene} (frames={len(img_paths)}) attempt {attempt+1}/2",
                          flush=True)
                finally:
                    gc.collect()
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
            if nfr is None:
                raise last_exc if last_exc is not None else RuntimeError("no output")

            os.makedirs(eval_dir, exist_ok=True)
            cmd = [sys.executable, args.eval_script,
                   "--pred_root", pred_dir, "--gt_root", gt_dense,
                   "--output_csv", eval_csv]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not (os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0):
                raise RuntimeError(f"eval failed rc={r.returncode}: {r.stderr[-600:]}")

            done += 1
            dt = time.time() - ts
            if done <= 3 or done % 20 == 0:
                rate = (time.time() - t_start) / max(done, 1)
                remaining = (len(my_scenes) - skipped - done - failed) * rate
                print(f"{tag} [{idx+1}/{len(my_scenes)}] {scene} frames={nfr} {dt:.1f}s | "
                      f"done={done} skip={skipped} fail={failed} ETA~{remaining/3600:.1f}h", flush=True)
        except Exception as e:
            failed += 1
            with open(fail_log, "a") as fh:
                fh.write(f"{scene}\t{repr(e)}\n")
            print(f"{tag} FAIL {scene}: {repr(e)[:200]}", flush=True)
            traceback.print_exc()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            _release_claim(claim_path, eval_csv)

    print(f"{tag} DONE done={done} skipped={skipped} failed={failed} "
          f"elapsed={(time.time()-t_start)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
