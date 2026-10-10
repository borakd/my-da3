#!/usr/bin/env python
"""CUT3R conditioned, at every timestep, on the SAME-TIMESTEP wrist pose predicted by the exterior-camera rig
(ext_cams/rig_trackon_eval.py on the Track-On-R v2 tracks), fed through the pretrained ray-map encoder — the
`gt` / `gt_all` conditioning of demo_ray.py with the rig's pose in place of the GT pose.

Per scene (only scenes the rig solved: RIG_DIR/<scene>/{record.json, anchors.npz}):
  window  = the rig window, frames t_ref .. t_end (t_ref = the rig's reference frame, where its pose equals the
            store GT pose by construction; t_end = the FK exit used by rig_trackon_eval --> the exact frames the
            rig evaluation covered). CUT3R's view 0 is t_ref, so every ray map is the pose RELATIVE TO t_ref,
            metric, robot-base frame -- the same convention the GT-ray training used (reference = first view).
  pose(t) = the rig's Kabsch-transferred lens pose at t (anchors.npz) when the rig solved t; on unsolved window
            frames the last solved rig pose is held (--gap hold; --gap mask feeds the masked ray token instead).
            Causal: the rig pose at t uses only frames <= t (modulo rig_track's whole-window pairing, as in the rig
            evaluation itself).
  arms    --pose_source rig | gt     x   --conditioning gt | gt_all | none | none_ts
            rig+gt      : finetuned GT-ray trunk (gtray_lr1e5) with rig rays, view 0 image-only   (main arm)
            rig+gt_all  : pretrained cut3r with rig rays on every view incl. view 0           (zero-shot arm)
            gt+gt       : the GT-ray oracle on the same window (ceiling)
            none / none_ts : no rays (plain finetune / pretrained controls), same window
Products: PRED_BASE/<scene>/{depth,camera}/000000.. (window-local numbering) + cond_poses.npz (frames, the poses
fed, source flag 1 = rig-solved / 0 = held, GT poses); EVAL_BASE/<scene>/eval_depth_pose_metrics.csv from the
official evaluator against GT_WIN_ROOT/<scene>/{depth,cam,...} (symlinks to the original frames, renumbered).
Resumable (eval csv present -> skip). Sharded by index modulo.
"""
import argparse
import contextlib
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch


def build_window_gt(scenes_root, scene, frames, gt_win_root):
    """GT_WIN_ROOT/<scene>/{depth,cam,outlier_mask,sky_mask}/<k:06d> -> dense/<dir>/<t:06d> for k, t in enumerate(frames)."""
    d = os.path.join(gt_win_root, scene)
    if os.path.isfile(os.path.join(d, "frames.json")):
        return d
    tmp = f"{d}.tmp.{os.getpid()}"
    dense = os.path.join(scenes_root, scene, "dense")
    for sub in ("depth", "cam", "outlier_mask", "sky_mask"):
        src_dir = os.path.join(dense, sub)
        if not os.path.isdir(src_dir):
            continue
        ext = next((os.path.splitext(f)[1] for f in os.listdir(src_dir) if f.startswith("000000")), None)
        if ext is None:
            continue
        os.makedirs(os.path.join(tmp, sub), exist_ok=True)
        for k, t in enumerate(frames):
            os.symlink(os.path.join(src_dir, f"{t:06d}{ext}"), os.path.join(tmp, sub, f"{k:06d}{ext}"))
    json.dump({"frames": [int(t) for t in frames]}, open(os.path.join(tmp, "frames.json"), "w"))
    try:
        os.rename(tmp, d)
    except OSError:  # another worker built it first
        shutil.rmtree(tmp, ignore_errors=True)
    return d


def rig_window(rig_dir, scene, gap):
    """(frames, cond_poses (n,4,4) c2w, src flags) from the rig products; None when the rig did not solve the scene."""
    rec_f, anc_f = os.path.join(rig_dir, scene, "record.json"), os.path.join(rig_dir, scene, "anchors.npz")
    if not (os.path.isfile(rec_f) and os.path.isfile(anc_f)):
        return None
    rec = json.load(open(rec_f))
    if rec.get("failure_reason") or not rec.get("windows") or rec["windows"][0].get("skipped"):
        return None
    w = rec["windows"][0]
    t_ref, (t0, t1) = int(w["t_ref"]), w["window"]
    z = np.load(anc_f)
    solved = {int(f): P.astype(np.float64) for f, P in zip(z["frames"], z["poses"])}
    if t_ref not in solved:
        return None
    frames = list(range(t_ref, int(t1) + 1))
    cond, src, last = [], [], None
    for t in frames:
        if t in solved:
            last = solved[t]
            src.append(1)
        else:
            src.append(0)
        cond.append(last)
    if gap == "mask":
        cond = [c if s else None for c, s in zip(cond, src)]
    return frames, cond, np.array(src, np.int8)


def load_window(scene_dir, frames, cond_poses, size, gt_ray):
    """Images + ray maps for the window, built exactly like demo_ray.load_frames_training_style but with the
    conditioning pose taken from cond_poses (rig) or from the store GT (gt_ray=True). Returns the demo_ray tuple
    plus a per-view list of 'has ray map' flags (False only for --gap mask holes)."""
    import PIL.Image
    import demo_ray
    from src.dust3r.datasets.base.base_multiview_dataset import get_ray_map
    from src.dust3r.datasets.utils.transforms import ImgNorm

    patch = 16
    images, ray_maps, Ks, gts, has = [], [], [], [], []
    ref = None
    for i, t in enumerate(frames):
        cam = np.load(os.path.join(scene_dir, "dense", "cam", f"{t:06d}.npz"))
        K = cam["intrinsic"].astype(np.float32).copy()
        gt = cam["pose"].astype(np.float32)
        img = PIL.Image.open(os.path.join(scene_dir, "dense", "rgb", f"{t:06d}.png")).convert("RGB")
        W1, H1 = img.size
        tw = max(patch, (int(size) // patch) * patch)
        th = ((H1 * tw + W1 * patch - 1) // (W1 * patch)) * patch
        img, K = demo_ray.crop_resize_training_style(img, K, (tw, th))
        pose = gt if gt_ray else (None if cond_poses[i] is None else cond_poses[i].astype(np.float32))
        if ref is None:
            ref = pose if pose is not None else gt
        if pose is None:  # masked hole: any finite map, ray_mask is switched off below
            ray_map = np.zeros((th, tw, 6), np.float32)
        else:
            ray_map = get_ray_map(ref, pose, K, th, tw).astype(np.float32)
        ray_maps.append(torch.from_numpy(ray_map).unsqueeze(0))
        Ks.append(torch.from_numpy(K).unsqueeze(0))
        gts.append(torch.from_numpy(gt).unsqueeze(0))
        has.append(pose is not None)
        images.append(dict(img=ImgNorm(img)[None], true_shape=np.int32([img.size[::-1]]), idx=i, instance=str(i)))
    return images, ray_maps, Ks, gts, has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--conditioning", required=True, choices=["gt", "gt_all", "none", "none_ts"])
    ap.add_argument("--pose_source", default="rig", choices=["rig", "gt"], help="pose the ray maps are built from")
    ap.add_argument("--gap", default="hold", choices=["hold", "mask"], help="window frames the rig did not solve")
    ap.add_argument("--rig_dir", required=True, help="rig_trackon_eval run dir (one sub-dir per scene)")
    ap.add_argument("--scenes_root", required=True)
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--pred_base", required=True)
    ap.add_argument("--eval_base", required=True)
    ap.add_argument("--gt_win_root", required=True)
    ap.add_argument("--eval_script", required=True)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(args.ckpt)
    import demo_ray
    from src.dust3r.inference import inference
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from infer_and_eval_worker_ray import save_depth_camera

    device = args.device if torch.cuda.is_available() else "cpu"
    scenes = [s.strip() for s in open(args.scene_list) if s.strip()]
    mine = [s for i, s in enumerate(scenes) if i % args.num_shards == args.shard_id]
    if args.limit > 0:
        mine = mine[: args.limit]
    tag = f"[{args.label} shard {args.shard_id}/{args.num_shards}]"
    print(f"{tag} {len(mine)} scenes, conditioning={args.conditioning} pose_source={args.pose_source} gap={args.gap} "
          f"ckpt={args.ckpt}", flush=True)
    t0 = time.time()
    model = ARCroco3DStereo.from_pretrained(args.ckpt).to(device)
    if getattr(model, "pose_gru", None) is not None:
        model.pose_gru = None
    model.eval()
    print(f"{tag} model loaded in {time.time() - t0:.1f}s; trained_conditioning="
          f"{getattr(model, 'trained_conditioning', '?')}", flush=True)
    os.makedirs(args.eval_base, exist_ok=True)
    fail_log = os.path.join(args.eval_base, f"_failures_shard{args.shard_id}.txt")
    done = skipped = failed = 0
    t_start = time.time()
    for idx, scene in enumerate(mine):
        eval_dir = os.path.join(args.eval_base, scene)
        eval_csv = os.path.join(eval_dir, "eval_depth_pose_metrics.csv")
        if os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0:
            skipped += 1
            continue
        win = rig_window(args.rig_dir, scene, args.gap)
        if win is None or len(win[0]) < 2:
            skipped += 1
            continue
        frames, cond, src = win
        scene_dir = os.path.join(args.scenes_root, scene)
        pred_dir = os.path.join(args.pred_base, scene)
        try:
            ts = time.time()
            gt_win = build_window_gt(args.scenes_root, scene, frames, args.gt_win_root)
            outputs = views = None
            try:
                with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                    if args.conditioning == "none":
                        img_paths = [os.path.join(scene_dir, "dense", "rgb", f"{t:06d}.png") for t in frames]
                        views = demo_ray.prepare_input_none(img_paths, args.size)
                    else:
                        images, ray_maps, Ks, gts, has = load_window(
                            scene_dir, frames, cond, args.size, gt_ray=(args.pose_source == "gt"))
                        views = demo_ray.prepare_input(images=images, ray_maps=ray_maps, intrinsics_list=Ks,
                                                       gt_poses=gts, conditioning=args.conditioning)
                        for v, h in zip(views, has):   # --gap mask: switch the ray token off on the holes
                            if not h:
                                v["ray_mask"] = torch.tensor(False).unsqueeze(0)
                with torch.no_grad():
                    outputs, _ = inference(views, model, device)
                nfr = save_depth_camera(outputs, pred_dir, pose_encoding_to_camera, estimate_focal_knowing_depth)
            finally:
                del outputs, views
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            np.savez(os.path.join(pred_dir, "cond_poses.npz"), frames=np.array(frames, np.int32),
                     poses=np.array([np.eye(4) if c is None else c for c in cond], np.float32), src=src,
                     gt=np.array([np.load(os.path.join(scene_dir, "dense", "cam", f"{t:06d}.npz"))["pose"] for t in frames], np.float32),
                     pose_source=args.pose_source, conditioning=args.conditioning, gap=args.gap)
            os.makedirs(eval_dir, exist_ok=True)
            r = subprocess.run([sys.executable, args.eval_script, "--pred_root", pred_dir, "--gt_root", gt_win,
                                "--output_csv", eval_csv], capture_output=True, text=True)
            if r.returncode != 0 or not (os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0):
                raise RuntimeError(f"eval failed rc={r.returncode}: {r.stderr[-500:]}")
            done += 1
            if done <= 3 or done % 25 == 0:
                rate = (time.time() - t_start) / max(done, 1)
                print(f"{tag} [{idx + 1}/{len(mine)}] {scene} frames={nfr} held={int((src == 0).sum())} "
                      f"{time.time() - ts:.1f}s | done={done} skip={skipped} fail={failed} "
                      f"ETA~{(len(mine) - idx - 1) * rate / 3600:.2f}h", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            with open(fail_log, "a") as fh:
                fh.write(f"{scene}\t{repr(e)}\n")
            print(f"{tag} FAIL {scene}: {repr(e)[:300]}", flush=True)
            traceback.print_exc()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    print(f"{tag} DONE done={done} skipped={skipped} failed={failed} elapsed={(time.time() - t_start) / 3600:.2f}h",
          flush=True)


if __name__ == "__main__":
    main()
