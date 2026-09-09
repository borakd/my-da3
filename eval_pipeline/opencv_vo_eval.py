#!/usr/bin/env python
"""Score OpenCV-VO preds with the SAME harness eval as every model arm (eval_depth_poses.py, default args),
one subprocess per scene, in parallel. Mirrors cg_fuse_fwd_bwd.py's eval step.

Expects <out_root>/<label>/preds/<scene>/{camera/*.npz, depth -> symlinked model depth}. Writes
<out_root>/<label>/eval/<scene>/eval_depth_pose_metrics.csv (skips scenes already scored).

Usage: opencv_vo_eval.py --out_root ... --label ... --scenes_root ... --scene_list ... [--workers 48]
"""
import argparse
import os
import subprocess
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_SCRIPT = os.path.join(os.path.dirname(HERE), "eval_bundle", "bin", "eval_depth_poses.py")


def one(task):
    scene, pred_dir, gt_root, eval_csv = task
    if os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0:
        return scene, "skip", ""
    if not os.path.isdir(os.path.join(pred_dir, "camera")) or not os.path.isdir(os.path.join(pred_dir, "depth")):
        return scene, "missing", "no camera/ or depth/ under preds"
    os.makedirs(os.path.dirname(eval_csv), exist_ok=True)
    cmd = [sys.executable, EVAL_SCRIPT, "--pred_root", pred_dir, "--gt_root", gt_root, "--output_csv", eval_csv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not (os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0):
        return scene, "fail", r.stderr[-400:]
    return scene, "ok", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--scenes_root", required=True); ap.add_argument("--scene_list", required=True)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    scenes = [l.strip() for l in open(a.scene_list) if l.strip()]
    tasks = [(s, os.path.join(a.out_root, a.label, "preds", s), os.path.join(a.scenes_root, s, "dense"),
              os.path.join(a.out_root, a.label, "eval", s, "eval_depth_pose_metrics.csv")) for s in scenes]
    counts = {}
    with Pool(a.workers) as p:
        for i, (scene, st, msg) in enumerate(p.imap_unordered(one, tasks, chunksize=4)):
            counts[st] = counts.get(st, 0) + 1
            if st in ("fail", "missing") and counts[st] <= 5:
                print(f"{st}: {scene}: {msg}", flush=True)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} {counts}", flush=True)
    print(f"{a.label}: {counts}", flush=True)


if __name__ == "__main__":
    main()
