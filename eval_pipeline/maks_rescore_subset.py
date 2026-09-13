#!/usr/bin/env python3
"""Re-score an existing prediction tree on a SUBSET of frames (the kept set),
with the unmodified eval script and its default args, so the numbers stay
comparable with every table row.

Builds <out_dir>/_kept_preds/{depth,camera} as symlinks to the pred files whose
index is NOT in the skip json for this scene, then runs eval_depth_poses.py
--pred_root <that dir> --gt_root <gt> --output_csv <out_dir>/eval_depth_pose_metrics.csv.

The evaluator renumbers files POSITIONALLY per camera stream, so a kept-GT
tree (<out_dir>/_kept_gt/{depth,cam}, same frame numbers) is built as well;
the CSV's local_timestep is then the position within K, not the original
frame number (kept_frames.json in out_dir maps local -> original). Sim3 is
re-fit on the kept centres and RPE pairs straddle the gaps (every arm scored
on K sees the same effect).
"""
import argparse
import json
import os
import subprocess
import sys


def build_kept_gt(gt_dense, kept_ids, dst):
    """Symlink GT depth/cam of kept_ids into dst/{depth,cam} (same frame numbers)."""
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
                os.symlink(os.path.abspath(os.path.join(src_dir, fn)), os.path.join(dst_dir, fn))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", required=True, help="<preds>/<scene> with depth/ and camera/")
    ap.add_argument("--gt_root", required=True, help="<scene>/dense")
    ap.add_argument("--skip_json", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eval_script", required=True)
    args = ap.parse_args()

    with open(args.skip_json) as f:
        bad = set(int(i) for i in json.load(f).get(args.scene, []))

    kept_root = os.path.join(args.out_dir, "_kept_preds")
    kept_ids = []
    n_kept = 0
    for sub, suf in (("depth", ".npy"), ("camera", ".npz")):
        src_dir = os.path.join(args.pred_root, sub)
        dst_dir = os.path.join(kept_root, sub)
        os.makedirs(dst_dir, exist_ok=True)
        for old in os.listdir(dst_dir):
            os.remove(os.path.join(dst_dir, old))
        for fn in sorted(os.listdir(src_dir)):
            stem, ext = os.path.splitext(fn)
            if ext != suf or not stem.isdigit():
                continue
            if int(stem) in bad:
                continue
            os.symlink(os.path.abspath(os.path.join(src_dir, fn)), os.path.join(dst_dir, fn))
            if sub == "depth":
                n_kept += 1
                kept_ids.append(int(stem))
    if n_kept < 2:
        raise SystemExit(f"only {n_kept} kept frames under {args.pred_root}")
    gt_kept = build_kept_gt(args.gt_root, kept_ids, os.path.join(args.out_dir, "_kept_gt"))
    with open(os.path.join(args.out_dir, "kept_frames.json"), "w") as f:
        json.dump(kept_ids, f)

    out_csv = os.path.join(args.out_dir, "eval_depth_pose_metrics.csv")
    cmd = [sys.executable, args.eval_script,
           "--pred_root", kept_root, "--gt_root", gt_kept, "--output_csv", out_csv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not (os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0):
        raise SystemExit(f"eval failed rc={r.returncode}: {r.stderr[-800:]}")
    print(f"[rescore] {args.scene}: kept {n_kept} frames (skipped {len(bad)}) -> {out_csv}",
          flush=True)


if __name__ == "__main__":
    main()
