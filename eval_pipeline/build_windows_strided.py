#!/usr/bin/env python3
"""Build N-frame sliding-window pseudo-scenes (configurable stride) from one
DROID scene, as symlink trees infer_and_eval_worker_ray.py treats as scenes.

Same layout/contract as the stride-1 `build_windows.py` under
outputs/cut3r_eval/wandb_stats_verification/scripts, plus --stride so a long
window (64 = the training num_views) stays affordable:

  <out_root>/windows_scenes/win_XXXX/dense/{rgb,cam,depth}/<orig basename>
  <out_root>/scene_list.txt

Original basenames are kept: the eval script pairs pred<->GT by local index
after the per-camera split, and demo_ray pairs rgb<->cam by basename.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dense", required=True, help="source scene .../dense dir")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()

    rgb_dir = os.path.join(args.src_dense, "rgb")
    cam_dir = os.path.join(args.src_dense, "cam")
    depth_dir = os.path.join(args.src_dense, "depth")

    frames = sorted(
        f for f in os.listdir(rgb_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    n, w, s = len(frames), args.window, args.stride
    if n < w:
        sys.exit(f"only {n} frames, need >= {w}")

    stems = [os.path.splitext(f)[0] for f in frames]
    missing = [st for st in stems
               if not os.path.isfile(os.path.join(cam_dir, st + ".npz"))
               or not os.path.isfile(os.path.join(depth_dir, st + ".npy"))]
    if missing:
        sys.exit(f"{len(missing)} frames missing cam/.npz or depth/.npy, e.g. {missing[:5]}")

    scenes_root = os.path.join(args.out_root, "windows_scenes")
    os.makedirs(scenes_root, exist_ok=True)

    names = []
    for i in range(0, n - w + 1, s):
        name = f"win_{i:04d}"
        names.append(name)
        dense = os.path.join(scenes_root, name, "dense")
        for sub in ("rgb", "cam", "depth"):
            os.makedirs(os.path.join(dense, sub), exist_ok=True)
        for j in range(i, i + w):
            for sub, fn in (("rgb", frames[j]),
                            ("cam", stems[j] + ".npz"),
                            ("depth", stems[j] + ".npy")):
                dst = os.path.join(dense, sub, fn)
                if not os.path.lexists(dst):
                    os.symlink(os.path.join(args.src_dense, sub, fn), dst)

    list_path = os.path.join(args.out_root, "scene_list.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"{n} frames -> {len(names)} windows of {w} (stride {s})")
    print(f"scenes_root: {scenes_root}")
    print(f"scene_list:  {list_path}")


if __name__ == "__main__":
    main()
