#!/usr/bin/env python3
"""Enumerate test scenes (deterministically) that have a non-empty dense/rgb dir.

Writes one scene name per line (sorted) to the output file.
"""
import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_root", required=True,
                    help="e.g. .../test/dl3dv_multi/wrist")
    ap.add_argument("--out", required=True, help="output scene-list txt path")
    args = ap.parse_args()

    names = sorted(os.listdir(args.scenes_root))
    kept = []
    for n in names:
        rgb = os.path.join(args.scenes_root, n, "dense", "rgb")
        if not os.path.isdir(rgb):
            continue
        # require at least 2 frames (pose RPE needs >=2)
        imgs = glob.glob(os.path.join(rgb, "*.png")) + glob.glob(os.path.join(rgb, "*.jpg"))
        if len(imgs) >= 2:
            kept.append(n)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for n in kept:
            f.write(n + "\n")
    print(f"Found {len(names)} entries, kept {len(kept)} scenes with >=2 frames.")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
