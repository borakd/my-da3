#!/usr/bin/env python
"""Aggregate RobotSeg scores over seeds: score/summary.json of robotseg_seed{0,1,2} (+ the unseeded first run in robotseg/).

Writes .../ext_cams/robotseg/seeds_summary.md and prints it: per res x cam x variant the mean and [min, max] over seeds of
J_ref, J&F (VRS protocol), pixel precision, FP frames, plus the unseeded value of J_ref.

    python robotseg_seed_table.py
"""
import json
import os

import numpy as np

ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams"
EP = "RAIL+80edfcb1+2023-07-14-14h-28m-45s"
SEEDS = [0, 1, 2]
ORDER = ["auto_f0_gripper", "auto_f0_arm", "auto_f0_robot", "auto_entry_gripper", "auto_vis_gripper", "image_gripper",
         "bbox_entry_gripper", "bbox_vis_gripper"]


def load(root, res):
    p = os.path.join(root, res, EP, "score", "summary.json")
    return json.load(open(p))["summary"] if os.path.isfile(p) else None


def main():
    lines = [f"# RobotSeg out of the box on {EP}: seeds {SEEDS} (mean [min, max]) + unseeded first run", "",
             "Reference = SAM3 box+click masks (not human GT). J_ref = mean IoU over reference-present frames; "
             "J&F = RobotSeg tools/benchmark.py protocol; FP = non-empty prediction on a reference-empty frame.", ""]
    for res in ("1280x720", "320x180"):
        seeds = [load(os.path.join(ROOT, f"robotseg_seed{k}"), res) for k in SEEDS]
        seeds = [s for s in seeds if s is not None]
        first = load(os.path.join(ROOT, "robotseg"), res)
        lines += [f"## {res} ({len(seeds)} seeds)", "",
                  "| cam | variant | J_ref | J&F (VRS) | pix prec | FP frames | J_ref unseeded |",
                  "|---|---|---|---|---|---|---|"]
        for cam in ("ext1", "ext2"):
            for v in ORDER:
                rows = [s[cam][v] for s in seeds if v in s[cam]]
                if not rows:
                    continue

                def agg(key, fmt="{:.2f}"):
                    x = np.array([r[key] for r in rows], float)
                    return (fmt + " [" + fmt + ", " + fmt + "]").format(x.mean(), x.min(), x.max())
                un = first[cam][v]["J_ref"] if first and v in first[cam] else None
                lines.append(f"| {cam} | {v} | {agg('J_ref')} | {agg('JF_vrs')} | {agg('pix_prec')} | {agg('FP', '{:.0f}')} | "
                             f"{'-' if un is None else f'{un:.2f}'} |")
        lines.append("")
    out = os.path.join(ROOT, "robotseg", "seeds_summary.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
