#!/usr/bin/env python3
"""Build the finetune-vs-regular sim3 comparison tex table (7 rows incl.
"Captain Cut3r"), with best per column bold and runner-up underlined.

Sources (all per-scene, averaged with nanmean across scenes):
  - existing labels: depth (absrel, a1) from summary/per_scene_<label>.csv
    (alignment-independent, identical across eval eras), pose from
    <label>/sim3_pose_both.csv under the chosen --agg.
  - captain: everything from summary/per_scene_captain_ray_gt_last_sim3both.csv
    (written by aggregate_sim3_both.py from its fresh default-args eval CSVs).

--agg rmse  -> current eval_depth_poses.py defaults (per-scene nan-RMSE pose agg)
--agg mean  -> the Jul-7 tex's aggregation (per-scene nanmean), for reference
"""
import argparse
import csv
import math
import os

ROWS = [
    ("InfiniteVGGT (zero-shot)", "infinitevggt"),
    ("CUT3R (zero-shot)", "regular"),
    ("Finetuned (mini, 50 epochs)", "final"),
    ("Finetuned (full, 8 epochs)", "augfull_last"),
    ("Finetuned (full, 14 epochs)", "augfull_last_ep14"),
    ("Finetuned (full, 50 epochs)", "augfull_final"),
    ("Captain Cut3r", "captain_ray_gt_last"),
]
METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
HIGHER_BETTER = {"a1"}

HEADER = r"""\documentclass[border=8pt,varwidth=25cm]{standalone}
\usepackage{booktabs}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\usepackage{xcolor}
\usepackage{newtxtext,newtxmath} % Times New Roman styling

\begin{document}
\footnotesize
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{lcccccc}
\toprule
Setup & n\_scenes & absrel $\downarrow$ & a1 $\uparrow$ & ate $\downarrow$ & rpe\_trans $\downarrow$ & rpe\_rot $\downarrow$ \\
\midrule
"""
FOOTER = r"""\bottomrule
\end{tabular}
\end{document}
"""


def _nanmean_col(path, col):
    vals = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                v = float(r[col])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(v):
                vals.append(v)
    return (sum(vals) / len(vals) if vals else float("nan")), len(vals)


def _nrows(path):
    with open(path) as f:
        return sum(1 for _ in csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--agg", choices=["mean", "rmse"], required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()

    summary = os.path.join(args.out_root, "summary")
    table = []
    for name, label in ROWS:
        if label == "captain_ray_gt_last":
            src = os.path.join(summary, f"per_scene_{label}_sim3both.csv")
            vals = {
                "absrel": _nanmean_col(src, "absrel")[0],
                "a1": _nanmean_col(src, "a1")[0],
                "ate": _nanmean_col(src, f"ate_{args.agg}")[0],
                "rpe_trans": _nanmean_col(src, f"rpe_trans_{args.agg}")[0],
                "rpe_rot": _nanmean_col(src, f"rpe_rot_{args.agg}")[0],
            }
            n = _nrows(src)
        else:
            depth_src = os.path.join(summary, f"per_scene_{label}.csv")
            pose_src = os.path.join(args.out_root, label, "sim3_pose_both.csv")
            vals = {
                "absrel": _nanmean_col(depth_src, "absrel")[0],
                "a1": _nanmean_col(depth_src, "a1")[0],
                "ate": _nanmean_col(pose_src, f"ate_{args.agg}")[0],
                "rpe_trans": _nanmean_col(pose_src, f"rpe_trans_{args.agg}")[0],
                "rpe_rot": _nanmean_col(pose_src, f"rpe_rot_{args.agg}")[0],
            }
            n = _nrows(depth_src)
        table.append((name, n, vals))

    # best (bold) / runner-up (underline) per metric
    marks = {}
    for m in METRICS:
        vals = [(row[2][m], i) for i, row in enumerate(table)
                if math.isfinite(row[2][m])]
        vals.sort(reverse=(m in HIGHER_BETTER))
        marks[m] = {}
        if vals:
            marks[m][vals[0][1]] = "bf"
        if len(vals) > 1:
            marks[m][vals[1][1]] = "ul"

    lines = []
    for i, (name, n, vals) in enumerate(table):
        cells = [name, str(n)]
        for m in METRICS:
            s = f"{vals[m]:.4f}"
            mk = marks[m].get(i)
            if mk == "bf":
                s = r"\textbf{" + s + "}"
            elif mk == "ul":
                s = r"\underline{" + s + "}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")

    with open(args.out_tex, "w") as f:
        f.write(HEADER + "\n".join(lines) + "\n" + FOOTER)
    print(f"Wrote {args.out_tex}  (agg={args.agg})")
    for (name, n, vals) in table:
        print(f"  {name:32s} n={n}  " +
              "  ".join(f"{m}={vals[m]:.4f}" for m in METRICS))


if __name__ == "__main__":
    main()
