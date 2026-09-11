#!/usr/bin/env python
"""Full-harness (all 4292 DROID test scenes) LaTeX table: CUT3R zero-shot / finetuned, with and without the
OpenCV-VO pose backbone. Format/style follows
maks_plots_augfull_lr1e5/eval/.../maks_windows100/tables/windows100_breakdown.tex (standalone, booktabs,
best/2nd/3rd cell colours per column, footnotes in a minipage).

Values: per-scene ALL/MEAN row of eval_depth_pose_metrics.csv (eval_depth_poses.py default args: depth
median-scale-aligned, pose Sim(3)-aligned, RMSE-reduced), nanmean over scenes -- identical to
eval_pipeline/aggregate_results.py.

Usage: build_opencv_full_table.py [--out_root ...] [--scene_list ...] [--tex ...]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aggregate_results import METRICS, read_scene_summary  # noqa: E402

OUT_DEFAULT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
ROWS = [
    ("CUT3R zero-shot (pretrained ckpt)", "cut3r_zeroshot"),
    ("Regular CUT3R (finetuned, augfull\\_lr1e5)", "augfull_lr1e5"),
    ("CUT3R zero-shot + OpenCV pose backbone", "vo_full_zs"),
    ("CUT3R Finetuned + OpenCV pose backbone", "vo_full_ft"),
    ("CUT3R Finetuned + OpenCV pose backbone, GT intrinsics", "vo_full_ftgt"),
]
HIGHER_BETTER = {"a1"}
COLORS = [r"\cellcolor{red!30}\textbf{", r"\cellcolor{orange!30}", r"\cellcolor{yellow!40}"]


def load(out_root, label, scenes):
    vals = {m: [] for m in METRICS}; n = 0
    for s in scenes:
        r = read_scene_summary(os.path.join(out_root, label, "eval", s, "eval_depth_pose_metrics.csv"))
        if r is None:
            continue
        n += 1
        for m in METRICS:
            vals[m].append(r[m])
    return n, {m: float(np.nanmean(vals[m])) if vals[m] else float("nan") for m in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=OUT_DEFAULT)
    ap.add_argument("--scene_list", default=os.path.join(OUT_DEFAULT, "scene_list.txt"))
    ap.add_argument("--tex", default=os.path.join(OUT_DEFAULT, "tables", "opencv_backbone_full4292.tex"))
    args = ap.parse_args()
    scenes = [l.strip() for l in open(args.scene_list) if l.strip()]

    data = [(name, lbl) + load(args.out_root, lbl, scenes) for name, lbl in ROWS]
    present = [d for d in data if d[2] > 0]
    # rank per column (ties share a rank), 3-decimal display values are what get compared, as in the reference
    disp = {(d[1], m): round(d[3][m], 4) for d in present for m in METRICS}
    rank = {}
    for m in METRICS:
        vs = sorted({disp[(d[1], m)] for d in present if np.isfinite(disp[(d[1], m)])}, reverse=(m in HIGHER_BETTER))
        for d in present:
            v = disp[(d[1], m)]
            rank[(d[1], m)] = vs.index(v) if np.isfinite(v) and v in vs else 99
    lines = []
    for name, lbl, n, v in data:
        if n == 0:
            lines.append(f"{name} & \\multicolumn{{5}}{{c}}{{(pending)}} \\\\")
            continue
        cells = []
        for m in METRICS:
            # 4 decimals: at 3 dp, 0.1197 and 0.1203 both render as "0.120" and the ATE /
            # RPE-trans columns look falsely identical. 4 dp matches summary/averages_table.txt.
            txt = f"{v[m]:.4f}"
            rk = rank[(lbl, m)]
            if rk < 3:
                txt = COLORS[rk] + txt + ("}" if rk == 0 else "")
            cells.append(txt)
        nn = f" (n={n})" if n != len(scenes) else ""
        lines.append(f"{name}{nn} & " + " & ".join(cells) + r" \\")
    n_full = max((d[2] for d in present), default=0)

    tex = r"""\documentclass[11pt,border=8pt]{standalone}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\newsavebox{\tblbox}
\begin{document}
\sbox{\tblbox}{%
\begin{tabular}{@{}l r r | r r r@{}}
\toprule
 & \multicolumn{2}{c|}{Depth} & \multicolumn{3}{c}{Pose metrics} \\
Method & AbsRel$\downarrow$ & $\delta{<}1.25\uparrow$
 & ATE$\downarrow$ & RPE\textsubscript{trans}$\downarrow$ & RPE\textsubscript{rot}$\downarrow$ \\
\midrule
""" + "\n".join(lines[:2]) + "\n\\midrule\n" + "\n".join(lines[2:]) + r"""
\bottomrule
\end{tabular}}%
\begin{minipage}{\wd\tblbox}
\usebox{\tblbox}\par\vspace{2pt}
{\footnotesize All """ + str(n_full) + r""" DROID wrist test scenes. Values = per-scene averages (eval\_depth\_poses.py default args), averaged over scenes. Depth = mean over frames (per-frame median-scaled); pose = Sim3-aligned RMSE over frames; RPE\textsubscript{rot} in degrees.\par}
{\footnotesize OpenCV pose backbone: Shi--Tomasi + LK tracks, parallax-gated essential-matrix bootstrap, PnP against a triangulated map, keyframes every 5\,px of parallax, outlier culling, re-bootstrap after 3 failed frames with scale hand-off, static-track lever on. Inputs at frame $t$: the images and CUT3R's predicted focal for frame $t$ (closed loop, inference-time); depth columns are the paired CUT3R model's own depth. The last row replaces the predicted focal with the calibrated intrinsics (privileged, diagnostic). One non-causal step is retained: frames before each (re-)bootstrap succeeds are posed retroactively by PnP against the map built at the bootstrap frame (typically 2--16 frames per scene, plus a few after each re-bootstrap); all other frames use only past information.\par}
{\footnotesize Highlighting: \colorbox{red!30}{\textbf{best}}, \colorbox{orange!30}{2nd}, \colorbox{yellow!40}{3rd} per column (ties share)\par}
\end{minipage}
\end{document}
"""
    os.makedirs(os.path.dirname(args.tex), exist_ok=True)
    open(args.tex, "w").write(tex)
    d, base = os.path.dirname(args.tex), os.path.splitext(os.path.basename(args.tex))[0]
    cmd = ("export TEXMFROOT=/apps/GPP/LATEX/20240430 && export TEXMFCNF=$TEXMFROOT:$TEXMFROOT/texmf-dist/web2c && "
           "export PATH=/apps/GPP/LATEX/20240430/bin/x86_64-linux:$PATH; "
           f"cd {d} && pdflatex -interaction=nonstopmode {base}.tex >{base}.build.log 2>&1 && "
           f"gs -sDEVICE=png16m -r300 -o {base}.png -dBATCH -dNOPAUSE {base}.pdf >/dev/null")
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"compile failed: see {d}/{base}.build.log")
    print("OK:", args.tex, "+ .pdf + .png")
    for name, lbl, n, v in data:
        print(f"  {name:55s} n={n:4d} " + " ".join(f"{m}={v[m]:.4f}" for m in METRICS))


if __name__ == "__main__":
    main()
