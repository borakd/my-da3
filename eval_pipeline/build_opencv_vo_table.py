#!/usr/bin/env python
"""LaTeX table (+ PDF + PNG) of the OpenCV VO control arm vs the CUT3R arms on the 12-scene smoke set.

Reads <OUT>/<label>/sim3_pose_both.csv (pose_sim3_both.py output; Sim(3)-aligned, per-scene means) and
<OUT>/<label>/preds/<scene>/summary.json (OpenCV diagnostics). Depth columns are not shown: the OpenCV arm
predicts poses only (AbsRel / delta1 would be inherited from augfull_lr1e5). Same standalone/booktabs/newtx
recipe as build_gru_vs_prevpred_teal.py; compiled with the MN5 latex module + /usr/bin/gs.

Usage: build_opencv_vo_table.py [--out_root ...] [--scene_list ...] [--tex ...]
"""
import argparse
import csv
import glob
import json
import os
import subprocess

import numpy as np

OUT_DEFAULT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"

# (display name, label, group). Groups are separated by \midrule.
ROWS = [
    (r"CUT3R zero-shot (pretrained ckpt)", "cut3r_zeroshot", "model"),
    (r"CUT3R finetuned (\texttt{augfull\_lr1e5})", "augfull_lr1e5", "model"),
    (r"Champion: gated fwd$\times$bwd fusion (\texttt{augfull\_cg\_fuse\_g7})", "augfull_cg_fuse_g7", "model"),
    (r"OpenCV VO v0 + zero-shot CUT3R focal at frame $t$", "vo6_zs_perframe", "opencv"),
    (r"OpenCV VO v0 + finetuned CUT3R focal at frame $t$", "vo6_ft_perframe", "opencv"),
    (r"OpenCV VO v0 + finetuned CUT3R focal, running median to $t$", "vo6_ft_causal", "opencv"),
    (r"OpenCV VO v0, fixed focal 203\,px (model-free)", "vo4_fnominal", "opencv"),
]


def load_label(out_root, label, scenes):
    rows = {r["scene"]: r for r in csv.DictReader(open(os.path.join(out_root, label, "sim3_pose_both.csv")))}
    rows = [rows[s] for s in scenes if s in rows]
    ate = np.array([float(r["ate_mean"]) for r in rows]); rt = np.array([float(r["rpe_trans_mean"]) for r in rows])
    rr = np.array([float(r["rpe_rot_mean"]) for r in rows])
    vals = dict(n=len(rows), ate=1000 * ate.mean(), ate_med=1000 * np.median(ate), rpe_t=1000 * rt.mean(), rpe_r=rr.mean())
    summ = [json.load(open(f)) for f in glob.glob(os.path.join(out_root, label, "preds", "*", "summary.json"))]
    summ = [s for s in summ if s["scene"] in set(scenes)]
    if summ:
        fr = sum(s["frames"] for s in summ)
        vals["failed"] = 100.0 * sum(s["failed_frames"] for s in summ) / fr
        vals["reboots"] = sum(s["reboots"] for s in summ)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=OUT_DEFAULT)
    ap.add_argument("--scene_list", default=os.path.join(os.path.dirname(__file__), "cg_smoke_scenes_12.txt"))
    ap.add_argument("--tex", default=os.path.join(OUT_DEFAULT, "tables", "opencv_vo_smoke12.tex"))
    args = ap.parse_args()
    scenes = [l.strip() for l in open(args.scene_list) if l.strip()]

    data = [(name, load_label(args.out_root, lbl, scenes), grp) for name, lbl, grp in ROWS]
    cols = ["ate", "ate_med", "rpe_t", "rpe_r"]
    best = {c: min(v[c] for _, v, _ in data) for c in cols}
    lines, prev = [], None
    for name, v, grp in data:
        if prev is not None and grp != prev:
            lines.append(r"\midrule")
        prev = grp
        cells = []
        for c in cols:
            txt = f"{v[c]:.1f}" if c != "rpe_r" else f"{v[c]:.2f}"
            cells.append(r"\textbf{" + txt + "}" if abs(v[c] - best[c]) < 1e-12 else txt)
        diag = f"{v['failed']:.0f}\\,\\% / {v['reboots']}" if "failed" in v else "--"
        lines.append(f"{name} & " + " & ".join(cells) + f" & {diag} \\\\")

    n = data[0][1]["n"]
    tex = r"""\documentclass[border=8pt,varwidth=25cm]{standalone}
\usepackage{booktabs}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\usepackage[table]{xcolor}
\usepackage{newtxtext,newtxmath}

\begin{document}
\begin{center}
{\normalsize\bfseries OpenCV visual-odometry control arm vs.\ CUT3R on the DROID wrist smoke set (""" + str(n) + r""" scenes)}\par\vspace{8pt}
\footnotesize
\setlength{\tabcolsep}{7pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{lccccc}
\toprule
 & \multicolumn{4}{c}{Pose (Sim(3)-aligned, per-scene mean, then mean over scenes)} & Diagnostics \\
\cmidrule(lr){2-5}\cmidrule(lr){6-6}
Method & ATE (mm) $\downarrow$ & ATE median (mm) $\downarrow$ & RPE$_{\mathrm{trans}}$ (mm) $\downarrow$ & RPE$_{\mathrm{rot}}$ ($^{\circ}$) $\downarrow$ & failed frames / re-bootstraps \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\par\vspace{6pt}
\begin{minipage}{0.97\linewidth}\scriptsize
OpenCV VO v0: Shi--Tomasi + LK tracks, parallax-gated essential-matrix bootstrap (0.5\,px), PnP against a
triangulated map, keyframes every 5\,px of parallax, outlier culling, re-bootstrap after 3 failed frames with
scale hand-off, static-track lever on; images only plus the stated focal, which at frame $t$ uses only what CUT3R
has produced up to $t$ (inference-time). ``Failed frames'':
pose held because PnP found $<30$ inliers; each re-bootstrap starts a new arbitrary-scale segment. Model rows
use the same scorer on the same scenes. Depth metrics not shown (the OpenCV arm predicts poses only).
\end{minipage}
\end{center}
\end{document}
"""
    os.makedirs(os.path.dirname(args.tex), exist_ok=True)
    open(args.tex, "w").write(tex)
    d, base = os.path.dirname(args.tex), os.path.splitext(os.path.basename(args.tex))[0]
    # Prefer the MN5 latex module when lmod is present (compute/older login nodes); fall back to /usr/bin/pdflatex.
    # The MN5 LaTeX tree is used directly (the system /usr/bin/pdflatex lacks standalone.cls); no lmod needed.
    modload = ("export TEXMFROOT=/apps/GPP/LATEX/20240430 && export TEXMFCNF=$TEXMFROOT:$TEXMFROOT/texmf-dist/web2c && "
               "export PATH=/apps/GPP/LATEX/20240430/bin/x86_64-linux:$PATH; ")
    cmd = (modload + f"cd {d} && pdflatex -interaction=nonstopmode {base}.tex >{base}.build.log 2>&1 && "
           f"gs -sDEVICE=png16m -r300 -o {base}.png -dBATCH -dNOPAUSE {base}.pdf >/dev/null")
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"compile failed: {r.stderr[-1500:]}\n{r.stdout[-800:]}")
    print("OK:", args.tex, "+ .pdf + .png")
    for name, v, _ in data:
        print(f"  {name:70s} ATE {v['ate']:6.1f} med {v['ate_med']:6.1f} RPEt {v['rpe_t']:5.2f} RPEr {v['rpe_r']:.3f}")


if __name__ == "__main__":
    main()
