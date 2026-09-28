#!/usr/bin/env python3
"""Full-4292 table for the uncgate v1 campaign: every finished arm under $OUT/unc_full4292/<arm>/eval,
paired per scene against augfull_lr1e5, with the frame gate and the self-conf aligned token gate as references.
Writes $OUT/unc_full4292/table/unc_gate_4292.{csv,md,tex,pdf,png}. Requires pdflatex on PATH for pdf/png.
"""
import csv, glob, json, math, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from maks_subset_compare import load_label, boot_ci  # per-scene MEAN rows, scene-level bootstrap

OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
ROOT = f"{OUT}/unc_full4292"; TAB = f"{ROOT}/table"; os.makedirs(TAB, exist_ok=True)
METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]; HIGHER = {"a1"}
scenes = [l.strip() for l in open(f"{OUT}/scene_list.txt") if l.strip()]
NAMES = {
    "augfull_lr1e5": "CUT3R finetuned (baseline)",
    "augfull_cg_g7ema": "Conf-Gate frame gate (g7ema)",
    "causal_posefix_tc": "Causal pose recalibration",
    "tok_al_q50_g50": "Aligned token gate, self-view conf (q.5, g.5)",
    "tok_uncA_q50_g50": "v1 head A (Gauss, clip 64): token gate q.5 g.5",
    "tok_uncA_q50_g50_ema7": "v1 head A + per-token EMA .7",
    "tok_uncB_q50_g50": "v1 head B (Laplace, clip 64): token gate q.5 g.5",
    "tok_uncB_q50_g50_ema7": "v1 head B + per-token EMA .7",
    "tok_uncC_q50_g50": "v1 head C (Gauss, clip 16): token gate q.5 g.5",
    "tok_uncC_q50_g50_ema7": "v1 head C + per-token EMA .7",
    "tok_cbG1_q50_g50": "Conf branch trained through the gate (gscale 1): hard rank rule q.5 g.5",
    "tok_cbG1_q50_g50_soft05": "Conf branch trained through the gate (gscale 1): soft rule T .5",
    "tok_cbG10_q50_g50": "Conf branch trained through the gate (gscale 10): hard rank rule q.5 g.5",
    "tok_cbG10_q50_g50_soft05": "Conf branch trained through the gate (gscale 10): soft rule T .5",
    "tok_cbG10L_q50_g50": "Conf branch trained through the gate (gscale 10, lr 1e-5): hard rank rule q.5 g.5",
    "tok_cbG10L_q50_g50_soft05": "Conf branch trained through the gate (gscale 10, lr 1e-5): soft rule T .5",
    "tok_cbG1L_q50_g50": "Conf branch trained through the gate (gscale 1, lr 1e-5): hard rank rule q.5 g.5",
    "tok_cbG1L_q50_g50_soft05": "Conf branch trained through the gate (gscale 1, lr 1e-5): soft rule T .5",
}
base = load_label(f"{OUT}/augfull_lr1e5/eval", scenes)
rows = []  # (label, n, means dict, deltas dict{metric:(mean, lo, hi, wins)})
def add(label, data):
    common = [s for s in scenes if s in data and s in base]
    if len(common) < 4292 * 0.999:
        print(f"skip {label}: {len(common)}/4292 scenes"); return
    means = {m: float(np.nanmean([data[s][m] for s in common])) for m in METRICS}
    deltas = {}
    for m in METRICS:
        d = np.array([data[s][m] - base[s][m] for s in common], float); d = d[np.isfinite(d)]
        lo, hi = boot_ci(d, n=10000); wins = int((d > 0).sum() if m in HIGHER else (d < 0).sum())
        deltas[m] = (float(d.mean()), lo, hi, wins, len(d))
    rows.append((label, len(common), means, deltas))
add("augfull_lr1e5", base)
for lab, path in (("augfull_cg_g7ema", f"{OUT}/augfull_cg_g7ema/eval"), ("causal_posefix_tc", f"{OUT}/causal_posefix_tc/eval")):
    if os.path.isdir(path): add(lab, load_label(path, scenes))
for d in sorted(glob.glob(f"{ROOT}/*/eval")):
    lab = d.split("/")[-2]
    add(lab, load_label(d, scenes))
# ---- csv / md
with open(f"{TAB}/unc_gate_4292.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["label", "n"] + METRICS + [f"d_{m}" for m in METRICS] + [f"sig_{m}" for m in METRICS])
    for lab, n, means, deltas in rows:
        w.writerow([lab, n] + [f"{means[m]:.6f}" for m in METRICS] + [f"{deltas[m][0]:+.6f}" for m in METRICS]
                   + [int(deltas[m][1] > 0 or deltas[m][2] < 0) for m in METRICS])
def fmt_delta(m, d):
    mean, lo, hi, wins, n = d; star = "*" if (lo > 0 or hi < 0) else ""
    return f"{mean:+.4f}{star}" if m != "rpe_rot" else f"{mean:+.3f}{star}"
md = ["| method | n | AbsRel | d<1.25 | ATE | RPE trans | RPE rot |", "|---|---|---|---|---|---|---|"]
for lab, n, means, deltas in rows:
    cells = []
    for m in METRICS:
        v = f"{means[m]:.4f}" if m != "rpe_rot" else f"{means[m]:.3f}"
        cells.append(v if lab == "augfull_lr1e5" else f"{v} ({fmt_delta(m, deltas[m])}, {deltas[m][3]}/{deltas[m][4]})")
    md.append(f"| {NAMES.get(lab, lab)} | {n} | " + " | ".join(cells) + " |")
open(f"{TAB}/unc_gate_4292.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
# ---- tex (standalone, booktabs); best / 2nd / 3rd shading among non-baseline rows
def esc(t): return t.replace("_", r"\_").replace("<", "$<$").replace("&", r"\&")
ranks = {}
for m in METRICS:
    vals = sorted([(r[2][m], r[0]) for r in rows], reverse=(m in HIGHER))
    ranks[m] = {lab: i for i, (v, lab) in enumerate(vals)}
COL = ["red!30", "orange!30", "yellow!40"]
tex = [r"\documentclass{standalone}", r"\usepackage{booktabs,xcolor,colortbl,amsmath}", r"\begin{document}",
       r"\small\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{l r r r r r}", r"\toprule",
       r"Method & AbsRel $\downarrow$ & $\delta<1.25$ $\uparrow$ & ATE $\downarrow$ & RPE$_\text{trans}$ $\downarrow$ & RPE$_\text{rot}$ $\downarrow$ \\", r"\midrule"]
for lab, n, means, deltas in rows:
    cells = []
    for m in METRICS:
        v = f"{means[m]:.4f}" if m != "rpe_rot" else f"{means[m]:.3f}"
        rk = ranks[m][lab]; cell = v if lab == "augfull_lr1e5" else f"{v} \\scriptsize({fmt_delta(m, deltas[m])})"
        if rk < 3: cell = f"\\cellcolor{{{COL[rk]}}}" + cell
        cells.append(cell)
    tex.append(esc(NAMES.get(lab, lab)) + " & " + " & ".join(cells) + r" \\")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{document}"]
# note line under the table
tex.insert(-2, r"\multicolumn{6}{l}{\scriptsize All 4292 DROID wrist test scenes; parentheses: paired delta vs the baseline, * = 95\% scene-bootstrap CI excludes 0; cell colour = best / 2nd / 3rd.} \\")
open(f"{TAB}/unc_gate_4292.tex", "w").write("\n".join(tex) + "\n")
try:
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "unc_gate_4292.tex"], cwd=TAB, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    # ghostscript is the converter the master table uses (ImageMagick is not on the compute nodes)
    subprocess.run(["gs", "-sDEVICE=png16m", "-r300", "-o", f"{TAB}/unc_gate_4292.png", "-dBATCH", "-dNOPAUSE",
                    f"{TAB}/unc_gate_4292.pdf"], check=True, timeout=300, stdout=subprocess.DEVNULL)
    print("wrote", f"{TAB}/unc_gate_4292.png")
except Exception as e:
    print("pdf/png build failed:", e)
json.dump({"rows": [(l, n, m, {k: list(v) for k, v in d.items()}) for l, n, m, d in rows]}, open(f"{TAB}/unc_gate_4292.json", "w"), indent=1)
