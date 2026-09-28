#!/usr/bin/env python3
"""The trained confidence head (G1L, hard rank rule q.5 g.5) evaluated with BOTH CUT3R backbones on all 4292
DROID wrist test scenes. Two groups, each with its own plain row as the paired baseline:

  pretrained backbone  cut3r_512_dpt_4_64.pth      plain = cut3r_zeroshot;  gated arms under $OUT/zs_full4292
  finetuned backbone   augfull_lr1e5 checkpoint     plain = augfull_lr1e5;   gated arms under $OUT/unc_full4292

Rows per group: plain; gate keyed by the backbone's own (untrained) self-view confidence; gate keyed by the G1L head.
Deltas in parentheses are paired per scene vs the group's plain row (* = 95% scene-bootstrap CI excludes 0); a second
line gives the G1L-head arm paired against the own-confidence arm, i.e. what the head's training bought on that backbone.
Writes $OUT/zs_full4292/table/conf_head_backbones_4292.{md,tex,pdf,png}.
"""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maks_subset_compare import load_label, boot_ci
OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
TAB = f"{OUT}/zs_full4292/table"; os.makedirs(TAB, exist_ok=True)
METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]; HIGHER = {"a1"}
scenes = [l.strip() for l in open(f"{OUT}/scene_list.txt") if l.strip()]
GROUPS = [
    ("Pretrained CUT3R backbone (no DROID training in the backbone)", [
        ("plain", "CUT3R zero-shot", f"{OUT}/cut3r_zeroshot/eval"),
        ("own", "+ token gate keyed by its own self-view confidence (untrained)", f"{OUT}/zs_full4292/tok_al_q50_g50/eval"),
        ("g1l", "+ token gate keyed by the G1L head (trained on the finetuned backbone)", f"{OUT}/zs_full4292/tok_cbG1L_q50_g50/eval")]),
    ("Finetuned CUT3R backbone (augfull, lr 1e-5)", [
        ("plain", "CUT3R finetuned", f"{OUT}/augfull_lr1e5/eval"),
        ("own", "+ token gate keyed by its own self-view confidence (untrained)", f"{OUT}/unc_full4292/tok_al_q50_g50/eval"),
        ("g1l", "+ token gate keyed by the G1L head (trained on this backbone)", f"{OUT}/unc_full4292/tok_cbG1L_q50_g50/eval")]),
]

def paired(a, b, common):
    out = {}
    for m in METRICS:
        d = np.array([a[s][m] - b[s][m] for s in common], float); d = d[np.isfinite(d)]
        lo, hi = boot_ci(d, n=10000); wins = int((d > 0).sum() if m in HIGHER else (d < 0).sum())
        out[m] = (float(d.mean()), lo, hi, wins, len(d))
    return out

def fd(m, d):
    mean, lo, hi, wins, n = d; star = "*" if (lo > 0 or hi < 0) else ""
    return f"{mean:+.4f}{star}" if m != "rpe_rot" else f"{mean:+.3f}{star}"

def esc(t): return t.replace("_", r"\_").replace("<", "$<$").replace("&", r"\&").replace("%", r"\%")

md = ["| backbone | method | n | AbsRel | d<1.25 | ATE | RPE trans | RPE rot |", "|---|---|---|---|---|---|---|---|"]
tex = [r"\documentclass{standalone}", r"\usepackage{booktabs,xcolor,colortbl,amsmath}", r"\begin{document}",
       r"\small\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{l r r r r r}", r"\toprule",
       r"Method & AbsRel $\downarrow$ & $\delta<1.25$ $\uparrow$ & ATE $\downarrow$ & RPE$_\text{trans}$ $\downarrow$ & RPE$_\text{rot}$ $\downarrow$ \\"]
for gname, arms in GROUPS:
    data = {k: load_label(p, scenes) for k, _, p in arms}
    common = [s for s in scenes if all(s in data[k] for k in data)]
    assert len(common) >= 4292 * 0.999, (gname, len(common))
    means = {k: {m: float(np.nanmean([data[k][s][m] for s in common])) for m in METRICS} for k in data}
    dplain = {k: paired(data[k], data["plain"], common) for k in ("own", "g1l")}
    dhead = paired(data["g1l"], data["own"], common)
    tex += [r"\midrule", r"\multicolumn{6}{l}{\emph{" + esc(gname) + r"}} \\"]
    for k, label, _ in arms:
        cells = []
        for m in METRICS:
            v = f"{means[k][m]:.4f}" if m != "rpe_rot" else f"{means[k][m]:.3f}"
            cells.append(v if k == "plain" else f"{v} \\scriptsize({fd(m, dplain[k][m])})")
        tex.append(r"\quad " + esc(label) + " & " + " & ".join(cells) + r" \\")
        md.append(f"| {gname.split(' (')[0]} | {label} | {len(common)} | " + " | ".join(
            (f"{means[k][m]:.4f}" if k == "plain" else f"{means[k][m]:.4f} ({fd(m, dplain[k][m])}, {dplain[k][m][3]}/{dplain[k][m][4]})") for m in METRICS) + " |")
    tex.append(r"\quad \scriptsize G1L head minus own confidence & " + " & ".join(f"\\scriptsize {fd(m, dhead[m])}" for m in METRICS) + r" \\")
    md.append(f"| {gname.split(' (')[0]} | G1L head minus own confidence (paired) | {len(common)} | " + " | ".join(
        f"{fd(m, dhead[m])} [{dhead[m][1]:+.4f},{dhead[m][2]:+.4f}] {dhead[m][3]}/{dhead[m][4]}" for m in METRICS) + " |")
tex += [r"\bottomrule",
        r"\multicolumn{6}{l}{\scriptsize All 4292 DROID wrist test scenes, single causal pass. Parentheses: paired per-scene delta vs the group's plain row; * = 95\% scene-bootstrap CI excludes 0.} \\",
        r"\multicolumn{6}{l}{\scriptsize Token gate: the 240 on-image state tokens ranked by pooled confidence, top half commit fully, rest at weight 0.5; registers always commit. G1L head: 0.44M-param copy of the} \\",
        r"\multicolumn{6}{l}{\scriptsize self-view conf conv stack, trained one epoch on DROID through the differentiable gate on the frozen finetuned backbone (gscale 1, lr 1e-5); hot-swapped onto the pretrained backbone.} \\",
        r"\end{tabular}", r"\end{document}"]
open(f"{TAB}/conf_head_backbones_4292.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
open(f"{TAB}/conf_head_backbones_4292.tex", "w").write("\n".join(tex) + "\n")
try:
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "conf_head_backbones_4292.tex"], cwd=TAB, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    subprocess.run(["gs", "-sDEVICE=png16m", "-r300", "-o", f"{TAB}/conf_head_backbones_4292.png", "-dBATCH", "-dNOPAUSE",
                    f"{TAB}/conf_head_backbones_4292.pdf"], check=True, timeout=300, stdout=subprocess.DEVNULL)
    print("wrote", f"{TAB}/conf_head_backbones_4292.png")
except Exception as e:
    print("pdf/png build failed:", e)
