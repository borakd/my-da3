#!/usr/bin/env python3
"""Result table for the write-gate variants (WRITE_GATE_VARIANTS.md), paired per scene against the
plain finetuned run (augfull_lr1e5) on ONE scene subset (the 430 list or the full 4292 list).

Rows (in this order; an arm whose eval tree is missing is skipped, one that is incomplete is skipped
unless --allow_partial): plain; V1 head, V1 dose-matched constant, V1 oracle; the same three for the
RELATIVE-residual V1 heads (fg_*_rel: trained on the consecutive-frame pose residual, train_wgate_heads.py
--target rel); V2 leverage probes (const 0, freeze after 8, const .5), V2 head, V2 constant, V2 oracle;
the V2 rel triple (mg_*_rel); variant C (consequence-trained gate: cg1_head, cg2_head, the two
_lr5 arms and the dose-matched constants cg1_const / cg2_const, under a "Variant C
(consequence-trained)" heading row); then any --ref rows.
Arm eval dirs: <pilot_dir>/<arm>/eval (as maks_sweep430.sbatch lays them out); the constants' values
are read from <pilot_dir>/<arm>/control.json when present and shown in the row label.

ONE common scene set: every complete row (baseline included) is aggregated over exactly the same
scenes = listed scenes that are scored in the baseline AND in every admitted complete arm (an arm is
"complete" when it covers >= --min_frac of the list; default .999 tolerates a handful of failed
scenes on the 4292 list, and those scenes are then dropped from ALL rows). Rows admitted only under
--allow_partial (dagger) are computed on their own scored subset of the common set and show their n.
Oracle arms: a scene with no entry in the arm's per-scene control.json (copied by the sbatch), or
listed in eval_pipeline/maks_arm_<arm>.skipped.txt (wgate_make_controls.py), ran PLAIN under that
arm's label and is treated as unscored for the arm. Skipped rows and dropped scenes are recorded in
the .json ("skipped", "dropped_scenes") and as a footnote in the .md / .tex.

Numbers: per-scene MEAN rows (maks_subset_compare.load_label), nanmean over the common scenes;
paired delta vs the baseline with a 10k scene-bootstrap 95% CI (boot_ci), * = CI excludes 0, and the
per-scene win count. Writes $OUT/wgate/table/wgate_<subset>.{csv,md,tex,pdf,png,json}
(tex -> pdf -> png needs pdflatex + gs, as unc_gate_table.py).

    python eval_pipeline/wgate_table.py --scene_list eval_pipeline/maks_subset430.txt [--pilot_dir $OUT/wgate430]
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maks_subset_compare import load_label, boot_ci  # noqa: E402  per-scene MEAN rows, scene-level bootstrap

OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
EP = os.path.dirname(os.path.abspath(__file__))
METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
HIGHER = {"a1"}
BASE = "augfull_lr1e5"
LATEX_BIN = "/apps/GPP/LATEX/20240430/bin/x86_64-linux"  # compute nodes have no pdflatex on PATH

# (arm dir under pilot_dir, tex label, markdown label, group). Groups get a \\midrule in the tex.
# "{c}" in a constant row's label is replaced by the value read from the arm's control.json.
ROWS = [
    ("fg_head", "V1 frame gate: FrameConfHead, $a_t=\\mathrm{clip}(\\sigma_\\mathrm{ref}/\\sigma_t,.5,1)$",
     "V1 frame gate: FrameConfHead, a_t = clip(sigma_ref/sigma_t, .5, 1)", "V1"),
    ("fg_const", "V1 dose-matched constant $a={c}$", "V1 dose-matched constant a = {c}", "V1"),
    ("fg_oracle", "V1 oracle: GT residual $\\to a_t$ (same map)", "V1 oracle: GT residual -> a_t (same map)", "V1"),
    ("fg_head_rel", "V1 frame gate, REL target: FrameConfHead on the consecutive-frame residual",
     "V1 frame gate, REL target: FrameConfHead on the consecutive-frame residual", "V1 rel"),
    ("fg_const_rel", "V1 rel dose-matched constant $a={c}$", "V1 rel dose-matched constant a = {c}", "V1 rel"),
    ("fg_oracle_rel", "V1 rel oracle: GT consecutive-frame residual $\\to a_t$",
     "V1 rel oracle: GT consecutive-frame residual -> a_t", "V1 rel"),
    ("mg_const0", "V2 probe: pose memory frozen ($b=0$)", "V2 probe: pose memory frozen (b = 0)", "V2 probes"),
    ("mg_freeze8", "V2 probe: memory frozen after frame 8", "V2 probe: memory frozen after frame 8", "V2 probes"),
    ("mg_const50", "V2 probe: constant $b=.5$", "V2 probe: constant b = .5", "V2 probes"),
    ("mg_head", "V2 mem gate: PoseSigmaHead, $b_t=\\mathrm{clip}(1/\\sigma_\\mathrm{comb},.5,1)$",
     "V2 mem gate: PoseSigmaHead, b_t = clip(1/sigma_comb, .5, 1)", "V2"),
    ("mg_const", "V2 dose-matched constant $b={c}$", "V2 dose-matched constant b = {c}", "V2"),
    ("mg_oracle", "V2 oracle: GT pose residual $\\to b_t$ (same map)", "V2 oracle: GT pose residual -> b_t (same map)", "V2"),
    ("mg_conf", "V2 mem gate keyed by the model's OWN self-view confidence (untrained)",
     "V2 mem gate keyed by the model's OWN self-view confidence (untrained)", "V2"),
    ("mg_head_rel", "V2 mem gate, REL target: PoseSigmaHead on the consecutive-frame residual",
     "V2 mem gate, REL target: PoseSigmaHead on the consecutive-frame residual", "V2 rel"),
    ("mg_const_rel", "V2 rel dose-matched constant $b={c}$", "V2 rel dose-matched constant b = {c}", "V2 rel"),
    ("mg_oracle_rel", "V2 rel oracle: GT consecutive-frame pose residual $\\to b_t$",
     "V2 rel oracle: GT consecutive-frame pose residual -> b_t", "V2 rel"),
    # Variant C (WRITE_GATE_VARIANTS.md, "the CONSEQUENCE-trained gate"): the head outputs the weight
    # itself and is trained by the gradient of a causal ATE/RPE loss through the memory (train_wgate_consequence.py);
    # cg1 = state write (frame_gate), cg2 = pose-memory write (mem_gate); _lr5 = the lr 1e-5 arm.
    # cg1_const / cg2_const are the dose-matched constants of the two lr 1e-4 head arms
    # (wgate_make_controls.py --fg_arm cg1_head --mg_arm cg2_head --fg_const_name cg1_const --mg_const_name cg2_const).
    ("cg1_head", "C1 state gate, consequence-trained (lr $10^{-4}$)", "C1 state gate, consequence-trained (lr 1e-4)", "C"),
    ("cg2_head", "C2 pose-memory gate, consequence-trained (lr $10^{-4}$)",
     "C2 pose-memory gate, consequence-trained (lr 1e-4)", "C"),
    ("cg1_head_lr5", "C1 state gate, consequence-trained (lr $10^{-5}$)", "C1 state gate, consequence-trained (lr 1e-5)", "C"),
    ("cg2_head_lr5", "C2 pose-memory gate, consequence-trained (lr $10^{-5}$)",
     "C2 pose-memory gate, consequence-trained (lr 1e-5)", "C"),
    ("cg1_const", "C1 dose-matched constant $a={c}$", "C1 dose-matched constant a = {c}", "C"),
    ("cg2_const", "C2 dose-matched constant $b={c}$", "C2 dose-matched constant b = {c}", "C"),
]
# Groups that get a printed heading row (md + tex) above their first row; the others only get a
# blank line / \midrule, as before.
GROUP_HEADINGS = {"C": "Variant C (consequence-trained)"}
NAMES = [r[0] for r in ROWS]   # arm dir names, in table order


def const_of(pilot_dir, arm):
    """The constant a dose-matched arm ran with (from the control.json the sbatch copied), or None."""
    try:
        c = json.load(open(os.path.join(pilot_dir, arm, "control.json"))).get("*", {})
        fg = c.get("frame_gate") or {}
        if fg.get("const") is not None:
            return float(fg["const"])
        if c.get("alpha_all") is not None:
            return float(c["alpha_all"])
        mg = c.get("mem_gate") or {}
        if mg.get("const") is not None:
            return float(mg["const"])
    except (OSError, ValueError):
        pass
    return None


def oracle_unscored(pilot_dir, arm, scenes):
    """Scenes of the list that ran PLAIN under a per-scene (oracle) arm: those absent from the arm's
    copied control.json when it has per-scene keys and no "*" entry, plus the generator's
    eval_pipeline/maks_arm_<arm>.skipped.txt sidecar. Empty for "*" (constant / head) arms."""
    out = set()
    try:
        c = json.load(open(os.path.join(pilot_dir, arm, "control.json")))
        if isinstance(c, dict) and c and "*" not in c:
            out |= {s for s in scenes if s not in c}
    except (OSError, ValueError):
        pass
    sk = os.path.join(EP, f"maks_arm_{arm}.skipped.txt")
    if os.path.isfile(sk):
        out |= {l.strip() for l in open(sk) if l.strip()}
    return out


def fmt_val(m, v):
    return f"{v:.3f}" if m == "rpe_rot" else f"{v:.4f}"


def fmt_delta(m, d):
    mean, lo, hi, wins, n = d
    star = "*" if (lo > 0 or hi < 0) else ""
    return (f"{mean:+.3f}" if m == "rpe_rot" else f"{mean:+.4f}") + star


def esc(t):
    return t.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene_list", default=os.path.join(EP, "maks_subset430.txt"), help="430 or 4292 scene list")
    ap.add_argument("--pilot_dir", default=f"{OUT}/wgate430", help="root with <arm>/eval per arm")
    ap.add_argument("--base_eval", default=f"{OUT}/{BASE}/eval", help="plain run's eval tree (baseline)")
    ap.add_argument("--subset", default="", help="name in the output files (default: number of listed scenes)")
    ap.add_argument("--table_dir", default=f"{OUT}/wgate/table")
    ap.add_argument("--ref", action="append", default=[], help="NAME=EVALDIR[=LABEL]: extra reference rows")
    ap.add_argument("--min_frac", type=float, default=0.999, help="skip arms scored on fewer scenes than this fraction")
    ap.add_argument("--allow_partial", action="store_true", help="keep incomplete arms (marked with a dagger)")
    ap.add_argument("--no_pdf", action="store_true")
    args = ap.parse_args()

    scenes = [l.strip() for l in open(args.scene_list) if l.strip()]
    subset = args.subset or str(len(scenes))
    os.makedirs(args.table_dir, exist_ok=True)
    stem = os.path.join(args.table_dir, f"wgate_{subset}")
    base = load_label(args.base_eval, scenes)
    print(f"baseline {BASE}: {len(base)}/{len(scenes)} scenes")

    # ---- pass 1: load + admit every arm (nothing is aggregated yet)
    loaded = []   # (arm, tex label, md label, group, data{scene: metrics}, partial)
    skipped = []  # (arm, reason) -> json + footnote

    def add(arm, label, mdlabel, group, eval_dir):
        if not os.path.isdir(eval_dir):
            skipped.append((arm, "missing"))
            print(f"  missing {arm}: {eval_dir}")
            return
        data = load_label(eval_dir, scenes)
        unscored = oracle_unscored(args.pilot_dir, arm, scenes) if group != "ref" else set()
        n_plain = sum(1 for s in unscored if s in data)
        for s in unscored:
            data.pop(s, None)
        if n_plain:
            print(f"  {arm}: {n_plain} scene(s) ran plain (no control entry) -> treated as unscored")
        n_ok = sum(1 for s in scenes if s in data and s in base)
        partial = n_ok < args.min_frac * len(scenes)
        if partial and not args.allow_partial:
            skipped.append((arm, f"{n_ok}/{len(scenes)} scored" + (f", {n_plain} ran plain" if n_plain else "")))
            print(f"  skip {arm}: {n_ok}/{len(scenes)} scenes scored (use --allow_partial)")
            return
        loaded.append((arm, label, mdlabel, group, data, partial, n_plain))

    for arm, label, mdlabel, group in ROWS:
        c = const_of(args.pilot_dir, arm)
        cs = f"{c:.3f}" if c is not None else "?"
        add(arm, label.replace("{c}", cs), mdlabel.replace("{c}", cs), group, os.path.join(args.pilot_dir, arm, "eval"))
    for r in args.ref:
        parts = r.split("=", 2)
        name, d = parts[0], parts[1]
        lab = parts[2] if len(parts) > 2 else name
        add(name, esc(lab), lab, "ref", d)

    # ---- the common scene set: listed AND in the baseline AND in every complete arm
    common = [s for s in scenes if s in base and all(s in data for _, _, _, _, data, partial, _ in loaded if not partial)]
    dropped = [s for s in scenes if s not in common]
    print(f"common scene set: {len(common)}/{len(scenes)} ({len(dropped)} dropped from all rows)")

    # ---- pass 2: aggregate every row on the common set (partial rows on their scored part of it)
    rows = []  # (arm, tex label, md label, group, n, partial, means, deltas{m: (mean, lo, hi, wins, n)})
    rows.append((BASE, "CUT3R finetuned (plain, augfull\\_lr1e5)", f"CUT3R finetuned (plain, {BASE})", "base",
                 len(common), False, {m: float(np.nanmean([base[s][m] for s in common])) for m in METRICS}, None))
    for arm, label, mdlabel, group, data, partial, n_plain in loaded:
        sc = [s for s in common if s in data]
        if not sc:
            skipped.append((arm, "0 scenes in the common set"))
            print(f"  skip {arm}: no scenes in the common set")
            continue
        means = {m: float(np.nanmean([data[s][m] for s in sc])) for m in METRICS}
        deltas = {}
        for m in METRICS:
            d = np.array([data[s][m] - base[s][m] for s in sc], float)
            d = d[np.isfinite(d)]
            if len(d) == 0:
                deltas[m] = (np.nan, np.nan, np.nan, 0, 0)
                continue
            lo, hi = boot_ci(d, n=10000)
            wins = int((d > 0).sum() if m in HIGHER else (d < 0).sum())
            deltas[m] = (float(d.mean()), lo, hi, wins, len(d))
        rows.append((arm, label, mdlabel, group, len(sc), partial, means, deltas))
        print(f"  {arm}: {len(sc)} scenes" + ("  (partial)" if partial else ""))

    note_skipped = "; ".join(f"{a} ({r})" for a, r in skipped)
    note_dropped = (f"{len(dropped)} listed scene(s) dropped from all rows (unscored in the baseline or a complete arm)"
                    if dropped else "")

    # ---- csv
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "n", "partial"] + METRICS + [f"d_{m}" for m in METRICS] + [f"lo_{m}" for m in METRICS]
                   + [f"hi_{m}" for m in METRICS] + [f"wins_{m}" for m in METRICS])
        for arm, label, mdlabel, group, n, partial, means, deltas in rows:
            rec = [arm, n, int(partial)] + [f"{means[m]:.6f}" for m in METRICS]
            if deltas:
                rec += [f"{deltas[m][0]:+.6f}" for m in METRICS] + [f"{deltas[m][1]:+.6f}" for m in METRICS] \
                    + [f"{deltas[m][2]:+.6f}" for m in METRICS] + [f"{deltas[m][3]}/{deltas[m][4]}" for m in METRICS]
            w.writerow(rec)

    # ---- md
    md = [f"# Write-gate variants, {subset} scenes, paired vs {BASE}", "",
          "Cell: mean (paired delta, per-scene wins/n); * = 95% scene-bootstrap CI excludes 0; dagger = incomplete arm.", "",
          "| method | n | AbsRel | d<1.25 | ATE | RPE trans | RPE rot |", "|---|---|---|---|---|---|---|"]
    last_group = None
    for arm, label, mdlabel, group, n, partial, means, deltas in rows:
        if last_group is not None and group != last_group:
            md.append("| | | | | | | |")
        if group != last_group and group in GROUP_HEADINGS:
            md.append(f"| **{GROUP_HEADINGS[group]}** | | | | | | |")
        last_group = group
        cells = []
        for m in METRICS:
            v = fmt_val(m, means[m])
            cells.append(v if deltas is None else f"{v} ({fmt_delta(m, deltas[m])}, {deltas[m][3]}/{deltas[m][4]})")
        md.append(f"| {mdlabel}{' (dagger)' if partial else ''} | {n} | " + " | ".join(cells) + " |")
    md.append("")
    md.append(f"Common scene set: {len(common)}/{len(scenes)} listed scenes" + (f"; {note_dropped}" if note_dropped else "") + ".")
    if note_skipped:
        md.append(f"Rows not shown: {note_skipped}.")
    open(stem + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- tex (standalone, booktabs); best / 2nd / 3rd shading per metric over all rows
    ranks = {}
    for m in METRICS:
        vals = sorted([(r[6][m], r[0]) for r in rows], reverse=(m in HIGHER))
        ranks[m] = {arm: i for i, (v, arm) in enumerate(vals)}
    COL = ["red!30", "orange!30", "yellow!40"]
    tex = [r"\documentclass{standalone}", r"\usepackage{booktabs,xcolor,colortbl,amsmath,amssymb}", r"\begin{document}",
           r"\small\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{l r r r r r r}", r"\toprule",
           r"Method & $n$ & AbsRel $\downarrow$ & $\delta<1.25$ $\uparrow$ & ATE $\downarrow$ & "
           r"RPE$_\text{trans}$ $\downarrow$ & RPE$_\text{rot}$ $\downarrow$ \\", r"\midrule"]
    last_group = None
    for arm, label, mdlabel, group, n, partial, means, deltas in rows:
        if last_group is not None and group != last_group:
            tex.append(r"\midrule")
        if group != last_group and group in GROUP_HEADINGS:
            tex.append(r"\multicolumn{7}{l}{\textit{" + esc(GROUP_HEADINGS[group]) + r"}} \\")
        last_group = group
        cells = []
        for m in METRICS:
            v = fmt_val(m, means[m])
            cell = v if deltas is None else f"{v} \\scriptsize({fmt_delta(m, deltas[m])}, {deltas[m][3]}/{deltas[m][4]})"
            rk = ranks[m][arm]
            if rk < 3:
                cell = f"\\cellcolor{{{COL[rk]}}}" + cell
            cells.append(cell)
        tex.append(label + (r"$^\dagger$" if partial else "") + f" & {n} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule",
            r"\multicolumn{7}{l}{\scriptsize " + esc(f"{subset} DROID wrist test scenes; parentheses: paired delta vs {BASE}, per-scene wins; ")
            + r"* = 95\% scene-bootstrap CI excludes 0; colour = best / 2nd / 3rd; $\dagger$ = incomplete arm.} \\"]
    foot = f"common scene set {len(common)}/{len(scenes)}" + (f"; {note_dropped}" if note_dropped else "") \
        + (f"; rows not shown: {note_skipped}" if note_skipped else "")
    tex += [r"\multicolumn{7}{l}{\scriptsize " + esc(foot) + r".} \\",
            r"\end{tabular}", r"\end{document}"]
    open(stem + ".tex", "w").write("\n".join(tex) + "\n")
    json.dump({"subset": subset, "scene_list": args.scene_list, "pilot_dir": args.pilot_dir,
               "n_listed": len(scenes), "common_n": len(common), "dropped_scenes": dropped,
               "skipped": [{"arm": a, "reason": r} for a, r in skipped],
               "rows": [{"arm": a, "label": ml, "group": g, "n": n, "partial": p, "means": mu,
                         "deltas": {k: list(v) for k, v in d.items()} if d else None}
                        for a, l, ml, g, n, p, mu, d in rows]}, open(stem + ".json", "w"), indent=1)
    if args.no_pdf:
        return
    env = dict(os.environ)
    if os.path.isdir(LATEX_BIN):
        env["PATH"] = LATEX_BIN + os.pathsep + env.get("PATH", "")
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", os.path.basename(stem) + ".tex"],
                       cwd=args.table_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300, env=env)
        # ghostscript, as the other tables (ImageMagick is not on the compute nodes)
        subprocess.run(["gs", "-sDEVICE=png16m", "-r300", "-o", stem + ".png", "-dBATCH", "-dNOPAUSE", stem + ".pdf"],
                       check=True, timeout=300, stdout=subprocess.DEVNULL, env=env)
        print("wrote", stem + ".png")
    except Exception as e:
        print("pdf/png build failed:", e, "-- md/tex/csv/json are written")


if __name__ == "__main__":
    main()
