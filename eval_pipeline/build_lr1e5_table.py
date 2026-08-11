#!/usr/bin/env python3
"""6-column LaTeX table from aggregate_results.py output, with top-3 cell shading.

    Method | AbsRel | delta<1.25 | ATE | RPE_trans | RPE_rot(deg)

Nothing in the repo did this: build_sim3_table.py emits 7 columns and needs
pose_sim3_both.py's CSVs, while build_gru_overfit_table.py / _teal.py are
6-column but read a SINGLE scene's raw eval CSV. This one reads the 4292-scene
means that aggregate_results.py already computed.

Source of truth: <out_root>/summary/averages_table.csv, whose columns
`setup,n_scenes,absrel,a1,ate,rpe_trans,rpe_rot` (aggregate_results.py:118) are
exactly the six needed. `--from_per_scene` instead re-reduces
summary/per_scene_<label>.csv with the same finite-mean aggregate_results.py
uses (isfinite filter + mean, aggregate_results.py:74-76) as a cross-check.

Shading (build_gru_overfit_table.py:53 convention): best = red!30,
2nd = orange!30, 3rd = yellow!40, ranked over DISTINCT displayed values so
ties share a colour. a1 is higher-better; every other metric is lower-better.

Usage:
  python eval_pipeline/build_lr1e5_table.py \
      --out_root /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval \
      --row augfull_lr1e5="CUT3R finetune (no ray conditioning)" \
      --row gtray_lr1e5="+ GT ray-map conditioning" \
      --row prevpred_lr1e5="+ previous-frame predicted rays" \
      --out_tex .../tables/lr1e5_3arm.tex [--compile]
"""
import argparse
import csv
import os
import subprocess
import sys

METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
HIGHER = {"a1"}                       # every other metric is lower-better
COLORS = [r"\cellcolor{red!30}", r"\cellcolor{orange!30}", r"\cellcolor{yellow!40}"]
DECIMALS = 4


def tex_escape(s):
    """Aggregate labels carry underscores; raw they break pdflatex."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def read_averages(out_root, labels):
    """Read the setup rows of summary/averages_table.csv."""
    path = os.path.join(out_root, "summary", "averages_table.csv")
    if not os.path.isfile(path):
        sys.exit(f"ERROR: {path} not found -- run aggregate_results.py first.")
    rows = {r["setup"]: r for r in csv.DictReader(open(path))}
    out = {}
    for lb in labels:
        if lb not in rows:
            sys.exit(f"ERROR: label {lb!r} not in {path} (has: {sorted(rows)})")
        r = rows[lb]
        out[lb] = {"n_scenes": int(float(r["n_scenes"])),
                   **{m: float(r[m]) for m in METRICS},
                   **{f"n_valid_{m}": int(float(r.get(f"n_valid_{m}", 0))) for m in METRICS}}
    return out


def read_per_scene(out_root, labels):
    """Independent path: re-reduce per_scene_<label>.csv exactly as
    aggregate_results.py:74-76 does (isfinite filter, then mean)."""
    import math
    out = {}
    for lb in labels:
        path = os.path.join(out_root, "summary", f"per_scene_{lb}.csv")
        if not os.path.isfile(path):
            sys.exit(f"ERROR: {path} not found.")
        rows = list(csv.DictReader(open(path)))
        d = {"n_scenes": len(rows)}
        for m in METRICS:
            vals = []
            for r in rows:
                try:
                    v = float(r[m])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    vals.append(v)
            d[m] = sum(vals) / len(vals) if vals else float("nan")
            d[f"n_valid_{m}"] = len(vals)
        out[lb] = d
    return out


def rank_colors(values):
    """values: list of displayed-string floats. Returns index -> color prefix."""
    fin = [(v, i) for i, v in enumerate(values) if v == v]  # drop nan
    if not fin:
        return {}
    distinct = sorted({v for v, _ in fin})
    order = distinct if not RANK_HIGHER else sorted(distinct, reverse=True)
    rank_of = {v: k for k, v in enumerate(order)}
    return {i: COLORS[rank_of[v]] for v, i in fin if rank_of[v] < len(COLORS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True,
                    help="cut3r_eval root (the dir holding summary/)")
    ap.add_argument("--row", action="append", required=True, metavar="LABEL=NAME",
                    help="repeatable, in table order; NAME may be omitted")
    ap.add_argument("--out_tex", required=True)
    ap.add_argument("--title", default="CUT3R conditioning arms on DROID "
                                       "(lr $10^{-5}$, 4292 test scenes)")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--from_per_scene", action="store_true",
                    help="re-reduce per_scene_<label>.csv instead of reading "
                         "averages_table.csv (cross-check; must agree)")
    ap.add_argument("--expect_scenes", type=int, default=4292,
                    help="fail unless every row aggregated this many scenes; 0 disables")
    ap.add_argument("--compile", action="store_true", help="run pdflatex + gs")
    args = ap.parse_args()

    labels, names = [], {}
    for spec in args.row:
        lb, _, nm = spec.partition("=")
        labels.append(lb)
        names[lb] = nm or lb

    data = (read_per_scene if args.from_per_scene else read_averages)(args.out_root, labels)

    # Refuse to render a silently-truncated table: aggregate_results.py skips
    # scenes with no CSV and still prints a table (aggregate_results.py:54-55).
    if args.expect_scenes:
        bad = {lb: d["n_scenes"] for lb, d in data.items()
               if d["n_scenes"] != args.expect_scenes}
        if bad:
            sys.exit(f"ERROR: expected {args.expect_scenes} scenes per label, got {bad}. "
                     f"Reconcile claims/failures before publishing (or pass --expect_scenes 0).")

    # Per-column top-3 shading over distinct displayed (rounded) values.
    global RANK_HIGHER
    shade = {}
    for m in METRICS:
        RANK_HIGHER = m in HIGHER
        disp = [round(data[lb][m], DECIMALS) for lb in labels]
        shade[m] = rank_colors(disp)

    n_all = sorted({d["n_scenes"] for d in data.values()})
    sub = args.subtitle or (
        f"sim3-aligned, RMSE-reduced pose; median-scaled depth; "
        f"{n_all[0] if len(n_all) == 1 else n_all} scenes per arm")

    L = []
    L.append(r"\documentclass[border=8pt,varwidth=25cm]{standalone}")
    L.append(r"\usepackage{booktabs}")
    L.append(r"\usepackage{amssymb}")
    L.append(r"\usepackage[T1]{fontenc}")
    L.append(r"\usepackage[table]{xcolor}")
    L.append(r"\usepackage{newtxtext,newtxmath} % Times New Roman styling")
    L.append("")
    L.append(r"\begin{document}")
    L.append(r"\begin{center}")
    L.append(r"{\normalsize\bfseries " + args.title + r"}\par\vspace{4pt}")
    L.append(r"{\footnotesize " + tex_escape(sub) + r"}\par\vspace{8pt}")
    L.append(r"\footnotesize")
    L.append(r"\setlength{\tabcolsep}{8pt}")
    L.append(r"\renewcommand{\arraystretch}{1.15}")
    L.append(r"\begin{tabular}{lccccc}")
    L.append(r"\toprule")
    L.append(r" & \multicolumn{2}{c}{Depth} & \multicolumn{3}{c}{Pose} \\")
    L.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}")
    L.append(r"Method & AbsRel $\downarrow$ & $\delta < 1.25$ $\uparrow$ & "
             r"ATE $\downarrow$ & RPE$_{\mathrm{trans}}$ $\downarrow$ & "
             r"RPE$_{\mathrm{rot}}$ ($^{\circ}$) $\downarrow$ \\")
    L.append(r"\midrule")
    for i, lb in enumerate(labels):
        cells = [tex_escape(names[lb])]
        for m in METRICS:
            v = data[lb][m]
            cells.append(shade[m].get(i, "") + f"{v:.{DECIMALS}f}")
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{center}")
    L.append(r"\end{document}")
    tex = "\n".join(L) + "\n"

    os.makedirs(os.path.dirname(os.path.abspath(args.out_tex)), exist_ok=True)
    with open(args.out_tex, "w") as fh:
        fh.write(tex)
    print(tex)
    print(f"wrote {args.out_tex}")

    if args.compile:
        # /usr/bin/pdflatex has no standalone.cls; the TeXLive tree under
        # /apps/GPP/LATEX does. build_gru_overfit_table.py:267-274 reaches it via
        # `module load latex/20240430`, but lmod is not initialised on the login
        # nodes (/etc/profile.d/lmod.sh is absent), so put the tree's bin on PATH
        # directly. MN5 has no ghostscript module; /usr/bin/gs is used for the png.
        outdir = os.path.dirname(os.path.abspath(args.out_tex))
        stem = os.path.splitext(os.path.basename(args.out_tex))[0]
        texbin = "/apps/GPP/LATEX/20240430/bin/x86_64-linux"
        cmd = (
            f"export PATH={texbin}:$PATH && "
            f"cd {outdir} && pdflatex -interaction=nonstopmode {stem}.tex >/dev/null && "
            f"gs -sDEVICE=png16m -r300 -o {stem}.png -dBATCH -dNOPAUSE {stem}.pdf >/dev/null"
        )
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"compile failed:\n{r.stderr[-2000:]}\n{r.stdout[-1000:]}", file=sys.stderr)
            return
        print(f"wrote {os.path.join(outdir, stem + '.pdf')}")
        print(f"wrote {os.path.join(outdir, stem + '.png')}")


RANK_HIGHER = False

if __name__ == "__main__":
    main()
