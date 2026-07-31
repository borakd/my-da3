#!/usr/bin/env python3
"""Rebuild the "PoseGRU grid vs. prev-frame predicted rays" single-test-scene table
from EVERY PoseGRU overfit run that has a checkpoint, and cross-reference the fresh
numbers against the previously published ones.

Successor to build_gru_vs_prevpred_teal.py: same table, same highlight scheme
(red!30 best / orange!30 2nd / yellow!40 3rd, ties share a colour), but the arm list
is DISCOVERED from the eval root instead of hard-coded, so the v3 R/F arms (and any
later a5 arm) appear automatically.

  --grid    eval root holding <label>/eval/<scene>/eval_depth_pose_metrics.csv
            (default: the fresh gru_grid_final sweep)
  --legacy  older eval root used only for cross-referencing (default: gru_grid)
  --prefer  which source wins for arms present in both (default: legacy, so the
            already-published rows keep their exact values and only genuinely new
            arms are added)

Prints a cross-reference table of fresh-vs-legacy-vs-published values first, then
writes the .tex and compiles PDF + PNG.

Usage: python build_gru_overfit_table.py [--dry-run]
"""
import argparse
import csv
import os
import re
import shutil
import subprocess

ROOT = "/scratch/bdursun25/cuteanything/outputs/cut3r_eval/overfit_test_scene"
SCENE = "RAIL+eh61f232+2023-10-26-17h-33m-59s/13062452+wrist"
PREV_PRED_CSV = ("/scratch/bdursun25/cuteanything/outputs/cut3r_eval/"
                 "demo_ray_smoketest/prev_pred/eval_depth_pose_metrics.csv")
TEX = os.path.join(ROOT, "tables", "cut3r_droid_single_test_scene_sim3_gru_vs_prevpred.tex")

METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
HIGHER = {"a1"}
COLORS = [r"\cellcolor{red!30}", r"\cellcolor{orange!30}", r"\cellcolor{yellow!40}"]

A_TEXT = {
    "a1": "direct, pose",
    "a2": "residual, pose",
    "a3": "direct, pose+delta",
    "a4": "residual, pose+delta",
    "a5": "split-anchor, pose+delta",
}
CELL_RE = re.compile(r"(?:\\cellcolor\{[^}]*\})?\s*([0-9]*\.?[0-9]+)\s*$")


def parse_label(label):
    """'v2_a1_g0' / 'gru_a1_g0' / 'v3_a4_g1_f1_r8' -> (sort key, display name, group).

    The display name follows the convention already in the table: 'PoseGRU A<N> G<M>
    (<head algebra>, <gru input>)', with the v3 levers appended as they appear in the
    run name (F1 = two-frame image features into the GRU, R<K> = K refinement iterations).
    """
    m = re.search(r"(a[1-5])_g([0-2])", label)
    if not m:
        return None
    a, g = m.group(1), m.group(2)
    f1 = "_f1" in label
    r = re.search(r"_r(\d+)(?:_|$)", label)
    bits = [f"PoseGRU {a.upper()} G{g}"]
    if f1:
        bits.append("F1")
    if r:
        bits.append(f"R{r.group(1)}")
    detail = [A_TEXT[a]]
    if r:
        detail.append(f"{r.group(1)} iters")
    if f1:
        detail.append("img feats")
    name = " ".join(bits) + " (" + ", ".join(detail) + ")"
    # Order: the v2 grid by G then A (as published), then the v3 arms.
    ver = 1 if (f1 or r) else 0
    return (ver, int(g), int(a[1]), int(f1), int(r.group(1)) if r else 0), name, f"g{g}_v{ver}"


def mean_row(csv_path):
    with open(csv_path) as f:
        row = next(r for r in csv.DictReader(f)
                   if r["camera_id"] == "ALL" and r["local_timestep"] == "MEAN")
    return [round(float(row[m]), 4) for m in METRICS]


def collect(grid):
    """label -> (sortkey, name, group, values, epochs) for every arm with an eval CSV."""
    out = {}
    if not os.path.isdir(grid):
        return out
    for label in sorted(os.listdir(grid)):
        d = os.path.join(grid, label)
        csv_path = os.path.join(d, "eval", SCENE, "eval_depth_pose_metrics.csv")
        if not os.path.isdir(d) or not os.path.isfile(csv_path) or not os.path.getsize(csv_path):
            continue
        p = parse_label(label)
        if p is None:
            continue
        # The "[N ep]" row note must be READ, never assumed: an eval run started
        # from a keep_freq snapshot instead of checkpoint-final would otherwise be
        # published as a 50-epoch arm. eval_gru_overfit_all.sbatch always writes
        # epochs.txt; older label dirs were backfilled from their training log.
        ep_file = os.path.join(d, "epochs.txt")
        if os.path.isfile(ep_file):
            epochs = open(ep_file).read().strip()
        else:
            epochs = "50"
            print(f"WARNING: {label} has no epochs.txt — labelling it '[50 ep]' "
                  f"UNVERIFIED. Write {ep_file} with the real epoch count.")
        key, name, group = p
        # 'gru_a1_g0_ep40'-style intermediate evals are keyed apart by their epoch.
        out[(key, epochs)] = (key, name, group, mean_row(csv_path), epochs, label)
    return out


def published_rows(tex_path):
    """name -> values, read back from the currently published .tex."""
    out = {}
    if not os.path.isfile(tex_path):
        return out
    for ln in open(tex_path):
        if "&" not in ln or r"\\" not in ln or "multicolumn" in ln or "cmidrule" in ln:
            continue
        parts = ln.split("&")
        if len(parts) != 6:
            continue
        name = parts[0].strip()
        vals = []
        for p in parts[1:]:
            m = CELL_RE.match(p.replace(r"\\", "").strip())
            if not m:
                vals = None
                break
            vals.append(float(m.group(1)))
        if vals and name and "Method" not in name:
            out[name] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=os.path.join(ROOT, "gru_grid_final"))
    ap.add_argument("--legacy", default=os.path.join(ROOT, "gru_grid"))
    ap.add_argument("--prefer", choices=["legacy", "fresh"], default="legacy")
    ap.add_argument("--tex", default=TEX, help="output .tex (default: the published table)")
    ap.add_argument("--baseline_csv", default=PREV_PRED_CSV)
    ap.add_argument("--baseline_name", default="Prev-frame predicted rays (overfit)")
    ap.add_argument("--subtitle", default="", help="italic line under the table title")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tex_path = args.tex

    fresh = collect(args.grid)
    legacy = collect(args.legacy)
    pub = published_rows(tex_path)

    # ---- cross-reference -----------------------------------------------------
    print(f"{'arm':52s} {'source':7s} " + " ".join(f"{m:>10s}" for m in METRICS))
    print("-" * 118)
    all_keys = sorted(set(fresh) | set(legacy), key=lambda k: (k[0], k[1]))
    n_diff = 0
    for k in all_keys:
        rec = fresh.get(k) or legacy[k]
        name = f"{rec[1]} [{rec[4]} ep]"
        for src, tbl in (("fresh", fresh), ("legacy", legacy)):
            if k in tbl:
                print(f"{name[:52]:52s} {src:7s} " +
                      " ".join(f"{v:10.4f}" for v in tbl[k][3]))
        if name in pub:
            print(f"{'':52s} {'pubtex':7s} " + " ".join(f"{v:10.4f}" for v in pub[name]))
        if k in fresh and k in legacy and fresh[k][3] != legacy[k][3]:
            n_diff += 1
            d = [f"{a - b:+.4f}" for a, b in zip(fresh[k][3], legacy[k][3])]
            print(f"{'':52s} {'DELTA':7s} " + " ".join(f"{x:>10s}" for x in d))
        if name in pub and rec[3] != [round(v, 4) for v in pub[name]]:
            print(f"{'':52s} {'!! published row differs from the chosen source':7s}")
    print(f"\ncross-reference: {len(fresh)} fresh arms, {len(legacy)} legacy arms, "
          f"{len(pub)} published rows, {n_diff} fresh-vs-legacy value differences\n")

    # ---- assemble the table --------------------------------------------------
    chosen = {}
    for k in all_keys:
        primary, secondary = ((legacy, fresh) if args.prefer == "legacy" else (fresh, legacy))
        chosen[k] = primary.get(k) or secondary[k]
    # Only full-length (final-checkpoint) rows go in the table; the older
    # intermediate-epoch evals are cross-referenced above but not published.
    best_ep = {}
    for k, rec in chosen.items():
        best_ep[k[0]] = max(best_ep.get(k[0], 0), int(rec[4]))
    rows = [(args.baseline_name, mean_row(args.baseline_csv), "base")]
    for k in all_keys:
        rec = chosen[k]
        if int(rec[4]) != best_ep[k[0]]:
            continue
        rows.append((f"{rec[1]} [{rec[4]} ep]", rec[3], rec[2]))

    col_colors = []
    for c, m in enumerate(METRICS):
        disp = sorted({v[c] for _, v, _ in rows}, reverse=m in HIGHER)
        col_colors.append({v: COLORS[r] for r, v in enumerate(disp[:3])})

    lines = []
    prev_group = None
    for name, vals, group in rows:
        if prev_group is not None and group != prev_group:
            lines.append(r"\midrule")
        prev_group = group
        cells = " & ".join(f"{col_colors[c].get(v, '')}{v:.4f}" for c, v in enumerate(vals))
        lines.append(f"{name} & {cells} \\\\")

    tex = r"""\documentclass[border=8pt,varwidth=25cm]{standalone}
\usepackage{booktabs}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\usepackage[table]{xcolor}
\usepackage{newtxtext,newtxmath} % Times New Roman styling

\begin{document}
\begin{center}
{\normalsize\bfseries PoseGRU grid vs.\ prev-frame predicted rays (single test scene)}\par""" + (
        "\\vspace{3pt}\n{\\footnotesize\\itshape " + args.subtitle + "}\\par" if args.subtitle else ""
    ) + r"""\vspace{8pt}
\footnotesize
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{lccccc}
\toprule
 & \multicolumn{2}{c}{Depth} & \multicolumn{3}{c}{Pose} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-6}
Method & AbsRel $\downarrow$ & $\delta < 1.25$ $\uparrow$ & ATE $\downarrow$ & RPE$_{\mathrm{trans}}$ $\downarrow$ & RPE$_{\mathrm{rot}}$ ($^{\circ}$) $\downarrow$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{center}
\end{document}
"""
    if args.dry_run:
        print(tex)
        return

    tables_dir = os.path.dirname(tex_path)
    base = os.path.splitext(os.path.basename(tex_path))[0]
    for ext in (".tex", ".pdf", ".png"):
        src = os.path.join(tables_dir, base + ext)
        if os.path.isfile(src) and not os.path.isfile(src + ".bak_pre_v3arms"):
            shutil.copy(src, src + ".bak_pre_v3arms")
    with open(tex_path, "w") as f:
        f.write(tex)

    cmd = (
        "source /etc/profile.d/lmod.sh && module load latex/2025 ghostscript && "
        "export TEXMFROOT=/opt/ohpc/pub/apps/latex/2025 && "
        "export TEXMFCNF=$TEXMFROOT:$TEXMFROOT/texmf-dist/web2c && "
        f"cd {tables_dir} && pdflatex -interaction=nonstopmode {base}.tex >/dev/null && "
        f"gs -sDEVICE=png16m -r300 -o {base}.png -dBATCH -dNOPAUSE {base}.pdf"
    )
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"compile failed: {r.stderr[-1500:]}\n{r.stdout[-500:]}")
    print(f"OK: {len(rows)} rows -> {tex_path} + .pdf + .png")


if __name__ == "__main__":
    main()
