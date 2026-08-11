#!/usr/bin/env python3
"""Build the simplified PoseGRU-vs-prev_pred single-test-scene table with
top-3 per-column highlights (red!30 / orange!30 / yellow!40, best first —
the same scheme as append_gru_grid_to_sim3_table.py; ties share a color,
all other cells unshaded), reading the MEAN rows straight from the eval
CSVs, then compile PDF + PNG. (Filename kept from the earlier teal-gradient
variant of this table.)

Usage: python build_gru_vs_prevpred_teal.py
"""
import csv
import os
import subprocess

ROOT = os.environ.get(
    "OVERFIT_EVAL_ROOT",
    os.path.join(
        os.environ.get("OUT_ROOT", "/gpfs/projects/etur59/koc821022/outputs"),
        "cut3r_eval", "overfit_test_scene",
    ),
)
GRID = os.path.join(ROOT, "gru_grid")
# MN5 drops the trailing camera component; override if evaluating elsewhere.
SCENE = os.environ.get("OVERFIT_SCENE", "RAIL+eh61f232+2023-10-26-17h-33m-59s")
# The published baseline row comes from a Jul-10 smoketest pass that predates
# every surviving prev_pred checkpoint -- it cannot be regenerated, only located.
PREV_PRED_CSV = os.environ.get(
    "PREV_PRED_CSV",
    os.path.join(
        os.environ.get("OUT_ROOT", "/gpfs/projects/etur59/koc821022/outputs"),
        "cut3r_eval", "demo_ray_smoketest", "prev_pred",
        "eval_depth_pose_metrics.csv",
    ),
)
TEX = os.path.join(ROOT, "tables", "cut3r_droid_single_test_scene_sim3_gru_vs_prevpred.tex")
METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
HIGHER = {"a1"}

# Arms with no eval CSV yet are skipped; \midrule separates baseline/G0/G1/G2.
ARMS = [
    ("gru_a1_g0", "PoseGRU A1 G0 (direct, pose)"),
    ("gru_a2_g0", "PoseGRU A2 G0 (residual, pose)"),
    ("gru_a3_g0", "PoseGRU A3 G0 (direct, pose+delta)"),
    ("gru_a4_g0", "PoseGRU A4 G0 (residual, pose+delta)"),
    ("gru_a1_g1", "PoseGRU A1 G1 (direct, pose)"),
    ("gru_a2_g1", "PoseGRU A2 G1 (residual, pose)"),
    ("gru_a3_g1", "PoseGRU A3 G1 (direct, pose+delta)"),
    ("gru_a4_g1", "PoseGRU A4 G1 (residual, pose+delta)"),
    ("gru_a1_g2", "PoseGRU A1 G2 (direct, pose)"),
    ("gru_a2_g2", "PoseGRU A2 G2 (residual, pose)"),
    ("gru_a3_g2", "PoseGRU A3 G2 (direct, pose+delta)"),
    ("gru_a4_g2", "PoseGRU A4 G2 (residual, pose+delta)"),
]


def mean_row(csv_path):
    with open(csv_path) as f:
        row = next(r for r in csv.DictReader(f)
                   if r["camera_id"] == "ALL" and r["local_timestep"] == "MEAN")
    return [round(float(row[m]), 4) for m in METRICS]


def main():
    rows = [("Prev-frame predicted rays (overfit)", mean_row(PREV_PRED_CSV), "base")]
    for label, name in ARMS:
        csv_path = os.path.join(GRID, label, "eval", SCENE, "eval_depth_pose_metrics.csv")
        if not os.path.isfile(csv_path):
            continue
        ep = open(os.path.join(GRID, label, "epochs.txt")).read().strip()
        rows.append((f"{name} [{ep} ep]", mean_row(csv_path), label[-2:]))

    # Top-3 per-column highlights over the distinct displayed values (same
    # scheme as append_gru_grid_to_sim3_table.py): best = red!30, 2nd =
    # orange!30, 3rd = yellow!40; ties share a color; other cells unshaded.
    COLORS = [r"\cellcolor{red!30}", r"\cellcolor{orange!30}", r"\cellcolor{yellow!40}"]
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
        cells = " & ".join(f"{col_colors[c].get(v, '')}{v:.4f}"
                           for c, v in enumerate(vals))
        lines.append(f"{name} & {cells} \\\\")

    tex = r"""\documentclass[border=8pt,varwidth=25cm]{standalone}
\usepackage{booktabs}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\usepackage[table]{xcolor}
\usepackage{newtxtext,newtxmath} % Times New Roman styling

\begin{document}
\begin{center}
{\normalsize\bfseries PoseGRU grid vs.\ prev-frame predicted rays (single test scene)}\par\vspace{8pt}
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
    os.makedirs(os.path.dirname(TEX) or ".", exist_ok=True)
    with open(TEX, "w") as f:
        f.write(tex)

    tables_dir = os.path.dirname(TEX)
    base = os.path.splitext(os.path.basename(TEX))[0]
    cmd = (
        "source /etc/profile.d/lmod.sh && module load latex/20240430 && "
        # MN5 has no ghostscript module; /usr/bin/gs is used for the `gs` step below.
        "export TEXMFROOT=/apps/GPP/LATEX/20240430 && "
        "export TEXMFCNF=$TEXMFROOT:$TEXMFROOT/texmf-dist/web2c && "
        f"cd {tables_dir} && pdflatex -interaction=nonstopmode {base}.tex >/dev/null && "
        f"gs -sDEVICE=png16m -r300 -o {base}.png -dBATCH -dNOPAUSE {base}.pdf"
    )
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"compile failed: {r.stderr[-1500:]}\n{r.stdout[-500:]}")
    print("OK:", TEX, "+ .pdf + .png")


if __name__ == "__main__":
    main()
