#!/usr/bin/env python3
"""Append the PoseGRU A1-A4 x G0/G1 single-test-scene rows to the sim3 tex table
and recompile it (PDF + PNG).

Reads MEAN rows from outputs/cut3r_eval/overfit_test_scene/gru_grid/<label>/eval/
<scene>/eval_depth_pose_metrics.csv (+ <label>/epochs.txt for the row note),
appends one \\midrule block of 8 rows (A1-A4 under G0, then A1-A4 under G1),
and recomputes the table-wide top-3 per-column highlights (red/orange/yellow)
over ALL rows, matching the table's existing convention. Ranking uses the
displayed 4-decimal values; equal displayed values share a color.

Usage: python append_gru_grid_to_sim3_table.py [--dry-run]
"""
import argparse
import csv
import os
import re
import shutil
import subprocess

ROOT = os.environ.get(
    "OVERFIT_EVAL_ROOT",
    os.path.join(
        os.environ.get("OUT_ROOT", "/gpfs/scratch/etur59/koc821022/outputs"),
        "cut3r_eval", "overfit_test_scene",
    ),
)
GRID = os.path.join(ROOT, "gru_grid")
TEX = os.path.join(ROOT, "tables", "cut3r_droid_single_test_scene_sim3.tex")
# MN5 drops the trailing camera component; override if evaluating elsewhere.
SCENE = os.environ.get("OVERFIT_SCENE", "RAIL+eh61f232+2023-10-26-17h-33m-59s")
OLD_PNG_WIDTH = 1661  # keep the recompiled PNG at the existing render width

# (label_dir, arm text) in required order: A1-A4 under G0, then G1, then G2.
# Arms whose eval CSV does not exist yet are skipped, and the tex is always
# rebuilt from the .bak_pre_gru_grid baseline -> re-running this script as
# more arms finish is safe and idempotent.
NEW_ROWS = [
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
METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
HIGHER_BETTER = {"a1"}
COLORS = [r"\cellcolor{red!30}", r"\cellcolor{orange!30}", r"\cellcolor{yellow!40}"]

CELL_RE = re.compile(r"(?:\\cellcolor\{[^}]*\})?\s*([0-9]*\.?[0-9]+)\s*$")


def read_new_rows():
    rows = []
    for label, name in NEW_ROWS:
        csv_path = os.path.join(GRID, label, "eval", SCENE, "eval_depth_pose_metrics.csv")
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path) as f:
            mean = next(r for r in csv.DictReader(f)
                        if r["camera_id"] == "ALL" and r["local_timestep"] == "MEAN")
        with open(os.path.join(GRID, label, "epochs.txt")) as f:
            epochs = f.read().strip()
        vals = [float(mean[m]) for m in METRICS]
        rows.append((f"{name} [{epochs} ep]", vals))
    return rows


def parse_existing(lines):
    """Yield (idx, name, [5 floats]) for existing data rows (5 '&'-separated values)."""
    out = []
    for i, ln in enumerate(lines):
        if "&" not in ln or r"\\" not in ln or "multicolumn" in ln or "cmidrule" in ln:
            continue
        parts = ln.split("&")
        if len(parts) != 6:
            continue
        name = parts[0].strip()
        vals = []
        ok = True
        for p in parts[1:]:
            m = CELL_RE.match(p.replace(r"\\", "").strip())
            if not m:
                ok = False
                break
            vals.append(float(m.group(1)))
        if ok and name and "Method" not in name:
            out.append((i, name, vals))
    return out


def fmt(v):
    return f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Always build from the pre-grid baseline so repeated runs stay idempotent.
    bak = TEX + ".bak_pre_gru_grid"
    if not os.path.isfile(bak):
        shutil.copy(TEX, bak)
    with open(bak) as f:
        lines = f.readlines()
    existing = parse_existing(lines)
    new_rows = read_new_rows()

    # Rank the DISPLAYED (4-dec) values over all rows, per column.
    all_named = [(name, vals) for _, name, vals in existing] + new_rows
    col_colors = []  # per column: {displayed_value: color}
    for c, metric in enumerate(METRICS):
        disp = sorted({round(v[c], 4) for _, v in all_named},
                      reverse=metric in HIGHER_BETTER)
        col_colors.append({v: COLORS[r] for r, v in enumerate(disp[:3])})

    def render_row(name, vals):
        cells = []
        for c, v in enumerate(vals):
            d = round(v, 4)
            cells.append(f"{col_colors[c].get(d, '')}{fmt(d)}")
        return f"{name} & " + " & ".join(cells) + r" \\" + "\n"

    # Re-render existing rows with fresh highlights (values untouched).
    for idx, name, vals in existing:
        lines[idx] = render_row(name, vals)

    # Insert the new block before \bottomrule.
    bottom = next(i for i, ln in enumerate(lines) if r"\bottomrule" in ln)
    block = ["\\midrule\n"] + [render_row(n, v) for n, v in new_rows]
    lines = lines[:bottom] + block + lines[bottom:]

    out = "".join(lines)
    if args.dry_run:
        print(out)
        return

    os.makedirs(os.path.dirname(TEX) or ".", exist_ok=True)
    with open(TEX, "w") as f:
        f.write(out)

    tables_dir = os.path.dirname(TEX)
    base = os.path.splitext(os.path.basename(TEX))[0]
    compile_cmd = (
        "source /etc/profile.d/lmod.sh && module load latex/20240430 && "
        # MN5 has no ghostscript module; /usr/bin/gs is used for the `gs` step below.
        # The latex module has a flat bin/ layout that breaks kpathsea
        # self-location ("can't find pdflatex.fmt") — point it at the real root.
        "export TEXMFROOT=/apps/GPP/LATEX/20240430 && "
        "export TEXMFCNF=$TEXMFROOT:$TEXMFROOT/texmf-dist/web2c && "
        f"cd {tables_dir} && pdflatex -interaction=nonstopmode {base}.tex >/dev/null && "
        # 300 dpi matches the table's previous PNG renders.
        f"gs -sDEVICE=png16m -r300 -o {base}.png -dBATCH -dNOPAUSE {base}.pdf"
    )
    r = subprocess.run(["bash", "-c", compile_cmd], capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        raise SystemExit(f"compile failed: {r.stderr[-2000:]}")
    print("OK:", TEX, "->", os.path.join(tables_dir, base + ".pdf"), "+ .png")


if __name__ == "__main__":
    main()
