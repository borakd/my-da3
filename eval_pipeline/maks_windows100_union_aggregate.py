#!/usr/bin/env python3
"""Scene-level comparison of the union blocking arms on the 100-scene set.

Rows (each scene counts once, paired over the scenes present in every arm):
  baseline                      augfull_lr1e5 plain run
  strongest window blocked      arm w100_W1  (rank-1 window only)
  its pre-window blocked        arm w100_P1
  all windows blocked           arm w100_WALL (union of up to 4 windows per scene)
  all pre-windows blocked       arm w100_PALL (union of their pre-window spans)

Paired deltas vs baseline with bootstrap 95% CIs and win counts, plus the direct
contrast all-pre-windows vs all-windows. Writes results_union.md and the
compiled table windows100_union.{tex,pdf,png} (3 decimals, top-3 colours).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maks_window_tables import table, compile_png, METRICS, HIGHER, fmt  # noqa: E402
from maks_windows100_aggregate import scene_row, boot_ci  # noqa: E402


def paired(rows_a, rows_b, scenes):
    out = {}
    for m in METRICS:
        d = np.array([rows_a[s][m] - rows_b[s][m] for s in scenes])
        lo, hi = boot_ci(d)
        better = (d > 0) if m in HIGHER else (d < 0)
        out[m] = {"delta": float(d.mean()), "lo": lo, "hi": hi, "sig": bool(lo > 0 or hi < 0),
                  "wins": int(better.sum()), "n": len(d),
                  "better": bool((d.mean() > 0) if m in HIGHER else (d.mean() < 0))}
    return out


def line(name, st):
    return f"- {name}: " + "; ".join(
        f"{m} {v['delta']:+.4f}{'*' if v['sig'] else ''} [{v['lo']:+.4f},{v['hi']:+.4f}] "
        f"{'↑' if v['better'] else '↓'} wins {v['wins']}/{v['n']}" for m, v in st.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline_eval", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    man = json.load(open(os.path.join(args.root, "manifest.json")))
    scenes_all = man["scenes"]
    arms = {"W1": "w100_W1", "P1": "w100_P1", "WALL": "w100_WALL", "PALL": "w100_PALL"}
    rows = {"B": {}, **{k: {} for k in arms}}
    for s in scenes_all:
        rows["B"][s] = scene_row(os.path.join(args.baseline_eval, s, "eval_depth_pose_metrics.csv"))
        for k, a in arms.items():
            rows[k][s] = scene_row(os.path.join(args.root, a, "eval", s, "eval_depth_pose_metrics.csv"))
    scenes = [s for s in scenes_all if all(rows[k].get(s) is not None and all(np.isfinite(rows[k][s][m]) for m in METRICS)
                                          for k in rows)]
    n_win = {}
    for w in man["windows"]:
        n_win[w["scene"]] = n_win.get(w["scene"], 0) + 1
    fr = {k: 0 for k in ("WALL", "PALL")}
    for k, a in (("WALL", "w100_WALL"), ("PALL", "w100_PALL")):
        spec = json.load(open(os.path.join(args.root, a, "control.json")))
        fr[k] = float(np.mean([len(spec[s]["block"]) for s in scenes if s in spec]))
    mean = {k: {m: float(np.mean([rows[k][s][m] for s in scenes])) for m in METRICS} for k in rows}

    L = [f"# Union blocking on the 100-scene set (scene-level, paired)", "",
         f"{len(scenes)} scenes with every arm present (of {len(scenes_all)}); each scene counts once. "
         f"Windows per scene: mean {np.mean([n_win[s] for s in scenes]):.2f} (cap {man['max_windows']}). "
         f"Frames blocked per scene: all-windows {fr['WALL']:.1f}, all-pre-windows {fr['PALL']:.1f}. "
         "Values = per-scene end-of-scene averages, mean over scenes; * = bootstrap 95% CI excludes 0.", "",
         "| row | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    names = {"B": "baseline", "W1": "strongest window blocked", "P1": "its pre-window blocked",
             "WALL": "all windows blocked", "PALL": "all pre-windows blocked"}
    for k in ("B", "W1", "P1", "WALL", "PALL"):
        L.append(f"| {names[k]} | " + " | ".join(fmt(m, mean[k][m]) for m in METRICS) + " |")
    L.append("")
    for k in ("W1", "P1", "WALL", "PALL"):
        L.append(line(f"{names[k]} − baseline", paired(rows[k], rows["B"], scenes)))
    L.append(line("all pre-windows − all windows", paired(rows["PALL"], rows["WALL"], scenes)))
    L.append(line("all pre-windows − its pre-window (rank 1 only)", paired(rows["PALL"], rows["P1"], scenes)))
    md = os.path.join(args.out_dir, "results_union.md")
    open(md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    note = (f"{len(scenes)} DROID test scenes, checkpoint augfull\\_lr1e5, scene-level averages (each scene once). "
            f"Up to {man['max_windows']} decline windows per scene; all-windows blocks {fr['WALL']:.0f} frames per scene on average, "
            f"all-pre-windows {fr['PALL']:.0f}. Blocked = frames decoded and scored but not written to memory.")
    trows = [("Regular CUT3R (all frames written)", mean["B"]),
             ("Strongest window excluded", mean["W1"]),
             ("Its pre-window excluded", mean["P1"]),
             ("All windows excluded", mean["WALL"]),
             ("All pre-windows excluded", mean["PALL"])]
    tex = os.path.join(args.out_dir, "windows100_union.tex")
    table(trows, note, tex); print(compile_png(tex)); print(f"wrote {md}")


if __name__ == "__main__":
    main()
