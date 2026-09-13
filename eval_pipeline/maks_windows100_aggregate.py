#!/usr/bin/env python3
"""Aggregate the window / pre-window blocking runs over ALL detected windows.

For every (scene, window) pair in <root>/manifest.json, read the scene-level
table numbers (per-scene MEAN row) of: the baseline run, the run with that
window blocked (arm w100_W<rank>) and the run with its pre-window span blocked
(arm w100_P<rank>). Pool over windows (each window counts once; a scene with
four windows contributes four paired triples):

  mean over windows of baseline / window-blocked / pre-blocked values,
  paired deltas vs baseline with bootstrap 95% CIs and per-window win rates,
  and the direct paired contrast pre-block vs window-block.

Also a breakdown by window position (terminal = window touches the scene end,
interior = it does not), since the single-scene result differed between them.

Outputs in --out_dir: results.md, windows100_main.{tex,pdf,png} (3 rows) and
windows100_breakdown.{tex,pdf,png} (6 rows), 3-decimal values, top-3 colours.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maks_window_tables import table, compile_png, METRICS, HIGHER, fmt  # noqa: E402


def scene_row(p):
    if not os.path.isfile(p):
        return None
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["camera_id"] == "ALL" and r["local_timestep"] == "MEAN":
            return {m: float(r[m]) for m in METRICS}
    for r in rows:
        if r["local_timestep"] == "MEAN":
            return {m: float(r[m]) for m in METRICS}
    return None


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stats(trip, key_a, key_b):
    """Paired a-b over windows -> per metric dict."""
    out = {}
    for m in METRICS:
        d = np.array([t[key_a][m] - t[key_b][m] for t in trip])
        lo, hi = boot_ci(d)
        better = (d > 0) if m in HIGHER else (d < 0)
        worse = (d < 0) if m in HIGHER else (d > 0)
        out[m] = {"delta": float(d.mean()), "lo": lo, "hi": hi, "sig": bool(lo > 0 or hi < 0),
                  "wins": int(better.sum()), "losses": int(worse.sum()), "n": len(d),
                  "better": bool((d.mean() > 0) if m in HIGHER else (d.mean() < 0))}
    return out


def mean_row(trip, key):
    return {m: float(np.mean([t[key][m] for t in trip])) for m in METRICS}


def md_stats(name, st):
    cells = []
    for m in METRICS:
        s = st[m]
        cells.append(f"{m} {s['delta']:+.4f}{'*' if s['sig'] else ''} [{s['lo']:+.4f},{s['hi']:+.4f}] "
                     f"{'↑' if s['better'] else '↓'} wins {s['wins']}/{s['n']}")
    return f"- {name}: " + "; ".join(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline_eval", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    man = json.load(open(os.path.join(args.root, "manifest.json")))

    trip, missing = [], 0
    for w in man["windows"]:
        s, k = w["scene"], w["rank"]
        B = scene_row(os.path.join(args.baseline_eval, s, "eval_depth_pose_metrics.csv"))
        W = scene_row(os.path.join(args.root, f"w100_W{k}", "eval", s, "eval_depth_pose_metrics.csv"))
        P = scene_row(os.path.join(args.root, f"w100_P{k}", "eval", s, "eval_depth_pose_metrics.csv"))
        if B is None or W is None or P is None or any(not math.isfinite(x[m]) for x in (B, W, P) for m in METRICS):
            missing += 1
            continue
        trip.append({"B": B, "W": W, "P": P, "terminal": w["terminal"], "scene": s, "rank": k, "len": w["len"]})
    n_scenes = len({t["scene"] for t in trip})
    term = [t for t in trip if t["terminal"]]
    inter = [t for t in trip if not t["terminal"]]

    L = [f"# Window vs pre-window blocking, pooled over windows", "",
         f"{len(man['scenes'])} test scenes (seed {man['seed']}), {len(trip)} windows scored "
         f"({missing} skipped for missing/NaN rows), up to {man['max_windows']} strongest windows per scene "
         f"(step-score Z ≥ {man['zmin']}, widths ≤ {man['max_width']}). Each window = one paired triple "
         "(baseline, window blocked, pre-window blocked); values are per-scene table numbers averaged over "
         "windows (a scene with four windows contributes four triples). Blocked = frames still decoded and "
         "scored but not written to state/memory. Pre-window = same-length span immediately before the window. "
         f"Terminal windows ({len(term)}) touch the scene end; interior ({len(inter)}) do not.", ""]

    def block(name, T):
        Bm, Wm, Pm = mean_row(T, "B"), mean_row(T, "W"), mean_row(T, "P")
        L.append(f"## {name} (n={len(T)} windows)"); L.append("")
        L.append("| row | " + " | ".join(METRICS) + " |"); L.append("|---|" + "---|" * len(METRICS))
        L.append("| baseline | " + " | ".join(fmt(m, Bm[m]) for m in METRICS) + " |")
        L.append("| window blocked | " + " | ".join(fmt(m, Wm[m]) for m in METRICS) + " |")
        L.append("| pre-window blocked | " + " | ".join(fmt(m, Pm[m]) for m in METRICS) + " |")
        L.append("")
        L.append(md_stats("window blocked − baseline", stats(T, "W", "B")))
        L.append(md_stats("pre-window blocked − baseline", stats(T, "P", "B")))
        L.append(md_stats("pre-window − window (negative = pre-block better, except a1)", stats(T, "P", "W")))
        L.append("")
        return Bm, Wm, Pm

    Bm, Wm, Pm = block("All windows", trip)
    tB, tW, tP = block("Terminal windows", term) if term else (None, None, None)
    iB, iW, iP = block("Interior windows", inter) if inter else (None, None, None)
    md = os.path.join(args.out_dir, "results.md")
    open(md, "w").write("\n".join(L))
    print("\n".join(L))

    note = (f"{len(man['scenes'])} DROID test scenes, {len(trip)} decline windows (up to {man['max_windows']} per scene), "
            "checkpoint augfull\\_lr1e5. Values = per-scene end-of-scene averages, averaged over windows "
            "(each window counts once). Blocked = frames decoded and scored but not written to memory; "
            "pre-window = same-length span immediately before the window.")
    rows = [(f"Regular CUT3R (all frames written)", Bm),
            (f"Window frames excluded from memory", Wm),
            (f"Pre-window frames excluded from memory", Pm)]
    tex = os.path.join(args.out_dir, "windows100_main.tex")
    table(rows, note, tex); print(compile_png(tex))
    if term and inter:
        rows = [(f"Terminal windows (n={len(term)}): Regular CUT3R", tB),
                ("Terminal windows: window excluded", tW),
                ("Terminal windows: pre-window excluded", tP),
                (f"Interior windows (n={len(inter)}): Regular CUT3R", iB),
                ("Interior windows: window excluded", iW),
                ("Interior windows: pre-window excluded", iP)]
        tex = os.path.join(args.out_dir, "windows100_breakdown.tex")
        table(rows, note + " Terminal = window touches the scene end.", tex); print(compile_png(tex))
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
