#!/usr/bin/env python3
"""Paired comparison of eval arms on ONE scene subset (all scenes from the
4292-scene test list), against a baseline label, with the table convention:
per-scene MEAN row (depth mean over frames; pose RMSE over frames, Sim3), then
nanmean across scenes. Every arm is aggregated over exactly the same scenes.

For each arm vs baseline and metric: mean delta, paired-bootstrap 95% CI
(10k resamples over scenes), wins / ties / losses, and a significance flag
(CI excludes 0). Reference labels (existing full-run trees, e.g. the conf-gate
winner) can be added with --ref; they are scored on the same scenes.

  python maks_subset_compare.py --scene_list L --baseline LABEL=EVALDIR \
      --arm NAME=EVALDIR [--arm ...] [--ref NAME=EVALDIR ...] --out_dir DIR
"""
import argparse
import csv
import math
import os

import numpy as np

METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
HIGHER = {"a1"}
SURFACE = "#fcfcfb"; GRID = "#e6e5e1"; TEXT = "#0b0b0b"; TEXT2 = "#52514e"
COLORS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]


def scene_row(csv_path):
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r.get("camera_id") == "ALL" and r.get("local_timestep") == "MEAN":
            return {m: float(r[m]) if r[m] not in ("", "nan") else math.nan for m in METRICS}
    for r in rows:
        if r.get("local_timestep") == "MEAN":
            return {m: float(r[m]) if r[m] not in ("", "nan") else math.nan for m in METRICS}
    return None


def load_label(eval_dir, scenes):
    out = {}
    for s in scenes:
        r = scene_row(os.path.join(eval_dir, s, "eval_depth_pose_metrics.csv"))
        if r is not None:
            out[s] = r
    return out


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(m, v):
    return f"{v:.2f}" if m == "rpe_rot" else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--baseline", required=True, help="LABEL=EVALDIR")
    ap.add_argument("--arm", action="append", default=[], help="NAME=EVALDIR (run in this experiment)")
    ap.add_argument("--ref", action="append", default=[], help="NAME=EVALDIR (existing full-run reference)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    scenes = [l.strip() for l in open(args.scene_list) if l.strip()]
    bname, bdir = args.baseline.split("=", 1)
    base = load_label(bdir, scenes)

    arms = [(n, load_label(d, scenes), "arm") for n, d in (a.split("=", 1) for a in args.arm)]
    arms += [(n, load_label(d, scenes), "ref") for n, d in (a.split("=", 1) for a in args.ref)]
    # common scene set: baseline and every ARM must have it (refs may miss scenes; report n)
    common = [s for s in scenes if s in base and all(s in r for n, r, k in arms if k == "arm")]
    L = [f"# Subset comparison: {args.title}", "",
         f"Scenes: {len(common)} of {len(scenes)} listed (all from the 4292-scene test list), identical set for every row. "
         f"Values = nanmean over scenes of the per-scene table numbers. Deltas are paired per scene vs `{bname}`; "
         "CI = paired bootstrap 95%; * = CI excludes 0.", ""]
    L += ["| row | n | absrel | a1 | ate | rpe_trans | rpe_rot | wins/5 |", "|---|---|---|---|---|---|---|---|"]
    bm = {m: float(np.nanmean([base[s][m] for s in common])) for m in METRICS}
    L.append(f"| {bname} (baseline) | {len(common)} | " + " | ".join(fmt(m, bm[m]) for m in METRICS) + " | – |")
    detail = []
    deltas_all = {}
    for name, rows, kind in arms:
        sc = [s for s in common if s in rows]
        cells, wins = [], 0
        deltas_all[name] = {}
        for m in METRICS:
            d = np.array([rows[s][m] - base[s][m] for s in sc if np.isfinite(rows[s][m]) and np.isfinite(base[s][m])])
            deltas_all[name][m] = d
            mean_arm = float(np.nanmean([rows[s][m] for s in sc]))
            lo, hi = boot_ci(d)
            sig = (lo > 0 or hi < 0)
            better = (d.mean() > 0) if m in HIGHER else (d.mean() < 0)
            wins += int(better)
            w = int(np.sum(d > 0)) if m in HIGHER else int(np.sum(d < 0))
            l = int(np.sum(d < 0)) if m in HIGHER else int(np.sum(d > 0))
            cells.append(f"{fmt(m, mean_arm)} ({'+' if d.mean() >= 0 else '−'}{fmt(m, abs(d.mean()))}{'*' if sig else ''}, {'↑' if better else '↓'} {w}/{len(d)})")
            detail.append(f"- {name} {m}: Δ={d.mean():+.5f} CI[{lo:+.5f}, {hi:+.5f}] wins {w} losses {l} ties {len(d)-w-l}")
        tag = "" if kind == "arm" else " (reference, existing full run)"
        L.append(f"| {name}{tag} | {len(sc)} | " + " | ".join(cells) + f" | {wins} |")
    L += ["", "Cell format: mean (paired Δ vs baseline, ↑ better / ↓ worse, per-scene wins/n). ", "", "## Per-metric detail", ""] + detail + [""]

    # figure: paired delta distributions per metric, one violin per arm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.2 * len(METRICS), 4.2))
    fig.patch.set_facecolor(SURFACE)
    names = [n for n, _, _ in arms]
    for ax, m in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        for sde in ("top", "right"): ax.spines[sde].set_visible(False)
        ax.grid(True, axis="y", color=GRID, lw=1); ax.tick_params(colors=TEXT2, labelsize=8, length=0)
        data = [deltas_all[n][m] for n in names]
        parts = ax.violinplot(data, showmeans=False, showextrema=False, widths=0.8)
        for body, c in zip(parts["bodies"], COLORS):
            body.set_facecolor(c); body.set_edgecolor("none"); body.set_alpha(0.45)
        for i, (d, c) in enumerate(zip(data, COLORS), 1):
            ax.scatter([i], [d.mean()], color=c, s=40, zorder=3, edgecolor=SURFACE, linewidth=1)
            lo, hi = boot_ci(d)
            ax.plot([i, i], [lo, hi], color=c, lw=2, zorder=3)
        ax.axhline(0, color=TEXT2, lw=1)
        ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Δ {m} vs {bname}" + ("  (↑ better)" if m in HIGHER else "  (↓ better)"), loc="left", fontsize=9, color=TEXT2)
    fig.suptitle(f"{args.title}: paired per-scene deltas (violin), mean and bootstrap 95% CI", x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = os.path.join(args.out_dir, "subset_compare.png")
    fig.savefig(png, dpi=130, facecolor=SURFACE); plt.close(fig)
    L.append(f"Figure: `{png}`")
    md = os.path.join(args.out_dir, "subset_compare.md")
    open(md, "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
