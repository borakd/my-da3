#!/usr/bin/env python3
"""Plot the five eval metrics over frame index for one scene.

Reads an eval_depth_pose_metrics.csv written by eval_bundle/bin/eval_depth_poses.py
(camera 0, numeric local_timestep rows; the per-camera MEAN row supplies the
scene-level reference value) and writes:

  <out_dir>/metrics_over_time.png            five stacked panels, shared frame axis
  <out_dir>/metrics_over_time_<metric>.png   one panel per metric

Per-frame values are exactly the CSV rows (depth: per-frame median-scaled absrel /
a1; pose: per-frame ATE after the scene-wide Sim3 fit, consecutive-pair RPE
attributed to the later frame). The scene reference line is the CSV's own MEAN
row (nanmean over frames for depth, nan-RMSE for pose), i.e. the number that
enters the tables.

Usage:
  python plot_metrics_over_time.py --csv <eval csv> --out_dir <dir> [--title <str>]
or from Python: plot_metrics_over_time(csv_path, out_dir, title=None) -> [png paths]
"""
import argparse
import csv
import math
import os

METRICS = [
    ("absrel", "AbsRel", "lower is better"),
    ("a1", "δ < 1.25 (a1)", "higher is better"),
    ("ate", "ATE (m, Sim3)", "lower is better"),
    ("rpe_trans", "RPE trans (m)", "lower is better"),
    ("rpe_rot", "RPE rot (deg)", "lower is better"),
]
SCENE_AGG = {"absrel": "mean", "a1": "mean", "ate": "RMSE", "rpe_trans": "RMSE", "rpe_rot": "RMSE"}

# Reference palette (dataviz skill, light mode): series-1 blue, recessive grid, text tokens.
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
REF = "#52514e"


def read_csv(csv_path):
    """Return (frames, {metric: [values]}, {metric: scene value}) for camera 0."""
    frames, per, scene = [], {m: [] for m, _, _ in METRICS}, {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            cam = (r.get("camera_id") or "").strip()
            ts = (r.get("local_timestep") or "").strip()
            if ts == "MEAN":
                if cam in ("0", "ALL") and not scene:
                    scene = {m: _f(r.get(m)) for m, _, _ in METRICS}
                continue
            if cam != "0":
                continue
            try:
                t = int(ts)
            except ValueError:
                continue
            frames.append(t)
            for m, _, _ in METRICS:
                per[m].append(_f(r.get(m)))
    order = sorted(range(len(frames)), key=lambda i: frames[i])
    frames = [frames[i] for i in order]
    per = {m: [v[i] for i in order] for m, v in per.items()}
    return frames, per, scene


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.grid(False, axis="x")
    ax.tick_params(colors=TEXT2, labelsize=8, length=0)
    ax.margins(x=0.01)


def _panel(ax, frames, vals, key, label, note, scene_val, show_xlabel):
    _style(ax)
    ax.plot(frames, vals, color=SERIES, linewidth=2, solid_joinstyle="round", solid_capstyle="round")
    if scene_val is not None and math.isfinite(scene_val):
        ax.axhline(scene_val, color=REF, linewidth=1, alpha=0.7)
        ax.annotate(f"scene {SCENE_AGG[key]} {scene_val:.4g}", xy=(1.0, scene_val),
                    xycoords=("axes fraction", "data"), xytext=(-4, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8, color=TEXT2,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.9))
    ax.set_ylabel(label, color=TEXT, fontsize=9)
    ax.set_title(f"{label}  ({note})", loc="left", fontsize=9, color=TEXT2)
    if key == "a1":
        ax.set_ylim(0, 1.02)
    else:
        ax.set_ylim(bottom=0)
    if show_xlabel:
        ax.set_xlabel("frame #", color=TEXT, fontsize=9)


def plot_metrics_over_time(csv_path, out_dir, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames, per, scene = read_csv(csv_path)
    if len(frames) < 2:
        raise ValueError(f"fewer than 2 per-frame rows in {csv_path}")
    os.makedirs(out_dir, exist_ok=True)
    title = title or os.path.basename(os.path.dirname(os.path.abspath(csv_path)))
    written = []

    # Combined figure: five panels, one shared frame axis, one y-axis each.
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(11, 2.2 * len(METRICS)), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, label, note) in zip(axes, METRICS):
        _panel(ax, frames, per[key], key, label, note, scene.get(key), ax is axes[-1])
    fig.suptitle(f"{title} — metrics over frames (n={len(frames)})", x=0.01, ha="left",
                 fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(out_dir, "metrics_over_time.png")
    fig.savefig(p, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    written.append(p)

    # One file per metric.
    for key, label, note in METRICS:
        fig, ax = plt.subplots(1, 1, figsize=(11, 3.2))
        fig.patch.set_facecolor(SURFACE)
        _panel(ax, frames, per[key], key, label, note, scene.get(key), True)
        ax.set_title(f"{title} — {label}  ({note})", loc="left", fontsize=10, color=TEXT)
        fig.tight_layout()
        p = os.path.join(out_dir, f"metrics_over_time_{key}.png")
        fig.savefig(p, dpi=130, facecolor=SURFACE)
        plt.close(fig)
        written.append(p)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="eval_depth_pose_metrics.csv")
    ap.add_argument("--out_dir", default=None, help="default: the CSV's directory")
    ap.add_argument("--title", default=None, help="default: the CSV's parent directory name (scene)")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    for p in plot_metrics_over_time(args.csv, out_dir, args.title):
        print(p)


if __name__ == "__main__":
    main()
