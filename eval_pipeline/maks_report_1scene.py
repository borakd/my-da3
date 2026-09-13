#!/usr/bin/env python3
"""Assemble the harmful-frame pilot table for one scene (HARMFUL_FRAME_SKIP_PLAN.md §9).

Layout under --root (written by maks_harmful_1scene.sbatch):
  p1/preds/<scene>, p1/eval/<scene>/eval_depth_pose_metrics.csv          P1 scored on F
  p1/eval_kept_<rule>/<scene>/eval_depth_pose_metrics.csv                 V0 (P1 preds on K)
  v1_<rule>/preds/<scene>, v1_<rule>/eval/<scene>/...                     V1 block, scored on F
  v1_<rule>/eval_kept/<scene>/...                                         V1 block, scored on K
  v2_<rule>/preds/<scene>, v2_<rule>/eval/<scene>/...                     V2 drop, scored on K
  masks/skip_<rule>.json, masks/flag_summary.json

Also runs the §8 sanity gates:
  - V1 parity: preds before the first flagged frame byte-identical to P1, first
    frame after it different (proves the update switch reached the model).
  - V2 file count == |K| with original indices.
Writes <root>/report_<scene>.md and a per-frame ATE plot if matplotlib is available.
"""
import argparse
import csv
import json
import math
import os

import numpy as np

METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]


def mean_row(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r.get("camera_id") == "ALL" and r.get("local_timestep") == "MEAN":
            return {m: float(r[m]) for m in METRICS}
    for r in rows:
        if r.get("local_timestep") == "MEAN":
            return {m: float(r[m]) for m in METRICS}
    return None


def per_frame(path, col, index_map=None):
    """{frame: value}. Kept-set CSVs number frames by POSITION within K, so pass
    index_map (list: local -> original frame number) to re-key them."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["camera_id"].strip() != "0":
                continue
            try:
                t = int(r["local_timestep"])
            except ValueError:
                continue
            if index_map is not None:
                t = index_map[t]
            try:
                out[t] = float(r[col])
            except (TypeError, ValueError):
                out[t] = math.nan
    return out


def same_file(a, b):
    if not (os.path.isfile(a) and os.path.isfile(b)):
        return None
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def fmt(v):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--rules", nargs="+", default=["ate15", "rpe2x", "absrel15", "any"])
    args = ap.parse_args()
    R, S = args.root, args.scene
    lines = [f"# Harmful-frame pilot: `{S}`", ""]

    p1_csv = os.path.join(R, "p1", "eval", S, "eval_depth_pose_metrics.csv")
    p1 = mean_row(p1_csv)
    n_frames = len(per_frame(p1_csv, "ate"))
    lines.append(f"Frames: {n_frames}. Eval: eval_depth_poses.py default args "
                 "(depth median-scaled per frame, pose Sim3 + RMSE over frames).")
    lines.append("")

    for rule in args.rules:
        skip = json.load(open(os.path.join(R, "masks", f"skip_{rule}.json"))).get(S, [])
        skip = sorted(int(i) for i in skip)
        kept = [t for t in range(n_frames) if t not in set(skip)]
        arms = [
            ("P1 (F)", p1),
            ("V0 P1 on K", mean_row(os.path.join(R, "p1", f"eval_kept_{rule}", S, "eval_depth_pose_metrics.csv"))),
            ("V1 block (F)", mean_row(os.path.join(R, f"v1_{rule}", "eval", S, "eval_depth_pose_metrics.csv"))),
            ("V1 block (K)", mean_row(os.path.join(R, f"v1_{rule}", "eval_kept", S, "eval_depth_pose_metrics.csv"))),
            ("V2 drop (K)", mean_row(os.path.join(R, f"v2_{rule}", "eval", S, "eval_depth_pose_metrics.csv"))),
        ]
        lines.append(f"## rule `{rule}`: {len(skip)}/{n_frames} frames flagged "
                     f"({100*len(skip)/max(n_frames,1):.1f}%), first {skip[0] if skip else None}")
        lines.append("")
        lines.append("| arm | " + " | ".join(METRICS) + " |")
        lines.append("|---|" + "---|" * len(METRICS))
        for name, row in arms:
            vals = [fmt(row[m]) if row else "missing" for m in METRICS]
            lines.append(f"| {name} | " + " | ".join(vals) + " |")
        lines.append("")
        # Deltas that mean something: V1(F) vs P1(F); V2(K) vs V0(K); V1(K) vs V0(K)
        def delta(a, b):
            if not (a and b):
                return "missing"
            parts = []
            for m in METRICS:
                d = a[m] - b[m]
                better = d > 0 if m == "a1" else d < 0
                parts.append(f"{m} {d:+.4f}{' better' if better else ' worse'}")
            return "; ".join(parts)
        lines.append(f"- V1 block (F) vs P1 (F): {delta(arms[2][1], arms[0][1])}")
        lines.append(f"- V1 block (K) vs V0 (K): {delta(arms[3][1], arms[1][1])}")
        lines.append(f"- V2 drop (K) vs V0 (K): {delta(arms[4][1], arms[1][1])}")
        lines.append("")

        # --- sanity gates -------------------------------------------------------
        gates = []
        if skip:
            f0 = skip[0]
            p1d = os.path.join(R, "p1", "preds", S, "camera")
            v1d = os.path.join(R, f"v1_{rule}", "preds", S, "camera")
            before = [same_file(os.path.join(p1d, f"{t:06d}.npz"), os.path.join(v1d, f"{t:06d}.npz"))
                      for t in range(0, f0 + 1)]  # the flagged frame itself is decoded identically
            after = [same_file(os.path.join(p1d, f"{t:06d}.npz"), os.path.join(v1d, f"{t:06d}.npz"))
                     for t in range(f0 + 1, n_frames)]
            ok_before = all(b is True for b in before)
            ok_after = any(a is False for a in after)
            gates.append(f"V1 parity: frames 0..{f0} identical to P1 = {ok_before}; "
                         f"some frame after {f0} differs = {ok_after} "
                         f"({'PASS' if ok_before and ok_after else 'FAIL: update switch did not take effect'})")
            v2d = os.path.join(R, f"v2_{rule}", "preds", S, "depth")
            v2_files = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(v2d)) if os.path.isdir(v2d) else []
            gates.append(f"V2 files: {len(v2_files)} == |K| {len(kept)} and indices match = "
                         f"{v2_files == kept} ({'PASS' if v2_files == kept else 'FAIL'})")
        else:
            gates.append("no frames flagged; V1/V2 are identical to P1 by construction")
        lines.extend(f"- {g}" for g in gates)
        lines.append("")

        # --- per-frame ATE plot ---------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            a_p1 = per_frame(p1_csv, "ate")
            a_v1 = per_frame(os.path.join(R, f"v1_{rule}", "eval", S, "eval_depth_pose_metrics.csv"), "ate")
            a_v2 = per_frame(os.path.join(R, f"v2_{rule}", "eval", S, "eval_depth_pose_metrics.csv"), "ate",
                             index_map=kept)
            fig, ax = plt.subplots(figsize=(11, 3.6))
            for t in skip:
                ax.axvspan(t - 0.5, t + 0.5, color="#f4c7c3", lw=0)
            ax.plot(sorted(a_p1), [a_p1[t] for t in sorted(a_p1)], label="P1 plain", color="#4c72b0")
            if a_v1:
                ax.plot(sorted(a_v1), [a_v1[t] for t in sorted(a_v1)], label="V1 block", color="#dd8452")
            if a_v2:
                ax.plot(sorted(a_v2), [a_v2[t] for t in sorted(a_v2)], label="V2 drop (kept only)",
                        color="#55a868", marker=".", ms=3, lw=0.8)
            ax.set_xlabel("frame")
            ax.set_ylabel("per-frame ATE (Sim3, m)")
            ax.set_title(f"{S} — rule {rule} (shaded = flagged)")
            ax.legend(loc="upper left", fontsize=8)
            fig.tight_layout()
            png = os.path.join(R, f"ate_per_frame_{rule}.png")
            fig.savefig(png, dpi=130)
            lines.append(f"Per-frame ATE plot: `{png}`")
            lines.append("")
        except Exception as e:  # matplotlib missing in env, or similar: not fatal
            lines.append(f"(plot skipped: {e!r})")
            lines.append("")

    out = os.path.join(R, f"report_{S}.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[report] wrote {out}")


if __name__ == "__main__":
    main()
