#!/usr/bin/env python3
"""Summarize the C1/C1b/B0 control+baseline rows on the 13 smoke scenes and check
protocol parity of the B0 re-runs against the cached per_scene_<label>.csv.

Reads <probe>/<label>/eval/<scene>/eval_depth_pose_metrics.csv (camera_id==ALL,
local_timestep==MEAN row, same selection as eval_pipeline/aggregate_results.py),
writes <probe>/summary_control_rows.csv (per-scene, all labels) and
<probe>/parity_<label>_vs_<cached>.csv, and prints mean rows + max abs diffs.
"""
import csv, os, sys
import numpy as np

PROBE = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe"
SUMMARY = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/summary"
SCENES = [l.strip() for l in open("/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt") if l.strip()]
METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
LABELS = ["B0_augfull_none", "B0_gtray_gt", "C1_ext_context", "C1b_ext_context_last"]
PARITY = {"B0_augfull_none": "augfull_lr1e5", "B0_gtray_gt": "gtray_lr1e5"}


def mean_row(p):
    if not os.path.isfile(p):
        return None
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        if r.get("camera_id") == "ALL" and r.get("local_timestep") == "MEAN":
            return {m: float(r[m]) for m in METRICS}
    for r in rows:
        if r.get("local_timestep") == "MEAN":
            return {m: float(r[m]) for m in METRICS}
    return None


res = {}
for lb in LABELS:
    res[lb] = {sc: mean_row(os.path.join(PROBE, lb, "eval", sc, "eval_depth_pose_metrics.csv")) for sc in SCENES}

with open(os.path.join(PROBE, "summary_control_rows.csv"), "w") as f:
    w = csv.writer(f); w.writerow(["label", "scene"] + METRICS)
    for lb in LABELS:
        for sc in SCENES:
            r = res[lb][sc]
            w.writerow([lb, sc] + ([f"{r[m]:.10g}" for m in METRICS] if r else [""] * 5))

print(f"{'label':22s} n/13  " + "  ".join(f"{m:>9s}" for m in METRICS))
for lb in LABELS:
    have = [r for r in res[lb].values() if r]
    if have:
        print(f"{lb:22s} {len(have):2d}/13  " + "  ".join(f"{np.mean([r[m] for r in have]):9.5f}" for m in METRICS))
    else:
        print(f"{lb:22s}  0/13")

for lb, cached in PARITY.items():
    cp = os.path.join(SUMMARY, f"per_scene_{cached}.csv")
    ref = {r["scene"]: {m: float(r[m]) for m in METRICS} for r in csv.DictReader(open(cp))}
    out = os.path.join(PROBE, f"parity_{lb}_vs_{cached}.csv")
    maxd = {m: 0.0 for m in METRICS}; n = 0
    with open(out, "w") as f:
        w = csv.writer(f); w.writerow(["scene"] + [f"d_{m}" for m in METRICS] + [f"new_{m}" for m in METRICS] + [f"cached_{m}" for m in METRICS])
        for sc in SCENES:
            r = res[lb][sc]
            if r is None or sc not in ref:
                w.writerow([sc] + ["MISSING"] * 15); continue
            n += 1
            d = {m: abs(r[m] - ref[sc][m]) for m in METRICS}
            for m in METRICS: maxd[m] = max(maxd[m], d[m])
            w.writerow([sc] + [f"{d[m]:.3e}" for m in METRICS] + [f"{r[m]:.10g}" for m in METRICS] + [f"{ref[sc][m]:.10g}" for m in METRICS])
    print(f"PARITY {lb} vs cached {cached}: {n}/13 scenes compared; max|diff| " +
          ", ".join(f"{m}={maxd[m]:.3e}" for m in METRICS) + f"  -> {out}")
