#!/usr/bin/env python
"""Aggregate the exterior-rig ray-conditioning arms on their COMMON scene set: per scene the evaluator's ALL/MEAN row
(same aggregation as the master table: unweighted nanmean over scenes), plus per-scene win rates of the first label
against each other label.    python rigray_summary.py OUT_DIR LABEL [LABEL ...]"""
import csv, os, sys
import numpy as np, pandas as pd
out, labels = sys.argv[1], sys.argv[2:]
M = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
rows = []
for L in labels:
    base = f"{out}/{L}/eval"
    for s in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        f = f"{base}/{s}/eval_depth_pose_metrics.csv"
        if not os.path.isfile(f): continue
        for r in csv.DictReader(open(f)):
            if r["camera_id"] == "ALL" and r["local_timestep"] == "MEAN":
                rows.append(dict(label=L, scene=s, **{m: float(r[m]) for m in M}))
d = pd.DataFrame(rows); n = d.groupby("label").scene.nunique()
common = set.intersection(*[set(d[d.label == L].scene) for L in labels]); d = d[d.scene.isin(common)]
L0 = labels[0]; p = d.pivot(index="scene", columns="label", values=M)
lines = [f"scenes per label: {n.to_dict()}; common: {len(common)}", "", "| label | " + " | ".join(M) + " |", "|---" * (len(M) + 1) + "|"]
for L in labels: lines.append(f"| {L} | " + " | ".join(f"{p[m][L].mean():.4f}" for m in M) + " |")
lines += ["", f"per-scene win rate of {L0} (lower is better) vs:"]
for L in labels[1:]: lines.append(f"- {L}: " + ", ".join(f"{m} {(p[m][L0] < p[m][L]).mean():.1%}" for m in M))
print("\n".join(lines)); open(f"{out}/{L0}/SUMMARY_{'_vs_'.join(labels)}.md", "w").write("\n".join(lines) + "\n")
p.to_csv(f"{out}/{L0}/per_scene_{'_vs_'.join(labels)}.csv")
