#!/usr/bin/env python
"""Score the rig-ray arms and their controls on the RIG-SOLVED frames only (src flag 1 in the main arm's cond_poses.npz),
with the master-table pose math (rig_trackon_compare.metrics: Sim3 Umeyama + RMSE ATE / RPE on consecutive solved
frames), against the store GT; the rig's own pose on the same frames is scored too.
    python rigray_solved_frames.py OUT_DIR RIG_DIR MAIN_LABEL LABEL [LABEL ...]"""
import os, sys, glob
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig_trackon_compare import metrics
out, rig_dir, labels = sys.argv[1], sys.argv[2], sys.argv[3:]
M = ["ate", "rpe_trans", "rpe_rot"]

def load_cams(d):
    return {int(os.path.basename(f)[:6]): np.load(f)["pose"].astype(np.float64) for f in glob.glob(f"{d}/camera/*.npz")}

def one(scene):
    z = np.load(f"{out}/{labels[0]}/preds/{scene}/cond_poses.npz")
    frames, src, gt = z["frames"], z["src"], z["gt"].astype(np.float64)
    ks = [k for k in range(len(frames)) if src[k]]
    G = {k: gt[k] for k in range(len(frames))}
    rows = []
    for L in labels:
        d = f"{out}/{L}/preds/{scene}"
        if not os.path.isdir(f"{d}/camera"): return []
        rows.append(dict(scene=scene, label=L, **metrics(load_cams(d), G, ks)))
    a = np.load(f"{rig_dir}/{scene}/anchors.npz"); rig = {int(f) - int(frames[0]): P.astype(np.float64) for f, P in zip(a["frames"], a["poses"])}
    rows.append(dict(scene=scene, label="rig", **metrics(rig, G, ks)))
    return rows

scenes = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{out}/{labels[0]}/preds/*/cond_poses.npz"))
with Pool(16) as p: rows = [r for rs in p.imap_unordered(one, scenes, chunksize=16) for r in rs]
d = pd.DataFrame(rows).dropna(); ok = d.groupby("scene").label.nunique() == len(labels) + 1; d = d[d.scene.isin(ok[ok].index)]
p = d.pivot(index="scene", columns="label", values=M)
lines = [f"rig-solved frames only; scenes {p.shape[0]}; solved frames per scene median {np.median([n for n in d[d.label=='rig'].n])}", "",
         "| label | ATE | RPE_trans | RPE_rot |", "|---|---|---|---|"]
for L in ["rig"] + labels: lines.append(f"| {L} | " + " | ".join(f"{p[m][L].mean():.4f}" for m in M) + " |")
lines += ["", f"per-scene win rate of {labels[0]} vs:"] + [f"- {L}: " + ", ".join(f"{m} {(p[m][labels[0]] < p[m][L]).mean():.1%}" for m in M) for L in ["rig"] + labels[1:]]
print("\n".join(lines)); open(f"{out}/{labels[0]}/SUMMARY_solved_frames.md", "w").write("\n".join(lines) + "\n"); p.to_csv(f"{out}/{labels[0]}/per_scene_solved_frames.csv")
