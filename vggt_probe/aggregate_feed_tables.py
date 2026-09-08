#!/usr/bin/env python3
"""Aggregate the CUT3R feed (F1-F8), control (C1/C1b), baseline re-run (B0_*) and cached
reference rows (augfull_lr1e5, gtray_lr1e5, prevgt_lr1e5, augfull_cg_fuse_g7) on the 13 smoke
scenes. Login-node safe: reads CSVs only.

Outputs (under <probe>/tables/):
  cut3r_feeds_per_scene.csv   label,scene,T,split,absrel,a1,ate,rpe_trans,rpe_rot
  cut3r_feeds_mean.csv        label,n,split(all/short/long),mean of each metric
  cut3r_feeds_wins.csv        per-label per-scene win counts vs augfull_lr1e5 and vs gtray_lr1e5
  cut3r_feeds_parity.csv      B0 re-runs vs cached rows, max abs diff per metric
  cut3r_feeds_tables.md       markdown rendering of all of the above
"""
import csv, os, sys
import numpy as np

PROBE = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe"
SUMMARY = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/summary"
STORE = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
WT = "/gpfs/home/koc/koc821022/vggt_features"
OUT = os.path.join(PROBE, "tables")
SCENES = ["RAIL+80edfcb1+2023-07-14-14h-28m-45s"] + [l.strip() for l in open(f"{WT}/eval_pipeline/cg_smoke_scenes_12.txt") if l.strip()]
METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
CACHED = ["augfull_lr1e5", "gtray_lr1e5", "prevgt_lr1e5", "augfull_cg_fuse_g7"]
FRESH = ["B0_augfull_none", "B0_gtray_gt", "C1_ext_context", "C1b_ext_context_last",
         "F1_prevgt_prevgt_raw", "F2_prevgt_prevgt_calib", "F3_prevgt_prevgt_oracle",
         "F4_gtray_gt_raw", "F5_gtray_gt_calib", "F6_gtray_gt_oracle",
         "F7_prevgt_prevgt_slide_calib", "F8_gtray_gt_slide_calib"]
PARITY = {"B0_augfull_none": "augfull_lr1e5", "B0_gtray_gt": "gtray_lr1e5"}
LONG_T = 200


def mean_row(p):
    if not os.path.isfile(p):
        return None
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        if r.get("camera_id") == "ALL" and r.get("local_timestep") == "MEAN":
            return {m: float(r[m]) for m in METRICS}
    return None


T = {sc: len([f for f in os.listdir(f"{STORE}/{sc}/dense/rgb") if f.endswith(".png")]) for sc in SCENES}
SPLIT = {sc: ("long" if T[sc] > LONG_T else "short") for sc in SCENES}

res, nfr = {}, {}
for lb in CACHED:
    ref = {r["scene"]: {m: float(r[m]) for m in METRICS} for r in csv.DictReader(open(f"{SUMMARY}/per_scene_{lb}.csv"))}
    res[lb] = {sc: ref.get(sc) for sc in SCENES}
for lb in FRESH:
    res[lb] = {sc: mean_row(f"{PROBE}/{lb}/eval/{sc}/eval_depth_pose_metrics.csv") for sc in SCENES}
    nfr[lb] = {}
    for sc in SCENES:
        cd = f"{PROBE}/{lb}/preds/{sc}/camera"
        nfr[lb][sc] = len(os.listdir(cd)) if os.path.isdir(cd) else 0
LABELS = CACHED + FRESH

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/cut3r_feeds_per_scene.csv", "w") as f:
    w = csv.writer(f); w.writerow(["label", "scene", "T", "split"] + METRICS)
    for lb in LABELS:
        for sc in SCENES:
            r = res[lb][sc]
            w.writerow([lb, sc, T[sc], SPLIT[sc]] + ([f"{r[m]:.10g}" for m in METRICS] if r else [""] * 5))


def agg(lb, split):
    rows = [res[lb][sc] for sc in SCENES if res[lb][sc] and (split == "all" or SPLIT[sc] == split)]
    if not rows:
        return 0, None
    return len(rows), {m: float(np.mean([r[m] for r in rows])) for m in METRICS}

md = []
with open(f"{OUT}/cut3r_feeds_mean.csv", "w") as f:
    w = csv.writer(f); w.writerow(["label", "split", "n"] + METRICS)
    for split in ["all", "short", "long"]:
        ntot = sum(1 for sc in SCENES if split == "all" or SPLIT[sc] == split)
        md.append(f"\n### Mean over {split} scenes (n={ntot}; short = T<={LONG_T}, long = T>{LONG_T})\n")
        md.append("| label | n | absrel | a1 | ATE | RPE-trans | RPE-rot |")
        md.append("|---|---|---|---|---|---|---|")
        for lb in LABELS:
            n, mrow = agg(lb, split)
            w.writerow([lb, split, n] + ([f"{mrow[m]:.6g}" for m in METRICS] if mrow else [""] * 5))
            if mrow:
                md.append(f"| {lb} | {n}/{ntot} | {mrow['absrel']:.4f} | {mrow['a1']:.4f} | {mrow['ate']:.4f} | {mrow['rpe_trans']:.5f} | {mrow['rpe_rot']:.3f} |")
            else:
                md.append(f"| {lb} | 0/{ntot} | - | - | - | - | - |")

# per-scene wins vs references (lower better except a1)
better = {m: (lambda a, b: a > b) if m == "a1" else (lambda a, b: a < b) for m in METRICS}
with open(f"{OUT}/cut3r_feeds_wins.csv", "w") as f:
    w = csv.writer(f); w.writerow(["label", "reference", "n_compared"] + [f"wins_{m}" for m in METRICS])
    md.append("\n### Per-scene win counts (label better than reference; lower is better except a1)\n")
    md.append("| label | vs | n | absrel | a1 | ATE | RPE-trans | RPE-rot |")
    md.append("|---|---|---|---|---|---|---|---|")
    for ref in ["augfull_lr1e5", "gtray_lr1e5"]:
        for lb in LABELS:
            if lb == ref:
                continue
            n = 0; wins = {m: 0 for m in METRICS}
            for sc in SCENES:
                a, b = res[lb][sc], res[ref][sc]
                if a and b:
                    n += 1
                    for m in METRICS:
                        wins[m] += int(better[m](a[m], b[m]))
            w.writerow([lb, ref, n] + [wins[m] for m in METRICS])
            if n:
                md.append(f"| {lb} | {ref} | {n} | " + " | ".join(f"{wins[m]}/{n}" for m in METRICS) + " |")

with open(f"{OUT}/cut3r_feeds_parity.csv", "w") as f:
    w = csv.writer(f); w.writerow(["label", "cached", "n"] + [f"maxabs_{m}" for m in METRICS])
    md.append("\n### Protocol parity: fresh B0 re-runs vs cached per_scene rows (max abs diff)\n")
    md.append("| label | cached | n | absrel | a1 | ATE | RPE-trans | RPE-rot |")
    md.append("|---|---|---|---|---|---|---|---|")
    for lb, ref in PARITY.items():
        n = 0; maxd = {m: 0.0 for m in METRICS}
        for sc in SCENES:
            a, b = res[lb][sc], res[ref][sc]
            if a and b:
                n += 1
                for m in METRICS:
                    maxd[m] = max(maxd[m], abs(a[m] - b[m]))
        w.writerow([lb, ref, n] + [f"{maxd[m]:.3e}" for m in METRICS])
        md.append(f"| {lb} | {ref} | {n} | " + " | ".join(f"{maxd[m]:.1e}" for m in METRICS) + " |")

md.append("\n### Per-scene ATE / RPE-rot / absrel (scene, T)\n")
hdr = "| scene | T | " + " | ".join(LABELS) + " |"
for m in ["ate", "rpe_rot", "absrel"]:
    md.append(f"\n**{m}**\n"); md.append(hdr); md.append("|---|---|" + "---|" * len(LABELS))
    for sc in SCENES:
        cells = []
        for lb in LABELS:
            r = res[lb][sc]
            cells.append(f"{r[m]:.4f}" if r else "-")
        md.append(f"| {sc} | {T[sc]} | " + " | ".join(cells) + " |")

md.append("\n### Coverage / frame counts (fresh labels: n scenes with eval CSV; saved camera npz count == store T?)\n")
md.append("| label | n eval CSV | frame-count mismatches |")
md.append("|---|---|---|")
for lb in FRESH:
    n = sum(1 for sc in SCENES if res[lb][sc])
    mism = [f"{sc}:{nfr[lb][sc]}!={T[sc]}" for sc in SCENES if nfr[lb][sc] and nfr[lb][sc] != T[sc]]
    md.append(f"| {lb} | {n}/13 | {', '.join(mism) if mism else 'none'} |")
# NaN check
nan = [(lb, sc, m) for lb in LABELS for sc in SCENES for m in METRICS if res[lb][sc] and not np.isfinite(res[lb][sc][m])]
md.append(f"\nNon-finite metric cells: {nan if nan else 'none'}")

open(f"{OUT}/cut3r_feeds_tables.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
