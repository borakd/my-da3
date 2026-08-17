#!/usr/bin/env python
"""Confidence-gate campaign: calibrate gate thresholds from a log-mode run.

Reads state_gate.json telemetry (STATE_GATE_MODE=log) for every scene in the
list, pools per-frame signals, and reports:
  - pooled percentiles of each signal (the tau grid candidates),
  - per-scene signal spread (does a global tau make sense, or is 'rel' needed),
  - Spearman correlation of per-scene signal stats vs the clean per-scene
    metrics (does low confidence actually mark bad scenes?).

Usage:
  python eval_pipeline/cg_calibrate.py --label diag_cg_log430 \
      --scene_list eval_pipeline/noise_oracle_subset_430.txt [--json out.json]
"""
import argparse
import csv
import json
import os

import numpy as np

METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
PCTS = [1, 5, 10, 20, 30, 50, 70, 90, 99]


def midrank(a):
    # Average ranks over tie blocks. Plain argsort-of-argsort breaks by input
    # order, and ties are COMMON here: log(conf-1+1e-8) floors at -18.42
    # wherever conf saturates at 1 (~20% of frames in the smoke data).
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = midrank(a), midrank(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--out_root",
                    default="/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval")
    ap.add_argument("--clean_csv", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    with open(args.scene_list) as f:
        scenes = [ln.strip() for ln in f if ln.strip()]

    keys = None
    pooled = {}
    scene_stats = {}
    for s in scenes:
        p = os.path.join(args.out_root, args.label, "preds", s, "state_gate.json")
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            d = json.load(f)
        if keys is None:
            keys = d["keys"][:-2]  # drop s, g
            for k in keys:
                pooled[k] = []
        arr = np.array(d["frames"], dtype=float)
        if arr.size == 0:
            continue
        st = {}
        for j, k in enumerate(keys):
            col = arr[:, j]
            pooled[k].append(col)
            st[k + "_mean"] = float(col.mean())
            st[k + "_min"] = float(col.min())
            st[k + "_p10"] = float(np.percentile(col, 10))
        st["n_frames"] = int(arr.shape[0])
        scene_stats[s] = st

    if not scene_stats:
        raise SystemExit(
            f"no state_gate.json telemetry under {args.out_root}/{args.label}/preds"
            " -- was the arm run with STATE_GATE_MODE=log through the ray worker?")
    print(f"telemetry: {len(scene_stats)}/{len(scenes)} scenes")
    out = {"n_scenes": len(scene_stats), "pooled": {}, "per_scene_spread": {},
           "correlations": {}}

    for k in keys:
        allv = np.concatenate(pooled[k])
        pc = {f"p{q}": float(np.percentile(allv, q)) for q in PCTS}
        out["pooled"][k] = dict(n=int(allv.size), mean=float(allv.mean()), **pc)
        print(f"{k:15s} n={allv.size}  mean {allv.mean():+.4f}  " +
              "  ".join(f"p{q} {pc[f'p{q}']:+.4f}" for q in PCTS))
        means = np.array([scene_stats[s][k + "_mean"] for s in scene_stats])
        out["per_scene_spread"][k] = dict(
            scene_mean_p10=float(np.percentile(means, 10)),
            scene_mean_p50=float(np.percentile(means, 50)),
            scene_mean_p90=float(np.percentile(means, 90)))

    clean_csv = args.clean_csv or os.path.join(
        args.out_root, "summary", "per_scene_augfull_lr1e5.csv")
    clean = {}
    with open(clean_csv) as f:
        for row in csv.DictReader(f):
            if row.get("scene") in scene_stats:
                clean[row["scene"]] = {m: float(row[m]) for m in METRICS}
    common = sorted(set(clean) & set(scene_stats))
    print(f"\ncorrelations over {len(common)} scenes (per-scene signal vs clean metric):")
    for k in keys:
        for stat in ("_mean", "_p10"):
            sig = np.array([scene_stats[s][k + stat] for s in common])
            row = {}
            for m in METRICS:
                met = np.array([clean[s][m] for s in common])
                ok = np.isfinite(sig) & np.isfinite(met)
                row[m] = spearman(sig[ok], met[ok])
            out["correlations"][k + stat] = row
            print(f"  {k + stat:22s} " +
                  "  ".join(f"{m} {row[m]:+.3f}" for m in METRICS))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
