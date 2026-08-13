#!/usr/bin/env python
"""Pinned Gate-3 instrument: per-scene win% vs arm C on ATE and RPE-rot.

Conventions (frozen in PREREG_VERDICT_RULES.md before the deliverable score
existed): strict '<' per scene, denominator = the 4292 common scenes, zero
tolerance for NaNs or missing scenes — any deviation is a loud failure, not a
silent shrink. Also reports the closure fraction (C - X) / (C - B) per metric
and the depth-family-band read.

Usage: python eval_pipeline/win_rate.py <label> [<label> ...]
"""
import csv
import sys

SUMMARY = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/summary"
C_LABEL, B_LABEL = "prevpred_lr1e5", "gtray_lr1e5"
METRICS = ["ate", "rpe_rot"]
TIER_PROGRESS, TIER_RESULT = 54.22, 63.02
BAND = {"ate": 4.0, "rpe_rot": 13.5}
DEPTH_BAND = {"absrel": (0.194, 0.201), "a1": (0.757, 0.773)}


def load(label):
    path = f"{SUMMARY}/per_scene_{label}.csv"
    rows = {}
    for r in csv.DictReader(open(path)):
        rows[r["scene"]] = r
    return rows


def main():
    C, B = load(C_LABEL), load(B_LABEL)
    for label in sys.argv[1:]:
        X = load(label)
        common = sorted(set(X) & set(C))
        assert len(common) == 4292, f"{label}: {len(common)} common scenes != 4292"
        print(f"== {label} (n={len(common)}) ==")
        for m in METRICS:
            xv = [float(X[s][m]) for s in common]
            cv = [float(C[s][m]) for s in common]
            bv = [float(B[s][m]) for s in common]
            assert all(v == v for v in xv + cv), f"{label}/{m}: NaN present"
            win = 100.0 * sum(x < c for x, c in zip(xv, cv)) / len(common)
            ties = sum(x == c for x, c in zip(xv, cv))
            mx, mc, mb = (sum(v) / len(v) for v in (xv, cv, bv))
            closure = (mc - mx) / (mc - mb) if mc != mb else float("nan")
            near = [t for t in (TIER_PROGRESS, TIER_RESULT) if abs(win - t) <= BAND[m]]
            print(f"  {m}: win% vs C = {win:.2f} (ties {ties})  mean {mx:.6f}"
                  f"  closure {100*closure:.1f}%"
                  f"{'  REPLICATE-BAND of tier ' + str(near) if near else ''}")
        for m, (lo, hi) in DEPTH_BAND.items():
            mv = sum(float(X[s][m]) for s in common) / len(common)
            ok = lo <= mv <= hi
            print(f"  {m}: mean {mv:.4f}  family band [{lo}, {hi}]: {'inside' if ok else 'OUTSIDE'}")


if __name__ == "__main__":
    main()
