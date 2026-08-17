#!/usr/bin/env python
"""Confidence-gate campaign: paired per-scene readout of gated arms vs clean.

Pairs each gated label's per-scene rows (camera_id==ALL, local_timestep==MEAN,
the aggregate_results.py reduction) against a clean reference, which is either
  --clean_label <diag label>   rows read from that label's eval/ tree, or
  --clean_csv   <per_scene_*.csv>  (default: summary/per_scene_augfull_lr1e5.csv,
                legitimate because the harness is byte-deterministic: the full
                run's per-scene numbers ARE the clean numbers for any subset).

For every metric: mean(clean), mean(gated), paired delta, seeded percentile
bootstrap CI, and directional win/tie/loss counts (a1 higher-better, all other
metrics lower-better). Also summarizes state_gate.json telemetry when present
(skip fraction per scene, pooled).

Usage:
  python eval_pipeline/cg_analyze.py --labels diag_cg_t1 diag_cg_t2 \
      --scene_list eval_pipeline/noise_oracle_subset_430.txt [--json out.json]
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
HIGHER_BETTER = {"a1"}


def read_scene_summary(path):
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("camera_id") == "ALL" and row.get("local_timestep") == "MEAN":
                out = {}
                for m in METRICS:
                    try:
                        out[m] = float(row[m])
                    except (KeyError, ValueError):
                        out[m] = float("nan")
                return out
    return None


def label_rows(out_root, label, scenes):
    rows = {}
    for s in scenes:
        p = os.path.join(out_root, label, "eval", s, "eval_depth_pose_metrics.csv")
        if os.path.isfile(p):
            r = read_scene_summary(p)
            if r is not None:
                rows[s] = r
    return rows


def csv_rows(path, scenes):
    keep = set(scenes)
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            s = row.get("scene")
            if s in keep:
                rows[s] = {m: float(row[m]) for m in METRICS}
    return rows


def boot_ci(d, seed=0, n=10000):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def gate_telemetry(out_root, label, scenes):
    fr_skip, atten, per_scene = [], [], {}
    for s in scenes:
        p = os.path.join(out_root, label, "preds", s, "state_gate.json")
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            d = json.load(f)
        g = np.array([fr[-1] for fr in d["frames"]], dtype=float)
        if len(g):
            per_scene[s] = float((g < 0.5).mean())
            fr_skip.append(per_scene[s])
            atten.append(float((1.0 - g).mean()))
    if not fr_skip:
        return None
    a = np.array(fr_skip)
    return {
        "n_scenes_with_telemetry": len(a),
        "mean_skip_frac": float(a.mean()),
        # soft mode: g is a continuous sigmoid, so also report the mean
        # attenuation 1-g (a heavily-attenuated scene can have skip_frac 0)
        "mean_attenuation": float(np.mean(atten)),
        "p50_skip_frac": float(np.percentile(a, 50)),
        "p90_skip_frac": float(np.percentile(a, 90)),
        "frac_scenes_any_skip": float((a > 0).mean()),
        "frac_scenes_skip_gt95": float((a > 0.95).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--out_root",
                    default="/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval")
    ap.add_argument("--clean_label", default=None)
    ap.add_argument("--clean_csv", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    with open(args.scene_list) as f:
        scenes = [ln.strip() for ln in f if ln.strip()]

    if args.clean_label:
        clean = label_rows(args.out_root, args.clean_label, scenes)
    else:
        path = args.clean_csv or os.path.join(
            args.out_root, "summary", "per_scene_augfull_lr1e5.csv")
        clean = csv_rows(path, scenes)
    print(f"clean reference: {len(clean)}/{len(scenes)} scenes")

    report = {}
    for label in args.labels:
        rows = label_rows(args.out_root, label, scenes)
        paired = sorted(set(rows) & set(clean))
        print(f"\n=== {label}: {len(rows)} scored, {len(paired)} paired ===")
        if not paired:
            print("  NO DATA -- label empty or misspelled; skipping")
            report[label] = {"n_paired": 0, "metrics": None, "error": "NO_DATA"}
            continue
        if len(rows) < len(scenes):
            # Cross-arm comparisons need a shared scene basis; a partially
            # scored label silently shifts its mean (house rule from
            # noise_oracle_analyze.py, which hard-exits on this).
            print(f"  WARNING: partial coverage {len(rows)}/{len(scenes)} -- "
                  f"cross-arm comparisons are NOT apples-to-apples")
        rep = {"n_paired": len(paired), "metrics": {}}
        for m in METRICS:
            c = np.array([clean[s][m] for s in paired])
            v = np.array([rows[s][m] for s in paired])
            ok = np.isfinite(c) & np.isfinite(v)
            c, v = c[ok], v[ok]
            d = v - c
            lo, hi = boot_ci(d)
            sign = -1.0 if m in HIGHER_BETTER else 1.0
            wins = int(((v - c) * sign < 0).sum())
            loss = int(((v - c) * sign > 0).sum())
            ties = int((v == c).sum())
            noop = (d == 0).all()
            better = None if noop else bool((d.mean() * sign) < 0)
            ci_excl0 = (lo > 0) or (hi < 0)
            rep["metrics"][m] = dict(
                n=int(ok.sum()), clean_mean=float(c.mean()),
                gated_mean=float(v.mean()), delta=float(d.mean()),
                ci=[lo, hi], wins=wins, ties=ties, losses=loss,
                better=better, significant=bool(ci_excl0))
            tag = "no-op " if noop else ("BETTER" if better else "worse ")
            sig = "*" if ci_excl0 else " "
            print(f"  {m:9s} clean {c.mean():+.6f}  gated {v.mean():+.6f}  "
                  f"delta {d.mean():+.6f} CI[{lo:+.6f},{hi:+.6f}]{sig} {tag} "
                  f"w/t/l {wins}/{ties}/{loss}")
        tel = gate_telemetry(args.out_root, label, scenes)
        if tel:
            rep["telemetry"] = tel
            print(f"  gate: mean skip {tel['mean_skip_frac']:.3f}, "
                  f"atten {tel['mean_attenuation']:.3f}, "
                  f"p90 {tel['p90_skip_frac']:.3f}, "
                  f"any-skip scenes {tel['frac_scenes_any_skip']:.3f}, "
                  f">95%-skip scenes {tel['frac_scenes_skip_gt95']:.3f}")
        report[label] = rep

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
