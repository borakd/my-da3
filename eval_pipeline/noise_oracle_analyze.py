#!/usr/bin/env python
"""Noise-tolerance oracle: compute the B1-B4 pre-registered readouts (Step 1).

Reads the diag_noise_* per-scene eval rows (camera_id==ALL, local_timestep==
MEAN — identical reduction to aggregate_results.py), pairs them per scene
against diag_noise_clean, and evaluates:

  B1 SHAPE   SHALLOW if ATE at 0.5*sigma_ref (white) loses <=25% of the
             subset B->A ATE gap; CLIFF if >=50% of the gap is lost by
             0.25*sigma_ref; else INTERMEDIATE.
  B2 POISON  per mode, smallest multiplier where noisy-B mean ATE >= arm A's
             subset mean ATE.
  B3 DRIFT   at 0.5*sigma_ref: D_ATE = dATE_walk/dATE_white, D_RPEt analog;
             drift-anchoring dominant iff D_ATE >= 2 and D_RPEt < 1.5.
  B4 NOTE    exposure-bias record, emitted when B1 == CLIFF.

Every delta carries a paired per-scene SE and a seeded percentile bootstrap CI.
diag rows never touch averages_table.csv.
"""
import argparse
import csv
import json
import os

import numpy as np

METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
MULTS = ["0125", "0250", "0500", "1000"]
MULTV = {"0125": 0.125, "0250": 0.25, "0500": 0.5, "1000": 1.0}


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


def paired_delta(noisy, clean, metric, seed=0):
    scenes = sorted(set(noisy) & set(clean))
    d = np.array([noisy[s][metric] - clean[s][metric] for s in scenes])
    d = d[np.isfinite(d)]
    if not len(d):
        return None
    rng = np.random.RandomState(seed)
    boots = [
        float(np.mean(d[rng.randint(0, len(d), len(d))])) for _ in range(2000)
    ]
    return dict(
        n=int(len(d)),
        mean=float(np.mean(d)),
        se=float(np.std(d, ddof=1) / np.sqrt(len(d))),
        ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval")
    ap.add_argument("--subset", default="eval_pipeline/noise_oracle_subset_430.txt")
    ap.add_argument("--a_per_scene", default="/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/summary/per_scene_augfull_lr1e5.csv")
    ap.add_argument("--calibration", default="eval_pipeline/evidence/noise_oracle_calibration.json")
    ap.add_argument("--out", default="eval_pipeline/evidence/noise_curve.json")
    a = ap.parse_args()

    scenes = [ln.strip() for ln in open(a.subset) if ln.strip()]
    clean = label_rows(a.out_root, "diag_noise_clean", scenes)

    arm_a = {}
    with open(a.a_per_scene) as f:
        for row in csv.DictReader(f):
            if row["scene"] in set(scenes):
                arm_a[row["scene"]] = {m: float(row[m]) for m in METRICS}

    common = sorted(set(clean) & set(arm_a))
    a_ate = float(np.nanmean([arm_a[s]["ate"] for s in common]))
    b_ate = float(np.nanmean([clean[s]["ate"] for s in common]))
    gap = a_ate - b_ate

    points, missing = {}, []
    for mode in ("white", "walk"):
        for mtag in MULTS:
            lb = f"diag_noise_{mode}_m{mtag}"
            rows = label_rows(a.out_root, lb, scenes)
            if len(rows) < 0.95 * len(scenes):
                missing.append((lb, len(rows)))
            pt = {"n_scenes": len(rows)}
            for m in METRICS:
                pt[f"mean_{m}"] = float(np.nanmean([r[m] for r in rows.values()])) if rows else float("nan")
                pd = paired_delta(rows, clean, m)
                pt[f"delta_{m}"] = pd
            pt["gap_fraction_ate"] = (
                pt["delta_ate"]["mean"] / gap if pt.get("delta_ate") and gap else None
            )
            points[lb] = pt

    def gf(mode, mtag):
        p = points.get(f"diag_noise_{mode}_m{mtag}", {})
        return p.get("gap_fraction_ate")

    def dm(mode, mtag, metric):
        p = points.get(f"diag_noise_{mode}_m{mtag}", {})
        d = p.get(f"delta_{metric}")
        return d["mean"] if d else None

    b1 = "UNDECIDED"
    if gf("white", "0500") is not None and gf("white", "0250") is not None:
        if gf("white", "0250") >= 0.5:
            b1 = "CLIFF"
        elif gf("white", "0500") <= 0.25:
            b1 = "SHALLOW"
        else:
            b1 = "INTERMEDIATE"

    b2 = {}
    for mode in ("white", "walk"):
        b2[mode] = None
        for mtag in MULTS:
            p = points.get(f"diag_noise_{mode}_m{mtag}", {})
            if p.get("mean_ate") is not None and np.isfinite(p.get("mean_ate", np.nan)):
                if p["mean_ate"] >= a_ate:
                    b2[mode] = MULTV[mtag]
                    break

    d_ate = d_rpet = None
    dw, db = dm("white", "0500", "ate"), dm("walk", "0500", "ate")
    if dw and db and dw != 0:
        d_ate = db / dw
    dwt, dbt = dm("white", "0500", "rpe_trans"), dm("walk", "0500", "rpe_trans")
    if dwt and dbt and dwt != 0:
        d_rpet = dbt / dwt
    b3 = dict(
        D_ATE=d_ate,
        D_RPEt=d_rpet,
        drift_anchoring_dominant=(
            d_ate is not None and d_rpet is not None and d_ate >= 2.0 and d_rpet < 1.5
        ),
    )

    cal = json.load(open(a.calibration))
    out = dict(
        subset=a.subset,
        n_subset=len(scenes),
        n_paired=len(common),
        calibration={k: cal[k] for k in ("sigma_ref_t", "sigma_ref_r_deg")},
        arm_a_subset_ate=a_ate,
        clean_b_subset_ate=b_ate,
        b_to_a_ate_gap=gap,
        points=points,
        missing_or_partial=missing,
        B1_shape=b1,
        B2_sigma_poison=b2,
        B3_drift=b3,
        B4_exposure_bias_note=(
            "CLIFF verdict: B's brittleness under conditioning error is a "
            "teacher-forcing artifact; standing prohibition on teacher-forcing "
            "GT into any conditioning path during training. The refine arm "
            "trains on its own in-loop outputs (satisfied by design)."
            if b1 == "CLIFF" else None
        ),
    )
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(dict(
        B1_shape=b1, B2_sigma_poison=b2, B3_drift=b3,
        arm_a_subset_ate=a_ate, clean_b_subset_ate=b_ate, gap=gap,
        gap_fraction_white={t: gf("white", t) for t in MULTS},
        gap_fraction_walk={t: gf("walk", t) for t in MULTS},
        missing=missing,
    ), indent=1))


if __name__ == "__main__":
    main()
