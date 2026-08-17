#!/usr/bin/env python
"""Build GT-oracle "bad frame" skip masks for the CUT3R memory-write-blocking
experiment.

For each scene in eval_pipeline/noise_oracle_subset_430.txt, reads per-frame
errors of the CLEAN (ungated) run from
  /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/diag_cg_log430/eval/<scene>/eval_depth_pose_metrics.csv
(rows with camera_id=='0' and numeric local_timestep; local_timestep t == view
index t) and emits three oracle masks, each {scene: [sorted frame indices to
BLOCK]}:

  1. eval_pipeline/evidence/cg_oracle_mask_ate15.json
       block frames with per-frame ate > 1.5 x scene median ate
  2. eval_pipeline/evidence/cg_oracle_mask_atetop20.json
       block the top 20% of frames by per-frame ate within each scene
  3. eval_pipeline/evidence/cg_oracle_mask_rpe2x.json
       block frames where rpe_trans > 2 x scene median rpe_trans OR
       rpe_rot > 2 x scene median rpe_rot (NaN never blocks)

Rules for all variants: never block frame 0; cap blocked fraction at 40% per
scene (if over cap, keep the worst frames by the variant's own criterion up to
the cap); every scene appears as a key (empty list allowed).
"""

import csv
import json
import math
import os
import sys

import numpy as np

REPO = "/gpfs/home/koc/koc821022/my-da3"
SCENE_LIST = os.path.join(REPO, "eval_pipeline", "noise_oracle_subset_430.txt")
EVAL_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/diag_cg_log430/eval"
OUT_DIR = os.path.join(REPO, "eval_pipeline", "evidence")

CAP_FRAC = 0.40
TOP_FRAC = 0.20


def load_scene(scene):
    """Return dict of numpy arrays (t, ate, rpe_trans, rpe_rot) for camera 0,
    numeric local_timestep rows, sorted by t. Returns None if CSV missing."""
    path = os.path.join(EVAL_ROOT, scene, "eval_depth_pose_metrics.csv")
    if not os.path.isfile(path):
        return None
    ts, ates, rts, rrs = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["camera_id"].strip() != "0":
                continue
            t_str = row["local_timestep"].strip()
            try:
                t = int(t_str)
            except ValueError:
                continue  # MEAN / ALL rows
            ts.append(t)
            ates.append(float(row["ate"]))
            rts.append(float(row["rpe_trans"]))
            rrs.append(float(row["rpe_rot"]))
    order = np.argsort(ts)
    return {
        "t": np.asarray(ts, dtype=np.int64)[order],
        "ate": np.asarray(ates, dtype=np.float64)[order],
        "rpe_trans": np.asarray(rts, dtype=np.float64)[order],
        "rpe_rot": np.asarray(rrs, dtype=np.float64)[order],
    }


def apply_rules(t, candidate_mask, severity, n_frames):
    """Common post-processing: drop frame 0, cap at CAP_FRAC of n_frames
    keeping the worst frames by `severity` (higher = worse).
    Returns (sorted list of ints, cap_hit: bool)."""
    keep = candidate_mask & (t != 0)
    idx = np.where(keep)[0]
    max_block = int(math.floor(CAP_FRAC * n_frames + 1e-9))
    cap_hit = len(idx) > max_block
    if cap_hit:
        # keep the worst max_block frames by severity
        idx = idx[np.argsort(-severity[idx], kind="stable")[:max_block]]
    return sorted(int(x) for x in t[idx]), cap_hit


def build_ate15(d):
    med = np.nanmedian(d["ate"])
    thr = 1.5 * med
    with np.errstate(invalid="ignore"):
        cand = d["ate"] > thr  # NaN compares False -> never blocks
    return apply_rules(d["t"], cand, d["ate"], len(d["t"]))


def build_atetop20(d):
    n = len(d["t"])
    k = int(math.floor(TOP_FRAC * n + 1e-9))
    ate = d["ate"].copy()
    ate[np.isnan(ate)] = -np.inf  # NaN never selected
    ate[d["t"] == 0] = -np.inf    # never block frame 0
    cand = np.zeros(n, dtype=bool)
    if k > 0:
        top = np.argsort(-ate, kind="stable")[:k]
        cand[top] = ate[top] > -np.inf
    return apply_rules(d["t"], cand, d["ate"], n)


def build_rpe2x(d):
    rt, rr = d["rpe_trans"], d["rpe_rot"]
    med_t = np.nanmedian(rt)
    med_r = np.nanmedian(rr)
    with np.errstate(invalid="ignore"):
        cand = (rt > 2.0 * med_t) | (rr > 2.0 * med_r)  # NaN -> False

    # severity for cap ranking: worst excess over the variant's own thresholds
    def ratio(v, med):
        out = np.full_like(v, -np.inf)
        ok = ~np.isnan(v)
        if med > 0:
            out[ok] = v[ok] / med
        else:  # degenerate median: any positive value is infinitely over
            out[ok] = np.where(v[ok] > 0, np.inf, 0.0)
        return out

    severity = np.maximum(ratio(rt, med_t), ratio(rr, med_r))
    return apply_rules(d["t"], cand, severity, len(d["t"]))


VARIANTS = {
    "cg_oracle_mask_ate15.json": build_ate15,
    "cg_oracle_mask_atetop20.json": build_atetop20,
    "cg_oracle_mask_rpe2x.json": build_rpe2x,
}


def main():
    with open(SCENE_LIST) as f:
        scenes = [ln.strip() for ln in f if ln.strip()]
    assert len(scenes) == 430, f"expected 430 scenes, got {len(scenes)}"

    data = {}
    missing = []
    for s in scenes:
        d = load_scene(s)
        if d is None or len(d["t"]) == 0:
            missing.append(s)
        else:
            # sanity: timesteps should be 0..n-1 exactly once
            assert d["t"][0] == 0 and np.all(np.diff(d["t"]) == 1), \
                f"non-contiguous timesteps in {s}"
        data[s] = d
    if missing:
        print(f"WARNING: {len(missing)} scenes missing/empty CSV -> empty masks: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

    os.makedirs(OUT_DIR, exist_ok=True)
    all_stats = {}
    for fname, builder in VARIANTS.items():
        masks, fracs, cap_hits, zeros = {}, [], 0, 0
        for s in scenes:
            d = data[s]
            if d is None or len(d["t"]) == 0:
                masks[s] = []
                fracs.append(0.0)
                zeros += 1
                continue
            blocked, cap_hit = builder(d)
            assert 0 not in blocked
            n = len(d["t"])
            assert len(blocked) <= math.floor(CAP_FRAC * n + 1e-9)
            masks[s] = blocked
            fracs.append(len(blocked) / n)
            cap_hits += int(cap_hit)
            zeros += int(len(blocked) == 0)

        out_path = os.path.join(OUT_DIR, fname)
        with open(out_path, "w") as f:
            json.dump(masks, f, indent=0, sort_keys=True)
        fr = np.asarray(fracs)
        stats = {
            "scenes_covered": len(masks),
            "mean_blocked_frac": float(fr.mean()),
            "median_blocked_frac": float(np.median(fr)),
            "scenes_at_cap": cap_hits,
            "scenes_zero_blocked": zeros,
        }
        all_stats[fname] = stats
        print(f"{fname}: scenes={stats['scenes_covered']} "
              f"mean_frac={stats['mean_blocked_frac']:.4f} "
              f"median_frac={stats['median_blocked_frac']:.4f} "
              f"at_cap={stats['scenes_at_cap']} "
              f"zero_blocked={stats['scenes_zero_blocked']} -> {out_path}")

    print(json.dumps(all_stats, indent=2))
    return all_stats


if __name__ == "__main__":
    main()
