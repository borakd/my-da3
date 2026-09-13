#!/usr/bin/env python3
"""Flag "harmful" frames of one scene from a clean run's per-frame eval CSV.

Reads eval_depth_pose_metrics.csv (camera 0, numeric local_timestep rows; the
MEAN/ALL rows are ignored) and writes one skip json per rule, in the
{scene: [sorted frame indices]} schema the worker's --skip_frames_file expects.

Rules (see HARMFUL_FRAME_SKIP_PLAN.md §5; the first two mirror
my-da3/eval_pipeline/cg_build_oracle_masks.py so the pilot is comparable):
  ate15     ate > 1.5*median(ate)
  rpe2x     rpe_trans > 2*median  OR  rpe_rot > 2*median
  absrel15  absrel > 1.5*median(absrel)          (depth-only, new)
  any       union of the three                    (hits the cap on most scenes)

Common rules: frame 0 is never flagged; NaN never flags; the flagged fraction
is capped at CAP_FRAC per scene (keep the worst by a rank-sum of the
normalised criteria when over the cap).
"""
import argparse
import csv
import json
import math
import os

import numpy as np

CAP_FRAC = 0.40
METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]


def load_rows(csv_path):
    out = {m: [] for m in METRICS}
    ts = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["camera_id"].strip() != "0":
                continue
            try:
                t = int(r["local_timestep"].strip())
            except ValueError:
                continue
            ts.append(t)
            for m in METRICS:
                try:
                    out[m].append(float(r[m]))
                except (TypeError, ValueError):
                    out[m].append(math.nan)
    order = np.argsort(ts)
    return np.asarray(ts)[order], {m: np.asarray(v)[order] for m, v in out.items()}


def _ratio(v):
    """v / nanmedian(v); NaN stays NaN; a zero median yields NaN (never flags)."""
    med = np.nanmedian(v) if np.any(np.isfinite(v)) else math.nan
    if not np.isfinite(med) or med <= 0:
        return np.full_like(v, math.nan, dtype=np.float64)
    return v / med


def flag(ts, m, rule):
    c_ate = (_ratio(m["ate"]), 1.5)
    c_rpe = [(_ratio(m["rpe_trans"]), 2.0), (_ratio(m["rpe_rot"]), 2.0)]
    c_abs = (_ratio(m["absrel"]), 1.5)
    crit = {"ate15": [c_ate], "rpe2x": c_rpe, "absrel15": [c_abs],
            "any": [c_ate] + c_rpe + [c_abs]}.get(rule)
    if crit is None:
        raise ValueError(rule)

    bad = np.zeros(len(ts), dtype=bool)
    score = np.zeros(len(ts), dtype=np.float64)
    for r, k in crit:
        hit = np.isfinite(r) & (r > k)
        bad |= hit
        score += np.where(np.isfinite(r), r / k, 0.0)
    bad[ts == 0] = False

    cap = int(math.floor(CAP_FRAC * len(ts)))
    if bad.sum() > cap:
        idx = np.where(bad)[0]
        keep = idx[np.argsort(-score[idx])][:cap]
        bad = np.zeros(len(ts), dtype=bool)
        bad[keep] = True
    return [int(t) for t in ts[bad]]


def runs(flagged):
    """Lengths of maximal runs of consecutive flagged indices."""
    out, cur, prev = [], 0, None
    for t in flagged:
        if prev is not None and t == prev + 1:
            cur += 1
        else:
            if cur:
                out.append(cur)
            cur = 1
        prev = t
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="clean run's eval_depth_pose_metrics.csv")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rules", nargs="+", default=["ate15", "rpe2x", "absrel15", "any"])
    args = ap.parse_args()

    ts, m = load_rows(args.csv)
    if len(ts) < 2:
        raise SystemExit(f"only {len(ts)} numeric rows in {args.csv}")
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {"scene": args.scene, "n_frames": int(len(ts)), "rules": {}}
    for rule in args.rules:
        flagged = flag(ts, m, rule)
        path = os.path.join(args.out_dir, f"skip_{rule}.json")
        with open(path, "w") as f:
            json.dump({args.scene: flagged}, f)
        rl = runs(flagged)
        summary["rules"][rule] = {
            "n_flagged": len(flagged),
            "frac": len(flagged) / len(ts),
            "n_runs": len(rl),
            "longest_run": max(rl) if rl else 0,
            "first_flagged": flagged[0] if flagged else None,
            "json": path,
        }
        print(f"[flag] {rule}: {len(flagged)}/{len(ts)} frames "
              f"({100*len(flagged)/len(ts):.1f}%), {len(rl)} runs, longest {max(rl) if rl else 0}, "
              f"first {flagged[0] if flagged else None} -> {path}", flush=True)
    with open(os.path.join(args.out_dir, "flag_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
