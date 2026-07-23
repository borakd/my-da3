#!/usr/bin/env python3
"""Aggregate per-scene sim3 eval CSVs into table averages, under BOTH per-scene
pose aggregations:

  - mean: nanmean over timesteps  (what the Jul-7 averages_finetune_vs_regular
          _table_sim3.tex rows used, via the then-default --pose_agg / the
          sim3_pose.csv files)
  - rmse: nan-RMSE over timesteps (eval_depth_poses.py's CURRENT default
          --pose_agg rmse)

Depth metrics (absrel, a1) are aggregated with nanmean over timesteps in both
modes (matching the script's own MEAN row, which is identical across eras).

Per-timestep rows are read from <out_root>/<label>/<evaldir>/<scene>/
eval_depth_pose_metrics.csv; stored MEAN rows are IGNORED and re-derived, so
old (mean-era) and new (rmse-era) CSVs are handled uniformly: per camera,
aggregate over its finite timestep values; then average across cameras (the
script's ALL row construction — trivially one camera for the wrist stream).

Validation: where <label>/sim3_pose.csv exists, the recomputed mean-agg pose
values are compared against it per scene (they came from the same per-timestep
values, so max|diff| should be ~1e-12).

Usage:
  python aggregate_sim3_both.py --out_root .../cut3r_eval --scene_list .../scene_list.txt \
      --spec regular=eval_sim3 final=eval_sim3 captain_ray_gt_last=eval
"""
import argparse
import csv
import math
import os

DEPTH = ["absrel", "a1"]
POSE = ["ate", "rpe_trans", "rpe_rot"]


def _finite(rows, col):
    out = []
    for r in rows:
        try:
            v = float(r[col])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def scene_metrics(csv_path):
    """Return ({metric: value} per aggregation mode, stored ALL/MEAN row dict),
    or None if unreadable. The stored row is what the eval script itself wrote
    (mean-agg in old CSVs, rmse-agg in current-defaults CSVs)."""
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    cams = {}
    stored = None
    for r in rows:
        if r.get("camera_id") == "ALL" and r.get("local_timestep") == "MEAN":
            stored = {}
            for m in DEPTH + POSE:
                try:
                    stored[m] = float(r[m])
                except (KeyError, TypeError, ValueError):
                    stored[m] = float("nan")
            continue
        if r.get("local_timestep") == "MEAN" or r.get("camera_id") == "ALL":
            continue
        cams.setdefault(r["camera_id"], []).append(r)
    if not cams:
        return None
    per_cam = {"mean": [], "rmse": []}
    for cam_rows in cams.values():
        agg = {"mean": {}, "rmse": {}}
        for m in DEPTH:
            v = _finite(cam_rows, m)
            d = sum(v) / len(v) if v else float("nan")
            agg["mean"][m] = agg["rmse"][m] = d
        for m in POSE:
            v = _finite(cam_rows, m)
            agg["mean"][m] = sum(v) / len(v) if v else float("nan")
            agg["rmse"][m] = math.sqrt(sum(x * x for x in v) / len(v)) if v else float("nan")
        for mode in ("mean", "rmse"):
            per_cam[mode].append(agg[mode])
    out = {}
    for mode in ("mean", "rmse"):
        out[mode] = {}
        for m in DEPTH + POSE:
            vals = [c[m] for c in per_cam[mode] if math.isfinite(c[m])]
            out[mode][m] = sum(vals) / len(vals) if vals else float("nan")
    return out, stored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--spec", nargs="+", required=True,
                    help="label=evaldirname pairs, e.g. regular=eval_sim3 captain_ray_gt_last=eval")
    ap.add_argument("--summary_dir", default=None,
                    help="default: <out_root>/summary")
    ap.add_argument("--check_mean_row", choices=["none", "mean", "rmse"], default="none",
                    help="Cross-check the CSVs' own stored ALL/MEAN row against our "
                    "recomputed aggregation of this mode (old CSVs stored mean-agg, "
                    "current-defaults CSVs store rmse-agg).")
    args = ap.parse_args()

    summary_dir = args.summary_dir or os.path.join(args.out_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    scenes = [l.strip() for l in open(args.scene_list) if l.strip()]

    results = {}
    for spec in args.spec:
        label, evaldir = spec.split("=", 1)
        per_scene = {}
        missing = []
        row_dmax, row_dn = 0.0, 0
        for scene in scenes:
            p = os.path.join(args.out_root, label, evaldir, scene,
                             "eval_depth_pose_metrics.csv")
            res = scene_metrics(p)
            if res is None:
                missing.append(scene)
                continue
            sm, stored = res
            per_scene[scene] = sm
            if args.check_mean_row != "none" and stored is not None:
                for m in DEPTH + POSE:
                    mine = sm[args.check_mean_row][m]
                    theirs = stored[m]
                    if math.isfinite(mine) and math.isfinite(theirs):
                        row_dmax = max(row_dmax, abs(mine - theirs))
                        row_dn += 1

        # validation vs stored per-scene mean-agg sim3 values
        val_path = os.path.join(args.out_root, label, "sim3_pose.csv")
        vmax, vn = 0.0, 0
        if os.path.isfile(val_path):
            ref = {r["scene"]: r for r in csv.DictReader(open(val_path))}
            for scene, sm in per_scene.items():
                r = ref.get(scene)
                if not r:
                    continue
                for m, col in [("ate", "ate_sim3"), ("rpe_trans", "rpe_trans_sim3"),
                               ("rpe_rot", "rpe_rot_sim3")]:
                    try:
                        d = abs(sm["mean"][m] - float(r[col]))
                    except (KeyError, ValueError):
                        continue
                    if math.isfinite(d):
                        vmax = max(vmax, d)
                        vn += 1

        # per-scene csv (both aggs)
        out_csv = os.path.join(summary_dir, f"per_scene_{label}_sim3both.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene"] + DEPTH + [f"{m}_mean" for m in POSE] +
                       [f"{m}_rmse" for m in POSE])
            for scene in scenes:
                sm = per_scene.get(scene)
                if sm is None:
                    continue
                w.writerow([scene] + [sm["mean"][m] for m in DEPTH] +
                           [sm["mean"][m] for m in POSE] +
                           [sm["rmse"][m] for m in POSE])

        # averages (nanmean across scenes)
        avgs = {}
        for mode in ("mean", "rmse"):
            avgs[mode] = {}
            for m in DEPTH + POSE:
                vals = [sm[mode][m] for sm in per_scene.values()
                        if math.isfinite(sm[mode][m])]
                avgs[mode][m] = sum(vals) / len(vals) if vals else float("nan")
        results[label] = {"n": len(per_scene), "avgs": avgs,
                          "missing": len(missing), "vmax": vmax, "vn": vn}
        print(f"{label:24s} n={len(per_scene):4d} missing={len(missing):4d} "
              f"validated={vn} max|mean-agg diff vs sim3_pose.csv|={vmax:.2e}")
        if args.check_mean_row != "none":
            print(f"    stored-MEAN-row check ({args.check_mean_row}-agg): "
                  f"n={row_dn} max|diff|={row_dmax:.2e}")
        if missing[:5]:
            print(f"    missing sample: {missing[:5]}")
        for mode in ("mean", "rmse"):
            a = avgs[mode]
            print(f"    [{mode}] absrel {a['absrel']:.4f}  a1 {a['a1']:.4f}  "
                  f"ate {a['ate']:.4f}  rpe_t {a['rpe_trans']:.4f}  rpe_r {a['rpe_rot']:.4f}")

    # machine-readable summary
    out = os.path.join(summary_dir, "averages_sim3_both.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "agg", "n_scenes"] + DEPTH + POSE)
        for label, r in results.items():
            for mode in ("mean", "rmse"):
                w.writerow([label, mode, r["n"]] +
                           [f"{r['avgs'][mode][m]:.6f}" for m in DEPTH + POSE])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
