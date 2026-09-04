#!/usr/bin/env python3
"""Aggregate per-scene eval CSVs into per-setup averages and a comparison table.

For each setup (best/final/regular):
  - read every <eval_base>/<scene>/eval_depth_pose_metrics.csv
  - take the scene-level summary row (camera_id == "ALL", local_timestep == "MEAN")
  - collect per-scene metrics -> per_scene_<label>.csv
  - average (nanmean) each metric across all scenes
Emit averages_table.{csv,md,txt} comparing the setups.

A label whose tree also carries <label>/eval_gru (written by the prev_pred_gru
worker: the GRU-refined pose scored by the identical eval script) automatically
gets a second row "<label>_gru", aggregated with exactly the same math.
"""
import argparse
import csv
import os

import numpy as np

METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]


def read_scene_summary(csv_path):
    """Return dict metric->float for the ALL/MEAN row, or None."""
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    chosen = None
    for r in rows:
        if r.get("camera_id") == "ALL" and r.get("local_timestep") == "MEAN":
            chosen = r
            break
    if chosen is None:
        # fall back to single-camera MEAN row
        for r in rows:
            if r.get("local_timestep") == "MEAN":
                chosen = r
                break
    if chosen is None:
        return None
    out = {}
    for m in METRICS:
        v = chosen.get(m, "")
        try:
            out[m] = float(v)
        except (TypeError, ValueError):
            out[m] = float("nan")
    return out


def aggregate_setup(label, eval_base, summary_dir, scene_list):
    per_scene = []
    for scene in scene_list:
        csv_path = os.path.join(eval_base, scene, "eval_depth_pose_metrics.csv")
        if not os.path.isfile(csv_path):
            continue
        s = read_scene_summary(csv_path)
        if s is None:
            continue
        per_scene.append((scene, s))

    # write per-scene csv
    out_csv = os.path.join(summary_dir, f"per_scene_{label}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene"] + METRICS)
        for scene, s in per_scene:
            w.writerow([scene] + [s[m] for m in METRICS])

    # averages (nanmean) + valid counts
    avgs = {}
    counts = {}
    for m in METRICS:
        vals = np.array([s[m] for _, s in per_scene], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        counts[m] = int(finite.size)
        avgs[m] = float(np.mean(finite)) if finite.size else float("nan")
    return {"label": label, "n_scenes": len(per_scene), "avgs": avgs, "counts": counts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True,
                    help="cut3r_eval root containing <label>/eval/")
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--labels", nargs="+", default=["best", "final", "regular"])
    args = ap.parse_args()

    summary_dir = os.path.join(args.out_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    with open(args.scene_list) as f:
        scene_list = [ln.strip() for ln in f if ln.strip()]

    results = []
    for label in args.labels:
        eval_base = os.path.join(args.out_root, label, "eval")
        res = aggregate_setup(label, eval_base, summary_dir, scene_list)
        results.append(res)
        print(f"{label}: aggregated {res['n_scenes']} scenes")
        # GRU-refined pose trajectory of the same run (prev_pred_gru arms):
        # scored per scene by the identical eval script into <label>/eval_gru,
        # aggregated here with the identical math as a companion row.
        gru_base = os.path.join(args.out_root, label, "eval_gru")
        if os.path.isdir(gru_base):
            gres = aggregate_setup(f"{label}_gru", gru_base, summary_dir, scene_list)
            results.append(gres)
            print(f"{label}_gru: aggregated {gres['n_scenes']} scenes")

    nice = {
        "absrel": "AbsRel (depth, lower better)",
        "a1": "delta<1.25 a1 (depth, higher better)",
        "ate": "ATE (pose, lower better)",
        "rpe_trans": "RPE-trans (pose, lower better)",
        "rpe_rot": "RPE-rot deg (pose, lower better)",
    }
    label_names = {
        "best": "Finetuned (best ckpt)",
        "final": "Finetuned (final ckpt)",
        "regular": "Regular CUT3R (pretrained)",
    }

    # CSV table
    table_csv = os.path.join(summary_dir, "averages_table.csv")
    with open(table_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setup", "n_scenes"] + METRICS +
                   [f"n_valid_{m}" for m in METRICS])
        for r in results:
            w.writerow(
                [r["label"], r["n_scenes"]]
                + [f"{r['avgs'][m]:.6f}" for m in METRICS]
                + [r["counts"][m] for m in METRICS]
            )

    # Markdown table
    table_md = os.path.join(summary_dir, "averages_table.md")
    with open(table_md, "w") as f:
        f.write("# CUT3R test-set evaluation (averaged over all test scenes)\n\n")
        f.write("Metrics are per-scene summary rows averaged (unweighted) across scenes. "
                "Eval: eval_depth_poses.py default args "
                "(depth median-scale-aligned, pose sim3-aligned, RMSE-reduced, "
                "single stream).\n\n")
        header = ["Setup", "Scenes"] + [nice[m] for m in METRICS]
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in results:
            row = [label_names.get(r["label"], r["label"]), str(r["n_scenes"])] + \
                  [f"{r['avgs'][m]:.4f}" for m in METRICS]
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n_Valid-scene counts per metric:_\n\n")
        f.write("| Setup | " + " | ".join(METRICS) + " |\n")
        f.write("|" + "|".join(["---"] * (len(METRICS) + 1)) + "|\n")
        for r in results:
            f.write("| " + label_names.get(r["label"], r["label"]) + " | " +
                    " | ".join(str(r["counts"][m]) for m in METRICS) + " |\n")

    # Plain text table
    table_txt = os.path.join(summary_dir, "averages_table.txt")
    with open(table_txt, "w") as f:
        f.write("CUT3R test-set evaluation -- averaged over all test scenes\n")
        f.write("Eval: eval_depth_poses.py default args "
                "(depth median-scale-aligned, pose sim3-aligned, RMSE-reduced).\n\n")
        colw = 26
        head = f"{'setup':<28}{'n_scenes':>10}" + "".join(f"{m:>14}" for m in METRICS)
        f.write(head + "\n")
        f.write("-" * len(head) + "\n")
        for r in results:
            line = f"{label_names.get(r['label'], r['label']):<28}{r['n_scenes']:>10}"
            line += "".join(f"{r['avgs'][m]:>14.4f}" for m in METRICS)
            f.write(line + "\n")

    print("\nWrote:")
    print(" ", table_csv)
    print(" ", table_md)
    print(" ", table_txt)
    print()
    with open(table_txt) as f:
        print(f.read())


if __name__ == "__main__":
    main()
