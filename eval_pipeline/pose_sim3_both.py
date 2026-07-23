#!/usr/bin/env python3
"""Fast POSE-ONLY Sim3 ATE + RPE over saved predictions, under BOTH per-scene
aggregations:

  mean: nanmean over per-frame values   (eval_depth_poses.py's OLD default
        --pose_agg; what sim3_pose.csv / the Jul-7 tex rows used)
  rmse: nan-RMSE over per-frame values  (eval_depth_poses.py's CURRENT default)

Same faithful pose path as pose_rpe_sim3.py (align=sim3, num_cameras=1,
pose_type=c2w): Umeyama-with-scale on trajectory centers, aligned =
(R@T_R, s*(R@t)+t0), per-frame ATE, consecutive-pair RPE (E = inv(d_gt)@d_pr).
Loads ONLY camera/*.npz poses (never depth); groups work BY SCENE so each
scene's GT poses are loaded once and shared across all labels.

Validation: where <label>/eval_sim3/<scene>/eval_depth_pose_metrics.csv exists
(full re-eval output, any era — per-timestep rows are era-independent), the
PER-TIMESTEP ate/rpe values are compared elementwise against ours, which
validates the whole pose path; the aggregations are then pure arithmetic.
Existing <label>/sim3_pose.csv (mean era) is also compared against our
mean-agg values. Writes <label>/sim3_pose_both.csv; touches nothing else.
"""
import argparse
import csv
import glob
import os
import re
from multiprocessing import Pool

import numpy as np

_INT = re.compile(r"^\d+$")


def _resolve_cam(root):
    for d in ("camera", "cam"):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            return p
    return os.path.join(root, "camera")


def _load_poses_sorted(cam_dir):
    poses = {}
    for f in glob.glob(os.path.join(cam_dir, "*.npz")):
        stem = os.path.splitext(os.path.basename(f))[0]
        if not _INT.match(stem):
            continue
        d = np.load(f)
        p = np.asarray(d["pose"], dtype=np.float64)
        if p.shape == (3, 4):
            o = np.eye(4)
            o[:3, :4] = p
            p = o
        poses[int(stem)] = p
    return poses


def _umeyama(src, dst, with_scale=True):
    n = src.shape[0]
    mu_s = src.mean(0); mu_d = dst.mean(0)
    src_c = src - mu_s; dst_c = dst - mu_d
    cov = (dst_c.T @ src_c) / max(n, 1)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    if with_scale:
        var_src = np.mean(np.sum(src_c * src_c, axis=1))
        scale = 1.0 if var_src <= 1e-15 else float(np.trace(np.diag(S) @ D) / var_src)
    else:
        scale = 1.0
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def _rotation_angle_deg(R):
    tr = float(np.trace(R))
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _agg(vals, mode):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if mode == "mean":
        return float(np.mean(arr))
    return float(np.sqrt(np.mean(arr * arr)))


def _sim3_pose_series(pred, gt):
    """Per-frame ATE list + consecutive-pair RPE lists (sim3-aligned)."""
    ts = sorted(set(pred.keys()) & set(gt.keys()))
    if not ts:
        return [], [], []
    pc = np.stack([pred[t][:3, 3] for t in ts], axis=0)
    gc = np.stack([gt[t][:3, 3] for t in ts], axis=0)
    s, R, tt = _umeyama(pc, gc, with_scale=True)
    aligned = {}
    for t in ts:
        T = pred[t]
        Tout = np.eye(4, dtype=np.float64)
        Tout[:3, :3] = R @ T[:3, :3]
        Tout[:3, 3] = s * (R @ T[:3, 3]) + tt
        aligned[t] = Tout
    ate = [float(np.linalg.norm(aligned[t][:3, 3] - gt[t][:3, 3])) for t in ts]
    rpe_trans, rpe_rot = [], []
    for i in range(len(ts) - 1):
        t0, t1 = ts[i], ts[i + 1]
        d_gt = np.linalg.inv(gt[t0]) @ gt[t1]
        d_pr = np.linalg.inv(aligned[t0]) @ aligned[t1]
        E = np.linalg.inv(d_gt) @ d_pr
        rpe_trans.append(float(np.linalg.norm(E[:3, 3])))
        rpe_rot.append(_rotation_angle_deg(E[:3, :3]))
    return ate, rpe_trans, rpe_rot


def _read_eval_timesteps(csv_path):
    """Per-timestep (ate, rpe_trans, rpe_rot) lists from a full-eval CSV
    (camera 0 rows, numeric timesteps, in order)."""
    try:
        rows = list(csv.DictReader(open(csv_path)))
    except Exception:
        return None
    out = []
    for r in rows:
        if r.get("camera_id") != "0" or not _INT.match(r.get("local_timestep") or ""):
            continue

        def _f(k):
            try:
                return float(r[k])
            except (TypeError, ValueError, KeyError):
                return float("nan")

        out.append((int(r["local_timestep"]), _f("ate"), _f("rpe_trans"), _f("rpe_rot")))
    out.sort()
    return out or None


def _one_scene(task):
    scene, gt_cam, label_dirs = task          # label_dirs: [(label, pred_cam, eval_csv)]
    gt = _load_poses_sorted(gt_cam)
    out = []
    for label, pred_cam, eval_csv in label_dirs:
        pred = _load_poses_sorted(pred_cam)
        ate, rpe_t, rpe_r = _sim3_pose_series(pred, gt)
        aggs = tuple(_agg(v, m) for m in ("mean", "rmse") for v in (ate, rpe_t, rpe_r))
        # elementwise validation vs an existing full-eval CSV, if present
        vmax = float("nan")
        if eval_csv and os.path.isfile(eval_csv):
            ref = _read_eval_timesteps(eval_csv)
            if ref is not None and len(ref) == len(ate):
                diffs = []
                for t, r_ate, r_rt, r_rr in ref:
                    if t < len(ate) and np.isfinite(r_ate) and np.isfinite(ate[t]):
                        diffs.append(abs(r_ate - ate[t]))
                    if 1 <= t <= len(rpe_t) and np.isfinite(r_rt):
                        diffs.append(abs(r_rt - rpe_t[t - 1]))
                    if 1 <= t <= len(rpe_r) and np.isfinite(r_rr):
                        diffs.append(abs(r_rr - rpe_r[t - 1]))
                if diffs:
                    vmax = max(diffs)
        out.append((label, aggs, vmax))
    return scene, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--scenes_root", required=True)
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    scenes = [l.strip() for l in open(args.scene_list) if l.strip()]
    tasks = []
    for scene in scenes:
        gt_cam = _resolve_cam(os.path.join(args.scenes_root, scene, "dense"))
        label_dirs = []
        for label in args.labels:
            pred_cam = _resolve_cam(os.path.join(args.out_root, label, "preds", scene))
            if not os.path.isdir(pred_cam):
                continue
            eval_csv = os.path.join(args.out_root, label, "eval_sim3", scene,
                                    "eval_depth_pose_metrics.csv")
            label_dirs.append((label, pred_cam, eval_csv))
        if label_dirs:
            tasks.append((scene, gt_cam, label_dirs))
    print(f"scene-tasks: {len(tasks)}  workers: {args.workers}", flush=True)

    results = {lbl: [] for lbl in args.labels}
    vdiffs = {lbl: [] for lbl in args.labels}
    with Pool(args.workers) as p:
        for i, (scene, out) in enumerate(p.imap_unordered(_one_scene, tasks, chunksize=4)):
            for label, aggs, vmax in out:
                results[label].append((scene,) + aggs)
                if np.isfinite(vmax):
                    vdiffs[label].append(vmax)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} scenes", flush=True)

    print("\nVALIDATION (elementwise per-timestep vs existing eval_sim3 CSVs):")
    for lbl in args.labels:
        d = np.asarray(vdiffs[lbl])
        if d.size:
            print(f"  {lbl}: n_scenes={d.size} max|diff|={d.max():.3e} "
                  f">1e-6:{int((d > 1e-6).sum())}", flush=True)
        else:
            print(f"  {lbl}: no eval_sim3 CSVs to validate against", flush=True)

    # secondary validation: mean-agg vs existing sim3_pose.csv (mean era)
    for lbl in args.labels:
        ref_path = os.path.join(args.out_root, lbl, "sim3_pose.csv")
        if not os.path.isfile(ref_path):
            continue
        ref = {r["scene"]: r for r in csv.DictReader(open(ref_path))}
        dmax, n = 0.0, 0
        for row in results[lbl]:
            r = ref.get(row[0])
            if not r:
                continue
            for mine, col in ((row[1], "ate_sim3"), (row[2], "rpe_trans_sim3"),
                              (row[3], "rpe_rot_sim3")):
                try:
                    d = abs(mine - float(r[col]))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(d):
                    dmax = max(dmax, d); n += 1
        print(f"  {lbl}: mean-agg vs sim3_pose.csv n={n} max|diff|={dmax:.3e}", flush=True)

    print("\nPER-LABEL sim3 pose metrics (mean | rmse):")
    for lbl in args.labels:
        rows = results[lbl]
        if not rows:
            continue
        outp = os.path.join(args.out_root, lbl, "sim3_pose_both.csv")
        with open(outp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene", "ate_mean", "rpe_trans_mean", "rpe_rot_mean",
                        "ate_rmse", "rpe_trans_rmse", "rpe_rot_rmse"])
            for row in rows:
                w.writerow(list(row))
        line = [f"{lbl}: n={len(rows)}"]
        for base, names in ((1, "mean"), (4, "rmse")):
            vals = []
            for j, name in enumerate(["ate", "rpe_t", "rpe_r"]):
                fin = [r[base + j] for r in rows if np.isfinite(r[base + j])]
                m = float(np.mean(fin)) if fin else float("nan")
                vals.append(f"{name}={m:.4f}")
            line.append(f"[{names}] " + " ".join(vals))
        print("  " + "  ".join(line) + f" -> {outp}", flush=True)


if __name__ == "__main__":
    main()
