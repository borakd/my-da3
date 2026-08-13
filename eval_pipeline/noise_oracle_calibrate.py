#!/usr/bin/env python
"""Calibrate sigma_ref for the noise-tolerance oracle (Step 1, GRU v4 campaign).

sigma_ref = the honest closed-loop head's own per-view pose error vs GT, in GT
units, measured on arm C's (prevpred_lr1e5) ALREADY-SAVED 4292-scene eval
predictions — zero GPU. Per scene: sim3-align (Umeyama with scale) the
predicted camera-center trajectory to GT, exactly the alignment family
eval_depth_poses.py scores with, then take per-view translation L2 (GT units)
and rotation geodesic error (deg, after applying the alignment rotation).
sigma_ref_{t,r} = median over every (scene, view>0).

Usage:
  python noise_oracle_calibrate.py \
      --pred_root  <OUT>/prevpred_lr1e5/preds \
      --scenes_root <...>/test/dl3dv_multi/wrist \
      --scene_list  eval_pipeline/noise_oracle_subset_430.txt \
      --out         eval_pipeline/evidence/noise_oracle_calibration.json
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np


def umeyama_sim3(src, dst):
    """Similarity transform (s, R, t) minimizing ||dst - (s R src + t)||^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s if var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def scene_errors(args):
    pred_dir, gt_dir = args
    preds = sorted(glob.glob(os.path.join(pred_dir, "*.npz")))
    et, er = [], []
    P, G = [], []
    for p in preds:
        base = os.path.basename(p)
        g = os.path.join(gt_dir, base)
        if not os.path.isfile(g):
            return None
        P.append(np.load(p)["pose"].astype(np.float64))
        G.append(np.load(g)["pose"].astype(np.float64))
    if len(P) < 3:
        return None
    P, G = np.stack(P), np.stack(G)
    s, R, t = umeyama_sim3(P[:, :3, 3], G[:, :3, 3])
    for i in range(1, len(P)):
        tp = s * R @ P[i, :3, 3] + t
        et.append(float(np.linalg.norm(tp - G[i, :3, 3])))
        Rrel = G[i, :3, :3].T @ (R @ P[i, :3, :3])
        c = np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0)
        er.append(float(np.degrees(np.arccos(c))))
    return et, er


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--scenes_root", required=True)
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    scenes = [ln.strip() for ln in open(a.scene_list) if ln.strip()]
    jobs = [
        (
            os.path.join(a.pred_root, s, "camera"),
            os.path.join(a.scenes_root, s, "dense", "cam"),
        )
        for s in scenes
    ]
    with Pool(a.workers) as pool:
        results = pool.map(scene_errors, jobs)

    ok = [r for r in results if r is not None]
    all_t = np.array([e for r in ok for e in r[0]])
    all_r = np.array([e for r in ok for e in r[1]])
    out = dict(
        pred_root=a.pred_root,
        scene_list=a.scene_list,
        scenes_requested=len(scenes),
        scenes_used=len(ok),
        views_used=int(len(all_t)),
        sigma_ref_t=float(np.median(all_t)),
        sigma_ref_r_deg=float(np.median(all_r)),
        trans_quantiles={q: float(np.quantile(all_t, float(q))) for q in ("0.25", "0.5", "0.75", "0.9")},
        rot_deg_quantiles={q: float(np.quantile(all_r, float(q))) for q in ("0.25", "0.5", "0.75", "0.9")},
        note=(
            "sim3(Umeyama+scale) per-scene alignment of pred camera centers to "
            "GT; per-view trans L2 in GT units and rotation geodesic deg for "
            "views>0; medians define sigma_ref for the noise grid."
        ),
    )
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("scenes_used", "views_used", "sigma_ref_t", "sigma_ref_r_deg")}, indent=1))


if __name__ == "__main__":
    main()
