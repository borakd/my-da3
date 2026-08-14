#!/usr/bin/env python
"""Fit the NGC drift warm-start (d_t, d_r) from existing closed-loop preds.

Theil-Sen slope of the mean per-view conditioning-error magnitude vs view
index over views 1..63 (the training horizon), residuals computed in the
view-0-relative frame: rel = inv(P_0) @ P_k for pred and GT separately;
translation residual ||t_pred - t_gt|| / s_seq (per-scene GT scale, so d_t is
in s_seq units/view); rotation residual = geodesic angle between the relative
rotations, in deg/view. Primary source: diag_gtrayB_prevpred430 (the collapse
whose first moment the training noise must cover); secondary (reported, not
used for the warm start): diag_refine_clean430.
"""
import glob
import json
import os

import numpy as np

OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
GT_ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
SUBSET = "eval_pipeline/noise_oracle_subset_430.txt"
H = 64  # training horizon


def rot_angle_deg(Ra, Rb):
    c = np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def scene_residuals(pred_dir, gt_dir):
    preds = sorted(glob.glob(os.path.join(pred_dir, "*.npz")))[:H]
    if len(preds) < 8:
        return None
    P, G = [], []
    for p in preds:
        g = os.path.join(gt_dir, os.path.basename(p))
        if not os.path.isfile(g):
            return None
        P.append(np.load(p)["pose"].astype(np.float64))
        G.append(np.load(g)["pose"].astype(np.float64))
    iP0, iG0 = np.linalg.inv(P[0]), np.linalg.inv(G[0])
    rels_p = [iP0 @ P[k] for k in range(len(P))]
    rels_g = [iG0 @ G[k] for k in range(len(G))]
    # Per-side scale normalization (p43 avg_dis convention): raw pred-vs-GT
    # differencing measures the ~4.8x head-scale mismatch instead of pose
    # error (the rung-0 / v1-instrument failure class). After dividing each
    # side by its own mean relative-translation norm, the GT side has unit
    # scale, so d_t is in sequence-scale units/view by construction.
    f_p = max(float(np.mean([np.linalg.norm(r[:3, 3]) for r in rels_p[1:]])), 1e-9)
    f_g = max(float(np.mean([np.linalg.norm(r[:3, 3]) for r in rels_g[1:]])), 1e-9)
    et = np.full(H, np.nan)
    er = np.full(H, np.nan)
    for k in range(1, len(P)):
        et[k] = np.linalg.norm(rels_p[k][:3, 3] / f_p - rels_g[k][:3, 3] / f_g)
        er[k] = rot_angle_deg(rels_p[k][:3, :3], rels_g[k][:3, :3])
    return et, er


def theil_sen(y):
    ks = np.arange(len(y))
    ok = np.isfinite(y)
    ks, y = ks[ok], y[ok]
    slopes = [
        (y[j] - y[i]) / (ks[j] - ks[i])
        for i in range(len(ks)) for j in range(i + 1, len(ks))
    ]
    return float(np.median(slopes))


def fit(label):
    scenes = [l.strip() for l in open(SUBSET) if l.strip()]
    ets, ers = [], []
    for s in scenes:
        r = scene_residuals(f"{OUT}/{label}/preds/{s}/camera", f"{GT_ROOT}/{s}/dense/cam")
        if r is not None:
            ets.append(r[0])
            ers.append(r[1])
    met = np.nanmean(np.stack(ets), 0)
    mer = np.nanmean(np.stack(ers), 0)
    naive_t = float((met[H - 1] - met[1]) / (H - 2)) if np.isfinite(met[H - 1]) else float("nan")
    return dict(n_scenes=len(ets), d_t=theil_sen(met), d_r=theil_sen(mer),
                naive_slope_t=naive_t,
                mean_curve_t=[round(float(x), 5) for x in met.tolist()],
                mean_curve_r=[round(float(x), 4) for x in mer.tolist()])


def main():
    out = dict(primary_label="diag_gtrayB_prevpred430",
               primary=fit("diag_gtrayB_prevpred430"),
               secondary_label="diag_refine_clean430",
               secondary=fit("diag_refine_clean430"))
    for arm in ("primary", "secondary"):
        f = out[arm]
        assert np.isfinite(f["d_t"]) and np.isfinite(f["d_r"]), f"{arm}: non-finite fit"
    p = out["primary"]
    assert p["d_t"] > 0 and p["d_r"] > 0, "primary drift slopes must be positive"
    with open("eval_pipeline/evidence/ngc_drift_warmstart.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("n_scenes", "d_t", "d_r", "naive_slope_t")}
                      for k, v in (("primary", out["primary"]), ("secondary", out["secondary"]))},
                     indent=1))


if __name__ == "__main__":
    main()
