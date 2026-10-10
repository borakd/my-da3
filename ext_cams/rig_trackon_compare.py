#!/usr/bin/env python
"""Score the Track-On rig, CUT3R zero-shot and CUT3R finetuned (augfull_lr1e5) on the SAME frames per scene with the
master-table pose math (eval_bundle/bin/eval_depth_poses.py defaults: Sim3 Umeyama on camera centres, then RMSE of
per-frame ATE and of consecutive-pair RPE_trans / RPE_rot), against the store GT wrist poses.
Frame sets per scene: 'rig' = frames the rig solved (anchors.npz); also 'full' = every frame (CUT3R only, sanity check
against master_4292.csv). Depth metrics do not apply to the rig (it predicts poses only).

    python rig_trackon_compare.py RUN_DIR OUT_CSV [--workers N]
"""
import argparse, glob, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/using_ext_cams/eval_bundle/bin"))
import rig_track as rt                                  # noqa: E402
import eval_depth_poses as edp                          # noqa: E402
O = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
SRC = {"zs": f"{O}/cut3r_zeroshot/preds", "ft": f"{O}/augfull_lr1e5/preds"}


def metrics(pred, gt, ts):
    ts = [t for t in ts if t in pred and t in gt]
    if len(ts) < 2: return dict(ate=np.nan, rpe_trans=np.nan, rpe_rot=np.nan, n=len(ts))
    al = edp._align_pred_poses_to_gt({t: pred[t] for t in ts}, {t: gt[t] for t in ts}, "sim3")
    ate = [np.linalg.norm(al[t][:3, 3] - gt[t][:3, 3]) for t in ts]
    rtr, rro = [], []
    for a, b in zip(ts[:-1], ts[1:]):
        E = np.linalg.inv(edp._relative_pose(gt[a], gt[b])) @ edp._relative_pose(al[a], al[b])
        rtr.append(np.linalg.norm(E[:3, 3])); rro.append(edp._rotation_angle_deg(E[:3, :3]))
    return dict(ate=edp._nanrmse(ate), rpe_trans=edp._nanrmse(rtr), rpe_rot=edp._nanrmse(rro), n=len(ts))


def one(d):
    ep = os.path.basename(d); f = f"{d}/anchors.npz"
    if not os.path.isfile(f): return []
    z = np.load(f); fr = [int(x) for x in z["frames"]]; rig = {int(t): P for t, P in zip(z["frames"], z["poses"])}
    store = f"{rt.STORE_ROOT}/{ep}/dense/cam"; T = rt.count_files(store, ".npz"); gt = rt.load_poses(store, T, "gt")
    out = [dict(episode=ep, method="rig", frames="rig", **metrics(rig, gt, fr))]
    for k, root in SRC.items():
        try: p = rt.load_poses(f"{root}/{ep}/camera", T, k)
        except rt.Failure: continue
        out.append(dict(episode=ep, method=k, frames="rig", **metrics(p, gt, fr)))
        out.append(dict(episode=ep, method=k, frames="full", **metrics(p, gt, sorted(gt))))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args(); dirs = sorted(d for d in glob.glob(f"{a.run}/*") if os.path.isdir(d))
    with Pool(a.workers) as p: rows = [r for rs in p.imap_unordered(one, dirs, chunksize=8) for r in rs]
    pd.DataFrame(rows).to_csv(a.out, index=False); print(len(rows), "rows")
