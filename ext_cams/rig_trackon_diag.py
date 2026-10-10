#!/usr/bin/env python
"""Why the Track-On rig fails / is inaccurate: per-scene diagnostics of the pairing stage and of the physical validity
of the pairs it finds. Same inputs, window and pairing code as rig_trackon_eval.py.
  pairing:   window length, observed frames per camera (Track-On visible & in-image), candidate pairs with >= 10 common
             frames, best epipolar residual quantiles, pairs at the scene's threshold and at flat 2.5 / 5 px.
  validity:  for every pair found, triangulate per frame and express the point in the GRIPPER frame with the store GT
             wrist pose: Y_t = inv(GT_t) X_t. A true physical correspondence gives a constant Y_t (drift_mm ~ triangulation
             noise); a pseudo-pair (two different surface points that merely satisfy the epipolar line) drifts as the
             gripper rotates. Also the model extent (cm) and the per-frame solvability (>= 3 / >= 4 pairs observed).
    python rig_trackon_diag.py --episode EP --tracks DIR --events CSV --out CSV_ROW"""
import argparse, csv, json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_track as rt, rig_trackon_eval as rte

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--episode", required=True); ap.add_argument("--tracks", required=True)
    ap.add_argument("--events", required=True); ap.add_argument("--out", required=True); a = ap.parse_args(); ep = a.episode
    r = dict(episode=ep)
    ev = [x for x in csv.DictReader(open(a.events)) if x["episode"] == ep and x["event"] == "0"]
    b, e = int(ev[0]["entry_frame"]), int(ev[0]["exit_frame"])
    cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS)
    tr, vis, q = {}, {}, {}
    for cam, _ in cams:
        z = np.load(f"{a.tracks}/{ep}/{cam}_f{b:05d}.npz"); tr[cam] = np.transpose(z["tracks"], (1, 0, 2)); vis[cam] = z["visibility"].T; q[cam] = z["queries"]
    store = f"{rt.STORE_ROOT}/{ep}/dense/cam"; T = min(rt.count_files(store, ".npz"), tr["ext1"].shape[1], tr["ext2"].shape[1])
    t_end = min(e, T) - 1; L = t_end - b + 1; gt = rt.load_poses(store, T, "gt")
    r.update(birth=b, t_end=t_end, window_len=L, n_q1=len(q["ext1"]), n_q2=len(q["ext2"]))
    obs, inimg = {}, {}
    for cam, _ in cams:
        x = tr[cam][:, :T]; ii = ~np.isnan(x[..., 0]) & (x[..., 0] >= 0) & (x[..., 0] < rt.W) & (x[..., 1] >= 0) & (x[..., 1] < rt.H)
        o = vis[cam][:, :T] & ii; o[:, :b] = False; o[:, t_end + 1:] = False; ii[:, :b] = False; ii[:, t_end + 1:] = False
        obs[cam] = o; inimg[cam] = ii
        r[f"vis_frac_{cam}"] = float(o[:, b:t_end + 1].mean()); r[f"inimg_frac_{cam}"] = float(ii[:, b:t_end + 1].mean())
        r[f"frames_any_{cam}"] = int(o.any(0).sum()); r[f"pts_obs_median_{cam}"] = float(np.median(o[:, b:t_end + 1].sum(0)))
    Xc = rt.triangulate_point(P["ext1"], P["ext2"], q["ext1"][:, 1:].mean(0), q["ext2"][:, 1:].mean(0))
    depth = {c: float((E[c][:3, :3] @ Xc + E[c][:3, 3])[2]) for c in E}
    sz = {c: (min(1.0, K[c][0, 0] / (max(depth[c], 0.05) * 100) / rt.PPCM_RAIL) if depth[c] > 0 else 1.0) for c in E}
    thr = rt.RAIL["epi_px"] * max(min(sz.values()), 0.4)
    R1, t1_, R2, t2_ = E["ext1"][:3, :3], E["ext1"][:3, 3], E["ext2"][:3, :3], E["ext2"][:3, 3]
    Rr = R2 @ R1.T; trel = t2_ - Rr @ t1_; F = np.linalg.inv(K["ext2"]).T @ rt.skew(trel) @ Rr @ np.linalg.inv(K["ext1"])
    com = (obs["ext1"][:, None, :] & obs["ext2"][None, :, :]).sum(2)
    r.update(thr_px=thr, size_scale=min(sz.values()), depth_m=min(depth.values()), cand_pairs=int((com >= rt.RAIL["min_common"]).sum()),
             best_common_frames=int(com.max()))
    pairs, res = rte.pair_tracks(tr["ext1"][:, :T], tr["ext2"][:, :T], obs["ext1"], obs["ext2"], F, thr)
    fin = res[np.isfinite(res)]
    r.update(pairs=len(pairs), res_min=float(fin.min()) if fin.size else np.nan, res_p10=float(np.percentile(fin, 10)) if fin.size else np.nan,
             pairs_flat2p5=len(rte.pair_tracks(tr["ext1"][:, :T], tr["ext2"][:, :T], obs["ext1"], obs["ext2"], F, 2.5)[0]),
             pairs_flat5=len(rte.pair_tracks(tr["ext1"][:, :T], tr["ext2"][:, :T], obs["ext1"], obs["ext2"], F, 5.0)[0]))
    if pairs:
        X = np.full((len(pairs), T, 3), np.nan); drift, Y0 = [], []
        for k, (i, j) in enumerate(pairs):
            cm = np.where(obs["ext1"][i] & obs["ext2"][j])[0]
            Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], tr["ext1"][i, cm].T.astype(np.float64), tr["ext2"][j, cm].T.astype(np.float64))
            Xw = (Xh[:3] / Xh[3]).T; X[k, cm] = Xw
            Y = np.array([np.linalg.inv(gt[t])[:3] @ np.r_[x, 1.0] for t, x in zip(cm, Xw)])   # point in the wrist-camera (gripper) frame
            drift.append(np.sqrt(((Y - np.median(Y, 0)) ** 2).sum(1).mean()) * 1000); Y0.append(np.median(Y, 0))
        Y0 = np.array(Y0); ntri = (~np.isnan(X[:, b:t_end + 1, 0])).sum(0)
        r.update(drift_mm_median=float(np.median(drift)), drift_mm_p90=float(np.percentile(drift, 90)), pairs_drift_gt5mm=int((np.array(drift) > 5).sum()),
                 extent_cm=float(np.linalg.norm(Y0 - Y0.mean(0), axis=1).max() * 100), frames_ge3=int((ntri >= 3).sum()), frames_ge4=int((ntri >= 4).sum()),
                 ntri_median=float(np.median(ntri)))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(r)); w.writeheader(); w.writerow(r)

if __name__ == "__main__":
    main()
