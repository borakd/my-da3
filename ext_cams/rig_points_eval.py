#!/usr/bin/env python
"""rig_points_eval: the rig_track.py evaluation with the gripper points taken from the kinematic labels
(pointworld-droid-gripper-points positions_v1) instead of SAM-mask-seeded LK tracks.

Unchanged from rig_track.py (functions imported from it): calibration (factory intrinsics at 1280x720 + PointWorld
optimized extrinsics), cv2 triangulation, growing body model, RANSAC + weighted Kabsch with the 5 mm inlier threshold,
lens transfer with the store GT wrist pose at the reference frame (the hand-eye stand-in), per-frame lens errors and
Sim3-aligned ATE / RPE (traj_metrics) against the store GT, CUT3R finetuned (augfull_lr1e5) on the same frames.
Changed: the points. Each labelled gripper point (fixed id) is a cross-view pair by construction; it is observed in a
camera on a frame when it is inside the image and faces that camera (labels' in_frame & front). Pixels are the labels'
320x180 coordinates scaled x4 to 1280x720.

Window modes:
  --window T0 T1 --t_ref TR   evaluate exactly this window and reference frame (to reuse a SAM run's frames)
  (default) label windows     every maximal run of frames with >= --min_pts points observed in BOTH cameras is a
                              separate evaluation; reference = first frame of the run with >= 4 such points (rig_track rule)
--links base keeps only the rigid gripper-base points (diagnostic: finger points move with the opening).
Also reports the probe's centroid-only variant (lens moved by the inlier-centroid displacement, Kabsch rotation).

    python rig_points_eval.py --episode EP --positions DIR --out DIR [--window T0 T1 --t_ref TR] [--min_pts 3]
"""
import argparse, csv, json, os, sys
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_track as rt   # noqa: E402  (verbatim geometry + inputs of the SAM evaluation)


def calib(ep, z):
    """rig_track's calibration (episode factory intrinsics at 1280x720 + PointWorld extrinsics). When the episode has no valid
    factory entry (133 test scenes), fall back to the intrinsics the labels were projected with (the camera serial's median
    over all DROID episodes, stored in the label file at 320x180, scaled x4). Returns the source per camera."""
    meta = json.load(open(f"{rt.RAW_ROOT}/{ep}/metadata_{ep}.json"))
    cams = [("ext1", str(meta["ext1_cam_serial"])), ("ext2", str(meta["ext2_cam_serial"]))]
    try: intr = rt.load_intrinsics(ep, rt.INTR_JSON)
    except rt.Failure: intr = {}
    camj = json.load(open(f"{rt.CAM_JSON}/{ep}_cameras.json"))
    K, E, P, src = {}, {}, {}, {}
    for cam, s in cams:
        e = intr.get(s)
        if e and e.get("width") and e["cameraMatrix"][0] > 0:
            fx, cx, fy, cy = e["cameraMatrix"]; sx, sy = rt.W / e["width"], rt.H / e.get("height", rt.H)
            K[cam] = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]]); src[cam] = "episode"
        else:
            K[cam] = np.diag([rt.W / 320, rt.H / 180, 1.0]) @ z[f"{s}_K"]; src[cam] = "label_K_serial_median"
        E[cam] = np.array(camj[s]["optimized_extrinsics"], np.float64); P[cam] = K[cam] @ E[cam][:3]
    return cams, K, E, P, src


def evaluate(t0, t1, t_ref, X, obs_both, gt, cut):
    """rig_track's alignment loop on a given window and reference frame. X (N, T, 3) triangulated points (nan if unobserved)."""
    X0 = np.full((X.shape[0], 3), np.nan); seed0 = ~np.isnan(X[:, t_ref, 0]); X0[seed0] = X[seed0, t_ref]
    P0 = gt[t_ref]; c0 = P0[:3, 3]
    rows, pred, cpred, gtl, cutl, pframes = [], [], [], [], [], []
    for t in range(t0, t1 + 1):
        if t not in gt: break
        have = ~np.isnan(X0[:, 0]) & ~np.isnan(X[:, t, 0])
        R, tt, inl = rt.ransac_kabsch(X0[have], X[have, t], np.ones(have.sum()), rt.RAIL["rigid_m"], iters=rt.RAIL["ransac_iters"]) \
            if have.sum() >= 3 else (None, None, None)
        if R is None:
            rows.append(dict(frame=t, n_pts=int(have.sum()), n_inl=0, pos_err_cm=np.nan, rot_err_deg=np.nan, centroid_pos_err_cm=np.nan,
                             rigid_resid_mm=np.nan)); continue
        Mt = np.eye(4); Mt[:3, :3] = R; Mt[:3, 3] = tt
        newp = np.isnan(X0[:, 0]) & ~np.isnan(X[:, t, 0])
        if newp.any(): X0[newp] = (X[newp, t] - tt) @ R
        Pt = Mt @ P0
        cen_t = (X0[have][inl] @ R.T + tt).mean(0); cen_ref = X0[have][inl].mean(0); Pc = Pt.copy(); Pc[:3, 3] = c0 + (cen_t - cen_ref)
        resid = np.linalg.norm((X0[have][inl] @ R.T + tt) - X[have, t][inl], axis=1)
        rows.append(dict(frame=t, n_pts=int(have.sum()), n_inl=int(inl.sum()), pos_err_cm=float(np.linalg.norm(Pt[:3, 3] - gt[t][:3, 3]) * 100),
                         rot_err_deg=float(rt.rot_angle_deg(Pt[:3, :3].T @ gt[t][:3, :3])),
                         centroid_pos_err_cm=float(np.linalg.norm(Pc[:3, 3] - gt[t][:3, 3]) * 100), rigid_resid_mm=float(np.median(resid) * 1000)))
        pred.append(Pt); cpred.append(Pc); gtl.append(gt[t]); pframes.append(t)
        if cut is not None and t in cut: cutl.append(cut[t])
    ok = [r for r in rows if r["n_inl"] >= 3]
    rec = dict(window=[t0, t1], window_len=t1 - t0 + 1, t_ref=t_ref, n_anchored=len(pred), ref_points=int(seed0.sum()),
               frames_both=int(obs_both[t0:t1 + 1].sum()))
    if len(pred) >= 2:
        pe = np.array([r["pos_err_cm"] for r in ok]); re_ = np.array([r["rot_err_deg"] for r in ok]); ce = np.array([r["centroid_pos_err_cm"] for r in ok])
        m = rt.traj_metrics(pred, gtl); mc = rt.traj_metrics(cpred, gtl)
        rec.update(pos_err_median_cm=float(np.median(pe)), pos_err_p90_cm=float(np.percentile(pe, 90)), pos_err_max_cm=float(pe.max()),
                   rot_err_median_deg=float(np.median(re_)), rot_err_p90_deg=float(np.percentile(re_, 90)), rot_err_max_deg=float(re_.max()),
                   centroid_pos_err_median_cm=float(np.median(ce)), rigid_resid_median_mm=float(np.median([r["rigid_resid_mm"] for r in ok])),
                   inliers_median=float(np.median([r["n_inl"] for r in ok])), rig_ate=m["ate"], rig_rpe_t=m["rpe_trans"], rig_rpe_rot=m["rpe_rot"],
                   centroid_ate=mc["ate"])
        if cut is not None and len(cutl) == len(pred):
            mk = rt.traj_metrics(cutl, gtl); rec.update(cut3r_ate_same_frames=mk["ate"], cut3r_rpe_t_same_frames=mk["rpe_trans"], cut3r_rpe_rot_same_frames=mk["rpe_rot"])
    return rec, rows, (pframes, pred, cpred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True); ap.add_argument("--positions", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, nargs=2); ap.add_argument("--t_ref", type=int); ap.add_argument("--min_pts", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--links", default="all", choices=["all", "base"], help="base = only robotiq_85_base_link points (rigid; diagnostic)")
    a = ap.parse_args(); ep = a.episode; os.makedirs(a.out, exist_ok=True)
    z = np.load(f"{a.positions}/{ep}_gripper.npz"); kept = [str(s) for s in z["serials"]]
    cams, K, E, P, ksrc = calib(ep, z)
    for _, s in cams:
        if s not in kept: raise SystemExit(f"{ep}: camera {s} has no labels (excluded)")
    store = f"{rt.STORE_ROOT}/{ep}/dense/cam"; n_store = rt.count_files(store, ".npz")
    T = min(int(z["T"]) - 1, n_store)               # frames with a video frame and a store pose
    gt = rt.load_poses(store, T, "store GT")
    try: cut = rt.load_poses(f"{rt.CUT3R_ROOT}/{ep}/camera", T, "CUT3R")
    except rt.Failure: cut = None
    uv = {c: z[f"{s}_uv"][:T].astype(np.float64) * np.array([rt.W / 320, rt.H / 180]) for c, s in cams}
    ob = {c: (z[f"{s}_in_frame"] & z[f"{s}_front"])[:T] for c, s in cams}
    both = ob["ext1"] & ob["ext2"]                  # (T, N): point observed in both cameras
    if a.links == "base": both = both & (z["link"] == "robotiq_85_base_link")[None]
    N = both.shape[1]; X = np.full((N, T, 3), np.nan)
    for k in range(N):
        tt = np.nonzero(both[:, k])[0]
        if len(tt) == 0: continue
        Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], uv["ext1"][tt, k].T, uv["ext2"][tt, k].T); X[k, tt] = (Xh[:3] / Xh[3]).T
    nboth = both.sum(1); obs_both = nboth >= a.min_pts
    if a.window:
        wins = [(a.window[0], min(a.window[1], T - 1), a.t_ref)]
    else:
        wins, t = [], 0
        while t < T:
            if not obs_both[t]: t += 1; continue
            s = t
            while t + 1 < T and obs_both[t + 1]: t += 1
            cand = [u for u in range(s, t + 1) if nboth[u] >= rt.RAIL["min_ref_pairs"]]
            wins.append((s, t, cand[0] if cand else None)); t += 1
    recs, allrows, poses = [], [], []
    for i, (t0, t1, tr) in enumerate(wins):
        if tr is None:
            recs.append(dict(episode=ep, tag=a.tag, window_idx=i, n_windows=len(wins), window=[t0, t1], window_len=t1 - t0 + 1, skipped="no frame with >= 4 points")); continue
        rec, rows, (pf, pp, cp) = evaluate(t0, t1, tr, X, obs_both, gt, cut)
        poses += [(i, f, P_, C_) for f, P_, C_ in zip(pf, pp, cp)]
        rec.update(episode=ep, tag=a.tag, window_idx=i, n_windows=len(wins)); recs.append(rec)
        allrows += [dict(r, window_idx=i) for r in rows]
    if poses:   # predicted wrist poses (c2w, robot base frame) per solved frame: Kabsch transfer and centroid-only variant
        np.savez(f"{a.out}/anchors.npz", window_idx=np.array([p[0] for p in poses]), frames=np.array([p[1] for p in poses]),
                 poses=np.array([p[2] for p in poses]), centroid_poses=np.array([p[3] for p in poses]))
    json.dump(dict(episode=ep, tag=a.tag, K_src=ksrc, T=T, frames_with_min_pts_both=int(obs_both.sum()), windows=recs), open(f"{a.out}/record.json", "w"), indent=1)
    if allrows:
        with open(f"{a.out}/per_frame.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(allrows[0])); w.writeheader(); w.writerows(allrows)
    print(json.dumps(recs)[:600])


if __name__ == "__main__":
    main()
