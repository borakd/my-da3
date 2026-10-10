#!/usr/bin/env python
"""rig_trackon_eval: the exterior-camera rig evaluation with the gripper points taken from the Track-On-R tracks of the
birth-frame pipeline (vggt_features, Leonardo v2: FK birth frame >= 50 % in both views -> RobotSeg mask -> Track-On-R).

Window (the user's constraint): from the FK birth frame (entry) to the frame before the FK exit, i.e. while the gripper
model stays >= 50 % inside BOTH exterior images (entry/exit from entry_events.py, event 0 == v2 birth_f050).
A Track-On point is observed in a camera on a frame when its visibility flag is set and it lies inside the image.

Unchanged, imported:
  * cross-view pairing = rig_track.py's accumulated symmetric epipolar residual (median over >= 10 common observed frames,
    one-to-one, mutual best), threshold 2.5 px x the gripper's apparent-size scale (floor 0.4), size from the triangulated
    centroid of the two cameras' birth-frame query points (rig_track's mask-centroid estimator);
  * triangulation with factory intrinsics at 1280x720 + PointWorld optimized extrinsics (rig_points_eval.calib, incl. its
    serial-median intrinsics fallback);
  * rig_points_eval.evaluate: reference = first window frame with >= 4 triangulated pairs, growing body model, RANSAC +
    weighted Kabsch (5 mm inliers), lens transfer with the store GT wrist pose at the reference frame (hand-eye stand-in,
    the only GT before scoring), per-frame errors + Sim3 ATE / RPE vs the store GT, CUT3R finetuned on the same frames.

    python rig_trackon_eval.py --episode EP --tracks DIR --events CSV --out DIR [--positions DIR]
Writes DIR/record.json (same layout as rig_points_eval, so rig_points_summary.py aggregates it), per_frame.csv, anchors.npz.
"""
import argparse, csv, json, os, re, sys
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_track as rt          # noqa: E402
import rig_points_eval as rpe   # noqa: E402

ZED_CALIB = "/gpfs/scratch/etur59/koc821022/outputs/droid_birth_frames/zed_calib"   # Leonardo's factory confs (last resort)
POSITIONS = "/gpfs/scratch/etur59/koc821022/gripper_points/positions_v1/positions"


def calib(ep, positions):
    """rig_points_eval.calib; when the label file is missing, intrinsics fall back to the ZED factory conf [LEFT_CAM_HD]."""
    f = f"{positions}/{ep}_gripper.npz"
    if os.path.isfile(f):
        return rpe.calib(ep, np.load(f))
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
            sec = open(f"{ZED_CALIB}/SN{s}.conf").read().split("[LEFT_CAM_HD]")[1].split("[")[0]
            d = dict(re.findall(r"(\w+)=([-\d.eE+]+)", sec))
            K[cam] = np.array([[float(d["fx"]), 0, float(d["cx"])], [0, float(d["fy"]), float(d["cy"])], [0, 0, 1]]); src[cam] = "zed_conf"
        E[cam] = np.array(camj[s]["optimized_extrinsics"], np.float64); P[cam] = K[cam] @ E[cam][:3]
    return cams, K, E, P, src


def pair_tracks(A1, A2, ok1, ok2, F, thr):
    """rig_track.py's cross-view pairing, verbatim logic: symmetric epipolar residual, median over common frames."""
    T = A1.shape[1]; n1, n2 = len(A1), len(A2)
    X2h = np.concatenate([np.nan_to_num(A2.astype(np.float64)), np.ones((n2, T, 1))], axis=2)
    L1 = X2h @ F
    res = np.full((n1, n2), np.inf)
    for i in range(n1):
        com = ok1[i][None, :] & ok2
        js = np.where(com.sum(1) >= rt.RAIL["min_common"])[0]
        if len(js) == 0: continue
        x1 = np.c_[np.nan_to_num(A1[i].astype(np.float64)), np.ones(T)]
        l2 = x1 @ F.T
        d2 = np.abs((X2h[js] * l2[None]).sum(2)) / np.hypot(l2[:, 0], l2[:, 1])[None]
        d1 = np.abs((x1[None] * L1[js]).sum(2)) / np.hypot(L1[js, :, 0], L1[js, :, 1])
        res[i, js] = np.nanmedian(np.where(com[js], 0.5 * (d1 + d2), np.nan), axis=1)
    pairs, used2 = [], set()
    for i in np.argsort(res.min(1)):
        j = int(np.argmin(res[i]))
        if res[i, j] < thr and j not in used2 and np.argmin(res[:, j]) == i:
            pairs.append((i, j)); used2.add(j)
    return pairs, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True); ap.add_argument("--tracks", required=True, help="trackon_run_v2/tracks")
    ap.add_argument("--events", required=True, help="entry_events_gap3.csv (event 0 = birth, exit_frame exclusive)")
    ap.add_argument("--out", required=True); ap.add_argument("--positions", default=POSITIONS); ap.add_argument("--tag", default="trackon_v2")
    ap.add_argument("--no_cut", action="store_true", help="diagnostic: window to the end of the tracks instead of the FK exit")
    ap.add_argument("--paired", action="store_true", help="tracks share point indexing across the two cameras (rig_match_track.py): "
                    "pairs are (k, k) by construction, the epipolar pairing stage is skipped")
    a = ap.parse_args(); ep = a.episode; os.makedirs(a.out, exist_ok=True)
    rec = dict(episode=ep, tag=a.tag)
    try:
        ev = [r for r in csv.DictReader(open(a.events)) if r["episode"] == ep and r["event"] == "0"]
        if not ev: raise rt.Failure("no event 0 (no birth frame / excluded)")
        b, e = int(ev[0]["entry_frame"]), int(ev[0]["exit_frame"])
        cams, K, E, P, ksrc = calib(ep, a.positions)
        tr, vis, q = {}, {}, {}
        for cam, _ in cams:
            f = f"{a.tracks}/{ep}/{cam}_f{b:05d}.npz"
            if not os.path.isfile(f): raise rt.Failure(f"track file missing: {os.path.basename(f)}")
            z = np.load(f); tr[cam] = np.transpose(z["tracks"], (1, 0, 2)); vis[cam] = z["visibility"].T; q[cam] = z["queries"]
        store = f"{rt.STORE_ROOT}/{ep}/dense/cam"; n_store = rt.count_files(store, ".npz")
        T = min(n_store, tr["ext1"].shape[1], tr["ext2"].shape[1])
        t_end = T - 1 if a.no_cut else min(e, T) - 1
        gt = rt.load_poses(store, T, "store GT")
        try: cut = rt.load_poses(f"{rt.CUT3R_ROOT}/{ep}/camera", T, "CUT3R")
        except rt.Failure: cut = None
        obs = {}
        for cam, _ in cams:
            x = tr[cam][:, :T]; o = vis[cam][:, :T] & ~np.isnan(x[..., 0]) & (x[..., 0] >= 0) & (x[..., 0] < rt.W) & (x[..., 1] >= 0) & (x[..., 1] < rt.H)
            o[:, :b] = False; o[:, t_end + 1:] = False; obs[cam] = o
        # apparent size scale from the birth-frame query centroids (rig_track's estimator, on Track-On's own seed points)
        qb = {c: (q[c][q[c][:, 0] == b, 1:] if (q[c][:, 0] == b).any() else q[c][:, 1:]) for c in q}   # points seeded at the birth frame only
        Xc = rt.triangulate_point(P["ext1"], P["ext2"], qb["ext1"].mean(0), qb["ext2"].mean(0))
        depth = {c: float((E[c][:3, :3] @ Xc + E[c][:3, 3])[2]) for c in E}
        sz = {c: (min(1.0, K[c][0, 0] / (max(depth[c], 0.05) * 100) / rt.PPCM_RAIL) if depth[c] > 0 else 1.0) for c in E}
        thr = rt.RAIL["epi_px"] * max(min(sz.values()), 0.4)
        R1, t1_, R2, t2_ = E["ext1"][:3, :3], E["ext1"][:3, 3], E["ext2"][:3, :3], E["ext2"][:3, 3]
        Rr = R2 @ R1.T; trel = t2_ - Rr @ t1_
        F = np.linalg.inv(K["ext2"]).T @ rt.skew(trel) @ Rr @ np.linalg.inv(K["ext1"])
        if a.paired:
            if tr["ext1"].shape[0] != tr["ext2"].shape[0]: raise rt.Failure("paired tracks must have the same N in both cameras")
            pairs = [(k, k) for k in range(tr["ext1"].shape[0]) if (obs["ext1"][k] & obs["ext2"][k]).sum() >= 2]
            epi_med = []
            for i, j in pairs:   # diagnostic only: the residual of the given correspondences over their common frames
                cm = obs["ext1"][i] & obs["ext2"][j]
                h1 = np.c_[tr["ext1"][i, :T][cm].astype(np.float64), np.ones(cm.sum())]; h2 = np.c_[tr["ext2"][j, :T][cm].astype(np.float64), np.ones(cm.sum())]
                l2 = h1 @ F.T; l1 = h2 @ F
                epi_med.append(np.median(0.5 * (np.abs((h2 * l2).sum(1)) / np.hypot(l2[:, 0], l2[:, 1]) + np.abs((h1 * l1).sum(1)) / np.hypot(l1[:, 0], l1[:, 1]))))
            epi_med = float(np.median(epi_med)) if epi_med else None
        else:
            pairs, res = pair_tracks(tr["ext1"][:, :T], tr["ext2"][:, :T], obs["ext1"], obs["ext2"], F, thr)
            epi_med = float(np.median([res[i, j] for i, j in pairs])) if pairs else None
        rec.update(K_src=ksrc, T=T, birth=b, exit=e, window_end=t_end, n_queries={c: int(len(q[c])) for c in q},
                   depth_birth_m=depth, size_scale=sz, epi_thr_px=thr, pairs=len(pairs), paired=bool(a.paired),
                   epipolar_px_median=epi_med)
        if len(pairs) < 3: raise rt.Failure(f"fewer than 3 cross-view pairs ({len(pairs)})")
        X = np.full((len(pairs), T, 3), np.nan); rp = []
        for k, (i, j) in enumerate(pairs):
            com = np.where(obs["ext1"][i] & obs["ext2"][j])[0]
            x1 = tr["ext1"][i, com].T.astype(np.float64); x2 = tr["ext2"][j, com].T.astype(np.float64)
            Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], x1, x2); Xw = (Xh[:3] / Xh[3]).T; X[k, com] = Xw
            for c, xx in (("ext1", x1), ("ext2", x2)):
                pr = P[c] @ np.c_[Xw, np.ones(len(Xw))].T; rp.append(np.linalg.norm(pr[:2] / pr[2] - xx, axis=0))
        rp = np.concatenate(rp); rec["reproj_px"] = dict(median=float(np.median(rp)), p90=float(np.percentile(rp, 90)))
        drift = []   # DIAGNOSTIC (uses GT, not part of the method): drift of each triangulated point in the gripper frame
        for k in range(len(pairs)):
            cm = np.where(~np.isnan(X[k, :, 0]))[0]
            if len(cm) < 5: continue
            Y = np.array([np.linalg.inv(gt[t])[:3] @ np.r_[X[k, t], 1.0] for t in cm if t in gt])
            if len(Y) < 5: continue
            drift.append(np.sqrt(((Y - np.median(Y, 0)) ** 2).sum(1).mean()) * 1000)
        rec["diag_drift_mm"] = dict(median=float(np.median(drift)), p90=float(np.percentile(drift, 90)), gt5mm_frac=float(np.mean(np.array(drift) > 5))) if drift else None
        ntri = (~np.isnan(X[:, :, 0])).sum(0); obs_both = ntri >= 3
        cand = [u for u in range(b, t_end + 1) if ntri[u] >= rt.RAIL["min_ref_pairs"]]
        if not cand: raise rt.Failure(f"no window frame with >= {rt.RAIL['min_ref_pairs']} triangulated pairs")
        w, rows, (pf, pp, cp) = rpe.evaluate(b, t_end, cand[0], X, obs_both, gt, cut)
        w.update(episode=ep, tag=a.tag, window_idx=0, n_windows=1, pairs=len(pairs))
        rec.update(frames_with_min_pts_both=int(obs_both[b:t_end + 1].sum()), windows=[w])
        if pp:
            np.savez(f"{a.out}/anchors.npz", frames=np.array(pf), poses=np.array(pp), centroid_poses=np.array(cp))
        if rows:
            with open(f"{a.out}/per_frame.csv", "w", newline="") as f:
                rr = [dict(r, window_idx=0) for r in rows]; wr = csv.DictWriter(f, list(rr[0])); wr.writeheader(); wr.writerows(rr)
    except rt.Failure as ex:
        rec.update(failure_reason=str(ex), windows=[dict(episode=ep, tag=a.tag, window_idx=0, n_windows=1, skipped=str(ex))])
        rec.setdefault("T", 0); rec.setdefault("frames_with_min_pts_both", 0)
    json.dump(rec, open(f"{a.out}/record.json", "w"), indent=1, default=float)
    print(json.dumps({k: rec[k] for k in rec if k != "windows"}, default=float)[:400]); print(json.dumps(rec["windows"], default=float)[:500])


if __name__ == "__main__":
    main()
