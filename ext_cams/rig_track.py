#!/usr/bin/env python
"""rig_track: per-episode wrist-camera anchors from the two STATIC exterior cameras, driven by any gripper-mask seeding
method. Command-line form of the validated rig_kabsch_probe.py (same algorithm, same thresholds):

  * seeds Shi-Tomasi corners inside the eroded gripper mask of each exterior view, tracks them FORWARD ONLY with
    pyramidal LK + a forward-backward check (1.5 px at 720p), keeps a track only while it stays inside the dilated mask,
    re-seeds whenever the live set is thin;
  * pairs tracks across the two views by the epipolar residual accumulated over >= 10 common frames (2.5 px), one-to-one;
  * triangulates every pair per frame with the PointWorld optimized extrinsics and the factory intrinsics;
  * reference frame = first frame with >= 4 triangulated pairs; the body model (point coordinates in the reference frame)
    grows as new pairs are first seen on a solved frame; every frame is aligned to the model with RANSAC + weighted Kabsch
    (5 mm inlier), frames with < 3 model points are unsolved;
  * the body motion is transferred to the lens with the kinematic GT pose at the reference frame (the declared hand-eye
    stand-in, the only GT used before scoring).

The visibility window runs from the first to the last frame where BOTH masks are non-empty; gaps inside it are allowed
(tracks survive a frame with an empty mask on the forward-backward check alone; nothing is re-seeded there).
Pixel-unit parameters scale DOWN with the gripper's apparent size when it is smaller than on RAIL (pixels per centimetre
from the depth of the mask-centroid triangulation at the seed frame); they never exceed the RAIL values.

    python rig_track.py --episode EP --mask_npz_ext1 PATH --mask_npz_ext2 PATH --out DIR [--video]

Writes DIR/record.json (always; failure_reason set and exit 0 when something is missing), DIR/anchors.npz,
DIR/per_frame.csv, optionally DIR/tracks.npz (--save_tracks) and DIR/rig_side_by_side.mp4 (--video).
Native 1280x720 only. Budget (login node, OMP_NUM_THREADS=1): RAIL, 128 frames, 5 s (+4 s with --video); the pairing
stage grows with the track count so episodes above ~500 frames go through sbatch (rig_track_array.sbatch).
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

TOOL_VERSION = "rig_track v1 (2026-09-29, from rig_kabsch_probe.py)"
OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
AUDIT = f"{OUT_ROOT}/vggt_probe/gt_audit_2026-09-08"
INTR_JSON = f"{AUDIT}/docs/hf_intrinsics.json"                       # 126 MB, all episodes; cached per episode below
INTR_CACHE = f"{OUT_ROOT}/ext_cams/seedstudy/intrinsics_cache"
CAM_JSON = f"{AUDIT}/pointworld/droid/cameras"                       # <EP>_cameras.json, optimized_extrinsics = world->cam
RAW_ROOT = "/gpfs/scratch/etur59/koc821022/vggt_cache/raw"
STORE_ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
CUT3R_ROOT = f"{OUT_ROOT}/augfull_lr1e5/preds"                       # finetuned backbone, c2w non-metric
W, H = 1280, 720

# ----------------------------------------------------------------------------- validated RAIL parameters (720p)
RAIL = dict(erode_px=11, dilate_px=31, lk_win_px=21, min_dist_px=6, excl_r_px=8, fb_px=1.5, epi_px=2.5,
            rigid_m=0.005, min_common=10, max_pts=120, feat_quality=0.005, area_frac=0.3, ransac_iters=300, min_ref_pairs=4)
# pixels per centimetre of the gripper on RAIL+80edfcb1+2023-07-14-14h-28m-45s (min over ext1/ext2) from the mask-centroid
# triangulation at its seed frame 41 (the same estimator the tool applies to every episode); gripper at 0.42-0.50 m there.
PPCM_RAIL = 10.7  # measured 10.72 (ext1) / 13.92 (ext2); rounded down so both RAIL cameras clamp to scale 1.0


# ----------------------------------------------------------------------------- geometry helpers (verbatim from the probe)
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def kabsch(A, B, w=None):
    """R, t minimising sum w |R a + t - b|^2 (A, B: N x 3)."""
    w = np.ones(len(A)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    ca, cb = (w[:, None] * A).sum(0), (w[:, None] * B).sum(0)
    Hm = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def ransac_kabsch(A, B, w, thr, iters=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(A)
    if n < 3:
        return None, None, np.zeros(n, bool)
    best_inl = np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        if np.linalg.matrix_rank(A[idx] - A[idx].mean(0)) < 2:
            continue
        R, t = kabsch(A[idx], B[idx])
        inl = np.linalg.norm((A @ R.T + t) - B, axis=1) < thr
        if inl.sum() > best_inl.sum():
            best_inl = inl
    if best_inl.sum() < 3:
        return None, None, best_inl
    R, t = kabsch(A[best_inl], B[best_inl], w[best_inl])
    return R, t, best_inl


def umeyama(src, dst):
    """Sim3 (s, R, t) with dst ~ s R src + t; src, dst: N x 3."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    Hm = S.T @ D / len(src)
    U, sig, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1, 1, d])
    R = Vt.T @ Dm @ U.T
    s = np.trace(np.diag(sig) @ Dm) / (S ** 2).sum() * len(src)
    return s, R, mu_d - s * R @ mu_s


def traj_metrics(pred, gt):
    """pred, gt: lists of 4x4 c2w. ATE = RMSE of camera centres after Sim3 alignment; RPE over consecutive entries on the
    aligned trajectory (translation in metres, rotation in degrees), mean over steps."""
    P, G = np.array(pred), np.array(gt)
    if len(P) < 2:
        return dict(ate=float("nan"), rpe_trans=float("nan"), rpe_rot=float("nan"), sim3_scale=float("nan"), n=len(P))
    s, R, t = umeyama(P[:, :3, 3], G[:, :3, 3])
    Pa = P.copy()
    Pa[:, :3, 3] = (s * (R @ P[:, :3, 3].T)).T + t
    Pa[:, :3, :3] = R @ P[:, :3, :3]
    ate = np.sqrt(np.mean(np.sum((Pa[:, :3, 3] - G[:, :3, 3]) ** 2, 1)))
    rt, rr = [], []
    for i in range(len(P) - 1):
        dg = np.linalg.inv(G[i]) @ G[i + 1]
        dp = np.linalg.inv(Pa[i]) @ Pa[i + 1]
        e = np.linalg.inv(dg) @ dp
        rt.append(np.linalg.norm(e[:3, 3])); rr.append(rot_angle_deg(e[:3, :3]))
    return dict(ate=float(ate), rpe_trans=float(np.mean(rt)), rpe_rot=float(np.mean(rr)), sim3_scale=float(s), n=len(P))


# ----------------------------------------------------------------------------- inputs
class Failure(Exception):
    pass


def load_mask_npz(path):
    """Mask contract: 'union' uint8 (T, H, W/8) bit-packed, 'shape' [T, H, W]. Returns uint8 (T, 720, 1280)."""
    if not os.path.isfile(path):
        raise Failure(f"mask file missing: {path}")
    try:
        m = np.load(path)
        T, h, w = [int(v) for v in m["shape"]]
        M = np.unpackbits(m["union"], axis=2)[:, :, :w].astype(np.uint8)
    except Exception as e:  # corrupt / wrong keys
        raise Failure(f"mask file unreadable ({e.__class__.__name__}: {e}): {path}")
    if M.shape[0] != T:
        raise Failure(f"mask shape/union mismatch in {path}: shape says T={T}, union has {M.shape[0]}")
    if (h, w) != (H, W):
        M = np.array([cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST) for x in M])
    return M


def read_frames(path, n_max, gray=True):
    """Decode up to n_max frames at 1280x720; grayscale by default (the tracker never needs colour; the video re-reads)."""
    if not os.path.isfile(path):
        raise Failure(f"mp4 missing: {path}")
    cap = cv2.VideoCapture(path)
    fr = []
    while len(fr) < n_max:
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        fr.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if gray else img)
    cap.release()
    if not fr:
        raise Failure(f"no frames decoded from {path}")
    return fr


def load_intrinsics(ep, intr_json):
    os.makedirs(INTR_CACHE, exist_ok=True)
    cache = f"{INTR_CACHE}/{ep}.json"
    if os.path.isfile(cache):
        return json.load(open(cache))
    if not os.path.isfile(intr_json):
        raise Failure(f"intrinsics json missing: {intr_json}")
    allI = json.load(open(intr_json))
    if ep not in allI:
        raise Failure(f"episode not in intrinsics json: {ep}")
    tmp = f"{cache}.tmp.{os.getpid()}"
    json.dump(allI[ep], open(tmp, "w")); os.replace(tmp, cache)
    return allI[ep]


def load_poses(d, T, what):
    if not os.path.isdir(d):
        raise Failure(f"{what} directory missing: {d}")
    poses = {}
    for t in range(T):
        f = f"{d}/{t:06d}.npz"
        if not os.path.isfile(f):
            break
        poses[t] = np.load(f)["pose"].astype(np.float64)
    if not poses:
        raise Failure(f"{what}: no pose files in {d}")
    return poses


def count_files(d, suffix):
    return len([f for f in os.listdir(d) if f.endswith(suffix)]) if os.path.isdir(d) else 0


# ----------------------------------------------------------------------------- the tracker
def triangulate_point(P1, P2, x1, x2):
    Xh = cv2.triangulatePoints(P1, P2, np.asarray(x1, np.float64).reshape(2, 1), np.asarray(x2, np.float64).reshape(2, 1))
    return (Xh[:3] / Xh[3]).ravel()


def scaled_params(sz):
    """Pixel-unit parameters for a gripper whose apparent size is sz (<= 1) times the RAIL one. sz = 1 -> RAIL values.
    Floors mirror the probe's own resolution-scaling rule (max(3,..)|1 kernels, fb floor at sz 0.5, epipolar floor at 0.4)."""
    return dict(erode=max(3, int(RAIL["erode_px"] * sz)) | 1, dilate=max(5, int(RAIL["dilate_px"] * sz)) | 1,
                lk_win=max(7, int(RAIL["lk_win_px"] * sz)) | 1, min_dist=max(3, int(RAIL["min_dist_px"] * sz)),
                excl_r=max(3, int(RAIL["excl_r_px"] * sz)), fb=RAIL["fb_px"] * max(sz, 0.5))


def run(a, rec):
    t_start = time.time()
    ep = a.episode
    raw = f"{RAW_ROOT}/{ep}"
    meta_f = f"{raw}/metadata_{ep}.json"
    if not os.path.isfile(meta_f):
        raise Failure(f"metadata missing: {meta_f}")
    meta = json.load(open(meta_f))
    serials = {"ext1": str(meta.get("ext1_cam_serial", "")), "ext2": str(meta.get("ext2_cam_serial", "")),
               "wrist": str(meta.get("wrist_cam_serial", ""))}
    if not serials["ext1"] or not serials["ext2"]:
        raise Failure("exterior serials missing from metadata")
    rec["serials"] = serials
    cams = [("ext1", serials["ext1"]), ("ext2", serials["ext2"])]

    # --- calibration
    intr = load_intrinsics(ep, a.intrinsics_json)
    cam_f = f"{CAM_JSON}/{ep}_cameras.json"
    if not os.path.isfile(cam_f):
        raise Failure(f"PointWorld cameras json missing: {cam_f}")
    camj = json.load(open(cam_f))
    K, E, P = {}, {}, {}
    for cam, s in cams:
        if s not in intr:
            raise Failure(f"serial {s} ({cam}) not in intrinsics")
        if s not in camj or "optimized_extrinsics" not in camj[s]:
            raise Failure(f"serial {s} ({cam}) has no optimized_extrinsics")
        fx, cx, fy, cy = intr[s]["cameraMatrix"]
        iw, ih = intr[s].get("width", W), intr[s].get("height", H)
        sx, sy = W / iw, H / ih
        K[cam] = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])
        E[cam] = np.array(camj[s]["optimized_extrinsics"], np.float64)  # world(base) -> camera
        P[cam] = K[cam] @ E[cam][:3]
    R1, t1_, R2, t2_ = E["ext1"][:3, :3], E["ext1"][:3, 3], E["ext2"][:3, :3], E["ext2"][:3, 3]
    Rr = R2 @ R1.T; tr = t2_ - Rr @ t1_
    F = np.linalg.inv(K["ext2"]).T @ skew(tr) @ Rr @ np.linalg.inv(K["ext1"])

    # --- masks, frames, GT, CUT3R; frame counts checked against each other (products, not exit codes)
    mask_paths = {"ext1": a.mask_npz_ext1, "ext2": a.mask_npz_ext2}
    M = {cam: load_mask_npz(mask_paths[cam]) for cam, _ in cams}
    rec["n_frames_mask"] = {c: int(len(M[c])) for c in M}
    store_cam = f"{STORE_ROOT}/{ep}/dense/cam"
    rec["n_frames_store"] = count_files(store_cam, ".npz")
    if rec["n_frames_store"] == 0:
        raise Failure(f"store cam directory empty or missing: {store_cam}")
    T = min(min(len(M[c]) for c in M), rec["n_frames_store"])
    mp4 = {cam: f"{raw}/recordings/MP4/{s}.mp4" for cam, s in cams}
    gray = {cam: read_frames(mp4[cam], T) for cam, _ in cams}
    rec["n_frames_mp4"] = {c: int(len(gray[c])) for c in gray}
    T = min(T, min(len(gray[c]) for c in gray))
    timing = {"load": time.time() - t_start}
    rec["n_frames"] = int(rec["n_frames_store"])
    rec["n_frames_used"] = int(T)
    if T < 2:
        raise Failure(f"fewer than 2 usable frames (T={T})")
    gt = load_poses(store_cam, T, "store GT")
    T = min(T, len(gt))
    cut3r_dir = f"{CUT3R_ROOT}/{ep}/camera"
    cut = None
    try:
        cut = load_poses(cut3r_dir, T, "CUT3R finetuned preds")
    except Failure as e:
        rec["cut3r_missing"] = str(e)
    rec["n_frames_cut3r"] = int(len(cut)) if cut else 0

    # --- visibility window: first..last frame where BOTH masks are non-empty; gaps allowed inside
    nonempty = {cam: np.array([M[cam][t].any() for t in range(T)]) for cam in M}
    rec["mask_frames"] = {c: int(nonempty[c].sum()) for c in nonempty}
    both = nonempty["ext1"] & nonempty["ext2"]
    rec["mask_frames_both"] = int(both.sum())
    if not both.any():
        raise Failure("gripper never visible in both exterior masks")
    t0 = int(np.argmax(both)); t1 = int(T - 1 - np.argmax(both[::-1]))
    rec["window"] = [t0, t1]; rec["window_len"] = t1 - t0 + 1
    rec["window_gap_frames"] = int((~both[t0:t1 + 1]).sum())

    area = {cam: np.array([M[cam][t].sum() for t in range(T)], float) for cam in M}
    med_area = {cam: float(np.median(area[cam][t0:t1 + 1][area[cam][t0:t1 + 1] > 0])) for cam in M}
    t_seed = t0
    while t_seed < t1 and any(area[c][t_seed] < RAIL["area_frac"] * med_area[c] for c in M):
        t_seed += 1
    rec["t_seed"] = int(t_seed)

    # --- apparent size: pixels per centimetre from the mask-centroid triangulation at the seed frame
    cen = {}
    for cam in M:
        ys, xs = np.nonzero(M[cam][t_seed])
        cen[cam] = np.array([xs.mean(), ys.mean()])
    Xc = triangulate_point(P["ext1"], P["ext2"], cen["ext1"], cen["ext2"])
    depth = {cam: float((E[cam][:3, :3] @ Xc + E[cam][:3, 3])[2]) for cam in M}
    ppcm = {cam: float(K[cam][0, 0] / (max(depth[cam], 0.05) * 100)) for cam in M}
    sz = {cam: (min(1.0, ppcm[cam] / PPCM_RAIL) if a.size_scale is None else a.size_scale) for cam in M}
    if any(depth[c] <= 0 for c in depth):
        rec["prepass_warning"] = "mask-centroid triangulation behind a camera; RAIL thresholds used"
        sz = {cam: 1.0 for cam in M}
    prm = {cam: scaled_params(sz[cam]) for cam in M}
    thr_ep = RAIL["epi_px"] * max(min(sz.values()), 0.4)
    rec.update(prepass_depth_m=depth, ppcm=ppcm, ppcm_rail=PPCM_RAIL, size_scale=sz,
               mask_area_seed_px={c: float(area[c][t_seed]) for c in M},
               thresholds=dict(epi_px=thr_ep, rigid_mm=RAIL["rigid_m"] * 1000, min_common=RAIL["min_common"],
                               **{f"{c}_{k}": v for c in prm for k, v in prm[c].items()}))

    # --- seed + track (forward only, per view, re-seeding); tracks survive an empty-mask frame on the fb check alone
    max_pts = RAIL["max_pts"]
    tracks, seed_events = {}, {}
    for cam in gray:
        p = prm[cam]
        er = np.ones((p["erode"],) * 2, np.uint8); dil = np.ones((p["dilate"],) * 2, np.uint8)
        lk = dict(winSize=(p["lk_win"],) * 2, maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        tr_ = np.full((0, T, 2), np.nan, np.float32)
        alive = np.zeros(0, bool)
        n_seed_events = 0
        for t in range(t_seed, t1 + 1):
            if t > t_seed and alive.any():
                idx = np.where(alive)[0]
                p_prev = tr_[idx, t - 1].reshape(-1, 1, 2)
                p_next, st, _ = cv2.calcOpticalFlowPyrLK(gray[cam][t - 1], gray[cam][t], p_prev, None, **lk)
                p_back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[cam][t], gray[cam][t - 1], p_next, None, **lk)
                fb = np.linalg.norm(p_back - p_prev, axis=2).ravel()
                ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < p["fb"])
                pn = p_next.reshape(-1, 2)
                if nonempty[cam][t]:
                    dm = cv2.dilate(M[cam][t], dil)
                    inm = np.array([0 <= x < W and 0 <= y < H and dm[int(y), int(x)] > 0 for x, y in pn])
                else:  # gap: no mask this frame, keep the track if it stays in the image
                    inm = np.array([0 <= x < W and 0 <= y < H for x, y in pn])
                ok &= inm
                tr_[idx[ok], t] = pn[ok]
                alive[idx[~ok]] = False
            if alive.sum() < max_pts // 2 and area[cam][t] >= RAIL["area_frac"] * med_area[cam]:
                seedmask = cv2.erode(M[cam][t], er)
                if alive.any():
                    for x, y in tr_[alive, t]:
                        cv2.circle(seedmask, (int(x), int(y)), p["excl_r"], 0, -1)
                pts = cv2.goodFeaturesToTrack(gray[cam][t], max_pts - int(alive.sum()), RAIL["feat_quality"], p["min_dist"], mask=seedmask)
                if pts is not None:
                    pts = pts.reshape(-1, 2).astype(np.float32)
                    new = np.full((len(pts), T, 2), np.nan, np.float32); new[:, t] = pts
                    tr_ = np.concatenate([tr_, new]); alive = np.concatenate([alive, np.ones(len(pts), bool)])
                    n_seed_events += 1
        tracks[cam] = tr_; seed_events[cam] = n_seed_events
    timing["track"] = time.time() - t_start - sum(timing.values())
    lens = {cam: (~np.isnan(tracks[cam][:, :, 0])).sum(1) for cam in tracks}
    rec["n_tracks"] = {c: int(len(tracks[c])) for c in tracks}
    rec["seed_events"] = seed_events
    rec["track_len_median"] = {c: (float(np.median(lens[c])) if len(lens[c]) else 0.0) for c in lens}
    if any(len(tracks[c]) == 0 for c in tracks):
        raise Failure("no tracks seeded in " + ",".join(c for c in tracks if len(tracks[c]) == 0))

    # --- cross-view pairing by accumulated epipolar residual (vectorised over the second view; same numbers as the probe)
    A1, A2 = tracks["ext1"], tracks["ext2"]
    ok1, ok2 = ~np.isnan(A1[:, :, 0]), ~np.isnan(A2[:, :, 0])
    n1, n2 = len(A1), len(A2)
    X2h = np.concatenate([np.nan_to_num(A2.astype(np.float64)), np.ones((n2, T, 1))], axis=2)   # (n2, T, 3)
    L1 = X2h @ F                                                                                   # epipolar lines in view 1
    res = np.full((n1, n2), np.inf)
    for i in range(n1):
        com = ok1[i][None, :] & ok2                                                                # (n2, T)
        js = np.where(com.sum(1) >= RAIL["min_common"])[0]                                         # candidates with overlap
        if len(js) == 0:
            continue
        x1 = np.c_[np.nan_to_num(A1[i].astype(np.float64)), np.ones(T)]                            # (T, 3)
        l2 = x1 @ F.T                                                                              # lines in view 2 (T, 3)
        d2 = np.abs((X2h[js] * l2[None]).sum(2)) / np.hypot(l2[:, 0], l2[:, 1])[None]             # (len(js), T)
        d1 = np.abs((x1[None] * L1[js]).sum(2)) / np.hypot(L1[js, :, 0], L1[js, :, 1])
        res[i, js] = np.nanmedian(np.where(com[js], 0.5 * (d1 + d2), np.nan), axis=1)
    pairs, used2 = [], set()
    for i in np.argsort(res.min(1)):
        j = int(np.argmin(res[i]))
        if res[i, j] < thr_ep and j not in used2 and np.argmin(res[:, j]) == i:
            pairs.append((i, j)); used2.add(j)
    timing["pair"] = time.time() - t_start - sum(timing.values())
    rec["pairs"] = len(pairs)
    rec["epipolar_px_median"] = float(np.median([res[i, j] for i, j in pairs])) if pairs else float("nan")
    if len(pairs) < 3:
        raise Failure(f"fewer than 3 cross-view pairs ({len(pairs)})")

    # --- triangulate per frame
    X = np.full((len(pairs), T, 3), np.nan)
    reproj = []
    for k, (i, j) in enumerate(pairs):
        com = np.where(ok1[i] & ok2[j])[0]
        x1 = A1[i, com].T.astype(np.float64); x2 = A2[j, com].T.astype(np.float64)
        Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], x1, x2)
        Xw = (Xh[:3] / Xh[3]).T
        X[k, com] = Xw
        for cam, xx in (("ext1", x1), ("ext2", x2)):
            pr = (P[cam] @ np.c_[Xw, np.ones(len(Xw))].T); pr = pr[:2] / pr[2]
            reproj.append(np.linalg.norm(pr - xx, axis=0))
    reproj = np.concatenate(reproj)
    rec["reproj_px"] = dict(median=float(np.median(reproj)), p90=float(np.percentile(reproj, 90)))

    # --- rigid alignment to the reference frame with a growing body model
    n_tri = (~np.isnan(X[:, :, 0])).sum(0)
    cand = np.where(n_tri >= RAIL["min_ref_pairs"])[0]
    if len(cand) == 0:
        raise Failure(f"no frame with >= {RAIL['min_ref_pairs']} triangulated pairs")
    t_ref = int(cand[0])
    X0 = np.full((len(pairs), 3), np.nan)
    seed0 = ~np.isnan(X[:, t_ref, 0])
    X0[seed0] = X[seed0, t_ref]
    spread = np.linalg.norm(X0[seed0] - X0[seed0].mean(0), axis=1)
    rec["t_ref"] = t_ref
    rec["ref_cloud"] = dict(n=int(seed0.sum()), extent_cm=float(spread.max() * 100))
    rec["ref_depth_m"] = {cam: float(np.linalg.norm(E[cam][:3, :3] @ X0[seed0].T + E[cam][:3, 3:], axis=0).mean()) for cam in E}
    rec["tri_pairs_per_frame"] = dict(median=float(np.median(n_tri[t0:t1 + 1])), min=int(n_tri[t0:t1 + 1].min()),
                                      frames_ge3=int((n_tri[t0:t1 + 1] >= 3).sum()))
    P0 = gt[t_ref]  # the only GT before scoring: lens pose at the reference frame (hand-eye stand-in)
    thr_in = RAIL["rigid_m"]
    rows, pred, pred_frames, gtl, cutl, n_inl_l, resid_l = [], [], [], [], [], [], []
    for t in range(t0, t1 + 1):
        have = ~np.isnan(X0[:, 0]) & ~np.isnan(X[:, t, 0])
        R, tt, inl = ransac_kabsch(X0[have], X[have, t], np.ones(have.sum()), thr_in, iters=RAIL["ransac_iters"]) \
            if have.sum() >= 3 else (None, None, None)
        if R is None:
            rows.append(dict(frame=t, n_pts=int(have.sum()), n_inl=0, rigid_resid_mm=np.nan, pos_err_cm=np.nan, rot_err_deg=np.nan,
                             mask_both=int(both[t])))
            continue
        Mt = np.eye(4); Mt[:3, :3] = R; Mt[:3, 3] = tt
        newp = np.isnan(X0[:, 0]) & ~np.isnan(X[:, t, 0])
        if newp.any():
            X0[newp] = (X[newp, t] - tt) @ R      # R^T (x - t): register into the reference frame
        Pt = Mt @ P0
        e_pos = np.linalg.norm(Pt[:3, 3] - gt[t][:3, 3]) * 100
        e_rot = rot_angle_deg(Pt[:3, :3].T @ gt[t][:3, :3])
        resid = np.linalg.norm((X0[have][inl] @ R.T + tt) - X[have, t][inl], axis=1)
        rows.append(dict(frame=t, n_pts=int(have.sum()), n_inl=int(inl.sum()), rigid_resid_mm=float(np.median(resid) * 1000),
                         pos_err_cm=float(e_pos), rot_err_deg=float(e_rot), mask_both=int(both[t])))
        pred.append(Pt); pred_frames.append(t); gtl.append(gt[t]); n_inl_l.append(int(inl.sum())); resid_l.append(float(np.median(resid) * 1000))
        if cut is not None and t in cut:
            cutl.append(cut[t])
    K_ = len(pred)
    rec["n_anchored"] = K_
    rec["coverage_frac"] = K_ / rec["n_frames"]
    rec["anchored_frac_window"] = K_ / rec["window_len"]
    rec["model_points_final"] = int((~np.isnan(X0[:, 0])).sum())
    rec["model_extent_cm"] = float(np.nanmax(np.linalg.norm(X0 - np.nanmean(X0, 0), axis=1)) * 100)
    if K_ == 0:
        raise Failure("no frame solved (no frame with >= 3 model points / RANSAC inliers)")
    pe = np.array([r["pos_err_cm"] for r in rows if r["n_inl"] >= 3]); re_ = np.array([r["rot_err_deg"] for r in rows if r["n_inl"] >= 3])
    n_inl_a = np.array(n_inl_l); n_pts_a = np.array([r["n_pts"] for r in rows if r["n_inl"] >= 3])
    rec["inliers"] = dict(median=float(np.median(n_inl_a)), min=int(n_inl_a.min()))
    rec["rigid_resid_mm"] = dict(median=float(np.median(resid_l)), p90=float(np.percentile(resid_l, 90)))
    rec["rigidity_bad_frac"] = float(np.mean(n_inl_a < 0.5 * n_pts_a))
    rec["rigidity_bad_def"] = "anchored frames where fewer than half of the visible model points pass the 5 mm rigid test"
    jumps_cm, jumps_deg, gaps = [], [], []
    for i in range(1, K_):
        jumps_cm.append(np.linalg.norm(pred[i][:3, 3] - pred[i - 1][:3, 3]) * 100)
        jumps_deg.append(rot_angle_deg(pred[i - 1][:3, :3].T @ pred[i][:3, :3]))
        gaps.append(pred_frames[i] - pred_frames[i - 1])
    rec["max_jump_cm"] = float(max(jumps_cm)) if jumps_cm else 0.0
    rec["max_jump_deg"] = float(max(jumps_deg)) if jumps_deg else 0.0
    rec["max_anchor_gap_frames"] = int(max(gaps)) if gaps else 0
    rec["pos_err_median_cm"] = float(np.median(pe)); rec["pos_err_p90_cm"] = float(np.percentile(pe, 90)); rec["pos_err_max_cm"] = float(pe.max())
    rec["rot_err_median_deg"] = float(np.median(re_)); rec["rot_err_p90_deg"] = float(np.percentile(re_, 90)); rec["rot_err_max_deg"] = float(re_.max())
    m_rig = traj_metrics(pred, gtl)
    rec["rig_ate"], rec["rig_rpe_t"], rec["rig_rpe_rot"], rec["sim3_scale"] = m_rig["ate"], m_rig["rpe_trans"], m_rig["rpe_rot"], m_rig["sim3_scale"]
    if cut is not None and len(cutl) == K_:
        m_cut = traj_metrics(cutl, gtl)
        rec["cut3r_ate_same_frames"], rec["cut3r_rpe_t_same_frames"], rec["cut3r_rpe_rot_same_frames"], rec["cut3r_sim3_scale_same_frames"] = \
            m_cut["ate"], m_cut["rpe_trans"], m_cut["rpe_rot"], m_cut["sim3_scale"]
    else:
        rec["cut3r_ate_same_frames"] = rec["cut3r_rpe_t_same_frames"] = rec["cut3r_rpe_rot_same_frames"] = rec["cut3r_sim3_scale_same_frames"] = None
        m_cut = None

    timing["solve"] = time.time() - t_start - sum(timing.values())
    rec["timing_s"] = timing
    # --- products
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/anchors.npz", frames=np.array(pred_frames, np.int32), poses=np.array(pred), n_inl=n_inl_a.astype(np.int32),
             rigid_resid_mm=np.array(resid_l), lens_ref=P0[:3, 3], model_centroid=np.nanmean(X0, 0),
             model_extent=float(np.nanmax(np.linalg.norm(X0 - np.nanmean(X0, 0), axis=1))), t_ref=t_ref,
             window=np.array([t0, t1]), episode=ep, serials=np.array([s for _, s in cams]))
    with open(f"{a.out}/per_frame.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[-1].keys())); wr.writeheader(); wr.writerows(rows)
    if a.save_tracks:
        np.savez_compressed(f"{a.out}/tracks.npz", X=X, X0=X0, pairs=np.array(pairs), t_ref=t_ref, t0=t0, t1=t1,
                            tr1=tracks["ext1"], tr2=tracks["ext2"], P_ext1=P["ext1"], P_ext2=P["ext2"], E_ext1=E["ext1"], E_ext2=E["ext2"])
    rec["elapsed_s"] = time.time() - t_start
    if a.video:
        tv = time.time()
        make_side_by_side(a, ep, T, t0, t1, both, mp4, M, tracks, pairs, rows, pred, pred_frames, P, gt, cut, m_rig, m_cut)
        rec["video"] = f"{a.out}/rig_side_by_side.mp4"; rec["video_s"] = time.time() - tv
        rec["video_bytes"] = os.path.getsize(rec["video"]) if os.path.isfile(rec["video"]) else 0
    return rec


# ----------------------------------------------------------------------------- 3-panel video (wrist + inset | ext1 | ext2), as in the probe
def make_side_by_side(a, ep, T, t0, t1, both, mp4, M, tracks, pairs, rows, pred, pred_frames, P, gt_all, cut_all, rig_m, cut_m):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sam3_gripper_masks import FfmpegWriter
    PW, PH = 640, 360
    caps = {cam: cv2.VideoCapture(mp4[cam]) for cam in mp4}
    pair1 = {i for i, _ in pairs}; pair2 = {j for _, j in pairs}
    pred_by_t = {int(f): p for f, p in zip(pred_frames, pred)}
    row_by_t = {r["frame"]: r for r in rows}
    sf = [int(f) for f in pred_frames]
    cut_c = None
    if cut_all is not None and all(t in cut_all for t in sf) and len(sf) >= 3:
        sc_, Rc, tc = umeyama(np.array([cut_all[t][:3, 3] for t in sf]), np.array([gt_all[t][:3, 3] for t in sf]))
        cut_c = {t: sc_ * Rc @ cut_all[t][:3, 3] + tc for t in cut_all}
    G = np.array([gt_all[t][:3, 3] for t in range(t0, t1 + 1)])
    lo, hi = G[:, :2].min(0) - 0.05, G[:, :2].max(0) + 0.05
    span = (hi - lo).max(); cen = (lo + hi) / 2; lo = cen - span / 2
    IW, IH, IX, IY = 230, 230, PW - 240, PH - 240

    def to_inset(xy):
        return int(IX + (xy[0] - lo[0]) / span * (IW - 1)), int(IY + (IH - 1) - (xy[1] - lo[1]) / span * (IH - 1))

    def banner(img, text, color):
        cv2.rectangle(img, (0, PH // 2 - 22), (PW, PH // 2 + 22), color, -1)
        cv2.putText(img, text, (PW // 2 - 7 * len(text), PH // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    rgb_dir = f"{STORE_ROOT}/{ep}/dense/rgb"
    writer = FfmpegWriter(f"{a.out}/rig_side_by_side.mp4", 3 * PW, PH, 15)
    cm_txt = (f"ATE rig {rig_m['ate']:.4f} vs CUT3R {cut_m['ate']:.4f}" if cut_m else f"ATE rig {rig_m['ate']:.4f}")
    rpe_txt = (f"RPE_t rig {rig_m['rpe_trans']:.4f} vs {cut_m['rpe_trans']:.4f}   RPE_rot {rig_m['rpe_rot']:.2f} vs {cut_m['rpe_rot']:.2f}"
               if cut_m else f"RPE_t rig {rig_m['rpe_trans']:.4f}   RPE_rot {rig_m['rpe_rot']:.2f}")
    for t in range(T):
        in_win = t0 <= t <= t1
        r = row_by_t.get(t)
        solved = t in pred_by_t
        wimg = cv2.imread(f"{rgb_dir}/{t:06d}.png")
        wimg = np.zeros((PH, PW, 3), np.uint8) if wimg is None else cv2.resize(wimg, (PW, PH), interpolation=cv2.INTER_CUBIC)
        cv2.putText(wimg, f"wrist f{t}  rig = 2 static cams, tracked gripper points, Kabsch", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ov = wimg.copy(); cv2.rectangle(ov, (IX - 6, IY - 6), (IX + IW + 6, IY + IH + 6), (30, 30, 30), -1)
        wimg = cv2.addWeighted(ov, 0.75, wimg, 0.25, 0)
        cv2.putText(wimg, "top view (base x,y)", (IX + 4, IY + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        if in_win:
            for tt in range(t0, min(t, t1) + 1):
                if tt > t0:
                    cv2.line(wimg, to_inset(gt_all[tt - 1][:3, 3]), to_inset(gt_all[tt][:3, 3]), (0, 220, 0), 2)
                    if cut_c is not None:
                        cv2.line(wimg, to_inset(cut_c[tt - 1]), to_inset(cut_c[tt]), (255, 120, 0), 1)
                    if tt in pred_by_t and (tt - 1) in pred_by_t:
                        cv2.line(wimg, to_inset(pred_by_t[tt - 1][:3, 3]), to_inset(pred_by_t[tt][:3, 3]), (0, 0, 255), 2)
            cv2.circle(wimg, to_inset(gt_all[t][:3, 3]), 4, (0, 220, 0), -1)
            if cut_c is not None:
                cv2.circle(wimg, to_inset(cut_c[t]), 4, (255, 120, 0), -1)
            if solved:
                cv2.circle(wimg, to_inset(pred_by_t[t][:3, 3]), 4, (0, 0, 255), -1)
        y = 44
        for text in ("green GT   red rig   blue CUT3R (Sim3-aligned)", f"window f{t0}-{t1}: {cm_txt}", rpe_txt):
            cv2.putText(wimg, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1); y += 20
        if solved:
            cv2.putText(wimg, f"this frame: lens pos err {r['pos_err_cm']:.1f} cm  rot err {r['rot_err_deg']:.1f} deg  inliers {r['n_inl']}/{r['n_pts']}",
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        panels = [wimg]
        for cam, tr_, pset in (("ext1", tracks["ext1"], pair1), ("ext2", tracks["ext2"], pair2)):
            ok_, img = caps[cam].read()
            if not ok_:
                img = np.zeros((H, W, 3), np.uint8)
            elif img.shape[1] != W or img.shape[0] != H:
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            cnts, _ = cv2.findContours(M[cam][t], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 200, 0), 2)
            n_tr = 0
            for n in range(len(tr_)):
                if np.isnan(tr_[n, t, 0]):
                    continue
                n_tr += 1
                x, y_ = tr_[n, t]
                cv2.circle(img, (int(x), int(y_)), 5, (255, 200, 0) if n in pset else (140, 140, 140), -1)

            def proj(pt):
                q = P[cam] @ np.r_[pt, 1]; return int(q[0] / q[2]), int(q[1] / q[2])
            if in_win:
                cv2.drawMarker(img, proj(gt_all[t][:3, 3]), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
                if cut_c is not None:
                    c = np.array(proj(cut_c[t])); cv2.rectangle(img, tuple(c - 9), tuple(c + 9), (255, 120, 0), 3)
                if solved:
                    cv2.circle(img, proj(pred_by_t[t][:3, 3]), 12, (0, 0, 255), 3)
            img = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_AREA)
            cv2.putText(img, f"{cam} f{t}  tracks {n_tr}  cyan=cross-view paired  + GT lens  o rig lens  [] CUT3R", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            panels.append(img)
        if not in_win:
            for pnl in panels:
                banner(pnl, "GRIPPER NOT IN ALL CAMERAS", (0, 0, 160))
        elif not solved:
            for pnl in panels:
                banner(pnl, "RIG NOT SOLVED (<3 tracked points)" if both[t] else "MASK GAP: RIG NOT SOLVED", (0, 140, 180))
        writer.write(np.concatenate(panels, axis=1))
    writer.close()
    for c in caps.values():
        c.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--mask_npz_ext1", required=True)
    ap.add_argument("--mask_npz_ext2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", action="store_true", help="write DIR/rig_side_by_side.mp4 (wrist+inset | ext1 | ext2)")
    ap.add_argument("--save_tracks", action="store_true", help="also write DIR/tracks.npz (2D tracks, pairs, 3D points, P/E)")
    ap.add_argument("--size_scale", type=float, default=None, help="override the apparent-size scale (diagnostics only)")
    ap.add_argument("--intrinsics_json", default=INTR_JSON)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rec = dict(episode=a.episode, tool=TOOL_VERSION, status="ok", failure_reason=None, out_dir=os.path.abspath(a.out),
               masks={"ext1": a.mask_npz_ext1, "ext2": a.mask_npz_ext2}, n_anchored=0, coverage_frac=0.0)
    t_start = time.time()
    try:
        run(a, rec)
    except Failure as e:
        rec["status"] = "failed"; rec["failure_reason"] = str(e)
    except Exception as e:  # never leave an episode without a record
        import traceback
        rec["status"] = "failed"; rec["failure_reason"] = f"unexpected {e.__class__.__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
    rec.setdefault("elapsed_s", time.time() - t_start)
    rec["elapsed_total_s"] = time.time() - t_start

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    json.dump(clean(rec), open(f"{a.out}/record.json", "w"), indent=1)
    keys = ["status", "failure_reason", "window", "mask_frames_both", "n_anchored", "coverage_frac", "pairs", "pos_err_median_cm",
            "rot_err_median_deg", "rig_ate", "rig_rpe_t", "rig_rpe_rot", "cut3r_ate_same_frames", "size_scale", "elapsed_s"]
    print(json.dumps({k: clean(rec.get(k)) for k in keys}))
    sys.exit(0)


if __name__ == "__main__":
    main()
