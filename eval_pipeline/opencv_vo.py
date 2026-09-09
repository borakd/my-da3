#!/usr/bin/env python
"""OpenCV monocular visual-odometry control arm (v0) for the DROID wrist harness.

Design of record: OPENCV_VO_DESIGN.md (repo root). Every decided choice is hard-coded;
every still-open decision is a CLI switch so the options can be scored head to head.

Pipeline (per scene, offline, images only + the model's own predicted focal):
  frames  : the eval loader's 320x192 cover frames (load_images_cover), grayscale
  K       : fx=fy = per-scene median of the model's predicted focal (preds camera/*.npz), pp = image centre
  frontend: Shi-Tomasi (1000, 0.01, 5) + pyramidal LK (OpenCV defaults) + forward-backward check < 1 px;
            persistent tracks, new corners only at keyframes (existing tracks masked out)
  bootstrap: frame 0 vs first frame whose median track displacement >= 5 px; findEssentialMat(RANSAC 0.5 px,
            0.999) + recoverPose; cheirality count >= min_inliers else keep waiting; triangulate (cheirality +
            reprojection < 2 px); frames 1..b-1 filled retroactively by PnP against the bootstrap map
  tracking: solvePnPRansac(ITERATIVE, 2 px, 0.999, 1000) + solvePnPRefineLM; no initial guess; no fallback solver
  keyframe: median track displacement since last keyframe >= 5 px -> triangulate new points (KF-to-KF),
            detect new corners
  output  : camera/%06d.npz with pose (c2w, 4x4) + intrinsics, for every frame; diag.csv per scene

Open-decision switches: --new_points, --cull, --min_inliers, --fail_policy, --refine, --lever (see argparse).
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import cv2
import PIL.Image

cv2.setNumThreads(1)

# ----------------------------------------------------------------------------- decided constants
MAX_CORNERS, QUALITY, MIN_DIST = 1000, 0.01, 5
FB_THRESH = 1.0
PARALLAX_PX = 5.0
E_THRESH, E_CONF = 0.5, 0.999
PNP_THRESH, PNP_CONF, PNP_ITERS = 2.0, 0.999, 1000   # PNP_THRESH may be overridden by --pnp_thresh
TRI_REPROJ = 2.0
MAX_TRACK_LEN_BEFORE_BOOT = 400


def load_gray_cover(path, size=320, patch=16):
    """Exactly load_images_cover's geometry (scale to cover 320x192, centre crop), then grayscale."""
    img = PIL.Image.open(path).convert("RGB")
    W1, H1 = img.size
    tw = max(patch, (int(size) // patch) * patch)
    th = ((H1 * tw + W1 * patch - 1) // (W1 * patch)) * patch
    scale = max(tw / W1, th / H1)
    new_w = max(tw, int(round(W1 * scale))); new_h = max(th, int(round(H1 * scale)))
    interp = PIL.Image.BICUBIC if scale >= 1 else PIL.Image.LANCZOS
    img = img.resize((new_w, new_h), interp)
    left = (new_w - tw) // 2; top = (new_h - th) // 2
    img = img.crop((left, top, left + tw, top + th))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)


def selfcal_focal(G, W, H, n_pairs=40, gap=6):
    """Self-calibration from the images alone: sweep candidate focals and keep the one under which the two-view
    geometry across many parallax-gated frame pairs is most consistent (most inliers at 1 px in findEssentialMat,
    tie-break by mean Sampson error of the inliers). Assumes pp at the image centre and square pixels.
    Returns (focal_px, n_pairs_used)."""
    n = len(G); pairs = []
    for i in np.linspace(0, max(0, n - gap - 1), n_pairs).astype(int):
        j = i + gap
        p0 = cv2.goodFeaturesToTrack(G[i], MAX_CORNERS, QUALITY, MIN_DIST)
        if p0 is None or len(p0) < 30: continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(G[i], G[j], p0, None); p0b, stb, _ = cv2.calcOpticalFlowPyrLK(G[j], G[i], p1, None)
        ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm((p0 - p0b)[:, 0], axis=1) < FB_THRESH)
        a, b = p0[ok, 0].astype(np.float64), p1[ok, 0].astype(np.float64)
        if len(a) < 30: continue
        disp = np.linalg.norm(b - a, axis=1)
        if np.median(disp) < PARALLAX_PX: continue
        keep = disp >= 1.0                                     # static-track lever, same rule as the main loop
        if keep.sum() < 30: continue
        pairs.append((a[keep], b[keep]))
    if len(pairs) < 5:
        return None, len(pairs)
    def score(f):
        K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]]); tot = 0
        for a, b in pairs:
            E, inl = cv2.findEssentialMat(a, b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None and E.shape == (3, 3) and inl is not None: tot += int(inl.sum())
        return tot
    coarse = np.arange(100, 501, 10.0); sc = [score(f) for f in coarse]
    f0 = coarse[int(np.argmax(sc))]
    fine = np.arange(f0 - 10, f0 + 10.1, 2.0); sf = [score(f) for f in fine]
    return float(fine[int(np.argmax(sf))]), len(pairs)


def model_K(preds_scene_dir, W, H, focal_mode="model", scene_dir=None):
    """Camera matrix in the 320x192 cover frame. focal_mode: 'model' = per-scene median of the model's
    predicted focal (decision 0.1b); a float = fixed nominal focal (strict zero-shot); 'gt' = calibrated focal
    from the scene's cam npz rescaled by the cover factor (DIAGNOSTIC ONLY, reads GT intrinsics)."""
    if focal_mode == "gt":
        z = np.load(os.path.join(scene_dir, "dense", "cam", "000000.npz"))
        Kn = z["intrinsic"]
        from PIL import Image
        W1, H1 = Image.open(sorted(glob.glob(os.path.join(scene_dir, "dense", "rgb", "*.png")))[0]).size
        f = float(Kn[0, 0]) * max(W / W1, H / H1)
    elif focal_mode == "selfcal":
        raise ValueError("selfcal is resolved in run_scene (needs the frames)")
    elif focal_mode == "model":
        fs = sorted(glob.glob(os.path.join(preds_scene_dir, "camera", "*.npz")))
        if not fs:
            raise FileNotFoundError(f"no model camera npz under {preds_scene_dir}")
        f = float(np.median([np.load(x)["intrinsics"][0, 0] for x in fs]))
    else:
        f = float(focal_mode)
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    return K


def rt_to_T(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(t).ravel(); return T


class Tracks:
    """Live 2D tracks (structure of arrays)."""

    def __init__(self):
        self.pos = np.zeros((0, 2), np.float32)     # current position
        self.kf_pos = np.zeros((0, 2), np.float32)  # position at the last keyframe (or at detection)
        self.mp = np.zeros(0, np.int64)             # map point id or -1
        self.ids = np.zeros(0, np.int64)
        self._next = 0

    def __len__(self): return len(self.pos)

    def add(self, pts):
        n = len(pts)
        self.pos = np.vstack([self.pos, pts.astype(np.float32)])
        self.kf_pos = np.vstack([self.kf_pos, pts.astype(np.float32)])
        self.mp = np.concatenate([self.mp, -np.ones(n, np.int64)])
        self.ids = np.concatenate([self.ids, np.arange(self._next, self._next + n)])
        self._next += n

    def keep(self, mask):
        self.pos, self.kf_pos, self.mp, self.ids = self.pos[mask], self.kf_pos[mask], self.mp[mask], self.ids[mask]


def detect(gray, tracks, W, H):
    mask = np.full((H, W), 255, np.uint8)
    for x, y in tracks.pos:
        cv2.circle(mask, (int(round(x)), int(round(y))), MIN_DIST, 0, -1)
    p = cv2.goodFeaturesToTrack(gray, MAX_CORNERS, QUALITY, MIN_DIST, mask=mask)
    return np.zeros((0, 2), np.float32) if p is None else p[:, 0]


def lk_step(g0, g1, tracks):
    if len(tracks) == 0:
        return np.zeros(0, bool)
    p0 = tracks.pos.reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None)
    p0b, stb, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None)
    ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm((p0 - p0b)[:, 0], axis=1) < FB_THRESH)
    H, W = g1.shape
    ok &= (p1[:, 0, 0] >= 0) & (p1[:, 0, 0] < W) & (p1[:, 0, 1] >= 0) & (p1[:, 0, 1] < H)
    tracks.pos = p1[:, 0]
    return ok


def triangulate_pair(K0, K1, T0_w2c, T1_w2c, a, b):
    """Triangulate world points from two w2c poses (each view with its own K); returns X (N,3), accept mask
    (cheirality in both cameras + reprojection < 2 px in both views)."""
    P0 = K0 @ T0_w2c[:3]; P1 = K1 @ T1_w2c[:3]
    Xh = cv2.triangulatePoints(P0, P1, a.T.astype(np.float64), b.T.astype(np.float64))
    X = (Xh[:3] / np.where(np.abs(Xh[3]) < 1e-12, 1e-12, Xh[3])).T
    Xw1 = np.c_[X, np.ones(len(X))]
    z0 = (T0_w2c @ Xw1.T)[2]; z1 = (T1_w2c @ Xw1.T)[2]
    def proj(P):
        x = P @ Xw1.T; return (x[:2] / np.where(np.abs(x[2]) < 1e-12, 1e-12, x[2])).T
    r0 = np.linalg.norm(proj(P0) - a, axis=1); r1 = np.linalg.norm(proj(P1) - b, axis=1)
    acc = (z0 > 0) & (z1 > 0) & (r0 < TRI_REPROJ) & (r1 < TRI_REPROJ) & np.isfinite(X).all(1)
    return X, acc


def pnp(K, X, x, min_inliers):
    """Returns (T_w2c or None, inlier index array or None)."""
    if len(X) < 6:
        return None, None
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(X.astype(np.float64), x.astype(np.float64), K, None,
                                                 flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=PNP_THRESH,
                                                 confidence=PNP_CONF, iterationsCount=PNP_ITERS)
    except cv2.error:
        return None, None
    if not ok or inl is None or len(inl) < min_inliers:
        return None, None
    inl = inl[:, 0]
    try:
        rvec, tvec = cv2.solvePnPRefineLM(X[inl].astype(np.float64), x[inl].astype(np.float64), K, None, rvec, tvec)
    except cv2.error:
        pass
    R, _ = cv2.Rodrigues(rvec)
    return rt_to_T(R, tvec), inl


# ----------------------------------------------------------------------------- optional local BA (3.4)
def local_ba(K, kf_T_w2c, kf_obs, mp_xyz, window):
    """Windowed bundle adjustment over the last `window` keyframes (oldest in window fixed) and the map
    points they observe. Reprojection residuals, Huber loss, scipy least_squares with a sparse Jacobian.
    kf_T_w2c: list of 4x4 (modified in place for the window); kf_obs: list of dict mp_id -> (u,v);
    mp_xyz: dict mp_id -> np.array(3) (modified in place)."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation as Rot
    kfs = list(range(max(0, len(kf_T_w2c) - window), len(kf_T_w2c)))
    if len(kfs) < 2:
        return
    fixed = kfs[0]; free = kfs[1:]
    pts = sorted({m for k in kfs for m in kf_obs[k] if m in mp_xyz})
    if len(pts) < 10:
        return
    pidx = {m: i for i, m in enumerate(pts)}; kidx = {k: i for i, k in enumerate(free)}
    obs = [(k, m, kf_obs[k][m]) for k in kfs for m in kf_obs[k] if m in pidx]
    x0 = np.concatenate([np.concatenate([Rot.from_matrix(kf_T_w2c[k][:3, :3]).as_rotvec(), kf_T_w2c[k][:3, 3]]) for k in free]
                        + [mp_xyz[m] for m in pts])
    n_cam = 6 * len(free)
    fixed_T = kf_T_w2c[fixed]
    def unpack(x):
        Ts = {fixed: fixed_T}
        for k in free:
            i = kidx[k]; rv = x[6 * i:6 * i + 3]; t = x[6 * i + 3:6 * i + 6]
            Ts[k] = rt_to_T(Rot.from_rotvec(rv).as_matrix(), t)
        P = x[n_cam:].reshape(-1, 3)
        return Ts, P
    def resid(x):
        Ts, P = unpack(x); out = np.empty(2 * len(obs))
        for i, (k, m, uv) in enumerate(obs):
            Xc = Ts[k][:3, :3] @ P[pidx[m]] + Ts[k][:3, 3]
            z = Xc[2] if abs(Xc[2]) > 1e-9 else 1e-9
            out[2 * i] = K[0, 0] * Xc[0] / z + K[0, 2] - uv[0]
            out[2 * i + 1] = K[1, 1] * Xc[1] / z + K[1, 2] - uv[1]
        return out
    S = lil_matrix((2 * len(obs), len(x0)), dtype=int)
    for i, (k, m, _) in enumerate(obs):
        if k in kidx:
            S[2 * i:2 * i + 2, 6 * kidx[k]:6 * kidx[k] + 6] = 1
        S[2 * i:2 * i + 2, n_cam + 3 * pidx[m]:n_cam + 3 * pidx[m] + 3] = 1
    sol = least_squares(resid, x0, jac_sparsity=S, loss="huber", f_scale=PNP_THRESH, max_nfev=30, x_scale="jac")
    Ts, P = unpack(sol.x)
    for k in free:
        kf_T_w2c[k] = Ts[k]
    for m in pts:
        mp_xyz[m] = P[pidx[m]]


# ----------------------------------------------------------------------------- the pipeline
def run_scene(scene_dir, preds_scene_dir, out_scene_dir, args):
    t_start = time.time()
    fs = sorted(glob.glob(os.path.join(scene_dir, "dense", "rgb", "*.png")))
    n = len(fs)
    G = [load_gray_cover(f) for f in fs]
    H, W = G[0].shape
    selfcal_pairs = None
    def K_of(f):
        return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    if args.focal == "selfcal":
        f_sc, selfcal_pairs = selfcal_focal(G, W, H)
        Ks = [K_of(f_sc) if f_sc else model_K(preds_scene_dir, W, H, "model", scene_dir)] * n
    elif args.focal.startswith("perframe:") or args.focal.startswith("causal:"):
        # INFERENCE-TIME focal: frame t uses only what the model has produced up to t.
        #   perframe:<preds_root> -> the model's focal for frame t itself
        #   causal:<preds_root>   -> running median of the model's focals for frames 0..t
        mode, root = args.focal.split(":", 1)
        scene_name = os.path.basename(scene_dir)
        fpf = []
        for f in range(n):
            z = np.load(os.path.join(root, scene_name, "camera", f"{f:06d}.npz"))
            fpf.append(float(z["intrinsics"][0, 0]))
        if mode == "causal":
            fpf = [float(np.median(fpf[:f + 1])) for f in range(n)]
        Ks = [K_of(v) for v in fpf]
    else:
        Ks = [model_K(preds_scene_dir, W, H, args.focal, scene_dir)] * n
    K = Ks[0]                                   # kept for the (unused in v0) local BA

    poses_c2w = [None] * n
    diag = []                                   # per-frame dicts
    tracks = Tracks()
    mp_xyz = {}                                 # map point id -> world xyz
    mp_outl = {}                                # map point id -> consecutive PnP-outlier count
    next_mp = 0
    kf_T = []                                   # keyframe w2c poses
    kf_obs = []                                 # keyframe: mp_id -> (u,v)
    kf_frames = []
    booted = False
    boot_anchor_T = np.eye(4)                   # w2c of the frame the current bootstrap is anchored at
    pre_hist = []                               # (ids, pos) per frame since anchor, for retro PnP
    anchor_frame = 0
    last_T = np.eye(4); prev_T = None           # w2c of last / previous frame (for const-velocity)
    n_reboot = 0; n_fail = 0; consec_fail = 0

    old_mp = {}                                 # track id -> world point in the PREVIOUS scale segment (for scale hand-off)

    def new_bootstrap(frame_idx):
        nonlocal booted, boot_anchor_T, pre_hist, anchor_frame, old_mp
        if args.scale_handoff:
            old_mp = {int(i): mp_xyz[int(m)] for i, m in zip(tracks.ids, tracks.mp) if m >= 0 and int(m) in mp_xyz}
            tracks.mp[:] = -1                       # keep the tracks alive, forget their (old-scale) points
            tracks.kf_pos = tracks.pos.copy()
        else:
            old_mp = {}
            tracks.keep(np.zeros(len(tracks), bool))
        tracks.add(detect(G[frame_idx], tracks, W, H))
        booted = False
        boot_anchor_T = last_T.copy()
        anchor_frame = frame_idx
        pre_hist = [(tracks.ids.copy(), tracks.pos.copy())]

    def declare_keyframe(frame_idx, T_w2c):
        """Triangulate KF-to-KF, register observations, detect new corners, reset kf_pos."""
        nonlocal next_mp
        if kf_T:
            T_prev = kf_T[-1]
            cand = (tracks.mp < 0)
            if args.new_points == "kf":
                pass
            if cand.sum() >= 1:
                X, acc = triangulate_pair(Ks[kf_frames[-1]], Ks[frame_idx], T_prev, T_w2c, tracks.kf_pos[cand].astype(np.float64), tracks.pos[cand].astype(np.float64))
                ci = np.where(cand)[0][acc]
                for row, Xi in zip(ci, X[acc]):
                    tracks.mp[row] = next_mp; mp_xyz[next_mp] = Xi; mp_outl[next_mp] = 0; next_mp += 1
        kf_T.append(T_w2c.copy()); kf_frames.append(frame_idx)
        kf_obs.append({int(m): (float(u), float(v)) for m, (u, v) in zip(tracks.mp, tracks.pos) if m >= 0})
        if args.refine == "ba" and len(kf_T) >= 3:
            try:
                local_ba(K, kf_T, kf_obs, mp_xyz, window=args.ba_window)
            except Exception as e:  # BA must never kill a scene
                diag.append(dict(frame=frame_idx, event=f"ba_error:{type(e).__name__}"))
        tracks.add(detect(G[frame_idx], tracks, W, H))
        tracks.kf_pos = tracks.pos.copy()

    # ---- frame 0
    poses_c2w[0] = np.eye(4)
    new_bootstrap(0)
    diag.append(dict(frame=0, n_tracks=len(tracks), n_mp=0, inliers=-1, failed=0, keyframe=1, booted=0))

    for f in range(1, n):
        prev_pos = tracks.pos.copy()
        ok = lk_step(G[f - 1], G[f], tracks)
        if args.lever and ok.any():
            # 1.2 lever: on frames with clear parallax (median per-frame displacement >= 2 px) a track that did not
            # move (< 1 px) is provably not scene geometry (image-static gripper): kill it.
            step = np.linalg.norm(tracks.pos - prev_pos, axis=1)
            if np.median(step[ok]) >= 2.0:
                ok &= step >= 1.0
        tracks.keep(ok)
        # dead map points can never be re-observed (no descriptors): drop them
        rec = dict(frame=f, n_tracks=len(tracks), n_mp=int((tracks.mp >= 0).sum()), inliers=-1, failed=0, keyframe=0, booted=int(booted))

        if not booted:
            pre_hist.append((tracks.ids.copy(), tracks.pos.copy()))
            disp = np.linalg.norm(tracks.pos - tracks.kf_pos, axis=1) if len(tracks) else np.zeros(0)
            T_w2c = None
            if len(tracks) >= args.min_inliers and np.median(disp) >= PARALLAX_PX:
                a = tracks.kf_pos.astype(np.float64); b = tracks.pos.astype(np.float64)
                Ka, Kb = Ks[anchor_frame], Ks[f]
                an = (a - Ka[:2, 2]) / Ka[0, 0]; bn = (b - Kb[:2, 2]) / Kb[0, 0]     # each view normalised by its own K
                I3 = np.eye(3)
                E, inl = cv2.findEssentialMat(an, bn, I3, method=cv2.RANSAC, prob=E_CONF,
                                              threshold=E_THRESH / (0.5 * (Ka[0, 0] + Kb[0, 0])))
                if E is not None and E.shape == (3, 3) and inl is not None:
                    npass, R, t, _ = cv2.recoverPose(E, an, bn, I3, mask=inl.copy())
                    if npass >= args.min_inliers:
                        T_rel = rt_to_T(R, t)                       # anchor cam -> this cam (unit translation)
                        T_w2c = T_rel @ boot_anchor_T
                        X, acc = triangulate_pair(Ka, Kb, boot_anchor_T, T_w2c, a, b)
                        acc &= inl[:, 0].astype(bool)
                        if old_mp and acc.sum() >= 10:
                            # 2.15 scale hand-off: match the new segment's scale to the previous one through tracks
                            # that carried a map point before the re-bootstrap (depth ratio in the anchor camera)
                            ratios = []
                            for row in np.where(acc)[0]:
                                tid = int(tracks.ids[row])
                                if tid in old_mp:
                                    zo = (boot_anchor_T @ np.r_[old_mp[tid], 1.0])[2]; zn = (boot_anchor_T @ np.r_[X[row], 1.0])[2]
                                    if zo > 0 and zn > 0: ratios.append(zo / zn)
                            if len(ratios) >= 10:
                                sc = float(np.median(ratios))
                                if 0.2 < sc < 5.0:
                                    T_rel = rt_to_T(R, sc * np.asarray(t).ravel()); T_w2c = T_rel @ boot_anchor_T
                                    X, acc = triangulate_pair(Ka, Kb, boot_anchor_T, T_w2c, a, b); acc &= inl[:, 0].astype(bool)
                                    rec["event"] = f"handoff:{sc:.2f}:{len(ratios)}"
                            old_mp = {}
                        if acc.sum() >= args.min_inliers:
                            for row in np.where(acc)[0]:
                                tracks.mp[row] = next_mp; mp_xyz[next_mp] = X[row]; mp_outl[next_mp] = 0; next_mp += 1
                            booted = True
                            # retro-fill frames anchor+1 .. f-1 by PnP against the fresh map (non-causal; --no_retro_fill disables)
                            id2mp = {int(i): int(m) for i, m in zip(tracks.ids, tracks.mp) if m >= 0}
                            for k, (ids_k, pos_k) in enumerate(pre_hist[1:-1] if not args.no_retro_fill else [], start=anchor_frame + 1):
                                sel = [j for j, i in enumerate(ids_k) if int(i) in id2mp]
                                if len(sel) >= 6:
                                    Xk = np.array([mp_xyz[id2mp[int(ids_k[j])]] for j in sel]); xk = pos_k[sel]
                                    Tk, inl_k = pnp(Ks[k], Xk, xk, 4)
                                    if Tk is not None:
                                        poses_c2w[k] = np.linalg.inv(Tk)
                                        if k < len(diag):
                                            diag[k]["failed"] = 0; diag[k]["inliers"] = int(len(inl_k)); diag[k]["event"] = "retro"
                                        n_fail -= 1; continue
                                poses_c2w[k] = poses_c2w[k - 1] if poses_c2w[k - 1] is not None else np.linalg.inv(boot_anchor_T)
                            kf_T.clear(); kf_obs.clear(); kf_frames.clear()
                            kf_T.append(boot_anchor_T.copy()); kf_frames.append(anchor_frame)
                            kf_obs.append({int(m): (float(u), float(v)) for m, (u, v) in zip(tracks.mp, tracks.kf_pos) if m >= 0})
                            declare_keyframe(f, T_w2c); rec["keyframe"] = 1; rec["booted"] = 1
                        else:
                            T_w2c = None
            if T_w2c is None:
                # not yet bootstrapped: hold the anchor pose, keep tracking; top up corners if tracks run low
                poses_c2w[f] = np.linalg.inv(last_T)
                rec["failed"] = 1; n_fail += 1
                if len(tracks) < args.min_inliers or len(pre_hist) > MAX_TRACK_LEN_BEFORE_BOOT:
                    new_bootstrap(f)
                diag.append(rec); continue
            prev_T, last_T = last_T, T_w2c
            poses_c2w[f] = np.linalg.inv(T_w2c)
            rec["inliers"] = int((tracks.mp >= 0).sum()); diag.append(rec); continue

        # ---- normal tracking: PnP against the map
        has = tracks.mp >= 0
        X = np.array([mp_xyz[int(m)] for m in tracks.mp[has]]) if has.any() else np.zeros((0, 3))
        T_w2c, inl = pnp(Ks[f], X, tracks.pos[has], args.min_inliers)
        if T_w2c is None:
            n_fail += 1; rec["failed"] = 1
            if args.fail_policy == "cv" and prev_T is not None:
                T_w2c = (last_T @ np.linalg.inv(prev_T)) @ last_T
            else:
                T_w2c = last_T.copy()
            poses_c2w[f] = np.linalg.inv(T_w2c)
            prev_T, last_T = last_T, T_w2c
            consec_fail += 1
            if has.sum() < args.min_inliers or (args.reboot_after_fails and consec_fail >= args.reboot_after_fails):
                # map is dead or persistently inconsistent with the tracks: re-bootstrap from here (new scale segment)
                n_reboot += 1; new_bootstrap(f); rec["event"] = "reboot"; consec_fail = 0
            diag.append(rec); continue
        consec_fail = 0
        rec["inliers"] = int(len(inl))
        if args.cull == "outlier":
            hi = np.where(has)[0]; inl_set = set(hi[inl].tolist())
            drop = np.zeros(len(tracks), bool)
            for row in hi:
                m = int(tracks.mp[row])
                if row in inl_set:
                    mp_outl[m] = 0
                else:
                    mp_outl[m] += 1
                    if mp_outl[m] >= args.cull_hits:
                        drop[row] = True; mp_xyz.pop(m, None)
            tracks.keep(~drop)
        poses_c2w[f] = np.linalg.inv(T_w2c)
        prev_T, last_T = last_T, T_w2c

        # ---- every-frame triangulation option (2.13)
        if args.new_points == "every" and kf_T:
            cand = (tracks.mp < 0) & (np.linalg.norm(tracks.pos - tracks.kf_pos, axis=1) >= PARALLAX_PX)
            if cand.sum() >= 1:
                Xn, acc = triangulate_pair(Ks[kf_frames[-1]], Ks[f], kf_T[-1], T_w2c, tracks.kf_pos[cand].astype(np.float64), tracks.pos[cand].astype(np.float64))
                for row, Xi in zip(np.where(cand)[0][acc], Xn[acc]):
                    tracks.mp[row] = next_mp; mp_xyz[next_mp] = Xi; mp_outl[next_mp] = 0; next_mp += 1

        # ---- keyframe trigger (2.12): parallax since last keyframe
        disp = np.linalg.norm(tracks.pos - tracks.kf_pos, axis=1) if len(tracks) else np.zeros(0)
        if len(tracks) and np.median(disp) >= PARALLAX_PX:
            declare_keyframe(f, T_w2c); rec["keyframe"] = 1
        diag.append(rec)

    # ---- write outputs
    cam_dir = os.path.join(out_scene_dir, "camera"); os.makedirs(cam_dir, exist_ok=True)
    last = np.eye(4)
    for f in range(n):
        if poses_c2w[f] is None:
            poses_c2w[f] = last
        last = poses_c2w[f]
        np.savez(os.path.join(cam_dir, f"{f:06d}.npz"), pose=poses_c2w[f].astype(np.float32), intrinsics=Ks[f].astype(np.float32))
    with open(os.path.join(out_scene_dir, "diag.csv"), "w", newline="") as fh:
        keys = ["frame", "n_tracks", "n_mp", "inliers", "failed", "keyframe", "booted", "event"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for r in diag:
            w.writerow({k: r.get(k, "") for k in keys})
    inl = [r["inliers"] for r in diag if r.get("inliers", -1) >= 0]
    summary = dict(scene=os.path.basename(scene_dir), frames=n, failed_frames=n_fail, reboots=n_reboot,
                   keyframes=len(kf_frames), median_inliers=float(np.median(inl)) if inl else 0.0,
                   map_points=len(mp_xyz), seconds=round(time.time() - t_start, 1), focal=float(np.median([k[0, 0] for k in Ks])), focal_min=float(min(k[0, 0] for k in Ks)), focal_max=float(max(k[0, 0] for k in Ks)), selfcal_pairs=selfcal_pairs)
    json.dump(summary, open(os.path.join(out_scene_dir, "summary.json"), "w"))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes_root", required=True)
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--preds_root", required=True, help="model preds root providing the per-scene focal (e.g. .../augfull_lr1e5/preds)")
    ap.add_argument("--out_root", required=True, help="harness out root; writes <out_root>/<label>/preds/<scene>/camera/")
    ap.add_argument("--label", required=True)
    ap.add_argument("--depth_link", default="", help="if set, symlink <preds_root-like>/<scene>/depth into each scene (inherited depth)")
    # open decisions
    ap.add_argument("--new_points", choices=["kf", "every"], default="kf", help="2.13")
    ap.add_argument("--cull", choices=["none", "outlier"], default="none", help="2.14")
    ap.add_argument("--cull_hits", type=int, default=3)
    ap.add_argument("--min_inliers", type=int, default=20, help="3.1")
    ap.add_argument("--fail_policy", choices=["hold", "cv"], default="hold", help="3.2")
    ap.add_argument("--refine", choices=["none", "ba"], default="none", help="3.4")
    ap.add_argument("--ba_window", type=int, default=5)
    ap.add_argument("--lever", dest="lever", action="store_true", default=True, help="1.2 reject_static_tracks lever (ON by default since 2026-09-05)")
    ap.add_argument("--no_lever", dest="lever", action="store_false", help="disable the static-track lever")
    ap.add_argument("--pnp_thresh", type=float, default=2.0, help="2.7 PnP RANSAC reprojection threshold (px)")
    ap.add_argument("--focal", default="model", help="0.1b: 'model' (per-scene median, default), a fixed focal in px at 320x192 (strict zero-shot), 'selfcal', 'gt' (diagnostic), 'perframe:<preds_root>' (the model's focal for frame t; inference-time), or 'causal:<preds_root>' (running median of the model's focals for frames 0..t; inference-time)")
    ap.add_argument("--no_retro_fill", action="store_true", help="strictly causal: do NOT re-pose pre-bootstrap frames against the bootstrap map (they keep the held anchor pose and stay flagged failed)")
    ap.add_argument("--scale_handoff", action="store_true", help="2.15: at a re-bootstrap keep the old tracks and match the new segment's scale to the old one")
    ap.add_argument("--reboot_after_fails", type=int, default=0, help="3.2b recovery: re-bootstrap after K consecutive PnP failures (0 = only when the map starves)")
    ap.add_argument("--shard_id", type=int, default=0); ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()
    global PNP_THRESH
    PNP_THRESH = args.pnp_thresh

    scenes = [l.strip() for l in open(args.scene_list) if l.strip()]
    scenes = scenes[args.shard_id::args.num_shards]
    out_preds = os.path.join(args.out_root, args.label, "preds")
    for s in scenes:
        out_scene = os.path.join(out_preds, s)
        if os.path.isfile(os.path.join(out_scene, "summary.json")):
            continue
        try:
            summ = run_scene(os.path.join(args.scenes_root, s), os.path.join(args.preds_root, s), out_scene, args)
            if args.depth_link:
                dst = os.path.join(out_scene, "depth"); src = os.path.join(args.depth_link, s, "depth")
                if not os.path.exists(dst) and os.path.isdir(src):
                    os.symlink(src, dst)
            print(json.dumps(summ), flush=True)
        except Exception as e:
            print(f"ERROR {s}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
