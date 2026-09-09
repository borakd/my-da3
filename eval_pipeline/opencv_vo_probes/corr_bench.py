#!/usr/bin/env python
"""Correspondence-method benchmark for decision 1.5 (frontend only).

For each frame pair (i, i+gap) of one scene, every method produces a set of
2D->2D correspondences. Each correspondence is scored against the GT flow
induced by GT depth + GT relative pose (native 320x180 frame, calibrated K;
GT is used ONLY for scoring). Also: essential-matrix rotation error from the
method's correspondences on pairs with GT rotation > 0.5 deg.

Usage: corr_bench.py <scene_dir> <out_json> [gap ...]
"""
import sys, os, glob, json, time
import numpy as np, cv2
cv2.setNumThreads(1)

scene, out = sys.argv[1], sys.argv[2]
gaps = [int(g) for g in sys.argv[3:]] or [1, 4]
d = os.path.join(scene, "dense")
fs = sorted(glob.glob(os.path.join(d, "rgb", "*.png")))
n = len(fs)
G = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in fs]
H, W = G[0].shape
cams = [np.load(os.path.join(d, "cam", "%06d.npz" % i)) for i in range(n)]
K = cams[0]["intrinsic"]; Kinv = np.linalg.inv(K)
c2w = [c["pose"] for c in cams]
D = [np.load(os.path.join(d, "depth", "%06d.npy" % i)) for i in range(n)]
M = [cv2.imread(os.path.join(d, "outlier_mask", "%06d.png" % i), cv2.IMREAD_GRAYSCALE) > 0 for i in range(n)]

def gt_flow(i, j, pts):
    """pts: (N,2) float pixel coords in frame i -> (N,2) GT location in frame j, valid mask."""
    u = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1); v = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    z = D[i][v, u]; valid = (z > 0) & (~M[i][v, u])
    X = (Kinv @ np.c_[pts, np.ones(len(pts))].T) * z  # 3,N
    T = np.linalg.inv(c2w[j]) @ c2w[i]
    Xj = T[:3, :3] @ X + T[:3, 3:4]
    valid &= Xj[2] > 1e-6
    pj = (K @ Xj); pj = (pj[:2] / np.maximum(pj[2], 1e-6)).T
    inside = (pj[:, 0] >= 0) & (pj[:, 0] < W) & (pj[:, 1] >= 0) & (pj[:, 1] < H)
    return pj, valid & inside

def skew(t): return np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
def sampson_gt(T, a, b):
    F = Kinv.T @ skew(T[:3, 3]) @ T[:3, :3] @ Kinv
    a1 = np.c_[a, np.ones(len(a))]; b1 = np.c_[b, np.ones(len(b))]
    Fa = (F @ a1.T).T; Ftb = (F.T @ b1.T).T; num = np.sum(b1 * Fa, 1) ** 2
    return num / (Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2)
def rot_angle(R): return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

# ---------------- methods: each returns (ptsA (N,2), ptsB (N,2)) ----------------
def m_lk(gi, gj, win=21, lvl=3, fb=1.0):
    p0 = cv2.goodFeaturesToTrack(gi, 1000, 0.01, 5)
    if p0 is None: return np.zeros((0, 2)), np.zeros((0, 2))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(gi, gj, p0, None, winSize=(win, win), maxLevel=lvl)
    p0b, stb, _ = cv2.calcOpticalFlowPyrLK(gj, gi, p1, None, winSize=(win, win), maxLevel=lvl)
    ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm((p0 - p0b)[:, 0], axis=1) < fb)
    return p0[ok, 0], p1[ok, 0]

def m_lk_nofb(gi, gj):
    p0 = cv2.goodFeaturesToTrack(gi, 1000, 0.01, 5)
    if p0 is None: return np.zeros((0, 2)), np.zeros((0, 2))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(gi, gj, p0, None)
    ok = st[:, 0] == 1
    return p0[ok, 0], p1[ok, 0]

_orb = cv2.ORB_create(nfeatures=1000)
_sift = cv2.SIFT_create(nfeatures=1000)
_bf_h = cv2.BFMatcher(cv2.NORM_HAMMING)
_bf_l2 = cv2.BFMatcher(cv2.NORM_L2)
def _match(det, bf, gi, gj, ratio=0.75):
    k0, d0 = det.detectAndCompute(gi, None); k1, d1 = det.detectAndCompute(gj, None)
    if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2: return np.zeros((0, 2)), np.zeros((0, 2))
    mm = bf.knnMatch(d0, d1, k=2)
    good = [m[0] for m in mm if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    a = np.array([k0[m.queryIdx].pt for m in good]); b = np.array([k1[m.trainIdx].pt for m in good])
    return a.reshape(-1, 2), b.reshape(-1, 2)
def m_orb(gi, gj): return _match(_orb, _bf_h, gi, gj)
def m_sift(gi, gj): return _match(_sift, _bf_l2, gi, gj)

_dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
def _dense_at_corners(flow, gi):
    p0 = cv2.goodFeaturesToTrack(gi, 1000, 0.01, 5)
    if p0 is None: return np.zeros((0, 2)), np.zeros((0, 2))
    a = p0[:, 0]; u = np.clip(np.round(a[:, 0]).astype(int), 0, W - 1); v = np.clip(np.round(a[:, 1]).astype(int), 0, H - 1)
    return a, a + flow[v, u]
def m_dis(gi, gj): return _dense_at_corners(_dis.calc(gi, gj, None), gi)
def m_farneback(gi, gj):
    return _dense_at_corners(cv2.calcOpticalFlowFarneback(gi, gj, None, 0.5, 3, 15, 3, 5, 1.2, 0), gi)
def m_dis_grid(gi, gj):
    flow = _dis.calc(gi, gj, None)
    vv, uu = np.mgrid[4:H:8, 4:W:8]; a = np.c_[uu.ravel(), vv.ravel()].astype(np.float32)
    return a, a + flow[vv.ravel(), uu.ravel()]

METHODS = {"lk_fb": m_lk, "lk_nofb": m_lk_nofb, "orb_ratio": m_orb, "sift_ratio": m_sift,
           "dis_corners": m_dis, "dis_grid": m_dis_grid, "farneback_corners": m_farneback}

res = {m: {g: dict(pairs=0, ncorr=[], ncorrect=[], epe=[], in1=[], in2=[], static=[], samp1=[], survival=[], time=[], rot_err=[], rot_pairs=0, e_inl=[]) for g in gaps} for m in METHODS}
for g in gaps:
    for i in range(0, n - g, 2):
        j = i + g; T = np.linalg.inv(c2w[j]) @ c2w[i]; gt_rot = rot_angle(T[:3, :3])
        for name, fn in METHODS.items():
            r = res[name][g]; t0 = time.time(); a, b = fn(G[i], G[j]); r["time"].append(time.time() - t0)
            r["pairs"] += 1; r["ncorr"].append(len(a))
            if len(a) == 0: continue
            pj, valid = gt_flow(i, j, a)
            if valid.sum() > 0:
                e = np.linalg.norm(b[valid] - pj[valid], axis=1)
                r["epe"].append(float(np.median(e))); r["in1"].append(float((e < 1).mean())); r["in2"].append(float((e < 2).mean()))
                r["ncorrect"].append(int((e < 2).sum()))
            disp = np.linalg.norm(b - a, axis=1); r["static"].append(float((disp < 1).mean()))
            if np.linalg.norm(T[:3, 3]) > 1e-3:
                r["samp1"].append(float((sampson_gt(T, a.astype(np.float64), b.astype(np.float64)) < 1).mean()))
            if name in ("lk_fb", "lk_nofb"):
                p0 = cv2.goodFeaturesToTrack(G[i], 1000, 0.01, 5); r["survival"].append(len(a) / max(1, len(p0)))
            if gt_rot > 0.5 and len(a) >= 8:
                E, inl = cv2.findEssentialMat(a.astype(np.float64), b.astype(np.float64), K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None and E.shape == (3, 3) and inl is not None:
                    _, R, t, _ = cv2.recoverPose(E, a.astype(np.float64), b.astype(np.float64), K, mask=inl.copy())
                    r["rot_err"].append(float(rot_angle(R.T @ T[:3, :3]))); r["rot_pairs"] += 1; r["e_inl"].append(float(inl.mean()))

def agg(v, f=np.median): return float(f(v)) if len(v) else None
summary = {m: {g: dict(pairs=r["pairs"], ncorr_med=agg(r["ncorr"]), ncorrect_med=agg(r["ncorrect"]), static_med=agg(r["static"]), samp1_med=agg(r["samp1"]), epe_med=agg(r["epe"]), in1_med=agg(r["in1"]), in2_med=agg(r["in2"]),
                       survival_med=agg(r["survival"]), ms_med=1000 * (agg(r["time"]) or 0), rot_pairs=r["rot_pairs"],
                       rot_err_med=agg(r["rot_err"]), rot_err_p90=agg(r["rot_err"], lambda x: np.percentile(x, 90)), e_inl_med=agg(r["e_inl"]),
                       raw=dict(ncorr=r["ncorr"], ncorrect=r["ncorrect"], epe=r["epe"], in2=r["in2"], static=r["static"], samp1=r["samp1"], rot_err=r["rot_err"]))
               for g, r in rg.items()} for m, rg in res.items()}
json.dump(dict(scene=os.path.basename(scene), n=n, summary=summary), open(out, "w"))
print(os.path.basename(scene), n, "done")
