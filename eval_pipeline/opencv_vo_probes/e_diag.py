#!/usr/bin/env python
"""Why is the essential-matrix translation direction ~35 deg off?  Diagnostic for 2.12 / 2.10.

Pairs (i, j=i+GAP), chained LK tracks with the static lever ON (disp>=1 px), parallax gated.
For each pair, several two-view pose estimates are scored vs the GT relative pose
(rotation error deg, translation-direction error deg):
  e2     : findEssentialMat RANSAC 2 px (pipeline)            + recoverPose
  e1     : RANSAC 1 px
  e05    : RANSAC 0.5 px
  magsac : cv2.USAC_MAGSAC, 2 px
  synth1 : SAME points, but correspondences replaced by GT flow (GT depth + GT pose) + 1 px Gaussian noise
           -> geometric conditioning of this motion, independent of the frontend
  synth0 : GT flow with no noise (sanity: should be ~0)
  refine : e2 inliers, then scipy least_squares on (rotvec, t-direction) minimizing Sampson distance
  gtR_t  : rotation fixed to GT, translation direction by least squares on the epipolar constraint
           (isolates whether t is bad because R is bad)
Also records GT motion: rotation deg, translation mm, and the ratio of median translational parallax to
median rotational flow (rotation-dominance).
Usage: e_diag.py <scene_dir> <out_json> [GAP]
"""
import sys, os, glob, json
import numpy as np, cv2
cv2.setNumThreads(1)
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

scene, out = sys.argv[1], sys.argv[2]
GAP = int(sys.argv[3]) if len(sys.argv) > 3 else 8
d = os.path.join(scene, "dense")
fs = sorted(glob.glob(os.path.join(d, "rgb", "*.png"))); n = len(fs)
G = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in fs]; H, W = G[0].shape
cams = [np.load(os.path.join(d, "cam", "%06d.npz" % i)) for i in range(n)]
K = cams[0]["intrinsic"].astype(np.float64); Kinv = np.linalg.inv(K)
c2w = [c["pose"] for c in cams]
D = [np.load(os.path.join(d, "depth", "%06d.npy" % i)) for i in range(n)]
M = [cv2.imread(os.path.join(d, "outlier_mask", "%06d.png" % i), cv2.IMREAD_GRAYSCALE) > 0 for i in range(n)]
rng = np.random.default_rng(0)

def rot_angle(R): return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
def tdir(t, tg):
    if np.linalg.norm(tg) < 2e-3 or np.linalg.norm(t) < 1e-12: return None
    return float(np.degrees(np.arccos(np.clip(np.dot(t.ravel(), tg) / np.linalg.norm(t) / np.linalg.norm(tg), -1, 1))))
def skew(t): return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
def lk_hop(gi, gj, p):
    p1, st, _ = cv2.calcOpticalFlowPyrLK(gi, gj, p, None); p0b, stb, _ = cv2.calcOpticalFlowPyrLK(gj, gi, p1, None)
    return p1, (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm((p - p0b)[:, 0], axis=1) < 1.0)
def lk_chain(i, j, p):
    ok = np.ones(len(p), bool); cur = p.copy()
    for f in range(i, j):
        nxt, okf = lk_hop(G[f], G[f + 1], cur); ok &= okf; cur = nxt
    return cur, ok

def est(a, b, method, thr):
    E, inl = cv2.findEssentialMat(a, b, K, method=method, prob=0.999, threshold=thr)
    if E is None or E.shape != (3, 3) or inl is None: return None
    _, R, t, _ = cv2.recoverPose(E, a, b, K, mask=inl.copy()); return R, t.reshape(3), inl[:, 0].astype(bool)

def sampson(R, t, a, b):
    F = Kinv.T @ skew(t) @ R @ Kinv
    a1 = np.c_[a, np.ones(len(a))]; b1 = np.c_[b, np.ones(len(b))]
    Fa = (F @ a1.T).T; Ftb = (F.T @ b1.T).T; num = np.sum(b1 * Fa, 1)
    return num / np.sqrt(Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2)

def refine(R0, t0, a, b):
    """least-squares on (rotvec, spherical t) minimizing Sampson distance."""
    th0 = np.array([np.arctan2(t0[1], t0[0]), np.arccos(np.clip(t0[2] / np.linalg.norm(t0), -1, 1))])
    x0 = np.r_[Rot.from_matrix(R0).as_rotvec(), th0]
    def unpack(x):
        R = Rot.from_rotvec(x[:3]).as_matrix(); ph, th = x[3], x[4]
        return R, np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    def resid(x):
        R, t = unpack(x); return sampson(R, t, a, b)
    sol = least_squares(resid, x0, loss="huber", f_scale=1.0, max_nfev=200)
    return unpack(sol.x)

def t_given_R(R, a, b):
    """with R known, epipolar constraint b^T [t]x R a = 0 is linear in t: solve by SVD."""
    an = (Kinv @ np.c_[a, np.ones(len(a))].T).T; bn = (Kinv @ np.c_[b, np.ones(len(b))].T).T
    Ra = (R @ an.T).T
    A = np.cross(bn, Ra)            # b x (R a) . t = 0  <=>  b^T [t]x (R a) = 0 up to sign
    _, _, Vt = np.linalg.svd(A); t = Vt[-1]
    return t

res = dict(pairs=0, gated=0, gt_rot=[], gt_t_mm=[], rot_dom=[], n_tracks=[], methods={m: dict(rot=[], tdir=[], inl=[]) for m in ("e2", "e1", "e05", "magsac", "synth1", "synth0", "refine", "gtR_t", "e2_rel", "e2_oracle", "e2_oracle_rel")})
for i in range(0, n - GAP, 2):
    j = i + GAP; res["pairs"] += 1
    p0 = cv2.goodFeaturesToTrack(G[i], 1000, 0.01, 5)
    if p0 is None or len(p0) < 20: continue
    p1, ok = lk_chain(i, j, p0)
    a = p0[ok, 0].astype(np.float64); b = p1[ok, 0].astype(np.float64)
    if len(a) < 20: continue
    disp = np.linalg.norm(b - a, axis=1)
    if np.median(disp) < 2.0: continue
    keep = disp >= 1.0
    if keep.sum() < 20: continue
    a, b = a[keep], b[keep]
    T = np.linalg.inv(c2w[j]) @ c2w[i]; Rg, tg = T[:3, :3], T[:3, 3]
    if np.linalg.norm(tg) < 2e-3: continue
    res["gated"] += 1; res["gt_rot"].append(float(rot_angle(Rg))); res["gt_t_mm"].append(float(1000 * np.linalg.norm(tg))); res["n_tracks"].append(int(len(a)))
    # synthetic GT correspondences at the same points (need GT depth)
    u = np.clip(np.round(a[:, 0]).astype(int), 0, W - 1); v = np.clip(np.round(a[:, 1]).astype(int), 0, H - 1)
    z = D[i][v, u]; zv = (z > 0) & (~M[i][v, u])
    X = (Kinv @ np.c_[a, np.ones(len(a))].T) * z; Xj = Rg @ X + tg[:, None]
    pj = (K @ Xj); pj = (pj[:2] / np.maximum(pj[2], 1e-9)).T
    # rotation-dominance: flow under pure rotation vs under pure translation at these points
    Xr = Rg @ X; pr = (K @ Xr); pr = (pr[:2] / np.maximum(pr[2], 1e-9)).T
    Xt = X + tg[:, None]; pt = (K @ Xt); pt = (pt[:2] / np.maximum(pt[2], 1e-9)).T
    if zv.sum() >= 10:
        rf = np.median(np.linalg.norm(pr[zv] - a[zv], axis=1)); tf = np.median(np.linalg.norm(pt[zv] - a[zv], axis=1))
        res["rot_dom"].append(float(rf / max(tf, 1e-6)))
    def rec(name, r):
        if r is None: return
        R, t, inl = r; m = res["methods"][name]
        m["rot"].append(float(rot_angle(R.T @ Rg))); td = tdir(t, tg)
        if td is not None: m["tdir"].append(td)
        m["inl"].append(float(inl.mean()) if inl is not None else 1.0)
    r2 = est(a, b, cv2.RANSAC, 2.0); rec("e2", r2)
    # relative static lever: drop tracks moving < 20% of the median (chained drift of static points exceeds 1 px)
    rel = np.linalg.norm(b - a, axis=1) >= 0.2 * np.median(np.linalg.norm(b - a, axis=1))
    if rel.sum() >= 20: rec("e2_rel", est(a[rel], b[rel], cv2.RANSAC, 2.0))
    # ORACLE gripper mask (diagnostic only): keep only tracks with valid GT depth (gripper region has none)
    if zv.sum() >= 20: rec("e2_oracle", est(a[zv], b[zv], cv2.RANSAC, 2.0))
    if (zv & rel).sum() >= 20: rec("e2_oracle_rel", est(a[zv & rel], b[zv & rel], cv2.RANSAC, 2.0))
    rec("e1", est(a, b, cv2.RANSAC, 1.0)); rec("e05", est(a, b, cv2.RANSAC, 0.5))
    try: rec("magsac", est(a, b, cv2.USAC_MAGSAC, 2.0))
    except cv2.error: pass
    if zv.sum() >= 20:
        as_, bs = a[zv], pj[zv]
        rec("synth0", est(as_, bs, cv2.RANSAC, 2.0))
        rec("synth1", est(as_, bs + rng.normal(0, 1.0, bs.shape), cv2.RANSAC, 2.0))
    if r2 is not None:
        R0, t0, inl = r2
        try:
            Rr, tr = refine(R0, t0, a[inl], b[inl]); rec("refine", (Rr, tr, inl))
        except Exception: pass
        rec("gtR_t", (Rg, t_given_R(Rg, a[inl], b[inl]), inl))
json.dump(dict(scene=os.path.basename(scene), n=n, gap=GAP, res=res), open(out, "w"))
print(os.path.basename(scene), n, "gated", res["gated"], "of", res["pairs"], "done")
