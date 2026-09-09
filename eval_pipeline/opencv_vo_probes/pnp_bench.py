#!/usr/bin/env python
"""PnP-variant benchmark for decision 2.4.

Same LK+FB frontend as decision 1.5. For each pair (i, i+gap): 3D points come from
GT depth on frame i (native 320x180, calibrated K; GT used for the 3D points and
the reference pose only). Each solver estimates frame j's pose from the 2D-3D
correspondences and is scored against the GT relative pose.

Two conditions:
  clean    : only tracks with valid GT depth (no identity-voting outliers)
  outliers : additionally, static tracks (disp < 1 px) with NO valid depth are
             assigned a plausible wrong 3D point (median scene depth), so they act
             as the identity-voting outliers a real map will contain.

Variants: solvePnPRansac with flag in {ITERATIVE, EPNP, P3P, AP3P, SQPNP},
each with and without solvePnPRefineLM on the inlier set.
Fixed: reprojectionError=2 px, confidence=0.999, iterationsCount=1000.

Usage: pnp_bench.py <scene_dir> <out_json> [gap ...]
"""
import sys, os, glob, json, time
import numpy as np, cv2
cv2.setNumThreads(1)

scene, out = sys.argv[1], sys.argv[2]
gaps = [int(g) for g in sys.argv[3:]] or [1, 4]
d = os.path.join(scene, "dense")
fs = sorted(glob.glob(os.path.join(d, "rgb", "*.png"))); n = len(fs)
G = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in fs]; H, W = G[0].shape
cams = [np.load(os.path.join(d, "cam", "%06d.npz" % i)) for i in range(n)]
K = cams[0]["intrinsic"].astype(np.float64); Kinv = np.linalg.inv(K)
c2w = [c["pose"] for c in cams]
D = [np.load(os.path.join(d, "depth", "%06d.npy" % i)) for i in range(n)]
M = [cv2.imread(os.path.join(d, "outlier_mask", "%06d.png" % i), cv2.IMREAD_GRAYSCALE) > 0 for i in range(n)]

def rot_angle(R): return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

def lk(gi, gj):
    p0 = cv2.goodFeaturesToTrack(gi, 1000, 0.01, 5)
    if p0 is None: return np.zeros((0, 2)), np.zeros((0, 2))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(gi, gj, p0, None)
    p0b, stb, _ = cv2.calcOpticalFlowPyrLK(gj, gi, p1, None)
    ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm((p0 - p0b)[:, 0], axis=1) < 1.0)
    return p0[ok, 0].astype(np.float64), p1[ok, 0].astype(np.float64)

FLAGS = {"iterative": cv2.SOLVEPNP_ITERATIVE, "epnp": cv2.SOLVEPNP_EPNP, "p3p": cv2.SOLVEPNP_P3P,
         "ap3p": cv2.SOLVEPNP_AP3P, "sqpnp": cv2.SOLVEPNP_SQPNP}
VARIANTS = [(f, r) for f in FLAGS for r in (False, True)]

def solve(X, x, flag, refine):
    """X: (N,3) in frame i camera coords; x: (N,2) pixels in frame j. Returns T_ji (4x4) or None, n_inliers."""
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(X, x, K, None, flags=FLAGS[flag], reprojectionError=2.0,
                                                 confidence=0.999, iterationsCount=1000)
    except cv2.error:
        return None, 0
    if not ok or inl is None or len(inl) < 4: return None, 0
    inl = inl[:, 0]
    if refine:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(X[inl], x[inl], K, None, rvec, tvec)
        except cv2.error:
            pass
    R, _ = cv2.Rodrigues(rvec); T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.ravel()
    return T, len(inl)

res = {f"{f}{'+lm' if r else ''}": {g: {c: dict(pairs=0, fail=0, rot=[], trans_mm=[], tdir=[], inl=[], ms=[]) for c in ("clean", "outliers")}
                                     for g in gaps} for f, r in VARIANTS}
for g in gaps:
    for i in range(0, n - g, 2):
        j = i + g; T_gt = np.linalg.inv(c2w[j]) @ c2w[i]
        a, b = lk(G[i], G[j])
        if len(a) < 8: continue
        u = np.clip(np.round(a[:, 0]).astype(int), 0, W - 1); v = np.clip(np.round(a[:, 1]).astype(int), 0, H - 1)
        z = D[i][v, u]; valid = (z > 0) & (~M[i][v, u])
        disp = np.linalg.norm(b - a, axis=1); static_nodepth = (disp < 1.0) & (~valid)
        if valid.sum() < 8: continue
        zfill = z.copy(); zfill[~valid] = np.median(z[valid])
        Xall = ((Kinv @ np.c_[a, np.ones(len(a))].T) * zfill).T  # N,3 in cam i
        for cond in ("clean", "outliers"):
            sel = valid if cond == "clean" else (valid | static_nodepth)
            X = np.ascontiguousarray(Xall[sel]); x = np.ascontiguousarray(b[sel])
            for f, r in VARIANTS:
                name = f"{f}{'+lm' if r else ''}"; rr = res[name][g][cond]; rr["pairs"] += 1
                t0 = time.time(); T, ninl = solve(X, x, f, r); rr["ms"].append(1000 * (time.time() - t0))
                if T is None: rr["fail"] += 1; continue
                E = np.linalg.inv(T_gt) @ T
                rr["rot"].append(float(rot_angle(E[:3, :3]))); rr["trans_mm"].append(float(1000 * np.linalg.norm(E[:3, 3])))
                tg = T_gt[:3, 3]; tp = T[:3, 3]
                if np.linalg.norm(tg) > 2e-3 and np.linalg.norm(tp) > 1e-9:
                    rr["tdir"].append(float(np.degrees(np.arccos(np.clip(np.dot(tg, tp) / np.linalg.norm(tg) / np.linalg.norm(tp), -1, 1)))))
                rr["inl"].append(int(ninl))

json.dump(dict(scene=os.path.basename(scene), n=n, res=res), open(out, "w"))
print(os.path.basename(scene), n, "done")
