#!/usr/bin/env python
"""Triangulation-acceptance benchmark for decision 2.11.

Triplets (i, j=i+G, k=j+G) with persistent LK tracks i->j->k, CHAINED frame-by-frame (same frontend as
1.3/1.5: Shi-Tomasi 1000/0.01/5, LK defaults, fwd-bwd < 1 px at each hop).
Relative pose i->j is either
  E  : findEssentialMat(RANSAC 2 px, 0.999) + recoverPose   (what the pipeline has)
  GT : the GT relative pose                                  (isolates the filter)
Triangulate every i<->j correspondence (cv2.triangulatePoints, P_i=K[I|0], P_j=K[R|t]),
apply each acceptance filter, then score the admitted "map":
  n_acc      : points admitted
  bad_frac   : admitted points whose depth (in cam i) is off from GT depth by >20%
               after a single median scale alignment (E has arbitrary scale; GT is metric
               but we align the same way for comparability)
  pnp rot/tdir: solvePnPRansac(ITERATIVE, 2 px, 0.999, 1000) + RefineLM of frame k
               against the admitted map using the k-observations of the same tracks;
               rotation error (deg) and translation-direction error (deg) vs GT.
  pnp_fail   : PnP returned < 20 inliers (the 3.1 floor)
Filters:
  none        : accept all
  cheir       : depth > 0 in both cameras
  cheir+rep   : + reprojection error < 2 px in both views
  cheir+par   : + parallax angle >= 1 deg
  all         : cheir + rep + par
GT depth / pose are used ONLY for scoring. Static tracks are kept (lever OFF).
Usage: tri_bench.py <scene_dir> <out_json> [G]
"""
import sys, os, glob, json
import numpy as np, cv2
cv2.setNumThreads(1)

scene, out = sys.argv[1], sys.argv[2]
GAP = int(sys.argv[3]) if len(sys.argv) > 3 else 4
KOFF = int(os.environ.get("KOFF", "0")) or 2 * GAP      # PnP test frame k = j + KOFF (default: k = j + GAP)
LEVER = os.environ.get("LEVER", "0") == "1"
E_THR = float(os.environ.get("E_THR", "2.0"))
KF_MODE = os.environ.get("KF_MODE", "stride")           # stride | parallax | ratio : how the second keyframe j is chosen
KF_VAL = float(os.environ.get("KF_VAL", "0"))            # parallax: median track displacement (px) since i; ratio: surviving-track fraction            # RANSAC threshold for the bootstrap essential matrix             # reject_static_tracks lever: drop tracks with disp(i->j) < 1 px
d = os.path.join(scene, "dense")
fs = sorted(glob.glob(os.path.join(d, "rgb", "*.png"))); n = len(fs)
G = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in fs]; H, W = G[0].shape
cams = [np.load(os.path.join(d, "cam", "%06d.npz" % i)) for i in range(n)]
K = cams[0]["intrinsic"].astype(np.float64); Kinv = np.linalg.inv(K)
c2w = [c["pose"] for c in cams]
D = [np.load(os.path.join(d, "depth", "%06d.npy" % i)) for i in range(n)]
M = [cv2.imread(os.path.join(d, "outlier_mask", "%06d.png" % i), cv2.IMREAD_GRAYSCALE) > 0 for i in range(n)]

def rot_angle(R): return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
def lk_hop(gi, gj, p):
    p1, st, _ = cv2.calcOpticalFlowPyrLK(gi, gj, p, None)
    p0b, stb, _ = cv2.calcOpticalFlowPyrLK(gj, gi, p1, None)
    ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm((p - p0b)[:, 0], axis=1) < 1.0)
    return p1, ok
def lk_chain(i, j, p):
    """Track p from frame i to frame j one frame at a time (pipeline reality), fwd-bwd check per step."""
    ok = np.ones(len(p), bool); cur = p.copy()
    for f in range(i, j):
        nxt, okf = lk_hop(G[f], G[f + 1], cur); ok &= okf; cur = nxt
    return cur, ok

def triangulate(R, t, a, b):
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))]); P1 = K @ np.hstack([R, t.reshape(3, 1)])
    Xh = cv2.triangulatePoints(P0, P1, a.T, b.T); X = (Xh[:3] / Xh[3]).T  # in cam i
    Xj = (R @ X.T + t.reshape(3, 1)).T
    def proj(P, X3):
        x = (P @ np.c_[X3, np.ones(len(X3))].T); return (x[:2] / x[2]).T
    ra = np.linalg.norm(proj(P0, X) - a, axis=1); rb = np.linalg.norm(proj(P1, X) - b, axis=1)
    # parallax angle between rays from the two centers (cam i at 0, cam j at -R^T t)
    Cj = -R.T @ t.reshape(3)
    r1 = X / np.linalg.norm(X, axis=1, keepdims=True); r2 = (X - Cj); r2 /= np.linalg.norm(r2, axis=1, keepdims=True)
    par = np.degrees(np.arccos(np.clip(np.sum(r1 * r2, 1), -1, 1)))
    return X, Xj, ra, rb, par

FILTERS = {
    "none":      lambda X, Xj, ra, rb, par: np.ones(len(X), bool),
    "cheir":     lambda X, Xj, ra, rb, par: (X[:, 2] > 0) & (Xj[:, 2] > 0),
    "cheir+rep": lambda X, Xj, ra, rb, par: (X[:, 2] > 0) & (Xj[:, 2] > 0) & (ra < 2) & (rb < 2),
    "cheir+par": lambda X, Xj, ra, rb, par: (X[:, 2] > 0) & (Xj[:, 2] > 0) & (par >= 1.0),
    "all":       lambda X, Xj, ra, rb, par: (X[:, 2] > 0) & (Xj[:, 2] > 0) & (ra < 2) & (rb < 2) & (par >= 1.0),
}
res = {pc: {f: dict(triplets=0, n_acc=[], bad_frac=[], pnp_rot=[], pnp_tdir=[], pnp_inl=[], pnp_fail=0, skipped=0)
            for f in FILTERS} for pc in ("E", "GT")}
gated = 0; total = 0
kf_gaps = []
for i in range(0, n - GAP - KOFF, 2):
    p0 = cv2.goodFeaturesToTrack(G[i], 1000, 0.01, 5)
    if p0 is None or len(p0) < 20: continue
    if KF_MODE == "stride":
        j = i + GAP; p1, ok1 = lk_chain(i, j, p0)
    else:
        # walk forward one frame at a time until the trigger fires (max 32 frames)
        cur = p0.copy(); ok1 = np.ones(len(p0), bool); j = None
        for f in range(i, min(i + 32, n - KOFF - 1)):
            nxt, okf = lk_hop(G[f], G[f + 1], cur); ok1 &= okf; cur = nxt
            if ok1.sum() < 20: break
            fired = (np.median(np.linalg.norm((cur - p0)[ok1, 0], axis=1)) >= KF_VAL) if KF_MODE == "parallax" else (ok1.mean() < KF_VAL)
            if fired: j = f + 1; break
        if j is None: continue
        p1 = cur
    k = j + KOFF; total += 1; kf_gaps.append(j - i)
    p2, ok2 = lk_chain(j, k, p1)
    ok = ok1 & ok2
    a = p0[ok, 0].astype(np.float64); b = p1[ok, 0].astype(np.float64); c = p2[ok, 0].astype(np.float64)
    if len(a) < 20: continue
    disp_ij = np.linalg.norm(b - a, axis=1)
    if np.median(disp_ij) < 2.0: continue   # parallax gate: only pairs where the scene moved
    if LEVER:
        keep = disp_ij >= 1.0
        if keep.sum() < 20: continue
        a, b, c = a[keep], b[keep], c[keep]
    gated += 1
    T_ji = np.linalg.inv(c2w[j]) @ c2w[i]; T_ki = np.linalg.inv(c2w[k]) @ c2w[i]
    u = np.clip(np.round(a[:, 0]).astype(int), 0, W - 1); v = np.clip(np.round(a[:, 1]).astype(int), 0, H - 1)
    zgt = D[i][v, u]; zvalid = (zgt > 0) & (~M[i][v, u])
    poses = {"GT": (T_ji[:3, :3], T_ji[:3, 3])}
    E, inl = cv2.findEssentialMat(a, b, K, method=cv2.RANSAC, prob=0.999, threshold=E_THR)
    if E is not None and E.shape == (3, 3) and inl is not None:
        _, Re, te, _ = cv2.recoverPose(E, a, b, K, mask=inl.copy()); poses["E"] = (Re, te.reshape(3))
    for pc, (R, t) in poses.items():
        X, Xj, ra, rb, par = triangulate(R, t, a, b)
        for fname, ffn in FILTERS.items():
            r = res[pc][fname]; r["triplets"] += 1
            sel = ffn(X, Xj, ra, rb, par) & np.isfinite(X).all(1)
            r["n_acc"].append(int(sel.sum()))
            if sel.sum() < 4: r["skipped"] += 1; continue
            # depth accuracy of admitted points (single median scale)
            sv = sel & zvalid
            if sv.sum() >= 5:
                s = np.median(zgt[sv] / np.maximum(X[sv, 2], 1e-9))
                rel = np.abs(s * X[sv, 2] - zgt[sv]) / zgt[sv]; r["bad_frac"].append(float((rel > 0.2).mean()))
            # PnP of frame k against the admitted map
            Xm = np.ascontiguousarray(X[sel]); xk = np.ascontiguousarray(c[sel])
            try:
                okp, rvec, tvec, inlp = cv2.solvePnPRansac(Xm, xk, K, None, flags=cv2.SOLVEPNP_ITERATIVE,
                                                           reprojectionError=2.0, confidence=0.999, iterationsCount=1000)
            except cv2.error:
                okp = False
            if not okp or inlp is None or len(inlp) < 20: r["pnp_fail"] += 1; continue
            inlp = inlp[:, 0]
            rvec, tvec = cv2.solvePnPRefineLM(Xm[inlp], xk[inlp], K, None, rvec, tvec)
            Rk, _ = cv2.Rodrigues(rvec); tk = tvec.reshape(3)
            r["pnp_rot"].append(float(rot_angle(Rk.T @ T_ki[:3, :3]))); r["pnp_inl"].append(int(len(inlp)))
            tg = T_ki[:3, 3]
            if np.linalg.norm(tg) > 2e-3 and np.linalg.norm(tk) > 1e-9:
                r["pnp_tdir"].append(float(np.degrees(np.arccos(np.clip(np.dot(tg, tk) / np.linalg.norm(tg) / np.linalg.norm(tk), -1, 1)))))
json.dump(dict(scene=os.path.basename(scene), n=n, gap=GAP, koff=KOFF, lever=LEVER, e_thr=E_THR, triplets_total=total, triplets_gated=gated, kf_mode=KF_MODE, kf_val=KF_VAL, kf_gaps=kf_gaps, res=res), open(out, "w"))
print(os.path.basename(scene), n, "gated", gated, "of", total, "done")
