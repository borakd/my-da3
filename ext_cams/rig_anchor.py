#!/usr/bin/env python
"""rig_anchor: exterior-camera rig that RE-ANCHORS to the birth model on every frame (idea 1, 2026-10-09).

Diagnosis this answers (match_lkr15 / match_ct): the guided birth matches are true points and give label-level poses for
~20 frames; then the tracker loses them, the pose rides on points appended through already-wrong poses (dead reckoning)
and the error grows to tens of degrees. Here the model never grows and the birth points are never lost for good:

  model     X_B = the guided cross-view DISK matches at the FK birth frame b (rig_match_track.py matches.npz, the same seeds
            as the lkr15 / ct / Track-On arms), triangulated once with the calibration. Fixed for the whole window.
  appear.   each model point keeps reference descriptors: the DISK dense descriptor at its birth pixel in both views, plus a
            small memory of descriptors taken at frames where it was a confident geometric inlier (no 3D is ever added).
  frame t   (causal) predict the motion (constant velocity, or hold), project X_B into both views, and search DISK's DENSE
            descriptor map in a window around each predicted pixel for the best match to the point's references (no
            keypoint detector involved, so detector repeatability does not matter). Observations = (model point, view, pixel).
            Robust multi-view pose: hypotheses {prediction, hold, per-view PnP-RANSAC, both-view triangulate + RANSAC-Kabsch},
            best by reprojection inliers in both views, then a Huber least-squares refinement of the 6-DoF motion on both
            views (multi-view PnP). After a run of unsolved frames the search window grows and finally becomes global
            (the whole gripper region) for re-localisation.
  output    lens pose(t) = M_t @ GT(b) (the hand-eye stand-in at the birth frame, as in every rig arm); frames that fail the
            solved test (>= --min_inl inliers spread >= --min_spread m) carry no pose.

GT-free inputs only: MP4s, calibration (factory intrinsics + PointWorld extrinsics), FK birth/exit frames, the birth seeds and,
for the global re-localisation crop only, the Track-On v2 points visible at t (image-based; already used by rig_match_track).
Evaluation (GT, after the fact): rig_points_eval conventions with t_ref = b, plus increment (frame-to-frame) errors.

    python rig_anchor.py --episode EP --out DIR [--seeds DIR] [options]
Writes DIR/record.json (windows[0] in the rig_points_eval layout), per_frame.csv, anchors.npz (frames, poses, motions).
"""
import argparse, csv, json, os, sys, time
import numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_track as rt            # noqa: E402
import rig_trackon_eval as rte    # noqa: E402

S = "/gpfs/scratch/etur59/koc821022"
RT_DIR = f"{S}/outputs/cut3r_eval/ext_cams/rig_trackon"
SEEDS = f"{RT_DIR}/match_ct/tracks"
V2 = f"{S}/outputs/droid_birth_frames/trackon_run_v2/tracks"
EVENTS = f"{S}/outputs/droid_birth_frames/entries/entry_events_gap3.csv"
MASKS = f"{S}/outputs/droid_birth_frames/birth_masks_v2_seeding"   # RobotSeg birth masks (items_v2.csv)
W, H = rt.W, rt.H


# ----------------------------------------------------------------------------- small geometry helpers
def ortho(R):
    U_, _, Vt = np.linalg.svd(R); return U_ @ np.diag([1, 1, np.sign(np.linalg.det(U_ @ Vt))]) @ Vt


def se3(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; return M


def region_mask(pts, dil):
    m = np.zeros((H, W), np.uint8)
    if len(pts) >= 3:
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts).astype(np.int32)), 255)
    elif len(pts):
        for x, y in pts: cv2.circle(m, (int(x), int(y)), dil, 255, -1)
    return cv2.dilate(m, np.ones((2 * dil + 1,) * 2, np.uint8))


# ----------------------------------------------------------------------------- DISK dense descriptors
class Dense:
    def __init__(self):
        import torch, kornia
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = kornia.feature.DISK.from_pretrained("depth").eval().to(self.dev)

    def map(self, g, box, scale, max_side=1280):
        """Dense L2-normalised DISK descriptors of the crop box=(x0,y0,x1,y1) of grey frame g, resized by `scale`.
        Returns (D (128,h,w) torch, (x0, y0, sx, sy)) with map coords m = (u - (x0, y0)) * (sx, sy)."""
        torch = self.torch
        x0, y0, x1, y1 = box; crop = g[y0:y1, x0:x1]
        s = min(scale, max_side / max(crop.shape))
        cw, ch = max(16, int(round(crop.shape[1] * s / 16)) * 16), max(16, int(round(crop.shape[0] * s / 16)) * 16)
        up = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_CUBIC if s >= 1 else cv2.INTER_AREA)
        with torch.no_grad():
            img = torch.from_numpy(up).to(self.dev).float().div(255)[None, None].repeat(1, 3, 1, 1)
            _, D = self.model.heatmap_and_dense_descriptors(img)
            D = torch.nn.functional.normalize(D[0], dim=0)
        return D, (x0, y0, cw / crop.shape[1], ch / crop.shape[0])

    def sample(self, D, geo, u):
        """Bilinear descriptors at full-res pixels u (N,2) -> (N,128) normalised torch."""
        torch = self.torch
        x0, y0, sx, sy = geo; h, w = D.shape[1:]
        m = (np.asarray(u, np.float64) - [x0, y0]) * [sx, sy]
        gx = 2 * (m[:, 0] + 0.5) / w - 1; gy = 2 * (m[:, 1] + 0.5) / h - 1          # align_corners=False convention
        grid = torch.from_numpy(np.stack([gx, gy], 1)).float().to(self.dev)[None, None]
        d = torch.nn.functional.grid_sample(D[None], grid, mode="bilinear", align_corners=False)[0, :, 0].T
        return torch.nn.functional.normalize(d, dim=1)

    def search(self, D, geo, refs, centers, radius, excl, chunk=48):
        """For each point k: best match to any of its reference descriptors refs[k] (M_k,128) inside a disc of `radius`
        full-res px around centers[k] (None = whole map). Returns pos (K,2) full-res, peak (K,), second (K,) = best
        similarity farther than `excl` full-res px from the peak (distinctiveness)."""
        torch = self.torch
        x0, y0, sx, sy = geo; h, w = D.shape[1:]; K = len(refs)
        pos = np.full((K, 2), np.nan); peak = np.full(K, -1.0); second = np.full(K, -1.0)
        if K == 0: return pos, peak, second
        Mmax = max(len(r) for r in refs)
        Rf = torch.zeros((K, Mmax, D.shape[0]), device=self.dev)
        for k, r in enumerate(refs): Rf[k, :len(r)] = r
        if centers is None:                                               # global: whole map, one matmul per chunk
            Df = D.reshape(D.shape[0], -1); chunk = max(1, int(1.2e8 // (Mmax * Df.shape[1])))
            for c0 in range(0, K, chunk):
                sim = torch.einsum("kmd,dn->kmn", Rf[c0:c0 + chunk], Df).max(1).values      # (k, h*w)
                self._peaks(sim.reshape(-1, h, w), np.zeros((len(sim), 2)), (sx + sy) / 2 * excl, pos, peak, second, c0, geo, None)
            return pos, peak, second
        R_ = int(np.ceil(radius * max(sx, sy))) + 1
        Dp = torch.nn.functional.pad(D, (R_, R_, R_, R_))                  # zero descriptors outside the crop -> sim 0
        mc = (np.asarray(centers, np.float64) - [x0, y0]) * [sx, sy]
        ci = np.round(mc).astype(int)
        oy, ox = np.mgrid[-R_:R_ + 1, -R_:R_ + 1]
        disc = torch.from_numpy((ox / sx) ** 2 + (oy / sy) ** 2 <= radius ** 2).to(self.dev)
        oyt = torch.from_numpy(oy.ravel()).to(self.dev); oxt = torch.from_numpy(ox.ravel()).to(self.dev)
        chunk = max(1, int(1.2e8 // (D.shape[0] * oy.size)))
        for c0 in range(0, K, chunk):
            cc = torch.from_numpy(ci[c0:c0 + chunk]).to(self.dev)
            yy = (cc[:, 1:2] + R_ + oyt[None]).clamp(0, Dp.shape[1] - 1); xx = (cc[:, 0:1] + R_ + oxt[None]).clamp(0, Dp.shape[2] - 1)
            win = Dp[:, yy, xx]                                            # (128, k, L)
            sim = torch.einsum("kmd,dkl->kml", Rf[c0:c0 + chunk], win).max(1).values      # (k, L)
            sim = torch.where(disc.reshape(1, -1), sim, torch.full_like(sim, -1.0))
            self._peaks(sim.reshape(-1, 2 * R_ + 1, 2 * R_ + 1), ci[c0:c0 + chunk] - R_, (sx + sy) / 2 * excl, pos, peak, second, c0, geo, None)
        return pos, peak, second

    def _peaks(self, sim, origin, excl_m, pos, peak, second, c0, geo, _):
        torch = self.torch
        x0, y0, sx, sy = geo; k, hh, ww = sim.shape
        flat = sim.reshape(k, -1); v, idx = flat.max(1)
        py, px = (idx // ww).cpu().numpy(), (idx % ww).cpu().numpy(); v = v.cpu().numpy()
        oy, ox = torch.meshgrid(torch.arange(hh, device=sim.device), torch.arange(ww, device=sim.device), indexing="ij")
        pyt = torch.from_numpy(py).to(sim.device)[:, None, None]; pxt = torch.from_numpy(px).to(sim.device)[:, None, None]
        far = ((oy[None] - pyt) ** 2 + (ox[None] - pxt) ** 2) > excl_m ** 2
        sec = torch.where(far, sim, torch.full_like(sim, -1.0)).reshape(k, -1).max(1).values.cpu().numpy()
        S_ = sim.cpu().numpy()
        for j in range(k):
            fy, fx = float(py[j]), float(px[j])
            if 0 < py[j] < hh - 1 and 0 < px[j] < ww - 1:                 # sub-pixel: 1D parabola per axis
                a, b_, c = S_[j, py[j], px[j] - 1], S_[j, py[j], px[j]], S_[j, py[j], px[j] + 1]
                den = a - 2 * b_ + c; fx += 0.5 * (a - c) / den if den < 0 else 0
                a, c = S_[j, py[j] - 1, px[j]], S_[j, py[j] + 1, px[j]]
                den = a - 2 * b_ + c; fy += 0.5 * (a - c) / den if den < 0 else 0
            mx, my = fx + origin[j][0], fy + origin[j][1]
            pos[c0 + j] = [mx / sx + x0, my / sy + y0]; peak[c0 + j] = v[j]; second[c0 + j] = sec[j]


# ----------------------------------------------------------------------------- learned optical flow (torchvision RAFT)
class Flow:
    def __init__(self, which="large", iters=12):
        import torch
        from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
        self.torch = torch; self.dev = "cuda" if torch.cuda.is_available() else "cpu"; self.iters = iters
        self.model = (raft_large(weights=Raft_Large_Weights.DEFAULT) if which == "large" else raft_small(weights=Raft_Small_Weights.DEFAULT)).eval().to(self.dev)

    def pair(self, g0, g1, box):
        """forward and backward flow (2,h,w numpy) on the crop box=(x0,y0,x1,y1) of grey frames g0 -> g1."""
        torch = self.torch; x0, y0, x1, y1 = box
        def prep(g):
            t_ = torch.from_numpy(g[y0:y1, x0:x1]).to(self.dev).float().div(127.5).sub(1.0)
            return t_[None, None].repeat(1, 3, 1, 1)
        a_, b_ = prep(g0), prep(g1)
        with torch.no_grad():
            f = self.model(torch.cat([a_, b_]), torch.cat([b_, a_]), num_flow_updates=self.iters)[-1]
        f = f.cpu().numpy(); return f[0], f[1]


def bilinear(F2, pts):
    """sample a (2,h,w) field at float pixel coords pts (n,2) (crop coords)."""
    h, w = F2.shape[1:]; x = np.clip(pts[:, 0], 0, w - 1.001); y = np.clip(pts[:, 1], 0, h - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); fx = x - x0; fy = y - y0
    out = np.zeros((len(pts), 2))
    for c in range(2):
        f = F2[c]
        out[:, c] = (f[y0, x0] * (1 - fx) * (1 - fy) + f[y0, x0 + 1] * fx * (1 - fy) + f[y0 + 1, x0] * (1 - fx) * fy + f[y0 + 1, x0 + 1] * fx * fy)
    return out


def voxel_keep(X, vox, nmax):
    """indices of one point per voxel (first occurrence), capped at nmax by uniform subsampling."""
    if len(X) == 0: return np.zeros(0, int)
    key = np.floor(X / vox).astype(np.int64); _, idx = np.unique(key, axis=0, return_index=True); idx = np.sort(idx)
    if len(idx) > nmax: idx = idx[np.linspace(0, len(idx) - 1, nmax).round().astype(int)]
    return idx


def pca_normals(X, k=12):
    """unit normals from the smallest principal axis of the k nearest neighbours (sign arbitrary)."""
    from scipy.spatial import cKDTree
    n = len(X)
    if n < 4: return np.tile([0.0, 0.0, 1.0], (n, 1))
    k = min(k, n); tree = cKDTree(X); _, nn = tree.query(X, k=k)
    P_ = X[nn] - X[nn].mean(1, keepdims=True); C = np.einsum("nki,nkj->nij", P_, P_)
    w_, v_ = np.linalg.eigh(C); return v_[:, :, 0]


def gn_pose(Xm, obs_k, obs_c, obs_u, Kc, Rc, tc, R0, t0, iters=5, huber=2.0):
    """Gauss-Newton on the 6-DoF motion (R, t) with X_t = R Xm + t, two cameras (Rc, tc, Kc world->cam), Huber weights."""
    R, t = R0.copy(), t0.copy()
    for _ in range(iters):
        Y = Xm[obs_k] @ R.T + t                                     # (n,3) world at t
        J = np.zeros((len(obs_k), 2, 6)); r = np.zeros((len(obs_k), 2))
        for c in (0, 1):
            m = obs_c == c
            if not m.any(): continue
            Z = Y[m] @ Rc[c].T + tc[c]; z = np.maximum(Z[:, 2], 1e-4); f = Kc[c]
            uv = Z[:, :2] / z[:, None] * [f[0, 0], f[1, 1]] + [f[0, 2], f[1, 2]]; r[m] = uv - obs_u[m]
            dpi = np.zeros((m.sum(), 2, 3)); dpi[:, 0, 0] = f[0, 0] / z; dpi[:, 0, 2] = -f[0, 0] * Z[:, 0] / z ** 2
            dpi[:, 1, 1] = f[1, 1] / z; dpi[:, 1, 2] = -f[1, 1] * Z[:, 1] / z ** 2
            RX = Y[m] - t                                          # R Xm
            dY = np.zeros((m.sum(), 3, 6)); dY[:, :, 3:] = np.eye(3)
            dY[:, 0, 1] = RX[:, 2]; dY[:, 0, 2] = -RX[:, 1]; dY[:, 1, 0] = -RX[:, 2]; dY[:, 1, 2] = RX[:, 0]; dY[:, 2, 0] = RX[:, 1]; dY[:, 2, 1] = -RX[:, 0]
            J[m] = np.einsum("nij,jk,nkl->nil", dpi, Rc[c], dY)
        e = np.linalg.norm(r, axis=1); w = np.where(e <= huber, 1.0, huber / np.maximum(e, 1e-9))
        Jw = (J * w[:, None, None]).reshape(-1, 6); rw = (r * w[:, None]).reshape(-1)
        H = Jw.T @ Jw + 1e-6 * np.eye(6); g = Jw.T @ rw
        try: d = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError: break
        R = ortho(Rotation.from_rotvec(d[:3]).as_matrix() @ R); t = Rotation.from_rotvec(d[:3]).as_matrix() @ t + d[3:]
        if np.linalg.norm(d) < 1e-7: break
    return R, t


def f2_ransac(Xc, tracks, frames, Kc, Rc, tc, n_samples=300, k_sample=5, tau=1.0, min_frac=0.8, seed=0):
    """GT-free phantom filter: candidates Xc (n,3) with plain LK chains tracks[c] (n, T, 2) (NaN = lost) over `frames` (sorted,
    frames[0] = reference where the pose is identity). Sample k candidates, solve the pose of every frame from the sample's
    two-view tracks (GN from the previous frame), score all candidates by their reprojection residuals; a candidate is
    consistent if in each view it is observed in >= min_frac of the frames with median residual <= tau and p90 <= 2.5 tau.
    Returns (inlier mask, poses dict) of the best sample (refit on its inliers)."""
    rng = np.random.default_rng(seed); n = len(Xc); best = (-1, None, None)
    obs_f = {}
    for f in frames[1:]:
        kk, cc, uu = [], [], []
        for c in (0, 1):
            p = tracks[c][:, f]; ok = ~np.isnan(p[:, 0])
            kk.append(np.where(ok)[0]); cc.append(np.full(ok.sum(), c)); uu.append(p[ok])
        obs_f[f] = (np.concatenate(kk), np.concatenate(cc), np.vstack(uu) if uu else np.zeros((0, 2)))
    alive = np.array([all(not np.isnan(tracks[c][i, frames[-1], 0]) for c in (0, 1)) for i in range(n)])
    pool = np.where(alive)[0]
    if len(pool) < k_sample: pool = np.arange(n)

    def run(sample, score=True):
        R, t = np.eye(3), np.zeros(3); poses = {frames[0]: (R, t)}; res = np.full((2, n, len(frames) - 1), np.nan)
        for j, f in enumerate(frames[1:]):
            k, c, u = obs_f[f]; m = np.isin(k, sample)
            if m.sum() < 4: return None, None
            R, t = gn_pose(Xc, k[m], c[m], u[m], Kc, Rc, tc, R, t); poses[f] = (R, t)
            Y = Xc[k] @ R.T + t
            for cv in (0, 1):
                mm = c == cv
                if not mm.any(): continue
                Z = Y[mm] @ Rc[cv].T + tc[cv]; z = np.maximum(Z[:, 2], 1e-4); fK = Kc[cv]
                uv = Z[:, :2] / z[:, None] * [fK[0, 0], fK[1, 1]] + [fK[0, 2], fK[1, 2]]; res[cv, k[mm], j] = np.linalg.norm(uv - u[mm], axis=1)
        nf = len(frames) - 1; ok = np.ones(n, bool)
        for cv in (0, 1):
            r = res[cv]; cnt = np.isfinite(r).sum(1); med = np.nanmedian(np.where(np.isfinite(r), r, np.nan), axis=1); p90 = np.nanpercentile(np.where(np.isfinite(r), r, np.nan), 90, axis=1)
            ok &= (cnt >= min_frac * nf) & (med <= tau) & (p90 <= 2.5 * tau)
        return ok, poses
    for _ in range(n_samples):
        sample = rng.choice(pool, size=min(k_sample, len(pool)), replace=False)
        ok, poses = run(sample)
        if ok is None: continue
        sc = int(ok.sum())
        if sc > best[0]: best = (sc, ok, poses)
    if best[1] is None or best[0] < 4: return np.zeros(n, bool), {}
    ok, poses = run(np.where(best[1])[0])        # refit on the consensus set
    if ok is None or ok.sum() < 4: return best[1], best[2]
    return ok, poses


# ----------------------------------------------------------------------------- the solver
class Anchor:
    def __init__(self, a, K, E, X_B):
        self.a = a; self.set_model(X_B)
        self.Kc = np.array([K[c] for c in ("ext1", "ext2")]); self.Ec = np.array([E[c] for c in ("ext1", "ext2")])
        self.Rc = self.Ec[:, :3, :3]; self.tc = self.Ec[:, :3, 3]
        self.P = np.array([self.Kc[i] @ self.Ec[i][:3] for i in range(2)])

    def set_model(self, X):
        from scipy.spatial import cKDTree
        self.X = X; self.tree = cKDTree(X); self.normals = pca_normals(X, 12) if len(X) >= 12 else None

    def assoc(self, R, t, S, max_m):
        """model index and distance for every stereo point S (world) under pose (R,t); -1 beyond max_m."""
        Y = (S - t) @ R; d, j = self.tree.query(Y, k=1, distance_upper_bound=max_m)
        ok = np.isfinite(d); j = np.where(ok, j, -1); return j, d, Y

    def project(self, ci, R, t, idx=None):
        X = self.X if idx is None else self.X[idx]
        Z = (X @ R.T + t) @ self.Rc[ci].T + self.tc[ci]
        z = Z[:, 2]; uv = Z[:, :2] / np.maximum(z, 1e-3)[:, None] * [self.Kc[ci][0, 0], self.Kc[ci][1, 1]] + [self.Kc[ci][0, 2], self.Kc[ci][1, 2]]
        return uv, z

    def resid(self, R, t, obs):
        """per-observation reprojection error (px); obs = (k, ci, u[, Lw]) with optional whitening Lw (n,2,2)."""
        k, ci, u = obs[:3]; Lw = obs[3] if len(obs) > 3 else None; e = np.full(len(k), np.inf)
        for c in (0, 1):
            m = ci == c
            if m.any():
                uv, z = self.project(c, R, t, k[m]); d = uv - u[m]
                if Lw is not None: d = np.einsum("nij,nj->ni", Lw[m], d)
                ee = np.linalg.norm(d, axis=1); ee[z <= 0.05] = np.inf; e[m] = ee
        return e

    def hypotheses(self, obs):
        k, ci, u = obs[:3]; out = []
        for c in (0, 1):                                                   # per-view PnP-RANSAC
            m = ci == c
            if m.sum() < 6: continue
            ok, rv, tv, inl = cv2.solvePnPRansac(self.X[k[m]].astype(np.float64), u[m].astype(np.float64), self.Kc[c], None,
                                                 iterationsCount=200, reprojectionError=self.a.tau_in, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
            if ok and inl is not None and len(inl) >= 4:
                Rp = cv2.Rodrigues(rv)[0]; out.append(("pnp%d" % (c + 1), ortho(self.Rc[c].T @ Rp), self.Rc[c].T @ (tv.ravel() - self.tc[c])))
        both = np.intersect1d(k[ci == 0], k[ci == 1])                      # both-view points: triangulate + RANSAC Kabsch
        if len(both) >= 3:
            u1 = np.array([u[(ci == 0) & (k == j)][0] for j in both]); u2 = np.array([u[(ci == 1) & (k == j)][0] for j in both])
            Xh = cv2.triangulatePoints(self.P[0], self.P[1], u1.T.astype(np.float64), u2.T.astype(np.float64)); Y = (Xh[:3] / Xh[3]).T
            R, t, inl = rt.ransac_kabsch(self.X[both], Y, np.ones(len(both)), self.a.kabsch_m, iters=200)
            if R is not None: out.append(("kabsch", R, t))
        return out

    def refine_t(self, R0, t0, obs):
        """Huber LS of the translation only (rotation held at R0), both views."""
        k, ci, u = obs[:3]; Lw = obs[3] if len(obs) > 3 else None
        def fun(x):
            r = []
            for c in (0, 1):
                m = ci == c
                if m.any():
                    uv, _ = self.project(c, R0, t0 + x, k[m]); d = uv - u[m]
                    if Lw is not None: d = np.einsum("nij,nj->ni", Lw[m], d)
                    r.append(d.ravel())
            return np.concatenate(r)
        res = least_squares(fun, np.zeros(3), method="trf", loss="huber", f_scale=self.a.huber_px, x_scale=1.0, max_nfev=50)
        return R0, t0 + res.x

    def refine(self, R0, t0, obs, prior=None, stereo=None):
        """Huber LS of the 6-DoF motion on all given observations (both views) around (R0, t0); optional motion prior =
        (R_pred, t_pred, sigma_deg, sigma_mm) on the deviation from the prediction (model centroid for translation);
        optional stereo = (S (n,3) world points of this frame, w_px_per_mm, assoc_m, iters): point-to-plane (or point-to-point
        when the model has no normals) residuals of S against the model, re-associated `iters` times (ICP)."""
        k, ci, u = obs[:3]; Lw = obs[3] if len(obs) > 3 else None; a = self.a; cen = self.X.mean(0)
        S, w3, amax, iters = stereo if stereo is not None else (None, 0.0, 0.0, 1)
        R0c, t0c = R0, t0
        for it in range(max(1, iters)):
            if S is not None and len(S):
                j, d, _ = self.assoc(R0c, t0c, S, amax); sel = np.where(j >= 0)[0]; Sm = S[sel]; Mm = self.X[j[sel]]
                Nm = self.normals[j[sel]] if self.normals is not None else None
            else:
                Sm = None
            def fun(x):
                Rd = Rotation.from_rotvec(x[:3]).as_matrix(); R = Rd @ R0c; t = Rd @ t0c + x[3:]
                r = []
                for c in (0, 1):
                    m = ci == c
                    if m.any():
                        uv, _ = self.project(c, R, t, k[m]); dd = uv - u[m]
                        if Lw is not None: dd = np.einsum("nij,nj->ni", Lw[m], dd)
                        r.append(dd.ravel())
                if Sm is not None and len(Sm):
                    Y = (Sm - t) @ R - Mm                                   # stereo points in the model frame minus their model match
                    r.append((np.einsum("ni,ni->n", Y, Nm) if Nm is not None else Y.ravel()) * 1000.0 * w3)
                if prior is not None:
                    Rp, tp, sd, sm = prior
                    r.append(np.degrees(Rotation.from_matrix(R @ Rp.T).as_rotvec()) / sd * a.huber_px)
                    r.append(((R @ cen + t) - (Rp @ cen + tp)) * 1000 / sm * a.huber_px)
                return np.concatenate(r) if r else np.zeros(1)
            if (not len(k)) and (Sm is None or not len(Sm)): break
            res = least_squares(fun, np.zeros(6), method="trf", loss="huber", f_scale=a.huber_px, x_scale=1.0, max_nfev=50)
            Rd = Rotation.from_rotvec(res.x[:3]).as_matrix(); R0c, t0c = ortho(Rd @ R0c), Rd @ t0c + res.x[3:]
        return R0c, t0c

    def stereo_stats(self, R, t, S, amax):
        """(n_assoc, rms_mm, median_mm) of the stereo points within amax of the model under (R,t); point-to-plane if normals."""
        if S is None or not len(S): return 0, np.nan, np.nan
        j, d, Y = self.assoc(R, t, S, amax); sel = j >= 0
        if not sel.any(): return 0, np.nan, np.nan
        e = np.abs(np.einsum("ni,ni->n", Y[sel] - self.X[j[sel]], self.normals[j[sel]])) if self.normals is not None else d[sel]
        return int(sel.sum()), float(np.sqrt(np.mean(e ** 2)) * 1000), float(np.median(e) * 1000)

    def spread(self, idx):
        if len(idx) < 3: return 0.0
        Xg = self.X[idx] - self.X[idx].mean(0); _, _, Vt = np.linalg.svd(Xg, full_matrices=False); p = Xg @ Vt[1]
        return float(p.max() - p.min())


def run(a):
    ep = a.episode; t_start = time.time(); rec = dict(episode=ep, tag=a.tag)
    ev = [r for r in csv.DictReader(open(a.events)) if r["episode"] == ep and r["event"] == "0"]
    if not ev: raise rt.Failure("no event 0 (no birth frame / excluded)")
    b, e = int(ev[0]["entry_frame"]), int(ev[0]["exit_frame"])
    cams, K, E, P, ksrc = rte.calib(ep, rte.POSITIONS)
    meta = json.load(open(f"{rt.RAW_ROOT}/{ep}/metadata_{ep}.json"))
    store = f"{rt.STORE_ROOT}/{ep}/dense/cam"; n_store = rt.count_files(store, ".npz")
    T = n_store if a.window == "full" else min(e, n_store)
    sf = f"{a.seeds}/{ep}/matches.npz"
    if not os.path.isfile(sf): raise rt.Failure("no birth seeds (fewer than 3 gated matches)")
    zs = np.load(sf); x = {"ext1": zs["x1"].astype(np.float64), "ext2": zs["x2"].astype(np.float64)}
    N = len(x["ext1"])
    if N < a.min_seeds: raise rt.Failure(f"fewer than {a.min_seeds} birth seeds ({N})")
    ext = None
    if a.obs in ("tracks", "both"):
        ext = {}
        for c in ("ext1", "ext2"):
            f = f"{a.obs_tracks}/{ep}/{c}_f{b:05d}.npz"
            if not os.path.isfile(f): raise rt.Failure(f"observation track file missing: {os.path.basename(f)}")
            zt = np.load(f)
            if zt["tracks"].shape[1] < N or np.nanmax(np.abs(zt["queries"][:N, 1:] - x[c])) > 1.0: raise rt.Failure("observation tracks do not match the seeds")
            ext[c] = (zt["tracks"][:, :N].astype(np.float64), zt["visibility"][:, :N].astype(bool))
    gray = {c: rt.read_frames(f"{rt.RAW_ROOT}/{ep}/recordings/MP4/{s}.mp4", T) for c, s in cams}
    T = min(T, min(len(gray[c]) for c in gray)); t_end = T - 1
    if t_end - b < 2: raise rt.Failure(f"window too short (b={b}, T={T})")
    # birth model (world = robot base frame at b)
    Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], x["ext1"].T, x["ext2"].T); X_B = (Xh[:3] / Xh[3]).T
    clouds = None; rec_stereo = None; clouds_src = None
    if a.stereo:
        cf = f"{a.stereo}/{ep}.npz"
        if not os.path.isfile(cf): raise rt.Failure("no stereo cloud file")
        zc = np.load(cf); okc = (zc["sim"] >= a.stereo_sim) & (zc["epi"] <= a.stereo_epi)
        if a.f1_m3d > 0 and "m3d" in zc.files:
            metac = json.loads(str(zc["meta"])); sc_ = {p_["t"]: p_.get("scale", float("nan")) for p_ in metac["per_frame"]}
            scale_pt = np.array([sc_.get(int(t_), np.nan) for t_ in zc["frame"]])
            okc &= (zc["m3d"] * scale_pt <= a.f1_m3d) & (zc["pconf"].min(1) >= a.f1_pconf)
        fr_ = zc["frame"][okc]; Xc = zc["X"][okc].astype(np.float64); x1c = zc["x1"][okc].astype(np.float64); x2c = zc["x2"][okc].astype(np.float64)
        order = np.argsort(fr_, kind="stable"); fr_, Xc, x1c, x2c = fr_[order], Xc[order], x1c[order], x2c[order]
        srcc = zc["src"][okc][order] if "src" in zc.files else np.ones(len(fr_), np.int8)
        starts = np.searchsorted(fr_, np.arange(0, T + 1)); clouds = {t_: (Xc[starts[t_]:starts[t_ + 1]], x1c[starts[t_]:starts[t_ + 1]], x2c[starts[t_]:starts[t_ + 1]]) for t_ in range(T) if starts[t_ + 1] > starts[t_]}
        clouds_src = {t_: srcc[starts[t_]:starts[t_ + 1]] for t_ in range(T) if starts[t_ + 1] > starts[t_]}
        rec_stereo = dict(frames_with_cloud=len(clouds), pts_median=float(np.median([len(v[0]) for v in clouds.values()])) if clouds else 0)
        if a.model in ("stereo", "both") and b in clouds and len(clouds[b][0]) >= a.min_seeds:
            Sb, s1, s2 = clouds[b]
            if a.model == "both": Sb, s1, s2 = np.vstack([X_B, Sb]), np.vstack([x["ext1"], s1]), np.vstack([x["ext2"], s2])
            keep = voxel_keep(Sb, a.model_vox * 1e-3, a.model_max)
            X_B = Sb[keep]; x = {"ext1": s1[keep], "ext2": s2[keep]}; N = len(X_B)
            rec_stereo["model_birth"] = int(N)
    cand_src = np.zeros(N, np.int8)      # 0 = DISK seed, 1/2/3 = stereo cloud source, 4 = extended-region DISK (in-process)
    if a.cands_ext > 0:
        import pandas as _pd, rig_match_track as _rmt
        it0 = _pd.read_csv(f"{MASKS}/items_v2.csv"); it0 = it0[(it0.episode == ep) & (it0.frame == b)]
        reg_ext = {}
        for c in ("ext1", "ext2"):
            r_ = it0[it0.cam == c]; mm = cv2.imread(f"{MASKS}/{r_.mask_relpath.iloc[0]}", 0) if len(r_) else None
            if mm is None: reg_ext = None; break
            if mm.shape != (H, W): mm = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)
            reg_ext[c] = cv2.dilate((mm > 0).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.cands_ext + 1,) * 2))
        if reg_ext is not None:
            R1_, t1_, R2_, t2_ = E["ext1"][:3, :3], E["ext1"][:3, 3], E["ext2"][:3, :3], E["ext2"][:3, 3]; Rr_ = R2_ @ R1_.T; trel_ = t2_ - Rr_ @ t1_
            F_ = np.linalg.inv(K["ext2"]).T @ rt.skew(trel_) @ Rr_ @ np.linalg.inv(K["ext1"])
            g1_ = rt.read_frames(f"{rt.RAW_ROOT}/{ep}/recordings/MP4/{meta['ext1_cam_serial']}.mp4", b + 1)[b]; g2_ = rt.read_frames(f"{rt.RAW_ROOT}/{ep}/recordings/MP4/{meta['ext2_cam_serial']}.mp4", b + 1)[b]
            xa, xb, cf_ = _rmt.match_disk_epi(g1_, g2_, reg_ext["ext1"], reg_ext["ext2"], F_, a.cands_ext_side, 2.5)
            sel_, _ = _rmt.gate(xa, xb, cf_, reg_ext["ext1"], reg_ext["ext2"], F_, P, E, 2.5, 0.3)
            if len(sel_):
                xa, xb = xa[sel_], xb[sel_]
                # drop candidates duplicating the existing seeds (within 2 px in view 1)
                if N: keep_ = np.array([np.min(np.linalg.norm(x["ext1"] - p_, axis=1)) > 2.0 for p_ in xa])
                else: keep_ = np.ones(len(xa), bool)
                xa, xb = xa[keep_], xb[keep_]
                Xh_ = cv2.triangulatePoints(P["ext1"], P["ext2"], xa.T, xb.T); Xe = (Xh_[:3] / Xh_[3]).T
                if a.cands == "ext_only" or N == 0:
                    X_B, x, N = Xe, {"ext1": xa, "ext2": xb}, len(Xe); cand_src = np.full(N, 4, np.int8)
                else:
                    X_B = np.vstack([X_B, Xe]); x = {"ext1": np.vstack([x["ext1"], xa]), "ext2": np.vstack([x["ext2"], xb])}
                    cand_src = np.concatenate([cand_src, np.full(len(Xe), 4, np.int8)]); N = len(X_B)
    if a.cands != "seeds" and a.stereo:
        cf = f"{a.stereo}/{ep}/{ep}.npz" if os.path.isdir(f"{a.stereo}/{ep}") else f"{a.stereo}/{ep}.npz"
        if os.path.isfile(cf):
            zc = np.load(cf); atb = (zc["frame"] == b) & (zc["sim"] >= a.stereo_sim) & (zc["epi"] <= a.stereo_epi)
            if a.f1_m3d > 0 and "m3d" in zc.files:
                metac = json.loads(str(zc["meta"])); sb_ = next((p_.get("scale", float("nan")) for p_ in metac["per_frame"] if p_["t"] == b), float("nan"))
                atb &= (zc["m3d"] * sb_ <= a.f1_m3d) & (zc["pconf"].min(1) >= a.f1_pconf)
            if a.cands == "recip": atb &= np.isin(zc["src"], [1, 3])
            if atb.any():
                order = np.argsort(-zc["sim"][atb])[: a.cands_max]; idx = np.where(atb)[0][order]
                Xn = zc["X"][idx].astype(np.float64); x1n = zc["x1"][idx].astype(np.float64); x2n = zc["x2"][idx].astype(np.float64)
                if a.cands in ("recip", "guided_only"):
                    X_B, x, N = Xn, {"ext1": x1n, "ext2": x2n}, len(Xn); cand_src = zc["src"][idx].astype(np.int8)
                else:
                    X_B = np.vstack([X_B, Xn]); x = {"ext1": np.vstack([x["ext1"], x1n]), "ext2": np.vstack([x["ext2"], x2n])}
                    cand_src = np.concatenate([cand_src, zc["src"][idx].astype(np.int8)]); N = len(X_B)
    if a.model_labels:   # GT DIAGNOSTIC ONLY: perfect model = FK label points (world frame at b); isolates tracker/solver from model errors
        zl = np.load(f"{rte.POSITIONS}/{ep}_gripper.npz"); s0 = str(zl["serials"][0])
        uvl = zl[f"{s0}_uv"][b].astype(np.float64); dl = zl[f"{s0}_z"][b].astype(np.float64); Kl = zl[f"{s0}_K"]; W2C = zl[f"{s0}_W2C"]; okl = zl[f"{s0}_front"][b] & zl[f"{s0}_in_frame"][b]
        if a.model_labels == "base": okl &= zl["link"] == "robotiq_85_base_link"
        Xcam = np.c_[(uvl[:, 0] - Kl[0, 2]) / Kl[0, 0] * dl, (uvl[:, 1] - Kl[1, 2]) / Kl[1, 1] * dl, dl]
        X_B = ((Xcam - W2C[:3, 3]) @ W2C[:3, :3])[okl]; N = len(X_B)
        x = {}
        for c in ("ext1", "ext2"):
            Z_ = X_B @ E[c][:3, :3].T + E[c][:3, 3]; x[c] = Z_[:, :2] / Z_[:, 2:3] * [K[c][0, 0], K[c][1, 1]] + [K[c][0, 2], K[c][1, 2]]
        cand_src = np.full(N, 9, np.int8)
        if a.labels_vis:   # GT DIAGNOSTIC: per-frame visibility of the label points (front & in_frame) per view
            lab_vis = {}
            for ci_, c in enumerate(("ext1", "ext2")):
                sc = str(meta[f"{c}_cam_serial"]); lab_vis[ci_] = (zl[f"{sc}_front"] & zl[f"{sc}_in_frame"])[:, okl] if f"{sc}_front" in zl.files else None
    # (b) GT-free F1 phantom filter on cloud candidates: MASt3R two-view consistency (scaled) and pointmap confidence
    bmask = {}
    if a.seed_mask or a.extend or a.mwt:
        import pandas as _pd
        it = _pd.read_csv(f"{MASKS}/items_v2.csv"); it = it[(it.episode == ep) & (it.frame == b)]
        for c in ("ext1", "ext2"):
            r_ = it[it.cam == c]
            if len(r_):
                mm = cv2.imread(f"{MASKS}/{r_.mask_relpath.iloc[0]}", 0)
                if mm is not None:
                    if mm.shape != (H, W): mm = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)
                    bmask[c] = mm > 0
    if a.seed_mask and len(bmask) == 2:
        keep = np.ones(N, bool)
        for c in ("ext1", "ext2"):
            md = cv2.dilate(bmask[c].astype(np.uint8), np.ones((2 * a.seed_mask_dil + 1,) * 2, np.uint8)) > 0
            xi = np.clip(np.round(x[c]).astype(int), [0, 0], [W - 1, H - 1]); keep &= md[xi[:, 1], xi[:, 0]]
        rec_seed_mask = dict(n_before=int(N), n_after=int(keep.sum()))
        if keep.sum() >= a.min_seeds:
            X_B = X_B[keep]; x = {c: x[c][keep] for c in x}; N = int(keep.sum())
            if ext is not None: ext = {c: (ext[c][0][:, keep], ext[c][1][:, keep]) for c in ext}
    else:
        rec_seed_mask = None
    an = Anchor(a, K, E, X_B)
    Xc_ = X_B.mean(0); lkw = {}
    for c in ("ext1", "ext2"):
        dep = float((E[c][:3, :3] @ Xc_ + E[c][:3, 3])[2]); sz = min(1.0, K[c][0, 0] / (max(dep, 0.05) * 100) / rt.PPCM_RAIL)
        lkw[c] = int(max(7, a.lk_win * sz)) | 1
    # Track-On v2 (image-based region, used for the birth crop = the seeds' own crop, and for global re-localisation)
    qz = {c: np.load(f"{a.tracks_v2}/{ep}/{c}_f{b:05d}.npz") for c in ("ext1", "ext2")}
    q = {c: qz[c]["queries"][:, 1:] for c in qz}; qtr = {c: qz[c]["tracks"] for c in qz}; qvis = {c: qz[c]["visibility"] for c in qz}

    def hull3d(t):
        """coarse 3D position of the gripper region at t: closest point between the two views' rays through the median of the
        visible Track-On v2 points (None if a view has < 3 visible points). Image-based, causal."""
        cs = []
        for c in ("ext1", "ext2"):
            if t >= len(qtr[c]): return None
            p = qtr[c][t]; ok = qvis[c][t] & ~np.isnan(p[:, 0]) & (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
            if ok.sum() < 3: return None
            cs.append(np.median(p[ok], 0))
        o, d = [], []
        for c, u in zip(("ext1", "ext2"), cs):
            Rc_, tc_ = E[c][:3, :3], E[c][:3, 3]; r_ = Rc_.T @ (np.linalg.inv(K[c]) @ np.r_[u, 1.0]); o.append(-Rc_.T @ tc_); d.append(r_ / np.linalg.norm(r_))
        o1, o2, d1, d2 = o[0], o[1], d[0], d[1]; A = np.array([[d1 @ d1, -d1 @ d2], [d1 @ d2, -d2 @ d2]]); rhs = np.array([(o2 - o1) @ d1, (o2 - o1) @ d2])
        try: s1, s2 = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError: return None
        if s1 <= 0.1 or s2 <= 0.1: return None
        return 0.5 * ((o1 + s1 * d1) + (o2 + s2 * d2))

    def v2_box(c, t, margin=20):
        if t < len(qtr[c]):
            p = qtr[c][t]; ok = qvis[c][t] & ~np.isnan(p[:, 0]) & (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
            if ok.sum() >= 3: return box_of(p[ok], margin)
        return None

    def box_of(pts, margin):
        x0, y0 = np.floor(pts.min(0) - margin).astype(int); x1, y1 = np.ceil(pts.max(0) + margin + 1).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
        return (x0, y0, x1, y1) if x1 - x0 >= 16 and y1 - y0 >= 16 else None

    dn = Dense() if a.obs in ("dense", "both", "mklt+dense") else None
    flw = Flow(a.flow_model, a.flow_iters) if a.obs == "mflow" else None
    # birth crop exactly as rig_match_track (_disk_features on region_mask(q, 8) bbox + 20 px, long side -> 640)
    refs = [[] for _ in range(N)]; s_b, zb = {}, {}
    for ci, c in (enumerate(("ext1", "ext2")) if dn is not None else []):
        m = region_mask(q[c], a.region_dilate); ys, xs = np.nonzero(m)
        box = (max(xs.min() - 20, 0), max(ys.min() - 20, 0), min(xs.max() + 21, W), min(ys.max() + 21, H))
        s_b[c] = a.side / max(box[2] - box[0], box[3] - box[1])
        D, geo = dn.map(gray[c][b], box, s_b[c])
        d = dn.sample(D, geo, x[c])
        for k in range(N): refs[k].append(d[k])
        zb[c] = float(an.project(ci, np.eye(3), np.zeros(3))[1].mean())
    n_birth_refs = 2
    import torch
    refs = [torch.stack(r) for r in refs] if dn is not None else refs

    ext_tracks = {}
    if a.extend and len(bmask) == 2 and a.obs in ("mklt", "mklt+dense"):
        for c in ("ext1", "ext2"):
            me = cv2.erode(bmask[c].astype(np.uint8), np.ones((5, 5), np.uint8))
            for xx, yy in np.round(x[c]).astype(int): cv2.circle(me, (int(xx), int(yy)), 6, 0, -1)
            cn = cv2.goodFeaturesToTrack(gray[c][b], maxCorners=a.extend_max, qualityLevel=0.01, minDistance=6, mask=me)
            if cn is None: continue
            tr_ = np.full((len(cn), a.extend + 1, 2), np.nan, np.float32); tr_[:, 0] = cn.reshape(-1, 2); alive = np.ones(len(cn), bool)
            for j in range(1, a.extend + 1):
                if b + j > t_end: break
                ii = np.where(alive)[0]
                if not len(ii): break
                p0_ = tr_[ii, j - 1].reshape(-1, 1, 2)
                lk_ = dict(winSize=(lkw[c],) * 2, maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
                p1_, st_, _ = cv2.calcOpticalFlowPyrLK(gray[c][b + j - 1], gray[c][b + j], p0_, None, **lk_)
                pb_, st2_, _ = cv2.calcOpticalFlowPyrLK(gray[c][b + j], gray[c][b + j - 1], p1_, None, **lk_)
                fb_ = np.linalg.norm(pb_ - p0_, axis=2).ravel(); ok_ = (st_.ravel() == 1) & (st2_.ravel() == 1) & (fb_ < a.lk_fb)
                tr_[ii[ok_], j] = p1_.reshape(-1, 2)[ok_]; alive[ii[~ok_]] = False
            ext_tracks[c] = tr_
    n_extended = 0
    v2m = None
    if a.v2_model > 0:   # per view: (M,) object-frame 3D (nan until triangulated), the raw v2 tracks, visibility
        v2m = {ci: dict(X=np.full((len(q[c]), 3), np.nan), tr=qtr[c], vis=qvis[c], done=False) for ci, c in enumerate(("ext1", "ext2"))}
    f2 = None
    if 'lab_vis' not in dir(): lab_vis = None
    if a.f2_frames > 0:
        f2 = dict(tracks={0: np.full((N, T, 2), np.nan, np.float32), 1: np.full((N, T, 2), np.nan, np.float32)}, alive=np.ones(N, bool), checks=[], n_in=[], done=False)
        f2["tracks"][0][:, b] = x["ext1"]; f2["tracks"][1][:, b] = x["ext2"]
        f2_cp = sorted({b + k for k in (a.f2_frames // 3, 2 * a.f2_frames // 3, a.f2_frames) if k >= 3})
        Kc_ = np.array([K[c] for c in ("ext1", "ext2")]); Ec_ = np.array([E[c] for c in ("ext1", "ext2")]); Rc_, tc_ = Ec_[:, :3, :3], Ec_[:, :3, 3]
        X_all = an.X.copy()                                # all candidates (the model may shrink to the consensus set)
        model_idx = np.arange(N)                           # which candidates form the current model
    Rft = None
    if a.rot_ft:
        ftp = rt.load_poses(f"{rt.CUT3R_ROOT}/{ep}/camera", t_end + 1, "CUT3R")
        Rgb = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", b + 1, "store GT")[b][:3, :3]   # birth frame only
        Rft = {t_: ortho(Rgb @ ftp[b][:3, :3].T @ ftp[t_][:3, :3] @ Rgb.T) for t_ in ftp if t_ >= b}
    Gb = {c: gray[c][b].astype(np.float32) for c in ("ext1", "ext2")}; Mb = {c: bmask[c].astype(np.float32) for c in bmask}
    if a.lk_clip > 0:   # saturate highlights: LK then follows edges and matte structure, not the specular blobs
        gray_lk = {c: [np.minimum(g, a.lk_clip).astype(np.uint8) for g in gray[c]] for c in gray}
    else:
        gray_lk = gray
    c_hull_b = hull3d(b) if a.hull_reloc else None
    R_s, t_s, solved = {b: np.eye(3)}, {b: np.zeros(3)}, {b: True}
    last = b; rows = []; kf = b; kf_inl = None; n_kf = 0; kfs = [b]; vel = None; n_acc = 0; w_hat = None; Rfu = {}; bad_cnt = None; n_retired = 0; n_coast = 0; blind = 0; n_v2 = 0
    pend = dict(X=np.zeros((0, 3)), t0=np.zeros(0, int), tr={0: np.zeros((0, T, 2), np.float32), 1: np.zeros((0, T, 2), np.float32)}, alive=np.zeros(0, bool))
    n_grow_cand = n_grow_ok = 0
    for t in range(b + 1, t_end + 1):
        fs = t - last
        if f2 is not None and not f2["done"] and t <= b + a.f2_frames:
            for ci, c in enumerate(("ext1", "ext2")):
                idx = np.where(f2["alive"])[0]
                if not len(idx): break
                p0 = f2["tracks"][ci][idx, t - 1].reshape(-1, 1, 2)
                lk_ = dict(winSize=(lkw[c],) * 2, maxLevel=a.lk_levels, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
                pn, st, _ = cv2.calcOpticalFlowPyrLK(gray_lk[c][t - 1], gray_lk[c][t], p0, None, **lk_)
                pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray_lk[c][t], gray_lk[c][t - 1], pn, None, **lk_)
                fb = np.linalg.norm(pb - p0, axis=2).ravel(); pn = pn.reshape(-1, 2)
                good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < a.lk_fb) & (pn[:, 0] >= 0) & (pn[:, 0] < W) & (pn[:, 1] >= 0) & (pn[:, 1] < H)
                f2["tracks"][ci][idx[good], t] = pn[good]; f2["alive"][idx[~good]] = False
            if t in f2_cp:
                frames_ = list(range(b, t + 1))
                ok_, poses_ = f2_ransac(X_all, f2["tracks"], frames_, Kc_, Rc_, tc_, n_samples=a.f2_samples, k_sample=a.f2_k, tau=a.f2_px, min_frac=a.f2_frac, seed=t)
                f2["checks"].append(int(t)); f2["n_in"].append(int(ok_.sum()))
                if ok_.sum() >= a.min_seeds:
                    model_idx = np.where(ok_)[0]; an.set_model(X_all[model_idx])
                    f2["kept_x1"] = x["ext1"][model_idx].round(1).tolist(); f2["kept_src"] = cand_src[model_idx].tolist()
                    if t == f2_cp[-1]: f2["done"] = True
        if fs == 1 and (t - 2) in solved and solved.get(t - 2) and (t - 2) >= b:
            if a.vel_beta < 1.0 and vel is not None:     # smoothed velocity (rotvec, centroid displacement per frame)
                dR = Rotation.from_rotvec(vel[0]).as_matrix(); cen0 = an.X.mean(0)
                Rp = ortho(dR @ R_s[t - 1]); tp = (R_s[t - 1] @ cen0 + t_s[t - 1] + vel[1]) - Rp @ cen0
            else:
                dR = R_s[t - 1] @ R_s[t - 2].T; Rp = ortho(dR @ R_s[t - 1]); tp = dR @ t_s[t - 1] + (t_s[t - 1] - dR @ t_s[t - 2])
        else:
            Rp, tp = R_s[last], t_s[last]
        if Rft is not None and t in Rft:
            cen0 = an.X.mean(0); c_last = R_s[last] @ cen0 + t_s[last]
            v_c = (c_last - (R_s[last - 1] @ cen0 + t_s[last - 1])) if (fs == 1 and solved.get(last - 1) and last - 1 >= b) else np.zeros(3)
            if a.rot_ft_beta < 1.0 and (t - 1) in Rft:
                w_meas = Rotation.from_matrix(Rft[t] @ Rft[t - 1].T).as_rotvec()
                w_hat = w_meas if w_hat is None else (1 - a.rot_ft_beta) * w_hat + a.rot_ft_beta * w_meas
                Rfu[t] = ortho(Rotation.from_rotvec(w_hat).as_matrix() @ Rfu.get(t - 1, Rft[t - 1]))
                if a.rot_ft_pull > 0:   # slow pull toward the absolute CUT3R rotation (removes the smoothing lag bias, keeps it causal)
                    dv = Rotation.from_matrix(Rft[t] @ Rfu[t].T).as_rotvec(); Rfu[t] = ortho(Rotation.from_rotvec(a.rot_ft_pull * dv).as_matrix() @ Rfu[t])
                Rp = Rfu[t]
            else:
                Rp = Rft[t]
            tp = c_last + v_c - Rp @ cen0
        glob = fs > a.lost_global and dn is not None
        radius = min(a.radius * (1 + a.radius_grow * (fs - 1)), a.radius_max)
        obs_k, obs_c, obs_u, obs_pk, obs_sec = [], [], [], [], []; maps = {}; obs_W = []; n_mwt = [0, 0]
        for ci, c in enumerate(("ext1", "ext2")):
            if a.obs == "mflow" and not glob:
                uvp, zp = an.project(ci, R_s[last], t_s[last]); uvn, zn = an.project(ci, Rp, tp)
                okp = (zp > 0.05) & (uvp[:, 0] >= 2) & (uvp[:, 0] < W - 2) & (uvp[:, 1] >= 2) & (uvp[:, 1] < H - 2)
                idx = np.where(okp)[0]
                if len(idx) >= 3:
                    allp = np.vstack([uvp[idx], uvn[idx]]); m_ = a.flow_margin
                    x0f, y0f = np.floor(allp.min(0) - m_).astype(int); x1f, y1f = np.ceil(allp.max(0) + m_).astype(int)
                    x0f, y0f = max(x0f, 0), max(y0f, 0); x1f, y1f = min(x1f, W), min(y1f, H)
                    wf, hf = (x1f - x0f) // 8 * 8, (y1f - y0f) // 8 * 8
                    if wf >= 64 and hf >= 64:
                        x1f, y1f = x0f + wf, y0f + hf
                        ff, fbw = flw.pair(gray[c][last], gray[c][t], (x0f, y0f, x1f, y1f))
                        pc = uvp[idx] - [x0f, y0f]; fv = bilinear(ff, pc); pn = pc + fv; bv = bilinear(fbw, pn)
                        fbe = np.linalg.norm(fv + bv, axis=1); pn = pn + [x0f, y0f]
                        good = (fbe < a.flow_fb) & (pn[:, 0] >= 0) & (pn[:, 0] < W) & (pn[:, 1] >= 0) & (pn[:, 1] < H)
                        for j in np.where(good)[0]:
                            obs_k.append(idx[j]); obs_c.append(ci); obs_u.append(pn[j]); obs_pk.append(0.0); obs_sec.append(0.0); obs_W.append(None)
                continue
            if a.obs in ("mklt", "mklt+dense") and not glob:
                if a.kf_multi:   # latest keyframe, birth, and the keyframe whose pose is closest to the prediction
                    srcs = [kf, b]
                    if len(kfs) > 2:
                        dd = [rt.rot_angle_deg(Rp @ R_s[f].T) + a.kf_cm_per_deg_inv * np.linalg.norm((Rp @ an.X.mean(0) + tp) - (R_s[f] @ an.X.mean(0) + t_s[f])) * 100 for f in kfs]
                        srcs.append(kfs[int(np.argmin(dd))])
                    srcs = list(dict.fromkeys(srcs))
                else:
                    srcs = [kf if a.kf else last]
                uvn, zn = an.project(ci, Rp, tp)
                for src in srcs:
                  uvp, zp = an.project(ci, R_s[src], t_s[src])
                  okp = (zp > 0.05) & (zn > 0.05) & (uvp[:, 0] >= 2) & (uvp[:, 0] < W - 2) & (uvp[:, 1] >= 2) & (uvp[:, 1] < H - 2)
                  if lab_vis is not None and lab_vis.get(ci) is not None and t < len(lab_vis[ci]) and len(an.X) == lab_vis[ci].shape[1]:
                      okp &= lab_vis[ci][t] & lab_vis[ci][src]
                  idx = np.where(okp)[0]
                  if len(idx):
                      p0 = uvp[idx].astype(np.float32).reshape(-1, 1, 2); p1 = uvn[idx].astype(np.float32).reshape(-1, 1, 2)
                      lk = dict(winSize=(lkw[c],) * 2, maxLevel=a.lk_levels, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01), minEigThreshold=a.lk_mineig)
                      if a.lk_noinit:
                          pn, st, err = cv2.calcOpticalFlowPyrLK(gray_lk[c][src], gray_lk[c][t], p0, None, **lk)
                          pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray_lk[c][t], gray_lk[c][src], pn, None, **lk)
                      else:
                          pn, st, err = cv2.calcOpticalFlowPyrLK(gray_lk[c][src], gray_lk[c][t], p0, p1.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
                          pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray_lk[c][t], gray_lk[c][src], pn, p0.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
                      fb = np.linalg.norm(pb - p0, axis=2).ravel(); pn = pn.reshape(-1, 2)
                      good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < a.lk_fb) & (pn[:, 0] >= 0) & (pn[:, 0] < W) & (pn[:, 1] >= 0) & (pn[:, 1] < H)
                      if a.aperture:
                          gx = cv2.Sobel(gray[c][src], cv2.CV_32F, 1, 0, ksize=3) / 8.0; gy = cv2.Sobel(gray[c][src], cv2.CV_32F, 0, 1, ksize=3) / 8.0
                      for j in np.where(good)[0]:
                          obs_k.append(idx[j]); obs_c.append(ci); obs_u.append(pn[j].astype(np.float64)); obs_pk.append(0.0); obs_sec.append(0.0)
                          if a.aperture:
                              h = lkw[c] // 2; x0_, y0_ = int(round(p0[j, 0, 0])), int(round(p0[j, 0, 1]))
                              wx = gx[max(y0_ - h, 0):y0_ + h + 1, max(x0_ - h, 0):x0_ + h + 1]; wy = gy[max(y0_ - h, 0):y0_ + h + 1, max(x0_ - h, 0):x0_ + h + 1]
                              G = np.array([[(wx * wx).sum(), (wx * wy).sum()], [(wx * wy).sum(), (wy * wy).sum()]], np.float64)
                              tr_ = np.trace(G); Wm = 2 * G / tr_ + a.aperture_eps * np.eye(2) if tr_ > 0 else np.eye(2)
                              obs_W.append(np.linalg.cholesky(Wm).T)   # residual r -> L^T r with Wm = L L^T
                          else:
                              obs_W.append(None)
                if a.mwt and c in bmask:   # masked, model-warped BIRTH templates -> absolute observations of the seeds
                    A_ = an.Rc[ci] @ Rp @ an.Rc[ci].T; a_ = an.Rc[ci] @ tp + an.tc[ci] - A_ @ an.tc[ci]
                    Kc_ = an.Kc[ci]; Ki_ = np.linalg.inv(Kc_); h_ = max(5, lkw[c] // 2); r_ = a.mwt_r
                    nm = 0
                    for k in range(N):
                        Xc_ = an.Rc[ci] @ an.X[k] + an.tc[ci]; d_ = float(np.linalg.norm(Xc_)); n_ = Xc_ / d_
                        Hk = Kc_ @ (A_ + np.outer(a_, n_) / d_) @ Ki_
                        v0 = Hk @ np.r_[x[c][k], 1.0]
                        if v0[2] <= 0: continue
                        v0 = v0[:2] / v0[2]; vx, vy = int(round(v0[0])), int(round(v0[1]))
                        if vx - h_ - r_ < 0 or vy - h_ - r_ < 0 or vx + h_ + r_ >= W or vy + h_ + r_ >= H: continue
                        gy_, gx_ = np.mgrid[vy - h_:vy + h_ + 1, vx - h_:vx + h_ + 1].astype(np.float64)
                        Hi = np.linalg.inv(Hk); q_ = Hi @ np.stack([gx_.ravel(), gy_.ravel(), np.ones(gx_.size)])
                        mx_ = (q_[0] / q_[2]).reshape(gx_.shape).astype(np.float32); my_ = (q_[1] / q_[2]).reshape(gx_.shape).astype(np.float32)
                        Tm = cv2.remap(Gb[c], mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        Mm = cv2.remap(Mb[c], mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        Mm = (Mm > 0.5).astype(np.float32)
                        if Mm.mean() < a.mwt_minmask or Tm[Mm > 0].std() < a.mwt_minstd: continue
                        reg_ = gray[c][t][vy - h_ - r_:vy + h_ + r_ + 1, vx - h_ - r_:vx + h_ + r_ + 1].astype(np.float32)
                        ncc = cv2.matchTemplate(reg_, Tm, cv2.TM_CCOEFF_NORMED, mask=Mm)
                        ncc = np.nan_to_num(ncc, nan=-1.0, posinf=-1.0, neginf=-1.0)
                        py_, px_ = np.unravel_index(int(ncc.argmax()), ncc.shape); pk_ = float(ncc[py_, px_])
                        if pk_ < a.mwt_ncc: continue
                        yy_, xx_ = np.mgrid[0:ncc.shape[0], 0:ncc.shape[1]]
                        sec_ = float(ncc[(yy_ - py_) ** 2 + (xx_ - px_) ** 2 > 4].max()) if ncc.size > 25 else -1.0
                        if pk_ - sec_ < a.mwt_gap: continue
                        fx_, fy_ = float(px_), float(py_)
                        if 0 < px_ < ncc.shape[1] - 1:
                            l_, c0_, r0_ = ncc[py_, px_ - 1], ncc[py_, px_], ncc[py_, px_ + 1]; den = l_ - 2 * c0_ + r0_
                            if den < 0: fx_ += 0.5 * (l_ - r0_) / den
                        if 0 < py_ < ncc.shape[0] - 1:
                            l_, c0_, r0_ = ncc[py_ - 1, px_], ncc[py_, px_], ncc[py_ + 1, px_]; den = l_ - 2 * c0_ + r0_
                            if den < 0: fy_ += 0.5 * (l_ - r0_) / den
                        u_obs = np.array([vx + fx_ - r_, vy + fy_ - r_]) + (v0 - [vx, vy])
                        obs_k.append(k); obs_c.append(ci); obs_u.append(u_obs); obs_pk.append(pk_); obs_sec.append(sec_)
                        obs_W.append(np.sqrt(a.mwt_w) * np.eye(2)); nm += 1
                    n_mwt[ci] = nm
                if a.obs == "mklt": continue
            if ext is not None:
                trc, vic = ext[c]
                if t < len(trc):
                    for k in np.where(vic[t] & ~np.isnan(trc[t][:, 0]))[0]:
                        u_ = trc[t][k]
                        if 0 <= u_[0] < W and 0 <= u_[1] < H:
                            obs_k.append(k); obs_c.append(ci); obs_u.append(u_); obs_pk.append(0.0); obs_sec.append(0.0); obs_W.append(None)
                if a.obs == "tracks": continue
            uv, z = an.project(ci, Rp, tp)
            vis = (z > 0.05) & (uv[:, 0] > -radius) & (uv[:, 0] < W + radius) & (uv[:, 1] > -radius) & (uv[:, 1] < H + radius)
            if glob:
                box = v2_box(c, t, a.global_margin)
                if box is None: continue
                idx = np.arange(N); scale = s_b[c]; D, geo = dn.map(gray[c][t], box, scale)
                pos, pk, sec = dn.search(D, geo, [refs[k] for k in idx], None, None, a.excl_px)
            else:
                idx = np.where(vis)[0]
                if len(idx) == 0: continue
                box = box_of(uv[idx], radius + 8)
                if box is None: continue
                scale = float(np.clip(s_b[c] * np.median(z[idx]) / zb[c], 0.5 * s_b[c], 2 * s_b[c]))
                D, geo = dn.map(gray[c][t], box, scale)
                pos, pk, sec = dn.search(D, geo, [refs[k] for k in idx], uv[idx], radius, a.excl_px)
            ok = (pk >= a.sim_min) & (pk - sec >= a.dist_min) & ~np.isnan(pos[:, 0])
            ok &= (pos[:, 0] >= 0) & (pos[:, 0] < W) & (pos[:, 1] >= 0) & (pos[:, 1] < H)
            for j in np.where(ok)[0]:
                obs_k.append(idx[j]); obs_c.append(ci); obs_u.append(pos[j]); obs_pk.append(pk[j]); obs_sec.append(sec[j]); obs_W.append(None)
            maps[ci] = (D, geo)
        v2_obs = ([], [], [])
        if v2m is not None and t > b + a.v2_model:
            for ci in (0, 1):
                d_ = v2m[ci]
                if t >= len(d_["tr"]): continue
                okv = ~np.isnan(d_["X"][:, 0]) & d_["vis"][t] & ~np.isnan(d_["tr"][t][:, 0])
                for k_ in np.where(okv)[0]:
                    u_ = d_["tr"][t, k_]
                    if 0 <= u_[0] < W and 0 <= u_[1] < H: v2_obs[0].append((ci, k_)); v2_obs[1].append(ci); v2_obs[2].append(u_)
        if v2_obs[0]:
            # temporary model: LK model points + the v2 points observed now (indices continue after len(an.X))
            Xv = np.array([v2m[ci_]["X"][k_] for ci_, k_ in v2_obs[0]]); base_X = an.X.copy()
            an.X = np.vstack([base_X, Xv]); kv = np.arange(len(base_X), len(an.X))
            obs_k = list(obs_k) + kv.tolist(); obs_c = list(obs_c) + v2_obs[1]; obs_u = list(obs_u) + v2_obs[2]; obs_W = list(obs_W) + [np.sqrt(a.v2_w) * np.eye(2)] * len(kv); obs_pk = list(obs_pk) + [0.0] * len(kv); obs_sec = list(obs_sec) + [0.0] * len(kv)
        else:
            base_X = None
        obs = (np.array(obs_k, int), np.array(obs_c, int), np.array(obs_u, np.float64).reshape(-1, 2))
        if len(obs_W) == len(obs[0]) and any(w_ is not None for w_ in obs_W):
            obs = obs + (np.array([np.eye(2) if w_ is None else w_ for w_ in obs_W]),)
        if a.grow_frames > 0 and len(pend["X"]):
            for ci, c in enumerate(("ext1", "ext2")):
                idx = np.where(pend["alive"])[0]
                if not len(idx): break
                p0 = pend["tr"][ci][idx, t - 1].reshape(-1, 1, 2)
                lk_ = dict(winSize=(lkw[c],) * 2, maxLevel=a.lk_levels, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
                pn, st, _ = cv2.calcOpticalFlowPyrLK(gray_lk[c][t - 1], gray_lk[c][t], p0, None, **lk_)
                pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray_lk[c][t], gray_lk[c][t - 1], pn, None, **lk_)
                fb = np.linalg.norm(pb - p0, axis=2).ravel(); pn = pn.reshape(-1, 2)
                good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < a.lk_fb) & (pn[:, 0] >= 0) & (pn[:, 0] < W) & (pn[:, 1] >= 0) & (pn[:, 1] < H)
                pend["tr"][ci][idx[good], t] = pn[good]; pend["alive"][idx[~good]] = False
        if v2m is not None and t == b + a.v2_model:
            for ci, c in enumerate(("ext1", "ext2")):
                Kinv = np.linalg.inv(an.Kc[ci]); Cw = -an.Rc[ci].T @ an.tc[ci]; d_ = v2m[ci]; n_ok = 0
                for k_ in range(len(d_["X"])):
                    fr_ = [f_ for f_ in range(b, t + 1) if solved.get(f_) and f_ < len(d_["tr"]) and d_["vis"][f_, k_] and not np.isnan(d_["tr"][f_, k_, 0])]
                    if len(fr_) < a.v2_min_obs: continue
                    P_, D_ = [], []
                    for f_ in fr_:
                        Rm, tm = R_s[f_], t_s[f_]; dw = an.Rc[ci].T @ (Kinv @ np.r_[d_["tr"][f_, k_], 1.0]); dw /= np.linalg.norm(dw)
                        P_.append(Rm.T @ (Cw - tm)); D_.append(Rm.T @ dw)
                    P_, D_ = np.array(P_), np.array(D_)
                    ang = np.degrees(np.arccos(np.clip(np.min(D_ @ D_.T), -1, 1)))
                    if ang < a.v2_min_parallax: continue
                    A_ = np.zeros((3, 3)); bb = np.zeros(3)
                    for pp, dd in zip(P_, D_): Pm = np.eye(3) - np.outer(dd, dd); A_ += Pm; bb += Pm @ pp
                    if np.linalg.eigvalsh(A_)[0] < 1e-6: continue
                    Xo = np.linalg.solve(A_, bb)
                    if np.linalg.norm(Xo - an.X.mean(0)) > a.v2_max_dist: continue
                    rp = []
                    for f_ in fr_:
                        Zc = an.Rc[ci] @ (R_s[f_] @ Xo + t_s[f_]) + an.tc[ci]
                        if Zc[2] <= 0.05: rp.append(1e9); continue
                        rp.append(np.linalg.norm((an.Kc[ci] @ (Zc / Zc[2]))[:2] - d_["tr"][f_, k_]))
                    if np.median(rp) < a.v2_px and np.percentile(rp, 90) < 2.5 * a.v2_px: d_["X"][k_] = Xo; n_ok += 1
                d_["done"] = True; n_v2 += n_ok
        S_t = clouds[t][0] if (clouds is not None and t in clouds) else None
        st_arg = (S_t, a.stereo_w, a.stereo_assoc * 1e-3, a.stereo_iters) if (S_t is not None and len(S_t) >= a.stereo_min and a.stereo_w > 0) else None
        row = dict(frame=t, fs=fs, kf=kf, n_st=0 if S_t is None else len(S_t), n_st_in=0, st_rms_mm=np.nan, n_model=len(an.X), n_mwt1=n_mwt[0], n_mwt2=n_mwt[1], mode="global" if glob else "guided", radius=radius, n_obs=len(obs[0]),
                   n_obs1=int((obs[1] == 0).sum()), n_obs2=int((obs[1] == 1).sum()), hyp="", n_inl=0, n_inl1=0, n_inl2=0,
                   reproj_px=np.nan, spread_cm=np.nan, solved=False, dev_pred_deg=np.nan)
        cv_pred = (fs == 1 and (t - 2) >= b and solved.get(t - 2, False)) and not (Rft is not None and t in Rft)
        if len(obs[0]) >= a.min_inl or st_arg is not None:
            if Rft is not None and t in Rft:
                cen_ = an.X.mean(0); cands = [("pred_ft", Rp, tp), ("hold_ft", Rp, (R_s[last] @ cen_ + t_s[last]) - Rp @ cen_)]
                for nm_, Rh_, th_ in (an.hypotheses(obs) if len(obs[0]) >= 6 else []):
                    cands.append((nm_ + "_ft", Rp, (Rh_ @ cen_ + th_) - Rp @ cen_))
                if c_hull_b is not None:
                    ch_ = hull3d(t)
                    if ch_ is not None: cands.append(("hull_ft", Rp, ch_ - Rp @ c_hull_b))
                if len(obs[0]) >= 3:   # translation-only robust fit from the prediction with a wide gate (fast gripper, zero initial velocity)
                    Rw_, tw_ = an.refine_t(Rp, tp, tuple(o[an.resid(Rp, tp, obs) < 4 * a.tau_out] for o in obs)) if (an.resid(Rp, tp, obs) < 4 * a.tau_out).sum() >= 3 else (Rp, tp)
                    cands.append(("wide_ft", Rw_, tw_))
            else:
                cands = ([] if glob else [("pred", Rp, tp), ("hold", R_s[last], t_s[last])]) + (an.hypotheses(obs) if len(obs[0]) >= 6 else [])
            best = None
            for name, Rh, th in cands:
                n_in = int((an.resid(Rh, th, obs) < a.tau_in).sum()) if len(obs[0]) else 0
                if st_arg is not None: n_in += an.stereo_stats(Rh, th, S_t, a.stereo_assoc * 1e-3)[0] * a.stereo_vote
                if best is None or n_in > best[0]: best = (n_in, name, Rh, th)
            if best is not None and (best[0] >= 3 or st_arg is not None):
                _, name, Rh, th = best
                e0 = an.resid(Rh, th, obs) if len(obs[0]) else np.zeros(0); gate = e0 < a.tau_out
                sub = tuple(o[gate] for o in obs)
                prior = (Rp, tp, a.prior_deg, a.prior_mm) if (a.prior_deg > 0 and not glob and cv_pred) else None
                Rr, tr_ = an.refine_t(Rh, th, sub) if (Rft is not None and t in Rft) else an.refine(Rh, th, sub, prior, st_arg)
                e1 = an.resid(Rr, tr_, obs) if len(obs[0]) else np.zeros(0); inl = e1 < a.tau_in
                kin = np.unique(obs[0][inl]) if len(obs[0]) else np.zeros(0, int); sp = an.spread(kin)
                n_st_in, st_rms, st_med = an.stereo_stats(Rr, tr_, S_t, a.stereo_assoc * 1e-3) if st_arg is not None else (0, np.nan, np.nan)
                ok_lk = inl.sum() >= a.min_inl and sp >= a.min_spread and ((inl & (obs[1] == 0)).sum() >= a.min_view and (inl & (obs[1] == 1)).sum() >= a.min_view)
                ok_st = st_arg is not None and n_st_in >= a.stereo_min and (np.isnan(st_rms) or st_rms <= a.stereo_ok_mm)
                ok = ok_lk or ok_st
                if not ok and a.rot_ft_fallback and Rft is not None and t in Rft and len(obs[0]) >= 3:
                    # translation-only solve did not pass: hold the fused rotation, re-solve translation from the last pose with a
                    # looser gate; if that fails too, hold the whole pose (coverage over precision for the consumer)
                    Rh2, th2 = Rp, t_s[last] + (Rp - R_s[last]) @ (-an.X.mean(0)) + (R_s[last] @ an.X.mean(0) + t_s[last]) - (R_s[last] @ an.X.mean(0) + t_s[last])
                    e2 = an.resid(Rh2, th2, obs); g2 = e2 < 2 * a.tau_out
                    if g2.sum() >= 3:
                        Rr, tr_ = an.refine_t(Rh2, th2, tuple(o[g2] for o in obs)); e1 = an.resid(Rr, tr_, obs); inl = e1 < 2 * a.tau_in
                        ok = inl.sum() >= 3; name = name + "+fb"
                    if not ok:
                        Rr, tr_ = Rp, tp; ok = True; name = name + "+hold"; inl = np.zeros(len(obs[0]), bool)
                dev = float(rt.rot_angle_deg(Rr @ Rp.T))
                if ok and not glob and cv_pred and a.jump_deg > 0 and dev > a.jump_deg: ok = False
                row.update(n_st_in=n_st_in, st_rms_mm=st_rms)
                row.update(hyp=name, n_inl=int(inl.sum()), n_inl1=int((inl & (obs[1] == 0)).sum()), n_inl2=int((inl & (obs[1] == 1)).sum()),
                           reproj_px=float(np.median(e1[inl])) if inl.any() else np.nan, spread_cm=sp * 100, solved=bool(ok), dev_pred_deg=dev)
                if ok and Rft is not None and t in Rft and base_X is not None and len(kv) >= a.v2_min_pull:
                    # absolute translation from the Track-On v2 observations alone (their 3D was motion-triangulated early)
                    mv = np.isin(obs[0], kv); subv = tuple(o[mv] for o in obs)
                    _, tv2 = an.refine_t(Rr, tr_, subv); ev2 = an.resid(Rr, tv2, obs)[mv]
                    if (ev2 < 2 * a.tau_in).sum() >= a.v2_min_pull:
                        cen0 = an.X[:len(base_X)].mean(0); c_lk = Rr @ cen0 + tr_; c_v2 = Rr @ cen0 + tv2
                        row.update(c_v2_x=c_v2[0], c_v2_y=c_v2[1], c_v2_z=c_v2[2], n_v2_in=int((ev2 < 2 * a.tau_in).sum()))
                        if a.v2_pull > 0 and np.linalg.norm(c_v2 - c_lk) < a.pull_gate * 1e-2: tr_ = (c_lk + a.v2_pull * (c_v2 - c_lk)) - Rr @ cen0
                if ok and c_hull_b is not None:
                    ch_ = hull3d(t)
                    if ch_ is not None:
                        cen0 = an.X[:len(base_X)].mean(0) if base_X is not None else an.X.mean(0); c_h = ch_ + Rr @ (cen0 - c_hull_b)
                        row.update(c_hull_x=c_h[0], c_hull_y=c_h[1], c_hull_z=c_h[2])
                        if a.hull_pull > 0:
                            c_lk = Rr @ cen0 + tr_
                            if np.linalg.norm(c_h - c_lk) < a.pull_gate * 1e-2: tr_ = (c_lk + a.hull_pull * (c_h - c_lk)) - Rr @ cen0
                if ok and Rft is not None and t in Rft and a.rot_ft_tbeta < 1.0 and fs == 1 and "+hold" not in name:
                    cen0 = an.X.mean(0); cm_ = Rr @ cen0 + tr_; cp_ = tp + Rp @ cen0        # tp was built as c_last + v_c - Rp cen0
                    cf_ = cp_ + a.rot_ft_tbeta * (cm_ - cp_); tr_ = cf_ - Rr @ cen0
                if ok and cv_pred and not glob and (a.fuse_rot < 1.0 or a.fuse_pos < 1.0):
                    # causal alpha-beta filter: blend the measured pose with the constant-velocity prediction
                    cen0 = an.X.mean(0)
                    dw = Rotation.from_matrix(Rr @ Rp.T).as_rotvec()
                    Rf = ortho(Rotation.from_rotvec(a.fuse_rot * dw).as_matrix() @ Rp)
                    cm, cp = Rr @ cen0 + tr_, Rp @ cen0 + tp
                    cf = cp + a.fuse_pos * (cm - cp)
                    Rr, tr_ = Rf, cf - Rf @ cen0
                if ok and a.retire > 0 and len(obs[0]):
                    bad_now = np.zeros(len(an.X), bool); seen = np.zeros(len(an.X), bool)
                    seen[obs[0]] = True; bad_now[obs[0][e1 > a.tau_out]] = True
                    if bad_cnt is None or len(bad_cnt) != len(an.X): bad_cnt = np.zeros(len(an.X), int)
                    bad_cnt = np.where(bad_now, bad_cnt + 1, np.where(seen, 0, bad_cnt))
                    drop = bad_cnt >= a.retire
                    if drop.any() and (len(an.X) - drop.sum()) >= a.min_seeds:
                        an.set_model(an.X[~drop]); bad_cnt = bad_cnt[~drop]; n_retired += int(drop.sum())
                if ok:
                    blind = 0
                    if fs == 1:                                   # velocity update from the accepted increment
                        cen0 = an.X.mean(0); w_ = Rotation.from_matrix(Rr @ R_s[t - 1].T).as_rotvec()
                        d_ = (Rr @ cen0 + tr_) - (R_s[t - 1] @ cen0 + t_s[t - 1])
                        vel = (w_, d_) if vel is None else ((1 - a.vel_beta) * vel[0] + a.vel_beta * w_, (1 - a.vel_beta) * vel[1] + a.vel_beta * d_)
                    else:
                        vel = None
                    R_s[t], t_s[t], solved[t] = Rr, tr_, True; last = t
                    if S_t is not None and len(S_t) and a.acc_frames > 0 and ((t - b) <= a.acc_frames or (a.acc_rms > 0 and not np.isnan(st_rms) and st_rms <= a.acc_rms)) and len(an.X) < a.model_max:
                        j, d, Y = an.assoc(Rr, tr_, S_t, a.acc_near * 1e-3)
                        new_pts = Y[(j >= 0) & (d > a.acc_dup * 1e-3)]
                        if len(new_pts):
                            keep = voxel_keep(new_pts, a.model_vox * 1e-3, a.model_max - len(an.X))
                            an.set_model(np.vstack([an.X, new_pts[keep]])); n_acc += len(keep)
                    if a.kf or a.kf_multi:
                        n_u = len({(int(k_), int(c_)) for k_, c_ in zip(obs[0][inl], obs[1][inl])})
                        if kf_inl is None: kf_inl = n_u
                        if (t - kf >= a.kf_gap) or (n_u < a.kf_frac * kf_inl):
                            kf = t; kf_inl = n_u; n_kf += 1; kfs.append(t)
                    if a.mem > 0 and dn is not None:                                          # appearance memory (no geometry is added)
                        pk_arr = np.array(obs_pk)
                        for c_, (Dm, gm) in maps.items():
                            sel = np.where(inl & (obs[1] == c_) & (pk_arr >= a.mem_sim) & (e1 < a.mem_px))[0]
                            if not len(sel): continue
                            dd = dn.sample(Dm, gm, obs[2][sel])
                            for j, kk in enumerate(obs[0][sel]):
                                r = refs[kk]
                                if float((r @ dd[j]).max()) < a.mem_novel:
                                    keep = r if len(r) - n_birth_refs < a.mem else torch.cat([r[:n_birth_refs], r[n_birth_refs + 1:]])
                                    refs[kk] = torch.cat([keep, dd[j][None]])
        if base_X is not None: an.X = base_X   # drop the temporary v2 rows (model tree/normals unchanged)
        if not solved.get(t) and a.coast >= 0 and Rft is not None and t in Rft:
            cen_ = an.X.mean(0); c_last_ = R_s[last] @ cen_ + t_s[last]
            steps = t - last
            v_ = (c_last_ - (R_s[last - 1] @ cen_ + t_s[last - 1])) if (solved.get(last - 1) and last - 1 >= b and (last - 1) in R_s) else np.zeros(3)
            ch_ = hull3d(t) if (c_hull_b is not None and a.hull_coast) else None
            if ch_ is not None:
                t_new = ch_ - Rp @ c_hull_b; tag_ = "+coast_hull"; blind = 0
            else:
                c_new = c_last_ + v_ * min(steps, a.coast); t_new = c_new - Rp @ cen_; tag_ = "+coast_hold"; blind = blind + 1
            if blind <= a.coast_emit:
                R_s[t], t_s[t], solved[t] = Rp, t_new, True; row.update(hyp=row.get("hyp", "") + tag_, solved=True); n_coast += 1
            # the tracker keeps anchoring on the last REAL solve: do not move `last`
        if not solved.get(t):
            solved[t] = False
        if a.grow_frames > 0 and solved.get(t):
            # (1) candidates whose chains are due: consistency with the SOLVED poses over their window, both views
            due = np.where(t - pend["t0"] >= a.grow_frames)[0]
            if len(due):
                keep_rows = np.ones(len(pend["X"]), bool); add = []
                for i in due:
                    keep_rows[i] = False; okc = True
                    for ci in (0, 1):
                        fs_ = [f for f in range(pend["t0"][i] + 1, t + 1) if solved.get(f) and not np.isnan(pend["tr"][ci][i, f, 0])]
                        if len(fs_) < a.grow_frac * a.grow_frames: okc = False; break
                        Y = np.array([R_s[f] @ pend["X"][i] + t_s[f] for f in fs_]); Z = Y @ an.Rc[ci].T + an.tc[ci]; z = np.maximum(Z[:, 2], 1e-4)
                        uv = Z[:, :2] / z[:, None] * [an.Kc[ci][0, 0], an.Kc[ci][1, 1]] + [an.Kc[ci][0, 2], an.Kc[ci][1, 2]]
                        e_ = np.linalg.norm(uv - pend["tr"][ci][i, fs_], axis=1)
                        if np.median(e_) > a.grow_px or np.percentile(e_, 90) > 2.5 * a.grow_px: okc = False; break
                    if okc: add.append(pend["X"][i])
                if add:
                    add = np.array(add); j_, d_, _ = an.assoc(np.eye(3), np.zeros(3), add, a.acc_dup * 1e-3) if len(an.X) else (np.full(len(add), -1), None, None)
                    add = add[j_ < 0]
                    if len(add) and len(an.X) + len(add) <= a.model_max: an.set_model(np.vstack([an.X, add])); n_grow_ok += len(add)
                pend = dict(X=pend["X"][keep_rows], t0=pend["t0"][keep_rows], tr={ci: pend["tr"][ci][keep_rows] for ci in (0, 1)}, alive=pend["alive"][keep_rows])
            # (2) fresh candidates from this frame's cloud (reciprocal matches), expressed in the model frame with the solved pose
            if clouds is not None and t in clouds and (t - b) % a.grow_every == 0 and len(pend["X"]) < a.grow_pending_max:
                Sw, s1, s2 = clouds[t]; srcs = clouds_src[t] if clouds_src is not None else np.ones(len(Sw), np.int8)
                selc = np.isin(srcs, [1, 3]) if a.grow_src == "recip" else np.ones(len(Sw), bool)
                if selc.any():
                    Xm = (Sw[selc] - t_s[t]) @ R_s[t]
                    near_model = an.assoc(np.eye(3), np.zeros(3), Xm, a.acc_dup * 1e-3)[0] >= 0 if len(an.X) else np.zeros(len(Xm), bool)
                    far = np.linalg.norm(Xm - an.X.mean(0), axis=1) > a.acc_near * 1e-3 * 10 if len(an.X) else np.zeros(len(Xm), bool)
                    pick = np.where(~near_model & ~far)[0][: a.grow_max]
                    if len(pick):
                        n_grow_cand += len(pick); m_ = len(pick)
                        trn = {ci: np.full((m_, T, 2), np.nan, np.float32) for ci in (0, 1)}
                        trn[0][:, t] = s1[selc][pick]; trn[1][:, t] = s2[selc][pick]
                        pend = dict(X=np.vstack([pend["X"], Xm[pick]]), t0=np.concatenate([pend["t0"], np.full(m_, t, int)]),
                                    tr={ci: np.concatenate([pend["tr"][ci], trn[ci]]) for ci in (0, 1)}, alive=np.concatenate([pend["alive"], np.ones(m_, bool)]))
        if ext_tracks and t == b + a.extend:
            newX = []
            for ci_, c in enumerate(("ext1", "ext2")):
                if c not in ext_tracks: continue
                Kinv = np.linalg.inv(an.Kc[ci_]); Cw = -an.Rc[ci_].T @ an.tc[ci_]
                for tr1 in ext_tracks[c]:
                    fr_ = [j for j in range(a.extend + 1) if not np.isnan(tr1[j, 0]) and solved.get(b + j)]
                    if len(fr_) < a.extend_min_obs: continue
                    P_, D_ = [], []
                    for j in fr_:
                        Rm, tm = R_s[b + j], t_s[b + j]
                        dw = an.Rc[ci_].T @ (Kinv @ np.r_[tr1[j], 1.0]); dw /= np.linalg.norm(dw)
                        P_.append(Rm.T @ (Cw - tm)); D_.append(Rm.T @ dw)
                    P_, D_ = np.array(P_), np.array(D_)
                    A_ = np.zeros((3, 3)); bb = np.zeros(3)
                    for pp, dd in zip(P_, D_):
                        Pm = np.eye(3) - np.outer(dd, dd); A_ += Pm; bb += Pm @ pp
                    ev = np.linalg.eigvalsh(A_)
                    ang = np.degrees(np.arccos(np.clip(np.min(D_ @ D_.T), -1, 1)))
                    if ang < a.extend_min_parallax or ev[0] < 1e-6: continue
                    Xo = np.linalg.solve(A_, bb)
                    if np.linalg.norm(Xo - an.X.mean(0)) > a.extend_max_dist: continue
                    rp = []
                    for j in fr_:
                        Zc = an.Rc[ci_] @ (R_s[b + j] @ Xo + t_s[b + j]) + an.tc[ci_]
                        if Zc[2] <= 0.05: rp.append(1e9); continue
                        uvp_ = (an.Kc[ci_] @ (Zc / Zc[2]))[:2]; rp.append(np.linalg.norm(uvp_ - tr1[j]))
                    if np.median(rp) < a.extend_px and np.max(rp) < 2.5 * a.extend_px: newX.append(Xo)
            if newX:
                an.set_model(np.vstack([an.X, np.array(newX)])); n_extended = len(newX)
        if solved.get(t):
            cen_e = an.X.mean(0); c_e = R_s[t] @ cen_e + t_s[t]; row.update(c_x=c_e[0], c_y=c_e[1], c_z=c_e[2])
        rows.append(row)
    if a.out_tbeta < 1.0 or a.out_rbeta < 1.0:   # causal output smoothing: state = previous OUTPUTS, prediction = constant velocity
        cen0 = an.X[:N].mean(0) if len(an.X) >= N else an.X.mean(0); Ro, to = {}, {}; w_f = None; v_f = None; prev = None
        for t_ in sorted(k_ for k_ in R_s if solved.get(k_)):
            Rm_, tm_ = R_s[t_], t_s[t_]; cm_ = Rm_ @ cen0 + tm_
            if prev is None or t_ - prev != 1:
                Ro[t_], to[t_] = Rm_, tm_; w_f = None; v_f = None; prev = t_; continue
            w_meas = Rotation.from_matrix(Rm_ @ Ro[prev].T).as_rotvec(); w_f = w_meas if w_f is None else (1 - a.out_rbeta) * w_f + a.out_rbeta * w_meas
            R_new = ortho(Rotation.from_rotvec(w_f).as_matrix() @ Ro[prev])
            if a.out_rpull > 0: R_new = ortho(Rotation.from_rotvec(a.out_rpull * Rotation.from_matrix(Rm_ @ R_new.T).as_rotvec()).as_matrix() @ R_new)
            c_prev = Ro[prev] @ cen0 + to[prev]; c_pred = c_prev + (v_f if v_f is not None else np.zeros(3))
            c_new = c_pred + a.out_tbeta * (cm_ - c_pred); v_f = c_new - c_prev
            Ro[t_], to[t_] = R_new, c_new - R_new @ cen0; prev = t_
        R_s.update(Ro); t_s.update(to)
    rec.update(K_src=ksrc, T=T, birth=b, exit=e, window_end=t_end, n_seeds=N, n_keyframes=n_kf, n_extended=n_extended, seed_mask=rec_seed_mask, stereo=rec_stereo, n_accumulated=n_acc, n_model_final=len(an.X), n_grow_cand=n_grow_cand, n_grow_ok=n_grow_ok, n_retired=n_retired, n_coast=n_coast, n_v2=n_v2, cand_src=np.bincount(cand_src, minlength=4).tolist(), f2=({k_: f2[k_] for k_ in ('checks', 'n_in', 'done', 'kept_x1', 'kept_src') if k_ in f2} if f2 is not None else None), elapsed_s=time.time() - t_start,
               refs_final_median=float(np.median([len(r) for r in refs])) if refs and len(refs[0]) else 0.0)
    return rec, rows, (R_s, t_s, solved, an.X[:N].mean(0)), b, t_end, store


def evaluate(rec, rows, sol, b, t_end, store):
    R_s, t_s, solved = sol[:3]
    gt = rt.load_poses(store, t_end + 1, "store GT")
    try: cut = rt.load_poses(f"{rt.CUT3R_ROOT}/{rec['episode']}/camera", t_end + 1, "CUT3R")
    except rt.Failure: cut = None
    P0 = gt[b]; frames = [t for t in range(b, t_end + 1) if solved.get(t) and t in gt]
    pred = {t: se3(R_s[t], t_s[t]) @ P0 for t in frames}
    byf = {r["frame"]: r for r in rows}
    cen0 = sol[3] if len(sol) > 3 else None
    if cen0 is not None:
        for r in rows:
            t = r["frame"]
            if t not in gt: continue
            Mg = gt[t] @ np.linalg.inv(gt[b]); c_true = Mg[:3, :3] @ cen0 + Mg[:3, 3]
            for key in ("c", "c_v2", "c_hull"):
                if f"{key}_x" in r and r[f"{key}_x"] == r[f"{key}_x"]:
                    r[f"{key}_err_cm"] = float(np.linalg.norm(np.array([r[f"{key}_x"], r[f"{key}_y"], r[f"{key}_z"]]) - c_true) * 100)
    for t in range(b + 1, t_end + 1):
        r = byf.get(t)
        if r is None: continue
        if t in pred:
            r["pos_err_cm"] = float(np.linalg.norm(pred[t][:3, 3] - gt[t][:3, 3]) * 100)
            r["rot_err_deg"] = float(rt.rot_angle_deg(pred[t][:3, :3].T @ gt[t][:3, :3]))
        else:
            r["pos_err_cm"] = r["rot_err_deg"] = np.nan
    fr = frames[1:] if len(frames) > 1 else []                              # b itself is exact by construction
    w = dict(window=[b, t_end], window_len=t_end - b + 1, t_ref=b, n_anchored=len(frames), ref_points=rec["n_seeds"])
    if len(frames) >= 2:
        pe = np.array([byf[t]["pos_err_cm"] for t in fr]) if fr else np.array([0.0]); re_ = np.array([byf[t]["rot_err_deg"] for t in fr]) if fr else np.array([0.0])
        m = rt.traj_metrics([pred[t] for t in frames], [gt[t] for t in frames])
        w.update(pos_err_median_cm=float(np.median(pe)), pos_err_p90_cm=float(np.percentile(pe, 90)), pos_err_max_cm=float(pe.max()),
                 rot_err_median_deg=float(np.median(re_)), rot_err_p90_deg=float(np.percentile(re_, 90)), rot_err_max_deg=float(re_.max()),
                 inliers_median=float(np.median([byf[t]["n_inl"] for t in fr])) if fr else np.nan,
                 rig_ate=m["ate"], rig_rpe_t=m["rpe_trans"], rig_rpe_rot=m["rpe_rot"])
        if cut is not None and all(t in cut for t in frames):
            mk = rt.traj_metrics([cut[t] for t in frames], [gt[t] for t in frames])
            w.update(cut3r_ate_same_frames=mk["ate"], cut3r_rpe_t_same_frames=mk["rpe_trans"], cut3r_rpe_rot_same_frames=mk["rpe_rot"])
        inc_r, inc_t = [], []                                              # consecutive solved frames: increment error
        for t in frames[1:]:
            if t - 1 in pred:
                dp = np.linalg.inv(pred[t - 1]) @ pred[t]; dg = np.linalg.inv(gt[t - 1]) @ gt[t]; d = np.linalg.inv(dg) @ dp
                inc_r.append(rt.rot_angle_deg(d[:3, :3])); inc_t.append(np.linalg.norm(d[:3, 3]) * 100)
                byf[t]["inc_rot_err_deg"] = inc_r[-1]; byf[t]["inc_pos_err_cm"] = inc_t[-1]
        if inc_r:
            w.update(inc_rot_err_median_deg=float(np.median(inc_r)), inc_rot_err_mean_deg=float(np.mean(inc_r)),
                     inc_rot_err_p90_deg=float(np.percentile(inc_r, 90)), inc_pos_err_median_cm=float(np.median(inc_t)), n_inc=len(inc_r))
    rec["windows"] = [dict(w, episode=rec["episode"], tag=rec["tag"], window_idx=0, n_windows=1)]
    return pred


ARGS = [
    ("--episode", dict(required=True)), ("--out", dict(required=True)), ("--tag", dict(default="anchor")),
    ("--seeds", dict(default=SEEDS)), ("--tracks_v2", dict(default=V2)), ("--events", dict(default=EVENTS)),
    ("--obs", dict(default="dense", choices=["dense", "tracks", "both", "mklt", "mklt+dense", "mflow"], help="observation source: DISK dense search, external tracks of the seeds, both, model-based LK (templates re-centred on the model projection every frame), or model-based LK + dense anchors")),
    ("--obs_tracks", dict(default=None, help="--obs tracks|both: <ep>/{ext1,ext2}_f<b>.npz seeded on --seeds (first N columns)")),
    ("--window", dict(default="full", choices=["full", "fk"], help="full = birth frame to the episode end (store frames); fk = to the FK exit")),
    ("--stereo", dict(default=None, help="DIR of stereo_clouds.py outputs (<ep>.npz): per-frame cross-view 3D points")),
    ("--model", dict(default="seeds", choices=["seeds", "stereo", "both"], help="birth model source: guided DISK seeds, the stereo birth cloud, or both")),
    ("--model_vox", dict(type=float, default=2.0, help="model voxel size (mm) for de-duplication")), ("--model_max", dict(type=int, default=3000)),
    ("--stereo_sim", dict(type=float, default=0.5)), ("--stereo_epi", dict(type=float, default=2.5)),
    ("--stereo_w", dict(type=float, default=1.0, help="weight of the stereo point-to-plane residual (px per mm); 0 = stereo not used in the solve")),
    ("--stereo_assoc", dict(type=float, default=12.0, help="association radius (mm) stereo point -> model")), ("--stereo_iters", dict(type=int, default=2)),
    ("--stereo_min", dict(type=int, default=20, help="min associated stereo points for a stereo-only solve / ok")), ("--stereo_ok_mm", dict(type=float, default=6.0)),
    ("--stereo_vote", dict(type=float, default=0.2, help="hypothesis score: each associated stereo point counts this much (LK inlier = 1)")),
    ("--acc_frames", dict(type=int, default=20, help="accumulate stereo points into the model during the first K frames (0 = never)")),
    ("--acc_rms", dict(type=float, default=0.0, help="also accumulate whenever the stereo rms (mm) is below this (0 = off)")),
    ("--acc_near", dict(type=float, default=15.0, help="new points must lie within this (mm) of the current model")), ("--acc_dup", dict(type=float, default=2.0)),
    ("--cands", dict(default="seeds", choices=["seeds", "recip", "both", "guided_only", "ext_only"], help="birth candidates: guided DISK seeds, MASt3R reciprocal matches of the v2 cloud file (--stereo DIR), both, or every cloud match")),
    ("--cands_max", dict(type=int, default=120)),
    ("--cands_ext", dict(type=int, default=0, help="in-process DISK-guided matching at birth inside the RobotSeg mask dilated by this many px (0 = off); adds candidates on the mount/housing/clips outside the black body")),
    ("--cands_ext_side", dict(type=int, default=640)),
    ("--retire", dict(type=int, default=0, help="retire a model point after this many consecutive solved frames with residual > tau_out in a view where it was observed (0 = never)")),
    ("--f2_frames", dict(type=int, default=0, help="GT-free phantom filter: causal RANSAC over the plain two-view LK chains of all candidates, checked at K/3, 2K/3 and K frames after birth (0 = off)")),
    ("--f2_samples", dict(type=int, default=300)), ("--f2_k", dict(type=int, default=5)), ("--f2_px", dict(type=float, default=1.0)), ("--f2_frac", dict(type=float, default=0.8)),
    ("--grow_frames", dict(type=int, default=0, help="model growth: fresh cloud matches become candidates, LK-chained for K frames and admitted when consistent with the solved poses (0 = off)")),
    ("--grow_every", dict(type=int, default=4)), ("--grow_max", dict(type=int, default=30)), ("--grow_pending_max", dict(type=int, default=300)),
    ("--grow_px", dict(type=float, default=1.0)), ("--grow_frac", dict(type=float, default=0.7)), ("--grow_src", dict(default="recip", choices=["recip", "all"])),
    ("--model_labels", dict(default=None, choices=[None, "all", "base"], help="GT DIAGNOSTIC: the model = FK label points at b (positions_v1)")),
    ("--labels_vis", dict(action="store_true", help="GT DIAGNOSTIC with --model_labels: only label points visible (front, in frame) in that view at both frames are tracked")),
    ("--f1_m3d", dict(type=float, default=0.0, help="GT-free phantom filter on cloud matches: keep s*m3d <= this (m); 0 = off")), ("--f1_pconf", dict(type=float, default=2.0)),
    ("--min_seeds", dict(type=int, default=4)),
    ("--flow_model", dict(default="large", choices=["large", "small"])), ("--flow_iters", dict(type=int, default=12)),
    ("--flow_margin", dict(type=int, default=48)), ("--flow_fb", dict(type=float, default=1.0, help="mflow forward-backward consistency threshold (px)")),
    ("--mwt", dict(action="store_true", help="add absolute observations: each seed's BIRTH patch (RobotSeg-masked) warped by the homography the predicted motion induces on a fronto-parallel local plane, matched by masked NCC")),
    ("--mwt_r", dict(type=int, default=12)), ("--mwt_ncc", dict(type=float, default=0.7)), ("--mwt_gap", dict(type=float, default=0.03)),
    ("--mwt_minmask", dict(type=float, default=0.25)), ("--mwt_minstd", dict(type=float, default=4.0)), ("--mwt_w", dict(type=float, default=1.0)),
    ("--seed_mask", dict(action="store_true", help="drop birth seeds outside the (dilated) RobotSeg birth mask in either view")), ("--seed_mask_dil", dict(type=int, default=4)),
    ("--extend", dict(type=int, default=0, help="model-LK only: corners inside the eroded RobotSeg birth mask are LK-tracked over frames b..b+K and triangulated in the object frame with the solved motions; appended to the model at b+K (0 = off)")),
    ("--extend_max", dict(type=int, default=150)), ("--extend_min_obs", dict(type=int, default=6)), ("--extend_min_parallax", dict(type=float, default=2.0)),
    ("--extend_max_dist", dict(type=float, default=0.15)), ("--extend_px", dict(type=float, default=1.0)),
    ("--lk_win", dict(type=int, default=31, help="model-LK window (px) at the RAIL apparent size, scaled down for smaller grippers")), ("--lk_levels", dict(type=int, default=3)),
    ("--lk_fb", dict(type=float, default=1.0, help="model-LK forward-backward threshold (px)")),
    ("--lk_mineig", dict(type=float, default=1e-4, help="OpenCV minEigThreshold: drop LK features whose spatial-gradient matrix is too weak (texture-less)")),
    ("--lk_noinit", dict(action="store_true", help="model-LK starts at the previous position instead of the constant-velocity prediction")),
    ("--lk_clip", dict(type=int, default=0, help="saturate gray levels above this before model-LK (0 = off)")),
    ("--aperture", dict(action="store_true", help="whiten model-LK residuals by the template structure tensor (edge points constrain only their normal)")), ("--aperture_eps", dict(type=float, default=0.05)),
    ("--kf", dict(action="store_true", help="model-LK from the current KEYFRAME image (templates at the model projection under the keyframe pose) instead of the previous frame")),
    ("--kf_gap", dict(type=int, default=1000, help="new keyframe at most every N frames")), ("--kf_multi", dict(action="store_true", help="model-LK from the latest keyframe + birth + the keyframe closest in pose to the prediction, pooled")),
    ("--kf_cm_per_deg_inv", dict(type=float, default=2.0, help="keyframe pose distance = deg + this x cm")), ("--kf_frac", dict(type=float, default=0.7, help="new keyframe when inliers drop below this fraction of the keyframe's first inlier count")), ("--region_dilate", dict(type=int, default=8)), ("--side", dict(type=int, default=640)),
    ("--radius", dict(type=float, default=20.0, help="full-res px search radius around the predicted pixel")),
    ("--radius_grow", dict(type=float, default=0.5)), ("--radius_max", dict(type=float, default=60.0)),
    ("--lost_global", dict(type=int, default=8, help="unsolved frames before the search becomes global (Track-On v2 region)")),
    ("--global_margin", dict(type=int, default=30)),
    ("--excl_px", dict(type=float, default=4.0, help="full-res px around the peak excluded from the second-best")),
    ("--sim_min", dict(type=float, default=0.5)), ("--dist_min", dict(type=float, default=0.02)),
    ("--tau_in", dict(type=float, default=4.0)), ("--tau_out", dict(type=float, default=10.0)), ("--huber_px", dict(type=float, default=2.0)),
    ("--kabsch_m", dict(type=float, default=0.01)),
    ("--min_inl", dict(type=int, default=6)), ("--min_view", dict(type=int, default=0)), ("--min_spread", dict(type=float, default=0.01)),
    ("--prior_deg", dict(type=float, default=0.0, help="motion prior sigma (deg) on the deviation from constant velocity; 0 = off")),
    ("--prior_mm", dict(type=float, default=3.0)),
    ("--fuse_rot", dict(type=float, default=1.0, help="filter gain on the measured rotation vs the constant-velocity prediction (1 = no filtering)")),
    ("--fuse_pos", dict(type=float, default=1.0, help="filter gain on the measured model-centroid position")),
    ("--rot_ft", dict(action="store_true", help="HYBRID: rotation per frame from CUT3R finetuned (augfull_lr1e5 online preds, relative to b, via GT(b)); the rig solves translation only")),
    ("--rot_ft_beta", dict(type=float, default=1.0, help="causal smoothing of the CUT3R rotation increments (1 = raw)")),
    ("--rot_ft_tbeta", dict(type=float, default=1.0, help="with --rot_ft: gain on the measured centroid vs the constant-velocity prediction (1 = raw)")),
    ("--out_tbeta", dict(type=float, default=1.0, help="causal OUTPUT smoothing of the centroid (gain on the measurement vs constant velocity; the tracker state is untouched)")),
    ("--out_rbeta", dict(type=float, default=1.0, help="causal OUTPUT smoothing of the rotation increments")), ("--out_rpull", dict(type=float, default=0.0)),
    ("--rot_ft_pull", dict(type=float, default=0.0, help="per-frame pull of the smoothed rotation toward the absolute CUT3R rotation")),
    ("--rot_ft_fallback", dict(action="store_true", help="with --rot_ft: never leave a frame unsolved (loose re-solve, then hold)")),
    ("--coast", dict(type=int, default=-1, help="with --rot_ft: emit the fused rotation on unsolved frames with constant-velocity translation for this many frames, then hold the position (-1 = off)")),
    ("--hull_reloc", dict(action="store_true", help="with --rot_ft: Track-On v2 hull position (closest point of the two median rays) as a translation hypothesis every frame and as the coast position when LK is lost")),
    ("--v2_model", dict(type=int, default=0, help="Track-On v2 points as model points: each single-view track over the first K solved frames is triangulated in the object frame (motion parallax); afterwards their Track-On positions are observations (0 = off)")),
    ("--v2_min_obs", dict(type=int, default=8)), ("--v2_min_parallax", dict(type=float, default=1.5)), ("--v2_max_dist", dict(type=float, default=0.2)), ("--v2_px", dict(type=float, default=1.0)),
    ("--v2_w", dict(type=float, default=0.5, help="weight of a Track-On observation relative to a model-LK observation (variance ratio)")),
    ("--v2_pull", dict(type=float, default=0.0, help="per-frame pull of the LK centroid toward the Track-On-only translation (0 = log only)")), ("--v2_min_pull", dict(type=int, default=6)),
    ("--pull_gate", dict(type=float, default=10.0, help="apply a pull only when the cue is within this many cm of the LK centroid")),
    ("--hull_pull", dict(type=float, default=0.0, help="per-frame pull of the LK centroid toward the Track-On hull position (0 = log only)")),
    ("--hull_coast", dict(action="store_true", help="also use the hull position as the coast position (unconfirmed; off = hull only as a re-acquisition hypothesis)")),
    ("--coast_emit", dict(type=int, default=1000000, help="emit at most this many consecutive BLIND coast frames (no hull position); later blind frames are left unsolved for the consumer to mask")),
    ("--vel_beta", dict(type=float, default=1.0, help="velocity smoothing for the prediction: v <- (1-beta) v + beta * last increment (1 = last increment only)")),
    ("--jump_deg", dict(type=float, default=0.0, help="reject a solve deviating more than this from the prediction; 0 = off")),
    ("--mem", dict(type=int, default=6, help="appearance memory size per point (0 = birth descriptors only)")),
    ("--mem_sim", dict(type=float, default=0.7)), ("--mem_px", dict(type=float, default=2.0)), ("--mem_novel", dict(type=float, default=0.95)),
]


def main():
    ap = argparse.ArgumentParser()
    for name, kw in ARGS: ap.add_argument(name, **kw)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    try:
        rec, rows, sol, b, t_end, store = run(a)
        pred = evaluate(rec, rows, sol, b, t_end, store)
        if pred:
            fr = sorted(pred); R_s, t_s = sol[0], sol[1]
            np.savez(f"{a.out}/anchors.npz", frames=np.array(fr), poses=np.array([pred[t] for t in fr]),
                     motions=np.array([se3(R_s[t], t_s[t]) for t in fr]), model_centroid=sol[3] if len(sol) > 3 else np.zeros(3))
        if rows:
            keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
            with open(f"{a.out}/per_frame.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, keys); wr.writeheader(); wr.writerows(rows)
    except rt.Failure as ex:
        rec = dict(episode=a.episode, tag=a.tag, failure_reason=str(ex),
                   windows=[dict(episode=a.episode, tag=a.tag, window_idx=0, n_windows=1, skipped=str(ex))])
    rec["args"] = vars(a)
    json.dump(rec, open(f"{a.out}/record.json", "w"), indent=1, default=float)
    w = rec["windows"][0]
    print(json.dumps({k: w.get(k) for k in ("n_anchored", "window_len", "pos_err_median_cm", "rot_err_median_deg", "inc_rot_err_median_deg", "rig_ate", "skipped")}, default=float))


if __name__ == "__main__":
    main()
