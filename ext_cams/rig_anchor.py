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


# ----------------------------------------------------------------------------- the solver
class Anchor:
    def __init__(self, a, K, E, X_B):
        self.a = a; self.X = X_B
        self.Kc = np.array([K[c] for c in ("ext1", "ext2")]); self.Ec = np.array([E[c] for c in ("ext1", "ext2")])
        self.Rc = self.Ec[:, :3, :3]; self.tc = self.Ec[:, :3, 3]
        self.P = np.array([self.Kc[i] @ self.Ec[i][:3] for i in range(2)])

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

    def refine(self, R0, t0, obs, prior=None):
        """Huber LS of the 6-DoF motion on all given observations (both views) around (R0, t0); optional motion prior =
        (R_pred, t_pred, sigma_deg, sigma_mm) on the deviation from the prediction (model centroid for translation)."""
        k, ci, u = obs[:3]; Lw = obs[3] if len(obs) > 3 else None; a = self.a; cen = self.X.mean(0)
        def fun(x):
            Rd = Rotation.from_rotvec(x[:3]).as_matrix(); R = Rd @ R0; t = Rd @ t0 + x[3:]
            r = []
            for c in (0, 1):
                m = ci == c
                if m.any():
                    uv, _ = self.project(c, R, t, k[m]); d = uv - u[m]
                    if Lw is not None: d = np.einsum("nij,nj->ni", Lw[m], d)
                    r.append(d.ravel())
            if prior is not None:
                Rp, tp, sd, sm = prior
                r.append(np.degrees(Rotation.from_matrix(R @ Rp.T).as_rotvec()) / sd * a.huber_px)
                r.append(((R @ cen + t) - (Rp @ cen + tp)) * 1000 / sm * a.huber_px)
            return np.concatenate(r)
        res = least_squares(fun, np.zeros(6), method="trf", loss="huber", f_scale=a.huber_px, x_scale=1.0, max_nfev=50)
        Rd = Rotation.from_rotvec(res.x[:3]).as_matrix()
        return ortho(Rd @ R0), Rd @ t0 + res.x[3:]

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
    T = min(e, n_store)
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
    Rft = None
    if a.rot_ft:
        ftp = rt.load_poses(f"{rt.CUT3R_ROOT}/{ep}/camera", t_end + 1, "CUT3R")
        Rgb = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", b + 1, "store GT")[b][:3, :3]   # birth frame only
        Rft = {t_: ortho(Rgb @ ftp[b][:3, :3].T @ ftp[t_][:3, :3] @ Rgb.T) for t_ in ftp if t_ >= b}
    Gb = {c: gray[c][b].astype(np.float32) for c in ("ext1", "ext2")}; Mb = {c: bmask[c].astype(np.float32) for c in bmask}
    R_s, t_s, solved = {b: np.eye(3)}, {b: np.zeros(3)}, {b: True}
    last = b; rows = []; kf = b; kf_inl = None; n_kf = 0; kfs = [b]; vel = None
    for t in range(b + 1, t_end + 1):
        fs = t - last
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
            Rp = Rft[t]; tp = c_last + v_c - Rp @ cen0
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
                  idx = np.where(okp)[0]
                  if len(idx):
                      p0 = uvp[idx].astype(np.float32).reshape(-1, 1, 2); p1 = uvn[idx].astype(np.float32).reshape(-1, 1, 2)
                      lk = dict(winSize=(lkw[c],) * 2, maxLevel=a.lk_levels, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
                      pn, st, err = cv2.calcOpticalFlowPyrLK(gray[c][src], gray[c][t], p0, p1.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
                      pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray[c][t], gray[c][src], pn, p0.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
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
        obs = (np.array(obs_k, int), np.array(obs_c, int), np.array(obs_u, np.float64).reshape(-1, 2))
        if len(obs_W) == len(obs[0]) and any(w_ is not None for w_ in obs_W):
            obs = obs + (np.array([np.eye(2) if w_ is None else w_ for w_ in obs_W]),)
        row = dict(frame=t, fs=fs, kf=kf, n_mwt1=n_mwt[0], n_mwt2=n_mwt[1], mode="global" if glob else "guided", radius=radius, n_obs=len(obs[0]),
                   n_obs1=int((obs[1] == 0).sum()), n_obs2=int((obs[1] == 1).sum()), hyp="", n_inl=0, n_inl1=0, n_inl2=0,
                   reproj_px=np.nan, spread_cm=np.nan, solved=False, dev_pred_deg=np.nan)
        if len(obs[0]) >= a.min_inl:
            cands = [("pred_ft", Rp, tp)] if (Rft is not None and t in Rft) else (([] if glob else [("pred", Rp, tp), ("hold", R_s[last], t_s[last])]) + an.hypotheses(obs))
            best = None
            for name, Rh, th in cands:
                n_in = int((an.resid(Rh, th, obs) < a.tau_in).sum())
                if best is None or n_in > best[0]: best = (n_in, name, Rh, th)
            if best is not None and best[0] >= 3:
                _, name, Rh, th = best
                e0 = an.resid(Rh, th, obs); gate = e0 < a.tau_out
                sub = tuple(o[gate] for o in obs)
                prior = (Rp, tp, a.prior_deg, a.prior_mm) if (a.prior_deg > 0 and not glob and fs == 1) else None
                Rr, tr_ = an.refine_t(Rh, th, sub) if (Rft is not None and t in Rft) else an.refine(Rh, th, sub, prior)
                e1 = an.resid(Rr, tr_, obs); inl = e1 < a.tau_in
                kin = np.unique(obs[0][inl]); sp = an.spread(kin)
                ok = inl.sum() >= a.min_inl and sp >= a.min_spread and ((inl & (obs[1] == 0)).sum() >= a.min_view and (inl & (obs[1] == 1)).sum() >= a.min_view)
                dev = float(rt.rot_angle_deg(Rr @ Rp.T))
                if ok and not glob and fs == 1 and a.jump_deg > 0 and dev > a.jump_deg: ok = False
                row.update(hyp=name, n_inl=int(inl.sum()), n_inl1=int((inl & (obs[1] == 0)).sum()), n_inl2=int((inl & (obs[1] == 1)).sum()),
                           reproj_px=float(np.median(e1[inl])) if inl.any() else np.nan, spread_cm=sp * 100, solved=bool(ok), dev_pred_deg=dev)
                if ok and fs == 1 and not glob and (a.fuse_rot < 1.0 or a.fuse_pos < 1.0):
                    # causal alpha-beta filter: blend the measured pose with the constant-velocity prediction
                    cen0 = an.X.mean(0)
                    dw = Rotation.from_matrix(Rr @ Rp.T).as_rotvec()
                    Rf = ortho(Rotation.from_rotvec(a.fuse_rot * dw).as_matrix() @ Rp)
                    cm, cp = Rr @ cen0 + tr_, Rp @ cen0 + tp
                    cf = cp + a.fuse_pos * (cm - cp)
                    Rr, tr_ = Rf, cf - Rf @ cen0
                if ok:
                    if fs == 1:                                   # velocity update from the accepted increment
                        cen0 = an.X.mean(0); w_ = Rotation.from_matrix(Rr @ R_s[t - 1].T).as_rotvec()
                        d_ = (Rr @ cen0 + tr_) - (R_s[t - 1] @ cen0 + t_s[t - 1])
                        vel = (w_, d_) if vel is None else ((1 - a.vel_beta) * vel[0] + a.vel_beta * w_, (1 - a.vel_beta) * vel[1] + a.vel_beta * d_)
                    else:
                        vel = None
                    R_s[t], t_s[t], solved[t] = Rr, tr_, True; last = t
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
        if not solved.get(t):
            solved[t] = False
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
                an.X = np.vstack([an.X, np.array(newX)]); n_extended = len(newX)
        rows.append(row)
    rec.update(K_src=ksrc, T=T, birth=b, exit=e, window_end=t_end, n_seeds=N, n_keyframes=n_kf, n_extended=n_extended, seed_mask=rec_seed_mask, elapsed_s=time.time() - t_start,
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
