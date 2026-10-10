#!/usr/bin/env python
"""rig_v0: PAIRING-FREE exterior-camera rig, v0 (2026-10-08).

Every tracked point, from either exterior camera, is a gripper point at an unknown depth d_i along its birth-frame ray;
one rigid motion T_t per frame moves all of them; depths and motions are solved by robust reprojection error in both
cameras at once. No cross-view pairing: the shared T_t is the only link between the views.

  model    X_i = o_c + d_i r_i          base frame at the birth frame b := gripper frame, so T_b = I
  observe  u_(i,t) ~ pi_c(T_t X_i)      c = the camera that tracks point i
  link     at b, each ray's depth interval = the first run of the ray that projects inside the OTHER camera's silhouette
           (two-view visual hull); d_i starts at near + ALPHA (far - near) (the point faces its own camera); rays that never
           enter the other silhouette are dropped; a soft prior keeps d_i inside the interval widened by PRIOR_MARGIN on both
           sides (the interval itself as the bound pins the points whose true depth sits at or past its ends and stalls
           the solve: 8-9 deg vs 0.01 deg on the exact-label check).
  frame t  (causal) one robust least-squares solve over the last W_FREE poses (free) and all live depths, using the
           observations of those frames plus older solved frames at frozen poses (depth evidence, up to W_HIST back).
           T_t is emitted once and never revised. Robust start per frame: the candidate (constant velocity, hold, per-camera
           PnP-RANSAC on the current model) with the most inliers; only observations within TAU_OUT of it enter the solve,
           so points are set aside per frame (fingers while the gripper closes) and come back when they fit again; a point
           > TAU_OUT px off on BAD_FRAMES consecutive frames is dropped for good.
  re-pick  every RESEARCH_EVERY frames each linked point's depth is re-chosen by a 1D grid search along its ray against
           the solved poses of the last W_HIST frames (truncated reprojection cost); taken when it halves the cost at the
           current depth (and a dropped point whose new depth fits is re-admitted). Local refinement cannot move a point
           that started in the wrong silhouette run (thin fingers: a phantom stretch of the two-view hull) several cm. Frame solved if >= MIN_INL inliers (< TAU_IN px) spread >= MIN_SPREAD m along their 2nd principal axis.

Sources
  --source trackon  Track-On-R v2 tracks (points seeded at the birth frame); silhouette = the RobotSeg birth masks
                    (--masks_root, mapped through the v2 worklist) or --silhouette seedhull (dilated hull of the seeds).
  --source labels   SOLVER CHECK: FK label points (positions_v1) as tracks, each camera's points treated as UNPAIRED;
                    silhouette = union of the per-link convex hulls of the projected labels. --init gt starts every
                    depth at the truth (solver only); --noise_px / --drop_frac perturb the tracks.
Evaluation (GT only here; rig_points_eval.evaluate conventions with t_ref = b): lens pose = T_t @ GT(b); per-frame
position / rotation error, Sim3 ATE / RPE (rig_track.traj_metrics), CUT3R finetuned on the same frames.

    python rig_v0.py --episode EP --source labels|trackon --out DIR [--masks_root ROOT | --silhouette seedhull]
Writes DIR/record.json (rig_points_eval layout, windows[0]), per_frame.csv, anchors.npz.
"""
import argparse, csv, json, os, sys, time
import numpy as np, cv2
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_track as rt          # noqa: E402
import rig_points_eval as rpe   # noqa: E402
import rig_trackon_eval as rte  # noqa: E402

S = "/gpfs/scratch/etur59/koc821022"
EVENTS = f"{S}/outputs/droid_birth_frames/entries/entry_events_gap3.csv"
TRACKS = f"{S}/outputs/droid_birth_frames/trackon_run_v2/tracks"
WORKLIST = f"{S}/outputs/droid_birth_frames/trackon_run_v2/worklist.json"
LEO_ROOT = "/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/"
W, H = rt.W, rt.H


# ----------------------------------------------------------------------------- inputs
def window(ep, events, T, no_cut):
    ev = [r for r in csv.DictReader(open(events)) if r["episode"] == ep and r["event"] == "0"]
    if not ev: raise rt.Failure("no event 0 (no birth frame / excluded)")
    b, e = int(ev[0]["entry_frame"]), int(ev[0]["exit_frame"])
    return b, (T - 1 if no_cut else min(e, T) - 1)


def load_labels(ep, a):
    """FK label points as per-camera tracks. Returns cams, K, E, T, U{c: (N,T,2)}, O{c: (N,T)}, sil_fn, extra."""
    z = np.load(f"{a.positions}/{ep}_gripper.npz")
    cams, K, E, P, ksrc = rpe.calib(ep, z)
    for _, s in cams:
        if s not in [str(x) for x in z["serials"]]: raise rt.Failure(f"camera {s} has no labels (excluded)")
    n_store = rt.count_files(f"{rt.STORE_ROOT}/{ep}/dense/cam", ".npz")
    T = min(int(z["T"]) - 1, n_store)
    sc = np.array([W / 320, H / 180]); link = z["link"]
    keep = np.ones(len(link), bool) if a.links == "all" else (link == "robotiq_85_base_link")
    rng = np.random.default_rng(a.seed)
    U, O, Zd, sil_pts = {}, {}, {}, {}
    for c, s in cams:
        uv = z[f"{s}_uv"][:T].astype(np.float64) * sc                      # (T, N, 2)
        ob = (z[f"{s}_in_frame"] & z[f"{s}_front"])[:T] & keep[None]
        if a.noise_px > 0: uv = uv + rng.normal(0, a.noise_px, uv.shape)
        if a.drop_frac > 0: ob = ob & (rng.random(ob.shape) >= a.drop_frac)
        U[c] = np.transpose(uv, (1, 0, 2)); O[c] = ob.T
        Zd[c] = z[f"{s}_z"][:T].astype(np.float64).T                         # camera-z depth (N, T), GT diagnostics only
        sil_pts[c] = (z[f"{s}_uv"][:T].astype(np.float64) * sc, (z[f"{s}_z"][:T] > 0))

    def silhouette(c, b):
        uv, front = sil_pts[c]; m = np.zeros((H, W), np.uint8)
        for name in np.unique(link):
            p = uv[b, (link == name) & front[b]]
            if len(p) >= 3: cv2.fillConvexPoly(m, cv2.convexHull(np.round(p).astype(np.int32)), 1)
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    return cams, K, E, T, U, O, silhouette, dict(K_src=ksrc, depth_gt=Zd)


def load_trackon(ep, a, b_hint=None):
    cams, K, E, P, ksrc = rte.calib(ep, a.positions)
    ev = [r for r in csv.DictReader(open(a.events)) if r["episode"] == ep and r["event"] == "0"]
    if not ev: raise rt.Failure("no event 0 (no birth frame / excluded)")
    b = int(ev[0]["entry_frame"])
    U, O, Q = {}, {}, {}
    for c, _ in cams:
        f = f"{a.tracks}/{ep}/{c}_f{b:05d}.npz"
        if not os.path.isfile(f): raise rt.Failure(f"track file missing: {os.path.basename(f)}")
        z = np.load(f); U[c] = np.transpose(z["tracks"], (1, 0, 2)).astype(np.float64); Q[c] = z["queries"]
        O[c] = z["visibility"].T.copy()
    n_store = rt.count_files(f"{rt.STORE_ROOT}/{ep}/dense/cam", ".npz")
    T = min(n_store, U["ext1"].shape[1], U["ext2"].shape[1])
    for c in U:
        x = U[c][:, :T]; o = O[c][:, :T] & ~np.isnan(x[..., 0]) & (x[..., 0] >= 0) & (x[..., 0] < W) & (x[..., 1] >= 0) & (x[..., 1] < H)
        o &= (Q[c][:, 0] == b)[:, None]                                         # points seeded at the birth frame only
        U[c] = np.nan_to_num(x); O[c] = o
    wl = None
    if a.silhouette == "masks":
        wl = {(w["episode"], w["cam"]): w["mask"] for w in json.load(open(a.worklist))}

    def silhouette(c, bb):
        if a.silhouette == "masks":
            rel = wl.get((ep, c))
            if rel is None: raise rt.Failure(f"no worklist mask for {c}")
            f = f"{a.masks_root}/{rel.replace(LEO_ROOT, '')}"
            m = cv2.imread(f, 0)
            if m is None: raise rt.Failure(f"mask missing: {f}")
            if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            return m > 0
        q = Q[c][Q[c][:, 0] == bb, 1:]                                          # seedhull: dilated convex hull of the seeds
        if len(q) < 3: raise rt.Failure(f"fewer than 3 seeds in {c}")
        dd = np.sqrt(((q[:, None] - q[None]) ** 2).sum(2)); np.fill_diagonal(dd, np.inf)
        r = int(np.ceil(np.median(dd.min(1)))) + 5                                # grid stride + the seeding erosion
        m = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(m, cv2.convexHull(np.round(q).astype(np.int32)), 1)
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    return cams, K, E, T, U, O, silhouette, dict(K_src=ksrc)


# ----------------------------------------------------------------------------- geometry
def project(Kc, Ec, X):
    Z = X @ Ec[:3, :3].T + Ec[:3, 3]; z = np.maximum(Z[:, 2], 1e-3)
    return (Z[:, :2] / z[:, None]) * [Kc[0, 0], Kc[1, 1]] + [Kc[0, 2], Kc[1, 2]], Z[:, 2]


def hull_intervals(o, r, Kc, Ec, M, ds):
    """First run of each ray (o + d r, d in ds) projecting inside silhouette M of the other camera. -> near, far (nan if none)."""
    n, D = len(o), len(ds)
    X = (o[:, None, :] + ds[None, :, None] * r[:, None, :]).reshape(-1, 3)
    uv, z = project(Kc, Ec, X)
    ins = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    ii = np.where(ins)[0]; ins[ii] = M[uv[ii, 1].astype(int), uv[ii, 0].astype(int)]
    ins = ins.reshape(n, D); near, far = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        k = np.where(ins[i])[0]
        if len(k) == 0: continue
        k0 = k[0]; gap = np.where(np.diff(k) > 1)[0]; k1 = k[gap[0]] if len(gap) else k[-1]
        near[i], far[i] = ds[k0], ds[k1]
    return near, far


def ortho(R):
    """Nearest rotation (SVD). Composing R1 R2^T R1 for the constant-velocity prediction amplifies round-off ~2.4x per frame."""
    U_, _, Vt = np.linalg.svd(R); D = np.diag([1, 1, np.sign(np.linalg.det(U_ @ Vt))]); return U_ @ D @ Vt


def se3(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; return M


# ----------------------------------------------------------------------------- the solver
class Rig:
    def __init__(self, a, cams, K, E, U, O, b, T):
        self.a = a; self.b = b; self.T = T
        self.Kc = np.array([K[c] for c, _ in cams]); self.Ec = np.array([E[c] for c, _ in cams])
        self.Rc = self.Ec[:, :3, :3]; self.tc = self.Ec[:, :3, 3]
        self.fc = np.stack([self.Kc[:, 0, 0], self.Kc[:, 1, 1]], 1); self.cc = np.stack([self.Kc[:, 0, 2], self.Kc[:, 1, 2]], 1)
        # one row per point seeded (observed) at the birth frame, both cameras stacked; (cam, index in that camera)
        rows = [(ci, k) for ci, (c, _) in enumerate(cams) for k in np.where(O[c][:, b])[0]]
        self.cam = np.array([r[0] for r in rows], int); self.src_idx = np.array([r[1] for r in rows], int)
        self.U = np.stack([U[cams[ci][0]][k] for ci, k in rows]); self.O = np.stack([O[cams[ci][0]][k] for ci, k in rows])
        n = len(rows)
        self.o = np.array([-self.Rc[ci].T @ self.tc[ci] for ci in self.cam])                # camera centres (base frame)
        Kinv = np.linalg.inv(self.Kc)
        rr = np.array([self.Rc[ci].T @ (Kinv[ci] @ np.r_[self.U[i, b], 1.0]) for i, ci in enumerate(self.cam)])
        self.r = rr / np.linalg.norm(rr, axis=1, keepdims=True)
        self.near, self.far = np.full(n, np.nan), np.full(n, np.nan)
        self.d = np.full(n, np.nan); self.active = np.zeros(n, bool); self.bad = np.zeros(n, int)
        self.Rs = {b: np.eye(3)}; self.ts = {b: np.zeros(3)}; self.solved = {b: True}
        self.G = np.zeros_like(self.O)          # observation used by the solves (inlier at its own frame)

    def link(self, sil, ds):
        """The cross-view step: each ray's depth interval inside the other camera's silhouette."""
        for ci in range(2):
            m = self.cam == ci
            if not m.any(): continue
            nr, fr = hull_intervals(self.o[m], self.r[m], self.Kc[1 - ci], self.Ec[1 - ci], sil[1 - ci], ds)
            self.near[m], self.far[m] = nr, fr
        ok = ~np.isnan(self.near)
        self.d[ok] = self.near[ok] + self.a.alpha * (self.far[ok] - self.near[ok]); self.active = ok.copy()
        self.near0, self.far0 = self.near.copy(), self.far.copy()                     # raw silhouette interval (diagnostics)
        self.near, self.far = self.near - self.a.prior_margin, self.far + self.a.prior_margin

    def X(self, idx=None):
        idx = np.arange(len(self.d)) if idx is None else idx
        return self.o[idx] + self.d[idx, None] * self.r[idx]

    def solve(self, t, R_init, t_init):
        a = self.a
        self.Rs[t], self.ts[t] = R_init, t_init
        free = [s for s in range(max(self.b + 1, t - a.w_free + 1), t + 1)]
        hist = [s for s in range(max(self.b + 1, free[0] - a.w_hist), free[0], a.hist_stride) if self.solved.get(s)]
        frames = np.array(hist + free); nh = len(hist); nf = len(free)
        pts = np.where(self.active)[0]
        sub = (self.O & self.G)[np.ix_(pts, frames)]
        pi, fj = np.nonzero(sub)
        if len(pi) == 0: return None
        u = self.U[pts[pi], frames[fj]]; cam = self.cam[pts[pi]]
        o, r = self.o[pts[pi]], self.r[pts[pi]]
        R0 = np.array([self.Rs[s] for s in frames]); t0 = np.array([self.ts[s] for s in frames])
        nfree_p = 6 * nf; npt = len(pts); w_prior = a.prior_px_per_mm * 1000.0
        near, far = self.near[pts], self.far[pts]
        Rc, tc, fc, cc = self.Rc[cam], self.tc[cam], self.fc[cam], self.cc[cam]

        def poses(x):
            w = x[:nfree_p].reshape(nf, 6); Rd = Rotation.from_rotvec(w[:, :3]).as_matrix()
            Rs, ts = R0.copy(), t0.copy()
            Rs[nh:] = Rd @ R0[nh:]; ts[nh:] = np.einsum("nij,nj->ni", Rd, t0[nh:]) + w[:, 3:]
            return Rs, ts

        def fun(x):
            Rs, ts = poses(x); d = x[nfree_p:]
            Xp = o + d[pi, None] * r
            Y = np.einsum("nij,nj->ni", Rs[fj], Xp) + ts[fj]
            Z = np.einsum("nij,nj->ni", Rc, Y) + tc
            uv = Z[:, :2] / np.maximum(Z[:, 2:3], 1e-3) * fc + cc
            ex = np.maximum(0, near - d) + np.maximum(0, d - far)
            return np.concatenate([(uv - u).ravel(), ex * w_prior])

        # motion prior: acceleration of the model centroid (mm/frame^2) and of the rotation (deg/frame^2) per free frame,
        # against the previous two frames (free in this solve or already emitted); Huber like the pixels, so a real change
        # of motion costs only linearly. Causal: only frames <= t.
        cen = self.X(pts).mean(0) if npt else np.zeros(3); acc = []
        if a.smooth_rot_deg > 0:
            for k, s_ in enumerate(free):
                if s_ - 2 < self.b: continue
                ref = []
                for q in (s_ - 1, s_ - 2):
                    if q in free: ref.append(("f", free.index(q)))
                    elif q in self.Rs: ref.append(("x", q))
                    else: ref = None; break
                if ref: acc.append((k, ref))
        nacc = len(acc)
        fixed = sorted({r_[1] for _, refs in acc for r_ in refs if r_[0] == "x"}); fpos = {q: nf + j for j, q in enumerate(fixed)}
        fR = np.array([self.Rs[q] for q in fixed]).reshape(-1, 3, 3); ft = np.array([self.ts[q] for q in fixed]).reshape(-1, 3)
        ix = lambda r_: r_[1] if r_[0] == "f" else fpos[r_[1]]
        i0 = np.array([k for k, _ in acc], int); i1 = np.array([ix(refs[0]) for _, refs in acc], int); i2 = np.array([ix(refs[1]) for _, refs in acc], int)

        def acc_res(Rs, ts):
            if not nacc: return np.zeros(0)
            AR = np.concatenate([Rs[nh:], fR]); At = np.concatenate([ts[nh:], ft])
            R0_, R1_, R2_ = AR[i0], AR[i1], AR[i2]
            w01 = Rotation.from_matrix(R0_ @ np.transpose(R1_, (0, 2, 1))).as_rotvec()
            w12 = Rotation.from_matrix(R1_ @ np.transpose(R2_, (0, 2, 1))).as_rotvec()
            c = AR @ cen + At
            return np.concatenate([np.degrees(w01 - w12) / a.smooth_rot_deg, (c[i0] - 2 * c[i1] + c[i2]) * 1000 / a.smooth_pos_mm], 1).ravel()

        fun_obs = fun
        def fun(x):
            Rs, ts = poses(x); return np.concatenate([fun_obs(x), acc_res(Rs, ts)])

        nobs = len(pi); rows, cols = [], []
        for k in range(2):                                   # observation rows: own frame's pose block (if free) + own depth
            rr_ = 2 * np.arange(nobs) + k
            isf = fj >= nh
            for q in range(6):
                rows.append(rr_[isf]); cols.append(6 * (fj[isf] - nh) + q)
            rows.append(rr_); cols.append(nfree_p + pi)
        rows.append(2 * nobs + np.arange(npt)); cols.append(nfree_p + np.arange(npt))
        for j, (k, refs) in enumerate(acc):                  # acceleration rows: this frame and its free predecessors
            r0 = 2 * nobs + npt + 6 * j + np.arange(6)
            for kk in [k] + [r_[1] for r_ in refs if r_[0] == "f"]:
                for q in range(6): rows.append(r0); cols.append(np.full(6, 6 * kk + q))
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        sp = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * nobs + npt + 6 * nacc, nfree_p + npt)).tocsr()
        x0 = np.r_[np.zeros(nfree_p), self.d[pts]]
        res = least_squares(fun, x0, jac_sparsity=sp, method="trf", loss="huber", f_scale=a.huber_px,
                            x_scale=1.0, max_nfev=a.max_nfev)   # x_scale="jac" stalls (ftol) far from the optimum
        Rs, ts = poses(res.x)
        for k, s in enumerate(free): self.Rs[s], self.ts[s] = ortho(Rs[nh + k]), ts[nh + k]
        self.d[pts] = res.x[nfree_p:]
        return res

    def research_depths(self, t, ds):
        """1D global depth search per linked point against the solved poses so far (causal). Returns #points changed."""
        a = self.a
        fr = [s for s in range(max(self.b + 1, t - a.w_hist), t + 1, a.hist_stride) if self.solved.get(s)]
        if len(fr) < 3: return 0
        cand = np.where(~np.isnan(self.near0))[0]; cap2 = a.research_cap_px ** 2; changed = 0
        for ci in range(2):
            idx = cand[self.cam[cand] == ci]
            if not len(idx): continue
            Xg = self.o[idx, None, :] + ds[None, :, None] * self.r[idx, None, :]                     # (n, D, 3)
            Xc = self.X(idx)                                                                          # current depths
            cost = np.zeros((len(idx), len(ds))); cur = np.zeros(len(idx)); nob = np.zeros(len(idx), int)
            for s in fr:
                ob = self.O[idx, s]
                if not ob.any(): continue
                Y = Xg @ self.Rs[s].T + self.ts[s]
                uv, _ = project(self.Kc[ci], self.Ec[ci], Y.reshape(-1, 3)); uv = uv.reshape(len(idx), len(ds), 2)
                e2 = np.minimum(((uv - self.U[idx, s][:, None]) ** 2).sum(2), cap2)
                cost += np.where(ob[:, None], e2, 0)
                uvc, _ = project(self.Kc[ci], self.Ec[ci], Xc @ self.Rs[s].T + self.ts[s])
                cur += np.where(ob, np.minimum(((uvc - self.U[idx, s]) ** 2).sum(1), cap2), 0); nob += ob
            k = cost.argmin(1); best = cost[np.arange(len(idx)), k]
            take = (nob >= 5) & (best < 0.5 * cur) & (best / np.maximum(nob, 1) < a.tau_in ** 2)
            for j in np.where(take)[0]:
                i = idx[j]; self.d[i] = ds[k[j]]; self.near[i] = min(self.near[i], self.d[i] - a.prior_margin)
                self.far[i] = max(self.far[i], self.d[i] + a.prior_margin); self.active[i] = True; self.bad[i] = 0; changed += 1
        return changed

    def residual_at(self, t, R=None, tt=None):
        """Reprojection error (px) at frame t of every active point observed at t (nan elsewhere), under (R, tt) or the
        stored pose of frame t."""
        R = self.Rs[t] if R is None else R; tt = self.ts[t] if tt is None else tt
        e = np.full(len(self.d), np.nan); m = self.active & self.O[:, t]
        if not m.any(): return e
        idx = np.where(m)[0]; Y = self.X(idx) @ R.T + tt
        for ci in range(2):
            mm = self.cam[idx] == ci
            if mm.any():
                uv, _ = project(self.Kc[ci], self.Ec[ci], Y[mm]); e[idx[mm]] = np.linalg.norm(uv - self.U[idx[mm], t], axis=1)
        return e

    def pnp_candidates(self, t):
        """Per-camera PnP + RANSAC on the current model (gripper frame) -> candidate motions T_t = E_c^-1 [R|t]_pnp."""
        out = []; m = self.active & self.O[:, t]
        for ci in range(2):
            idx = np.where(m & (self.cam == ci))[0]
            if len(idx) < 6: continue
            ok, rv, tv, inl = cv2.solvePnPRansac(self.X(idx).astype(np.float64), self.U[idx, t].astype(np.float64), self.Kc[ci], None,
                                                 iterationsCount=100, reprojectionError=self.a.tau_in, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
            if not ok or inl is None or len(inl) < 4: continue
            Rp_ = cv2.Rodrigues(rv)[0]; tp_ = tv.ravel()
            out.append((ortho(self.Rc[ci].T @ Rp_), self.Rc[ci].T @ (tp_ - self.tc[ci])))
        return out

    def spread(self, idx):
        if len(idx) < 3: return 0.0
        Xg = self.X(idx); Xg = Xg - Xg.mean(0)
        _, _, Vt = np.linalg.svd(Xg, full_matrices=False); p = Xg @ Vt[1]
        return float(p.max() - p.min())

    def run(self, t_end):
        a = self.a; stats = {}; ds = np.arange(a.dmin, a.dmax, a.dstep); self.n_repicked = 0
        for t in range(self.b + 1, t_end + 1):
            p1 = t - 1; p2 = t - 2
            Rp, tp = self.Rs[p1], self.ts[p1]
            if p2 >= self.b and self.solved.get(p1) and self.solved.get(p2):   # constant velocity in the base frame
                dR = self.Rs[p1] @ self.Rs[p2].T; dt = self.ts[p1] - dR @ self.ts[p2]
                Rcv, tcv = ortho(dR @ Rp), dR @ tp + dt
            else:
                Rcv, tcv = Rp, tp
            nobs_t = int((self.active & self.O[:, t]).sum())
            if nobs_t < a.min_inl or not self.active.any():
                self.Rs[t], self.ts[t] = Rp, tp; self.solved[t] = False
                stats[t] = dict(n_obs=nobs_t, n_inl=0, n_inl_ext1=0, n_inl_ext2=0, spread_cm=np.nan, reproj_px=np.nan, solved=False, retry=False)
                continue
            # robust start: constant velocity, hold, per-camera PnP-RANSAC; keep the one most points agree with, and let only
            # the observations within TAU_OUT of it into the solve (non-rigid fingers while the gripper closes, slid tracks)
            # the prediction is kept unless an alternative has clearly more inliers (single-camera PnP on a small object is
            # noisy: with 1.5 px noise a plain argmax picked it on 67 % of frames and the error followed)
            e0 = self.residual_at(t, Rcv, tcv); best = (int(np.nansum(e0 < a.tau_in)), 0, Rcv, tcv, e0)
            for k, (Rc_, tc_) in enumerate([(Rp, tp)] + self.pnp_candidates(t), start=1):
                e1 = self.residual_at(t, Rc_, tc_); n1 = int(np.nansum(e1 < a.tau_in))
                if n1 >= max(a.switch_ratio * best[0], best[0] + a.switch_min): best = (n1, k, Rc_, tc_, e1)
            _, kbest, Rb, tb, e0 = best; retry = kbest >= 2
            gate = ~np.isnan(e0) & (e0 < a.tau_out)
            if gate.sum() < a.min_inl:
                self.Rs[t], self.ts[t] = Rb, tb; self.solved[t] = False
                stats[t] = dict(n_obs=nobs_t, n_inl=0, n_inl_ext1=0, n_inl_ext2=0, spread_cm=np.nan, reproj_px=np.nan, solved=False, retry=retry)
                continue
            self.G[:, t] = gate
            self.solve(t, Rb, tb); e = self.residual_at(t)
            obs = ~np.isnan(e); inl = obs & (e < a.tau_in); self.G[:, t] = obs & (e < a.tau_out)
            self.bad[obs & (e > a.tau_out)] += 1; self.bad[obs & (e <= a.tau_out)] = 0
            self.active &= self.bad < a.bad_frames
            sp = self.spread(np.where(inl)[0]); ok = (inl.sum() >= a.min_inl) and (sp >= a.min_spread)
            self.solved[t] = bool(ok)
            if a.research_every and (t - self.b) % a.research_every == 0: self.n_repicked += self.research_depths(t, ds)
            stats[t] = dict(n_obs=nobs_t, n_inl=int(inl.sum()), n_inl_ext1=int((inl & (self.cam == 0)).sum()),
                            n_inl_ext2=int((inl & (self.cam == 1)).sum()), spread_cm=sp * 100,
                            reproj_px=float(np.nanmedian(e[inl])) if inl.any() else np.nan, solved=bool(ok), retry=retry)
        return stats


# ----------------------------------------------------------------------------- evaluation (GT only here)
def evaluate(rig, stats, b, t_end, gt, cut):
    P0 = gt[b]; rows, pred, gtl, cutl, pf = [], [], [], [], []
    for t in range(b, t_end + 1):
        if t not in gt: break
        st = stats.get(t, dict(n_obs=int(rig.O[:, b].sum()), n_inl=int(rig.active.sum()), solved=True))
        row = dict(frame=t, **{k: v for k, v in st.items()})
        if st["solved"]:
            Pt = se3(rig.Rs[t], rig.ts[t]) @ P0
            row.update(pos_err_cm=float(np.linalg.norm(Pt[:3, 3] - gt[t][:3, 3]) * 100), rot_err_deg=float(rt.rot_angle_deg(Pt[:3, :3].T @ gt[t][:3, :3])))
            pred.append(Pt); gtl.append(gt[t]); pf.append(t)
            if cut is not None and t in cut: cutl.append(cut[t])
        else:
            row.update(pos_err_cm=np.nan, rot_err_deg=np.nan)
        rows.append(row)
    w = dict(window=[b, t_end], window_len=t_end - b + 1, t_ref=b, n_anchored=len(pred))
    ok = [r for r in rows if r["solved"] and r["frame"] > b]
    if len(pred) >= 2 and ok:
        pe = np.array([r["pos_err_cm"] for r in ok]); re_ = np.array([r["rot_err_deg"] for r in ok]); m = rt.traj_metrics(pred, gtl)
        w.update(pos_err_median_cm=float(np.median(pe)), pos_err_p90_cm=float(np.percentile(pe, 90)), pos_err_max_cm=float(pe.max()),
                 rot_err_median_deg=float(np.median(re_)), rot_err_p90_deg=float(np.percentile(re_, 90)), rot_err_max_deg=float(re_.max()),
                 inliers_median=float(np.median([r["n_inl"] for r in ok])), reproj_px_median=float(np.nanmedian([r["reproj_px"] for r in ok])),
                 both_cams_frac=float(np.mean([(r["n_inl_ext1"] >= 1) and (r["n_inl_ext2"] >= 1) for r in ok])),
                 rig_ate=m["ate"], rig_rpe_t=m["rpe_trans"], rig_rpe_rot=m["rpe_rot"])
        if cut is not None and len(cutl) == len(pred):
            mk = rt.traj_metrics(cutl, gtl); w.update(cut3r_ate_same_frames=mk["ate"], cut3r_rpe_t_same_frames=mk["rpe_trans"], cut3r_rpe_rot_same_frames=mk["rpe_rot"])
    return w, rows, pf, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", default="v0")
    ap.add_argument("--source", choices=["labels", "trackon"], required=True)
    ap.add_argument("--positions", default=rte.POSITIONS); ap.add_argument("--events", default=EVENTS); ap.add_argument("--tracks", default=TRACKS)
    ap.add_argument("--silhouette", choices=["masks", "seedhull"], default="masks", help="trackon only")
    ap.add_argument("--masks_root", default=f"{S}/outputs/droid_birth_frames"); ap.add_argument("--worklist", default=WORKLIST)
    ap.add_argument("--no_cut", action="store_true")
    # labels-only solver-check knobs
    ap.add_argument("--links", choices=["all", "base"], default="all"); ap.add_argument("--init", choices=["hull", "gt"], default="hull")
    ap.add_argument("--noise_px", type=float, default=0.0); ap.add_argument("--drop_frac", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    # method parameters (720p pixels, metres)
    ap.add_argument("--alpha", type=float, default=0.3); ap.add_argument("--dmin", type=float, default=0.1); ap.add_argument("--dmax", type=float, default=3.0)
    ap.add_argument("--dstep", type=float, default=0.002); ap.add_argument("--prior_px_per_mm", type=float, default=1.0)
    ap.add_argument("--prior_margin", type=float, default=0.03, help="m added to both ends of the silhouette interval for the depth prior")
    ap.add_argument("--w_free", type=int, default=10); ap.add_argument("--w_hist", type=int, default=60); ap.add_argument("--hist_stride", type=int, default=2)
    ap.add_argument("--huber_px", type=float, default=2.0); ap.add_argument("--tau_in", type=float, default=4.0); ap.add_argument("--tau_out", type=float, default=10.0)
    ap.add_argument("--bad_frames", type=int, default=30, help="consecutive outlier frames before a point is dropped for good"); ap.add_argument("--min_inl", type=int, default=6); ap.add_argument("--min_spread", type=float, default=0.01)
    ap.add_argument("--max_nfev", type=int, default=30)
    ap.add_argument("--smooth_rot_deg", type=float, default=0.0, help="motion prior: rotation acceleration scale (deg/frame^2); 0 = off")
    ap.add_argument("--smooth_pos_mm", type=float, default=2.0, help="motion prior: centroid acceleration scale (mm/frame^2)")
    ap.add_argument("--switch_ratio", type=float, default=1.2); ap.add_argument("--switch_min", type=int, default=5)
    ap.add_argument("--research_every", type=int, default=5, help="0 disables the 1D depth re-pick"); ap.add_argument("--research_cap_px", type=float, default=10.0)
    a = ap.parse_args(); ep = a.episode; os.makedirs(a.out, exist_ok=True)
    rec = dict(episode=ep, tag=a.tag, source=a.source, args=vars(a)); t_start = time.time()
    try:
        loader = load_labels if a.source == "labels" else load_trackon
        cams, K, E, T, U, O, silhouette, extra = loader(ep, a)
        b, t_end = window(ep, a.events, T, a.no_cut)
        if t_end <= b: raise rt.Failure("empty window")
        store = f"{rt.STORE_ROOT}/{ep}/dense/cam"; gt = rt.load_poses(store, T, "store GT")
        try: cut = rt.load_poses(f"{rt.CUT3R_ROOT}/{ep}/camera", T, "CUT3R")
        except rt.Failure: cut = None
        for c in U: O[c][:, :b] = False; O[c][:, t_end + 1:] = False
        rig = Rig(a, cams, K, E, U, O, b, T)
        if len(rig.d) == 0: raise rt.Failure("no point observed at the birth frame")
        sil = [silhouette(c, b) for c, _ in cams]
        rig.link(sil, np.arange(a.dmin, a.dmax, a.dstep))
        diag = dict(n_seeds={c: int((rig.cam == ci).sum()) for ci, (c, _) in enumerate(cams)},
                    n_linked={c: int((rig.active & (rig.cam == ci)).sum()) for ci, (c, _) in enumerate(cams)},
                    interval_cm_median=float(np.nanmedian(rig.far0 - rig.near0) * 100) if rig.active.any() else None)
        if a.source == "labels":                     # GT depth along each ray (diagnostic; and the --init gt oracle)
            dgt = np.array([extra["depth_gt"][cams[ci][0]][k, b] / (rig.Rc[ci] @ rig.r[i])[2] for i, (ci, k) in enumerate(zip(rig.cam, rig.src_idx))])
            m = rig.active
            if m.any():
                al = (dgt[m] - rig.near0[m]) / np.maximum(rig.far0[m] - rig.near0[m], 1e-6)
                diag.update(alpha_true_median=float(np.median(al)), alpha_true_p10=float(np.percentile(al, 10)), alpha_true_p90=float(np.percentile(al, 90)),
                            true_in_interval_frac=float(np.mean((al >= 0) & (al <= 1))), depth_err_init_mm=float(np.median(np.abs(rig.d[m] - dgt[m])) * 1000))
            if a.init == "gt":
                rig.d = dgt.copy(); rig.near, rig.far = dgt - 0.005, dgt + 0.005; rig.active = np.ones(len(dgt), bool)
        rec.update(K_src=extra["K_src"], T=T, birth=b, window_end=t_end, link=diag)
        if rig.active.sum() < a.min_inl: raise rt.Failure(f"fewer than {a.min_inl} linked points ({int(rig.active.sum())})")
        stats = rig.run(t_end)
        if a.source == "labels":
            m = rig.active
            if m.any(): diag.update(depth_err_end_mm=float(np.median(np.abs(rig.d[m] - dgt[m])) * 1000), active_end=int(m.sum()))
        w, rows, pf, pred = evaluate(rig, stats, b, t_end, gt, cut)
        w.update(episode=ep, tag=a.tag, window_idx=0, n_windows=1); rec.update(windows=[w], active_end=int(rig.active.sum()), n_repicked=int(rig.n_repicked),
                                                                              frames_with_min_pts_both=int(sum(r.get("solved", False) for r in rows)))
        if pred:
            Tg = np.array([se3(rig.Rs[t], rig.ts[t]) for t in pf])
            np.savez(f"{a.out}/anchors.npz", frames=np.array(pf), poses=np.array(pred), centroid_poses=np.array(pred), gripper_motion=Tg)
        with open(f"{a.out}/per_frame.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, list(rows[-1])); wr.writeheader(); wr.writerows([{k: r.get(k) for k in rows[-1]} for r in rows])
    except rt.Failure as ex:
        rec.update(failure_reason=str(ex), windows=[dict(episode=ep, tag=a.tag, window_idx=0, n_windows=1, skipped=str(ex))])
        rec.setdefault("T", 0); rec.setdefault("frames_with_min_pts_both", 0)
    rec["seconds"] = time.time() - t_start
    json.dump(rec, open(f"{a.out}/record.json", "w"), indent=1, default=float)
    print(json.dumps({k: rec[k] for k in rec if k not in ("windows", "args")}, default=float)[:600]); print(json.dumps(rec["windows"], default=float)[:600])


if __name__ == "__main__":
    main()
