#!/usr/bin/env python
"""seed_cheap_verifiers (idea 4, 2026-09-29): seed_verified_motion.py (the round-3 candidate, copied verbatim: MOG2 background
model, moving components, cross-view DLT triangulation of the candidate centroids, reach / height / size / dark / mutual
silhouette gates, distal-preference Selector, box crop emission, verified_points.npz) PLUS four cheap, switchable verifiers
whose purpose is the IDENTIFICATION of the gripper among the verified moving things (the round-3 failure: 27 % of the seeder's
own verified 3D picks are > 0.20 m from the lens, i.e. an elbow, a joint ring, a manipulated object or a cloth).
CPU only, no learned model, no ground truth, causal (frame t uses frames <= t; emitted masks and points are final).

Tests, evaluated per cross-view pair that already passed the round-3 gates (so the round-3 verified set is the base set and
every test can only REMOVE pairs); each is logged per frame in seed_info.json and switchable (--twotone/--size2/--terminal/
--postfilter 0|1):
  (a) TWO-TONE SIGNATURE (--twotone): the DROID wrist is a black Robotiq 2F-85 body with a light-coloured block directly above
      it in the image up-direction of the arm (the white ZED-Mini housing + white Franka flange on the RAIL rig, a light-tan
      housing on the AUTOLab rigs; the Franka links are white on every rig). Per view: dark body = the candidate's dark-achromatic
      pixels (opened); proximal = pixels of the parent moving component whose arm-distance (geodesic from where the component
      enters the image, see below) is smaller than the dark body's median arm-distance; bright = luminance >= bright_thresh and
      chroma <= bright_chroma_max; the bright proximal pixels within twotone_adj_m (0.04 m-equivalent at the triangulated depth)
      of the dark body form the signature; its equivalent diameter sqrt(area) * depth / f must lie in [twotone_min, twotone_max]
      = [0.03, 0.08] m. Passes when >= twotone_views (1) of the two views pass (the housing can be hidden behind the gripper
      in one view). A black joint ring up the arm passes this test (white link on its proximal side): (c) is the test for that.
  (b) SIZE-AT-DEPTH (--size2): per view, at the triangulated depth in that view, the candidate's longer bbox side must lie in
      [size2_min, size_max] = [0.08, 0.30] m (the round-3 window was [0.07, 0.30]) AND its shorter bbox side must be >= size2_minor
      (0.03 m: the Robotiq body is ~ 8.5 cm wide, so no silhouette of it is thinner than 3 cm; cables and cloth edges are).
      Both views must pass.
  (c) TERMINAL NODE (--terminal): the gripper is the end of the kinematic chain. Per view, arm-distance = geodesic distance
      inside the parent foreground component from where it enters the image (border contacts, else the pixel nearest the
      projected base origin, else the top-most pixel; Euclidean fallback when the geodesic is unavailable). The moving-foreground
      pixels of the parent component (component & dilated frame-difference) must not extend beyond the candidate's farthest
      pixel by more than terminal_m (0.10 m-equivalent at the triangulated depth): max(arm-distance over moving pixels) -
      max(arm-distance over candidate pixels) <= 0.10 m. Both views must pass. Rejects joint rings, the elbow, the forearm;
      keeps the gripper (fingers and a held object within 10 cm pass; a long held object does not, accepted by design).
  (d) POST-FILTER (--postfilter): masks are emitted only on frames whose pick passed every enabled test (the tracker cannot seed
      elsewhere). In this seeder a frame without an accepted pick is empty by construction, so the post-filter is verified
      (n_removed logged, expected 0) rather than active; the tracker-side complement "no fresh segment / no link on a frame
      without a verified point" lives in rig_track2_cv.py --require_vp_start (a copy of rig_track2.py with that one flag) and is
      measured as a separate variant.

Declared constants (written to seed_info.json declared_constants): bright_thresh 140 / bright_chroma_max 100 (light-coloured
housing and flange: an object prior from the rig hardware, the same on every DROID rig), twotone_adj_m 0.04, twotone size
window [0.03, 0.08] m, size2 window [0.08, 0.30] m with minor side >= 0.03 m, terminal_m 0.10 (the numbers of the task
specification, taken from the hardware dimensions: Robotiq 2F-85 ~ 8.5 x 15 cm with housing, ZED Mini 12.5 x 3 cm); nothing
was tuned on the smoke scenes (run once, report). Everything else is the round-3 seeder's parameter set, untouched.

Cost: the tests add a per-candidate distance transform on a padded crop and a few reductions; measured per frame in
seed_info.json timing (round-3 seeder 120-240 ms/frame for both cameras at 640x360 on one login-node core).

Causality: frame t uses frames <= t only (MOG2 state, previous gray, Selector's last pick). GT-free: the store is opened only to
count frames; calibration = declared PointWorld optimized_extrinsics + factory intrinsics.

Outputs per the mask contract under <out_root>/<EP>/: <cam>_<serial>__<method>_masks.npz (union uint8 (T,720,160) packbits,
shape [T,720,1280], frames_present (T,)), verified_points.npz (frames, xyz base-frame, accepted), seed_info.json (params,
declared constants, per-frame log incl. the test values of the pick and the reason the round-3 pick was rejected, timing).

    OMP_NUM_THREADS=1 python seed_cheap_verifiers.py --episode EP [--method cheap_verifiers] [--twotone 1 --size2 1 --terminal 1 --postfilter 1]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

cv2.setNumThreads(1)  # login nodes cap CPU time per process across threads
from skimage.graph import MCP_Geometric  # noqa: E402

RAW_ROOT = "/gpfs/scratch/etur59/koc821022/vggt_cache/raw"
STORE_ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
AUDIT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/gt_audit_2026-09-08"
OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas/cheap_verifiers"
METHOD_DEFAULT = "cheap_verifiers"
NATIVE_W, NATIVE_H = 1280, 720
CAMS = ("ext1", "ext2")
TOOL_VERSION = "seed_cheap_verifiers v1 (2026-09-29 idea 4: seed_verified_motion round-3 seeder + tests a/b/c/d)"


# ----------------------------------------------------------------------------------------------- calibration (declared)
def load_calib(ep, serials):
    """Per camera: native-pixel intrinsics, world(base)->camera extrinsics, projection matrix, projected base origin and the
    plausible gripper depth (mean of the median depth of a workspace grid and the camera-to-base distance; used only to
    turn the candidate cap lengths into pixels). Declared calibration only, no GT."""
    intr_all = json.load(open(f"{AUDIT}/docs/hf_intrinsics.json"))[ep]
    camj = json.load(open(f"{AUDIT}/pointworld/droid/cameras/{ep}_cameras.json"))
    grid = np.array([[x, y, z, 1.0] for x in (0.3, 0.5, 0.7) for y in (-0.3, 0, 0.3) for z in (0.1, 0.3, 0.5)])
    out = {}
    for c, s in serials.items():
        fx, cx, fy, cy = intr_all[s]["cameraMatrix"]
        iw, ih = intr_all[s].get("width", NATIVE_W), intr_all[s].get("height", NATIVE_H)
        sx, sy = NATIVE_W / iw, NATIVE_H / ih
        fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        E = np.array(camj[s]["optimized_extrinsics"], float)
        P = K @ E[:3]
        Xc = E[:3, 3]  # base origin in camera coordinates
        base_px = [float(fx * Xc[0] / Xc[2] + cx), float(fy * Xc[1] / Xc[2] + cy)] if Xc[2] > 0.1 else None
        cam_c = -E[:3, :3].T @ E[:3, 3]
        z_grid = float(np.median((E @ grid.T)[2]))
        d_base = float(np.linalg.norm(cam_c))
        out[c] = dict(serial=s, fx=fx, fy=fy, cx=cx, cy=cy, K=K, E=E, P=P, base_px=base_px, cam_to_base=d_base,
                      z_grid_median=z_grid, plausible_depth=0.5 * (z_grid + d_base), cam_centre=cam_c.tolist())
    return out


def triangulate(P1, P2, x1, x2):
    """DLT triangulation of two native-pixel points. Returns X (3,) in the base frame, the mean reprojection residual (px)
    and the per-view residuals; inf when the point is behind either camera."""
    Xh = cv2.triangulatePoints(P1, P2, np.array(x1, float).reshape(2, 1), np.array(x2, float).reshape(2, 1))
    if abs(Xh[3, 0]) < 1e-12:
        return np.zeros(3), float("inf"), [float("inf"), float("inf")]
    X = (Xh[:3, 0] / Xh[3, 0]).ravel()
    r = []
    for P, x in ((P1, x1), (P2, x2)):
        p = P @ np.r_[X, 1.0]
        if p[2] <= 1e-6:
            return X, float("inf"), [float("inf"), float("inf")]
        r.append(float(np.hypot(p[0] / p[2] - x[0], p[1] / p[2] - x[1])))
    return X, float(np.mean(r)), r


def project(cal, X):
    """(x, y, depth) of a base-frame point in a camera, native pixels."""
    Xc = cal["E"][:3, :3] @ X + cal["E"][:3, 3]
    z = float(Xc[2])
    if z <= 1e-6:
        return None, None, z
    return float(cal["fx"] * Xc[0] / z + cal["cx"]), float(cal["fy"] * Xc[1] / z + cal["cy"]), z


# ----------------------------------------------------------------------------------------------- image helpers
def geodesic_small(comp, roots_rc, g):
    """Geodesic distance (8-connected) inside the boolean component from the root pixels, computed on a g-times
    downsampled grid (any-pooling keeps thin parts connected) and returned at the component's resolution in its pixels;
    inf outside / unreachable. None when no root lands inside the component."""
    H, W = comp.shape
    small = cv2.resize(comp.astype(np.uint8) * 255, (W // g, H // g), interpolation=cv2.INTER_AREA) > 0
    rs, cs = np.nonzero(small)
    if len(rs) == 0:
        return None
    r0, r1, c0, c1 = rs.min(), rs.max() + 1, cs.min(), cs.max() + 1
    sub = small[r0:r1, c0:c1]
    cost = np.where(sub, 1.0, np.inf)
    st = {(int(r // g - r0), int(c // g - c0)) for r, c in roots_rc}
    st = [p for p in st if 0 <= p[0] < sub.shape[0] and 0 <= p[1] < sub.shape[1] and sub[p]]
    if not st:
        return None
    cum, _ = MCP_Geometric(cost, fully_connected=True).find_costs(st)
    out_small = np.full(small.shape, np.inf)
    out_small[r0:r1, c0:c1] = cum
    out = cv2.resize(out_small.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float64) * g
    out[~comp] = np.inf
    return out


def border_pixels(comp):
    edge = np.zeros_like(comp)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    return np.argwhere(comp & edge)


def component_containing(mask, pt):
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    k = lab[pt[0], pt[1]]
    return mask if k == 0 else (lab == k)


def fill_small_holes(mask, max_hole_px):
    inv = (~mask).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    H, W = mask.shape
    out = mask.copy()
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < max_hole_px and x > 0 and y > 0 and x + w < W and y + h < H:
            out[lab == k] = True
    return out


# ----------------------------------------------------------------------------------------------- silhouette consistency
class ViewGeom:
    """Back-projection of native pixels to base-frame rays and projection into a view (declared calibration)."""

    def __init__(self, cal):
        self.Kinv = np.linalg.inv(cal["K"])
        self.R, self.t = cal["E"][:3, :3], cal["E"][:3, 3]
        self.C = -self.R.T @ self.t
        self.fx, self.fy, self.cx, self.cy = cal["fx"], cal["fy"], cal["cx"], cal["cy"]

    def points_along(self, pts, depths):
        """pts (N,2) native px, depths (D,) -> X (N,D,3) base-frame points at those camera depths."""
        h = np.c_[pts, np.ones(len(pts))]
        d = (self.R.T @ (self.Kinv @ h.T)).T  # (N,3): unit-depth directions
        return self.C[None, None] + depths[None, :, None] * d[:, None, :]

    def project(self, X):
        """X (...,3) -> (px, py, z) native."""
        Xc = X @ self.R.T + self.t
        z = Xc[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.fx * Xc[..., 0] / z + self.cx, self.fy * Xc[..., 1] / z + self.cy, z


def sample_points(cand, s, k):
    """k native-pixel sample points of a candidate: centroid, tip (if any) and evenly spaced mask pixels."""
    ys, xs = np.nonzero(cand["mask"])
    idx = np.linspace(0, len(ys) - 1, max(k - 2, 1)).astype(int)
    pts = [(cand["cx"], cand["cy"])] + ([(cand["tip"][0] * s, cand["tip"][1] * s)] if "tip" in cand else []) + \
          [((x + 0.5) * s - 0.5, (y + 0.5) * s - 0.5) for y, x in zip(ys[idx], xs[idx])]
    return np.array(pts[:k], float)


def dt_small(mask, g):
    """Distance (px, at the mask resolution / g) to the nearest mask pixel, on a g-times downsampled grid."""
    H, W = mask.shape
    m = cv2.resize(mask.astype(np.uint8) * 255, (W // g, H // g), interpolation=cv2.INTER_AREA) > 0
    return cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)


def silhouette_frac(pts, geom_from, geom_to, dt_to, depths, a, s, g):
    """Fraction of the sample points of a candidate in view `from` whose epipolar segment (depths inside the reach sphere)
    passes within hull_tol_px (native) of the other view's candidate (dt_to at proc/g resolution). None if fewer than 3
    samples are testable (their segments leave the other image)."""
    X = geom_from.points_along(pts, depths)  # (K,D,3)
    valid = (np.linalg.norm(X, axis=2) < a.reach) & (X[..., 2] > a.zmin)
    px, py, z = geom_to.project(X)
    Hs, Ws = dt_to.shape
    ix, iy = np.floor((px + 0.5) / (s * g)).astype(int), np.floor((py + 0.5) / (s * g)).astype(int)
    inside = valid & np.isfinite(px) & np.isfinite(py) & (z > a.min_cam_depth) & (ix >= 0) & (ix < Ws) & (iy >= 0) & (iy < Hs)
    d = np.full(px.shape, np.inf)
    d[inside] = dt_to[iy[inside], ix[inside]] * s * g
    testable = inside.any(axis=1)
    if testable.sum() < 3:
        return None
    hit = (d <= a.hull_tol_px).any(axis=1)
    return float(hit[testable].mean())


# ----------------------------------------------------------------------------------------------- per-view motion + candidates
class ViewMotion:
    """MOG2 background model + moving-pixel map at processing resolution (as seed_motion_only.py)."""

    def __init__(self, a):
        self.a = a
        self.mog = cv2.createBackgroundSubtractorMOG2(history=a.history, varThreshold=a.var_thresh, detectShadows=not a.no_shadows)
        self.mog.setShadowThreshold(a.shadow_tau)
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.close_px, a.close_px))
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.open_px, a.open_px))
        self.k_move = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.move_dilate_px, a.move_dilate_px))
        self.prev_gray = None
        self.t = 0

    def step(self, bgr):
        """Returns (fg bool, moving bool, moving_dil bool, gray, chroma) or None during warm-up."""
        a = self.a
        gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        chroma = (bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16))
        t = self.t
        self.t += 1
        if t < a.warmup_frames:
            self.mog.apply(bgr, learningRate=a.warmup_lr)
            self.prev_gray = gray
            return None
        fg = self.mog.apply(bgr, learningRate=1.0 / a.history)  # 255 fg, 127 shadow, 0 bg
        fg = ((fg == 255) | ((fg == 127) & (gray < a.dark_thresh) & (chroma < a.chroma_thresh))).astype(np.uint8)
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_open).astype(bool)
        moving = (cv2.absdiff(gray, self.prev_gray) > a.move_thresh) if self.prev_gray is not None else np.zeros_like(fg)
        moving_dil = cv2.dilate(moving.astype(np.uint8), self.k_move).astype(bool)
        self.prev_gray = gray
        return fg, moving, moving_dil, gray, chroma


def candidate_record(mask, kind, parent, gray, chroma, s, a):
    """Native-pixel geometry of a processing-resolution candidate mask."""
    ys, xs = np.nonzero(mask)
    cx, cy = (xs.mean() + 0.5) * s - 0.5, (ys.mean() + 0.5) * s - 0.5
    bw, bh = (xs.max() - xs.min() + 1) * s, (ys.max() - ys.min() + 1) * s
    dark = float(((gray[ys, xs] < a.dark_thresh) & (chroma[ys, xs] < a.chroma_thresh)).mean())
    return dict(kind=kind, parent=parent, mask=mask, cx=float(cx), cy=float(cy), bw=float(bw), bh=float(bh),
                size_px=float(max(bw, bh)), minor_px=float(min(bw, bh)), area=float(len(xs) * s * s), dark=dark)


def candidate_extras(rec, pctx, gray, chroma, a, k_dark):
    """Idea-4 per-candidate quantities (processing resolution, depth-free; converted to metres per pair with the triangulated
    depth): g_max = farthest arm-distance of the candidate (test c); dark body = the candidate's dark-achromatic pixels (opened;
    the candidate itself when empty), g_dark = its median arm-distance; tt_dt = distances (proc px) of the parent's BRIGHT and
    PROXIMAL (arm-distance < g_dark) pixels to the dark body, inside a padded crop (test a)."""
    m, geo = rec["mask"], pctx["geo"]
    fin = np.isfinite(geo) & m
    rec["g_max"] = float(geo[fin].max()) if fin.any() else float("nan")
    dark = (m & (gray < a.dark_thresh) & (chroma < a.chroma_thresh)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k_dark).astype(bool)
    if not dark.any():
        dark = m
    fd = dark & np.isfinite(geo)
    g_dark = float(np.median(geo[fd])) if fd.any() else float("nan")
    rec["g_dark"] = g_dark
    prox = pctx["bright"] & ((geo < g_dark) if np.isfinite(g_dark) else ~m)
    ys, xs = np.nonzero(dark)
    pad = a.twotone_pad_px
    H, W = m.shape
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    sub_dark = dark[y0:y1, x0:x1]
    sub_prox = prox[y0:y1, x0:x1]
    if sub_prox.any():
        dt = cv2.distanceTransform((~sub_dark).astype(np.uint8), cv2.DIST_L2, 3)
        rec["tt_dt"] = np.sort(dt[sub_prox].astype(np.float32))
    else:
        rec["tt_dt"] = np.zeros(0, np.float32)
    rec["dark_body_px"] = int(dark.sum())
    return rec


def extract_candidates(fg, moving, moving_dil, gray, chroma, cal, a, s, px_per_m):
    """Candidates of one view at frame t: fg components with motion, their moving fragments and their distal caps
    (round-3 seeder), plus the idea-4 per-parent context (arm-distance field, moving extent, bright map) and per-candidate
    extras. Returns (parents, cands, ctx)."""
    H, W = fg.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    min_area, min_area_mov = a.min_area_frac * W * H, a.min_area_mov_frac * W * H
    parents, cands, ctx = {}, [], {}
    k_dark = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.dark_open_px, a.dark_open_px))
    bright_map = (gray >= a.bright_thresh) & (chroma <= a.bright_chroma_max)
    base_pt = None
    if cal["base_px"] is not None:
        base_pt = ((cal["base_px"][1] + 0.5) / s - 0.5, (cal["base_px"][0] + 0.5) / s - 0.5)  # (row, col) at proc res, may be outside
    rr, cc = np.mgrid[0:H, 0:W]
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = lab == k
        mv = comp & moving
        if not mv.any():
            continue  # a ghost / an object the arm moved earlier: no longer moving
        parents[k] = comp
        # arm-distance field: geodesic from where the component enters the image (round-3 roots), Euclidean fallback
        roots = border_pixels(comp)
        if len(roots) == 0:
            rs, cs = np.nonzero(comp)
            if base_pt is not None:
                i = int(np.argmin((rs - base_pt[0]) ** 2 + (cs - base_pt[1]) ** 2))
            else:
                i = int(np.argmin(rs))  # top-most pixel: the arm enters from above when the base is not projectable
            roots = np.array([[rs[i], cs[i]]])
        geo = geodesic_small(comp, roots, a.geo_grid)
        field = "geodesic"
        if geo is None:
            if base_pt is not None:
                geo = np.where(comp, np.hypot(rr - base_pt[0], cc - base_pt[1]), np.inf); field = "euclid_base"
            else:
                geo = np.where(comp, rr.astype(np.float64), np.inf); field = "row"
        mv_dil = comp & moving_dil
        fm = mv_dil & np.isfinite(geo)
        pctx = dict(geo=geo, field=field, g_mov_max=float(geo[fm].max()) if fm.any() else float("nan"),
                    bright=comp & bright_map, n_bright=int((comp & bright_map).sum()))
        ctx[k] = pctx
        cands.append(candidate_extras(candidate_record(comp, "fg", k, gray, chroma, s, a), pctx, gray, chroma, a, k_dark))
        # moving fragments inside the component (the max_frag largest)
        frag = mv_dil.astype(np.uint8)
        nf, labf, statsf, _ = cv2.connectedComponentsWithStats(frag, connectivity=8)
        order = sorted(range(1, nf), key=lambda j: -statsf[j, cv2.CC_STAT_AREA])[:a.max_frag]
        for j in order:
            if statsf[j, cv2.CC_STAT_AREA] >= min_area_mov:
                cands.append(candidate_extras(candidate_record(labf == j, "mov", k, gray, chroma, s, a), pctx, gray, chroma, a, k_dark))
        # dark-achromatic components inside the component (the black gripper on a white arm), opened to drop cables
        dark = (comp & (gray < a.dark_thresh) & (chroma < a.chroma_thresh)).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k_dark)
        nd, labd, statsd, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        order = sorted(range(1, nd), key=lambda j: -statsd[j, cv2.CC_STAT_AREA])[:a.max_frag]
        for j in order:
            if statsd[j, cv2.CC_STAT_AREA] >= min_area_mov:
                cands.append(candidate_extras(candidate_record(fill_small_holes(labd == j, a.max_hole_px), "dark", k, gray, chroma, s, a), pctx, gray, chroma, a, k_dark))
        # distal caps along the component (round-3: only with a geodesic field)
        if field != "geodesic":
            continue
        tip_sel = mv & np.isfinite(geo)
        if not tip_sel.any():
            continue
        tip = tuple(int(v) for v in np.unravel_index(np.argmax(np.where(tip_sel, geo, -1.0)), geo.shape))
        geo_tip = geodesic_small(comp, np.array([tip]), a.geo_grid)
        if geo_tip is None:
            continue
        comp_area = int(comp.sum())
        for L_m in a.cap_lengths:
            cap = comp & (geo_tip <= L_m * px_per_m)
            if not cap.any():
                continue
            cap = fill_small_holes(component_containing(cap, tip), a.max_hole_px)
            if cap.sum() < min_area_mov or cap.sum() >= 0.95 * comp_area:
                continue
            rec = candidate_record(cap, f"cap{L_m:g}", k, gray, chroma, s, a)
            rec["tip"] = [tip[1], tip[0]]
            cands.append(candidate_extras(rec, pctx, gray, chroma, a, k_dark))
    return parents, cands, ctx


# ----------------------------------------------------------------------------------------------- 3D verification + selection
def idea4_tests(rec, u, v, cal1, cal2, ctx1, ctx2, a, s):
    """Tests (a) two-tone, (b) size-at-depth, (c) terminal node on a pair that passed the round-3 gates. Per-view values are
    converted to metres with the pair's triangulated depth in that view (rec['depths']). Fills rec['twotone_m'],
    rec['ok_twotone'], rec['minor_m'], rec['ok_size2'], rec['beyond_m'], rec['ok_terminal'], rec['fails']."""
    fails = []
    tt, ok_tt = [], []
    minor, ok_sz = [], []
    beyond, ok_tm = [], []
    for cand, cal, ctx, z in ((u, cal1, ctx1, rec["depths"][0]), (v, cal2, ctx2, rec["depths"][1])):
        f, pc = cal["fx"], ctx[cand["parent"]]
        m_per_ppx = s * z / f                      # metres per processing-resolution pixel at this view's triangulated depth
        # (a) two-tone: bright proximal pixels within twotone_adj_m of the dark body -> equivalent diameter
        band_px = a.twotone_adj_m / m_per_ppx
        n_px = int(np.searchsorted(cand["tt_dt"], band_px, side="right")) if len(cand["tt_dt"]) else 0
        d_m = float(np.sqrt(n_px) * m_per_ppx)
        tt.append(round(d_m, 4)); ok_tt.append(bool(a.twotone_min <= d_m <= a.twotone_max))
        # (b) size-at-depth: longer side in [size2_min, size_max], shorter side >= size2_minor
        long_m, minor_m = cand["size_px"] * z / f, cand["minor_px"] * z / f
        minor.append(round(float(minor_m), 4))
        ok_sz.append(bool(a.size2_min <= long_m <= a.size_max and minor_m >= a.size2_minor))
        # (c) terminal node: moving foreground of the parent beyond the candidate's farthest pixel
        b = (pc["g_mov_max"] - cand["g_max"]) * m_per_ppx if (np.isfinite(pc["g_mov_max"]) and np.isfinite(cand["g_max"])) else float("nan")
        beyond.append(None if not np.isfinite(b) else round(float(b), 4))
        ok_tm.append(bool((not np.isfinite(b)) or b <= a.terminal_m))
    rec["twotone_m"], rec["minor_m"], rec["beyond_m"] = tt, minor, beyond
    rec["ok_twotone"] = bool(sum(ok_tt) >= a.twotone_views)
    rec["ok_size2"] = bool(all(ok_sz))
    rec["ok_terminal"] = bool(all(ok_tm))
    if a.twotone and not rec["ok_twotone"]:
        fails.append("twotone")
    if a.size2 and not rec["ok_size2"]:
        fails.append("size2")
    if a.terminal and not rec["ok_terminal"]:
        fails.append("terminal")
    rec["fails"] = fails
    return not fails


def verify_pairs(c1, c2, cal1, cal2, geom1, geom2, ctx1, ctx2, a, s):
    """All cross-view candidate pairs with their triangulation and gate values; 'ok_base' when every round-3 gate passes,
    'ok' when the enabled idea-4 tests pass as well. The silhouette test (the expensive gate) runs only on pairs that pass
    the cheap gates; the idea-4 tests only on pairs that pass the round-3 gates; distance transforms are cached."""
    out = []
    depths = np.linspace(a.min_cam_depth, a.max_depth, a.depth_samples)
    dts1, dts2, pts1, pts2 = {}, {}, {}, {}
    for i, u in enumerate(c1):
        for j, v in enumerate(c2):
            X, r, rv = triangulate(cal1["P"], cal2["P"], (u["cx"], u["cy"]), (v["cx"], v["cy"]))
            dist = float(np.linalg.norm(X))
            rec = dict(i=i, j=j, kinds=[u["kind"], v["kind"]], X=[round(float(x), 4) for x in X], r=round(r, 2), dist=round(dist, 3),
                       ok_reach=bool(dist < a.reach), ok_z=bool(X[2] > a.zmin), sizes=[], depths=[], ok_size=True, ok_front=True, ok_ray=True,
                       dark=[round(u["dark"], 2), round(v["dark"], 2)], ok_dark=bool(min(u["dark"], v["dark"]) >= a.dark_min), hull=[None, None], ok_hull=None)
            for cand, cal, rr in ((u, cal1, rv[0]), (v, cal2, rv[1])):
                _, _, z = project(cal, X)
                rec["depths"].append(round(z, 3))
                if z < a.min_cam_depth:
                    rec["ok_front"] = False
                    rec["sizes"].append(None)
                    continue
                if rr > a.ray_tol_m * cal["fx"] / z or (a.r_max_px > 0 and rr > a.r_max_px):
                    rec["ok_ray"] = False
                size_m = cand["size_px"] * z / cal["fx"]
                rec["sizes"].append(round(float(size_m), 3))
                if not (a.size_min <= size_m <= a.size_max):
                    rec["ok_size"] = False
            cheap = rec["ok_ray"] and rec["ok_reach"] and rec["ok_z"] and rec["ok_size"] and rec["ok_front"] and rec["ok_dark"]
            if cheap and a.hull_min > 0:
                if i not in pts1:
                    pts1[i], dts1[i] = sample_points(u, s, a.sample_pts), dt_small(u["mask"], a.hull_grid)
                if j not in pts2:
                    pts2[j], dts2[j] = sample_points(v, s, a.sample_pts), dt_small(v["mask"], a.hull_grid)
                f12 = silhouette_frac(pts1[i], geom1, geom2, dts2[j], depths, a, s, a.hull_grid)
                f21 = silhouette_frac(pts2[j], geom2, geom1, dts1[i], depths, a, s, a.hull_grid)
                rec["hull"] = [None if f12 is None else round(f12, 2), None if f21 is None else round(f21, 2)]
                fr = [f for f in (f12, f21) if f is not None]
                thr = a.hull_min if len(fr) == 2 else a.hull_min_single  # one testable direction only: stricter
                rec["ok_hull"] = bool(fr) and all(f >= thr for f in fr)
            rec["ok_base"] = bool(cheap and (a.hull_min <= 0 or rec["ok_hull"]))
            rec["ok"] = bool(idea4_tests(rec, u, v, cal1, cal2, ctx1, ctx2, a, s)) if rec["ok_base"] else False
            out.append(rec)
    return out


class Selector:
    """Distal preference + temporal consistency over the verified pairs (causal state: last pick only). Unchanged."""

    def __init__(self, a):
        self.a = a
        self.X_prev, self.t_prev = None, None
        self.tracks = []

    def pick(self, t, verified):
        a = self.a
        if not verified:
            return None, "none"
        alive = self.X_prev is not None and (t - self.t_prev) <= a.coast_frames
        mode = "new"
        if alive:
            gap = t - self.t_prev
            cons = [v for v in verified if np.linalg.norm(np.array(v["X"]) - self.X_prev) <= a.jump_m * gap]
            if cons:
                best, mode = max(cons, key=lambda v: v["dist"]), "track"
            else:
                best, mode = max(verified, key=lambda v: v["dist"]), "switch"
        else:
            best = max(verified, key=lambda v: v["dist"])
        if mode != "track":
            self.tracks.append(dict(track=len(self.tracks), frame=int(t), mode=mode, **{k: best[k] for k in ("kinds", "X", "r", "dist", "sizes", "depths", "dark", "hull", "twotone_m", "minor_m", "beyond_m")}))
        self.X_prev, self.t_prev = np.array(best["X"]), t
        return best, mode


def emit_mask(cand, parent, cal, X, a, s):
    """Native-resolution mask: the parent component (or the candidate) cropped to the box_m box around the projection of X.
    Returns (mask_native bool, box [x0, y0, x1, y1], (px, py, z))."""
    px, py, z = project(cal, X)
    src = parent if a.emit == "parent" else cand["mask"]
    nat = cv2.resize(src.astype(np.uint8), (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_NEAREST).astype(bool)
    hx, hy = 0.5 * a.box_m * cal["fx"] / z, 0.5 * a.box_m * cal["fy"] / z
    x0, x1 = int(max(0, np.floor(px - hx))), int(min(NATIVE_W, np.ceil(px + hx) + 1))
    y0, y1 = int(max(0, np.floor(py - hy))), int(min(NATIVE_H, np.ceil(py + hy) + 1))
    out = np.zeros_like(nat)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = nat[y0:y1, x0:x1]
    return out, [x0, y0, x1, y1], (px, py, z)


# ----------------------------------------------------------------------------------------------- video io
def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open {path}")
    return cap


def read_pair(caps, proc_w, proc_h):
    frames = {}
    for c, cap in caps.items():
        ok, f = cap.read()
        if not ok:
            return None
        frames[c] = (f, cv2.resize(f, (proc_w, proc_h), interpolation=cv2.INTER_AREA))
    return frames


def write_overlay(method, cams, serials, videos, packed, logs, out_dir, fps, png_frames, T):
    """Streaming two-panel overlay: emitted mask (green), crop box (magenta), candidate centroids (grey = rejected,
    yellow = verified, red = picked, blue = passed round 3 but failed an idea-4 test), projected 3D point (red cross)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sam3_gripper_masks import FfmpegWriter
    pw, ph = 640, 360
    path = os.path.join(out_dir, f"{method}_ext_side_by_side.mp4")
    writer = FfmpegWriter(path, pw * len(cams), ph, fps)
    caps = {c: open_video(videos[c]) for c in cams}
    for t in range(T):
        fr = read_pair(caps, pw, ph)
        if fr is None:
            break
        info = logs[t]
        panels = []
        for c in cams:
            img = fr[c][0].copy()
            m = np.unpackbits(packed[c][t], axis=1)[:, :NATIVE_W].astype(bool)
            if m.any():
                ov = img.copy()
                ov[m] = (0, 255, 0)
                img = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
            for cd in info["cands"].get(c, []):
                col = (0, 0, 255) if cd["picked"] else ((0, 255, 255) if cd["verified"] else ((255, 128, 0) if cd.get("base") else (160, 160, 160)))
                cv2.circle(img, (int(cd["cx"]), int(cd["cy"])), 8 if cd["picked"] else 5, col, -1)
            box = info.get("box", {}).get(c)
            if box is not None:
                cv2.rectangle(img, (box[0], box[1]), (box[2] - 1, box[3] - 1), (255, 0, 255), 2)
                px, py = info["proj"][c][:2]
                cv2.drawMarker(img, (int(px), int(py)), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            tst = info.get("tests", {})
            hdr = f"{c}/{serials[c]} {method} f{t}/{T - 1} {info['mode']} nc={len(info['cands'].get(c, []))} base={tst.get('n_base')} ok={info['n_verified']}"
            if info.get("X") is not None:
                hdr += f" |X|={info['dist']:.2f} sz={info['sizes']} tt={info.get('twotone_m')} bey={info.get('beyond_m')}"
            elif tst.get("rejected_best"):
                hdr += f" REJ {tst['rejected_best']['kinds']} {tst['rejected_best']['fails']}"
            cv2.rectangle(img, (0, 0), (NATIVE_W, 26), (0, 0, 0), -1)
            cv2.putText(img, hdr, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA))
        row = np.concatenate(panels, axis=1)
        writer.write(row)
        if t in png_frames:
            cv2.imwrite(os.path.join(out_dir, f"{method}_check_f{t:04d}.png"), row)
    writer.close()
    for cap in caps.values():
        cap.release()
    return path


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--method", default=METHOD_DEFAULT, help="mask file suffix / method name (ablations: cv_twotone, cv_size2, cv_terminal)")
    ap.add_argument("--proc_w", type=int, default=640, help="processing width (height = 9/16); geometry and outputs are native 1280x720")
    # background model (seed_motion_only defaults)
    ap.add_argument("--history", type=int, default=300)
    ap.add_argument("--var_thresh", type=float, default=25.0)
    ap.add_argument("--no_shadows", action="store_true")
    ap.add_argument("--shadow_tau", type=float, default=0.3)
    ap.add_argument("--dark_thresh", type=int, default=70)
    ap.add_argument("--chroma_thresh", type=int, default=15)
    ap.add_argument("--warmup_frames", type=int, default=5)
    ap.add_argument("--warmup_lr", type=float, default=0.3)
    ap.add_argument("--close_px", type=int, default=9)
    ap.add_argument("--open_px", type=int, default=5)
    ap.add_argument("--move_thresh", type=int, default=10)
    ap.add_argument("--move_dilate_px", type=int, default=7)
    # candidates
    ap.add_argument("--min_area_frac", type=float, default=0.0015, help="foreground components below this image fraction are dropped")
    ap.add_argument("--min_area_mov_frac", type=float, default=0.0004, help="moving fragments / caps below this image fraction are dropped")
    ap.add_argument("--cap_lengths", default="0.1,0.15,0.2,0.25", help="distal cap lengths (m, via the plausible depth)")
    ap.add_argument("--geo_grid", type=int, default=2, help="geodesics on a grid downsampled by this factor")
    ap.add_argument("--min_depth", type=float, default=0.35, help="floor on the plausible depth for the cap pixel scale")
    ap.add_argument("--max_hole_px", type=int, default=400)
    ap.add_argument("--max_frag", type=int, default=8, help="keep at most this many moving / dark fragments per component (largest first)")
    ap.add_argument("--dark_open_px", type=int, default=5, help="opening of the dark-achromatic map before its components (drops cables)")
    # 3D verification (round 3, unchanged)
    ap.add_argument("--ray_tol_m", type=float, default=0.04, help="the two centroid rays must meet within this (m): per-view residual < ray_tol_m * f / depth")
    ap.add_argument("--r_max_px", type=float, default=0.0, help="additional hard cap on the per-view residual (px, native); 0 = none")
    ap.add_argument("--reach", type=float, default=0.9, help="max distance of the point from the base origin (m)")
    ap.add_argument("--zmin", type=float, default=-0.15, help="min height of the point in the base frame (m, z up)")
    ap.add_argument("--size_min", type=float, default=0.07, help="round-3 apparent size window (m): longer bbox side * depth / f")
    ap.add_argument("--size_max", type=float, default=0.30)
    ap.add_argument("--min_cam_depth", type=float, default=0.15, help="the point must be at least this far in front of each camera (m)")
    ap.add_argument("--max_depth", type=float, default=1.8, help="epipolar segments are sampled up to this camera depth (m)")
    ap.add_argument("--depth_samples", type=int, default=40)
    ap.add_argument("--dark_min", type=float, default=0.15, help="both candidates must be dark-achromatic in at least this fraction of their pixels")
    ap.add_argument("--hull_min", type=float, default=0.6, help="mutual silhouette consistency threshold (fraction of samples); 0 disables the test")
    ap.add_argument("--hull_min_single", type=float, default=0.85, help="threshold when only one direction is testable")
    ap.add_argument("--hull_tol_px", type=float, default=8.0, help="a sample hits when its epipolar segment passes within this of the other mask (px, native)")
    ap.add_argument("--sample_pts", type=int, default=12)
    ap.add_argument("--hull_grid", type=int, default=2, help="distance transforms on the processing grid downsampled by this")
    # idea 4: cheap verifiers (declared constants, see the docstring)
    ap.add_argument("--twotone", type=int, default=1, help="(a) two-tone signature test on/off")
    ap.add_argument("--bright_thresh", type=int, default=140, help="(a) bright = luminance >= this (light housing / white Franka flange)")
    ap.add_argument("--bright_chroma_max", type=int, default=100, help="(a) bright = chroma (max-min channel) <= this (white to light tan)")
    ap.add_argument("--twotone_adj_m", type=float, default=0.04, help="(a) adjacency band around the dark body (m-equivalent at the triangulated depth)")
    ap.add_argument("--twotone_min", type=float, default=0.03, help="(a) equivalent diameter window of the bright proximal band (m)")
    ap.add_argument("--twotone_max", type=float, default=0.08)
    ap.add_argument("--twotone_views", type=int, default=1, help="(a) number of views that must pass (1: the housing may be hidden in one view)")
    ap.add_argument("--twotone_pad_px", type=int, default=60, help="(a) crop padding for the distance transform (proc px; 4 cm at 0.25 m is 42 px)")
    ap.add_argument("--size2", type=int, default=1, help="(b) size-at-depth test on/off")
    ap.add_argument("--size2_min", type=float, default=0.08, help="(b) longer bbox side >= this (m) at the triangulated depth, per view")
    ap.add_argument("--size2_minor", type=float, default=0.03, help="(b) shorter bbox side >= this (m), per view")
    ap.add_argument("--terminal", type=int, default=1, help="(c) terminal-node test on/off")
    ap.add_argument("--terminal_m", type=float, default=0.10, help="(c) moving foreground may extend at most this beyond the candidate along the arm (m-equivalent)")
    ap.add_argument("--postfilter", type=int, default=1, help="(d) restrict masks to frames with an accepted pick (verified; a no-op by construction here)")
    # selection + emission (round 3, unchanged)
    ap.add_argument("--jump_m", type=float, default=0.12, help="temporal consistency radius per frame of gap (m)")
    ap.add_argument("--coast_frames", type=int, default=15, help="a track stays alive this many frames without a pick")
    ap.add_argument("--box_m", type=float, default=0.30, help="crop box side, metres-equivalent at the point's depth")
    ap.add_argument("--emit", default="parent", choices=["parent", "candidate"], help="what the box crops: the foreground component holding the pick, or the pick itself")
    # outputs
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--png_frames", default="", help="comma list of frame indices dumped as check PNGs (needs --video)")
    ap.add_argument("--log_pairs", action="store_true", help="keep every pair's verification record in the per-frame log (large)")
    a = ap.parse_args()
    a.cap_lengths = [float(x) for x in a.cap_lengths.split(",") if x.strip()]
    method = a.method

    ep = a.episode
    raw = os.path.join(RAW_ROOT, ep)
    meta = json.load(open(os.path.join(raw, f"metadata_{ep}.json")))
    cams = list(CAMS)
    serials = {c: str(meta[f"{c}_cam_serial"]) for c in cams}
    videos = {c: os.path.join(raw, "recordings", "MP4", f"{serials[c]}.mp4") for c in cams}
    for c, p in videos.items():
        if not os.path.exists(p):
            sys.exit(f"missing video for {c}: {p}")
    out_dir = os.path.join(a.out_root, ep)
    os.makedirs(out_dir, exist_ok=True)
    proc_w, proc_h = a.proc_w, a.proc_w * 9 // 16
    s = NATIVE_W / proc_w
    T_store = len([f for f in os.listdir(os.path.join(STORE_ROOT, ep, "dense", "cam")) if f.endswith(".npz")])
    cal = load_calib(ep, serials)
    px_per_m = {c: cal[c]["fx"] / s / max(cal[c]["plausible_depth"], a.min_depth) for c in cams}
    print(f"{TOOL_VERSION}\nepisode {ep}\nvideos {videos}\nstore frames {T_store}\nout {out_dir}\ntests twotone={a.twotone} size2={a.size2} terminal={a.terminal} postfilter={a.postfilter}", flush=True)
    for c in cams:
        k = cal[c]
        print(f"[{c}/{serials[c]}] fx={k['fx']:.1f} base_px={k['base_px']} cam_to_base={k['cam_to_base']:.2f} z_grid={k['z_grid_median']:.2f} "
              f"plausible_depth={k['plausible_depth']:.2f} -> {px_per_m[c]:.0f} px/m at {proc_w}", flush=True)

    t_all = time.time()
    views = {c: ViewMotion(a) for c in cams}
    caps = {c: open_video(videos[c]) for c in cams}
    sel = Selector(a)
    geom = {c: ViewGeom(cal[c]) for c in cams}
    packed = {c: np.zeros((T_store, NATIVE_H, NATIVE_W // 8), np.uint8) for c in cams}
    present = {c: np.zeros(T_store, bool) for c in cams}
    logs = []
    modes = {}
    n_pairs_total = n_base_total = n_verified_total = 0
    test_counts = dict(pairs_base=0, fail_twotone=0, fail_size2=0, fail_terminal=0, frames_base_nonempty=0, frames_all_rejected=0,
                       fail_twotone_only=0, fail_size2_only=0, fail_terminal_only=0)
    t_motion = t_verify = 0.0
    t = 0
    while t < T_store:
        fr = read_pair(caps, proc_w, proc_h)
        if fr is None:
            break
        t0 = time.time()
        info = dict(t=t, mode="warmup", cands={}, n_pairs=0, n_verified=0)
        cands, parents, ctxs = {}, {}, {}
        for c in cams:
            st = views[c].step(fr[c][1])
            if st is None:
                continue
            fg, moving, moving_dil, gray, chroma = st
            parents[c], cands[c], ctxs[c] = extract_candidates(fg, moving, moving_dil, gray, chroma, cal[c], a, s, px_per_m[c])
        t1 = time.time()
        t_motion += t1 - t0
        if all(c in cands for c in cams):
            pairs = verify_pairs(cands[cams[0]], cands[cams[1]], cal[cams[0]], cal[cams[1]], geom[cams[0]], geom[cams[1]], ctxs[cams[0]], ctxs[cams[1]], a, s)
            base = [p for p in pairs if p["ok_base"]]
            verified = [p for p in base if p["ok"]]
            n_pairs_total += len(pairs)
            n_base_total += len(base)
            n_verified_total += len(verified)
            test_counts["pairs_base"] += len(base)
            for p in base:
                for f_ in p["fails"]:
                    test_counts[f"fail_{f_}"] += 1
                if len(p["fails"]) == 1:
                    test_counts[f"fail_{p['fails'][0]}_only"] += 1
            best, mode = sel.pick(t, verified)
            tests = dict(n_base=len(base), n_ok=len(verified),
                         n_fail_twotone=sum(1 for p in base if "twotone" in p["fails"]), n_fail_size2=sum(1 for p in base if "size2" in p["fails"]),
                         n_fail_terminal=sum(1 for p in base if "terminal" in p["fails"]))
            if base:
                test_counts["frames_base_nonempty"] += 1
                if not verified:
                    test_counts["frames_all_rejected"] += 1
                    rb = max(base, key=lambda v: v["dist"])   # the pair the round-3 Selector would have taken (most distal; no track state)
                    tests["rejected_best"] = dict(kinds=rb["kinds"], X=rb["X"], dist=rb["dist"], fails=rb["fails"], twotone_m=rb["twotone_m"],
                                                  minor_m=rb["minor_m"], beyond_m=rb["beyond_m"], sizes=rb["sizes"])
            info.update(mode=mode, n_pairs=len(pairs), n_verified=len(verified), tests=tests)
            if a.log_pairs:
                info["pairs"] = pairs
            picked_idx = {cams[0]: best["i"], cams[1]: best["j"]} if best else {}
            ver_idx = {cams[0]: {p["i"] for p in verified}, cams[1]: {p["j"] for p in verified}}
            base_idx = {cams[0]: {p["i"] for p in base}, cams[1]: {p["j"] for p in base}}
            for c in cams:
                info["cands"][c] = [dict(kind=cd["kind"], cx=round(cd["cx"], 1), cy=round(cd["cy"], 1), size_px=round(cd["size_px"], 1),
                                         dark=round(cd["dark"], 2), verified=bool(i in ver_idx[c]), base=bool(i in base_idx[c]), picked=bool(picked_idx.get(c) == i))
                                    for i, cd in enumerate(cands[c])]
            if best is not None:
                X = np.array(best["X"])
                info.update(X=best["X"], r=best["r"], dist=best["dist"], sizes=best["sizes"], depths=best["depths"], kinds=best["kinds"], dark=best["dark"], hull=best["hull"],
                            twotone_m=best["twotone_m"], minor_m=best["minor_m"], beyond_m=best["beyond_m"],
                            ok_twotone=best["ok_twotone"], ok_size2=best["ok_size2"], ok_terminal=best["ok_terminal"],
                            field=[ctxs[c][cands[c][idx]["parent"]]["field"] for c, idx in picked_idx.items()], box={}, proj={})
                for c, idx in picked_idx.items():
                    cd = cands[c][idx]
                    m, box, proj = emit_mask(cd, parents[c][cd["parent"]], cal[c], X, a, s)
                    packed[c][t] = np.packbits(m, axis=1)
                    present[c][t] = bool(m.any())
                    info["box"][c] = box
                    info["proj"][c] = [round(proj[0], 1), round(proj[1], 1), round(proj[2], 3)]
                    info.setdefault("area", {})[c] = int(m.sum())
        t_verify += time.time() - t1
        modes[info["mode"]] = modes.get(info["mode"], 0) + 1
        logs.append(info)
        if t % 100 == 0:
            print(f"  f{t}: mode={info['mode']} cands={[len(info['cands'].get(c, [])) for c in cams]} pairs={info['n_pairs']} base={info.get('tests', {}).get('n_base')} "
                  f"ok={info['n_verified']} X={info.get('X')} sizes={info.get('sizes')} tt={info.get('twotone_m')} beyond={info.get('beyond_m')} ({time.time() - t_all:.0f}s)", flush=True)
        t += 1
    for cap in caps.values():
        cap.release()
    T_mp4 = t
    if T_mp4 < T_store:
        print(f"WARNING: mp4 pair yielded {T_mp4} frames < store {T_store}; remaining frames are empty", flush=True)
        for k in range(T_mp4, T_store):
            logs.append(dict(t=k, mode="no_frame", cands={}, n_pairs=0, n_verified=0))
    t_seed = time.time() - t_all

    # (d) post-filter: masks only on frames whose pick passed every enabled test (accepted = a pick exists at t)
    accepted = np.zeros(T_store, bool)
    for lg in logs:
        if lg.get("X") is not None and 0 <= lg["t"] < T_store:
            accepted[lg["t"]] = True
    postfilter = dict(enabled=bool(a.postfilter), n_removed={}, note="masks are emitted only for accepted picks, so the post-filter removes nothing by construction; verified here")
    for c in cams:
        rm = present[c] & ~accepted
        postfilter["n_removed"][c] = int(rm.sum())
        if a.postfilter and rm.any():
            packed[c][rm] = 0
            present[c][rm] = False

    products, stats = {}, {}
    for c in cams:
        path = os.path.join(out_dir, f"{c}_{serials[c]}__{method}_masks.npz")
        np.savez_compressed(path, union=packed[c], shape=np.array([T_store, NATIVE_H, NATIVE_W]), frames_present=present[c])
        chk = np.load(path)
        u = np.unpackbits(chk["union"], axis=2)[:, :, :NATIVE_W].astype(bool)
        nonempty = u.reshape(T_store, -1).any(1)
        ok = tuple(int(v) for v in chk["shape"]) == (T_store, NATIVE_H, NATIVE_W) and u.shape == (T_store, NATIVE_H, NATIVE_W) \
            and bool((chk["frames_present"] == nonempty).all()) and int(nonempty.sum()) == int(present[c].sum())
        pr = present[c]
        products[c] = dict(npz=path, T=T_store, store_frames=T_store, mp4_frames_read=T_mp4, bytes=os.path.getsize(path), frames_present=int(pr.sum()), product_ok=bool(ok))
        areas = [lg.get("area", {}).get(c) for lg in logs if lg.get("area", {}).get(c)]
        stats[c] = dict(cam=c, serial=serials[c], n_frames=T_store, frames_present=int(pr.sum()), frac_present=float(pr.mean()) if T_store else 0.0,
                        first_present=int(np.argmax(pr)) if pr.any() else None, last_present=int(T_store - 1 - np.argmax(pr[::-1])) if pr.any() else None,
                        mean_area_frac_present=float(np.mean(areas) / (NATIVE_W * NATIVE_H)) if areas else None,
                        calib=dict(fx=round(cal[c]["fx"], 1), fy=round(cal[c]["fy"], 1), base_px=cal[c]["base_px"], cam_to_base=round(cal[c]["cam_to_base"], 3),
                                   z_grid_median=round(cal[c]["z_grid_median"], 3), plausible_depth=round(cal[c]["plausible_depth"], 3), px_per_m_proc=round(px_per_m[c], 1)))
        print(f"[{c}] wrote {path} T={T_store} present={int(pr.sum())} ok={ok}", flush=True)
    # per-frame verified 3D point (base frame, metres) of the picked pair; NaN / accepted=False where no pick.
    vp_xyz = np.full((T_store, 3), np.nan)
    vp_acc = np.zeros(T_store, bool)
    for lg in logs:
        if lg.get("X") is not None and 0 <= lg["t"] < T_store:
            vp_xyz[lg["t"]] = np.array(lg["X"], float)
            vp_acc[lg["t"]] = True
    vp_path = os.path.join(out_dir, "verified_points.npz")
    np.savez(vp_path, frames=np.arange(T_store, dtype=np.int32), xyz=vp_xyz, accepted=vp_acc,
             note=f"{method}: base-frame (m) DLT point of the picked pair per frame after the idea-4 tests; NaN / accepted=False where "
                  "nothing passed; causal (frame t uses frames <= t); calibration only, no GT")
    chk = np.load(vp_path)
    vp_ok = (chk["xyz"].shape == (T_store, 3) and chk["accepted"].shape == (T_store,) and len(chk["frames"]) == T_store
             and bool((np.isfinite(chk["xyz"]).all(1) == chk["accepted"]).all()) and int(chk["accepted"].sum()) == sum(1 for lg in logs if lg.get("X") is not None))
    verified_points = dict(npz=vp_path, T=T_store, accepted=int(vp_acc.sum()), bytes=os.path.getsize(vp_path), product_ok=bool(vp_ok))
    print(f"[verified_points] wrote {vp_path} T={T_store} accepted={int(vp_acc.sum())} ok={vp_ok}", flush=True)
    picked_kinds = {}
    for lg in logs:
        if lg.get("kinds"):
            key = "+".join(lg["kinds"])
            picked_kinds[key] = picked_kinds.get(key, 0) + 1
    rejected_kinds, rejected_fails = {}, {}
    for lg in logs:
        rb = lg.get("tests", {}).get("rejected_best")
        if rb:
            key = "+".join(rb["kinds"]); rejected_kinds[key] = rejected_kinds.get(key, 0) + 1
            fk = "+".join(rb["fails"]); rejected_fails[fk] = rejected_fails.get(fk, 0) + 1
    video_path = None
    t_video = 0.0
    if a.video:
        png = [int(x) for x in a.png_frames.split(",") if x.strip()]
        t0 = time.time()
        video_path = write_overlay(method, cams, serials, videos, packed, logs, out_dir, a.fps, png, T_store)
        t_video = time.time() - t0
        print(f"overlay {video_path} in {t_video:.1f}s", flush=True)
    total = time.time() - t_all
    n_picked = sum(1 for lg in logs if lg.get("X") is not None)
    info = dict(episode=ep, method=method, tool=TOOL_VERSION, serials=serials, params=vars(a), proc_res=[proc_w, proc_h], native_res=[NATIVE_W, NATIVE_H],
                store_frames=T_store, mp4_frames_read=T_mp4, causal=True, uses_gt=False, training_free=True,
                calibration_use="PointWorld optimized_extrinsics + factory intrinsics: cross-view triangulation, reach/height/size gates, crop box, "
                                "metre conversion of the idea-4 tests at the triangulated depth; plausible depth for the candidate cap lengths only",
                declared_constants=dict(
                    object_prior="DROID wrist = black Robotiq 2F-85 body with a light-coloured (white / light-tan) camera housing and the white Franka flange directly above it; "
                                 "Franka links white with black joint rings; the gripper is the terminal node of the arm",
                    bright=dict(luminance_min=a.bright_thresh, chroma_max=a.bright_chroma_max),
                    twotone=dict(adj_m=a.twotone_adj_m, equiv_diameter_window_m=[a.twotone_min, a.twotone_max], views_required=a.twotone_views,
                                 dark_body="candidate's dark-achromatic pixels (dark_thresh %d, chroma_thresh %d, opened %d px)" % (a.dark_thresh, a.chroma_thresh, a.dark_open_px)),
                    size2=dict(long_side_window_m=[a.size2_min, a.size_max], minor_side_min_m=a.size2_minor, views_required=2),
                    terminal=dict(beyond_max_m=a.terminal_m, views_required=2, arm_distance="geodesic inside the parent foreground component from its image entry (border contacts, else nearest the projected base, else top-most); Euclidean fallback"),
                    source="task specification + hardware dimensions (Robotiq 2F-85 ~8.5 x 15 cm incl. housing, ZED Mini 12.5 x 3 cm); not tuned on any scene"),
                tests_enabled=dict(twotone=bool(a.twotone), size2=bool(a.size2), terminal=bool(a.terminal), postfilter=bool(a.postfilter)),
                verification=dict(ray_tol_m=a.ray_tol_m, r_max_px=a.r_max_px, reach_m=a.reach, zmin_m=a.zmin, size_window_m=[a.size_min, a.size_max],
                                  dark_min=a.dark_min, hull_min=a.hull_min, hull_min_single=a.hull_min_single, hull_tol_px=a.hull_tol_px, box_m=a.box_m,
                                  jump_m_per_frame=a.jump_m, coast_frames=a.coast_frames, emit=a.emit),
                summary=dict(frames_picked=n_picked, frac_picked=round(n_picked / max(T_store, 1), 4), modes=modes, picked_kinds=picked_kinds,
                             n_tracks=len(sel.tracks), pairs_tested=n_pairs_total, pairs_verified_round3=n_base_total, pairs_verified=n_verified_total,
                             frames_with_any_round3_verified=sum(1 for lg in logs if lg.get("tests", {}).get("n_base", 0) > 0),
                             frames_with_any_verified=sum(1 for lg in logs if lg["n_verified"] > 0),
                             test_counts=test_counts, rejected_best_kinds=rejected_kinds, rejected_best_fails=rejected_fails),
                postfilter=postfilter,
                tracks=sel.tracks, per_cam=stats, products=products, verified_points=verified_points, overlay_mp4=video_path,
                timing=dict(total_seconds=round(total, 1), seeding_seconds=round(t_seed, 1), motion_and_candidates_seconds=round(t_motion, 1),
                            verification_seconds=round(t_verify, 1), overlay_seconds=round(t_video, 1), cpu_seconds_process=round(time.process_time(), 1),
                            ms_per_frame_seeding=round(1000 * t_seed / max(T_mp4, 1), 1)),
                per_frame=logs)
    json.dump(info, open(os.path.join(out_dir, "seed_info.json"), "w"), indent=1, default=str)
    print(f"DONE {ep} in {total:.1f}s (seeding {t_seed:.1f}s = {info['timing']['ms_per_frame_seeding']} ms/frame, process CPU {time.process_time():.1f}s); "
          f"picked {n_picked}/{T_store} frames (round-3 base non-empty on {info['summary']['frames_with_any_round3_verified']}), tracks {len(sel.tracks)}, modes {modes}, "
          f"kinds {picked_kinds}; tests {test_counts}; rejected-best fails {rejected_fails}; postfilter removed {postfilter['n_removed']}; "
          f"products ok: {all(p['product_ok'] for p in products.values())}; verified_points ok: {verified_points['product_ok']}", flush=True)


if __name__ == "__main__":
    main()
