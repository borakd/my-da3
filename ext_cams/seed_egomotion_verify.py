#!/usr/bin/env python
"""seed_egomotion_verify (IDEA 2, 2026-09-29): seed_verified_motion.py's candidate generation (MOG2 moving components,
cross-view DLT triangulation, reach / size / dark / silhouette gates) with the Selector's PREFERENCE RULE ("most distal
verified pair") REPLACED by an EGO-MOTION CONSISTENCY TEST against CUT3R's own camera-centre track (closed loop):

  * Candidate identity over time: at every frame the verified 3D points (base frame, metres) are clustered by single
    linkage at cluster_m = 0.10 m (candidates of one gripper-sized object: the fg / mov / cap / dark variants of the same
    blob have centroids within about half the 0.2 m Robotiq + wrist-mount length; the forearm link is 0.38 m, so an elbow
    and the gripper land in different clusters) and the clusters are associated greedily (nearest first) to the alive
    candidate tracks within jump_m x gap (0.12 m per frame of gap, as round 3); unmatched clusters start new tracks,
    tracks unseen for coast_frames (15) die. Track observation = the cluster's mean point. Causal state only.
  * Motion test at frame t for every track observed at t with >= min_obs (8) observations inside the causal window of
    the last W = 15 frames (frames t-14..t) whose 3D travel (diameter of the window's points, robust to triangulation
    jitter) is >= min_travel (0.03 m): a similarity (scale, rotation, translation; Umeyama 1991) is fitted from CUT3R's
    finetuned camera centres on the SAME frames (preds/EP/camera/NNNNNN.npz 'pose' c2w, non-metric: the model's own
    output, read lazily at frame t so nothing after t is ever touched) to the track's points; the score is the RMS
    residual (m) per unit of travel (m). The track with the LOWEST score is picked (no absolute threshold, as specified);
    the emitted pair is the most distal verified pair of that track's cluster at t. A grasped object moves rigidly with
    the wrist camera and scores low; a person, a chair, a cloth or an elbow under a rotating wrist does not.
  * Fallback: when no track has enough motion (or no CUT3R pose is available for the window) the round-3 rule decides
    (temporal consistency to the last pick + most distal); the last pick is shared by both rules so the fallback stays
    consistent with the motion picks.
  Everything else (background model, candidates, verification gates, the emitted mask = the foreground component that
  holds the picked candidate cropped to the box_m box around the projected 3D point, verified_points.npz) is identical to
  seed_verified_motion.py, so the only difference to round 3 is WHICH verified pair is picked.

Causality: frame t uses exterior frames <= t and CUT3R poses of frames <= t (available online: CUT3R runs on the wrist
stream at frame t). GT-free: no store pose is read (the store is opened only to count frames); the CUT3R preds are the
model's output, not GT. Training-free, CPU only, no DROID-trained component.
Declared constants (seed_info.json 'declared_constants'): W = 15 frames, min_travel = 0.03 m, min_obs = 8, cluster_m = 0.10 m,
plus the round-3 verification constants; none was tuned on the 13 scenes (set once from the physical arguments above).
Cost: the round-3 seeder (120-240 ms/frame for both cameras at 640x360 on one login-node core, i.e. 4-8 fps) plus about
1 ms/frame for the clustering, association and Umeyama fits; below the 10-20 fps target by the base seeder's cost, the
test itself is free. seed_info.json 'timing' has the measured numbers.

    OMP_NUM_THREADS=1 python seed_egomotion_verify.py --episode EP [--out_root .../ideas/egomotion_verify] [--video]

--- the round-3 seeder this is built on (unchanged parts) ---
Per frame t (both cameras in lockstep; pixel work at 640x360, ALL geometry in native 1280x720 pixels):
  1. background: per camera an OpenCV MOG2 mixture updated online with a fixed learning rate after a short warm-up (as
     seed_motion_only.py, incl. the dark-and-achromatic exception to the shadow label). Foreground = pixels off the model;
     median blur + close + open. "moving" = |gray_t - gray_{t-1}| > move_thresh, dilated.
  2. candidates per camera (only inside foreground components that contain moving pixels; ghosts of absorbed poses do not
     move and are dropped):  fg  = the component itself;  mov = connected fragments of (component & moving);
     capL = distal caps of the component: geodesic distance inside the component from where it enters the image (border
     contacts, else the pixel nearest the projected base origin), tip = farthest moving pixel, cap = pixels within L metres
     (0.10/0.15/0.20/0.25, converted with the calibration's plausible depth) of the tip along the component.
     dark = connected components of the dark-achromatic pixels inside the component (the DROID gripper is a black
     Robotiq 2F-85 with a black wrist-camera mount on every rig: an object prior, not ground truth).
     Each candidate carries its centroid, bbox and area in native pixels and its dark-achromatic fraction.
  3. 3D verification of every cross-view candidate pair (declared calibration only: PointWorld optimized_extrinsics =
     world(base)->camera, factory intrinsics): DLT triangulation of the two centroids, accepted only if
       - the centroid rays meet: per-view reprojection residual of the triangulated point < ray_tol_m * f / depth
         (ray_tol_m = 0.04 m; the round-2 spec's 6 px gate is 7 mm at 0.6 m and passed 7-10% of the true gripper pairs
         on the 44bb9c36 rig in the round-2 diagnostic: silhouette centroids of a 15 cm object seen ~90 deg apart differ by cm),
       - |X| < reach (0.9 m from the base origin) and X_z > zmin (-0.15 m, z up), the point in front of both cameras,
       - in each view the candidate's apparent size (longer bbox side * depth / f) lies in [size_min, size_max] m,
       - both candidates are dark-achromatic in at least dark_min of their pixels,
       - mutual silhouette consistency: sample_pts pixels of candidate A are back-projected along their rays at depths
         inside the reach sphere and projected into the other view; the fraction whose epipolar segment passes within
         hull_tol_px of candidate B must be >= hull_min, and the same from B to A (a candidate whose samples leave the
         other image is not testable in that direction; with one testable direction the threshold is hull_min_single).
         The projection of any point of a solid object lies inside its silhouette, so true pairs pass whatever the
         viewpoint; an elbow paired with the gripper by a chance epipolar coincidence of the centroids fails.
  4. selection (ROUND-3 RULE, here the FALLBACK only): while a track is alive (a verified pick within the last coast_frames),
     the verified pairs within jump_m * (frames since the last pick) of the previous 3D point are the consistent set and the
     most DISTAL of them (largest |X|) is picked; otherwise the most distal verified pair starts a new track.
  5. emission per view: the foreground component that holds the picked candidate (or the candidate itself, --emit
     candidate), cropped to a box of box_m = 0.30 m equivalent (half-size 0.15 * f / depth px per axis) centred on the
     projection of the triangulated point. Manipulated objects far from the gripper are cut off by the box. Frames
     without a verified pick get an EMPTY mask (frames_present false): there is no hold and no propagation.

Ground truth is never read here; the store is opened only to count frames (product check). Writes, per the mask file
contract, <out_root>/<EP>/<cam>_<serial>__egomotion_verify_masks.npz (union uint8 (T,720,160) = packbits(mask, axis=2),
shape [T,720,1280], frames_present bool (T,)) for both cameras, <EP>/seed_info.json (parameters, declared constants,
calibration summary, per-camera stats, tracks, per-frame log incl. the motion-test values, timing) and optionally
<EP>/egomotion_verify_ext_side_by_side.mp4 (--video) and check PNGs (--png_frames).
<EP>/verified_points.npz: frames (T,) int, xyz (T,3) float = the base-frame DLT point of the PICKED verified candidate pair
at each frame (NaN where nothing was picked) and accepted (T,) bool; the per-frame verified 3D point the tracker consumes
through --verified_points (rig_track2.py wrong-body gate). Causal: row t is the pick made at frame t from frames <= t.
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
OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas/egomotion_verify"
CUT3R_PREDS = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/augfull_lr1e5/preds"  # the model's own output (closed loop)
METHOD = "egomotion_verify"
NATIVE_W, NATIVE_H = 1280, 720
CAMS = ("ext1", "ext2")


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
    pts = [(cand["cx"], cand["cy"])] + ([(cand["tip"][0] * s, cand["tip"][1] * s)] if "tip" in cand else []) +           [((x + 0.5) * s - 0.5, (y + 0.5) * s - 0.5) for y, x in zip(ys[idx], xs[idx])]
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
                size_px=float(max(bw, bh)), area=float(len(xs) * s * s), dark=dark)


def extract_candidates(fg, moving, moving_dil, gray, chroma, cal, a, s, px_per_m):
    """Candidates of one view at frame t: fg components with motion, their moving fragments and their distal caps."""
    H, W = fg.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    min_area, min_area_mov = a.min_area_frac * W * H, a.min_area_mov_frac * W * H
    parents, cands = {}, []
    k_dark = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.dark_open_px, a.dark_open_px))
    base_pt = None
    if cal["base_px"] is not None:
        base_pt = ((cal["base_px"][1] + 0.5) / s - 0.5, (cal["base_px"][0] + 0.5) / s - 0.5)  # (row, col) at proc res, may be outside
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = lab == k
        mv = comp & moving
        if not mv.any():
            continue  # a ghost / an object the arm moved earlier: no longer moving
        parents[k] = comp
        cands.append(candidate_record(comp, "fg", k, gray, chroma, s, a))
        # moving fragments inside the component (the max_frag largest)
        frag = (comp & moving_dil).astype(np.uint8)
        nf, labf, statsf, _ = cv2.connectedComponentsWithStats(frag, connectivity=8)
        order = sorted(range(1, nf), key=lambda j: -statsf[j, cv2.CC_STAT_AREA])[:a.max_frag]
        for j in order:
            if statsf[j, cv2.CC_STAT_AREA] >= min_area_mov:
                cands.append(candidate_record(labf == j, "mov", k, gray, chroma, s, a))
        # dark-achromatic components inside the component (the black gripper on a white arm), opened to drop cables
        dark = (comp & (gray < a.dark_thresh) & (chroma < a.chroma_thresh)).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k_dark)
        nd, labd, statsd, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        order = sorted(range(1, nd), key=lambda j: -statsd[j, cv2.CC_STAT_AREA])[:a.max_frag]
        for j in order:
            if statsd[j, cv2.CC_STAT_AREA] >= min_area_mov:
                cands.append(candidate_record(fill_small_holes(labd == j, a.max_hole_px), "dark", k, gray, chroma, s, a))
        # distal caps along the component
        roots = border_pixels(comp)
        if len(roots) == 0:
            rs, cs = np.nonzero(comp)
            if base_pt is not None:
                i = int(np.argmin((rs - base_pt[0]) ** 2 + (cs - base_pt[1]) ** 2))
            else:
                i = int(np.argmin(rs))  # top-most pixel: the arm enters from above when the base is not projectable
            roots = np.array([[rs[i], cs[i]]])
        geo = geodesic_small(comp, roots, a.geo_grid)
        if geo is None:
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
            cands.append(rec)
    return parents, cands


# ----------------------------------------------------------------------------------------------- 3D verification + selection
def verify_pairs(c1, c2, cal1, cal2, geom1, geom2, a, s):
    """All cross-view candidate pairs with their triangulation and gate values; 'ok' when every gate passes. The
    silhouette test (the expensive gate) runs only on pairs that pass the cheap gates; distance transforms are cached."""
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
            rec["ok"] = bool(cheap and (a.hull_min <= 0 or rec["ok_hull"]))
            out.append(rec)
    return out


class Selector:
    """ROUND-3 RULE (the fallback here): distal preference + temporal consistency over the verified pairs (causal state:
    last pick only). rule() is pure (no state change) so the ego-motion selector can log what round 3 would have picked;
    commit() records the pick actually made (by either rule)."""

    def __init__(self, a):
        self.a = a
        self.X_prev, self.t_prev = None, None
        self.tracks = []

    def rule(self, t, verified):
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
        return best, mode

    def consistency_mode(self, t, best):
        """'track' if best is within jump_m x gap of the last pick (alive), 'switch' if alive but not, 'new' otherwise."""
        a = self.a
        if self.X_prev is None or (t - self.t_prev) > a.coast_frames:
            return "new"
        return "track" if np.linalg.norm(np.array(best["X"]) - self.X_prev) <= a.jump_m * (t - self.t_prev) else "switch"

    def commit(self, t, best, mode):
        if mode != "track":
            self.tracks.append(dict(track=len(self.tracks), frame=int(t), mode=mode, **{k: best[k] for k in ("kinds", "X", "r", "dist", "sizes", "depths", "dark", "hull")}))
        self.X_prev, self.t_prev = np.array(best["X"]), t

    def pick(self, t, verified):
        best, mode = self.rule(t, verified)
        if best is not None:
            self.commit(t, best, mode)
        return best, mode


# ----------------------------------------------------------------------------------------------- ego-motion consistency (idea 2)
def umeyama(src, dst):
    """Least-squares similarity dst ~ s R src + t (Umeyama 1991, with scale). src, dst (N,3). Returns s, R, t and the
    RMS residual in dst units. A degenerate src (no spread) gives s = 0 and the residual = the spread of dst."""
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = float((xs ** 2).sum() / n)
    s = float((D * np.diag(S)).sum() / var_s) if var_s > 1e-12 else 0.0
    t = mu_d - s * (R @ mu_s)
    res = dst - (s * (R @ src.T).T + t)
    return s, R, t, float(np.sqrt((res ** 2).sum(1).mean()))


def single_linkage(pts, radius):
    """Clusters (lists of indices) of the (N,3) points under single linkage at the given radius (union-find)."""
    n = len(pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n > 1:
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        for i, j in zip(*np.nonzero(np.triu(d <= radius, 1))):
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


class Cut3rCentres:
    """CUT3R finetuned camera centres (c2w 'pose' translation, non-metric, the model's own output), read LAZILY: the file
    of frame t is opened the first time a frame >= t asks for it, so the test at frame t never touches a later frame."""

    def __init__(self, ep, root):
        self.dir = os.path.join(root, ep, "camera")
        self.cache = {}
        self.n_files = len([f for f in os.listdir(self.dir) if f.endswith(".npz")]) if os.path.isdir(self.dir) else 0
        self.missing = 0

    def get(self, t):
        if t not in self.cache:
            p = os.path.join(self.dir, f"{t:06d}.npz")
            if os.path.isfile(p):
                self.cache[t] = np.load(p)["pose"][:3, 3].astype(np.float64)
            else:
                self.cache[t] = None
                self.missing += 1
        return self.cache[t]


class EgoMotionSelector:
    """Idea 2: candidate tracks (clusters of verified 3D points associated over time) scored by the Umeyama residual per
    unit of travel between the track's window and CUT3R's camera-centre track on the same frames; lowest score wins;
    round-3 rule (self.r3) when no track has enough motion. Causal state: alive tracks (their last W observations) and
    the round-3 selector's last pick."""

    def __init__(self, a, cut3r):
        self.a = a
        self.cut3r = cut3r
        self.r3 = Selector(a)
        self.tracks = []          # alive candidate tracks: dict(id, ts, Xs, last_t, members)
        self.next_id = 0
        self.picked_track = None  # id of the track picked last (for the switch log)
        self.switches = []        # (frame, from_id, to_id) of motion picks that changed track
        self.n_tests = 0

    def _update_tracks(self, t, verified):
        a = self.a
        self.tracks = [tr for tr in self.tracks if (t - tr["last_t"]) <= a.coast_frames]  # coast, then die
        if not verified:
            return []
        pts = np.array([v["X"] for v in verified], float)
        clusters = single_linkage(pts, a.cluster_m)
        obs = [dict(mean=pts[c].mean(0), members=c) for c in clusters]
        pairs = []
        for oi, o in enumerate(obs):
            for ti, tr in enumerate(self.tracks):
                gap = t - tr["last_t"]
                d = float(np.linalg.norm(o["mean"] - tr["Xs"][-1]))
                if d <= a.jump_m * gap:
                    pairs.append((d, oi, ti))
        pairs.sort()
        used_o, used_t, assigned = set(), set(), {}
        for d, oi, ti in pairs:
            if oi in used_o or ti in used_t:
                continue
            used_o.add(oi), used_t.add(ti)
            assigned[oi] = ti
        current = []
        for oi, o in enumerate(obs):
            if oi in assigned:
                tr = self.tracks[assigned[oi]]
            else:
                tr = dict(id=self.next_id, ts=[], Xs=[], last_t=t, members=[])
                self.next_id += 1
                self.tracks.append(tr)
            tr["ts"].append(int(t)), tr["Xs"].append(o["mean"])
            tr["ts"], tr["Xs"] = tr["ts"][-a.window:], tr["Xs"][-a.window:]  # keep the causal window only
            tr["last_t"], tr["members"] = int(t), o["members"]
            current.append(tr)
        return current

    def _test(self, t, tr):
        """Motion test of one track at frame t. Returns a record; 'counts' False when the track has too few observations
        in the window, too little travel or a missing CUT3R pose."""
        a = self.a
        t_lo = t - a.window + 1
        idx = [k for k, ts in enumerate(tr["ts"]) if ts >= t_lo]
        rec = dict(track=tr["id"], n_obs=len(idx), counts=False, why=None)
        if len(idx) < a.min_obs:
            rec["why"] = "few_obs"
            return rec
        X = np.array([tr["Xs"][k] for k in idx])
        travel = float(np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2).max())
        rec["travel"] = round(travel, 4)
        if travel < a.min_travel:
            rec["why"] = "still"
            return rec
        C = [self.cut3r.get(tr["ts"][k]) for k in idx]
        if any(c is None for c in C):
            rec["why"] = "no_cut3r_pose"
            return rec
        s, R, tt, rms = umeyama(np.array(C), X)
        rec.update(counts=True, scale=round(s, 4), rms=round(rms, 4), score=round(rms / travel, 4))
        return rec

    def pick(self, t, verified):
        """Returns (best pair or None, mode, per-frame log dict)."""
        a = self.a
        current = self._update_tracks(t, verified)
        r3_best, r3_mode = self.r3.rule(t, verified)
        log = dict(n_tracks_alive=len(self.tracks), tests=[], r3_X=None if r3_best is None else r3_best["X"])
        best, mode, chosen = None, "none", None
        if current:
            tests = [(self._test(t, tr), tr) for tr in current]
            log["tests"] = [rec for rec, _ in tests]
            counted = [(rec, tr) for rec, tr in tests if rec["counts"]]
            self.n_tests += len(counted)
            if counted:
                rec, tr = min(counted, key=lambda rt: rt[0]["score"])
                best = max((verified[m] for m in tr["members"]), key=lambda v: v["dist"])  # most distal pair of the cluster
                mode, chosen = "motion", tr["id"]
                log.update(picked_track=tr["id"], score=rec["score"], travel=rec["travel"], rms=rec["rms"], scale=rec["scale"], n_obs=rec["n_obs"],
                           n_counted=len(counted), changed_vs_r3=bool(r3_best is not None and np.linalg.norm(np.array(best["X"]) - np.array(r3_best["X"])) > a.cluster_m))
                if self.picked_track is not None and self.picked_track != tr["id"]:
                    self.switches.append((int(t), self.picked_track, tr["id"]))
                self.picked_track = tr["id"]
        if best is None and r3_best is not None:
            best, mode = r3_best, "fb_" + r3_mode  # fallback: round-3 rule
            for tr in current:
                if best["i"] in [verified[m]["i"] for m in tr["members"]] and best["j"] in [verified[m]["j"] for m in tr["members"]]:
                    chosen = tr["id"]
            log["picked_track"] = chosen
        if best is not None:
            self.r3.commit(t, best, self.r3.consistency_mode(t, best))  # shared last-pick state for both rules
        return best, mode, log


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


def write_overlay(ep, cams, serials, videos, packed, logs, cal, out_dir, fps, png_frames, T):
    """Streaming two-panel overlay: emitted mask (green), crop box (magenta), candidate centroids (grey = rejected,
    yellow = verified, red = picked), projected 3D point (red cross)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sam3_gripper_masks import FfmpegWriter
    pw, ph = 640, 360
    path = os.path.join(out_dir, f"{METHOD}_ext_side_by_side.mp4")
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
                col = (0, 0, 255) if cd["picked"] else ((0, 255, 255) if cd["verified"] else (160, 160, 160))
                cv2.circle(img, (int(cd["cx"]), int(cd["cy"])), 8 if cd["picked"] else 5, col, -1)
            box = info.get("box", {}).get(c)
            if box is not None:
                cv2.rectangle(img, (box[0], box[1]), (box[2] - 1, box[3] - 1), (255, 0, 255), 2)
                px, py = info["proj"][c][:2]
                cv2.drawMarker(img, (int(px), int(py)), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            hdr = f"{c}/{serials[c]} {METHOD} f{t}/{T - 1} {info['mode']} nc={len(info['cands'].get(c, []))} nv={info['n_verified']}"
            if info.get("X") is not None:
                hdr += f" |X|={info['dist']:.2f} r={info['r']:.1f} sz={info['sizes']} hull={info.get('hull')} dark={info.get('dark')}"
            cv2.rectangle(img, (0, 0), (NATIVE_W, 26), (0, 0, 0), -1)
            cv2.putText(img, hdr, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA))
        row = np.concatenate(panels, axis=1)
        writer.write(row)
        if t in png_frames:
            cv2.imwrite(os.path.join(out_dir, f"{METHOD}_check_f{t:04d}.png"), row)
    writer.close()
    for cap in caps.values():
        cap.release()
    return path


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out_root", default=OUT_ROOT)
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
    # 3D verification
    ap.add_argument("--ray_tol_m", type=float, default=0.04, help="the two centroid rays must meet within this (m): per-view residual < ray_tol_m * f / depth")
    ap.add_argument("--r_max_px", type=float, default=0.0, help="additional hard cap on the per-view residual (px, native); 0 = none. 6 px was the round-2 spec")
    ap.add_argument("--reach", type=float, default=0.9, help="max distance of the point from the base origin (m)")
    ap.add_argument("--zmin", type=float, default=-0.15, help="min height of the point in the base frame (m, z up)")
    ap.add_argument("--size_min", type=float, default=0.07, help="apparent size window (m): longer bbox side * depth / f")
    ap.add_argument("--size_max", type=float, default=0.30)
    ap.add_argument("--min_cam_depth", type=float, default=0.15, help="the point must be at least this far in front of each camera (m)")
    ap.add_argument("--max_depth", type=float, default=1.8, help="epipolar segments are sampled up to this camera depth (m)")
    ap.add_argument("--depth_samples", type=int, default=40)
    ap.add_argument("--dark_min", type=float, default=0.15, help="both candidates must be dark-achromatic in at least this fraction of their pixels")
    ap.add_argument("--hull_min", type=float, default=0.6, help="mutual silhouette consistency threshold (fraction of samples); 0 disables the test")
    ap.add_argument("--hull_min_single", type=float, default=0.85, help="threshold when only one direction is testable (the other candidate's samples leave the image)")
    ap.add_argument("--hull_tol_px", type=float, default=8.0, help="a sample hits when its epipolar segment passes within this of the other mask (px, native)")
    ap.add_argument("--sample_pts", type=int, default=12)
    ap.add_argument("--hull_grid", type=int, default=2, help="distance transforms on the processing grid downsampled by this")
    # selection + emission
    ap.add_argument("--jump_m", type=float, default=0.12, help="temporal consistency radius per frame of gap (m)")
    ap.add_argument("--coast_frames", type=int, default=15, help="a track stays alive this many frames without a pick")
    ap.add_argument("--box_m", type=float, default=0.30, help="crop box side, metres-equivalent at the point's depth")
    ap.add_argument("--emit", default="parent", choices=["parent", "candidate"], help="what the box crops: the foreground component holding the pick, or the pick itself")
    # idea 2: ego-motion consistency test (declared constants, set from physical reasoning, not tuned on the 13 scenes)
    ap.add_argument("--cut3r_root", default=CUT3R_PREDS, help="CUT3R finetuned preds root (<root>/EP/camera/NNNNNN.npz 'pose' c2w): the model's own output, read causally")
    ap.add_argument("--window", type=int, default=15, help="W: causal window (frames t-W+1..t) of the motion test")
    ap.add_argument("--min_travel", type=float, default=0.03, help="a track must span at least this (m, diameter of its window points) for the test to count")
    ap.add_argument("--min_obs", type=int, default=8, help="a track must have at least this many observations inside the window (more than half of W)")
    ap.add_argument("--cluster_m", type=float, default=0.10, help="single-linkage radius (m) grouping the verified 3D points of one frame into one candidate object")
    # outputs
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--png_frames", default="", help="comma list of frame indices dumped as check PNGs (needs --video)")
    ap.add_argument("--log_pairs", action="store_true", help="keep every pair's verification record in the per-frame log (large)")
    a = ap.parse_args()
    a.cap_lengths = [float(x) for x in a.cap_lengths.split(",") if x.strip()]

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
    print(f"episode {ep}\nvideos {videos}\nstore frames {T_store}\nout {out_dir}", flush=True)
    for c in cams:
        k = cal[c]
        print(f"[{c}/{serials[c]}] fx={k['fx']:.1f} base_px={k['base_px']} cam_to_base={k['cam_to_base']:.2f} z_grid={k['z_grid_median']:.2f} "
              f"plausible_depth={k['plausible_depth']:.2f} -> {px_per_m[c]:.0f} px/m at {proc_w}", flush=True)

    t_all = time.time()
    views = {c: ViewMotion(a) for c in cams}
    caps = {c: open_video(videos[c]) for c in cams}
    cut3r = Cut3rCentres(ep, a.cut3r_root)
    print(f"cut3r preds {cut3r.dir}: {cut3r.n_files} files (store {T_store}); read lazily per frame (causal)", flush=True)
    sel = EgoMotionSelector(a, cut3r)
    geom = {c: ViewGeom(cal[c]) for c in cams}
    packed = {c: np.zeros((T_store, NATIVE_H, NATIVE_W // 8), np.uint8) for c in cams}
    present = {c: np.zeros(T_store, bool) for c in cams}
    logs = []
    modes = {}
    n_pairs_total = n_verified_total = 0
    t_motion = t_verify = 0.0
    t = 0
    while t < T_store:
        fr = read_pair(caps, proc_w, proc_h)
        if fr is None:
            break
        t0 = time.time()
        info = dict(t=t, mode="warmup", cands={}, n_pairs=0, n_verified=0)
        cands, parents = {}, {}
        for c in cams:
            st = views[c].step(fr[c][1])
            if st is None:
                continue
            fg, moving, moving_dil, gray, chroma = st
            parents[c], cands[c] = extract_candidates(fg, moving, moving_dil, gray, chroma, cal[c], a, s, px_per_m[c])
        t1 = time.time()
        t_motion += t1 - t0
        if all(c in cands for c in cams):
            pairs = verify_pairs(cands[cams[0]], cands[cams[1]], cal[cams[0]], cal[cams[1]], geom[cams[0]], geom[cams[1]], a, s)
            verified = [p for p in pairs if p["ok"]]
            n_pairs_total += len(pairs)
            n_verified_total += len(verified)
            best, mode, mlog = sel.pick(t, verified)
            info.update(mode=mode, n_pairs=len(pairs), n_verified=len(verified), motion=mlog)
            if a.log_pairs:
                info["pairs"] = pairs
            picked_idx = {cams[0]: best["i"], cams[1]: best["j"]} if best else {}
            ver_idx = {cams[0]: {p["i"] for p in verified}, cams[1]: {p["j"] for p in verified}}
            for c in cams:
                info["cands"][c] = [dict(kind=cd["kind"], cx=round(cd["cx"], 1), cy=round(cd["cy"], 1), size_px=round(cd["size_px"], 1),
                                         dark=round(cd["dark"], 2), verified=bool(i in ver_idx[c]), picked=bool(picked_idx.get(c) == i))
                                    for i, cd in enumerate(cands[c])]
            if best is not None:
                X = np.array(best["X"])
                info.update(X=best["X"], r=best["r"], dist=best["dist"], sizes=best["sizes"], depths=best["depths"], kinds=best["kinds"], dark=best["dark"], hull=best["hull"], box={}, proj={})
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
            print(f"  f{t}: mode={info['mode']} cands={[len(info['cands'].get(c, [])) for c in cams]} pairs={info['n_pairs']} verified={info['n_verified']} "
                  f"X={info.get('X')} r={info.get('r')} sizes={info.get('sizes')} ({time.time() - t_all:.0f}s)", flush=True)
        t += 1
    for cap in caps.values():
        cap.release()
    T_mp4 = t
    if T_mp4 < T_store:
        print(f"WARNING: mp4 pair yielded {T_mp4} frames < store {T_store}; remaining frames are empty", flush=True)
        for k in range(T_mp4, T_store):
            logs.append(dict(t=k, mode="no_frame", cands={}, n_pairs=0, n_verified=0))
    t_seed = time.time() - t_all

    products, stats = {}, {}
    for c in cams:
        path = os.path.join(out_dir, f"{c}_{serials[c]}__{METHOD}_masks.npz")
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
    # round 3: per-frame verified 3D point (base frame, metres) of the picked pair; NaN / accepted=False where no pick.
    vp_xyz = np.full((T_store, 3), np.nan)
    vp_acc = np.zeros(T_store, bool)
    for lg in logs:
        if lg.get("X") is not None and 0 <= lg["t"] < T_store:
            vp_xyz[lg["t"]] = np.array(lg["X"], float)
            vp_acc[lg["t"]] = True
    vp_path = os.path.join(out_dir, "verified_points.npz")
    np.savez(vp_path, frames=np.arange(T_store, dtype=np.int32), xyz=vp_xyz, accepted=vp_acc,
             note="base-frame (m) DLT point of the picked verified candidate pair per frame; NaN / accepted=False where nothing "
                  "was verified at that frame; causal (frame t uses frames <= t); calibration only, no GT")
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
    video_path = None
    t_video = 0.0
    if a.video:
        png = [int(x) for x in a.png_frames.split(",") if x.strip()]
        t0 = time.time()
        video_path = write_overlay(ep, cams, serials, videos, packed, logs, cal, out_dir, a.fps, png, T_store)
        t_video = time.time() - t0
        print(f"overlay {video_path} in {t_video:.1f}s", flush=True)
    total = time.time() - t_all
    n_picked = sum(1 for lg in logs if lg.get("X") is not None)
    n_motion = sum(1 for lg in logs if lg.get("mode") == "motion")
    n_changed = sum(1 for lg in logs if lg.get("mode") == "motion" and lg["motion"].get("changed_vs_r3"))
    n_multi = sum(1 for lg in logs if lg.get("mode") == "motion" and lg["motion"].get("n_counted", 0) > 1)
    why = {}
    for lg in logs:
        for rec in lg.get("motion", {}).get("tests", []):
            if not rec["counts"]:
                why[rec["why"]] = why.get(rec["why"], 0) + 1
    scores = [lg["motion"]["score"] for lg in logs if lg.get("mode") == "motion"]
    info = dict(episode=ep, method=METHOD, serials=serials, params=vars(a), proc_res=[proc_w, proc_h], native_res=[NATIVE_W, NATIVE_H],
                store_frames=T_store, mp4_frames_read=T_mp4, causal=True, uses_gt=False, training_free=True, cpu_only=True,
                closed_loop_signal=dict(what="CUT3R finetuned (augfull_lr1e5) camera centres, c2w 'pose' translation, non-metric; the model's own output",
                                        root=cut3r.dir, n_files=cut3r.n_files, n_missing_requested=cut3r.missing, read="lazily at frame t (frames <= t only)"),
                declared_constants=dict(window_frames=a.window, min_travel_m=a.min_travel, min_obs=a.min_obs, cluster_m=a.cluster_m, jump_m_per_frame=a.jump_m,
                                        coast_frames=a.coast_frames, score="umeyama_rms_m / travel_m (travel = diameter of the track's window points)",
                                        pick="lowest score among tracks observed at t that count; no absolute threshold; fallback = round-3 rule",
                                        per_rig_constants="none (no exemplar bank, no hand-eye offset)", tuned_on_smoke13=False),
                calibration_use="PointWorld optimized_extrinsics + factory intrinsics: cross-view triangulation, reach/height/size gates, crop box; "
                                "plausible depth for the candidate cap lengths only",
                verification=dict(ray_tol_m=a.ray_tol_m, r_max_px=a.r_max_px, reach_m=a.reach, zmin_m=a.zmin, size_window_m=[a.size_min, a.size_max],
                                  dark_min=a.dark_min, hull_min=a.hull_min, hull_min_single=a.hull_min_single, hull_tol_px=a.hull_tol_px, box_m=a.box_m,
                                  jump_m_per_frame=a.jump_m, coast_frames=a.coast_frames, emit=a.emit),
                summary=dict(frames_picked=n_picked, frac_picked=round(n_picked / max(T_store, 1), 4), modes=modes, picked_kinds=picked_kinds,
                             n_tracks=len(sel.r3.tracks), pairs_tested=n_pairs_total, pairs_verified=n_verified_total,
                             frames_with_any_verified=sum(1 for lg in logs if lg["n_verified"] > 0),
                             frames_motion_pick=n_motion, frames_fallback_pick=n_picked - n_motion,
                             frames_motion_pick_with_several_counted=n_multi, frames_motion_pick_changed_vs_round3_rule=n_changed,
                             tests_counted=sel.n_tests, tests_not_counted_why=why, candidate_tracks_created=sel.next_id,
                             motion_pick_track_switches=sel.switches,
                             score_median=(round(float(np.median(scores)), 4) if scores else None), score_p90=(round(float(np.percentile(scores, 90)), 4) if scores else None)),
                tracks=sel.r3.tracks, per_cam=stats, products=products, verified_points=verified_points, overlay_mp4=video_path,
                timing=dict(total_seconds=round(total, 1), seeding_seconds=round(t_seed, 1), motion_and_candidates_seconds=round(t_motion, 1),
                            verification_seconds=round(t_verify, 1), overlay_seconds=round(t_video, 1), cpu_seconds_process=round(time.process_time(), 1),
                            ms_per_frame_seeding=round(1000 * t_seed / max(T_mp4, 1), 1)),
                per_frame=logs)
    json.dump(info, open(os.path.join(out_dir, "seed_info.json"), "w"), indent=1, default=str)
    print(f"DONE {ep} in {total:.1f}s (seeding {t_seed:.1f}s = {info['timing']['ms_per_frame_seeding']} ms/frame, process CPU {time.process_time():.1f}s); "
          f"picked {n_picked}/{T_store} frames (motion {n_motion}, changed vs round-3 rule {n_changed}, fallback {n_picked - n_motion}), r3 tracks {len(sel.r3.tracks)}, "
          f"cand tracks {sel.next_id}, modes {modes}, kinds {picked_kinds}; products ok: {all(p['product_ok'] for p in products.values())}; "
          f"verified_points ok: {verified_points['product_ok']}", flush=True)


if __name__ == "__main__":
    main()
