#!/usr/bin/env python
"""seed_silhouette_model (IDEA 5): model-based silhouette tracking of the DROID end effector in the two STATIC exterior
cameras. A coarse rigid model of the end effector, expressed in the WRIST-CAMERA frame (OpenCV: x right, y down, z optical
axis), is rendered with the declared calibration K[R|t] into each view and its 6-DoF pose is refined per frame by RAPiD-style
edge alignment. The product is the metric pose of the wrist camera itself (c2w in the base frame) on every solved frame,
plus the rendered silhouette as a mask (suffix silhouette_model) so the mosaic and rig_track2 can run on it.

METHOD (per frame t, both views jointly)
  1. Prediction. When frame t-1 was solved: T_pred = T_{t-1}; when the round-3 rig tracker (rig2/anchors.npz, GT-free) has
     anchors at t and at an earlier solved frame t_a of the SAME segment: T_pred = M_t M_{t_a}^{-1} T_{t_a} (M = the anchor's
     rigid motion X_t = R_t X_ref + trans_t, so this carries the last solved pose along the tracker's own rigid motion, no GT).
     When nothing is tracked: global initialisation (step 4).
  2. Silhouette edges. For every model box the silhouette edges (edges between a front- and a back-facing face for that
     camera centre) are sampled every ~6 px along their projection; samples that fall > 2 px inside another box's projected
     convex hull are dropped (union outline). Each sample keeps its 3D model-frame point and its 2D outward normal.
  3. Refinement. Canny edges (thresholds 40/120) + Sobel gradient on a ROI around the projection; for each sample the nearest
     Canny pixel along the normal within the search radius whose gradient direction is within 55 deg of the normal
     (|cos| >= 0.57) AND, for the black boxes (body, fingers), with the image darker 3 px inside than 3 px outside by >= 15
     grey levels (edge polarity: the Robotiq 2F-85 is black on every DROID rig, the same object prior the round-3 seeder
     uses with gray < 70), gives the signed residual r_i; Gauss-Newton on se(3) (body-frame update) with Tukey weights (c = 8 px)
     and a Levenberg damping of 1e-4, both views stacked. Schedule: 3 iterations at radius 30 px, then 5 at 15 px (the
     task's "iterate 5 times, 15 px"; the wider first pass only widens the basin after a global init or a prediction gap).
     Gate: the frame is SOLVED when the pooled median |r_i| over both views <= 4 px, >= 50 % of the samples found an edge,
     each view has >= 12 samples with an edge, and in each view >= 25 % of a 3x3x3 lattice inside the body box projects on
     dark pixels (gray < 70; the seeder's dark-achromatic prior); otherwise REJECTED (no pose, empty mask). After 8 consecutive rejections
     (or a rejection with no prediction source) the track is lost and step 4 runs at the next frame with masks.
  4. Global initialisation (GT-free, mask-driven). At a frame where the seeding masks are non-empty (>= 1500 px) in both
     views: a two-view visual hull of the masks (1 cm voxels in a 0.5 m cube around the masks' triangulated centroid,
     >= 150 voxels) gives a centroid h. Hypotheses: wrist z axis on a 100-point Fibonacci sphere x roll in 12 steps of
     30 deg x body-box centre at h and h +/- 3 cm along each base axis (8400 candidates). Each is scored by the silhouette
     IoU of the DARK boxes (body, fingers) rendered at 1/8 resolution against the mask, mean of the two views (the masks,
     SAM3 on the shelf and the dark-achromatic motion seeder on the 13 scenes, cover the black gripper, not the camera
     housing, so the ZED box is left out of the mask comparison), times (0.5 + 0.5 x housing brightness) where housing
     brightness = clip((mean gray on 5 points along the ZED bar - 60) / 100, 0, 1). This is the term that breaks the
     roll-180 symmetry of the black boxes about the body centre-line: the camera housing is a bright bar on every DROID rig
     seen (white mount on RAIL, gold anodised bar on both AUTOLab rigs) while the gripper is black, so the pose whose ZED
     box lands on the bright bar wins over its mirror image whose bar lands on the far side of the body. The best 12 are refined first against the MASK BOUNDARY
     (dark boxes only; the mask is the initialisation signal, cleaner than Canny) and then against the image edges (all
     boxes, step 3); the refined candidate with the highest IoU x brightness that passes the gate with edge support >= 0.6,
     IoU >= 0.35 in both views (0.7 x the shelf median IoU at the GT pose, 0.47-0.53) and housing brighter than the body
     by >= 30 gray levels in every view where the housing projects inside the image (>= 1 such view) starts a new track. A failed init is
     retried at most every 3rd frame. The masks are used ONLY here (initialisation), never in the per-frame path.
  5. Products (--out DIR): DIR/anchors_silhouette.npz (frames, R (K,3,3), t (K,3): c2w of the WRIST CAMERA, metric, base
     frame; track_id, resid_px, edge_frac, init_frame), DIR/<cam>_<serial>__silhouette_model_masks.npz (rendered silhouette,
     mask contract), DIR/verified_points.npz (frames, xyz = the model body-box centre in the base frame, accepted; also
     lens_xyz) for rig_track2 --verified_points, DIR/seed_info.json (declared constants, parameters, per-frame log, timing,
     cost), DIR/score.json (--diag: pose errors vs the store GT, Sim3 ATE/RPE with rig_track2.traj_metrics), optional
     DIR/silhouette_side_by_side.mp4 (--video).

DECLARED RIG CONSTANTS (measured ONCE on the shelf episode RAIL+80edfcb1+2023-07-14-14h-28m-45s from the hand-prompted SAM3
masks + the kinematic GT, `--measure`; 1 cm voxel visual hull in the wrist-camera frame over the 91 frames with both masks):
  body box    x [-0.02, 0.08]  y [0.01, 0.09]  z [0.02, 0.12] m   (hull at >= 85 % of the observations; the Robotiq 2F-85
              body hangs below (+y) and in front of (+z) the lens, offset +3 cm in x = the lens is the LEFT ZED lens)
  finger box  x [ 0.00, 0.08]  y [0.03, 0.09]  z [0.12, 0.16] m   (hull at >= 60-70 %: the fingers taper, they open/close)
  ZED Mini    x [-0.031, 0.094] y [-0.015, 0.015] z [-0.027, 0.0] m (from the housing spec 124.5 x 30.5 x 26.5 mm around the
              left lens; the hull cannot resolve a 3 cm slab at the mask boundary, so this one is spec-derived)
  The task's nominal 0.09 x 0.15 x 0.20 m Robotiq body is NOT what the hull measures (the mask covers ~0.10 x 0.08 x 0.14 m
  incl. fingers); the measured boxes are used. All three are in seed_info.json as declared_constants.

CAUSALITY: frame t uses frames <= t only (Canny at t, the pose at t-1, rig anchors at <= t, masks at t for the init).
GT-FREE: the store GT is opened only under --diag/--video for scoring/overlay (and by --measure, the one-off shelf
measurement). No DROID-trained component; no learned model at all. Exterior geometry at native 1280x720.
COST: reported in seed_info.json (ms per frame, tracking vs init vs decode); see the study report.

    OMP_NUM_THREADS=1 python seed_silhouette_model.py --episode EP --out DIR \
        --mask_npz_ext1 P1 --mask_npz_ext2 P2 [--anchors rig2/anchors.npz] [--diag] [--video]
    OMP_NUM_THREADS=1 python seed_silhouette_model.py --measure     # re-derive the hull boxes on the shelf episode (GT)
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
import cv2
import numpy as np

cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig_track2 import traj_metrics, rot_angle_deg, load_mask_npz, load_intrinsics, umeyama  # noqa: E402  (read-only reuse)

TOOL_VERSION = "seed_silhouette_model v1 (2026-09-29, idea 5: two-box (+fingers) model, RAPiD edge refinement, hull-PCA global init)"
OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
AUDIT = f"{OUT_ROOT}/vggt_probe/gt_audit_2026-09-08"
INTR_JSON = f"{AUDIT}/docs/hf_intrinsics.json"
CAM_JSON = f"{AUDIT}/pointworld/droid/cameras"
RAW_ROOT = "/gpfs/scratch/etur59/koc821022/vggt_cache/raw"
STORE_ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
SHELF_EP = "RAIL+80edfcb1+2023-07-14-14h-28m-45s"
SHELF_MASKS = f"{OUT_ROOT}/ext_cams/gripper_sam3/{SHELF_EP}"
W, H = 1280, 720
METHOD = "silhouette_model"

# ----------------------------------------------------------------------------- declared rig constants (wrist-camera frame, metres)
MODEL_BOXES = {  # name: (min corner, max corner)
    "body":    ([-0.020, 0.010, 0.020], [0.080, 0.090, 0.120]),
    "fingers": ([0.000, 0.030, 0.120], [0.080, 0.090, 0.160]),
    "zed":     ([-0.031, -0.015, -0.027], [0.094, 0.015, 0.000]),
}
DARK_BOXES = ("body", "fingers")   # object prior: the Robotiq 2F-85 is black on every DROID rig (gray < 70, as the round-3 seeder)
MODEL_SOURCE = ("body/fingers: 1 cm visual hull of the shelf SAM3 masks in the GT wrist frame (RAIL, 91 frames, >= 85 % / >= 60 % of the "
                "observations); zed: ZED Mini housing spec 124.5 x 30.5 x 26.5 mm around the left lens (not resolvable in the hull)")
PARAMS = dict(canny_lo=40, canny_hi=120, sample_px=6.0, search_px=(30, 30, 30, 15, 15, 15, 15, 15), grad_cos_min=0.57, tukey_c=8.0,
              lm_damping=1e-4, gate_med_px=4.0, gate_edge_frac=0.5, gate_min_per_view=12, lost_after=8, roi_pad=60,
              init_dirs=100, init_roll_n=12, init_pos_step=0.03, init_top=12, init_edge_frac=0.6, init_iou_min=0.35, init_scale=8,
              init_retry_every=3, hull_voxel=0.01, hull_half=0.25, hull_min_voxels=150, mask_min_px=1500,
              dark_thresh=70, dark_min=0.25, polarity_px=3, polarity_min=15,
              zed_dark_gray=60, zed_bright_span=100, zed_contrast_min=30)


# ----------------------------------------------------------------------------- se(3) helpers
def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def rot_from_z_roll(z, roll):
    """Rotation whose third column is the unit vector z; roll (rad) about z from a canonical x."""
    z = z / np.linalg.norm(z)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R0 = np.c_[x, y, z]
    return R0 @ so3_exp(np.array([0, 0, roll]))


# ----------------------------------------------------------------------------- model
class BoxModel:
    """Axis-aligned boxes in the wrist frame. Faces indexed (axis j, sign s in {-1,+1}); 12 edges per box."""

    def __init__(self, boxes):
        self.names = list(boxes)
        self.lo = np.array([boxes[n][0] for n in self.names], float)
        self.hi = np.array([boxes[n][1] for n in self.names], float)
        self.c = 0.5 * (self.lo + self.hi); self.h = 0.5 * (self.hi - self.lo)
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        self.verts = self.c[:, None, :] + signs[None] * self.h[:, None, :]  # (B, 8, 3)
        # edges: (axis k, other axes (a, b), signs (sa, sb))
        self.edges = []
        for k in range(3):
            a, b = [j for j in range(3) if j != k]
            for sa in (-1, 1):
                for sb in (-1, 1):
                    self.edges.append((k, a, b, sa, sb))
        self.body_centre = self.c[self.names.index("body")]
        self.dark = np.array([n in DARK_BOXES for n in self.names])
        self.dark_idx = [i for i, n in enumerate(self.names) if n in DARK_BOXES]
        bi = self.names.index("body"); g = np.linspace(-0.6, 0.6, 3)
        self.body_lattice = self.c[bi] + np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) * self.h[bi]  # 27 interior points
        zi = self.names.index("zed"); gz = np.linspace(-0.7, 0.7, 5)
        self.zed_lattice = self.c[zi] + np.stack(np.meshgrid(gz, [0.0], [0.0], indexing="ij"), -1).reshape(-1, 3) * self.h[zi]  # 5 points along the bar

    def silhouette_samples(self, R, t, cam, sample_px, subset=None):
        """Samples on the union silhouette outline for the pose (R, t) (wrist c2w) in camera cam -> dict with X (N,3 model
        frame), uv (N,2), n (N,2 outward unit normals), box (N,). subset: box indices to use (default all)."""
        o_m = R.T @ (cam["centre"] - t)  # camera centre in the model frame
        P = cam["P"]
        Rw = P[:, :3] @ R; tw = P[:, :3] @ t + P[:, 3]  # model -> image homogeneous
        Xs, uvs, ns, bx = [], [], [], []
        polys = []
        use = list(range(len(self.names))) if subset is None else list(subset)
        for bi in range(len(self.names)):
            V = self.verts[bi] @ Rw.T + tw
            if bi not in use or (V[:, 2] <= 1e-3).any():
                polys.append(None); continue
            uv = V[:, :2] / V[:, 2:3]
            polys.append(cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2))
        for bi in range(len(self.names)):
            if polys[bi] is None:
                continue
            c, h = self.c[bi], self.h[bi]
            front = {(j, s): (s * o_m[j] > s * c[j] + h[j]) for j in range(3) for s in (-1, 1)}
            centre_uv = polys[bi].mean(0)
            for (k, a, b, sa, sb) in self.edges:
                if front[(a, sa)] == front[(b, sb)]:
                    continue
                p0 = c.copy(); p0[a] += sa * h[a]; p0[b] += sb * h[b]; p1 = p0.copy()
                p0[k] -= h[k]; p1[k] += h[k]
                q0 = Rw @ p0 + tw; q1 = Rw @ p1 + tw
                if q0[2] <= 1e-3 or q1[2] <= 1e-3:
                    continue
                u0 = q0[:2] / q0[2]; u1 = q1[:2] / q1[2]
                L = np.linalg.norm(u1 - u0)
                n_s = int(np.clip(np.floor(L / sample_px), 2, 60))
                lam = (np.arange(n_s) + 0.5) / n_s
                X = p0[None] + lam[:, None] * (p1 - p0)[None]
                q = X @ Rw.T + tw; uv = q[:, :2] / q[:, 2:3]
                d = (u1 - u0) / max(L, 1e-9); nrm = np.array([-d[1], d[0]])
                if np.dot(nrm, uv.mean(0) - centre_uv) < 0:
                    nrm = -nrm
                Xs.append(X); uvs.append(uv); ns.append(np.repeat(nrm[None], n_s, 0)); bx.append(np.full(n_s, bi))
        if not Xs:
            return None
        X = np.concatenate(Xs); uv = np.concatenate(uvs); n = np.concatenate(ns); bx = np.concatenate(bx)
        keep = np.ones(len(X), bool)
        for bj in range(len(self.names)):
            if polys[bj] is None:
                continue
            other = bx != bj
            if not other.any():
                continue
            pts = uv[other].astype(np.float32)
            inside = np.array([cv2.pointPolygonTest(polys[bj], (float(p[0]), float(p[1])), True) for p in pts]) > 2.0
            keep[np.where(other)[0][inside]] = False
        keep &= (uv[:, 0] >= 1) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 1) & (uv[:, 1] < H - 1)
        if keep.sum() < 4:
            return None
        return dict(X=X[keep], uv=uv[keep], n=n[keep], box=bx[keep], polys=polys)

    def render(self, R, t, cam, shape=(H, W), only=None, subset=None):
        m = np.zeros(shape, np.uint8)
        P = cam["P"]; Rw = P[:, :3] @ R; tw = P[:, :3] @ t + P[:, 3]
        sc = shape[1] / W
        for bi in ([only] if only is not None else (subset if subset is not None else range(len(self.names)))):
            V = self.verts[bi] @ Rw.T + tw
            if (V[:, 2] <= 1e-3).any():
                continue
            uv = (V[:, :2] / V[:, 2:3]) * sc
            hull = cv2.convexHull(np.round(uv).astype(np.int32).reshape(-1, 1, 2))
            cv2.fillConvexPoly(m, hull, 1)
        return m

    def lattice_mean(self, R, t, cam, gray, lattice):
        """Mean gray of the projected lattice points (nan when off-image)."""
        P = cam["P"]; q = lattice @ (P[:, :3] @ R).T + (P[:, :3] @ t + P[:, 3])
        if (q[:, 2] <= 1e-3).any():
            return float("nan")
        uv = q[:, :2] / q[:, 2:3]
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if ok.sum() < max(3, len(lattice) // 3):
            return float("nan")
        return float(gray[uv[ok, 1].astype(int), uv[ok, 0].astype(int)].mean())

    def dark_frac(self, R, t, cam, gray, thresh):
        """Fraction of the projected body-box lattice points that are dark (object prior); nan when off-image."""
        P = cam["P"]; q = self.body_lattice @ (P[:, :3] @ R).T + (P[:, :3] @ t + P[:, 3])
        if (q[:, 2] <= 1e-3).any():
            return float("nan")
        uv = q[:, :2] / q[:, 2:3]
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if ok.sum() < 9:
            return float("nan")
        return float((gray[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] < thresh).mean())

    def bbox(self, R, t, cam):
        P = cam["P"]; Rw = P[:, :3] @ R; tw = P[:, :3] @ t + P[:, 3]
        V = self.verts.reshape(-1, 3) @ Rw.T + tw
        if (V[:, 2] <= 1e-3).any():
            return None
        uv = V[:, :2] / V[:, 2:3]
        return uv.min(0), uv.max(0)


# ----------------------------------------------------------------------------- calibration
def load_calib(ep, cams):
    intr = load_intrinsics(ep, INTR_JSON)
    camj = json.load(open(f"{CAM_JSON}/{ep}_cameras.json"))
    out = {}
    for cam, s in cams:
        fx, cx, fy, cy = intr[s]["cameraMatrix"]
        iw, ih = intr[s].get("width", W), intr[s].get("height", H)
        sx, sy = W / iw, H / ih
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])
        E = np.array(camj[s]["optimized_extrinsics"], np.float64)
        out[cam] = dict(serial=s, K=K, E=E, P=K @ E[:3], centre=-E[:3, :3].T @ E[:3, 3], fx=K[0, 0], fy=K[1, 1])
    return out


def triangulate(P1, P2, x1, x2):
    Xh = cv2.triangulatePoints(P1, P2, np.asarray(x1, np.float64).reshape(2, 1), np.asarray(x2, np.float64).reshape(2, 1))
    return (Xh[:3] / Xh[3]).ravel()


# ----------------------------------------------------------------------------- image features per frame
class FrameEdges:
    """Canny edge map + gradient direction on a ROI of one view (computed once per frame, shared by all iterations)."""

    def __init__(self, gray, roi, lo, hi):
        x0, y0, x1, y1 = roi
        self.x0, self.y0 = x0, y0
        sub = gray[y0:y1, x0:x1]
        self.gray = sub
        self.edge = cv2.Canny(sub, lo, hi) > 0
        gx = cv2.Sobel(sub, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(sub, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy) + 1e-6
        self.gx, self.gy = gx / mag, gy / mag
        self.h, self.w = self.edge.shape

    def search(self, uv, n, radius, cos_min, polarity=None, pol_px=3, pol_min=15):
        """Nearest edge along +/- normal within radius whose gradient is within acos(cos_min) of the normal; for samples
        with polarity True the image must be darker pol_px inside than pol_px outside by >= pol_min (black object).
        Returns the signed distance (nan when none)."""
        N = len(uv)
        s = np.arange(-radius, radius + 1)
        order = np.argsort(np.abs(s), kind="stable"); s = s[order]
        px = uv[:, 0:1] + s[None] * n[:, 0:1] - self.x0; py = uv[:, 1:2] + s[None] * n[:, 1:2] - self.y0
        xi = np.round(px).astype(int); yi = np.round(py).astype(int)
        ok = (xi >= 0) & (xi < self.w) & (yi >= 0) & (yi < self.h)
        xi = np.clip(xi, 0, self.w - 1); yi = np.clip(yi, 0, self.h - 1)
        hit = ok & self.edge[yi, xi]
        cosn = np.abs(self.gx[yi, xi] * n[:, 0:1] + self.gy[yi, xi] * n[:, 1:2])
        hit &= cosn >= cos_min
        if polarity is not None and polarity.any():
            xo = np.clip(np.round(px + pol_px * n[:, 0:1]).astype(int), 0, self.w - 1); yo = np.clip(np.round(py + pol_px * n[:, 1:2]).astype(int), 0, self.h - 1)
            xn = np.clip(np.round(px - pol_px * n[:, 0:1]).astype(int), 0, self.w - 1); yn = np.clip(np.round(py - pol_px * n[:, 1:2]).astype(int), 0, self.h - 1)
            darker = (self.gray[yo, xo].astype(np.int16) - self.gray[yn, xn].astype(np.int16)) >= pol_min
            hit &= darker | ~polarity[:, None]
        first = np.argmax(hit, axis=1)
        found = hit[np.arange(N), first]
        r = np.where(found, s[first].astype(float), np.nan)
        return r


class MaskEdges(FrameEdges):
    """Edge map = the boundary of a binary mask (initialisation only), gradient from the mask itself; polarity uses the real gray."""

    def __init__(self, gray, mask):
        self.x0 = self.y0 = 0
        m = mask.astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        self.edge = (cv2.dilate(m, k) - cv2.erode(m, k)) > 0
        mf = cv2.GaussianBlur(m.astype(np.float32), (5, 5), 0)
        gx = cv2.Sobel(mf, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(mf, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy) + 1e-6
        self.gx, self.gy = gx / mag, gy / mag
        self.gray = gray; self.full_gray = gray
        self.h, self.w = self.edge.shape


# ----------------------------------------------------------------------------- refinement (RAPiD / Gauss-Newton on se(3))
def refine(model, R, t, cams, edges, P, subset=None):
    """Returns (R, t, stats). Body-frame update: X_w = R (X_m + dtheta x X_m + dt) + t. subset: boxes whose silhouette is used."""
    stats = dict(med=np.nan, frac=0.0, per_view=[], n=0)
    for it, radius in enumerate(P["search_px"]):
        Js, rs = [], []
        per_view = []
        for cam in ("ext1", "ext2"):
            c = cams[cam]
            smp = model.silhouette_samples(R, t, c, P["sample_px"], subset)
            if smp is None or edges[cam] is None:
                per_view.append((0, 0)); continue
            r = edges[cam].search(smp["uv"], smp["n"], radius, P["grad_cos_min"], model.dark[smp["box"]], P["polarity_px"], P["polarity_min"])
            ok = np.isfinite(r)
            per_view.append((int(ok.sum()), int(len(r))))
            if ok.sum() < 3:
                continue
            X = smp["X"][ok]; n = smp["n"][ok]; r = r[ok]
            # jacobian: du/dxi = (du/dXc) (Rk) (dXw/dxi); dXw/dtheta = -R [X]_x, dXw/dt = R
            Rk = c["E"][:3, :3]; Xc = (Rk @ (R @ X.T + t[:, None]) + c["E"][:3, 3:]).T
            z = Xc[:, 2]; fx, fy = c["fx"], c["fy"]
            A = np.zeros((len(X), 2, 3))
            A[:, 0, 0] = fx / z; A[:, 0, 2] = -fx * Xc[:, 0] / z ** 2
            A[:, 1, 1] = fy / z; A[:, 1, 2] = -fy * Xc[:, 1] / z ** 2
            skew = np.zeros((len(X), 3, 3))
            skew[:, 0, 1] = -X[:, 2]; skew[:, 0, 2] = X[:, 1]; skew[:, 1, 0] = X[:, 2]; skew[:, 1, 2] = -X[:, 0]; skew[:, 2, 0] = -X[:, 1]; skew[:, 2, 1] = X[:, 0]
            dXw = np.concatenate([-(R[None] @ skew), np.repeat(R[None], len(X), 0)], axis=2)  # (N,3,6): [theta | t]
            Ju = A @ (Rk[None] @ dXw)  # (N,2,6)
            J = (n[:, :, None] * Ju).sum(1)  # (N,6)
            Js.append(J); rs.append(r)
        if not Js:
            stats.update(med=np.nan, frac=0.0, per_view=per_view, n=0); return R, t, stats
        J = np.concatenate(Js); r = np.concatenate(rs)
        n_tot = sum(v[1] for v in per_view); n_ok = sum(v[0] for v in per_view)
        u = np.abs(r) / P["tukey_c"]
        w = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
        JtW = J.T * w
        Hm = JtW @ J + P["lm_damping"] * np.eye(6) * (1.0 + np.trace(JtW @ J) / 6)
        g = JtW @ r
        try:
            xi = np.linalg.solve(Hm, g)
        except np.linalg.LinAlgError:
            break
        xi[:3] = np.clip(xi[:3], -0.35, 0.35); xi[3:] = np.clip(xi[3:], -0.08, 0.08)
        dR = so3_exp(xi[:3])
        # X_w = R (dR X + dt) + t  ->  R' = R dR, t' = R dt + t
        t = R @ xi[3:] + t; R = R @ dR
        stats.update(med=float(np.median(np.abs(r))), frac=float(n_ok / max(n_tot, 1)), per_view=per_view, n=int(n_ok))
    # final evaluation at the converged pose with the last radius (+ the dark-interior object prior per view)
    Js, rs, per_view, dark = [], [], [], []
    box_ok = np.zeros(len(model.names)); box_n = np.zeros(len(model.names))
    for cam in ("ext1", "ext2"):
        c = cams[cam]
        smp = model.silhouette_samples(R, t, c, P["sample_px"], subset)
        if smp is None or edges[cam] is None:
            per_view.append((0, 0)); dark.append(float("nan")); continue
        r = edges[cam].search(smp["uv"], smp["n"], P["search_px"][-1], P["grad_cos_min"], model.dark[smp["box"]], P["polarity_px"], P["polarity_min"])
        ok = np.isfinite(r); per_view.append((int(ok.sum()), int(len(r)))); rs.append(r[ok])
        dark.append(model.dark_frac(R, t, c, edges[cam].full_gray, P["dark_thresh"]))
        box_ok += np.bincount(smp["box"][ok], minlength=len(model.names)); box_n += np.bincount(smp["box"], minlength=len(model.names))
    r = np.concatenate(rs) if rs else np.zeros(0)
    n_tot = sum(v[1] for v in per_view); n_ok = sum(v[0] for v in per_view)
    stats.update(med=float(np.median(np.abs(r))) if len(r) else np.nan, frac=float(n_ok / max(n_tot, 1)), per_view=per_view, n=int(n_ok), dark=dark,
                 box_frac=[float(box_ok[i] / box_n[i]) if box_n[i] else float("nan") for i in range(len(model.names))])
    return R, t, stats


def gate_ok(stats, P):
    return (np.isfinite(stats["med"]) and stats["med"] <= P["gate_med_px"] and stats["frac"] >= P["gate_edge_frac"]
            and all(v[0] >= P["gate_min_per_view"] for v in stats["per_view"])
            and all(np.isfinite(d) and d >= P["dark_min"] for d in stats.get("dark", [])))


def make_edges(model, R, t, cams, gray, P):
    out = {}
    for cam in ("ext1", "ext2"):
        bb = model.bbox(R, t, cams[cam])
        if bb is None:
            out[cam] = None; continue
        pad = P["roi_pad"] + max(P["search_px"])
        x0, y0 = int(max(0, bb[0][0] - pad)), int(max(0, bb[0][1] - pad))
        x1, y1 = int(min(W, bb[1][0] + pad)), int(min(H, bb[1][1] + pad))
        if x1 - x0 < 8 or y1 - y0 < 8:
            out[cam] = None; continue
        out[cam] = FrameEdges(gray[cam], (x0, y0, x1, y1), P["canny_lo"], P["canny_hi"])
        out[cam].full_gray = gray[cam]
    return out


# ----------------------------------------------------------------------------- global initialisation from the masks
def two_view_hull(masks, cams, P):
    """Voxel visual hull of the two masks around their triangulated centroids -> (centroid, major axis, n_voxels, seed point)."""
    cen = {}
    for cam in ("ext1", "ext2"):
        ys, xs = np.nonzero(masks[cam])
        if len(xs) < 50:
            return None
        cen[cam] = (xs.mean(), ys.mean())
    seed = triangulate(cams["ext1"]["P"], cams["ext2"]["P"], cen["ext1"], cen["ext2"])
    if not np.all(np.isfinite(seed)) or np.linalg.norm(seed) > 1.5:
        return None
    ax = np.arange(-P["hull_half"], P["hull_half"] + 1e-9, P["hull_voxel"])
    G = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3) + seed
    inside = np.ones(len(G), bool)
    for cam in ("ext1", "ext2"):
        q = cams[cam]["P"] @ np.r_[G.T, np.ones((1, len(G)))]
        z = q[2]; u = q[0] / np.where(z > 1e-6, z, 1e-6); v = q[1] / np.where(z > 1e-6, z, 1e-6)
        ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui = np.clip(u, 0, W - 1).astype(int); vi = np.clip(v, 0, H - 1).astype(int)
        inside &= ok & masks[cam][vi, ui]
    pts = G[inside]
    if len(pts) < 30:
        return None
    hc = pts.mean(0)
    ev, evec = np.linalg.eigh(np.cov((pts - hc).T))
    a = evec[:, -1]
    if np.dot(a, hc - np.zeros(3)) < 0:  # orient the axis away from the base origin (distal end = fingers), both signs are tried anyway
        a = -a
    return dict(centroid=hc, axis=a, n=int(len(pts)), seed=seed, extent=np.sqrt(np.maximum(ev, 0)))


def fibonacci_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def iou_bool(a, b):
    u = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / u if u else 0.0


def global_init(model, cams, gray, masks, P):
    """Rasterised silhouette-IoU search (dark boxes vs the mask, 1/init_scale resolution) over a uniform rotation grid x a
    small position grid around the two-view hull centroid, then two-stage refinement of the best init_top candidates
    (mask boundary with the dark boxes, then image edges with all boxes). Returns (R, t, stats, info) or None."""
    for cam in ("ext1", "ext2"):
        if np.count_nonzero(masks[cam]) < P["mask_min_px"]:
            return None
    hull = two_view_hull(masks, cams, P)
    if hull is None or hull["n"] < P["hull_min_voxels"]:
        return None
    sc = P["init_scale"]; shp = (H // sc, W // sc)
    m_lo = {cam: cv2.resize(masks[cam].astype(np.uint8), (shp[1], shp[0]), interpolation=cv2.INTER_NEAREST) > 0 for cam in cams}
    dirs = fibonacci_dirs(P["init_dirs"])
    rolls = np.arange(P["init_roll_n"]) * (2 * np.pi / P["init_roll_n"])
    st = P["init_pos_step"]
    pos = [hull["centroid"] + d for d in ([np.zeros(3)] + [np.eye(3)[k] * sg * st for k in range(3) for sg in (-1, 1)])]
    bc = model.body_centre
    dv = model.verts[model.dark_idx].reshape(-1, 3)  # dark boxes' vertices (model frame)
    nb = len(model.dark_idx)
    Pm = {cam: (cams[cam]["P"][:, :3], cams[cam]["P"][:, 3]) for cam in cams}
    cands = []
    for z in dirs:
        for roll in rolls:
            R = rot_from_z_roll(z, roll)
            dvR = dv @ R.T; Rbc = R @ bc
            for pb in pos:
                t = pb - Rbc
                sc_sum, nv = 0.0, 0
                for cam in ("ext1", "ext2"):
                    A, b = Pm[cam]
                    q = (dvR + t) @ A.T + b
                    if (q[:, 2] <= 1e-3).any():
                        break
                    uv = (q[:, :2] / q[:, 2:3]) / sc
                    m = np.zeros(shp, np.uint8)
                    for k in range(nb):
                        hull_k = cv2.convexHull(np.round(uv[8 * k:8 * k + 8]).astype(np.int32).reshape(-1, 1, 2))
                        cv2.fillConvexPoly(m, hull_k, 1)
                    iz = model.lattice_mean(R, t, cams[cam], gray[cam], model.zed_lattice)
                    zb = 0.0 if not np.isfinite(iz) else float(np.clip((iz - P["zed_dark_gray"]) / P["zed_bright_span"], 0.0, 1.0))
                    sc_sum += iou_bool(m > 0, m_lo[cam]) * (0.5 + 0.5 * zb); nv += 1
                if nv == 2:
                    cands.append((sc_sum / 2, R, t))
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    best = None
    medges = {cam: MaskEdges(gray[cam], masks[cam]) for cam in ("ext1", "ext2")}
    tried = []
    for sc0, R, t in cands[:P["init_top"]]:
        R1, t1, _ = refine(model, R, t, cams, medges, P, subset=model.dark_idx)   # stage 1: dark boxes vs the mask boundary
        edges = make_edges(model, R1, t1, cams, gray, P)
        R2, t2, st = refine(model, R1, t1, cams, edges, P)                        # stage 2: all boxes vs the image edges
        ious = [iou_bool(model.render(R2, t2, cams[cam], (H // 4, W // 4), subset=model.dark_idx) > 0,
                         cv2.resize(masks[cam].astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_NEAREST) > 0) for cam in ("ext1", "ext2")]
        tried.append(dict(coarse=round(float(sc0), 3), iou=[round(v, 3) for v in ious], med=st["med"], frac=round(st["frac"], 2), dark=[round(d, 2) if np.isfinite(d) else None for d in st["dark"]], gate=bool(gate_ok(st, P)),
                          box_frac=[round(v, 2) if np.isfinite(v) else None for v in st["box_frac"]], R=R2.tolist(), t=t2.tolist()))
        contrast = [model.lattice_mean(R2, t2, cams[cam], gray[cam], model.zed_lattice) - model.lattice_mean(R2, t2, cams[cam], gray[cam], model.body_lattice) for cam in ("ext1", "ext2")]
        tried[-1]["zed_minus_body"] = [round(v, 1) if np.isfinite(v) else None for v in contrast]
        testable = [v for v in contrast if np.isfinite(v)]   # a view where the housing projects off-image is neutral, not a failure
        if gate_ok(st, P) and st["frac"] >= P["init_edge_frac"] and min(ious) >= P["init_iou_min"] and len(testable) >= 1 and all(v >= P["zed_contrast_min"] for v in testable):
            key = float(np.mean(ious)) * (0.5 + 0.5 * float(np.clip(np.mean(testable) / P["zed_bright_span"], 0, 1)))
            if best is None or key > best[0]:
                best = (key, R2, t2, st, dict(coarse_score=float(sc0), iou_after=ious, hull_n=hull["n"], hull_extent_cm=(hull["extent"] * 100).round(1).tolist(),
                                              n_candidates=len(cands), seed=hull["seed"].tolist()))
    if best is None:
        return dict(failed=True, tried=tried, hull_n=hull["n"], n_candidates=len(cands))
    best[4]["tried"] = tried
    return best[1], best[2], best[3], best[4]


# ----------------------------------------------------------------------------- video (frames read again, overlay)
def write_video(out_mp4, ep, cams, model, poses, T, gt, masks_init):
    from sam3_gripper_masks import FfmpegWriter
    caps = {cam: cv2.VideoCapture(f"{RAW_ROOT}/{ep}/recordings/MP4/{cams[cam]['serial']}.mp4") for cam in cams}
    PW, PH = 640, 360
    wr = FfmpegWriter(out_mp4, 2 * PW, PH + 24, 15)
    for t in range(T):
        panels = []
        for cam in ("ext1", "ext2"):
            ok, fr = caps[cam].read()
            if not ok:
                fr = np.zeros((H, W, 3), np.uint8)
            if t in poses:
                R, tt = poses[t]
                m = model.render(R, tt, cams[cam])
                fr[m > 0] = (0.55 * fr[m > 0] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(fr, cnts, -1, (0, 255, 0), 2)
                q = cams[cam]["P"] @ np.r_[tt, 1]
                if q[2] > 0:
                    cv2.circle(fr, (int(q[0] / q[2]), int(q[1] / q[2])), 7, (255, 0, 255), 2)
            if gt is not None and t in gt:
                q = cams[cam]["P"] @ np.r_[gt[t][:3, 3], 1]
                if q[2] > 0:
                    cv2.circle(fr, (int(q[0] / q[2]), int(q[1] / q[2])), 6, (0, 0, 255), -1)
            if masks_init is not None and masks_init[cam][t].any():
                cnts, _ = cv2.findContours(masks_init[cam][t].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(fr, cnts, -1, (255, 200, 0), 1)
            panels.append(cv2.resize(fr, (PW, PH), interpolation=cv2.INTER_AREA))
        img = np.zeros((PH + 24, 2 * PW, 3), np.uint8); img[24:] = np.hstack(panels)
        cv2.putText(img, f"{ep[:40]} f{t}  green=model silhouette  magenta=est lens  red=GT lens (diag)  yellow=init mask", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        wr.write(img)
    wr.close()
    for c in caps.values():
        c.release()


# ----------------------------------------------------------------------------- one-off measurement on the shelf episode (GT)
def measure(args):
    ep = SHELF_EP
    meta = json.load(open(f"{RAW_ROOT}/{ep}/metadata_{ep}.json"))
    cams_l = [("ext1", str(meta["ext1_cam_serial"])), ("ext2", str(meta["ext2_cam_serial"]))]
    cams = load_calib(ep, cams_l)
    M = {cam: load_mask_npz(f"{SHELF_MASKS}/{cam}_{s}__gripper_boxclick_masks.npz").astype(bool) for cam, s in cams_l}
    T = min(len(M["ext1"]), len(M["ext2"]))
    gt = np.array([np.load(f"{STORE_ROOT}/{ep}/dense/cam/{t:06d}.npz")["pose"] for t in range(T)])
    ax = np.arange(-0.30, 0.30 + 1e-9, 0.01)
    G = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    cnt = np.zeros(len(G), int); tot = np.zeros(len(G), int)
    frames = [t for t in range(T) if M["ext1"][t].any() and M["ext2"][t].any()]
    for t in frames:
        Xw = (gt[t][:3, :3] @ G.T).T + gt[t][:3, 3]
        for cam in ("ext1", "ext2"):
            q = cams[cam]["P"] @ np.r_[Xw.T, np.ones((1, len(Xw)))]
            z = q[2]; u = q[0] / z; v = q[1] / z
            ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            tot += ok
            cnt += ok & M[cam][t][np.clip(v, 0, H - 1).astype(int), np.clip(u, 0, W - 1).astype(int)]
    frac = cnt / np.maximum(tot, 1); seen = tot >= 0.8 * 2 * len(frames)
    rep = dict(episode=ep, frames=len(frames), voxel_m=0.01)
    for th in (0.95, 0.85, 0.7, 0.6):
        pts = G[seen & (frac >= th)]
        rep[f"hull_{th}"] = dict(n=int(len(pts)), p2=np.percentile(pts, 2, axis=0).round(3).tolist(), p98=np.percentile(pts, 98, axis=0).round(3).tolist()) if len(pts) else None
    print(json.dumps(rep, indent=1))
    print("declared MODEL_BOXES:", json.dumps(MODEL_BOXES))


# ----------------------------------------------------------------------------- main tracking
def run(args):
    t_all = time.time()
    ep = args.episode
    P = dict(PARAMS)
    meta = json.load(open(f"{RAW_ROOT}/{ep}/metadata_{ep}.json"))
    cams_l = [("ext1", str(meta["ext1_cam_serial"])), ("ext2", str(meta["ext2_cam_serial"]))]
    cams = load_calib(ep, cams_l)
    model = BoxModel(MODEL_BOXES)
    store_cam = f"{STORE_ROOT}/{ep}/dense/cam"
    T_store = len([f for f in os.listdir(store_cam) if f.endswith(".npz")])
    Mi = {cam: load_mask_npz(p).astype(bool) for cam, p in (("ext1", args.mask_npz_ext1), ("ext2", args.mask_npz_ext2))}
    T = min(T_store, len(Mi["ext1"]), len(Mi["ext2"]))
    anchors = None
    if args.anchors and os.path.isfile(args.anchors):
        an = np.load(args.anchors)
        anchors = {int(f): (int(g), np.asarray(R), np.asarray(tr)) for f, g, R, tr in zip(an["frames"], an["segment_id"], an["R"], an["trans"])}
    gt = None
    if args.diag or args.video:
        gt = {t: np.load(f"{store_cam}/{t:06d}.npz")["pose"].astype(np.float64) for t in range(T)}
    caps = {cam: cv2.VideoCapture(f"{RAW_ROOT}/{ep}/recordings/MP4/{cams[cam]['serial']}.mp4") for cam in cams}
    for c in caps.values():
        if hasattr(cv2, "CAP_PROP_N_THREADS"):
            c.set(cv2.CAP_PROP_N_THREADS, 1)
    poses, log = {}, []
    masks_out = {cam: np.zeros((T, H, W // 8), np.uint8) for cam in cams}
    present = {cam: np.zeros(T, bool) for cam in cams}
    vp_xyz = np.full((T, 3), np.nan); lens_xyz = np.full((T, 3), np.nan)
    track_id, cur_track = -1, None
    last_solved = None            # (t, R, t) of the last solved frame
    seg_last = {}                 # segment id -> (t_a, R, t) last solved frame that is anchored in that segment
    n_reject_streak = 0
    last_init_try = -10 ** 6
    timing = dict(decode=0.0, track=0.0, init=0.0, render=0.0, n_init_calls=0, n_track_calls=0)
    init_frames = []
    for t in range(T):
        t0 = time.time()
        gray = {}
        for cam in cams:
            ok, fr = caps[cam].read()
            if not ok:
                fr = np.zeros((H, W, 3), np.uint8)
            gray[cam] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        timing["decode"] += time.time() - t0
        row = dict(frame=t, status="idle", source=None)
        # ---- prediction
        pred = None
        if last_solved is not None and cur_track is not None:
            if anchors is not None and t in anchors and anchors[t][0] in seg_last:
                g, Rt, trt = anchors[t]
                ta, Ra_, ta_ = seg_last[g]
                _, Ra, tra = anchors[ta]
                # M_t M_ta^{-1} T_a : X = Rt (Ra^T (Y - tra)) + trt
                Rrel = Rt @ Ra.T; trel = trt - Rrel @ tra
                pred = (Rrel @ Ra_, Rrel @ ta_ + trel); row["source"] = f"anchor(seg{g},from f{ta})"
            elif t - last_solved[0] <= P["lost_after"]:
                pred = (last_solved[1], last_solved[2]); row["source"] = f"prev(f{last_solved[0]})"
        t1 = time.time()
        solved = None
        if pred is not None:
            edges = make_edges(model, pred[0], pred[1], cams, gray, P)
            R, tt, st = refine(model, pred[0], pred[1], cams, edges, P)
            timing["n_track_calls"] += 1
            if gate_ok(st, P):
                solved = (R, tt, st)
                row["status"] = "solved"
            else:
                row["status"] = "rejected"; row.update(med_px=st["med"], edge_frac=st["frac"])
                n_reject_streak += 1
                if n_reject_streak >= P["lost_after"]:
                    cur_track = None; row["status"] = "lost"
        timing["track"] += time.time() - t1
        if solved is None and cur_track is None and Mi["ext1"][t].any() and Mi["ext2"][t].any() and (t - last_init_try >= P["init_retry_every"]):
            t2 = time.time(); last_init_try = t
            res = global_init(model, cams, gray, {cam: Mi[cam][t] for cam in cams}, P)
            timing["init"] += time.time() - t2; timing["n_init_calls"] += 1
            if res is not None and not isinstance(res, dict):
                R, tt, st, info = res
                track_id += 1; cur_track = track_id; seg_last = {}
                solved = (R, tt, st); row["status"] = "solved"; row["source"] = "init"; row["init"] = info
                init_frames.append(t)
            else:
                row["status"] = "init_failed" if row["status"] in ("idle", "lost") else row["status"]
                row["init_fail"] = res
        if solved is not None:
            R, tt, st = solved
            n_reject_streak = 0
            last_solved = (t, R, tt)
            if anchors is not None and t in anchors:
                seg_last[anchors[t][0]] = (t, R, tt)
            poses[t] = (R, tt)
            row.update(track=cur_track, med_px=st["med"], edge_frac=st["frac"], n_edges=st["n"], per_view=st["per_view"], dark=st.get("dark"))
            t3 = time.time()
            for cam in cams:
                m = model.render(R, tt, cams[cam])
                masks_out[cam][t] = np.packbits(m, axis=1); present[cam][t] = bool(m.any())
            timing["render"] += time.time() - t3
            vp_xyz[t] = R @ model.body_centre + tt; lens_xyz[t] = tt
            if gt is not None:
                row["pos_err_cm"] = float(np.linalg.norm(tt - gt[t][:3, 3]) * 100)
                row["rot_err_deg"] = float(rot_angle_deg(R.T @ gt[t][:3, :3]))
        log.append(row)
    for c in caps.values():
        c.release()
    elapsed = time.time() - t_all
    os.makedirs(args.out, exist_ok=True)
    for cam in cams:
        np.savez_compressed(f"{args.out}/{cam}_{cams[cam]['serial']}__{METHOD}_masks.npz", union=masks_out[cam], shape=np.array([T, H, W]), frames_present=present[cam])
    fr = np.array(sorted(poses), int)
    Rs = np.array([poses[t][0] for t in fr]).reshape(-1, 3, 3); ts = np.array([poses[t][1] for t in fr]).reshape(-1, 3)
    rows_by = {r["frame"]: r for r in log}
    tid = np.array([rows_by[t].get("track", -1) for t in fr], int)
    np.savez(f"{args.out}/anchors_silhouette.npz", frames=fr, R=Rs, t=ts, track_id=tid,
             resid_px=np.array([rows_by[t]["med_px"] for t in fr]), edge_frac=np.array([rows_by[t]["edge_frac"] for t in fr]),
             init_frames=np.array(init_frames, int), episode=ep, serials=np.array([cams[c]["serial"] for c in ("ext1", "ext2")]),
             product_note="GT-FREE: R, t = c2w pose of the WRIST CAMERA (left ZED lens) in the robot base frame, metres, on solved frames only")
    np.savez(f"{args.out}/verified_points.npz", frames=np.arange(T), xyz=vp_xyz, accepted=np.isfinite(vp_xyz[:, 0]), lens_xyz=lens_xyz,
             note="xyz = model body-box centre (base frame) on solved frames, NaN otherwise; lens_xyz = estimated wrist-camera position")
    n_solved = int(len(fr))
    info = dict(episode=ep, method=METHOD, tool=TOOL_VERSION, serials={c: cams[c]["serial"] for c in cams}, n_frames=T, n_frames_store=T_store,
                causal=True, uses_gt=bool(args.diag or args.video), gt_use="scoring/overlay only (--diag/--video)",
                declared_constants=dict(model_boxes_wrist_frame_m=MODEL_BOXES, source=MODEL_SOURCE, calibration="PointWorld optimized_extrinsics (world->cam) + factory intrinsics"),
                params={k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()},
                inputs=dict(mask_npz_ext1=args.mask_npz_ext1, mask_npz_ext2=args.mask_npz_ext2, anchors=args.anchors if anchors is not None else None,
                            mask_role="global initialisation only"),
                summary=dict(n_solved=n_solved, coverage=n_solved / T, n_tracks=track_id + 1, init_frames=init_frames,
                             n_rejected=sum(1 for r in log if r["status"] == "rejected"), n_lost=sum(1 for r in log if r["status"] == "lost"),
                             n_init_failed=sum(1 for r in log if r["status"] == "init_failed"),
                             sources={s: sum(1 for r in log if r["status"] == "solved" and str(r.get("source", "")).split("(")[0] == s) for s in ("prev", "anchor", "init")},
                             resid_px_median=float(np.median([rows_by[t]["med_px"] for t in fr])) if n_solved else None),
                timing=dict(elapsed_s=elapsed, decode_s=timing["decode"], track_s=timing["track"], init_s=timing["init"], render_s=timing["render"],
                            n_init_calls=timing["n_init_calls"], n_track_calls=timing["n_track_calls"],
                            ms_per_frame_total=1000 * elapsed / T, ms_per_frame_decode=1000 * timing["decode"] / T,
                            ms_per_track_call=(1000 * timing["track"] / timing["n_track_calls"]) if timing["n_track_calls"] else None,
                            ms_per_init_call=(1000 * timing["init"] / timing["n_init_calls"]) if timing["n_init_calls"] else None,
                            ms_per_frame_excl_init=1000 * (elapsed - timing["init"]) / T),
                products=dict(masks=[f"{args.out}/{cam}_{cams[cam]['serial']}__{METHOD}_masks.npz" for cam in cams], anchors=f"{args.out}/anchors_silhouette.npz",
                              verified_points=f"{args.out}/verified_points.npz"),
                per_frame=log)
    # ---- --diag scoring (GT)
    if args.diag and n_solved:
        pe = np.array([rows_by[t]["pos_err_cm"] for t in fr]); re_ = np.array([rows_by[t]["rot_err_deg"] for t in fr])
        pred = [np.r_[np.c_[poses[t][0], poses[t][1]], [[0, 0, 0, 1]]] for t in fr]
        gtl = [gt[t] for t in fr]
        m_sim3 = traj_metrics(pred, gtl, tid) if n_solved >= 2 else None
        # metric (no alignment) ATE and per-track Sim3
        ate_metric = float(np.sqrt(np.mean(pe ** 2))) / 100
        per_track = []
        for g in sorted(set(tid.tolist())):
            idx = np.where(tid == g)[0]
            if len(idx) >= 2:
                mt = traj_metrics([pred[i] for i in idx], [gtl[i] for i in idx])
                per_track.append(dict(track=int(g), n=int(len(idx)), ate=mt["ate"], rpe_trans=mt["rpe_trans"], rpe_rot=mt["rpe_rot"], sim3_scale=mt["sim3_scale"],
                                      pos_err_median_cm=float(np.median(pe[idx])), rot_err_median_deg=float(np.median(re_[idx]))))
        # CUT3R on the same frames (finetuned preds) for the score rule
        cut_dir = f"{OUT_ROOT}/augfull_lr1e5/preds/{ep}/camera"
        m_cut = None
        if os.path.isdir(cut_dir) and n_solved >= 2:
            try:
                cut = [np.load(f"{cut_dir}/{t:06d}.npz")["pose"].astype(np.float64) for t in fr]
                m_cut = traj_metrics(cut, gtl, tid)
            except Exception as e:  # noqa: BLE001
                m_cut = dict(error=repr(e))
        wb = float((np.linalg.norm(ts - np.array([gt[t][:3, 3] for t in fr]), axis=1) > 0.20).mean())
        score = dict(note="GT USED (scoring only): pose product vs kinematic GT; Sim3 metrics via rig_track2.traj_metrics (tracks concatenated, RPE skips track changes)",
                     n_solved=n_solved, coverage=n_solved / T, pos_err_median_cm=float(np.median(pe)), pos_err_p90_cm=float(np.percentile(pe, 90)),
                     rot_err_median_deg=float(np.median(re_)), rot_err_p90_deg=float(np.percentile(re_, 90)),
                     ate_metric_no_alignment=ate_metric, wrong_body_frac_gt=wb, sim3=m_sim3, cut3r_same_frames=m_cut, per_track=per_track,
                     max_jump_cm=float(max([np.linalg.norm(ts[i] - ts[i - 1]) * 100 for i in range(1, n_solved) if tid[i] == tid[i - 1] and fr[i] == fr[i - 1] + 1] or [0])))
        info["score"] = {k: v for k, v in score.items() if k != "per_track"}
        json.dump(clean(score), open(f"{args.out}/score.json", "w"), indent=1)
    json.dump(clean(info), open(f"{args.out}/seed_info.json", "w"))
    if args.video:
        write_video(f"{args.out}/silhouette_side_by_side.mp4", ep, cams, model, poses, T, gt, Mi)
    s = info["summary"]; tm = info["timing"]
    print(f"{ep} T={T} solved={n_solved} ({100 * n_solved / T:.0f}%) tracks={s['n_tracks']} init_frames={init_frames[:8]} rejected={s['n_rejected']} "
          f"resid_med={s['resid_px_median']} | {tm['ms_per_frame_total']:.1f} ms/frame total, {tm['ms_per_frame_excl_init']:.1f} excl init, "
          f"track {tm['ms_per_track_call']} ms/call, init {tm['ms_per_init_call']} ms/call x{tm['n_init_calls']}"
          + (f" | pos {info['score']['pos_err_median_cm']:.2f} cm rot {info['score']['rot_err_median_deg']:.2f} deg wb {info['score']['wrong_body_frac_gt']:.3f} ATE(sim3) {info['score']['sim3']['ate'] if info['score']['sim3'] else None}" if "score" in info else ""), flush=True)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode"); ap.add_argument("--out")
    ap.add_argument("--mask_npz_ext1"); ap.add_argument("--mask_npz_ext2")
    ap.add_argument("--anchors", default=None, help="rig_track2 anchors.npz (round-3 tracker) for the anchor-based prediction")
    ap.add_argument("--diag", action="store_true"); ap.add_argument("--video", action="store_true")
    ap.add_argument("--measure", action="store_true", help="re-derive the model boxes on the shelf episode (GT; one-off)")
    a = ap.parse_args()
    if a.measure:
        measure(a); return
    if not (a.episode and a.out and a.mask_npz_ext1 and a.mask_npz_ext2):
        ap.error("--episode, --out, --mask_npz_ext1, --mask_npz_ext2 are required")
    run(a)


if __name__ == "__main__":
    main()
