#!/usr/bin/env python
"""seed_verified_sam3: ROUND-2 seeding method `verified_sam3`: gripper masks on the two STATIC exterior DROID cameras of one
episode. GPU, GT-free, strictly causal, training-free (the only learned component is the pretrained generic SAM 3).

Per frame t, both views in lockstep (only frames <= t are read; whatever is emitted at t is final, never revised):
  CANDIDATES per view, each a compact blob with its image centroid, box, long side and a skeleton distance from the arm's
  entry (the ROOT: from the declared calibration, the border-contact cluster nearest to the projected robot base, or the
  component pixel nearest to the base when the base projects inside the image, e.g. 44bb9c36 ext1; without a base
  projection (RAIL ext2: base behind the camera) all border contacts are roots, as in seed_motion_only.py):
    (i)  motion: OpenCV MOG2 running background model (static cameras; seed_motion_only.py settings incl. the dark-
         achromatic shadow rule), restricted to pixels that moved within the last --recent_k frames (frame difference
         with the global exposure-step shift removed; this kills MOG2 ghosts and flicker) -> foreground components ->
         per component: 'cap' = the distal cap (pixels within a geodesic length L of the component's far end, measured
         from the root; the far end is taken among currently moving pixels), 'compact' = the whole component when it is
         small enough to be a hand-sized object at the near end of the workspace, 'dark' = compact dark achromatic
         sub-blobs of the component (the black gripper body / wrist camera inside the white arm; a 9 px opening cuts the
         black cable off), with the skeleton distance of their far end.
    (ii) text (only while no verified track is active, every --text_every frames): SAM 3 image model, phrase "robotic arm"
         -> far end of the arm mask along its skeleton (arm_tip of seed_text_arm_sam3.py) -> 'text_tip' = the arm pixels
         within L of that end, 'text_beyond' = the moving foreground just beyond it (the text mask often stops at the
         wrist flange and excludes the black gripper).
    (iii) cross-view transfer (GPU mode, seeking only): a black gripper over a black table is invisible to (i) in one
         view (44bb9c36 ext1) while the other view sees it against the bright background. A strong-view candidate ('dark'
         first, then farthest along the skeleton, --transfer_k per direction) is carried along its ray into the weak view;
         the depth is the one at which the epipolar segment passes nearest (< --transfer_anchor_m at that depth) to a
         weak-view candidate centroid (the wrist-flange cap, a dark blob); the weak-view proposal is a synthetic box of
         --transfer_size_m at that depth with one click at its centre, and SAM decides the object. Such a pair passes the
         geometric pre-test by construction, ranks after every real verified pair, and is accepted ONLY if the two SAM
         masks pass the 3D test below.
  VERIFICATION with the declared calibration only (PointWorld optimized_extrinsics = world(base)->camera, factory
  intrinsics; no GT anywhere): every (ext1 candidate, ext2 candidate) pair is triangulated from the two centroids and
  accepted only when
    - the symmetric epipolar residual (mean of the two point-to-epipolar-line distances, native px) is < --epi_max (6),
    - the 3D point lies within --reach m of the base origin (0.9) and above the table (z > --z_min = -0.15 m),
    - it is in front of both cameras, and
    - its apparent size in each view (long side of the candidate's box x depth / f) is consistent with a
      --size_min..--size_max m object (0.10-0.25, widened by --size_tol).
  SELECTION among verified pairs: real pairs before transfer proposals; pairs temporally consistent with the previous
  accepted 3D position (within --jump_m per frame, --memory_frames) first; then the largest skeleton distance from the
  entry (the end effector is the arm's distal end). A new acquisition from a real pair also needs a verified pair on one
  of the --persist previous frames within the jump limit.
  PROMPT: box + up to 3 positive clicks (distance-transform maxima of the candidate blob and of its two principal-axis
  halves) on the SAM 3 SAM2-style tracker (build_sam3_video_model, the seed_motion_sam3.py / sam3_gripper_boxclick_3cams.py
  recipe), in each view on the SAME frame. The two prompt-frame masks must pass a sanity check (area window, centroid inside
  the box, box IoU) AND the same 3D test on their centroids; otherwise the prompt is cleared in both views and the next
  verified pair is tried (--max_pair_tries per frame).
  TRACKING: forward only, one frame per call in both views (propagate_in_video(start_frame_idx=t, max_frame_num_to_track=0)).
  The emitted mask is the SAM object (largest component plus fragments >= --frag_keep of it), never the moving blob.
  Every frame the two mask centroids are re-verified with the same 3D test; masks are still emitted while the test fails
  for <= --lost_frames consecutive frames; after more failures, or when a mask disappears, both tracker states are reset
  (frames already emitted stay as they were) and seeding restarts from the next verified candidate pair.
Ground truth is never read. The store is read only for the frame count (product check).

Outputs (MASK FILE CONTRACT): <out_root>/<EP>/<cam>_<serial>__verified_sam3_masks.npz ('union' uint8 (T,720,160) =
np.packbits(mask, axis=2), 'shape' [T,720,1280], 'frames_present' bool (T,)); <EP>/seed_info.json (the verified candidate
pair of every accepted and rejected (re)prompt with its 3D verification values, drops, per-cam stats, timing);
<EP>/per_frame.csv (per-frame verification values); with --video the side-by-side overlay <EP>/verified_sam3_side_by_side.mp4
and look/*.png at the prompt frames.

    python seed_verified_sam3.py --episode EP --out_root .../seedstudy2/verified_sam3 [--video]
    OMP_NUM_THREADS=1 python seed_verified_sam3.py --episode EP --cue_only [--max_frames N]   # CPU: motion candidates + verification only
"""
import argparse
import collections
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

cv2.setNumThreads(1)  # login nodes cap CPU time per process across threads
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam3_gripper_masks import FfmpegWriter, RAW_ROOT, read_frames, render  # noqa: E402
from seed_motion_only import argmax_geo, border_pixels, component_containing, fill_small_holes, geodesic  # noqa: E402
from seed_motion_sam3 import check_product, interior_point, mask_sane, save_masks, to_out  # noqa: E402
from seed_text_arm_sam3 import ArmDetector, arm_tip, largest_cc, point_at, select_arm  # noqa: E402

STORE_ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
AUDIT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/gt_audit_2026-09-08"
INTR_CACHE = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/intrinsics_cache"
OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy2/verified_sam3"
METHOD = "verified_sam3"
NW, NH = 1280, 720
DIAG = float(np.hypot(NW, NH))


# ----------------------------------------------------------------------------------------------- declared calibration
def load_intrinsics(ep):
    cache = f"{INTR_CACHE}/{ep}.json"
    if os.path.isfile(cache):
        return json.load(open(cache))
    allI = json.load(open(f"{AUDIT}/docs/hf_intrinsics.json"))
    os.makedirs(INTR_CACHE, exist_ok=True)
    tmp = f"{cache}.tmp.{os.getpid()}"
    json.dump(allI[ep], open(tmp, "w"))
    os.replace(tmp, cache)
    return allI[ep]


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)


def fundamental(K1, E1, K2, E2):
    """F with x2^T F x1 = 0 for the world->camera extrinsics E1, E2 (X2 = R X1 + t)."""
    R1, t1, R2, t2 = E1[:3, :3], E1[:3, 3], E2[:3, :3], E2[:3, 3]
    R = R2 @ R1.T
    t = t2 - R @ t1
    F = np.linalg.inv(K2).T @ skew(t) @ R @ np.linalg.inv(K1)
    return F / np.linalg.norm(F)


def sym_epi(F, x1, x2):
    """Mean of the two point-to-epipolar-line distances (px)."""
    a = np.array([x1[0], x1[1], 1.0])
    b = np.array([x2[0], x2[1], 1.0])
    l2, l1 = F @ a, F.T @ b
    return 0.5 * (abs(l2 @ b) / max(np.hypot(l2[0], l2[1]), 1e-9) + abs(l1 @ a) / max(np.hypot(l1[0], l1[1]), 1e-9))


def triangulate(P1, P2, x1, x2):
    Xh = cv2.triangulatePoints(P1, P2, np.asarray(x1, np.float64).reshape(2, 1), np.asarray(x2, np.float64).reshape(2, 1))
    return (Xh[:3] / Xh[3]).ravel()


def load_geometry(ep, serials, cams):
    intr = load_intrinsics(ep)
    camj = json.load(open(f"{AUDIT}/pointworld/droid/cameras/{ep}_cameras.json"))
    g = {}
    for c in cams:
        s = str(serials[c])
        fx, cx, fy, cy = intr[s]["cameraMatrix"]
        iw, ih = intr[s].get("width", NW), intr[s].get("height", NH)
        fx, cx, fy, cy = fx * NW / iw, cx * NW / iw, fy * NH / ih, cy * NH / ih
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        E = np.array(camj[s]["optimized_extrinsics"], float)
        root = None  # arm root pixel: the lowest point of the robot's base column that projects in front of the camera
        for z in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            p = E[:3, :3] @ np.array([0.0, 0.0, z]) + E[:3, 3]
            if p[2] >= 0.15:
                x, y = fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy
                if -3 * NW <= x <= 4 * NW and -3 * NH <= y <= 4 * NH:
                    root = [float(x), float(y), z]
                    break
        # plausible gripper depth for the metres-to-pixels scale (seed_motion_only.plausible_depth, calibration only)
        grid = np.array([[x, y, z, 1.0] for x in (0.3, 0.5, 0.7) for y in (-0.3, 0, 0.3) for z in (0.1, 0.3, 0.5)])
        zg = (E @ grid.T)[2]
        z_grid = float(np.median(zg))
        cam_c = -E[:3, :3].T @ E[:3, 3]
        depth = 0.5 * (z_grid + float(np.linalg.norm(cam_c)))
        depth_near = max(0.3, float(np.percentile(zg, 25)))  # near end of the workspace: upper bound on apparent sizes
        g[c] = dict(K=K, E=E, P=K @ E[:3], fx=fx, fy=fy, cx=cx, cy=cy, root=root, depth=depth, depth_near=depth_near, cam_center=cam_c.round(3).tolist(),
                    base_depth=float(E[2, 3]))
    c1, c2 = cams
    g["F"] = fundamental(g[c1]["K"], g[c1]["E"], g[c2]["K"], g[c2]["E"])
    return g


def verify(geom, cams, d1, d2, a):
    """The 3D test on two descriptors (centroid, long_side in native px). Returns the values and ok."""
    g1, g2 = geom[cams[0]], geom[cams[1]]
    x1, x2 = d1["centroid"], d2["centroid"]
    epi = float(sym_epi(geom["F"], x1, x2))
    X = triangulate(g1["P"], g2["P"], x1, x2)
    dist = float(np.linalg.norm(X))
    z1 = float((g1["E"][:3, :3] @ X + g1["E"][:3, 3])[2])
    z2 = float((g2["E"][:3, :3] @ X + g2["E"][:3, 3])[2])
    s1 = d1["long_side"] * z1 / g1["fx"]
    s2 = d2["long_side"] * z2 / g2["fx"]
    lo, hi = a.size_min / a.size_tol, a.size_max * a.size_tol
    why = []
    if not epi < a.epi_max:
        why.append(f"epi {epi:.1f}")
    if not dist < a.reach:
        why.append(f"reach {dist:.2f}")
    if not X[2] > a.z_min:
        why.append(f"below table z {X[2]:.2f}")
    if not (z1 > 0.1 and z2 > 0.1):
        why.append("behind camera")
    if not (lo <= s1 <= hi and lo <= s2 <= hi):
        why.append(f"size {s1:.2f}/{s2:.2f}")
    return dict(ok=not why, why="; ".join(why), epi=round(epi, 2), X=[round(float(v), 3) for v in X], dist=round(dist, 3),
                z_base=round(float(X[2]), 3), z1=round(z1, 3), z2=round(z2, 3), size1=round(float(s1), 3), size2=round(float(s2), 3))


# ----------------------------------------------------------------------------------------------- candidates
def blob_desc(blob, sx=1.0, sy=1.0):
    """centroid / bbox / long side of a boolean blob, scaled to native px; border = the blob touches the image border."""
    ys, xs = np.nonzero(blob)
    x0, y0, x1, y1 = xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy
    return dict(centroid=(float(xs.mean() * sx), float(ys.mean() * sy)), bbox=[float(x0), float(y0), float(x1), float(y1)],
                long_side=float(max(x1 - x0, y1 - y0)), area=float(len(xs) * sx * sy),
                border=bool(blob[:2].any() or blob[-2:].any() or blob[:, :2].any() or blob[:, -2:].any()))


class MotionCue:
    """MOG2 foreground restricted to recently moving pixels, at processing resolution -> per-frame candidates per
    component: 'cap' (distal cap from the far end), 'compact' (small whole component), 'dark' (compact dark achromatic
    sub-blobs: the black gripper body inside the white arm)."""

    def __init__(self, gc, a, pw, ph):
        self.a, self.pw, self.ph = a, pw, ph
        self.sx, self.sy = NW / pw, NH / ph
        self.px_per_m = gc["fx"] * (pw / NW) / max(gc["depth"], a.min_depth)  # proc px per metre at the plausible depth
        self.cap_px = a.cap_m * self.px_per_m
        self.compact_max_px = a.compact_max_m * gc["fx"] * (pw / NW) / max(gc["depth_near"], a.min_depth)  # at the NEAR workspace depth
        self.root_px = None if gc["root"] is None else (gc["root"][0] / self.sx, gc["root"][1] / self.sy)
        self.mog = cv2.createBackgroundSubtractorMOG2(history=a.history, varThreshold=a.var_thresh, detectShadows=True)
        self.mog.setShadowThreshold(a.shadow_tau)
        self.min_area = a.min_area_frac * pw * ph
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.close_px, a.close_px))
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.open_px, a.open_px))
        self.k_move = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.move_dilate_px, a.move_dilate_px))
        self.k_recent = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.recent_dilate_px, a.recent_dilate_px))
        self.k_dark_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.dark_open_px, a.dark_open_px))  # cuts the black cable off the gripper body
        self.k_dark_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.recent = collections.deque(maxlen=max(1, a.recent_k))
        self.prev_gray = None
        self.t = 0

    def roots_for(self, comp):
        """(roots (r,c) array, reference point (x,y) in proc px) for the geodesic 'distance along the arm from its entry'."""
        bp = border_pixels(comp)
        if self.root_px is not None:
            rx, ry = self.root_px
            inside = 0 <= rx < self.pw and 0 <= ry < self.ph
            if inside or len(bp) == 0:
                rs, cs = np.nonzero(comp)
                i = int(np.argmin((cs - rx) ** 2 + (rs - ry) ** 2))
                return np.array([[rs[i], cs[i]]]), (float(cs[i]), float(rs[i]))
            d = np.hypot(bp[:, 1] - rx, bp[:, 0] - ry)
            i = int(np.argmin(d))
            keep = bp[np.hypot(bp[:, 1] - bp[i, 1], bp[:, 0] - bp[i, 0]) <= self.a.root_cluster_px]
            return keep, (float(bp[i, 1]), float(bp[i, 0]))
        if len(bp):
            return bp, (float(bp[:, 1].mean()), float(bp[:, 0].mean()))
        rs, cs = np.nonzero(comp)
        i = int(np.argmin(rs))
        return np.array([[rs[i], cs[i]]]), (float(cs[i]), float(rs[i]))

    def make(self, blob, kind, skel_px, moving, extra=None):
        d = blob_desc(blob, self.sx, self.sy)
        d.update(cue="motion", kind=kind, skel=float(skel_px * self.sx), move_frac=float(moving[blob].mean()), blob=blob, blob_res="proc")
        if extra:
            d.update(extra)
        return d

    def component_candidates(self, comp, moving, gray, chroma):
        a = self.a
        out = []
        roots, ref = self.roots_for(comp)
        ys, xs = np.nonzero(comp)
        long_side = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        if long_side <= self.compact_max_px:
            out.append(self.make(comp, "compact", np.hypot(xs.mean() - ref[0], ys.mean() - ref[1]), moving))
        geo = geodesic(comp, roots)
        fin = comp & np.isfinite(geo) if geo is not None else np.zeros_like(comp)

        def skel_of(blob):
            g = geo[blob & fin] if geo is not None else np.zeros(0)
            if len(g):
                return float(np.max(g))
            by, bx = np.nonzero(blob)
            return float(np.hypot(bx.mean() - ref[0], by.mean() - ref[1]))

        # dark achromatic sub-blobs (the black gripper body / wrist camera inside the white arm)
        if a.max_dark > 0:
            dark = (comp & (gray < a.dark_cand_gray) & (chroma < a.dark_cand_chroma)).astype(np.uint8)
            dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, self.k_dark_open)
            dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, self.k_dark_close)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
            dk = sorted([(int(stats[k, cv2.CC_STAT_AREA]), k) for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= a.min_cand_px], reverse=True)
            for _, k in dk[:a.max_dark]:
                blob = lab == k
                by, bx = np.nonzero(blob)
                if max(bx.max() - bx.min() + 1, by.max() - by.min() + 1) > self.compact_max_px:
                    continue
                out.append(self.make(blob, "dark", skel_of(blob), moving))
        if geo is None or not fin.any():
            return out
        gmax = float(geo[fin].max())
        if gmax < 0.5 * self.cap_px:
            return out
        sel = fin & moving
        tip = argmax_geo(sel if sel.sum() >= a.min_tip_moving_px else fin, geo)
        geo_tip = geodesic(fin, np.array([tip]))
        if geo_tip is None:
            return out
        L = self.cap_px
        cap = fill_small_holes(component_containing(fin & (geo_tip <= L), tip), a.max_hole_px)
        if cap.sum() >= a.min_cand_px:
            out.append(self.make(cap, "cap", geo[tip], moving, dict(tip=[float(tip[1] * self.sx), float(tip[0] * self.sy)], geo_tip=round(float(geo[tip]), 1))))
        if a.cap2:
            cap2 = fin & (geo_tip > L) & (geo_tip <= 2 * L)
            if cap2.sum() >= a.min_cand_px:
                cap2 = largest_cc(cap2)
                if cap2.sum() >= a.min_cand_px:
                    out.append(self.make(cap2, "cap2", max(geo[tip] - L, 0.0), moving))
        return out

    def step(self, bgr_native):
        a = self.a
        small = cv2.resize(bgr_native, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self.t < a.warmup_frames:
            self.mog.apply(small, learningRate=a.warmup_lr)
            self.prev_gray = gray
            self.t += 1
            return [], None, None
        fg = self.mog.apply(small, learningRate=1.0 / a.history)  # 255 fg, 127 shadow
        chroma = small.max(axis=2).astype(np.int16) - small.min(axis=2).astype(np.int16)
        fg = ((fg == 255) | ((fg == 127) & (gray < a.dark_thresh) & (chroma < a.chroma_thresh))).astype(np.uint8)
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_open)
        # frame difference with the global (exposure-step) shift removed; recently moving = OR over the last recent_k
        d = cv2.absdiff(gray, self.prev_gray).astype(np.int16)
        d = np.clip(d - int(np.median(d)), 0, 255).astype(np.uint8)
        mv = (d > a.move_thresh).astype(np.uint8)
        moving = cv2.dilate(mv, self.k_move).astype(bool)
        self.recent.append(mv)
        rec = cv2.dilate(np.max(np.stack(self.recent), 0), self.k_recent)
        fg_used = cv2.morphologyEx(fg & rec, cv2.MORPH_CLOSE, self.k_close)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg_used, connectivity=8)
        comps = sorted([(int(stats[k, cv2.CC_STAT_AREA]), k) for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= self.min_area], reverse=True)
        cands = []
        for _, k in comps[:a.max_comps]:
            cands += self.component_candidates(lab == k, moving, gray, chroma)
        self.prev_gray = gray
        self.t += 1
        return cands, fg_used.astype(bool), moving


class TextCue:
    """SAM 3 text prompt 'robotic arm' -> far end of the arm mask -> 'text_tip' and 'text_beyond' candidates (native res)."""

    def __init__(self, a, gc):
        self.a = a
        self.det = ArmDetector(a.phrase, version=a.det_version, thresh=a.thresh)
        self.L = a.cap_m * gc["fx"] / max(gc["depth"], a.min_depth)  # native px
        self.n_calls = self.n_fired = self.n_valid = 0

    def __call__(self, frame, fg_native):
        a = self.a
        self.n_calls += 1
        masks, sc = self.det(frame)
        arm, kept = select_arm(masks, sc, a.min_area_frac_text, a.big_area_frac_text)
        if arm is None:
            return [], None
        self.n_fired += 1
        arm = largest_cc(arm)
        ti = arm_tip(arm)
        if ti is None or ti["path_len"] < a.min_path_w * ti["w"]:
            return [], arm
        self.n_valid += 1
        out = []
        tip = np.array(ti["tip"])
        # cap = arm pixels within L of the tip along the mask (0.5-scale geodesic)
        from skimage.graph import MCP_Geometric
        ms = cv2.resize(arm.astype(np.uint8), (NW // 2, NH // 2), interpolation=cv2.INTER_NEAREST) > 0
        ty, tx = int(np.clip(tip[1] / 2, 0, NH // 2 - 1)), int(np.clip(tip[0] / 2, 0, NW // 2 - 1))
        if not ms[ty, tx]:
            ys, xs = np.nonzero(ms)
            i = int(np.argmin((ys - ty) ** 2 + (xs - tx) ** 2))
            ty, tx = int(ys[i]), int(xs[i])
        D, _ = MCP_Geometric(np.where(ms, 1.0, np.inf), fully_connected=True).find_costs([(ty, tx)])
        cap = cv2.resize((np.isfinite(D) & (D <= self.L / 2)).astype(np.uint8), (NW, NH), interpolation=cv2.INTER_NEAREST).astype(bool) & arm
        if cap.sum() >= a.min_cand_px * 4:
            d = blob_desc(cap)
            d.update(cue="text", kind="text_tip", skel=float(ti["path_len"]), move_frac=None, blob=cap, blob_res="native", tip=[float(tip[0]), float(tip[1])], w=ti["w"])
            out.append(d)
        # beyond the tip: moving foreground in a box ahead of the tip, outside the arm mask
        if fg_native is not None:
            w = ti["w"]
            back = np.array(point_at(ti["path"], ti["cum"], 2.0 * w))
            dv = tip - back
            n = np.linalg.norm(dv)
            dv = dv / n if n > 1e-6 else np.array([0.0, 1.0])
            cen = tip + 2.0 * w * dv
            r = 1.8 * w
            x0, y0 = int(max(0, cen[0] - r)), int(max(0, cen[1] - r))
            x1, y1 = int(min(NW, cen[0] + r)), int(min(NH, cen[1] + r))
            reg = np.zeros((NH, NW), bool)
            reg[y0:y1, x0:x1] = fg_native[y0:y1, x0:x1] & ~arm[y0:y1, x0:x1]
            if reg.sum() >= a.min_cand_px * 4:
                reg = largest_cc(reg)
                if reg.sum() >= a.min_cand_px * 4:
                    d = blob_desc(reg)
                    d.update(cue="text", kind="text_beyond", skel=float(ti["path_len"] + 2 * w), move_frac=None, blob=reg, blob_res="native", tip=[float(tip[0]), float(tip[1])], w=ti["w"])
                    out.append(d)
        return out, arm


def transfer_proposals(strong, weak, gs, gw, cams_sw, a):
    """Cross-view proposals: a strong-view candidate (dark first, then farthest along the skeleton) is carried along its
    ray into the weak view; the depth is the one at which the epipolar segment passes nearest to a weak-view candidate
    centroid (within --transfer_anchor_m metres at that depth). The weak-view proposal is a synthetic box of
    --transfer_size_m at that depth (no blob); SAM decides the object, and the pair is accepted only if the two SAM masks
    pass the 3D test. Returns [(strong cand, weak proposal cand)]."""
    order = sorted(range(len(strong)), key=lambda i: (strong[i]["kind"] == "dark", strong[i]["skel"]), reverse=True)[:a.transfer_k]
    Rs, ts = gs["E"][:3, :3], gs["E"][:3, 3]
    Cs = -Rs.T @ ts
    Kinv = np.linalg.inv(gs["K"])
    out = []
    for i in order:
        cs = strong[i]
        d = Rs.T @ Kinv @ np.array([cs["centroid"][0], cs["centroid"][1], 1.0])
        d /= np.linalg.norm(d)
        best = None
        for z in np.linspace(a.transfer_zmin, a.transfer_zmax, 41):
            X = Cs + z * d
            if np.linalg.norm(X) > a.reach or X[2] < a.z_min:
                continue
            p = gw["P"] @ np.r_[X, 1.0]
            if p[2] < 0.15:
                continue
            xw, yw, zw = p[0] / p[2], p[1] / p[2], p[2]
            if not (0 <= xw < NW and 0 <= yw < NH):
                continue
            for j, cw in enumerate(weak):
                dm = np.hypot(xw - cw["centroid"][0], yw - cw["centroid"][1]) * zw / gw["fx"]
                if dm < a.transfer_anchor_m and (best is None or dm < best[0]):
                    best = (dm, xw, yw, zw, z, j)
        if best is None:
            continue
        dm, xw, yw, zw, z, j = best
        side = a.transfer_size_m * gw["fx"] / zw
        bbox = [max(0.0, xw - side / 2), max(0.0, yw - side / 2), min(NW - 1.0, xw + side / 2), min(NH - 1.0, yw + side / 2)]
        ct = dict(cue="transfer", kind="transfer", centroid=(float(xw), float(yw)), bbox=[float(v) for v in bbox], long_side=float(side), area=float(side * side),
                  border=False, skel=float(weak[j]["skel"]), move_frac=None, blob=None, blob_res=None, anchor_kind=weak[j]["kind"], anchor_m=round(float(dm), 3),
                  depth_strong=round(float(z), 3), from_view=cams_sw[0], from_kind=cs["kind"])
        out.append((cs, ct))
    return out


def native_blob(c):
    if c["blob_res"] == "native":
        return c["blob"]
    return cv2.resize(c["blob"].astype(np.uint8), (NW, NH), interpolation=cv2.INTER_NEAREST).astype(bool)


def build_prompt(c, pad):
    """box + up to 3 positive clicks (native px) from a candidate blob: deepest points of the blob and of its two halves."""
    if c.get("blob") is None:  # synthetic proposal: its box and one click at the centre
        x0, y0, x1, y1 = [int(round(v)) for v in c["bbox"]]
        return dict(box=[x0, y0, x1, y1], clicks=[[int(c["centroid"][0]), int(c["centroid"][1])]])
    blob = native_blob(c)
    ys, xs = np.nonzero(blob)
    box = [int(max(0, xs.min() - pad)), int(max(0, ys.min() - pad)), int(min(NW - 1, xs.max() + pad)), int(min(NH - 1, ys.max() + pad))]
    pts = np.stack([xs, ys], 1).astype(float)
    mu = pts.mean(0)
    cov = np.cov((pts - mu).T) if len(pts) > 2 else np.eye(2)
    u = np.linalg.eigh(cov)[1][:, -1]
    proj = (pts - mu) @ u
    ha, hb = np.zeros_like(blob), np.zeros_like(blob)
    ha[ys[proj >= 0], xs[proj >= 0]] = True
    hb[ys[proj < 0], xs[proj < 0]] = True
    clicks = []
    for m in (blob, ha, hb):
        p = interior_point(m.astype(np.uint8))
        if p is not None and all(np.hypot(p[0] - q[0], p[1] - q[1]) >= 10 for q in clicks):
            clicks.append(p)
    return dict(box=box, clicks=clicks[:3])


def cand_public(c):
    return {k: v for k, v in c.items() if k not in ("blob", "blob_res")}


def clean_mask(m, frac):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [k for k in range(1, n) if areas[k - 1] >= frac * areas.max()]
    return np.isin(lab, keep)


# ----------------------------------------------------------------------------------------------- SAM 3 pair tracker
class PairTracker:
    def __init__(self, tracker, videos, cams, T, offload):
        import torch
        self.torch, self.tracker, self.cams = torch, tracker, cams
        self.states = {}
        for c in cams:
            st = tracker.init_state(video_path=videos[c], offload_video_to_cpu=bool(offload))
            assert st["num_frames"] >= T, (c, st["num_frames"], T)
            assert (st["video_height"], st["video_width"]) == (NH, NW), (st["video_height"], st["video_width"])
            self.states[c] = st
        self.active, self.need_preflight = False, False

    def prompt(self, t, prompts):
        masks = {}
        for c in self.cams:
            pr = prompts[c]
            pts = self.torch.tensor([[x / NW, y / NH] for x, y in pr["clicks"]], dtype=self.torch.float32)
            labels = self.torch.ones(len(pr["clicks"]), dtype=self.torch.int32)
            x0, y0, x1, y1 = pr["box"]
            box = np.array([[x0 / NW, y0 / NH, x1 / NW, y1 / NH]], dtype=np.float32)
            _, _, _, vm = self.tracker.add_new_points_or_box(inference_state=self.states[c], frame_idx=int(t), obj_id=1, points=pts, labels=labels, box=box)
            masks[c] = (vm[0][0] > 0).cpu().numpy()
        return masks

    def undo_prompt(self, t):
        for c in self.cams:
            self.tracker.clear_all_points_in_frame(self.states[c], int(t), 1, need_output=False)

    def propagate(self, t):
        masks, scores = {}, {}
        for c in self.cams:
            got = None
            for fidx, _, _, vm, sc in self.tracker.propagate_in_video(self.states[c], start_frame_idx=int(t), max_frame_num_to_track=0, reverse=False,
                                                                       propagate_preflight=self.need_preflight, tqdm_disable=True):
                assert int(fidx) == t, (fidx, t)
                got = (vm[0][0] > 0).cpu().numpy()
                try:
                    scores[c] = float(sc.reshape(-1)[0].item())
                except Exception:  # noqa: BLE001
                    scores[c] = float("nan")
            masks[c] = got
        self.need_preflight = False
        return masks, scores

    def reset(self):
        for c in self.cams:
            self.tracker._reset_tracking_results(self.states[c])
        self.active, self.need_preflight = False, False

    def close(self):
        for c in self.cams:
            del self.states[c]
        self.torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------- overlay
def draw_cands(img, cands, color=(255, 0, 255)):
    for c in cands:
        x0, y0, x1, y1 = [int(v) for v in c["bbox"]]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
        cx, cy = c["centroid"]
        cv2.circle(img, (int(cx), int(cy)), 4, color, -1)


def draw_prompt(img, pr, color=(0, 255, 255)):
    x0, y0, x1, y1 = pr["box"]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    for x, y in pr["clicks"]:
        cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.circle(img, (int(x), int(y)), 6, (0, 0, 0), 1)


def panel(frame, cam, serial, t, T, mask, cands, prompt, txt):
    img = render(frame, to_out(mask, float("nan")), f"{cam}/{serial} {METHOD} f{t}/{T - 1} {txt}")
    draw_cands(img, cands)
    if prompt is not None:
        draw_prompt(img, prompt)
    return cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--cams", default="ext1,ext2")
    ap.add_argument("--proc_w", type=int, default=640, help="motion-cue processing width (height = 9/16)")
    ap.add_argument("--cue_only", action="store_true", help="CPU only: motion candidates + verification, no SAM, no products")
    ap.add_argument("--max_frames", type=int, default=0, help="cue_only: stop after N frames")
    ap.add_argument("--video", action="store_true", help="side-by-side overlay mp4 + look PNGs")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--offload_video", type=int, default=1, help="SAM3 init_state(offload_video_to_cpu)")
    # motion cue (seed_motion_only.py settings)
    ap.add_argument("--history", type=int, default=300)
    ap.add_argument("--var_thresh", type=float, default=25.0)
    ap.add_argument("--shadow_tau", type=float, default=0.3)
    ap.add_argument("--dark_thresh", type=int, default=70)
    ap.add_argument("--chroma_thresh", type=int, default=15)
    ap.add_argument("--warmup_frames", type=int, default=5)
    ap.add_argument("--warmup_lr", type=float, default=0.3)
    ap.add_argument("--close_px", type=int, default=9)
    ap.add_argument("--open_px", type=int, default=5)
    ap.add_argument("--min_area_frac", type=float, default=0.0015)
    ap.add_argument("--move_thresh", type=int, default=10)
    ap.add_argument("--move_dilate_px", type=int, default=7)
    ap.add_argument("--max_comps", type=int, default=5, help="largest foreground components considered per frame")
    ap.add_argument("--cap_m", type=float, default=0.20, help="distal cap length along the arm (metres at the plausible depth)")
    ap.add_argument("--compact_max_m", type=float, default=0.35, help="a whole component is a 'compact' candidate below this long side")
    ap.add_argument("--min_depth", type=float, default=0.30)
    ap.add_argument("--min_tip_moving_px", type=int, default=20)
    ap.add_argument("--min_cand_px", type=int, default=150, help="min candidate blob area (proc px)")
    ap.add_argument("--max_hole_px", type=int, default=400)
    ap.add_argument("--root_cluster_px", type=int, default=40)
    ap.add_argument("--cand_move_frac", type=float, default=0.05, help="motion candidates must have moved this fraction since t-1 (ghost filter)")
    ap.add_argument("--recent_k", type=int, default=8, help="foreground is restricted to pixels that moved within the last N frames (ghost / flicker filter)")
    ap.add_argument("--recent_dilate_px", type=int, default=15)
    ap.add_argument("--cap2", type=int, default=0, help="also emit the segment behind the cap (arm chunk by construction; off)")
    ap.add_argument("--max_dark", type=int, default=2, help="dark achromatic sub-blobs per component (0 = off)")
    ap.add_argument("--dark_cand_gray", type=int, default=80)
    ap.add_argument("--dark_cand_chroma", type=int, default=40)
    ap.add_argument("--dark_open_px", type=int, default=9)
    # cross-view transfer proposals
    ap.add_argument("--transfer_k", type=int, default=2, help="strong-view candidates carried into the other view per direction (0 = off)")
    ap.add_argument("--transfer_every", type=int, default=2, help="seeking frames between transfer attempts")
    ap.add_argument("--transfer_anchor_m", type=float, default=0.15, help="the epipolar segment must pass within this distance of a weak-view candidate")
    ap.add_argument("--transfer_size_m", type=float, default=0.20, help="side of the synthetic prompt box (metres at the transferred depth)")
    ap.add_argument("--transfer_zmin", type=float, default=0.25)
    ap.add_argument("--transfer_zmax", type=float, default=1.05)
    # text cue
    ap.add_argument("--no_text", action="store_true")
    ap.add_argument("--phrase", default="robotic arm")
    ap.add_argument("--det_version", default="sam3.1", choices=["sam3.1", "sam3"])
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--text_every", type=int, default=2, help="run the text detector every N frames while seeking")
    ap.add_argument("--min_area_frac_text", type=float, default=0.006)
    ap.add_argument("--big_area_frac_text", type=float, default=0.02)
    ap.add_argument("--min_path_w", type=float, default=3.0)
    # verification
    ap.add_argument("--epi_max", type=float, default=6.0, help="symmetric epipolar residual (px, native)")
    ap.add_argument("--reach", type=float, default=0.9, help="max distance from the base origin (m)")
    ap.add_argument("--z_min", type=float, default=-0.15, help="table level in the base frame (m)")
    ap.add_argument("--size_min", type=float, default=0.10)
    ap.add_argument("--size_max", type=float, default=0.25)
    ap.add_argument("--size_tol", type=float, default=1.25, help="size band = [size_min/tol, size_max*tol]")
    ap.add_argument("--jump_m", type=float, default=0.10, help="temporal consistency: max 3D motion per frame (m)")
    ap.add_argument("--memory_frames", type=int, default=30, help="previous accepted position is remembered this long")
    ap.add_argument("--persist", type=int, default=2, help="new acquisition needs a verified pair on one of the previous N frames (0 = off)")
    # SAM prompt / tracking
    ap.add_argument("--box_pad", type=int, default=8)
    ap.add_argument("--min_mask", type=int, default=800)
    ap.add_argument("--max_mask_frac", type=float, default=0.12)
    ap.add_argument("--min_box_iou", type=float, default=0.25)
    ap.add_argument("--frag_keep", type=float, default=0.15)
    ap.add_argument("--retry_gap", type=int, default=2)
    ap.add_argument("--max_pair_tries", type=int, default=4)
    ap.add_argument("--lost_frames", type=int, default=5, help="drop the track after MORE than this many consecutive verification failures")
    a = ap.parse_args()

    ep = a.episode
    raw = os.path.join(RAW_ROOT, ep)
    meta = json.load(open(os.path.join(raw, f"metadata_{ep}.json")))
    cams = [c for c in a.cams.split(",") if c]
    assert len(cams) == 2, "verified_sam3 needs exactly two exterior views"
    serials = {c: str(meta[f"{c}_cam_serial"]) for c in cams}
    videos = {c: os.path.join(raw, "recordings", "MP4", f"{serials[c]}.mp4") for c in cams}
    for c, p in videos.items():
        if not os.path.exists(p):
            sys.exit(f"missing video for {c}: {p}")
    T_store = len([f for f in os.listdir(os.path.join(STORE_ROOT, ep, "dense", "cam")) if f.endswith(".npz")])
    out_dir = os.path.join(a.out_root, ep)
    os.makedirs(os.path.join(out_dir, "look"), exist_ok=True)
    t_all = time.time()

    def log(s):
        print(s, flush=True)

    log(f"episode {ep}  store frames {T_store}\nvideos {videos}\nout {out_dir}")
    geom = load_geometry(ep, serials, cams)
    for c in cams:
        g = geom[c]
        log(f"  {c} ({serials[c]}): fx {g['fx']:.0f}, camera centre (base frame) {g['cam_center']}, root px {None if g['root'] is None else [round(v, 0) for v in g['root']]}, "
            f"plausible depth {g['depth']:.2f} m")
    T = T_store if not (a.cue_only and a.max_frames > 0) else min(T_store, a.max_frames)
    frames = {}
    t0 = time.time()
    for c in cams:
        fr = read_frames(videos[c])
        if len(fr) < T_store:
            sys.exit(f"{c}: mp4 has {len(fr)} frames < store {T_store}")
        if len(fr) > T:
            fr = fr[:T]
        frames[c] = fr
        assert fr[0].shape[:2] == (NH, NW), fr[0].shape
    t_load = time.time() - t0
    log(f"frames read in {t_load:.0f}s ({T} per cam)")
    pw, ph = a.proc_w, a.proc_w * 9 // 16
    motion = {c: MotionCue(geom[c], a, pw, ph) for c in cams}
    log(f"motion cue: proc {pw}x{ph}, cap px " + ", ".join(f"{c} {motion[c].cap_px:.0f}" for c in cams) + ", compact max px " + ", ".join(f"{c} {motion[c].compact_max_px:.0f}" for c in cams)
        + ", depth near " + ", ".join(f"{c} {geom[c]['depth_near']:.2f}" for c in cams))

    text = pt = None
    if not a.cue_only:
        import torch
        from sam3.model_builder import build_sam3_video_model
        t0 = time.time()
        model = build_sam3_video_model()
        tracker = model.tracker
        tracker.backbone = model.detector.backbone
        if not a.no_text:
            text = TextCue(a, geom[cams[0]])
        log(f"models built in {time.time() - t0:.0f}s (tracker sam3{'' if a.no_text else ', text detector ' + a.det_version}); gpu {torch.cuda.get_device_name(0)}")
        t0 = time.time()
        pt = PairTracker(tracker, videos, cams, T, a.offload_video)
        t_load_sam = time.time() - t0
        log(f"tracker states loaded in {t_load_sam:.0f}s")
    else:
        t_load_sam = 0.0

    c1, c2 = cams
    masks_out = {c: {} for c in cams}
    rows = []
    prompts_ok, prompts_rej, drops = [], [], []
    hist = collections.deque(maxlen=max(a.persist, 1))  # (t, X) of the best verified pair on seeking frames
    prev = None  # (t, X) of the last accepted / verified position
    n_fail, last_try = 0, -10 ** 9
    tm = dict(motion=0.0, text=0.0, sam=0.0, verify=0.0)
    n_verified_frames = n_pairs_total = 0
    min_epi_log = []  # per seeking frame: min epi among pairs passing every other test (threshold diagnostics)
    cands_hist = {c: [None] * T for c in cams}
    pairs_hist = [None] * T
    prompt_by_frame = {}

    for t in range(T):
        t0 = time.time()
        cands, fg = {}, {}
        for c in cams:
            cands[c], fg[c], _ = motion[c].step(frames[c][t])
        tm["motion"] += time.time() - t0
        m_out = {c: None for c in cams}
        ver = None
        event = ""
        # ---- tracking + re-verification
        if pt is not None and pt.active:
            t0 = time.time()
            m, sc = pt.propagate(t)
            tm["sam"] += time.time() - t0
            mc = {c: (clean_mask(m[c], a.frag_keep) if m[c] is not None and m[c].sum() >= a.min_mask else None) for c in cams}
            if all(mc[c] is not None for c in cams):
                ver = verify(geom, cams, blob_desc(mc[c1]), blob_desc(mc[c2]), a)
                if ver["ok"]:
                    n_fail = 0
                    prev = (t, np.array(ver["X"]))
                    n_verified_frames += 1
                else:
                    n_fail += 1
                if n_fail <= a.lost_frames:
                    m_out = mc
                else:
                    drops.append(dict(frame=t, reason=f"verification failed {n_fail} frames: {ver['why']}", last=ver))
                    event = "drop"
                    log(f"  DROP at f{t}: {drops[-1]['reason']}")
                    pt.reset()
            else:
                gone = [c for c in cams if mc[c] is None]
                drops.append(dict(frame=t, reason=f"mask disappeared in {gone}"))
                event = "drop"
                log(f"  DROP at f{t}: {drops[-1]['reason']}")
                pt.reset()
        # ---- seeking
        if pt is None or not pt.active:
            if text is not None and t % a.text_every == 0:
                t0 = time.time()
                for c in cams:
                    fgn = None if fg[c] is None else cv2.resize(fg[c].astype(np.uint8), (NW, NH), interpolation=cv2.INTER_NEAREST).astype(bool)
                    tc, _ = text(frames[c][t], fgn)
                    cands[c] += tc
                tm["text"] += time.time() - t0
            t0 = time.time()
            use = {c: [x for x in cands[c] if x["move_frac"] is None or x["move_frac"] >= a.cand_move_frac] for c in cams}
            pairs = []
            best_epi = None
            for i, x in enumerate(use[c1]):
                for j, y in enumerate(use[c2]):
                    v = verify(geom, cams, x, y, a)
                    n_pairs_total += 1
                    if v["ok"]:
                        pairs.append((i, j, v))
                    elif v["why"].startswith("epi") and ";" not in v["why"]:
                        best_epi = v["epi"] if best_epi is None else min(best_epi, v["epi"])
            n_real = len(pairs)
            if a.transfer_k > 0 and t % a.transfer_every == 0:
                for s_, w_ in ((c1, c2), (c2, c1)):
                    for cs, ct in transfer_proposals(use[s_], use[w_], geom[s_], geom[w_], (s_, w_), a):
                        use[w_].append(ct)
                        x, y = (cs, ct) if s_ == c1 else (ct, cs)
                        i, j = (use[c1].index(x), len(use[c2]) - 1) if s_ == c1 else (len(use[c1]) - 1, use[c2].index(y))
                        v = verify(geom, cams, x, y, a)
                        n_pairs_total += 1
                        if v["ok"]:
                            v["transfer"] = True
                            pairs.append((i, j, v))
            if pairs:
                best_epi = min(p[2]["epi"] for p in pairs)
            if best_epi is not None:
                min_epi_log.append((t, best_epi))

            def rank(p):
                i, j, v = p
                X = np.array(v["X"])
                cons = prev is not None and (t - prev[0]) <= a.memory_frames and np.linalg.norm(X - prev[1]) <= a.jump_m * max(1, t - prev[0])
                score = 0.5 * (use[c1][i]["skel"] + use[c2][j]["skel"]) / DIAG
                return (0 if v.get("transfer") else 1, 1 if cons else 0, score)

            pairs.sort(key=rank, reverse=True)
            pairs_hist[t] = [dict(i=i, j=j, rank=rank((i, j, v)), kinds=[use[c1][i]["kind"], use[c2][j]["kind"]], **v) for i, j, v in pairs]
            tm["verify"] += time.time() - t0
            if pairs:
                n_verified_frames += int(n_real > 0)
                X = np.array(pairs[0][2]["X"])
                persist_ok = a.persist == 0 or pairs[0][2].get("transfer") or any(1 <= t - th <= a.persist and np.linalg.norm(X - Xh) <= a.jump_m * (t - th) for th, Xh in hist)
                if n_real > 0:
                    hist.append((t, X))
                if pt is not None and persist_ok and t - last_try >= a.retry_gap:
                    last_try = t
                    for k, (i, j, v) in enumerate(pairs[:a.max_pair_tries]):
                        pr = {c1: build_prompt(use[c1][i], a.box_pad), c2: build_prompt(use[c2][j], a.box_pad)}
                        t0 = time.time()
                        m = pt.prompt(t, pr)
                        tm["sam"] += time.time() - t0
                        mc = {c: clean_mask(m[c], a.frag_keep) if m[c].any() else m[c] for c in cams}
                        sane = {c: mask_sane(mc[c], pr[c]["box"], a) for c in cams}
                        v2 = verify(geom, cams, blob_desc(mc[c1]), blob_desc(mc[c2]), a) if all(mc[c].any() for c in cams) else dict(ok=False, why="empty mask")
                        rec = dict(frame=t, kind="reprompt" if prompts_ok else "initial", attempt=k, cands={c1: cand_public(use[c1][i]), c2: cand_public(use[c2][j])},
                                   verification=v, prompts=pr, sam_sanity={c: sane[c][1] for c in cams}, sam_verification=v2,
                                   sam_area={c: int(mc[c].sum()) for c in cams}, n_pairs=len(pairs), rank=rank((i, j, v)))
                        if all(sane[c][0] for c in cams) and v2["ok"]:
                            prompts_ok.append(rec)
                            prompt_by_frame[t] = pr
                            pt.active, pt.need_preflight = True, True
                            n_fail = 0
                            prev = (t, np.array(v2["X"]))
                            m_out = mc
                            event = "prompt"
                            log(f"  PROMPT ({rec['kind']}) at f{t} try {k}: {c1} {use[c1][i]['cue']}/{use[c1][i]['kind']} {c2} {use[c2][j]['cue']}/{use[c2][j]['kind']} "
                                f"cand epi {v['epi']} dist {v['dist']} z {v['z_base']} size {v['size1']}/{v['size2']} | sam epi {v2['epi']} dist {v2['dist']} size {v2['size1']}/{v2['size2']} area {rec['sam_area']}")
                            break
                        rec["why"] = "; ".join([f"{c}: {sane[c][1]}" for c in cams if not sane[c][0]] + ([v2["why"]] if not v2["ok"] else []))
                        prompts_rej.append(rec)
                        pt.undo_prompt(t)
                        log(f"  prompt at f{t} try {k} REJECTED: {rec['why']}")
                    ver = ver or (pairs[0][2] if event != "prompt" else v2)
                elif ver is None:
                    ver = pairs[0][2]
        for c in cams:
            if m_out[c] is not None:
                masks_out[c][t] = m_out[c]
            cands_hist[c][t] = [dict(bbox=x["bbox"], centroid=x["centroid"], kind=x["kind"], cue=x["cue"], skel=round(x["skel"], 1), move_frac=x["move_frac"],
                                     border=x.get("border"), long_side=round(x["long_side"], 1)) for x in (use.get(c, cands[c]) if (pt is None or not pt.active) else cands[c])]
        rows.append(dict(frame=t, active=int(pt is not None and pt.active), n_cand1=len(cands[c1]), n_cand2=len(cands[c2]),
                         present1=int(m_out[c1] is not None), present2=int(m_out[c2] is not None), n_fail=n_fail, event=event,
                         **({k: ver[k] for k in ("ok", "epi", "dist", "z_base", "size1", "size2", "why")} if ver else {}),
                         **(dict(X0=ver["X"][0], X1=ver["X"][1], X2=ver["X"][2]) if ver else {})))
        if t % 100 == 0 or t == T - 1:
            log(f"f{t}/{T - 1}: cands {len(cands[c1])}/{len(cands[c2])} active={int(pt is not None and pt.active)} prompts {len(prompts_ok)} drops {len(drops)} "
                f"present {sum(1 for r in rows if r['present1'])}/{sum(1 for r in rows if r['present2'])} | {time.time() - t_all:.0f}s")

    t_loop = time.time() - t_all - t_load - t_load_sam
    if text is not None:
        tm["text_detector"] = text.det.t_total
    epi_vals = np.array([e for _, e in min_epi_log])
    info = dict(episode=ep, method=METHOD, serials=serials, T=T, store_frames=T_store, args=vars(a), causal=True, uses_gt=False,
                calibration_use="PointWorld optimized_extrinsics + factory intrinsics: epipolar/triangulation verification, reach, table height, apparent-size test, arm root pixel, metres-to-pixels scale",
                geometry={c: dict(fx=geom[c]["fx"], cam_center=geom[c]["cam_center"], root_px=geom[c]["root"], plausible_depth=round(geom[c]["depth"], 3), base_depth=round(geom[c]["base_depth"], 3)) for c in cams},
                prompts=prompts_ok, rejected=prompts_rej, drops=drops, n_prompts=len(prompts_ok), n_reprompts=max(0, len(prompts_ok) - 1),
                n_frames_with_verified_pair=n_verified_frames, n_pairs_tested=n_pairs_total,
                seeking_min_epi_px=dict(n=int(len(epi_vals)), median=float(np.median(epi_vals)) if len(epi_vals) else None,
                                        p25=float(np.percentile(epi_vals, 25)) if len(epi_vals) else None, frac_below_6=float((epi_vals < 6).mean()) if len(epi_vals) else None),
                text_cue=None if text is None else dict(calls=text.n_calls, fired=text.n_fired, valid=text.n_valid, seconds=round(text.det.t_total, 1)),
                timing=dict(total_seconds=round(time.time() - t_all, 1), read_frames=round(t_load, 1), sam_state_load=round(t_load_sam, 1), loop=round(t_loop, 1),
                            ms_per_frame_loop=round(1000 * t_loop / max(T, 1), 1), **{k: round(v, 1) for k, v in tm.items()}),
                cams={})
    with open(os.path.join(out_dir, "per_frame.csv" if not a.cue_only else "per_frame_cue_only.csv"), "w", newline="") as f:
        keys = ["frame", "active", "n_cand1", "n_cand2", "present1", "present2", "n_fail", "event", "ok", "epi", "dist", "z_base", "size1", "size2", "X0", "X1", "X2", "why"]
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

    if a.cue_only:
        info["cue_only"] = True
        info["cands"] = cands_hist
        info["pairs"] = pairs_hist
        json.dump(info, open(os.path.join(out_dir, "cue_info.json"), "w"), indent=1, default=str)
        log(f"CUE_ONLY: frames with a verified pair {n_verified_frames}/{T}, pairs tested {n_pairs_total}, min-epi over seeking frames: {info['seeking_min_epi_px']}, "
            f"timing {info['timing']}")
        if a.video:
            picks = [t for t in range(T) if rows[t].get("ok")][::max(1, n_verified_frames // 6)][:8]
            for t in picks:
                pan = [panel(frames[c][t], c, serials[c], t, T, None, cands_hist[c][t], None, f"cue epi={rows[t]['epi']} d={rows[t]['dist']} z={rows[t]['z_base']}") for c in cams]
                cv2.imwrite(os.path.join(out_dir, "look", f"cue_f{t:04d}.png"), np.concatenate(pan, axis=1))
        return

    # ---- products
    for c in cams:
        t0 = time.time()
        p = os.path.join(out_dir, f"{c}_{serials[c]}__{METHOD}_masks.npz")
        union, present = save_masks(p, masks_out[c], T, NH, NW)
        chk = check_product(p, T_store)
        info["cams"][c] = dict(serial=serials[c], masks_npz=p, **chk, t_write=round(time.time() - t0, 1),
                               mean_area_frac=chk["mean_area_frac"], frac_present=chk["n_present"] / T)
        log(f"  wrote {p}: {chk}")
    pt.close()
    json.dump(info, open(os.path.join(out_dir, "seed_info.json"), "w"), indent=1, default=str)
    if a.video:
        t0 = time.time()
        path = os.path.join(out_dir, f"{METHOD}_side_by_side.mp4")
        vw = FfmpegWriter(path, 1280, 360, a.fps)
        for t in range(T):
            r = rows[t]
            txt = f"{'TRACK' if r['active'] else 'seek'} nfail={r['n_fail']} " + (f"epi={r['epi']} d={r['dist']} z={r['z_base']} s={r['size1']}/{r['size2']} {'OK' if r['ok'] else r['why']}" if "epi" in r else "") + (f" [{r['event']}]" if r["event"] else "")
            pan = [panel(frames[c][t], c, serials[c], t, T, masks_out[c].get(t), cands_hist[c][t], (prompt_by_frame.get(t) or {}).get(c), txt) for c in cams]
            row = np.concatenate(pan, axis=1)
            vw.write(row)
            if t in prompt_by_frame or (drops and any(d["frame"] == t for d in drops)):
                cv2.imwrite(os.path.join(out_dir, "look", f"f{t:04d}_{r['event'] or 'x'}.png"), row)
        vw.close()
        info["overlay_mp4"] = path
        info["timing"]["overlay"] = round(time.time() - t0, 1)
        json.dump(info, open(os.path.join(out_dir, "seed_info.json"), "w"), indent=1, default=str)
        log(f"overlay {path} in {time.time() - t0:.0f}s")
    log("\nSUMMARY")
    for c in cams:
        i = info["cams"][c]
        log(f"  {c}: present {i['n_present']}/{T} (first {i['first']}, last {i['last']}) mean area {i['mean_area_frac']:.4f}")
    log(f"  prompts {len(prompts_ok)} (rejected {len(prompts_rej)}), drops {len(drops)}, frames with verified pair {n_verified_frames}/{T}, "
        f"seeking min-epi median {info['seeking_min_epi_px']['median']} px; timing {info['timing']}")
    log(f"DONE {out_dir}")


if __name__ == "__main__":
    main()
