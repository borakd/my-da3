#!/usr/bin/env python
"""seed_arm_skeleton_base (IDEA 1): base-projection-anchored arm skeleton -> end-effector leaf -> cross-view verified
3D pick -> gripper mask, on the two STATIC exterior cameras of one DROID episode. Causal, training-free, GT-free.

METHOD (per frame t, both cameras in lockstep; all geometry in native 1280x720 pixels, skeletons at 640x360)
  1. Base projection. The world origin IS the robot base, so it projects to K [R|t] (0,0,0,1) in each view (declared
     PointWorld optimized_extrinsics + factory intrinsics). The point is allowed OUTSIDE the image (it is, on most rigs:
     the arm enters the frame from the base side). When the base is beside/behind the camera (depth < 2 cm, RAIL ext2)
     the anchor is placed at infinity along the base's image-plane direction. This anchor is the ROOT reference.
  2. Arm mask, primary: the SAM 3 image model with the text prompt "robotic arm", per frame, single image (strictly
     causal), after seed_text_arm_sam3.py's instance selection (area / border rule, largest component, white-arm
     brightness rule). --stage detect (GPU) runs that detector on every frame and keeps ALL arm components (the largest
     under the white-arm brightness rule plus border-touching others: the arm chain leaves and RE-ENTERS the image on
     the 44bb9c36 ext1 view); --arm_source detect feeds those to stage cpu. --arm_source cached instead reads the
     per-frame masks the same detector wrote in round 1 (<text_arm_root>/EP/<cam>_<serial>__text_arm_full_masks.npz),
     saved after largest_cc, so re-entering pieces are lost (the CPU-only fallback when no GPU is available).
     Arm mask, fallback (frames where the text mask is empty or its leaf fails verification): the MOG2 moving
     foreground component (seed_verified_motion.ViewMotion, same background-model constants) that CONTAINS the base
     neighbourhood (within base_touch_m = 0.30 m-equivalent at the base depth of the base projection) when the base is
     in view, or ENTERS the image on the base side (touches the border within base_touch_m of the border point nearest
     the base projection) otherwise; no component qualifies -> no fallback. (Runs A/B accepted the component NEAREST
     the anchor, which let a moving forearm + held object far from the base through; see ITERATION.)
  3. Skeleton. Mask at 640x360, 5x5 close, largest component, skimage skeletonize (cv2.ximgproc is not available in
     either env). ROOT = skeleton pixel nearest the base anchor. Geodesic path length along the skeleton from the root
     (skimage MCP_Geometric, cost 1 on skeleton pixels). ROOT RULE (iteration 1, see ITERATION below): the base
     projection is the root only when it lies inside the image and the mask reaches it (nearest skeleton pixel within
     base_touch_m = 0.30 m-equivalent at the base depth); otherwise the arm is anchored where the mask ENTERS the image:
     the widest THICK border contact (largest connected border run whose distance transform reaches thick_frac x
     half-width; a cable's border contact is thin); with no thick border contact, the skeleton pixel nearest the base
     anchor. Leaves = skeleton endpoints (one 8-neighbour) reachable from
     the root; when the base is off-image the root is the THICK border contact nearest the base projection (the arm
     enters from the base side). CABLES (the text mask always includes the thick black cable): removed BEFORE skeletonising by a
     morphological opening with a disc of open_m = 0.025 m-equivalent diameter at the plausible depth (cables are
     <= 1.5 cm across, every arm link and the gripper body >= 8 cm; the 2 cm fingers may go too, then the leaf sits at
     the palm). CHAIN WALK: the END EFFECTOR = the leaf farthest from the root by path length; when that leaf lies in
     the border zone the arm LEAVES the image and the chain continues in the largest other component touching the
     border (its re-entry = the thick border contact nearest the exit), at most max_hops times, until a leaf is
     interior; "at the border" = edge distance <= 1.5 x the leaf's half-width (a truncated tube's medial axis ends
     about one half-width before the edge). A walk that ends on a border leaf yields NO candidate (the end effector is
     not visible). The next two leaves of the final component are backups for step 4 (rank recorded).
  4. Verification of a leaf pair (view 1 leaf, view 2 leaf), tried in order text-text, motion-motion, text-motion,
     motion-text, within each by rank sum (0,0),(0,1),(1,0),...; the first pair that passes is the pick:
       - 2D reach: each leaf inside the projected reach sphere around the base: radius f x tan(asin(reach_m / d)) around
         the base projection, d = camera-to-base distance; VACUOUS when d <= reach_m (the camera is inside the reach
         sphere, every pixel is in reach: true on all 13 smoke rigs, d = 0.49-0.89 m), recorded as reach2d_mode;
       - the two leaves triangulate (DLT): per-view reprojection residual < ray_tol_m x f / depth (ray_tol_m = 0.04 m:
         a medial-axis endpoint seen from two views ~90 deg apart is the same physical end only to within about half
         the gripper width; the round-2 6 px gate passed 7-10% of TRUE pairs on the 44bb9c36 rig). The literal 6 px
         outcome is RECORDED per frame (ok_6px) but not used;
       - |X| < reach_m (0.9) from the base, X_z > z_min (-0.10 m: the fingertips cannot be below the table top, 10 cm
         of calibration slack), in front of both cameras (depth > 0.15 m).
     The verified 3D leaf (xyz_leaf) is the end effector. verified_points.npz 'xyz' (the tracker's wrong-body gate
     input) = the triangulation of the skeleton-path points lens_back_m = 0.10 m-equivalent behind each leaf, a DECLARED
     hand-eye stand-in (the DROID wrist camera sits about 0.10 m behind the Robotiq 2F-85 fingertips along the tool
     axis; allowed by the study rules as a declared per-rig constant), with the leaf itself as the fallback when that
     second triangulation fails its gates (back_ok). No temporal state: every frame is verified on its own.
  5. Gripper mask.
     CPU variant (suffix arm_skeleton_base_cpu, this script --stage cpu): the arm-mask pixels within a 0.25 m-equivalent
     radius (0.25 x f / leaf depth px) of the leaf, per verified frame; unverified frames are EMPTY (no hold).
     Primary (suffix arm_skeleton_base, --stage sam3, GPU): SAM 3 video tracker (build_sam3_video_model, SAM 2-style
     box + clicks), prompted at the verified leaf with seed_text_arm_sam3.make_prompts's two hypotheses (A: gripper
     inside the arm mask behind the tip; B: gripper beyond the tip; the last-link direction comes from the skeleton
     path), selected by round 1's causal rules (area window, adjacency, static-mask rejection when the arm moves, A
     unless B is clearly darker: the Robotiq gripper is black, the Franka links white), propagated FORWARD only.
     Re-prompt (causal) at a verified frame when the tracked mask is lost (empty / negative object score), when its
     centroid drifts more than drift_m (0.25 m-equivalent) from the frame's verified leaf, or every reprompt_every
     frames, with min_gap frames between prompts (round-1 values 10 / 5). Non-cond memory around a re-prompt is
     cleared (tracker.clear_non_cond_mem_around_input), as in round 1.

CAUSALITY: frame t uses frames <= t only (detector: frame t; MOG2: online model; skeleton/triangulation: frame t;
SAM3: memory of past frames, forward propagation; the tracker's init_state decodes the MP4 up front for I/O only).
GT-FREE: the store is opened only to count frames; kinematic poses are never read. No DROID-trained component
(SAM 3 is a generic pretrained model). DECLARED CONSTANTS (also in seed_info.json): PointWorld extrinsics + factory
intrinsics (calibration), reach_m 0.9, z_min -0.10, ray_tol_m 0.04, min_cam_depth 0.15, radius_m 0.25, drift_m 0.25,
thick_frac 0.6 (border-contact thickness), open_m 0.025, min_comp_frac 0.002, border_zone_px 12, max_hops 2, w_stat 65, top_k 3, seed_verified_motion's MOG2 constants, seed_text_arm_sam3's
prompt geometry and hypothesis rules. Nothing was tuned on the 13 scenes.
ITERATION (the one documented iteration, first run 2026-09-29 22:03 on RAIL: 0/128 verified, empty masks): (a) the 2D
reach test was formulated as a circle of radius reach_m x f / plausible_depth around the base projection, which is
wrong when the camera is inside the reach sphere (all rigs): it rejected every pair; (b) with the base off-image the
"skeleton pixel nearest the base projection" landed at the DISTAL end whenever the arm entered the image from the
opposite border (RAIL ext1: base projects bottom-left, the arm rises above the camera and enters from the top), so
the farthest leaf was the arm's entry; (c) after (a)+(b), check PNGs on one scene per rig showed two more
formulation failures: the thin-run cable rule (relative to the arm's 65th-percentile half-width) pruned the WHOLE
distal arm on the 0d4edc83 rig, where the camera sits near the shoulder and the proximal links are ~3x thicker in
pixels than the wrist (leaf at the elbow), and on the 44bb9c36 rig the arm leaves the image at the top and the gripper
re-enters as a SEPARATE component, so largest-component + farthest-leaf returned the exit point. Replaced by the
metric opening (cables) and the chain walk (re-entry); (d) the first FULL cached run over the 13 scenes ("run A",
kept under ideas/arm_skeleton_base_cached_runA, 22% wrong-body anchors) and its check PNGs on 11-23-18h-32 showed
the root at the WRONG border contact on 44bb9c36 ext2 (the widest thick contact is the shoulder link at the left edge,
the arm enters from the bottom-left where the base projects) and exits missed by the 12 px border zone (the medial
axis of a truncated tube ends ~ one half-width inside the edge, 48 px here), so the walk never hopped to the
re-entering gripper piece: root = thick contact nearest the base anchor, at_border relative to the half-width,
no candidate when the walk ends on a border leaf ("run B"); (e) on the same scene the remaining wrong picks were
the MOTION fallback landing on the cloth held by the gripper (the component never came near the base): the loosened
"nearest the anchor" acceptance was replaced by the task's literal rule (contains the base neighbourhood / enters on
the base side) ("run C"); (f) f191 of the same scene: the opened ext2 mask was in two pieces and the walk stopped at the
interior CUT end of the shoulder piece; the chain now also hops across small gaps (an unused component within two
half-widths of an interior leaf) ("run D", the reported cached-CPU variant). All are fixes of the construction, found
by LOOKING at the overlays; no threshold was changed to move a number. Runs A, B and C are reported next to D.
COST: see seed_info.json timing (stage cpu per frame; stage sam3 per frame); the round-1 detector costs ~55 ms per
frame per camera on the H100 (measured in round 1, seed_info.json detector_seconds), the CPU part is measured here.

OUTPUTS (<out_root>/EP/; MASK FILE CONTRACT: 'union' uint8 (T,720,160) = packbits(mask, axis=2), 'shape', 'frames_present'):
  <cam>_<serial>__arm_skeleton_base_textarm_masks.npz text-detector arm masks, all components (stage detect)
  <cam>_<serial>__arm_skeleton_base_cpu_masks.npz     CPU variant masks (stage cpu)
  <cam>_<serial>__arm_skeleton_base_arm_masks.npz     the arm mask used at each verified frame (text or motion; diagnostic + stage-sam3 input)
  <cam>_<serial>__arm_skeleton_base_masks.npz         SAM3-refined masks (stage sam3; the PRIMARY)
  verified_points.npz                                 frames, xyz (T,3) base frame m, accepted (T,) bool, + source/rank/residual
  skeleton_leaves.json                                per-frame record: leaves, verification values, prompts (stage-sam3 input)
  seed_info.json                                      parameters, declared constants, per-camera stats, timing (both stages)
  check_f*.png (stage cpu, a few frames), arm_skeleton_base_side_by_side.mp4 (stage sam3 --video)

    OMP_NUM_THREADS=1 python seed_arm_skeleton_base.py --episode EP --stage cpu [--arm_source cached|detect]   # login node, < 300 s CPU
    python seed_arm_skeleton_base.py --episode EP --stage detect | sam3                                     # GPU (seed_arm_skeleton_base.sbatch: detect -> cpu -> sam3)
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

cv2.setNumThreads(1)
from skimage.graph import MCP_Geometric  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seed_verified_motion import AUDIT, NATIVE_H, NATIVE_W, RAW_ROOT, STORE_ROOT, ViewMotion, load_calib, project, triangulate  # noqa: E402
from seed_text_arm_sam3 import load_masks, make_prompts, point_at, save_masks  # noqa: E402

METHOD = "arm_skeleton_base"
METHOD_CPU = "arm_skeleton_base_cpu"
METHOD_ARM = "arm_skeleton_base_arm"
METHOD_TEXT = "arm_skeleton_base_textarm"
IDEAS_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas"
TEXT_ARM_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/text_arm_sam3"
CAMS = ("ext1", "ext2")
# seed_verified_motion.py's background-model defaults (inherited, declared)
MOG_DEFAULTS = dict(history=300, var_thresh=25.0, no_shadows=False, shadow_tau=0.3, dark_thresh=70, chroma_thresh=15, warmup_frames=5,
                    warmup_lr=0.3, close_px=9, open_px=5, move_thresh=10, move_dilate_px=7)
# seed_text_arm_sam3.py's prompt geometry / hypothesis rules (inherited, declared)
PROMPT_DEFAULTS = dict(box_back=4.5, box_fwd=1.0, box_half=2.0, neg_back=7.0, area_min_w2=0.3, area_max_w2=14.0, adj_max_w=5.0,
                       motion_thr=15, motion_min=0.10, motion_ratio=0.4, a_not_dark=90.0, b_darker_by=30.0)


# ----------------------------------------------------------------------------------------------- geometry
def base_anchor(cal):
    """Pixel anchor of the robot base in this view (may lie outside the image). 'projected' when the base has depth
    > 2 cm; otherwise 'direction': a point at infinity along the base's image-plane direction (base beside the camera)."""
    Xc = cal["E"][:3, 3]
    z = float(Xc[2])
    if z > 0.02:
        return np.array([cal["fx"] * Xc[0] / z + cal["cx"], cal["fy"] * Xc[1] / z + cal["cy"]], float), "projected"
    d = np.array([Xc[0], Xc[1]], float)
    n = np.linalg.norm(d)
    d = d / n if n > 1e-6 else np.array([0.0, 1.0])
    return np.array([cal["cx"], cal["cy"]], float) + 1.0e4 * d, "direction"


def largest_cc(m):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m
    return lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def _leaves_of(comp_sk, dt, root, a, s, w_s, h_s):
    """Leaves of one component's skeleton from a root pixel: endpoints reachable from the root, sorted by geodesic path
    length (farthest first). Returns (leaves list, n_endpoints, reachable skeleton) or None."""
    cost = np.where(comp_sk, 1.0, np.inf)
    mcp = MCP_Geometric(cost, fully_connected=True)
    D, _ = mcp.find_costs([root])
    reach = comp_sk & np.isfinite(D)
    if reach.sum() < 3:
        return None
    nb = cv2.filter2D(comp_sk.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT).astype(int) - comp_sk.astype(int)
    ends = reach & (nb <= 1)
    ends[root] = False
    ey, ex = np.nonzero(ends)
    cands = [(float(D[y, x]), int(y), int(x)) for y, x in zip(ey, ex)]
    fallback = False
    if not cands:  # cycle: the farthest reachable skeleton pixel
        Dm = np.where(reach, D, -1.0)
        y, x = np.unravel_index(int(np.argmax(Dm)), Dm.shape)
        cands, fallback = [(float(D[y, x]), int(y), int(x))], True
    cands.sort(key=lambda c: -c[0])
    rr = int(round(a.tip_r * s))
    leaves = []
    for rank, (dist, y, x) in enumerate(cands[:a.top_k]):
        tb = mcp.traceback((y, x))  # root -> leaf
        pts = np.array(tb[::-1], float)  # leaf -> root (row, col)
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])) if len(pts) > 1 else np.zeros(0)
        cum = (np.concatenate([[0.0], np.cumsum(seg)]) / s).tolist()
        path = [(float(c / s), float(r_ / s)) for r_, c in pts]
        y0, y1, x0, x1 = max(0, y - rr), min(h_s, y + rr + 1), max(0, x - rr), min(w_s, x + rr + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = (yy - y) ** 2 + (xx - x) ** 2 <= rr * rr
        w_half = float(np.clip(dt[y0:y1, x0:x1][disc].max() / s, 12.0, 120.0))
        d_edge = min(path[0][0], path[0][1], NATIVE_W - 1 - path[0][0], NATIVE_H - 1 - path[0][1])  # native px
        at_border = bool(d_edge <= max(a.border_zone_px, a.border_w_factor * w_half))
        leaves.append(dict(rank=rank, tip=path[0], path=path, cum=cum, w=w_half, path_len=float(cum[-1]), at_border=at_border, leaf_fallback=fallback))
    return leaves, len(cands), reach


def _thick_border_contacts(comp, dt, thick_thr, s):
    """Connected border runs of the component whose neighbourhood is thick (an arm link crossing the image edge; a
    cable's contact is thin). List of dict(cx, cy, area, thick) in small px, possibly empty."""
    h_s, w_s = comp.shape
    b = max(1, int(round(4 * s)))
    border = np.zeros_like(comp)
    border[:b] = border[-b:] = True
    border[:, :b] = border[:, -b:] = True
    contact = comp & border
    if not contact.any():
        return []
    n, lab, stats, cents = cv2.connectedComponentsWithStats(contact.astype(np.uint8), connectivity=8)
    out = []
    for k in range(1, n):
        ys_r, xs_r = np.nonzero(lab == k)
        rr_ = int(np.ceil(2 * thick_thr)) + 1
        y0, y1, x0, x1 = max(0, ys_r.min() - rr_), min(h_s, ys_r.max() + rr_ + 1), max(0, xs_r.min() - rr_), min(w_s, xs_r.max() + rr_ + 1)
        thick = float(dt[y0:y1, x0:x1].max())
        if thick >= thick_thr:
            out.append(dict(cx=float(cents[k][0]), cy=float(cents[k][1]), area=int(stats[k, cv2.CC_STAT_AREA]), thick=thick))
    return out


def skeleton_leaves(mask_native, anchor, cal, a):
    """Arm skeleton rooted at the base (or the arm's image entry), walked to the end effector (iteration-1 formulation):
      1. mask at 640x360, 5x5 close, then a morphological OPENING with a disc of open_m (0.025 m) equivalent diameter at
         the plausible depth: removes cables (<= 1.5 cm) and thin clutter, keeps every arm link and the gripper body
         (>= 8 cm; the 2 cm fingers may go, the leaf then sits at the palm); components below min_comp_frac are dropped.
      2. skeleton of the opened mask (skimage), distance transform; ROOT RULE on the union skeleton: the base projection
         when it lies inside the image (+ margin) and the skeleton comes within base_touch_m of it; else the THICK border
         contact NEAREST the base anchor (the arm enters from the base side; a cable's contact is thin); else the
         skeleton pixel nearest the base anchor.
      3. CHAIN WALK: from the root, the farthest leaf of the root's component by path length. A leaf is AT THE BORDER
         when its edge distance is <= max(border_zone_px, border_w_factor x its half-width): the medial axis of a
         tube cut by the image edge ends about one half-width before the edge. If the farthest leaf is at the border
         the arm LEAVES the image and the chain continues in the unused component whose thick border contact (its
         re-entry) is nearest the exit leaf. If the farthest leaf is interior but an unused component lies within
         gap_w_factor (2) half-widths of it, the leaf is a CUT end (occlusion, or the opening at a thin junction) and
         the chain hops into that component (rooted at its skeleton pixel nearest the leaf). Up to max_hops hops; the
         walk ends at an interior leaf with no component nearby: its component's leaves (farthest first, top_k) are the
         end-effector candidates. If the walk ends on a border leaf the end effector is NOT VISIBLE in this view: no
         candidate (end_not_visible).
    Returns dict(leaves, root, root_mode, chain, ...) with paths in native px (leaf -> root of its component), or None."""
    s = a.skel_scale
    w_s, h_s = int(round(NATIVE_W * s)), int(round(NATIVE_H * s))
    ms = cv2.resize(mask_native.astype(np.uint8), (w_s, h_s), interpolation=cv2.INTER_NEAREST)
    ms = cv2.morphologyEx(ms, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    d_open = int(round(a.open_m * cal["fx"] / max(cal["plausible_depth"], 0.2) * s))
    d_open = max(3, d_open | 1)
    ms = cv2.morphologyEx(ms, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_open, d_open))) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ms.astype(np.uint8), connectivity=8)
    min_comp = a.min_comp_frac * w_s * h_s
    keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= min_comp]
    if not keep:
        return None
    ms = np.isin(lab, keep)
    dt = cv2.distanceTransform(ms.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(ms)
    ys, xs = np.nonzero(sk)
    if len(ys) < 5:
        return None
    anchor_px, anchor_mode = anchor
    ax, ay = anchor_px[0] * s, anchor_px[1] * s
    w_arm = float(np.percentile(dt[sk], a.w_stat))
    thick_thr = max(2.0, a.thick_frac * w_arm)
    root, root_mode = None, None
    if anchor_mode == "projected" and -a.base_margin_px * s <= ax < w_s + a.base_margin_px * s and -a.base_margin_px * s <= ay < h_s + a.base_margin_px * s:
        r = int(np.argmin((xs - ax) ** 2 + (ys - ay) ** 2))
        z_base = max(float(cal["E"][2, 3]), 0.05)
        if np.hypot(xs[r] - ax, ys[r] - ay) <= a.base_touch_m * cal["fx"] / z_base * s:
            root, root_mode = (int(ys[r]), int(xs[r])), "base"
    if root is None:
        cts = _thick_border_contacts(ms, dt, thick_thr, s)
        if cts:  # the arm enters the image from the base side: the thick border contact nearest the base anchor
            c_ = min(cts, key=lambda c: (c["cx"] - ax) ** 2 + (c["cy"] - ay) ** 2)
            r = int(np.argmin((xs - c_["cx"]) ** 2 + (ys - c_["cy"]) ** 2))
            root, root_mode = (int(ys[r]), int(xs[r])), "border"
    if root is None:
        r = int(np.argmin((xs - ax) ** 2 + (ys - ay) ** 2))
        root, root_mode = (int(ys[r]), int(xs[r])), "anchor_nearest"
    # chain walk over components
    comp_id = int(lab[root])
    used = {comp_id}
    chain = []
    leaves, n_ends, reach_all = None, 0, np.zeros_like(sk)
    cur_root = root
    for hop in range(a.max_hops + 1):
        comp = lab == comp_id
        res = _leaves_of(sk & comp, dt, cur_root, a, s, w_s, h_s)
        if res is None:
            break
        lv, ne, reach = res
        n_ends += ne
        reach_all |= reach
        chain.append(dict(comp=comp_id, area=int(stats[comp_id, cv2.CC_STAT_AREA]), root=(float(cur_root[1] / s), float(cur_root[0] / s)),
                          farthest_at_border=lv[0]["at_border"], path_len=round(lv[0]["path_len"])))
        leaves = lv
        ex, ey = lv[0]["tip"][0] * s, lv[0]["tip"][1] * s
        nxt = None
        if lv[0]["at_border"]:
            # the arm leaves the image here: continue in the unused component whose thick border contact (its re-entry)
            # is nearest the exit leaf
            for k in keep:
                if k in used:
                    continue
                for c_ in _thick_border_contacts(lab == k, dt, thick_thr, s):
                    d2 = (c_["cx"] - ex) ** 2 + (c_["cy"] - ey) ** 2
                    if nxt is None or d2 < nxt[0]:
                        nxt = (d2, k, (c_["cx"], c_["cy"]), "border")
        else:
            # an interior leaf may be a CUT end (occlusion, the opening at a thin junction): the chain continues into an
            # unused component lying within gap_w_factor half-widths of the leaf; otherwise it is the end effector
            gap = a.gap_w_factor * lv[0]["w"] * s
            for k in keep:
                if k in used:
                    continue
                ky, kx = np.nonzero(lab == k)
                d2 = float(np.min((kx - ex) ** 2 + (ky - ey) ** 2))
                if d2 <= gap * gap and (nxt is None or d2 < nxt[0]):
                    nxt = (d2, k, (ex, ey), "gap")
        if nxt is None:
            break
        _, comp_id, (px_, py_), hop = nxt
        chain[-1]["hop"] = hop
        used.add(comp_id)
        cy_, cx_ = np.nonzero(sk & (lab == comp_id))
        if len(cy_) == 0:
            break
        r = int(np.argmin((cx_ - px_) ** 2 + (cy_ - py_) ** 2))
        cur_root = (int(cy_[r]), int(cx_[r]))
    if leaves is None:
        return None
    end_not_visible = bool(leaves[0]["at_border"])
    if end_not_visible:  # the arm leaves the image and nothing re-enters: the end effector is not visible here
        leaves = []
    return dict(leaves=leaves, end_not_visible=end_not_visible, root=(float(root[1] / s), float(root[0] / s)), root_mode=root_mode, chain=chain, n_hops=len(chain) - 1,
                w_arm=float(w_arm / s), n_endpoints=n_ends, n_pruned_thin=0, skeleton_px=int(reach_all.sum()), mask_px=int(ms.sum()) * int(round(1 / s)) ** 2,
                open_px=int(round(d_open / s)), n_components=len(keep))


def verify_pair(l1, l2, cal1, cal2, anchors, a):
    """3D verification of a leaf pair (native px). Returns the record (ok flag + every gate value)."""
    X, r, rv = triangulate(cal1["P"], cal2["P"], l1["tip"], l2["tip"])
    dist = float(np.linalg.norm(X))
    rec = dict(X=[round(float(v), 4) for v in X], r_px=round(float(r), 2) if np.isfinite(r) else None, rv_px=[round(float(v), 2) if np.isfinite(v) else None for v in rv],
               dist=round(dist, 3), ok_reach3d=bool(dist < a.reach_m), ok_z=bool(X[2] > a.z_min), depths=[], ok_front=True, ok_ray=True, ok_reach2d=True,
               ok_6px=bool(np.isfinite(r) and max(rv) < 6.0))
    for leaf, cal, (anc, mode), rr in ((l1, cal1, anchors[0], rv[0]), (l2, cal2, anchors[1], rv[1])):
        _, _, z = project(cal, X)
        rec["depths"].append(round(float(z), 3))
        if not (z > a.min_cam_depth):
            rec["ok_front"] = False
            continue
        if not (rr < a.ray_tol_m * cal["fx"] / z):
            rec["ok_ray"] = False
        d_cb = float(cal["cam_to_base"])
        if d_cb <= a.reach_m:
            rec.setdefault("reach2d_mode", []).append("inside_sphere")  # camera inside the reach sphere: every pixel is in reach
        elif mode == "projected":
            R2 = cal["fx"] * np.tan(np.arcsin(a.reach_m / d_cb))
            rec.setdefault("reach2d_mode", []).append(f"circle_{R2:.0f}px")
            if np.hypot(leaf["tip"][0] - anc[0], leaf["tip"][1] - anc[1]) > R2:
                rec["ok_reach2d"] = False
        else:
            rec.setdefault("reach2d_mode", []).append("anchor_at_infinity")
    rec["ok"] = bool(rec["ok_front"] and rec["ok_ray"] and rec["ok_reach3d"] and rec["ok_z"] and rec["ok_reach2d"])
    return rec


def unpack_frame(packed_t, W=NATIVE_W):
    return np.unpackbits(packed_t, axis=1)[:, :W].astype(bool)


# ----------------------------------------------------------------------------------------------- stage cpu
def fallback_component(fg, moving, anchor, cal, a, s_proc, min_area_px):
    """MOG2 foreground component (with moving pixels) that CONTAINS the base neighbourhood or ENTERS the image on the
    base side (the task's rule; runs A/B used 'nearest the anchor', which accepted a moving forearm + held object far
    from the base). Base in view: the component must come within base_touch_m-equivalent (at the base depth) of the
    base projection. Base off-image: the component must touch the image border within base_touch_m-equivalent of the
    border point nearest the base anchor. Returns (native bool mask, distance px) or (None, None)."""
    anchor_px, anchor_mode = anchor
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    H, W = fg.shape
    ax, ay = anchor_px[0] / s_proc, anchor_px[1] / s_proc
    z_base = max(float(cal["E"][2, 3]), 0.05) if anchor_mode == "projected" else 0.05
    touch_px = a.base_touch_m * cal["fx"] / z_base / s_proc
    in_view = anchor_mode == "projected" and -a.base_margin_px / s_proc <= ax < W + a.base_margin_px / s_proc and -a.base_margin_px / s_proc <= ay < H + a.base_margin_px / s_proc
    bx, by = float(np.clip(ax, 0, W - 1)), float(np.clip(ay, 0, H - 1))  # image point nearest the anchor
    best, best_d = None, np.inf
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area_px:
            continue
        comp = lab == k
        if not (comp & moving).any():
            continue
        ys, xs = np.nonzero(comp)
        if in_view:
            d = float(np.sqrt(np.min((xs - ax) ** 2 + (ys - ay) ** 2)))
        else:
            edge = (xs <= 1) | (ys <= 1) | (xs >= W - 2) | (ys >= H - 2)
            if not edge.any():
                continue
            d = float(np.sqrt(np.min((xs[edge] - bx) ** 2 + (ys[edge] - by) ** 2)))
        if d <= touch_px and d < best_d:
            best, best_d = comp, d
    if best is None:
        return None, None
    nat = cv2.resize(best.astype(np.uint8), (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return nat, float(best_d * s_proc)


def draw_check(img, arm, leaves_info, pick_leaf, mask_cpu, anchor, mode, hdr):
    out = img.copy()
    if arm is not None and arm.any():
        cnts, _ = cv2.findContours(arm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 0, 255), 1)
    if mask_cpu is not None and mask_cpu.any():
        ov = out.copy()
        ov[mask_cpu] = (0, 255, 0)
        out = cv2.addWeighted(ov, 0.45, out, 0.55, 0)
    if leaves_info is not None:
        for lf in leaves_info["leaves"]:
            p = np.array(lf["path"], np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [p], False, (255, 128, 0), 1)
        rx, ry = leaves_info["root"]
        cv2.circle(out, (int(rx), int(ry)), 7, (0, 255, 255), -1)
        for ch in leaves_info.get("chain", [])[1:]:
            cv2.circle(out, (int(ch["root"][0]), int(ch["root"][1])), 7, (0, 200, 255), 2)
        for lf in leaves_info["leaves"]:
            cv2.circle(out, (int(lf["tip"][0]), int(lf["tip"][1])), 5, (200, 200, 200), 2)
    if pick_leaf is not None:
        cv2.circle(out, (int(pick_leaf["tip"][0]), int(pick_leaf["tip"][1])), 9, (0, 0, 255), 2)
    cx, cy = NATIVE_W / 2, NATIVE_H / 2
    d = np.array(anchor) - np.array([cx, cy])
    n = np.linalg.norm(d)
    if n > 1e-6:
        d = d / n
        cv2.arrowedLine(out, (int(cx), int(cy)), (int(cx + 120 * d[0]), int(cy + 120 * d[1])), (0, 255, 255), 2)
    cv2.rectangle(out, (0, 0), (NATIVE_W, 24), (0, 0, 0), -1)
    cv2.putText(out, hdr + f" base:{mode}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def stage_cpu(ep, a, out_dir, log):
    raw = os.path.join(RAW_ROOT, ep)
    meta = json.load(open(os.path.join(raw, f"metadata_{ep}.json")))
    serials = {c: str(meta[f"{c}_cam_serial"]) for c in CAMS}
    videos = {c: os.path.join(raw, "recordings", "MP4", f"{serials[c]}.mp4") for c in CAMS}
    for c, p in videos.items():
        if not os.path.exists(p):
            sys.exit(f"missing video for {c}: {p}")
    T = len([f for f in os.listdir(os.path.join(STORE_ROOT, ep, "dense", "cam")) if f.endswith(".npz")])
    cal = load_calib(ep, serials)
    anchors = {c: base_anchor(cal[c]) for c in CAMS}
    for c in CAMS:
        log(f"[{c}/{serials[c]}] base anchor {anchors[c][1]} px={[round(float(v)) for v in anchors[c][0]]} base_depth={cal[c]['E'][2, 3]:.3f} "
            f"fx={cal[c]['fx']:.0f} plausible_depth={cal[c]['plausible_depth']:.2f} cam_to_base={cal[c]['cam_to_base']:.2f} "
            f"reach2d={'vacuous (camera inside the reach sphere)' if cal[c]['cam_to_base'] <= a.reach_m else 'circle'}")
    # arm masks (cached text detector)
    text = {}
    for c in CAMS:
        if a.arm_source == "cached":
            p = os.path.join(a.text_arm_root, ep, f"{c}_{serials[c]}__text_arm_full_masks.npz")
        else:  # detect: this script's own --stage detect product (all arm components, not only the largest)
            p = os.path.join(out_dir, f"{c}_{serials[c]}__{METHOD_TEXT}_masks.npz")
        if not os.path.isfile(p):
            sys.exit(f"arm masks missing for --arm_source {a.arm_source}: {p}")
        z = np.load(p)
        assert tuple(int(v) for v in z["shape"]) == (T, NATIVE_H, NATIVE_W), (p, z["shape"], T)
        text[c] = dict(packed=z["union"], present=z["frames_present"].astype(bool), path=p)
        log(f"[{c}] text-arm masks ({a.arm_source}) {p}: present {int(text[c]['present'].sum())}/{T}")
    mog_args = SimpleNamespace(**MOG_DEFAULTS)
    views = {c: ViewMotion(mog_args) for c in CAMS}
    proc_w, proc_h = 640, 360
    s_proc = NATIVE_W / proc_w
    min_area_px = 0.0015 * proc_w * proc_h
    caps = {c: cv2.VideoCapture(videos[c]) for c in CAMS}
    packed_cpu = {c: np.zeros((T, NATIVE_H, NATIVE_W // 8), np.uint8) for c in CAMS}
    packed_arm = {c: np.zeros((T, NATIVE_H, NATIVE_W // 8), np.uint8) for c in CAMS}
    present = {c: np.zeros(T, bool) for c in CAMS}
    vp_xyz, vp_acc = np.full((T, 3), np.nan), np.zeros(T, bool)
    vp_leaf, vp_back_ok = np.full((T, 3), np.nan), np.zeros(T, bool)
    vp_src, vp_rank, vp_res = np.full(T, "", dtype="<U12"), np.full((T, 2), -1, np.int8), np.full(T, np.nan)
    records = []
    counts = dict(verified=0, by_source={}, text_leaf_both=0, motion_leaf_both=0, no_frame=0, ok_6px=0, lit_6px_would_pass=0, root_mode={c: {} for c in CAMS})
    t_dec = t_mog = t_skel = t_ver = t_emit = 0.0
    t_all = time.time()
    png_frames = set(int(x) for x in a.png_frames.split(",") if x.strip())
    for t in range(T):
        t0 = time.time()
        frames = {}
        for c in CAMS:
            ok, f = caps[c].read()
            if not ok:
                frames = None
                break
            frames[c] = f
        if frames is None:
            counts["no_frame"] += 1
            records.append(dict(t=t, verified=False, why="no_frame"))
            continue
        t1 = time.time()
        t_dec += t1 - t0
        st = {}
        for c in CAMS:
            small = cv2.resize(frames[c], (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            st[c] = views[c].step(small)
        t2 = time.time()
        t_mog += t2 - t1
        rec = dict(t=t, verified=False, cams={})
        leaves = {c: {} for c in CAMS}
        for c in CAMS:
            cr = dict(text_present=bool(text[c]["present"][t]))
            if text[c]["present"][t]:
                arm = unpack_frame(text[c]["packed"][t])
                li = skeleton_leaves(arm, anchors[c], cal[c], a)
                if li is not None:
                    leaves[c]["text"] = (li, arm)
                    counts["root_mode"][c][li["root_mode"]] = counts["root_mode"][c].get(li["root_mode"], 0) + 1
                    counts.setdefault("hops", {}).setdefault(c, {})
                    counts["hops"][c][str(li["n_hops"])] = counts["hops"][c].get(str(li["n_hops"]), 0) + 1
                    cr["text"] = dict(n_leaves=len(li["leaves"]), n_endpoints=li["n_endpoints"], n_pruned_thin=li["n_pruned_thin"], w_arm=round(li["w_arm"], 1),
                                      root=[round(v) for v in li["root"]], root_mode=li["root_mode"], n_hops=li["n_hops"], n_comp=li["n_components"], tips=[[round(v) for v in lf["tip"]] for lf in li["leaves"]],
                                      path_len=[round(lf["path_len"]) for lf in li["leaves"]], w=[round(lf["w"], 1) for lf in li["leaves"]])
            if st[c] is not None:
                fg, moving, moving_dil, gray, chroma = st[c]
                comp, dmin = fallback_component(fg, moving, anchors[c], cal[c], a, s_proc, min_area_px)
                if comp is not None:
                    cr["motion_comp_px"] = int(comp.sum())
                    cr["motion_comp_dist_to_anchor_px"] = round(dmin)
                    leaves[c]["motion_raw"] = comp
            rec["cams"][c] = cr
        t3 = time.time()
        t_skel += t3 - t2
        # verification: text-text first; motion skeletons are computed lazily
        pick = None
        if all("text" in leaves[c] for c in CAMS):
            counts["text_leaf_both"] += 1

        def get(c, src):
            if src == "text":
                return leaves[c].get("text")
            if "motion" not in leaves[c]:
                comp = leaves[c].get("motion_raw")
                if comp is None:
                    leaves[c]["motion"] = None
                else:
                    li = skeleton_leaves(comp, anchors[c], cal[c], a)
                    leaves[c]["motion"] = None if li is None else (li, comp)
                    if li is not None:
                        rec["cams"][c]["motion"] = dict(n_leaves=len(li["leaves"]), n_endpoints=li["n_endpoints"], n_pruned_thin=li["n_pruned_thin"], w_arm=round(li["w_arm"], 1),
                                                        root=[round(v) for v in li["root"]], root_mode=li["root_mode"], n_hops=li["n_hops"], n_comp=li["n_components"], tips=[[round(v) for v in lf["tip"]] for lf in li["leaves"]],
                                                        path_len=[round(lf["path_len"]) for lf in li["leaves"]], w=[round(lf["w"], 1) for lf in li["leaves"]])
            return leaves[c]["motion"]

        tried = []
        for s1, s2 in (("text", "text"), ("motion", "motion"), ("text", "motion"), ("motion", "text")):
            g1, g2 = get(CAMS[0], s1), get(CAMS[1], s2)
            if g1 is None or g2 is None:
                continue
            L1, L2 = g1[0]["leaves"], g2[0]["leaves"]
            pairs = sorted(((i, j) for i in range(len(L1)) for j in range(len(L2))), key=lambda ij: (ij[0] + ij[1], ij))
            for i, j in pairs:
                vr = verify_pair(L1[i], L2[j], cal[CAMS[0]], cal[CAMS[1]], (anchors[CAMS[0]], anchors[CAMS[1]]), a)
                tried.append(dict(src=[s1, s2], rank=[i, j], **{k: vr[k] for k in ("ok", "ok_ray", "ok_6px", "ok_reach3d", "ok_z", "ok_front", "ok_reach2d", "r_px", "dist")}))
                if vr["ok"]:
                    pick = dict(src=(s1, s2), rank=(i, j), leaves=(L1[i], L2[j]), arms=(g1[1], g2[1]), ver=vr)
                    break
            if pick is not None:
                break
        rec["tried"] = tried
        t4 = time.time()
        t_ver += t4 - t3
        if pick is not None:
            X = np.array(pick["ver"]["X"], float)
            rec.update(verified=True, src=list(pick["src"]), rank=list(pick["rank"]), ver=pick["ver"], prompts={})
            counts["verified"] += 1
            key = "+".join(pick["src"])
            counts["by_source"][key] = counts["by_source"].get(key, 0) + 1
            counts["ok_6px"] += int(pick["ver"]["ok_6px"])
            vp_leaf[t], vp_acc[t], vp_src[t], vp_rank[t], vp_res[t] = X, True, key, pick["rank"], pick["ver"]["r_px"]
            # declared hand-eye stand-in: the wrist camera sits lens_back_m behind the fingertips along the tool axis;
            # the skeleton-path points lens_back_m-equivalent behind each leaf are triangulated (same ray tolerance)
            back_pts = []
            for k, c in enumerate(CAMS):
                leaf = pick["leaves"][k]
                _, _, z_ = project(cal[c], X)
                back_pts.append(point_at(leaf["path"], leaf["cum"], a.lens_back_m * cal[c]["fx"] / max(z_, a.min_cam_depth)))
            Xb, rb, rvb = triangulate(cal[CAMS[0]]["P"], cal[CAMS[1]]["P"], back_pts[0], back_pts[1])
            back_ok = bool(np.isfinite(rb) and np.linalg.norm(Xb) < a.reach_m and Xb[2] > a.z_min)
            for k, c in enumerate(CAMS):
                _, _, zb = project(cal[c], Xb)
                back_ok = back_ok and zb > a.min_cam_depth and rvb[k] < a.ray_tol_m * cal[c]["fx"] / max(zb, a.min_cam_depth)
            vp_back_ok[t] = bool(back_ok)
            vp_xyz[t] = Xb if back_ok else X
            rec["back"] = dict(pts=[[round(v, 1) for v in bp] for bp in back_pts], X=[round(float(v), 4) for v in Xb], r_px=round(float(rb), 2) if np.isfinite(rb) else None, ok=bool(back_ok))
            for k, c in enumerate(CAMS):
                leaf, arm = pick["leaves"][k], pick["arms"][k]
                px, py, z = project(cal[c], X)
                rad = a.radius_m * cal[c]["fx"] / max(z, a.min_cam_depth)
                yy, xx = np.ogrid[:NATIVE_H, :NATIVE_W]
                disc = (xx - leaf["tip"][0]) ** 2 + (yy - leaf["tip"][1]) ** 2 <= rad * rad
                m = arm & disc
                packed_cpu[c][t] = np.packbits(m, axis=1)
                packed_arm[c][t] = np.packbits(arm, axis=1)
                present[c][t] = bool(m.any())
                prs = make_prompts(dict(tip=leaf["tip"], path=leaf["path"], cum=leaf["cum"], w=leaf["w"]), NATIVE_H, NATIVE_W, SimpleNamespace(**PROMPT_DEFAULTS))
                rec["prompts"][c] = dict(tip=[round(v, 1) for v in leaf["tip"]], w=round(leaf["w"], 1), proj=[round(px, 1), round(py, 1), round(z, 3)], radius_px=round(rad, 1),
                                         area_cpu=int(m.sum()), A=prs["A"], B=prs["B"], dir=prs["A"]["dir"], path_len=round(leaf["path_len"]), at_border=leaf["at_border"])
        if any(tr["ok_6px"] and tr["ok_reach3d"] and tr["ok_z"] and tr["ok_front"] and tr["ok_reach2d"] for tr in tried):
            counts["lit_6px_would_pass"] += 1
        t_emit += time.time() - t4
        records.append(rec)
        if t in png_frames:
            panels = []
            for k, c in enumerate(CAMS):
                li = None
                arm = None
                if pick is not None:
                    src = pick["src"][k]
                    li, arm = leaves[c][src]
                elif "text" in leaves[c]:
                    li, arm = leaves[c]["text"]
                m = unpack_frame(packed_cpu[c][t]) if present[c][t] else None
                hdr = f"{c}/{serials[c]} f{t}/{T - 1} root:{li['root_mode'] if li else '-'} " + (f"VERIFIED {'+'.join(pick['src'])} rank {pick['rank']} |X|={pick['ver']['dist']} r={pick['ver']['r_px']}px" if pick else "not verified")
                panels.append(draw_check(frames[c], arm, li, pick["leaves"][k] if pick else None, m, anchors[c][0], anchors[c][1], hdr))
            cv2.imwrite(os.path.join(out_dir, f"check_f{t:04d}.png"), cv2.resize(np.concatenate(panels, axis=1), (1280, 360), interpolation=cv2.INTER_AREA))
        if t % 100 == 0:
            log(f"  f{t}: verified={rec['verified']} src={rec.get('src')} rank={rec.get('rank')} X={rec.get('ver', {}).get('X')} r={rec.get('ver', {}).get('r_px')} "
                f"tried={len(tried)} ({time.time() - t_all:.0f}s)")
    for cap in caps.values():
        cap.release()
    total = time.time() - t_all
    products = {}
    for c in CAMS:
        p_cpu = os.path.join(out_dir, f"{c}_{serials[c]}__{METHOD_CPU}_masks.npz")
        np.savez_compressed(p_cpu, union=packed_cpu[c], shape=np.array([T, NATIVE_H, NATIVE_W]), frames_present=present[c])
        p_arm = os.path.join(out_dir, f"{c}_{serials[c]}__{METHOD_ARM}_masks.npz")
        arm_present = np.array([packed_arm[c][t].any() for t in range(T)])
        np.savez_compressed(p_arm, union=packed_arm[c], shape=np.array([T, NATIVE_H, NATIVE_W]), frames_present=arm_present)
        chk = np.load(p_cpu)
        u = np.unpackbits(chk["union"], axis=2)[:, :, :NATIVE_W].astype(bool)
        ne = u.reshape(T, -1).any(1)
        ok = u.shape == (T, NATIVE_H, NATIVE_W) and bool((chk["frames_present"] == ne).all()) and int(ne.sum()) == int(present[c].sum())
        products[c] = dict(cpu_npz=p_cpu, arm_npz=p_arm, T=T, frames_present_cpu=int(present[c].sum()), frames_present_arm=int(arm_present.sum()),
                           first_present=int(np.argmax(present[c])) if present[c].any() else None, product_ok=bool(ok),
                           mean_area_frac_present=float(u[present[c]].mean()) if present[c].any() else None)
        log(f"[{c}] wrote {p_cpu} present={int(present[c].sum())}/{T} ok={ok}")
    vp_path = os.path.join(out_dir, "verified_points.npz")
    np.savez(vp_path, frames=np.arange(T, dtype=np.int32), xyz=vp_xyz, accepted=vp_acc, source=vp_src, rank=vp_rank, resid_px=vp_res,
             xyz_leaf=vp_leaf, back_ok=vp_back_ok, lens_back_m=a.lens_back_m,
             note="xyz (used by the tracker gate) = base-frame (m) DLT point of the skeleton-path points lens_back_m behind the verified "
                  "leaf pair (declared hand-eye stand-in: the wrist camera sits ~0.10 m behind the fingertips), falling back to the leaf "
                  "itself when that triangulation fails (back_ok False); xyz_leaf = the raw verified leaf (the end effector). NaN / "
                  "accepted=False where nothing was verified; causal (frame t uses frames <= t); declared calibration only, no GT")
    chk = np.load(vp_path)
    vp_ok = bool((np.isfinite(chk["xyz"]).all(1) == chk["accepted"]).all()) and int(chk["accepted"].sum()) == counts["verified"]
    log(f"[verified_points] wrote {vp_path} accepted={int(vp_acc.sum())}/{T} back_ok={int(vp_back_ok.sum())} ok={vp_ok}")
    json.dump(dict(episode=ep, T=T, serials=serials, anchors={c: dict(px=[float(v) for v in anchors[c][0]], mode=anchors[c][1]) for c in CAMS}, records=records),
              open(os.path.join(out_dir, "skeleton_leaves.json"), "w"), default=float)
    timing = dict(total_s=round(total, 1), per_frame_ms=round(1000 * total / max(T, 1), 1), decode_ms=round(1000 * t_dec / max(T, 1), 1), mog2_ms=round(1000 * t_mog / max(T, 1), 1),
                  skeleton_ms=round(1000 * t_skel / max(T, 1), 1), verify_ms=round(1000 * t_ver / max(T, 1), 1), emit_ms=round(1000 * t_emit / max(T, 1), 1),
                  cpu_time_s=round(time.process_time(), 1), note="stage cpu, login node, OMP_NUM_THREADS=1; the text detector (~55 ms/frame/cam on H100, round 1) is NOT included")
    info = dict(episode=ep, method=METHOD, stage_cpu=dict(done=time.strftime("%Y-%m-%d %H:%M:%S"), params=vars(a), counts=counts, products=products,
                                                          verified_points=dict(npz=vp_path, accepted=int(vp_acc.sum()), back_ok=int(vp_back_ok.sum()), product_ok=vp_ok), timing=timing),
                serials=serials, store_frames=T, causal=True, uses_gt=False, arm_source=a.arm_source, text_arm_root=a.text_arm_root,
                declared_constants=dict(calibration="PointWorld optimized_extrinsics (world=base -> camera) + factory intrinsics, native 1280x720",
                                        base_anchor={c: dict(px=[round(float(v), 1) for v in anchors[c][0]], mode=anchors[c][1]) for c in CAMS},
                                        reach_m=a.reach_m, z_min=a.z_min, ray_tol_m=a.ray_tol_m, min_cam_depth=a.min_cam_depth, radius_m=a.radius_m, lens_back_m=a.lens_back_m,
                                        thick_frac=a.thick_frac, open_m=a.open_m, min_comp_frac=a.min_comp_frac, border_zone_px=a.border_zone_px, border_w_factor=a.border_w_factor, max_hops=a.max_hops, gap_w_factor=a.gap_w_factor, w_stat=a.w_stat, top_k=a.top_k, tip_r=a.tip_r, skel_scale=a.skel_scale, base_touch_m=a.base_touch_m, base_margin_px=a.base_margin_px,
                                        mog2=MOG_DEFAULTS, prompts=PROMPT_DEFAULTS, sam3=dict(drift_m=a.drift_m, reprompt_every=a.reprompt_every, min_gap=a.min_gap),
                                        note="all set from physical reasoning / inherited from round 1-3 scripts before the run; nothing tuned on the 13 scenes"))
    p_info = os.path.join(out_dir, "seed_info.json")
    if os.path.isfile(p_info):
        old = json.load(open(p_info))
        old.update(info)
        info = old
    json.dump(info, open(p_info, "w"), indent=1, default=str)
    log(f"SUMMARY stage cpu {ep}: verified {counts['verified']}/{T} by_source {counts['by_source']} text_leaf_both {counts['text_leaf_both']} "
        f"ok_6px_among_picks {counts['ok_6px']} literal_6px_would_pass {counts['lit_6px_would_pass']} | {timing}")
    return info


# ----------------------------------------------------------------------------------------------- stage detect (GPU)
def stage_detect(ep, a, out_dir, log, models):
    """SAM 3 text detector "robotic arm" per frame (single image, causal), seed_text_arm_sam3.select_arm instance rule,
    then ALL components >= min_area_frac are kept: the largest component under round 1's white-arm brightness rule, plus
    any other component that touches the image border (a re-entering piece of the arm chain). Round 1 saved only the
    largest component, which loses the gripper when the arm leaves and re-enters the image (44bb9c36 ext1)."""
    import torch
    from sam3_gripper_masks import read_frames
    from seed_text_arm_sam3 import select_arm
    det = models["detector"]
    # SAM3's video predictor enters a global bf16 autocast (sam3_multiplex_video_predictor.bf16_context) that round 1
    # inherited by building the video model first; the image model needs it explicitly (bf16 activations vs fp32 weights)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raw = os.path.join(RAW_ROOT, ep)
    meta = json.load(open(os.path.join(raw, f"metadata_{ep}.json")))
    serials = {c: str(meta[f"{c}_cam_serial"]) for c in CAMS}
    T = len([f for f in os.listdir(os.path.join(STORE_ROOT, ep, "dense", "cam")) if f.endswith(".npz")])
    info_cams = {}
    for c in CAMS:
        t0 = time.time()
        frames = read_frames(os.path.join(raw, "recordings", "MP4", f"{serials[c]}.mp4"))
        assert len(frames) == T, (len(frames), T)
        H, W = frames[0].shape[:2]
        masks = np.zeros((T, H, W), bool)
        n_fired = n_valid = n_extra = 0
        t_det = 0.0
        for t in range(T):
            t1 = time.time()
            with autocast:
                m_all, sc = det(frames[t])
            t_det += time.time() - t1
            if len(m_all):
                n_fired += 1
            union, kept = select_arm(m_all, sc, a.min_area_frac, a.big_area_frac)
            if union is None:
                continue
            g = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
            g_med = float(np.median(g))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(union.astype(np.uint8), connectivity=8)
            if n <= 1:
                continue
            order = sorted(range(1, n), key=lambda k: -stats[k, cv2.CC_STAT_AREA])
            big = lab == order[0]
            if float(g[big].mean()) < max(a.arm_gray_abs, a.arm_gray_rel * g_med):
                continue
            out = big.copy()
            n_valid += 1
            for k in order[1:]:
                if stats[k, cv2.CC_STAT_AREA] < a.min_area_frac * H * W:
                    break
                comp = lab == k
                if comp[:4].any() or comp[-4:].any() or comp[:, :4].any() or comp[:, -4:].any():
                    out |= comp
                    n_extra += 1
            masks[t] = out
        p_out = os.path.join(out_dir, f"{c}_{serials[c]}__{METHOD_TEXT}_masks.npz")
        save_masks(p_out, masks)
        sec = time.time() - t0
        info_cams[c] = dict(serial=serials[c], n_frames=T, detector_fired=n_fired, frames_present=int(masks.reshape(T, -1).any(1).sum()), n_valid=n_valid, n_extra_components=n_extra,
                            seconds=round(sec, 1), detector_ms_per_frame=round(1000 * t_det / T, 1), per_frame_ms=round(1000 * sec / T, 1), npz=p_out)
        log(f"[{c}] detect done: {sec:.0f}s, detector {1000 * t_det / T:.0f} ms/frame, fired {n_fired}/{T}, present {info_cams[c]['frames_present']}/{T}, extra border components {n_extra} -> {p_out}")
        del frames
    p_info = os.path.join(out_dir, "seed_info.json")
    info = json.load(open(p_info)) if os.path.isfile(p_info) else dict(episode=ep, method=METHOD)
    info["stage_detect"] = dict(done=time.strftime("%Y-%m-%d %H:%M:%S"), cams=info_cams, params=dict(phrase="robotic arm", det_version=a.det_version, thresh=a.thresh, min_area_frac=a.min_area_frac,
                                                                                                     big_area_frac=a.big_area_frac, arm_gray_abs=a.arm_gray_abs, arm_gray_rel=a.arm_gray_rel),
                                note="per-frame single-image text detector (causal); largest component under the white-arm brightness rule + border-touching extra components")
    json.dump(info, open(p_info, "w"), indent=1, default=str)


# ----------------------------------------------------------------------------------------------- stage sam3 (GPU)
def stage_sam3(ep, a, out_dir, log, models):
    import torch
    from sam3_gripper_masks import FfmpegWriter, read_frames
    tracker = models["tracker"]
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # as round 1 (global context of the video predictor)
    lv = json.load(open(os.path.join(out_dir, "skeleton_leaves.json")))
    T, serials = lv["T"], lv["serials"]
    recs = {r["t"]: r for r in lv["records"]}
    raw = os.path.join(RAW_ROOT, ep)
    videos = {c: os.path.join(raw, "recordings", "MP4", f"{serials[c]}.mp4") for c in CAMS}
    info_cams = {}
    masks_all = {}
    pa = SimpleNamespace(**PROMPT_DEFAULTS)
    for c in CAMS:
        t_cam = time.time()
        frames = read_frames(videos[c])
        assert len(frames) == T, (len(frames), T)
        H, W = frames[0].shape[:2]
        arm_all, _ = load_masks(os.path.join(out_dir, f"{c}_{serials[c]}__{METHOD_ARM}_masks.npz"))
        masks_track = np.zeros((T, H, W), bool)
        scores = np.full(T, np.nan)
        prompts = []
        gray_cache = {}

        def gray(t):
            if t not in gray_cache:
                gray_cache[t] = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
                for k in [k for k in gray_cache if k < t - 2]:
                    del gray_cache[k]
            return gray_cache[t]

        def sam_on_frame(t, pr):
            pts = torch.tensor([[x / W, y / H] for x, y in pr["points"]], dtype=torch.float32)
            labels = torch.tensor(pr["labels"], dtype=torch.int32)
            bx = pr["box"]
            box = np.array([[bx[0] / W, bx[1] / H, bx[2] / W, bx[3] / H]], dtype=np.float32)
            with autocast:
                _, _, _, vmasks = tracker.add_new_points_or_box(inference_state=state, frame_idx=int(t), obj_id=1, points=pts, labels=labels, box=box)
            return (vmasks[0][0] > 0).cpu().numpy()

        def apply_prompt(t, reason):
            """Round-1 hypothesis rules at the verified leaf of frame t (see module docstring)."""
            pr_all = recs[t]["prompts"][c]
            g = gray(t)
            w = pr_all["w"]
            tip = np.array(pr_all["tip"])
            arm = arm_all[t]
            moving, mot_arm, diff = None, None, None
            if t >= 1:
                diff = cv2.absdiff(g, gray(t - 1)) > pa.motion_thr
                x0, y0 = int(max(0, tip[0] - 6 * w)), int(max(0, tip[1] - 6 * w))
                x1, y1 = int(min(W, tip[0] + 6 * w)), int(min(H, tip[1] + 6 * w))
                near = np.zeros_like(arm)
                near[y0:y1, x0:x1] = arm[y0:y1, x0:x1]
                mot_arm = float(diff[near].mean()) if near.any() else 0.0
                moving = mot_arm >= pa.motion_min
            cands = {}
            order = ["A", "B"]
            for name in order:
                m = sam_on_frame(t, pr_all[name])
                area = float(m.sum())
                ys, xs = np.nonzero(m)
                if area > 0:
                    cen = np.array([xs.mean(), ys.mean()])
                    dist_tip = float(np.linalg.norm(cen - tip))
                    gmean = float(g[m].mean())
                    mot = float(diff[m].mean()) if diff is not None else None
                else:
                    dist_tip, gmean, mot = float("inf"), 255.0, None
                ok = (pa.area_min_w2 * w * w <= area <= pa.area_max_w2 * w * w) and dist_tip <= pa.adj_max_w * w
                why = "" if ok else "area/adjacency"
                if ok and moving and mot is not None and mot < pa.motion_ratio * mot_arm:
                    ok, why = False, f"static (mot {mot:.2f} vs arm {mot_arm:.2f})"
                cands[name] = dict(area_w2=round(area / (w * w), 2), dist_tip_w=round(dist_tip / w, 2), gray=round(gmean, 1),
                                   mot=None if mot is None else round(mot, 3), ok=bool(ok), why=why)
            valid = [n for n in order if cands[n]["ok"]]
            if not valid:
                log(f"[{c}] prompt @ f{t} ({reason}): both hypotheses rejected {json.dumps(cands)}")
                tracker.clear_all_points_in_frame(state, int(t), 1, need_output=False)
                return None
            if "A" in valid and "B" in valid:
                pick = "B" if (cands["A"]["gray"] > pa.a_not_dark and cands["B"]["gray"] < cands["A"]["gray"] - pa.b_darker_by) else "A"
            else:
                pick = valid[0]
            if pick != order[-1]:
                sam_on_frame(t, pr_all[pick])
            pr = pr_all[pick]
            prompts.append(dict(frame=int(t), reason=reason, mode=pick, box=pr["box"], points=pr["points"], labels=pr["labels"], w=w, tip=pr_all["tip"],
                                cands=cands, moving=moving, mot_arm=None if mot_arm is None else round(mot_arm, 3), src=recs[t]["src"], rank=recs[t]["rank"]))
            log(f"[{c}] prompt @ f{t} ({reason}): pick {pick} moving={moving} | A {cands['A']} | B {cands['B']} | w {w:.0f} tip {[int(v) for v in tip]} src {recs[t]['src']}")
            return pick

        with autocast:
            state = tracker.init_state(video_path=videos[c], offload_video_to_cpu=a.offload_video)
        assert state["num_frames"] == T, (state["num_frames"], T)
        verified_frames = [t for t in range(T) if recs.get(t, {}).get("verified") and c in recs[t].get("prompts", {})]
        first = None
        for t0 in verified_frames:
            if apply_prompt(t0, "seed") is not None:
                first = t0
                break
        n_pass = 0
        n_reprompt = dict(lost=0, drift=0, periodic=0, rejected=0)
        if first is not None:
            cur, last_prompt = first, first
            vset = set(verified_frames)
            while cur < T:
                n_pass += 1
                gen = tracker.propagate_in_video(state, start_frame_idx=int(cur), max_frame_num_to_track=T, reverse=False, propagate_preflight=True, tqdm_disable=True)
                reprompt = None
                autocast.__enter__()
                for fidx, _, _, vmasks, obj_scores in gen:
                    fidx = int(fidx)
                    m = (vmasks[0][0] > 0).cpu().numpy()
                    s_ = obj_scores
                    try:
                        s_ = float(np.asarray(s_.detach().float().cpu().numpy() if hasattr(s_, "detach") else s_).reshape(-1)[0])
                    except Exception:  # noqa: BLE001
                        s_ = float("nan")
                    masks_track[fidx], scores[fidx] = m, s_
                    if fidx == cur or fidx not in vset or fidx - last_prompt < a.min_gap:
                        continue
                    lost = (not m.any()) or (np.isfinite(s_) and s_ < 0)
                    pr = recs[fidx]["prompts"][c]
                    drift = False
                    if not lost:
                        ys, xs = np.nonzero(m)
                        dpx = float(np.hypot(xs.mean() - pr["tip"][0], ys.mean() - pr["tip"][1]))
                        drift = dpx > a.drift_m / a.radius_m * pr["radius_px"]
                    periodic = a.reprompt_every > 0 and fidx - last_prompt >= a.reprompt_every
                    if lost or drift or periodic:
                        reprompt = (fidx, "lost" if lost else ("drift %.0fpx" % dpx if drift else "periodic"))
                        break
                autocast.__exit__(None, None, None)
                if reprompt is None:
                    break
                gen.close()
                fidx, reason = reprompt
                n_reprompt[reason.split()[0]] += 1
                if apply_prompt(fidx, reason) is None:
                    n_reprompt["rejected"] += 1
                cur, last_prompt = fidx, fidx
        del state
        torch.cuda.empty_cache()
        sec = time.time() - t_cam
        n_present = int(masks_track.reshape(T, -1).any(1).sum())
        p_out = os.path.join(out_dir, f"{c}_{serials[c]}__{METHOD}_masks.npz")
        save_masks(p_out, masks_track)
        masks_all[c] = masks_track
        info_cams[c] = dict(serial=serials[c], n_frames=T, first_prompt_frame=first, n_verified_frames=len(verified_frames), n_prompts=len(prompts), prompts=prompts,
                            reprompts=n_reprompt, n_propagation_passes=n_pass, modes="".join(p["mode"] for p in prompts), frames_present=n_present,
                            mean_area_present=float(masks_track[masks_track.reshape(T, -1).any(1)].mean()) if n_present else None,
                            obj_score_min=float(np.nanmin(scores)) if np.isfinite(scores).any() else None, seconds=round(sec, 1), per_frame_ms=round(1000 * sec / T, 1), npz=p_out)
        log(f"[{c}] sam3 done: {sec:.0f}s ({1000 * sec / T:.0f} ms/frame incl. decode), first prompt f{first}, {len(prompts)} prompts {n_reprompt}, {n_pass} passes, present {n_present}/{T} -> {p_out}")
        del frames
    if a.video:
        t0 = time.time()
        fr = {c: read_frames(videos[c]) for c in CAMS}
        pw, ph = 640, 360
        writer = FfmpegWriter(os.path.join(out_dir, f"{METHOD}_side_by_side.mp4"), pw * 2, ph, 15)
        pbf = {c: {p["frame"]: p for p in info_cams[c]["prompts"]} for c in CAMS}
        for t in range(T):
            panels = []
            for c in CAMS:
                img = fr[c][t].copy()
                m = masks_all[c][t]
                if m.any():
                    ov = img.copy()
                    ov[m] = (0, 255, 0)
                    img = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
                r = recs.get(t, {})
                if r.get("verified") and c in r.get("prompts", {}):
                    pr = r["prompts"][c]
                    cv2.circle(img, (int(pr["tip"][0]), int(pr["tip"][1])), 8, (0, 0, 255), 2)
                    cv2.circle(img, (int(pr["proj"][0]), int(pr["proj"][1])), int(pr["radius_px"]), (255, 0, 255), 1)
                p = pbf[c].get(t)
                if p:
                    x0, y0, x1, y1 = [int(round(v)) for v in p["box"]]
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 3)
                    for (x, y), l in zip(p["points"], p["labels"]):
                        cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0) if l else (0, 0, 255), -1)
                cv2.rectangle(img, (0, 0), (NATIVE_W, 24), (0, 0, 0), -1)
                cv2.putText(img, f"{c}/{serials[c]} {METHOD} f{t}/{T - 1} " + ("verified " + "+".join(r["src"]) if r.get("verified") else "unverified") + (f" PROMPT {p['mode']} {p['reason']}" if p else ""),
                            (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA))
            writer.write(np.concatenate(panels, axis=1))
        writer.close()
        log(f"video written in {time.time() - t0:.0f}s")
    p_info = os.path.join(out_dir, "seed_info.json")
    info = json.load(open(p_info)) if os.path.isfile(p_info) else dict(episode=ep, method=METHOD)
    info["stage_sam3"] = dict(done=time.strftime("%Y-%m-%d %H:%M:%S"), cams=info_cams, params=dict(drift_m=a.drift_m, reprompt_every=a.reprompt_every, min_gap=a.min_gap),
                              total_seconds=round(sum(info_cams[c]["seconds"] for c in CAMS), 1), per_frame_ms_both_cams=round(1000 * sum(info_cams[c]["seconds"] for c in CAMS) / T, 1),
                              note="SAM3 tracker stage on GPU; per_frame includes MP4 decode + init_state; the text detector (~55 ms/frame/cam, round 1) and stage cpu are extra")
    json.dump(info, open(p_info, "w"), indent=1, default=str)
    log(f"SUMMARY stage sam3 {ep}: " + " | ".join(f"{c}: first f{info_cams[c]['first_prompt_frame']} prompts {info_cams[c]['n_prompts']} present {info_cams[c]['frames_present']}/{T} {info_cams[c]['seconds']}s" for c in CAMS))


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", default=None)
    ap.add_argument("--episodes", default=None, help="file with one episode per line (stage sam3: models are built once)")
    ap.add_argument("--out_root", default=os.path.join(IDEAS_ROOT, METHOD))
    ap.add_argument("--stage", default="cpu", choices=["detect", "cpu", "sam3"])
    ap.add_argument("--arm_source", default="cached", choices=["cached", "detect"], help="stage cpu: round-1 cached per-frame text masks (largest component only) or this script's --stage detect product (all components)")
    ap.add_argument("--text_arm_root", default=TEXT_ARM_ROOT)
    # skeleton (declared)
    ap.add_argument("--skel_scale", type=float, default=0.5)
    ap.add_argument("--thick_frac", type=float, default=0.6, help="thick skeleton pixel: distance transform >= this x the arm half-width")
    ap.add_argument("--open_m", type=float, default=0.025, help="opening disc diameter (m-equivalent at the plausible depth): removes cables (<= 1.5 cm), keeps links / gripper body (>= 8 cm)")
    ap.add_argument("--min_comp_frac", type=float, default=0.002, help="opened-mask components below this image fraction are dropped")
    ap.add_argument("--border_zone_px", type=float, default=12, help="a leaf within max(this, border_w_factor x its half-width) of the image edge is 'at the border' (the arm leaves the image)")
    ap.add_argument("--border_w_factor", type=float, default=1.5, help="the medial axis of a tube cut by the image edge ends about one half-width before the edge")
    ap.add_argument("--max_hops", type=int, default=2, help="chain walk: at most this many hops into other components (border re-entry or gap)")
    ap.add_argument("--gap_w_factor", type=float, default=2.0, help="chain walk: an interior leaf with an unused component within this many half-widths is a cut end; the chain hops across")
    ap.add_argument("--w_stat", type=float, default=65, help="percentile of the distance transform on the skeleton = arm half-width")
    ap.add_argument("--top_k", type=int, default=3, help="leaves per view tried in the cross-view verification, farthest first")
    ap.add_argument("--tip_r", type=float, default=100, help="half-width near the leaf = max distance transform within this radius (native px)")
    ap.add_argument("--base_touch_m", type=float, default=0.30, help="root = base projection only if the skeleton comes within this (m-equivalent at the base depth) of it")
    ap.add_argument("--base_margin_px", type=float, default=50, help="... and the base projection lies inside the image (+ this margin)")
    # verification (declared)
    ap.add_argument("--reach_m", type=float, default=0.9)
    ap.add_argument("--z_min", type=float, default=-0.10)
    ap.add_argument("--ray_tol_m", type=float, default=0.04)
    ap.add_argument("--min_cam_depth", type=float, default=0.15)
    # emission
    ap.add_argument("--radius_m", type=float, default=0.25, help="CPU variant: arm-mask pixels within this metre-equivalent radius of the leaf")
    ap.add_argument("--lens_back_m", type=float, default=0.10, help="declared hand-eye stand-in: verified_points.xyz = the point this far behind the leaf along the skeleton path (triangulated)")
    ap.add_argument("--det_version", default="sam3.1"); ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--min_area_frac", type=float, default=0.006); ap.add_argument("--big_area_frac", type=float, default=0.02)
    ap.add_argument("--arm_gray_abs", type=float, default=50.0); ap.add_argument("--arm_gray_rel", type=float, default=0.7)
    ap.add_argument("--drift_m", type=float, default=0.25, help="stage sam3: re-prompt when the tracked centroid is farther than this from the verified leaf")
    ap.add_argument("--reprompt_every", type=int, default=10)
    ap.add_argument("--min_gap", type=int, default=5)
    ap.add_argument("--offload_video", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--png_frames", default="", help="stage cpu: comma list of frames dumped as check PNGs")
    a = ap.parse_args()
    eps = [a.episode] if a.episode else [l.strip() for l in open(a.episodes) if l.strip()]
    models = {}
    if a.stage == "sam3":
        import torch  # noqa: F401
        from sam3.model_builder import build_sam3_video_model
        t0 = time.time()
        vm = build_sam3_video_model()
        tracker = vm.tracker
        tracker.backbone = vm.detector.backbone
        tracker.clear_non_cond_mem_around_input = True
        tracker.iter_use_prev_mask_pred = False
        models = dict(tracker=tracker)
        print(f"sam3 video model built in {time.time() - t0:.0f}s", flush=True)
    elif a.stage == "detect":
        from seed_text_arm_sam3 import ArmDetector
        t0 = time.time()
        models = dict(detector=ArmDetector("robotic arm", version=a.det_version, thresh=a.thresh))
        print(f"sam3 text detector built in {time.time() - t0:.0f}s", flush=True)
    for ep in eps:
        out_dir = os.path.join(a.out_root, ep)
        os.makedirs(out_dir, exist_ok=True)
        logf = open(os.path.join(out_dir, f"seed_log_{a.stage}.txt"), "a")

        def log(msg):
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()

        log(f"== {METHOD} stage {a.stage} {ep} {time.strftime('%Y-%m-%d %H:%M:%S')} args {vars(a)}")
        try:
            if a.stage == "cpu":
                stage_cpu(ep, a, out_dir, log)
            elif a.stage == "detect":
                stage_detect(ep, a, out_dir, log, models)
            else:
                stage_sam3(ep, a, out_dir, log, models)
        except Exception as e:  # noqa: BLE001
            import traceback
            log(f"FAILED {ep}: {e!r}\n{traceback.format_exc()}")
            if a.episode:
                raise
        logf.close()


if __name__ == "__main__":
    main()
