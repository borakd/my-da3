#!/usr/bin/env python
"""METHOD text_arm_sam3: training-free, causal gripper masks on the two DROID exterior cameras (1280x720).

Per exterior camera of one episode, frame by frame, forward only:
  1. SAM 3 image model, text prompt "robotic arm" (fired on the RAIL exterior frames in the 2026-09-28 probe) -> instance
     masks on the current frame alone. Keep the instances that look like an arm (area / border-contact rules, see
     `select_arm`; the probe showed a permanent 0.3%-area false hit), union them, keep the largest connected component,
     and require it to be bright (all DROID rigs are white Franka arms: mean gray >= max(50, 0.7 x image median); the
     RAIL false hits were a black arm-like object at 16-44 gray against 77-102 for the arm).
  2. Find the gripper end of the arm mask (2x downscaled). The arm ENTERS the image where the mask touches the border
     (largest border-contact run; if the mask floats, the mask pixel nearest the border). Tip = the mask pixel
     geodesically farthest from the entry inside the mask (skimage MCP) among pixels whose thin run (geodesic distance
     to the nearest 'core' pixel, distance transform >= 0.9 x the arm half-width scale = 65th pct of the distance
     transform on the skeleton) is at most 1 scale: in effect the far end of the thick body (gripper body / wrist),
     never the thick black cable that the text mask always includes nor the finger tips; pixels within 24 px of the
     image edge are excluded. (CPU sweep on 5 scenes against the projected kinematic wrist-camera centre, GT used for
     scoring only: the centre lies inside prompt box A on 91.7% of detector frames with these values, 65% with the
     first skeleton version.) Half-width w = the largest
     distance-transform value within 100 px of the tip (the gripper body behind the finger tips); last-link direction
     = tip minus the geodesic-path point 2w behind it.
  3. Prompt at the tip. The text mask often EXCLUDES the black gripper and ends at the wrist flange (RAIL ext2), so two
     box+click hypotheses are tried with the SAM 2-style tracker of build_sam3_video_model() on that frame:
     A "gripper inside the arm mask" (box 4.5w behind the tip .. 1w ahead, positive clicks 0.5w and 2w behind the tip on
     the path, negative click 7w back or at the entry end of the path), B "gripper beyond the tip" (box 0.5w behind .. 4.5w ahead along the last link,
     positive clicks 1.2w / 3w ahead, negative click 3w back). Link 7 + flange + gripper + wrist camera are one rigid
     body, so a mask that reaches up to the flange is fine. Selection, causal: (i) a hypothesis mask must have a sane
     area (0.3..14 w^2) and centroid within 5w of the tip; (ii) when the arm is moving (frame difference t-1 -> t over
     the arm near the tip), a hypothesis whose mask is static (black curtains, drawer fronts) is rejected; (iii) A is
     preferred; B wins only when A's mask is not dark (mean gray > 90) and B's is darker by > 30 (the Robotiq gripper is
     black, the Franka links white: a hand-set hardware prior, nothing trained). The winner is propagated FORWARD only.
  4. Automatic re-prompting, causal: the detector runs on every frame (--detect_every). The tracker is re-prompted on
     frame t (both hypotheses again; no memory, non-cond memory of the +-7 surrounding frames cleared) when the tracked
     mask is empty / has a negative object score, when less than --inside_min of it lies inside the (dilated) box of
     the current mode built from frame t's tip, or every --reprompt_every frames; propagation continues from t.
  Fallback product (no SAM refinement): the raw text-arm mask clipped to box A, per frame where the detector ran.

Causality: at frame t only frames <= t are used (detector: frame t alone; motion test: t-1 and t; tracker: memory of
past frames + the prompt frame). The tracker's init_state decodes the whole MP4 up front for I/O convenience only;
nothing from frames > t enters the computation for frame t. Ground truth is not used anywhere in this script.

Outputs (MASK FILE CONTRACT: 'union' uint8 (T,H,W/8) = packbits of the bool (T,720,1280) mask along axis 2, 'shape',
'frames_present'):
  <out_root>/EP/<cam>_<serial>__text_arm_sam3_masks.npz   tracker masks (the METHOD)
  <out_root>/EP/<cam>_<serial>__text_arm_box_masks.npz    fallback: text-arm mask inside box A
  <out_root>/EP/<cam>_<serial>__text_arm_full_masks.npz   diagnostic: the selected arm text mask per frame
  <out_root>/EP/seed_info.json                            prompts, hypotheses, re-prompts, detector stats, timing,
                                                          optional IoU vs reference masks (diagnostic only)
  <out_root>/EP/text_arm_sam3_side_by_side.mp4 (+ snap PNGs) unless --no_video
Needs a GPU (SAM 3 hard-codes CUDA); see seed_text_arm_sam3.sbatch.
"""
import argparse
import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam3_gripper_masks import FfmpegWriter, RAW_ROOT, read_frames, render  # noqa: E402

SEED_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/text_arm_sam3"
STORE = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
METHOD = "text_arm_sam3"
FALLBACK = "text_arm_box"
CAMS = ("ext1", "ext2")


# ------------------------------------------------------------------------------------------------ mask file contract
def save_masks(path, masks_bool):
    """masks_bool: (T, H, W) bool. Writes the seedstudy contract."""
    T, H, W = masks_bool.shape
    np.savez_compressed(path, union=np.packbits(masks_bool.astype(np.uint8), axis=2), shape=np.array([T, H, W]),
                        frames_present=masks_bool.reshape(T, -1).any(1))


def load_masks(path):
    z = np.load(path)
    T, H, W = [int(v) for v in z["shape"]]
    return np.unpackbits(z["union"], axis=2)[:, :, :W].astype(bool), z["frames_present"].astype(bool)


# ------------------------------------------------------------------------------------------------ arm detector
class ArmDetector:
    """SAM 3 image model + text prompt. Per-frame, single-image: strictly causal."""

    def __init__(self, phrase, version="sam3.1", thresh=0.3):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
        ckpt = download_ckpt_from_hf(version) if version == "sam3.1" else None  # resolves in the offline HF cache
        self.model = build_sam3_image_model(checkpoint_path=ckpt)
        self.proc = Sam3Processor(self.model, confidence_threshold=thresh)
        self.phrase = phrase
        with torch.inference_mode():
            self.text_out = self.model.backbone.forward_text([phrase], device="cuda")
        self.n_calls, self.t_total = 0, 0.0

    def __call__(self, frame_bgr):
        import PIL.Image
        import torch
        t0 = time.time()
        pil = PIL.Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            st = self.proc.set_image(pil)
            st["backbone_out"].update(self.text_out)
            st["geometric_prompt"] = self.model._get_dummy_prompt()
            st = self.proc._forward_grounding(st)
            masks = st["masks"].reshape(-1, st["original_height"], st["original_width"]).cpu().numpy().astype(bool)
            scores = st["scores"].reshape(-1).float().cpu().numpy()
        self.n_calls += 1
        self.t_total += time.time() - t0
        return masks, scores


def select_arm(masks, scores, min_area_frac=0.006, big_area_frac=0.02, border_px=4):
    """Instances that can be a robot arm: area >= min_area_frac and (touches the border or area >= big_area_frac).
    Returns (union bool mask or None, list of kept instance indices)."""
    if len(masks) == 0:
        return None, []
    H, W = masks[0].shape
    keep = []
    for i, m in enumerate(masks):
        a = m.mean()
        if a < min_area_frac:
            continue
        touches = m[:border_px].any() or m[-border_px:].any() or m[:, :border_px].any() or m[:, -border_px:].any()
        if touches or a >= big_area_frac:
            keep.append(i)
    if not keep:
        return None, []
    union = np.zeros((H, W), bool)
    for i in keep:
        union |= masks[i]
    return union, keep


def largest_cc(m):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


# ------------------------------------------------------------------------------------------------ arm geometry
def arm_tip(arm_mask, scale=0.5, border_px=4, thick_frac=0.9, thin_run_w=1.0, border_zone=24, tip_r=100, w_stat=65):
    """Gripper end of the arm mask = the mask pixel geodesically farthest (inside the mask) from where the arm enters
    the image, among pixels whose 'thin run' (geodesic distance to the nearest thick pixel, dt >= thick_frac x arm
    half-width) is at most thin_run_w half-widths. That keeps the short thin wrist stub and the fingers (they end a
    tip) but rejects the thick black cable the text mask always includes (a long thin run to the image border).
    Returns dict(tip=(x,y), path=[(x,y),...] tip->entry in full-res px, cum=[geodesic px], w=half-width px,
    entry=(x,y), entry_mode, path_len) or None."""
    from skimage.graph import MCP_Geometric
    from skimage.morphology import skeletonize
    H, W = arm_mask.shape
    ms = cv2.resize(arm_mask.astype(np.uint8), (int(round(W * scale)), int(round(H * scale))), interpolation=cv2.INTER_NEAREST)
    ms = cv2.morphologyEx(ms, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ms = largest_cc(ms > 0)
    h, w = ms.shape
    if ms.sum() < 30:
        return None
    dt = cv2.distanceTransform(ms.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(ms)
    w_arm = float(np.percentile(dt[sk], w_stat)) if sk.any() else float(dt.max())  # arm half-width scale (w_stat-th pct of dt on the skeleton)
    thick = ms & (dt >= max(2.0, thick_frac * w_arm))
    b = max(1, int(round(border_px * scale)))
    border = np.zeros_like(ms)
    border[:b] = border[-b:] = True
    border[:, :b] = border[:, -b:] = True
    contact = ms & border
    if contact.any():
        n, lab, stats, cents = cv2.connectedComponentsWithStats(contact.astype(np.uint8), connectivity=8)
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        ys, xs = np.nonzero(lab == k)
        entry, entry_mode = cents[k], "border"
    else:
        ys0, xs0 = np.nonzero(ms)
        d = np.minimum(np.minimum(xs0, w - 1 - xs0), np.minimum(ys0, h - 1 - ys0))
        i = int(np.argmin(d))
        ys, xs = np.array([ys0[i]]), np.array([xs0[i]])
        entry, entry_mode = np.array([xs0[i], ys0[i]], float), "nearest_border"
    costs = np.where(ms, 1.0, np.inf)
    mcp = MCP_Geometric(costs, fully_connected=True)
    D, _ = mcp.find_costs(list(zip(ys.tolist(), xs.tolist())))
    D = np.where(ms & np.isfinite(D), D, -1.0)
    if thick.any():
        ty_, tx_ = np.nonzero(thick)
        Dthin, _ = MCP_Geometric(costs, fully_connected=True).find_costs(list(zip(ty_.tolist(), tx_.tolist())))
        Dthin = np.where(np.isfinite(Dthin), Dthin, np.inf)
    else:
        Dthin = np.zeros_like(D)
    bz = max(2 * b, int(round(border_zone * scale)))
    cand = np.where(Dthin <= thin_run_w * w_arm, D, -1.0)
    cand[:bz] = cand[-bz:] = -1.0
    cand[:, :bz] = cand[:, -bz:] = -1.0
    if cand.max() <= 0:
        cand = np.where(Dthin <= thin_run_w * w_arm, D, -1.0)
    if cand.max() <= 0:
        cand = D
    if cand.max() <= 0:
        return None
    ty, tx = np.unravel_index(int(np.argmax(cand)), cand.shape)
    tb = mcp.traceback((int(ty), int(tx)))  # start -> tip
    pts = np.array(tb[::-1], float)  # tip -> entry, (row, col)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])) if len(pts) > 1 else np.zeros(0)
    cum = np.concatenate([[0.0], np.cumsum(seg)]) / scale
    path = [(float(c / scale), float(rr / scale)) for rr, c in pts]
    # half-width at the tip: the geodesic path hugs the inner boundary at bends, so the distance transform along it
    # under-reports the width; use the largest distance-transform value within tip_r px of the tip instead (the palm /
    # gripper body behind the finger tips, the forearm end behind a thin wrist stub)
    rr = int(round(tip_r * scale))
    y0, y1, x0, x1 = max(0, ty - rr), min(h, ty + rr + 1), max(0, tx - rr), min(w, tx + rr + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = (yy - ty) ** 2 + (xx - tx) ** 2 <= rr * rr
    w_half = float(np.clip(dt[y0:y1, x0:x1][disc].max() / scale, 12.0, 120.0))
    return dict(tip=path[0], path=path, cum=cum.tolist(), w=w_half, entry=(float(entry[0] / scale), float(entry[1] / scale)),
                entry_mode=entry_mode, path_len=float(cum[-1]), w_arm=float(w_arm / scale), thin_run=float(Dthin[ty, tx] / scale))


def point_at(path, cum, s):
    """Point on the path at geodesic distance s from the tip (clamped)."""
    if s <= 0:
        return path[0]
    if s >= cum[-1]:
        return path[-1]
    k = int(np.searchsorted(cum, s))
    k = min(max(k, 1), len(path) - 1)
    a, b = np.array(path[k - 1]), np.array(path[k])
    f = (s - cum[k - 1]) / max(cum[k] - cum[k - 1], 1e-6)
    return tuple((a + f * (b - a)).tolist())


def _box_from(p_a, p_b, half, H, W):
    d = p_b - p_a
    n = np.linalg.norm(d)
    perp = np.array([-d[1], d[0]]) / n if n > 1e-6 else np.array([1.0, 0.0])
    corners = np.stack([p_a + half * perp, p_a - half * perp, p_b + half * perp, p_b - half * perp])
    x0, y0 = np.clip(corners.min(0), 0, [W - 1, H - 1])
    x1, y1 = np.clip(corners.max(0), 0, [W - 1, H - 1])
    return [float(x0), float(y0), float(x1), float(y1)]


def make_prompts(tipinfo, H, W, args):
    """Two box+click hypotheses (full-res px) from the tip geometry: A gripper inside the arm mask, B beyond the tip."""
    path, cum, w = tipinfo["path"], tipinfo["cum"], tipinfo["w"]
    tip = np.array(path[0])
    back_pt = np.array(point_at(path, cum, 2.0 * w))
    d = tip - back_pt
    n = np.linalg.norm(d)
    d = d / n if n > 1e-6 else np.array([0.0, 1.0])

    def clip(p):
        return [float(np.clip(p[0], 0, W - 1)), float(np.clip(p[1], 0, H - 1))]

    boxA = _box_from(np.array(point_at(path, cum, args.box_back * w)), tip + args.box_fwd * w * d, args.box_half * w, H, W)
    ptsA = [list(point_at(path, cum, 0.5 * w)), list(point_at(path, cum, 2.0 * w))]
    labA = [1, 1]
    # negative click on the arm behind the box: at neg_back w when the path is long enough, else at the entry end of
    # the path if that is clear of the box (without it SAM segments the whole visible arm, RAIL ext1)
    s_neg = args.neg_back * w if cum[-1] > args.neg_back * w + w else cum[-1]
    if s_neg > (args.box_back + 1.0) * w:
        pn = point_at(path, cum, s_neg)
        if not (boxA[0] <= pn[0] <= boxA[2] and boxA[1] <= pn[1] <= boxA[3]):
            ptsA.append(list(pn))
            labA.append(0)
    boxB = _box_from(tip - 0.5 * w * d, tip + 4.5 * w * d, args.box_half * w, H, W)
    ptsB = [clip(tip + 1.2 * w * d), clip(tip + 3.0 * w * d)]
    labB = [1, 1]
    if cum[-1] > 3.0 * w + w:
        ptsB.append(list(point_at(path, cum, 3.0 * w)))
        labB.append(0)
    common = dict(w=float(w), tip=[float(tip[0]), float(tip[1])], dir=[float(d[0]), float(d[1])])
    return dict(A=dict(box=boxA, points=[[float(x), float(y)] for x, y in ptsA], labels=labA, **common),
                B=dict(box=boxB, points=[[float(x), float(y)] for x, y in ptsB], labels=labB, **common))


def inside_frac(mask, box, w, H, W):
    if not mask.any():
        return 0.0
    x0, y0, x1, y1 = box
    x0, y0 = int(max(0, x0 - w)), int(max(0, y0 - w))
    x1, y1 = int(min(W - 1, x1 + w)), int(min(H - 1, y1 + w))
    return float(mask[y0:y1 + 1, x0:x1 + 1].sum() / mask.sum())


# ------------------------------------------------------------------------------------------------ per camera
def run_camera(cam, serial, mp4, frames, detector, tracker, args, log):
    import torch
    T = len(frames)
    H, W = frames[0].shape[:2]
    t_cam = time.time()
    det_cache = {}   # frame -> dict(valid, arm(bool HxW)|None, prompts{A,B}|None, ...)
    masks_track = np.zeros((T, H, W), bool)
    masks_fb = np.zeros((T, H, W), bool)
    masks_arm = np.zeros((T, H, W), bool)   # selected text-arm mask per detector frame (diagnostic product, lets the geometry be tuned on CPU)
    scores = np.full(T, np.nan)
    n_det_fired = n_det_valid = 0
    gray_cache = {}

    def gray(t):
        if t not in gray_cache:
            gray_cache[t] = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
            for k in [k for k in gray_cache if k < t - 2]:
                del gray_cache[k]
        return gray_cache[t]

    def detect(t):
        nonlocal n_det_fired, n_det_valid
        if t in det_cache:
            return det_cache[t]
        masks, sc = detector(frames[t])
        arm, kept = select_arm(masks, sc, args.min_area_frac, args.big_area_frac)
        d = dict(frame=t, n_inst=int(len(masks)), kept=[int(k) for k in kept], probs=[round(float(p), 3) for p in sc], valid=False,
                 arm=None, prompts=None)
        if len(masks):
            n_det_fired += 1
        if arm is not None:
            arm = largest_cc(arm)
            g = gray(t)
            g_arm, g_med = float(g[arm].mean()), float(np.median(g))
            d.update(arm_gray=g_arm, img_median_gray=g_med)
            if g_arm < max(args.arm_gray_abs, args.arm_gray_rel * g_med):
                d.update(reject=f"arm too dark: gray {g_arm:.0f} < max({args.arm_gray_abs}, {args.arm_gray_rel} x median {g_med:.0f})")
                det_cache[t] = d
                return d
            masks_arm[t] = arm
            tip = arm_tip(arm, thick_frac=args.thick_frac, thin_run_w=args.thin_run_w, w_stat=args.w_stat)
            if tip is not None and tip["path_len"] >= args.min_path_w * tip["w"]:
                prs = make_prompts(tip, H, W, args)
                d.update(valid=True, arm=arm, prompts=prs, entry=tip["entry"], entry_mode=tip["entry_mode"], w=tip["w"],
                         path_len=tip["path_len"], arm_area_frac=float(arm.mean()), tip=prs["A"]["tip"])
                x0, y0, x1, y1 = [int(round(v)) for v in prs["A"]["box"]]
                fb = np.zeros_like(arm)
                fb[y0:y1 + 1, x0:x1 + 1] = arm[y0:y1 + 1, x0:x1 + 1]
                masks_fb[t] = fb
                n_det_valid += 1
            elif tip is not None:
                d.update(reject=f"path_len {tip['path_len']:.0f} < {args.min_path_w} w ({tip['w']:.0f})")
        det_cache[t] = d
        return d

    prompts = []

    def sam_on_frame(t, pr):
        pts = torch.tensor([[x / W, y / H] for x, y in pr["points"]], dtype=torch.float32)
        labels = torch.tensor(pr["labels"], dtype=torch.int32)
        bx = pr["box"]
        box = np.array([[bx[0] / W, bx[1] / H, bx[2] / W, bx[3] / H]], dtype=np.float32)
        _, _, _, vmasks = tracker.add_new_points_or_box(inference_state=state, frame_idx=int(t), obj_id=1, points=pts, labels=labels, box=box)
        return (vmasks[0][0] > 0).cpu().numpy()

    def apply_prompt(t, d, reason):
        """Try both hypotheses on frame t, keep one by the causal rules in the module docstring (re-issued last so the
        tracker state holds it). Returns the mode or None."""
        g = gray(t)
        w = d["w"]
        tip = np.array(d["tip"])
        moving, mot_arm, diff = None, None, None
        if t >= 1:
            diff = cv2.absdiff(g, gray(t - 1)) > args.motion_thr
            x0, y0 = int(max(0, tip[0] - 6 * w)), int(max(0, tip[1] - 6 * w))
            x1, y1 = int(min(W, tip[0] + 6 * w)), int(min(H, tip[1] + 6 * w))
            near = np.zeros_like(d["arm"])
            near[y0:y1, x0:x1] = d["arm"][y0:y1, x0:x1]
            mot_arm = float(diff[near].mean()) if near.any() else 0.0
            moving = mot_arm >= args.motion_min
        cands = {}
        order = ["A", "B"]
        for name in order:
            m = sam_on_frame(t, d["prompts"][name])
            area = float(m.sum())
            ys, xs = np.nonzero(m)
            if area > 0:
                cen = np.array([xs.mean(), ys.mean()])
                dist_tip = float(np.linalg.norm(cen - tip))
                gmean = float(g[m].mean())
                mot = float(diff[m].mean()) if diff is not None else None
            else:
                dist_tip, gmean, mot = float("inf"), 255.0, None
            ok = (args.area_min_w2 * w * w <= area <= args.area_max_w2 * w * w) and dist_tip <= args.adj_max_w * w
            why = "" if ok else "area/adjacency"
            if ok and moving and mot is not None and mot < args.motion_ratio * mot_arm:
                ok, why = False, f"static (mot {mot:.2f} vs arm {mot_arm:.2f})"
            cands[name] = dict(area_w2=round(area / (w * w), 2), dist_tip_w=round(dist_tip / w, 2), gray=round(gmean, 1),
                               mot=None if mot is None else round(mot, 3), ok=bool(ok), why=why)
        valid = [n for n in order if cands[n]["ok"]]
        if not valid:
            log(f"[{cam}] prompt @ f{t} ({reason}): both hypotheses rejected {json.dumps(cands)}")
            tracker.clear_all_points_in_frame(state, int(t), 1, need_output=False)  # undo: no stale prompt left on this frame
            return None
        if "A" in valid and "B" in valid:
            pick = "B" if (cands["A"]["gray"] > args.a_not_dark and cands["B"]["gray"] < cands["A"]["gray"] - args.b_darker_by) else "A"
        else:
            pick = valid[0]
        if pick != order[-1]:
            sam_on_frame(t, d["prompts"][pick])  # re-issue so the tracker holds the winner
        pr = d["prompts"][pick]
        prompts.append(dict(frame=int(t), reason=reason, mode=pick, box=pr["box"], points=pr["points"], labels=pr["labels"], w=pr["w"],
                            tip=pr["tip"], cands=cands, moving=moving, mot_arm=None if mot_arm is None else round(mot_arm, 3),
                            arm_area_frac=d["arm_area_frac"], entry_mode=d["entry_mode"]))
        log(f"[{cam}] prompt @ f{t} ({reason}): pick {pick} moving={moving} | A {cands['A']} | B {cands['B']} | w {w:.0f} tip {[int(v) for v in tip]} arm {d['arm_area_frac']*100:.1f}%")
        return pick

    state = tracker.init_state(video_path=mp4, offload_video_to_cpu=args.offload_video)
    assert state["num_frames"] == T, (state["num_frames"], T)
    assert (state["video_height"], state["video_width"]) == (H, W), (state["video_height"], state["video_width"], H, W)

    # phase 1: first valid seed, scanning forward
    t0, mode = 0, None
    while t0 < T:
        if detect(t0)["valid"]:
            mode = apply_prompt(t0, det_cache[t0], "seed")
            if mode is not None:
                break
        t0 += 1
    if t0 >= T:
        log(f"[{cam}] no valid seed on any frame; empty products")
        return dict(masks_track=masks_track, masks_fb=masks_fb, masks_arm=masks_arm, scores=scores, prompts=prompts, det=det_cache, first_frame=None,
                    n_det_calls=len(det_cache), n_det_fired=n_det_fired, n_det_valid=n_det_valid, sec=time.time() - t_cam, n_pass=0)
    # phase 2: forward propagation with causal re-prompting
    cur, last_prompt = t0, t0
    n_pass = 0
    while cur < T:
        n_pass += 1
        gen = tracker.propagate_in_video(state, start_frame_idx=int(cur), max_frame_num_to_track=T, reverse=False,
                                         propagate_preflight=True, tqdm_disable=True)
        reprompt = None
        for fidx, _, _, vmasks, obj_scores in gen:
            fidx = int(fidx)
            m = (vmasks[0][0] > 0).cpu().numpy()
            s = obj_scores
            try:
                s = float(np.asarray(s.detach().float().cpu().numpy() if hasattr(s, "detach") else s).reshape(-1)[0])
            except Exception:
                s = float("nan")
            masks_track[fidx], scores[fidx] = m, s
            if fidx == cur:
                continue
            lost = (not m.any()) or (np.isfinite(s) and s < 0)
            want_det = (args.detect_every > 0 and fidx % args.detect_every == 0) or lost
            if not want_det:
                continue
            d = detect(fidx)
            if fidx - last_prompt < args.min_gap or not d["valid"]:
                continue
            fr = inside_frac(m, d["prompts"][mode]["box"], d["w"], H, W)
            periodic = args.reprompt_every > 0 and fidx - last_prompt >= args.reprompt_every
            if lost or fr < args.inside_min or periodic:
                reprompt = (fidx, d, "lost" if lost else ("drift inside=%.2f" % fr if fr < args.inside_min else "periodic"))
                break
        if reprompt is None:
            break
        gen.close()
        fidx, d, reason = reprompt
        new_mode = apply_prompt(fidx, d, reason)
        # the generator was closed at fidx; frames > fidx are untracked. Resume from fidx: with a new prompt it is a
        # cond frame, after a rejection (prompt undone) it is simply re-run from memory. Either way last_prompt = fidx
        # is the cooldown before the next attempt.
        if new_mode is not None:
            mode = new_mode
        cur, last_prompt = fidx, fidx
    del state
    torch.cuda.empty_cache()
    sec = time.time() - t_cam
    log(f"[{cam}] done: {sec:.0f}s, {n_pass} propagation passes, {len(prompts)} prompts, detector calls {len(det_cache)} "
        f"(fired {n_det_fired}, valid {n_det_valid}), tracked frames {int(masks_track.reshape(T,-1).any(1).sum())}/{T}, "
        f"fallback frames {int(masks_fb.reshape(T,-1).any(1).sum())}/{T}")
    return dict(masks_track=masks_track, masks_fb=masks_fb, masks_arm=masks_arm, scores=scores, prompts=prompts, det=det_cache, first_frame=t0,
                n_det_calls=len(det_cache), n_det_fired=n_det_fired, n_det_valid=n_det_valid, sec=sec, n_pass=n_pass)


# ------------------------------------------------------------------------------------------------ overlay
def draw_overlay(frame, m, s, d, prompt_by_frame, fidx, header, mode):
    out = None
    if m.any():
        p = 1.0 / (1.0 + np.exp(-s)) if np.isfinite(s) else 1.0
        out = dict(ids=np.array([1]), probs=np.array([p]), boxes=np.zeros((1, 4)), masks=m[None])
    img = render(frame, out, header)
    if d is not None and d.get("arm") is not None:
        cnts, _ = cv2.findContours(d["arm"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (255, 0, 255), 1)
        if d.get("prompts"):
            p = prompt_by_frame.get(fidx)
            pr = d["prompts"][p["mode"]] if p else d["prompts"][mode or "A"]
            x0, y0, x1, y1 = [int(round(v)) for v in pr["box"]]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255) if p else (255, 200, 0), 3 if p else 1)
            if p:
                for (x, y), l in zip(pr["points"], pr["labels"]):
                    cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0) if l else (0, 0, 255), -1)
                cv2.putText(img, f"PROMPT {p['mode']} ({p['reason']})", (x0, max(y1 + 18, 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (int(pr["tip"][0]), int(pr["tip"][1])), 5, (255, 255, 255), 2)
    return img


def write_overlays(ep, out_dir, serials, frames, res, args):
    cams = list(res)
    T = min(len(frames[c]) for c in cams)
    pw, ph = 640, 360
    writer = FfmpegWriter(os.path.join(out_dir, f"{METHOD}_side_by_side.mp4"), pw * len(cams), ph, args.fps)
    tracked = [i for i in range(T) if any(res[c]["masks_track"][i].any() for c in cams)]
    snaps = set()
    if tracked:
        snaps = {tracked[int(len(tracked) * q)] for q in (0.1, 0.5, 0.9)}
    for i in range(T):
        panels = []
        for c in cams:
            r = res[c]
            d = r["det"].get(i)
            pbf = {p["frame"]: p for p in r["prompts"]}
            mode = None
            for p in r["prompts"]:
                if p["frame"] <= i:
                    mode = p["mode"]
            hdr = f"{c}/{serials[c]} {METHOD} f{i}/{T-1} " + (f"mode {mode} " if mode else "") + ("" if d is None else f"det n={d['n_inst']} " + ("valid" if d["valid"] else "novalid"))
            img = draw_overlay(frames[c][i], r["masks_track"][i], r["scores"][i], d, pbf, i, hdr, mode)
            if i in snaps:
                cv2.imwrite(os.path.join(out_dir, f"snap_{c}_f{i:04d}.png"), img)
            panels.append(cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA))
        writer.write(np.concatenate(panels, axis=1))
    writer.close()
    return sorted(snaps)


def iou_vs_reference(masks, ref_path):
    if not os.path.isfile(ref_path):
        return None
    ref, ref_present = load_masks(ref_path)
    if ref.shape != masks.shape:
        return dict(note=f"shape mismatch {ref.shape} vs {masks.shape}")
    both = ref_present & masks.reshape(len(masks), -1).any(1)
    ious = [float((ref[i] & masks[i]).sum() / max((ref[i] | masks[i]).sum(), 1)) for i in np.nonzero(both)[0]]
    return dict(ref=ref_path, frames_ref=int(ref_present.sum()), frames_ours=int(masks.reshape(len(masks), -1).any(1).sum()), frames_both=int(both.sum()),
                iou_mean=float(np.mean(ious)) if ious else None, iou_median=float(np.median(ious)) if ious else None,
                iou_min=float(np.min(ious)) if ious else None, frac_iou_ge_0p5=float(np.mean(np.array(ious) >= 0.5)) if ious else None)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out_root", default=SEED_ROOT)
    ap.add_argument("--cams", default="ext1,ext2")
    ap.add_argument("--phrase", default="robotic arm")
    ap.add_argument("--det_version", default="sam3.1", choices=["sam3.1", "sam3"], help="weights for the text detector (the probe evidence is sam3.1)")
    ap.add_argument("--thresh", type=float, default=0.3, help="detector confidence threshold (probe used 0.3)")
    ap.add_argument("--min_area_frac", type=float, default=0.006)
    ap.add_argument("--big_area_frac", type=float, default=0.02)
    ap.add_argument("--arm_gray_abs", type=float, default=50.0, help="reject arm masks darker than this mean gray (white Franka prior)")
    ap.add_argument("--arm_gray_rel", type=float, default=0.7, help="... or darker than this fraction of the image median gray")
    ap.add_argument("--thick_frac", type=float, default=0.9, help="a mask pixel is 'thick' when its distance transform is >= this x the arm half-width scale (CPU sweep 2026-09-29 on 5 scenes: 0.9/1.0/65 best)")
    ap.add_argument("--thin_run_w", type=float, default=1.0, help="tip candidates may be at most this many half-width scales (geodesic) away from thick pixels; rejects cables")
    ap.add_argument("--w_stat", type=float, default=65, help="percentile of the distance transform on the skeleton that defines the arm half-width scale")
    ap.add_argument("--min_path_w", type=float, default=2.0, help="reject arm masks whose skeleton path is shorter than this many half-widths")
    ap.add_argument("--box_back", type=float, default=4.5)
    ap.add_argument("--box_fwd", type=float, default=1.0)
    ap.add_argument("--box_half", type=float, default=2.0)
    ap.add_argument("--neg_back", type=float, default=7.0)
    ap.add_argument("--area_min_w2", type=float, default=0.3, help="hypothesis mask must cover at least this many w^2")
    ap.add_argument("--area_max_w2", type=float, default=14.0, help="... and at most this many w^2 (gripper body + link 7 is about 9 w^2; the whole arm is far more)")
    ap.add_argument("--adj_max_w", type=float, default=5.0, help="hypothesis mask centroid must be within this many w of the tip")
    ap.add_argument("--motion_thr", type=int, default=15, help="abs frame-difference threshold (gray levels) for the motion test")
    ap.add_argument("--motion_min", type=float, default=0.10, help="arm is 'moving' when this fraction of arm pixels near the tip changed")
    ap.add_argument("--motion_ratio", type=float, default=0.4, help="a hypothesis mask must move at least this fraction of the arm's motion")
    ap.add_argument("--a_not_dark", type=float, default=90.0, help="B can win only if A's mask mean gray exceeds this")
    ap.add_argument("--b_darker_by", type=float, default=30.0, help="... and B's mask is darker than A's by this much")
    ap.add_argument("--detect_every", type=int, default=1, help="run the text detector every N frames (0 = only when lost)")
    ap.add_argument("--inside_min", type=float, default=0.5, help="re-prompt when less than this fraction of the tracked mask lies inside the dilated mode box")
    ap.add_argument("--min_gap", type=int, default=5, help="minimum frames between prompts")
    ap.add_argument("--reprompt_every", type=int, default=10, help="periodic re-prompt (0 = off)")
    ap.add_argument("--offload_video", action="store_true", help="keep the tracker's decoded frames on CPU")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--ref_root", default="/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/gripper_sam3",
                    help="where <EP>/<cam>_<serial>__gripper_boxclick_masks.npz reference masks may exist (diagnostic IoU only)")
    args = ap.parse_args()

    ep = args.episode
    raw = os.path.join(RAW_ROOT, ep)
    meta = json.load(open(os.path.join(raw, f"metadata_{ep}.json")))
    serials = dict(ext1=meta["ext1_cam_serial"], ext2=meta["ext2_cam_serial"])
    cams = [c for c in args.cams.split(",") if c]
    out_dir = os.path.join(args.out_root, ep)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "seed_log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"== {METHOD} {ep} {time.strftime('%Y-%m-%d %H:%M:%S')} args {vars(args)}")
    cam_dir = os.path.join(STORE, ep, "dense", "cam")
    n_store = len([f for f in os.listdir(cam_dir) if f.endswith(".npz")]) if os.path.isdir(cam_dir) else None
    videos = {c: os.path.join(raw, "recordings", "MP4", f"{serials[c]}.mp4") for c in cams}
    for c, p in videos.items():
        if not os.path.exists(p):
            sys.exit(f"missing video for {c}: {p}")
    t_read = time.time()
    frames = {c: read_frames(videos[c]) for c in cams}
    for c in cams:
        log(f"{c} {serials[c]}: {len(frames[c])} frames {frames[c][0].shape[1]}x{frames[c][0].shape[0]} (store cam count {n_store})")
        if n_store is not None and len(frames[c]) != n_store:
            log(f"WARNING {c}: MP4 frame count {len(frames[c])} != store {n_store}")
    log(f"frames read in {time.time()-t_read:.0f}s")

    import torch
    from sam3.model_builder import build_sam3_video_model
    t0 = time.time()
    detector = ArmDetector(args.phrase, version=args.det_version, thresh=args.thresh)
    vm = build_sam3_video_model()
    tracker = vm.tracker
    tracker.backbone = vm.detector.backbone
    tracker.clear_non_cond_mem_around_input = True   # a re-prompt wipes the stale non-cond memory of the +-7 surrounding frames
    tracker.iter_use_prev_mask_pred = False           # a re-prompt does not get the (possibly drifted) previous mask as a prior
    log(f"models built in {time.time()-t0:.0f}s (detector {args.det_version}, tracker sam3)")

    res, info = {}, dict(episode=ep, method=METHOD, serials=serials, args=vars(args), store_frames=n_store, cams={})
    for c in cams:
        r = run_camera(c, serials[c], videos[c], frames[c], detector, tracker, args, log)
        res[c] = r
        T = len(frames[c])
        p_method = os.path.join(out_dir, f"{c}_{serials[c]}__{METHOD}_masks.npz")
        p_fb = os.path.join(out_dir, f"{c}_{serials[c]}__{FALLBACK}_masks.npz")
        save_masks(p_method, r["masks_track"])
        save_masks(p_fb, r["masks_fb"])
        save_masks(os.path.join(out_dir, f"{c}_{serials[c]}__text_arm_full_masks.npz"), r["masks_arm"])  # diagnostic: selected arm text mask
        ref = os.path.join(args.ref_root, ep, f"{c}_{serials[c]}__gripper_boxclick_masks.npz")
        ci = dict(serial=serials[c], n_frames=T, first_prompt_frame=r["first_frame"], prompts=r["prompts"], n_prompts=len(r["prompts"]),
                  n_reprompts=max(0, len(r["prompts"]) - 1), n_propagation_passes=r.get("n_pass"),
                  modes=[p["mode"] for p in r["prompts"]],
                  detector_calls=r["n_det_calls"], detector_fired=r["n_det_fired"], detector_valid=r["n_det_valid"],
                  frames_present_method=int(r["masks_track"].reshape(T, -1).any(1).sum()),
                  frames_present_fallback=int(r["masks_fb"].reshape(T, -1).any(1).sum()),
                  mean_area_method=float(r["masks_track"].mean()), mean_area_fallback=float(r["masks_fb"].mean()),
                  obj_score_min=float(np.nanmin(r["scores"])) if np.isfinite(r["scores"]).any() else None,
                  obj_score_mean=float(np.nanmean(r["scores"])) if np.isfinite(r["scores"]).any() else None,
                  seconds=r["sec"], detector_seconds=detector.t_total, masks_npz=p_method, fallback_npz=p_fb,
                  iou_vs_reference_method=iou_vs_reference(r["masks_track"], ref), iou_vs_reference_fallback=iou_vs_reference(r["masks_fb"], ref),
                  per_frame=[dict(frame=t, valid=d["valid"], n_inst=d["n_inst"], probs=d["probs"], w=d.get("w"), arm_area_frac=d.get("arm_area_frac"),
                                  arm_gray=d.get("arm_gray"), entry_mode=d.get("entry_mode"), tip=d.get("tip"), reject=d.get("reject")) for t, d in sorted(r["det"].items())])
        info["cams"][c] = ci
        detector.t_total = 0.0
    snaps = []
    if not args.no_video:
        t0 = time.time()
        snaps = write_overlays(ep, out_dir, serials, frames, res, args)
        log(f"overlay written in {time.time()-t0:.0f}s, snaps {snaps}")
    info["snap_frames"] = snaps
    info["total_seconds"] = sum(info["cams"][c]["seconds"] for c in cams)
    json.dump(info, open(os.path.join(out_dir, "seed_info.json"), "w"), indent=1, default=str)
    log("SUMMARY")
    for c in cams:
        ci = info["cams"][c]
        iou = ci["iou_vs_reference_method"]
        log(f"  {c}: seed f{ci['first_prompt_frame']} prompts {ci['n_prompts']} modes {''.join(ci['modes'])} present {ci['frames_present_method']}/{ci['n_frames']} "
            f"fallback {ci['frames_present_fallback']}/{ci['n_frames']} det valid {ci['detector_valid']}/{ci['detector_calls']} "
            f"{ci['seconds']:.0f}s" + (f" IoU-vs-ref mean {iou['iou_mean']:.3f} (n={iou['frames_both']})" if iou and iou.get("iou_mean") is not None else ""))
    log(f"DONE {out_dir}")


if __name__ == "__main__":
    main()
