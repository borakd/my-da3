#!/usr/bin/env python
"""motion_sam3: automatic gripper masks on the two STATIC exterior DROID cameras of one episode, seeded from motion.

Method (strictly causal, training-free, no ground truth in the method path):
  1. Motion blob per exterior view at frame t from frames t-2, t-1, t only: three-frame differencing
     (|I_t-I_{t-1}| & |I_t-I_{t-2}|, max over BGR), open/close/fill, largest component = the moving arm. Its geodesic
     diameter (BFS on a 1/4-res grid) gives two extremities e1, e2 and whether the blob touches the image border near each.
  2. Joint 3D decision (declared exterior calibration only: PointWorld optimized_extrinsics + factory intrinsics):
     the four extremity pairings across the two views are triangulated; a pairing is valid when its reprojection residual
     is < --r_max px and the point lies inside the robot's reach (< --reach m from the base, plausible height). The valid
     pairing farthest from the base is the gripper tip in both views (the gripper is the moving part farthest from the
     base); the per-view border rule (the arm enters from out of frame) can override within a 10 cm margin. Frames with
     no valid pairing (background people, single-view motion) yield no prompt: this is the person filter.
  3. Prompt = distal cap (blob pixels within geodesic distance L of the tip): its bounding box + up to 3 interior clicks
     (distance-transform maxima). Prompts are gated by the wrist camera's own frame difference (the arm is moving).
  4. The first gated frame prompts the SAM3 SAM2-style tracker (build_sam3_video_model, box + clicks, relative coords).
     The prompt-frame mask must pass a sanity check (area window, centroid inside the box, bbox overlap); otherwise the
     prompt is cleared and the next candidate is tried.
  5. Propagation is FORWARD ONLY, one frame per call (propagate_in_video(start_frame_idx=t, max_frame_num_to_track=0,
     reverse=False)); frames before the prompt frame are never read by the tracker and get empty masks. The tracker's
     memory at frame t holds conditioning frames <= t and the previous num_maskmem-1 frames; object pointers are
     restricted to t' <= t (sam3_tracker_base._prepare_memory_conditioned_features).
  6. Re-prompt: when the tracked mask has been absent (empty or outside the area window) for more than --lost_frames
     consecutive frames and a gated candidate exists at the current (not yet tracked) frame, a new box+clicks prompt is
     added as a fresh conditioning frame and propagation continues from it.
  7. Online consistency: a tracked gripper must stay put when the wrist camera sees no motion (wrist frame difference
     < w_still) and, when the wrist sees motion AND this view holds a large moving blob, the mask must either move or be
     part of that blob. Over the last 12 voting frames, >= 7 disagreements drop the track (tracker state reset, no
     revision of already-emitted frames) and force a re-prompt. Fine manipulation gives no votes: the wrist-image
     difference stays high at close range while the exterior blob is small.

Outputs (MASK FILE CONTRACT): <out_root>/<EP>/<cam>_<serial>__motion_sam3_masks.npz with 'union' uint8 (T,H,W/8)
= np.packbits(mask, axis=2), 'shape' = [T,H,W], 'frames_present' bool (T,); plus seed_info.json, per-cam csv, overlay
mp4s (per cam and side-by-side) and look/*.png frames for eyeballing. Exterior frames at the native 1280x720.

    python seed_motion_sam3.py --episode EP --out_root .../seedstudy/motion_sam3
    python seed_motion_sam3.py --episode EP --cue_only          # CPU: motion cue + candidate prompts only
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam3_gripper_masks import FfmpegWriter, RAW_ROOT, read_frames, render, side_by_side  # noqa: E402

STORE_ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
AUDIT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/gt_audit_2026-09-08"
OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/motion_sam3"
METHOD = "motion_sam3"
K3 = np.ones((3, 3), np.uint8)


# ----------------------------------------------------------------------------------------------- calibration (declared)
def load_geometry(ep, serials):
    """(P, base): projection matrices K [R|t] (world = robot base -> camera) and the base-origin pixel per cam, from the
    declared calibration; (None, {}) when unavailable. base[c]["ok"] only when the projection is well conditioned."""
    try:
        intr = json.load(open(f"{AUDIT}/docs/hf_intrinsics.json"))[ep]
        cams = json.load(open(f"{AUDIT}/pointworld/droid/cameras/{ep}_cameras.json"))
        P, base = {}, {}
        for c, s in serials.items():
            fx, cx, fy, cy = intr[s]["cameraMatrix"]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
            E = np.array(cams[s]["optimized_extrinsics"], float)
            P[c] = K @ E[:3]
            Xc = E[:3, 3]  # robot base origin in camera coordinates
            W, H = intr[s].get("width", 1280), intr[s].get("height", 720)
            if Xc[2] >= 0.3:
                x, y = fx * Xc[0] / Xc[2] + cx, fy * Xc[1] / Xc[2] + cy
                base[c] = dict(x=float(x), y=float(y), depth=float(Xc[2]), ok=bool(-W <= x <= 2 * W and -H <= y <= 2 * H))
            else:
                base[c] = dict(x=float("nan"), y=float("nan"), depth=float(Xc[2]), ok=False)
        return P, base
    except Exception as e:  # noqa: BLE001
        print(f"[calib] unavailable for {ep}: {e!r}", flush=True)
        return None, {}


def triangulate(P1, P2, x1, x2):
    """x1, x2: (x, y) px. Returns X (3,) in the base frame and the mean reprojection residual in px."""
    Xh = cv2.triangulatePoints(P1, P2, np.array(x1, float).reshape(2, 1), np.array(x2, float).reshape(2, 1))
    X = (Xh[:3] / Xh[3]).ravel()
    r = []
    for P, x in ((P1, x1), (P2, x2)):
        p = P @ np.r_[X, 1.0]
        if p[2] <= 1e-6:
            return X, float("inf")
        r.append(np.hypot(p[0] / p[2] - x[0], p[1] / p[2] - x[1]))
    return X, float(np.mean(r))


# ----------------------------------------------------------------------------------------------- motion blob (per view)
def fill_holes(m):
    h, w = m.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    inv = (1 - m).astype(np.uint8)
    for (x, y) in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if inv[y, x] == 1:
            cv2.floodFill(inv, ff, (x, y), 2)
    return ((m == 1) | (inv == 1)).astype(np.uint8)


def geodesic(mask, seed_yx, max_iter=4000):
    dist = np.full(mask.shape, -1, np.int32)
    front = np.zeros(mask.shape, np.uint8)
    front[seed_yx] = 1
    d = 0
    while front.any() and d < max_iter:
        dist[front > 0] = d
        front = cv2.dilate(front, K3) & mask & (dist < 0).astype(np.uint8)
        d += 1
    return dist


def touches_border(ml, e_yx, radius=15, margin=6):
    h, w = ml.shape
    y, x = e_yx
    sub = ml[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)] > 0
    ys, xs = np.nonzero(sub)
    if len(ys) == 0:
        return False
    ys, xs = ys + max(0, y - radius), xs + max(0, x - radius)
    return bool(np.min(np.minimum(np.minimum(ys, h - 1 - ys), np.minimum(xs, w - 1 - xs))) <= margin)


def interior_point(mask_u8):
    """Deepest interior point (distance transform with a zero frame so the image border is not 'interior')."""
    if not mask_u8.any():
        return None
    m = mask_u8.copy()
    m[0, :] = m[-1, :] = 0
    m[:, 0] = m[:, -1] = 0
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return [int(x), int(y)] if dt[y, x] > 0 else None


def blob_stage(m, a):
    """m: uint8 (H,W) motion mask. Largest component, its geodesic extremities and border flags; None if too small."""
    H, W = m.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[k, cv2.CC_STAT_AREA])
    if area < a.min_blob or area > 0.4 * H * W:
        return None
    blob = (lab == k).astype(np.uint8)
    s = a.grid
    ml = (cv2.resize(blob * 255, (W // s, H // s), interpolation=cv2.INTER_AREA) > 100).astype(np.uint8)
    ys, xs = np.nonzero(ml)
    if len(ys) < 4:
        return None
    j = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
    d0 = geodesic(ml, (ys[j], xs[j]))
    e1 = np.unravel_index(int(np.argmax(d0)), d0.shape)
    d1 = geodesic(ml, e1)
    e2 = np.unravel_index(int(np.argmax(d1)), d1.shape)
    d2 = geodesic(ml, e2)
    px = lambda e: [int(e[1] * s + s // 2), int(e[0] * s + s // 2)]  # noqa: E731
    return dict(area=area, blob=np.packbits(blob, axis=1), ml=ml, d=(d1, d2), e_px=(px(e1), px(e2)), chain_len=float(d1[e2] * s),
                touch=(touches_border(ml, e1), touches_border(ml, e2)),
                bbox=[int(stats[k, 0]), int(stats[k, 1]), int(stats[k, 0] + stats[k, 2] - 1), int(stats[k, 1] + stats[k, 3] - 1)])


def cap_stage(st, dist_idx, a, H, W):
    """Distal cap around extremity dist_idx (0/1): box + clicks. None when the cap is too small."""
    s = a.grid
    dD = st["d"][dist_idx]
    L = float(np.clip(a.cap_frac * st["chain_len"], a.cap_min, a.cap_max))
    blob = np.unpackbits(st["blob"], axis=1)[:, :W]
    cap = cv2.resize(((dD >= 0) & (dD <= L / s)).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) & blob
    cap_area = int(cap.sum())
    if cap_area < a.min_cap:
        return None
    ys, xs = np.nonzero(cap)
    p = a.box_pad
    box = [int(max(0, xs.min() - p)), int(max(0, ys.min() - p)), int(min(W - 1, xs.max() + p)), int(min(H - 1, ys.max() + p))]
    tip = cv2.resize(((dD >= 0) & (dD <= L / (2 * s))).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) & cap
    clicks = []
    for c in (interior_point(cap), interior_point(tip), interior_point(cap & (1 - tip))):
        if c is not None and all(np.hypot(c[0] - q[0], c[1] - q[1]) >= 10 for q in clicks):
            clicks.append(c)
    if not clicks:
        return None
    return dict(box=box, clicks=clicks, cap_area=cap_area, blob_area=st["area"], chain_len=round(st["chain_len"], 1),
                dist=st["e_px"][dist_idx], prox=st["e_px"][1 - dist_idx], L=L)


def motion_stages(frames, a):
    """Causal per-frame blob stages: stage[t] depends on frames t-2, t-1, t only."""
    k_open = np.ones((5, 5), np.uint8)
    k_close = np.ones((a.close_k, a.close_k), np.uint8)
    stages = [None] * len(frames)
    prev1 = prev2 = None
    for t, f in enumerate(frames):
        g = cv2.GaussianBlur(f, (5, 5), 0)
        if t >= 2:
            d1 = cv2.absdiff(g, prev1).max(axis=2)
            d2 = cv2.absdiff(g, prev2).max(axis=2)
            if a.thresh_dark < a.thresh:  # dark-adaptive: a black gripper on a dark background moves with small absolute differences
                thr = np.clip(a.thresh_rel * (g.max(axis=2).astype(np.float32) + 25.0), a.thresh_dark, a.thresh)
            else:
                thr = a.thresh
            m = ((d1 > thr) & (d2 > thr)).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
            if m.sum() >= a.min_blob:
                stages[t] = blob_stage(fill_holes(m), a)
        prev2, prev1 = prev1, g
    return stages


def joint_decision(st, P, base, cams, a):
    """st: {cam: stage or None}. Decide which extremity is the gripper tip in each view and whether the frame may prompt.
    Per view: border rule (the extremity where the blob touches the image border is the arm side), else the calibration
    rule (the extremity farther from the projected base origin, when that projection is well conditioned). Undecided
    views take the end of the smallest-residual 3D pairing with the other view. The frame is valid only when the decided
    ends triangulate (declared calibration) with residual < r_max inside the robot's reach: the person filter."""
    c1, c2 = cams
    out = dict(valid=False, rule="none", distal={})
    s1, s2 = st.get(c1), st.get(c2)
    decided = {}
    for c, s in ((c1, s1), (c2, s2)):
        if s is None:
            continue
        t1, t2 = s["touch"]
        if t1 != t2:
            decided[c] = (1 if t1 else 0, "border")
        elif base.get(c, {}).get("ok"):
            b = np.array([base[c]["x"], base[c]["y"]])
            d = [np.linalg.norm(np.array(e) - b) for e in s["e_px"]]
            decided[c] = (int(np.argmax(d)), "calib")
    if s1 is None or s2 is None or P is None:
        for c, s in ((c1, s1), (c2, s2)):  # single-view fallback (no prompt unless the calibration is missing)
            if s is not None:
                out["distal"][c] = decided.get(c, ((1 if s["e_px"][0][1] <= s["e_px"][1][1] else 0), "higher"))
        out["valid"] = P is None and bool(out["distal"])
        return out
    hyps = []
    for i in range(2):
        for j in range(2):
            X, r = triangulate(P[c1], P[c2], s1["e_px"][i], s2["e_px"][j])
            dist = float(np.linalg.norm(X))
            ok = r < a.r_max and dist < a.reach and a.zmin < X[2] < a.zmax
            cons = all(decided.get(c) is None or decided[c][0] == idx for c, idx in ((c1, i), (c2, j)))
            hyps.append(dict(i=i, j=j, X=[round(float(v), 3) for v in X], r=round(r, 1), dist=round(dist, 3), ok=bool(ok), cons=bool(cons)))
    out["hyps"] = hyps
    cand = [h for h in hyps if h["ok"] and h["cons"]]
    if not cand:
        return out
    best = min(cand, key=lambda h: h["r"])
    if not decided and best["dist"] < a.tip_min_dist:  # nothing but 3D: a corresponding pair close to the base is not a tip
        return out
    rule = "+".join(sorted({decided[c][1] for c in decided})) + ("+3d" if len(decided) < 2 else "")
    out.update(valid=True, rule=rule, X=best["X"], dist=best["dist"], r=best["r"],
               distal={c1: (best["i"], decided.get(c1, (None, "3d"))[1]), c2: (best["j"], decided.get(c2, (None, "3d"))[1])})
    return out


def wrist_motion(ep, T):
    """Causal wrist-camera frame difference (store 320x180 rgb, top rows only: the fingers sit at the bottom)."""
    d = os.path.join(STORE_ROOT, ep, "dense", "rgb")
    w = np.zeros(T)
    prev = None
    for t in range(T):
        f = cv2.imread(os.path.join(d, f"{t:06d}.png"))
        if f is None:
            return None
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        if prev is not None:
            w[t] = float(np.abs(g - prev)[:int(g.shape[0] * 0.83)].mean())
        prev = g
    return w


# ----------------------------------------------------------------------------------------------- SAM3 tracking
def mask_sane(m, box, a):
    H, W = m.shape
    area = int(m.sum())
    if area < a.min_mask or area > a.max_mask_frac * H * W:
        return False, f"area {area}"
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean(), ys.mean()
    x0, y0, x1, y1 = box
    if not (x0 - 20 <= cx <= x1 + 20 and y0 - 20 <= cy <= y1 + 20):
        return False, f"centroid ({cx:.0f},{cy:.0f}) outside box {box}"
    ix = max(0, min(x1, xs.max()) - max(x0, xs.min())); iy = max(0, min(y1, ys.max()) - max(y0, ys.min()))
    inter = ix * iy; ua = (x1 - x0) * (y1 - y0) + (xs.max() - xs.min()) * (ys.max() - ys.min()) - inter
    iou = inter / max(ua, 1)
    if iou < a.min_box_iou:
        return False, f"bbox iou {iou:.2f}"
    return True, f"area {area} iou {iou:.2f}"


def valid_track_mask(m, a):
    area = int(m.sum())
    return a.min_mask <= area <= a.max_mask_frac * m.size


def mask_motion(m_prev, m):
    """(centroid shift px, IoU) between two boolean masks."""
    yp, xp = np.nonzero(m_prev); y, x = np.nonzero(m)
    shift = float(np.hypot(x.mean() - xp.mean(), y.mean() - yp.mean()))
    iou = float((m_prev & m).sum() / max((m_prev | m).sum(), 1))
    return shift, iou


def track_camera(tracker, video_path, frames, cands, w, blobs, a, log):
    """Forward-only, one frame at a time, with re-prompting and the online wrist-consistency drop."""
    import torch
    T = len(frames)
    H, W = frames[0].shape[:2]
    t0 = time.time()
    state = tracker.init_state(video_path=video_path)
    n = state["num_frames"]
    log(f"  tracker state: {n} frames loaded in {time.time() - t0:.0f}s (mp4 {T} frames)")
    assert n >= T, (n, T)
    info = dict(prompts=[], rejected=[], drops=[], n_reprompts=0, t_load=time.time() - t0)
    masks, scores = {t: None for t in range(T)}, {t: float("nan") for t in range(T)}

    def try_prompt(t, cand, kind):
        pts = torch.tensor([[x / W, y / H] for x, y in cand["clicks"]], dtype=torch.float32)
        labels = torch.ones(len(cand["clicks"]), dtype=torch.int32)
        x0, y0, x1, y1 = cand["box"]
        box = np.array([[x0 / W, y0 / H, x1 / W, y1 / H]], dtype=np.float32)
        _, _, _, vmasks = tracker.add_new_points_or_box(inference_state=state, frame_idx=int(t), obj_id=1, points=pts, labels=labels, box=box)
        m = (vmasks[0][0] > 0).cpu().numpy()
        ok, why = mask_sane(m, cand["box"], a)
        rec = dict(frame=int(t), kind=kind, accepted=bool(ok), why=why, mask_area=int(m.sum()), **{k: v for k, v in cand.items() if k != "L"})
        if ok:
            info["prompts"].append(rec)
            log(f"  prompt ({kind}) at f{t} accepted: box {cand['box']} clicks {cand['clicks']} rule={cand['rule']} cap={cand['cap_area']} mask={int(m.sum())}")
        else:
            tracker.clear_all_points_in_frame(state, int(t), 1, need_output=False)
            info["rejected"].append(rec)
            log(f"  prompt ({kind}) at f{t} REJECTED ({why})")
        return ok

    t_tr = time.time()
    active, need_preflight = False, False
    n_lost, last_try, last_prompt = a.lost_frames + 1, -10 ** 9, -10 ** 9
    votes = collections.deque(maxlen=12)
    for t in range(T):
        c = cands[t]
        if c is not None and t - last_try >= a.retry_gap and t - last_prompt >= a.reprompt_gap and (not active or n_lost > a.lost_frames):
            is_re = bool(info["prompts"])
            budget_ok = (info["n_reprompts"] < a.max_reprompts) if is_re else (len(info["rejected"]) < a.max_attempts)
            if budget_ok:
                last_try = t
                if try_prompt(t, c, "reprompt" if is_re else "initial"):
                    if is_re:
                        info["n_reprompts"] += 1
                    active, need_preflight, last_prompt, n_lost = True, True, t, 0
                    votes.clear()
        if not active:
            continue
        got = None
        for fidx, _, _, vmasks, obj_scores in tracker.propagate_in_video(state, start_frame_idx=int(t), max_frame_num_to_track=0, reverse=False,
                                                                          propagate_preflight=need_preflight, tqdm_disable=True):
            assert int(fidx) == t, (fidx, t)
            got = (vmasks[0][0] > 0).cpu().numpy()
            try:
                scores[t] = float(obj_scores.reshape(-1)[0].item())
            except Exception:  # noqa: BLE001
                pass
        need_preflight = False
        assert got is not None
        if valid_track_mask(got, a):
            masks[t] = got
            # online consistency vote against the wrist-motion signal (see the module docstring, step 7)
            if w is not None and masks.get(t - 1) is not None:
                shift, iou = mask_motion(masks[t - 1], got)
                if w[t] < a.w_still:  # arm still: a mask that travels is not the gripper
                    votes.append(1 if shift > a.mask_shift else 0)
                elif w[t] > a.w_moving and blobs[t] is not None and blobs[t][1] >= a.min_blob_prompt:
                    # arm moving AND this view shows a large moving blob: a static mask that is not part of it is not the gripper
                    blob = np.unpackbits(blobs[t][0], axis=1)[:, :W].astype(bool)
                    overlap = (got & blob).sum() / max(got.sum(), 1)
                    votes.append(1 if (shift < a.mask_static and overlap < a.mask_blob_overlap) else 0)
            n_lost = 0
        else:
            n_lost += 1
        if len(votes) >= 10 and sum(votes) >= a.drop_votes:
            info["drops"].append(dict(frame=int(t), disagreements=int(sum(votes)), n_votes=len(votes)))
            log(f"  DROP track at f{t}: {sum(votes)}/{len(votes)} wrist-consistency disagreements; resetting tracker state")
            tracker._reset_tracking_results(state)
            active, n_lost = False, a.lost_frames + 1
            votes.clear()
    info["t_track"] = time.time() - t_tr
    info["prompt_frame"] = info["prompts"][0]["frame"] if info["prompts"] else None
    n_tracked = T - info["prompt_frame"] if info["prompt_frame"] is not None else 0
    info["fps_track"] = n_tracked / max(info["t_track"], 1e-6)
    if not info["prompts"]:
        log("  NO PROMPT ACCEPTED: all frames get empty masks")
    return masks, scores, info, state


# ----------------------------------------------------------------------------------------------- outputs
def to_out(m, score):
    if m is None or not m.any():
        return dict(ids=np.zeros(0, int), probs=np.zeros(0), boxes=np.zeros((0, 4)), masks=np.zeros((0, 1, 1), bool))
    p = 1.0 / (1.0 + np.exp(-score)) if np.isfinite(score) else 1.0
    return dict(ids=np.array([1]), probs=np.array([p]), boxes=np.zeros((1, 4)), masks=m[None])


def draw_prompt(img, cand, color=(0, 255, 255)):
    x0, y0, x1, y1 = cand["box"]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    for x, y in cand["clicks"]:
        cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.circle(img, (int(x), int(y)), 6, (0, 0, 0), 1)
    if "prox" in cand:
        cv2.circle(img, tuple(int(v) for v in cand["prox"]), 8, (255, 0, 0), 2)
        cv2.circle(img, tuple(int(v) for v in cand["dist"]), 8, (0, 0, 255), 2)
    return img


def save_masks(path, masks, T, H, W):
    union = np.zeros((T, H, W), bool)
    for t in range(T):
        if masks.get(t) is not None:
            union[t] = masks[t]
    present = union.any(axis=(1, 2))
    np.savez_compressed(path, union=np.packbits(union.astype(np.uint8), axis=2), shape=np.array([T, H, W]), frames_present=present)
    return union, present


def check_product(path, T_store):
    z = np.load(path)
    T, H, W = [int(v) for v in z["shape"]]
    u = np.unpackbits(z["union"], axis=2)[:, :, :W].astype(bool)
    assert u.shape == (T, H, W) == (T_store, 720, 1280), (u.shape, T_store)
    fp = z["frames_present"]
    assert fp.shape == (T,) and np.array_equal(fp, u.any(axis=(1, 2))), "frames_present inconsistent with union"
    return dict(T=T, H=H, W=W, n_present=int(fp.sum()), first=int(np.argmax(fp)) if fp.any() else None,
                last=int(T - 1 - np.argmax(fp[::-1])) if fp.any() else None, mean_area_frac=float(u[fp].mean()) if fp.any() else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="RAIL+80edfcb1+2023-07-14-14h-28m-45s")
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--cams", default="ext1,ext2")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--overlay", type=int, default=1, help="write per-cam and side-by-side overlay mp4s")
    ap.add_argument("--cue_only", action="store_true", help="CPU only: motion cue + candidate prompts, no SAM")
    ap.add_argument("--no_wrist", action="store_true", help="do not use the wrist camera's motion signal (gate + consistency)")
    # motion cue
    ap.add_argument("--thresh", type=int, default=30, help="per-pixel difference threshold (0-255, max over BGR)")
    ap.add_argument("--thresh_dark", type=int, default=12, help="lower bound of the dark-adaptive threshold; == thresh disables it")
    ap.add_argument("--thresh_rel", type=float, default=0.25, help="dark-adaptive threshold = clip(thresh_rel * (I + 25), thresh_dark, thresh)")
    ap.add_argument("--close_k", type=int, default=25)
    ap.add_argument("--grid", type=int, default=4, help="downsampling factor for the geodesic computation")
    ap.add_argument("--min_blob", type=int, default=4000, help="min moving-blob area (px at 1280x720)")
    ap.add_argument("--min_blob_prompt", type=int, default=10000, help="min moving-blob area for a view to take part in a prompt (px)")
    ap.add_argument("--min_cap", type=int, default=3000, help="min distal-cap area to allow a prompt (px)")
    ap.add_argument("--cap_frac", type=float, default=0.30)
    ap.add_argument("--cap_min", type=float, default=120.0)
    ap.add_argument("--cap_max", type=float, default=260.0)
    ap.add_argument("--box_pad", type=int, default=8)
    # 3D pairing test (declared calibration)
    ap.add_argument("--r_max", type=float, default=40.0, help="max reprojection residual of an extremity pairing (px)")
    ap.add_argument("--reach", type=float, default=1.1, help="max distance of the tip from the base (m)")
    ap.add_argument("--zmin", type=float, default=-0.4)
    ap.add_argument("--zmax", type=float, default=1.3)
    ap.add_argument("--tip_min_dist", type=float, default=0.4, help="3D-only decisions need the tip at least this far from the base (m)")
    ap.add_argument("--persist_gap", type=int, default=3, help="a prompt needs a valid pairing within this many frames before it")
    ap.add_argument("--persist_m", type=float, default=0.15, help="... whose tip moved less than this per frame (m)")
    # wrist gate / consistency
    ap.add_argument("--w_gate", type=float, default=4.0, help="prompt only when the wrist frame difference exceeds this")
    ap.add_argument("--w_moving", type=float, default=6.0)
    ap.add_argument("--w_still", type=float, default=2.0)
    ap.add_argument("--mask_shift", type=float, default=4.0, help="centroid shift (px) above which a mask counts as travelling")
    ap.add_argument("--mask_static", type=float, default=1.5, help="centroid shift (px) below which a mask counts as static")
    ap.add_argument("--mask_blob_overlap", type=float, default=0.05, help="mask fraction inside the moving blob below which it is not part of it")
    ap.add_argument("--drop_votes", type=int, default=7)
    # SAM acceptance / tracking
    ap.add_argument("--min_mask", type=int, default=800)
    ap.add_argument("--max_mask_frac", type=float, default=0.12)
    ap.add_argument("--min_box_iou", type=float, default=0.25)
    ap.add_argument("--retry_gap", type=int, default=3)
    ap.add_argument("--max_attempts", type=int, default=12)
    ap.add_argument("--lost_frames", type=int, default=5, help="re-prompt after MORE than this many consecutive lost frames")
    ap.add_argument("--reprompt_gap", type=int, default=8)
    ap.add_argument("--max_reprompts", type=int, default=30)
    a = ap.parse_args()

    ep = a.episode
    raw = os.path.join(RAW_ROOT, ep)
    meta = json.load(open(os.path.join(raw, f"metadata_{ep}.json")))
    cams = [c for c in a.cams.split(",") if c]
    serials = {c: meta[f"{c}_cam_serial"] for c in cams}
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
    P, base = load_geometry(ep, serials)
    w = None if a.no_wrist else wrist_motion(ep, T_store)
    if w is not None:
        log(f"wrist motion signal: quantiles 5/50/95 = {np.percentile(w[1:], [5, 50, 95]).round(2).tolist()}, frames > gate {int((w > a.w_gate).sum())}/{T_store}")
    frames, stages, H, W = {}, {}, 720, 1280
    for c in cams:
        t0 = time.time()
        fr = read_frames(videos[c])
        if len(fr) < T_store:
            sys.exit(f"{c}: mp4 has {len(fr)} frames < store {T_store}")
        if len(fr) > T_store:
            log(f"  {c}: mp4 has {len(fr)} frames > store {T_store}; truncating to the store count")
            fr = fr[:T_store]
        frames[c] = fr
        H, W = fr[0].shape[:2]
        t1 = time.time()
        stages[c] = motion_stages(fr, a)
        log(f"{c} ({serials[c]}): {len(fr)} frames {W}x{H}, read {t1 - t0:.0f}s, motion blobs {time.time() - t1:.0f}s; "
            f"frames with a blob: {sum(s is not None for s in stages[c])}/{len(fr)}")
    # joint 3D decision + per-view candidates
    t0 = time.time()
    cands = {c: [None] * T_store for c in cams}
    joint = [None] * T_store
    n_valid, n_gated = 0, 0
    last_valid = None  # (t, X) of the previous valid frame, for the persistence test
    for t in range(T_store):
        jd = joint_decision({c: (stages[c][t] if stages[c][t] is not None and stages[c][t]["area"] >= a.min_blob_prompt else None) for c in cams},
                            P, base, cams, a)
        joint[t] = jd
        if not jd["valid"]:
            continue
        n_valid += 1
        persist = True
        if P is not None:
            persist = last_valid is not None and t - last_valid[0] <= a.persist_gap and \
                np.linalg.norm(np.array(jd["X"]) - np.array(last_valid[1])) < a.persist_m * (t - last_valid[0])
            last_valid = (t, jd["X"])
        if not persist or (w is not None and w[t] <= a.w_gate):
            continue
        n_gated += 1
        for c in cams:
            idx, rule = jd["distal"][c]
            cp = cap_stage(stages[c][t], idx, a, H, W)
            if cp is not None:
                cp.update(rule=rule, X=jd.get("X"), dist_base=jd.get("dist"), resid=jd.get("r"))
                cands[c][t] = cp
    blobs = {c: [(st["blob"], st["area"]) if st is not None else None for st in stages[c]] for c in cams}  # packed, for the consistency vote
    for c in cams:
        for st in stages[c]:
            if st is not None:
                st.pop("blob", None)
    firsts = {c: next((t for t in range(T_store) if cands[c][t] is not None), None) for c in cams}
    log(f"joint 3D decision {time.time() - t0:.0f}s: frames with a valid pairing {n_valid}/{T_store}, after persistence + wrist gate {n_gated}; "
        f"first candidates {firsts}; calibration {'ok' if P is not None else 'MISSING (single-view rules only)'}; base px {base}")
    for c in cams:
        if firsts[c] is not None:
            log(f"  {c} first candidate: {json.dumps(cands[c][firsts[c]])}")

    if a.cue_only:
        for c in cams:
            for t in [firsts[c]] + [t for t in range(T_store) if cands[c][t] is not None][::25]:
                if t is None:
                    continue
                x = cands[c][t]
                img = draw_prompt(frames[c][t].copy(), x)
                cv2.putText(img, f"{c} f{t} rule={x['rule']} cap={x['cap_area']} chain={x['chain_len']:.0f} dist={x['dist_base']} r={x['resid']}", (10, 40), 0, 0.9, (0, 255, 255), 2)
                cv2.imwrite(os.path.join(out_dir, "look", f"cue_{c}_f{t:04d}.png"), cv2.resize(img, (640, 360)))
        json.dump(dict(episode=ep, cue_only=True, n_valid=n_valid, n_gated=n_gated, firsts=firsts,
                       cands={c: {t: x for t, x in enumerate(cands[c]) if x} for c in cams}), open(os.path.join(out_dir, "cue_info.json"), "w"), indent=1)
        log("CUE_ONLY done")
        return

    import torch
    from sam3.model_builder import build_sam3_video_model
    t0 = time.time()
    model = build_sam3_video_model()
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    log(f"tracker built in {time.time() - t0:.0f}s; gpu {torch.cuda.get_device_name(0)}")

    seed_info = dict(episode=ep, method=METHOD, serials=serials, T=T_store, args=vars(a), cams={}, causal=True, uses_gt=False,
                     calibration_use="declared exterior calibration for the 3D pairing test (distal end + person filter)" if P is not None else "none (missing)",
                     n_joint_valid=n_valid, n_joint_gated=n_gated, wrist_gate=w is not None)
    outputs_all, label = {}, {}
    for c in cams:
        log(f"\n=== {c} ({serials[c]}) ===")
        masks, scores, info, state = track_camera(tracker, videos[c], frames[c], cands[c], w, blobs[c], a, log)
        del state
        torch.cuda.empty_cache()
        t0 = time.time()
        base = os.path.join(out_dir, f"{c}_{serials[c]}__{METHOD}")
        union, present = save_masks(base + "_masks.npz", masks, T_store, H, W)
        with open(base + "_frames.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "present", "area_frac", "obj_score_logit", "cand_cap_area", "cand_rule", "wrist_motion"])
            for t in range(T_store):
                cd = cands[c][t]
                wr.writerow([t, int(present[t]), f"{union[t].mean():.5f}", f"{scores.get(t, float('nan')):.3f}",
                             cd["cap_area"] if cd else 0, cd["rule"] if cd else "", f"{w[t]:.2f}" if w is not None else ""])
        chk = check_product(base + "_masks.npz", T_store)
        info.update(chk, t_write=time.time() - t0, n_candidates=sum(x is not None for x in cands[c]))
        seed_info["cams"][c] = info
        log(f"  wrote {base}_masks.npz: {chk}")
        outs = {t: to_out(masks.get(t), scores.get(t, float("nan"))) for t in range(T_store)}
        outputs_all[c], label[c] = outs, METHOD
        pf = [p["frame"] for p in info["prompts"]]
        pres = np.nonzero(present)[0]
        picks = list(pf[:3])
        if len(pres):
            picks += [int(pres[len(pres) // 2]), int(pres[-1])]
        for t in dict.fromkeys(picks):
            img = render(frames[c][t], outs[t], f"{c}/{serials[c]} {METHOD} f{t}/{T_store - 1} {'PROMPT' if t in pf else ''}")
            cd = next((p for p in info["prompts"] if p["frame"] == t), None)
            if cd is not None:
                draw_prompt(img, cd)
            cv2.imwrite(os.path.join(out_dir, "look", f"{c}_f{t:04d}.png"), cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
        if a.overlay:
            t0 = time.time()
            vw = FfmpegWriter(base + ".mp4", W, H, a.fps)
            for t in range(T_store):
                img = render(frames[c][t], outs[t], f"{c}/{serials[c]} {METHOD} f{t}/{T_store - 1} present={int(present[t])} w={w[t]:.1f}" if w is not None else
                             f"{c}/{serials[c]} {METHOD} f{t}/{T_store - 1} present={int(present[t])}")
                cd = cands[c][t]
                if cd is not None:
                    x0, y0, x1, y1 = cd["box"]
                    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 255), 1)
                p = next((p for p in info["prompts"] if p["frame"] == t), None)
                if p is not None:
                    draw_prompt(img, p)
                vw.write(img)
            vw.close()
            info["t_overlay"] = time.time() - t0
        json.dump(seed_info, open(os.path.join(out_dir, "seed_info.json"), "w"), indent=1, default=str)
    if a.overlay and len(cams) > 1:
        side_by_side(cams, frames, outputs_all, label, os.path.join(out_dir, f"{METHOD}_side_by_side.mp4"), a.fps)
    seed_info["t_total"] = time.time() - t_all
    json.dump(seed_info, open(os.path.join(out_dir, "seed_info.json"), "w"), indent=1, default=str)
    log("\nSUMMARY")
    for c in cams:
        i = seed_info["cams"][c]
        log(f"  {c}: prompt f{i.get('prompt_frame')} reprompts={i.get('n_reprompts')} rejected={len(i.get('rejected', []))} drops={len(i.get('drops', []))} "
            f"present {i.get('n_present')}/{T_store} (first {i.get('first')}, last {i.get('last')}) mean area {i.get('mean_area_frac', 0):.4f} "
            f"track {i.get('t_track', 0):.0f}s ({i.get('fps_track', 0):.1f} fps)")
    log(f"total {seed_info['t_total']:.0f}s\nDONE {out_dir}")


if __name__ == "__main__":
    main()
