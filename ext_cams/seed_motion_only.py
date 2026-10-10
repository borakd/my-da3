#!/usr/bin/env python
"""seed_motion_only: gripper seed masks for the two STATIC exterior cameras of one DROID episode, from motion alone.
No SAM, no learned model, CPU only, causal by construction (frame t uses frames <= t; nothing is propagated backward).

Per camera and frame (processing resolution 640x360, output upsampled to the native 1280x720):
  1. background: OpenCV MOG2 (per-pixel Gaussian mixture) updated online with a FIXED learning rate 1/history after a
     short warm-up (OpenCV's default 1/min(t, history) absorbs a paused arm within a few frames early in an episode).
     Foreground = pixels off the background model; shadow-labelled pixels (darker than the background by a factor in
     [0.3, 1], same chroma) are dropped unless they are dark AND achromatic in absolute terms (the black hand over a
     grey background is otherwise labelled shadow and lost; a cast shadow on wood keeps the wood's chroma).
  2. clean: median blur, morphological close + open, drop components below a minimum area.
  3. arm component: the moving component that touches the arm's ENTRY SIDE of the image border (large enough, or
     overlapping the previous gripper mask; the one overlapping it most wins, else the largest). Only when no such
     component exists is a moving component overlapping the previous gripper mask taken (temporal continuity: the
     arm paused, MOG2 absorbed all but the moving wrist). The entry side starts from a prior (`--entry_prior top`: the ZEDs sit 0.4-0.6 m from the base at table
     height, the upper arm always leaves the frame at the top) and is overridden causally when the accumulated border
     contact of motion-selected arm components says otherwise (`--entry_prior auto` = data only). Components not linked
     to the arm this way (people walking through, background flicker) are rejected.
  4. distal part: geodesic distance INSIDE the arm component from its border contacts (multi-source, all sides by
     default since the upper arm may leave the frame through two borders; the gripper's own contact, if it touches the
     border, is excluded) = distance along the arm from where it enters. Tip = farthest pixel
     among the pixels that moved between t-1 and t (a MOG2 ghost of an earlier arm pose, or a drawer or cloth the arm
     just moved, does not move any more, so it cannot be the tip), with a jump limit to the previous tip. Gripper mask = the tip's connected part of the pixels whose
     geodesic distance lies within CAP of the tip; CAP = 0.25 m converted to pixels with a plausible depth = mean of
     (median depth of a workspace grid in front of the base) and (camera-to-base distance), from the PointWorld
     extrinsics and factory intrinsics: calibration only, no GT.
  5. hold: when nothing is detected but the previous gripper mask is still static against the image at the start of
     the hold (no accumulated change under it), the previous mask is carried forward (the arm paused and MOG2
     absorbed it, or only a static fragment of it is still seen).

Ground truth is not used anywhere in this file. The store is read only to check the frame count (product check).

Writes, per the mask file contract, <out_root>/<EP>/<cam>_<serial>__motion_only_masks.npz with
  union: uint8 (T, 720, 160) = np.packbits(mask, axis=2); shape: [T, 720, 1280]; frames_present: bool (T,)
plus <EP>/seed_info.json (parameters, per-camera stats, entry side, per-frame log, timing) and optionally an overlay mp4
(--video) and check PNGs (--png_frames).

    OMP_NUM_THREADS=1 python seed_motion_only.py --episode EP --out_root .../seedstudy/motion_only --video
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
OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/motion_only"
METHOD = "motion_only"
NATIVE_W, NATIVE_H = 1280, 720
SIDES = ("top", "bottom", "left", "right")


# ----------------------------------------------------------------------------------------------- calibration (scale only)
def plausible_depth(ep, serial):
    """Plausible depth (m) of the gripper in this exterior camera = mean of the median depth of a 3x3x3 workspace grid
    in front of the base and the camera-to-base distance. Calibration only (PointWorld world-to-camera extrinsics,
    factory intrinsics); no GT. Returns (depth, fx_native, details)."""
    intr = json.load(open(f"{AUDIT}/docs/hf_intrinsics.json"))[ep][serial]
    E = np.array(json.load(open(f"{AUDIT}/pointworld/droid/cameras/{ep}_cameras.json"))[serial]["optimized_extrinsics"])
    grid = np.array([[x, y, z, 1.0] for x in (0.3, 0.5, 0.7) for y in (-0.3, 0, 0.3) for z in (0.1, 0.3, 0.5)])
    z_grid = float(np.median((E @ grid.T)[2]))
    cam_c = -E[:3, :3].T @ E[:3, 3]
    d_base = float(np.linalg.norm(cam_c))
    fx = float(intr["cameraMatrix"][0])
    return 0.5 * (z_grid + d_base), fx, dict(z_grid_median=round(z_grid, 3), cam_to_base=round(d_base, 3))


# ----------------------------------------------------------------------------------------------- helpers
def border_contacts(comp):
    """Per-side count of component pixels on the image border."""
    return dict(top=int(comp[0].sum()), bottom=int(comp[-1].sum()), left=int(comp[:, 0].sum()), right=int(comp[:, -1].sum()))


def border_pixels(comp, exclude=None):
    """(r, c) list of component pixels lying on the image border, minus those inside `exclude`."""
    edge = np.zeros_like(comp)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    sel = comp & edge
    if exclude is not None:
        sel &= ~exclude
    return np.argwhere(sel)


def geodesic(comp, starts):
    """Geodesic distance (8-connected, diagonal = sqrt2) inside the boolean component from the start pixels; inf elsewhere.
    Computed on the component's bounding box for speed."""
    rs, cs = np.nonzero(comp)
    r0, r1, c0, c1 = rs.min(), rs.max() + 1, cs.min(), cs.max() + 1
    sub = comp[r0:r1, c0:c1]
    cost = np.where(sub, 1.0, np.inf)
    mcp = MCP_Geometric(cost, fully_connected=True)
    st = [(int(r - r0), int(c - c0)) for r, c in starts if sub[r - r0, c - c0]]
    if not st:
        return None
    cum, _ = mcp.find_costs(st)
    out = np.full(comp.shape, np.inf)
    out[r0:r1, c0:c1] = cum
    return out


def disk(shape, center, radius):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2


def component_containing(mask, pt):
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    k = lab[pt[0], pt[1]]
    if k == 0:
        return mask
    return lab == k


def fill_small_holes(mask, max_hole_px):
    """Fill background holes enclosed by the mask that are smaller than max_hole_px."""
    inv = (~mask).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    H, W = mask.shape
    out = mask.copy()
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < max_hole_px and x > 0 and y > 0 and x + w < W and y + h < H:  # enclosed (does not reach the border)
            out[lab == k] = True
    return out


def argmax_geo(sel, geo):
    return tuple(int(v) for v in np.unravel_index(np.argmax(np.where(sel, geo, -1.0)), geo.shape))


# ----------------------------------------------------------------------------------------------- the seeder
class MotionOnlySeeder:
    def __init__(self, W, H, fx_proc, depth, a):
        self.W, self.H = W, H
        self.a = a
        self.px_per_m = fx_proc / max(depth, a.min_depth)
        self.cap_px = a.cap_m * self.px_per_m
        self.jump_px = a.jump_m * self.px_per_m
        self.mog = cv2.createBackgroundSubtractorMOG2(history=a.history, varThreshold=a.var_thresh, detectShadows=not a.no_shadows)
        self.mog.setShadowThreshold(a.shadow_tau)
        self.min_area = a.min_area_frac * W * H
        self.min_area_first = a.min_area_first_frac * W * H
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.close_px, a.close_px))
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.open_px, a.open_px))
        self.k_move = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.move_dilate_px, a.move_dilate_px))
        self.prev_gray = None
        self.prev_grip = None
        self.prev_tip = None
        self.hold_ref = None  # gray image at the start of the current hold
        self.hold_len = 0
        self.stale = 0
        self.side_acc = dict((s, 0) for s in SIDES)
        self.n_side_frames = 0
        self.entry_side = None if a.entry_prior == "auto" else a.entry_prior
        self.entry_from_prior = self.entry_side is not None
        self.t = 0

    # ---- helpers
    def _static_under(self, mask, gray):
        """Fraction of mask pixels whose gray level changed by more than move_thresh since the hold reference."""
        ref = self.hold_ref if self.hold_ref is not None else self.prev_gray
        if ref is None or not mask.any():
            return 1.0
        return float((cv2.absdiff(gray, ref) > self.a.move_thresh)[mask].mean())

    def _update_entry_side(self, comp):
        bc = border_contacts(comp)
        for s in SIDES:
            self.side_acc[s] += bc[s]
        self.n_side_frames += 1
        if self.a.entry_prior == "auto":
            self.entry_side = max(SIDES, key=lambda s: self.side_acc[s])
        elif self.n_side_frames >= self.a.entry_override_frames:
            best = max(SIDES, key=lambda s: self.side_acc[s])
            if best != self.entry_side and self.side_acc[best] > self.a.entry_override_ratio * max(self.side_acc[self.entry_side], 1):
                self.entry_side, self.entry_from_prior = best, False

    def _finish(self, grip, arm, info, gray):
        if grip is None:
            self.prev_grip, self.prev_tip, self.hold_ref, self.hold_len = None, None, None, 0
            info["mode"] = "none"
        else:
            self.prev_grip = grip
            info["grip_area"] = int(grip.sum())
        info["entry"] = self.entry_side
        self.prev_gray = gray
        self.t += 1
        return grip, arm, info

    # ---- per-frame
    def step(self, bgr):
        a = self.a
        gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        info = dict(t=self.t, mode="none", n_comp=0, n_rej=0, comp_area=0, grip_area=0, tip=None, geo_tip=None, entry=self.entry_side)
        if self.t < a.warmup_frames:  # background model warm-up: no detection
            self.mog.apply(bgr, learningRate=a.warmup_lr)
            info["mode"] = "warmup"
            return self._finish(None, None, info, gray)
        fg = self.mog.apply(bgr, learningRate=1.0 / a.history)  # 255 = foreground, 127 = shadow, 0 = background
        # shadow-labelled pixels are dropped unless dark AND achromatic (the black hand); a cast shadow on a wooden
        # table keeps the wood's chroma
        chroma = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
        fg = ((fg == 255) | ((fg == 127) & (gray < a.dark_thresh) & (chroma < a.chroma_thresh))).astype(np.uint8)
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_open)
        moving = cv2.dilate((cv2.absdiff(gray, self.prev_gray) > a.move_thresh).astype(np.uint8), self.k_move).astype(bool)

        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        comps = [(k, int(stats[k, cv2.CC_STAT_AREA])) for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= self.min_area]
        comps.sort(key=lambda kv: -kv[1])
        info["n_comp"] = len(comps)
        prev_area = int(self.prev_grip.sum()) if self.prev_grip is not None else 0

        # ---- arm component selection.
        # 1. ARM: a moving component touching the entry side (large enough, or overlapping the previous gripper mask).
        #    Preferred over continuity so that a released object (a cloth on the table) does not keep the label.
        # 2. FRAGMENT: no arm found; a component overlapping the previous gripper mask. Accepted as the arm only when it
        #    is at least fragment_frac of the previous mask and moving; smaller or static fragments go to the hold logic.
        chosen, roots_mode = None, "entry"
        cands = []
        for k, area in comps:
            comp = lab == k
            if not (comp & moving).any():
                continue
            bc = border_contacts(comp)
            touches = [sd for sd in SIDES if bc[sd] > 0]
            on_entry = (self.entry_side in touches) if self.entry_side is not None else bool(touches)
            ov = int((comp & self.prev_grip).sum()) if self.prev_grip is not None else 0
            if on_entry and (area >= self.min_area_first or ov >= a.continuity_frac * prev_area > 0):
                cands.append((ov, area, k))
        if cands:
            cands.sort(reverse=True)
            chosen, info["mode"] = cands[0][2], "arm"
        elif self.prev_grip is not None:
            best, best_ov = None, 0
            for k, area in comps:
                ov = int(((lab == k) & self.prev_grip).sum())
                if ov > best_ov:
                    best, best_ov = k, ov
            if best is not None and best_ov >= a.continuity_frac * prev_area:
                comp = lab == best
                if comp.sum() >= a.fragment_frac * prev_area and (comp & moving).any():
                    chosen, info["mode"], roots_mode = best, "continuity", "nearest"
                else:
                    info["fragment"] = int(comp.sum())
        info["n_rej"] = len(comps) - (1 if chosen is not None else 0)

        grip, arm = None, None
        if chosen is not None:
            comp = lab == chosen
            arm = comp
            info["comp_area"] = int(comp.sum())
            # roots = where the arm enters: entry-side border contacts (minus the previous gripper's own contact),
            # else any border contact, else the pixel nearest the entry-side border line
            excl = None if self.prev_grip is None else cv2.dilate(self.prev_grip.astype(np.uint8), self.k_close).astype(bool)
            roots = np.zeros((0, 2), int)
            if roots_mode == "entry":
                side = self.entry_side or "top"
                roots = border_pixels(comp, excl)
                if len(roots) and a.roots == "entry":
                    on_side = dict(top=roots[:, 0] == 0, bottom=roots[:, 0] == self.H - 1, left=roots[:, 1] == 0, right=roots[:, 1] == self.W - 1)[side]
                    roots = roots[on_side] if on_side.any() else roots
            if len(roots) == 0:
                rs, cs = np.nonzero(comp)
                side = self.entry_side or "top"
                d = dict(top=rs, bottom=self.H - 1 - rs, left=cs, right=self.W - 1 - cs)[side]
                i = int(np.argmin(d))
                roots = np.array([[rs[i], cs[i]]])
            geo = geodesic(comp, roots)
            if geo is not None:
                fin = comp & np.isfinite(geo)
                cand = fin & moving
                mode_tip = "motion"
                if self.prev_tip is not None and cand.any():
                    near = cand & disk(comp.shape, self.prev_tip, self.jump_px)
                    gl = argmax_geo(cand, geo)
                    if near.any():
                        far_jump = np.hypot(gl[0] - self.prev_tip[0], gl[1] - self.prev_tip[1]) > self.jump_px
                        self.stale = self.stale + 1 if far_jump else 0
                        if self.stale > a.stale_frames:  # the jump-limited tip has been contradicted for a while: re-acquire
                            self.stale, mode_tip = 0, "motion_reacq"
                        else:
                            cand, mode_tip = near, "motion_near"
                if cand.any():
                    tip = argmax_geo(cand, geo)
                elif self.prev_tip is not None:  # arm paused: keep the tip where it was, snapped onto the component
                    rs, cs = np.nonzero(fin)
                    i = int(np.argmin((rs - self.prev_tip[0]) ** 2 + (cs - self.prev_tip[1]) ** 2))
                    tip, mode_tip = (int(rs[i]), int(cs[i])), "hold_tip"
                else:
                    tip = None
                if tip is not None:
                    gt_ = float(geo[tip])
                    L = min(self.cap_px, a.max_cap_frac * gt_) if gt_ > 0 else self.cap_px
                    # distal part = pixels within L of the tip ALONG the component (geodesic from the tip, not from the
                    # root: side lobes of a wide elbow are as far from the root as the fingertip)
                    geo_tip = geodesic(fin, np.array([tip]))
                    g = fill_small_holes(fin & (geo_tip <= L), a.max_hole_px)
                    if g.sum() >= a.min_grip_px:
                        grip = g
                        if mode_tip.startswith("motion") and info["mode"] == "arm":
                            self._update_entry_side(comp)
                        self.prev_tip = tip
                        info.update(tip=[tip[1], tip[0]], geo_tip=round(gt_, 1), mode=info["mode"] + "/" + mode_tip, cap_px=round(L, 1))
        # ---- hold logic against the reference image at the start of the hold (no slow drift)
        if self.prev_grip is not None:
            if grip is None and self.hold_len < a.hold_max and self._static_under(self.prev_grip, gray) < a.hold_move_frac:
                # nothing (or only a static fragment) detected while the previous gripper region is unchanged: carry it
                grip, info["mode"] = self.prev_grip, "hold_mask" if "fragment" not in info else "hold_fragment"
                info["tip"] = [self.prev_tip[1], self.prev_tip[0]] if self.prev_tip is not None else None
                self.hold_ref = self.hold_ref if self.hold_ref is not None else self.prev_gray
                self.hold_len += 1
            elif grip is not None:
                self.hold_ref, self.hold_len = None, 0
        return self._finish(grip, arm, info, gray)


# ----------------------------------------------------------------------------------------------- video io
def iter_frames(path, proc_w, proc_h, want_native=False):
    cap = cv2.VideoCapture(path)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield (f if want_native else None), cv2.resize(f, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    cap.release()


def longest_run(b):
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def run_camera(cam, serial, video, ep, a, proc_w, proc_h):
    depth, fx, dd = plausible_depth(ep, serial)
    seeder = MotionOnlySeeder(proc_w, proc_h, fx * proc_w / NATIVE_W, depth, a)
    masks, arms, log = [], [], []
    t0 = time.time()
    for _, small in iter_frames(video, proc_w, proc_h):
        grip, arm, info = seeder.step(small)
        masks.append(grip)
        arms.append(arm)
        log.append(info)
    dt = time.time() - t0
    T = len(masks)
    present = np.array([m is not None for m in masks])
    stats = dict(cam=cam, serial=serial, n_frames=T, frames_present=int(present.sum()), frac_present=float(present.mean()) if T else 0.0,
                 plausible_depth_m=round(depth, 3), depth_details=dd, fx_native=round(fx, 1), px_per_m_proc=round(seeder.px_per_m, 1),
                 cap_px_proc=round(seeder.cap_px, 1), jump_px_proc=round(seeder.jump_px, 1), entry_side=seeder.entry_side,
                 entry_from_prior=seeder.entry_from_prior, entry_side_acc=seeder.side_acc,
                 modes={m: sum(1 for x in log if x["mode"] == m) for m in sorted(set(x["mode"] for x in log))},
                 mean_grip_area_frac=float(np.mean([m.mean() for m in masks if m is not None])) if present.any() else None,
                 first_present=int(np.argmax(present)) if present.any() else None,
                 longest_present_run=longest_run(present), seconds=round(dt, 1), ms_per_frame=round(1000 * dt / max(T, 1), 1))
    print(f"[{cam}/{serial}] {T} frames, present {present.sum()}/{T}, entry={seeder.entry_side} (prior={seeder.entry_from_prior}) {seeder.side_acc}, "
          f"depth={depth:.2f} m {dd}, cap={seeder.cap_px:.0f}px@{proc_w}, modes={stats['modes']}, {dt:.1f}s ({stats['ms_per_frame']} ms/frame)", flush=True)
    return masks, arms, log, stats


def save_masks(path, masks):
    T = len(masks)
    full = np.zeros((T, NATIVE_H, NATIVE_W), bool)
    for i, m in enumerate(masks):
        if m is not None:
            full[i] = cv2.resize(m.astype(np.uint8), (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_NEAREST).astype(bool)
    present = full.reshape(T, -1).any(1)
    np.savez_compressed(path, union=np.packbits(full, axis=2), shape=np.array([T, NATIVE_H, NATIVE_W]), frames_present=present)
    return present


def write_overlay(ep, cams, serials, videos, masks, arms, logs, out_dir, fps, png_frames, proc_w, proc_h):
    """Streaming two-panel overlay: gripper mask (id 1), arm component (id 2), tip (red circle)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sam3_gripper_masks import FfmpegWriter, render
    pw, ph = 640, 360
    path = os.path.join(out_dir, f"{METHOD}_ext_side_by_side.mp4")
    writer = FfmpegWriter(path, pw * len(cams), ph, fps)
    its = [iter_frames(videos[c], proc_w, proc_h, want_native=True) for c in cams]
    T = min(len(masks[c]) for c in cams)
    for i in range(T):
        panels = []
        for c, it in zip(cams, its):
            native, _ = next(it)
            m, arm, info = masks[c][i], arms[c][i], logs[c][i]
            ids, probs, ms = [], [], []
            if arm is not None:
                ids.append(2); probs.append(1.0); ms.append(cv2.resize(arm.astype(np.uint8), (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_NEAREST).astype(bool))
            if m is not None:
                ids.append(1); probs.append(1.0); ms.append(cv2.resize(m.astype(np.uint8), (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_NEAREST).astype(bool))
            out = None if not ids else dict(ids=np.array(ids), probs=np.array(probs), boxes=np.zeros((len(ids), 4)), masks=np.stack(ms))
            img = render(native, out, f"{c}/{serials[c]} {METHOD} f{i}/{T-1} {info['mode']} entry={info.get('entry')}")
            if info.get("tip") is not None:
                x, y = info["tip"]
                cv2.circle(img, (int(x * NATIVE_W / proc_w), int(y * NATIVE_H / proc_h)), 10, (0, 0, 255), 3)
            panels.append(cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA))
        row = np.concatenate(panels, axis=1)
        writer.write(row)
        if i in png_frames:
            cv2.imwrite(os.path.join(out_dir, f"{METHOD}_check_f{i:04d}.png"), row)
    writer.close()
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--cams", default="ext1,ext2")
    ap.add_argument("--proc_w", type=int, default=640, help="processing width (height = 9/16); masks are upsampled to 1280x720")
    # background model
    ap.add_argument("--history", type=int, default=300, help="MOG2 history: fixed learning rate 1/history after warm-up")
    ap.add_argument("--var_thresh", type=float, default=25.0, help="MOG2 varThreshold")
    ap.add_argument("--no_shadows", action="store_true", help="MOG2 detectShadows=False (cast shadows on the table then join the arm component)")
    ap.add_argument("--shadow_tau", type=float, default=0.3, help="MOG2 shadow threshold: pixel is shadow if its brightness is in [tau, 1] x background")
    ap.add_argument("--dark_thresh", type=int, default=70, help="shadow-labelled pixels darker than this gray level are kept as foreground (black hand)")
    ap.add_argument("--chroma_thresh", type=int, default=15, help="... and only if achromatic: max(BGR)-min(BGR) below this")
    ap.add_argument("--roots", default="all", choices=["entry", "all"], help="geodesic roots: all border contacts of the arm component (minus the gripper's own), or entry-side contacts only")
    ap.add_argument("--warmup_frames", type=int, default=5, help="frames used only to initialise the background model (no detection)")
    ap.add_argument("--warmup_lr", type=float, default=0.3)
    # cleaning and selection
    ap.add_argument("--close_px", type=int, default=9)
    ap.add_argument("--open_px", type=int, default=5)
    ap.add_argument("--min_area_frac", type=float, default=0.0015, help="drop foreground components below this image fraction")
    ap.add_argument("--min_area_first_frac", type=float, default=0.01, help="first acquisition (no continuity): the arm must be at least this big")
    ap.add_argument("--continuity_frac", type=float, default=0.2, help="component keeps the arm label if it covers this fraction of the previous gripper mask")
    ap.add_argument("--entry_prior", default="top", choices=["top", "bottom", "left", "right", "auto"], help="entry side prior; auto = data only")
    ap.add_argument("--entry_override_frames", type=int, default=30, help="motion-selected frames before the data may override the prior")
    ap.add_argument("--entry_override_ratio", type=float, default=3.0, help="another side must have this times more accumulated contact to override")
    ap.add_argument("--move_thresh", type=int, default=10, help="gray-level difference counted as motion / change")
    ap.add_argument("--move_dilate_px", type=int, default=7, help="small: a static object the arm just moved (drawer, cloth) must not become a tip candidate")
    # distal part
    ap.add_argument("--cap_m", type=float, default=0.25, help="distal cap along the arm, metres-equivalent")
    ap.add_argument("--max_cap_frac", type=float, default=1.0, help="cap never exceeds this fraction of the tip's geodesic length (1 = metric cap only)")
    ap.add_argument("--jump_m", type=float, default=0.2, help="tip jump limit per frame, metres-equivalent")
    ap.add_argument("--stale_frames", type=int, default=15)
    ap.add_argument("--min_depth", type=float, default=0.35, help="floor on the plausible depth used for the pixel scale")
    ap.add_argument("--min_grip_px", type=int, default=150, help="at processing resolution")
    ap.add_argument("--max_hole_px", type=int, default=400, help="fill enclosed holes smaller than this (processing resolution)")
    # hold
    ap.add_argument("--hold_move_frac", type=float, default=0.3, help="hold the previous gripper mask while less than this fraction of it changed since the hold began")
    ap.add_argument("--hold_max", type=int, default=150, help="longest hold in frames")
    ap.add_argument("--fragment_frac", type=float, default=0.5, help="a continuity component smaller than this fraction of the previous mask is a fragment: hold, never a new mask")
    # outputs
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--video", action="store_true", help="write the two-panel overlay mp4")
    ap.add_argument("--png_frames", default="", help="comma list of frame indices to also dump as check PNGs (needs --video)")
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
    out_dir = os.path.join(a.out_root, ep)
    os.makedirs(out_dir, exist_ok=True)
    proc_w = a.proc_w
    proc_h = proc_w * 9 // 16
    n_store = len([f for f in os.listdir(os.path.join(STORE_ROOT, ep, "dense", "cam")) if f.endswith(".npz")])
    print(f"episode {ep}\nvideos {videos}\nstore frames {n_store}\nout {out_dir}", flush=True)

    t_all = time.time()
    masks, arms, logs, stats, products = {}, {}, {}, {}, {}
    for c in cams:
        masks[c], arms[c], logs[c], stats[c] = run_camera(c, serials[c], videos[c], ep, a, proc_w, proc_h)
        path = os.path.join(out_dir, f"{c}_{serials[c]}__{METHOD}_masks.npz")
        present = save_masks(path, masks[c])
        chk = np.load(path)
        T = int(chk["shape"][0])
        ok = (T == n_store) and (tuple(chk["shape"]) == (T, NATIVE_H, NATIVE_W)) and bool((chk["frames_present"] == present).all()) \
            and int(chk["frames_present"].sum()) == stats[c]["frames_present"]
        products[c] = dict(npz=path, T=T, store_frames=n_store, bytes=os.path.getsize(path), frames_present=int(chk["frames_present"].sum()), product_ok=bool(ok))
        print(f"[{c}] wrote {path} T={T} (store {n_store}) present={int(chk['frames_present'].sum())} ok={ok}", flush=True)
    video_path = None
    if a.video:
        png = [int(x) for x in a.png_frames.split(",") if x.strip()]
        t0 = time.time()
        video_path = write_overlay(ep, cams, serials, videos, masks, arms, logs, out_dir, a.fps, png, proc_w, proc_h)
        print(f"overlay {video_path} in {time.time()-t0:.1f}s", flush=True)
    total = time.time() - t_all
    info = dict(episode=ep, method=METHOD, serials=serials, params=vars(a), proc_res=[proc_w, proc_h], native_res=[NATIVE_W, NATIVE_H],
                store_frames=n_store, causal=True, uses_gt=False,
                calibration_use="PointWorld extrinsics + factory intrinsics only for the metres-to-pixels scale of the cap and jump limit",
                prompts=None, note="no prompts, boxes or re-prompts: motion-only seeding has none",
                per_cam=stats, products=products, overlay_mp4=video_path,
                timing=dict(total_seconds=round(total, 1), cpu_seconds_process=round(time.process_time(), 1),
                            seeding_seconds={c: stats[c]["seconds"] for c in cams}),
                per_frame={c: logs[c] for c in cams})
    json.dump(info, open(os.path.join(out_dir, "seed_info.json"), "w"), indent=1, default=str)
    print(f"DONE {ep} in {total:.1f}s (process CPU {time.process_time():.1f}s); products ok: {all(p['product_ok'] for p in products.values())}", flush=True)


if __name__ == "__main__":
    main()
