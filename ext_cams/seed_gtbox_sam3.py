#!/usr/bin/env python
"""gtbox_sam3 -- REFERENCE / CEILING seeding of the gripper masks in the two exterior cameras (PRIVILEGED, NOT A CANDIDATE).

Label everything produced here as `reference`: the prompts are built from the kinematic GROUND-TRUTH wrist pose
(store cam/*.npz) at the frame being prompted. This method exists to be the IoU reference for the training-free
candidate seeding methods; it can never be a method row.

Recipe (per episode, per exterior camera, at the native 1280x720):
  1. Project the GT lens (wrist camera centre, base frame) into the exterior image with the PointWorld
     optimized_extrinsics (world-to-camera, used directly) and the factory intrinsics (hf_intrinsics.json).
  2. At the FIRST frame where the projected GT lens is inside the image with a margin, prompt the SAM3
     SAM 2-style tracker (build_sam3_video_model, the recipe of sam3_gripper_boxclick_3cams.py: box + positive
     clicks in relative coordinates) with a box centred slightly below the lens point -- at the projection of
     lens + fwd_m * (wrist-camera +z axis), which on the validated RAIL masks is where the mask centroid sits
     (0.083 / 0.081 m along +z in ext1 / ext2) -- sized from the gripper's expected apparent size,
     box_w_m x box_h_m metres at the lens depth through the intrinsics, plus positive clicks along the same axis.
  3. Propagate FORWARD ONLY (frames > t use memory of frames <= t; nothing is propagated backwards).
  4. Re-prompt from the GT projection whenever the tracker has lost the object for more than `lost_frames`
     consecutive frames while the GT lens is in view (loss = empty mask). Optional drift rule (on by default,
     reported separately): re-prompt when the mask sits mostly outside a 1.5x-expanded GT box for the same run
     length. A re-prompt resets the tracker memory (clear_all_points_in_video) and restarts at the current frame,
     so the mask at t still depends only on frames <= t.

Outputs (out_root/EP/): <cam>_<serial>__gtbox_sam3_masks.npz following the MASK FILE CONTRACT
    'union' uint8 (T, H, W/8) = np.packbits(mask.astype(uint8), axis=2) of the (T, 720, 1280) boolean mask,
    'shape' = [T, 720, 1280], 'frames_present' bool (T,) true where the mask is non-empty
  <cam>_<serial>__gtbox_sam3_frames.csv (per-frame presence, area, object logit, GT in-view flag, lens pixel),
  <cam>_<serial>__gtbox_sam3.mp4 overlay (mask + GT lens cross + prompt boxes), prompt_check_<cam>_f<t>.png per
  prompt event, all_ext_gtbox_sam3_side_by_side.mp4, seed_info.json (prompt frames / boxes / re-prompts / timing).

    python seed_gtbox_sam3.py --episode EP --out_root .../seedstudy/gtbox_sam3        # GPU (sbatch)
    OMP_NUM_THREADS=1 python seed_gtbox_sam3.py --episode EP --dry_run                # CPU geometry check only
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam3_gripper_masks import FfmpegWriter, RAW_ROOT, render, side_by_side  # noqa: E402

METHOD = "gtbox_sam3"
ROLE = "reference/ceiling (privileged: prompts built from the kinematic GT wrist pose; not a candidate method)"
OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/gtbox_sam3"
AUDIT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/gt_audit_2026-09-08"
STORE_ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
CAMS = ("ext1", "ext2")


# ----------------------------------------------------------------------------------------------- geometry
def load_episode(ep):
    meta = json.load(open(os.path.join(RAW_ROOT, ep, f"metadata_{ep}.json")))
    serials = {c: str(meta[f"{c}_cam_serial"]) for c in CAMS}
    intr = json.load(open(f"{AUDIT}/docs/hf_intrinsics.json"))[ep]
    cams = json.load(open(f"{AUDIT}/pointworld/droid/cameras/{ep}_cameras.json"))
    K, E, WH = {}, {}, {}
    for c in CAMS:
        s = serials[c]
        fx, cx, fy, cy = intr[s]["cameraMatrix"]
        K[c] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        WH[c] = (int(intr[s]["width"]), int(intr[s]["height"]))
        E[c] = np.array(cams[s]["optimized_extrinsics"], float)  # world(base) -> camera
    cam_dir = os.path.join(STORE_ROOT, ep, "dense", "cam")
    files = sorted(f for f in os.listdir(cam_dir) if f.endswith(".npz"))
    T = len(files)
    assert [int(f[:6]) for f in files] == list(range(T)), "store cam files are not 0..T-1"
    gt = np.stack([np.load(os.path.join(cam_dir, f))["pose"].astype(np.float64) for f in files])
    return serials, K, E, WH, gt, T


def project(K, E, X):
    p = E[:3, :3] @ X + E[:3, 3]
    q = K @ p
    return q[0] / q[2], q[1] / q[2], p[2]


def prompt_geometry(K, E, pose, W, H, a):
    """Box + clicks for one frame from the GT wrist pose. Returns dict with pixel coordinates (native resolution)."""
    lens = pose[:3, 3]
    zax = pose[:3, :3][:, 2]  # wrist-camera optical axis in the base frame (OpenCV convention, audited)
    ul, vl, zl = project(K, E, lens)
    centre = lens + a.fwd_m * zax
    uc, vc, zc = project(K, E, centre)
    bw = K[0, 0] * a.box_w_m / max(zc, 1e-3)
    bh = K[1, 1] * a.box_h_m / max(zc, 1e-3)
    box_raw = [uc - bw / 2, vc - bh / 2, uc + bw / 2, vc + bh / 2]
    box = [min(max(box_raw[0], 0), W - 1), min(max(box_raw[1], 0), H - 1), min(max(box_raw[2], 0), W - 1), min(max(box_raw[3], 0), H - 1)]
    clicks = []
    for f in a.click_fwd:
        u, v, z = project(K, E, lens + f * zax)
        if z > 0 and 0 <= u < W and 0 <= v < H:
            clicks.append([float(u), float(v)])
    m = a.margin
    in_view = bool(zl > 0.05 and zc > 0.05 and m <= ul < W - m and m <= vl < H - m and 0 <= uc < W and 0 <= vc < H)
    return dict(lens_px=[float(ul), float(vl)], lens_z=float(zl), centre_px=[float(uc), float(vc)], centre_z=float(zc),
                box=[float(x) for x in box], box_raw=[float(x) for x in box_raw], clicks=clicks, in_view=in_view)


def mask_frac_in_box(m, box, W, H, expand):
    if not m.any():
        return 0.0
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) / 2 * expand, (box[3] - box[1]) / 2 * expand
    x0, y0 = int(max(cx - hw, 0)), int(max(cy - hh, 0))
    x1, y1 = int(min(cx + hw, W - 1)) + 1, int(min(cy + hh, H - 1)) + 1
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(m[y0:y1, x0:x1].sum() / m.sum())


# ----------------------------------------------------------------------------------------------- drawing
def draw_prompt(img, g, present_mask=None, color=(0, 255, 255)):
    x0, y0, x1, y1 = [int(round(v)) for v in g["box"]]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    for x, y in g["clicks"]:
        cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.circle(img, (int(x), int(y)), 6, (0, 0, 0), 1)
    return img


def draw_gt(img, g):
    u, v = g["lens_px"]
    H, W = img.shape[:2]
    if 0 <= u < W and 0 <= v < H:
        cv2.drawMarker(img, (int(u), int(v)), (0, 255, 0) if g["in_view"] else (0, 140, 255), cv2.MARKER_CROSS, 26, 2)
    uc, vc = g["centre_px"]
    if 0 <= uc < W and 0 <= vc < H:
        cv2.circle(img, (int(uc), int(vc)), 5, (255, 0, 255), -1)
    return img


def read_frames_list(path, T):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < T:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


# ----------------------------------------------------------------------------------------------- tracking (GPU)
def track_camera(tracker, video_path, geoms, T, W, H, a, cam, out_dir, frames, log):
    import torch
    t_load = time.time()
    state = tracker.init_state(video_path=video_path, offload_video_to_cpu=a.offload_video)
    n = state["num_frames"]
    log(f"[{cam}] tracker state: {n} frames {state['video_width']}x{state['video_height']} loaded in {time.time() - t_load:.1f}s")
    assert n == T, f"MP4 frames {n} != store frames {T}"
    assert (state["video_width"], state["video_height"]) == (W, H), (state["video_width"], state["video_height"], W, H)
    masks = np.zeros((T, H, W), bool)
    logits = np.full(T, np.nan)
    events = []
    in_view = np.array([g["in_view"] for g in geoms])

    def prompt_at(t, reason):
        tracker.clear_all_points_in_video(state)  # fresh memory: the restart at t depends on frame t only
        g = geoms[t]
        bx = g["box"]
        box = np.array([[bx[0] / W, bx[1] / H, bx[2] / W, bx[3] / H]], dtype=np.float32)
        pts = torch.tensor([[x / W, y / H] for x, y in g["clicks"]], dtype=torch.float32) if g["clicks"] else None
        labels = torch.ones(len(g["clicks"]), dtype=torch.int32) if g["clicks"] else None
        _, oids, _, vmasks = tracker.add_new_points_or_box(inference_state=state, frame_idx=int(t), obj_id=1, points=pts, labels=labels, box=box)
        m = (vmasks[0][0] > 0).cpu().numpy()
        ev = dict(frame=int(t), reason=reason, box=bx, clicks=g["clicks"], lens_px=g["lens_px"], lens_z=g["lens_z"],
                  centre_px=g["centre_px"], mask_area_frac=float(m.mean()), mask_area_px=int(m.sum()))
        events.append(ev)
        if len(events) <= a.max_check_png and frames is not None:
            out = dict(ids=np.array([1]), probs=np.array([1.0]), boxes=np.zeros((1, 4)), masks=m[None])
            img = render(frames[t], out, f"{cam} REFERENCE gtbox_sam3 prompt #{len(events)} ({reason}) frame {t}  box+{len(g['clicks'])} clicks  mask area {m.mean() * 100:.2f}%")
            draw_prompt(img, g)
            draw_gt(img, g)
            cv2.imwrite(os.path.join(out_dir, f"prompt_check_{cam}_f{t:04d}.png"), img)
        log(f"[{cam}] prompt ({reason}) on frame {t}: lens z {g['lens_z']:.2f} m, box {[round(v) for v in bx]}, {len(g['clicks'])} clicks, mask area {m.mean() * 100:.2f}% ({int(m.sum())} px)")
        return m

    if not in_view.any():
        log(f"[{cam}] GT lens never in view with margin {a.margin}px: no prompt, all masks empty")
        return masks, logits, events, in_view
    t0 = int(np.argmax(in_view))
    prompt_at(t0, "initial")
    cur = t0
    lost_run = drift_run = 0
    t_track = time.time()
    while cur < T:
        restart = None
        for fidx, oids, _, vmasks, obj_scores in tracker.propagate_in_video(state, start_frame_idx=int(cur), max_frame_num_to_track=T,
                                                                          reverse=False, propagate_preflight=True, tqdm_disable=True):
            m = (vmasks[0][0] > 0).cpu().numpy()
            masks[fidx] = m
            try:
                logits[fidx] = float(np.asarray(obj_scores.detach().cpu().numpy() if hasattr(obj_scores, "detach") else obj_scores).reshape(-1)[0])
            except Exception:
                pass
            present = m.sum() >= a.min_area_px
            if in_view[fidx]:
                lost_run = 0 if present else lost_run + 1
                if present and a.drift_frac > 0:
                    drift_run = drift_run + 1 if mask_frac_in_box(m, geoms[fidx]["box"], W, H, a.drift_expand) < a.drift_frac else 0
                else:
                    drift_run = 0
                if lost_run > a.lost_frames or drift_run > a.lost_frames:
                    restart = (int(fidx), "lost" if lost_run > a.lost_frames else "drift")
                    break
        if restart is None:
            break
        if len(events) > a.max_reprompts:
            log(f"[{cam}] re-prompt cap {a.max_reprompts} reached at frame {restart[0]}; continuing without further re-prompts")
            # finish the remaining frames with the current memory (no more restarts)
            for fidx, oids, _, vmasks, obj_scores in tracker.propagate_in_video(state, start_frame_idx=int(restart[0]), max_frame_num_to_track=T,
                                                                              reverse=False, propagate_preflight=False, tqdm_disable=True):
                masks[fidx] = (vmasks[0][0] > 0).cpu().numpy()
            break
        lost_run = drift_run = 0
        prompt_at(restart[0], restart[1])
        cur = restart[0]
    log(f"[{cam}] tracked {T} frames in {time.time() - t_track:.1f}s; prompts {len(events)} "
        f"(initial 1, lost {sum(e['reason'] == 'lost' for e in events)}, drift {sum(e['reason'] == 'drift' for e in events)})")
    del state
    torch.cuda.empty_cache()
    return masks, logits, events, in_view


# ----------------------------------------------------------------------------------------------- outputs
def write_contract_npz(path, masks):
    T, H, W = masks.shape
    assert masks.dtype == bool and (H, W) == (720, 1280), (masks.dtype, H, W)
    union = np.packbits(masks.astype(np.uint8), axis=2)
    np.savez_compressed(path, union=union, shape=np.array([T, H, W]), frames_present=masks.any(axis=(1, 2)))
    chk = np.load(path)
    back = np.unpackbits(chk["union"], axis=2)[:, :, :W].astype(bool)
    assert back.shape == (T, H, W) and np.array_equal(back, masks) and np.array_equal(chk["frames_present"], masks.any(axis=(1, 2)))


def write_overlay(path, frames, masks, geoms, events, cam, serial, fps):
    T = len(frames)
    H, W = frames[0].shape[:2]
    ev_by_t = {e["frame"]: e for e in events}
    writer = FfmpegWriter(path, W, H, fps)
    for t in range(T):
        m = masks[t]
        out = dict(ids=np.array([1]), probs=np.array([1.0]), boxes=np.zeros((1, 4)), masks=m[None]) if m.any() else None
        tag = "  PROMPT(" + ev_by_t[t]["reason"] + ")" if t in ev_by_t else ""
        img = render(frames[t], out, f"{cam}/{serial}  REFERENCE gtbox_sam3  frame {t}/{T - 1}  GT lens {'in view' if geoms[t]['in_view'] else 'out of view'}{tag}")
        draw_gt(img, geoms[t])
        if t in ev_by_t:
            draw_prompt(img, geoms[t])
        writer.write(img)
    writer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--box_w_m", type=float, default=0.15, help="box width in metres at the lens depth")
    ap.add_argument("--box_h_m", type=float, default=0.20, help="box height in metres at the lens depth")
    ap.add_argument("--fwd_m", type=float, default=0.08, help="box centre = lens + fwd_m along the wrist-camera +z axis ('slightly below' the lens)")
    ap.add_argument("--click_fwd", type=float, nargs="+", default=[0.02, 0.05, 0.09], help="positive clicks at lens + f * z for each f")
    ap.add_argument("--margin", type=int, default=24, help="in-view margin in pixels for the projected GT lens")
    ap.add_argument("--lost_frames", type=int, default=5, help="re-prompt when lost for MORE than this many consecutive in-view frames")
    ap.add_argument("--min_area_px", type=int, default=30, help="masks smaller than this count as lost")
    ap.add_argument("--drift_frac", type=float, default=0.35, help="drift rule: mask fraction inside the expanded GT box below this counts as drift (0 = off)")
    ap.add_argument("--drift_expand", type=float, default=1.5)
    ap.add_argument("--max_reprompts", type=int, default=60)
    ap.add_argument("--max_check_png", type=int, default=8)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--offload_video", action="store_true", help="keep the decoded frames on CPU inside the tracker state")
    ap.add_argument("--no_overlay", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="CPU only: geometry, in-view frames, first-prompt PNGs, no SAM3")
    a = ap.parse_args()

    t_start = time.time()
    ep = a.episode
    out_dir = os.path.join(a.out_root, ep)
    os.makedirs(out_dir, exist_ok=True)
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    serials, K, E, WH, gt, T = load_episode(ep)
    log(f"episode {ep}: {T} store frames; serials {serials}; METHOD {METHOD} = {ROLE}")
    geoms = {c: [prompt_geometry(K[c], E[c], gt[t], WH[c][0], WH[c][1], a) for t in range(T)] for c in CAMS}
    info = dict(episode=ep, method=METHOD, role=ROLE, causal="forward-only propagation; the mask at t depends on frames <= t and the GT pose at the prompt frame only",
                uses_gt=True, gt_use="kinematic wrist pose (store cam/*.npz) projected with PointWorld optimized_extrinsics + factory intrinsics: prompt box/clicks, in-view test, re-prompt decisions",
                serials=serials, T=T, resolution=dict(WH), params=vars(a), cams={}, timing={})
    for c in CAMS:
        iv = np.array([g["in_view"] for g in geoms[c]])
        first = int(np.argmax(iv)) if iv.any() else None
        info["cams"][c] = dict(serial=serials[c], gt_in_view_frames=int(iv.sum()), first_in_view_frame=first,
                               first_prompt_geometry=geoms[c][first] if first is not None else None)
        log(f"[{c}] GT lens in view (margin {a.margin}px) on {iv.sum()}/{T} frames; first in-view frame {first}"
            + (f"; box {[round(v) for v in geoms[c][first]['box']]} at z {geoms[c][first]['lens_z']:.2f} m, {len(geoms[c][first]['clicks'])} clicks" if first is not None else ""))

    videos = {c: os.path.join(RAW_ROOT, ep, "recordings", "MP4", f"{serials[c]}.mp4") for c in CAMS}
    for c in CAMS:
        assert os.path.isfile(videos[c]), videos[c]

    if a.dry_run:
        for c in CAMS:
            first = info["cams"][c]["first_in_view_frame"]
            if first is None:
                continue
            cap = cv2.VideoCapture(videos[c])
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            ok, img = cap.read()
            cap.release()
            if ok:
                draw_prompt(img, geoms[c][first])
                draw_gt(img, geoms[c][first])
                cv2.putText(img, f"{c} DRY RUN prompt geometry frame {first}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imwrite(os.path.join(out_dir, f"dryrun_prompt_geometry_{c}_f{first:04d}.png"), img)
        info["timing"]["elapsed_s"] = time.time() - t_start
        json.dump(info, open(os.path.join(out_dir, "seed_info_dryrun.json"), "w"), indent=1)
        log("DRY_RUN_OK " + out_dir)
        return

    import torch
    from sam3.model_builder import build_sam3_video_model
    t0 = time.time()
    model = build_sam3_video_model()
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    info["timing"]["model_build_s"] = time.time() - t0
    log(f"tracker built in {info['timing']['model_build_s']:.0f}s on {torch.cuda.get_device_name(0)}")

    frames_all, outputs_all, labels = {}, {}, {}
    for c in CAMS:
        W, H = WH[c]
        t0 = time.time()
        frames = None if a.no_overlay else read_frames_list(videos[c], T)
        if frames is not None:
            assert len(frames) == T and frames[0].shape[:2] == (H, W), (len(frames), frames[0].shape, T, H, W)
        t_read = time.time() - t0
        t0 = time.time()
        masks, logits, events, in_view = track_camera(tracker, videos[c], geoms[c], T, W, H, a, c, out_dir, frames, log)
        t_track = time.time() - t0
        t0 = time.time()
        base = os.path.join(out_dir, f"{c}_{serials[c]}__{METHOD}")
        write_contract_npz(base + "_masks.npz", masks)
        present = masks.any(axis=(1, 2))
        area = masks.reshape(T, -1).mean(1)
        ev_by_t = {e["frame"]: e["reason"] for e in events}
        with open(base + "_frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "present", "area_frac", "obj_logit", "gt_in_view", "lens_u", "lens_v", "lens_z", "prompt_event"])
            for t in range(T):
                g = geoms[c][t]
                w.writerow([t, int(present[t]), f"{area[t]:.5f}", f"{logits[t]:.3f}" if np.isfinite(logits[t]) else "", int(g["in_view"]),
                            f"{g['lens_px'][0]:.1f}", f"{g['lens_px'][1]:.1f}", f"{g['lens_z']:.3f}", ev_by_t.get(t, "")])
        if frames is not None:
            write_overlay(base + ".mp4", frames, masks, geoms[c], events, c, serials[c], a.fps)
            frames_all[c] = frames
            outputs_all[c] = {t: (dict(ids=np.array([1]), probs=np.array([1.0]), boxes=np.zeros((1, 4)), masks=masks[t][None]) if present[t] else None) for t in range(T)}
            labels[c] = "REFERENCE gtbox_sam3"
        t_write = time.time() - t0
        info["cams"][c].update(dict(frames_present=int(present.sum()), frames_present_and_in_view=int((present & in_view).sum()),
                                    in_view_without_mask=int((in_view & ~present).sum()), mean_area_frac_present=float(area[present].mean()) if present.any() else None,
                                    prompts=events, n_prompts=len(events), n_reprompt_lost=sum(e["reason"] == "lost" for e in events),
                                    n_reprompt_drift=sum(e["reason"] == "drift" for e in events), masks_npz=base + "_masks.npz",
                                    timing=dict(read_frames_s=t_read, track_s=t_track, write_s=t_write)))
        log(f"[{c}] mask present on {present.sum()}/{T} frames ({(present & in_view).sum()} of the {in_view.sum()} GT-in-view frames); "
            f"read {t_read:.0f}s track {t_track:.0f}s write {t_write:.0f}s")
    if frames_all:
        t0 = time.time()
        side_by_side(list(CAMS), frames_all, outputs_all, labels, os.path.join(out_dir, f"all_ext_{METHOD}_side_by_side.mp4"), a.fps)
        info["timing"]["side_by_side_s"] = time.time() - t0
    info["timing"]["elapsed_s"] = time.time() - t_start
    info["log"] = log_lines
    json.dump(info, open(os.path.join(out_dir, "seed_info.json"), "w"), indent=1, default=str)
    log(f"DONE {out_dir} in {info['timing']['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
