#!/usr/bin/env python
"""Gripper masklets on all three DROID cameras of one episode with the recipe that WORKED in the 2026-09-17 session
(smoke/sam3_verify/wrist_gripper_video.py, smoke/gripper_ablation/export_gripper_masks*.py, export_dynamic_masks.py):
SAM 2-style tracker of build_sam3_video_model(), backbone = detector backbone, box + positive clicks in RELATIVE
coordinates on a high-contrast frame, propagated backward then forward from the prompt frame (propagate_preflight=True).
Text prompts are not used anywhere (they never found the gripper).

Per camera: prompt(s) -> prompt-frame check PNG (box, clicks, resulting mask) -> propagation -> overlay mp4 + union-mask
npz + per-frame csv (obj score per frame; frames where the tracker returns an empty mask are 'NO DETECTION').
Needs a GPU; see sam3_gripper_boxclick_3cams.sbatch.

RESOLUTION POLICY (user, 2026-09-28): default to the PointWorld store / scoring resolution, 320x180. The wrist is tracked on
the 320x180 clip and only upsampled for display. The exterior cameras have no PointWorld counterpart; their 1280x720 tracks
were approved as-is for this scene, but new exterior processing should default to 320x180 unless stated otherwise.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from sam3_gripper_masks import FfmpegWriter, OUT_ROOT, RAW_ROOT, read_frames, render, side_by_side, write_camera_outputs

# Prompts in NATIVE pixel coordinates of the 1280x720 MP4s (relative coords are what the tracker gets).
# wrist: the validated recipe was given in 320x180 pixels of the store clip; relative coords are identical.
# The wrist recipe was validated on the 320x180 store clip (outputs/cut3r_eval/wrist_videos/...); the tracker's masks depend
# on the input it sees, so the wrist is TRACKED on that exact clip and the masks are upsampled for the 1280x720 overlay.
WRIST_CLIP_320 = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/wrist_videos/RAIL+80edfcb1+2023-07-14-14h-28m-45s_wrist_13062452.mp4"
REFERENCE_WRIST_MASKS = "/gpfs/scratch/etur59/koc821022/smoke/gripper_ablation/masks/masks.npz"  # 2026-09-17 validated masks, for an identity check
PROMPTS = {
    "wrist": dict(ref_wh=(320, 180), track_video=WRIST_CLIP_320, prompts=[
        dict(frame=127, box=[60, 100, 305, 180], points=[[105, 165], [150, 172], [265, 135]]),
    ]),
    "ext1": dict(ref_wh=(1280, 720), prompts=[
        dict(frame=96, box=[670, 120, 760, 295], points=[[712, 175], [695, 255], [725, 250]]),   # gripper holding the bunny, bright background
        dict(frame=64, box=[605, 205, 705, 355], points=[[655, 250], [632, 315], [672, 320]]),   # gripper in the pot
    ]),
    "ext2": dict(ref_wh=(1280, 720), prompts=[
        dict(frame=112, box=[808, 170, 950, 305], points=[[870, 210], [920, 205], [862, 280]]),  # gripper holding the bunny
        dict(frame=80, box=[765, 198, 930, 335], points=[[820, 235], [880, 230], [815, 305]]),   # gripper in the pot
    ]),
}


def draw_prompt(img, box, points, ref_wh):
    W, H = img.shape[1], img.shape[0]
    sx, sy = W / ref_wh[0], H / ref_wh[1]
    x0, y0, x1, y1 = [int(v * s) for v, s in zip(box, (sx, sy, sx, sy))]
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
    for x, y in points:
        cv2.circle(img, (int(x * sx), int(y * sy)), 6, (0, 255, 0), -1)
        cv2.circle(img, (int(x * sx), int(y * sy)), 6, (0, 0, 0), 1)
    return img


def track_camera(tracker, video_path, frames, spec, out_dir, cam, serial):
    import torch
    rw, rh = spec["ref_wh"]
    track_video = spec.get("track_video", video_path)
    state = tracker.init_state(video_path=track_video)
    n = state["num_frames"]
    assert n == len(frames), (n, len(frames))
    H, W = frames[0].shape[:2]

    def to_render_res(m):  # tracker masks are at the tracked clip's resolution
        return m if m.shape == (H, W) else cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    checks = []
    for p in spec["prompts"]:
        pts = torch.tensor([[x / rw, y / rh] for x, y in p["points"]], dtype=torch.float32)
        labels = torch.ones(len(p["points"]), dtype=torch.int32)
        bx = p["box"]
        box = np.array([[bx[0] / rw, bx[1] / rh, bx[2] / rw, bx[3] / rh]], dtype=np.float32)
        _, oids, _, vmasks = tracker.add_new_points_or_box(inference_state=state, frame_idx=int(p["frame"]), obj_id=1,
                                                           points=pts, labels=labels, box=box)
        m = to_render_res((vmasks[0][0] > 0).cpu().numpy())
        out = dict(ids=np.array([1]), probs=np.array([1.0]), boxes=np.zeros((1, 4)), masks=m[None])
        img = render(frames[p["frame"]], out, f"{cam}/{serial} PROMPT frame {p['frame']}  box+{len(p['points'])} clicks  mask area {m.mean()*100:.1f}%")
        draw_prompt(img, bx, p["points"], (rw, rh))
        path = os.path.join(out_dir, f"prompt_check_{cam}_f{p['frame']:03d}.png")
        cv2.imwrite(path, img)
        checks.append(dict(frame=int(p["frame"]), mask_area_frac=float(m.mean()), png=path))
        print(f"[{cam}] prompt on frame {p['frame']}: mask area {m.mean()*100:.2f}%", flush=True)
    masks, scores = {}, {}
    starts = [p["frame"] for p in spec["prompts"]]  # later prompt first, like export_dynamic_masks.py
    for start in starts:
        for rev in (True, False):
            for fidx, oids, _, vmasks, obj_scores in tracker.propagate_in_video(state, start_frame_idx=int(start), max_frame_num_to_track=n,
                                                                              reverse=rev, propagate_preflight=True):
                if fidx in masks:
                    continue
                masks[fidx] = (vmasks[0][0] > 0).cpu().numpy()  # native tracking resolution
                s = obj_scores
                try:
                    s = float(np.asarray(s.detach().cpu().numpy() if hasattr(s, "detach") else s).reshape(-1)[0])
                except Exception:
                    s = float("nan")
                scores[fidx] = s
    assert sorted(masks) == list(range(n)), (len(masks), n)
    if cam == "wrist" and os.path.isfile(REFERENCE_WRIST_MASKS):  # identity check against the 2026-09-17 masks
        ref = np.load(REFERENCE_WRIST_MASKS)["masks"].astype(bool)
        if ref.shape == (n,) + masks[0].shape:
            iou = [float((ref[i] & masks[i]).sum() / max((ref[i] | masks[i]).sum(), 1)) for i in range(n)]
            print(f"[wrist] IoU vs 2026-09-17 reference masks: min {min(iou):.4f} mean {np.mean(iou):.4f} (identical frames: {sum(v == 1.0 for v in iou)}/{n})", flush=True)
        else:
            print(f"[wrist] reference masks shape {ref.shape} != {(n,) + masks[0].shape}; no identity check", flush=True)
    outputs = {}
    for i in range(n):
        m = to_render_res(masks[i])
        prob = 1.0 / (1.0 + np.exp(-scores[i])) if np.isfinite(scores[i]) else 1.0
        if m.any():
            outputs[i] = dict(ids=np.array([1]), probs=np.array([prob]), boxes=np.zeros((1, 4)), masks=m[None])
        else:
            outputs[i] = dict(ids=np.zeros(0, int), probs=np.zeros(0), boxes=np.zeros((0, 4)), masks=np.zeros((0,) + m.shape, bool))
    del state
    torch.cuda.empty_cache()
    return outputs, scores, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="RAIL+80edfcb1+2023-07-14-14h-28m-45s")
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--cams", default="wrist,ext1,ext2")
    args = ap.parse_args()
    ep = args.episode
    raw = os.path.join(RAW_ROOT, ep)
    meta = json.load(open(os.path.join(raw, f"metadata_{ep}.json")))
    serials = dict(wrist=meta["wrist_cam_serial"], ext1=meta["ext1_cam_serial"], ext2=meta["ext2_cam_serial"])
    cams = [c for c in args.cams.split(",") if c]
    videos = {c: os.path.join(raw, "recordings", "MP4", f"{serials[c]}.mp4") for c in cams}
    out_dir = os.path.join(args.out_root, ep)
    os.makedirs(out_dir, exist_ok=True)
    frames = {c: read_frames(videos[c]) for c in cams}
    for c in cams:
        print(f"{c}: {len(frames[c])} frames {frames[c][0].shape[1]}x{frames[c][0].shape[0]}", flush=True)

    from sam3.model_builder import build_sam3_video_model
    t0 = time.time()
    model = build_sam3_video_model()
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    print(f"tracker built in {time.time()-t0:.0f}s", flush=True)

    summaries, outputs_all, label = [], {}, {}
    for c in cams:
        t0 = time.time()
        outs, scores, checks = track_camera(tracker, videos[c], frames[c], PROMPTS[c], out_dir, c, serials[c])
        print(f"[{c}] tracked in {time.time()-t0:.0f}s", flush=True)
        s = write_camera_outputs(c, serials[c], "box+clicks", "gripper_boxclick", frames[c], outs, out_dir, args.fps,
                                 extra=dict(method="sam3 SAM2-style tracker, box + clicks (2026-09-17 recipe)", prompts=PROMPTS[c], prompt_checks=checks,
                                            obj_score_min=float(np.nanmin(list(scores.values()))), obj_score_max=float(np.nanmax(list(scores.values())))))
        # per-frame object scores next to the csv
        with open(os.path.join(out_dir, f"{c}_{serials[c]}__gripper_boxclick_objscore.csv"), "w") as f:
            f.write("frame,obj_score_logit,mask_nonempty\n")
            for i in range(len(frames[c])):
                f.write(f"{i},{scores[i]:.4f},{int(len(outs[i]['ids']) > 0)}\n")
        summaries.append(s)
        outputs_all[c], label[c] = outs, "box+clicks"
    if len(cams) > 1:
        side_by_side(cams, frames, outputs_all, label, os.path.join(out_dir, "all_cams_gripper_boxclick_side_by_side.mp4"), args.fps)
    json.dump(dict(episode=ep, serials=serials, summaries=summaries), open(os.path.join(out_dir, "summary_boxclick.json"), "w"), indent=1, default=str)
    print("\nSUMMARY")
    for s in summaries:
        print(f"  {s['cam']:5s} frames with mask {s['frames_with_detection']}/{s['n_frames']}  mean area {s['mean_area_frac']}")
    print("DONE", out_dir)


if __name__ == "__main__":
    main()
