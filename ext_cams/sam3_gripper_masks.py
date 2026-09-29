#!/usr/bin/env python
"""SAM 3 gripper masks on the three DROID cameras (wrist, ext1, ext2) of one episode.

Phase 1  probe:   for each camera, run several text phrases on several frames (detector only) and
                  record what fires and with what probability.
Phase 2  track:   for each camera, pick the best-firing GRIPPER phrase (and, separately, the best ARM
                  phrase as a diagnostic), prompt it on its best frame and propagate through the whole
                  video with the SAM 3.1 multiplex predictor (text stays active: instances that appear
                  later still get detected).
Phase 3  fallback: on the wrist camera, if no gripper phrase fires, use the box + 3-click recipe that was
                  validated in /gpfs/scratch/etur59/koc821022/smoke/sam3_verify (SAM 2-style tracker,
                  relative coordinates, so it transfers from 320x180 to 1280x720 unchanged).

Per camera it writes an overlay mp4 (h264, 15 fps = DROID's native rate), a compressed npz with the
union mask per frame, a per-frame csv, plus probe.json / summary.json for the episode and one
three-panel side-by-side mp4.  Needs a GPU (the SAM 3 code hard-codes CUDA); see the .sbatch next to it.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

FFMPEG = "/home/koc/koc821022/.conda/envs/cuteanything/bin/ffmpeg"  # has libx264; the module ffmpeg is not on PATH under sbatch
RAW_ROOT = "/gpfs/scratch/etur59/koc821022/vggt_cache/raw"
OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/gripper_sam3"

GRIPPER_PHRASES = ["robot gripper", "gripper", "robotic gripper", "robot hand", "end effector", "robot claw", "gripper fingers", "black gripper", "robot end effector", "parallel jaw gripper", "robot fingers", "mechanical claw", "wrist camera"]
ARM_PHRASES = ["robot arm", "robotic arm", "robot"]
# BGR palette, one colour per object id (mod len)
PALETTE = [(0, 200, 255), (0, 255, 0), (255, 80, 0), (255, 0, 255), (0, 128, 255), (255, 255, 0), (128, 0, 255), (0, 255, 255)]
# validated wrist recipe (smoke/sam3_verify/wrist_gripper_video.py): frame 127, box + 3 positive clicks, 320x180 pixel coords
WRIST_FALLBACK = dict(frame=127, points_px=[[105, 165], [150, 172], [265, 135]], box_px=[60, 100, 305, 180], ref_wh=(320, 180))


# ----------------------------------------------------------------------------------------------- helpers
def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def clean_out(out):
    """Normalise one predictor output dict to numpy (ids int, probs float, boxes rel xywh, masks bool N,H,W)."""
    ids = to_np(out["out_obj_ids"]).astype(int).reshape(-1)
    probs = to_np(out["out_probs"]).astype(float).reshape(-1)
    boxes = to_np(out["out_boxes_xywh"]).astype(float).reshape(-1, 4)
    masks = to_np(out["out_binary_masks"]).astype(bool)
    if masks.ndim == 2:
        masks = masks[None]
    return dict(ids=ids, probs=probs, boxes=boxes, masks=masks)


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames


class FfmpegWriter:
    def __init__(self, path, w, h, fps):
        self.proc = subprocess.Popen(
            [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)
        self.w, self.h = w, h

    def write(self, frame):
        assert frame.shape[1] == self.w and frame.shape[0] == self.h, (frame.shape, self.w, self.h)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with {rc}")


def render(frame, out, header, alpha=0.45):
    """Overlay masks (colour by id), outlines, boxes and labels on a BGR frame; banner when nothing is detected."""
    img = frame.copy()
    H, W = img.shape[:2]
    if out is not None and len(out["ids"]):
        overlay = img.copy()
        for k, (oid, p, box, m) in enumerate(zip(out["ids"], out["probs"], out["boxes"], out["masks"])):
            col = PALETTE[int(oid) % len(PALETTE)]
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay[m] = col
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, col, 2)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        for oid, p, box, m in zip(out["ids"], out["probs"], out["boxes"], out["masks"]):
            col = PALETTE[int(oid) % len(PALETTE)]
            x, y, w, h = box
            x0, y0, x1, y1 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
            if w > 0 and h > 0:
                cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            else:  # no box given (fallback tracker): derive from the mask
                ys, xs = np.nonzero(m)
                if len(xs):
                    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
                    cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            label = f"id{int(oid)} p={p:.2f} area={m.mean()*100:.1f}%" if p >= 0 else f"id{int(oid)} p=lost area={m.mean()*100:.1f}%"
            cv2.putText(img, label, (max(x0, 2), max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (max(x0, 2), max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (0, H // 2 - 22), (W, H // 2 + 22), (0, 0, 160), -1)
        cv2.putText(img, "NO DETECTION", (W // 2 - 110, H // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    # header strip
    cv2.rectangle(img, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(img, header, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def write_camera_outputs(cam, serial, phrase, tag, frames, outputs, out_dir, fps, extra=None):
    """outputs: {frame_idx: cleaned out or None}. Writes mp4 + npz + csv, returns summary dict."""
    N = len(frames)
    H, W = frames[0].shape[:2]
    base = os.path.join(out_dir, f"{cam}_{serial}__{tag}")
    writer = FfmpegWriter(base + ".mp4", W, H, fps)
    union = np.zeros((N, H, W), dtype=bool)
    rows, n_present, probs_all, area_all = [], 0, [], []
    for i in range(N):
        out = outputs.get(i)
        if out is not None and len(out["ids"]):
            n_present += 1
            union[i] = out["masks"].any(0)
            if (out["probs"] > 0).any():  # -10000 = tracker's "object dropped after this frame" sentinel
                probs_all.append(float(out["probs"][out["probs"] > 0].max()))
            area_all.append(float(union[i].mean()))
            rows.append([i, len(out["ids"]), " ".join(map(str, out["ids"].tolist())), " ".join(f"{p:.3f}" for p in out["probs"]),
                         f"{union[i].mean():.5f}", " ".join(f"{v:.4f}" for v in out["boxes"].reshape(-1))])
        else:
            rows.append([i, 0, "", "", "0", ""])
        header = f"{cam}/{serial}  '{phrase}'  frame {i}/{N-1}  n={0 if out is None else len(out['ids'])}"
        writer.write(render(frames[i], out, header))
    writer.close()
    np.savez_compressed(base + "_masks.npz", union=np.packbits(union, axis=-1), shape=np.array(union.shape), frames_present=np.array([r[1] > 0 for r in rows]))
    with open(base + "_frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "n_obj", "ids", "probs", "union_area_frac", "boxes_xywh_rel"])
        w.writerows(rows)
    summ = dict(cam=cam, serial=serial, tag=tag, phrase=phrase, n_frames=N, frames_with_detection=n_present,
                frac_frames_with_detection=n_present / N, mean_max_prob=float(np.mean(probs_all)) if probs_all else None,
                mean_area_frac=float(np.mean(area_all)) if area_all else None, mp4=base + ".mp4", masks_npz=base + "_masks.npz")
    if extra:
        summ.update(extra)
    print(f"[{cam}/{tag}] '{phrase}': detected on {n_present}/{N} frames, mean max prob {summ['mean_max_prob']}, mean area {summ['mean_area_frac']}", flush=True)
    return summ


def side_by_side(cams, frames_by_cam, outputs_by_cam, phrase_by_cam, out_path, fps, panel_w=640, panel_h=360):
    N = min(len(frames_by_cam[c]) for c in cams)
    writer = FfmpegWriter(out_path, panel_w * len(cams), panel_h, fps)
    for i in range(N):
        panels = []
        for c in cams:
            out = outputs_by_cam[c].get(i)
            n = 0 if out is None else len(out["ids"])
            img = render(frames_by_cam[c][i], out, f"{c} '{phrase_by_cam[c]}' f{i} n={n}")
            panels.append(cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA))
        writer.write(np.concatenate(panels, axis=1))
    writer.close()


# ----------------------------------------------------------------------------------------------- SAM 3 calls
def probe_phrases(predictor, sid, phrases, probe_frames, thresh):
    rows = []
    for phrase in phrases:
        for f in probe_frames:
            predictor.handle_request(request=dict(type="reset_session", session_id=sid))
            out = clean_out(predictor.handle_request(request=dict(type="add_prompt", session_id=sid, frame_index=int(f), text=phrase,
                                                                  output_prob_thresh=thresh))["outputs"])
            rows.append(dict(phrase=phrase, frame=int(f), n=int(len(out["ids"])), probs=[round(float(p), 3) for p in out["probs"]],
                             boxes_xywh=[[round(float(v), 3) for v in b] for b in out["boxes"]],
                             areas=[round(float(m.mean()), 4) for m in out["masks"]]))
            print(f"  probe '{phrase}' f{f}: n={rows[-1]['n']} probs={rows[-1]['probs']} areas={rows[-1]['areas']}", flush=True)
    return rows


def best_phrase(rows, phrases):
    best = None
    for r in rows:
        if r["phrase"] in phrases and r["n"] > 0:
            score = max(r["probs"])
            if best is None or score > best[0]:
                best = (score, r["phrase"], r["frame"])
    return best  # (prob, phrase, frame) or None


def track_phrase(predictor, sid, phrase, prompt_frame, thresh):
    """prompt_frame is informational only (where the probe fired): the multiplex tracker applies the text prompt
    to ALL frames, only propagates forward from start_frame_index (a backward pass just reads a cache), and
    prompting on frame k>0 while propagating from 0 trips an assertion at frame k when the tracker holds more
    objects than the cached single-frame detection ('trk_masks and trk_obj_ids ... 1 vs 2', job 46755541).
    So: text prompt on frame 0, one forward propagation from 0 (the notebook-verified flow)."""
    predictor.handle_request(request=dict(type="reset_session", session_id=sid))
    predictor.handle_request(request=dict(type="add_prompt", session_id=sid, frame_index=0, text=phrase, output_prob_thresh=thresh))
    outputs = {}
    for r in predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=sid, start_frame_index=0,
                                                          propagation_direction="forward", output_prob_thresh=thresh)):
        outputs[int(r["frame_index"])] = clean_out(r["outputs"])
    return outputs


def wrist_fallback(video_path, n_frames):
    """SAM 2-style tracker with the validated box + clicks on frame 127 (relative coords)."""
    import torch
    from sam3.model_builder import build_sam3_video_model
    rw, rh = WRIST_FALLBACK["ref_wh"]
    pts = torch.tensor([[x / rw, y / rh] for x, y in WRIST_FALLBACK["points_px"]], dtype=torch.float32)
    labels = torch.tensor([1] * len(WRIST_FALLBACK["points_px"]), dtype=torch.int32)
    bx = WRIST_FALLBACK["box_px"]
    box = np.array([[bx[0] / rw, bx[1] / rh, bx[2] / rw, bx[3] / rh]], dtype=np.float32)
    pf = min(WRIST_FALLBACK["frame"], n_frames - 1)
    model = build_sam3_video_model()
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    state = tracker.init_state(video_path=video_path)
    tracker.add_new_points_or_box(inference_state=state, frame_idx=pf, obj_id=1, points=pts, labels=labels, box=box)
    outputs = {}
    for rev in (True, False):
        for fidx, oids, _, vmasks, _ in tracker.propagate_in_video(state, start_frame_idx=pf, max_frame_num_to_track=state["num_frames"],
                                                                  reverse=rev, propagate_preflight=True):
            m = (vmasks[0][0] > 0).cpu().numpy()
            if m.any():
                outputs[int(fidx)] = dict(ids=np.array([1]), probs=np.array([1.0]), boxes=np.zeros((1, 4)), masks=m[None])
            else:
                outputs[int(fidx)] = dict(ids=np.zeros(0, int), probs=np.zeros(0), boxes=np.zeros((0, 4)), masks=np.zeros((0,) + m.shape, bool))
    del model, tracker, state
    torch.cuda.empty_cache()
    return outputs, pf


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="RAIL+80edfcb1+2023-07-14-14h-28m-45s")
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--thresh", type=float, default=0.3, help="output_prob_thresh for probe and tracking")
    ap.add_argument("--probe_frames", default="0,32,64,96,127")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--wrist_fallback", choices=["auto", "always", "never"], default="auto")
    ap.add_argument("--cams", default="wrist,ext1,ext2")
    ap.add_argument("--selftest", action="store_true", help="CPU-only: render a few frames with a fake mask through ffmpeg and exit")
    args = ap.parse_args()

    ep = args.episode
    raw = os.path.join(RAW_ROOT, ep)
    meta = json.load(open(os.path.join(raw, f"metadata_{ep}.json")))
    serials = dict(wrist=meta["wrist_cam_serial"], ext1=meta["ext1_cam_serial"], ext2=meta["ext2_cam_serial"])
    cams = [c for c in args.cams.split(",") if c]
    videos = {c: os.path.join(raw, "recordings", "MP4", f"{serials[c]}.mp4") for c in cams}
    for c, p in videos.items():
        if not os.path.exists(p):
            sys.exit(f"missing video for {c}: {p}")
    out_dir = os.path.join(args.out_root, ep)
    os.makedirs(out_dir, exist_ok=True)
    print(f"episode {ep}\nvideos {videos}\nout {out_dir}", flush=True)

    frames = {c: read_frames(videos[c]) for c in cams}
    for c in cams:
        print(f"{c}: {len(frames[c])} frames {frames[c][0].shape[1]}x{frames[c][0].shape[0]}", flush=True)

    if args.selftest:
        c = cams[0]
        H, W = frames[c][0].shape[:2]
        fake = {}
        for i in range(6):
            m = np.zeros((H, W), bool)
            m[H // 2:, W // 3:(2 * W) // 3] = True
            fake[i] = dict(ids=np.array([3]), probs=np.array([0.87]), boxes=np.array([[1 / 3, 0.5, 1 / 3, 0.5]]), masks=m[None]) if i % 3 else None
        s = write_camera_outputs(c, serials[c], "selftest", "selftest", frames[c][:6], fake, out_dir, args.fps)
        side_by_side(cams, {k: v[:6] for k, v in frames.items()}, {k: fake for k in cams}, {k: "selftest" for k in cams},
                     os.path.join(out_dir, "selftest_side_by_side.mp4"), args.fps)
        print("SELFTEST_OK", json.dumps(s, indent=1))
        return

    import torch
    from sam3.model_builder import build_sam3_predictor
    t0 = time.time()
    predictor = build_sam3_predictor(version="sam3.1")
    print(f"predictor built in {time.time()-t0:.0f}s", flush=True)

    probe_frames = [int(x) for x in args.probe_frames.split(",")]
    probe_all, summaries, outputs_primary, phrase_primary = {}, [], {}, {}
    for c in cams:
        n = len(frames[c])
        pf = [f for f in probe_frames if f < n]
        print(f"\n=== {c} ({serials[c]}) ===", flush=True)
        sid = predictor.handle_request(request=dict(type="start_session", resource_path=videos[c]))["session_id"]
        t0 = time.time()
        rows = probe_phrases(predictor, sid, GRIPPER_PHRASES + ARM_PHRASES, pf, args.thresh)
        probe_all[c] = rows
        print(f"probe done in {time.time()-t0:.0f}s", flush=True)
        for tag, phrases in (("gripper", GRIPPER_PHRASES), ("arm", ARM_PHRASES)):
            best = best_phrase(rows, phrases)
            if best is None:
                print(f"[{c}/{tag}] no phrase fired at thresh {args.thresh}", flush=True)
                summaries.append(dict(cam=c, serial=serials[c], tag=tag, phrase=None, frames_with_detection=0, n_frames=n, note="no text phrase fired"))
                if tag == "gripper":
                    outputs_primary[c], phrase_primary[c] = {}, "none"
                continue
            prob, phrase, f = best
            t0 = time.time()
            try:
                outs = track_phrase(predictor, sid, phrase, f, args.thresh)
            except Exception as e:  # keep going: one failed pass must not kill the other cameras
                import traceback; traceback.print_exc()
                print(f"[{c}/{tag}] TRACKING FAILED for '{phrase}' (prompt frame {f}): {e!r}", flush=True)
                summaries.append(dict(cam=c, serial=serials[c], tag=tag, phrase=phrase, frames_with_detection=0, n_frames=n, note=f"tracking failed: {e!r}"))
                if tag == "gripper":
                    outputs_primary[c], phrase_primary[c] = {}, phrase + " (failed)"
                continue
            print(f"[{c}/{tag}] tracked '{phrase}' (prompt frame {f}, probe prob {prob:.2f}) in {time.time()-t0:.0f}s", flush=True)
            summaries.append(write_camera_outputs(c, serials[c], phrase, tag, frames[c], outs, out_dir, args.fps,
                                                  extra=dict(prompt_frame=f, probe_prob=prob, method="sam3.1 text prompt + propagate")))
            if tag == "gripper":
                outputs_primary[c], phrase_primary[c] = outs, phrase
        predictor.handle_request(request=dict(type="close_session", session_id=sid))

    json.dump(probe_all, open(os.path.join(out_dir, "probe.json"), "w"), indent=1)

    if "wrist" in cams and (args.wrist_fallback == "always" or (args.wrist_fallback == "auto" and not outputs_primary.get("wrist"))):
        print("\n=== wrist fallback: SAM 2-style tracker, box + clicks ===", flush=True)
        del predictor
        torch.cuda.empty_cache()
        t0 = time.time()
        outs, pf = wrist_fallback(videos["wrist"], len(frames["wrist"]))
        print(f"fallback tracked in {time.time()-t0:.0f}s", flush=True)
        summaries.append(write_camera_outputs("wrist", serials["wrist"], "box+clicks(fallback)", "gripper_boxclick", frames["wrist"], outs, out_dir, args.fps,
                                              extra=dict(prompt_frame=pf, method="sam3 tracker, box+3 clicks (validated recipe)", recipe=WRIST_FALLBACK)))
        if not outputs_primary.get("wrist"):
            outputs_primary["wrist"], phrase_primary["wrist"] = outs, "box+clicks"

    if len(cams) > 1:
        side_by_side(cams, frames, outputs_primary, phrase_primary, os.path.join(out_dir, "all_cams_gripper_side_by_side.mp4"), args.fps)
    json.dump(dict(episode=ep, serials=serials, thresh=args.thresh, summaries=summaries), open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=str)
    print("\nSUMMARY")
    for s in summaries:
        print(f"  {s['cam']:5s} {s['tag']:18s} phrase={s.get('phrase')!r:22} frames={s.get('frames_with_detection')}/{s.get('n_frames')}")
    print("DONE", out_dir)


if __name__ == "__main__":
    main()
