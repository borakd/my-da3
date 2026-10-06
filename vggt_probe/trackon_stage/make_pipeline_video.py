"""Showcase video of the full pipeline for one DROID scene, both exterior cameras side by side:
  frames < birth frame    -> raw frame, "waiting for birth frame" (kinematic visibility check)
  birth frame (held)      -> RobotSeg gripper mask overlay + the query points sampled inside it
                              (the ONLY frame RobotSeg ever ran on)
  frames > birth frame    -> Track-On2 causal tracks (dots + short trails; hollow = not visible)
Usage: make_pipeline_video.py <episode> [--out <mp4>] [--hold 20]
"""
import argparse
import json
import os
import sys

import av
import cv2
import imageio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # vggt_probe: birth_frames.py
import birth_frames as bf  # noqa: E402

O = bf.OUT_ROOT
RAW = "/leonardo_scratch/large/userexternal/bdursun0/robotseg_demo/raw"
PANEL = (960, 540)
HDR = 64
FPS = 15.0
TRAIL = 12


def decode_all(mp4):
    with av.open(mp4) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        return [f.to_ndarray(format="rgb24") for f in c.decode(s)]


def colours(n):
    hsv = np.zeros((n, 1, 3), np.uint8)
    hsv[:, 0, 0] = np.linspace(0, 179, n).astype(np.uint8)
    hsv[:, 0, 1:] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[:, 0, :]


def put(img, text, xy, scale=0.7, col=(255, 255, 255), th=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hold", type=int, default=20, help="output frames to hold on the birth frame")
    ap.add_argument("--tracks_root", default=f"{O}/trackon_run/tracks", help="dir holding <ep>/<cam>_f<t>.npz")
    ap.add_argument("--label", default="Track-On-R", help="tracker name shown in the overlay")
    args = ap.parse_args()
    ep = args.episode
    births = {r["episode"]: r for r in json.load(open(f"{O}/birth_frames.json"))}
    b = births[ep]
    B = b["birth_f050"]
    meta = json.load(open(f"{bf.META_DIR}/{ep}.json"))
    cams = {}
    for cam in ("ext1", "ext2"):
        sn = meta[f"{cam}_cam_serial"]
        frames = decode_all(f"{RAW}/{ep}/recordings/MP4/{sn}.mp4")
        mask = cv2.imread(f"{O}/masks_run/masks/{ep}/{cam}_f{B:05d}.png", 0) > 0
        z = np.load(f"{args.tracks_root}/{ep}/{cam}_f{B:05d}.npz")
        cams[cam] = dict(frames=frames, mask=mask, tracks=z["tracks"], vis=z["visibility"], queries=z["queries"],
                         kin=None)
    curves = bf.analyze_episode(ep)["curves"]
    T = min(len(cams["ext1"]["frames"]), len(cams["ext2"]["frames"]), b["n_frames"])
    out = args.out or f"{O}/examples/{ep}__pipeline.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    W = 2 * PANEL[0]
    sx, sy = PANEL[0] / 1280, PANEL[1] / 720

    def panel(cam, t, stage):
        c = cams[cam]
        img = cv2.resize(c["frames"][t], PANEL, interpolation=cv2.INTER_AREA).copy()
        k1 = curves[cam][t] if t < len(curves[cam]) else float("nan")
        if stage == "before":
            img = (img * 0.55).astype(np.uint8)
            put(img, f"{cam}   frame {t}   gripper model in view: {100*k1:.0f}%", (10, 30), 0.7, (200, 200, 200))
            put(img, "waiting for the birth frame (kinematic check, no model run)", (10, PANEL[1] - 16), 0.6, (200, 200, 200))
        elif stage == "birth":
            m = cv2.resize(c["mask"].astype(np.uint8), PANEL, interpolation=cv2.INTER_NEAREST) > 0
            img[m] = (0.4 * img[m] + 0.6 * np.array([255, 120, 0])).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (255, 255, 255), 2)
            for (x, y) in c["queries"][:, 1:]:
                cv2.circle(img, (int(x * sx), int(y * sy)), 3, (0, 255, 0), -1)
            put(img, f"{cam}   BIRTH FRAME {t}   gripper model in view: {100*k1:.0f}%", (10, 30), 0.7, (255, 220, 0))
            put(img, f"RobotSeg gripper mask (the only frame segmented) -> {len(c['queries'])} query points", (10, PANEL[1] - 16), 0.6, (255, 220, 0))
        else:
            tr, vi = c["tracks"], c["vis"]
            N = tr.shape[1]
            cols = colours(N)
            for n in range(N):
                for s in range(max(B, t - TRAIL), t):
                    p0, p1 = tr[s, n], tr[s + 1, n]
                    if np.isnan(p0).any() or np.isnan(p1).any():
                        continue
                    cv2.line(img, (int(p0[0] * sx), int(p0[1] * sy)), (int(p1[0] * sx), int(p1[1] * sy)),
                             tuple(int(v) for v in cols[n]), 1, cv2.LINE_AA)
                p = tr[t, n]
                if np.isnan(p).any():
                    continue
                col = tuple(int(v) for v in cols[n])
                if vi[t, n]:
                    cv2.circle(img, (int(p[0] * sx), int(p[1] * sy)), 4, col, -1, cv2.LINE_AA)
                else:
                    cv2.circle(img, (int(p[0] * sx), int(p[1] * sy)), 4, col, 1, cv2.LINE_AA)
            nv = int(vi[t].sum())
            put(img, f"{cam}   frame {t}   {args.label} (causal): {nv}/{N} points visible", (10, 30), 0.7, (120, 255, 120))
            put(img, "tracks started from the birth-frame mask; no re-segmentation", (10, PANEL[1] - 16), 0.6, (120, 255, 120))
        return img

    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=1) as wr:
        for t in range(T):
            stage = "before" if t < B else ("birth" if t == B else "after")
            reps = args.hold if stage == "birth" else 1
            row = np.hstack([panel("ext1", t, stage), panel("ext2", t, stage)])
            hdr = np.zeros((HDR, W, 3), np.uint8)
            put(hdr, f"{ep}", (10, 26), 0.7, (255, 255, 255), 1)
            steps = "1) birth frame = first frame with >=50% of the kinematic gripper model inside BOTH exterior views" \
                    + f" (B={B})   2) RobotSeg gripper mask on frame B only   3) {args.label} from B to the end ({T} frames)"
            put(hdr, steps, (10, 52), 0.52, (180, 220, 255), 1)
            frame = np.vstack([hdr, row])
            for _ in range(reps):
                wr.append_data(frame)
    print("wrote", out, "frames", T, "birth", B)


if __name__ == "__main__":
    main()
