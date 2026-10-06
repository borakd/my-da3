"""Side-by-side qualitative comparison video of two Track-On checkpoints on one scene.
Layout: 2 rows (checkpoints) x 2 columns (ext1, ext2); identical query points; tracks drawn from the
birth frame on (dots + short trails, hollow = tracker says not visible); frames before birth dimmed.
Usage: make_comparison_video.py <episode> --ckpts track_on_r trackon2 [--labels "Track-On-R" "Track-On2"]
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
CMP = f"{O}/comparisons"
RAW = "/leonardo_scratch/large/userexternal/bdursun0/robotseg_demo/raw"
PANEL = (960, 540)
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
    ap.add_argument("--ckpts", nargs=2, default=["track_on_r", "trackon2"])
    ap.add_argument("--labels", nargs=2, default=["Track-On-R (track_on_r.pt)", "Track-On2 (trackon2_dinov3.pt)"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ep = args.episode
    B = {r["episode"]: r for r in json.load(open(f"{O}/birth_frames.json"))}[ep]["birth_f050"]
    meta = json.load(open(f"{bf.META_DIR}/{ep}.json"))
    frames = {cam: decode_all(f"{RAW}/{ep}/recordings/MP4/{meta[f'{cam}_cam_serial']}.mp4") for cam in ("ext1", "ext2")}
    data = {(c, cam): np.load(f"{CMP}/{c}/tracks/{ep}/{cam}_f{B:05d}.npz") for c in args.ckpts for cam in ("ext1", "ext2")}
    T = min(len(frames["ext1"]), len(frames["ext2"]))
    sx, sy = PANEL[0] / 1280, PANEL[1] / 720
    out = args.out or f"{CMP}/{ep}__compare.mp4"
    HDR = 40

    def panel(c, label, cam, t):
        img = cv2.resize(frames[cam][t], PANEL, interpolation=cv2.INTER_AREA).copy()
        if t < B:
            img = (img * 0.55).astype(np.uint8)
            put(img, f"{label}  {cam}  frame {t}  (before birth frame {B})", (10, 30), 0.65, (200, 200, 200))
            return img
        z = data[(c, cam)]
        tr, vi = z["tracks"], z["visibility"]
        N = tr.shape[1]
        cols = colours(N)
        for n in range(N):
            for s in range(max(B, t - TRAIL), min(t, len(tr) - 1)):
                p0, p1 = tr[s, n], tr[s + 1, n]
                if np.isnan(p0).any() or np.isnan(p1).any():
                    continue
                cv2.line(img, (int(p0[0] * sx), int(p0[1] * sy)), (int(p1[0] * sx), int(p1[1] * sy)), tuple(int(v) for v in cols[n]), 1, cv2.LINE_AA)
            if t >= len(tr):
                continue
            p = tr[t, n]
            if np.isnan(p).any():
                continue
            col = tuple(int(v) for v in cols[n])
            if vi[t, n]:
                cv2.circle(img, (int(p[0] * sx), int(p[1] * sy)), 4, col, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (int(p[0] * sx), int(p[1] * sy)), 4, col, 1, cv2.LINE_AA)
        nv = int(vi[t].sum()) if t < len(vi) else 0
        put(img, f"{label}  {cam}  frame {t}  visible {nv}/{N}", (10, 30), 0.65, (120, 255, 120))
        return img

    with imageio.get_writer(out, fps=15.0, codec="libx264", quality=7, macro_block_size=1) as wr:
        for t in range(T):
            rows = [np.hstack([panel(c, lab, "ext1", t), panel(c, lab, "ext2", t)]) for c, lab in zip(args.ckpts, args.labels)]
            hdr = np.zeros((HDR, 2 * PANEL[0], 3), np.uint8)
            put(hdr, f"{ep}   birth frame {B}   same RobotSeg birth-frame mask and query points for both trackers; rows = checkpoints", (10, 27), 0.6, (255, 255, 255), 1)
            wr.append_data(np.vstack([hdr] + rows))
    print("wrote", out)


if __name__ == "__main__":
    main()
