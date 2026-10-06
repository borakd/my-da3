#!/usr/bin/env python
"""Review videos for the scenes where the gripper is never visible to both exterior cameras.

Each video stacks 10 scenes vertically; each row shows the three camera feeds side by side
(wrist | ext1 | ext2) playing simultaneously from frame 0. Shorter episodes hold their last
frame. The row label carries the episode name, the frame counter, and the kinematic visibility
fraction of each exterior camera for that frame (what the detector believed).

Usage: make_failed_scenes_videos.py --list failed_scenes.txt --out_dir <dir> [--tile_w 320]
"""
import argparse
import json
import os

import cv2
import numpy as np

import birth_frames as bf

RAW_ROOT = "/leonardo_scratch/large/userexternal/bdursun0/robotseg_demo/raw"
FPS = 15


class Feed:
    def __init__(self, mp4):
        self.cap = cv2.VideoCapture(mp4)
        self.last = None
        self.ended = False

    def read(self):
        if not self.ended:
            ok, f = self.cap.read()
            if ok:
                self.last = f
            else:
                self.ended = True
        return self.last


def kin_curves(ep):
    r = bf.analyze_episode(ep)
    return r.get("curves", {"ext1": [], "ext2": []}), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_video", type=int, default=10)
    ap.add_argument("--tile_w", type=int, default=320)
    args = ap.parse_args()
    eps = [l.strip() for l in open(args.list) if l.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    tw, th = args.tile_w, args.tile_w * 9 // 16
    bar = 22
    import imageio

    for vi in range(0, len(eps), args.per_video):
        group = eps[vi:vi + args.per_video]
        feeds, curves, lens, labels = [], [], [], []
        for ep in group:
            meta = json.load(open(f"{bf.META_DIR}/{ep}.json"))
            sns = [meta["wrist_cam_serial"], meta["ext1_cam_serial"], meta["ext2_cam_serial"]]
            feeds.append([Feed(f"{RAW_ROOT}/{ep}/recordings/MP4/{sn}.mp4") for sn in sns])
            c, r = kin_curves(ep)
            curves.append(c)
            lens.append(r.get("n_frames", 0))
            labels.append(f"{ep}  ext1 max {r.get('ext1_max_frac', 0):.2f}  ext2 max {r.get('ext2_max_frac', 0):.2f}")
        n = max(lens)
        out = f"{args.out_dir}/failed_scenes_video_{vi // args.per_video + 1}.mp4"
        H = len(group) * (th + bar)
        W = 3 * tw
        print(f"{out}: {len(group)} scenes, {n} frames, {W}x{H}", flush=True)
        with imageio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=1) as wr:
            for t in range(n):
                rows = []
                for i, ep in enumerate(group):
                    tiles = []
                    for f in feeds[i]:
                        img = f.read()
                        if img is None:
                            img = np.zeros((720, 1280, 3), np.uint8)
                        tiles.append(cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA))
                    row = np.hstack(tiles)
                    lab = np.zeros((bar, W, 3), np.uint8)
                    k1 = curves[i]["ext1"][t] if t < len(curves[i]["ext1"]) else float("nan")
                    k2 = curves[i]["ext2"][t] if t < len(curves[i]["ext2"]) else float("nan")
                    held = "  (held)" if t >= lens[i] else ""
                    txt = f"{labels[i]}   f{min(t, lens[i]-1):4d}/{lens[i]}{held}   kin ext1 {k1:.2f}  ext2 {k2:.2f}"
                    cv2.putText(lab, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    for j, name in enumerate(("wrist", "ext1", "ext2")):
                        cv2.putText(row, name, (j * tw + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                    rows.append(np.vstack([lab, row]))
                frame = np.vstack(rows)
                wr.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for fl in feeds:
            for f in fl:
                f.cap.release()
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
