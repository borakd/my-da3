#!/usr/bin/env python
"""One mosaic video per seeding method: all 13 smoke scenes playing simultaneously, each tile = ext1 | ext2 at 320x180
with the method's gripper mask filled green (contour in bright green) and the projected kinematic GT wrist-camera lens as a
red dot (reference only). Scenes shorter than the longest freeze on their last frame with an END banner. Empty masks are
tagged NO MASK; missing files are tagged NO MASK FILE.

    OMP_NUM_THREADS=1 python method_mosaic.py --method motion_only --root $OUT/ext_cams/seedstudy/motion_only \
        --suffix motion_only --out $OUT/ext_cams/mosaics/motion_only_smoke13.mp4
"""
import argparse, json, os, sys
import numpy as np, cv2
cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam3_gripper_masks import FfmpegWriter  # noqa: E402

OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
AUDIT = f"{OUT}/vggt_probe/gt_audit_2026-09-08"
RAW = "/gpfs/scratch/etur59/koc821022/vggt_cache/raw"
STORE = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
SCENES = "/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt"
TW, TH = 320, 180          # per-camera tile
COLS, ROWS = 3, 5
BANNER = 30
GREEN, RED, WHITE = (0, 200, 0), (0, 0, 255), (255, 255, 255)


def short(ep):
    lab, uid, ts = ep.split("+")
    return f"{lab[:4]}+{uid[:4]}+{ts[5:16]}"


class Scene:
    def __init__(self, ep, root, suffix):
        self.ep, self.name = ep, short(ep)
        meta = json.load(open(f"{RAW}/{ep}/metadata_{ep}.json"))
        self.serials = [meta["ext1_cam_serial"], meta["ext2_cam_serial"]]
        self.T = len([f for f in os.listdir(f"{STORE}/{ep}/dense/cam") if f.endswith(".npz")])
        self.caps = [cv2.VideoCapture(f"{RAW}/{ep}/recordings/MP4/{s}.mp4") for s in self.serials]
        self.masks = []
        for cam, s in zip(("ext1", "ext2"), self.serials):
            p = f"{root}/{ep}/{cam}_{s}__{suffix}_masks.npz"
            if os.path.exists(p):
                d = np.load(p)
                self.masks.append((d["union"], int(d["shape"][2])))
            else:
                self.masks.append(None)
        intr = json.load(open(f"{AUDIT}/docs/hf_intrinsics.json"))[ep]
        cams = json.load(open(f"{AUDIT}/pointworld/droid/cameras/{ep}_cameras.json"))
        self.P = []
        for s in self.serials:
            fx, cx, fy, cy = intr[s]["cameraMatrix"]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            self.P.append(K @ np.array(cams[s]["optimized_extrinsics"])[:3])
        self.gt = np.array([np.load(f"{STORE}/{ep}/dense/cam/{t:06d}.npz")["pose"][:3, 3] for t in range(self.T)])
        self.last = None
        self.t = -1

    def tile(self, t):
        if t >= self.T:
            if self.last is None:
                self.last = np.zeros((TH, 2 * TW, 3), np.uint8)
            img = self.last.copy()
            cv2.rectangle(img, (0, TH // 2 - 12), (2 * TW, TH // 2 + 12), (60, 60, 60), -1)
            cv2.putText(img, "END", (2 * TW // 2 - 18, TH // 2 + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
            return img
        panels = []
        for i in range(2):
            ok, fr = self.caps[i].read()
            if not ok:
                fr = np.zeros((720, 1280, 3), np.uint8)
            fr = cv2.resize(fr, (TW, TH), interpolation=cv2.INTER_AREA)
            m = self.masks[i]
            tag = None
            if m is None:
                tag = "NO MASK FILE"
            else:
                union, W = m
                if t < union.shape[0]:
                    mk = np.unpackbits(union[t], axis=1)[:, :W].astype(np.uint8)
                    mk = cv2.resize(mk, (TW, TH), interpolation=cv2.INTER_NEAREST)
                    if mk.any():
                        fill = fr.copy(); fill[mk > 0] = (0.55 * fill[mk > 0] + 0.45 * np.array(GREEN)).astype(np.uint8)
                        fr = fill
                        cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(fr, cnts, -1, (0, 255, 0), 1)
                    else:
                        tag = "NO MASK"
                else:
                    tag = "NO MASK"
            # GT lens (reference)
            q = self.P[i] @ np.r_[self.gt[t], 1]
            if q[2] > 0:
                u, v = q[0] / q[2] / 4, q[1] / q[2] / 4
                if 0 <= u < TW and 0 <= v < TH:
                    cv2.circle(fr, (int(u), int(v)), 4, RED, -1); cv2.circle(fr, (int(u), int(v)), 4, WHITE, 1)
            if tag:
                cv2.rectangle(fr, (0, TH - 18), (TW, TH), (0, 0, 140), -1)
                cv2.putText(fr, tag, (6, TH - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
            panels.append(fr)
        img = np.hstack(panels)
        cv2.rectangle(img, (0, 0), (2 * TW, 16), (0, 0, 0), -1)
        cv2.putText(img, f"{self.name}  f{t}/{self.T}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        self.last = img
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--root", required=True); ap.add_argument("--suffix", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--fps", type=int, default=15); ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()
    eps = [l.strip() for l in open(SCENES) if l.strip()]
    scenes = [Scene(ep, a.root, a.suffix) for ep in eps]
    scenes.sort(key=lambda s: s.T)
    Tmax = max(s.T for s in scenes)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    W, H = COLS * 2 * TW, BANNER + ROWS * TH
    writer = FfmpegWriter(a.out, W, H, a.fps)
    for t in range(0, Tmax, a.step):
        canvas = np.zeros((H, W, 3), np.uint8)
        cv2.putText(canvas, f"{a.method}   green = method mask   red dot = GT wrist-camera lens (reference)   frame {t}/{Tmax}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        for k, s in enumerate(scenes):
            r, c = divmod(k, COLS)
            if a.step > 1 and t < s.T:
                for _ in range(a.step - 1):   # keep the captures in step with t
                    for cap in s.caps: cap.grab()
            y0, x0 = BANNER + r * TH, c * 2 * TW
            canvas[y0:y0 + TH, x0:x0 + 2 * TW] = s.tile(t)
        writer.write(canvas)
    writer.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
