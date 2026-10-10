#!/usr/bin/env python
"""DIAGNOSTIC scoring of seedstudy gripper masks against the kinematic ground truth (GT is used here for scoring ONLY).

The wrist camera sits on the gripper's rigid body, so its centre (store cam/NNNNNN.npz 'pose', camera-to-base) projected
with the PointWorld optimized_extrinsics (base-to-camera) and the factory intrinsics must fall on / next to a correct
gripper mask. Per scene and exterior camera, for a mask file following the seedstudy contract:
  present   fraction of frames with a non-empty mask
  hit@r     fraction of frames (with the GT point in view) where the mask dilated by r px contains the projected point
  miss_far  fraction of present frames whose mask centroid is > far px from the projected point (mask on something else)
  d_med     median distance (px) from the mask centroid to the projected point over present frames
  spill     mean fraction of mask pixels farther than `far` px from the projected point (over-inclusion: whole arm)
  area      mean mask area (fraction of the image)

    OMP_NUM_THREADS=1 python seedstudy_score_gt.py --method text_arm_sam3 [--suffix __text_arm_box_masks.npz] [--draw EP]
"""
import argparse
import json
import os

import cv2
import numpy as np

SEEDSTUDY = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy"
STORE = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
RAW = "/gpfs/scratch/etur59/koc821022/vggt_cache/raw"
AUDIT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/gt_audit_2026-09-08"


def load_masks(path):
    z = np.load(path)
    T, H, W = [int(v) for v in z["shape"]]
    return np.unpackbits(z["union"], axis=2)[:, :, :W].astype(bool)


def gt_projections(ep, serial):
    """(T, 2) pixel coords of the wrist-camera centre in the exterior camera <serial>, plus (T,) in-view flag."""
    intr = json.load(open(os.path.join(AUDIT, "docs", "hf_intrinsics.json")))[ep][serial]
    fx, cx, fy, cy = intr["cameraMatrix"]
    W, H = intr["width"], intr["height"]
    E = np.array(json.load(open(os.path.join(AUDIT, "pointworld", "droid", "cameras", f"{ep}_cameras.json")))[serial]["optimized_extrinsics"], float)
    cam_dir = os.path.join(STORE, ep, "dense", "cam")
    files = sorted(f for f in os.listdir(cam_dir) if f.endswith(".npz"))
    uv, inview = np.zeros((len(files), 2)), np.zeros(len(files), bool)
    for i, f in enumerate(files):
        c = np.load(os.path.join(cam_dir, f))["pose"][:3, 3]
        p = E[:3, :3] @ c + E[:3, 3]
        if p[2] > 0.05:
            u, v = fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy
            uv[i] = (u, v)
            inview[i] = 0 <= u < W and 0 <= v < H
    return uv, inview


def score_masks(masks, uv, inview, r=25, far=150):
    T = min(len(masks), len(uv))
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    present = np.array([masks[t].any() for t in range(T)])
    hit = np.zeros(T, bool)
    dist = np.full(T, np.nan)
    spill = np.full(T, np.nan)   # fraction of mask pixels farther than `far` px from the projected point (over-inclusion, e.g. the whole arm)
    area = np.full(T, np.nan)
    for t in range(T):
        if not present[t]:
            continue
        ys, xs = np.nonzero(masks[t])
        dist[t] = float(np.hypot(xs.mean() - uv[t, 0], ys.mean() - uv[t, 1]))
        if inview[t]:
            spill[t] = float(np.mean(np.hypot(xs - uv[t, 0], ys - uv[t, 1]) > far))
        area[t] = len(xs) / (masks.shape[1] * masks.shape[2])
        if inview[t]:
            u, v = int(round(uv[t, 0])), int(round(uv[t, 1]))
            x0, y0, x1, y1 = max(0, u - r), max(0, v - r), min(masks.shape[2], u + r + 1), min(masks.shape[1], v + r + 1)
            hit[t] = masks[t, y0:y1, x0:x1].any()
    n_view = int(inview[:T].sum())
    return dict(T=T, in_view=n_view, present=float(present.mean()), present_inview=float(present[inview[:T]].mean()) if n_view else None,
                hit=float(hit[inview[:T]].mean()) if n_view else None,
                miss_far=float(np.nanmean(dist[present] > far)) if present.any() else None,
                d_med=float(np.nanmedian(dist[present])) if present.any() else None,
                spill=float(np.nanmean(spill)) if np.isfinite(spill).any() else None,
                area=float(np.nanmean(area)) if np.isfinite(area).any() else None,
                hit_frames=int(hit.sum()), present_frames=int(present.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--suffix", default=None, help="default __<method>_masks.npz")
    ap.add_argument("--scenes", default="/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt")
    ap.add_argument("--r", type=int, default=25)
    ap.add_argument("--far", type=float, default=150)
    ap.add_argument("--draw", default=None, help="episode: write <root>/<EP>/gtproj_<cam>_f*.jpg with the projected point and the mask")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()
    root = args.root or os.path.join(SEEDSTUDY, args.method)
    suffix = args.suffix or f"__{args.method}_masks.npz"
    eps = [l.strip() for l in open(args.scenes) if l.strip()]
    allrows = {}
    agg = dict(hit=[], present=[], miss_far=[], d_med=[], spill=[], area=[])
    for ep in eps:
        meta = json.load(open(os.path.join(RAW, ep, f"metadata_{ep}.json")))
        line = ep[:40]
        for cam in ("ext1", "ext2"):
            ser = meta[f"{cam}_cam_serial"]
            path = os.path.join(root, ep, f"{cam}_{ser}{suffix}")
            if not os.path.isfile(path):
                line += f" | {cam} MISSING"
                continue
            masks = load_masks(path)
            uv, inview = gt_projections(ep, ser)
            s = score_masks(masks, uv, inview, args.r, args.far)
            allrows[f"{ep}/{cam}"] = s
            for k in agg:
                if s.get(k) is not None:
                    agg[k].append(s[k])
            line += (f" | {cam} view {s['in_view']}/{s['T']} present {s['present']:.2f} hit@{args.r} {s['hit'] if s['hit'] is None else round(s['hit'], 2)} "
                     f"far {s['miss_far'] if s['miss_far'] is None else round(s['miss_far'], 2)} d_med {s['d_med'] if s['d_med'] is None else round(s['d_med'])} "
                     f"spill {s['spill'] if s['spill'] is None else round(s['spill'], 2)} area {s['area'] if s['area'] is None else round(100 * s['area'], 1)}%")
            if args.draw == ep:
                cap = cv2.VideoCapture(os.path.join(RAW, ep, "recordings", "MP4", f"{ser}.mp4"))
                T = s["T"]
                for f in sorted({int(T * q) for q in (0.1, 0.3, 0.5, 0.7, 0.9)}):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                    ok, img = cap.read()
                    if not ok:
                        continue
                    ov = img.copy()
                    ov[masks[f]] = (255, 128, 0)
                    img = cv2.addWeighted(ov, 0.4, img, 0.6, 0)
                    if inview[f]:
                        cv2.circle(img, (int(uv[f, 0]), int(uv[f, 1])), 12, (0, 255, 0), 3)
                        cv2.circle(img, (int(uv[f, 0]), int(uv[f, 1])), args.r, (0, 255, 0), 1)
                    cv2.putText(img, f"{cam} f{f} GT wrist-cam centre=green mask=orange", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imwrite(os.path.join(root, ep, f"gtproj_{cam}_f{f:04d}.jpg"), cv2.resize(img, (960, 540)))
        print(line)
    print(f"\nMEAN over {len(agg['hit'])} scene-cams ({args.method}{suffix}): hit@{args.r} {np.mean(agg['hit']):.3f}  present {np.mean(agg['present']):.3f}  "
          f"miss_far {np.mean(agg['miss_far']):.3f}  d_med {np.median(agg['d_med']):.0f}px  spill>{args.far:.0f}px {np.mean(agg['spill']):.3f}  area {100 * np.mean(agg['area']):.2f}%")
    if args.json_out:
        json.dump(dict(rows=allrows, mean=dict(hit=float(np.mean(agg['hit'])), present=float(np.mean(agg['present'])), miss_far=float(np.mean(agg['miss_far'])),
                                                d_med=float(np.median(agg['d_med']))), r=args.r, far=args.far), open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
