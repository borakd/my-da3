#!/usr/bin/env python
"""DIAGNOSTIC overlay: one exterior frame with the METHOD mask (orange), the REFERENCE mask (cyan, overlap = white) and
the projected kinematic GT lens (green circle; GT is scoring-only). Explains a low IoU-vs-reference at a glance.

    OMP_NUM_THREADS=1 python seedstudy_overlay_vs_ref.py --method motion_sam3 --ref_method gtbox_sam3 \
        --episode EP --cam ext2 [--frames 100,300] [--out DIR]     -> DIR/<cam>_f<frame>_vs_<ref>.jpg (960x540)
"""
import argparse, json, os
import cv2
import numpy as np
from seedstudy_summary import RAW_ROOT, STORE_ROOT, AUDIT, SEEDSTUDY, W, H, load_intrinsics, unpack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--ref_method", default="gtbox_sam3")
    ap.add_argument("--episode", required=True); ap.add_argument("--cam", default="ext1")
    ap.add_argument("--frames", default=None, help="comma list; default 5 evenly spaced frames where the method mask is non-empty")
    ap.add_argument("--out", default=None, help="default <root>/mask_accuracy/overlays/<EP>")
    ap.add_argument("--root", default=None, help="method root; default SEEDSTUDY/<method>")
    ap.add_argument("--ref_root", default=None, help="reference root; default SEEDSTUDY/<ref_method>")
    a = ap.parse_args()
    ep, cam = a.episode, a.cam
    root = a.root or f"{SEEDSTUDY}/{a.method}"
    ref_root = a.ref_root or f"{SEEDSTUDY}/{a.ref_method}"
    s = str(json.load(open(f"{RAW_ROOT}/{ep}/metadata_{ep}.json"))[f"{cam}_cam_serial"])
    u, _, (T, _, _) = unpack(f"{root}/{ep}/{cam}_{s}__{a.method}_masks.npz")
    ref_p = f"{ref_root}/{ep}/{cam}_{s}__{a.ref_method}_masks.npz"
    v = unpack(ref_p)[0] if os.path.isfile(ref_p) else None
    intr = load_intrinsics(ep)[s]; fx, cx, fy, cy = intr["cameraMatrix"]
    E = np.array(json.load(open(f"{AUDIT}/pointworld/droid/cameras/{ep}_cameras.json"))[s]["optimized_extrinsics"], np.float64)
    nonempty = np.where(u.reshape(T, -1).any(1))[0]
    frames = [int(x) for x in a.frames.split(",")] if a.frames else sorted({int(nonempty[int(q * (len(nonempty) - 1))]) for q in (0.05, 0.3, 0.5, 0.7, 0.95)})
    out = a.out or f"{root}/mask_accuracy/overlays/{ep}"
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(f"{RAW_ROOT}/{ep}/recordings/MP4/{s}.mp4")
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        if not ok:
            print("no frame", f); continue
        ov = img.copy()
        m, r = u[f], (v[f] if v is not None and f < len(v) else np.zeros_like(u[f]))
        ov[m & ~r] = (0, 128, 255); ov[r & ~m] = (255, 255, 0); ov[m & r] = (255, 255, 255)
        img = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
        c = np.load(f"{STORE_ROOT}/{ep}/dense/cam/{f:06d}.npz")["pose"][:3, 3]
        p = E[:3, :3] @ c + E[:3, 3]
        if p[2] > 0.05:
            px, py = fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy
            cv2.circle(img, (int(px), int(py)), 14, (0, 255, 0), 3)
        inter, uni = int((m & r).sum()), int((m | r).sum())
        cv2.putText(img, f"{cam} f{f} {a.method}=orange {a.ref_method}=cyan overlap=white GTlens=green IoU={inter / max(uni, 1):.2f} area m={int(m.sum())} r={int(r.sum())}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        path = f"{out}/{cam}_f{f:04d}_vs_{a.ref_method}.jpg"
        cv2.imwrite(path, cv2.resize(img, (960, 540))); print("wrote", path)


if __name__ == "__main__":
    main()
