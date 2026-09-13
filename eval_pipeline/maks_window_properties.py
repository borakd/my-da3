#!/usr/bin/env python3
"""How does every measurable property of the INPUT relate to where CUT3R's
metrics decline? 100 scenes, 367 decline windows.

Successor to maks_window_physics.py. That script measured 8 properties; this one
measures 24, grouped into five families, and reports them with full names rather
than column abbreviations.

Every property is computed from ground-truth poses/depth and the pixels alone --
no model output is involved in any feature, so nothing here can be circular.

Dropped, with cause:
  outlier_mask fraction   byte-identical to (depth <= 0) on every frame checked,
                          so it duplicated 'no depth measurement'
  sky fraction            sky_mask is identically zero (indoor wrist camera)
  image appearance        measured once over all 100 scenes and found inert, so
                          not recomputed. Cohen's d inside a window was sharpness
                          -0.14, over/under-exposed 0.24, contrast 0.12, texture
                          density -0.06, entropy -0.06, brightness -0.02,
                          saturation -0.01 -- every one far below the motion and
                          overlap effects. Blur, exposure and texture do not
                          explain where the metrics decline; the slight blur that
                          does appear is a symptom of the camera speed.

For each property: the standardised difference (Cohen's d) between frames inside
a decline window and frames elsewhere, the same for the pre-window spans, and the
Spearman correlation with the per-frame aggregate badness. Features are z-scored
WITHIN each scene before the effect sizes so that scene-level differences (one
episode being closer to the table than another) cannot masquerade as an effect.

Outputs <out_dir>/window_properties.{md,png} and per_frame_properties.npz
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_decline_windows import load, badness, METRICS  # noqa: E402

ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"

# (key, printable name, family). Order here is the order in the figure.
PROPS = [
    ("translation_speed",      "Translation speed",                      "Camera motion"),
    ("rotation_speed",         "Rotation speed",                         "Camera motion"),
    ("forward_motion",         "Motion along the viewing direction",     "Camera motion"),
    ("sideways_motion",        "Motion across the viewing direction",    "Camera motion"),
    ("translation_accel",      "Translation acceleration",               "Camera motion"),
    ("rotation_accel",         "Rotation acceleration",                  "Camera motion"),
    ("heading_change",         "Change of travel direction",             "Camera motion"),

    ("flow_magnitude",         "Optical flow magnitude",                 "Apparent motion in the image"),
    ("flow_inconsistency",     "Optical flow forward-backward mismatch", "Apparent motion in the image"),
    ("covisibility",           "Overlap with the previous frame",        "Apparent motion in the image"),
    ("intensity_change",       "Frame-to-frame intensity change",        "Apparent motion in the image"),

    ("depth_median",           "Distance to the scene",                  "Scene geometry"),
    ("depth_spread",           "Depth range within the frame",           "Scene geometry"),
    ("near_fraction",          "Fraction of pixels closer than 12 cm",   "Scene geometry"),
    ("invalid_fraction",       "Fraction of pixels with no depth",       "Scene geometry"),
    ("surface_roughness",      "Surface roughness of the depth map",     "Scene geometry"),


    ("position_in_sequence",   "Position in the sequence",               "Sequence position"),
    ("distance_travelled",     "Distance travelled so far",              "Sequence position"),
]
KEYS = [p[0] for p in PROPS]


def rot_angle(A, B):
    c = (np.trace(A.T @ B) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def scene_properties(scene):
    d = f"{ROOT}/{scene}/dense"
    rgb = sorted(f for f in os.listdir(f"{d}/rgb") if f.endswith(".png"))
    n = len(rgb)
    F = {k: np.full(n, np.nan) for k in KEYS}
    prev = {}
    travelled = 0.0
    for i, fn in enumerate(rgb):
        stem = os.path.splitext(fn)[0]
        bgr = cv2.imread(f"{d}/rgb/{fn}")
        if bgr is None:
            continue
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g = grey.astype(np.float32)

        # ---- apparent motion --------------------------------------------
        if "grey" in prev:
            F["intensity_change"][i] = float(np.abs(g - prev["grey"]).mean() / 255.0)
            fw = cv2.calcOpticalFlowFarneback(prev["grey8"], grey, None,
                                              0.5, 3, 15, 3, 5, 1.2, 0)
            bw = cv2.calcOpticalFlowFarneback(grey, prev["grey8"], None,
                                              0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(fw, axis=2)
            F["flow_magnitude"][i] = float(np.median(mag))
            H, W = grey.shape
            ys, xs = np.mgrid[0:H, 0:W]
            xf = np.clip(np.round(xs + fw[..., 0]).astype(int), 0, W - 1)
            yf = np.clip(np.round(ys + fw[..., 1]).astype(int), 0, H - 1)
            resid = np.linalg.norm(fw + bw[yf, xf], axis=2)
            F["flow_inconsistency"][i] = float(np.median(resid))
        prev["grey"] = g
        prev["grey8"] = grey

        # ---- scene geometry ---------------------------------------------
        z = None
        dp = f"{d}/depth/{stem}.npy"
        if os.path.isfile(dp):
            z = np.load(dp).astype(np.float32)
            valid = z > 0
            F["invalid_fraction"][i] = float(1.0 - valid.mean())
            if valid.any():
                zv = z[valid]
                F["depth_median"][i] = float(np.median(zv))
                F["near_fraction"][i] = float((zv < 0.12).mean())
                F["depth_spread"][i] = float(np.percentile(zv, 95) - np.percentile(zv, 5))
                zm = np.where(valid, z, np.nan)
                dzx = np.abs(np.diff(zm, axis=1))
                dzy = np.abs(np.diff(zm, axis=0))
                both = np.concatenate([dzx[np.isfinite(dzx)], dzy[np.isfinite(dzy)]])
                if both.size:
                    F["surface_roughness"][i] = float(np.mean(both))

        # ---- camera motion + sequence position --------------------------
        cam = f"{d}/cam/{stem}.npz"
        if os.path.isfile(cam):
            npz = np.load(cam)
            P = npz["pose"].astype(np.float64)
            if P.shape == (3, 4):
                P = np.vstack([P, [0, 0, 0, 1]])
            c, R = P[:3, 3], P[:3, :3]
            K = npz["intrinsic"].astype(np.float64)
            if "c" in prev:
                v = c - prev["c"]
                sp = float(np.linalg.norm(v))
                F["translation_speed"][i] = sp
                F["rotation_speed"][i] = rot_angle(prev["R"], R)
                axis = prev["R"][:, 2]                    # optical axis of the previous frame
                fwd = float(abs(v @ axis))
                F["forward_motion"][i] = fwd
                F["sideways_motion"][i] = float(math.sqrt(max(sp * sp - fwd * fwd, 0.0)))
                travelled += sp
                if "v" in prev:
                    F["translation_accel"][i] = float(np.linalg.norm(v - prev["v"]))
                    F["rotation_accel"][i] = abs(F["rotation_speed"][i] - prev["rs"])
                    nv, npv = np.linalg.norm(v), np.linalg.norm(prev["v"])
                    if nv > 1e-9 and npv > 1e-9:
                        F["heading_change"][i] = float(np.degrees(np.arccos(
                            np.clip((v @ prev["v"]) / (nv * npv), -1, 1))))
                prev["v"], prev["rs"] = v, F["rotation_speed"][i]

                # overlap: reproject this frame's points into the previous camera
                if z is not None:
                    valid = z > 0
                    step = 4
                    vm = valid[::step, ::step]
                    if vm.any():
                        H, W = z.shape
                        ys, xs = np.mgrid[0:H:step, 0:W:step]
                        zz = z[::step, ::step][vm]
                        X = (xs[vm] - K[0, 2]) / K[0, 0] * zz
                        Y = (ys[vm] - K[1, 2]) / K[1, 1] * zz
                        pts = np.stack([X, Y, zz], 1)
                        world = (R @ pts.T).T + c
                        Pp = np.eye(4); Pp[:3, :3] = prev["R"]; Pp[:3, 3] = prev["c"]
                        loc = (np.linalg.inv(Pp)[:3, :3] @ world.T).T + np.linalg.inv(Pp)[:3, 3]
                        front = loc[:, 2] > 1e-6
                        u = K[0, 0] * loc[front, 0] / loc[front, 2] + K[0, 2]
                        vv = K[1, 1] * loc[front, 1] / loc[front, 2] + K[1, 2]
                        inb = (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
                        F["covisibility"][i] = float(inb.sum() / max(len(loc), 1))
            F["distance_travelled"][i] = travelled
            prev["c"], prev["R"] = c, R
        F["position_in_sequence"][i] = i / max(n - 1, 1)
    return scene, n, F


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return math.nan
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline_eval", default=f"{OUT}/augfull_lr1e5/eval")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    man = json.load(open(a.manifest))
    scenes = man["scenes"]
    with Pool(a.procs) as pool:
        feats = {s: (n, F) for s, n, F in pool.map(scene_properties, scenes)}
    print(f"features computed for {len(feats)} scenes", flush=True)

    per_win = [(w["scene"], w["window"][0], w["window"][1], w["pre"][0], w["pre"][1])
               for w in man["windows"]]
    allF = {k: [] for k in KEYS}
    lab, bad = [], []
    for s in scenes:
        n, F = feats[s]
        ts, raw = load(f"{a.baseline_eval}/{s}/eval_depth_pose_metrics.csv")
        z = {m: badness(raw[m], m)[0] for m in METRICS}
        B = np.nanmean(np.vstack([z[m] for m in METRICS]), axis=0)
        Bfull = np.full(n, np.nan); Bfull[ts] = B
        L = np.zeros(n, dtype=int)          # 0 elsewhere, 1 window, 2 pre-window
        for sc, w0, w1, p0, p1 in per_win:
            if sc != s:
                continue
            L[p0:p1 + 1] = np.where(L[p0:p1 + 1] == 1, 1, 2)
            L[w0:w1 + 1] = 1
        for k in KEYS:
            allF[k].append(F[k])
        lab.append(L); bad.append(Bfull)
    allF = {k: np.concatenate(v) for k, v in allF.items()}
    lab = np.concatenate(lab); bad = np.concatenate(bad)
    np.savez(os.path.join(a.out_dir, "per_frame_properties.npz"), label=lab, badness=bad, **allF)

    # per-scene z-scoring so between-scene differences cannot create an effect
    zF = {}
    for k in KEYS:
        x = allF[k]; zx = np.full_like(x, np.nan); start = 0
        for s in scenes:
            n = feats[s][0]; seg = x[start:start + n]; ok = np.isfinite(seg)
            if ok.sum() > 5 and seg[ok].std() > 0:
                zx[start:start + n] = (seg - seg[ok].mean()) / seg[ok].std()
            start += n
        zF[k] = zx

    win, pre, oth = lab == 1, lab == 2, lab == 0

    def cohen_d(x, m1, m0):
        A, Bv = x[m1 & np.isfinite(x)], x[m0 & np.isfinite(x)]
        if len(A) < 5 or len(Bv) < 5:
            return math.nan
        sp = math.sqrt((A.var() + Bv.var()) / 2) or 1e-12
        return (A.mean() - Bv.mean()) / sp

    rows = []
    for key, name, fam in PROPS:
        rows.append({
            "key": key, "name": name, "family": fam,
            "d_window": cohen_d(zF[key], win, oth),
            "d_pre": cohen_d(zF[key], pre, oth),
            "rho": spearman(allF[key], bad),
            "mean_elsewhere": float(np.nanmean(allF[key][oth])),
            "mean_window": float(np.nanmean(allF[key][win])),
            "mean_pre": float(np.nanmean(allF[key][pre])),
        })
    json.dump(rows, open(os.path.join(a.out_dir, "window_properties.json"), "w"), indent=1)

    L_ = [f"# Input properties vs metric decline ({len(scenes)} scenes, {len(per_win)} windows)", "",
          f"{len(lab)} frames: {win.sum()} inside a decline window, {pre.sum()} in the span immediately "
          f"before one, {oth.sum()} elsewhere. Every property is computed from ground-truth poses, "
          "ground-truth depth and the pixels only -- no model output enters any feature. Effect sizes "
          "use per-scene z-scored values so that differences between episodes cannot masquerade as an "
          "effect. Cohen's d is positive when the property is LARGER inside that group than elsewhere. "
          "The correlation is Spearman's rho against the per-frame aggregate badness (mean robust z "
          "over the five metrics, sign-flipped for delta<1.25).", "",
          "| property | family | mean elsewhere | mean in window | d (window) | d (pre-window) | rho with badness |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        L_.append(f"| {r['name']} | {r['family']} | {r['mean_elsewhere']:.4g} | {r['mean_window']:.4g} "
                  f"| {r['d_window']:+.2f} | {r['d_pre']:+.2f} | {r['rho']:+.2f} |")
    md = os.path.join(a.out_dir, "window_properties.md")
    open(md, "w").write("\n".join(L_) + "\n")
    print("\n".join(L_))
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
