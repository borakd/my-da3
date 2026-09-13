#!/usr/bin/env python3
"""Are decline windows a data-side phenomenon? Per-frame PHYSICAL features from
GT + pixels only (no model), compared inside decline windows, in the pre-window
spans, and elsewhere, over the 100-scene manifest.

Features per frame t (all causal / local):
  trans_speed   ||c_t - c_{t-1}||            GT camera centre step (m)
  rot_speed     angle(R_{t-1}^T R_t)         GT rotation step (deg)
  sharpness     variance of Laplacian of the grey image (blur proxy)
  int_change    mean |I_t - I_{t-1}| / 255   appearance change (motion/lighting)
  depth_median  median valid GT depth (m)    scene proximity
  near_frac     fraction of pixels with GT depth < 0.12 m (gripper / object at the lens)
  invalid_frac  fraction of GT depth pixels <= 0 (no GT)
  outlier_frac  fraction of pixels flagged in dense/outlier_mask (GT quality)

Per feature: mean in window frames vs non-window frames, standardised
difference (Cohen's d over frames), the same for pre-window spans, and the
Spearman correlation of the feature with the per-frame aggregate badness from
the baseline run. Also the fraction of windows whose peak of each feature
(inside window or pre-window) exceeds the scene's 90th percentile.

Outputs: <out_dir>/window_physics.md, window_physics.png, per_frame_features.npz
"""
import argparse
import csv
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
FEATS = ["trans_speed", "rot_speed", "sharpness", "int_change", "depth_median", "near_frac", "invalid_frac", "outlier_frac"]


def scene_features(scene):
    d = f"{ROOT}/{scene}/dense"
    rgb = sorted(f for f in os.listdir(f"{d}/rgb") if f.endswith(".png"))
    n = len(rgb)
    F = {k: np.full(n, np.nan) for k in FEATS}
    prev_gray, prev_c, prev_R = None, None, None
    for i, fn in enumerate(rgb):
        stem = os.path.splitext(fn)[0]
        img = cv2.imread(f"{d}/rgb/{fn}", cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        F["sharpness"][i] = cv2.Laplacian(img, cv2.CV_64F).var()
        if prev_gray is not None:
            F["int_change"][i] = np.abs(img.astype(np.float32) - prev_gray).mean() / 255.0
        prev_gray = img.astype(np.float32)
        cam = f"{d}/cam/{stem}.npz"
        if os.path.isfile(cam):
            P = np.load(cam)["pose"].astype(np.float64)
            if P.shape == (3, 4):
                P = np.vstack([P, [0, 0, 0, 1]])
            c, R = P[:3, 3], P[:3, :3]
            if prev_c is not None:
                F["trans_speed"][i] = float(np.linalg.norm(c - prev_c))
                cosang = (np.trace(prev_R.T @ R) - 1) / 2
                F["rot_speed"][i] = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
            prev_c, prev_R = c, R
        dp = f"{d}/depth/{stem}.npy"
        if os.path.isfile(dp):
            z = np.load(dp)
            valid = z > 0
            F["invalid_frac"][i] = 1.0 - valid.mean()
            if valid.any():
                zv = z[valid]
                F["depth_median"][i] = float(np.median(zv))
                F["near_frac"][i] = float((zv < 0.12).mean())
        om = f"{d}/outlier_mask/{stem}.png"
        if os.path.isfile(om):
            m = cv2.imread(om, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                F["outlier_frac"][i] = float((m > 0).mean())
    return scene, n, F


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return math.nan
    ra = np.argsort(np.argsort(a[ok])); rb = np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline_eval", default=f"{OUT}/augfull_lr1e5/eval")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    man = json.load(open(args.manifest))
    scenes = man["scenes"]
    with Pool(args.procs) as pool:
        feats = dict((s, (n, F)) for s, n, F in pool.map(scene_features, scenes))

    # per-frame labels + baseline badness
    allF = {k: [] for k in FEATS}; lab = []; bad = []; scene_id = []
    per_win = []  # (scene, a, b, pa, pb)
    for w in man["windows"]:
        per_win.append((w["scene"], w["window"][0], w["window"][1], w["pre"][0], w["pre"][1]))
    for s in scenes:
        n, F = feats[s]
        ts, raw = load(f"{args.baseline_eval}/{s}/eval_depth_pose_metrics.csv")
        z = {m: badness(raw[m], m)[0] for m in METRICS}
        B = np.nanmean(np.vstack([z[m] for m in METRICS]), axis=0)
        Bfull = np.full(n, np.nan); Bfull[ts] = B
        L = np.zeros(n, dtype=int)  # 0 other, 1 window, 2 pre-window (window wins on overlap)
        for sc, a, b, pa, pb in per_win:
            if sc != s:
                continue
            L[pa:pb + 1] = np.where(L[pa:pb + 1] == 1, 1, 2)
            L[a:b + 1] = 1
        for k in FEATS:
            allF[k].append(F[k])
        lab.append(L); bad.append(Bfull); scene_id += [s] * n
    allF = {k: np.concatenate(v) for k, v in allF.items()}
    lab = np.concatenate(lab); bad = np.concatenate(bad)
    np.savez(os.path.join(args.out_dir, "per_frame_features.npz"), label=lab, badness=bad, **allF)

    def d_eff(x, m1, m0):
        a, b = x[m1 & np.isfinite(x)], x[m0 & np.isfinite(x)]
        sp = math.sqrt((a.var() + b.var()) / 2) or 1e-12
        return (a.mean() - b.mean()) / sp, a.mean(), b.mean()

    # per-scene z-scored features so that scene-level differences don't dominate
    zF = {}
    for k in FEATS:
        x = allF[k].copy(); zx = np.full_like(x, np.nan)
        start = 0
        for s in scenes:
            n = feats[s][0]; seg = x[start:start + n]; ok = np.isfinite(seg)
            if ok.sum() > 5 and seg[ok].std() > 0:
                zx[start:start + n] = (seg - seg[ok].mean()) / seg[ok].std()
            start += n
        zF[k] = zx

    win, pre, oth = lab == 1, lab == 2, lab == 0
    L = ["# Decline windows vs physical frame features (100 scenes, 367 windows)", "",
         f"{len(lab)} frames: {win.sum()} in windows, {pre.sum()} in pre-window spans, {oth.sum()} elsewhere. "
         "Features from GT poses/depth/masks and pixels only. d = Cohen's d on per-scene z-scored features "
         "(positive = larger in that group than in 'elsewhere' frames); rho = Spearman between the raw feature "
         "and the baseline's per-frame aggregate badness over all frames.", "",
         "| feature | elsewhere mean | window mean | d(window) | pre-window mean | d(pre) | rho(badness) |",
         "|---|---|---|---|---|---|---|"]
    rows_fig = []
    for k in FEATS:
        dw, mw, mo = d_eff(zF[k], win, oth)
        dp, mp, _ = d_eff(zF[k], pre, oth)
        raw_o = np.nanmean(allF[k][oth]); raw_w = np.nanmean(allF[k][win]); raw_p = np.nanmean(allF[k][pre])
        rho = spearman(allF[k], bad)
        L.append(f"| {k} | {raw_o:.4g} | {raw_w:.4g} | {dw:+.2f} | {raw_p:.4g} | {dp:+.2f} | {rho:+.2f} |")
        rows_fig.append((k, dw, dp, rho))
    L.append("")
    # window-level: fraction of windows whose window / pre-window peak exceeds the scene p90
    L += ["## Fraction of windows containing an extreme frame (feature above the scene's 90th percentile)", "",
          "| feature | inside window | in pre-window | either | chance level for a span of that length |", "|---|---|---|---|---|"]
    start_of = {}; s0 = 0
    for s in scenes:
        start_of[s] = s0; s0 += feats[s][0]
    for k in FEATS:
        hit_w = hit_p = hit_e = 0; chance = []
        for sc, a, b, pa, pb in per_win:
            n, F = feats[sc]; x = F[k]; ok = np.isfinite(x)
            if ok.sum() < 10:
                continue
            thr = np.nanpercentile(x, 90)
            hw = np.nanmax(x[a:b + 1]) > thr if np.isfinite(x[a:b + 1]).any() else False
            hp = np.nanmax(x[pa:pb + 1]) > thr if np.isfinite(x[pa:pb + 1]).any() else False
            hit_w += hw; hit_p += hp; hit_e += (hw or hp)
            chance.append(1 - 0.9 ** (b - a + 1))
        m = len(chance) or 1
        L.append(f"| {k} | {hit_w/m:.0%} | {hit_p/m:.0%} | {hit_e/m:.0%} | {np.mean(chance):.0%} |")
    L.append("")
    md = os.path.join(args.out_dir, "window_physics.md")
    open(md, "w").write("\n".join(L)); print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2)); fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    y = np.arange(len(rows_fig))
    ax.barh(y - 0.2, [r[1] for r in rows_fig], 0.35, color="#e34948", label="inside window vs elsewhere")
    ax.barh(y + 0.2, [r[2] for r in rows_fig], 0.35, color="#eda100", label="pre-window vs elsewhere")
    for i, r in enumerate(rows_fig):
        ax.text(max(r[1], r[2], 0) + 0.02, i, f"rho={r[3]:+.2f}", va="center", fontsize=8, color="#52514e")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows_fig]); ax.axvline(0, color="#52514e", lw=1)
    ax.set_xlabel("Cohen's d on per-scene z-scored feature"); ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title("Do decline windows coincide with physical frame conditions? (100 scenes, 367 windows)", loc="left", fontsize=10)
    fig.tight_layout(); png = os.path.join(args.out_dir, "window_physics.png"); fig.savefig(png, dpi=130, facecolor="#fcfcfb")
    print(f"wrote {md}\nwrote {png}")


if __name__ == "__main__":
    main()
