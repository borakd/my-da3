#!/usr/bin/env python3
"""Render an mp4 of the Kendall-Gal confidence token gate acting on one wrist video.

Per frame (default 1280x720 canvas, 10 fps):
  top-left    RGB frame with the per-token state-write gate drawn on the 20x12 patch grid the
              240 on-image state tokens live on: tokens that commit in full are left clear with a
              thin green edge, tokens that commit at weight g_min are tinted red.
  top-middle  the (trained) self-view confidence map that keys the gate, log scale, inferno.
  top-right   predicted depth (inverse-depth colouring, robust range per scene).
  bottom-left top-down trajectory: GT (green), this arm (blue) and, if stored, the plain
              finetune (red), each prediction placed in the GT frame by the evaluator's own
              whole-sequence Sim(3) fit; cursor at the current frame.
  bottom-right per-frame ATE sparkline for the arm (and the plain finetune) with the scene means.

Inputs: the arm's preds dir written by infer_and_eval_worker.py with DUMP_GATE=1
(depth/, camera/, gate/{conf_self, tok_gate_w}), its eval CSV, the GT dense dir.

  python maks_render_conf_gate_video.py --scene <name> --pilot viz_cbG1L --arm tok_cbG1L_q50_g50 --out x.mp4
"""
import argparse
import csv
import glob
import os
import subprocess

import cv2
import numpy as np

OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
W_PE, PH, PW = 28, 12, 20          # RoPE state grid width; on-image patch grid at 320x192 / 16
BG, INK, MUTE = (250, 250, 248), (60, 60, 58), (150, 150, 146)
GT_COL, BASE_COL, OURS_COL = (90, 170, 90), (70, 70, 225), (215, 140, 40)   # BGR


def put(img, s, org, scale=0.5, col=INK, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def load_poses(d, n, key):
    P = np.full((n, 4, 4), np.nan)
    for t in range(n):
        p = f"{d}/{t:06d}.npz"
        if os.path.isfile(p):
            P[t] = np.asarray(np.load(p)[key], float)
    return P


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, sv, Vt = np.linalg.svd(D.T @ S / len(src))
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    var = (S ** 2).sum() / len(src)
    s = float(np.trace(np.diag(sv) @ W) / var) if var > 0 else 1.0
    return s, R, md - s * (R @ ms)


def align_to_gt(P, gtP):
    ok = np.isfinite(P[:, 0, 0]) & np.isfinite(gtP[:, 0, 0])
    s, R, t = umeyama(P[ok][:, :3, 3], gtP[ok][:, :3, 3])
    C = np.full((len(P), 3), np.nan)
    C[ok] = (s * (R @ P[ok][:, :3, 3].T)).T + t
    return C


def per_frame(csv_path, n, key):
    out = np.full(n, np.nan)
    try:
        for r in csv.DictReader(open(csv_path)):
            if r.get("camera_id", "").strip() == "0":
                out[int(r["local_timestep"])] = float(r[key])
    except Exception:
        pass
    return out


def scene_means(csv_path):
    try:
        rows = [r for r in csv.DictReader(open(csv_path)) if r.get("camera_id", "").strip() == "0"]
        m = {k: np.nanmean([float(r[k]) for r in rows]) for k in ("absrel", "a1", "ate", "rpe_trans", "rpe_rot")}
        m["ate"] = float(np.sqrt(np.nanmean([float(r["ate"]) ** 2 for r in rows])))   # the evaluator reduces ATE by RMSE
        return m
    except Exception:
        return {}


def colormap(x, lo, hi, cmap=cv2.COLORMAP_INFERNO):
    y = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
    return cv2.applyColorMap((y * 255).astype(np.uint8), cmap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pilot", default="viz_cbG1L")
    ap.add_argument("--arm", default="tok_cbG1L_q50_g50")
    ap.add_argument("--base_label", default="augfull_lr1e5", help="plain finetune preds dir under OUT (optional)")
    ap.add_argument("--title", default="Kendall-Gal conf token gate, conf head trained through the gate (G1L, hard rank q.5 g.5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()

    gt = f"{ROOT}/{a.scene}/dense"
    pred = f"{OUT}/{a.pilot}/{a.arm}/preds/{a.scene}"
    ecsv = f"{OUT}/{a.pilot}/{a.arm}/eval/{a.scene}/eval_depth_pose_metrics.csv"
    rgb_files = sorted(glob.glob(f"{gt}/rgb/*.png"))
    n = len(rgb_files)
    assert n > 0 and os.path.isdir(f"{pred}/gate"), (n, pred)

    gtP = load_poses(f"{gt}/cam", n, "pose")
    prP = load_poses(f"{pred}/camera", n, "pose")
    ours = align_to_gt(prP, gtP)
    gtC = gtP[:, :3, 3]
    base = None
    bcsv = f"{OUT}/{a.base_label}/eval/{a.scene}/eval_depth_pose_metrics.csv"
    if os.path.isdir(f"{OUT}/{a.base_label}/preds/{a.scene}/camera"):
        base = align_to_gt(load_poses(f"{OUT}/{a.base_label}/preds/{a.scene}/camera", n, "pose"), gtP)
    ate_ours, ate_base = per_frame(ecsv, n, "ate"), per_frame(bcsv, n, "ate")
    m_ours, m_base = scene_means(ecsv), scene_means(bcsv)

    # top-down plane: the two principal axes of the GT trajectory
    ok = np.isfinite(gtC[:, 0])
    mu = gtC[ok].mean(0)
    _, _, Vt = np.linalg.svd(gtC[ok] - mu)
    proj = lambda C: (C - mu) @ Vt[:2].T
    pts = [proj(gtC[ok]), proj(ours[np.isfinite(ours[:, 0])])] + ([proj(base[np.isfinite(base[:, 0])])] if base is not None else [])
    allp = np.concatenate(pts)
    lo, hi = allp.min(0), allp.max(0)
    span = max((hi - lo).max(), 1e-3) * 1.15
    ctr = (lo + hi) / 2

    # depth colour range, robust over the scene
    dsample = np.concatenate([np.load(f"{pred}/depth/{t:06d}.npy").ravel()[::97] for t in range(0, n, max(n // 8, 1))])
    dsample = dsample[np.isfinite(dsample) & (dsample > 0)]
    inv_lo, inv_hi = np.percentile(1.0 / dsample, [2, 98])
    # confidence colour range (log conf), robust over the scene
    csample = np.concatenate([np.load(f"{pred}/gate/{t:06d}.npz")["conf_self"].ravel()[::97] for t in range(0, n, max(n // 8, 1))])
    c_lo, c_hi = np.percentile(np.log(np.maximum(csample, 1.0)), [2, 98])

    CW, CH = 1280, 720
    PW_, PH_ = 400, 225                       # top tiles (16:9 of a 320x180 frame)
    x0, y0 = 20, 60
    gap = 20
    vw = cv2.VideoWriter(a.out + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (CW, CH))
    for t in range(n):
        canvas = np.full((CH, CW, 3), BG, np.uint8)
        put(canvas, a.title, (x0, 30), 0.6, INK, 1)
        put(canvas, f"{a.scene}    frame {t:3d}/{n}", (x0, 50), 0.45, MUTE, 1)

        # --- RGB + gate overlay
        img = cv2.imread(rgb_files[t])
        g = np.load(f"{pred}/gate/{t:06d}.npz")
        tok = g["tok_gate_w"].reshape(-1)
        rr, cc = np.meshgrid(np.arange(PH), np.arange(PW), indexing="ij")
        grid = tok[rr * W_PE + cc]                                     # token i sits at (i // 28, i % 28) on the RoPE grid
        full = grid >= 0.999
        ov = cv2.resize(img, (PW_, PH_), interpolation=cv2.INTER_LINEAR)
        cell_w, cell_h = PW_ / PW, PH_ / PH
        tint = ov.copy()
        for r in range(PH):
            for c in range(PW):
                x1, y1 = int(round(c * cell_w)), int(round(r * cell_h))
                x2, y2 = int(round((c + 1) * cell_w)), int(round((r + 1) * cell_h))
                if not full[r, c]:
                    cv2.rectangle(tint, (x1, y1), (x2, y2), (40, 40, 220), -1)
        ov = cv2.addWeighted(ov, 0.55, tint, 0.45, 0)
        for r in range(PH):
            for c in range(PW):
                if full[r, c]:
                    x1, y1 = int(round(c * cell_w)), int(round(r * cell_h))
                    x2, y2 = int(round((c + 1) * cell_w)), int(round((r + 1) * cell_h))
                    cv2.rectangle(ov, (x1, y1), (x2 - 1, y2 - 1), (90, 200, 90), 1)
        canvas[y0:y0 + PH_, x0:x0 + PW_] = ov
        put(canvas, f"state write gate: {int(full.sum())}/{PH * PW} image tokens full, rest x{grid.min():.1f}", (x0, y0 + PH_ + 18), 0.42, INK)
        put(canvas, "green edge = full write, red = attenuated; registers commit", (x0, y0 + PH_ + 36), 0.38, MUTE)

        # --- confidence map
        conf = g["conf_self"]
        cm = colormap(np.log(np.maximum(conf, 1.0)), c_lo, c_hi)
        cm = cv2.resize(cm, (PW_, PH_), interpolation=cv2.INTER_NEAREST)
        xc = x0 + PW_ + gap
        canvas[y0:y0 + PH_, xc:xc + PW_] = cm
        put(canvas, f"self-view confidence, trained head (log, [{c_lo:.2f},{c_hi:.2f}])", (xc, y0 + PH_ + 18), 0.42, INK)
        put(canvas, "pooled to 20x12 tokens; top half by rank -> full write", (xc, y0 + PH_ + 36), 0.38, MUTE)

        # --- depth
        d = np.load(f"{pred}/depth/{t:06d}.npy")
        inv = np.where(np.isfinite(d) & (d > 0), 1.0 / np.maximum(d, 1e-6), inv_lo)
        dm = colormap(inv, inv_lo, inv_hi, cv2.COLORMAP_TURBO)
        dm = cv2.resize(dm, (PW_, PH_), interpolation=cv2.INTER_NEAREST)
        xd = xc + PW_ + gap
        canvas[y0:y0 + PH_, xd:xd + PW_] = dm
        put(canvas, "predicted depth (inverse depth colouring)", (xd, y0 + PH_ + 18), 0.42, INK)
        if m_ours:
            put(canvas, f"scene: AbsRel {m_ours['absrel']:.3f}  d<1.25 {m_ours['a1']:.3f}", (xd, y0 + PH_ + 36), 0.38, MUTE)

        # --- trajectory (bottom-left)
        ty0 = y0 + PH_ + 60
        TW, TH = 600, CH - ty0 - 20
        tile = np.full((TH, TW, 3), (244, 244, 241), np.uint8)
        def to_px(p):
            u = (p[:, 0] - ctr[0]) / span + 0.5
            v = (p[:, 1] - ctr[1]) / span + 0.5
            return np.stack([u * (TW - 40) + 20, (1 - v) * (TH - 40) + 20], 1).astype(int)
        def draw(C, col, thick):
            okc = np.isfinite(C[:, 0])
            idx = np.where(okc)[0]
            idx = idx[idx <= t]
            if len(idx) < 2:
                return
            P = to_px(proj(C[idx]))
            for i in range(1, len(P)):
                cv2.line(tile, tuple(P[i - 1]), tuple(P[i]), col, thick, cv2.LINE_AA)
            cv2.circle(tile, tuple(P[-1]), 5, col, -1, cv2.LINE_AA)
        # full GT path faint, then the growing paths
        okg = np.isfinite(gtC[:, 0])
        Pg = to_px(proj(gtC[okg]))
        for i in range(1, len(Pg)):
            cv2.line(tile, tuple(Pg[i - 1]), tuple(Pg[i]), (205, 225, 205), 1, cv2.LINE_AA)
        if base is not None:
            draw(base, BASE_COL, 2)
        draw(gtC, GT_COL, 2)
        draw(ours, OURS_COL, 2)
        put(tile, "top-down trajectory (GT principal plane), Sim(3)-aligned as the evaluator scores it", (12, 18), 0.4, INK)
        put(tile, "GT", (12, 40), 0.45, GT_COL, 2)
        put(tile, "this arm", (50, 40), 0.45, OURS_COL, 2)
        if base is not None:
            put(tile, "plain finetune", (140, 40), 0.45, BASE_COL, 2)
        canvas[ty0:ty0 + TH, x0:x0 + TW] = tile

        # --- ATE sparkline (bottom-right)
        sx = x0 + TW + gap
        SW = CW - sx - 20
        sp = np.full((TH, SW, 3), (244, 244, 241), np.uint8)
        series = [(ate_ours, OURS_COL)] + ([(ate_base, BASE_COL)] if np.isfinite(ate_base).any() else [])
        vmax = np.nanmax([np.nanmax(s) for s, _ in series if np.isfinite(s).any()] + [1e-3])
        def sp_px(i, v):
            return int(20 + i / max(n - 1, 1) * (SW - 40)), int(TH - 30 - v / vmax * (TH - 70))
        for s, col in series:
            pts_ = [sp_px(i, s[i]) for i in range(n) if np.isfinite(s[i])]
            for i in range(1, len(pts_)):
                cv2.line(sp, pts_[i - 1], pts_[i], col, 2 if col == OURS_COL else 1, cv2.LINE_AA)
        cx = sp_px(t, 0)[0]
        cv2.line(sp, (cx, 40), (cx, TH - 30), (120, 120, 120), 1)
        put(sp, "per-frame ATE (m), Sim(3) over the whole sequence; scene ATE = RMSE of this curve", (12, 18), 0.4, INK)
        put(sp, f"0 .. {vmax:.3f} m", (12, TH - 10), 0.36, MUTE)
        if m_ours:
            put(sp, f"this arm: ATE {m_ours['ate']:.4f}  RPE t {m_ours['rpe_trans']:.4f}  RPE rot {m_ours['rpe_rot']:.3f} deg", (12, 40), 0.42, OURS_COL, 1)
        if m_base:
            put(sp, f"plain finetune: ATE {m_base['ate']:.4f}  RPE t {m_base['rpe_trans']:.4f}  RPE rot {m_base['rpe_rot']:.3f} deg", (12, 60), 0.42, BASE_COL, 1)
        if np.isfinite(ate_ours[t]):
            put(sp, f"frame ATE {ate_ours[t]:.4f}", (cx + 6, 80), 0.4, OURS_COL, 1)
        canvas[ty0:ty0 + TH, sx:sx + SW] = sp

        vw.write(canvas)
    vw.release()
    # re-encode to H.264 so it plays everywhere; fall back to the mp4v file if ffmpeg is missing
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out + ".tmp.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", a.out], check=True)
        os.remove(a.out + ".tmp.mp4")
    except Exception as e:
        print(f"ffmpeg re-encode failed ({e!r}); keeping mp4v file")
        os.replace(a.out + ".tmp.mp4", a.out)
    print(f"wrote {a.out} ({n} frames)")


if __name__ == "__main__":
    main()
