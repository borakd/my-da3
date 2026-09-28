#!/usr/bin/env python3
"""MP4 overlay for one per-frame write-dose oracle search (perframe_dose_search.py): the scene's input video
as background with a HUD on top -- chosen a_t / b_t gauge, the ATE-cost heatmap of every candidate weight
at every frame (revealed up to the current frame), the current frame's cost column, the chosen-weight
timeline, and per-frame ATE / RPE-rot of the searched schedule vs plain CUT3R and the scene-best constant.

    python eval_pipeline/perframe_dose_video.py [--backbone zs] [--site state] [--tag best] [--pass_ 2] [--fps 8]

Writes $OUT/perframe_dose/<bb>/perframe_dose_<site>_<tag>_p<pass>.mp4 (H.264 via the env's ffmpeg).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
DATA = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
PLAIN = {"zs": f"{OUT}/cut3r_zeroshot/eval", "ft": f"{OUT}/augfull_lr1e5/eval"}

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", default="zs", choices=["zs", "ft"])
ap.add_argument("--site", default="state", choices=["state", "mem"])
ap.add_argument("--tag", default="best")
ap.add_argument("--pass_", type=int, default=2)
ap.add_argument("--fps", type=float, default=8)
ap.add_argument("--out", default="")
args = ap.parse_args()

ROOT = f"{OUT}/perframe_dose/{args.backbone}"
S = json.load(open(f"{ROOT}/search_{args.site}_{args.tag}.json"))
scene, grid, T = S["scene"], np.array(S["grid"]), S["T"]
P = next(p for p in S["passes"] if p["pass"] == args.pass_)
sched = np.array(P["schedule"])                      # (T,) chosen weight per frame; [0] = 1
L = np.array(P["landscape"])                         # (T-2, K): scene ATE if frame t=i+1 took grid[k]
K = len(grid)
D = np.full((K, T), np.nan)                          # cost above the best weight at that frame
D[:, 1:1 + L.shape[0]] = (L - L.min(1, keepdims=True)).T
vmax = np.nanpercentile(D, 97)
wsym = "a" if args.site == "state" else "b"


def per_frame(d):
    a = np.genfromtxt(f"{d}/{scene}/eval_depth_pose_metrics.csv", delimiter=",", names=True)
    a = a[np.isfinite(a["camera_id"]) & np.isfinite(a["local_timestep"])]  # drop the ALL / MEAN summary rows
    return {c: np.asarray(a[c], float) for c in ("ate", "rpe_rot", "absrel")}


def scene_ate(m):
    return float(np.sqrt(np.nanmean(m["ate"] ** 2)))


curves = [("plain CUT3R", per_frame(PLAIN[args.backbone]), "#bbbbbb"),
          (f"constant {wsym}={S['const_best']['w']:g}", per_frame(f"{ROOT}/pf_{args.site}_constbest/eval"), "#f5c542"),
          (f"per-frame {wsym}_t (oracle)", per_frame(f"{ROOT}/pf_{args.site}_{args.tag}_p{args.pass_}/eval"), "#34e0e0")]
for lab, m, _ in curves:
    assert len(m["ate"]) == T, (lab, len(m["ate"]), T)
imgs = sorted(glob.glob(f"{DATA}/{scene}/dense/rgb/*.png"))[:T]
assert len(imgs) == T, len(imgs)

W, H, DPI = 1280, 720, 100
FG, PANEL = "white", (0.05, 0.05, 0.08, 0.72)
plt.rcParams.update({"font.size": 8, "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
                     "ytick.color": FG, "axes.edgecolor": "#888888"})


def panel(fig, rect, title=None, pad=(0.032, 0.045, 0.008, 0.035)):
    """Axes over the video, with a translucent backing box that also covers its tick labels and title
    (pad = left, bottom, right, top in figure fractions)."""
    x, y, w, h = rect
    l, b, r, t = pad
    fig.patches.append(Rectangle((x - l, y - b), w + l + r, h + b + t, transform=fig.transFigure,
                                 facecolor=PANEL, edgecolor="none", zorder=0.5))
    ax = fig.add_axes(rect, zorder=1)
    ax.set_facecolor("none")
    if title:
        ax.set_title(title, fontsize=8.5, color=FG, loc="left", pad=3)
    return ax


def smooth(y, k=5):
    y = np.where(np.isfinite(y), y, np.nan)
    out = np.array([np.nanmean(y[max(0, i - k // 2): i + k // 2 + 1]) if np.isfinite(y[max(0, i - k // 2): i + k // 2 + 1]).any()
                    else np.nan for i in range(len(y))])
    return out


def frame(t):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    bg = fig.add_axes([0, 0, 1, 1], zorder=-1); bg.axis("off")
    bg.imshow(plt.imread(imgs[t]), aspect="auto", interpolation="bilinear")

    # ---- HUD text + gauge
    hud = panel(fig, [0.015, 0.855, 0.36, 0.13], pad=(0, 0, 0, 0)); hud.set_xticks([]); hud.set_yticks([])
    hud.axis("off")
    fin = scene_ate(curves[2][1])
    txt = (f"{scene}\n{'zero-shot' if args.backbone == 'zs' else 'finetuned'} CUT3R, "
           f"{'state' if args.site == 'state' else 'pose-memory'} write, oracle search (GT, not causal)\n"
           f"frame {t:3d} / {T - 1}\n"
           f"scene ATE: plain {scene_ate(curves[0][1]):.4f} | const {scene_ate(curves[1][1]):.4f} | per-frame {fin:.4f}")
    hud.text(0.03, 0.5, txt, va="center", ha="left", fontsize=8, family="monospace", transform=hud.transAxes)
    g = panel(fig, [0.045, 0.36, 0.03, 0.36], pad=(0.032, 0.015, 0.012, 0.065))
    g.set_xlim(0, 1); g.set_ylim(0, 1); g.set_xticks([])
    g.set_yticks([0, 0.25, 0.5, 0.75, 1]); g.set_yticklabels(["0", ".25", ".5", ".75", "1"], fontsize=7)
    g.add_patch(Rectangle((0.1, 0), 0.8, sched[t], color="#34e0e0", alpha=0.9))
    g.axhline(S["const_best"]["w"], color="#f5c542", lw=1, ls=":")
    g.set_title(f"{wsym}_t\n{sched[t]:.2f}", fontsize=9, color="#34e0e0", pad=3)

    # ---- heatmap (revealed up to t)
    hm = panel(fig, [0.53, 0.56, 0.44, 0.36], f"ATE cost of each {wsym} at each frame (scene ATE - best at that frame)")
    hm.imshow(D, aspect="auto", origin="lower", cmap="magma_r", vmin=0, vmax=vmax, extent=[-0.5, T - 0.5, -0.5, K - 0.5])
    hm.add_patch(Rectangle((t + 0.5, -0.5), T - t - 1, K, color=(0.05, 0.05, 0.08), alpha=0.8))
    ks = np.array([int(np.argmin(np.abs(grid - w))) for w in sched])
    hm.plot(np.arange(1, t + 1), ks[1:t + 1], ".", color="#34e0e0", ms=3)
    hm.axvline(t, color="white", lw=1)
    hm.set_yticks(range(K)); hm.set_yticklabels([f"{v:g}" for v in grid], fontsize=6.5)
    hm.set_ylabel("weight"); hm.set_xlabel("frame t", labelpad=1)

    # ---- the current frame's cost column
    cb = panel(fig, [0.53, 0.30, 0.20, 0.19], f"frame {t}: scene ATE if {wsym}_{t} = w")
    if 1 <= t <= L.shape[0]:
        col = L[t - 1]
        cols = ["#34e0e0" if k == ks[t] else "#8a8aa0" for k in range(K)]
        cb.barh(range(K), col, color=cols, height=0.75)
        lo, hi = col.min(), col.max()
        cb.set_xlim(lo - 0.15 * (hi - lo + 1e-9), hi + 0.05 * (hi - lo + 1e-9))
        cb.axvline(scene_ate(curves[0][1]), color="#bbbbbb", lw=0.8, ls="--")
    else:
        cb.text(0.5, 0.5, "frame 0 always written in full" if t == 0 else "last frame not searched",
                ha="center", va="center", transform=cb.transAxes)
    cb.set_yticks(range(K)); cb.set_yticklabels([f"{v:g}" for v in grid], fontsize=6)
    cb.tick_params(axis="x", labelsize=6)

    # ---- bottom strip: chosen weight, per-frame ATE, per-frame RPE rot
    x = np.arange(T)
    specs = [("chosen weight " + f"{wsym}_t", None), ("per-frame ATE (one Sim(3) fit)", "ate"),
             ("RPE rot, deg (5-frame mean)", "rpe_rot")]
    for i, (title, key) in enumerate(specs):
        ax = panel(fig, [0.05 + i * 0.315, 0.05, 0.28, 0.18], title)
        if key is None:
            ax.step(x, sched, where="mid", color="#34e0e0", alpha=0.25, lw=1)
            ax.step(x[:t + 1], sched[:t + 1], where="mid", color="#34e0e0", lw=1.5)
            ax.axhline(S["const_best"]["w"], color="#f5c542", lw=1, ls=":")
            ax.set_ylim(-0.05, 1.05)
        else:
            for lab, m, c in curves:
                y = m[key] if key == "ate" else smooth(m[key])
                ax.plot(x, y, color=c, alpha=0.25, lw=1)
                ax.plot(x[:t + 1], y[:t + 1], color=c, lw=1.5, label=lab)
            if i == 1:
                ax.legend(fontsize=6, loc="upper left", facecolor=PANEL, edgecolor="none", labelcolor=FG)
        ax.axvline(t, color="white", lw=0.8)
        ax.set_xlim(0, T - 1); ax.tick_params(labelsize=6)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def ffmpeg_bin():
    cand = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    return cand if os.path.isfile(cand) else shutil.which("ffmpeg")


out = args.out or f"{ROOT}/perframe_dose_{args.site}_{args.tag}_p{args.pass_}.mp4"
proc = subprocess.Popen([ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                         "-crf", "18", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
for t in range(T):
    fr = frame(t)
    assert fr.shape == (H, W, 3), fr.shape
    proc.stdin.write(fr.tobytes())
proc.stdin.close()
assert proc.wait() == 0
print("wrote", out)
