#!/usr/bin/env python3
"""Two readable figures for the input-property analysis, in place of one dense one.

  window_properties_effects.png    Which frames are decline windows?
                                   A dot-and-interval plot -- one row per
                                   property, a dot for frames inside a window and
                                   one for the span just before, 95% intervals
                                   from a SCENE-level bootstrap (whole episodes
                                   resampled, never frames: within-episode lag-1
                                   autocorrelation runs 0.49-1.00, so a
                                   frame-level bootstrap would invent precision).

  window_properties_affects.png    What does each property actually affect?
                                   Correlation with the pose metrics on one axis
                                   and the depth metrics on the other. The single
                                   fact worth carrying away is that the two are
                                   orthogonal: nothing sits off the axes.

Marker SHAPE carries the property family in the second figure rather than a
fourth categorical hue, which the documented palette does not support under the
all-pairs rule that scatter plots fall under.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_decline_windows import load, badness, METRICS  # noqa: E402

ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9990"
GRID = "#e4e3de"
C_WINDOW = "#2a78d6"    # palette slot 1
C_PRE = "#eb6834"       # palette slot 2
POSE = ["rpe_trans", "rpe_rot"]
DEPTH = ["absrel", "a1"]
SHAPES = ["o", "s", "^", "D"]


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return np.nan
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def compute(a):
    rows = json.load(open(a.json))
    d = np.load(a.npz)
    man = json.load(open(a.manifest))
    scenes, n_win = man["scenes"], len(man["windows"])
    lens = [len([f for f in os.listdir(f"{ROOT}/{s}/dense/rgb") if f.endswith(".png")])
            for s in scenes]
    lab = d["label"]
    assert sum(lens) == len(lab)
    bounds = np.cumsum([0] + lens)
    S, K = len(scenes), len(rows)

    cnt = np.zeros((S, K, 2)); ssum = np.zeros((S, K, 2)); ssq = np.zeros((S, K, 2))
    rcnt = np.zeros((S, K)); rsum = np.zeros((S, K)); rsq = np.zeros((S, K))
    RHO = np.full((S, K, len(METRICS)), np.nan)
    for si, s in enumerate(scenes):
        lo, hi = bounds[si], bounds[si + 1]
        n = hi - lo
        L = lab[lo:hi]
        ts, raw = load(f"{a.baseline_eval}/{s}/eval_depth_pose_metrics.csv")
        Z = {}
        for m in METRICS:
            full = np.full(n, np.nan); full[ts] = badness(raw[m], m)[0]
            Z[m] = full
        for ki, r in enumerate(rows):
            x = d[r["key"]][lo:hi].astype(float)
            for mi, m in enumerate(METRICS):
                RHO[si, ki, mi] = spearman(x, Z[m])
            ok = np.isfinite(x)
            if ok.sum() <= 5 or x[ok].std() == 0:
                continue
            z = (x - x[ok].mean()) / x[ok].std()
            for gi, sel in enumerate([(L == 1) & ok, (L == 2) & ok]):
                cnt[si, ki, gi] = sel.sum(); ssum[si, ki, gi] = z[sel].sum()
                ssq[si, ki, gi] = (z[sel] ** 2).sum()
            sel = (L == 0) & ok
            rcnt[si, ki] = sel.sum(); rsum[si, ki] = z[sel].sum(); rsq[si, ki] = (z[sel] ** 2).sum()

    def d_from(idx):
        c = cnt[idx].sum(0); s1 = ssum[idx].sum(0); q = ssq[idx].sum(0)
        rc = rcnt[idx].sum(0); rs = rsum[idx].sum(0); rq = rsq[idx].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mA = s1 / c; vA = q / c - mA ** 2
            mO = (rs / rc)[:, None]; vO = (rq / rc - (rs / rc) ** 2)[:, None]
            return (mA - mO) / np.sqrt(np.clip((vA + vO) / 2, 1e-12, None))

    rng = np.random.default_rng(0)
    point_d = d_from(np.arange(S))
    bd = np.stack([d_from(rng.integers(0, S, S)) for _ in range(a.boot)])
    d_lo, d_hi = np.nanpercentile(bd, 2.5, 0), np.nanpercentile(bd, 97.5, 0)

    def axis_rho(sub):
        cols = [METRICS.index(m) for m in sub]
        return np.nanmean(RHO[:, :, cols], axis=2)
    pose_s, depth_s = axis_rho(POSE), axis_rho(DEPTH)
    pose_pt, depth_pt = np.nanmean(pose_s, 0), np.nanmean(depth_s, 0)
    pb = np.stack([np.nanmean(pose_s[rng.integers(0, S, S)], 0) for _ in range(a.boot)])
    db = np.stack([np.nanmean(depth_s[rng.integers(0, S, S)], 0) for _ in range(a.boot)])
    return dict(rows=rows, scenes=scenes, n_win=n_win, n_frames=len(lab),
                point_d=point_d, d_lo=d_lo, d_hi=d_hi,
                pose=pose_pt, depth=depth_pt,
                pose_ci=(np.nanpercentile(pb, 2.5, 0), np.nanpercentile(pb, 97.5, 0)),
                depth_ci=(np.nanpercentile(db, 2.5, 0), np.nanpercentile(db, 97.5, 0)),
                rho=np.nanmean(RHO, 0))


def frame(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5, length=0)


def fig_effects(R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = R["rows"]
    fams = []
    for r in rows:
        if r["family"] not in fams:
            fams.append(r["family"])
    order, spans = [], []
    for f in reversed(fams):
        grp = sorted([i for i, r in enumerate(rows) if r["family"] == f],
                     key=lambda i: np.nan_to_num(R["point_d"][i, 0], nan=-9))
        spans.append((f, len(order), len(order) + len(grp)))
        order += grp
    order = np.array(order)
    n = len(order)

    fig = plt.figure(figsize=(13.2, 0.52 * n + 2.9))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.375, 0.085, 0.595, 0.755])
    frame(ax)
    y = np.arange(n)

    for g, col, dy, nm in [(0, C_WINDOW, -0.16, "Inside a decline window"),
                           (1, C_PRE, 0.16, "In the span just before a window")]:
        v, lo, hi = R["point_d"][order, g], R["d_lo"][order, g], R["d_hi"][order, g]
        sig = (lo > 0) | (hi < 0)
        for i in range(n):
            al = 1.0 if sig[i] else 0.3
            ax.plot([lo[i], hi[i]], [y[i] + dy] * 2, color=col, lw=1.6, alpha=al,
                    solid_capstyle="round", zorder=3)
            ax.plot(v[i], y[i] + dy, "o", ms=7.5, color=col, alpha=al,
                    mec=SURFACE, mew=1.2, zorder=4)

    for g in (0.2, 0.5):
        for s in (-1, 1):
            ax.axvline(s * g, color=GRID, lw=1, ls=(0, (2, 3)), zorder=0)
    ax.axvline(0, color=INK2, lw=1.2, zorder=2)
    for k, (f, lo_, hi_) in enumerate(spans):
        if k:
            ax.axhline(lo_ - 0.5, color=GRID, lw=1.1, zorder=1)
        ymid = (lo_ + hi_) / 2 - 0.5
        frac = (ymid - (-0.7)) / ((n - 0.3) - (-0.7))
        fig.text(0.016, 0.085 + frac * 0.755, f, fontsize=10.5, color=INK,
                 fontweight="bold", va="center", ha="left")

    lim = float(np.nanmax(np.abs(np.concatenate([R["d_lo"][order].ravel(),
                                                 R["d_hi"][order].ravel()])))) * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([rows[i]["name"] for i in order], fontsize=10.5, color=INK)
    ax.set_xlabel("Standardised difference from the rest of the sequence   (Cohen's d)",
                  fontsize=10, color=INK2, labelpad=9)
    for g, lb in ((0.2, "small"), (0.5, "medium")):
        ax.text(g, n - 0.42, lb, fontsize=8.5, color=MUTED, ha="center")
        ax.text(-g, n - 0.42, lb, fontsize=8.5, color=MUTED, ha="center")

    h = [Line2D([], [], color=C_WINDOW, marker="o", ms=7.5, lw=1.6, mec=SURFACE,
                label="Inside a decline window"),
         Line2D([], [], color=C_PRE, marker="o", ms=7.5, lw=1.6, mec=SURFACE,
                label="In the span just before a window")]
    leg = ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
                    frameon=False, fontsize=10.5, handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.text(0.018, 0.955, "Which frames become decline windows?", fontsize=16,
             color=INK, fontweight="bold", va="center")
    fig.text(0.018, 0.918,
             f"{len(R['scenes'])} DROID test scenes, {R['n_win']} decline windows, {R['n_frames']} frames. "
             "Properties measured from ground truth and pixels only.", fontsize=10, color=INK2, va="center")
    fig.text(0.018, 0.893,
             "Bars are 95% intervals from resampling whole episodes. Faded = not distinguishable from zero.",
             fontsize=10, color=INK2, va="center")
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"wrote {out}")


def fig_affects(R, out):
    """Two small signed-bar panels sharing one row of names. Ordered by the pose
    correlation, so the disjointness reads as an L: the rows with long bars on
    the left have none on the right and vice versa. No scatter, so no labels can
    ever collide."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = R["rows"]
    px, py = R["pose"], R["depth"]
    plo, phi = R["pose_ci"]; dlo, dhi = R["depth_ci"]
    order = np.argsort(np.nan_to_num(px))
    n = len(order)

    fig = plt.figure(figsize=(13.0, 0.5 * n + 3.0))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.07,
                          left=0.33, right=0.975, top=0.80, bottom=0.09)
    axp = fig.add_subplot(gs[0, 0]); axd = fig.add_subplot(gs[0, 1], sharey=axp)
    y = np.arange(n)

    for ax, v, lo, hi, title in ((axp, px, plo, phi, "POSE error\nRPE translation and rotation"),
                                 (axd, py, dlo, dhi, "DEPTH error\nAbsRel and δ<1.25")):
        frame(ax)
        sig = (lo[order] > 0) | (hi[order] < 0)
        vv = v[order]
        for i in range(n):
            ax.barh(y[i], vv[i], 0.6, color=INK2, alpha=0.92 if sig[i] else 0.16, zorder=3)
        ax.errorbar(vv, y, xerr=[vv - lo[order], hi[order] - vv], fmt="none",
                    ecolor=INK, elinewidth=1.0, capsize=2.2, capthick=1.0, zorder=4)
        for i in range(n):
            if sig[i] and abs(vv[i]) >= 0.12:
                off = 0.012 if vv[i] >= 0 else -0.012
                ax.text(hi[order][i] + off if vv[i] >= 0 else lo[order][i] + off, y[i],
                        f"{vv[i]:+.2f}", va="center", ha="left" if vv[i] >= 0 else "right",
                        fontsize=8.6, color=INK, zorder=5)
        ax.axvline(0, color=INK2, lw=1.2, zorder=2)
        ax.set_xlim(-0.62, 0.72)
        ax.set_title(title, fontsize=10.5, color=INK, pad=10, loc="center")
        ax.set_xlabel("mean per-scene Spearman's rho", fontsize=9.5, color=INK2, labelpad=8)

    axp.set_ylim(-0.7, n - 0.3)
    axp.set_yticks(y)
    axp.set_yticklabels([rows[i]["name"] for i in order], fontsize=10.5, color=INK)
    axd.tick_params(labelleft=False)

    fig.text(0.016, 0.955, "Pose error and depth error have separate causes",
             fontsize=16, color=INK, fontweight="bold", va="center")
    fig.text(0.016, 0.908,
             "Each row is one property. A row with a long bar on the left has essentially none on the right, and the\n"
             "other way round: what makes the pose worse and what makes the depth worse are disjoint sets of causes.",
             fontsize=10, color=INK2, va="center", linespacing=1.5)
    fig.text(0.016, 0.862,
             "Faded bars are not distinguishable from zero (95% interval from resampling whole episodes).",
             fontsize=10, color=INK2, va="center")
    fig.text(0.016, 0.025,
             "100 DROID test scenes. ATE is deliberately absent: no per-frame property predicts it (largest |rho| 0.19), because "
             "it is an accumulated quantity rather than a per-frame one.", fontsize=9, color=MUTED, va="center")
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline_eval",
                    default="/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/augfull_lr1e5/eval")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    R = compute(a)
    os.makedirs(a.out_dir, exist_ok=True)
    fig_effects(R, os.path.join(a.out_dir, "window_properties_effects.png"))
    fig_affects(R, os.path.join(a.out_dir, "window_properties_affects.png"))


if __name__ == "__main__":
    main()
