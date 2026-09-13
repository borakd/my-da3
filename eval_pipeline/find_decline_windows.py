#!/usr/bin/env python3
"""Find the frame windows that contribute most to metric DECLINE in one scene,
aggregated across all five eval metrics, at any window width.

Input: a per-scene eval_depth_pose_metrics.csv (eval_depth_poses.py, default
args; camera 0, numeric local_timestep rows).

Method
------
1. Orient + robust-standardise every metric into a "badness" z-series, higher = worse:
       b_m(t) = (x_m(t) - median_m) / (1.4826 * MAD_m)        (a1 uses 1 - a1)
   so a 0.01 m ATE bump and a 1° RPE spike are on the same footing, and the
   scene's own noise level sets the unit. NaNs (frame 0 RPE) are ignored.
2. Aggregate B(t) = mean over available metrics of b_m(t) (equal weights).
3. Multi-scale step detector. For every onset t and width w in [1, W]:
       D_w(t) = mean(B[t .. t+w-1]) - mean(B[t-w .. t-1])
   i.e. "how much worse is the next w frames than the previous w". Under the
   scene's noise (sigma = robust std of first differences / sqrt 2) the null
   spread of D_w is sigma*sqrt(2/w), so the score
       Z_w(t) = D_w(t) / (sigma * sqrt(2 / w))
   is comparable across widths; a 5-frame ramp scores highest at w~5, a single
   spike at w=1. This is a Haar / matched-filter change detector, the same
   statistic binary-segmentation change-point methods maximise.
4. Non-maximum suppression over (t, w): repeatedly take the highest remaining Z,
   report [t, t+w) as a decline window, and mask every (t', w') whose window
   overlaps [t-w, t+w). Stop below --zmin.
5. For each window report the per-metric before/after change in raw units and
   in z, and the window's share of each metric's scene aggregate ("error mass":
   sum of e^2 inside / total for the RMSE metrics, sum of x for absrel,
   sum of 1-a1 for a1), so "where it got worse" and "where it is worst" are
   both visible. Recovery (improvement) windows are also listed, from -Z.

Outputs: <out_dir>/decline_windows.json, decline_windows.md, decline_windows.png
"""
import argparse
import csv
import json
import math
import os

import numpy as np

METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
LABEL = {"absrel": "AbsRel", "a1": "a1 (δ<1.25)", "ate": "ATE (m)",
         "rpe_trans": "RPE trans (m)", "rpe_rot": "RPE rot (°)"}
SERIES = "#2a78d6"; AGG = "#4a3aa7"; DECL = "#e34948"; RECOV = "#1baf7a"
SURFACE = "#fcfcfb"; GRID = "#e6e5e1"; TEXT = "#0b0b0b"; TEXT2 = "#52514e"


def load(csv_path):
    ts, cols = [], {m: [] for m in METRICS}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["camera_id"].strip() != "0":
                continue
            try:
                t = int(r["local_timestep"].strip())
            except ValueError:
                continue
            ts.append(t)
            for m in METRICS:
                try:
                    cols[m].append(float(r[m]))
                except (TypeError, ValueError):
                    cols[m].append(math.nan)
    o = np.argsort(ts)
    return np.asarray(ts)[o], {m: np.asarray(v, float)[o] for m, v in cols.items()}


def badness(x, metric):
    """Oriented (higher = worse) robust z-series; NaN preserved."""
    v = 1.0 - x if metric == "a1" else x.copy()
    ok = np.isfinite(v)
    med = np.median(v[ok])
    mad = np.median(np.abs(v[ok] - med)) * 1.4826
    if mad <= 0:
        mad = np.std(v[ok]) or 1.0
    z = np.full_like(v, np.nan)
    z[ok] = (v[ok] - med) / mad
    return z, v


def step_scores(B, W):
    """Z[w-1, t] for onset t (window [t, t+w) vs [t-w, t)); NaN where undefined."""
    n = len(B)
    d = np.diff(B); d = d[np.isfinite(d)]
    sigma = 1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2) if len(d) else 1.0
    sigma = sigma if sigma > 0 else (np.nanstd(B) or 1.0)
    Z = np.full((W, n), np.nan)
    for w in range(1, W + 1):
        for t in range(w, n - w + 1):
            after, before = B[t:t + w], B[t - w:t]
            if np.isfinite(after).sum() < max(1, w // 2) or np.isfinite(before).sum() < max(1, w // 2):
                continue
            Z[w - 1, t] = (np.nanmean(after) - np.nanmean(before)) / (sigma * math.sqrt(2.0 / w))
    return Z, sigma


def nms(Z, zmin, max_windows):
    """Greedy non-maximum suppression over (t, w); returns [(t, w, z), ...]."""
    Zc = Z.copy()
    out = []
    while len(out) < max_windows:
        if not np.any(np.isfinite(Zc)):
            break
        idx = np.nanargmax(Zc)
        w1, t = np.unravel_index(idx, Zc.shape)
        z = Zc[w1, t]
        if not np.isfinite(z) or z < zmin:
            break
        w = w1 + 1
        out.append((int(t), int(w), float(z)))
        lo, hi = t - w, t + w  # suppress anything whose window overlaps the before+after span
        for w2 in range(1, Zc.shape[0] + 1):
            for t2 in range(Zc.shape[1]):
                if t2 < hi and t2 + w2 > lo:
                    Zc[w2 - 1, t2] = np.nan
    return out


def describe(win, ts, raw, z, agg_kind, B):
    t, w, score = win
    Bs = np.convolve(np.nan_to_num(B, nan=np.nanmedian(B)), np.ones(3) / 3, mode="same")
    lo, hi = max(0, t - w), min(len(ts), t + w)
    row = {"onset_frame": int(ts[t]), "window": [int(ts[t]), int(ts[min(t + w, len(ts)) - 1])],
           "width": w, "z": round(score, 2), "metrics": {}, "_B": Bs}
    for m in METRICS:
        b, a = raw[m][lo:t], raw[m][t:hi]
        zb, za = z[m][lo:t], z[m][t:hi]
        if np.isfinite(b).sum() == 0 or np.isfinite(a).sum() == 0:
            continue
        v = raw[m]; ok = np.isfinite(v)
        if agg_kind[m] == "rmse":
            mass = float(np.nansum(v[t:hi] ** 2) / np.nansum(v[ok] ** 2)) if np.nansum(v[ok] ** 2) > 0 else 0.0
        elif m == "a1":
            e = 1.0 - v
            mass = float(np.nansum(e[t:hi]) / np.nansum(e[ok])) if np.nansum(e[ok]) > 0 else 0.0
        else:
            mass = float(np.nansum(v[t:hi]) / np.nansum(v[ok])) if np.nansum(v[ok]) > 0 else 0.0
        row["metrics"][m] = {
            "before": float(np.nanmean(b)), "after": float(np.nanmean(a)),
            "delta": float(np.nanmean(a) - np.nanmean(b)),
            "delta_z": float(np.nanmean(za) - np.nanmean(zb)),
            "share_of_scene_error_in_window": round(mass, 4),
        }
    # Ramp: within [t-w, t+w) find where the (3-frame smoothed) aggregate crosses
    # 10% and 90% of the before->after level change. That is the span over which
    # the decline actually unfolds; the window width w only says how long the
    # contrast is sustained.
    Bs = row.pop("_B")
    seg = Bs[lo:hi]
    lvl_b, lvl_a = np.nanmean(Bs[lo:t]), np.nanmean(Bs[t:hi])
    if np.isfinite(lvl_b) and np.isfinite(lvl_a) and lvl_a != lvl_b:
        frac = (seg - lvl_b) / (lvl_a - lvl_b)
        k = np.arange(len(seg))
        i10 = next((int(i) for i in k if np.isfinite(frac[i]) and frac[i] >= 0.1 and i >= (t - lo) - w // 2), t - lo)
        i90 = next((int(i) for i in k if i >= i10 and np.isfinite(frac[i]) and frac[i] >= 0.9), min(len(seg) - 1, t - lo + w - 1))
        row["ramp"] = [int(ts[lo + i10]), int(ts[lo + i90])]
        row["ramp_frames"] = int(i90 - i10 + 1)
    dz = {m: d["delta_z"] for m, d in row["metrics"].items()}
    tot = sum(abs(x) for x in dz.values()) or 1.0
    row["drivers"] = {m: round(x / tot, 3) for m, x in sorted(dz.items(), key=lambda kv: -abs(kv[1]))}
    return row


def fmt_delta(m, d):
    s = "+" if d >= 0 else "−"
    if m == "a1":
        return f"{s}{abs(d):.3f}"
    if m == "rpe_rot":
        return f"{s}{abs(d):.2f}°"
    return f"{s}{abs(d):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default=None, help="default: CSV's directory")
    ap.add_argument("--max_width", type=int, default=20, help="largest window width W (frames)")
    ap.add_argument("--zmin", type=float, default=3.0, help="min noise-relative step score")
    ap.add_argument("--max_windows", type=int, default=12)
    ap.add_argument("--title", default=None)
    ap.add_argument("--blocked", default=None,
                    help="json of frames whose memory write was blocked ({scene: [..]} or [..]); "
                    "those frames are shaded and the lines drawn grey over them")
    args = ap.parse_args()
    blocked = set()
    if args.blocked and os.path.isfile(args.blocked):
        bj = json.load(open(args.blocked))
        blocked = set(int(x) for x in (next(iter(bj.values())) if isinstance(bj, dict) else bj))
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)
    title = args.title or os.path.basename(os.path.dirname(os.path.abspath(args.csv)))

    ts, raw = load(args.csv)
    n = len(ts)
    z, oriented = {}, {}
    for m in METRICS:
        z[m], oriented[m] = badness(raw[m], m)
    Bmat = np.vstack([z[m] for m in METRICS])
    B = np.nanmean(Bmat, axis=0)
    agg_kind = {"absrel": "mean", "a1": "mean", "ate": "rmse", "rpe_trans": "rmse", "rpe_rot": "rmse"}

    Z, sigma = step_scores(B, args.max_width)
    declines = nms(Z, args.zmin, args.max_windows)
    recoveries = nms(-Z, args.zmin, args.max_windows)
    dec_rows = [describe(w, ts, raw, z, agg_kind, B) for w in declines]
    rec_rows = [describe(w, ts, raw, z, agg_kind, -B) for w in recoveries]

    # ---- markdown report --------------------------------------------------------
    L = [f"# Decline windows: `{title}`", "",
         f"{n} frames. Badness = robust z per metric (higher = worse, a1 flipped), aggregate B = mean "
         f"over metrics. Step score Z_w(t) = (mean B over [t, t+w) − mean B over [t−w, t)) / "
         f"(σ·√(2/w)), σ = {sigma:.3f} (robust noise of B), widths 1..{args.max_width}, "
         f"threshold Z ≥ {args.zmin}, non-maximum suppression over overlapping (t, w).", ""]
    L += ["## Decline windows (ranked by Z)", "",
          "| # | onset | window | width | ramp (frames) | Z | drivers (signed share of z-change) | per-metric change (mean after − mean before) | share of scene error inside window |",
          "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(dec_rows, 1):
        drv = ", ".join(f"{m} {100*s:.0f}%" for m, s in list(r["drivers"].items())[:3])
        ch = "; ".join(f"{m} {fmt_delta(m, d['delta'])}" for m, d in r["metrics"].items())
        mass = "; ".join(f"{m} {100*d['share_of_scene_error_in_window']:.0f}%" for m, d in r["metrics"].items())
        ramp = f"{r['ramp'][0]}–{r['ramp'][1]} ({r['ramp_frames']})" if "ramp" in r else "–"
        L.append(f"| {i} | {r['onset_frame']} | {r['window'][0]}–{r['window'][1]} | {r['width']} | {ramp} | {r['z']} | {drv} | {ch} | {mass} |")
    L += ["", "## Recovery windows (improvements, same detector on −Z)", "",
          "| # | onset | window | width | ramp (frames) | Z | drivers | per-metric change |", "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rec_rows, 1):
        drv = ", ".join(f"{m} {100*s:.0f}%" for m, s in list(r["drivers"].items())[:3])
        ch = "; ".join(f"{m} {fmt_delta(m, d['delta'])}" for m, d in r["metrics"].items())
        ramp = f"{r['ramp'][0]}–{r['ramp'][1]} ({r['ramp_frames']})" if "ramp" in r else "–"
        L.append(f"| {i} | {r['onset_frame']} | {r['window'][0]}–{r['window'][1]} | {r['width']} | {ramp} | {r['z']} | {drv} | {ch} |")
    L.append("")
    md = os.path.join(out_dir, "decline_windows.md")
    open(md, "w").write("\n".join(L))
    with open(os.path.join(out_dir, "decline_windows.json"), "w") as f:
        json.dump({"scene": title, "n_frames": n, "sigma_B": sigma, "max_width": args.max_width,
                   "zmin": args.zmin, "declines": dec_rows, "recoveries": rec_rows}, f, indent=2)
    print("\n".join(L))

    # ---- figure ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(METRICS) + 2, 1, figsize=(12, 2.0 * (len(METRICS) + 2)), sharex=True,
                             gridspec_kw={"height_ratios": [1.4] + [1] * len(METRICS) + [1.2]})
    fig.patch.set_facecolor(SURFACE)

    BLOCK = "#6e6e6e"
    isb = np.array([int(t) in blocked for t in ts])
    def line(ax, y, color, lw=2, **kw):
        """Plot y vs ts; segments leading INTO a blocked frame are grey."""
        if not blocked:
            ax.plot(ts, y, color=color, lw=lw, **kw); return
        y = np.asarray(y, float)
        for i in range(1, len(ts)):
            if not (np.isfinite(y[i - 1]) and np.isfinite(y[i])):
                continue
            ax.plot(ts[i - 1:i + 1], y[i - 1:i + 1], color=(BLOCK if isb[i] else color), lw=lw,
                    solid_capstyle="round", **kw)
        ax.plot([], [], color=color, lw=lw, **kw)
    def shade_blocked(ax):
        if not blocked:
            return
        for i, t in enumerate(ts):
            if isb[i]:
                ax.axvspan(t - 0.5, t + 0.5, color=BLOCK, alpha=0.10, lw=0)
    def style(ax):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.grid(True, axis="y", color=GRID, linewidth=1); ax.grid(False, axis="x")
        ax.tick_params(colors=TEXT2, labelsize=8, length=0); ax.margins(x=0.01)

    def shade(ax):
        for r in dec_rows:
            ax.axvspan(r["window"][0] - 0.5, r["window"][1] + 0.5, color=DECL, alpha=0.12, lw=0)
            if "ramp" in r:
                ax.axvspan(r["ramp"][0] - 0.5, r["ramp"][1] + 0.5, color=DECL, alpha=0.22, lw=0)
        for r in rec_rows:
            ax.axvspan(r["window"][0] - 0.5, r["window"][1] + 0.5, color=RECOV, alpha=0.14, lw=0)

    ax = axes[0]; style(ax); shade(ax); shade_blocked(ax)
    line(ax, B, AGG)
    ax.axhline(0, color=TEXT2, lw=1, alpha=0.6)
    for i, r in enumerate(dec_rows, 1):
        ax.annotate(f"D{i} Z={r['z']:.1f}", xy=(r["onset_frame"], np.nanmax(B[r['window'][0] - ts[0]: r['window'][1] - ts[0] + 1])),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=DECL)
    ax.set_ylabel("aggregate badness B\n(mean robust z)", color=TEXT, fontsize=9)
    ax.set_title("aggregate across 5 metrics; red = decline windows (darker = ramp), green = recovery windows"
                 + ("; grey line / grey columns = memory write blocked" if blocked else ""), loc="left", fontsize=9, color=TEXT2)

    for ax, m in zip(axes[1:-1], METRICS):
        style(ax); shade(ax); shade_blocked(ax)
        line(ax, raw[m], SERIES)
        ax.set_ylabel(LABEL[m], color=TEXT, fontsize=9)
        if m == "a1": ax.set_ylim(0, 1.02)
        else: ax.set_ylim(bottom=0)

    ax = axes[-1]; style(ax)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        Zmax = np.nanmax(Z, axis=0)
    ax.fill_between(ts, 0, np.where(np.isfinite(Zmax), Zmax, 0), color=DECL, alpha=0.5, lw=0, step=None)
    Zmin = np.nanmin(Z, axis=0)
    ax.fill_between(ts, np.where(np.isfinite(Zmin), Zmin, 0), 0, color=RECOV, alpha=0.5, lw=0)
    ax.axhline(args.zmin, color=TEXT2, lw=1, alpha=0.6); ax.axhline(-args.zmin, color=TEXT2, lw=1, alpha=0.6)
    ax.set_ylabel("best step score\nover widths", color=TEXT, fontsize=9)
    ax.set_xlabel("frame #", color=TEXT, fontsize=9)
    fig.suptitle(f"{title} — where the metrics decline (window detector, widths 1–{args.max_width})",
                 x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    png = os.path.join(out_dir, "decline_windows.png")
    fig.savefig(png, dpi=130, facecolor=SURFACE); plt.close(fig)
    print(f"\nwrote {md}\nwrote {png}")


if __name__ == "__main__":
    main()
