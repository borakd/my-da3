#!/usr/bin/env python3
"""Render an analysis video for one scene: the RGB frames with a live overlay of
the five eval metrics, their goodness/badness, and the detected decline /
recovery windows, so the video can be read side by side with
metrics_over_time.png and decline_windows.png.

Layout (per frame, default 1280x760):
  left   RGB frame (upscaled) with a border whose colour tracks the aggregate
         badness of the CURRENT frame (green good -> grey -> red bad); a badge in
         the frame's top-left names the active window ("DECLINE D1 ramp",
         "RECOVERY R2") and the top-right shows frame #.
  right  six tiles: aggregate badness + one per metric. Each tile: name, current
         value (large), delta vs the scene reference, a sparkline of the whole
         scene with the scene reference line and a cursor; tile tint = that
         metric's robust z for the current frame.
  bottom timeline of the aggregate badness with the decline (red) / recovery
         (green) windows shaded exactly as in decline_windows.png, cursor at the
         current frame, window labels.

Inputs: the scene's frames dir (dense/rgb), its eval_depth_pose_metrics.csv
and the decline_windows.json written by find_decline_windows.py (optional; the
timeline is drawn without shading if absent).

  python render_metrics_video.py --frames <dense/rgb> --csv <eval csv> \
      --windows <decline_windows.json> --out <mp4> [--fps 6] [--hold_first 8]
"""
import argparse
import csv
import glob
import json
import math
import os

import cv2
import numpy as np

METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
LABEL = {"absrel": "AbsRel", "a1": "a1 (d<1.25)", "ate": "ATE (m)",
         "rpe_trans": "RPE trans (m)", "rpe_rot": "RPE rot (deg)"}
BETTER = {"absrel": "lower", "a1": "higher", "ate": "lower", "rpe_trans": "lower", "rpe_rot": "lower"}
FMT = {"absrel": "{:.3f}", "a1": "{:.3f}", "ate": "{:.4f}", "rpe_trans": "{:.4f}", "rpe_rot": "{:.2f}"}

# palette (BGR for cv2)
SURFACE = (251, 252, 252)
PANEL = (243, 243, 241)
GRID = (225, 229, 230)
TEXT = (11, 11, 11)
TEXT2 = (78, 81, 82)
BLUE = (214, 120, 42)
VIOLET = (167, 58, 74)
RED = (72, 73, 227)
GREEN = (122, 175, 27)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ----------------------------------------------------------------------------- data
def load_csv(path):
    ts, cols, scene = [], {m: [] for m in METRICS}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            cam, lt = r["camera_id"].strip(), r["local_timestep"].strip()
            if lt == "MEAN":
                if cam in ("0", "ALL") and not scene:
                    scene = {m: _f(r[m]) for m in METRICS}
                continue
            if cam != "0":
                continue
            try:
                t = int(lt)
            except ValueError:
                continue
            ts.append(t)
            for m in METRICS:
                cols[m].append(_f(r[m]))
    o = np.argsort(ts)
    return np.asarray(ts)[o], {m: np.asarray(v, float)[o] for m, v in cols.items()}, scene


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def robust_z(x, metric):
    v = 1.0 - x if metric == "a1" else x.copy()
    ok = np.isfinite(v)
    med = np.median(v[ok])
    mad = np.median(np.abs(v[ok] - med)) * 1.4826
    if mad <= 0:
        mad = np.std(v[ok]) or 1.0
    z = np.full_like(v, np.nan)
    z[ok] = (v[ok] - med) / mad
    return z


def badness_color(z, lo=-1.0, hi=2.0):
    """Robust z -> BGR: green at <= lo, neutral grey at 0, red at >= hi."""
    if not np.isfinite(z):
        return (170, 170, 170)
    if z < 0:
        a = min(1.0, z / lo)
        c0, c1 = np.array((170, 170, 170)), np.array(GREEN)
    else:
        a = min(1.0, z / hi)
        c0, c1 = np.array((170, 170, 170)), np.array(RED)
    return tuple(int(v) for v in (c0 * (1 - a) + c1 * a))


def tint(color, strength):
    """Blend a colour into the panel colour."""
    c = np.array(color, float); p = np.array(PANEL, float)
    return tuple(int(v) for v in (p * (1 - strength) + c * strength))


# ----------------------------------------------------------------------------- drawing
def text(img, s, org, scale=0.5, color=TEXT, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def sparkline(img, x0, y0, w, h, vals, cur, ref=None, color=BLUE, ylim=None):
    """Draw a full-scene sparkline in box (x0,y0,w,h) with cursor at index cur."""
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), GRID, 1)
    ok = np.isfinite(vals)
    if ok.sum() < 2:
        return
    lo, hi = (0.0, float(np.nanmax(vals)) * 1.05 or 1.0) if ylim is None else ylim
    if hi <= lo:
        hi = lo + 1e-9
    n = len(vals)
    xs = x0 + (np.arange(n) / max(n - 1, 1)) * w
    ys = y0 + h - (np.clip(vals, lo, hi) - lo) / (hi - lo) * h
    if ref is not None and np.isfinite(ref):
        yr = int(y0 + h - (np.clip(ref, lo, hi) - lo) / (hi - lo) * h)
        cv2.line(img, (x0, yr), (x0 + w, yr), TEXT2, 1, cv2.LINE_AA)
    pts = [(int(xs[i]), int(ys[i])) for i in range(n) if ok[i]]
    # past = solid, future = faint
    past = [p for i, p in enumerate([(int(xs[i]), int(ys[i])) if ok[i] else None for i in range(n)]) if p and i <= cur]
    fut = [p for i, p in enumerate([(int(xs[i]), int(ys[i])) if ok[i] else None for i in range(n)]) if p and i >= cur]
    if len(fut) > 1:
        cv2.polylines(img, [np.array(fut, np.int32)], False, tint(color, 0.35), 1, cv2.LINE_AA)
    if len(past) > 1:
        cv2.polylines(img, [np.array(past, np.int32)], False, color, 2, cv2.LINE_AA)
    cx = int(xs[cur])
    cv2.line(img, (cx, y0), (cx, y0 + h), TEXT2, 1, cv2.LINE_AA)
    if ok[cur]:
        cv2.circle(img, (cx, int(ys[cur])), 4, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, int(ys[cur])), 4, SURFACE, 1, cv2.LINE_AA)


def tile(img, x0, y0, w, h, name, value_s, delta_s, z, vals, cur, ref, ylim=None, color=BLUE):
    strength = min(0.55, abs(z) / 2.5) if np.isfinite(z) else 0.0
    bg = tint(badness_color(z), strength)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), bg, -1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), GRID, 1)
    # left colour bar = current goodness
    cv2.rectangle(img, (x0, y0), (x0 + 6, y0 + h), badness_color(z), -1)
    text(img, name, (x0 + 14, y0 + 17), 0.46, TEXT2)
    text(img, value_s, (x0 + 14, y0 + 48), 0.85, TEXT, 2)
    text(img, delta_s, (x0 + 14, y0 + 68), 0.40, TEXT2)
    sparkline(img, x0 + 150, y0 + 24, w - 162, h - 32, vals, cur, ref, color, ylim)


def timeline(img, x0, y0, w, h, B, cur, windows, ts, blocked=()):
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), PANEL, -1)
    n = len(B)
    def X(i):
        return int(x0 + (i / max(n - 1, 1)) * w)
    BLOCK = (110, 110, 110)
    for i in range(n):
        if int(ts[i]) in blocked:
            # faint grey column for the blocked frame + dark bar at the bottom
            cv2.rectangle(img, (X(i), y0), (X(i + 1) if i + 1 < n else x0 + w, y0 + h), (232, 232, 232), -1)
            cv2.rectangle(img, (X(i), y0 + h - 10), (X(i + 1) if i + 1 < n else x0 + w, y0 + h), BLOCK, -1)
    for kind, rows, col in (("D", windows.get("declines", []), RED), ("R", windows.get("recoveries", []), GREEN)):
        for k, r in enumerate(rows, 1):
            a, b = r["window"]
            ia, ib = int(np.searchsorted(ts, a)), int(np.searchsorted(ts, b))
            cv2.rectangle(img, (X(ia), y0), (X(ib + 1) if ib + 1 < n else x0 + w, y0 + h), tint(col, 0.28), -1)
            if "ramp" in r and kind == "D":
                ra, rb = r["ramp"]
                ia2, ib2 = int(np.searchsorted(ts, ra)), int(np.searchsorted(ts, rb))
                cv2.rectangle(img, (X(ia2), y0), (X(ib2 + 1) if ib2 + 1 < n else x0 + w, y0 + h), tint(col, 0.45), -1)
            row = (0 if kind == "D" else 2) + (k % 2)
            text(img, f"{kind}{k} Z={r['z']:.1f}", (X(ia) + 3, y0 + 13 + 12 * row), 0.38, col, 1)
    lo, hi = float(np.nanmin(B)), float(np.nanmax(B))
    pad = (hi - lo) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad
    y_of = lambda v: int(y0 + h - (v - lo) / (hi - lo) * h)
    cv2.line(img, (x0, y_of(0.0)), (x0 + w, y_of(0.0)), TEXT2, 1, cv2.LINE_AA)
    # aggregate line: violet where memory was written, grey over blocked frames
    # (a segment is grey when the frame it leads INTO was blocked)
    for i in range(1, n):
        if not (np.isfinite(B[i - 1]) and np.isfinite(B[i])):
            continue
        col = BLOCK if int(ts[i]) in blocked else VIOLET
        cv2.line(img, (X(i - 1), y_of(B[i - 1])), (X(i), y_of(B[i])), col, 2, cv2.LINE_AA)
    if blocked:
        text(img, "grey line / dark bar = memory write BLOCKED", (x0 + w - 300, y0 + h - 4), 0.4, BLOCK)
    cv2.line(img, (X(cur), y0), (X(cur), y0 + h), TEXT, 2, cv2.LINE_AA)
    if np.isfinite(B[cur]):
        cv2.circle(img, (X(cur), y_of(B[cur])), 5, BLOCK if int(ts[cur]) in blocked else VIOLET, -1, cv2.LINE_AA)
        cv2.circle(img, (X(cur), y_of(B[cur])), 5, SURFACE, 1, cv2.LINE_AA)
    text(img, "aggregate badness (mean robust z of 5 metrics); red = decline window (darker = ramp), green = recovery",
         (x0, y0 + h + 16), 0.42, TEXT2)
    # frame ticks
    for f in range(0, n, 10):
        text(img, str(int(ts[f])), (X(f) - 6, y0 + h + 32), 0.36, TEXT2)


def active_windows(t, windows):
    out = []
    for kind, rows in (("DECLINE", windows.get("declines", [])), ("RECOVERY", windows.get("recoveries", []))):
        for k, r in enumerate(rows, 1):
            a, b = r["window"]
            in_ramp = "ramp" in r and r["ramp"][0] <= t <= r["ramp"][1]
            if a <= t <= b or in_ramp:
                drv = ", ".join(f"{m} {int(round(100*abs(s)))}%" for m, s in list(r["drivers"].items())[:2])
                out.append((kind, k, in_ramp, drv))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="<scene>/dense/rgb")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--windows", default=None, help="decline_windows.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--hold_first", type=int, default=6, help="repeat frame 0 this many times")
    ap.add_argument("--title", default=None)
    ap.add_argument("--blocked", default=None,
                    help="json of frames whose memory write was blocked ({scene: [..]} or [..]); "
                    "drawn as a grey band on the timeline and a badge on those frames")
    ap.add_argument("--snap", type=int, nargs="*", default=None,
                    help="frame numbers to also save as PNG (default: first, middle)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frames, "*.png")) + glob.glob(os.path.join(args.frames, "*.jpg")))
    ts, raw, scene = load_csv(args.csv)
    n = len(ts)
    assert len(paths) >= n, f"{len(paths)} frames on disk, {n} rows in CSV"
    windows = json.load(open(args.windows)) if args.windows and os.path.isfile(args.windows) else {}
    blocked = set()
    if args.blocked and os.path.isfile(args.blocked):
        bj = json.load(open(args.blocked))
        blocked = set(int(x) for x in (next(iter(bj.values())) if isinstance(bj, dict) else bj))
    z = {m: robust_z(raw[m], m) for m in METRICS}
    B = np.nanmean(np.vstack([z[m] for m in METRICS]), axis=0)
    title = args.title or os.path.basename(os.path.dirname(os.path.abspath(args.csv)))

    # canvas geometry
    W, H = 1280, 760
    FW, FH = 900, 506           # frame area (16:9 of 320x180 -> x2.8125)
    fx0, fy0 = 20, 44
    tx0, tw = fx0 + FW + 16, W - (fx0 + FW + 16) - 16
    th = 76
    tl_y0, tl_h = fy0 + FH + 24, 120

    ffmpeg = None
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    if ffmpeg:
        import subprocess
        proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", args.out],
            stdin=subprocess.PIPE)
        write = lambda fr: proc.stdin.write(fr.tobytes())
    else:
        vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        write = vw.write

    for i in range(n):
        frame = cv2.imread(paths[i])
        canvas = np.full((H, W, 3), SURFACE, np.uint8)
        text(canvas, f"{title}", (fx0, 28), 0.6, TEXT, 1)
        # ---- frame with badness border
        col = badness_color(B[i])
        big = cv2.resize(frame, (FW, FH), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(canvas, (fx0 - 8, fy0 - 8), (fx0 + FW + 8, fy0 + FH + 8), col, -1)
        canvas[fy0:fy0 + FH, fx0:fx0 + FW] = big
        # badge(s)
        acts = active_windows(int(ts[i]), windows)
        yb = fy0 + 12
        status_col = col
        if not acts:
            lbl = "steady" if abs(B[i]) < 0.75 else ("BAD (high error)" if B[i] > 0 else "GOOD (low error)")
            cv2.rectangle(canvas, (fx0 + 10, yb), (fx0 + 10 + 250, yb + 30), SURFACE, -1)
            cv2.rectangle(canvas, (fx0 + 10, yb), (fx0 + 10 + 250, yb + 30), status_col, 2)
            text(canvas, lbl, (fx0 + 18, yb + 21), 0.6, TEXT, 2)
        for kind, k, in_ramp, drv in acts:
            c = RED if kind == "DECLINE" else GREEN
            s = f"{kind} {'D' if kind == 'DECLINE' else 'R'}{k}" + (" - ramp" if in_ramp else "")
            cv2.rectangle(canvas, (fx0 + 10, yb), (fx0 + 10 + 420, yb + 48), SURFACE, -1)
            cv2.rectangle(canvas, (fx0 + 10, yb), (fx0 + 10 + 420, yb + 48), c, 3)
            text(canvas, s, (fx0 + 18, yb + 21), 0.65, c, 2)
            text(canvas, f"driven by {drv}", (fx0 + 18, yb + 40), 0.45, TEXT, 1)
            yb += 56
        if int(ts[i]) in blocked:
            cv2.rectangle(canvas, (fx0 + 10, yb), (fx0 + 10 + 420, yb + 30), (60, 60, 60), -1)
            text(canvas, "MEMORY WRITE BLOCKED (predict only, state frozen)", (fx0 + 18, yb + 21), 0.5, SURFACE, 1)
            yb += 38
        # frame counter + aggregate value, top-right of frame
        fc = f"frame {int(ts[i])} / {int(ts[-1])}"
        (tw_, _), _ = cv2.getTextSize(fc, FONT, 0.6, 2)
        cv2.rectangle(canvas, (fx0 + FW - tw_ - 26, fy0 + 12), (fx0 + FW - 10, fy0 + 42), SURFACE, -1)
        text(canvas, fc, (fx0 + FW - tw_ - 18, fy0 + 34), 0.6, TEXT, 2)

        # ---- tiles
        y = fy0 - 8
        bz = B[i]
        tile(canvas, tx0, y, tw, th, "aggregate badness (robust z, 5 metrics)",
             f"{bz:+.2f}" if np.isfinite(bz) else "n/a",
             "0 = scene-typical; >0 worse than typical", bz, B, i, 0.0,
             ylim=(float(np.nanmin(B)) - 0.2, float(np.nanmax(B)) + 0.2), color=VIOLET)
        y += th + 8
        for m in METRICS:
            v = raw[m][i]
            ref = scene.get(m, math.nan)
            val_s = FMT[m].format(v) if np.isfinite(v) else "n/a"
            if np.isfinite(v) and np.isfinite(ref):
                d = v - ref
                good = d < 0 if BETTER[m] == "lower" else d > 0
                delta_s = f"{'+' if d >= 0 else '-'}{FMT[m].format(abs(d))} vs scene ({'better' if good else 'worse'})"
            else:
                delta_s = "no value (frame 0 has no RPE)" if not np.isfinite(v) else ""
            ylim = (0.0, 1.02) if m == "a1" else None
            tile(canvas, tx0, y, tw, th, f"{LABEL[m]}  [{BETTER[m]} is better]", val_s, delta_s,
                 z[m][i], raw[m], i, ref, ylim)
            y += th + 8

        # ---- timeline
        timeline(canvas, fx0, tl_y0, W - 2 * fx0, tl_h, B, i, windows, ts, blocked)


        reps = args.hold_first if i == 0 else 1
        for _ in range(reps):
            write(canvas)
        snaps = args.snap if args.snap is not None else [int(ts[0]), int(ts[n // 2])]
        if int(ts[i]) in snaps:
            cv2.imwrite(os.path.splitext(args.out)[0] + f"_frame{int(ts[i]):03d}.png", canvas)

    if ffmpeg:
        proc.stdin.close(); proc.wait()
    else:
        vw.release()
    print(f"wrote {args.out} ({n} frames @ {args.fps} fps, {W}x{H})")


if __name__ == "__main__":
    main()
