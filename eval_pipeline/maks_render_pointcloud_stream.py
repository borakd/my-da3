#!/usr/bin/env python3
"""Side-by-side 3D point-cloud streams: ground truth vs CUT3R vs CUT3R + causal
pose recalibration, accumulating frame by frame.

Why depth + pose and not a directly emitted pointmap: the recalibration modifies
ONLY the camera poses (depth is symlinked through untouched and scores
bit-identically). A world-frame pointmap emitted by the model would therefore be
completely insensitive to the method. Unprojecting the predicted depth through
the predicted pose is the one representation in which the correction is visible,
and it is also exactly what eval_depth_poses.py scores.

Each predicted stream is placed in the GT frame by the SAME similarity transform
the evaluator uses: Umeyama with scale on the camera centres, fit over the whole
sequence, then applied to the points and the trajectory alike. No per-frame
fudge, so accumulated drift shows up as smear exactly as the metric sees it.

Rendering is a hand-rolled painter's-algorithm splat (far to near) through one
fixed virtual camera shared by all panels, at 2x then downsampled, so the three
clouds are directly comparable pixel for pixel.

    python eval_pipeline/maks_render_pointcloud_stream.py --scene <name> --out x.mp4
"""
import argparse
import csv
import math
import os

import cv2
import numpy as np

OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"

BG = (250, 250, 248)
INK = (60, 60, 58)
MUTE = (150, 150, 146)
GT_COL = (90, 170, 90)      # BGR
BASE_COL = (70, 70, 225)
OURS_COL = (215, 140, 40)


# ---------------------------------------------------------------- geometry ---
def load_poses(d, n, key="pose"):
    P = np.full((n, 4, 4), np.nan)
    for t in range(n):
        p = f"{d}/{t:06d}.npz"
        if not os.path.isfile(p):
            continue
        z = np.load(p)
        M = np.asarray(z[key] if key in z else z[list(z.keys())[0]], float)
        if M.shape == (3, 4):
            M = np.vstack([M, [0, 0, 0, 1]])
        P[t] = M
    return P


def umeyama(src, dst):
    """Similarity (s, R, t) minimising ||s R src + t - dst||, as the evaluator fits it."""
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


def unproject(depth, K, stride):
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[ys, xs].astype(np.float64)
    ok = np.isfinite(z) & (z > 0)
    xs, ys, z = xs[ok], ys[ok], z[ok]
    x = (xs - K[0, 2]) / K[0, 0] * z
    y = (ys - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], 1), (ys, xs)


# ----------------------------------------------------------------- renderer ---
class View:
    """One fixed look-at camera shared by every panel."""

    def __init__(self, centre, radius, W, H, az=35.0, el=28.0, dist=2.8, ss=2, fill=0.62):
        self.W, self.H, self.ss = W, H, ss
        a, e = math.radians(az), math.radians(el)
        d = np.array([math.cos(e) * math.cos(a), math.sin(e), math.cos(e) * math.sin(a)])
        eye = centre + d * radius * dist
        f = (centre - eye)
        f /= np.linalg.norm(f)
        up = np.array([0.0, 1.0, 0.0])
        r = np.cross(f, up)
        r /= np.linalg.norm(r) or 1.0
        u = np.cross(r, f)
        self.Rv = np.stack([r, u, f])          # world -> view rows
        self.eye = eye
        # `dist` sets perspective strength only: the focal length scales with it so the
        # subject subtends a constant angle. `fill` is the actual zoom -- the fraction
        # of the half-panel the scene radius occupies.
        self.foc = 0.5 * min(W, H) * ss * dist * fill
        self.cx, self.cy = W * ss / 2.0, H * ss / 2.0

    def project(self, P):
        V = (P - self.eye) @ self.Rv.T
        z = V[:, 2]
        ok = z > 1e-6
        u = self.cx + self.foc * V[:, 0] / np.where(ok, z, 1)
        v = self.cy - self.foc * V[:, 1] / np.where(ok, z, 1)
        return u, v, z, ok

    def new_buffer(self):
        Wp, Hp = int(self.W * self.ss), int(self.H * self.ss)
        return (np.full((Hp, Wp, 3), BG, np.uint8), np.full(Hp * Wp, np.inf, np.float64))

    def splat_into(self, buf, P, C):
        """Depth-test one frame's points into a persistent (image, zbuffer) pair.

        Incremental: cost is O(points in THIS frame), not O(cloud so far), which
        is what makes a 700-frame sequence tractable. Result is identical to
        re-splatting the whole accumulated cloud far-to-near.
        """
        img, zb = buf
        Wp, Hp = int(self.W * self.ss), int(self.H * self.ss)
        if len(P) == 0:
            return buf
        u, v, z, ok = self.project(P)
        x = np.round(u).astype(np.int64)
        y = np.round(v).astype(np.int64)
        ok &= (x >= 0) & (x < Wp) & (y >= 0) & (y < Hp)
        x, y, z, c = x[ok], y[ok], z[ok], C[ok]
        if len(x) == 0:
            return buf
        idx = y * Wp + x
        order = np.argsort(z)                       # near first
        idx, z, c = idx[order], z[order], c[order]
        uniq, first = np.unique(idx, return_index=True)   # nearest per pixel
        zu, cu = z[first], c[first]
        win = zu < zb[uniq]
        zb[uniq[win]] = zu[win]
        img.reshape(-1, 3)[uniq[win]] = cu[win]
        return img, zb

    def polyline(self, img, P, colour, thick=2):
        if len(P) < 2:
            return img
        u, v, z, ok = self.project(P)
        pts = np.stack([u, v], 1)[ok].astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, colour, thick, cv2.LINE_AA)
        return img

    def finish(self, img):
        return cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)


# -------------------------------------------------------------------- text ---
def put(img, s, org, scale=0.42, col=INK, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def per_frame_ate(csv_path, n):
    out = np.full(n, np.nan)
    try:
        rows = list(csv.DictReader(open(csv_path)))
    except Exception:
        return out
    for r in rows:
        if r.get("camera_id", "").strip() != "0":
            continue
        try:
            t = int(r["local_timestep"].strip())
            out[t] = float(r["ate"])
        except (ValueError, TypeError, KeyError):
            continue
    return out


# -------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--gt_root", default=ROOT)
    ap.add_argument("--base_label", default="augfull_lr1e5")
    ap.add_argument("--ours_label", default="causal_posefix_tc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--panel", type=int, nargs=2, default=[520, 390])
    ap.add_argument("--az", type=float, default=35.0)
    ap.add_argument("--el", type=float, default=26.0)
    ap.add_argument("--dist", type=float, default=3.3, help="perspective strength, not zoom")
    ap.add_argument("--fill", type=float, default=0.46,
                    help="zoom: fraction of the half-panel the scene radius fills")
    ap.add_argument("--still", action="store_true",
                    help="write only the final composite as a PNG (for framing tuning)")
    ap.add_argument("--depth_pct", type=float, default=97.0,
                    help="drop the far tail of each depth map at this percentile")
    a = ap.parse_args()

    d = f"{a.gt_root}/{a.scene}/dense"
    rgbs = sorted(f for f in os.listdir(f"{d}/rgb") if f.endswith(".png"))
    n = len(rgbs)
    Pw, Ph = a.panel

    gtP = load_poses(f"{d}/cam", n)
    streams = {}
    for name, lab in (("base", a.base_label), ("ours", a.ours_label)):
        P = load_poses(f"{OUT}/{lab}/preds/{a.scene}/camera", n)
        ok = [t for t in range(n) if np.all(np.isfinite(P[t])) and np.all(np.isfinite(gtP[t]))]
        s, R, t = umeyama(P[ok][:, :3, 3], gtP[ok][:, :3, 3])
        streams[name] = {"P": P, "sim3": (s, R, t), "label": lab,
                         "ate": per_frame_ate(f"{OUT}/{lab}/eval/{a.scene}/eval_depth_pose_metrics.csv", n)}
        print(f"[{name}] {lab}: sim3 scale {s:.4f}, {len(ok)}/{n} frames")

    gtK = np.load(f"{d}/cam/000000.npz")["intrinsic"].astype(float)
    prK = np.load(f"{OUT}/{a.base_label}/preds/{a.scene}/camera/000000.npz")["intrinsics"].astype(float)

    # --- pass 1: GT cloud, to fix the shared view box -----------------------
    def gt_points(t):
        z = np.load(f"{d}/depth/{t:06d}.npy").astype(np.float64)
        z[z <= 0] = np.nan
        hi = np.nanpercentile(z, a.depth_pct) if np.isfinite(z).any() else 0
        z[z > hi] = np.nan
        pts, (ys, xs) = unproject(np.nan_to_num(z, nan=-1), gtK, a.stride)
        img = cv2.imread(f"{d}/rgb/{t:06d}.png")
        img = cv2.resize(img, (z.shape[1], z.shape[0]))
        return (gtP[t][:3, :3] @ pts.T).T + gtP[t][:3, 3], img[ys, xs]

    gt_cache = {}
    for t in range(n):
        if np.all(np.isfinite(gtP[t])):
            gt_cache[t] = gt_points(t)
    allgt = np.concatenate([v[0] for v in gt_cache.values()])
    lo, hi = np.percentile(allgt, [4, 96], axis=0)
    centre = (lo + hi) / 2
    radius = float(np.max(hi - lo)) / 2
    view = View(centre, radius, Pw, Ph, a.az, a.el, a.dist, fill=a.fill)
    print(f"view centre {np.round(centre,3)} radius {radius:.3f}")

    def pred_points(t, name):
        z = np.load(f"{OUT}/{streams[name]['label']}/preds/{a.scene}/depth/{t:06d}.npy").astype(np.float64)
        hi_ = np.nanpercentile(z, a.depth_pct) if np.isfinite(z).any() else 0
        z = np.where((z > 0) & (z <= hi_), z, -1)
        pts, (ys, xs) = unproject(z, prK, a.stride)
        img = cv2.imread(f"{d}/rgb/{t:06d}.png")
        img = cv2.resize(img, (z.shape[1], z.shape[0]))
        P = streams[name]["P"][t]
        w = (P[:3, :3] @ pts.T).T + P[:3, 3]
        s, R, tr = streams[name]["sim3"]
        return (s * (R @ w.T).T + tr), img[ys, xs]

    HDR, ROWLAB = 96, 22
    Wtot = Pw * 3 + 4 * 4
    Htot = HDR + 2 * (Ph + ROWLAB) + 10
    vw = None
    if not a.still:
        vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (Wtot, Htot))
        if not vw.isOpened():
            raise SystemExit(f"cannot open writer for {a.out}")

    KEYS = ("gt", "base", "ours")
    buf = {(k, m): view.new_buffer() for k in KEYS for m in ("rgb", "time")}

    def tcol(t):
        """Frame index -> BGR, so a surface seen early and late shows as two offset copies."""
        v = np.uint8([[int(255 * t / max(n - 1, 1))]])
        return cv2.applyColorMap(v, cv2.COLORMAP_TURBO)[0, 0].astype(np.uint8)
    gt_c_traj, base_c_traj, ours_c_traj = [], [], []
    for t in range(n):
        ct = tcol(t)
        if t in gt_cache:
            p, c = gt_cache[t]
            buf[("gt", "rgb")] = view.splat_into(buf[("gt", "rgb")], p, c)
            buf[("gt", "time")] = view.splat_into(buf[("gt", "time")], p, np.repeat(ct[None], len(p), 0))
            gt_c_traj.append(gtP[t][:3, 3])
        for name, traj in (("base", base_c_traj), ("ours", ours_c_traj)):
            P = streams[name]["P"][t]
            if not np.all(np.isfinite(P)):
                continue
            p, c = pred_points(t, name)
            buf[(name, "rgb")] = view.splat_into(buf[(name, "rgb")], p, c)
            buf[(name, "time")] = view.splat_into(buf[(name, "time")], p, np.repeat(ct[None], len(p), 0))
            s, R, tr = streams[name]["sim3"]
            traj.append(s * (R @ P[:3, 3]) + tr)

        canvas = np.full((Htot, Wtot, 3), BG, np.uint8)
        titles = [("Ground truth", "gt", GT_COL, None),
                  ("CUT3R finetuned", "base", BASE_COL, "base"),
                  ("+ causal pose recalibration", "ours", OURS_COL, "ours")]
        for r, mode in enumerate(("rgb", "time")):
            y0 = HDR + r * (Ph + ROWLAB)
            for i, (title, key, col, mkey) in enumerate(titles):
                img = buf[(key, mode)][0].copy()
                view.polyline(img, np.array(gt_c_traj), GT_COL, 3)
                if key != "gt":
                    view.polyline(img, np.array(base_c_traj if key == "base" else ours_c_traj), col, 3)
                img = view.finish(img)
                cv2.rectangle(img, (0, 0), (Pw - 1, Ph - 1), (225, 225, 222), 1)
                x0 = 4 + i * (Pw + 4)
                canvas[y0:y0 + Ph, x0:x0 + Pw] = img
                if r == 0:
                    put(canvas, title, (x0 + 8, HDR - 10), 0.5, col, 1)
                if mkey and r == 1:
                    e = streams[mkey]["ate"]
                    cur = e[t] if t < len(e) else np.nan
                    run = np.sqrt(np.nanmean(e[:t + 1] ** 2)) if np.isfinite(e[:t + 1]).any() else np.nan
                    cv2.rectangle(canvas, (x0, y0 + Ph - 22), (x0 + Pw, y0 + Ph), BG, -1)
                    put(canvas, f"ATE now {cur:.4f} m    RMS so far {run:.4f} m",
                        (x0 + 8, y0 + Ph - 7), 0.42, INK)
            lab = ("true colour --- ghosting = the same surface landing in two places"
                   if mode == "rgb" else
                   "coloured by frame index (blue = early, red = late) --- drift separates the copies")
            put(canvas, lab, (8, y0 + Ph + 16), 0.40, MUTE)

        put(canvas, a.scene, (10, 26), 0.52, INK, 1)
        put(canvas, f"frame {t+1}/{n}   |   points accumulate through the sequence; "
                    f"each predicted stream is Sim3-aligned to GT exactly as eval_depth_poses.py aligns it",
            (10, 48), 0.40, MUTE)
        eb = streams["base"]["ate"]; eo = streams["ours"]["ate"]
        rb = np.sqrt(np.nanmean(eb[:t + 1] ** 2)) if np.isfinite(eb[:t + 1]).any() else np.nan
        ro = np.sqrt(np.nanmean(eo[:t + 1] ** 2)) if np.isfinite(eo[:t + 1]).any() else np.nan
        if np.isfinite(rb) and np.isfinite(ro) and rb > 0:
            put(canvas, f"ATE so far   baseline {rb:.4f} m    ours {ro:.4f} m    ({(ro-rb)/rb:+.1%})",
                (10, 72), 0.46, OURS_COL if ro < rb else BASE_COL, 1)
        bar = int((t + 1) / n * (Wtot - 20))
        cv2.rectangle(canvas, (10, Htot - 6), (10 + bar, Htot - 3), INK, -1)
        if vw is not None:
            vw.write(canvas)
        if (t + 1) % 50 == 0:
            print(f"  {t+1}/{n}", flush=True)
    if vw is not None:
        vw.release()
        print(f"wrote {a.out}  ({n} frames, {Wtot}x{Htot})")
    else:
        cv2.imwrite(a.out, canvas)
        print(f"wrote {a.out}  (still, {Wtot}x{Htot})")


if __name__ == "__main__":
    main()
