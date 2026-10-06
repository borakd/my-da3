"""Track gripper points from the birth frame to the end of each DROID episode with Track-On (causal).

Input: the work list built from the RobotSeg birth-frame masks
    $WORK/bora/outputs/droid_birth_frames/trackon_run/worklist.json
    (one item per (scene, exterior camera): birth frame t_b, mask PNG at 1280x720, MP4 path)
Per item:
  1. sample query points inside the (eroded) birth-frame mask on a uniform grid; adaptive stride so
     that the count lands near --target points (cap --max_points, stride >= --min_stride)
  2. decode the MP4 with PyAV from frame 0, skip frames < t_b (no model calls), then feed frames
     t_b .. end to CausalTracker: queries injected at t_b, nothing is re-segmented later
  3. save tracks/<ep>/<cam>_f<t_b>.npz:
       tracks      (T, N, 2) float32, pixel coords in the 1280x720 frame, NaN for frames < t_b
       visibility  (T, N) bool (False before t_b)
       queries     (N, 3) float32 [t_b, x, y]
       meta        json string: episode, cam, birth_frame, n_frames_store, n_frames_mp4, frame size,
                   fps, mask path, store_scale (0.25 -> the 320x180 pointworld_droid_ext frames)
Resumable: items whose npz exists are skipped. Sharded with --shard/--nshards (index modulo).
Optionally renders a QA video for every --render_every-th item.
"""
import argparse
import json
import os
import sys
import time

import av
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # causal_track.py sits alongside
from causal_track import CausalTracker  # noqa: E402

OUT_ROOT = "/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/trackon_run"
STORE_SCALE = 0.25  # 1280x720 -> 320x180 (pointworld_droid_ext uses the same MP4 frames, bilinear)


def sample_mask_points(mask, target=64, max_points=100, min_stride=4, erode=5):
    """Uniform grid inside the eroded mask; stride chosen so ~target points survive."""
    m = mask.astype(np.uint8)
    if erode > 0:
        er = cv2.erode(m, np.ones((erode, erode), np.uint8)) > 0
        if er.sum() < 20:  # tiny mask: do not erode it away
            er = m > 0
    else:
        er = m > 0
    ys, xs = np.where(er)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32), 0
    area = len(xs)
    stride = max(min_stride, int(round(np.sqrt(area / target))))
    while True:
        gy = np.arange(ys.min(), ys.max() + 1, stride)
        gx = np.arange(xs.min(), xs.max() + 1, stride)
        GX, GY = np.meshgrid(gx, gy)
        keep = er[GY.ravel(), GX.ravel()]
        pts = np.stack([GX.ravel()[keep], GY.ravel()[keep]], 1).astype(np.float32)
        if len(pts) <= max_points or stride >= 64:
            break
        stride += 1
    if len(pts) > max_points:  # still too many at the stride cap: subsample deterministically
        idx = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
        pts = pts[idx]
    return pts, stride


def render_video(frames_rgb, tracks, vis, t_b, path, fps=15.0):
    import imageio
    T, N, _ = tracks.shape
    cols = (np.stack([np.linspace(0, 1, N)] * 3, 1) * 0 + 1)
    hsv = np.zeros((N, 1, 3), np.uint8)
    hsv[:, 0, 0] = (np.linspace(0, 179, N)).astype(np.uint8)
    hsv[:, 0, 1:] = 255
    cols = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[:, 0, :]
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1) as wr:
        for t, fr in enumerate(frames_rgb):
            img = fr.copy()
            if t >= t_b:
                for n in range(N):
                    x, y = tracks[t, n]
                    if np.isnan(x):
                        continue
                    c = tuple(int(v) for v in cols[n])
                    if vis[t, n]:
                        cv2.circle(img, (int(x), int(y)), 4, c, -1)
                    else:
                        cv2.circle(img, (int(x), int(y)), 4, c, 1)
            cv2.putText(img, f"f{t}" + ("" if t >= t_b else "  (before birth)"), (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            wr.append_data(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", default=f"{OUT_ROOT}/worklist.json")
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--items", nargs="*", default=None, help="explicit '<ep>:<cam>' items (pilot)")
    ap.add_argument("--target", type=int, default=64)
    ap.add_argument("--max_points", type=int, default=100)
    ap.add_argument("--min_stride", type=int, default=4)
    ap.add_argument("--support_grid", type=int, default=20)
    ap.add_argument("--render_every", type=int, default=0, help="render a QA mp4 for every k-th item (0 = never)")
    ap.add_argument("--render_all_items", action="store_true")
    args = ap.parse_args()

    work = json.load(open(args.worklist))
    if args.items:
        want = set(args.items)
        work = [w for w in work if f"{w['episode']}:{w['cam']}" in want]
    else:
        work = [w for i, w in enumerate(work) if i % args.nshards == args.shard]
    os.makedirs(f"{args.out}/tracks", exist_ok=True)
    os.makedirs(f"{args.out}/qa_videos", exist_ok=True)
    log_path = f"{args.out}/logs/track_shard{args.shard}.jsonl"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"shard {args.shard}/{args.nshards}: {len(work)} items", flush=True)

    tracker = CausalTracker(support_grid_size=args.support_grid)
    t_start = time.time()
    n_done = n_skip = 0
    frames_total = 0
    for k, w in enumerate(work):
        ep, cam, t_b = w["episode"], w["cam"], int(w["frame"])
        stem = f"{cam}_f{t_b:05d}"
        out_npz = f"{args.out}/tracks/{ep}/{stem}.npz"
        if os.path.isfile(out_npz):
            n_skip += 1
            continue
        t0 = time.time()
        rec = dict(episode=ep, cam=cam, birth_frame=t_b, n_frames_store=w["n_frames"])
        mask = cv2.imread(w["mask"], 0)
        if mask is None:
            rec["status"] = "mask_missing"
            open(log_path, "a").write(json.dumps(rec) + "\n")
            continue
        mask = mask > 0
        pts, stride = sample_mask_points(mask, args.target, args.max_points, args.min_stride)
        rec.update(n_points=int(len(pts)), grid_stride=int(stride), mask_area=float(mask.mean()))
        if len(pts) == 0:
            rec["status"] = "empty_mask"
            open(log_path, "a").write(json.dumps(rec) + "\n")
            continue
        try:
            container = av.open(w["mp4"])
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"open_failed:{e}"
            open(log_path, "a").write(json.dumps(rec) + "\n")
            continue
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else 15.0
        tracks, viss, frames_for_render = [], [], []
        render = args.render_all_items or (args.render_every and (k % args.render_every == 0))
        tracker.reset()
        H = W = None
        t = -1
        tracker_error = None
        try:
            for t, frame in enumerate(container.decode(stream)):
                if t < t_b:
                    if render:
                        frames_for_render.append(frame.to_ndarray(format="rgb24"))
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                try:
                    if H is None:
                        H, W = rgb.shape[:2]
                        sx, sy = W / mask.shape[1], H / mask.shape[0]
                        q = pts * np.array([sx, sy], np.float32)  # mask coords -> frame coords
                        p, v = tracker.step(rgb, new_points=q.tolist())
                    else:
                        p, v = tracker.step(rgb)
                except Exception as e:  # tracker failure: the item is NOT saved (resume will redo it)
                    tracker_error = str(e)[:300]
                    break
                tracks.append(p.numpy())
                viss.append(v.numpy())
                if render:
                    frames_for_render.append(rgb)
        except Exception as e:  # PyAV decode error: keep what was tracked, record it
            rec["decode_error"] = str(e)[:200]
        finally:
            container.close()
        n_mp4 = t + 1
        if tracker_error is not None:
            rec.update(status="tracker_error", error=tracker_error, n_frames_mp4=n_mp4, tracked_frames=len(tracks))
            open(log_path, "a").write(json.dumps(rec) + "\n")
            print(f"  TRACKER ERROR on {ep} {cam}: {tracker_error}", flush=True)
            torch.cuda.empty_cache()
            continue
        if not tracks:
            rec["status"] = "no_frames_after_birth"
            rec["n_frames_mp4"] = n_mp4
            open(log_path, "a").write(json.dumps(rec) + "\n")
            continue
        N = len(pts)
        T = n_mp4
        tr = np.full((T, N, 2), np.nan, np.float32)
        vi = np.zeros((T, N), bool)
        tr[t_b:t_b + len(tracks)] = np.stack(tracks)
        vi[t_b:t_b + len(viss)] = np.stack(viss)
        meta = dict(episode=ep, cam=cam, birth_frame=t_b, n_frames_store=w["n_frames"], n_frames_mp4=n_mp4,
                    n_frames_tracked=len(tracks), mp4_shorter_than_store=bool(n_mp4 < w["n_frames"]),
                    decode_error=rec.get("decode_error"),
                    frame_hw=[H, W], mp4_header_fps=fps, store_fps=15.0,
                    time_axis="row i == MP4 frame i == store frame i (DROID 15 Hz); mp4_header_fps is the container's nominal rate, not the sample rate",
                    mask=w["mask"], store_scale=STORE_SCALE,
                    support_grid=args.support_grid, delta_v=float(getattr(tracker.predictor, "delta_v", 0.8)),
                    ckpt=os.environ.get("TRACKON_CKPT", ""), grid_stride=stride, coords="pixels in the decoded MP4 frame (1280x720); multiply by store_scale for pointworld_droid_ext 320x180")
        os.makedirs(os.path.dirname(out_npz), exist_ok=True)
        tmp_npz = out_npz[:-4] + ".tmp.npz"
        np.savez_compressed(tmp_npz, tracks=tr, visibility=vi,
                            queries=np.concatenate([np.full((N, 1), t_b, np.float32), pts * np.array([sx, sy], np.float32)], 1),
                            meta=json.dumps(meta))
        os.replace(tmp_npz, out_npz)
        if render:
            try:
                render_video(frames_for_render, tr, vi, t_b, f"{args.out}/qa_videos/{ep}__{stem}.mp4", fps=15.0)
            except Exception as e:  # noqa: BLE001
                rec["render_error"] = str(e)[:200]
        dt = time.time() - t0
        frames_total += len(tracks)
        rec.update(status="ok", n_frames_mp4=n_mp4, tracked_frames=len(tracks), seconds=round(dt, 2),
                   fps_tracking=round(len(tracks) / max(dt, 1e-6), 1),
                   gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
                   vis_frac_mean=float(vi[t_b:].mean()), vis_frac_last=float(vi[t_b + len(viss) - 1].mean()))
        open(log_path, "a").write(json.dumps(rec) + "\n")
        n_done += 1
        if n_done % 20 == 0:
            el = time.time() - t_start
            print(f"  {k+1}/{len(work)} items | done {n_done} skipped {n_skip} | {el/60:.1f} min | {frames_total/el:.1f} frames/s overall", flush=True)
    el = time.time() - t_start
    print(f"shard {args.shard} finished: {n_done} tracked, {n_skip} skipped, {frames_total} frames in {el/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
