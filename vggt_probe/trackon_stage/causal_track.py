"""
Causal (online, frame-by-frame) point tracking with Track-On.

The video is decoded one frame at a time and each frame is fed to
``Predictor.forward_frame``; the model only ever sees past and current
frames, so the output is strictly causal. New query points may be added
at any frame.

Importable API
--------------
    from causal_track import CausalTracker

    tracker = CausalTracker(ckpt="/path/to/track_on_r.pt")   # device="cuda"
    tracker.reset()
    for frame in frames:                       # frame: (H, W, 3) uint8 RGB numpy or torch
        pts, vis = tracker.step(frame, new_points=[(x, y), ...] or None)
        # pts: (N_active, 2) float32 (x, y) in input-pixel coordinates
        # vis: (N_active,) bool

CLI
---
    python causal_track.py --video in.mp4 --grid 10                       # 10x10 grid at frame 0
    python causal_track.py --video in.mp4 --points "190,190;200,190"      # (x,y) at frame 0
    python causal_track.py --video in.mp4 --points "190,190,0;300,120,15" # (x,y,t)
    python causal_track.py --video in.mp4 --queries-file q.json           # [[t,x,y], ...]

Outputs (in --output-dir):
    tracks.npz   tracks (T, N, 2) float32, NaN before a query's start frame
                 visibility (T, N) bool, queries (N, 3) [t, x, y]
    tracks.mp4   rendered visualisation (unless --no-video)
"""

import argparse
import json
import os
import sys
import time
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Upstream Track-On code (model/, utils/) lives in the clone at $TRACKON_ROOT (exported by env.sh).
TRACKON_ROOT = os.environ.get("TRACKON_ROOT", os.path.expanduser("~/track_on"))
sys.path.insert(0, TRACKON_ROOT)

from model.trackon_predictor import Predictor  # noqa: E402
from utils.coord_utils import get_points_on_a_grid  # noqa: E402
from utils.train_utils import load_args_from_yaml  # noqa: E402

DEFAULT_CKPT = os.environ.get(
    "TRACKON_CKPT",
    "/leonardo_work/AIFAC_S07_110/bora/checkpoints/track_on/track_on_r.pt",
)


# --------------------------------------------------------------------------- #
#  Streaming tracker
# --------------------------------------------------------------------------- #
class CausalTracker:
    """Thin stateful wrapper around ``Predictor.forward_frame``.

    ``support_grid_size`` > 0 adds a hidden uniform grid of auxiliary points at
    the first frame (as the authors do for sparse user queries). They are
    tracked internally but never returned.
    """

    def __init__(
        self,
        ckpt: str = DEFAULT_CKPT,
        config: Optional[str] = None,
        device: Optional[str] = None,
        support_grid_size: int = 20,
        delta_v: Optional[float] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        model_args = load_args_from_yaml(config) if config else None
        self.predictor = Predictor(model_args, checkpoint_path=ckpt, support_grid_size=0)
        self.predictor.to(self.device).eval()
        if delta_v is not None:
            self.predictor.delta_v = delta_v
        self.support_grid_size = support_grid_size
        self.reset()

    # ---- state ----------------------------------------------------------- #
    def reset(self):
        self.predictor.reset()
        self.t = 0
        self.n_user = 0            # number of user queries added so far
        self._keep = []            # per active slot: True if user query, False if support
        self._support_added = False

    @property
    def num_points(self) -> int:
        return self.n_user

    # ---- helpers --------------------------------------------------------- #
    @staticmethod
    def _to_tensor_frame(frame, device) -> torch.Tensor:
        """(H, W, 3) uint8 RGB -> (1, 3, H, W) float32 in [0, 255]."""
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(np.ascontiguousarray(frame))
        if frame.ndim == 3 and frame.shape[-1] == 3:
            frame = frame.permute(2, 0, 1)
        if frame.ndim == 3:
            frame = frame.unsqueeze(0)
        return frame.to(device, non_blocking=True).float()

    # ---- main step ------------------------------------------------------- #
    @torch.no_grad()
    def step(self, frame, new_points: Optional[Sequence[Sequence[float]]] = None
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one frame.

        :param frame: (H, W, 3) uint8 RGB (numpy or torch) or (1, 3, H, W) float
        :param new_points: iterable of (x, y) pixel coords to start tracking now
        :return: (pts, vis) for *all user points added so far*, in insertion
                 order; pts (N, 2) float32 on CPU, vis (N,) bool on CPU
        """
        f = self._to_tensor_frame(frame, self.device)
        _, _, H, W = f.shape

        new_q = []
        if new_points is not None and len(new_points) > 0:
            new_q.append(torch.as_tensor(np.asarray(new_points, dtype=np.float32)).reshape(-1, 2))
            self._keep.extend([True] * new_q[-1].shape[0])
            self.n_user += new_q[-1].shape[0]

        # Support grid is injected together with the first batch of user queries
        if self.support_grid_size > 0 and not self._support_added and self.n_user > 0:
            grid = get_points_on_a_grid(self.support_grid_size, (H, W), self.device)[0]  # (S^2, 2)
            new_q.append(grid.cpu())
            self._keep.extend([False] * grid.shape[0])
            self._support_added = True

        new_q_t = torch.cat(new_q, 0).to(self.device) if new_q else None

        p, v = self.predictor.forward_frame(f, new_queries=new_q_t)
        self.t += 1

        if p.shape[0] == 0:
            return torch.empty(0, 2), torch.empty(0, dtype=torch.bool)

        keep = torch.tensor(self._keep, dtype=torch.bool, device=p.device)
        return p[keep].cpu(), v[keep].cpu()


# --------------------------------------------------------------------------- #
#  Video decoding (streaming)
# --------------------------------------------------------------------------- #
def iter_video_frames(path: str, max_frames: Optional[int] = None) -> Iterator[np.ndarray]:
    """Yield (H, W, 3) uint8 RGB frames one at a time using PyAV."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if max_frames is not None and i >= max_frames:
                break
            yield frame.to_ndarray(format="rgb24")


def video_fps(path: str, default: float = 10.0) -> float:
    import av

    with av.open(path) as container:
        r = container.streams.video[0].average_rate
        return float(r) if r else default


# --------------------------------------------------------------------------- #
#  Query parsing
# --------------------------------------------------------------------------- #
def parse_queries(args, H: int, W: int) -> np.ndarray:
    """Return (N, 3) float array of (t, x, y)."""
    if args.queries_file:
        if args.queries_file.endswith(".npy"):
            q = np.load(args.queries_file)
        else:
            with open(args.queries_file) as fh:
                q = np.asarray(json.load(fh), dtype=np.float32)
        q = np.asarray(q, dtype=np.float32).reshape(-1, 3)
        return q
    if args.points:
        rows = []
        for tok in args.points.replace(" ", "").split(";"):
            if not tok:
                continue
            vals = [float(v) for v in tok.split(",")]
            if len(vals) == 2:
                x, y = vals
                t = 0.0
            elif len(vals) == 3:
                x, y, t = vals
            else:
                raise ValueError(f"Bad point spec '{tok}', expected x,y or x,y,t")
            rows.append([t, x, y])
        return np.asarray(rows, dtype=np.float32)
    # default: grid at frame 0
    g = get_points_on_a_grid(args.grid, (H, W), torch.device("cpu"))[0].numpy()  # (S^2, 2)
    return np.concatenate([np.zeros((g.shape[0], 1), np.float32), g.astype(np.float32)], 1)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Causal point tracking with Track-On")
    p.add_argument("--video", required=True)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--config", default=None, help="model YAML (default: built-in Track-On2/R params)")
    p.add_argument("--output-dir", default="outputs/causal_track")
    q = p.add_mutually_exclusive_group()
    q.add_argument("--grid", type=int, default=10, help="SxS uniform grid of queries at frame 0 (default 10)")
    q.add_argument("--points", type=str, help='"x,y;x,y" or "x,y,t;..." pixel coords')
    q.add_argument("--queries-file", type=str, help=".json or .npy with rows [t, x, y]")
    p.add_argument("--support-grid", type=int, default=None,
                   help="hidden auxiliary grid size (default: 20 for --points/--queries-file, 0 for --grid)")
    p.add_argument("--delta-v", type=float, default=None, help="visibility threshold (default 0.8)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--point-size", type=int, default=60)
    p.add_argument("--fps", type=float, default=None, help="output video fps (default: input fps)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"video not found: {args.video}")
    if not os.path.isfile(args.ckpt):
        sys.exit(f"checkpoint not found: {args.ckpt}")
    os.makedirs(args.output_dir, exist_ok=True)

    sparse = bool(args.points or args.queries_file)
    support = args.support_grid if args.support_grid is not None else (20 if sparse else 0)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}  ckpt={args.ckpt}")
    t0 = time.time()
    tracker = CausalTracker(args.ckpt, args.config, device, support_grid_size=support, delta_v=args.delta_v)
    print(f"[info] model loaded in {time.time() - t0:.1f}s")

    frames_iter = iter_video_frames(args.video, args.max_frames)
    first = next(frames_iter)
    H, W = first.shape[:2]
    queries = parse_queries(args, H, W)                 # (N, 3) t,x,y
    N = queries.shape[0]
    order = np.argsort(queries[:, 0], kind="stable")     # insertion order == time order
    q_sorted = queries[order]
    inv = np.empty(N, dtype=np.int64)
    inv[order] = np.arange(N)                            # slot -> original index
    print(f"[info] {N} queries, first frame {H}x{W}, support grid {support}x{support}")

    tracks, vis, frames_kept = [], [], []
    n_added = 0
    t = 0
    tic = time.time()

    def _frames():
        yield first
        yield from frames_iter

    for frame in _frames():
        # queries whose start time is this frame
        new = []
        while n_added < N and int(q_sorted[n_added, 0]) <= t:
            new.append(q_sorted[n_added, 1:3])
            n_added += 1
        pts, v = tracker.step(frame, new_points=new if new else None)

        row = np.full((N, 2), np.nan, np.float32)
        vrow = np.zeros((N,), bool)
        if pts.shape[0] > 0:
            idx = inv[: pts.shape[0]]
            row[idx] = pts.numpy()
            vrow[idx] = v.numpy()
        tracks.append(row)
        vis.append(vrow)
        if not args.no_video:
            frames_kept.append(frame)
        t += 1
        if t % 50 == 0:
            print(f"[info] frame {t}  ({t / (time.time() - tic):.1f} fps)")

    tracks = np.stack(tracks)  # (T, N, 2)
    vis = np.stack(vis)        # (T, N)
    el = time.time() - tic
    print(f"[info] tracked {t} frames in {el:.1f}s  ({t / max(el, 1e-6):.1f} fps, causal)")

    out_npz = os.path.join(args.output_dir, "tracks.npz")
    np.savez(out_npz, tracks=tracks, visibility=vis, queries=queries, video=os.path.abspath(args.video))
    print(f"[done] saved {out_npz}  tracks {tracks.shape}  visibility {vis.shape}")

    if not args.no_video:
        from utils.vis_utils import plot_tracks_wo_tail, save_video

        rgb = np.stack(frames_kept)
        tr = tracks.transpose(1, 0, 2).copy()                      # (N, T, 2)
        # back-fill frames before a query's birth with its first position so the
        # colour map (based on frame-0 y) is sane; they stay hidden via occlusion
        for n in range(N):
            ok = ~np.isnan(tr[n, :, 0])
            if ok.any():
                tr[n, ~ok] = tr[n, ok.argmax()]
            else:
                tr[n] = 0.0
        occ = (~vis).T.astype(np.float32)                          # (N, T), 1 = occluded/not born
        rendered = plot_tracks_wo_tail(rgb, tr, occ, point_size=args.point_size)
        fps = args.fps or video_fps(args.video)
        out_mp4 = os.path.join(args.output_dir, "tracks.mp4")
        save_video(rendered.astype(np.uint8), out_mp4, fps=fps)


if __name__ == "__main__":
    main()
