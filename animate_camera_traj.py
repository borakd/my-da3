"""Animated GIF of a predicted camera trajectory progressively drawn against GT.

The ground-truth trajectory is shown in full (dashed black) from the first
frame; the aligned prediction is revealed pose-by-pose with a moving marker.
Loading, association, and alignment are shared with plot_camera_traj.py, so a
GIF made here corresponds exactly to the static plot made there.

Example:
    python animate_camera_traj.py \
        --pred "regular=outputs/cut3r_eval/demo_ray_smoketest/regular/camera" \
        --gt   <scene>/dense/cam --out traj_plots/regular.gif --align sim3
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evo.core import metrics, sync
from matplotlib.animation import FuncAnimation, PillowWriter

from plot_camera_traj import align_anchored, load_traj, parse_pred_arg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", required=True, help="'Label=path' (pose dir or TUM .txt)")
    ap.add_argument("--gt", required=True, help="ground-truth pose dir or TUM .txt")
    ap.add_argument("--out", required=True, help="output GIF path")
    ap.add_argument(
        "--align",
        choices=["sim3", "se3", "sim3_anchored", "se3_anchored", "first_pose", "none"],
        default="sim3",
    )
    ap.add_argument("--title", default=None)
    ap.add_argument("--color", default="tab:blue")
    ap.add_argument("--n_frames", type=int, default=120, help="animation frames (excl. hold)")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--hold", type=int, default=15, help="frames to hold the finished plot")
    args = ap.parse_args()

    label, path = parse_pred_arg(args.pred)
    gt = load_traj(args.gt)
    traj = load_traj(path)
    gt, traj = sync.associate_trajectories(gt, traj)
    if args.align == "first_pose":
        traj.align_origin(gt)
    elif args.align.endswith("_anchored"):
        align_anchored(traj, gt, with_scale=args.align.startswith("sim3"))
    elif args.align != "none":
        traj.align(gt, correct_scale=(args.align == "sim3"))
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt, traj))
    ate = ape.get_statistic(metrics.StatisticsType.rmse)
    print(f"{label}: ATE RMSE ({args.align}) = {ate:.4f} ({traj.num_poses} poses)")

    g, p = gt.positions_xyz, traj.positions_xyz

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)"), ax.set_zlabel("z (m)")

    # Fixed, equal-aspect bounds over both trajectories so the view never jumps.
    both = np.concatenate([g, p])
    center = (both.max(0) + both.min(0)) / 2
    half = (both.max(0) - both.min(0)).max() / 2 * 1.05
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass

    ax.plot(g[:, 0], g[:, 1], g[:, 2], "--", color="black", label="Ground Truth")
    ax.scatter(*g[0], color="black", marker="o", s=40, zorder=5)
    (line,) = ax.plot([], [], [], "-", color=args.color, label=f"{label} (ATE {ate:.3f})")
    (head,) = ax.plot([], [], [], "o", color=args.color, markersize=6, zorder=5)
    ax.legend(frameon=True)
    ax.set_title(args.title or label)

    counts = np.unique(np.linspace(1, len(p), min(args.n_frames, len(p))).astype(int))
    counts = np.concatenate([counts, np.full(args.hold, len(p))])

    def update(k):
        line.set_data(p[:k, 0], p[:k, 1])
        line.set_3d_properties(p[:k, 2])
        head.set_data(p[k - 1 : k, 0], p[k - 1 : k, 1])
        head.set_3d_properties(p[k - 1 : k, 2])
        return line, head

    anim = FuncAnimation(fig, update, frames=counts, blit=False)
    out = args.out if args.out.endswith(".gif") else args.out + ".gif"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=100)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
