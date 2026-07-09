"""Static 3D camera-trajectory comparison plots (evo-style) from per-frame pose files.

Plots one or more predicted trajectories against ground truth in a single 3D
figure. Poses are read either from a directory of per-frame ``.npz`` files
(``pose`` key — the format written by ``save_geometry_outputs_cuteanything`` in
``with_cut3r_v1.py``, the eval_pipeline workers, and the DL3DV_Multi GT
``dense/cam`` dirs) or from a TUM-format ``.txt`` file.

Each prediction is associated to the GT frame-by-frame (via the frame index in
the filename), optionally Umeyama-aligned (Sim(3) or SE(3)) independently, and
its post-alignment ATE RMSE is reported in the legend and on stdout.

Examples (run in the `cuteanything` env; no GPU needed):
    python plot_camera_traj.py \
        --pred "Captain Ray=outputs/cut3r_eval/overfit_test_scene/captain_ray/camera" \
               "CUT3R zero-shot=outputs/cut3r_eval/overfit_test_scene/regular/camera" \
        --gt   <scene>/dense/cam \
        --out  traj_plots/overfit_test_scene.png --align sim3

    python plot_camera_traj.py --pred outputs/.../camera --out traj.png  # single, unlabeled

Related: src/CUT3R/eval/relpose/evo_utils.py has the upstream single-trajectory
``plot_trajectory`` this is modeled on.
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface, plot
from scipy.spatial.transform import Rotation


def load_traj(path):
    """Load a trajectory as an evo PoseTrajectory3D.

    path: directory of per-frame .npz/.npy 4x4 c2w poses (frame id = filename
    stem, used as the timestamp), or a TUM-format .txt file.
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.npz"))) or sorted(
            glob.glob(os.path.join(path, "*.npy"))
        )
        if not files:
            raise FileNotFoundError(f"no .npz/.npy pose files in {path}")
        poses, stamps = [], []
        for f in files:
            if f.endswith(".npz"):
                with np.load(f) as d:
                    key = "pose" if "pose" in d else list(d.keys())[0]
                    poses.append(d[key].reshape(4, 4))
            else:
                poses.append(np.load(f).reshape(4, 4))
            stem = os.path.splitext(os.path.basename(f))[0]
            stamps.append(float(int(stem)) if stem.isdigit() else float(len(stamps)))
        c2w = np.stack(poses)
        quat_xyzw = Rotation.from_matrix(c2w[:, :3, :3]).as_quat()
        return PoseTrajectory3D(
            positions_xyz=c2w[:, :3, 3],
            orientations_quat_wxyz=quat_xyzw[:, [3, 0, 1, 2]],
            timestamps=np.asarray(stamps),
        )
    return file_interface.read_tum_trajectory_file(path)


def align_anchored(traj, ref, with_scale):
    """Umeyama fit constrained to pin traj's first position onto ref's first position.

    Minimizes sum ||s*R*(p_i - p_0) - (g_i - g_0)||^2 over R (and s if with_scale),
    i.e. standard Procrustes centered on the first points instead of the centroids,
    then t = g_0 - s*R*p_0. Uses only frame 0's *position* as the anchor — rotation
    and scale come from the whole trajectory (unlike align_origin, which trusts
    frame 0's predicted rotation).
    """
    from evo.core import lie_algebra as lie

    p = traj.positions_xyz - traj.positions_xyz[0]
    g = ref.positions_xyz - ref.positions_xyz[0]
    cov = (g.T @ p) / len(p)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    s = 1.0
    if with_scale:
        var = float(np.mean((p * p).sum(axis=1)))
        if var > 1e-15:
            s = float(np.trace(np.diag(S) @ D) / var)
    t = ref.positions_xyz[0] - s * (R @ traj.positions_xyz[0])
    traj.scale(s)
    traj.transform(lie.se3(R, t))


def parse_pred_arg(arg):
    """'Label=path' -> (label, path); bare 'path' -> ('Predicted', path)."""
    if "=" in arg and not os.path.exists(arg):
        label, path = arg.split("=", 1)
        return label, path
    return "Predicted", arg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--pred",
        required=True,
        nargs="+",
        help="one or more 'Label=path' (pose dir of per-frame .npz, or TUM .txt)",
    )
    ap.add_argument("--gt", default=None, help="ground-truth pose dir or TUM .txt")
    ap.add_argument("--out", required=True, help="output PNG path (or prefix without .png)")
    ap.add_argument(
        "--align",
        choices=["sim3", "se3", "sim3_anchored", "se3_anchored", "first_pose", "none"],
        default="sim3",
        help="alignment of each pred onto GT: Umeyama over all poses (sim3 = with "
        "scale, se3 = rigid); *_anchored = same fit constrained to pin the start "
        "position onto GT's start; first_pose = rigid gt_0 @ inv(pred_0), no fit",
    )
    ap.add_argument("--stride", type=int, default=1, help="subsample poses")
    ap.add_argument("--title", default=None)
    ap.add_argument(
        "--colors",
        nargs="+",
        default=None,
        help="matplotlib color per --pred entry (default: tab10 cycle)",
    )
    args = ap.parse_args()

    gt = load_traj(args.gt) if args.gt else None
    if gt is not None and args.stride > 1:
        gt.reduce_to_ids(np.arange(0, gt.num_poses, args.stride))

    preds = []
    for arg in args.pred:
        label, path = parse_pred_arg(arg)
        traj = load_traj(path)
        if args.stride > 1:
            traj.reduce_to_ids(np.arange(0, traj.num_poses, args.stride))
        ate = None
        if gt is not None:
            gt_sync, traj = sync.associate_trajectories(gt, traj)
            if args.align == "first_pose":
                traj.align_origin(gt_sync)
            elif args.align.endswith("_anchored"):
                align_anchored(traj, gt_sync, with_scale=args.align.startswith("sim3"))
            elif args.align != "none":
                traj.align(gt_sync, correct_scale=(args.align == "sim3"))
            ape = metrics.APE(metrics.PoseRelation.translation_part)
            ape.process_data((gt_sync, traj))
            ate = ape.get_statistic(metrics.StatisticsType.rmse)
            print(f"{label}: ATE RMSE ({args.align}) = {ate:.4f} ({traj.num_poses} poses)")
        preds.append((label, traj, ate))

    fig = plt.figure(figsize=(10, 10))
    plot_mode = plot.PlotMode.xyz
    ax = plot.prepare_axis(fig, plot_mode)
    if gt is not None:
        plot.traj(ax, plot_mode, gt, "--", "black", "Ground Truth")
        ax.scatter(*gt.positions_xyz[0], color="black", marker="o", s=40, zorder=5)
    cycle = plt.get_cmap("tab10").colors
    for i, (label, traj, ate) in enumerate(preds):
        color = args.colors[i % len(args.colors)] if args.colors else cycle[i % len(cycle)]
        legend = label if ate is None else f"{label} (ATE {ate:.3f})"
        plot.traj(ax, plot_mode, traj, "-", color, legend)
        ax.scatter(*traj.positions_xyz[0], color=color, marker="o", s=40, zorder=5)
    try:
        ax.set_aspect("equal", adjustable="box")
    except NotImplementedError:
        pass
    ax.legend(frameon=True)
    if args.title:
        ax.set_title(args.title)

    out = args.out if args.out.endswith(".png") else args.out + ".png"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
