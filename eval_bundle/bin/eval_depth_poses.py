#!/usr/bin/env python3
"""
Evaluate depth and pose metrics from two sequence roots.

Expected layout:
  pred_root/depth/000000.npy
  pred_root/camera/000000.npz
  gt_root/depth/000000.npy
  gt_root/camera/000000.npz or gt_root/cam/000000.npz

The evaluator supports interleaved multi-camera streams and splits files by:
  global_idx % num_cameras == camera_id

Depth metrics (per local timestep):
  - absrel
  - a1
  - optional depth scale alignment (default: scale)

Pose metrics (per local timestep):
  - ate (after per-camera trajectory alignment)
  - rpe_trans (for local timestep > 0)
  - rpe_rot (degrees, for local timestep > 0)

Outputs:
  - eval_depth_pose_metrics.csv
  - eval_metrics_readable.txt
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate depth/pose metrics from prediction and GT sequence folders."
    )
    parser.add_argument("--pred_root", type=Path, required=True, help="Prediction root.")
    parser.add_argument(
        "--gt_root",
        type=Path,
        default="/frozen/avg/bora_data/droid_datasets/training_data/pointworld_droid_wrist_test/dl3dv_multi/RAIL+eh61f232+2023-10-26-17h-33m-59s/13062452+wrist/dense",
        help="Ground-truth root."
    )
    parser.add_argument(
        "--pred_pose_type",
        type=str,
        choices=("c2w", "w2c"),
        default="c2w",
        help="Pose convention in prediction pose files.",
    )
    parser.add_argument(
        "--gt_pose_type",
        type=str,
        choices=("c2w", "w2c"),
        default="c2w",
        help="Pose convention in GT pose files.",
    )
    parser.add_argument(
        "--align",
        type=str,
        choices=("none", "se3", "sim3"),
        default="sim3",
        help="Global trajectory alignment per camera stream for ATE/RPE.",
    )
    parser.add_argument(
        "--num_cameras",
        type=int,
        default=1,
        help=(
            "Number of interleaved cameras in the stream (used with modulo split). "
            "Default 1 = monocular/single-stream evaluation (all frames form one "
            "sequence in index order). Set >1 only for interleaved multi-camera data."
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <pred_root>/eval_depth_pose_metrics.csv",
    )
    parser.add_argument(
        "--depth_eps",
        type=float,
        default=1e-9,
        help="Minimum positive depth threshold for valid pixels.",
    )
    parser.add_argument(
        "--depth_scale_align",
        type=str,
        choices=("none", "median", "scale"),
        default="median",
        help="Depth scale alignment mode before metric computation.",
    )
    parser.add_argument(
        "--pose_reduce",
        type=str,
        choices=("rmse", "mean"),
        default="rmse",
        help="How to summarize pose metrics in MEAN rows.",
    )
    parser.add_argument(
        "--eval_like_cut3r",
        action="store_true",
        default=False,
        help=(
            "Compute the metrics (ate, rpe_trans, rpe_rot, absrel, a1) with the exact "
            "math used by CUT3R's evaluation code. The input/output pipeline (file "
            "discovery, multi-camera split, CSV/readable outputs) is unchanged; only "
            "the metric computation differs. When set, --align, --depth_scale_align "
            "and --pose_reduce are ignored in favor of CUT3R's conventions: pose uses "
            "evo APE/RPE with Sim3 alignment (align + correct_scale) reported as RMSE; "
            "depth pools all valid pixels per camera stream with a single joint "
            "alignment. Requires the 'evo', 'torch', 'cv2' and 'scipy' packages."
        ),
    )
    parser.add_argument(
        "--cut3r_depth_align",
        type=str,
        choices=("scale", "scale_shift", "median", "metric"),
        default="scale",
        help=(
            "Depth alignment used only with --eval_like_cut3r. 'scale' = Weiszfeld IRLS "
            "scale-only (CUT3R video_depth run.sh default); 'scale_shift' = LAD affine "
            "(align_with_lad2); 'median' = median(gt)/median(pred); 'metric' = none."
        ),
    )
    parser.add_argument(
        "--cut3r_max_depth",
        type=float,
        default=70.0,
        help=(
            "Max GT depth for the valid mask (gt>0 & gt<max) with --eval_like_cut3r. "
            "CUT3R uses 70 for sintel/bonn. Pass a value <= 0 to disable the cap "
            "(gt>0 only, as CUT3R does for KITTI)."
        ),
    )
    parser.add_argument(
        "--cut3r_post_clip_max",
        type=float,
        default=None,
        help=(
            "Optional clamp on aligned predicted depth (post_clip_max) with "
            "--eval_like_cut3r. CUT3R's run.sh 'scale' path leaves this unset."
        ),
    )
    parser.add_argument(
        "--cut3r_use_gpu",
        action="store_true",
        default=False,
        help="Run CUT3R depth alignment/metrics on GPU (matches use_gpu=True).",
    )
    return parser.parse_args()


def _list_indexed_files(folder: Path, suffix: str) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    if not folder.is_dir():
        return out
    pat = re.compile(r"^\d+$")
    for p in sorted(folder.glob(f"*{suffix}")):
        if pat.match(p.stem):
            out[int(p.stem)] = p
    return out


def _resolve_cam_folder(root: Path) -> Path:
    camera = root / "camera"
    cam = root / "cam"
    if camera.is_dir():
        return camera
    if cam.is_dir():
        return cam
    return camera


def _split_stream_by_camera(
    indexed_files: Dict[int, Path], num_cameras: int
) -> Dict[int, Dict[int, Tuple[int, Path]]]:
    out: Dict[int, Dict[int, Tuple[int, Path]]] = {cid: {} for cid in range(num_cameras)}
    local_counts = [0 for _ in range(num_cameras)]
    for global_idx in sorted(indexed_files.keys()):
        cid = global_idx % num_cameras
        local_idx = local_counts[cid]
        out[cid][local_idx] = (global_idx, indexed_files[global_idx])
        local_counts[cid] += 1
    return out


def _to_4x4(mat: np.ndarray) -> np.ndarray:
    if mat.shape == (4, 4):
        return mat.astype(np.float64, copy=False)
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = mat
        return out
    raise ValueError(f"Unsupported pose shape {mat.shape}, expected (4,4) or (3,4).")


def _load_pose_as_c2w(path: Path, pose_type: str) -> np.ndarray:
    data = np.load(path)
    if "pose" not in data:
        raise KeyError(f"'pose' key missing in {path}")
    pose = _to_4x4(np.asarray(data["pose"]))
    if pose_type == "w2c":
        pose = np.linalg.inv(pose)
    return pose


def _rotation_angle_deg(R: np.ndarray) -> float:
    tr = float(np.trace(R))
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _robust_scalar_align(pred_vals: np.ndarray, gt_vals: np.ndarray, eps: float) -> float:
    s = 1.0
    for _ in range(25):
        r = s * pred_vals - gt_vals
        w = 1.0 / np.maximum(np.abs(r), eps)
        num = np.sum(w * pred_vals * gt_vals)
        den = np.sum(w * pred_vals * pred_vals)
        if den <= eps:
            break
        s_new = float(num / den)
        if abs(s_new - s) <= 1e-9 * max(1.0, abs(s)):
            s = s_new
            break
        s = s_new
    return s


def _resize_depth_bilinear(depth: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    src_h, src_w = depth.shape
    tgt_h, tgt_w = target_shape
    if (src_h, src_w) == (tgt_h, tgt_w):
        return depth

    y = np.linspace(0.0, src_h - 1.0, tgt_h)
    x = np.linspace(0.0, src_w - 1.0, tgt_w)

    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)

    wy = y - y0
    wx = x - x0

    Ia = depth[y0[:, None], x0[None, :]]
    Ib = depth[y1[:, None], x0[None, :]]
    Ic = depth[y0[:, None], x1[None, :]]
    Id = depth[y1[:, None], x1[None, :]]

    wa = (1.0 - wy)[:, None] * (1.0 - wx)[None, :]
    wb = wy[:, None] * (1.0 - wx)[None, :]
    wc = (1.0 - wy)[:, None] * wx[None, :]
    wd = wy[:, None] * wx[None, :]

    wa = wa * np.isfinite(Ia)
    wb = wb * np.isfinite(Ib)
    wc = wc * np.isfinite(Ic)
    wd = wd * np.isfinite(Id)

    numer = (
        wa * np.nan_to_num(Ia, nan=0.0)
        + wb * np.nan_to_num(Ib, nan=0.0)
        + wc * np.nan_to_num(Ic, nan=0.0)
        + wd * np.nan_to_num(Id, nan=0.0)
    )
    denom = wa + wb + wc + wd

    out = np.full((tgt_h, tgt_w), np.nan, dtype=np.float64)
    valid = denom > 0.0
    out[valid] = numer[valid] / denom[valid]
    return out


def _compute_absrel_a1(
    pred_depth: np.ndarray, gt_depth: np.ndarray, eps: float, scale_align: str
) -> Tuple[float, float]:
    pred = np.squeeze(np.asarray(pred_depth, dtype=np.float64))
    gt = np.squeeze(np.asarray(gt_depth, dtype=np.float64))

    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError(f"Depth must be 2D after squeeze, got pred={pred.shape}, gt={gt.shape}")
    if pred.shape != gt.shape:
        pred = _resize_depth_bilinear(pred, gt.shape)

    mask = np.isfinite(pred) & np.isfinite(gt) & (pred > eps) & (gt > eps)
    if not np.any(mask):
        return np.nan, np.nan

    p = pred[mask]
    g = gt[mask]

    if scale_align == "median":
        pred_med = float(np.median(p))
        gt_med = float(np.median(g))
        if not np.isfinite(pred_med) or pred_med <= eps or not np.isfinite(gt_med):
            return np.nan, np.nan
        p = p * (gt_med / pred_med)
    elif scale_align == "scale":
        s = _robust_scalar_align(p, g, eps=eps)
        if not np.isfinite(s):
            return np.nan, np.nan
        p = p * s

    absrel = np.mean(np.abs(p - g) / g)
    ratio = np.maximum(p / g, g / p)
    a1 = np.mean(ratio < 1.25)
    return float(absrel), float(a1)


def _estimate_umeyama(
    src_xyz: np.ndarray, dst_xyz: np.ndarray, with_scale: bool
) -> Tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src_xyz, dtype=np.float64)
    dst = np.asarray(dst_xyz, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Invalid center shapes: src={src.shape}, dst={dst.shape}")
    n = src.shape[0]
    if n < 1:
        raise ValueError("Need at least 1 point for alignment")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / max(n, 1)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt

    if with_scale:
        var_src = np.mean(np.sum(src_c * src_c, axis=1))
        if var_src <= 1e-15:
            scale = 1.0
        else:
            scale = float(np.trace(np.diag(S) @ D) / var_src)
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def _align_pred_poses_to_gt(
    pred_c2w: Dict[int, np.ndarray], gt_c2w: Dict[int, np.ndarray], mode: str
) -> Dict[int, np.ndarray]:
    if mode == "none":
        return dict(pred_c2w)

    common_ts = sorted(set(pred_c2w.keys()) & set(gt_c2w.keys()))
    if not common_ts:
        return dict(pred_c2w)

    pred_centers = np.stack([pred_c2w[t][:3, 3] for t in common_ts], axis=0)
    gt_centers = np.stack([gt_c2w[t][:3, 3] for t in common_ts], axis=0)

    with_scale = mode == "sim3"
    s, R, t = _estimate_umeyama(pred_centers, gt_centers, with_scale=with_scale)

    aligned: Dict[int, np.ndarray] = {}
    for ts, T in pred_c2w.items():
        Tout = np.eye(4, dtype=np.float64)
        Tout[:3, :3] = R @ T[:3, :3]
        Tout[:3, 3] = s * (R @ T[:3, 3]) + t
        aligned[ts] = Tout
    return aligned


def _relative_pose(Ta: np.ndarray, Tb: np.ndarray) -> np.ndarray:
    return np.linalg.inv(Ta) @ Tb


def _nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def _nanrmse(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(arr * arr)))


def _reduce_pose(values: Iterable[float], mode: str) -> float:
    if mode == "rmse":
        return _nanrmse(values)
    return _nanmean(values)


def _fmt3(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    return f"{v:.3f}"


# ---------------------------------------------------------------------------
# CUT3R-identical metric computation (only used with --eval_like_cut3r).
#
# These reproduce, verbatim, the math in CUT3R's evaluation code:
#   - depth: eval/video_depth/tools.py :: depth_evaluation
#   - pose:  eval/relpose/{utils.py,evo_utils.py} :: get_tum_poses / eval_metrics
# Heavy deps (torch / cv2 / evo / scipy) are imported lazily so the default
# (non-CUT3R) path keeps working without them.
# ---------------------------------------------------------------------------


def _cut3r_align_depth(pred_vals, gt_vals, align: str):
    """Return (s, t) such that aligned = s * pred + t, matching CUT3R.

    pred_vals / gt_vals are 1-D torch tensors of pooled valid pixels.
    """
    import torch

    if align == "metric":
        return 1.0, 0.0

    if align == "scale_shift":
        # CUT3R align_with_lad2: LAD affine fit via Adam, init s=median(gt)/median(pred).
        s_init = (torch.median(gt_vals) / torch.median(pred_vals)).item()
        s = torch.tensor(
            [s_init], requires_grad=True, device=pred_vals.device, dtype=pred_vals.dtype
        )
        t = torch.tensor(
            [0.0], requires_grad=True, device=pred_vals.device, dtype=pred_vals.dtype
        )
        optimizer = torch.optim.Adam([s, t], lr=1e-4)
        prev_loss = None
        for _ in range(1000):
            optimizer.zero_grad()
            loss = torch.sum(torch.abs(s * pred_vals + t - gt_vals))
            loss.backward()
            optimizer.step()
            if prev_loss is not None and torch.abs(prev_loss - loss) < 1e-6:
                break
            prev_loss = loss.item()
        return float(s.detach().item()), float(t.detach().item())

    if align == "scale":
        # CUT3R align_with_scale: Weiszfeld IRLS scale-only, init s=mean(gt)/mean(pred).
        s = torch.nanmean(gt_vals) / torch.nanmean(pred_vals)
        for _ in range(10):
            residuals = s * pred_vals - gt_vals
            abs_residuals = residuals.abs() + 1e-8
            weights = 1.0 / abs_residuals
            s = torch.sum(weights * pred_vals * gt_vals) / torch.sum(
                weights * pred_vals**2
            )
        s = s.clamp(min=1e-3).detach()
        return float(s.item()), 0.0

    # CUT3R default branch: median scaling.
    s = torch.median(gt_vals) / torch.median(pred_vals)
    return float(s.item()), 0.0


def _cut3r_depth_metrics(pred_aligned, gt_vals):
    """absrel and a1 (delta < 1.25) over a set of pixels, exactly as CUT3R does."""
    import torch

    absrel = torch.mean(torch.abs(pred_aligned - gt_vals) / gt_vals).item()
    # CUT3R clamps the aligned prediction to min=1e-5 before the delta ratios
    # (abs_rel above is computed on the unclamped values).
    pred_clamped = torch.clamp(pred_aligned, min=1e-5)
    max_ratio = torch.maximum(pred_clamped / gt_vals, gt_vals / pred_clamped)
    a1 = torch.mean((max_ratio < 1.25).float()).item()
    return float(absrel), float(a1)


def _cut3r_eval_camera_depth(
    pred_map: Dict[int, Tuple[int, Path]],
    gt_map: Dict[int, Tuple[int, Path]],
    common_local: List[int],
    max_depth,
    align: str,
    use_gpu: bool,
    post_clip_max,
) -> Tuple[Dict[int, Tuple[float, float]], Tuple[float, float, int]]:
    """Per-camera depth metrics matching CUT3R.

    Returns (per_frame_metrics, (absrel_pooled, a1_pooled, valid_pixels)).

    Pixels are pooled across the whole camera stream and aligned jointly (one
    s,t for the sequence) exactly as CUT3R stacks (N,H,W) -> flattens -> masks.
    The per-frame rows reuse the same global (s,t) for diagnostics; the MEAN row
    uses the pooled values (so it is identical to CUT3R's reported number).
    """
    import cv2
    import torch

    per_frame_pred: Dict[int, np.ndarray] = {}
    per_frame_gt: Dict[int, np.ndarray] = {}
    pool_pred: List[np.ndarray] = []
    pool_gt: List[np.ndarray] = []

    for local_ts in common_local:
        pred_d = np.squeeze(np.load(pred_map[local_ts][1]))
        gt_d = np.squeeze(np.load(gt_map[local_ts][1]))
        if pred_d.ndim != 2 or gt_d.ndim != 2:
            raise ValueError(
                f"Depth must be 2D after squeeze, got pred={pred_d.shape}, gt={gt_d.shape}"
            )
        if pred_d.shape != gt_d.shape:
            # CUT3R resizes the prediction to the GT size with bicubic interpolation.
            resize_src = pred_d
            if resize_src.dtype not in (np.float32, np.float64):
                resize_src = resize_src.astype(np.float32)
            pred_d = cv2.resize(
                resize_src,
                (gt_d.shape[1], gt_d.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        if max_depth is not None:
            mask = (gt_d > 0) & (gt_d < max_depth)
        else:
            mask = gt_d > 0

        p = np.ascontiguousarray(pred_d[mask].reshape(-1))
        g = np.ascontiguousarray(gt_d[mask].reshape(-1))
        per_frame_pred[local_ts] = p
        per_frame_gt[local_ts] = g
        pool_pred.append(p)
        pool_gt.append(g)

    if not pool_pred or sum(p.size for p in pool_pred) == 0:
        per_frame = {lt: (np.nan, np.nan) for lt in common_local}
        return per_frame, (np.nan, np.nan, 0)

    P = torch.from_numpy(np.concatenate(pool_pred))
    G = torch.from_numpy(np.concatenate(pool_gt))
    if use_gpu:
        P = P.cuda()
        G = G.cuda()

    s, t = _cut3r_align_depth(P, G, align)
    P_aligned = s * P + t
    if post_clip_max is not None:
        P_aligned = torch.clamp(P_aligned, max=post_clip_max)
    absrel_pool, a1_pool = _cut3r_depth_metrics(P_aligned, G)
    valid_pixels = int(G.numel())

    per_frame: Dict[int, Tuple[float, float]] = {}
    for local_ts in common_local:
        g_np = per_frame_gt[local_ts]
        if g_np.size == 0:
            per_frame[local_ts] = (np.nan, np.nan)
            continue
        p_t = torch.from_numpy(per_frame_pred[local_ts])
        g_t = torch.from_numpy(g_np)
        if use_gpu:
            p_t = p_t.cuda()
            g_t = g_t.cuda()
        p_aligned = s * p_t + t
        if post_clip_max is not None:
            p_aligned = torch.clamp(p_aligned, max=post_clip_max)
        per_frame[local_ts] = _cut3r_depth_metrics(p_aligned, g_t)

    return per_frame, (absrel_pool, a1_pool, valid_pixels)


def _c2w_to_tum_wxyz(c2w: np.ndarray) -> np.ndarray:
    """c2w 4x4 -> [x, y, z, qw, qx, qy, qz] (CUT3R's c2w_to_tumpose)."""
    from scipy.spatial.transform import Rotation

    xyz = c2w[:3, -1]
    qx, qy, qz, qw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
    return np.concatenate([xyz, [qw, qx, qy, qz]])


def _cut3r_eval_camera_pose(
    pred_c2w: Dict[int, np.ndarray],
    gt_c2w: Dict[int, np.ndarray],
    common_local: List[int],
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Tuple[float, float, float]]:
    """Per-camera ATE / RPE matching CUT3R's evo-based eval_metrics.

    ATE  = APE(translation_part, align + correct_scale) RMSE
    RPE  = RPE(delta=1 frame, all_pairs, align + correct_scale) RMSE
           (translation_part for rpe_trans, rotation_angle_deg for rpe_rot)

    Returns (ate_by_local, rpe_trans_by_local, rpe_rot_by_local,
             (ate_rmse, rpe_trans_rmse, rpe_rot_rmse)).
    """
    import evo.main_ape as main_ape
    import evo.main_rpe as main_rpe
    from evo.core import sync
    from evo.core.metrics import PoseRelation, Unit
    from evo.core.trajectory import PoseTrajectory3D

    ts = sorted(common_local)
    ate_by: Dict[int, float] = {}
    rpe_trans_by: Dict[int, float] = {}
    rpe_rot_by: Dict[int, float] = {}
    if len(ts) == 0:
        return ate_by, rpe_trans_by, rpe_rot_by, (np.nan, np.nan, np.nan)

    pred_tum = np.stack([_c2w_to_tum_wxyz(pred_c2w[t]) for t in ts], axis=0)
    gt_tum = np.stack([_c2w_to_tum_wxyz(gt_c2w[t]) for t in ts], axis=0)
    stamps = np.arange(len(ts)).astype(float)

    pred_traj = PoseTrajectory3D(
        positions_xyz=pred_tum[:, :3],
        orientations_quat_wxyz=pred_tum[:, 3:],
        timestamps=stamps,
    )
    gt_traj = PoseTrajectory3D(
        positions_xyz=gt_tum[:, :3],
        orientations_quat_wxyz=gt_tum[:, 3:],
        timestamps=stamps,
    )
    # Identical (integer) timestamps -> association is identity, matching CUT3R.
    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    ate = rpe_trans = rpe_rot = np.nan

    try:
        ate_res = main_ape.ape(
            gt_traj,
            pred_traj,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True,
        )
        ate = float(ate_res.stats["rmse"])
        arr = ate_res.np_arrays["error_array"]
        for i in range(min(len(arr), len(ts))):
            ate_by[ts[i]] = float(arr[i])
    except Exception as exc:  # pragma: no cover - evo edge cases
        print(f"[eval_like_cut3r] ATE failed: {exc}")

    if len(ts) >= 2:
        try:
            rot_res = main_rpe.rpe(
                gt_traj,
                pred_traj,
                est_name="traj",
                pose_relation=PoseRelation.rotation_angle_deg,
                align=True,
                correct_scale=True,
                delta=1,
                delta_unit=Unit.frames,
                rel_delta_tol=0.01,
                all_pairs=True,
            )
            rpe_rot = float(rot_res.stats["rmse"])
            arr = rot_res.np_arrays["error_array"]
            # delta=1 RPE errors correspond to consecutive pairs (ts[i], ts[i+1]);
            # attribute each to the second timestep, matching the existing CSV layout.
            for j in range(min(len(arr), len(ts) - 1)):
                rpe_rot_by[ts[j + 1]] = float(arr[j])
        except Exception as exc:  # pragma: no cover
            print(f"[eval_like_cut3r] RPE-rot failed: {exc}")

        try:
            trans_res = main_rpe.rpe(
                gt_traj,
                pred_traj,
                est_name="traj",
                pose_relation=PoseRelation.translation_part,
                align=True,
                correct_scale=True,
                delta=1,
                delta_unit=Unit.frames,
                rel_delta_tol=0.01,
                all_pairs=True,
            )
            rpe_trans = float(trans_res.stats["rmse"])
            arr = trans_res.np_arrays["error_array"]
            for j in range(min(len(arr), len(ts) - 1)):
                rpe_trans_by[ts[j + 1]] = float(arr[j])
        except Exception as exc:  # pragma: no cover
            print(f"[eval_like_cut3r] RPE-trans failed: {exc}")

    return ate_by, rpe_trans_by, rpe_rot_by, (ate, rpe_trans, rpe_rot)


def _weighted_nanmean(values: Iterable[float], weights: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=np.float64)
    wts = np.asarray(list(weights), dtype=np.float64)
    keep = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    if not np.any(keep):
        return np.nan
    return float(np.sum(vals[keep] * wts[keep]) / np.sum(wts[keep]))


def main() -> None:
    args = parse_args()
    if args.num_cameras <= 0:
        raise ValueError("--num_cameras must be >= 1")

    pred_depth_files = _list_indexed_files(args.pred_root / "depth", ".npy")
    gt_depth_files = _list_indexed_files(args.gt_root / "depth", ".npy")
    pred_cam_files = _list_indexed_files(args.pred_root / "camera", ".npz")
    gt_cam_files = _list_indexed_files(_resolve_cam_folder(args.gt_root), ".npz")

    if len(pred_depth_files) == 0 and len(pred_cam_files) == 0:
        raise FileNotFoundError(
            f"No prediction depth/camera files found under {args.pred_root}"
        )
    if len(gt_depth_files) == 0 and len(gt_cam_files) == 0:
        raise FileNotFoundError(
            f"No ground-truth depth/camera files found under {args.gt_root}"
        )

    pred_depth_by_cam = _split_stream_by_camera(pred_depth_files, args.num_cameras)
    gt_depth_by_cam = _split_stream_by_camera(gt_depth_files, args.num_cameras)
    pred_pose_by_cam = _split_stream_by_camera(pred_cam_files, args.num_cameras)
    gt_pose_by_cam = _split_stream_by_camera(gt_cam_files, args.num_cameras)

    cut3r = args.eval_like_cut3r
    cut3r_max_depth = args.cut3r_max_depth if args.cut3r_max_depth > 0 else None

    rows: List[dict] = []
    per_frame_rows: List[dict] = []
    camera_summaries: List[dict] = []
    # CUT3R-style per-camera summaries (depth pooled metrics + valid pixels; pose RMSE).
    cam_depth_summaries: List[Tuple[float, float, int]] = []
    cam_pose_summaries: List[Tuple[float, float, float]] = []

    for cam_id in range(args.num_cameras):
        cam_rows: List[dict] = []

        depth_common_local = sorted(
            set(pred_depth_by_cam[cam_id].keys()) & set(gt_depth_by_cam[cam_id].keys())
        )
        depth_metrics: Dict[int, Tuple[float, float]] = {}
        depth_summary: Tuple[float, float, int] = (np.nan, np.nan, 0)
        if cut3r:
            depth_metrics, depth_summary = _cut3r_eval_camera_depth(
                pred_depth_by_cam[cam_id],
                gt_depth_by_cam[cam_id],
                depth_common_local,
                max_depth=cut3r_max_depth,
                align=args.cut3r_depth_align,
                use_gpu=args.cut3r_use_gpu,
                post_clip_max=args.cut3r_post_clip_max,
            )
        else:
            for local_ts in depth_common_local:
                _, pred_path = pred_depth_by_cam[cam_id][local_ts]
                _, gt_path = gt_depth_by_cam[cam_id][local_ts]
                pred_d = np.load(pred_path)
                gt_d = np.load(gt_path)
                depth_metrics[local_ts] = _compute_absrel_a1(
                    pred_d, gt_d, eps=args.depth_eps, scale_align=args.depth_scale_align
                )
        cam_depth_summaries.append(depth_summary)

        pose_common_local = sorted(
            set(pred_pose_by_cam[cam_id].keys()) & set(gt_pose_by_cam[cam_id].keys())
        )
        pred_poses_c2w = {
            local_ts: _load_pose_as_c2w(
                pred_pose_by_cam[cam_id][local_ts][1], args.pred_pose_type
            )
            for local_ts in pose_common_local
        }
        gt_poses_c2w = {
            local_ts: _load_pose_as_c2w(
                gt_pose_by_cam[cam_id][local_ts][1], args.gt_pose_type
            )
            for local_ts in pose_common_local
        }

        ate_by_local: Dict[int, float] = {}
        rpe_trans_by_local: Dict[int, float] = {}
        rpe_rot_by_local: Dict[int, float] = {}
        pose_summary: Tuple[float, float, float] = (np.nan, np.nan, np.nan)
        if cut3r:
            (
                ate_by_local,
                rpe_trans_by_local,
                rpe_rot_by_local,
                pose_summary,
            ) = _cut3r_eval_camera_pose(
                pred_poses_c2w, gt_poses_c2w, pose_common_local
            )
        else:
            aligned_pred_poses = _align_pred_poses_to_gt(
                pred_poses_c2w, gt_poses_c2w, args.align
            )
            for local_ts in pose_common_local:
                p = aligned_pred_poses[local_ts][:3, 3]
                g = gt_poses_c2w[local_ts][:3, 3]
                ate_by_local[local_ts] = float(np.linalg.norm(p - g))
            for i in range(len(pose_common_local) - 1):
                t0 = pose_common_local[i]
                t1 = pose_common_local[i + 1]
                d_gt = _relative_pose(gt_poses_c2w[t0], gt_poses_c2w[t1])
                d_pr = _relative_pose(aligned_pred_poses[t0], aligned_pred_poses[t1])
                E = np.linalg.inv(d_gt) @ d_pr
                rpe_trans_by_local[t1] = float(np.linalg.norm(E[:3, 3]))
                rpe_rot_by_local[t1] = _rotation_angle_deg(E[:3, :3])
        cam_pose_summaries.append(pose_summary)

        all_local_ts = sorted(set(depth_common_local) | set(pose_common_local))
        for local_ts in all_local_ts:
            absrel, a1 = depth_metrics.get(local_ts, (np.nan, np.nan))
            row = {
                "camera_id": cam_id,
                "local_timestep": local_ts,
                "absrel": absrel,
                "a1": a1,
                "ate": ate_by_local.get(local_ts, np.nan),
                "rpe_trans": rpe_trans_by_local.get(local_ts, np.nan),
                "rpe_rot": rpe_rot_by_local.get(local_ts, np.nan),
            }
            rows.append(row)
            cam_rows.append(row)
            per_frame_rows.append(row)

        if cut3r:
            # MEAN row uses CUT3R's pooled depth metrics and evo RMSE pose metrics,
            # NOT a reduction over the per-frame rows.
            cam_mean = {
                "camera_id": cam_id,
                "local_timestep": "MEAN",
                "absrel": depth_summary[0],
                "a1": depth_summary[1],
                "ate": pose_summary[0],
                "rpe_trans": pose_summary[1],
                "rpe_rot": pose_summary[2],
            }
        else:
            cam_mean = {
                "camera_id": cam_id,
                "local_timestep": "MEAN",
                "absrel": _nanmean(r["absrel"] for r in cam_rows),
                "a1": _nanmean(r["a1"] for r in cam_rows),
                "ate": _reduce_pose((r["ate"] for r in cam_rows), args.pose_reduce),
                "rpe_trans": _reduce_pose((r["rpe_trans"] for r in cam_rows), args.pose_reduce),
                "rpe_rot": _reduce_pose((r["rpe_rot"] for r in cam_rows), args.pose_reduce),
            }
        rows.append(cam_mean)
        camera_summaries.append(cam_mean)

    if cut3r:
        # CUT3R aggregates depth across sequences weighted by valid-pixel count
        # (np.average(..., weights=valid_pixels)) and pose by a plain mean of the
        # per-sequence RMSEs (calculate_averages). Cameras are the sequences here.
        depth_vpix = [s[2] for s in cam_depth_summaries]
        rows.append(
            {
                "camera_id": "ALL",
                "local_timestep": "MEAN",
                "absrel": _weighted_nanmean(
                    (s[0] for s in cam_depth_summaries), depth_vpix
                ),
                "a1": _weighted_nanmean(
                    (s[1] for s in cam_depth_summaries), depth_vpix
                ),
                "ate": _nanmean(s[0] for s in cam_pose_summaries),
                "rpe_trans": _nanmean(s[1] for s in cam_pose_summaries),
                "rpe_rot": _nanmean(s[2] for s in cam_pose_summaries),
            }
        )
    else:
        rows.append(
            {
                "camera_id": "ALL",
                "local_timestep": "MEAN",
                "absrel": _nanmean(r["absrel"] for r in per_frame_rows),
                "a1": _nanmean(r["a1"] for r in per_frame_rows),
                "ate": _reduce_pose((r["ate"] for r in per_frame_rows), args.pose_reduce),
                "rpe_trans": _reduce_pose((r["rpe_trans"] for r in per_frame_rows), args.pose_reduce),
                "rpe_rot": _reduce_pose((r["rpe_rot"] for r in per_frame_rows), args.pose_reduce),
            }
        )

    out_csv = args.output_csv or (args.pred_root / "eval_depth_pose_metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "camera_id",
        "local_timestep",
        "absrel",
        "a1",
        "ate",
        "rpe_trans",
        "rpe_rot",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    readable_txt = out_csv.with_name("eval_metrics_readable.txt")
    with readable_txt.open("w") as f:
        if cut3r:
            f.write(
                "Args: eval_like_cut3r=True, "
                f"pred_pose_type={args.pred_pose_type}, "
                f"gt_pose_type={args.gt_pose_type}, "
                "pose_align=sim3 (evo APE/RPE, align+correct_scale, RMSE), "
                f"cut3r_depth_align={args.cut3r_depth_align}, "
                f"cut3r_max_depth={cut3r_max_depth}, "
                f"cut3r_post_clip_max={args.cut3r_post_clip_max}\n\n"
            )
        else:
            f.write(
                "Args: "
                f"pred_pose_type={args.pred_pose_type}, "
                f"gt_pose_type={args.gt_pose_type}, "
                f"alignment={args.align}, "
                f"depth_scale_align={args.depth_scale_align}, "
                f"pose_reduce={args.pose_reduce}\n\n"
            )
        for cam_mean in camera_summaries:
            f.write(f"***** Camera {cam_mean['camera_id']} *****\n")
            f.write(
                f"Depth: absrel = {_fmt3(cam_mean['absrel'])}, "
                f"a1 = {_fmt3(cam_mean['a1'])}\n"
            )
            f.write(
                f"Pose: ate = {_fmt3(cam_mean['ate'])}, "
                f"rpe_trans = {_fmt3(cam_mean['rpe_trans'])}, "
                f"rpe_rot = {_fmt3(cam_mean['rpe_rot'])}\n\n"
            )

    print("Evaluation completed.")
    print(f"Num cameras: {args.num_cameras}")
    if cut3r:
        print("Eval mode: CUT3R-identical (--eval_like_cut3r)")
        print(f"Depth alignment: {args.cut3r_depth_align} (joint per camera stream)")
        print(f"Depth max_depth: {cut3r_max_depth}")
        print("Pose: evo APE/RPE, Sim3 align (align+correct_scale), RMSE")
    else:
        print(f"Depth scale alignment: {args.depth_scale_align}")
        print(f"Pose summary reduction: {args.pose_reduce}")
        print(f"Alignment mode: {args.align}")
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote readable metrics: {readable_txt}")


if __name__ == "__main__":
    main()
