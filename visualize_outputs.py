#!/usr/bin/env python3
"""
Minimal visualization utility for with_cut3r.py outputs.

Input layout (default):
  outputs/with_cut3r/decoded/<key>/*.npy

Output layout:
  outputs/with_cut3r/visualizations/<key>/*.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CUT3R bridge outputs.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="outputs/with_cut3r/decoded",
        help="Directory that contains per-key .npy folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/with_cut3r/visualizations",
        help="Directory where PNG visualizations are saved.",
    )
    parser.add_argument(
        "--p_lo",
        type=float,
        default=1.0,
        help="Lower percentile for robust normalization.",
    )
    parser.add_argument(
        "--p_hi",
        type=float,
        default=99.0,
        help="Upper percentile for robust normalization.",
    )
    return parser.parse_args()


def squeeze_batch(arr: np.ndarray) -> np.ndarray:
    out = arr
    while out.ndim > 0 and out.shape[0] == 1:
        out = out[0]
    return out


def robust_normalize(x: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    lo = np.percentile(x[finite], p_lo)
    hi = np.percentile(x[finite], p_hi)
    if hi <= lo:
        hi = lo + 1e-6
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    y[~finite] = 0.0
    return y


def to_uint8_img01(x01: np.ndarray) -> np.ndarray:
    return (np.clip(x01, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_rgb(arr: np.ndarray, out_png: str) -> None:
    x = np.asarray(arr, dtype=np.float32)
    if np.nanmin(x) >= -1.1 and np.nanmax(x) <= 1.1:
        x = (x + 1.0) * 0.5
    else:
        x = robust_normalize(x, 1.0, 99.0)
    Image.fromarray(to_uint8_img01(x)).save(out_png)


def save_scalar(arr2d: np.ndarray, out_png: str, p_lo: float, p_hi: float, cmap: str = "viridis") -> None:
    x01 = robust_normalize(arr2d, p_lo, p_hi)
    rgba = plt.get_cmap(cmap)(x01)  # H,W,4
    rgb = rgba[..., :3]
    Image.fromarray(to_uint8_img01(rgb)).save(out_png)


def list_key_dirs(input_dir: str) -> Iterable[str]:
    dirs = sorted(glob.glob(os.path.join(input_dir, "*")))
    return [d for d in dirs if os.path.isdir(d)]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def visualize_key(
    key: str,
    npy_files: list[str],
    key_out_dir: str,
    p_lo: float,
    p_hi: float,
    pose_rows: list[list[float]],
) -> None:
    ensure_dir(key_out_dir)
    for f in npy_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        arr = squeeze_batch(np.load(f))

        # Camera pose is saved as numeric table + simple translation plot later.
        if key == "camera_pose":
            vals = np.asarray(arr, dtype=np.float32).reshape(-1).tolist()
            pose_rows.append([float(stem)] + vals)
            continue

        # RGB-like tensors.
        if arr.ndim == 3 and arr.shape[-1] == 3 and key == "rgb":
            save_rgb(arr, os.path.join(key_out_dir, f"{stem}.png"))
            continue

        # Confidence maps and generic scalar maps.
        if key.startswith("conf") and arr.ndim == 2:
            save_scalar(arr, os.path.join(key_out_dir, f"{stem}.png"), p_lo, p_hi, cmap="magma")
            continue

        # 3D points: show Z channel + norm map.
        if arr.ndim == 3 and arr.shape[-1] == 3 and key.startswith("pts3d"):
            z = arr[..., 2]
            n = np.linalg.norm(arr, axis=-1)
            save_scalar(z, os.path.join(key_out_dir, f"{stem}_z.png"), p_lo, p_hi, cmap="turbo")
            save_scalar(n, os.path.join(key_out_dir, f"{stem}_norm.png"), p_lo, p_hi, cmap="viridis")
            continue

        # Generic fallback for unknown arrays: try first channel/slice as scalar image.
        fallback = arr
        if fallback.ndim == 3:
            fallback = fallback[..., 0]
        if fallback.ndim == 2:
            save_scalar(fallback, os.path.join(key_out_dir, f"{stem}.png"), p_lo, p_hi, cmap="viridis")


def save_pose_summary(output_dir: str, pose_rows: list[list[float]]) -> None:
    if not pose_rows:
        return

    pose_rows = sorted(pose_rows, key=lambda r: r[0])
    csv_path = os.path.join(output_dir, "camera_pose_values.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        max_len = max(len(r) for r in pose_rows)
        header = ["frame_idx"] + [f"v{i}" for i in range(max_len - 1)]
        writer.writerow(header)
        for r in pose_rows:
            writer.writerow(r + [""] * (max_len - len(r)))

    # Minimal translation plot using first 3 values after frame_idx when available.
    arr = np.array([r + [np.nan] * (8 - len(r)) for r in pose_rows], dtype=np.float32)
    # arr columns: frame_idx, v0, v1, v2, ...
    if arr.shape[1] >= 4:
        x = arr[:, 0]
        tx, ty, tz = arr[:, 1], arr[:, 2], arr[:, 3]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, tx, label="t_x")
        ax.plot(x, ty, label="t_y")
        ax.plot(x, tz, label="t_z")
        ax.set_title("camera_pose first 3 values vs frame")
        ax.set_xlabel("frame")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "camera_pose_plot.png"), dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    key_dirs = list_key_dirs(args.input_dir)
    if not key_dirs:
        raise ValueError(f"No key folders found under {args.input_dir}")

    print(f"Input keys: {[os.path.basename(k) for k in key_dirs]}")
    pose_rows: list[list[float]] = []

    for key_dir in key_dirs:
        key = os.path.basename(key_dir)
        npy_files = sorted(glob.glob(os.path.join(key_dir, "*.npy")))
        if not npy_files:
            continue
        print(f"Visualizing key='{key}' ({len(npy_files)} files)")
        key_out_dir = os.path.join(args.output_dir, key)
        visualize_key(key, npy_files, key_out_dir, args.p_lo, args.p_hi, pose_rows)

    save_pose_summary(args.output_dir, pose_rows)
    print(f"Saved visualizations to: {args.output_dir}")


if __name__ == "__main__":
    main()
