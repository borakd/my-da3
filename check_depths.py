import csv
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


def valid_mask(pred, gt):
    m = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    return m

def median_scale_align(pred, gt, mask, eps=1e-8):
    # Scale pred so median(pred) matches median(gt) on valid pixels
    s = np.median(gt[mask]) / (np.median(pred[mask]) + eps)
    return pred * s

def compute_depth_metrics(pred, gt, mask, eps=1e-8):
    # pred, gt are 2D arrays (same shape), mask is boolean valid mask
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)

    abs_rel = np.mean(np.abs(g - p) / (g + eps))
    # rmse = np.sqrt(np.mean((g - p) ** 2))

    ratio = np.maximum(g / (p + eps), p / (g + eps))
    a1 = np.mean(ratio < 1.25)
    # a2 = np.mean(ratio < 1.25 ** 2)
    # a3 = np.mean(ratio < 1.25 ** 3)

    return {
        "abs_rel": float(abs_rel),
        # "abs_rel_per_px": float(abs_rel_per_px)
        # "rmse": float(rmse),
        "a1 acc": float(a1),
        # "a2 acc": float(a2),
        # "a3 acc": float(a3),
    }


def compute_depth_metrics_maps(pred, gt, mask, eps=1e-8):
    """
    pred, gt: (H, W)
    mask: bool (H, W)
    returns per-pixel maps with NaN outside mask
    """
    absrel_map = np.full_like(gt, np.nan, dtype=np.float32)
    a1_map = np.full_like(gt, np.nan, dtype=np.float32)

    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)

    absrel_vals = np.abs(g - p) / (g + eps)
    ratio = np.maximum(g / (p + eps), p / (g + eps))
    a1_vals = (ratio < 1.25).astype(np.float32)

    absrel_map[mask] = absrel_vals.astype(np.float32)
    a1_map[mask] = a1_vals
    return absrel_map, a1_map


def mean_metrics(metric_list):
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}


def to_depth_2d(depth):
    """Remove singleton dimensions so depth is (H, W)."""
    d = np.squeeze(np.asarray(depth))
    if d.ndim != 2:
        raise ValueError(
            f"depth must be 2D or squeezable to 2D, got shape {np.asarray(depth).shape}"
        )
    return d


def _list_depth_npy_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, "**", "*.npy"), recursive=True))
    return files


def align_depth_to_gt(pred_2d, gt_2d):
    """Uses linear interpolation to align the predicted depth to the ground truth depth.
    """
    pred_2d = to_depth_2d(pred_2d)
    gt_2d = to_depth_2d(gt_2d)
    gt_h, gt_w = gt_2d.shape
    pred_2d = cv2.resize(pred_2d.astype(np.float32), (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
    return pred_2d


def evaluate_depth(pred_depth_path, gt_depth_path):
    """Given a paths to folders containing predicted and ground truth depth files, computes raw
    and median-scale aligned AbsRel and A1 accuracy per-file.
    """
    pred_depth_files = _list_depth_npy_files(pred_depth_path)
    gt_depth_files = _list_depth_npy_files(gt_depth_path)

    assert len(pred_depth_files) == len(gt_depth_files), f"Number of predicted and ground truth depth files must match: {len(pred_depth_files)} != {len(gt_depth_files)}"

    static_raw, static_aligned, static_aligned_depths = [], [], []
    dynamic_raw, dynamic_aligned, dynamic_aligned_depths = [], [], []
    per_frame_aligned = []

    for i, (pred_depth_file, gt_depth_file) in enumerate(zip(pred_depth_files, gt_depth_files)):
        print(f"Evaluating depth {i+1}/{len(pred_depth_files)}")
        stem = os.path.splitext(os.path.basename(pred_depth_file))[0]
        role = "static" if i % 2 == 0 else "dynamic"

        pred_depth = np.load(pred_depth_file)
        gt_depth = np.load(gt_depth_file)

        # Align dims
        pred_depth = align_depth_to_gt(pred_depth, gt_depth)
        gt_depth = to_depth_2d(gt_depth)

        # Sanity-check for invalid pixels
        m = valid_mask(pred_depth, gt_depth)
        if m.sum() == 0:
            print(f"[WARN] no valid pixels at idx={i}, skipping")
            per_frame_aligned.append(
                {
                    "frame_idx": i,
                    "stem": stem,
                    "role": role,
                    "skipped": True,
                    "abs_rel": None,
                    "a1 acc": None,
                }
            )
            continue

        # raw metrics
        raw = compute_depth_metrics(pred_depth, gt_depth, m)

        # median-scale aligned metrics
        pred_aligned = median_scale_align(pred_depth, gt_depth, m)
        aligned = compute_depth_metrics(pred_aligned, gt_depth, m)

        per_frame_aligned.append(
            {
                "frame_idx": i,
                "stem": stem,
                "role": role,
                "skipped": False,
                "abs_rel": aligned["abs_rel"],
                "a1 acc": aligned["a1 acc"],
            }
        )

        # even/odd split: static/dynamic
        if i % 2 == 0:
            static_raw.append(raw)
            static_aligned.append(aligned)
            static_aligned_depths.append(pred_aligned.copy())
        else:
            dynamic_raw.append(raw)
            dynamic_aligned.append(aligned)
            dynamic_aligned_depths.append(pred_aligned.copy())

    # Summary
    static_aligned_mean = mean_metrics(static_aligned)
    dynamic_aligned_mean = mean_metrics(dynamic_aligned)

    return (
        static_raw, dynamic_raw,
        static_aligned, dynamic_aligned,
        static_aligned_mean, dynamic_aligned_mean,
        static_aligned_depths, dynamic_aligned_depths,
        per_frame_aligned,
    )


def visualize_depth(
    input_path: str,
    output_path: str,
    num_views: int,
    static_aligned_depths: list[np.ndarray],
    dynamic_aligned_depths: list[np.ndarray],
):
    """Visualizes the depth maps in the given folder.
    """
    inputs = _list_depth_npy_files(input_path)
    os.makedirs(output_path, exist_ok=True)

    view_paths = []
    for n in range(num_views):
        vp = os.path.join(output_path, f"view_{n}")
        view_paths.append(vp)
        os.makedirs(vp, exist_ok=True)
        os.makedirs(os.path.join(vp, "colorbar"), exist_ok=True)
        os.makedirs(os.path.join(vp, "median_scale_aligned"), exist_ok=True)

    first_path = os.path.join(input_path, inputs[0])
    first_d = to_depth_2d(np.load(first_path))

    second_path = os.path.join(input_path, inputs[1])
    second_d = to_depth_2d(np.load(second_path))

    # raw bounds (keep as-is)
    vmin_static = float(np.percentile(first_d, 1))
    vmax_static = float(np.percentile(first_d, 99))
    vmin_dynamic = float(np.percentile(second_d, 1))
    vmax_dynamic = float(np.percentile(second_d, 99))

    # NEW: aligned bounds
    a0 = static_aligned_depths[0]
    a1 = dynamic_aligned_depths[0]
    aligned_static_vmin = float(np.percentile(a0, 1))
    aligned_static_vmax = float(np.percentile(a0, 99))
    aligned_dynamic_vmin = float(np.percentile(a1, 1))
    aligned_dynamic_vmax = float(np.percentile(a1, 99))

    # print("Using fixed depth range from first frame:", vmin_static, "to", vmax_static)
    # print("Using fixed depth range from second frame:", vmin_dynamic, "to", vmax_dynamic)

    for i, input_name in enumerate(inputs):
        # Interleave order is static first
        if i % 2 == 0:
            raw_vmin, raw_vmax = vmin_static, vmax_static
            aligned_d = static_aligned_depths[i // 2]
            aligned_vmin, aligned_vmax = aligned_static_vmin, aligned_static_vmax
        else:
            raw_vmin, raw_vmax = vmin_dynamic, vmax_dynamic
            aligned_d = dynamic_aligned_depths[i // 2]
            aligned_vmin, aligned_vmax = aligned_dynamic_vmin, aligned_dynamic_vmax

        view_count = i % num_views
        out_dir = view_paths[view_count]

        # Load the depth
        d = to_depth_2d(np.load(os.path.join(input_path, input_name)))

        # 1. Directly save depth
        out_path = os.path.join(out_dir, f"{i:06d}_depth_vis.png")
        plt.imsave(out_path, d, cmap="viridis", vmin=raw_vmin, vmax=raw_vmax)
        print(f"Saved depth to {out_path}")

        # 2. Save depth with colorbar
        plt.figure(figsize=(6, 6))
        im = plt.imshow(d, cmap="viridis", vmin=raw_vmin, vmax=raw_vmax)
        plt.colorbar(im, label="Depth (m)")
        plt.axis("off")
        plt.tight_layout()
        out_path = os.path.join(out_dir, "colorbar", f"{i:06d}_depth_vis.jpg")
        plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Saved depth with colorbar to {out_path}")

        # 3. Save median-scale aligned depth with colorbar
        plt.figure(figsize=(6, 6))
        im = plt.imshow(aligned_d, cmap="viridis", vmin=aligned_vmin, vmax=aligned_vmax)
        plt.colorbar(im, label="Depth (m)")
        plt.axis("off")
        plt.tight_layout()
        out_path = os.path.join(out_dir, "median_scale_aligned", f"{i:06d}_depth_vis.jpg")
        plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Saved scaled depth with colorbar to {out_path}")

        print(f"Saved depth {i}/{len(inputs)}")
        print("*"*80)
        print()


def visualize_depth_errors(
    pred_depth_path: str,
    gt_depth_path: str,
    output_path: str,
    num_views: int,
    static_aligned_depths=None,
    dynamic_aligned_depths=None,
    use_aligned: bool = True,
):
    """
    Save per-frame pixelwise AbsRel and A1 maps.
    If use_aligned=True and aligned depth lists are provided, uses those depths.
    Otherwise uses raw predicted depths from pred_depth_path.
    Outputs go under output_path/view_{k}/errorbar/absrel and .../a1 (k = i % num_views).
    """
    pred_files = _list_depth_npy_files(pred_depth_path)
    gt_files = _list_depth_npy_files(gt_depth_path)
    assert len(pred_files) == len(gt_files), "pred/gt file count mismatch"

    view_errorbar_dirs = []
    for n in range(num_views):
        vp = os.path.join(output_path, f"view_{n}", "errorbar")
        absrel_dir = os.path.join(vp, "absrel")
        a1_dir = os.path.join(vp, "a1")
        os.makedirs(absrel_dir, exist_ok=True)
        os.makedirs(a1_dir, exist_ok=True)
        view_errorbar_dirs.append((absrel_dir, a1_dir))

    for i, (pred_f, gt_f) in enumerate(zip(pred_files, gt_files)):
        name = os.path.splitext(os.path.basename(pred_f))[0]

        gt = to_depth_2d(np.load(gt_f)).astype(np.float32)

        # choose depth to evaluate
        if use_aligned and static_aligned_depths is not None and dynamic_aligned_depths is not None:
            if i % 2 == 0:
                pred_eval = to_depth_2d(static_aligned_depths[i // 2]).astype(np.float32)
            else:
                pred_eval = to_depth_2d(dynamic_aligned_depths[i // 2]).astype(np.float32)
        else:
            pred_eval = to_depth_2d(np.load(pred_f)).astype(np.float32)

        # match GT resolution
        if pred_eval.shape != gt.shape:
            pred_eval = cv2.resize(
                pred_eval, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR
            )

        valid = np.isfinite(pred_eval) & np.isfinite(gt) & (gt > 0)
        if valid.sum() == 0:
            print(f"[WARN] no valid pixels for {name}, skipping")
            continue

        p = pred_eval[valid].astype(np.float64)
        g = gt[valid].astype(np.float64)

        absrel_map = np.full_like(gt, np.nan, dtype=np.float32)
        a1_map = np.full_like(gt, np.nan, dtype=np.float32)

        absrel_vals = np.abs(g - p) / (g + 1e-8)
        ratio = np.maximum(g / (p + 1e-8), p / (g + 1e-8))
        a1_vals = (ratio < 1.25).astype(np.float32)

        absrel_map[valid] = absrel_vals.astype(np.float32)
        a1_map[valid] = a1_vals

        # robust visualization range for absrel
        vmax_absrel = float(np.percentile(absrel_vals, 99))
        vmax_absrel = max(vmax_absrel, 1e-6)

        view_count = i % num_views
        absrel_dir, a1_dir = view_errorbar_dirs[view_count]

        # save absrel heatmap
        plt.figure(figsize=(6, 6))
        im = plt.imshow(absrel_map, cmap="viridis", vmin=0.0, vmax=vmax_absrel)
        plt.colorbar(im, label="AbsRel")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(absrel_dir, f"{name}_absrel.jpg"), dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()

        # save a1 correctness map
        plt.figure(figsize=(6, 6))
        im = plt.imshow(a1_map, cmap="viridis", vmin=0.0, vmax=1.0)
        plt.colorbar(im, label="A1 (1=correct, 0=incorrect)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(a1_dir, f"{name}_a1.jpg"), dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"[error maps] saved {i+1}/{len(pred_files)}: {name}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_depth_path", type=str, required=True) # Effectively the input path
    parser.add_argument("--gt_depth_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=2)
    args = parser.parse_args()

    # Obtain dictionaries with AbsRel and A1 Accuracy
    (
        static_raw, dynamic_raw,
        static_aligned, dynamic_aligned,
        static_aligned_mean, dynamic_aligned_mean,
        static_aligned_depths, dynamic_aligned_depths,
        per_frame_aligned,
    ) = evaluate_depth(args.pred_depth_path, args.gt_depth_path)

    midpoint = len(static_aligned) // 2
    print(f"Index: {midpoint}")
    print(f"static_aligned[{midpoint}]: {static_aligned[midpoint]['abs_rel']:.4f}, {static_aligned[midpoint]['a1 acc']:.4f}")

    # Create output directories
    output_base = args.output_path
    os.makedirs(output_base, exist_ok=True)
    depth_errors_path = os.path.join(output_base, "depth_errors")
    os.makedirs(depth_errors_path, exist_ok=True)

    # Visualize depths
    visualize_depth(
        args.pred_depth_path,
        args.output_path,
        args.num_views,
        static_aligned_depths,
        dynamic_aligned_depths
    )

    # Visualize depth error metrics
    visualize_depth_errors(
        args.pred_depth_path,
        args.gt_depth_path,
        depth_errors_path,
        args.num_views,
        static_aligned_depths=static_aligned_depths,
        dynamic_aligned_depths=dynamic_aligned_depths,
        use_aligned=True,
    )

    print("STATIC (median-scale-aligned)")
    print(f"{{'abs_rel': {static_aligned_mean['abs_rel']:.4f}, 'a1 acc': {static_aligned_mean['a1 acc']:.4f}}}")
    print("DYNAMIC (median-scale-aligned)")
    print(f"{{'abs_rel': {dynamic_aligned_mean['abs_rel']:.4f}, 'a1 acc': {dynamic_aligned_mean['a1 acc']:.4f}}}")
    # Save means and per-frame metrics (same table in .txt and machine-readable .csv)
    _hdr = ["frame_idx", "stem", "role", "skipped", "abs_rel", "a1_acc"]
    _rows = [
        [
            row["frame_idx"],
            row["stem"],
            row["role"],
            row["skipped"],
            "" if row["abs_rel"] is None else f"{row['abs_rel']:.6f}",
            "" if row["a1 acc"] is None else f"{row['a1 acc']:.6f}",
        ]
        for row in per_frame_aligned
    ]
    metrics_path = os.path.join(output_base, "depth_metrics.txt")
    with open(metrics_path, "w", newline="") as f:
        f.write(f"STATIC (median-scale-aligned): {{'abs_rel': {static_aligned_mean['abs_rel']:.4f}, 'a1 acc': {static_aligned_mean['a1 acc']:.4f}}}\n")
        f.write(f"DYNAMIC (median-scale-aligned): {{'abs_rel': {dynamic_aligned_mean['abs_rel']:.4f}, 'a1 acc': {dynamic_aligned_mean['a1 acc']:.4f}}}\n")
        f.write("\n--- per-frame (median-scale aligned) ---\n")
        w = csv.writer(f)
        w.writerow(_hdr)
        w.writerows(_rows)

    per_frame_csv = os.path.join(output_base, "depth_metrics_per_frame.csv")
    with open(per_frame_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_hdr)
        w.writerows(_rows)