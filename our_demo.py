# Basic usage copied from https://github.com/ByteDance-Seed/Depth-Anything-3

import glob, os, torch
from depth_anything_3.api import DepthAnything3
import argparse
import numpy as np
from dataclasses import fields
import matplotlib.pyplot as plt
import trimesh
from depth_anything_3.utils.visualize import visualize_depth
from depth_anything_3.specs import Prediction
from PIL import Image
import imageio

from utils import homogenize_poses, invert_poses, make_relative_poses


def parse_args():
    """Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run DA3 inference on a sequence of images."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the directory containing the input images.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the directory to save the inference outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--visualize_only",
        action="store_true",
        help="Visualize and don't do inference.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Use only the first K frames from the stream (0 means all).",
    )
    parser.add_argument(
        "--invert_poses",
        action="store_true",
        help="Invert the extrinsics (flips c2w to w2c and vice versa) before saving."
    )
    parser.add_argument(
        "--relative_poses",
        action="store_true",
        help="Convert poses from absolute to relative by anchoring at the first pose."
    )
    parser.add_argument(
        "--use_ray_pose",
        action="store_true",
        help="Use ray-based pose estimation instead of camera decoder.",
    )
    parser.add_argument(
        "--num_views_per_step",
        type=int,
        default=2,
        help="Number of views to process per step.",
    )
    return parser.parse_args()


def visualize_depth_mine(input_path: str, output_path: str):
    """Helper function to visualize depths with consistent scale."""
    inputs = sorted(os.listdir(input_path))
    os.makedirs(output_path, exist_ok=True)

    # Load first frame and set vmin/vmax
    first_path = os.path.join(input_path, inputs[0])
    first_all = np.load(first_path)      # (N, H, W)
    first_d = first_all[0]              # use view 0
    # you can use min/max or robust percentiles; pick one:
    # vmin, vmax = float(first_d.min()), float(first_d.max())
    vmin = float(np.percentile(first_d, 1))
    vmax = float(np.percentile(first_d, 99))
    print("Using fixed depth range from first frame:", vmin, "to", vmax)

    # Visualize all frames with same vmin/vmax as the first frame
    for input_name in inputs:
        full_path = os.path.join(input_path, input_name)
        depth_all = np.load(full_path)       # (N, H, W)
        # print("depth_all shape:", depth_all.shape)

        N = depth_all.shape[0]

        for view_idx in range(N):
            d = depth_all[view_idx]

            plt.figure(figsize=(6, 6))
            im = plt.imshow(d, cmap="viridis", vmin=vmin, vmax=vmax)
            plt.colorbar(im, label="Depth (m)")
            plt.axis("off")
            plt.tight_layout()

            # Make the per-view output directory
            os.makedirs(os.path.join(output_path, f"view_{view_idx}"), exist_ok=True)
            view_idx_depth_path = os.path.join(output_path, f"view_{view_idx}", f"{input_name}_depth_vis.jpg")
            plt.savefig(view_idx_depth_path, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close()
        print(f"Saved depth visualization for {input_name}")



def visualize_conf():
    """Helper function to visualize confidences
    """
    raise NotImplementedError("Not implemented yet")


def visualize_points_from_arrays(depth, K_all, ext_all, rgb_all, downsample=4):
    """Create point cloud from saved depth / intrinsics / extrinsics / images."""
    # depth: (N, H, W)
    # K_all: (N, 3, 3)
    # ext_all: (N, 3, 4) w2c
    # rgb_all: (N, H, W, 3)

    all_pts = []
    all_colors = []

    N, H, W = depth.shape

    # pixel grid (downsampled for size)
    us, vs = np.meshgrid(
        np.arange(0, W, downsample),
        np.arange(0, H, downsample),
        indexing="xy",
    )
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1)      # (h', w', 3)
    pix_flat = pix.reshape(-1, 3).T             # (3, P)

    for i in range(N):
        Z = depth[i, ::downsample, ::downsample]       # (h', w')
        Z_flat = Z.reshape(-1)                         # (P,)

        K = K_all[i]                                   # (3, 3)
        w2c = ext_all[i]                               # (3, 4)
        R = w2c[:, :3]
        t = w2c[:, 3:4]                                # (3, 1)

        # camera -> world transform
        R_c2w = R.T
        t_c2w = -R.T @ t

        K_inv = np.linalg.inv(K)
        rays = K_inv @ pix_flat                        # (3, P)
        X_cam = rays * Z_flat[None, :]                 # (3, P)
        X_world = R_c2w @ X_cam + t_c2w                # (3, P)

        # colors
        rgb = rgb_all[i, ::downsample, ::downsample, :]    # (h', w', 3)
        rgb_flat = rgb.reshape(-1, 3)

        all_pts.append(X_world.T)                      # (P, 3)
        all_colors.append(rgb_flat)

    pts = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    return pts, colors


def save_points(points, colors, output_path):
    """Hepler function to save point cloud as a GLB file using the points and colors outputted by visualize_points
    """
    cloud = trimesh.points.PointCloud(points, colors=colors)
    glb_bytes = trimesh.exchange.gltf.export_glb(cloud)
    with open(output_path, "wb") as f:
        f.write(glb_bytes)


def main():
    args = parse_args()
    device = torch.device(args.device)
    input_path = args.input_path
    output_path = args.output_path

    # Visualize and don't do inference
    # Use the same input and output paths from args to get the inference results and store the visualizations
    # IGNORE THIS CODE, IT IS OUTDATED, DO NOT USE --visualize_only
    if args.visualize_only:
        print(f"Visualizing depth...")
        depth_path = os.path.join(input_path, "depth")
        visualize_depth_mine(depth_path, output_path)
        print(f"Done!")

        # TODO: visualize confidences

        print("Generating point clouds...")
        depth_dir = os.path.join(input_path, "depth")
        intr_dir = os.path.join(input_path, "intrinsics")
        extr_dir = os.path.join(input_path, "extrinsics")
        img_dir = os.path.join(input_path, "processed_images")

        os.makedirs(output_path, exist_ok=True)

        depth_files = sorted(os.listdir(depth_dir))
        for fname in depth_files:
            prefix = fname.replace('_depth.npy', '')
            depth = np.load(os.path.join(depth_dir, fname))                    # (N, H, W)
            intr = np.load(os.path.join(intr_dir, f"{prefix}_intrinsics.npy"))   # (N, 3, 3)
            extr = np.load(os.path.join(extr_dir, f"{prefix}_extrinsics.npy"))   # (N, 3, 4)
            rgb  = np.load(os.path.join(img_dir, f"{prefix}_processed_images.npy"))  # (N, H, W, 3)

            pts, cols = visualize_points_from_arrays(depth, intr, extr, rgb, downsample=4)
            glb_out = os.path.join(output_path, f"{prefix}.glb")
            save_points(pts, cols, glb_out)
            print(f"Saved point cloud: {glb_out}")
        print(f"Done!")

        return

    # Load the model
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    model = model.to(device=device)
    model.eval()
    print(f"Model loaded, using device {device}")

    # Load input images
    print(f"Loading images from: {input_path}")
    # NOTE: use png for droid and jpg for robomimic.
    images = sorted(glob.glob(os.path.join(input_path, "*.png")))
    print(f"Found {len(images)} images")
    if len(images) == 0:
        raise ValueError(f"No images found in {input_path}")

    # Run inference on each input image individually
    predictions = []
    incr = args.num_views_per_step
    print(f"Using increment: {incr}")
    print(f"Ray pose estimation: {args.use_ray_pose}")

    for i in range(0, len(images) - 1, incr):
        if i >= args.max_frames:
            print(f"Reached max frames ({args.max_frames}), stopping inference")
            break
        # inference expects a list of images
        current_images = images[i:i+incr]
        print(f"Current images: {len(current_images)}, {current_images}")
        # Try difference reference view strategies
        # Options: "first", "middle", "saddle_balanced", "saddle_sim_range".
        # For single-view, saddle_balanced is preferred
        prediction = model.inference(
            current_images,
            use_ray_pose=args.use_ray_pose,
            ref_view_strategy="first",
            # export_feat_layers=(-1,),   # export features from the final layer
        )
        predictions.append(prediction)

        # print(f"pred: {pred}")
        # Prediction object attributes
        if i == 0:
            print(f"Prediction object fields:")
            for f in fields(prediction):
                print(f"    {f.name}")

        # # prediction.processed_images : [N, H, W, 3] uint8   array
        # print(pred.processed_images.shape)
        # # prediction.depth            : [N, H, W]    float32 array
        # print(pred.depth.shape)  
        # # prediction.conf             : [N, H, W]    float32 array
        # print(pred.conf.shape)  
        # # prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
        # print(pred.extrinsics.shape)
        # # prediction.intrinsics       : [N, 3, 3]    float32 array
        # print(pred.intrinsics.shape)
    print(f"Finished inference on {len(predictions)} tuples")

    # Save predictions to HDD
    os.makedirs(output_path, exist_ok=True)
    output_base = output_path

    # Create subfolders for each field
    depth_vis_path = os.path.join(output_base, "depth_vis")
    depth_path = os.path.join(output_base, "depth")
    conf_path = os.path.join(output_base, "conf")
    extrinsics_path = os.path.join(output_base, "extrinsics")
    intrinsics_path = os.path.join(output_base, "intrinsics")
    processed_images_path = os.path.join(output_base, "processed_images")
    os.makedirs(depth_vis_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(conf_path, exist_ok=True)
    os.makedirs(extrinsics_path, exist_ok=True)
    os.makedirs(intrinsics_path, exist_ok=True)
    os.makedirs(processed_images_path, exist_ok=True)

    # Need to save different initial vmin and vmax for each view
    vmin, vmax = [], []

    # Loop over all predictions and perform postprocessing steps before saving
    for i, prediction in enumerate(predictions):
        print(f"Postprocessing prediction {i}/{len(predictions)}")
        depth = prediction.depth
        conf = prediction.conf
        extrinsics = prediction.extrinsics
        intrinsics = prediction.intrinsics
        processed_images = prediction.processed_images

        # Save a visualization of the depth
        # Return the 95th percentile min and max from the first image and use the same values for all subsequent depths
        for view_idx in range(depth.shape[0]):

            if i == 0:
                depth_vis, depth_min, depth_max = visualize_depth(
                    depth[view_idx],
                    percentile=95,
                    ret_minmax=True,
                    ret_type=np.uint8,
                )
                print(f"Setting vmin[{view_idx}] = {depth_min}, vmax[{view_idx}] = {depth_max}")
                vmin.append(depth_min)
                vmax.append(depth_max)

            else:
                depth_vis = visualize_depth(
                    depth[view_idx],
                    depth_min=vmin[view_idx],
                    depth_max=vmax[view_idx],
                    ret_type=np.uint8,
                )

            view_path = os.path.join(depth_vis_path, f"view_{view_idx}")
            os.makedirs(view_path, exist_ok=True)

            depth_vis_save_path = os.path.join(view_path, f"{i:06d}_depth_vis.png")
            Image.fromarray(depth_vis).save(depth_vis_save_path)
            # print(f"[vmin = {vmin[view_idx]}, vmax = {vmax[view_idx]}] Saved depth visualization for view {view_idx} to {depth_vis_save_path}")


        # Always homogenize poses
        extrinsics = homogenize_poses(extrinsics)

        # NOTE: there appears to be some logic in the code which makes it so that if use_ray_pose is enabled, the model saves poses as c2w.
        if args.invert_poses:
            print("Inverting poses before saving...")
            extrinsics = invert_poses(extrinsics)
            print("Done")

        if args.relative_poses:
            # Anchor everything to the first static camera
            if i == 0:
                ref_pose_static = extrinsics[0].copy()
            print("Making relative poses before saving...")
            extrinsics = make_relative_poses(
                ref_pose_static,
                extrinsics,
            )
            print("Done")

        print(f"Extrinsics final shape: {extrinsics.shape}")

        np.save(os.path.join(depth_path, f"{i:06d}_depth.npy"), depth)
        np.save(os.path.join(conf_path, f"{i:06d}_conf.npy"), conf)
        np.save(os.path.join(extrinsics_path, f"{i:06d}_extrinsics.npy"), extrinsics)
        np.save(os.path.join(intrinsics_path, f"{i:06d}_intrinsics.npy"), intrinsics)
        np.save(os.path.join(processed_images_path, f"{i:06d}_processed_images.npy"), processed_images)
    print(f"Done! Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
