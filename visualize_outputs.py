# Helper script to visualize depth from .npy files using fixed vmax/vmin


import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def visualize_depth_da3(input_path: str, output_path: str):
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
    for i, input_name in enumerate(inputs):
        
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


def visualize_depth_gt(input_path: str, output_path: str):
    """Helper function to visualize depths with consistent scale.
    
    Args:
        input_path: path to the input depth map
        output_path: path to the output directory
        view: name of the view (static or dynamic)
    """
    inputs = sorted(os.listdir(input_path))
    os.makedirs(output_path, exist_ok=True)

    # Load first frame and set vmin/vmax
    first_path = os.path.join(input_path, inputs[0])
    first_d = np.load(first_path)      # (H, W)
    
    # you can use min/max or robust percentiles; pick one:
    # vmin, vmax = float(first_d.min()), float(first_d.max())
    vmin = float(np.percentile(first_d, 1))
    vmax = float(np.percentile(first_d, 99))
    print("Using fixed depth range from first frame:", vmin, "to", vmax)

    # Visualize all frames with same vmin/vmax as the first frame
    for i, input_name in enumerate(inputs):
        print(f"DEBUG input_name: {input_name}")
        full_path = os.path.join(input_path, input_name)
        d = np.load(full_path)       # (H, W)
        # print("depth_all shape:", depth_all.shape)

        plt.figure(figsize=(6, 6))
        im = plt.imshow(d, cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="Depth (m)")
        plt.axis("off")
        plt.tight_layout()

        out_path = os.path.join(output_path, f"{input_name}_depth_vis.jpg")
        plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()
    print(f"Saved depth visualization for {input_name}")


def visualize_depth(input_path: str, output_path: str, num_views: int):
    inputs = sorted(os.listdir(input_path))
    os.makedirs(output_path, exist_ok=True)

    view_paths = []
    for n in range(num_views):
        vp = os.path.join(output_path, f"view_{n}")
        view_paths.append(vp)
        os.makedirs(vp, exist_ok=True)

    first_path = os.path.join(input_path, inputs[0])
    first_d = np.load(first_path)
    if first_d.ndim == 3 and first_d.shape[0] == 1:
        first_d = np.squeeze(first_d, axis=0)

    second_path = os.path.join(input_path, inputs[1])
    second_d = np.load(second_path)
    if second_d.ndim == 3 and second_d.shape[0] == 1:
        second_d = np.squeeze(second_d, axis=0)

    vmin_static = float(np.percentile(first_d, 1))
    vmax_static = float(np.percentile(first_d, 99))
    vmin_dynamic = float(np.percentile(second_d, 1))
    vmax_dynamic = float(np.percentile(second_d, 99))
    print("Using fixed depth range from first frame:", vmin_static, "to", vmax_static)
    print("Using fixed depth range from second frame:", vmin_dynamic, "to", vmax_dynamic)

    for i, input_name in enumerate(inputs):
        # For even-numbered inputs, since interleave order is static first
        if i % 2 == 0:
            vmin = vmin_static
            vmax = vmax_static
        else:
            vmin = vmin_dynamic
            vmax = vmax_dynamic
        view_count = i % num_views
        out_dir = view_paths[view_count]

        d = np.load(os.path.join(input_path, input_name))
        if d.ndim == 3 and d.shape[0] == 1:
            d = np.squeeze(d, axis=0)

        # Saves exactly HxW (e.g., 504x504) instead of rendering on a larger figure.
        out_path = os.path.join(out_dir, f"{input_name}_depth_vis.png")
        plt.imsave(out_path, d, cmap="viridis", vmin=vmin, vmax=vmax)

        print(f"Saved depth visualization for {input_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize_depth", action="store_true", help="Visualize depth maps.")
    parser.add_argument("--imgs_to_vid", action="store_true", help="Convert images to video.")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_views", type=int, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    num_views = args.num_views
    model_type = args.model_type

    if visualize_depth:
        # debugging
        l = sorted(os.listdir(input_path))
        x = np.load(os.path.join(input_path, l[0]), allow_pickle=True)
        print(x.shape, x.dtype, type(x))

        print(f"Visualizing depths from {input_path}...")
        if model_type == 'da3' or model_type == 'vggt':
            visualize_depth_da3(input_path, output_path)    # use this for da3/vggt
        elif model_type == 'gt':
            visualize_depth_gt(input_path, output_path)
        elif model_type == 'ttt3r' or model_type == 'cut3r':
            visualize_depth(input_path, output_path, num_views)    # use this for ttt3r/cut3r
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        print(f"Done! Visuals saved to {output_path}")
