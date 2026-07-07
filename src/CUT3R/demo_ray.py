#!/usr/bin/env python3
"""
3D Point Cloud Inference with GT-pose ray-map conditioning (captain_ray oracle).

Functionally identical to demo.py, with one addition: alongside the RGB stream
(--seq_path, e.g. a DL3DV ``dense/rgb`` directory), a pose stream is loaded
(--pose_path, the matching ``dense/cam`` directory of per-frame ``.npz`` files
carrying ``pose`` (camera-to-world) and ``intrinsic``). Frames are paired by
basename (``000123.png`` <-> ``000123.npz``).

For every frame after the first, the GT camera relative to the FIRST frame is
converted to a ray map with the same get_ray_map used by the training loader,
and fed to the encoder with ray_mask=True — exactly what training with
``feed_gt_ray_map=True`` (config captain_ray) does. Frame 0 is the reference
view: its relative pose is identity, so it stays image-only (ray_mask=False),
also exactly as in training. Intrinsics are adjusted to the load_images_cover
resize+crop so the ray maps live at the model's input resolution.

Usage:
    python demo_ray.py --model_path MODEL_PATH --seq_path RGB_DIR
                       [--pose_path CAM_DIR] [--size 320] [--device cuda]
                       [--output_dir OUT_DIR] [--disable_viewer]

If --pose_path is omitted and --seq_path ends in ``.../rgb``, the sibling
``.../cam`` directory is used.
"""

import os
import numpy as np
import torch
import time
import glob
import random
import argparse
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
import PIL.Image

# Set random seed for reproducibility.
random.seed(42)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference conditioned on GT-pose ray maps."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="/frozen/avg/bora_data/droid_datasets/training_data/pointworld_droid_wrist/dl3dv_multi/AUTOLab+0d4edc83+2023-10-21-19h-06m-10s/18026681+wrist/dense/rgb",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--pose_path",
        type=str,
        default=None,
        help="Directory of per-frame cam .npz files ('pose' c2w + 'intrinsic'), "
        "paired with images by basename. Defaults to the 'cam' directory "
        "next to an 'rgb' --seq_path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="320",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.0,
        help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--disable_viewer",
        action="store_true",
        help="Disable the point cloud viewer.",
    )

    return parser.parse_args()


def crop_resize_training_style(image, K, resolution):
    """Replicate BaseMultiViewDataset._crop_resize_if_necessary (aug_crop=0).

    Same three steps, same cropping utilities, same rounding as the training
    loader (base_multiview_dataset.py:468): (1) crop to a window centered on
    the principal point, (2) Lanczos-rescale to cover `resolution`, (3) final
    crop to exactly `resolution` centered on the principal point. Returns the
    transformed PIL image and adjusted intrinsics.
    """
    import src.dust3r.datasets.utils.cropping as cropping

    W, H = image.size
    depthmap = np.zeros((H, W), dtype=np.float32)  # unused, keeps the API happy

    cx, cy = K[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    assert min_margin_x > W / 5, f"Bad principal point {cx},{cy} for {W}x{H}"
    assert min_margin_y > H / 5, f"Bad principal point {cx},{cy} for {W}x{H}"
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    image, depthmap, K = cropping.crop_image_depthmap(image, depthmap, K, (l, t, r, b))

    image, depthmap, K = cropping.rescale_image_depthmap(
        image, depthmap, K, np.array(resolution)
    )

    K2 = cropping.camera_matrix_of_crop(K, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(K, K2, resolution)
    image, depthmap, K2 = cropping.crop_image_depthmap(image, depthmap, K, crop_bbox)
    return image, K2


def load_frames_training_style(img_paths, pose_path, size):
    """Load images + GT ray maps exactly as the training loader does.

    Each image and its basename-matched cam npz ('pose' c2w, 'intrinsic') go
    through the training crop/resize (adjusting the intrinsics identically),
    then the GT camera relative to the FIRST frame becomes a ray map via the
    same get_ray_map used in training. The target resolution is (--size floored
    to a multiple of 16) wide, with height from the native aspect rounded UP to
    a multiple of 16 — for the 320x180 DROID frames at size=320 this is exactly
    the 320x192 training resolution.

    Returns (images, ray_maps): load_images-style dicts and (1, H, W, 6)
    float32 tensors, in image order.
    """
    from src.dust3r.datasets.base.base_multiview_dataset import get_ray_map
    from src.dust3r.datasets.utils.transforms import ImgNorm

    patch = 16
    ref_pose = None
    images = []
    ray_maps = []
    for i, img_path in enumerate(img_paths):
        base = os.path.splitext(os.path.basename(img_path))[0]
        npz_path = os.path.join(pose_path, base + ".npz")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(
                f"No pose file for frame '{base}': expected {npz_path}"
            )
        cam = np.load(npz_path)
        if "pose" not in cam or "intrinsic" not in cam:
            raise KeyError(
                f"{npz_path} must contain 'pose' and 'intrinsic', has {list(cam.keys())}"
            )
        pose = cam["pose"].astype(np.float32)
        K = cam["intrinsic"].astype(np.float32).copy()

        img = PIL.Image.open(img_path).convert("RGB")
        W1, H1 = img.size
        tw = max(patch, (int(size) // patch) * patch)
        th = ((H1 * tw + W1 * patch - 1) // (W1 * patch)) * patch  # ceil to patch
        img, K = crop_resize_training_style(img, K, (tw, th))

        if ref_pose is None:
            ref_pose = pose
        ray_map = get_ray_map(ref_pose, pose, K, th, tw).astype(np.float32)
        ray_maps.append(torch.from_numpy(ray_map).unsqueeze(0))  # (1, H, W, 6)
        images.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=i,
                instance=str(i),
            )
        )
    return images, ray_maps


def prepare_input(images, ray_maps):
    """
    Prepare input views for inference: every view carries its image, and every
    view after the first additionally carries its GT ray map (ray_mask=True),
    matching training with feed_gt_ray_map=True.

    Args:
        images (list): load_images-style dicts from load_frames_training_style.
        ray_maps (list): Per-frame (1, H, W, 6) GT ray maps, same order.

    Returns:
        list: A list of view dictionaries.
    """
    assert len(images) == len(ray_maps), (
        f"{len(images)} images vs {len(ray_maps)} ray maps — streams out of sync"
    )
    views = []
    for i in range(len(images)):
        h, w = images[i]["img"].shape[-2:]
        assert ray_maps[i].shape == (1, h, w, 6), (
            f"frame {i}: ray map {tuple(ray_maps[i].shape)} vs image {h}x{w}"
        )
        view = {
            "img": images[i]["img"],
            "ray_map": ray_maps[i],
            "true_shape": torch.from_numpy(images[i]["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            # View 0 is the reference frame (identity relative pose): image-only,
            # exactly as in training. All later views are ray-conditioned.
            "ray_mask": torch.tensor(i > 0).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    return views


def prepare_output(outputs, outdir, revisit=1, use_pose=True):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]
    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1).cpu() + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    for f_id in range(len(pts3ds_self)):
        depth = depths_tosave[f_id].cpu().numpy()
        conf = conf_self_tosave[f_id].cpu().numpy()
        color = colors_tosave[f_id].cpu().numpy()
        c2w = cam2world_tosave[f_id].cpu().numpy()
        intrins = intrinsics_tosave[f_id].cpu().numpy()
        np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
        np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
        iio.imwrite(
            os.path.join(outdir, "color", f"{f_id:06d}.png"),
            (color * 255).astype(np.uint8),
        )
        np.savez(
            os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
            pose=c2w,
            intrinsics=intrins,
        )

    return pts3ds_other, colors, conf_other, cam_dict


def parse_seq_path(p):
    """List image files in a directory (videos are not supported here, since a
    matching per-frame pose stream is required)."""
    if not os.path.isdir(p):
        raise ValueError(
            f"--seq_path must be a directory of frames (got {p}); "
            "video input has no matching pose stream."
        )
    img_paths = sorted(
        f for f in glob.glob(f"{p}/*") if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    return img_paths


def default_pose_path(seq_path):
    """For a DL3DV-style '<...>/dense/rgb' stream, use the sibling 'cam' dir."""
    parent, leaf = os.path.split(os.path.normpath(seq_path))
    if leaf == "rgb":
        return os.path.join(parent, "cam")
    return None


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference, inference_recurrent
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import PointCloudViewer

    # Prepare image file paths.
    img_paths = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    # Resolve and load the paired pose stream.
    pose_path = args.pose_path or default_pose_path(args.seq_path)
    if pose_path is None or not os.path.isdir(pose_path):
        print(
            f"No pose directory found (got {pose_path!r}). Pass --pose_path "
            "explicitly, or point --seq_path at a '.../dense/rgb' directory."
        )
        return

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    print(f"Loading frames + GT poses from {pose_path} (training-style crop)...")
    images, ray_maps = load_frames_training_style(img_paths, pose_path, args.size)

    # Prepare input views.
    print("Preparing input views...")
    views = prepare_input(images=images, ray_maps=ray_maps)
    print(
        f"Ray conditioning: view 0 image-only (reference frame), "
        f"views 1..{len(views) - 1} image + GT ray map."
    )

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, state_args = inference(views, model, device)
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    pts3ds_other, colors, conf, cam_dict = prepare_output(
        outputs, args.output_dir, 1, True
    )

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c.cpu().numpy() for c in colors]
    edge_colors = [None] * len(pts3ds_to_vis)

    # Create and run the point cloud viewer.
    if not args.disable_viewer:
        print("Launching point cloud viewer...")
        viewer = PointCloudViewer(
            model,
            state_args,
            pts3ds_to_vis,
            colors_to_vis,
            conf,
            cam_dict,
            device=device,
            edge_color_list=edge_colors,
            show_camera=True,
            vis_threshold=args.vis_threshold,
            size=args.size,
        )
        viewer.run()
    else:
        print("Point cloud viewer disabled. Skipping...")


def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
