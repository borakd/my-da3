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

--conditioning selects which variant to reproduce at inference:
  gt        (default) view x carries its own GT ray map — feed_gt_ray_map
            training (config captain_ray).
  prev_gt   view x carries view x-1's GT ray map (one-step lag) —
            feed_prev_gt_ray_map training (config captain_ray_prev_gt).
  prev_pred no ray maps are fed from data; the model builds view x's ray map
            from its own pose prediction at step x-1 (model.feed_prev_pred) —
            config captain_ray_prev_pred. Poses are still loaded for the
            intrinsics the ray-map builder needs. A PoseGRU carried by the
            checkpoint is STRIPPED here, so this is the raw closed loop (the
            GRU-less ablation of a GRU checkpoint).
  prev_pred_gru
            same closed loop as prev_pred, but the PoseGRU refiner stays
            active: it filters the fed-back pose before it is rendered into
            the conditioning ray map. The module and all of its
            hyperparameters (mode / input / hidden_dim / img_feat / iters) are
            restored from the checkpoint by load_model — nothing to pass.
            Use this for every captain_gru_v2 / captain_gru_v3 checkpoint.
  none      no ray conditioning at all — byte-for-byte demo.py behavior
            (images via load_images_cover, ray_mask=False everywhere, no pose
            files needed / --pose_path ignored). Use for regular CUT3R
            checkpoints, or as the unconditioned control for any checkpoint.

Usage:
    python demo_ray.py --model_path MODEL_PATH --seq_path RGB_DIR
                       [--pose_path CAM_DIR]
                       [--conditioning gt|prev_gt|prev_pred|prev_pred_gru|none]
                       [--size 320] [--device cuda]
                       [--output_dir OUT_DIR] [--disable_viewer]

If --pose_path is omitted and --seq_path ends in ``.../rgb``, the sibling
``.../cam`` directory is used.

PoseGRU checkpoints (checkpoints/captain_gru_overfit/*): the run-name levers are
all recorded inside the checkpoint, so the ONLY argument that varies is nothing —
every arm runs with ``--conditioning prev_pred_gru``. The naming convention maps
to ckpt["args"] entries that load_model reads back automatically:

    captain_gru_v2_a{1,2,3,4}_g{0,1,2}      captain_gru_v3_a4_g{1,2}[_f1]_r8
    ------------------------------------    --------------------------------
    a1  pose_gru_mode=direct    input=pose        f1  pose_gru_img_feat=input
    a2  pose_gru_mode=residual  input=pose            (dim 32, frames 2)
    a3  pose_gru_mode=direct    input=pose_delta  r8  pose_gru_iters=8
    a4  pose_gru_mode=residual  input=pose_delta      (iter_gamma 0.8)
    g0  bptt=False e2e=False    g1  bptt=True e2e=False    g2  bptt=False e2e=True

A/F levers are also cross-checked against the weight shapes (weight_ih width
7 vs 14, plus img_proj), and load_state_dict hard-fails on any pose_gru
mismatch — a mis-sniffed GRU can never evaluate as random init. The G levers
only route training-time gradients and are inert under inference.
"""

import argparse
import glob
import os
import random
import time
import imageio.v2 as iio
import numpy as np
import PIL.Image
import torch
from add_ckpt_path import add_path_to_dust3r

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
        default=os.environ.get(
            "DEMO_SEQ_PATH",
            # MN5 default; the old /frozen wrist store is not on this cluster.
            "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/"
            "wrist/AUTOLab+0d4edc83+2023-10-21-19h-11m-38s/dense/rgb",
        ),
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
        "--conditioning",
        type=str,
        default="gt",
        choices=["gt", "prev_gt", "prev_pred", "prev_pred_gru", "none"],
        help="Ray conditioning variant: 'gt' = each view its own GT ray map "
        "(feed_gt_ray_map), 'prev_gt' = view x gets view x-1's GT ray map "
        "(feed_prev_gt_ray_map), 'prev_pred' = the model builds view x's ray "
        "map from its own step-(x-1) pose prediction (feed_prev_pred), with any "
        "checkpoint PoseGRU stripped, 'prev_pred_gru' = same closed loop with "
        "the checkpoint's PoseGRU refiner active (all its hyperparameters are "
        "restored from the ckpt), 'none' = no ray conditioning, exactly demo.py "
        "(regular CUT3R checkpoints; needs no pose files).",
    )
    parser.add_argument(
        "--oracle",
        type=str,
        default="off",
        choices=["off", "gt"],
        help="pose_gru_oracle run mode (DIAGNOSTIC, not an arm): 'gt' feeds the "
        "GRU the CURRENT view's GT pose instead of the fed-back prediction — "
        "requires --conditioning prev_pred_gru and a pose stream (the GT "
        "camera_pose goes into the views). A correctly wired zero-init residual "
        "GRU is then an exact pass-through. Default 'off' = honest closed loop, "
        "even for checkpoints TRAINED with the oracle (a loud warning is printed "
        "for those either way).",
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

    image, depthmap, K = cropping.rescale_image_depthmap(image, depthmap, K, np.array(resolution))

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

    Returns (images, ray_maps, intrinsics_list, poses): load_images-style
    dicts, (1, H, W, 6) float32 ray maps, (1, 3, 3) intrinsics and (1, 4, 4)
    GT c2w poses, all in image order. The poses are what the ray maps were
    built from — hand them to prepare_input(gt_poses=...) so the views carry
    real GT camera_pose (required by the pose_gru_oracle diagnostic; inert
    data for every honest arm).
    """
    from src.dust3r.datasets.base.base_multiview_dataset import get_ray_map
    from src.dust3r.datasets.utils.transforms import ImgNorm

    patch = 16
    ref_pose = None
    images = []
    ray_maps = []
    intrinsics_list = []
    poses = []
    for i, img_path in enumerate(img_paths):
        base = os.path.splitext(os.path.basename(img_path))[0]
        npz_path = os.path.join(pose_path, base + ".npz")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"No pose file for frame '{base}': expected {npz_path}")
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
        intrinsics_list.append(torch.from_numpy(K).unsqueeze(0))  # (1, 3, 3)
        poses.append(torch.from_numpy(pose).unsqueeze(0))  # (1, 4, 4) GT c2w
        images.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=i,
                instance=str(i),
            )
        )
    return images, ray_maps, intrinsics_list, poses


def prepare_input(images, ray_maps, intrinsics_list, conditioning="gt", gt_poses=None):
    """
    Prepare input views for inference. Every view carries its image; the ray
    conditioning depends on the mode:

    - "gt": every view after the first carries its own GT ray map
      (ray_mask=True), matching training with feed_gt_ray_map=True.
    - "prev_gt": every view after the first carries the PREVIOUS view's GT ray
      map (ray_mask=True), matching feed_prev_gt_ray_map=True (the loader's
      one-step shift).
    - "prev_pred" / "prev_pred_gru": no data-side rays (ray_mask=False
      everywhere); the model builds view x's ray map from its own step-(x-1)
      pose prediction (model.feed_prev_pred must be set). Views must carry
      camera_intrinsics. The two modes build IDENTICAL views — the PoseGRU
      acts inside the decoder step, not in the inputs — so they differ only in
      whether run_inference keeps the checkpoint's refiner.

    Args:
        images (list): load_images-style dicts from load_frames_training_style.
        ray_maps (list): Per-frame (1, H, W, 6) GT ray maps, same order.
        intrinsics_list (list): Per-frame (1, 3, 3) intrinsics, same order.
        conditioning (str): "gt", "prev_gt", "prev_pred" or "prev_pred_gru".
        gt_poses (list|None): Per-frame (1, 4, 4) GT c2w poses (the 4th return
            of load_frames_training_style). When given, each view carries its
            REAL camera_pose instead of an identity placeholder — required by
            the pose_gru_oracle diagnostic (model.pose_gru_oracle='gt', which
            reads views[x]['camera_pose'] via gt_pose_encoding) and inert data
            for every honest arm (the model never reads camera_pose otherwise;
            losses are not run at inference).

    Returns:
        list: A list of view dictionaries.
    """
    assert len(images) == len(ray_maps) == len(intrinsics_list), (
        f"{len(images)} images vs {len(ray_maps)} ray maps vs "
        f"{len(intrinsics_list)} intrinsics — streams out of sync"
    )
    assert conditioning in ("gt", "prev_gt", "prev_pred", "prev_pred_gru"), conditioning
    # The GRU changes nothing on the input side; collapse it so the ray_mask /
    # map-index logic below stays single-branch.
    if conditioning == "prev_pred_gru":
        conditioning = "prev_pred"
    views = []
    for i in range(len(images)):
        h, w = images[i]["img"].shape[-2:]
        # prev_gt: view i is conditioned on view i-1's camera — its entire GT
        # ray map, exactly like the training loader's one-step shift. View 0
        # keeps its own map but is masked off below.
        map_idx = max(i - 1, 0) if conditioning == "prev_gt" else i
        assert ray_maps[map_idx].shape == (
            1,
            h,
            w,
            6,
        ), f"frame {i}: ray map {tuple(ray_maps[map_idx].shape)} vs image {h}x{w}"
        view = {
            "img": images[i]["img"],
            "ray_map": ray_maps[map_idx],
            "true_shape": torch.from_numpy(images[i]["true_shape"]),
            "idx": i,
            "instance": str(i),
            # Real GT pose when the caller provides it (oracle-capable views);
            # identity placeholder otherwise. Only the pose_gru_oracle path
            # ever reads this at inference.
            "camera_pose": (
                gt_poses[i].float()
                if gt_poses is not None
                else torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0)
            ),
            # Needed by feed_prev_pred's in-loop ray-map builder; harmless
            # (unused by the model) in the other modes.
            "camera_intrinsics": intrinsics_list[i],
            "img_mask": torch.tensor(True).unsqueeze(0),
            # View 0 is the reference frame: image-only, exactly as in training
            # (gt: its relative pose is identity; prev_gt: it has no
            # predecessor). For prev_pred no data-side rays are fed at all.
            "ray_mask": torch.tensor(i > 0 and conditioning != "prev_pred").unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    return views


def prepare_input_none(img_paths, size):
    """Build views with NO ray conditioning — a verbatim replica of demo.py's
    images-only ``prepare_input`` branch (load_images_cover, NaN placeholder
    ray_map, ray_mask=False everywhere). No pose files are read, so this works
    for regular CUT3R checkpoints exactly like demo.py does.
    """
    from src.dust3r.utils.image import load_images_cover

    images = load_images_cover(img_paths, size=size, square_ok=True)
    views = []
    for i in range(len(images)):
        view = {
            "img": images[i]["img"],
            "ray_map": torch.full(
                (
                    images[i]["img"].shape[0],
                    6,
                    images[i]["img"].shape[-2],
                    images[i]["img"].shape[-1],
                ),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(images[i]["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
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
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.camera import pose_encoding_to_camera
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
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu() for pred in outputs["pred"]
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

    colors = [0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]]

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
        [0.5 * (output["img"].permute(0, 2, 3, 1).cpu() + 1.0) for output in outputs["views"]]
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
    img_paths = sorted(f for f in glob.glob(f"{p}/*") if f.lower().endswith(IMAGE_EXTENSIONS))
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
    from src.dust3r.inference import inference
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import PointCloudViewer

    # Prepare image file paths.
    img_paths = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    if args.conditioning == "none":
        # Backwards-compatible unconditioned path: exactly demo.py. No pose
        # stream is needed (or read); images go through load_images_cover.
        print(f"Found {len(img_paths)} images in {args.seq_path}.")
        print("Preparing input views (no ray conditioning — demo.py-equivalent)...")
        views = prepare_input_none(img_paths, args.size)
        print(
            f"Ray conditioning: NONE — all {len(views)} views image-only "
            "(ray_mask=False), identical to demo.py."
        )
    else:
        # Resolve and load the paired pose stream.
        pose_path = args.pose_path or default_pose_path(args.seq_path)
        if pose_path is None or not os.path.isdir(pose_path):
            print(
                f"No pose directory found (got {pose_path!r}). Pass --pose_path "
                "explicitly, point --seq_path at a '.../dense/rgb' directory, "
                "or use --conditioning none (no poses needed)."
            )
            return

        print(f"Found {len(img_paths)} images in {args.seq_path}.")
        print(f"Loading frames + GT poses from {pose_path} (training-style crop)...")
        images, ray_maps, intrinsics_list, gt_poses = load_frames_training_style(
            img_paths, pose_path, args.size
        )

        # Prepare input views. The GT poses ride along as plain data so the
        # views are oracle-capable; no honest arm reads them.
        print("Preparing input views...")
        views = prepare_input(
            images=images,
            ray_maps=ray_maps,
            intrinsics_list=intrinsics_list,
            conditioning=args.conditioning,
            gt_poses=gt_poses,
        )
        if args.conditioning == "gt":
            print(
                f"Ray conditioning: view 0 image-only (reference frame), "
                f"views 1..{len(views) - 1} image + own GT ray map."
            )
        elif args.conditioning == "prev_gt":
            print(
                f"Ray conditioning: view 0 image-only (no predecessor), "
                f"views 1..{len(views) - 1} image + PREVIOUS view's GT ray map."
            )
        else:
            print(
                f"Ray conditioning: no data-side rays; the model builds view x's "
                f"ray map from its own step-(x-1) pose prediction "
                f"(views 1..{len(views) - 1})."
            )

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    if args.conditioning in ("prev_pred", "prev_pred_gru"):
        # Closed-loop conditioning happens inside _forward_decoder_group_step;
        # this model-side flag turns it on (same flag training sets from the
        # captain_ray_prev_pred config).
        model.feed_prev_pred = True
    # PoseGRU handling. load_model already materialized the refiner and restored
    # every hyperparameter from ckpt["args"] + the weight shapes, so the module
    # is present iff the checkpoint was trained with one. All that is left is to
    # honor the requested arm — kept identical to
    # eval_pipeline/infer_and_eval_worker_ray.py so demo_ray and the batch worker
    # never diverge on the same --conditioning string.
    if args.conditioning == "prev_pred_gru":
        if getattr(model, "pose_gru", None) is None:
            # A ckpt with no pose_gru weights still runs, but as an
            # identity-init refiner it is NOT a trained arm — say so loudly.
            print(
                "NOTE: checkpoint has no pose_gru weights — enabling an "
                "identity-init residual GRU (== plain prev_pred). Smoke-test "
                "only; results are not a trained-GRU arm."
            )
            model.enable_pose_gru()
            model.pose_gru.to(device)
        gru = model.pose_gru
        print(
            f"PoseGRU active (from ckpt): mode={gru.mode}, input={gru.input_mode}, "
            f"hidden_dim={gru.hidden_dim}, img_feat={gru.img_feat}"
            + (
                # img_feat_dim is the APPENDED cell width: the projector dim
                # when proj=True, the full src_dim*blocks width when proj=False
                # (blocks == frames for every source but corr, which is a
                # pair statistic: frames=2 semantically, ONE appended block).
                # src is the SOURCE sub-lever (pooled/resnet18/dinov2_vits14/
                # corr) — getattr'd so pickled pre-lever modules still print.
                f" (src={getattr(gru, 'img_feat_src', 'pooled')}, "
                f"appended={gru.img_feat_dim}, frames={gru.img_feat_frames}, "
                f"proj={gru.img_feat_proj})"
                if gru.img_feat == "input"
                else ""
            )
            + f", iters={gru.iters}"
            # P3.4 input_gain: a per-dimension scale on the cell input, carried
            # as a BUFFER. Restored by the load_model sniff like every other
            # lever, but unlike mode/iters a lost gain changes no shape and
            # raises no error — it would just silently evaluate a different
            # module. Print it (kept identical to
            # eval_pipeline/infer_and_eval_worker_ray.py, per the note above).
            + (
                ""
                if getattr(gru, "input_gain", None) is None
                else f", input_gain={[round(v, 4) for v in gru.input_gain.tolist()]}"
            )
            + "."
        )
    elif getattr(model, "pose_gru", None) is not None:
        # The GRU runs whenever the module exists and feed_prev_pred is on, so a
        # GRU checkpoint evaluated under any other arm must have it removed —
        # otherwise "prev_pred" would silently still be the refined closed loop.
        print(
            f"Checkpoint carries pose_gru but --conditioning {args.conditioning} "
            "— disabling the GRU for this run (GRU-less ablation)."
        )
        model.pose_gru = None

    # Run-mode cross-checks against how the checkpoint was TRAINED (stashed on
    # the net by load_model from ckpt['args']). Mismatches are legal ablations,
    # but never silent ones.
    trained_cond = getattr(model, "trained_conditioning", "none")
    if trained_cond != args.conditioning:
        print(
            f"WARNING: checkpoint was trained with conditioning "
            f"'{trained_cond}' but this run uses '{args.conditioning}' — an "
            "out-of-distribution probe, not the checkpoint's honest arm. "
            "Label the results accordingly."
        )
    trained_oracle = getattr(model, "trained_pose_gru_oracle", "off")
    if args.oracle == "gt":
        # DIAGNOSTIC: GT pose at the GRU input. Needs the refiner active and
        # views that carry REAL GT camera_pose (prepare_input(gt_poses=...) —
        # every conditioned mode above builds them; 'none' fills identities,
        # which would make the "oracle" silently feed identity poses).
        assert args.conditioning == "prev_pred_gru", (
            "--oracle gt refines the GRU input — it requires "
            "--conditioning prev_pred_gru"
        )
        model.pose_gru_oracle = "gt"
        print(
            "*** pose_gru_oracle=gt — DIAGNOSTIC RUN, NOT an arm. GT pose is "
            "injected at the GRU input (views carry the real GT camera_pose); "
            "a zero-init residual GRU is an exact pass-through. Outputs are "
            "GT-derived — never compare against honest runs. ***"
        )
    elif trained_oracle != "off":
        print(
            f"WARNING: checkpoint was TRAINED with pose_gru_oracle="
            f"'{trained_oracle}' (GT-injection diagnostic arm) but this run "
            "keeps the oracle OFF: the GRU now sees predicted poses it never "
            "trained on. Honest-loop metrics of an oracle-trained arm are an "
            "out-of-distribution probe (pass --oracle gt to reproduce the "
            "training-time wiring)."
        )
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
    pts3ds_other, colors, conf, cam_dict = prepare_output(outputs, args.output_dir, 1, True)

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
