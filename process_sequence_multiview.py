import os
import glob
import shutil
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import trimesh

from depth_anything_3.api import DepthAnything3

# TODO: use provided point cloud conversion and GLB saving functions

def depth_to_point_cloud(depth, intrinsics, extrinsics, image=None):
    """
    Convert depth map to point cloud in world coordinates.
    
    Args:
        depth: [H, W] numpy array
        intrinsics: [3, 3] numpy array
        extrinsics: [3, 4] numpy array (world-to-camera)
        image: [H, W, 3] numpy array (optional, for colors)
    
    Returns:
        points: [N, 3] numpy array (world coordinates)
        colors: [N, 3] numpy array (if image provided)
    """
    H, W = depth.shape
    
    # Create pixel coordinates
    u = np.arange(W).astype(np.float32)
    v = np.arange(H).astype(np.float32)
    u, v = np.meshgrid(u, v)
    ones = np.ones_like(u)
    pixel_coords = np.stack([u, v, ones], axis=-1)  # [H, W, 3]
    
    # Convert to camera coordinates
    intrinsics_inv = np.linalg.inv(intrinsics)
    camera_coords = np.einsum('ij,hwj->hwi', intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth[..., np.newaxis]  # [H, W, 3]
    
    # Convert to homogeneous coordinates
    camera_coords_homo = np.concatenate([
        camera_coords,
        np.ones((H, W, 1), dtype=np.float32)
    ], axis=-1)  # [H, W, 4]
    
    # Convert extrinsics to 4x4
    extrinsics_4x4 = np.eye(4, dtype=np.float32)
    extrinsics_4x4[:3, :] = extrinsics
    
    # World-to-camera to camera-to-world
    c2w = np.linalg.inv(extrinsics_4x4)
    
    # Transform to world coordinates
    world_coords_homo = np.einsum('ij,hwj->hwi', c2w, camera_coords_homo)
    points = world_coords_homo[..., :3]  # [H, W, 3]
    
    # Reshape to [N, 3]
    points = points.reshape(-1, 3)
    
    # Get colors if image provided
    colors = None
    if image is not None:
        colors = image.reshape(-1, 3)  # [N, 3]
    
    return points, colors


def save_point_cloud_glb(points, colors, filepath, conf_mask=None):
    """
    Save point cloud to GLB file.
    
    Args:
        points: [N, 3] numpy array
        colors: [N, 3] numpy array (0-255 uint8) or None
        filepath: Output file path
        conf_mask: [N] boolean array for filtering points
    """
    if conf_mask is not None:
        points = points[conf_mask]
        if colors is not None:
            colors = colors[conf_mask]
    
    # Create trimesh point cloud
    if colors is not None:
        # Normalize colors to 0-1 range
        colors_normalized = colors.astype(np.float32) / 255.0
        pc = trimesh.PointCloud(vertices=points, colors=colors_normalized)
    else:
        pc = trimesh.PointCloud(vertices=points)
    
    # Create scene and export to GLB
    scene = trimesh.Scene()
    scene.add_geometry(pc)
    scene.export(filepath)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Process multi-view image sequence with DA3")
    parser.add_argument("--episode_dir", type=str, required=True,
                       help="Episode directory (e.g., /path/to/episodes/episode_000000)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save outputs")
    parser.add_argument("--model", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                       help="DA3 model name or HuggingFace path")
    parser.add_argument("--no-depth", action="store_true",
                       help="Don't save depth maps")
    parser.add_argument("--no-pc", action="store_true",
                       help="Don't save point clouds")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on (cuda/cpu)")
    
    args = parser.parse_args()

    # Get per view image paths
    static1_paths = []
    static2_paths = []
    dyamic_paths = []

    # Use episode directory directly
    episode_dir = args.episode_dir
    print(f"Episode directory: {episode_dir}")

    static1_paths.extend(glob.glob(os.path.join(episode_dir, "exterior_image_1_left/images", "*.png")))
    static2_paths.extend(glob.glob(os.path.join(episode_dir, "exterior_image_2_left/images", "*.png")))
    dyamic_paths.extend(glob.glob(os.path.join(episode_dir, "wrist_image_left/images", "*.png")))

    static1_paths = sorted(list(set(static1_paths)))
    static2_paths = sorted(list(set(static2_paths)))
    dyamic_paths = sorted(list(set(dyamic_paths)))

    print(f"Found {len(static1_paths)} images")
    print(f"Found {len(static2_paths)} images")
    print(f"Found {len(dyamic_paths)} images")

    # Load model once (not inside the loop)
    print(f"Loading model: {args.model}")
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device=args.device)
    model.eval()

    # Create output directories once (don't clear them in the loop)
    if not args.no_depth:
        depth_maps_dir = os.path.join(args.output_dir, "depth_maps")
        os.makedirs(depth_maps_dir, exist_ok=True)
    if not args.no_pc:
        point_clouds_dir = os.path.join(args.output_dir, "point_clouds")
        os.makedirs(point_clouds_dir, exist_ok=True)

    reference_scale = None

    # TODO: set the depth percentile to be fixed beforehand
    max_depth_value = 0

    # Process each time step as multi-view (3 views together)
    for i in range(len(static1_paths)):
        # image_paths = [dyamic_paths[i], static1_paths[i], static2_paths[i]]
        image_paths = [static1_paths[i], static2_paths[i], dyamic_paths[i]]
        
        print(f"\nProcessing time step {i+1}/{len(static1_paths)}: {[Path(p).name for p in image_paths]}")
        
        # Run multi-view inference (all 3 views together)
        prediction = model.inference(
            image=image_paths,
            export_dir=None,
            ref_view_strategy="first",
        )
        
        # Compute scale factor for this frame based on reference view (dynamic, index 0)
        if reference_scale is None:
            # Set reference scale from first frame, reference view (dynamic)
            ref_depth = prediction.depth[0]
            valid_depth = ref_depth[ref_depth > 0]
            if len(valid_depth) > 0:
                reference_scale = np.percentile(valid_depth, 95)
                print(f"  Setting reference depth scale (95th percentile): {reference_scale:.4f}")
        
        # Compute single scale factor for this entire frame
        # frame_scale_factor = 1.0
        # if reference_scale is not None:
        #     ref_depth = prediction.depth[0]  # Reference view (dynamic)
        #     valid_depth = ref_depth[ref_depth > 0]
        #     if len(valid_depth) > 0:
        #         current_scale = np.percentile(valid_depth, 95)
        #         if current_scale > 1e-6:
        #             frame_scale_factor = reference_scale / current_scale
        #             print(f"  Frame {i+1} scale factor: {frame_scale_factor:.4f}")
        
        # Apply the same scale factor to all views in this frame
        # if frame_scale_factor != 1.0:
        #     prediction.depth = prediction.depth
        #     prediction.extrinsics = prediction.extrinsics.copy()
        #     prediction.extrinsics[:, :3] = prediction.extrinsics[:, :3]
        
        # Save results for each view
        for view_idx, (img_path, depth, intrinsics, extrinsics, image) in enumerate(zip(
            image_paths,
            prediction.depth,
            prediction.intrinsics,
            prediction.extrinsics,
            prediction.processed_images
        )):
            camera_name = Path(img_path).parent.parent.name
            base_name = Path(img_path).stem
            
            # Save depth map (already scaled)
            if not args.no_depth:
                camera_depth_dir = os.path.join(args.output_dir, "depth_maps", camera_name)
                os.makedirs(camera_depth_dir, exist_ok=True)
                
                depth_file = os.path.join(camera_depth_dir, f"{base_name}_depth.npy")
                np.save(depth_file, depth)
                
                # TODO: debug depth visualization
                # try fixing max_depth, min_depth (need to precompute max_depth)
                # try fixing percentile and not setting min or max depth
                from depth_anything_3.utils.visualize import visualize_depth
                depth_vis = visualize_depth(depth, percentile=95)
                depth_vis_file = os.path.join(camera_depth_dir, f"{base_name}_depth_vis.jpg")
                Image.fromarray(depth_vis).save(depth_vis_file)
            
            # Save point cloud (already scaled)
            if not args.no_pc:
                camera_pc_dir = os.path.join(args.output_dir, "point_clouds", camera_name)
                os.makedirs(camera_pc_dir, exist_ok=True)
                
                points, colors = depth_to_point_cloud(depth, intrinsics, extrinsics, image)
                glb_file = os.path.join(camera_pc_dir, f"{base_name}_pc.glb")
                save_point_cloud_glb(points, colors, glb_file, conf_mask=None)
                print(f"  Saved point cloud for {camera_name}: {glb_file} ({len(points)} points)")

    print(f"\nProcessing complete! Results saved to: {args.output_dir}")