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


def process_image_sequence(
    image_paths,
    output_dir,
    model_name="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    batch_size=1,
    save_depth_maps=True,
    save_point_clouds=True,
    conf_threshold_percentile=40.0,
    device="cuda"
):
    """
    Process a sequence of images and save depth maps and point clouds per image.
    
    Args:
        image_paths: List of input images (numpy arrays, PIL Images, or file paths)
        output_dir: Directory to save outputs
        model_name: DA3 model name or HuggingFace path
        batch_size: Number of images to process at once
        save_depth_maps: Whether to save depth maps
        save_point_clouds: Whether to save point clouds
        conf_threshold_percentile: Confidence threshold for filtering point clouds
        device: Device to run inference on
    """
    # Clear existing outputs and create output directories
    if save_depth_maps:
        depth_maps_dir = os.path.join(output_dir, "depth_maps")
        if os.path.exists(depth_maps_dir):
            shutil.rmtree(depth_maps_dir)
        os.makedirs(depth_maps_dir, exist_ok=True)
    if save_point_clouds:
        point_clouds_dir = os.path.join(output_dir, "point_clouds")
        if os.path.exists(point_clouds_dir):
            shutil.rmtree(point_clouds_dir)
        os.makedirs(point_clouds_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {model_name}")
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    model.eval()
    
    # Process images in batches
    num_images = len(image_paths)
    print(f"Processing {num_images} images in batches of {batch_size}")
    
    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        batch_paths = image_paths[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(num_images-1)//batch_size + 1}")
        print(f"Images {batch_start} to {batch_end-1}")
        
        # Run inference
        prediction = model.inference(
            image=batch_paths,
            export_dir=None,  # We'll handle saving ourselves
        )
        
        # Process each image in the batch
        for i, (img_path, idx) in enumerate(zip(batch_paths, range(batch_start, batch_end))):
            # Get outputs for this image
            depth = prediction.depth[i]  # [H, W]
            intrinsics = prediction.intrinsics[i]  # [3, 3]
            extrinsics = prediction.extrinsics[i]  # [3, 4]
            image = prediction.processed_images[i]  # [H, W, 3] uint8
            conf = prediction.conf[i] if prediction.conf is not None else None  # [H, W]
            
            # Get base filename
            base_name = Path(img_path).stem
            
            # Save depth map
            if save_depth_maps:
                depth_file = os.path.join(output_dir, "depth_maps", f"{base_name}_depth.npy")
                np.save(depth_file, depth)
                print(f"  Saved depth map: {depth_file}")
                
                # Also save as visualization
                from depth_anything_3.utils.visualize import visualize_depth
                depth_vis = visualize_depth(depth)
                depth_vis_file = os.path.join(output_dir, "depth_maps", f"{base_name}_depth_vis.jpg")
                Image.fromarray(depth_vis).save(depth_vis_file)
            
            # Save point cloud
            if save_point_clouds:
                # Convert depth to point cloud
                points, colors = depth_to_point_cloud(depth, intrinsics, extrinsics, image)
                
                # Save raw point cloud as GLB (all points, no filtering)
                glb_file = os.path.join(output_dir, "point_clouds", f"{base_name}_pc.glb")
                save_point_cloud_glb(points, colors, glb_file, conf_mask=None)
                print(f"  Saved raw point cloud: {glb_file} (points: {len(points)})")
    
    print(f"\nProcessing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Process image sequence with DA3")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save outputs")
    parser.add_argument("--model", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                       help="DA3 model name or HuggingFace path")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Number of images to process at once")
    parser.add_argument("--pattern", type=str, default="*.jpg",
                       help="Image file pattern (e.g., '*.jpg', '*.png')")
    parser.add_argument("--no-depth", action="store_true",
                       help="Don't save depth maps")
    parser.add_argument("--no-pc", action="store_true",
                       help="Don't save point clouds")
    parser.add_argument("--conf_thresh", type=float, default=0.0,
                       help="Confidence threshold percentile for point cloud filtering")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on (cuda/cpu)")
    
    args = parser.parse_args()
    
    # # Get image paths - search for common image formats
    # image_paths = []
    # for pattern in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
    #     image_paths.extend(glob.glob(os.path.join(args.input_dir, pattern)))
    # print("Image paths: ", image_paths)
    
    # # Remove duplicates and sort
    # image_paths = sorted(list(set(image_paths)))
    
    # if not image_paths:
    #     print(f"No images found in {args.input_dir}")
    #     exit(1)
    
    # Get per view image paths
    static1_paths = []
    static2_paths = []
    dyamic_paths = []

    static1_paths.extend(glob.glob(os.path.join(args.input_dir, "exterior_image_1_left/images", "*.png")))
    static2_paths.extend(glob.glob(os.path.join(args.input_dir, "exterior_image_2_left/images", "*.png")))
    dyamic_paths.extend(glob.glob(os.path.join(args.input_dir, "wrist_image_left/images", "*.png")))

    static1_paths = sorted(list(set(static1_paths)))
    static2_paths = sorted(list(set(static2_paths)))
    dyamic_paths = sorted(list(set(dyamic_paths)))
    
    print(f"Found {len(static1_paths)} images")
    print(f"Found {len(static2_paths)} images")
    print(f"Found {len(dyamic_paths)} images")

    # print(static1_paths)
    # print(static2_paths)
    # print(dyamic_paths)

    # exit()
    
    # # Process sequence
    # process_image_sequence(
    #     image_paths=image_paths,
    #     output_dir=args.output_dir,
    #     model_name=args.model,
    #     batch_size=args.batch_size,
    #     save_depth_maps=not args.no_depth,
    #     save_point_clouds=not args.no_pc,
    #     conf_threshold_percentile=args.conf_thresh,
    #     device=args.device
    # )

    # Process sequence as multi-view
    for i in range(len(static1_paths)):
        image_paths = [static1_paths[i], static2_paths[i], dyamic_paths[i]]

        process_image_sequence(
            image_paths=image_paths,
            output_dir=args.output_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            save_depth_maps=not args.no_depth,
            save_point_clouds=not args.no_pc,
            conf_threshold_percentile=args.conf_thresh,
            device=args.device
        )

    # Usage:
    # python process_sequence.py --input_dir "/media/bora/Extreme Pro/new_proj/episodes/episode_000000" --output_dir "/media/bora/Extreme Pro/DA3/outputs/da3_1.1/episode_000004"


