import glob
import os
import torch
import numpy as np
import imageio
from pathlib import Path
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth
from process_sequence import process_image_sequence

if __name__ == "__main__":
    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    model = model.to(device=device)
    model.eval()

    # File paths EDIT THESE
    input_path = "/media/bora/Extreme Pro/new_proj/episodes/episode_000000/wrist_image_left/images"
    output_dir = "outputs/da3_1.1/episode_000000"

    # Create depth_vis directory
    depth_vis_dir = os.path.join(output_dir, "depth_vis")
    os.makedirs(depth_vis_dir, exist_ok=True)

    # Get all images
    images = sorted(glob.glob(os.path.join(input_path, "*.png")))
    print(f"Processing {len(images)} images")

    # Process each image individually
    for idx, image_path in enumerate(images):
        print(f"\nProcessing {idx+1}/{len(images)}: {os.path.basename(image_path)}")
        
        # Wrap image path in a list - API requires list
        # prediction = model.inference(
        #     image=[image_path],
        #     export_dir=output_dir,
        #     export_format="glb",
        # )
        
        process_image_sequence(
            image_paths=image_path,
            output_dir=output_dir,
            model_name=model,
            batch_size=1,
            save_depth_maps=False,
            save_point_clouds=True,
            conf_threshold_percentile=0.0,
            device="cuda",
        )

        # Print shapes for the first image
        if idx == 0:
            print(f"  processed_images shape: {prediction.processed_images.shape}")
            print(f"  depth shape: {prediction.depth.shape}")
            print(f"  conf shape: {prediction.conf.shape}")
            print(f"  extrinsics shape: {prediction.extrinsics.shape}")
            print(f"  intrinsics shape: {prediction.intrinsics.shape}")


        # Manually save depth visualization with unique filename
        base_name = Path(image_path).stem
        depth = prediction.depth[0]  # Single image, so index 0
        image_vis = prediction.processed_images[0]  # Single image
        
        # Create depth visualization
        depth_vis = visualize_depth(depth)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        image_vis = image_vis.astype(np.uint8)
        
        # Concatenate side-by-side
        vis_image = np.concatenate([image_vis, depth_vis], axis=1)
        
        # Save with unique filename based on image name
        save_path = os.path.join(depth_vis_dir, f"{base_name}_depth_vis.jpg")
        imageio.imwrite(save_path, vis_image, quality=95)
        print(f"  Saved depth visualization: {save_path}")
    
    print(f"\nComplete! Results saved to: {output_dir}")
