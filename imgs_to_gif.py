from PIL import Image
import os
import glob

def create_gif_pillow(image_dir, output_path, duration=100, loop=0):
    """
    Create a GIF using PIL/Pillow.
    
    Args:
        image_dir: Directory containing the images
        output_path: Path to save the output GIF
        duration: Time per frame in milliseconds
        loop: Number of loops (0 = infinite)
    """
    # Get all image files sorted
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')) + 
                        glob.glob(os.path.join(image_dir, '*.png')))
    
    # Open all images
    images = [Image.open(img) for img in image_files]
    
    # Save as GIF
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop
        )
        print(f"GIF created: {output_path}")

# Example usage
create_gif_pillow(
    image_dir='/media/bora/Extreme Pro/DA3/outputs/da3/episode_000000/depth_maps',
    output_path='/media/bora/Extreme Pro/DA3/outputs/da3/episode_000000/depth_animation.gif',
    duration=100  # 100ms per frame
)