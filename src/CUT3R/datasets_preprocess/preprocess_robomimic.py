# Preprocess Robomimic ground truth and convert to DL3DV_Multi

import argparse
import os
import sys
import numpy as np
import cv2
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    # parser.add_argument("--num_views", type=int, default=2, required=True)
    return parser.parse_args()


#############################################################################################
### Helper functions to load robomimic ground truths and save them in DL3DV_Multi format. ###
#############################################################################################

def convert_intrinsics_and_poses(input_path, output_path):
    """Load robomimic intrinsics and poses from input path and save them to output path.
    """
    dynamic_poses_path = os.path.join(input_path, "extrinsics", "robot0_eye_in_hand", "demo_0")
    dynamic_poses_list = sorted(os.listdir(dynamic_poses_path))

    # These are individual files since they are constant at every frame.
    dynamic_intrinsic_path = os.path.join(input_path, "intrinsics", "robot0_eye_in_hand_intrinsics.npy")
    static_intrinsic_path = os.path.join(input_path, "intrinsics", "agentview_intrinsics.npy")
    static_pose_path = os.path.join(input_path, "extrinsics", "agentview_extrinsics.npy")

    cam_output_path = os.path.join(output_path, "dense", "cam")
    os.makedirs(cam_output_path, exist_ok=True)

    # Load the single files once here instead of repeatedly in the loop
    dynamic_intrinsic = np.load(dynamic_intrinsic_path)
    static_intrinsic = np.load(static_intrinsic_path)
    static_pose = np.load(static_pose_path)

    # Save intrinsics and poses to a single npz, interleave them by input view.
    print(len(dynamic_poses_list))
    for t, fname in enumerate(dynamic_poses_list):
        dynamic_pose = np.load(os.path.join(dynamic_poses_path, fname))
        np.savez(os.path.join(cam_output_path, f"{2*t:06d}.npz"),
                intrinsic=static_intrinsic, pose=static_pose)
        print(f"Saved static pose to {2*t:06d}.npz")
        np.savez(os.path.join(cam_output_path, f"{2*t+1:06d}.npz"),
                intrinsic=dynamic_intrinsic, pose=dynamic_pose)
        print(f"Saved dynamic pose to {2*t+1:06d}.npz")

    print(f"Saved intrinsics and poses to {cam_output_path}")


def convert_depths(input_path, output_path):
    """Load robomimic depths from input path and save them to output path.
    """
    depths_path = os.path.join(input_path, "depth", "interleaved_depths")
    depths_list = sorted(os.listdir(depths_path))

    # Since depths are already saved as npy, just copy them to output path but interleave them
    # as static, dynamic, repeat.
    depths_output_path = os.path.join(output_path, "dense", "depth")
    os.makedirs(depths_output_path, exist_ok=True)

    for i in range(len(depths_list)):
        shutil.copy2(os.path.join(depths_path, depths_list[i]), depths_output_path)
    
    print(f"Saved depths to {depths_output_path}")
    
    return 0


def convert_rgb(input_path, output_path):
    """Loads robomimic rgb from path.
    """
    static_path = os.path.join(input_path, "video", "agentview")
    dynamic_path = os.path.join(input_path, "video", "robot0_eye_in_hand")

    static_rgb_vid_path = os.path.join(static_path, "demo_0_rgb.mp4")
    dynamic_rgb_vid_path = os.path.join(dynamic_path, "demo_0_rgb.mp4")

    rgb_output_path = os.path.join(output_path, "dense", "rgb")
    os.makedirs(rgb_output_path, exist_ok=True)

    static_cap = cv2.VideoCapture(static_rgb_vid_path)
    dynamic_cap = cv2.VideoCapture(dynamic_rgb_vid_path)

    if not static_cap.isOpened():
        raise RuntimeError(f"Could not open static video: {static_rgb_vid_path}")
    if not dynamic_cap.isOpened():
        raise RuntimeError(f"Could not open dynamic video: {dynamic_rgb_vid_path}")

    t = 0
    while True:
        ok_s, frame_s = static_cap.read()
        ok_d, frame_d = dynamic_cap.read()

        # stop cleanly when either stream ends
        if not ok_s or not ok_d:
            break

        # interleave by timestep: even=static, odd=dynamic
        cv2.imwrite(os.path.join(rgb_output_path, f"{2*t:06d}.png"), frame_s)
        cv2.imwrite(os.path.join(rgb_output_path, f"{2*t+1:06d}.png"), frame_d)
        t += 1

    static_cap.release()
    dynamic_cap.release()

    print(f"Saved {2*t} RGB frames to {rgb_output_path}")
    return 0


def create_masks(length, output_path):
    """Create blank masks, all zeros 512x512.
    """
    sky_mask_output_path = os.path.join(output_path, "dense", "sky_mask")
    outlier_mask_output_path = os.path.join(output_path, "dense", "outlier_mask")
    os.makedirs(sky_mask_output_path, exist_ok=True)
    os.makedirs(outlier_mask_output_path, exist_ok=True)
    img = np.zeros((512, 512), dtype=np.uint8)
    for i in range(length):
        cv2.imwrite(os.path.join(sky_mask_output_path, f"{i:06d}.png"), img)
        cv2.imwrite(os.path.join(outlier_mask_output_path, f"{i:06d}.png"), img)
    print(f"Saved {len(os.listdir(sky_mask_output_path))} masks to {sky_mask_output_path}")
    print(f"Saved {len(os.listdir(outlier_mask_output_path))} masks to {outlier_mask_output_path}")
    return 0

def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load intrinsics, poses, depths, and rgb from input_dir and save them in DL3DV_Multi format.
    # Note: current implementation assumes that input_dir contains ground truths for a single scene.
    convert_intrinsics_and_poses(input_dir, output_dir)
    convert_depths(input_dir, output_dir)
    convert_rgb(input_dir, output_dir)
    create_masks(118, output_dir)
    
    print(f"Saved Robomimic ground truths in DL3DV_Multi format to {output_dir}")

if __name__ == "__main__":
    main()