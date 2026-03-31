# Preprocess Robomimic ground truth and convert to DL3DV_Multi on VALAR
# WARNING: some parameters are hard-coded, this file was designed to be run from:
#       /scratch/bdursun25/cuteanything/my-da3
# with:
#       python src/CUT3R/datasets_preprocess/preprocess_robomimic_valar.py \
#       --input_dir "scratch/bdursun25/robomimic_dataset" \
#       --output_dir "scratch/bdursun25/cuteanything/my-da3/scenes/robomimic_full/dl3dv_multi"

import argparse
import os
import sys
import numpy as np
import cv2
import shutil
import json
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mask_res", type=int, default=512)
    parser.add_argument("--tasks", type=str, default="can,lift,square,tool_hang,transport")
    return parser.parse_args()


#################################################################################################
### Helper functions for loading the entire robomimic dataset on VALAR in DL3DV_Multi format. ###
#################################################################################################

def load_all_robomimic_data(dataset_base_dir, output_dir, tasks):
    """Loads robomimic demo folders, assuming the default format after unpacking the hdf5 files.
    """
    # Unpacked data format is like this:
    # task_name/ph/extracted_npy/demo_*
    # (ph is 'proficient human')

    # First we get the task directories from the arg tasks_base_dir: can, lift, square, tool_hang, transport
    # From each task dir, we need to step through the 'ph' and 'extracted_npy' subdirectories.
    # From there we can collect the individual demo folders.

    # Count the total number of frames in the dataset
    total_frames = 0

    # can, lift, square, tool_hang, transport
    task_dirs = [d for d in sorted(os.listdir(dataset_base_dir))]
    print(f"Found tasks: {task_dirs}")
    task_paths = [os.path.join(dataset_base_dir, task_dir) for task_dir in task_dirs]

    # Store all data in this dictionary as:
    # {task_dir: {demo_folder: {data_type: [path1, path2, ...]}}}
    master_dict = {'can': {}, 'lift': {}, 'square': {}, 'tool_hang': {}, 'transport': {}} # keys are task dirs, values are demo folders

    for i, task_path in enumerate(task_paths):

        if task_dirs[i] not in tasks:
            print(f"Skipping {task_dirs[i]} not in tasks: {tasks}")
            continue
        print(f"Processing {task_dirs[i]}")

        demos_base_dir = os.path.join(task_path, "ph", "extracted_npy")
        demo_folders = sorted(os.listdir(demos_base_dir))
        for demo_folder in tqdm(demo_folders, desc=f"{task_dirs[i]} demos", leave=True):
            demo_path = os.path.join(demos_base_dir, demo_folder)

            # Call the helper which reads demo folders
            # data_paths[0] <--> depths
            # data_paths[1] <--> extrinsics
            # data_paths[2] <--> intrinsics
            # data_paths[3] <--> rgb

            # When doing 'transport', read_demo will return a interleaved demo contents paths as a list of length 2
            # where each list is for a separate camera pairing for the transport task.
            # if 'transport' in task_dirs[i]:
            #     interleaved_demo_contents_paths, num_frames = read_demo(demo_path, task_dirs[i])
            #     total_frames += num_frames
            #     master_dict['transport0'][demo_folder] = interleaved_demo_contents_paths
            #     master_dict['transport1'][demo_folder] = interleaved_demo_contents_paths
            
            interleaved_demo_contents_paths, num_frames = read_demo(demo_path, task_dirs[i])
            total_frames += num_frames
            master_dict[task_dirs[i]][demo_folder] = interleaved_demo_contents_paths

    return master_dict, total_frames

    
def read_demo(demo_base_path, task_dir):
    """Returns paths to the contents of a demo folder. Reduces clutter in load_all_robomimic_data.
    """
    # Each demo folder contains the following subdirectories:
    # Separate folders for each camera, e.g. agentview, robot0_eye_in_hand
    # Each camera folder contains depth, extrinsics, intrinsics, rgb

    camera_folders = sorted(os.listdir(demo_base_path))
    camera_folder_paths = [os.path.join(demo_base_path, camera_folder) for camera_folder in camera_folders]

    # Build the master list, where each element of the list is a list containing the paths to each .npy file
    demo_contents_paths = {'depth': [], 'extrinsics': [], 'intrinsics': [], 'rgb': []}
    # transport_second_half = {'depth': [], 'extrinsics': [], 'intrinsics': [], 'rgb': []}
    
    for camera_folder_path in camera_folder_paths:
        data_subdirs = sorted(os.listdir(camera_folder_path))
        assert len(data_subdirs) == 4, f"Demo's per-camera folders should contain 4 subdirectories, found {len(data_subdirs)}"
        
        # Now get the actual data for depth, extrinsics, intrinsics, rgb
        for data_subdir in data_subdirs:
            data_subdir_path = os.path.join(camera_folder_path, data_subdir)
            data_files = sorted(os.listdir(data_subdir_path))
            data_files_paths = [os.path.join(data_subdir_path, data_file) for data_file in data_files]
            demo_contents_paths[data_subdir].append(data_files_paths)

    # For each data type; depth, extrinsics, intrinsics, rgb,
    # interleave the static and dynamic camera paths and count the total frames
    num_frames = 0
    for data_type in demo_contents_paths.keys():
        
        # Length of demo_contents_paths[data_type] is the number of cameras
        demo_contents_paths[data_type] = interleave_data_contents(demo_contents_paths[data_type])
        num_frames += len(demo_contents_paths[data_type])

    # Return a dictionary with per-data type lists of interleaved .npy's
    # e.g.
    # for demo_0: {'depth': all files, 'extrinsics': all files, 'intrinsics': all files', 'rgb': all files}
    
    return demo_contents_paths, num_frames


def interleave_data_contents(data_paths):
    """Helper function to interleave the data paths such that all the data is in a single collapsed directory
    where the files alternate between static and dynamic camera ground truth (static always comes first).
    """
    # Use these to index which list is for the static and dynamic camera
    static_index, dynamic_index = -1, -1

    if 'agentview' in data_paths[0][0]:
        static_index = 0
        dynamic_index = 1
        dynamic_path = data_paths[1]
    elif 'sideview' in data_paths[1][0]:
        static_index = 1
        dynamic_index = 0
    elif 'transport' in data_paths[0][0]:
        static_index = 2    # shouldercamera0
        dynamic_index = 0   # robot0_eye_in_hand
    else:
        raise ValueError(f"Invalid data paths configuration, found paths: {data_paths}")

    # Interleave the data paths and return
    interleaved_data_paths = []
    transport_second_half = []

    for i in range(len(data_paths[0])):
        interleaved_data_paths.append(data_paths[static_index][i])
        interleaved_data_paths.append(data_paths[dynamic_index][i])

        # transport has four cameras:
        # robot0_eye_in_hand, robot1_eye_in_hand, shouldercamera0, shouldercamera1
        # First do both 0-numbered cams, and then do the 1-numbered cams here
        if 'transport' in data_paths[0][0]:
            transport_second_half.append(data_paths[static_index+1][i])
            transport_second_half.append(data_paths[dynamic_index+1][i])

    return interleaved_data_paths, transport_second_half


#############################################################################################
### Helper functions to load robomimic ground truths and save them in DL3DV_Multi format. ###
#############################################################################################

def convert_intrinsics_and_poses(intrinsics_path, poses_path, output_path):
    """Load robomimic intrinsics and poses from input path and save them to output path.
    """
    print("Saving intrinsics and poses to DL3DV_Multi format...")
    output_path = os.path.join(output_path, "cam")
    os.makedirs(output_path, exist_ok=True)

    # Save intrinsics and poses to a single npz. Note: they are already interleaved and we have full filepaths
    for i, (intrinsic_path, pose_path) in enumerate(zip(intrinsics_path, poses_path)):
        # print(f"Loading intrinsic from {intrinsic_path} and pose from {pose_path}")
        intrinsic = np.load(intrinsic_path)
        pose = np.load(pose_path)
        np.savez(os.path.join(output_path, f"{i:06d}.npz"),
                intrinsic=intrinsic, pose=pose)
        if i % 50 == 0:
            print(f"Saved intrinsics and poses to {os.path.join(output_path, f'{i:06d}.npz')}")
    print("Saved {} intrinsics and poses to {}".format(i+1, output_path))
    return 0


def convert_depths(input_path, output_path):
    """Load robomimic depths from input path and save them to output path.
    """
    # Since depths are already saved as npy, just copy them to output path.
    print("Saving depths to DL3DV_Multi format...")
    output_path = os.path.join(output_path, "depth")
    os.makedirs(output_path, exist_ok=True)

    for i, src_path in enumerate(input_path):
        shutil.copy2(src_path, output_path)
        if i % 50 == 0:
            print(f"saved depth from {src_path} to {os.path.join(output_path, f'{i:06d}.npy')}")
    print("Saved {} depths to {}".format(i+1, output_path))
    return 0


def convert_rgb(input_path, output_path):
    """Loads robomimic rgb from path.
    """
    print("Saving RGB frames to DL3DV_Multi format...")
    output_path = os.path.join(output_path, "rgb")
    os.makedirs(output_path, exist_ok=True)

    # Convert from .npy to .png and save only the .png to the output path
    for i, src_path in enumerate(input_path):
        rgb = np.load(src_path)
        cv2.imwrite(os.path.join(output_path, f"{i:06d}.png"), rgb)
        if i % 50 == 0:
            print(f"saved RGB frame from {src_path} to {os.path.join(output_path, f'{i:06d}.png')}")
    print("Saved {} RGB frames to {}".format(i+1, output_path))
    return 0


def create_masks(length, output_path, res):
    """Create blank masks, all zeros resxres.
    """
    print("Creating masks for DL3DV_Multi format...")
    sky_mask_output_path = os.path.join(output_path, "sky_mask")
    outlier_mask_output_path = os.path.join(output_path, "outlier_mask")
    os.makedirs(sky_mask_output_path, exist_ok=True)
    os.makedirs(outlier_mask_output_path, exist_ok=True)

    img = np.zeros((res, res), dtype=np.uint8)
    for i in range(length):
        cv2.imwrite(os.path.join(sky_mask_output_path, f"{i:06d}.png"), img)
        cv2.imwrite(os.path.join(outlier_mask_output_path, f"{i:06d}.png"), img)
    print(f"Saved {len(os.listdir(sky_mask_output_path))} masks to {sky_mask_output_path}")
    print(f"Saved {len(os.listdir(outlier_mask_output_path))} masks to {outlier_mask_output_path}")
    return 0


def main_valar(input_dir, output_dir, tasks):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing robomimic dataset from {input_dir} and saving to {output_dir}")

    # Load all robomimic data from input_dir and save them in Valar format.
    # Returns a giant dictionary:
    # {task_dir: {demo_folder: {data_type: [path1, path2, ...]}}}
    master_dict, total_frames = load_all_robomimic_data(input_dir, output_dir, tasks)

    # Verification
    for task_dir, demo_folders in master_dict.items():
        print(f"*************** {task_dir} ***************")
        print(f"Length of demo_folders: {len(demo_folders)}")
        for demo_folder, data_types in demo_folders.items():
            print(f"    {demo_folder}:", end = "")
            for data_type, data_paths in data_types.items():
                print(f"        {data_type} - {len(data_paths)}", end = "") # , example paths: [{data_paths[0]}, {data_paths[1]}, ... {data_paths[-2]}, {data_paths[-1]}]")
            print()

    print(f"Total frames in robomimic dataset: {total_frames}")
    exit()

    # Load the data to DL3DV_Multi format
    for task_dir, demo_folders in master_dict.items():
        for demo_folder, data_types in demo_folders.items():
            # Verification
            assert (
                len(data_types['depth']) == 
                len(data_types['intrinsics']) == 
                len(data_types['extrinsics']) == 
                len(data_types['rgb'])
            ), (
                f"Mismatched lengths: depth({len(data_types['depth'])}), "
                f"intrinsics({len(data_types['intrinsics'])}), "
                f"extrinsics({len(data_types['extrinsics'])}), "
                f"rgb({len(data_types['rgb'])})"
            )

            # Create the output base directory
            output_base_dir = os.path.join(output_dir, task_dir, demo_folder, "dense")
            print(f"Creating output base directory: {output_base_dir}")
            os.makedirs(output_base_dir, exist_ok=True)

            convert_intrinsics_and_poses(data_types['intrinsics'], data_types['extrinsics'], output_base_dir)
            convert_depths(data_types['depth'], output_base_dir)
            convert_rgb(data_types['rgb'], output_base_dir)
            create_masks(len(data_types['depth']), output_base_dir, 512)

    print(f"Saved DL3DV_Multi format robomimic ground truths to {output_dir}")
    return 0

if __name__ == "__main__":
    args = parse_args()
    main_valar(
        args.input_dir,
        args.output_dir,
        args.tasks,
    )
