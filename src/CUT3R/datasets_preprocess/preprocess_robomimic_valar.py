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
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=(
            "Comma-separated subset to write (default: all). "
            "Tokens: depth, rgb, intrinsics, extrinsics, cam, masks. "
            "intrinsics and/or extrinsics (or cam) write dense/cam/*.npz; "
            "depth writes dense/depth; rgb writes dense/rgb; masks writes sky_mask and outlier_mask."
        ),
    )
    return parser.parse_args()


def parse_save_flags(only: str | None):
    """Returns which converters to run. If only is None/empty, save everything."""
    allowed = {"depth", "rgb", "intrinsics", "extrinsics", "cam", "masks"}
    if only is None or not str(only).strip():
        return {"depth": True, "rgb": True, "cam": True, "masks": True}
    raw = {x.strip().lower() for x in only.split(",") if x.strip()}
    unknown = raw - allowed
    if unknown:
        raise ValueError(
            f"Unknown --only token(s): {sorted(unknown)}. Allowed: {sorted(allowed)}"
        )
    return {
        "depth": "depth" in raw,
        "rgb": "rgb" in raw,
        "cam": bool(raw & {"intrinsics", "extrinsics", "cam"}),
        "masks": "masks" in raw,
    }


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
    # interleave camera paths (transport uses a 4-camera round-robin per timestep).
    for data_type in demo_contents_paths.keys():
        lists = demo_contents_paths[data_type]
        if task_dir == "transport":
            demo_contents_paths[data_type] = interleave_transport_data_contents(lists)
        else:
            demo_contents_paths[data_type] = interleave_data_contents(lists)

    # One interleaved index == one exported view (same length for every modality).
    num_frames = len(demo_contents_paths["depth"])

    # Return a dictionary with per-data type lists of interleaved .npy's
    # e.g.
    # for demo_0: {'depth': all files, 'extrinsics': all files, 'intrinsics': all files', 'rgb': all files}
    
    return demo_contents_paths, num_frames


def _find_camera_list_index(data_paths, must_contain: str) -> int:
    for idx, path_list in enumerate(data_paths):
        if path_list and must_contain in path_list[0]:
            return idx
    raise ValueError(f"No camera list containing {must_contain!r} (example paths: {data_paths})")


def interleave_transport_data_contents(data_paths):
    """Per timestep: shouldercamera0 -> robot0_eye_in_hand -> shouldercamera1 -> robot1_eye_in_hand."""
    sc0 = _find_camera_list_index(data_paths, "shouldercamera0")
    r0 = _find_camera_list_index(data_paths, "robot0_eye_in_hand")
    sc1 = _find_camera_list_index(data_paths, "shouldercamera1")
    r1 = _find_camera_list_index(data_paths, "robot1_eye_in_hand")
    cam_order = (sc0, r0, sc1, r1)
    n = len(data_paths[sc0])
    for idx in cam_order:
        if len(data_paths[idx]) != n:
            raise ValueError(
                f"Transport camera length mismatch: cam {idx} has {len(data_paths[idx])}, expected {n}"
            )
    interleaved = []
    for i in range(n):
        for idx in cam_order:
            interleaved.append(data_paths[idx][i])
    return interleaved


def interleave_data_contents(data_paths):
    """Helper function to interleave the data paths such that all the data is in a single collapsed directory
    where the files alternate between static and dynamic camera ground truth (static always comes first).
    """
    static_index, dynamic_index = -1, -1

    if "agentview" in data_paths[0][0]:
        static_index = 0
        dynamic_index = 1
    elif "sideview" in data_paths[1][0]:
        static_index = 1
        dynamic_index = 0
    else:
        raise ValueError(f"Invalid data paths configuration, found paths: {data_paths}")

    interleaved_data_paths = []
    for i in range(len(data_paths[0])):
        interleaved_data_paths.append(data_paths[static_index][i])
        interleaved_data_paths.append(data_paths[dynamic_index][i])

    return interleaved_data_paths


#############################################################################################
### Helper functions to load robomimic ground truths and save them in DL3DV_Multi format. ###
#############################################################################################

def convert_intrinsics_and_poses(intrinsics_path, poses_path, output_path):
    """Load robomimic intrinsics and poses from input path and save them to output path.

    output_path: scene root (…/task/demo_*) — writes under output_path/dense/cam/.
    """
    print("Saving intrinsics and poses to DL3DV_Multi format...")
    output_path = os.path.join(output_path, "dense", "cam")
    os.makedirs(output_path, exist_ok=True)

    # Save intrinsics and poses to a single npz. Note: they are already interleaved and we have full filepaths
    for i, (intrinsic_path, pose_path) in enumerate(zip(intrinsics_path, poses_path)):
        # print(f"Loading intrinsic from {intrinsic_path} and pose from {pose_path}")
        intrinsic = np.load(intrinsic_path)
        pose = np.load(pose_path)
        np.savez(os.path.join(output_path, f"{i:06d}.npz"),
                intrinsic=intrinsic, pose=pose)
        # if i % 50 == 0:
            # print(f"Saved intrinsics and poses to {os.path.join(output_path, f'{i:06d}.npz')}")
    print("Saved {} intrinsics and poses to {}".format(i+1, output_path))
    return 0


def convert_depths(input_path, output_path):
    """Load robomimic depths from input path and save them to output path.

    output_path: scene root (…/task/demo_*) — writes under output_path/dense/depth/.
    """
    # Since depths are already saved as npy, just copy them to output path.
    print("Saving depths to DL3DV_Multi format...")
    output_path = os.path.join(output_path, "dense", "depth")
    os.makedirs(output_path, exist_ok=True)

    for i, src_path in enumerate(input_path):
        dst = os.path.join(output_path, f"{i:06d}.npy")
        shutil.copy2(src_path, dst)
        # if i % 50 == 0:
            # print(f"saved depth from {src_path} to {dst}")
    print("Saved {} depths to {}".format(i+1, output_path))
    return 0


def convert_rgb(input_path, output_path):
    """Loads robomimic rgb from path.

    output_path: scene root (…/task/demo_*) — writes under output_path/dense/rgb/.
    Matches preprocess_robomimic.py: cv2.imwrite expects BGR (same as VideoCapture frames).
    Extracted .npy observations are RGB (H, W, 3); convert before writing.
    """
    print("Saving RGB frames to DL3DV_Multi format...")
    output_path = os.path.join(output_path, "dense", "rgb")
    os.makedirs(output_path, exist_ok=True)

    for i, src_path in enumerate(input_path):
        img = np.load(src_path)
        if img.dtype != np.uint8:
            if float(img.max()) <= 1.0:
                img = (np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0) * 255.0).astype(
                    np.uint8
                )
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, f"{i:06d}.png"), img)
        # if i % 50 == 0:
            # print(f"saved RGB frame from {src_path} to {os.path.join(output_path, f'{i:06d}.png')}")
    print("Saved {} RGB frames to {}".format(i+1, output_path))
    return 0


def create_masks(length, output_path, res):
    """Create blank masks, all zeros resxres.

    output_path: scene root (…/task/demo_*) — writes under output_path/dense/{sky_mask,outlier_mask}/.
    """
    print("Creating masks for DL3DV_Multi format...")
    sky_mask_output_path = os.path.join(output_path, "dense", "sky_mask")
    outlier_mask_output_path = os.path.join(output_path, "dense", "outlier_mask")
    os.makedirs(sky_mask_output_path, exist_ok=True)
    os.makedirs(outlier_mask_output_path, exist_ok=True)

    img = np.zeros((res, res), dtype=np.uint8)
    for i in range(length):
        cv2.imwrite(os.path.join(sky_mask_output_path, f"{i:06d}.png"), img)
        cv2.imwrite(os.path.join(outlier_mask_output_path, f"{i:06d}.png"), img)
    print(f"Saved {len(os.listdir(sky_mask_output_path))} masks to {sky_mask_output_path}")
    print(f"Saved {len(os.listdir(outlier_mask_output_path))} masks to {outlier_mask_output_path}")
    return 0


def main_valar(input_dir, output_dir, tasks, save_flags, mask_res):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing robomimic dataset from {input_dir} and saving to {output_dir}")
    print(f"Save flags: {save_flags}")

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

            # Scene root: …/output_dir/<task>/<demo> (DL3DV layout: dense/ lives under this)
            output_scene_dir = os.path.join(output_dir, task_dir, demo_folder)
            dense_dir = os.path.join(output_scene_dir, "dense")
            print(f"Creating output scene directory: {output_scene_dir}")
            os.makedirs(dense_dir, exist_ok=True)

            if save_flags["cam"]:
                convert_intrinsics_and_poses(
                    data_types["intrinsics"], data_types["extrinsics"], output_scene_dir
                )
            if save_flags["depth"]:
                convert_depths(data_types["depth"], output_scene_dir)
            if save_flags["rgb"]:
                convert_rgb(data_types["rgb"], output_scene_dir)
            if save_flags["masks"]:
                create_masks(len(data_types["depth"]), output_scene_dir, mask_res)

    print(f"Saved DL3DV_Multi format robomimic ground truths to {output_dir}")
    return 0

if __name__ == "__main__":
    args = parse_args()
    save_flags = parse_save_flags(args.only)
    main_valar(
        args.input_dir,
        args.output_dir,
        args.tasks,
        save_flags,
        args.mask_res,
    )
