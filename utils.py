# Utility functions for DA3/CUT3R

import numpy as np


def homogenize_poses(pose):
    """Convert poses from (..., 3, 4) to (..., 4, 4)."""
    homo_poses = []
    for view_idx in range(pose.shape[0]):
        print(f"Homogenizing view {view_idx}")
        bottom = np.array([[0, 0, 0, 1]], dtype=pose[view_idx].dtype)
        homo_pose = np.vstack([pose[view_idx], bottom])
        homo_poses.append(homo_pose)
    homo_poses = np.array(homo_poses)
    return homo_poses


def invert_poses(poses):
    """Rigid inverse for batched camera transforms."""
    # This method assumes poses is already homogenized!
    inv_poses = []
    for view_idx in range(poses.shape[0]):
        print(f"Inverting view {view_idx}")
        inv_pose = np.linalg.inv(poses[view_idx])
        inv_poses.append(inv_pose)
    inv_poses = np.array(inv_poses)
    return inv_poses


def make_relative_poses(abs_pose_ref, abs_pose_tgt):
    """
    Align target pose(s) to the reference pose.
    Assumes all inputs are already homogeneous 4x4 transforms.

    Args:
        abs_pose_ref: Reference pose, shape (4,4).
        abs_pose_tgt: Target pose(s), shape (4,4) or (N,4,4).

    Returns:
        Relative/aligned pose(s) in homogeneous form.
    """
    ref = np.asarray(abs_pose_ref)
    tgt = np.asarray(abs_pose_tgt)
    if ref.shape != (4, 4):
        raise ValueError(f"Expected abs_pose_ref shape (4,4), got {ref.shape}")
    ref_inv = np.linalg.inv(ref)

    # Single-target relative transform
    if tgt.ndim == 2:
        if tgt.shape != (4, 4):
            raise ValueError(f"Expected abs_pose_tgt shape (4,4), got {tgt.shape}")
        return tgt @ ref_inv

    # Batched-target alignment: use view-0 of target chunk as static anchor
    if tgt.ndim != 3 or tgt.shape[-2:] != (4, 4):
        raise ValueError(f"Expected abs_pose_tgt shape (N,4,4), got {tgt.shape}")
    tgt_static = tgt[0]
    # Find right-mult transform X so tgt_static @ X == ref
    X = np.linalg.inv(tgt_static) @ ref
    return tgt @ X
