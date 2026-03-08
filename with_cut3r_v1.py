"""
Bridging DA3 transformer tokens with CUT3R memory.

1. Follow DA3's inference path up until transformer tokens are outputted.
2. Create an artificial "view" object for DA3 transformer tokens.
3. Modify CUT3R's dimensions to match DA3's.
4. Feed the artificial view objects into CUT3R's encoder and run the model forward.
5. Do this with training enabled. For the dataset, provide a single robomimic scene to overfit to.
"""

from __future__ import annotations
import glob
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any
import argparse
from dataclasses import fields
import numpy as np
import torch
import torch.nn.functional as F

# TODO: fix import paths
from src.depth_anything_3.api import DepthAnything3
from src.CUT3R.src.dust3r.model import ARCroco3DStereo


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Run DA3 tokens through CUT3R memory.")
    parser.add_argument("--input_path", type=str, required=True, help="Input image directory.")
    parser.add_argument("--output_path", type=str, default="outputs/with_cut3r", help="Output directory.")
    parser.add_argument(
        "--da3_model",
        type=str,
        default="depth-anything/DA3-GIANT-1.1",
        help="DA3 model id/path for DepthAnything3.from_pretrained().",
    )
    parser.add_argument(
        "--cut3r_model",
        type=str,
        default="src/CUT3R/src/cut3r_512_dpt_4_64.pth",
        help="CUT3R checkpoint path or HuggingFace model id.",
    )
    parser.add_argument("--da3_size", type=int, default=504, help="DA3 process resolution.")
    parser.add_argument("--cut3r_size", type=int, default=512, help="CUT3R image loading size.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument(
        "--views_per_step",
        type=int,
        default=2,
        help="Number of synchronized views per timestep in the interleaved stream.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train CUT3R with DA3 features.",
    )
    return parser.parse_args()


##############################
### Helper Functions Below ###
##############################

def save_geometry_outputs(outputs, out_dir):
    """
    Saves geometry outputs exactly as CUT3R does. Some modifications were made to account for the fact that
    we cannot supply "views" like the original implementation.
    """
    from src.CUT3R.src.dust3r.utils.camera import pose_encoding_to_camera
    from src.CUT3R.src.dust3r.post_process import estimate_focal_knowing_depth

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/pts3d_self", exist_ok=True)
    os.makedirs(f"{out_dir}/pts3d_other", exist_ok=True)
    os.makedirs(f"{out_dir}/conf", exist_ok=True)
    os.makedirs(f"{out_dir}/depth", exist_ok=True)
    os.makedirs(f"{out_dir}/camera", exist_ok=True)  # <-- camera npz folder

    for i, pred in enumerate(outputs.ress):
        if "pts3d_in_self_view" in pred:
            pself_t = pred["pts3d_in_self_view"]      # [1,H,W,3] tensor
            pself = pself_t.detach().cpu().numpy()
            np.save(f"{out_dir}/pts3d_self/{i:06d}.npy", pself)
            np.save(f"{out_dir}/depth/{i:06d}.npy", pself[..., 2])

            # ----- CUT3R-style intrinsics estimation -----
            B, H, W, _ = pself_t.shape
            pp = torch.tensor([W // 2, H // 2], device=pself_t.device).float().repeat(B, 1)
            focal = estimate_focal_knowing_depth(pself_t, pp, focal_mode="weiszfeld")

            intrinsics = torch.eye(3, device=pself_t.device).unsqueeze(0).repeat(B, 1, 1)
            intrinsics[:, 0, 0] = focal
            intrinsics[:, 1, 1] = focal
            intrinsics[:, 0, 2] = pp[:, 0]
            intrinsics[:, 1, 2] = pp[:, 1]

            # ----- CUT3R-style pose decoding -----
            if "camera_pose" in pred:
                c2w = pose_encoding_to_camera(pred["camera_pose"].clone())  # [1,4,4]
                np.savez(
                    f"{out_dir}/camera/{i:06d}.npz",
                    pose=c2w[0].detach().cpu().numpy(),
                    intrinsics=intrinsics[0].detach().cpu().numpy(),
                )

        if "pts3d_in_other_view" in pred:
            pother = pred["pts3d_in_other_view"].detach().cpu().numpy()
            np.save(f"{out_dir}/pts3d_other/{i:06d}.npy", pother)

        if "conf" in pred:
            np.save(f"{out_dir}/conf/{i:06d}.npy", pred["conf"].detach().cpu().numpy())
        elif "conf_self" in pred:
            np.save(f"{out_dir}/conf/{i:06d}.npy", pred["conf_self"].detach().cpu().numpy())
    
    print(f"Saved geometry outputs to {out_dir}")


# TODO: should true_size and shape use 504 (DA3 resolution) or 512 (CUT3R) resolution?
def get_cut3r_shape(num_views, size: int = 504):
    shape = (torch.tensor([[size, size]], dtype=torch.int32),) * num_views
    return shape


def get_cut3r_feat_ls(tokens):
    B, H, W, C = tokens.shape
    return tokens.reshape(B, H * W, C)


def get_cut3r_pos(tokens):
    """Get CUT3R-style positional encoding for DA3 tokens.
    """
    # Apply CUT3R-style positional encoding to the DA3 tokens
    B, H, W, C = tokens.shape  # (2, 36, 36, 1536)

    ys = torch.arange(H, device=tokens.device)
    xs = torch.arange(W, device=tokens.device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")   # Y,X: (H, W)

    pos_hw2 = torch.stack([Y, X], dim=-1)          # (H, W, 2)
    pos = pos_hw2.view(1, H * W, 2).repeat(B, 1, 1)  # (B, H*W, 2)
    return pos


def create_views_from_da3(da3_tokens):
    """Given DA3 transformer tokens, returns a "view" object as CUT3R normally expects as input.
    """
    # A view is a dictionary containing the following elements:
    # img, ray_map, true_shape, idx, instance, camera_pose, img_mask, ray_mask, update, reset
    
    # Split the DA3 tokens by the view dimension
    # [2, 36, 35, 1536] -> [[1, 36, 36, 1536], [1, 36, 36, 1536]]
    num_views = da3_tokens.shape[0]
    views = torch.split(da3_tokens, 1, dim=0)
    print(f"len(views): {len(views)}")
    for view in views:
        print(f"view.shape: {view.shape}, type(view): {type(view)}")

    # Create the view object for each view.
    # TODO: track idx and instance globally, currently they are always 0 and 1
    view_dicts = []
    for i in range(num_views):
        view_dict = {
            "img": None,
            "ray_map": None,
            "true_shape": torch.tensor([512, 512], dtype=torch.int32),
            "idx": i,
            "instance": i,
            "camera_pose": torch.eye(4),
            "img_mask": torch.tensor([True]),
            "ray_mask": torch.tensor([False]),
            "update": torch.tensor([True]),
            "reset": torch.tensor([False]),
        }
        view_dicts.append(view_dict)

    # CUT3R's views are stored as a list of dicts.
    print(f"len(view_dicts): {len(view_dicts)}")
    for view_dict in view_dicts:
        print(f"view_dict: {view_dict}")

    return view_dicts


def get_cut3r_encoder_outputs(da3_tokens, size: int = 504):
    """Returns the would-be encoder outputs from CUT3R.
    """
    shape = get_cut3r_shape(size)
    feat_ls = get_cut3r_feat_ls(da3_tokens)
    pos = get_cut3r_pos(da3_tokens)
    return shape, feat_ls, pos


# Use this in train_mine.py to replace the entire input preprocessing + encoder phase of CUT3R.
def get_cut3r_encoder_outputs_from_da3(
    batch,
    da3_model="depth-anything/DA3-GIANT-1.1",
    feat_layers=[39],
    da3_size=504,
    device="cuda",
):
    """
    batch: list[dict], CUT3R views from dataloader
    returns: shape, feat_ls, pos   (CUT3R encoder-output contract)
    """
    # 1) convert batched npy images back to RGB images for DA3
    # batch[i]["img"]: [B,3,H,W] in [-1,1], usually B=1
    imgs_for_da3 = []
    for view in batch:
        x = view["img"]                      # [B,3,H,W]
        assert view["img"].shape[0] == 1, "Current DA3 bridge assumes batch_size=1"
        x = x[0].detach().float()            # [3,H,W]
        x = (x + 1.0) / 2.0                  # [0,1]
        x = x.clamp(0, 1)
        x = (x.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")  # HWC uint8
        imgs_for_da3.append(x)

    # 2) run DA3 once on list of views (batch)
    pred = da3_model.inference(
        imgs_for_da3,
        ref_view_strategy="first",
        export_feat_layers=feat_layers,
        process_res=da3_size,
    )

    # 3) pull tokens from final transformer layer
    tokens = torch.from_numpy(pred.aux[f"feat_layer_{feat_layers[0]}"]).to(device)
    # expected: [V, Ht, Wt, C]

    # 4) build CUT3R-style outputs
    V, Ht, Wt, C = tokens.shape
    feat_ls = [tokens[i : i + 1].reshape(1, Ht * Wt, C) for i in range(V)]

    ys = torch.arange(Ht, device=tokens.device)
    xs = torch.arange(Wt, device=tokens.device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    p = torch.stack([Y, X], dim=-1).reshape(1, Ht * Wt, 2)
    pos = [p.clone() for _ in range(V)]

    # choose shape consistent with token grid / downstream head expectations
    patch = 14
    H_img, W_img = Ht * patch, Wt * patch
    assert H_img % patch == 0 and W_img % patch == 0

    shape = [
        torch.tensor([[H_img, W_img]], dtype=torch.int32, device=tokens.device)
        for _ in range(V)
    ]

    return shape, feat_ls, pos



############################
### Model Forward Passes ###
############################

def da3_forward_pass(model, inputs, feat_layers: list[int]):
    """Run DA3 inference and return selected transformer feature tokens."""
    prediction = model.inference(
        inputs,
        ref_view_strategy="first",
        export_feat_layers=feat_layers,
    )
    print(f"Prediction object fields:")
    for f in fields(prediction):
        print(f"    {f.name}")

    tokens = torch.from_numpy(prediction.aux[f"feat_layer_{feat_layers[0]}"]).to(model.device)
    return tokens


def cut3r_forward_pass(shape, feat_ls, pos, model, device):
    """Performs a CUT3R memory update using the final-layer transformer tokens from DA3 backbone."""
    # from src.CUT3R.src.dust3r.inference import da3_forward
    outputs, state_args = model.da3_forward(shape, feat_ls, pos, ret_state=True)
    return outputs, state_args


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # Load frozen DA3
    da3_model = DepthAnything3.from_pretrained(args.da3_model)
    da3_model.to(args.device)
    da3_model.eval()
    print(f"Loaded DA3 model from {args.da3_model}")

    # Load frozen CUT3R
    # TODO: later run with this unfrozen
    cut3r_model = ARCroco3DStereo.from_pretrained(args.cut3r_model)
    cut3r_model.to(args.device)
    # cut3r_model.eval()
    cut3r_model.train()
    print(f"Loaded CUT3R model from {args.cut3r_model}")

    # Preprocess input pairs
    # robomimic uses jpg
    inputs = sorted(glob.glob(os.path.join(args.input_path, "*.jpg")))
    pairs = []
    for i in range(0, len(inputs) - 1, args.views_per_step):
        pairs.append(inputs[i:i+args.views_per_step])
    print(f"Found {len(pairs)} pairs")

    #################################
    ### 1. Collect all DA3 tokens ###
    #################################

    feat_layers = [39]
    for pair in pairs:
        # Run DA3 forward pass, get the final layer transformer outputs for the current image pair.
        da3_tokens = da3_forward_pass(da3_model, pair, feat_layers)
        print(f"DEBUG da3_tokens.shape: {da3_tokens.shape}")
        print(f"DEBUG type(da3_tokens): {type(da3_tokens)}")


        ######################################################
        ### 2. Create artifical "views" from DA3 features. ###
        ######################################################

        # This views object will be a list of dictionaries where the length of the list is the
        # number of views given as input to DA3.
        views = create_views_from_da3(da3_tokens)
        shape, feat_ls, pos = get_cut3r_encoder_outputs(da3_tokens)
        print(f"shape: {shape}")
        print(f"feat_ls: {feat_ls}")
        print(f"pos: {pos}")
        print(f"len(shape): {len(shape)}, len(feat_ls): {len(feat_ls)}, len(pos): {len(pos)}")
        print(f"type(shape): {type(shape)}, type(feat_ls): {type(feat_ls)}, type(pos): {type(pos)}")
        
        ##########################################################
        ### 3. Call CUT3R forward pass WITH GRADIENTS ENABLED. ###
        ##########################################################

        # TODO: replace with training code
        # Run forward and save outputs
        with torch.enable_grad():
            # Since DA3 inference is run over multiple views simultaneously, once we get to CUT3R
            # we should pass each view individually since it expects monocular inputs.
            for view in views:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.device == "cuda")):
                    outputs, state_args = cut3r_forward_pass(shape, feat_ls, pos, cut3r_model, args.device)
        save_geometry_outputs(outputs, args.output_path)


if __name__ == "__main__":
    main()
