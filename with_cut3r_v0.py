"""
Bridging DA3 transformer tokens with CUT3R memory.

1. Follow DA3's inference path up until transformer tokens are outputted.
2. Adapt DA3 tokens for compatibility with CUT3R's encoder.
3. Reuse CUT3R's encoder, state update, and decoder.
4. Save decoded tensors and diagnostics for per-timestep analysis (robomimic has 2 views per unit of time).
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
from depth_anything_3.api import DepthAnything3
from CUT3R.src.dust3r.model import ARCroco3DStereo


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
    parser.add_argument("--da3_process_res", type=int, default=504, help="DA3 process resolution.")
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


def preprocess_tokens_old(tokens):
    """Preprocesses transformer tokens to be comptaible with CUT3R state and memory dimensions."""
    # CUT3R requires three variables in order to update the memory: "shape, feat_ls, pos"
    # shape: list of length equal to total number of inputs, each element of the list is a tensor([[512, 512]])
    # feat_ls: [1, 1024, 1024], 32x32 = 1024
    # pos (positional encodings): [1, 1024, 2]
    # NOTE: each of these is returned as a list

    # Get shape
    # TODO: modify this to be total number of input images?
    # num_inputs = tokens.shape[0]
    # shape = [torch.tensor([[512, 512]])] * num_inputs
    # print(f"DA3 shape: {shape}")

    # # Get feat_ls
    # B, H, W, C = tokens.shape
    # feat_ls = tokens.reshape(B, H * W, C)
    # feat_ls = []
    # print(f"DA3 feat_ls[0].shape: {feat_ls[0].shape}")

    # # Get pos
    # print(f"DA3 tokens.shape: {tokens.shape}")
    # pos = get_cut3r_pos(tokens)
    # print(f"DA3 pos.shape: {pos[0].shape}")

    B, H, W, C = tokens.shape  # B=2
    tokens_flat = tokens.reshape(B, H * W, C)  # [2,1296,1536]
    pos_all = get_cut3r_pos(tokens)            # [2,1296,2]

    feat_ls = [tokens_flat[i:i+1] for i in range(B)]   # two tensors [1,1296,1536]
    pos = [pos_all[i:i+1] for i in range(B)]            # two tensors [1,1296,2]
    shape = [torch.tensor([[504, 504]], device=tokens.device, dtype=torch.int32) for _ in range(B)]

    print(f"DA3 len(shape): {len(shape)}, shape[0].shape: {shape[0].shape}")
    print(f"DA3 len(feat_ls): {len(feat_ls)}, feat_ls[0].shape: {feat_ls[0].shape}, feat_ls[1].shape: {feat_ls[1].shape}")
    print(f"DA3 len(pos): {len(pos)}, pos[0].shape: {pos[0].shape}, pos[1].shape: {pos[1].shape}")

    return shape, feat_ls, pos


def preprocess_tokens(tokens, cut3r_image_hw=(512, 512), conv_1x1=None):
    # 1) Adapt DA3 [B,36,36,1536] -> CUT3R [B,1024,1024]
    # Old approach: use interpolation, not trainable just math
    # adapted = adapt_da3_tokens_to_cut3r(
    #     da3_tokens=tokens,
    #     target_grid_hw=(32, 32),
    #     target_dim=1024,
    #     projection="orthogonal",   # or "truncate"/"none"
    #     seed=0,
    # )  # [B, 32*32, 1024]

    # New approach: use 1x1 convolution, it will be initially random but it is trainable
    # TODO: while CUT3R is being trained, make sure this layer is as well. Overfit it to a single scene!!!
    adapted, conv_1x1 = adapt_da3_tokens_to_cut3r_conv(
        da3_tokens=tokens,
        target_grid_hw=(32, 32),
        target_dim=1024,
        conv_1x1=conv_1x1,
    )

    B = adapted.shape[0]

    # 2) Split per-view (CUT3R-style list of tensors)
    feat_ls = [adapted[i:i+1] for i in range(B)]   # each [1,1024,1024]

    # 3) Build pos for target grid (32x32), not original 36x36
    ys, xs = torch.meshgrid(
        torch.arange(32, device=tokens.device),
        torch.arange(32, device=tokens.device),
        indexing="ij",
    )
    pos_hw2 = torch.stack([ys, xs], dim=-1).view(1, 32 * 32, 2)  # [1,1024,2]
    pos = [pos_hw2.clone() for _ in range(B)]                     # list, each [1,1024,2]

    # 4) Shape metadata per view
    h, w = cut3r_image_hw
    shape = [
        torch.tensor([[h, w]], device=tokens.device, dtype=torch.int32)
        for _ in range(B)
    ]

    print(f"DA3 -> CUT3R len(shape): {len(shape)}, shape[0].shape: {shape[0].shape}")
    print(f"DA3 -> CUT3R len(feat_ls): {len(feat_ls)}, feat_ls[0].shape: {feat_ls[0].shape}, feat_ls[1].shape: {feat_ls[1].shape}")
    print(f"DA3 -> CUT3R len(pos): {len(pos)}, pos[0].shape: {pos[0].shape}, pos[1].shape: {pos[1].shape}")

    return shape, feat_ls, pos, conv_1x1


def adapt_da3_tokens_to_cut3r(
    da3_tokens: torch.Tensor,
    target_grid_hw: tuple[int, int] = (32, 32),
    target_dim: int = 1024,
    projection: str = "orthogonal",   # "orthogonal" | "truncate" | "none"
    seed: int = 0,
) -> torch.Tensor:
    """
    Convert DA3 tokens [B, H, W, C] to CUT3R-compatible tokens [B, Ht*Wt, Ct].

    Steps:
      1) spatial resize: HxW -> Ht x Wt (bilinear on channel-first grid)
      2) channel adapt: C -> Ct
         - orthogonal: fixed random orthonormal projection (no training)
         - truncate: keep first Ct channels
         - none: require C == Ct
    """
    if da3_tokens.ndim != 4:
        raise ValueError(f"Expected da3_tokens [B,H,W,C], got {tuple(da3_tokens.shape)}")

    B, H, W, C = da3_tokens.shape
    Ht, Wt = target_grid_hw
    device = da3_tokens.device
    dtype = da3_tokens.dtype

    # [B,H,W,C] -> [B,C,H,W]
    x = da3_tokens.permute(0, 3, 1, 2).contiguous()

    # Spatial adapt
    if (H, W) != (Ht, Wt):
        x = torch.nn.functional.interpolate(
            x.float(), size=(Ht, Wt), mode="bilinear", align_corners=False
        )

    # Channel adapt
    if projection == "none":
        if C != target_dim:
            raise ValueError(f"projection='none' requires C==target_dim, got {C} vs {target_dim}")
        x = x
    elif projection == "truncate":
        if C < target_dim:
            raise ValueError(f"Cannot truncate up from C={C} to target_dim={target_dim}")
        x = x[:, :target_dim, :, :]
    elif projection == "orthogonal":
        # Build fixed orthonormal projection W: [C, target_dim]
        # Deterministic from seed; no training.
        if C == target_dim:
            pass
        elif C > target_dim:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            W = torch.randn(C, target_dim, device=device, dtype=torch.float32, generator=g)
            W, _ = torch.linalg.qr(W, mode="reduced")  # columns orthonormal
            # [B,C,Ht,Wt] -> [B,Ht,Wt,C] -> matmul -> [B,Ht,Wt,target_dim]
            x = x.permute(0, 2, 3, 1) @ W
            x = x.permute(0, 3, 1, 2)
        else:  # C < target_dim
            pad = target_dim - C
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))
    else:
        raise ValueError(f"Unknown projection='{projection}'")

    # Back to token sequence: [B,Ct,Ht,Wt] -> [B,Ht*Wt,Ct]
    x = x.permute(0, 2, 3, 1).contiguous().view(B, Ht * Wt, target_dim)
    return x.to(dtype=dtype)


def adapt_da3_tokens_to_cut3r_conv(
    da3_tokens: torch.Tensor,                 # [B,H,W,C]
    target_grid_hw: tuple[int, int] = (32, 32),
    target_dim: int = 1024,
    conv_1x1: torch.nn.Conv2d | None = None, # in=1536,out=1024
):
    """
    Convert DA3 tokens [B,H,W,C] to CUT3R-compatible tokens [B,Ht*Wt,Ct] using a 1x1 convolution.
    """
    
    # TODO: initialize weights?

    B, H, W, C = da3_tokens.shape
    Ht, Wt = target_grid_hw

    x = da3_tokens.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]
    # print(f"DEBUG x.shape: {x.shape}")    # [2, 1536, 36, 36]

    if conv_1x1 is None:
        conv_1x1 = torch.nn.Conv2d(C, target_dim, kernel_size=1, bias=True).to(x.device)

    x = conv_1x1(x)  # [B,target_dim,H,W]

    if (H, W) != (Ht, Wt):
        x = torch.nn.functional.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)

    x = x.permute(0, 2, 3, 1).contiguous().view(B, Ht * Wt, target_dim)  # [B,1024,1024]
    return x, conv_1x1


def cut3r_forward_pass(shape, feat_ls, pos, model, device):
    """Performs a CUT3R memory update using the final-layer transformer tokens from DA3 backbone."""
    # from src.CUT3R.src.dust3r.inference import da3_forward
    outputs, state_args = model.da3_forward(shape, feat_ls, pos, ret_state=True)
    return outputs, state_args

    # TODO code for saving outputs


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
    cut3r_model.eval()
    # cut3r_model.train()
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
    all_da3_tokens = []
    for pair in pairs:
        # Run DA3 forward pass, get the final layer transformer outputs for the current image pair.
        da3_tokens = da3_forward_pass(da3_model, pair, feat_layers)
        all_da3_tokens.append(da3_tokens.cpu())
        print(f"DEBUG da3_tokens.shape: {da3_tokens.shape}")

    del da3_model
    torch.cuda.empty_cache()


    ##################################################
    ### 2. Build stream of CUT3R-compatible tokens ###
    ##################################################

    all_shape, all_feat_ls, all_pos = [], [], []
    conv_1x1 = None
    for tok_cpu in all_da3_tokens:
        tok_gpu = tok_cpu.to(args.device)
        # Added convolutional layer
        # shape, feat_ls, pos, conv_1x1 = preprocess_tokens(
        #     tok_gpu,
        #     cut3r_image_hw=(512, 512),
        #     conv_1x1=conv_1x1,
        # )
        shape, feat_ls, pos = tok_gpu
        all_shape.extend(shape)
        all_feat_ls.extend(feat_ls)
        all_pos.extend(pos)

    del all_da3_tokens
    torch.cuda.empty_cache()

    
    #############################################################
    ### 3. Call CUT3R inference once, recurrence is internal. ###
    #############################################################

    # Sanity checks
    print(len(all_feat_ls), len(all_pos), len(all_shape))
    print(all_feat_ls[0].shape, all_pos[0].shape, all_shape[0].shape)
    print(all_feat_ls[-1].shape, all_pos[-1].shape, all_shape[-1].shape)

    assert len(all_feat_ls) == len(pairs) * args.views_per_step
    assert len(all_pos) == len(all_feat_ls) == len(all_shape)
    assert all(f.shape == (1, 1024, 1024) for f in all_feat_ls)
    assert all(p.shape == (1, 1024, 2) for p in all_pos)

    # Run forward and save outputs
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.device == "cuda")):
            outputs, state_args = cut3r_forward_pass(all_shape, all_feat_ls, all_pos, cut3r_model, args.device)
    save_geometry_outputs(outputs, args.output_path)
        

    # # Export final layer features
    # feat_layers = [39]
    # all_shape, all_feat_ls, all_pos = [], [], []
    # for i, pair in enumerate(pairs):
    #     print(f"[{i}] Processing pair: {pair}")

    #     # Run DA3 forward pass, get the final layer transformer outputs for the current image pair.
    #     da3_tokens = da3_forward_pass(da3_model, pair, feat_layers)
    #     print(f"DEBUG da3_tokens.shape: {da3_tokens.shape}")

    #     del da3_model
    #     torch.cuda.empty_cache()
                
    #     # Get CUT3R-style variables
    #     shape, feat_ls, pos = preprocess_tokens(da3_tokens, cut3r_image_hw=(512, 512))
    #     all_shape.extend(shape)
    #     all_feat_ls.extend(feat_ls)
    #     all_pos.extend(pos)

    #     # Run CUT3R forward pass and save outputs
    #     # NOTE: since I am passing None for views, I have to modify the output structure.
    #     #       We mainly care about depth, conf, and camera parameters though which are all intact.
    #     outputs, state_args = cut3r_forward_pass(shape, feat_ls, pos, cut3r_model, args.device)
    #     save_geometry_outputs(outputs, args.output_path)

    #     # TODO: add a branch for training

    #     # TODO: make a modular pipeline which either uses DA3 or CUT3R decoder to read from CUT3R memory.
    #     # if args.da3_decode:
    #     #     decoder_outputs = da3_decoder(model, transformer_tokens)
    #     # elif args.cut3r_decode:
    #     #     decoder_outputs = cut3r_decoder(model, transformer_tokens)
    #     # else:
    #     #     raise ValueError("Invalid decoder type")
    #     # return decoder_outputs


    #     # break   # just do one pair for testing


if __name__ == "__main__":
    main()
