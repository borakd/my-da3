"""
DA3 -> CUT3R memory bridge with explicit, research-oriented control flow.

Design goals:
1. Follow DA3's native preprocessing/tokenization path exactly up to transformer tokens.
2. Adapt DA3 tokens to CUT3R encoder contracts (token count + channel width).
3. Reuse CUT3R's own recurrent state transition and memory-write functions.
4. Save decoded tensors and diagnostics so behavior can be audited frame by frame.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DA3TokenBundle:
    """
    Container for DA3 transformer outputs in multi-view stream mode.

    tokens shape: [T, V, N, C]
      T: timestep count in stream
      V: views-per-step (e.g., 2 cameras)
      N: patch tokens
      C: token channels
    """

    tokens: torch.Tensor
    patch_grid_hw: tuple[int, int]
    patch_size: int
    views_per_step: int


@dataclass
class CUT3RRuntime:
    """
    Runtime tensors produced by CUT3R state initialization.

    These tensors are the exact variables consumed by CUT3R recurrent decoding:
    _recurrent_rollout -> pose_retriever.update_mem -> _downstream_head.
    """

    model: Any
    views: list[dict[str, Any]]
    pos: list[torch.Tensor]
    shape: list[torch.Tensor]
    init_state_feat: torch.Tensor
    init_mem: torch.Tensor
    state_feat: torch.Tensor
    state_pos: torch.Tensor
    mem: torch.Tensor
    patch_grid_hw: tuple[int, int]
    token_dim: int


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Run DA3 tokens through CUT3R memory.")
    parser.add_argument("--input_path", type=str, required=True, help="Input image directory.")
    parser.add_argument("--output_path", type=str, default="outputs/with_cut3r", help="Output directory.")
    parser.add_argument(
        "--da3_model",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="DA3 model id/path for DepthAnything3.from_pretrained().",
    )
    parser.add_argument(
        "--cut3r_model",
        type=str,
        default="src/CUT3R/src/cut3r_512_dpt_4_64.pth",
        help="CUT3R checkpoint path or HuggingFace model id.",
    )
    parser.add_argument("--da3_process_res", type=int, default=504, help="DA3 process resolution.")
    parser.add_argument(
        "--da3_process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize", "upper_bound_crop", "lower_bound_crop"],
        help="DA3 process resolution policy.",
    )
    parser.add_argument(
        "--da3_ref_view_strategy",
        type=str,
        default="first",
        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
        help="Reference-view strategy passed into DA3 backbone attention routing.",
    )
    parser.add_argument("--cut3r_size", type=int, default=512, help="CUT3R image loading size.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument(
        "--views_per_step",
        type=int,
        default=2,
        help="Number of synchronized views per timestep in the interleaved stream.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Use only the first K frames from the stream (0 means all).",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print step-by-step tensor diagnostics and run sanity checks.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="How often to print per-frame bridge stats.",
    )
    return parser.parse_args()


def add_repo_python_paths(repo_root: pathlib.Path) -> None:
    """
    Adds local repo paths so both DA3 and CUT3R packages are importable.

    Why this is correct in this codebase:
    - DA3 python package lives under `src/depth_anything_3`.
    - CUT3R modules are imported as `dust3r.*` from `src/CUT3R/src`.
    """
    src_root = repo_root / "src"
    cut3r_root = src_root / "CUT3R"
    cut3r_src = cut3r_root / "src"

    for p in (src_root, cut3r_root, cut3r_src):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def resolve_device(device_arg: str) -> torch.device:
    """Resolves `--device` with CUDA fallback behavior."""
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    return device


def collect_input_images(input_path: str) -> list[str]:
    """Collects sorted image paths with extensions used across DA3/CUT3R demos."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(input_path, ext)))
    paths = sorted(paths)
    if not paths:
        raise ValueError(f"No images found in {input_path}")
    return paths


def chunk_interleaved_stream(image_paths: list[str], views_per_step: int) -> list[list[str]]:
    """
    Groups a flat interleaved stream into timesteps.

    Expected order for `views_per_step=2`:
      [view1_t1, view2_t1, view1_t2, view2_t2, ...]
    """
    if views_per_step <= 0:
        raise ValueError("--views_per_step must be > 0.")
    if len(image_paths) % views_per_step != 0:
        raise ValueError(
            f"Image count {len(image_paths)} is not divisible by views_per_step={views_per_step}. "
            "Provide a complete interleaved stream."
        )
    return [image_paths[i : i + views_per_step] for i in range(0, len(image_paths), views_per_step)]


def log_step(title: str) -> None:
    """Consistent stage delimiter for an implementation-from-scratch style trace."""
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def tensor_summary(name: str, tensor: torch.Tensor) -> None:
    """Prints shape/dtype/device/range to verify data contracts at each stage."""
    if tensor.numel() == 0:
        print(f"[verify] {name}: empty tensor shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")
        return
    t = tensor.detach()
    # Avoid OOM during verification on very large tensors by using sampled stats.
    max_stat_elems = 1_000_000
    sampled = False
    if t.numel() > max_stat_elems:
        sampled = True
        step = max(t.numel() // max_stat_elems, 1)
        t_stat = t.flatten()[::step][:max_stat_elems]
    else:
        t_stat = t
    # Run statistics on CPU to avoid temporary large GPU allocations.
    t_stat = t_stat.to("cpu")

    if t.is_floating_point():
        finite = torch.isfinite(t_stat)
        finite_ratio = float(finite.float().mean().cpu())
        min_val = float(t_stat[finite].min().cpu()) if finite.any() else float("nan")
        max_val = float(t_stat[finite].max().cpu()) if finite.any() else float("nan")
    else:
        finite_ratio = 1.0
        min_val = float(t_stat.min().cpu())
        max_val = float(t_stat.max().cpu())
    sample_note = " sampled" if sampled else ""
    print(
        f"[verify] {name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
        f"min={min_val:.5g} max={max_val:.5g} finite_ratio={finite_ratio:.4f}{sample_note}"
    )


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Hard sanity check for floating tensors."""
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values.")


def describe_prediction_dict(pred: dict[str, torch.Tensor], prefix: str = "") -> None:
    """Print decoded key names and tensor shapes for quick correctness checks."""
    key_info = []
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            key_info.append(f"{k}:{tuple(v.shape)}")
        else:
            key_info.append(f"{k}:{type(v).__name__}")
    print(f"{prefix}{', '.join(key_info)}")


def move_view_to_device(view: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """
    Moves CUT3R view tensors to the target device.

    CUT3R inference utilities do this with similar tensor-only transfer logic in
    `src/CUT3R/src/dust3r/inference.py`.
    """
    ignore = {"idx", "instance"}
    out: dict[str, Any] = {}
    for key, value in view.items():
        if key in ignore:
            out[key] = value
        elif isinstance(value, torch.Tensor):
            out[key] = value.to(device=device, non_blocking=True)
        else:
            out[key] = value
    return out


def get_da3_backbone_owner(da3_model_obj: torch.nn.Module) -> torch.nn.Module:
    """
    Returns the model that owns `.backbone`.

    For nested checkpoints (`NestedDepthAnything3Net` in `model/da3.py`), the any-view
    branch is at `.da3`; for single-branch checkpoints, `.model` already exposes backbone.
    """
    if hasattr(da3_model_obj, "da3"):
        return da3_model_obj.da3
    return da3_model_obj


def as_patch_hw(patch_size: Any) -> tuple[int, int]:
    """Normalizes scalar/tuple patch-size values to `(h, w)` form."""
    if isinstance(patch_size, tuple):
        return int(patch_size[0]), int(patch_size[1])
    return int(patch_size), int(patch_size)


def get_da3_patch_size(da3_owner: torch.nn.Module) -> int:
    """
    Reads DA3 patch size from backbone internals instead of hard-coding.

    The DINOv2 backbone in this repo uses `pretrained.patch_size` from
    `model/dinov2/vision_transformer.py`.
    """
    pretrained = getattr(getattr(da3_owner, "backbone", None), "pretrained", None)
    if pretrained is None:
        return 14
    patch_size = getattr(pretrained, "patch_size", 14)
    patch_h, _ = as_patch_hw(patch_size)
    return patch_h


def resize_and_match_tokens(
    da3_frame_tokens: torch.Tensor,
    src_grid_hw: tuple[int, int],
    tgt_grid_hw: tuple[int, int],
    tgt_dim: int,
) -> torch.Tensor:
    """
    Adapts DA3 per-frame tokens to CUT3R encoder contract.

    Transformation:
    [B, N, C_src] -> [B, H_src, W_src, C_src] -> interpolate to H_tgt/W_tgt
    -> channel truncate/pad to C_tgt -> flatten -> [B, N_tgt, C_tgt].
    """
    bsz, _, src_dim = da3_frame_tokens.shape
    src_h, src_w = src_grid_hw
    tgt_h, tgt_w = tgt_grid_hw

    x = da3_frame_tokens.reshape(bsz, src_h, src_w, src_dim).permute(0, 3, 1, 2).float()
    if (src_h, src_w) != (tgt_h, tgt_w):
        x = F.interpolate(x, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)

    if src_dim > tgt_dim:
        x = x[:, :tgt_dim]
    elif src_dim < tgt_dim:
        x = F.pad(x, (0, 0, 0, 0, 0, tgt_dim - src_dim))

    return x.permute(0, 2, 3, 1).reshape(bsz, tgt_h * tgt_w, tgt_dim)


def build_cut3r_views(image_paths: list[str], cut3r_size: int) -> list[dict[str, Any]]:
    """
    Builds CUT3R view dictionaries compatible with `_forward_encoder`.

    `dust3r.utils.image.load_images` performs CUT3R-native preprocessing:
    resize/crop + normalization to [-1, 1], producing tensors keyed as expected
    by CUT3R model internals.
    """
    from dust3r.utils.image import load_images

    images = load_images(image_paths, size=cut3r_size, verbose=False)
    views: list[dict[str, Any]] = []
    for i, img_dict in enumerate(images):
        img = img_dict["img"]  # shape [1, 3, H, W]
        h, w = img.shape[-2:]

        # We run image-only mode (no ray maps). CUT3R expects both fields to exist,
        # so we provide NaN ray maps + ray_mask=False exactly like CUT3R demo setup.
        views.append(
            {
                "img": img,
                "ray_map": torch.full((img.shape[0], 6, h, w), torch.nan, dtype=img.dtype),
                "true_shape": torch.from_numpy(img_dict["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
        )
    return views


def extract_da3_tokens(
    args: argparse.Namespace,
    device: torch.device,
    timesteps: list[list[str]],
    verify: bool = True,
) -> DA3TokenBundle:
    """
    Executes DA3 per-timestep multi-view encoding and returns transformer tokens.

    Call-path alignment with repo internals:
    - `DepthAnything3._preprocess_inputs` -> `InputProcessor` pipeline.
    - `DepthAnything3._prepare_model_inputs` for device + batch layout.
    - `DepthAnything3._normalize_extrinsics` for camera-token stability.
    - `DepthAnything3Net.backbone(...)` for transformer token extraction.
    """
    from depth_anything_3.api import DepthAnything3

    if verify:
        log_step("Step 1/4 (DA3): Per-timestep multi-view encoding")

    da3 = DepthAnything3.from_pretrained(args.da3_model).to(device=device).eval()
    da3_owner = get_da3_backbone_owner(da3.model)
    da3_patch_size = get_da3_patch_size(da3_owner)

    all_tokens: list[torch.Tensor] = []
    src_grid_hw: tuple[int, int] | None = None
    expected_views = args.views_per_step

    for t, step_paths in enumerate(timesteps):
        if len(step_paths) != expected_views:
            raise ValueError(
                f"Timestep {t} has {len(step_paths)} views, expected {expected_views}."
            )

        imgs_cpu, ex_cpu, in_cpu = da3._preprocess_inputs(
            step_paths,
            process_res=args.da3_process_res,
            process_res_method=args.da3_process_res_method,
        )
        imgs, ex_t, in_t = da3._prepare_model_inputs(imgs_cpu, ex_cpu, in_cpu)
        ex_t_norm = da3._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)

        if ex_t_norm is not None and getattr(da3_owner, "cam_enc", None) is not None:
            with torch.autocast(device_type=imgs.device.type, enabled=False):
                cam_token = da3_owner.cam_enc(ex_t_norm, in_t, imgs.shape[-2:])
        else:
            cam_token = None

        with torch.no_grad():
            use_cuda_amp = imgs.device.type == "cuda"
            amp_dtype = torch.bfloat16 if use_cuda_amp and torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=imgs.device.type, dtype=amp_dtype, enabled=use_cuda_amp):
                da3_feats, _ = da3_owner.backbone(
                    imgs,
                    cam_token=cam_token,
                    export_feat_layers=[],
                    ref_view_strategy=args.da3_ref_view_strategy,
                )

        # shape [B, V, N, C]; B is 1 for this script.
        step_tokens = da3_feats[-1][0].float()
        if step_tokens.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {step_tokens.shape[0]} at timestep {t}.")
        step_tokens = step_tokens.squeeze(0)  # [V, N, C]

        src_h = imgs.shape[-2] // da3_patch_size
        src_w = imgs.shape[-1] // da3_patch_size
        if src_h * src_w != step_tokens.shape[1]:
            raise ValueError(
                f"DA3 token count mismatch at t={t}: N={step_tokens.shape[1]}, "
                f"expected {src_h}x{src_w}={src_h * src_w}"
            )

        if src_grid_hw is None:
            src_grid_hw = (src_h, src_w)
        elif src_grid_hw != (src_h, src_w):
            raise ValueError(
                f"Inconsistent DA3 patch grid: first={src_grid_hw}, current={(src_h, src_w)} at timestep {t}."
            )

        all_tokens.append(step_tokens)
        # Keep long-stream token buffers on CPU; move per-frame chunks to GPU later.
        all_tokens[-1] = all_tokens[-1].cpu()
        if verify:
            print(f"[verify] timestep={t:04d} paths={step_paths}")
            tensor_summary("da3.step_tokens", step_tokens)
            assert_finite("da3.step_tokens", step_tokens)

    if not all_tokens:
        raise ValueError("No DA3 tokens were extracted.")
    tokens = torch.stack(all_tokens, dim=0)  # [T, V, N, C]
    # Release DA3 model memory before CUT3R stage when running on CUDA.
    del da3
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if verify:
        log_step("Step 2/4 (DA3): Verify stacked multi-view stream tokens")
        tensor_summary("da3.tokens_stream", tokens)
        assert_finite("da3.tokens_stream", tokens)

    print(
        "DA3 transformer stream:",
        f"T={tokens.shape[0]} V={tokens.shape[1]} N={tokens.shape[2]} C={tokens.shape[3]}",
    )
    assert src_grid_hw is not None
    return DA3TokenBundle(
        tokens=tokens,
        patch_grid_hw=src_grid_hw,
        patch_size=da3_patch_size,
        views_per_step=expected_views,
    )


def initialize_cut3r_runtime(
    args: argparse.Namespace,
    device: torch.device,
    image_paths: list[str],
    verify: bool = True,
) -> CUT3RRuntime:
    """
    Builds CUT3R state tensors by calling `_forward_encoder`.

    Why this entry point:
    `_forward_encoder` is the official internal split used by CUT3R's TBPTT/inference
    utilities before recurrent decoding; it returns exactly the state variables we need.
    """
    from dust3r.model import ARCroco3DStereo

    if args.cut3r_model.endswith(".pth") and not os.path.exists(args.cut3r_model):
        raise FileNotFoundError(
            f"CUT3R checkpoint not found: {args.cut3r_model}. "
            "Pass --cut3r_model to a valid .pth file."
        )

    if verify:
        log_step("Step 3/4 (CUT3R): Initialize recurrent state from the same image stream")

    model = ARCroco3DStereo.from_pretrained(args.cut3r_model).to(device=device).eval()
    if not getattr(model, "pose_head_flag", False):
        raise ValueError("CUT3R checkpoint has pose_head disabled; memory bridge requires pose retriever.")

    views = build_cut3r_views(image_paths, args.cut3r_size)
    views = [move_view_to_device(v, device) for v in views]

    if verify and views:
        tensor_summary("cut3r.view0.img", views[0]["img"])
        tensor_summary("cut3r.view0.ray_map", views[0]["ray_map"])

    with torch.no_grad():
        (feats, pos, shape), (
            init_state_feat,
            init_mem,
            state_feat,
            state_pos,
            mem,
        ) = model._forward_encoder(views)

    token_dim = feats[0].shape[-1]
    patch_h, patch_w = as_patch_hw(model.patch_embed.patch_size)
    cut_patch_h = views[0]["img"].shape[-2] // patch_h
    cut_patch_w = views[0]["img"].shape[-1] // patch_w

    print(f"CUT3R encoder target: patch grid={cut_patch_h}x{cut_patch_w}, C={token_dim}")
    if verify:
        tensor_summary("cut3r.state_feat_init", state_feat)
        tensor_summary("cut3r.mem_init", mem)
        assert_finite("cut3r.state_feat_init", state_feat)
        assert_finite("cut3r.mem_init", mem)
    return CUT3RRuntime(
        model=model,
        views=views,
        pos=pos,
        shape=shape,
        init_state_feat=init_state_feat,
        init_mem=init_mem,
        state_feat=state_feat,
        state_pos=state_pos,
        mem=mem,
        patch_grid_hw=(cut_patch_h, cut_patch_w),
        token_dim=token_dim,
    )


def validate_bridge_compatibility(
    da3_tokens: DA3TokenBundle,
    cut3r: CUT3RRuntime,
    verify: bool = True,
) -> None:
    """
    Enforces minimal compatibility assumptions before recurrent rollout.
    """
    timesteps, views_per_step = da3_tokens.tokens.shape[:2]
    expected_frames = timesteps * views_per_step
    if expected_frames != len(cut3r.views):
        raise ValueError(
            f"Frame count mismatch: DA3 stream={expected_frames} "
            f"(T={timesteps} x V={views_per_step}) CUT3R={len(cut3r.views)}"
        )
    if cut3r.state_feat.shape[0] != 1:
        raise ValueError(f"CUT3R state batch must be 1 for this script, got {cut3r.state_feat.shape[0]}.")
    if da3_tokens.views_per_step != views_per_step:
        raise ValueError(
            f"Internal views_per_step mismatch: bundle={da3_tokens.views_per_step} tensor={views_per_step}"
        )
    if verify:
        log_step("Step 4/4 (Bridge): Compatibility checks passed")
        print(
            "[verify] DA3 tokens and CUT3R runtime are shape-compatible:",
            f"T={timesteps} V={views_per_step} total_frames={expected_frames}",
        )


def apply_update_and_reset_masks(
    view: dict[str, Any],
    state_feat: torch.Tensor,
    mem: torch.Tensor,
    new_state_feat: torch.Tensor,
    new_mem: torch.Tensor,
    init_state_feat: torch.Tensor,
    init_mem: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies CUT3R's update/reset policy, matching `_forward_decoder_step`.
    """
    img_mask = view["img_mask"]
    update = view.get("update", None)
    if update is not None:
        update_mask = img_mask & update
    else:
        update_mask = img_mask
    update_mask = update_mask[:, None, None].float()

    state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
    mem = new_mem * update_mask + mem * (1 - update_mask)

    reset_mask = view["reset"]
    if reset_mask is not None:
        reset_mask = reset_mask[:, None, None].float()
        state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
        mem = init_mem * reset_mask + mem * (1 - reset_mask)
    return state_feat, mem


def bridge_tokens_and_decode(
    da3_tokens: DA3TokenBundle,
    cut3r: CUT3RRuntime,
    verify: bool = True,
    print_every: int = 1,
) -> tuple[list[dict[str, torch.Tensor]], list[float]]:
    """
    Core bridge loop: token adaptation -> recurrent rollout -> memory update -> decode.
    """
    decoded_outputs: list[dict[str, torch.Tensor]] = []
    memory_norms: list[float] = []

    model = cut3r.model
    state_feat = cut3r.state_feat
    mem = cut3r.mem

    timesteps, views_per_step = da3_tokens.tokens.shape[:2]
    frame_counter = 0

    for t in range(timesteps):
        for v in range(views_per_step):
            frame_idx = t * views_per_step + v
            da3_i = da3_tokens.tokens[t, v].unsqueeze(0).to(state_feat.device, non_blocking=True)  # [1, N, C]
            da3_i_reshaped = resize_and_match_tokens(
                da3_i,
                src_grid_hw=da3_tokens.patch_grid_hw,
                tgt_grid_hw=cut3r.patch_grid_hw,
                tgt_dim=cut3r.token_dim,
            )
            # CUT3R memory key input uses image-level pooled feature (`_get_img_level_feat`).
            da3_global_i = da3_i_reshaped.mean(dim=1, keepdim=True)
            if verify and (frame_counter % max(print_every, 1) == 0):
                print(f"[verify] timestep={t:04d} view={v:02d} frame_idx={frame_idx:04d}")
                tensor_summary("bridge.da3_i", da3_i)
                tensor_summary("bridge.da3_i_reshaped", da3_i_reshaped)
                tensor_summary("bridge.da3_global_i", da3_global_i)
                assert_finite("bridge.da3_i_reshaped", da3_i_reshaped)

            pos_i = cut3r.pos[frame_idx]
            shape_i = cut3r.shape[frame_idx]

            # Match CUT3R `_forward_decoder_step` behavior:
            # frame 0 uses learned pose token; later frames query memory.
            if frame_idx == 0:
                pose_feat_i = model.pose_token.expand(da3_i_reshaped.shape[0], -1, -1)
            else:
                pose_feat_i = model.pose_retriever.inquire(da3_global_i, mem)
            # CUT3R's decoder step uses -1 sentinel pose positions. That works with
            # the CUDA-compiled RoPE path, but the slow PyTorch fallback in
            # `src/CUT3R/src/croco/models/pos_embed.py` indexes embeddings directly
            # and does not accept negative indices. We therefore use zeros here to
            # keep the bridge executable when cuRoPE is unavailable.
            pose_pos_i = torch.zeros(
                da3_i_reshaped.shape[0],
                1,
                2,
                device=da3_i_reshaped.device,
                dtype=pos_i.dtype,
            )

            new_state_feat, dec = model._recurrent_rollout(
                state_feat,
                cut3r.state_pos,
                da3_i_reshaped,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                cut3r.init_state_feat,
                img_mask=cut3r.views[frame_idx]["img_mask"],
                reset_mask=cut3r.views[frame_idx]["reset"],
                update=cut3r.views[frame_idx].get("update", None),
            )

            # Memory write follows LocalMemory.update_mem contract:
            # key: pooled image feature, value: decoder pose token.
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = model.pose_retriever.update_mem(mem, da3_global_i, out_pose_feat_i)

            # Reconstruct the exact 4-scale head input used by CUT3R decoder step.
            head_input = [
                dec[0].float(),
                dec[model.dec_depth * 2 // 4][:, 1:].float(),
                dec[model.dec_depth * 3 // 4][:, 1:].float(),
                dec[model.dec_depth].float(),
            ]
            decoded_outputs.append(model._downstream_head(head_input, shape_i, pos=pos_i))

            state_feat, mem = apply_update_and_reset_masks(
                cut3r.views[frame_idx],
                state_feat,
                mem,
                new_state_feat,
                new_mem,
                cut3r.init_state_feat,
                cut3r.init_mem,
            )
            memory_norms.append(float(mem.norm().detach().cpu()))
            if verify and (frame_counter % max(print_every, 1) == 0):
                describe_prediction_dict(decoded_outputs[-1], prefix="[verify] decoded keys/shapes: ")
                print(f"[verify] mem_norm={memory_norms[-1]:.6f}")
            frame_counter += 1

    return decoded_outputs, memory_norms


def save_decoded_outputs(decoded_outputs: list[dict[str, torch.Tensor]], output_path: str) -> None:
    """
    Writes per-frame tensors as `.npy` files grouped by output key.
    """
    os.makedirs(output_path, exist_ok=True)
    frame_dir = os.path.join(output_path, "decoded")
    os.makedirs(frame_dir, exist_ok=True)

    for i, pred in enumerate(decoded_outputs):
        for key, value in pred.items():
            if not isinstance(value, torch.Tensor):
                continue
            key_dir = os.path.join(frame_dir, key)
            os.makedirs(key_dir, exist_ok=True)
            np.save(os.path.join(key_dir, f"{i:06d}.npy"), value.detach().cpu().numpy())


def save_run_metadata(
    args: argparse.Namespace,
    da3_tokens: DA3TokenBundle,
    cut3r: CUT3RRuntime,
    memory_norms: list[float],
    output_path: str,
) -> None:
    """Saves compact run metadata for reproducibility and analysis."""
    metadata = {
        "num_timesteps": int(da3_tokens.tokens.shape[0]),
        "views_per_step": int(da3_tokens.tokens.shape[1]),
        "num_frames": int(da3_tokens.tokens.shape[0] * da3_tokens.tokens.shape[1]),
        "batch_size": 1,
        "da3_patch_size": int(da3_tokens.patch_size),
        "da3_patch_grid_hw": list(da3_tokens.patch_grid_hw),
        "da3_token_dim": int(da3_tokens.tokens.shape[-1]),
        "cut3r_patch_grid_hw": list(cut3r.patch_grid_hw),
        "cut3r_token_dim": int(cut3r.token_dim),
        "da3_model": args.da3_model,
        "cut3r_model": args.cut3r_model,
        "da3_ref_view_strategy": args.da3_ref_view_strategy,
        "memory_norm_first": float(memory_norms[0]) if memory_norms else None,
        "memory_norm_last": float(memory_norms[-1]) if memory_norms else None,
    }
    with open(os.path.join(output_path, "bridge_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    repo_root = pathlib.Path(__file__).resolve().parent
    add_repo_python_paths(repo_root)

    device = resolve_device(args.device)
    image_paths = collect_input_images(args.input_path)
    if args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    if not image_paths:
        raise ValueError("No frames available after applying --max_frames.")
    timesteps = chunk_interleaved_stream(image_paths, args.views_per_step)
    print(f"Found {len(image_paths)} input images")
    print(f"Interpreting stream as {len(timesteps)} timesteps x {args.views_per_step} views (interleaved).")
    if args.verify:
        print(f"[verify] output_path={args.output_path}")
        print(f"[verify] device={device}")
        print(f"[verify] first_frame={image_paths[0]}")
        print(f"[verify] last_frame={image_paths[-1]}")

    # Stage 1: DA3 token extraction from native DA3 preprocessing + backbone path.
    da3_tokens = extract_da3_tokens(args, device, timesteps, verify=args.verify)

    # Stage 2: CUT3R state initialization from native CUT3R image preprocessing path.
    cut3r = initialize_cut3r_runtime(args, device, image_paths, verify=args.verify)

    # Stage 3: Bridge rollout (adapt DA3 tokens, update CUT3R memory, decode outputs).
    validate_bridge_compatibility(da3_tokens, cut3r, verify=args.verify)
    decoded_outputs, memory_norms = bridge_tokens_and_decode(
        da3_tokens,
        cut3r,
        verify=args.verify,
        print_every=args.print_every,
    )

    # Stage 4: Persist frame-level outputs and diagnostics.
    save_decoded_outputs(decoded_outputs, args.output_path)
    np.save(os.path.join(args.output_path, "memory_norms.npy"), np.asarray(memory_norms, dtype=np.float32))
    save_run_metadata(args, da3_tokens, cut3r, memory_norms, args.output_path)

    print(f"Saved decoded outputs to: {args.output_path}")


if __name__ == "__main__":
    main()
