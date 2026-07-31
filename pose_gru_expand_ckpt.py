#!/usr/bin/env python
"""
pose_gru_expand_ckpt.py — offline surgery: widen a trained v2 PoseGRU
checkpoint (cell input 7/14) into the v3 image-feature layout (7/14 + D) so
an F-lever run can warm-start from it.

What it does, exactly:
  - cell.weight_ih: old columns copied verbatim into the FIRST base_width
    columns; the new D feature columns get a fresh default GRUCell init
    (U(-1/sqrt(hidden), 1/sqrt(hidden))). They must NOT be zero: the new
    img_proj is zero-init, and zero columns on both sides would deadlock the
    feature channel's gradients permanently.
  - cell.weight_hh / bias_ih / bias_hh / head.*: copied verbatim.
  - pose_gru.img_proj.{weight,bias}: ZERO (function-preserving at load: the
    feature term is exactly 0, so the widened GRU computes exactly what the
    v2 GRU computed).
  - pose_gru.img_norm.{weight,bias}: LayerNorm defaults (ones/zeros).
  - ckpt args gain pose_gru_img_feat/_dim/_frames (and optionally
    pose_gru_iters) so load_model's sniff and the trainer agree.

The result is bit-equivalent in function to the source checkpoint until
img_proj moves away from zero.

Usage:
  python pose_gru_expand_ckpt.py --src <v2_ckpt.pth> --dst <out.pth> \
      [--img-feat-dim 32] [--frames 2] [--enc-dim 1024] [--iters 8]
"""
import argparse
import math

import torch
from omegaconf import OmegaConf, open_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--img-feat-dim", type=int, default=32)
    ap.add_argument("--frames", type=int, default=2, choices=[1, 2])
    ap.add_argument("--enc-dim", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=None,
                    help="also set args.pose_gru_iters (R lever) in the ckpt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    prefix = "pose_gru." if any(k.startswith("pose_gru.") for k in sd) else (
        "module.pose_gru." if any(k.startswith("module.pose_gru.") for k in sd) else None
    )
    assert prefix is not None, f"{args.src}: no pose_gru weights found"
    assert prefix + "img_proj.weight" not in sd, (
        f"{args.src}: already carries an img_proj — refusing to re-expand"
    )

    w_ih = sd[prefix + "cell.weight_ih"]
    hidden3, base_width = w_ih.shape
    hidden = sd[prefix + "cell.weight_hh"].shape[1]
    assert base_width in (7, 14), f"unexpected cell input width {base_width}"
    D = args.img_feat_dim
    assert D > 0

    torch.manual_seed(args.seed)
    # Fresh default GRUCell init for the new feature columns (see module doc).
    bound = 1.0 / math.sqrt(hidden)
    new_cols = torch.empty(hidden3, D, dtype=w_ih.dtype).uniform_(-bound, bound)
    sd[prefix + "cell.weight_ih"] = torch.cat([w_ih, new_cols], dim=1)

    src_total = args.enc_dim * args.frames
    sd[prefix + "img_proj.weight"] = torch.zeros(D, src_total, dtype=w_ih.dtype)
    sd[prefix + "img_proj.bias"] = torch.zeros(D, dtype=w_ih.dtype)
    sd[prefix + "img_norm.weight"] = torch.ones(args.enc_dim, dtype=w_ih.dtype)
    sd[prefix + "img_norm.bias"] = torch.zeros(args.enc_dim, dtype=w_ih.dtype)

    train_args = ckpt.get("args")
    if train_args is not None:
        if OmegaConf.is_config(train_args):
            with open_dict(train_args):
                train_args.pose_gru_img_feat = "input"
                train_args.pose_gru_img_feat_dim = D
                train_args.pose_gru_img_feat_frames = args.frames
                if args.iters is not None:
                    train_args.pose_gru_iters = args.iters
        else:
            setattr(train_args, "pose_gru_img_feat", "input")
            setattr(train_args, "pose_gru_img_feat_dim", D)
            setattr(train_args, "pose_gru_img_feat_frames", args.frames)
            if args.iters is not None:
                setattr(train_args, "pose_gru_iters", args.iters)

    torch.save(ckpt, args.dst)
    print(
        f"expanded {args.src} -> {args.dst}: cell input {base_width} -> "
        f"{base_width + D}, img_proj ({D}, {src_total}) zero-init, frames={args.frames}"
        + (f", args.pose_gru_iters={args.iters}" if args.iters is not None else "")
    )


if __name__ == "__main__":
    main()
