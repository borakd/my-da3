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
    # ORDER MATTERS: these source-lever refusals must run BEFORE the generic
    # img_proj/img_norm asserts below — an encoder/corr checkpoint carries
    # img_proj too, and the generic assert would refuse it first with a
    # misleading 'already expanded' reason.
    # Encoder-source F checkpoints (pose_gru_img_feat_src=resnet18/dinov2)
    # carry a frozen img_encoder whose features — not pooled pre-ray tokens —
    # are what the cell columns were trained on. Splicing the pooled-layout
    # projector (enc_dim-wide img_norm/img_proj) onto that cell would wire the
    # wrong source at the wrong width.
    assert not any(k.startswith(prefix + "img_encoder.") for k in sd), (
        f"{args.src}: carries a frozen img_encoder — this is an "
        f"encoder-source F checkpoint (pose_gru_img_feat_src != pooled). "
        f"Refusing to expand it into the pooled layout."
    )
    # Same refusal via the recorded args: corr checkpoints have no encoder
    # keys, so key-sniffing alone would miss them. Anything not pooled
    # (None covers pre-lever ckpts and non-struct OmegaConf misses) must not
    # be expanded into the pooled layout.
    feat_src = getattr(ckpt.get("args") or object(), "pose_gru_img_feat_src", None)
    assert feat_src in (None, "pooled"), (
        f"{args.src}: trained with pose_gru_img_feat_src={feat_src!r} — its "
        f"cell consumes that source's features, not pooled pre-ray tokens. "
        f"Expanding it into the pooled layout would produce a module that is "
        f"neither arm. Refusing."
    )

    assert prefix + "img_proj.weight" not in sd, (
        f"{args.src}: already carries an img_proj — refusing to re-expand"
    )
    # img_proj absence is NO LONGER proof the F lever is off: a checkpoint
    # trained with pose_gru_img_feat_proj=False has the features wired
    # straight into the cell and carries img_norm but no projector. Splicing a
    # projector onto that cell would produce a module that is neither arm.
    assert prefix + "img_norm.weight" not in sd, (
        f"{args.src}: carries img_norm but no img_proj — this is an F "
        f"checkpoint trained with pose_gru_img_feat_proj=False, whose cell "
        f"already consumes the raw features. Refusing to expand it."
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
