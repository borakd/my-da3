# Frozen image-feature sources for the PoseGRU F lever (pose_gru_img_feat_src).
#
# Three non-"pooled" sources live here:
#   - "resnet18"      : frozen torchvision resnet18 (ImageNet), avgpool output (512-d)
#   - "dinov2_vits14" : frozen official DINOv2 ViT-S/14 (vendored arch under
#                       dust3r/vendor/dinov2), final-norm CLS token (384-d)
#   - "corr"          : hand-crafted correlation/flow statistics between the
#                       current and previous view's CUT3R pre-ray token sets
#                       (54-d, no learned params) — see corr_motion_stats().
#
# The encoders are feature EXTRACTORS, not trainable modules: every parameter is
# requires_grad=False and the module is locked in eval mode (train() is a no-op
# that keeps .training False — resnet BN must never see batch-stats mode or
# update its running stats). Their weights ARE part of the enclosing model's
# state_dict (pose_gru.img_encoder.*), so checkpoints are self-contained and
# eval nodes never need the pretrained files below.

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Width of the corr_motion_stats() feature: 16 cells x (mean dx, mean dy, mean m)
# + 6 global stats (mean/std of dx, dy, m).
CORR_FEAT_DIM = 54

# Output width of each frozen encoder (one block as the GRU cell sees it).
ENCODER_DIMS = {"resnet18": 512, "dinov2_vits14": 384}

# Pretrained init files (train-time only; eval restores from ckpt["model"]).
_WEIGHTS_FILES = {
    "resnet18": "resnet18-f37072fd.pth",
    "dinov2_vits14": "dinov2_vits14_pretrain.pth",
}
_WEIGHTS_URLS = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
}

# ImageNet normalization, applied after mapping CUT3R's [-1, 1] back to [0, 1].
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def default_weights_path(name):
    """Default local path of the pretrained init for encoder `name`
    (src/CUT3R/src/pretrained_encoders/<file>)."""
    assert name in _WEIGHTS_FILES, f"unknown encoder {name!r}"
    return Path(__file__).parent.parent / "pretrained_encoders" / _WEIGHTS_FILES[name]


class FrozenImageEncoder(nn.Module):
    """Frozen, eval-locked feature extractor mapping a CUT3R-normalized image
    (B, 3, H, W) in [-1, 1] to a (B, out_dim) fp32 detached feature vector."""

    def __init__(self, name, pretrained=True, weights_path=None):
        super().__init__()
        assert name in ENCODER_DIMS, (
            f"unknown encoder {name!r}, expected one of {sorted(ENCODER_DIMS)}"
        )
        self.name = name
        self.out_dim = ENCODER_DIMS[name]

        # pretrained=False => random init; used by load_model, where
        # ckpt["model"] overwrites every pose_gru.img_encoder.* key anyway.
        state_dict = None
        if pretrained:
            path = Path(weights_path) if weights_path is not None else default_weights_path(name)
            if not path.is_file():
                raise RuntimeError(
                    f"pretrained weights for {name!r} not found at {path}.\n"
                    f"Compute nodes have no internet — fetch once from a LOGIN node:\n"
                    f"    curl -L -o {default_weights_path(name)} {_WEIGHTS_URLS[name]}"
                )
            state_dict = torch.load(path, map_location="cpu", weights_only=True)

        if name == "resnet18":
            import torchvision

            net = torchvision.models.resnet18(weights=None)
            if state_dict is not None:
                net.load_state_dict(state_dict, strict=True)  # full net incl. fc
            # Drop the classifier AFTER loading: forward then returns the
            # (B, 512) avgpool output directly, and the surviving keys keep
            # their native torchvision names (no nn.Sequential renaming).
            net.fc = nn.Identity()
            self.net = net
        else:  # dinov2_vits14
            from .vendor.dinov2 import vit_small

            # Must match the official dinov2_vits14 hub model exactly.
            net = vit_small(patch_size=14, img_size=518, init_values=1.0, block_chunks=0)
            if state_dict is not None:
                missing, unexpected = net.load_state_dict(state_dict, strict=False)
                assert missing == [] and set(unexpected) <= {"mask_token"}, (
                    f"dinov2_vits14 weights mismatch: missing={missing}, unexpected={unexpected}"
                )
            self.net = net

        # Buffers (persistent=False: they are constants, NOT state — they must
        # not appear in state_dict / checkpoints).
        self.register_buffer(
            "imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

        # Freeze and lock in eval mode.
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode=True):
        # Eval-locked: parent .train() calls must never flip this module back
        # to training mode (resnet BN would use batch stats and update its
        # running estimates).
        return super().train(False)

    def forward(self, img):
        """img: (B, 3, H, W) fp32 in CUT3R's [-1, 1] normalization.
        Returns (B, out_dim) float32, detached."""
        # Caller may sit inside an fp16/bf16 autocast region: run the frozen
        # encoder in fp32, gradient-free, regardless.
        with torch.no_grad(), torch.autocast(device_type=img.device.type, enabled=False):
            x01 = img.float() * 0.5 + 0.5
            y = (x01 - self.imagenet_mean) / self.imagenet_std
            if self.name == "dinov2_vits14":
                H, W = y.shape[-2:]
                Ht = int(math.ceil(H / 14) * 14)
                Wt = int(math.ceil(W / 14) * 14)
                if (Ht, Wt) != (H, W):
                    # e.g. 192x320 -> 196x322 at the training resolution
                    y = F.interpolate(
                        y, size=(Ht, Wt), mode="bilinear", align_corners=False, antialias=True
                    )
                out = self.net.forward_features(y)["x_norm_clstoken"]
            else:
                out = self.net(y)  # fc is Identity => (B, 512) avgpool output
        return out.float().detach()


def corr_motion_stats(cur_tokens, prev_tokens, grid_hw, cells=(4, 4)):
    """Correlation/flow statistics between two views' CUT3R pre-ray token sets.

    cur_tokens/prev_tokens: (B, N, C) detached fp32 tokens, N == H'*W' with
    (H', W') = grid_hw ((12, 20) at 192x320, patch 16), row-major token order.
    For each current token, its best-matching previous token (cosine similarity
    argmax) defines a normalized displacement (dx, dy) on the token grid in the
    cur->prev direction (approximately negative optical flow) and a match
    confidence m. Returns (B, 54) float32:
      - 48 dims: mean dx, mean dy, mean m per cell of a `cells` (4x4) partition
        of the grid (ceil-divided blocks), row-major cell order, (dx, dy, m)
        fastest;
      - 6 dims: global mean/std (population) of dx, dy, m, appended as
        (mean_dx, std_dx, mean_dy, std_dy, mean_m, std_m).
    Pure tensor ops, no learned params; the argmax is non-differentiable, which
    is fine — the inputs are detached data by construction.

    KNOWN LIMIT of m: it is the max cosine similarity, not a two-peak match
    confidence. Textureless regions score m ~= 1 against MANY previous tokens,
    so their argmax displacement is arbitrary while m still reads "confident".
    The downstream GRU has to learn that per-cell (dx, dy) is only meaningful
    where the cell's tokens are distinctive; a peak-ratio confidence would fix
    this at the statistic level but changes CORR_FEAT_DIM — a new arm, not a
    patch to this one."""
    Hg, Wg = grid_hw
    B, N, C = cur_tokens.shape
    assert N == Hg * Wg, f"token count {N} != grid {Hg}x{Wg}"
    assert prev_tokens.shape == cur_tokens.shape, (
        f"shape mismatch: cur {tuple(cur_tokens.shape)} vs prev {tuple(prev_tokens.shape)}"
    )

    # The caller (the F-lever site in _forward_decoder_group_step) sits inside
    # the trainer's fp16 autocast but OUTSIDE the GRU block's fp32 island, and
    # matmul is autocast-listed (both the sim matrix and the cell-aggregation
    # below) — without this guard the statistic computes in fp16 during
    # training and fp32 at eval (autocast off), a silent train/eval feature
    # mismatch. Same island the frozen encoders open internally.
    with torch.autocast(device_type=cur_tokens.device.type, enabled=False):
        cur = F.normalize(cur_tokens.detach().float(), dim=2)
        prev = F.normalize(prev_tokens.detach().float(), dim=2)
        sim = cur @ prev.transpose(1, 2)  # (B, N, N)
        m, j = sim.max(dim=2)  # best prev match per current token, both (B, N)

        idx = torch.arange(N, device=cur.device)
        xs = (idx % Wg).float()  # (N,) token grid coords, row-major
        ys = (idx // Wg).float()
        dx = (xs[j] - xs) / Wg  # (B, N) normalized displacement, cur->prev
        dy = (ys[j] - ys) / Hg

        # Per-cell aggregation over a ceil-divided cells[0] x cells[1] partition.
        ch, cw = cells
        cell_h = -(-Hg // ch)  # ceil division
        cell_w = -(-Wg // cw)
        cell_id = (idx // Wg) // cell_h * cw + (idx % Wg) // cell_w  # (N,) in [0, ch*cw)
        onehot = F.one_hot(cell_id, num_classes=ch * cw).float()  # (N, cells)
        counts = onehot.sum(dim=0).clamp(min=1.0)  # (cells,)
        cell_means = torch.stack([dx, dy, m], dim=2).transpose(1, 2) @ onehot / counts
        # (B, 3, cells) -> row-major cell order with (dx, dy, m) fastest:
        per_cell = cell_means.transpose(1, 2).reshape(B, 3 * ch * cw)  # (B, 48)

        glob = torch.stack(
            [
                dx.mean(dim=1),
                dx.std(dim=1, correction=0),
                dy.mean(dim=1),
                dy.std(dim=1, correction=0),
                m.mean(dim=1),
                m.std(dim=1, correction=0),
            ],
            dim=1,
        )  # (B, 6)

        out = torch.cat([per_cell, glob], dim=1)
    assert out.shape == (B, CORR_FEAT_DIM)
    return out.float()
