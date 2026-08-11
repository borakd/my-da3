# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Vendored subset of the official DINOv2 repo
# (https://github.com/facebookresearch/dinov2, cached hub snapshot
# facebookresearch_dinov2_main) for CUT3R's frozen-encoder F-source lever:
# just the ViT backbone + its layers, with
#   - `from dinov2.layers import ...` rewritten to relative imports,
#   - xformers hard-disabled (plain-torch fallback paths only),
#   - the training-only dino_head dropped.
# The official dinov2_vits14 hub model is
#   vit_small(patch_size=14, img_size=518, init_values=1.0, block_chunks=0)
# and its published checkpoint loads with strict=False (extra key: mask_token).

from .models.vision_transformer import DinoVisionTransformer, vit_small  # noqa: F401
