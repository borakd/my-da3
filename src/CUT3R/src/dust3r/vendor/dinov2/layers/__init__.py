# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Vendored for CUT3R: dino_head (training-only DINOHead) is not vendored, so its
# import is dropped; everything else matches the upstream package.
from .layer_scale import LayerScale
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused, SwiGLUFFNAligned
from .block import NestedTensorBlock, CausalAttentionBlock
from .attention import Attention, MemEffAttention
