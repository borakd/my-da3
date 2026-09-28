"""uncgate heads: per-pixel (v1) and per-token (v2) heteroscedastic registration-uncertainty heads.

Both heads sit on a FROZEN CUT3R and predict an unbounded 2-channel log-variance
``s = (s_u, s_v)`` of the backward optical flow induced by the model's own pose + depth.
They are trained with the Kendall & Gal heteroscedastic (aleatoric) Gaussian NLL
against the flow residual ``eps = flow_pred - flow_gt`` (the SURE-Map target).

Equations (per pixel / token, diagonal 2x2 Sigma = diag(exp(s_u), exp(s_v))):

    Gaussian NLL (K&G Eq. 7):
        L = eps_u^2 * exp(-s_u) + eps_v^2 * exp(-s_v) + s_u + s_v
    Laplace NLL (Sec. 4 variant, b = exp(s/2)):
        L = |eps_u| * exp(-s_u/2) + |eps_v| * exp(-s_v/2) + (s_u + s_v)/2
    Scalar confidence score (higher = more confident):
        score = -0.5 * (s_u + s_v)

Everything here is training-side; at inference the heads only consume features.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dust3r.utils.path_to_croco  # noqa: F401  (puts croco/ on sys.path, as linear_head.py does)
from models.blocks import Mlp  # noqa: E402  croco.models.blocks.Mlp

__all__ = [
    "PixelUncHead",
    "TokenUncHead",
    "gaussian_nll_2d",
    "laplace_nll_2d",
    "scalar_score",
    "align_image_tokens_to_state",
    "rope_token_map_local",
]

# Clamp range for s inside exp() -- numerical stability only; s itself is unbounded.
_S_CLAMP = 10.0


class PixelUncHead(nn.Module):
    """v1: per-pixel log-variance head.

    ``Mlp(dec_dim -> mlp_ratio*dec_dim -> out_ch*patch_size**2)`` with GELU, then the same
    token->pixel reshape as ``dust3r.heads.linear_head.LinearPts3d``:

        feat = proj(tokens)                                  # (B, S, out_ch*p*p)
        feat = feat.transpose(-1, -2).view(B, -1, H//p, W//p)  # (B, out_ch*p*p, H/p, W/p)
        s    = F.pixel_shuffle(feat, p)                      # (B, out_ch, H, W)

    Output is the RAW log-variance ``s``: no activation, no floor.  The last layer is
    initialised with zero bias and small weights (std 1e-3) so that initially
    ``s ~ 0``, i.e. ``sigma ~ 1 px``.
    """

    def __init__(self, dec_dim=768, patch_size=16, out_ch=2, mlp_ratio=4.0, hidden=None):
        super().__init__()
        self.dec_dim = dec_dim
        self.patch_size = patch_size
        self.out_ch = out_ch
        if hidden is None:
            hidden = int(mlp_ratio * dec_dim)
        self.hidden = hidden
        self.proj = Mlp(
            dec_dim,
            hidden_features=hidden,
            out_features=out_ch * patch_size**2,
            act_layer=nn.GELU,
        )
        nn.init.normal_(self.proj.fc2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.proj.fc2.bias)

    def forward(self, tokens, img_shape):
        """tokens (B, S, D) with S = (H/p)*(W/p); img_shape (H, W) -> s (B, out_ch, H, W)."""
        H, W = img_shape
        B, S, D = tokens.shape
        p = self.patch_size
        assert D == self.dec_dim, f"expected token dim {self.dec_dim}, got {D}"
        assert S == (H // p) * (W // p), f"S={S} != (H//p)*(W//p)={(H // p) * (W // p)}"
        feat = self.proj(tokens)  # (B, S, out_ch*p*p)
        feat = feat.transpose(-1, -2).view(B, -1, H // p, W // p)
        s = F.pixel_shuffle(feat, p)  # (B, out_ch, H, W)
        return s


class TokenUncHead(nn.Module):
    """v2: per-state-token log-variance head.

    Input per token: ``concat[state_tok (state_dim), img_tok (img_dim), on_flag (1)]``,
    LayerNorm on the concat, then ``MLP -> hidden -> hidden -> out_ch`` with GELU.
    For off-image state tokens ``img_tok`` is zeros and ``on_flag`` is 0
    (see :func:`align_image_tokens_to_state`).  Output is the RAW log-variance
    ``s`` (B, N, out_ch); last layer initialised to zero bias / small weights so ``s ~ 0``.
    """

    def __init__(self, state_dim=768, img_dim=768, hidden=384, out_ch=2):
        super().__init__()
        self.state_dim = state_dim
        self.img_dim = img_dim
        self.hidden = hidden
        self.out_ch = out_ch
        in_dim = state_dim + img_dim + 1
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_ch)
        self.act = nn.GELU()
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state_tokens, img_tokens_aligned, on_flag):
        """state_tokens (B,N,state_dim), img_tokens_aligned (B,N,img_dim), on_flag (B,N,1) -> s (B,N,out_ch)."""
        B, N, Ds = state_tokens.shape
        assert Ds == self.state_dim
        assert img_tokens_aligned.shape == (B, N, self.img_dim)
        assert on_flag.shape == (B, N, 1)
        x = torch.cat(
            [state_tokens, img_tokens_aligned, on_flag.to(state_tokens.dtype)], dim=-1
        )
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


def _masked_mean(per_elem, valid):
    """Mean of per_elem over entries where valid is True; 0 (with grad path) if none valid."""
    valid = valid.to(torch.bool)
    per_elem, valid = torch.broadcast_tensors(per_elem, valid)
    n = valid.sum()
    if n == 0:
        return per_elem.sum() * 0.0
    return per_elem[valid].sum() / n.to(per_elem.dtype)


def gaussian_nll_2d(s, eps, valid):
    """Diagonal 2D Gaussian NLL (SURE-Map / Kendall & Gal Eq. 7), mean over valid entries.

        L = eps_u^2 exp(-s_u) + eps_v^2 exp(-s_v) + s_u + s_v

    The ``s_u + s_v`` log-determinant term prevents the trivial infinite-variance solution.
    ``s`` and ``eps`` are broadcastable ``(B, ..., 2)``; ``valid`` is ``(B, ...)`` bool.
    ``s`` is clamped to ``[-10, 10]`` inside ``exp`` for numerical stability only.
    Returns 0 if no entry is valid.
    """
    s, eps = torch.broadcast_tensors(s, eps)
    s_c = s.clamp(-_S_CLAMP, _S_CLAMP)
    per = (eps**2 * torch.exp(-s_c)).sum(-1) + s.sum(-1)
    return _masked_mean(per, valid)


def laplace_nll_2d(s, eps, valid):
    """Diagonal 2D Laplace NLL (Sec. 4 variant, scale b = exp(s/2)), mean over valid entries.

        L = |eps_u| exp(-s_u/2) + |eps_v| exp(-s_v/2) + (s_u + s_v)/2
    """
    s, eps = torch.broadcast_tensors(s, eps)
    s_c = s.clamp(-_S_CLAMP, _S_CLAMP)
    per = (eps.abs() * torch.exp(-0.5 * s_c)).sum(-1) + 0.5 * s.sum(-1)
    return _masked_mean(per, valid)


def scalar_score(s):
    """``score = -0.5 * (s_u + s_v)`` -- log-precision-like; HIGHER = more confident.

    Matches the token gate's keep-if-large convention.  s (B, ..., 2) -> (B, ...).
    """
    return -0.5 * s.sum(-1)


def rope_token_map_local(n_state, ph, pw):
    """RoPE alignment of the model's state tokens to the image patch grid (local copy of
    ``flow_target.rope_token_map``, re-implemented to avoid a cross-module import).

        w_pe = int(sqrt(n_state)); w_pe += 1 if odd
        token i sits at (row, col) = (i // w_pe, i % w_pe)
        on = (row < ph) & (col < pw)

    For n_state=768, ph=12, pw=20 -> w_pe=28 and exactly 240 on-image tokens.
    Returns (on (n_state,) bool, r_idx (n_state,) long, c_idx (n_state,) long).
    """
    w_pe = int(math.sqrt(n_state))
    if w_pe % 2 == 1:
        w_pe += 1
    i = torch.arange(n_state)
    r_idx = i // w_pe
    c_idx = i % w_pe
    on = (r_idx < ph) & (c_idx < pw)
    return on, r_idx, c_idx


def align_image_tokens_to_state(img_tokens, pos, n_state, ph, pw):
    """Scatter image tokens onto the state-token index grid via the RoPE map.

    img_tokens (B, Nimg, D); pos (B, Nimg, 2) with rows = pos[..., 0], cols = pos[..., 1]
    (patch coordinates, asserted < (ph, pw)).  Image token at patch (r, c) lands at state
    index ``r * w_pe + c``.  Returns
        img_aligned (B, n_state, D): zeros at off-image state indices,
        on_flag     (B, n_state, 1): 1 where an image token was placed, else 0.
    """
    B, Nimg, D = img_tokens.shape
    assert pos.shape == (B, Nimg, 2), f"pos shape {tuple(pos.shape)} != {(B, Nimg, 2)}"
    pos = pos.long()
    assert bool((pos[..., 0] >= 0).all() and (pos[..., 0] < ph).all()), "row pos out of [0, ph)"
    assert bool((pos[..., 1] >= 0).all() and (pos[..., 1] < pw).all()), "col pos out of [0, pw)"
    on, r_idx, c_idx = rope_token_map_local(n_state, ph, pw)
    w_pe = int(math.sqrt(n_state))
    if w_pe % 2 == 1:
        w_pe += 1
    state_idx = pos[..., 0] * w_pe + pos[..., 1]  # (B, Nimg)
    assert bool((state_idx < n_state).all()), "computed state index >= n_state"

    img_aligned = img_tokens.new_zeros(B, n_state, D)
    on_flag = img_tokens.new_zeros(B, n_state, 1)
    img_aligned.scatter_(1, state_idx.unsqueeze(-1).expand(B, Nimg, D), img_tokens)
    on_flag.scatter_(1, state_idx.unsqueeze(-1), torch.ones(B, Nimg, 1, dtype=on_flag.dtype, device=on_flag.device))
    return img_aligned, on_flag
