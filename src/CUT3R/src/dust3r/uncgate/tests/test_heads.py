"""Tests for dust3r.uncgate.heads (plain asserts; run as a script on CPU)."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))  # .../CUT3R/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402

from dust3r.uncgate.heads import (  # noqa: E402
    PixelUncHead,
    TokenUncHead,
    align_image_tokens_to_state,
    gaussian_nll_2d,
    laplace_nll_2d,
    rope_token_map_local,
    scalar_score,
)

torch.manual_seed(0)
B, H, W, P, D = 2, 192, 320, 16, 768
PH, PW = H // P, W // P  # 12, 20
S = PH * PW  # 240
N_STATE = 768


def test_pixel_head_shapes_and_init():
    head = PixelUncHead(dec_dim=D, patch_size=P, out_ch=2)
    tokens = torch.randn(B, S, D)
    s = head(tokens, (H, W))
    assert s.shape == (B, 2, H, W), s.shape
    assert torch.isfinite(s).all()
    # small init -> s ~ 0 (sigma ~ 1 px)
    assert s.abs().max() < 0.1, s.abs().max()
    assert head.proj.fc1.in_features == D and head.proj.fc1.out_features == 4 * D
    assert head.proj.fc2.out_features == 2 * P * P


def test_pixel_head_reshape_matches_linear_head():
    """token k <-> patch (k // PW, k % PW); pixel_shuffle places channel c*P*P + dy*P + dx at (dy, dx)."""
    head = PixelUncHead(dec_dim=D, patch_size=P, out_ch=2)
    tokens = torch.randn(B, S, D)
    s = head(tokens, (H, W))
    feat = head.proj(tokens)  # (B, S, 2*P*P)
    k = 5 * PW + 7  # patch row 5, col 7
    for c in range(2):
        for dy, dx in [(0, 0), (3, 11), (15, 15)]:
            ref = feat[0, k, c * P * P + dy * P + dx]
            got = s[0, c, 5 * P + dy, 7 * P + dx]
            assert torch.allclose(ref, got), (c, dy, dx, ref, got)


def test_token_head_shapes():
    head = TokenUncHead(state_dim=D, img_dim=D, hidden=384, out_ch=2)
    st = torch.randn(B, N_STATE, D)
    img = torch.randn(B, N_STATE, D)
    flag = torch.ones(B, N_STATE, 1)
    s = head(st, img, flag)
    assert s.shape == (B, N_STATE, 2), s.shape
    assert torch.isfinite(s).all()
    assert s.abs().max() < 0.1


def test_nll_decreases_when_s_matches_variance():
    eps = torch.randn(B, H, W, 2) * 2.0  # eps ~ N(0, 4)
    valid = torch.ones(B, H, W, dtype=torch.bool)
    s0 = torch.zeros(B, H, W, 2)
    s_match = torch.full((B, H, W, 2), float(torch.log(torch.tensor(4.0))))
    l0 = gaussian_nll_2d(s0, eps, valid)
    l1 = gaussian_nll_2d(s_match, eps, valid)
    assert l1 < l0, (l1, l0)
    # analytic: at s=log(4): 2*(4/4 + log 4) = 2 + 2 log 4 ~ 4.77 ; at s=0: 2*(4 + 0) = 8
    assert abs(l1.item() - (2 + 2 * torch.log(torch.tensor(4.0)).item())) < 0.1, l1
    assert abs(l0.item() - 8.0) < 0.2, l0
    # laplace variant also improves toward the matched scale
    la0 = laplace_nll_2d(s0, eps, valid)
    la1 = laplace_nll_2d(s_match, eps, valid)
    assert la1 < la0, (la1, la0)
    # validity mask: no valid -> 0
    none = torch.zeros(B, H, W, dtype=torch.bool)
    assert gaussian_nll_2d(s0, eps, none).item() == 0.0
    assert laplace_nll_2d(s0, eps, none).item() == 0.0
    # mask actually excludes entries: huge eps on invalid pixels does not change the loss
    eps2 = eps.clone()
    valid2 = valid.clone()
    valid2[:, :10] = False
    eps2[:, :10] = 1e6
    assert torch.allclose(gaussian_nll_2d(s0, eps2, valid2), gaussian_nll_2d(s0, eps, valid2))
    # broadcasting: s (B,1,1,2) against eps (B,H,W,2)
    sb = torch.zeros(B, 1, 1, 2)
    assert torch.allclose(gaussian_nll_2d(sb, eps, valid), l0)


def test_gradient_flows_to_head_params():
    head = PixelUncHead(dec_dim=D, patch_size=P, out_ch=2)
    tokens = torch.randn(B, S, D)
    s = head(tokens, (H, W)).permute(0, 2, 3, 1)  # (B,H,W,2)
    eps = torch.randn(B, H, W, 2)
    valid = torch.rand(B, H, W) > 0.3
    loss = gaussian_nll_2d(s, eps, valid)
    loss.backward()
    for n, p in head.named_parameters():
        assert p.grad is not None, n
        assert torch.isfinite(p.grad).all(), n
    assert head.proj.fc2.weight.grad.abs().sum() > 0

    thead = TokenUncHead()
    st = torch.randn(B, N_STATE, D)
    img = torch.randn(B, N_STATE, D)
    flag = torch.ones(B, N_STATE, 1)
    s2 = thead(st, img, flag)
    loss2 = laplace_nll_2d(s2, torch.randn(B, N_STATE, 2), torch.ones(B, N_STATE, dtype=torch.bool))
    loss2.backward()
    for n, p in thead.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_scalar_score_ordering():
    s_conf = torch.tensor([[-2.0, -2.0]])  # small variance -> confident
    s_unc = torch.tensor([[3.0, 1.0]])  # large variance -> uncertain
    sc = scalar_score(torch.stack([s_conf, s_unc], 0))  # (2,1)
    assert sc.shape == (2, 1)
    assert sc[0, 0] > sc[1, 0]
    assert torch.allclose(sc[0, 0], torch.tensor(2.0)) and torch.allclose(sc[1, 0], torch.tensor(-2.0))
    s = torch.randn(B, H, W, 2)
    assert scalar_score(s).shape == (B, H, W)
    assert torch.allclose(scalar_score(s), -0.5 * (s[..., 0] + s[..., 1]))


def test_rope_map_local():
    on, r, c = rope_token_map_local(N_STATE, PH, PW)
    assert on.shape == (N_STATE,) and on.sum().item() == 240
    assert (r[0].item(), c[0].item()) == (0, 0)
    assert (r[28].item(), c[28].item()) == (1, 0)
    assert bool(on[0]) and bool(on[28]) and not bool(on[767]) and not bool(on[20])


def test_align_image_tokens_to_state():
    img = torch.randn(B, S, D)
    rows = torch.arange(S) // PW
    cols = torch.arange(S) % PW
    pos = torch.stack([rows, cols], -1).unsqueeze(0).expand(B, S, 2)
    aligned, flag = align_image_tokens_to_state(img, pos, N_STATE, PH, PW)
    assert aligned.shape == (B, N_STATE, D) and flag.shape == (B, N_STATE, 1)
    k10 = PW * 1 + 0  # image token index with pos (1,0)
    assert tuple(pos[0, k10].tolist()) == (1, 0)
    assert torch.equal(aligned[:, 28], img[:, k10])
    assert torch.equal(aligned[:, 0], img[:, 0])
    assert flag[:, 28, 0].eq(1).all() and flag[:, 0, 0].eq(1).all()
    assert aligned[:, 767].abs().sum() == 0 and flag[:, 767, 0].eq(0).all()
    assert aligned[:, 20].abs().sum() == 0 and flag[:, 20, 0].eq(0).all()  # col 20 is off-image
    assert flag.sum().item() == B * 240
    on, _, _ = rope_token_map_local(N_STATE, PH, PW)
    assert torch.equal(flag[0, :, 0].bool(), on)
    # a shuffled token order lands in the same places
    perm = torch.randperm(S)
    aligned2, flag2 = align_image_tokens_to_state(img[:, perm], pos[:, perm], N_STATE, PH, PW)
    assert torch.equal(aligned2, aligned) and torch.equal(flag2, flag)
    # out-of-range pos asserts
    bad = pos.clone()
    bad[0, 0, 0] = PH
    try:
        align_image_tokens_to_state(img, bad, N_STATE, PH, PW)
        raise RuntimeError("expected AssertionError")
    except AssertionError:
        pass
    # end-to-end into the token head
    head = TokenUncHead()
    st = torch.randn(B, N_STATE, D)
    assert head(st, aligned, flag).shape == (B, N_STATE, 2)


if __name__ == "__main__":
    test_pixel_head_shapes_and_init()
    test_pixel_head_reshape_matches_linear_head()
    test_token_head_shapes()
    test_nll_decreases_when_s_matches_variance()
    test_gradient_flows_to_head_params()
    test_scalar_score_ordering()
    test_rope_map_local()
    test_align_image_tokens_to_state()
    print("heads tests ok")
