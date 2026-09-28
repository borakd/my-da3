"""Tests for dust3r.uncgate.flow_target (plain asserts; run as a script, CPU only).

    OMP_NUM_THREADS=1 PYTHONPATH=src/CUT3R/src timeout 250 python src/CUT3R/src/dust3r/uncgate/tests/test_flow_target.py

Test (2), analytic derivation (pure translation along x, constant depth z, shared K)
-------------------------------------------------------------------------------------
Let ``R_cur = R_prev = I``, ``K_cur = K_prev = K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`` and
``D(u) = z`` for every pixel. Unprojection gives ``p_cur = z K^{-1}[x, y, 1]^T =
((x-cx) z/fx, (y-cy) z/fy, z)``. With ``t_cur = (tx, 0, 0)`` and ``t_prev = 0``:

    p_prev = R_prev^T (R_cur p_cur + t_cur - t_prev) = p_cur + (tx, 0, 0)
    pi(K p_prev) = (fx (X + tx)/z + cx,  fy Y/z + cy) = (x + fx tx / z,  y)
    f(u) = pi(K p_prev) - u = (+fx tx / z, 0)

i.e. a camera that moved by ``+tx`` (in the previous camera's axes) between ``i-1`` and
``i`` saw every scene point ``fx tx / z`` pixels further RIGHT in the previous image. The
mirror case ``t_cur = 0, t_prev = (tx, 0, 0)`` gives ``p_prev = p_cur - (tx, 0, 0)`` and
``f(u) = (-fx tx / z, 0)``; both signs are checked numerically to 1e-4 px. Depth-dependence
(``1/z``) is checked with two different constant depths.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # .../src/CUT3R/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import math  # noqa: E402

import torch  # noqa: E402

import dust3r.heads  # noqa: F401,E402  (resolves camera<->heads circular import)
from dust3r.uncgate.flow_target import (  # noqa: E402
    backward_flow_from_points,
    flow_residual,
    gt_backward_flow,
    patches_to_tokens,
    points_from_depth,
    pool_to_patches,
    pool_to_patches_masked,
    pred_backward_flow,
    rope_token_map,
)

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

B, H, W = 2, 192, 320
FX, FY, CX, CY = 250.0, 250.0, (W - 1) / 2, (H - 1) / 2


def make_K(b=B):
    K = torch.tensor([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
    return K[None].repeat(b, 1, 1)


def eye4(b=B):
    return torch.eye(4)[None].repeat(b, 1, 1)


def enc_from_c2w(c2w):
    """absT_quaR encoding (t_xyz, q_wxyz) of a (B,4,4) c2w with R already a rotation."""
    from dust3r.utils.camera import pose_encoding_to_camera  # noqa: F401

    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3]
    # quaternion from rotation matrix (w, x, y, z), valid for the rotations used here
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    qw = torch.sqrt((1 + tr).clamp(min=1e-12)) / 2
    qx = (R[:, 2, 1] - R[:, 1, 2]) / (4 * qw)
    qy = (R[:, 0, 2] - R[:, 2, 0]) / (4 * qw)
    qz = (R[:, 1, 0] - R[:, 0, 1]) / (4 * qw)
    return torch.cat((t, torch.stack((qw, qx, qy, qz), dim=-1)), dim=-1)


def rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# ---------------------------------------------------------------------------------------
# (1) identity poses + any depth -> zero flow, all valid
# ---------------------------------------------------------------------------------------
def test_identity_zero_flow():
    K = make_K()
    depth = 0.5 + 5.0 * torch.rand(B, H, W)
    flow, valid = gt_backward_flow(depth, K, K, eye4(), eye4(), torch.ones(B, H, W, dtype=torch.bool))
    assert flow.shape == (B, H, W, 2) and valid.shape == (B, H, W)
    assert valid.all(), "identity poses must give all-valid flow"
    assert flow.abs().max().item() < 1e-9, f"identity flow not zero: {flow.abs().max().item()}"
    # same through the pts3d path
    p = points_from_depth(depth, K)
    assert torch.allclose(p[..., 2], depth)
    flow2, valid2 = backward_flow_from_points(p, eye4(), eye4(), K)
    assert valid2.all() and flow2.abs().max().item() < 1e-9


# ---------------------------------------------------------------------------------------
# (2) pure x-translation, constant depth: flow = (+/- fx*tx/z, 0)  (derivation in docstring)
# ---------------------------------------------------------------------------------------
def test_pure_translation_x():
    K = make_K()
    tx = 0.2
    for z in (2.0, 5.0):
        depth = torch.full((B, H, W), z)
        expected = FX * tx / z
        # camera moved by +tx: previous image saw points +fx*tx/z to the right
        c2w_cur = eye4()
        c2w_cur[:, 0, 3] = tx
        flow, valid = gt_backward_flow(depth, K, K, c2w_cur, eye4(), torch.ones(B, H, W, dtype=torch.bool))
        err = (flow[..., 0] - expected).abs().max().item()
        assert err < 1e-4, f"x-flow mismatch (t_cur=+tx, z={z}): {err}"
        assert flow[..., 1].abs().max().item() < 1e-4
        # valid exactly where the shifted pixel stays inside the image
        xs = torch.arange(W, dtype=torch.get_default_dtype())
        inside = (xs + expected <= W - 1)[None, None].expand(B, H, W)
        assert torch.equal(valid, inside), "valid mask must be the inside-image mask"
        # mirror case: previous camera at +tx -> flow = -fx*tx/z
        c2w_prev = eye4()
        c2w_prev[:, 0, 3] = tx
        flow_m, _ = gt_backward_flow(depth, K, K, eye4(), c2w_prev, torch.ones(B, H, W, dtype=torch.bool))
        err_m = (flow_m[..., 0] + expected).abs().max().item()
        assert err_m < 1e-4, f"x-flow mismatch (t_prev=+tx, z={z}): {err_m}"
        assert flow_m[..., 1].abs().max().item() < 1e-4


# ---------------------------------------------------------------------------------------
# (3) predicted vs GT with identical inputs -> zero residual; common scale 3.7 -> same flow
# ---------------------------------------------------------------------------------------
def test_pred_equals_gt_and_scale_invariance():
    K = make_K()
    depth = 1.0 + 3.0 * torch.rand(B, H, W)
    c2w_cur = eye4()
    c2w_cur[:, :3, :3] = rot_y(0.03)
    c2w_cur[:, :3, 3] = torch.tensor([0.05, -0.02, 0.04])
    c2w_prev = eye4()
    c2w_prev[:, :3, :3] = rot_y(-0.02)
    c2w_prev[:, :3, 3] = torch.tensor([-0.03, 0.01, 0.0])
    valid_cur = torch.ones(B, H, W, dtype=torch.bool)

    flow_gt, valid_gt = gt_backward_flow(depth, K, K, c2w_cur, c2w_prev, valid_cur)
    assert valid_gt.float().mean().item() > 0.5, "test motion should keep most pixels valid"

    pts = points_from_depth(depth, K)
    enc_cur, enc_prev = enc_from_c2w(c2w_cur), enc_from_c2w(c2w_prev)
    # encoding round-trip sanity
    from dust3r.utils.camera import pose_encoding_to_camera

    assert torch.allclose(pose_encoding_to_camera(enc_cur), c2w_cur, atol=1e-9)
    flow_pred, valid_pred = pred_backward_flow(pts, enc_cur, enc_prev, K)
    eps, valid = flow_residual(flow_pred, flow_gt, valid_pred, valid_gt)
    assert torch.equal(valid, valid_gt & valid_pred)
    assert eps[valid].abs().max().item() < 1e-8, f"identical inputs must give zero residual: {eps[valid].abs().max()}"

    # common scale s=3.7 on pts3d AND both translations leaves flow unchanged
    s = 3.7
    enc_cur_s, enc_prev_s = enc_cur.clone(), enc_prev.clone()
    enc_cur_s[:, :3] *= s
    enc_prev_s[:, :3] *= s
    flow_s, valid_s = pred_backward_flow(pts * s, enc_cur_s, enc_prev_s, K)
    assert torch.equal(valid_s, valid_pred)
    d = (flow_s - flow_pred)[valid_pred].abs().max().item()
    assert d < 1e-8, f"flow must be scale invariant, diff={d}"
    # ... and scaling only the points DOES change it (guards against a trivially-zero test)
    flow_bad, _ = pred_backward_flow(pts * s, enc_cur, enc_prev, K)
    assert (flow_bad - flow_pred)[valid_pred].abs().max().item() > 1e-2


# ---------------------------------------------------------------------------------------
# (4) rope_token_map(768,12,20): 240 on-image tokens; token 0 <-> (0,0), token 28 <-> (1,0)
# ---------------------------------------------------------------------------------------
def test_rope_token_map():
    on, r_idx, c_idx = rope_token_map(768, 12, 20)
    assert on.shape == (768,) and on.dtype == torch.bool
    assert int(on.sum()) == 240, f"expected 240 on-image tokens, got {int(on.sum())}"
    assert on[0] and (r_idx[0], c_idx[0]) == (0, 0)
    assert on[28] and (r_idx[28], c_idx[28]) == (1, 0)
    assert not on[20] and (r_idx[20], c_idx[20]) == (0, 20)  # col 20 is off-image (pw=20)
    assert not on[767]  # 767 // 28 = 27 >= ph
    # patches_to_tokens scatters (r, c) -> token r*28 + c and leaves off-image NaN
    pooled = torch.arange(12 * 20, dtype=torch.get_default_dtype()).reshape(1, 1, 12, 20)
    tok = patches_to_tokens(pooled, 768)
    assert tok.shape == (1, 1, 768)
    assert tok[0, 0, 0].item() == 0.0 and tok[0, 0, 28].item() == 20.0 and tok[0, 0, 28 + 5].item() == 25.0
    assert torch.isnan(tok[0, 0, ~on]).all() and torch.isfinite(tok[0, 0, on]).all()


# ---------------------------------------------------------------------------------------
# (5) masked pooling ignores invalid pixels
# ---------------------------------------------------------------------------------------
def test_masked_pooling():
    ph, pw = 12, 20
    x = torch.rand(B, 2, H, W)
    valid = torch.rand(B, H, W) > 0.3
    # kill one whole patch (rows 16..31, cols 32..47) -> patch (1,2) has no valid pixel
    valid[:, 16:32, 32:48] = False
    pooled, frac = pool_to_patches_masked(x, valid, ph, pw)
    assert pooled.shape == (B, 2, ph, pw) and frac.shape == (B, 1, ph, pw)
    # brute-force reference on a few patches
    for (i, j) in [(0, 0), (3, 7), (11, 19), (1, 2)]:
        xs, ms = x[:, :, 16 * i:16 * (i + 1), 16 * j:16 * (j + 1)], valid[:, None, 16 * i:16 * (i + 1), 16 * j:16 * (j + 1)]
        n = ms.sum(dim=(2, 3))  # (B,1)
        ref = (xs * ms).sum(dim=(2, 3)) / n.clamp(min=1)
        ref = torch.where(n > 0, ref, torch.zeros_like(ref))
        assert torch.allclose(pooled[:, :, i, j], ref, atol=1e-10), f"patch {(i, j)} mismatch"
        assert torch.allclose(frac[:, 0, i, j], n[:, 0] / 256.0)
    assert (frac[:, 0, 1, 2] == 0).all() and (pooled[:, :, 1, 2] == 0).all()
    # invalid pixels really are ignored: overwrite them with garbage, pooled must not change
    x_g = torch.where(valid[:, None], x, torch.full_like(x, 1e6))
    pooled_g, _ = pool_to_patches_masked(x_g, valid, ph, pw)
    assert torch.allclose(pooled_g, pooled, atol=1e-8)
    # unmasked pooling with an all-valid mask equals plain pooling
    p_all, f_all = pool_to_patches_masked(x, torch.ones(B, H, W, dtype=torch.bool), ph, pw)
    assert torch.allclose(p_all, pool_to_patches(x, ph, pw), atol=1e-10) and (f_all == 1).all()


if __name__ == "__main__":
    test_identity_zero_flow()
    test_pure_translation_x()
    test_pred_equals_gt_and_scale_invariance()
    test_rope_token_map()
    test_masked_pooling()
    print("flow_target tests ok")
