"""uncgate Module A: SURE-Map backward-flow targets for the registration-uncertainty heads.

Everything here is training-side geometry: it turns (depth or predicted pts3d, poses,
intrinsics) into the *backward optical flow* from the current frame ``i`` to the previous
frame ``i-1``, so that a Kendall & Gal heteroscedastic head can be trained on the residual
between the flow induced by the frozen model's own pose+depth and the GT flow.

Conventions
-----------
* Images are ``(H, W)`` (``(192, 320)`` at train/eval; patch 16 -> ``ph=12, pw=20``).
* A pixel is ``u = (x, y)`` with ``x`` along ``W`` (column) and ``y`` along ``H`` (row);
  pixel centres sit at integer coordinates, ``u = (0, 0)`` is the top-left pixel.
* ``c2w`` matrices ``(B, 4, 4)`` map camera -> world; ``R_i = c2w_i[:3, :3]``,
  ``t_i = c2w_i[:3, 3]``.
* ``K`` is the ``(B, 3, 3)`` pinhole intrinsic of the *already resized/cropped* image.

Backward flow (SURE-Map)
------------------------
For a pixel ``u`` of the current frame ``i`` with depth ``D_i(u)``::

    p_cur  = D_i(u) * K_i^{-1} [u, 1]^T                      (3D point, current camera frame)
    p_prev = R_{i-1}^T (R_i p_cur + t_i - t_{i-1})           (same point, previous camera frame)
    f(u)   = pi(K_{i-1} p_prev) - u,   pi([x, y, z]) = (x/z, y/z)

so ``f(u)`` is the pixel displacement that carries ``u`` to where the same scene point was
imaged in frame ``i-1``. In matrix form ``p_prev = (c2w_{i-1}^{-1} c2w_i) p_cur``.

Because the model's depth and camera translation share ONE arbitrary global scale ``s``,
and ``pi`` is homogeneous (``pi(s K p) = pi(K p)``), the flow induced by the model's own
``pts3d_in_self_view`` and ``camera_pose`` is invariant to that scale: no depth/pose
alignment to GT is needed before comparing predicted and GT flow.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

# ``dust3r.utils.camera`` and ``dust3r.heads`` import each other; importing ``dust3r.heads``
# first is the order the rest of the code base relies on (``dust3r.model`` does the same).
import dust3r.heads  # noqa: F401  (resolves the camera <-> heads circular import)
from dust3r.utils.camera import pose_encoding_to_camera
from dust3r.utils.geometry import geotrf, inv

__all__ = [
    "backward_flow_from_points",
    "points_from_depth",
    "gt_backward_flow",
    "pred_backward_flow",
    "flow_residual",
    "pool_to_patches",
    "pool_to_patches_masked",
    "rope_token_map",
    "patches_to_tokens",
]

_Z_EPS = 1e-6
_INSIDE_TOL = 1e-3  # px


def _pixel_grid(H: int, W: int, device, dtype) -> torch.Tensor:
    """Return the ``(H, W, 2)`` grid of pixel coordinates ``u = (x, y)``, ``x`` along ``W``."""
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((xs, ys), dim=-1)


def backward_flow_from_points(
    p_cur: torch.Tensor,
    c2w_cur: torch.Tensor,
    c2w_prev: torch.Tensor,
    K_prev: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward flow ``f(u) = pi(K_prev p_prev) - u`` from 3D points in the current camera.

    ``p_prev = R_prev^T (R_cur p_cur + t_cur - t_prev) = (c2w_prev^{-1} c2w_cur) p_cur``.

    Args:
        p_cur:    ``(B, H, W, 3)`` 3D points in the CURRENT camera frame (any unit).
        c2w_cur:  ``(B, 4, 4)`` camera-to-world of the current frame.
        c2w_prev: ``(B, 4, 4)`` camera-to-world of the previous frame.
        K_prev:   ``(B, 3, 3)`` intrinsics of the previous frame's image.

    Returns:
        flow:  ``(B, H, W, 2)`` displacement in pixels ``(dx, dy)``.
        valid: ``(B, H, W)`` bool, ``z_prev > 1e-6`` and the projected point lies inside
               ``[0, W-1] x [0, H-1]`` (with a 1e-3 px round-off tolerance) and the flow is finite.
    """
    assert p_cur.ndim == 4 and p_cur.shape[-1] == 3, f"p_cur must be (B,H,W,3), got {tuple(p_cur.shape)}"
    B, H, W, _ = p_cur.shape
    dtype = p_cur.dtype
    rel = inv(c2w_prev.to(dtype)) @ c2w_cur.to(dtype)  # (B, 4, 4): current cam -> previous cam
    p_prev = geotrf(rel, p_cur)  # (B, H, W, 3)
    z_prev = p_prev[..., 2]
    proj = geotrf(K_prev.to(dtype), p_prev)  # (B, H, W, 3) homogeneous pixel coords
    z_safe = torch.where(z_prev.abs() > _Z_EPS, z_prev, torch.full_like(z_prev, _Z_EPS))
    uv_prev = proj[..., :2] / z_safe[..., None]
    grid = _pixel_grid(H, W, p_cur.device, dtype)[None]
    flow = uv_prev - grid
    # tolerance absorbs round-off at the image border (identity poses reproject
    # x = W-1 to 319.0000000001 or y = 0 to -2e-15); 1e-3 px is far below any real motion
    tol = _INSIDE_TOL
    inside = (
        (uv_prev[..., 0] >= -tol)
        & (uv_prev[..., 0] <= W - 1 + tol)
        & (uv_prev[..., 1] >= -tol)
        & (uv_prev[..., 1] <= H - 1 + tol)
    )
    valid = (z_prev > _Z_EPS) & inside & torch.isfinite(flow).all(dim=-1)
    flow = torch.where(torch.isfinite(flow), flow, torch.zeros_like(flow))
    return flow, valid


def points_from_depth(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Unproject a depth map with a pinhole ``K`` (no distortion).

    ``p(u) = D(u) * K^{-1} [x, y, 1]^T``, i.e. ``X = (x - cx) D / fx``, ``Y = (y - cy) D / fy``,
    ``Z = D`` (``K`` may also carry a skew term; the general ``K^{-1}`` is used).

    Args:
        depth: ``(B, H, W)``.
        K:     ``(B, 3, 3)``.
    Returns:
        ``(B, H, W, 3)`` points in the camera frame.
    """
    assert depth.ndim == 3, f"depth must be (B,H,W), got {tuple(depth.shape)}"
    B, H, W = depth.shape
    dtype = depth.dtype
    grid = _pixel_grid(H, W, depth.device, dtype)  # (H, W, 2)
    hom = torch.cat((grid, torch.ones_like(grid[..., :1])), dim=-1)[None].expand(B, H, W, 3)
    rays = geotrf(inv(K.to(dtype)), hom)  # (B, H, W, 3), z-component == 1
    return rays * depth[..., None]


def gt_backward_flow(
    depth_cur: torch.Tensor,
    K_cur: torch.Tensor,
    K_prev: torch.Tensor,
    c2w_cur: torch.Tensor,
    c2w_prev: torch.Tensor,
    valid_cur: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GT backward flow from GT depth + GT poses.

    ``flow_gt = backward_flow_from_points(points_from_depth(depth_cur, K_cur), c2w_cur, c2w_prev, K_prev)``
    and ``valid_gt = valid_proj & valid_cur & (depth_cur > 0)``.

    Args:
        depth_cur: ``(B, H, W)`` metric (or any-unit) GT depth of the current frame.
        K_cur, K_prev: ``(B, 3, 3)``.
        c2w_cur, c2w_prev: ``(B, 4, 4)`` GT camera-to-world (any common world frame).
        valid_cur: ``(B, H, W)`` bool GT validity mask of the current frame.
    Returns:
        ``(flow_gt (B,H,W,2), valid_gt (B,H,W) bool)``.
    """
    p_cur = points_from_depth(depth_cur, K_cur)
    flow, valid = backward_flow_from_points(p_cur, c2w_cur, c2w_prev, K_prev)
    valid = valid & valid_cur.bool() & (depth_cur > 0)
    return flow, valid


def pred_backward_flow(
    pts3d_self_cur: torch.Tensor,
    pose_enc_cur: torch.Tensor,
    pose_enc_prev: torch.Tensor,
    K_prev: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward flow induced by the frozen model's own ``pts3d_in_self_view`` and poses.

    The poses are ``(B, 7)`` ``absT_quaR`` encodings (``[t_x, t_y, t_z, q_w, q_x, q_y, q_z]``)
    decoded with :func:`dust3r.utils.camera.pose_encoding_to_camera` into ``c2w`` matrices in
    the view-0 frame. The model's depth and translation share one arbitrary scale and the
    flow is invariant to a common scale of ``(p_cur, t_cur, t_prev)``, so NO alignment to GT
    is performed here.

    Args:
        pts3d_self_cur: ``(B, H, W, 3)`` points in the current camera frame (model units).
        pose_enc_cur, pose_enc_prev: ``(B, 7)``.
        K_prev: ``(B, 3, 3)`` intrinsics of the previous frame's image.
    Returns:
        ``(flow_pred (B,H,W,2), valid_pred (B,H,W) bool)``.
    """
    c2w_cur = pose_encoding_to_camera(pose_enc_cur.to(pts3d_self_cur.dtype))
    c2w_prev = pose_encoding_to_camera(pose_enc_prev.to(pts3d_self_cur.dtype))
    return backward_flow_from_points(pts3d_self_cur, c2w_cur, c2w_prev, K_prev)


def flow_residual(
    flow_pred: torch.Tensor,
    flow_gt: torch.Tensor,
    valid_pred: torch.Tensor,
    valid_gt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Residual ``eps = flow_pred - flow_gt`` ``(B, H, W, 2)`` with ``valid = valid_pred & valid_gt``."""
    eps = flow_pred - flow_gt
    valid = valid_pred.bool() & valid_gt.bool()
    return eps, valid


def pool_to_patches(x: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    """``adaptive_avg_pool2d`` of ``x (B, C, H, W)`` over ``(H, W)`` -> ``(B, C, ph, pw)``."""
    assert x.ndim == 4, f"x must be (B,C,H,W), got {tuple(x.shape)}"
    return F.adaptive_avg_pool2d(x, (ph, pw))


def pool_to_patches_masked(
    x: torch.Tensor, valid: torch.Tensor, ph: int, pw: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Patch-average ``x`` over VALID pixels only.

    ``pooled[b, c, i, j] = sum_{u in patch(i,j)} valid(u) x(u) / max(sum_{u} valid(u), eps)`` and
    ``frac_valid[b, 0, i, j] = mean_{u in patch(i,j)} valid(u)``. Patches with no valid pixel
    get ``pooled = 0`` and ``frac_valid = 0`` (the caller decides what to do with them).

    Args:
        x:     ``(B, C, H, W)``.
        valid: ``(B, H, W)`` or ``(B, 1, H, W)`` bool/float.
    Returns:
        ``(pooled (B, C, ph, pw), frac_valid (B, 1, ph, pw))``.
    """
    assert x.ndim == 4, f"x must be (B,C,H,W), got {tuple(x.shape)}"
    if valid.ndim == 3:
        valid = valid[:, None]
    m = valid.to(x.dtype)
    frac_valid = F.adaptive_avg_pool2d(m, (ph, pw))  # (B, 1, ph, pw)
    num = F.adaptive_avg_pool2d(x * m, (ph, pw))  # (B, C, ph, pw)
    pooled = num / frac_valid.clamp(min=torch.finfo(x.dtype).tiny)
    pooled = torch.where(frac_valid > 0, pooled, torch.zeros_like(pooled))
    return pooled, frac_valid


def rope_token_map(n_state: int, ph: int, pw: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The state-token <-> image-patch alignment implied by the model's RoPE grid.

    ``w_pe = int(sqrt(n_state))``, ``+1`` if odd; state token ``i`` sits at RoPE position
    ``(i // w_pe, i % w_pe)``; image patches sit at ``(row, col)`` in the same units, so
    token ``i`` is *on-image* iff ``row < ph`` and ``col < pw``. For ``n_state=768, ph=12,
    pw=20``: ``w_pe = 28`` and exactly ``240`` tokens are on-image.

    Returns:
        ``on (n_state,) bool``, ``r_idx (n_state,) long``, ``c_idx (n_state,) long``.
    """
    w_pe = int(math.sqrt(n_state))
    w_pe = w_pe + 1 if w_pe % 2 == 1 else w_pe
    idx = torch.arange(n_state)
    r_idx, c_idx = idx // w_pe, idx % w_pe
    on = (r_idx < ph) & (c_idx < pw)
    return on, r_idx, c_idx


def patches_to_tokens(pooled: torch.Tensor, n_state: int) -> torch.Tensor:
    """Scatter a ``(B, C, ph, pw)`` patch grid onto the ``n_state`` state tokens.

    On-image tokens receive their patch value via :func:`rope_token_map`; off-image tokens
    (the global registers) are filled with ``NaN`` so the caller must decide how to treat them.

    Returns:
        ``(B, C, n_state)``.
    """
    assert pooled.ndim == 4, f"pooled must be (B,C,ph,pw), got {tuple(pooled.shape)}"
    B, C, ph, pw = pooled.shape
    on, r_idx, c_idx = rope_token_map(n_state, ph, pw)
    on, r_idx, c_idx = on.to(pooled.device), r_idx.to(pooled.device), c_idx.to(pooled.device)
    out = torch.full((B, C, n_state), float("nan"), dtype=pooled.dtype, device=pooled.device)
    out[:, :, on] = pooled[:, :, r_idx[on], c_idx[on]]
    return out
