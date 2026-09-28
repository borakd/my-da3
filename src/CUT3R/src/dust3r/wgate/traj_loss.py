"""Differentiable causal trajectory loss for the CONSEQUENCE-trained write gate (variant C).

Design + contract: ``WRITE_GATE_VARIANTS.md`` (repo root), section "Variant C: the CONSEQUENCE-trained
gate".  Variants 1 and 2 train a Kendall & Gal sigma that PREDICTS a frame's own error and use it as a
proxy for the value of that frame's write; their oracles closed that family.  Variant C drops the proxy:
the gate weight is trained by its CONSEQUENCE, i.e. the gradient of the trajectory error the benchmark
actually scores, flowing back through the memory and the frozen decoder.  This module is that error.

What it computes
----------------
``causal_ate(pred, gt)`` is the benchmark's ATE restricted to the frames seen SO FAR: the evaluator's
Sim(3) Umeyama alignment (scale included, fitted on the camera centres) followed by the RMSE of the
residual translations.  Called with frames ``0..t`` it is "ATE as of frame t", which is what a causal
gate can be held responsible for.  ``chunk_traj_loss`` averages it over the frames of one TBPTT chunk
(the pre-chunk history enters DETACHED, so the alignment sees the real drift context but no gradient
leaks past the chunk boundary), optionally plus a consecutive-frame (RPE-like) translation + rotation
term.

Agreement with the evaluator
----------------------------
``eval_bundle/bin/eval_depth_poses.py`` scores ATE either with its own ``_estimate_umeyama`` (Sim(3) on
the camera centres, ``align="sim3"``) or, under ``--eval_like_cut3r``, with evo's
``main_ape.ape(..., translation_part, align=True, correct_scale=True)``; both are the same Umeyama fit
and the same RMSE reduction.  ``umeyama_sim3`` here is a line-by-line torch port of
``_estimate_umeyama``, and ``tests/test_traj_loss.py`` asserts ``causal_ate`` over a real scene's stored
predictions against the per-frame ``ate`` column of that scene's ``eval_depth_pose_metrics.csv``
(RAIL+80edfcb1+2023-07-14-14h-28m-45s, 128 frames, ATE 0.025207676419: float64 agrees to 3e-18 and
per frame to 2e-13, float32 to 4e-10).

Precision
---------
Everything runs in float32 or better with autocast DISABLED (a bf16 SVD of the 3x3 covariance would cost
more accuracy than the whole alignment is worth).  float64 inputs are preserved, so the tests can
gradcheck in double.

The guard is this module's own responsibility, not the caller's: every autocast-eligible op here (the
four matmuls, in ``umeyama_sim3``, ``apply_sim3``, ``_svd_well_conditioned`` and ``_rpe_terms``) sits
inside a ``_no_autocast`` context, so ``causal_ate`` AND ``chunk_traj_loss`` return bit-identical
numbers inside and outside ``torch.autocast(..., bfloat16)``.  ``test_traj_loss.py`` asserts exactly
that, for the ATE term and for the RPE term.  (Before 2026-09-22 only ``causal_ate`` was covered: the
RPE term's ``dp @ R^T`` ran in bf16 under autocast and was wrong by ~5000x relative -- masked at the
time because ``train_wgate_consequence.py`` wraps the loss call in ``autocast(enabled=False)`` and the
eval worker uses no autocast, so no launched run produced a wrong number.)

Gradient through the alignment (DEPARTURE 7, see the Run log)
-------------------------------------------------------------
The contract says "``umeyama_sim3`` ... differentiable (``torch.linalg.svd``)" and it is: call it on
tensors that require grad and the gradient flows through the SVD.  But ``causal_ate`` does NOT use that
path by default, because it does not have to:

    ATE(x)^2 = (1/N) * min_T sum_i || T(x_i) - y_i ||^2 ,

i.e. the alignment is the argmin of the very quantity being differentiated.  By Danskin's envelope
theorem the total derivative equals the partial derivative at the optimum with T held FIXED -- the
``dT*/dx`` term is multiplied by ``d/dT`` of the objective, which is exactly zero at the optimum.  So
computing the Umeyama fit under ``no_grad`` on detached inputs and applying it as a constant similarity
gives the SAME gradient, while never running the SVD backward.  That matters because the SVD backward
carries ``1 / (sigma_i^2 - sigma_j^2)`` terms that are infinite whenever two singular values coincide --
which is exactly the degenerate case the contract asks us to survive (2 or 3 collinear camera centres,
i.e. a wrist that has not turned yet, which is the FIRST thing a causal loss sees at small t).

``align_grad="danskin"`` (default) is that path; ``align_grad="svd"`` forces the literal
differentiate-through-the-SVD path.  ``test_traj_loss.py`` checks both against
``torch.autograd.gradcheck`` and against each other (they agree to ~1e-12 on well-conditioned input),
and checks that only the default one survives collinear centres.

The envelope argument covers the ATE term ONLY.  The optional consecutive-frame (RPE) term is not the
quantity the Sim(3) minimises, so its alignment IS differentiated through (``RPE_ALIGN_GRAD = "svd"``);
dropping that term costs ~1% of the gradient, which the central-difference test catches.

Other numerical guards (all documented where they are used):
  * ``ridge``: the covariance gets ``ridge * max|cov| * I`` before the SVD so the forward is defined for
    an all-identical point set.  It shifts every singular value equally and therefore does NOT separate
    degenerate ones -- it is a forward guard, not a backward one (see above).
  * the Umeyama scale divides by the source variance, clamped at ``var_eps`` (and set to 1 below it,
    matching the evaluator).
  * ``sqrt`` is never taken at exactly zero: the mean squared error is clamped at ``ATE_EPS`` (a
    perfectly aligned trajectory would otherwise give an infinite ``d sqrt / d mse``), and vector norms
    use ``sqrt(sum x^2 + eps)``.
"""

import torch

__all__ = [
    "ATE_EPS",
    "DEFAULT_ALIGN_GRAD",
    "NORM_EPS",
    "RPE_ALIGN_GRAD",
    "RPE_DEGEN_TOL",
    "RIDGE",
    "VAR_EPS",
    "apply_sim3",
    "causal_ate",
    "chunk_traj_loss",
    "gt_centres",
    "gt_pose_encodings",
    "pred_centres",
    "pred_pose_encodings",
    "quat_geodesic",
    "sim3_align",
    "umeyama_sim3",
]

# --- numerical constants (see the module docstring) --------------------------------------------
RIDGE = 1e-12  # * max|cov|, added to the covariance diagonal before the SVD (forward guard)
VAR_EPS = 1e-15  # source-variance floor of the Umeyama scale (the evaluator's threshold)
ATE_EPS = 1e-16  # floor of the mean squared error before sqrt  -> ATE floor 1e-8
NORM_EPS = 1e-20  # inside sqrt() of every vector norm
DEFAULT_ALIGN_GRAD = "danskin"  # "danskin" (envelope theorem, exact + safe) | "svd" (literal)
RPE_ALIGN_GRAD = "svd"  # the RPE term is NOT the minimised objective -> Danskin does not apply to it
RPE_DEGEN_TOL = 1e-6  # relative singular-value gap below which the RPE term falls back to "danskin"


def _no_autocast(t):
    """Context manager: autocast OFF for the device of tensor ``t`` (cpu or cuda)."""
    return torch.autocast(device_type=t.device.type, enabled=False)


def _as_float(x):
    """Tensor in float32 or better (float64 preserved); non-tensors are converted."""
    x = torch.as_tensor(x)
    if x.dtype not in (torch.float32, torch.float64):
        x = x.float()
    return x


def _safe_norm(x, dim=-1, eps=NORM_EPS):
    """``sqrt(sum(x^2) + eps)`` -- equals ``x.norm()`` to ~1e-10 and is differentiable at x = 0."""
    return torch.sqrt((x * x).sum(dim=dim) + eps)


# ----------------------------------------------------------------------------- Sim(3) alignment
def umeyama_sim3(src, dst, with_scale=True, ridge=RIDGE, var_eps=VAR_EPS):
    """Least-squares similarity ``(s, R, t)`` with ``s * R @ src_i + t ~ dst_i``.

    A torch port of ``eval_bundle/bin/eval_depth_poses.py::_estimate_umeyama`` (Umeyama 1991), i.e.
    the SAME similarity the evaluator fits for ATE, with the scale estimated on the camera centres.
    Fully differentiable in ``src`` / ``dst`` through ``torch.linalg.svd``; read the module docstring
    before relying on that gradient (``causal_ate`` deliberately does not).

    Args:
        src: (..., N, 3) points to be moved (the PREDICTED camera centres).
        dst: (..., N, 3) targets (the GT camera centres).  Same shape as ``src``.
        with_scale: fit the scale too (Sim(3), the evaluator's default) or force ``s = 1`` (SE(3)).
        ridge: the covariance gets ``ridge * max|cov| * I`` before the SVD, so an all-identical point
            set still has a defined rotation.  Relative, so it is scale-free; at the default 1e-12 it
            moves ``R`` by ~2e-12 and the fitted ATE by far less than the 1e-6 the evaluator-agreement
            test allows (``ridge=0`` reproduces ``_estimate_umeyama`` bit-for-bit, checked in the
            tests).  It shifts all three singular values EQUALLY and therefore does NOT separate
            degenerate ones: it is a forward guard only, and the backward safety comes from
            ``align_grad="danskin"``.
        var_eps: below this source variance the scale is set to 1.0 (the evaluator's behaviour) and
            the division is clamped, so a static trajectory cannot produce inf/NaN.

    Returns:
        ``(s, R, t)`` with shapes ``(...)``, ``(..., 3, 3)``, ``(..., 3)``.  ``R`` is a proper rotation:
        the reflection case (``det(U V^T) < 0``, e.g. a mirrored trajectory) is corrected by flipping
        the sign of the last column, exactly as the evaluator does.  The sign itself is taken with
        ``sign(...).detach()`` -- it is piecewise constant, so its derivative is zero anyway, and
        detaching keeps the backward defined on the measure-zero set where the determinant vanishes.
    """
    src = _as_float(src)
    dst = _as_float(dst)
    if src.shape != dst.shape or src.dim() < 2 or src.shape[-1] != 3:
        raise ValueError(f"umeyama_sim3: bad shapes src={tuple(src.shape)} dst={tuple(dst.shape)}")
    n = src.shape[-2]
    if n < 1:
        raise ValueError("umeyama_sim3: need at least 1 point")
    if dst.dtype != src.dtype:  # promote both to the wider dtype
        dt = torch.promote_types(src.dtype, dst.dtype)
        src, dst = src.to(dt), dst.to(dt)

    with _no_autocast(src):
        mu_s = src.mean(dim=-2, keepdim=True)  # (..., 1, 3)
        mu_d = dst.mean(dim=-2, keepdim=True)
        sc = src - mu_s
        dc = dst - mu_d

        cov = dc.transpose(-1, -2) @ sc / n  # (..., 3, 3) = dst_c^T src_c / n
        if ridge and ridge > 0:
            eye = torch.eye(3, dtype=cov.dtype, device=cov.device)
            scale_ref = cov.detach().abs().amax(dim=(-2, -1), keepdim=True).clamp_min(var_eps)
            cov = cov + (ridge * scale_ref) * eye

        U, S, Vh = torch.linalg.svd(cov)
        det = torch.linalg.det(U) * torch.linalg.det(Vh)  # = det(U @ Vh)
        sgn = torch.sign(det).detach()
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        ones = torch.ones_like(sgn)
        d = torch.stack([ones, ones, sgn], dim=-1)  # (..., 3) = diag of the correction D
        R = (U * d.unsqueeze(-2)) @ Vh  # U @ diag(d) @ Vh

        if with_scale:
            var_src = (sc * sc).sum(dim=-1).mean(dim=-1)  # (...)
            trace_SD = (S * d).sum(dim=-1)  # trace(diag(S) @ D)
            s = trace_SD / var_src.clamp_min(var_eps)
            s = torch.where(var_src > var_eps, s, torch.ones_like(s))
        else:
            s = torch.ones_like(cov[..., 0, 0])

        t = mu_d.squeeze(-2) - s[..., None] * (R @ mu_s.squeeze(-2).unsqueeze(-1)).squeeze(-1)
    return s, R, t


def apply_sim3(s, R, t, x):
    """``s * R @ x_i + t`` for ``x`` of shape (..., N, 3); broadcasting over the leading dims.

    The matmul is autocast-eligible, so it runs under ``_no_autocast`` like every other matmul in this
    module -- a bf16 ``x @ R^T`` moves an aligned centre by ~3e-3, which is larger than the ATE itself.
    """
    with _no_autocast(x):
        return s[..., None, None] * (x @ R.transpose(-1, -2)) + t[..., None, :]


def sim3_align(pred, gt, align_grad=DEFAULT_ALIGN_GRAD, **kw):
    """The evaluator's Sim(3) from ``pred`` onto ``gt``; ``align_grad`` picks the gradient path.

    ``"danskin"`` (default): fit under ``no_grad`` on detached inputs, so the returned ``(s, R, t)`` are
    constants.  By the envelope theorem this loses NOTHING for any loss that is itself the minimised
    alignment error (``causal_ate``), and it avoids the SVD backward entirely -- see the module
    docstring.  ``"svd"``: the literal differentiable path through ``umeyama_sim3``.
    """
    if align_grad == "danskin":
        with torch.no_grad():
            return umeyama_sim3(pred.detach(), gt.detach(), **kw)
    if align_grad == "svd":
        return umeyama_sim3(pred, gt, **kw)
    raise ValueError(f"align_grad must be 'danskin' or 'svd', got {align_grad!r}")


# ----------------------------------------------------------------------------- causal ATE
def causal_ate(
    pred_centres,
    gt_centres,
    t=None,
    align_grad=DEFAULT_ALIGN_GRAD,
    reduce="mean",
    eps=ATE_EPS,
    ridge=RIDGE,
):
    """ATE "so far": RMSE of the residual translations after Sim(3)-aligning ``pred`` onto ``gt``.

    Called with frames ``0..t`` this is the ATE a causal system has earned by frame t; called with the
    whole trajectory it is the number in ``eval_depth_pose_metrics.csv`` (verified in the tests).

    Args:
        pred_centres: (..., N, 3) predicted camera centres (``pred["camera_pose"][:, :3]``).
        gt_centres:   (..., N, 3) GT camera centres in the same reference frame.  The metric is
            invariant to any rigid transform applied to BOTH and to any similarity applied to
            ``pred`` alone, so "same reference frame" is a convenience, not a requirement.
        t: optional INCLUSIVE last frame index -- frames ``0..t`` are used (``None`` = all).
        align_grad: "danskin" (default, exact and degeneracy-safe) or "svd"; see ``sim3_align``.
        reduce: "mean" -> scalar (mean over the leading/batch dims), "none" -> (...,) per trajectory.
        eps: the mean squared error is clamped at this value before ``sqrt``.  ``d sqrt(x) / dx`` is
            infinite at 0, and an exactly aligned trajectory is a realistic input (test (a)); with the
            clamp active ``clamp_min`` passes zero gradient, which is the correct limit.

    Returns: scalar tensor (``reduce="mean"``) or (...,) tensor, float32/float64, carrying grad.
    """
    p = _as_float(pred_centres)
    g = _as_float(gt_centres)
    if t is not None:
        p = p[..., : int(t) + 1, :]
        g = g[..., : int(t) + 1, :]
    with _no_autocast(p):
        s, R, tr = sim3_align(p, g, align_grad=align_grad, ridge=ridge)
        aligned = apply_sim3(s, R, tr, p)
        err2 = ((aligned - g) ** 2).sum(dim=-1)  # (..., N) squared per-frame ATE
        ate = torch.sqrt(err2.mean(dim=-1).clamp_min(eps))  # (...)
    if reduce == "mean":
        return ate.mean()
    if reduce == "none":
        return ate
    raise ValueError(f"reduce must be 'mean' or 'none', got {reduce!r}")


# ----------------------------------------------------------------------------- quaternions
def _q_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _q_mul(a, b):
    """Hamilton product, real part first -- the convention of ``dust3r.utils.camera`` (absT_quaR)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_geodesic(q1, q2, eps=NORM_EPS):
    """Geodesic angle in RADIANS between two unit quaternions (real part first), (..., 4) -> (...).

    ``angle = 2 * atan2(|vec(q1^-1 q2)|, |w(q1^-1 q2)|)`` -- the ``abs`` on w resolves the double cover
    (q and -q are the same rotation), giving a result in [0, pi].  Inputs are re-normalised first (the
    pose head's quaternion is not exactly unit); the norm uses ``sqrt(x^2 + eps)`` so the derivative
    stays finite at a zero rotation.
    """
    q1 = _as_float(q1)
    q2 = _as_float(q2)
    q1 = q1 / _safe_norm(q1).unsqueeze(-1)
    q2 = q2 / _safe_norm(q2).unsqueeze(-1)
    qe = _q_mul(_q_conj(q1), q2)
    w = qe[..., 0].abs().clamp(max=1.0)
    v = _safe_norm(qe[..., 1:], eps=eps)
    return 2.0 * torch.atan2(v, w)


# ----------------------------------------------------------------------------- chunk loss
def _split_centres_enc(x, name):
    """Accept (..., T, 3) centres or (..., T, 7) absT_quaR encodings -> (centres, enc_or_None)."""
    if x is None:
        return None, None
    x = _as_float(x)
    if x.shape[-1] == 3:
        return x, None
    if x.shape[-1] == 7:
        return x[..., :3], x
    raise ValueError(f"{name}: last dim must be 3 (centres) or 7 (absT_quaR), got {tuple(x.shape)}")


def _svd_well_conditioned(src, dst, ridge, tol=RPE_DEGEN_TOL):
    """Is the SVD BACKWARD of this Umeyama fit safe?  (Detached, cheap, returns a plain bool.)

    The SVD backward carries ``1 / (sigma_i^2 - sigma_j^2)``, so it is finite exactly while the three
    singular values of the covariance are distinct and nonzero.  This checks the smallest relative gap
    (and the smallest relative singular value) against ``tol``; a batch is safe only if every item is.
    A near-linear trajectory typically sits at 1e-3..1e-4 relative, i.e. a large-but-finite gradient
    that the trainer's ``clip_grad_norm_(1.0)`` handles; only genuine degeneracy trips this.
    """
    with torch.no_grad(), _no_autocast(src):
        src = src.detach()
        dst = dst.detach()
        n = src.shape[-2]
        sc = src - src.mean(dim=-2, keepdim=True)
        dc = dst - dst.mean(dim=-2, keepdim=True)
        cov = dc.transpose(-1, -2) @ sc / n
        if ridge and ridge > 0:
            eye = torch.eye(3, dtype=cov.dtype, device=cov.device)
            cov = cov + (ridge * cov.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(VAR_EPS)) * eye
        sv = torch.linalg.svdvals(cov)  # (..., 3), descending
        top = sv[..., :1].clamp_min(torch.finfo(sv.dtype).tiny)
        rel = sv / top
        gaps = torch.cat([rel[..., :-1] - rel[..., 1:], rel[..., -1:]], dim=-1)
        return bool(torch.isfinite(sv).all() and (gaps.min() > tol))


def _rpe_terms(full_p, full_g, full_pe, full_ge, lo, align_grad, ridge):
    """Mean consecutive-frame translation error (aligned) and rotation error (radians) over t >= lo.

    Definition (matches ``eval_pipeline/wgate_make_controls.py::gt_pose_errors_rel``, the rel-oracle of
    the same campaign, which in turn matches the evaluator's ``rpe_trans`` / ``rpe_rot`` up to its
    per-pair reduction):
      * translation: ``|| s R (C_t - C_{t-1}) - (G_t - G_{t-1}) ||`` with the Sim(3) fitted on ALL
        frames available here -- a wrong global scale or offset cancels, the per-step motion does not.
      * rotation: the geodesic angle between the predicted and GT frame-to-frame rotations, which is
        invariant to the alignment rotation (it cancels in ``R_{t-1}^T R_t``), so no encodings need to
        be transformed.

    IMPORTANT (``align_grad`` here is NOT the caller's default): Danskin's envelope theorem applies to
    ``causal_ate`` because the Sim(3) is the argmin of exactly that quantity.  It does NOT apply here --
    the alignment minimises the ABSOLUTE error, not this consecutive-frame one, so ``d(rpe)/dT*`` is
    nonzero and dropping it would give a systematically wrong gradient (measured: ~1% on a 12-frame
    chunk, caught by the central-difference test).  This term therefore differentiates THROUGH the
    Umeyama SVD by default (``rpe_align_grad="svd"``), which is also why it wants a non-degenerate
    chunk; pass ``rpe_align_grad="danskin"`` to trade that gradient term for degeneracy safety.
    """
    T = full_p.shape[-2]
    if lo < 1 or lo > T - 1:
        z = full_p.sum() * 0.0
        return z, z, 0, False
    # autocast OFF for the whole term, not just for the Umeyama fit: ``dp @ R^T`` below is an
    # autocast-eligible matmul, and computing the consecutive-frame translation error in bf16 is
    # wrong by orders of magnitude (measured 1.1e-6 -> 5.7e-3 on a 12-frame chunk).
    with _no_autocast(full_p):
        fallback = False
        if align_grad == "svd" and not _svd_well_conditioned(full_p, full_g, ridge):
            align_grad, fallback = "danskin", True  # a biased gradient beats a NaN; in stats
        s, R, tr = sim3_align(full_p, full_g, align_grad=align_grad, ridge=ridge)
        dp = full_p[..., lo:, :] - full_p[..., lo - 1 : -1, :]
        dg = full_g[..., lo:, :] - full_g[..., lo - 1 : -1, :]
        dp_al = s[..., None, None] * (dp @ R.transpose(-1, -2))  # translation only: no +t
        e_trans = _safe_norm(dp_al - dg)  # (..., K)

        qp_prev, qp = full_pe[..., lo - 1 : -1, 3:], full_pe[..., lo:, 3:]
        qg_prev, qg = full_ge[..., lo - 1 : -1, 3:], full_ge[..., lo:, 3:]
        q_rel_p = _q_mul(_q_conj(qp_prev), qp)
        q_rel_g = _q_mul(_q_conj(qg_prev), qg)
        e_rot = quat_geodesic(q_rel_g, q_rel_p)  # (..., K) radians
        return e_trans.mean(), e_rot.mean(), int(e_trans.shape[-1]), fallback


def chunk_traj_loss(
    pred_centres_chunk,
    gt_centres_chunk,
    prev_pred_centres=None,
    prev_gt_centres=None,
    ate_min_t=4,
    rpe_w=0.0,
    pred_enc_chunk=None,
    gt_enc_chunk=None,
    prev_pred_enc=None,
    prev_gt_enc=None,
    align_grad=DEFAULT_ALIGN_GRAD,
    rpe_align_grad=RPE_ALIGN_GRAD,
    eps=ATE_EPS,
    ridge=RIDGE,
):
    """The variant-C training loss on one TBPTT chunk: mean causal ATE, optionally + an RPE term.

    ``loss = mean_{t in chunk, t >= ate_min_t} causal_ate(centres[0..t])``
             ``+ rpe_w * (mean consecutive-frame translation error + mean rotation error)``

    where ``t`` is the GLOBAL frame index, ``len(prev) + i``.  The pre-chunk ("prev") centres are the
    history the rollout already committed; they enter the alignment DETACHED, so every ATE term sees the
    real accumulated drift while no gradient leaks past the TBPTT boundary.  Below ``ate_min_t`` the
    Sim(3) fit on 1-4 nearly collinear centres is degenerate and its ATE is ~0 by construction, so those
    frames contribute nothing but noise and are skipped (the default 4 is the contract's).

    Args:
        pred_centres_chunk: (..., Tc, 3) predicted centres of this chunk, WITH grad.  May instead be
            (..., Tc, 7) absT_quaR encodings, in which case the encodings are also used for the RPE
            rotation term and ``pred_enc_chunk`` is not needed.
        gt_centres_chunk: (..., Tc, 3) or (..., Tc, 7) GT, same reference frame and same length.
        prev_pred_centres / prev_gt_centres: (..., Tp, 3) or (..., Tp, 7) history, or None/empty for the
            first chunk.  Detached internally whatever is passed in.
        ate_min_t: skip chunk frames whose GLOBAL index is below this.
        rpe_w: weight of the consecutive-frame term; 0 (default) disables it.  When it is > 0 the 7-d
            encodings are REQUIRED (the rotation error has no definition without them) -- pass them
            either inside the centre arguments (last dim 7) or as the ``*_enc`` arguments; a missing
            encoding raises ValueError rather than silently dropping the rotation term.
        pred_enc_chunk / gt_enc_chunk / prev_pred_enc / prev_gt_enc: (..., T, 7) absT_quaR encodings,
            only needed when ``rpe_w > 0`` and the centre arguments are 3-d.
        align_grad / eps / ridge: passed through to ``causal_ate`` / ``sim3_align``.
        rpe_align_grad: the gradient path of the RPE term's OWN Sim(3) fit, "svd" by default because
            Danskin's theorem does not cover that term -- see ``_rpe_terms``.

    Returns:
        ``(loss, stats)``.  ``loss`` is a scalar tensor carrying grad (0 * the chunk's sum when no frame
        qualifies, so ``backward()`` is always safe).  ``stats`` is a dict of plain floats:
        ``ate`` (the mean ATE term), ``ate_last`` (the last qualifying frame's ATE, i.e. the newest
        causal ATE), ``n_ate``, ``rpe_trans``, ``rpe_rot`` (radians), ``rpe_rot_deg``, ``n_rpe``,
        ``rpe_align_fallback`` (1 when this chunk's Sim(3) was too degenerate to differentiate through
        and the RPE term used the stop-gradient fit instead -- worth logging as a rate),
        ``t_first`` / ``t_last`` (global index range of the chunk) and ``loss``.
    """
    p_chunk, pe_chunk = _split_centres_enc(pred_centres_chunk, "pred_centres_chunk")
    g_chunk, ge_chunk = _split_centres_enc(gt_centres_chunk, "gt_centres_chunk")
    if pe_chunk is None:
        _, pe_chunk = _split_centres_enc(pred_enc_chunk, "pred_enc_chunk")
    if ge_chunk is None:
        _, ge_chunk = _split_centres_enc(gt_enc_chunk, "gt_enc_chunk")
    if p_chunk.shape != g_chunk.shape:
        raise ValueError(
            f"chunk shapes differ: pred {tuple(p_chunk.shape)} gt {tuple(g_chunk.shape)}"
        )

    prev_p, prev_pe = _split_centres_enc(prev_pred_centres, "prev_pred_centres")
    prev_g, prev_ge = _split_centres_enc(prev_gt_centres, "prev_gt_centres")
    if prev_pe is None:
        _, prev_pe = _split_centres_enc(prev_pred_enc, "prev_pred_enc")
    if prev_ge is None:
        _, prev_ge = _split_centres_enc(prev_gt_enc, "prev_gt_enc")

    n_prev = 0
    full_p, full_g, full_pe, full_ge = p_chunk, g_chunk, pe_chunk, ge_chunk
    if prev_p is not None and prev_p.shape[-2] > 0:
        if prev_g is None or prev_g.shape != prev_p.shape:
            raise ValueError("prev_gt_centres must have the same shape as prev_pred_centres")
        n_prev = int(prev_p.shape[-2])
        full_p = torch.cat([prev_p.detach(), p_chunk], dim=-2)
        full_g = torch.cat([prev_g.detach(), g_chunk], dim=-2)
        if pe_chunk is not None and prev_pe is not None:
            full_pe = torch.cat([prev_pe.detach(), pe_chunk], dim=-2)
        else:
            full_pe = None
        if ge_chunk is not None and prev_ge is not None:
            full_ge = torch.cat([prev_ge.detach(), ge_chunk], dim=-2)
        else:
            full_ge = None

    n_chunk = int(p_chunk.shape[-2])
    stats = {"t_first": n_prev, "t_last": n_prev + n_chunk - 1}

    ates = []
    for i in range(n_chunk):
        tg = n_prev + i
        if tg < int(ate_min_t):
            continue
        ates.append(
            causal_ate(full_p, full_g, t=tg, align_grad=align_grad, eps=eps, ridge=ridge)
        )
    if ates:
        ate_stack = torch.stack(ates)
        ate_term = ate_stack.mean()
        stats["ate"] = float(ate_term.detach())
        stats["ate_last"] = float(ate_stack[-1].detach())
    else:
        ate_term = p_chunk.sum() * 0.0  # keeps the graph alive; value 0
        stats["ate"] = 0.0
        stats["ate_last"] = float("nan")
    stats["n_ate"] = len(ates)

    loss = ate_term
    if rpe_w:
        if full_pe is None or full_ge is None:
            raise ValueError(
                "chunk_traj_loss: rpe_w > 0 needs the 7-d absT_quaR encodings for every frame "
                "(chunk AND history) -- pass 7-d centre arguments or the *_enc arguments"
            )
        lo = max(1, n_prev)
        rpe_t, rpe_r, n_rpe, fallback = _rpe_terms(
            full_p, full_g, full_pe, full_ge, lo, rpe_align_grad, ridge
        )
        loss = loss + float(rpe_w) * (rpe_t + rpe_r)
        stats["rpe_trans"] = float(rpe_t.detach())
        stats["rpe_rot"] = float(rpe_r.detach())
        stats["rpe_rot_deg"] = float(rpe_r.detach()) * 180.0 / 3.141592653589793
        stats["n_rpe"] = n_rpe
        stats["rpe_align_fallback"] = int(fallback)
    stats["loss"] = float(loss.detach())
    return loss, stats


# ----------------------------------------------------------------------------- rollout helpers
def pred_pose_encodings(preds):
    """List of T prediction dicts -> (B, T, 7) absT_quaR, WITH grad, float32/64.

    ``pred["camera_pose"]`` is (B, 7) straight out of ``PoseDecoder`` + ``postprocess_pose``: the
    camera-to-world pose of the frame relative to frame 0, translation first.
    """
    return torch.stack([_as_float(p["camera_pose"]) for p in preds], dim=1)


def pred_centres(preds):
    """List of T prediction dicts -> (B, T, 3) predicted camera centres (``camera_pose[:, :3]``)."""
    return pred_pose_encodings(preds)[..., :3]


def gt_pose_encodings(views, ref_idx=0, ref_pose=None):
    """List of T view dicts -> (B, T, 7) GT absT_quaR relative to the reference frame, DETACHED.

    ``camera_to_pose_encoding(inv(ref) @ view["camera_pose"])``, the same expression
    ``dust3r/losses.py::get_all_pts3d`` uses for ``gt_poses`` (there with ``in_camera1``), so predictions
    and GT live in the same frame-0-relative convention.

    NOTE on chunked rollouts: pass ``ref_pose=batch[0]["camera_pose"]`` when ``views`` is only the
    current chunk, exactly as the criterion is called with ``camera1=batch[0]["camera_pose"]``.
    Otherwise ``views[ref_idx]`` is the CHUNK's first frame and each chunk gets its own reference.
    (ATE is invariant to that choice -- a different reference is a rigid transform of both trajectories
    -- but the RPE rotation term and any cross-chunk comparison are not, so be explicit.)

    No normalisation factor is applied: Sim(3) ATE is scale-invariant, so the loss does not need the
    ``?avg_dis`` point-norm the finetune pose loss divides by.
    """
    # dust3r.utils.camera cannot be imported before dust3r.heads (circular import); do it here so the
    # pure-math part of this module stays importable with nothing but torch.
    try:
        from dust3r.utils.camera import camera_to_pose_encoding
        from dust3r.utils.geometry import inv
    except ImportError:  # pragma: no cover - import-order fixup, as dust3r's own tests do it
        import dust3r.heads  # noqa: F401
        from dust3r.utils.camera import camera_to_pose_encoding
        from dust3r.utils.geometry import inv

    ref = _as_float(views[ref_idx]["camera_pose"] if ref_pose is None else ref_pose).detach()
    with torch.no_grad(), _no_autocast(ref):
        ref_inv = inv(ref)
        encs = [
            camera_to_pose_encoding(ref_inv @ _as_float(v["camera_pose"]).detach()) for v in views
        ]
        return torch.stack(encs, dim=1)


def gt_centres(views, ref_idx=0, ref_pose=None):
    """List of T view dicts -> (B, T, 3) GT camera centres relative to the reference frame."""
    return gt_pose_encodings(views, ref_idx=ref_idx, ref_pose=ref_pose)[..., :3]
