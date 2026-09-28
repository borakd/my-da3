"""Tests for dust3r.wgate.traj_loss (plain asserts; CPU only; run as a script or under pytest).

    cd src/CUT3R/src && OMP_NUM_THREADS=1 timeout 280 python dust3r/wgate/tests/test_traj_loss.py

Covers the contract of WRITE_GATE_VARIANTS.md "Variant C" / component B:
  (0) umeyama_sim3 against a numpy transcription of the evaluator's ``_estimate_umeyama``, including
      the reflection (det < 0) branch and the recovery of a known (s, R, t);
  (a) EXACTNESS      -- pred = s*R*gt + t with random s, R, t  =>  causal_ate ~ 0 (< 1e-5);
  (b) SCALE INVARIANCE -- multiplying every predicted centre by 7 leaves the loss unchanged;
  (c) GRADCHECK      -- torch.autograd.gradcheck on causal_ate (8 frames, float64, BOTH align_grad
      paths) + a central-difference check of d(chunk_traj_loss)/d(one frame's centre);
  (d) AGREEMENT WITH THE EVALUATOR -- causal_ate over all 128 frames of a real scene vs the RMSE of
      the per-frame ``ate`` column of that scene's eval_depth_pose_metrics.csv;
  (e) DEGENERACY     -- 2 or 3 collinear (and identical) centres: no NaN/inf in forward OR backward;
plus: the two gradient paths agree, the TBPTT history really is detached, ate_min_t skipping, the RPE
term (zero on a perfect prediction, local support, alignment invariance, missing-encoding error), the
quaternion helpers against dust3r.utils.camera, the rollout helpers on synthetic preds/views, and
AUTOCAST INERTNESS -- every stat, the loss and the gradient of chunk_traj_loss are bit-identical inside
and outside torch.autocast(bfloat16) (the RPE term used to be wrong there by ~5000x relative).
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))  # .../CUT3R/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dust3r.wgate.traj_loss import (  # noqa: E402
    ATE_EPS,
    _q_conj,
    _q_mul,
    apply_sim3,
    causal_ate,
    chunk_traj_loss,
    gt_centres,
    gt_pose_encodings,
    pred_centres,
    pred_pose_encodings,
    quat_geodesic,
    umeyama_sim3,
)

torch.manual_seed(0)

SCENE = "RAIL+80edfcb1+2023-07-14-14h-28m-45s"
PRED_DIR = f"/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/augfull_lr1e5/preds/{SCENE}/camera"
GT_DIR = (
    "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist/"
    f"{SCENE}/dense/cam"
)
CSV = f"/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/augfull_lr1e5/eval/{SCENE}/eval_depth_pose_metrics.csv"


# ----------------------------------------------------------------------------- helpers
def _rand_rotation(seed=None, dtype=torch.float64):
    """A proper rotation matrix from the QR of a random matrix."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    A = torch.randn(3, 3, generator=g, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R))  # fix the QR sign ambiguity
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _numpy_umeyama(src, dst, with_scale=True):
    """Verbatim transcription of eval_bundle/bin/eval_depth_poses.py::_estimate_umeyama."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = (dst_c.T @ src_c) / max(n, 1)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    if with_scale:
        var_src = np.mean(np.sum(src_c * src_c, axis=1))
        scale = 1.0 if var_src <= 1e-15 else float(np.trace(np.diag(S) @ D) / var_src)
    else:
        scale = 1.0
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def _synthetic_traj(T=24, seed=0, dtype=torch.float64):
    """A GT trajectory (smooth-ish random walk) and a perturbed prediction of it."""
    g = torch.Generator().manual_seed(seed)
    steps = torch.randn(T, 3, generator=g, dtype=dtype) * 0.1
    gt = torch.cumsum(steps, dim=0)
    gt = gt - gt[0]
    noise = torch.randn(T, 3, generator=g, dtype=dtype) * 0.02
    pred = gt + noise
    return gt, pred


def _rand_quats(T, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T, 4, generator=g, dtype=dtype)
    return q / q.norm(dim=-1, keepdim=True)


# ----------------------------------------------------------------------------- (0) umeyama
def test_umeyama_recovers_known_transform():
    torch.manual_seed(3)
    src = torch.randn(40, 3, dtype=torch.float64)
    R_true = _rand_rotation(seed=7)
    s_true = torch.tensor(2.75, dtype=torch.float64)
    t_true = torch.tensor([0.3, -1.2, 5.0], dtype=torch.float64)
    dst = s_true * (src @ R_true.T) + t_true
    s, R, t = umeyama_sim3(src, dst)
    assert abs(float(s) - float(s_true)) < 1e-8, (float(s), float(s_true))
    assert torch.allclose(R, R_true, atol=1e-8), (R, R_true)
    assert torch.allclose(t, t_true, atol=1e-8), (t, t_true)
    assert abs(float(torch.det(R)) - 1.0) < 1e-12
    print("(0) umeyama recovers a known Sim(3): s err %.2e, R err %.2e" % (
        abs(float(s) - float(s_true)), float((R - R_true).abs().max())))


def test_umeyama_matches_evaluator_numpy():
    torch.manual_seed(4)
    for seed in (0, 1, 2):
        g = torch.Generator().manual_seed(seed)
        src = torch.randn(37, 3, generator=g, dtype=torch.float64)
        dst = torch.randn(37, 3, generator=g, dtype=torch.float64) * 3.0 + 1.0
        # ridge = 0 must reproduce the evaluator bit-for-bit (it IS the same arithmetic)
        s, R, t = umeyama_sim3(src, dst, ridge=0.0)
        s_np, R_np, t_np = _numpy_umeyama(src.numpy(), dst.numpy())
        assert abs(float(s) - s_np) < 1e-14, (float(s), s_np)
        assert np.abs(R.numpy() - R_np).max() < 1e-14
        assert np.abs(t.numpy() - t_np).max() < 1e-14
        # the default ridge is a negligible perturbation of it (amplified by close singular values,
        # hence the looser bound -- the rotation of two nearly equal singular values is ill-posed)
        sd, Rd, td = umeyama_sim3(src, dst)
        assert abs(float(sd) - s_np) < 1e-8 and np.abs(Rd.numpy() - R_np).max() < 1e-8
        assert np.abs(td.numpy() - t_np).max() < 1e-8
        # SE(3) branch too
        s2, R2, t2 = umeyama_sim3(src, dst, with_scale=False, ridge=0.0)
        s_np2, R_np2, t_np2 = _numpy_umeyama(src.numpy(), dst.numpy(), with_scale=False)
        assert float(s2) == 1.0 and np.abs(R2.numpy() - R_np2).max() < 1e-14
        assert np.abs(t2.numpy() - t_np2).max() < 1e-14
    print("(0) umeyama_sim3(ridge=0) == the evaluator's _estimate_umeyama (sim3 + se3) to < 1e-14; "
          "the default ridge moves it by < 1e-8")


def test_umeyama_reflection_branch():
    """A mirrored target must still give a PROPER rotation (the det correction fires)."""
    torch.manual_seed(5)
    src = torch.randn(30, 3, dtype=torch.float64)
    dst = src.clone()
    dst[:, 2] = -dst[:, 2]  # mirror: the unconstrained Procrustes solution has det = -1
    s, R, t = umeyama_sim3(src, dst, ridge=0.0)
    s_np, R_np, t_np = _numpy_umeyama(src.numpy(), dst.numpy())
    assert float(torch.det(R)) > 0.99, float(torch.det(R))
    assert np.abs(R.numpy() - R_np).max() < 1e-12, "reflection branch differs from the evaluator"
    assert np.abs(t.numpy() - t_np).max() < 1e-12
    print("(0) reflection branch: det(R) = %.6f, matches the evaluator" % float(torch.det(R)))


def test_umeyama_batched():
    torch.manual_seed(6)
    src = torch.randn(2, 5, 20, 3, dtype=torch.float64)
    dst = torch.randn(2, 5, 20, 3, dtype=torch.float64)
    s, R, t = umeyama_sim3(src, dst)
    assert s.shape == (2, 5) and R.shape == (2, 5, 3, 3) and t.shape == (2, 5, 3)
    s0, R0, t0 = umeyama_sim3(src[1, 3], dst[1, 3])
    assert abs(float(s[1, 3]) - float(s0)) < 1e-12
    assert torch.allclose(R[1, 3], R0, atol=1e-12) and torch.allclose(t[1, 3], t0, atol=1e-12)
    a = apply_sim3(s, R, t, src)
    assert a.shape == src.shape
    print("(0) batched umeyama / apply_sim3 shapes + per-item agreement OK")


# ----------------------------------------------------------------------------- (a) exactness
def test_a_exactness():
    gt, _ = _synthetic_traj(T=24, seed=11)
    for seed in (0, 1, 2, 3):
        g = torch.Generator().manual_seed(seed)
        R = _rand_rotation(seed=100 + seed)
        s = torch.rand(1, generator=g, dtype=torch.float64) * 4.9 + 0.1
        t = torch.randn(3, generator=g, dtype=torch.float64) * 2.0
        pred = s * (gt @ R.T) + t
        a = causal_ate(pred, gt)
        assert float(a) < 1e-5, (seed, float(a))
        loss, st = chunk_traj_loss(pred, gt, ate_min_t=4)
        assert float(loss) < 1e-5, (seed, float(loss))
        assert st["n_ate"] == 24 - 4
        # with an RPE term as well (needs encodings): rotate the quaternions by the same R
        q = _rand_quats(24, seed=7)
        enc_gt = torch.cat([gt, q], dim=-1)
        qR = _q_mul(_rot_to_quat(R).expand_as(q), q)
        enc_pred = torch.cat([pred, qR], dim=-1)
        loss2, st2 = chunk_traj_loss(enc_pred, enc_gt, ate_min_t=4, rpe_w=0.1)
        assert float(loss2) < 1e-5, (seed, float(loss2), st2)
        assert st2["rpe_trans"] < 1e-7 and st2["rpe_rot"] < 1e-7, st2
    print("(a) exactness: causal_ate < 1e-5 and the full loss (ate + rpe) < 1e-5 on 4 random Sim(3)s")


def _rot_to_quat(R):
    """(3,3) proper rotation -> (1,4) unit quaternion, real part first (Shepperd, w-branch is enough
    for the well-conditioned random rotations used here; falls back to the largest-diagonal branch)."""
    R = R.double()
    tr = float(R.trace())
    if tr > 0:
        w = math.sqrt(1.0 + tr) / 2.0
        x = float(R[2, 1] - R[1, 2]) / (4 * w)
        y = float(R[0, 2] - R[2, 0]) / (4 * w)
        z = float(R[1, 0] - R[0, 1]) / (4 * w)
    else:
        i = int(torch.argmax(torch.diagonal(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        r = math.sqrt(1.0 + float(R[i, i] - R[j, j] - R[k, k]))
        v = [0.0, 0.0, 0.0]
        v[i] = r / 2
        v[j] = float(R[j, i] + R[i, j]) / (2 * r)
        v[k] = float(R[k, i] + R[i, k]) / (2 * r)
        w = float(R[k, j] - R[j, k]) / (2 * r)
        x, y, z = v
    q = torch.tensor([[w, x, y, z]], dtype=torch.float64)
    return q / q.norm()


# ----------------------------------------------------------------------------- (b) scale invariance
def test_b_scale_invariance():
    gt, pred = _synthetic_traj(T=32, seed=12)
    a1 = float(causal_ate(pred, gt))
    a7 = float(causal_ate(pred * 7.0, gt))
    assert abs(a1 - a7) < 1e-12 * max(1.0, a1), (a1, a7)
    l1, s1 = chunk_traj_loss(pred[8:], gt[8:], pred[:8], gt[:8], ate_min_t=4)
    l7, s7 = chunk_traj_loss(pred[8:] * 7.0, gt[8:], pred[:8] * 7.0, gt[:8], ate_min_t=4)
    assert abs(float(l1) - float(l7)) < 1e-12, (float(l1), float(l7))
    # ... and the RPE translation term is scale-invariant too (the fitted scale absorbs it)
    q = _rand_quats(32, seed=3)
    enc_gt = torch.cat([gt, q], dim=-1)
    enc_p1 = torch.cat([pred, q], dim=-1)
    enc_p7 = torch.cat([pred * 7.0, q], dim=-1)
    r1 = chunk_traj_loss(enc_p1[8:], enc_gt[8:], enc_p1[:8], enc_gt[:8], rpe_w=0.1)[1]
    r7 = chunk_traj_loss(enc_p7[8:], enc_gt[8:], enc_p7[:8], enc_gt[:8], rpe_w=0.1)[1]
    assert abs(r1["rpe_trans"] - r7["rpe_trans"]) < 1e-12, (r1, r7)
    # a global rotation of the prediction is also free
    Rg = _rand_rotation(seed=21)
    aR = float(causal_ate(pred @ Rg.T + 3.0, gt))
    assert abs(a1 - aR) < 1e-11, (a1, aR)
    print("(b) scale invariance: ate %.12f vs x7 %.12f vs rotated+shifted %.12f" % (a1, a7, aR))


# ----------------------------------------------------------------------------- (c) gradients
def test_c_gradcheck_causal_ate():
    gt, pred = _synthetic_traj(T=8, seed=13)
    for mode in ("danskin", "svd"):
        p = pred.clone().requires_grad_(True)
        ok = torch.autograd.gradcheck(
            lambda x: causal_ate(x, gt, align_grad=mode), (p,), eps=1e-6, atol=1e-7, rtol=1e-4
        )
        assert ok, mode
    print("(c) gradcheck on causal_ate (8 frames, float64): PASS for align_grad = danskin AND svd")


def test_c_gradcheck_with_t_and_batch():
    gt, pred = _synthetic_traj(T=10, seed=14)
    p = pred.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda x: causal_ate(x, gt, t=7), (p,), eps=1e-6, atol=1e-7, rtol=1e-4
    )
    gt_b = torch.stack([gt, gt.flip(0)])
    pred_b = torch.stack([pred, pred.flip(0) * 0.5])
    pb = pred_b.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda x: causal_ate(x, gt_b), (pb,), eps=1e-6, atol=1e-7, rtol=1e-4
    )
    print("(c) gradcheck with an explicit t and with a batch dim: PASS")


def _fd_check(f, x0, cells, h=1e-6):
    """Central difference of ``f`` vs autograd at ``x0``; returns (worst abs err, worst rel err)."""
    x = x0.clone().requires_grad_(True)
    f(x).backward()
    grad = x.grad.clone()
    wa = wr = 0.0
    for (k, j) in cells:
        xp = x0.clone()
        xp[k, j] += h
        xm = x0.clone()
        xm[k, j] -= h
        num = (float(f(xp)) - float(f(xm))) / (2 * h)
        ana = float(grad[k, j])
        wa = max(wa, abs(num - ana))
        wr = max(wr, abs(num - ana) / max(1e-12, abs(num)))
    return wa, wr


def test_c_finite_difference_chunk_loss():
    """d(chunk_traj_loss)/d(one frame's centre) against a central difference of the FULL forward.

    The ATE-only case is the numerical proof of the envelope theorem: the default gradient path never
    differentiates the alignment, yet a finite difference of the forward (which re-fits it every time)
    agrees.  The ate+rpe case is what forced RPE_ALIGN_GRAD = "svd" (see the last assertion).
    """
    gt, pred = _synthetic_traj(T=20, seed=15)
    q = _rand_quats(20, seed=5)
    enc_gt = torch.cat([gt, q], dim=-1)
    prev_enc_p = torch.cat([pred[:8], q[:8]], dim=-1)
    cells = [(0, 0), (3, 1), (7, 2), (11, 0)]

    def f(chunk, rpe_w=0.1, **kw):
        enc_chunk = torch.cat([chunk, q[8:]], dim=-1)
        loss, _ = chunk_traj_loss(
            enc_chunk, enc_gt[8:], prev_enc_p, enc_gt[:8], ate_min_t=4, rpe_w=rpe_w, **kw
        )
        return loss

    wa, wr = _fd_check(lambda x: f(x, rpe_w=0.0), pred[8:], cells)
    assert wa < 1e-7, ("ate only", wa, wr)
    print("(c) central difference vs autograd, ATE only  : worst abs %.2e rel %.2e (Danskin holds)" % (wa, wr))

    wa2, wr2 = _fd_check(f, pred[8:], cells)
    assert wa2 < 1e-7, ("ate + rpe", wa2, wr2)
    print("(c) central difference vs autograd, ATE + RPE : worst abs %.2e rel %.2e" % (wa2, wr2))

    # and the record of WHY: stop-gradient on the RPE term's alignment is measurably wrong
    wa3, wr3 = _fd_check(lambda x: f(x, rpe_align_grad="danskin"), pred[8:], cells)
    assert wr3 > 1e-3, ("the danskin shortcut should be visibly wrong on the rpe term", wa3, wr3)
    print("(c) ... with rpe_align_grad='danskin' (stop-grad on the fit): worst rel err %.2e -- which is"
          " why it is not the default for that term" % wr3)


def test_c_danskin_equals_svd_gradient():
    """The envelope theorem, numerically: the two alignment gradient paths give the same gradient."""
    gt, pred = _synthetic_traj(T=16, seed=16)
    grads = {}
    for mode in ("danskin", "svd"):
        p = pred.clone().requires_grad_(True)
        causal_ate(p, gt, align_grad=mode).backward()
        grads[mode] = p.grad.clone()
    diff = float((grads["danskin"] - grads["svd"]).abs().max())
    scale = float(grads["svd"].abs().max())
    assert diff < 1e-10 * max(1.0, scale), (diff, scale)
    print("(c) danskin vs svd gradient: max |diff| %.2e (grad scale %.3e) -- envelope theorem holds" % (diff, scale))


def test_c_backward_at_exact_alignment_is_finite():
    """ATE = 0 is a realistic input (a perfect gate); d sqrt/dx must not blow up there."""
    gt, _ = _synthetic_traj(T=12, seed=17)
    p = gt.clone().requires_grad_(True)
    a = causal_ate(p, gt)
    a.backward()
    assert torch.isfinite(p.grad).all(), p.grad
    assert float(a.detach()) <= math.sqrt(ATE_EPS) * 1.001 + 1e-12, float(a.detach())
    print("(c) backward at a perfectly aligned trajectory: ate %.2e, grad finite (max %.2e)" % (
        float(a.detach()), float(p.grad.abs().max())))


# ----------------------------------------------------------------------------- (d) evaluator
def _load_scene():
    import csv
    import glob

    pf = sorted(glob.glob(os.path.join(PRED_DIR, "*.npz")))
    gf = sorted(glob.glob(os.path.join(GT_DIR, "*.npz")))
    if not pf or not gf or not os.path.exists(CSV):
        return None
    P = np.stack([np.load(f)["pose"] for f in pf])
    G = np.stack([np.load(f)["pose"] for f in gf])
    rows = [
        r
        for r in csv.DictReader(open(CSV))
        if r["camera_id"] == "0" and r["local_timestep"].isdigit()
    ]
    ate_col = np.array([float(r["ate"]) for r in rows], dtype=np.float64)
    n = min(len(P), len(G), len(ate_col))
    return P[:n], G[:n], ate_col[:n]


def test_d_agreement_with_evaluator():
    data = _load_scene()
    if data is None:
        print("(d) SKIPPED: the stored scene %s is not on this filesystem" % SCENE)
        return
    P, G, ate_col = data
    ref = float(np.sqrt(np.mean(ate_col ** 2)))  # the CSV's RMSE reduction of the per-frame column
    pc = torch.from_numpy(P[:, :3, 3].astype(np.float64))
    gc = torch.from_numpy(G[:, :3, 3].astype(np.float64))

    ours64 = float(causal_ate(pc, gc))
    per_frame = _per_frame_ate(pc, gc)
    d64 = abs(ours64 - ref)
    dpf = float(np.abs(per_frame - ate_col).max())

    ours32 = float(causal_ate(pc.float(), gc.float()))
    d32 = abs(ours32 - ref)
    print(
        "(d) %s, %d frames: CSV RMSE(ate) %.12f | causal_ate f64 %.12f (diff %.2e, per-frame max %.2e)"
        " | f32 %.12f (diff %.2e)" % (SCENE, len(ate_col), ref, ours64, d64, dpf, ours32, d32)
    )
    assert d64 < 1e-6, d64
    assert dpf < 1e-6, dpf
    assert d32 < 1e-6, d32
    # the causal prefix at the last frame IS the full-trajectory ATE
    assert abs(float(causal_ate(pc, gc, t=len(ate_col) - 1)) - ours64) < 1e-15


def _per_frame_ate(pred, gt):
    from dust3r.wgate.traj_loss import sim3_align

    s, R, t = sim3_align(pred, gt)
    err = apply_sim3(s, R, t, pred) - gt
    return err.norm(dim=-1).numpy()


def test_d_causal_prefix_is_monotone_in_information():
    """Sanity on the real scene: the causal ATE at t uses only frames 0..t (changing later frames
    cannot change it) -- the property the whole causal loss rests on."""
    data = _load_scene()
    if data is None:
        print("(d2) SKIPPED: scene not present")
        return
    P, G, _ = data
    pc = torch.from_numpy(P[:, :3, 3].astype(np.float64))
    gc = torch.from_numpy(G[:, :3, 3].astype(np.float64))
    a40 = float(causal_ate(pc, gc, t=40))
    pc2 = pc.clone()
    pc2[41:] += 10.0
    a40b = float(causal_ate(pc2, gc, t=40))
    assert abs(a40 - a40b) < 1e-15, (a40, a40b)
    print("(d2) causal prefix at t=40 unchanged (%.12f) when frames 41.. are wrecked" % a40)


# ----------------------------------------------------------------------------- (e) degeneracy
def test_e_degenerate_collinear():
    """2 / 3 collinear centres, and an all-identical set: forward AND backward stay finite."""
    base = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    uneven = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64)
    wobble = base + torch.tensor([[0.0, 0.0, 0.0], [0.0, 1e-7, 0.0], [0.0, -1e-7, 0.0]], dtype=torch.float64)
    cases = {
        "2 collinear": (base[:2], base[:2] * 0.5 + 0.1),
        "3 collinear (exact)": (base, base * 2.0),
        "3 collinear, ate > 0": (base, uneven),  # no similarity maps one onto the other
        "3 near-collinear": (wobble, uneven),  # rank 2, sigma_3 ~ 0, sigma_2 ~ 1e-7
        "3 identical": (torch.zeros(3, 3, dtype=torch.float64), base),
        "1 point": (base[:1], base[:1] + 1.0),
    }
    nonzero_grad = 0
    for name, (pred0, gt0) in cases.items():
        p = pred0.clone().requires_grad_(True)
        a = causal_ate(p, gt0)
        assert torch.isfinite(a), (name, a)
        a.backward()
        assert torch.isfinite(p.grad).all(), (name, p.grad)
        gmax = float(p.grad.abs().max())
        nonzero_grad += gmax > 0
        print("    (e) %-22s ate %.3e, max|grad| %.3e" % (name, float(a.detach()), gmax))
    assert nonzero_grad >= 3, "the degenerate cases must not all be trivially zero-gradient"
    # the same through the chunk loss with ate_min_t = 0 (every frame counts, including t = 0/1)
    gt3 = base * 2.0
    p = base.clone().requires_grad_(True)
    loss, st = chunk_traj_loss(p, gt3, ate_min_t=0)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all(), (loss, p.grad)
    assert st["n_ate"] == 3
    # and with a degenerate (collinear) HISTORY followed by a real chunk
    gt, pred = _synthetic_traj(T=10, seed=18)
    prev_p = torch.zeros(3, 3, dtype=torch.float64)
    prev_g = base.clone()
    p2 = pred.clone().requires_grad_(True)
    loss2, _ = chunk_traj_loss(p2, gt, prev_p, prev_g, ate_min_t=4)
    loss2.backward()
    assert torch.isfinite(loss2) and torch.isfinite(p2.grad).all()
    print("(e) degeneracy: forward + backward finite in all %d cases (and through chunk_traj_loss)" % len(cases))


def test_e_rpe_degeneracy_falls_back_instead_of_nan():
    """rpe_align_grad='svd' + a collinear chunk: the guard must swap in the stop-gradient fit."""
    T = 8
    line = torch.zeros(T, 3, dtype=torch.float64)
    line[:, 0] = torch.arange(T, dtype=torch.float64)
    q = torch.zeros(T, 4, dtype=torch.float64)
    q[:, 0] = 1.0  # no rotation at all -> a perfectly degenerate chunk
    enc_g = torch.cat([line * 2.0, q], dim=-1)
    pred = line.clone()
    pred[3, 0] += 0.4  # a real error, so the loss is not trivially zero
    enc_p = torch.cat([pred, q], dim=-1)
    x = enc_p.clone().requires_grad_(True)
    loss, st = chunk_traj_loss(x, enc_g, ate_min_t=4, rpe_w=0.1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all(), (loss, x.grad)
    assert st["rpe_align_fallback"] == 1, st
    assert st["rpe_trans"] > 0.0, st
    # a healthy chunk does NOT take the fallback
    gt, pr = _synthetic_traj(T=16, seed=25)
    qq = _rand_quats(16, seed=26)
    _, st2 = chunk_traj_loss(
        torch.cat([pr, qq], -1), torch.cat([gt, qq], -1), ate_min_t=4, rpe_w=0.1
    )
    assert st2["rpe_align_fallback"] == 0, st2
    print("(e) RPE on a collinear chunk: fallback=1, loss %.4f, grad finite; healthy chunk fallback=0"
          % float(loss.detach()))


def test_e_svd_path_is_the_unsafe_one():
    """Documents WHY danskin is the default: the literal SVD backward NaNs on collinear centres."""
    base = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    p = base.clone().requires_grad_(True)
    causal_ate(p, base * 2.0, align_grad="svd").backward()
    finite = bool(torch.isfinite(p.grad).all())
    print("(e) align_grad='svd' on 3 collinear centres -> grad finite = %s (danskin path is the "
          "default for exactly this reason)" % finite)


# ----------------------------------------------------------------------------- chunk semantics
def test_chunk_history_is_detached_and_min_t_skips():
    gt, pred = _synthetic_traj(T=24, seed=19)
    prev_p = pred[:8].clone().requires_grad_(True)
    chunk = pred[8:].clone().requires_grad_(True)
    loss, st = chunk_traj_loss(chunk, gt[8:], prev_p, gt[:8], ate_min_t=4)
    loss.backward()
    assert prev_p.grad is None, "the TBPTT history must not receive gradient"
    assert chunk.grad is not None and torch.isfinite(chunk.grad).all()
    assert st["n_ate"] == 16 and st["t_first"] == 8 and st["t_last"] == 23
    # every chunk frame counts because they are all >= ate_min_t = 4
    loss0, st0 = chunk_traj_loss(pred[:6], gt[:6], ate_min_t=4)
    assert st0["n_ate"] == 2, st0
    loss_none, st_none = chunk_traj_loss(pred[:3], gt[:3], ate_min_t=4)
    assert st_none["n_ate"] == 0 and float(loss_none) == 0.0
    p3 = pred[:3].clone().requires_grad_(True)
    chunk_traj_loss(p3, gt[:3], ate_min_t=4)[0].backward()  # must not raise
    assert p3.grad is not None and float(p3.grad.abs().max()) == 0.0
    # the history CHANGES the value (it is the drift context, not decoration)
    l_ctx, _ = chunk_traj_loss(pred[8:], gt[8:], pred[:8], gt[:8], ate_min_t=4)
    l_noctx, _ = chunk_traj_loss(pred[8:], gt[8:], ate_min_t=4)
    assert abs(float(l_ctx) - float(l_noctx)) > 1e-4, (float(l_ctx), float(l_noctx))
    # ... and it equals the explicit prefix ATEs
    manual = torch.stack([causal_ate(pred[: t + 1], gt[: t + 1]) for t in range(8, 24)]).mean()
    assert abs(float(manual) - float(l_ctx)) < 1e-12, (float(manual), float(l_ctx))
    print("(chunk) history detached, ate_min_t honoured, n_ate correct, value == explicit prefix ATEs")


def test_chunk_stats_and_last():
    gt, pred = _synthetic_traj(T=20, seed=20)
    loss, st = chunk_traj_loss(pred[12:], gt[12:], pred[:12], gt[:12], ate_min_t=4)
    assert abs(st["ate_last"] - float(causal_ate(pred, gt))) < 1e-12
    assert abs(st["loss"] - float(loss)) < 1e-12
    assert set(["ate", "ate_last", "n_ate", "loss", "t_first", "t_last"]).issubset(st)
    assert all(isinstance(v, (int, float)) for v in st.values())
    print("(chunk) stats: ate_last == the full-trajectory ATE, all values plain numbers")


# ----------------------------------------------------------------------------- RPE term
def test_rpe_zero_on_perfect_and_local_on_a_single_frame():
    T = 16
    gt, _ = _synthetic_traj(T=T, seed=21)
    q = _rand_quats(T, seed=6)
    enc_gt = torch.cat([gt, q], dim=-1)
    _, st = chunk_traj_loss(enc_gt, enc_gt, ate_min_t=4, rpe_w=1.0)
    assert st["rpe_trans"] < 1e-9 and st["rpe_rot"] < 1e-9, st
    # perturb ONE predicted centre -> the two hops touching it dominate
    k = 9
    pred = gt.clone()
    pred[k] += torch.tensor([0.05, -0.02, 0.01], dtype=torch.float64)
    enc_p = torch.cat([pred, q], dim=-1)
    e_t = _rpe_trans_per_hop(enc_p, enc_gt)
    order = np.argsort(-e_t)[:2] + 1  # hop j is the pair (j-1, j)
    assert set(order.tolist()) == {k, k + 1}, (order, e_t)
    # perturb ONE predicted rotation -> EXACTLY the two hops touching it (alignment-free)
    qq = q.clone()
    dq = torch.tensor([[math.cos(0.1), math.sin(0.1), 0.0, 0.0]], dtype=torch.float64)
    qq[k] = _q_mul(qq[k : k + 1], dq)[0]
    enc_p2 = torch.cat([gt, qq], dim=-1)
    e_r = _rpe_rot_per_hop(enc_p2, enc_gt)
    nz = np.where(e_r > 1e-9)[0] + 1
    assert set(nz.tolist()) == {k, k + 1}, (nz, e_r)
    assert abs(e_r[k - 1] - 0.2) < 1e-9 and abs(e_r[k] - 0.2) < 1e-9, e_r[k - 1 : k + 1]
    print("(rpe) zero on a perfect prediction; a single bad frame shows up on exactly its two hops "
          "(rotation error 0.2 rad = 2 * 0.1 as constructed)")


def _rpe_trans_per_hop(enc_p, enc_g):
    from dust3r.wgate.traj_loss import _rpe_terms, sim3_align  # noqa: F401

    p, g = enc_p[..., :3], enc_g[..., :3]
    s, R, t = sim3_align(p, g)
    dp = (p[1:] - p[:-1]) @ R.T * s
    dg = g[1:] - g[:-1]
    return (dp - dg).norm(dim=-1).numpy()


def _rpe_rot_per_hop(enc_p, enc_g):
    qp, qg = enc_p[..., 3:], enc_g[..., 3:]
    rel_p = _q_mul(_q_conj(qp[:-1]), qp[1:])
    rel_g = _q_mul(_q_conj(qg[:-1]), qg[1:])
    return quat_geodesic(rel_g, rel_p).numpy()


def test_rpe_requires_encodings_and_is_weighted():
    gt, pred = _synthetic_traj(T=12, seed=22)
    try:
        chunk_traj_loss(pred, gt, ate_min_t=4, rpe_w=0.1)
    except ValueError as exc:
        assert "encodings" in str(exc), exc
    else:
        raise AssertionError("rpe_w > 0 without encodings must raise")
    q = _rand_quats(12, seed=9)
    enc_gt = torch.cat([gt, q], dim=-1)
    enc_p = torch.cat([pred, q], dim=-1)
    l0, s0 = chunk_traj_loss(enc_p, enc_gt, ate_min_t=4, rpe_w=0.0)
    l1, s1 = chunk_traj_loss(enc_p, enc_gt, ate_min_t=4, rpe_w=0.1)
    assert abs(float(l1) - (float(l0) + 0.1 * (s1["rpe_trans"] + s1["rpe_rot"]))) < 1e-12
    assert "rpe_trans" not in s0 and s1["n_rpe"] == 11 and s1["rpe_align_fallback"] == 0
    assert abs(s1["rpe_rot_deg"] - math.degrees(s1["rpe_rot"])) < 1e-9
    # a history shifts the hop window: only the chunk's own hops are scored
    l2, s2 = chunk_traj_loss(enc_p[6:], enc_gt[6:], enc_p[:6], enc_gt[:6], rpe_w=0.1)
    assert s2["n_rpe"] == 6, s2  # hops (5,6) .. (10,11)
    print("(rpe) missing encodings raise; weighting exact; hop window follows the chunk (n_rpe 11 / 6)")


def test_rpe_rotation_is_alignment_invariant():
    T = 14
    gt, pred = _synthetic_traj(T=T, seed=23)
    q = _rand_quats(T, seed=10)
    enc_gt = torch.cat([gt, q], dim=-1)
    Rg = _rand_rotation(seed=24)
    qg = _rot_to_quat(Rg)
    enc_p = torch.cat([pred, q], dim=-1)
    enc_pR = torch.cat([pred @ Rg.T * 3.0 + 1.0, _q_mul(qg.expand(T, 4), q)], dim=-1)
    s1 = chunk_traj_loss(enc_p, enc_gt, ate_min_t=4, rpe_w=0.1)[1]
    s2 = chunk_traj_loss(enc_pR, enc_gt, ate_min_t=4, rpe_w=0.1)[1]
    assert abs(s1["rpe_rot"] - s2["rpe_rot"]) < 1e-12, (s1["rpe_rot"], s2["rpe_rot"])
    assert abs(s1["rpe_trans"] - s2["rpe_trans"]) < 1e-11, (s1["rpe_trans"], s2["rpe_trans"])
    assert abs(s1["loss"] - s2["loss"]) < 1e-11
    print("(rpe) a global Sim(3) on the prediction changes nothing (rot %.6f, trans %.6f)" % (
        s1["rpe_rot"], s1["rpe_trans"]))


# ----------------------------------------------------------------------------- quaternions
def test_quaternion_helpers_match_dust3r():
    import dust3r.heads  # noqa: F401  (resolves the camera<->heads circular import)
    from dust3r.utils.camera import (
        quaternion_conjugate,
        quaternion_multiply,
        relative_pose_absT_quatR,
        quaternion_to_matrix,
    )

    q1 = _rand_quats(5, seed=31).float()
    q2 = _rand_quats(5, seed=32).float()
    assert torch.allclose(_q_mul(q1, q2), quaternion_multiply(q1, q2), atol=1e-6)
    assert torch.allclose(_q_conj(q1), quaternion_conjugate(q1), atol=1e-7)
    t1 = torch.randn(5, 3)
    t2 = torch.randn(5, 3)
    _, q_rel = relative_pose_absT_quatR(t1, q1, t2, q2)
    assert torch.allclose(_q_mul(_q_conj(q1), q2), q_rel, atol=1e-6)
    # geodesic against the matrix form acos((tr - 1) / 2)
    R1 = quaternion_to_matrix(q1.double())
    R2 = quaternion_to_matrix(q2.double())
    tr = torch.einsum("bij,bij->b", R1, R2)  # trace(R1^T R2)
    ang = torch.acos(((tr - 1) / 2).clamp(-1, 1))
    ours = quat_geodesic(q1.double(), q2.double())
    assert torch.allclose(ours, ang, atol=1e-9), (ours, ang)
    # double cover: q and -q are the same rotation
    assert float(quat_geodesic(q1.double(), -q1.double()).abs().max()) < 1e-9
    assert float(quat_geodesic(q1.double(), q1.double()).abs().max()) < 1e-9
    # differentiable at a zero rotation
    qz = q1[:1].double().clone().requires_grad_(True)
    quat_geodesic(qz, q1[:1].double()).sum().backward()
    assert torch.isfinite(qz.grad).all()
    print("(quat) _q_mul / _q_conj / quat_geodesic agree with dust3r.utils.camera and the matrix form")


# ----------------------------------------------------------------------------- rollout helpers
def test_rollout_helpers():
    import dust3r.heads  # noqa: F401
    from dust3r.utils.camera import camera_to_pose_encoding, pose_encoding_to_camera

    B, T = 2, 6
    torch.manual_seed(41)
    q = torch.randn(B * T, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    enc = torch.cat([torch.randn(B * T, 3), q], dim=-1)
    c2w = pose_encoding_to_camera(enc).reshape(B, T, 4, 4)
    views = [{"camera_pose": c2w[:, t]} for t in range(T)]
    preds = [{"camera_pose": enc.reshape(B, T, 7)[:, t].clone().requires_grad_(True)} for t in range(T)]

    pc = pred_centres(preds)
    pe = pred_pose_encodings(preds)
    assert pc.shape == (B, T, 3) and pe.shape == (B, T, 7)
    assert pc.requires_grad, "pred_centres must stay in the graph"
    assert torch.allclose(pc, enc.reshape(B, T, 7)[..., :3])

    gc = gt_centres(views)
    ge = gt_pose_encodings(views)
    assert gc.shape == (B, T, 3) and ge.shape == (B, T, 7) and not ge.requires_grad
    # frame 0 is the reference => identity encoding
    assert float(ge[:, 0, :3].abs().max()) < 1e-6
    assert torch.allclose(ge[:, 0, 3:].abs(), torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(B, 4), atol=1e-6)
    # explicit reference: chunk views + ref_pose == the same frame-0-relative encodings
    ge_chunk = gt_pose_encodings(views[3:], ref_pose=views[0]["camera_pose"])
    assert torch.allclose(ge_chunk, ge[:, 3:], atol=1e-5)
    # against the criterion's own expression
    from dust3r.utils.geometry import inv

    ref_inv = inv(views[0]["camera_pose"])
    manual = torch.stack([camera_to_pose_encoding(ref_inv @ v["camera_pose"]) for v in views], dim=1)
    assert torch.allclose(ge, manual, atol=1e-6)
    # end to end: a pred that IS the gt gives ~0 loss through the helpers
    preds_perfect = [{"camera_pose": ge[:, t].clone()} for t in range(T)]
    loss, st = chunk_traj_loss(pred_centres(preds_perfect), gt_centres(views), ate_min_t=4)
    assert float(loss) < 1e-5, (float(loss), st)
    print("(helpers) pred_centres / gt_centres shapes, grad, frame-0 reference, ref_pose, criterion parity")


# ----------------------------------------------------------------------------- misc
def test_dtype_and_autocast():
    gt, pred = _synthetic_traj(T=12, seed=51)
    a64 = causal_ate(pred, gt)
    a32 = causal_ate(pred.float(), gt.float())
    assert a64.dtype == torch.float64 and a32.dtype == torch.float32
    assert abs(float(a64) - float(a32)) < 1e-6, (float(a64), float(a32))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        a_ac = causal_ate(pred.float(), gt.float())
    assert a_ac.dtype == torch.float32, a_ac.dtype
    assert abs(float(a_ac) - float(a32)) == 0.0, (float(a_ac), float(a32))
    # float16 inputs are promoted, not rejected
    a16 = causal_ate(pred.half(), gt.half())
    assert a16.dtype == torch.float32
    print("(misc) dtypes preserved (f64/f32), bf16 autocast is inert, f16 promoted; f64 vs f32 %.2e" % (
        abs(float(a64) - float(a32))))


def test_autocast_is_inert_on_the_whole_chunk_loss():
    """REGRESSION: every term of chunk_traj_loss -- not just the ATE one -- must ignore autocast.

    Until 2026-09-22 only ``causal_ate`` opened ``_no_autocast``; ``_rpe_terms`` did not, so under
    ``torch.autocast(bfloat16)`` its ``dp @ R^T`` ran in bf16 and ``stats['rpe_trans']`` came out
    1.1e-6 -> 5.7e-3 (reldiff 5e+03) on this very input while ``ate`` and ``rpe_rot`` were unchanged.
    """
    T = 12
    gt, pred = _synthetic_traj(T=T, seed=77)
    qp, qg = _rand_quats(T, seed=78), _rand_quats(T, seed=79)
    enc_p = torch.cat([pred, qp], dim=-1).float()
    enc_g = torch.cat([gt, qg], dim=-1).float()
    n_prev = 4

    def _fwd():
        x = enc_p.clone().requires_grad_(True)
        loss, st = chunk_traj_loss(
            x[n_prev:], enc_g[n_prev:],
            prev_pred_centres=x[:n_prev], prev_gt_centres=enc_g[:n_prev],
            ate_min_t=4, rpe_w=0.1,
        )
        return loss, st, x

    # the FORWARD is what this module controls; backward is run outside the region, which is both
    # torch's documented pattern and what train_wgate_consequence.py does (accelerator.backward is
    # dedented out of every autocast block).  Backward INSIDE a live bf16 region re-autocasts the
    # matmuls of the backward graph itself -- that is the caller's business, not this module's, and
    # it is asserted against below so the distinction stays on the record.
    l0, s0, x0 = _fwd()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        l1, s1, x1 = _fwd()
    l0.backward()
    l1.backward()
    g0, g1 = x0.grad, x1.grad

    assert l0.dtype == torch.float32 and l1.dtype == torch.float32, (l0.dtype, l1.dtype)
    assert set(s0) == set(s1), (sorted(s0), sorted(s1))
    for k in sorted(s0):
        assert s0[k] == s1[k], "stats[%r] differs under autocast: %r vs %r" % (k, s0[k], s1[k])
    assert float(l0.detach()) == float(l1.detach()), (float(l0), float(l1))
    assert torch.equal(g0, g1), float((g0 - g1).abs().max())
    assert s0["rpe_trans"] > 0.0 and s0["n_rpe"] > 0, s0  # the term is actually exercised

    # backward called INSIDE the region does drift (bf16 matmul backward): documented, not ours
    l2, _, x2 = _fwd()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        l2.backward()
    d_in = float((x2.grad - g0).abs().max())
    assert d_in > 0.0, "expected the in-region backward to differ; torch semantics changed?"

    # the two matmul helpers the term is built from, on their own
    sc, R, tr = umeyama_sim3(pred.float(), gt.float())
    a0 = apply_sim3(sc, R, tr, pred.float())
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        a1 = apply_sim3(sc, R, tr, pred.float())
        s2, R2, t2 = umeyama_sim3(pred.float(), gt.float())
    assert torch.equal(a0, a1), float((a0 - a1).abs().max())
    assert torch.equal(R, R2) and torch.equal(tr, t2) and torch.equal(sc, s2)
    print("(misc) bf16 autocast is inert on ALL of chunk_traj_loss: %d stats + loss + grad "
          "bit-identical (rpe_trans %.4e, n_rpe %d); backward inside the region drifts %.2e, "
          "which is why the trainer backwards outside it" % (
              len(s0), s0["rpe_trans"], s0["n_rpe"], d_in))


def test_bad_inputs_raise():
    gt, pred = _synthetic_traj(T=6, seed=52)
    for bad in (
        lambda: causal_ate(pred, gt[:5]),
        lambda: causal_ate(pred, gt, reduce="sum"),
        lambda: causal_ate(pred, gt, align_grad="nope"),
        lambda: chunk_traj_loss(pred[..., :2], gt[..., :2]),
        lambda: chunk_traj_loss(pred, gt, pred[:3], gt[:2]),
    ):
        try:
            bad()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("expected a ValueError from %r" % bad)
    print("(misc) shape / option errors raise instead of silently doing something else")


# ----------------------------------------------------------------------------- runner
def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    order = [
        test_umeyama_recovers_known_transform,
        test_umeyama_matches_evaluator_numpy,
        test_umeyama_reflection_branch,
        test_umeyama_batched,
        test_a_exactness,
        test_b_scale_invariance,
        test_c_gradcheck_causal_ate,
        test_c_gradcheck_with_t_and_batch,
        test_c_finite_difference_chunk_loss,
        test_c_danskin_equals_svd_gradient,
        test_c_backward_at_exact_alignment_is_finite,
        test_d_agreement_with_evaluator,
        test_d_causal_prefix_is_monotone_in_information,
        test_e_degenerate_collinear,
        test_e_rpe_degeneracy_falls_back_instead_of_nan,
        test_e_svd_path_is_the_unsafe_one,
        test_chunk_history_is_detached_and_min_t_skips,
        test_chunk_stats_and_last,
        test_rpe_zero_on_perfect_and_local_on_a_single_frame,
        test_rpe_requires_encodings_and_is_weighted,
        test_rpe_rotation_is_alignment_invariant,
        test_quaternion_helpers_match_dust3r,
        test_rollout_helpers,
        test_dtype_and_autocast,
        test_autocast_is_inert_on_the_whole_chunk_loss,
        test_bad_inputs_raise,
    ]
    assert len(order) == len(fns), (len(order), len(fns), sorted(f.__name__ for f in fns))
    for fn in order:
        fn()
    print("\ntest_traj_loss: %d/%d PASS" % (len(order), len(order)))


if __name__ == "__main__":
    main()
