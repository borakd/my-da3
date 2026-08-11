#!/usr/bin/env python3
"""Code-level verification of the pose_gru_oracle diagnostic. CPU, no data, no
checkpoint — runs on a login node in a few seconds.

    conda activate cuteanything
    PYTHONPATH=$PWD:$PWD/src:$PWD/src/CUT3R python verify_gru_oracle.py

What it proves, in order:
  1. gt_pose_encoding() reproduces PoseGRULoss's target construction EXACTLY
     (same reference view, same inverse, same encoding). This is the one thing
     that, if wrong, makes the oracle read nonzero for a perfectly good GRU.
  2. A zero-init residual PoseGRU is a bit-exact pass-through for every
     (iters, img_feat) combination — i.e. the F and R levers cannot perturb the
     oracle's expected answer.
  3. Fed the raw GT pose, PoseGRULoss reports gru_quat_loss == 0 while
     gru_trans_loss stays NONZERO (the head's scene scale). This is the claim
     the config header makes; it is checked here so nobody takes it on faith.
  4. PoseGRULoss(norm_mode='') — the extra_test_criteria readout — returns
     exactly 0.0, and did NOT before the (B,1) shape fix in
     get_norm_factor_poses.
  5. An all-masked batch returns zero loss with empty details.
  6. fp16 round-tripping the injected GT would break the zero — the reason the
     oracle block stays in fp32.
"""
import sys

import torch

from dust3r.losses import PoseGRULoss
from dust3r.model import PoseGRU, gt_pose_encoding, pose_delta_encoding
from dust3r.utils.camera import camera_to_pose_encoding
from dust3r.utils.geometry import inv

B, V = 4, 5
HEAD_SCALE = 0.37  # the model's scene scale != GT scale (is_metric=False data)
FAILED = []


def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")
    if not ok:
        FAILED.append(name)


def rand_c2w(n, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    r, i, j, k = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (j * j + k * k), 2 * (i * j - k * r), 2 * (i * k + j * r),
            2 * (i * j + k * r), 1 - 2 * (i * i + k * k), 2 * (j * k - i * r),
            2 * (i * k - j * r), 2 * (j * k + i * r), 1 - 2 * (i * i + j * j),
        ],
        dim=-1,
    ).reshape(n, 3, 3)
    c2w = torch.eye(4).repeat(n, 1, 1)
    c2w[:, :3, :3] = R
    c2w[:, :3, 3] = torch.randn(n, 3, generator=g) * 2.0
    return c2w


def build_views(seed=0):
    return [
        {
            "camera_pose": rand_c2w(B, seed * 100 + v),
            "is_metric": torch.zeros(B, dtype=torch.bool),
        }
        for v in range(V)
    ]


views = build_views()

print("== 1. gt_pose_encoding == PoseGRULoss's target construction ==")
# The loss's own algebra, verbatim from PoseGRULoss.compute_loss.
in_camera1 = inv(views[0]["camera_pose"].float()).detach()
loss_targets = [
    camera_to_pose_encoding(in_camera1 @ v["camera_pose"].float()).detach() for v in views
]
worst = 0.0
for i in range(V):
    enc, ok = gt_pose_encoding(views, i)
    worst = max(worst, float((enc - loss_targets[i]).abs().max()))
    if not bool(ok.all()):
        FAILED.append(f"ok flag false at view {i}")
check("bit-exact match to the loss target", worst == 0.0, f"max|diff|={worst:.3e}")
check("idx<0 returns (None, None)", gt_pose_encoding(views, -1) == (None, None))

print("== 2. zero-init residual PoseGRU is a bit-exact pass-through ==")
for img_feat in ("none", "input"):
    for iters in (1, 4, 8):
        gru = PoseGRU(
            hidden_dim=128,
            mode="residual",
            input_mode="pose_delta",
            img_feat=img_feat,
            img_feat_dim=32,
            img_feat_frames=2,
            img_feat_src_dim=1024,
            iters=iters,
        )
        est = loss_targets[2].clone()
        delta = pose_delta_encoding(loss_targets[1], loss_targets[2])
        feat = torch.randn(B, 2 * 1024) if img_feat == "input" else None
        hidden = None
        for _ in range(iters):
            out, hidden = gru(torch.cat([est, delta], -1).float(), hidden, img_feat=feat)
            est = out.detach()  # pose_gru_iter_detach, as the model does
        d = float((out.detach() - loss_targets[2]).abs().max())
        check(f"pass-through F={img_feat} R={iters}", d == 0.0, f"max|diff|={d:.3e}")

print("== 3. oracle readout under the production criterion ==")


def make_preds(gru_pose_fn, n_views=V):
    """preds[i]['camera_pose'] is the head's pose at HEAD_SCALE; gru_pose is
    whatever the oracle would have produced."""
    preds = []
    for i in range(n_views):
        p = {"camera_pose": loss_targets[i].clone()}
        p["camera_pose"][:, :3] *= HEAD_SCALE
        gp = gru_pose_fn(i)
        if gp is not None:
            p["gru_pose"] = gp
        preds.append(p)
    return preds


# Oracle: the GRU emits exactly the GT pose it was handed (views 1.. only,
# mirroring the model, which never runs the GRU at view 0).
preds = make_preds(lambda i: loss_targets[i].clone() if i > 0 else None)
loss, det = PoseGRULoss()(views, preds, camera1=views[0]["camera_pose"])
for k in sorted(det):
    print(f"      {k:26s} = {det[k]:.6e}" if isinstance(det[k], float) else f"      {k:26s} = {det[k]}")
check("gru_quat_loss == 0", det["gru_quat_loss"] < 2e-7)
check(
    "gru_trans_loss NONZERO by design (head scale error)",
    det["gru_trans_loss"] > 1e-2,
    f"={det['gru_trans_loss']:.4f}",
)

print("== 3b. a perturbed GRU must NOT read zero (negative control) ==")
bad = make_preds(lambda i: loss_targets[i] + 0.05 if i > 0 else None)
_, det_bad = PoseGRULoss()(views, bad, camera1=views[0]["camera_pose"])
# NB: gru_trans_loss is NOT a usable control here. Under the oracle it is
# dominated by the head-vs-GT scale mismatch (2.13), so a 0.05 pose error moves
# it by less than a percent and can even move it DOWN (measured 2.1284 ->
# 2.1085). The quaternion term is factor-free and therefore clean, and the
# un-normalized loss in check 4 is the other honest discriminator.
check(
    "perturbation detected (quaternion term)",
    det_bad["gru_quat_loss"] > 1e-3,
    f"{det['gru_quat_loss']:.2e} -> {det_bad['gru_quat_loss']:.4f}",
)

print("== 4. PoseGRULoss(norm_mode='') — the extra_test_criteria readout ==")
loss0, det0 = PoseGRULoss(norm_mode="")(views, preds, camera1=views[0]["camera_pose"])
check("un-normalized oracle loss == 0", float(loss0) < 1e-6, f"={float(loss0):.3e}")
loss0b, _ = PoseGRULoss(norm_mode="")(views, bad, camera1=views[0]["camera_pose"])
check("...and > 0 when perturbed", float(loss0b) > 1e-3, f"={float(loss0b):.4f}")

print("== 4b. the TRAIN gauge: norm_mode='' + iter_gamma (R8 sequence loss) ==")
# gru_pose_iters is the stacked (N, B, 7) iterate tensor. Under a correct oracle
# EVERY iterate equals the injected GT (the pass-through is a fixed point), so
# the gamma-weighted intermediate mass must also be exactly 0.
preds_it = make_preds(lambda i: loss_targets[i].clone() if i > 0 else None)
for i, p in enumerate(preds_it):
    if "gru_pose" in p:
        p["gru_pose_iters"] = loss_targets[i].clone().unsqueeze(0).repeat(8, 1, 1)
li, di = PoseGRULoss(norm_mode="", iter_gamma=0.8)(
    views, preds_it, camera1=views[0]["camera_pose"]
)
check("train-gauge loss == 0 with R8 iterates", float(li) < 1e-6, f"={float(li):.3e}")
check(
    "gru_pose_loss_iters == 0",
    abs(di.get("gru_pose_loss_iters", 0.0)) < 1e-6,
    f"={di.get('gru_pose_loss_iters', 0.0):.3e}",
)

print("== 4c. exactly-zero loss has finite, exactly-zero gradient ==")
# This is what keeps the pass-through from drifting: if torch.norm's backward
# were NaN at 0 (a classic trap) the run would blow up; if it were merely
# nonzero the zero-init head would move and the oracle would decay after step 0.
gru = PoseGRU(hidden_dim=128, mode="residual", input_mode="pose_delta", iters=1)
gp = []
for i in range(V):
    if i == 0:
        gp.append(None)
        continue
    est = loss_targets[i].clone()
    d = pose_delta_encoding(loss_targets[i - 1], loss_targets[i])
    out, _ = gru(torch.cat([est, d], -1).float(), None)
    gp.append(out)
it = iter(gp)
g_preds = make_preds(lambda i: gp[i])
lg, _ = PoseGRULoss(norm_mode="")(views, g_preds, camera1=views[0]["camera_pose"])
lg.backward()
grads = [p.grad for p in gru.parameters() if p.grad is not None]
allfinite = all(bool(torch.isfinite(g).all()) for g in grads)
allzero = all(float(g.abs().max()) == 0.0 for g in grads)
check("loss is exactly 0 through the real module", float(lg) == 0.0, f"={float(lg):.3e}")
check("gradients finite (no NaN from norm at 0)", allfinite)
check("gradients exactly 0 -> zero-init head cannot drift", allzero)

print("== 5. vacuous-zero guard ==")
static = [
    {"camera_pose": torch.eye(4).repeat(B, 1, 1), "is_metric": torch.zeros(B, dtype=torch.bool)}
    for _ in range(V)
]
sp = [
    {"camera_pose": torch.zeros(B, 7), "gru_pose": torch.randn(B, 7) * 9.0} for _ in range(V)
]
for p in sp:
    p["camera_pose"][:, 3] = 1.0
sl, sd = PoseGRULoss()(static, sp, camera1=static[0]["camera_pose"])
check(
    "all-masked batch -> zero loss, empty details",
    sd == {} and float(sl) == 0.0,
    f"loss={float(sl)} details={sd}",
)

print("== 6. fp16 round-trip would break the zero (why the block stays fp32) ==")
half = make_preds(lambda i: loss_targets[i].half().float() if i > 0 else None)
_, det_h = PoseGRULoss()(views, half, camera1=views[0]["camera_pose"])
check(
    "fp16 GT is NOT exactly zero",
    det_h["gru_quat_loss"] > 1e-6,
    f"gru_quat_loss={det_h['gru_quat_loss']:.3e} (fp32 path gives 0.0)",
)

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASSED")
