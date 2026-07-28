"""Step-3 verification for PoseGRULoss (dust3r/losses.py) — synthetic, CPU-only.

Run from the worktree root:
    PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src" \
        conda run -n cuteanything python verify_pose_gru_loss.py
"""
import torch

import dust3r.heads  # noqa: F401  import-order gotcha: heads before utils.camera
from dust3r.losses import PoseGRULoss
from dust3r.utils.camera import (
    camera_to_pose_encoding,
    quaternion_to_matrix,
)

torch.manual_seed(0)
B, N = 3, 5
S = 0.37  # simulated model scene scale vs GT scale

PASS = []


def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name)
    print(f"  PASS  {name}")


def make_c2w(b):
    q = torch.nn.functional.normalize(torch.randn(b, 4), dim=-1)
    c2w = torch.eye(4).repeat(b, 1, 1)
    c2w[:, :3, :3] = quaternion_to_matrix(q)
    c2w[:, :3, 3] = torch.randn(b, 3)
    return c2w


def build_batch(is_metric=False, scale=S, gru_offset=0.0):
    """gts/preds mimicking one TBPTT chunk that starts at view 0."""
    gt_c2w = [make_c2w(B) for _ in range(N)]
    inv0 = torch.linalg.inv(gt_c2w[0])
    encs = [camera_to_pose_encoding(inv0 @ c) for c in gt_c2w]  # view-0-relative GT
    gts, preds = [], []
    for i in range(N):
        gts.append(
            {
                "camera_pose": gt_c2w[i].clone(),
                "is_metric": torch.full((B,), bool(is_metric)),
            }
        )
        # head prediction: GT in the model's own scale (translation * scale)
        cp = torch.cat([scale * encs[i][:, :3], encs[i][:, 3:]], dim=-1)
        pred = {"camera_pose": cp.clone()}
        if i > 0:
            pred["gru_pose"] = cp.clone() + gru_offset
        preds.append(pred)
    return gts, preds, gt_c2w


print("== 1. no gru_pose anywhere -> zero loss, empty details ==")
gts, preds, _ = build_batch()
for p in preds:
    p.pop("gru_pose", None)
loss, details = PoseGRULoss()(gts, preds)
check("zero loss without gru_pose", float(loss) == 0.0 and details == {})

print("== 2. perfect GRU (GT in model scale) -> ~0 loss despite scale mismatch ==")
gts, preds, _ = build_batch(is_metric=False, scale=S)
loss, details = PoseGRULoss()(gts, preds)
check(f"scale-invariant zero (loss={float(loss):.2e})", float(loss) < 1e-5)
check("details keys", set(details) == {"gru_pose_loss", "gru_trans_loss", "gru_quat_loss"})

print("== 3. perturbed GRU -> positive loss ==")
gts, preds, _ = build_batch(gru_offset=0.05)
loss_p, _ = PoseGRULoss()(gts, preds)
check(f"perturbation detected (loss={float(loss_p):.3f})", float(loss_p) > 1e-3)

print("== 4. metric samples: pred factor := gt factor -> scale mismatch now penalized ==")
gts, preds, _ = build_batch(is_metric=True, scale=S)
loss_m, _ = PoseGRULoss()(gts, preds)
check(f"metric true-scale supervision (loss={float(loss_m):.3f})", float(loss_m) > 1e-2)
gts, preds, _ = build_batch(is_metric=True, scale=1.0)
loss_m1, _ = PoseGRULoss()(gts, preds)
check(f"metric + scale 1 -> ~0 (loss={float(loss_m1):.2e})", float(loss_m1) < 1e-5)

print("== 5. camera1 kwarg path == no-kwarg path ==")
gts, preds, _ = build_batch(gru_offset=0.03)
l_none, _ = PoseGRULoss()(gts, preds)
l_cam1, _ = PoseGRULoss()(gts, preds, camera1=gts[0]["camera_pose"])
check("camera1 equivalence", torch.allclose(l_none, l_cam1))

print("== 6. gradient isolation: only gru_pose's graph gets gradients ==")
gts, preds, _ = build_batch()
gru_param = torch.nn.Parameter(torch.randn(7) * 0.01)
head_leaves = []
for i, p in enumerate(preds):
    leaf = p["camera_pose"].clone().requires_grad_(True)
    head_leaves.append(leaf)
    p["camera_pose"] = leaf
    if "gru_pose" in p:
        p["gru_pose"] = p["gru_pose"] + gru_param  # graph reaches only gru_param
loss, _ = PoseGRULoss()(gts, preds)
loss.backward()
check("gru param got grad", gru_param.grad is not None and gru_param.grad.abs().sum() > 0)
check("head poses got NO grad (detached)", all(l.grad is None for l in head_leaves))

print("== 7. NaN GT pose: sample dropped, loss stays finite ==")
gts, preds, _ = build_batch(gru_offset=0.03)
gts[2]["camera_pose"][1] = float("nan")  # sample 1, view 2
loss_nan, _ = PoseGRULoss()(gts, preds)
check(f"finite under NaN GT (loss={float(loss_nan):.3f})", torch.isfinite(loss_nan))
for g in gts:
    g["camera_pose"][:] = float("nan")
loss_allnan, det = PoseGRULoss()(gts, preds)
check("all-NaN -> zero loss", float(loss_allnan) == 0.0 and det == {})

print("== 8. MultiLoss composition: alpha weighting and chaining ==")
gts, preds, _ = build_batch(gru_offset=0.03)
l1, _ = PoseGRULoss()(gts, preds)
l2, _ = (2.0 * PoseGRULoss())(gts, preds)
check("alpha=2 doubles loss", torch.allclose(2 * l1, l2))
lc, dc = (PoseGRULoss() + PoseGRULoss())(gts, preds)
check("chained sum", torch.allclose(2 * l1, lc) and "gru_pose_loss" in dc)

print("== 9. config-string eval (trainer namespace) ==")
from dust3r.losses import *  # noqa: F401,F403  same wildcard the trainer relies on

crit = eval(
    "ConfLoss(Regr3DPoseBatchList(L21, norm_mode='?avg_dis'), alpha=0.2)"
    " + RGBLoss(MSE) + 1.0*PoseGRULoss()"
)
check("criterion string builds", "PoseGRULoss" in repr(crit))

print(f"\nALL {len(PASS)} CHECKS PASSED")
