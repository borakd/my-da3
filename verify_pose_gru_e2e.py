#!/usr/bin/env python
"""
verify_pose_gru_e2e.py — end-to-end GPU verification of the PoseGRU line
(steps 1-5) on the REAL pretrained checkpoint and REAL wrist_test data.

Stages:
  A  determinism floor: identical prev_pred rollouts twice
  B  residual-init parity: pose_gru(residual) == plain prev_pred
     (view 0 bit-identical; later views within fp-renorm tolerance;
      gru_pose == raw stashed pose: translation bit-equal, quat renormalized)
  C  aux-loss gradient flow + isolation on the real model
  D  full config train criterion: finite loss, gru details, joint grads
  E  TBPTT-style chunked backward: no double-backward crash, grads per chunk
  F  falsifiers: PREV_PRED_RAY_SHUFFLE perturbs; POSE_GRU_HIDDEN_SHUFFLE is a
     no-op at zero-init (negative control) and perturbs once the head is live
  G  direct mode: view 0 untouched, later views substantially changed
  H  checkpoint round-trip: loader guard raises without enable, loads strict
     with enable

Submit via:  sbatch verify_pose_gru_e2e.sbatch   (1x L40S)
"""
import os
import sys

WORKTREE = "/scratch/bdursun25/cuteanything/captain_gru_v3"
for p in [
    os.path.join(WORKTREE, "src"),
    os.path.join(WORKTREE, "src/CUT3R"),
    os.path.join(WORKTREE, "src/CUT3R/src"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

import dust3r.heads  # noqa: F401  MUST precede dust3r.utils.camera (circular import)
from dust3r.datasets.dl3dv import DL3DV_Multi
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig
from dust3r.losses import (  # noqa: F401  names used by the criterion string
    ConfLoss,
    L21,
    MSE,
    PoseGRULoss,
    RGBLoss,
    Regr3DPoseBatchList,
)
from torch.utils.data._utils.collate import default_collate

# ----------------------------------------------------------------------------- config
DATA_ROOT = (
    "/frozen/avg/bora_data/droid_datasets/training_data/"
    "pointworld_droid_wrist_test/dl3dv_multi"
)
CKPT = "/scratch/bdursun25/cuteanything/my-da3/src/CUT3R/src/cut3r_512_dpt_4_64.pth"
MODEL_STR = (
    "ARCroco3DStereo(ARCroco3DStereoConfig(freeze='encoder', state_size=768, "
    "state_pe='2d', pos_embed='RoPE100', rgb_head=True, pose_head=True, "
    "patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', "
    "output_mode='pts3d+pose', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), "
    "pose_mode=('exp', -inf, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, "
    "dec_embed_dim=768, dec_depth=12, dec_num_heads=12, landscape_only=False))"
)
TRAIN_CRIT = (
    "ConfLoss(Regr3DPoseBatchList(L21, norm_mode='?avg_dis'), alpha=0.2)"
    " + RGBLoss(MSE) + 1.0*PoseGRULoss()"
)
NV = 8
RES = (320, 192)  # (W, H)
IDXS = [5, 100]  # batch of 2 (falsifiers need B>1)
SEED = 777
DEVICE = "cuda"

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
np.set_printoptions(precision=5, suppress=True, linewidth=200)

CHECKS = []


def banner(t):
    print("\n" + "=" * 96 + f"\n {t}\n" + "=" * 96, flush=True)


def check(gid, desc, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {gid}: {desc}" + (f"\n         -> {detail}" if detail else ""), flush=True)
    CHECKS.append((gid, desc, bool(ok), detail))


def maxdiff(a, b):
    return float((a.double() - b.double()).abs().max())


def to_device(view):
    return {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in view.items()}


def rollout(model, batch, grad=False, chunk_groups=None, chunk_criterion=None):
    """Mirror loss_of_one_batch_tbptt: encoder no_grad, sequential group steps.
    If chunk_groups is set, detach state at chunk boundaries and (optionally)
    backward chunk_criterion per chunk. Returns list of res dicts."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    ress = []
    with ctx:
        with torch.no_grad():
            (feat, pos, shape), (isf, imem, sf, sp, mem) = model._forward_encoder(batch)
        feat = [f.detach() for f in feat]
        sf, mem = sf.detach(), mem.detach()
        isf, imem = isf.detach(), imem.detach()
        chunk_preds, chunk_views = [], []
        for v in range(len(batch)):
            res_group, (sf, mem) = model._forward_decoder_group_step(
                views=batch,
                view_indices=[v],
                feat_group=[feat[v]],
                pos_group=[pos[v]],
                shape_group=[shape[v]],
                init_state_feat=isf,
                init_mem=imem,
                state_feat=sf,
                state_pos=sp,
                mem=mem,
            )
            ress.append(res_group[0])
            chunk_preds.append(res_group[0])
            chunk_views.append(batch[v])
            boundary = chunk_groups is not None and (
                (v + 1) % chunk_groups == 0 or v == len(batch) - 1
            )
            if boundary:
                if chunk_criterion is not None:
                    loss, _ = chunk_criterion(
                        chunk_views, chunk_preds, camera1=batch[0]["camera_pose"]
                    )
                    if loss.requires_grad:
                        loss.backward()
                sf, mem = sf.detach(), mem.detach()  # inference.py:142-144
                chunk_preds, chunk_views = [], []
    return ress


def snap(ress):
    """Detach the keys we compare."""
    out = []
    for r in ress:
        out.append({k: r[k].detach().clone() for k in ("camera_pose", "pts3d_in_self_view")})
        if "gru_pose" in r:
            out[-1]["gru_pose"] = r["gru_pose"].detach().clone()
    return out


def diff_by_view(a, b, key):
    return [maxdiff(ra[key], rb[key]) for ra, rb in zip(a, b)]


# =============================================================================
banner("STAGE 0 — setup: real data, real ckpt, trainer-identical model build (GPU)")
# =============================================================================
assert torch.cuda.is_available(), "this verification requires a GPU"
print(f"  device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

ds = DL3DV_Multi(
    split="test",
    ROOT=DATA_ROOT,
    resolution=[RES],
    transform=ImgNorm,
    num_views=NV,
    n_corres=0,
    aug_crop=0,
    allow_repeat=True,
    force_consecutive_frame_sampling=True,
    seed=SEED,
)
view_lists = [ds[i] for i in IDXS]
batch = [to_device(default_collate([vl[v] for vl in view_lists])) for v in range(NV)]
print(f"  batch: {len(IDXS)} samples x {NV} consecutive views, labels "
      f"{[view_lists[b][0]['label'] for b in range(len(IDXS))]}")

inf = float("inf")
model = eval(MODEL_STR)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
missing = model.load_state_dict(ckpt["model"], strict=False)
print(f"  ckpt loaded strict=False: {len(missing.missing_keys)} missing, "
      f"{len(missing.unexpected_keys)} unexpected")
del ckpt
model = model.to(DEVICE).eval()
model.views_per_step = 1
model.feed_prev_pred = True
train_criterion = eval(TRAIN_CRIT).to(DEVICE)

# =============================================================================
banner("STAGE A — determinism floor: plain prev_pred twice")
# =============================================================================
base1 = snap(rollout(model, batch))
base2 = snap(rollout(model, batch))
noise = max(max(diff_by_view(base1, base2, "camera_pose")),
            max(diff_by_view(base1, base2, "pts3d_in_self_view")))
check("A1", "plain prev_pred rollout is run-to-run deterministic", noise == 0.0,
      f"max run-to-run diff = {noise:.3e}")

# =============================================================================
banner("STAGE B — residual-init parity: pose_gru(residual) vs plain prev_pred")
# =============================================================================
model.enable_pose_gru(hidden_dim=128, mode="residual")
model.pose_gru.to(DEVICE)
gru_res = rollout(model, batch)
gru_snap = snap(gru_res)

d0_pose = maxdiff(gru_snap[0]["camera_pose"], base1[0]["camera_pose"])
d0_pts = maxdiff(gru_snap[0]["pts3d_in_self_view"], base1[0]["pts3d_in_self_view"])
check("B1", "view 0 identical up to determinism floor (GRU cannot influence it)",
      d0_pose <= noise and d0_pts <= noise,
      f"pose diff {d0_pose:.3e}, pts diff {d0_pts:.3e}, floor {noise:.1e}")

dp = diff_by_view(gru_snap, base1, "camera_pose")
dx = diff_by_view(gru_snap, base1, "pts3d_in_self_view")
check("B2", "later views match plain prev_pred within fp-renorm tolerance",
      max(dp) < 1e-3 and max(dx) < 1e-2,
      f"per-view pose diffs {['%.1e' % d for d in dp]}; max pts diff {max(dx):.1e}")

t_exact = all(
    torch.equal(gru_snap[v]["gru_pose"][:, :3], gru_snap[v - 1]["camera_pose"][:, :3])
    for v in range(1, NV)
)
q_close = all(
    maxdiff(
        gru_snap[v]["gru_pose"][:, 3:],
        torch.nn.functional.normalize(gru_snap[v - 1]["camera_pose"][:, 3:], dim=-1),
    ) < 1e-6
    for v in range(1, NV)
)
check("B3", "gru_pose == raw stash (t bit-equal, q = renormalized quat)", t_exact and q_close)
check("B4", "gru_pose attached on views 1..N-1 only",
      "gru_pose" not in gru_res[0] and all("gru_pose" in r for r in gru_res[1:]))

# =============================================================================
banner("STAGE C — aux-loss gradients on the real model: flow + isolation")
# =============================================================================
model.zero_grad(set_to_none=True)
ress = rollout(model, batch, grad=True)
loss, details = PoseGRULoss()(batch, ress)
loss.backward()
gru_names = {n for n, _ in model.pose_gru.named_parameters()}
head_grads = [p.grad for n, p in model.pose_gru.named_parameters() if n.startswith("head.")]
cell_grads = [p.grad for n, p in model.pose_gru.named_parameters() if n.startswith("cell.")]
other_touched = [
    n for n, p in model.named_parameters()
    if not n.startswith("pose_gru.") and p.grad is not None
]
check("C1", "aux loss finite and positive", torch.isfinite(loss) and float(loss) > 0,
      f"loss={float(loss):.4f}, details={details}")
check("C2", "head grads nonzero", all(g is not None and g.abs().sum() > 0 for g in head_grads))
check("C3", "cell grads zero at zero-init head (known one-step quirk)",
      all(g is None or float(g.abs().sum()) == 0.0 for g in cell_grads))
check("C4", "NO non-GRU parameter received gradient from the aux loss",
      len(other_touched) == 0, f"touched: {other_touched[:5]}")
# once the head is live, the cell must start learning:
model.zero_grad(set_to_none=True)
with torch.no_grad():
    model.pose_gru.head.weight.normal_(0, 1e-3)
ress = rollout(model, batch, grad=True)
loss, _ = PoseGRULoss()(batch, ress)
loss.backward()
cell_ok = all(
    p.grad is not None and p.grad.abs().sum() > 0
    for n, p in model.pose_gru.named_parameters()
    if n.startswith("cell.")
)
check("C5", "cell grads nonzero once head is nonzero", cell_ok)
with torch.no_grad():
    model.pose_gru.head.weight.zero_()  # restore identity init

# =============================================================================
banner("STAGE D — full config train criterion on the real batch")
# =============================================================================
model.zero_grad(set_to_none=True)
ress = rollout(model, batch, grad=True)
loss, details = train_criterion(batch, ress, camera1=batch[0]["camera_pose"])
loss.backward()
dec_grad = any(
    p.grad is not None and p.grad.abs().sum() > 0
    for n, p in model.named_parameters() if n.startswith("dec_blocks.")
)
gru_grad = any(
    p.grad is not None and p.grad.abs().sum() > 0 for p in model.pose_gru.parameters()
)
check("D1", "full criterion finite", bool(torch.isfinite(loss)), f"loss={float(loss):.4f}")
check("D2", "gru details present in loss details",
      all(k in details for k in ("gru_pose_loss", "gru_trans_loss", "gru_quat_loss")),
      f"gru_pose_loss={details.get('gru_pose_loss')}")
check("D3", "decoder AND gru both receive grads (main + aux coexist)", dec_grad and gru_grad)
model.zero_grad(set_to_none=True)

# =============================================================================
banner("STAGE E — TBPTT-style chunked backward (chunk=2 groups, 4 chunks)")
# =============================================================================
try:
    rollout(model, batch, grad=True, chunk_groups=2, chunk_criterion=train_criterion)
    e_ok, e_msg = True, ""
except RuntimeError as e:
    e_ok, e_msg = False, str(e)[:150]
gru_grad = any(
    p.grad is not None and p.grad.abs().sum() > 0 for p in model.pose_gru.parameters()
)
check("E1", "per-chunk backward completes (hidden/pose detaches hold at boundaries)", e_ok, e_msg)
check("E2", "gru accumulated grads across chunks", gru_grad)
model.zero_grad(set_to_none=True)
model.eval()

# =============================================================================
banner("STAGE F — falsifiers")
# =============================================================================
os.environ["PREV_PRED_RAY_SHUFFLE"] = "1"
shuf = snap(rollout(model, batch))
os.environ.pop("PREV_PRED_RAY_SHUFFLE")
d0 = maxdiff(shuf[0]["camera_pose"], gru_snap[0]["camera_pose"])
d_later = max(diff_by_view(shuf, gru_snap, "camera_pose")[1:])
check("F1", "PREV_PRED_RAY_SHUFFLE: view 0 untouched, later views perturbed",
      d0 <= noise and d_later > 1e-4, f"view0 {d0:.1e}, max later {d_later:.3e}")

os.environ["POSE_GRU_HIDDEN_SHUFFLE"] = "1"
hshuf0 = snap(rollout(model, batch))
os.environ.pop("POSE_GRU_HIDDEN_SHUFFLE")
d_zero = max(diff_by_view(hshuf0, gru_snap, "camera_pose"))
check("F2", "HIDDEN_SHUFFLE is a no-op at zero-init head (negative control)", d_zero <= noise,
      f"max diff {d_zero:.1e}, floor {noise:.1e}")

torch.manual_seed(1234)
with torch.no_grad():
    model.pose_gru.head.weight.normal_(0, 5e-2)
live = snap(rollout(model, batch))
os.environ["POSE_GRU_HIDDEN_SHUFFLE"] = "1"
hshuf1 = snap(rollout(model, batch))
os.environ.pop("POSE_GRU_HIDDEN_SHUFFLE")
d_live = max(diff_by_view(hshuf1, live, "camera_pose"))
check("F3", "HIDDEN_SHUFFLE perturbs once head is live (falsifier reaches computation)",
      d_live > 1e-6, f"max diff {d_live:.3e}")
with torch.no_grad():
    model.pose_gru.head.weight.zero_()

# =============================================================================
banner("STAGE G — direct mode sanity")
# =============================================================================
model.enable_pose_gru(hidden_dim=128, mode="direct")
model.pose_gru.to(DEVICE)
direct = snap(rollout(model, batch))
d0 = maxdiff(direct[0]["camera_pose"], base1[0]["camera_pose"])
d_later = max(diff_by_view(direct, base1, "camera_pose")[1:])
qn = max(
    float((direct[v]["gru_pose"][:, 3:].norm(dim=-1) - 1).abs().max()) for v in range(1, NV)
)
check("G1", "direct mode: view 0 untouched, later views substantially changed",
      d0 <= noise and d_later > 1e-3, f"view0 {d0:.1e}, max later {d_later:.3e}")
check("G2", "direct-mode gru_pose quats unit-norm", qn < 1e-5, f"max |norm-1| {qn:.1e}")

# =============================================================================
banner("STAGE H — checkpoint round-trip and loader guard")
# =============================================================================
sd = {k: v.cpu() for k, v in model.state_dict().items()}
fresh = eval(MODEL_STR)
guard_raised = False
try:
    fresh.load_state_dict(sd)
except RuntimeError as e:
    guard_raised = "enable_pose_gru" in str(e)
check("H1", "loader guard refuses gru ckpt without enable_pose_gru", guard_raised)
fresh.enable_pose_gru(hidden_dim=128, mode="direct")
out = fresh.load_state_dict(sd)
check("H2", "loads strict with module enabled", str(out) == "<All keys matched successfully>",
      str(out)[:100])

# =============================================================================
banner("SUMMARY")
# =============================================================================
n_fail = sum(1 for *_, ok, _ in [(c[0], c[1], c[2], c[3]) for c in CHECKS] if not ok)
for gid, desc, ok, _ in CHECKS:
    print(f"  {'PASS' if ok else 'FAIL':4s}  {gid:3s}  {desc}")
print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
sys.exit(1 if n_fail else 0)
