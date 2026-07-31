#!/usr/bin/env python
"""
verify_gru_grid.py — GPU verification of the A1-A4 x G0-G3 lever grid
(pose_delta input, pose_gru_bptt, pose_gru_e2e) on the REAL pretrained
checkpoint, REAL wrist_test data, and the REAL loss_of_one_batch_tbptt path.

G arms: G0 = (bptt False, e2e False), G1 = (True, False), G2 = (False, True),
G3 = (True, True) — the SUPERSET arm, everything G1 does plus the main
reconstruction loss reaching the GRU. Note G2 is NOT a superset of G1: it is
G0 + e2e. A3/A4's G2 configs were converted in place to G3.

Stages:
  CFG   the 12 grid configs compose; lever keys per arm; twins differ only in levers
  UNIT  PoseGRU input widths; zero-init residual identity; pose_delta_encoding math
  ROLL  integration rollouts: delta wiring (captured GRU inputs vs recomputed),
        A4 zero-init parity, coherent PREV_PRED_RAY_SHUFFLE, direct+delta sanity
  GRAD  real tbptt: G0 seal, G1 within-chunk tape + boundary detach + no
        freed-graph crash + grad difference, G2 main-loss-only grads reach GRU,
        G3 superset: G1's tape AND G2's main-loss path in one pass
  EQ    forward equivalence: G flags never change forward math
  LOAD  load_model auto-enable of input_mode from weights; param-group split

Submit via:  sbatch verify_gru_grid.sbatch   (1x L40S)
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

# The GRAD stage compares gradient tensors between arms, so run-to-run
# nondeterminism in the backward (atomics in cuBLAS/cuDNN reductions) is the
# noise floor every differential check is measured against. Pin it down rather
# than raising thresholds to accommodate it. warn_only: a few ops in the DPT
# heads have no deterministic kernel — those stay nondeterministic and are
# absorbed by the max-over-repeats floor in the GRAD stage.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)

import dust3r.heads  # noqa: F401  MUST precede dust3r.utils.camera (circular import)
from dust3r.datasets.dl3dv import DL3DV_Multi
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.inference import loss_of_one_batch_tbptt
from dust3r.model import (
    ARCroco3DStereo,
    ARCroco3DStereoConfig,
    PoseGRU,
    load_model,
    pose_delta_encoding,
)
from dust3r.utils.camera import camera_to_pose_encoding, pose_encoding_to_camera
from dust3r.losses import (  # noqa: F401  names used by the criterion strings
    ConfLoss,
    L21,
    MSE,
    PoseGRULoss,
    RGBLoss,
    Regr3DPoseBatchList,
)
from torch.utils.data._utils.collate import default_collate

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
FULL_CRIT = (
    "ConfLoss(Regr3DPoseBatchList(L21, norm_mode='?avg_dis'), alpha=0.2)"
    " + RGBLoss(MSE) + 1.0*PoseGRULoss()"
)
MAIN_CRIT = "ConfLoss(Regr3DPoseBatchList(L21, norm_mode='?avg_dis'), alpha=0.2) + RGBLoss(MSE)"
NV = 8
RES = (320, 192)
IDXS = [5, 100]
SEED = 777
DEVICE = "cuda"
CFG_DIR = f"{WORKTREE}/src/CUT3R/config"
GRID = {  # arm -> (mode, input, bptt, e2e)
    "a1_g0": ("direct", "pose", False, False),
    "a1_g1": ("direct", "pose", True, False),
    "a1_g2": ("direct", "pose", False, True),
    "a2_g0": ("residual", "pose", False, False),
    "a2_g1": ("residual", "pose", True, False),
    "a2_g2": ("residual", "pose", False, True),
    "a3_g0": ("direct", "pose_delta", False, False),
    "a3_g1": ("direct", "pose_delta", True, False),
    # A3/A4's G2 arms were converted in place to G3 = G1 + G2 (bptt AND e2e),
    # the superset arm; A1/A2 keep the original G2 (G0 + e2e) for the record.
    "a3_g3": ("direct", "pose_delta", True, True),
    "a4_g0": ("residual", "pose_delta", False, False),
    "a4_g1": ("residual", "pose_delta", True, False),
    "a4_g3": ("residual", "pose_delta", True, True),
}
LEVER_KEYS = ("pose_gru_mode", "pose_gru_input", "pose_gru_bptt", "pose_gru_e2e")

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


# =============================================================================
banner("STAGE CFG — the 12 grid configs")
# =============================================================================
from hydra import compose, initialize_config_dir  # noqa: E402

with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
    cfgs = {arm: compose(config_name=f"captain_gru_v2_{arm}") for arm in GRID}
    base_cfg = compose(config_name="captain_gru_v2")

ok_keys, bad = True, []
for arm, (mode, inp, bptt, e2e) in GRID.items():
    c = cfgs[arm]
    got = (c.pose_gru_mode, c.pose_gru_input, bool(c.pose_gru_bptt), bool(c.pose_gru_e2e))
    if got != (mode, inp, bptt, e2e) or c.exp_name != f"captain_gru_v2_{arm}":
        ok_keys, bad = False, bad + [f"{arm}: {got} exp={c.exp_name}"]
check("CFG1", "all 12 arms compose with correct lever keys + exp_name", ok_keys, "; ".join(bad))

names = {cfgs[a].exp_name for a in GRID} | {cfgs[a].output_dir for a in GRID}
check("CFG2", "exp_names and output_dirs unique across arms", len(names) == 24)

allowed = set(LEVER_KEYS) | {"exp_name", "logdir", "output_dir"}
ok_diff, bad = True, []
for arm in GRID:
    dk = {
        k
        for k in set(cfgs[arm].keys()) | set(cfgs["a2_g0"].keys())
        if k != "hydra" and str(cfgs[arm].get(k)) != str(cfgs["a2_g0"].get(k))
    }
    if not dk <= allowed:
        ok_diff, bad = False, bad + [f"{arm}: {sorted(dk - allowed)}"]
check("CFG3", "every arm differs from a2_g0 ONLY in lever keys (+exp/dirs)", ok_diff, "; ".join(bad))

dk = {
    k
    for k in set(cfgs["a2_g0"].keys()) | set(base_cfg.keys())
    if k != "hydra" and str(cfgs["a2_g0"].get(k)) != str(base_cfg.get(k))
}
extra = dk - {"exp_name", "logdir", "output_dir", "pose_gru_input", "pose_gru_bptt", "pose_gru_e2e"}
a2 = cfgs["a2_g0"]
sem_same = (
    a2.pose_gru_input == "pose" and not a2.pose_gru_bptt and not a2.pose_gru_e2e
    and a2.pose_gru_mode == base_cfg.pose_gru_mode
)
check("CFG4", "a2_g0 == running captain_gru_v2 config (new keys at code defaults)",
      len(extra) == 0 and sem_same, f"unexpected diffs: {sorted(extra)}")

# =============================================================================
banner("STAGE UNIT — PoseGRU widths, zero-init identity, pose_delta_encoding math")
# =============================================================================
g7 = PoseGRU(input_mode="pose")
g14 = PoseGRU(input_mode="pose_delta")
check("U1", "input widths: pose -> 7, pose_delta -> 14",
      g7.cell.weight_ih.shape[1] == 7 and g14.cell.weight_ih.shape[1] == 14
      and g7.input_dim == 7 and g14.input_dim == 14)

pose = torch.randn(4, 7)
delta = torch.randn(4, 7)
exp = torch.cat([pose[:, :3], torch.nn.functional.normalize(pose[:, 3:7], dim=-1)], -1)
o7, _ = g7(pose, None)
o14, _ = g14(torch.cat([pose, delta], -1), None)
check("U2", "zero-init residual == renormalized input POSE part (7-d and 14-d)",
      maxdiff(o7, exp) < 1e-6 and maxdiff(o14, exp) < 1e-6,
      f"7d {maxdiff(o7, exp):.1e}, 14d {maxdiff(o14, exp):.1e}")

ident = pose_delta_encoding(None, pose)
check("U3", "first-step delta is identity motion [0,0,0,1,0,0,0]",
      torch.equal(ident, torch.tensor([[0, 0, 0, 1, 0, 0, 0]] * 4, dtype=ident.dtype)))

torch.manual_seed(42)
q = torch.nn.functional.normalize(torch.randn(16, 4), dim=-1) * torch.rand(16, 1).add(0.5)
enc_prev = torch.cat([torch.randn(16, 3), q[:8].repeat(2, 1)[:16]], -1)
q2 = torch.nn.functional.normalize(torch.randn(16, 4), dim=-1) * torch.rand(16, 1).add(0.5)
enc_cur = torch.cat([torch.randn(16, 3), q2], -1)
got = pose_delta_encoding(enc_prev, enc_cur)
C1 = pose_encoding_to_camera(enc_prev.float())
C2 = pose_encoding_to_camera(enc_cur.float())
ref = camera_to_pose_encoding(torch.linalg.inv(C1) @ C2)
check("U4", "pose_delta_encoding == encoding of inv(C_prev) @ C_cur (unnormalized quats ok)",
      maxdiff(got, ref) < 1e-4, f"maxdiff {maxdiff(got, ref):.2e}")
rt = pose_encoding_to_camera(got)
check("U5", "delta round-trips to a valid rigid transform",
      maxdiff(rt, torch.linalg.inv(C1) @ C2) < 1e-4)

# =============================================================================
banner("STAGE ROLL — setup: real data + real ckpt")
# =============================================================================
assert torch.cuda.is_available(), "this verification requires a GPU"
print(f"  device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}", flush=True)

ds = DL3DV_Multi(
    split="test", ROOT=DATA_ROOT, resolution=[RES], transform=ImgNorm,
    num_views=NV, n_corres=0, aug_crop=0, allow_repeat=True,
    force_consecutive_frame_sampling=True, seed=SEED,
)
view_lists = [ds[i] for i in IDXS]
batch = [to_device(default_collate([vl[v] for vl in view_lists])) for v in range(NV)]

inf = float("inf")
model = eval(MODEL_STR)  # trainer-identical build (freeze='encoder', both heads)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
missing = model.load_state_dict(ckpt["model"], strict=False)
print(f"  ckpt loaded strict=False: {len(missing.missing_keys)} missing, "
      f"{len(missing.unexpected_keys)} unexpected", flush=True)
del ckpt
model = model.to(DEVICE).eval()
model.views_per_step = 1
model.feed_prev_pred = True


class GRUSpy:
    """Wrap pose_gru.forward: record (gru_in, hidden_arg, raw stashes) per call.

    v3 note: the call site now always passes img_feat= (None for the F0 arms
    this file exercises); with iters>1 (R lever) there are N calls per view —
    every check below assumes the default N=1, which all v2 grid arms use.
    The v3 levers have their own verifier: verify_gru_v3_levers.py."""

    def __init__(self, model):
        self.model, self.calls = model, []
        self.orig = model.pose_gru.forward

    def __enter__(self):
        m = self.model

        def spy(gru_in, hidden, img_feat=None):
            # Exact cross-view credit probe. The hidden ARGUMENT is the only
            # tensor that carries credit from this call back into the previous
            # view's call (the returned hidden is also consumed by this view's
            # own head, so hooking that cannot separate the two sources).
            # Under a per-step detach the argument is tape-free and no hook can
            # fire; under BPTT it has a tape and a gradient physically arrives
            # during backward. Binary, so nondeterminism cannot blur it.
            rec = dict(
                    hidden_grad_absum=None,
                    gru_in=gru_in.detach().clone(),
                    img_feat=(None if img_feat is None else img_feat.detach().clone()),
                    hidden_has_tape=(hidden is not None and hidden.grad_fn is not None),
                    hidden_requires_grad=(hidden is not None and hidden.requires_grad),
                    stash_prev=(
                        None if m._prev_pred_pose_enc is None
                        else m._prev_pred_pose_enc.detach().clone()
                    ),
                    stash_prev_prev=(
                        None if getattr(m, "_prev_prev_pred_pose_enc", None) is None
                        else m._prev_prev_pred_pose_enc.detach().clone()
                    ),
            )
            if hidden is not None and hidden.requires_grad and hidden.grad_fn is not None:
                hidden.register_hook(
                    lambda g, _r=rec: _r.__setitem__(
                        "hidden_grad_absum", float(g.detach().abs().sum())
                    )
                )
            self.calls.append(rec)
            return self.orig(gru_in, hidden, img_feat=img_feat)

        m.pose_gru.forward = spy
        return self

    def __exit__(self, *a):
        self.model.pose_gru.forward = self.orig


def rollout(model, batch):
    """Plain sequential no-grad rollout (all G flags forward-identical)."""
    ress = []
    with torch.no_grad():
        (feat, pos, shape), (isf, imem, sf, sp, mem) = model._forward_encoder(batch)
        for v in range(len(batch)):
            res_group, (sf, mem) = model._forward_decoder_group_step(
                views=batch, view_indices=[v],
                feat_group=[feat[v]], pos_group=[pos[v]], shape_group=[shape[v]],
                init_state_feat=isf, init_mem=imem,
                state_feat=sf, state_pos=sp, mem=mem,
            )
            ress.append({k: t.detach().clone() for k, t in res_group[0].items()})
    return ress


base = rollout(model, batch)          # no GRU: plain prev_pred reference
base2 = rollout(model, batch)
noise = max(maxdiff(a["camera_pose"], b["camera_pose"]) for a, b in zip(base, base2))
check("R1", "determinism floor (plain prev_pred twice)", noise == 0.0, f"floor {noise:.1e}")

model.enable_pose_gru(hidden_dim=128, mode="residual", input_mode="pose_delta")
model.pose_gru.to(DEVICE)
with GRUSpy(model) as spy:
    a4 = rollout(model, batch)

dp = max(maxdiff(a["camera_pose"], b["camera_pose"]) for a, b in zip(a4, base))
dx = max(
    maxdiff(a["pts3d_in_self_view"], b["pts3d_in_self_view"]) for a, b in zip(a4, base)
)
check("R2", "A4 zero-init parity: residual+delta == plain prev_pred (fp tolerance)",
      dp < 1e-3 and dx < 1e-2, f"pose {dp:.1e}, pts {dx:.1e}")

c0 = spy.calls[0]["gru_in"]
id_enc = torch.zeros_like(c0[:, 7:14])
id_enc[:, 3] = 1.0
check("R3", "view-1 GRU input: pose part == stashed P(0), delta part == identity motion",
      torch.equal(c0[:, :7], spy.calls[0]["stash_prev"].float())
      and torch.equal(c0[:, 7:14], id_enc))

ok_pose = all(
    torch.equal(spy.calls[v - 1]["gru_in"][:, :7], a4[v - 1]["camera_pose"].float())
    for v in range(2, NV)
)
ok_delta = all(
    maxdiff(
        spy.calls[v - 1]["gru_in"][:, 7:14],
        pose_delta_encoding(a4[v - 2]["camera_pose"], a4[v - 1]["camera_pose"]),
    ) < 1e-6
    for v in range(2, NV)
)
ok_stash = all(
    torch.equal(spy.calls[v - 1]["stash_prev_prev"], a4[v - 2]["camera_pose"])
    for v in range(2, NV)
)
check("R4", "view x>=2 GRU input: pose == P(x-1), delta == enc(inv(P(x-2))@P(x-1)), "
      "stash rotation exact", ok_pose and ok_delta and ok_stash)

os.environ["PREV_PRED_RAY_SHUFFLE"] = "1"
with GRUSpy(model) as sspy:
    shuf = rollout(model, batch)
os.environ.pop("PREV_PRED_RAY_SHUFFLE")
ok_roll = all(
    torch.equal(c["gru_in"][:, :7], torch.roll(c["stash_prev"], 1, 0).float())
    and (
        c["stash_prev_prev"] is None
        or maxdiff(
            c["gru_in"][:, 7:14],
            pose_delta_encoding(
                torch.roll(c["stash_prev_prev"], 1, 0), torch.roll(c["stash_prev"], 1, 0)
            ),
        ) < 1e-6
    )
    for c in sspy.calls
)
d0 = maxdiff(shuf[0]["camera_pose"], a4[0]["camera_pose"])
dl = max(maxdiff(a["camera_pose"], b["camera_pose"]) for a, b in zip(shuf[1:], a4[1:]))
check("R5", "PREV_PRED_RAY_SHUFFLE rolls BOTH stashes coherently (pose AND delta)",
      ok_roll, f"{len(sspy.calls)} calls checked")
check("R6", "shuffle: view 0 untouched, later views perturbed",
      d0 <= noise and dl > 1e-4, f"view0 {d0:.1e}, later {dl:.2e}")

model.enable_pose_gru(hidden_dim=128, mode="direct", input_mode="pose_delta")
model.pose_gru.to(DEVICE)
a3 = rollout(model, batch)
d0 = maxdiff(a3[0]["camera_pose"], base[0]["camera_pose"])
dl = max(maxdiff(a["camera_pose"], b["camera_pose"]) for a, b in zip(a3[1:], base[1:]))
check("R7", "A3 (direct+delta): view 0 untouched, later views substantially changed",
      d0 <= noise and dl > 1e-3, f"view0 {d0:.1e}, later {dl:.2e}")

# =============================================================================
banner("STAGE GRAD — real loss_of_one_batch_tbptt: G0 seal, G1 tape, G2 e2e")
# =============================================================================
from accelerate import Accelerator  # noqa: E402
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa: E402

accelerator = Accelerator()
model.enable_pose_gru(hidden_dim=128, mode="residual", input_mode="pose_delta")
model.pose_gru.to(DEVICE)
torch.manual_seed(1234)
with torch.no_grad():  # nudge the head: at zero-init the cell gets no gradient (C3 quirk)
    model.pose_gru.head.weight.normal_(0, 1e-3)
gru_sd0 = {k: v.detach().clone() for k, v in model.pose_gru.state_dict().items()}
full_crit = eval(FULL_CRIT).to(DEVICE)
main_crit = eval(MAIN_CRIT).to(DEVICE)


def run_tbptt(criterion, bptt, e2e):
    """One real tbptt pass (8 views, chunk 4 -> 2 grad chunks). lr=0 optimizer
    so weights never move; grads snapshotted just before each zero_grad."""
    model.pose_gru_bptt = bptt
    model.pose_gru_e2e = e2e
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)
    scaler = NativeScaler(accelerator=accelerator)
    snaps, spy_calls = [], []
    orig_zero = opt.zero_grad

    def zero_spy(*a, **k):
        snaps.append({
            n: (p.grad.detach().float().clone() if p.grad is not None else None)
            for n, p in model.pose_gru.named_parameters()
        })
        orig_zero(*a, **k)

    opt.zero_grad = zero_spy
    err = None
    try:
        with GRUSpy(model) as spy:
            loss_of_one_batch_tbptt(
                batch, model, criterion, 4, scaler, opt, accelerator, inference=False
            )
            spy_calls = spy.calls
    except RuntimeError as e:
        err = str(e)[:200]
    model.train(False)
    model.pose_gru_bptt = False
    model.pose_gru_e2e = False
    model.zero_grad(set_to_none=True)
    return snaps, spy_calls, err


def gsum(snaps, prefix=""):
    tot = 0.0
    for s in snaps:
        for n, g in s.items():
            if g is not None and n.startswith(prefix):
                tot += float(g.abs().sum())
    return tot


def gcat(snaps, name):
    return torch.stack([
        s[name] if s[name] is not None else torch.zeros_like(gru_sd0[name]) for s in snaps
    ])


# --- G0: seal + hidden always tape-free
s_g0, calls_g0, err = run_tbptt(full_crit, bptt=False, e2e=False)
tape_flags = [c["hidden_has_tape"] for c in calls_g0[1:]]  # call 0 gets hidden=None
check("T1", "G0 tbptt completes; aux grads reach head AND cell",
      err is None and gsum(s_g0, "head.") > 0 and gsum(s_g0, "cell.") > 0,
      err or f"|head| {gsum(s_g0, 'head.'):.2e}, |cell| {gsum(s_g0, 'cell.'):.2e}")
check("T2", "G0: hidden entering every GRU call is tape-free (per-step seal)",
      not any(tape_flags), f"tape flags: {tape_flags}")

# Nondeterminism floor. EVERY differential check below is measured against
# this, so a single noisy sample makes the whole GRAD stage flaky: run
# 1409920 drew floor 5.89e-07 and run 1428021 drew 1.25e-06 while the signals
# they gate were bit-stable across both (T6 was 1.51e-05 in each). Two fixes:
# deterministic kernels at import time (see the torch setup near the top) to
# shrink the noise, and a max over REPEATS here so the estimate is an upper
# bound rather than one draw.
G0_REPEATS = 3
s_g0_reps = [s_g0] + [
    run_tbptt(full_crit, bptt=False, e2e=False)[0] for _ in range(G0_REPEATS - 1)
]


def pair_floor(a, b):
    return max(
        (maxdiff(a[i][n], b[i][n])
         for i in range(len(a)) for n in a[i]
         if a[i][n] is not None and b[i][n] is not None),
        default=0.0,
    )


floor = max(
    pair_floor(s_g0_reps[i], s_g0_reps[j])
    for i in range(len(s_g0_reps)) for j in range(i + 1, len(s_g0_reps))
)
s_g0b = s_g0_reps[1]
check("T3", f"grad determinism baseline (G0 x{G0_REPEATS} -> near-identical grads)",
      floor < 1e-6, f"floor {floor:.2e} (max over {G0_REPEATS} repeats)")

# --- G1: within-chunk tape, boundary detach, grads actually differ
s_g1, calls_g1, err = run_tbptt(full_crit, bptt=True, e2e=False)
# GRU calls happen at views 1..7; chunks are views 0-3 / 4-7. Hidden entering
# view v was written at v-1: in-chunk for v in {2,3, 5,6,7} (tape expected),
# boundary-crossing for v=4 (detached by inference.py -> tape-free).
tape = {v: calls_g1[v - 1]["hidden_has_tape"] for v in range(2, NV)}
expect = {v: (v != 4) for v in range(2, NV)}
check("T4", "G1 tbptt completes WITHOUT freed-graph crash (boundary detach works)",
      err is None, err or "")
check("T5", "G1: hidden tape spans steps IN-chunk, cut exactly at the chunk boundary",
      tape == expect, f"got {tape}, expected {expect}")
dcell = maxdiff(gcat(s_g1, "cell.weight_hh"), gcat(s_g0, "cell.weight_hh"))
check("T6", "G1 vs G0: cell.weight_hh grads differ (multi-step credit is real)",
      dcell > max(1e-9, 10 * floor), f"maxdiff {dcell:.2e} vs floor {floor:.2e}")

# --- G2: main loss alone reaches the GRU; sealed under G0
s_seal, _, err0 = run_tbptt(main_crit, bptt=False, e2e=False)
s_e2e, s_e2e_calls, err2 = run_tbptt(main_crit, bptt=False, e2e=True)
check("T7", "main-only criterion under G0: pose_gru grads exactly zero (seal intact)",
      err0 is None and gsum(s_seal) == 0.0, err0 or f"|grads| {gsum(s_seal):.2e}")
check("T8", "main-only criterion under G2: pose_gru head AND cell get grads via ray build",
      err2 is None and gsum(s_e2e, "head.") > 0 and gsum(s_e2e, "cell.") > 0,
      err2 or f"|head| {gsum(s_e2e, 'head.'):.2e}, |cell| {gsum(s_e2e, 'cell.'):.2e}")
frozen_live = [
    n for n, p in model.named_parameters()
    if ("enc_blocks_ray_map" in n or "patch_embed_ray_map" in n) and p.requires_grad
]
s_g2f, _, errf = run_tbptt(full_crit, bptt=False, e2e=True)
check("T9", "G2 with the full criterion completes; ray-encoder freeze intact "
      "(requires_grad=False, so e2e grads pass THROUGH it, never INTO it)",
      errf is None and not frozen_live, errf or f"unfrozen: {frozen_live[:3]}")

# --- G3 = G1 + G2: the superset arm. Everything G1 does (within-chunk hidden
# tape, boundary-detached) PLUS everything G2 does (main loss reaching the GRU
# through the ray build). The two levers are independent booleans, so this
# combination was never exercised before the G3 arms were created — these
# checks are what license spending GPU time on them.
s_g3, calls_g3, err3 = run_tbptt(full_crit, bptt=True, e2e=True)
check("T10", "G3 (bptt+e2e) tbptt completes — no freed-graph crash when the "
      "hidden tape and the ray-build tape coexist in one chunk backward",
      err3 is None, err3 or "")
tape3 = {v: calls_g3[v - 1]["hidden_has_tape"] for v in range(2, NV)}
check("T11", "G3 inherits G1's tape topology exactly (in-chunk tape, cut at the "
      "chunk boundary)", err3 is None and tape3 == expect, f"got {tape3}, expected {expect}")
# The main loss must reach the GRU under G3 just as it does under G2: run the
# main-only criterion so the aux term cannot mask a dead e2e path.
s_g3m, calls_g3m, err3m = run_tbptt(main_crit, bptt=True, e2e=True)
check("T12", "G3 inherits G2's main-loss path (main-only criterion still grads "
      "head AND cell)",
      err3m is None and gsum(s_g3m, "head.") > 0 and gsum(s_g3m, "cell.") > 0,
      err3m or f"|head| {gsum(s_g3m, 'head.'):.2e}, |cell| {gsum(s_g3m, 'cell.'):.2e}")
# And it must be a STRICT superset. Tested by ORTHOGONALITY, not by
# differencing gradient tensors: `floor` is measured on G0 under the full
# criterion, where the GRU's gradient is almost entirely the aux PoseGRULoss
# -- a tiny, nearly deterministic path (2.9e-07). Every e2e arm instead routes
# gradient through the ray build, ray encoder, decoder and DPT heads, and THAT
# backward is ~70x noisier (a G2-vs-G2 repeat differs by 2.1e-05, larger than
# the G3-vs-G2 difference of 1.9e-05). Gating an e2e comparison on the G0
# floor silently over-claims, and gating it on the honest paired floor is
# simply inconclusive. So test the mechanism instead of its magnitude.
#
# The e2e half is exactly decidable: with the MAIN-ONLY criterion an arm
# without e2e must receive EXACTLY zero GRU gradient (T7 shows this for G0).
# G1 must therefore also be exactly zero, and G3 must not be -- that is G3
# owning G2's mechanism, with no threshold anywhere. T14 below does the same
# for the BPTT half. Together they pin the superset from both sides.
s_g1m, _, err1m = run_tbptt(main_crit, bptt=True, e2e=False)
g1m_tot, g3m_tot = gsum(s_g1m), gsum(s_g3m)
d31 = maxdiff(gcat(s_g3, "cell.weight_hh"), gcat(s_g1, "cell.weight_hh"))
d32 = maxdiff(gcat(s_g3, "cell.weight_hh"), gcat(s_g2f, "cell.weight_hh"))
check("T13", "G3 owns G2's mechanism exactly: under the main-only criterion G1 "
      "gets zero GRU gradient (no e2e path) while G3 does not",
      err1m is None and g1m_tot == 0.0 and g3m_tot > 0.0,
      err1m or f"G1 |grads| {g1m_tot:.2e} (must be 0), G3 |grads| {g3m_tot:.2e}; "
      f"FYI noisy tensor diffs vs G1 {d31:.2e}, vs G2 {d32:.2e}")
# Multi-step credit under e2e: with the main-only criterion, G3 must differ
# from G2 -- that difference IS the reconstruction loss reaching earlier
# steps' cell calls through the hidden, which is the whole point of the arm.
# EXACT, not statistical. Two earlier attempts to settle this by differencing
# gradient tensors both failed on noise, and the failure is structural rather
# than fixable by tuning: the term travels the e2e backward, whose own
# run-to-run spread (2.1e-05) is LARGER than the term itself (7.3e-06). No
# threshold on that comparison can be both honest and decisive.
#   - run 1428195 amplified the head x100 to lift the signal: signal +48x but
#     paired noise +300x, SNR 6x -> 3.8x. Wild poses make ray maps, and hence
#     the backward's reduction order, far more variable. Do not retry.
#   - run 1428210 measured at the normal head vs a paired floor: 7.28e-06 vs
#     2.11e-05. Inconclusive, correctly reported as a FAIL.
# So ask the question the gradient itself answers. The hidden ARGUMENT of a
# call is the sole route by which credit leaves this view for the previous
# one. Under G2 it is detached, so no hook can ever fire on it; under G3 it
# carries a tape and a gradient physically arrives. With the MAIN-ONLY
# criterion, a gradient arriving there is proof that the reconstruction loss
# -- not the aux loss -- reached an earlier step's cell call. Binary.
in_chunk = [v for v in range(2, NV) if v != 4]  # chunks are views 0-3 / 4-7
g3_arrivals = {v: calls_g3m[v - 1]["hidden_grad_absum"] for v in in_chunk}
g2_arrivals = {v: s_e2e_calls[v - 1]["hidden_grad_absum"] for v in in_chunk}
g3_live = [v for v, a in g3_arrivals.items() if a is not None and a > 0.0]
g2_live = [v for v, a in g2_arrivals.items() if a is not None and a > 0.0]
check("T14", "under the main-only criterion the reconstruction loss physically "
      "reaches EARLIER steps' cell calls through the hidden under G3, and "
      "cannot under G2",
      len(g3_live) == len(in_chunk) and not g2_live,
      f"G3 gradient arrived at views {g3_live} (expected {in_chunk}); "
      f"G2 arrived at {g2_live} (expected none)")

# =============================================================================
banner("STAGE EQ — G flags change gradients only, never forward math")
# =============================================================================
with torch.no_grad():
    model.pose_gru.load_state_dict(gru_sd0)  # nudged-head state, same for all arms
model.eval()
outs = {}
G_ARMS = {"g0": (False, False), "g1": (True, False), "g2": (False, True), "g3": (True, True)}
for name, (bptt, e2e) in G_ARMS.items():
    model.pose_gru_bptt, model.pose_gru_e2e = bptt, e2e
    outs[name] = rollout(model, batch)
model.pose_gru_bptt = model.pose_gru_e2e = False
eq = all(
    torch.equal(outs["g0"][v][k], outs[g][v][k])
    for g in ("g1", "g2", "g3") for v in range(NV) for k in ("camera_pose", "pts3d_in_self_view")
)
check("E1", "no-grad rollouts bit-identical across G0/G1/G2/G3 (eval-time equivalence)", eq)

# =============================================================================
banner("STAGE LOAD — load_model auto-enable of input_mode; param-group split")
# =============================================================================
mini = {
    "model": {f"pose_gru.{k}": v.cpu() for k, v in model.pose_gru.state_dict().items()},
    "args": cfgs["a4_g0"],  # residual + pose_delta
}
mini_path = os.path.join(
    os.environ.get("SCRATCH_DIR", "/scratch/bdursun25/cuteanything/captain_gru_v3"),
    "tmp_gru_grid_mini_ckpt.pth",
)
torch.save(mini, mini_path)
net = load_model(mini_path, device="cpu", verbose=True)
g = net.pose_gru
check("L1", "load_model auto-enable: input_mode=pose_delta from weights, mode from args",
      g is not None and g.input_mode == "pose_delta" and g.mode == "residual"
      and g.cell.weight_ih.shape[1] == 14)
check("L2", "trained GRU weights actually loaded",
      g is not None and torch.equal(g.cell.weight_ih, mini["model"]["pose_gru.cell.weight_ih"]))
os.remove(mini_path)
del net

from train_cut3r_baseline import split_pose_gru_param_groups  # noqa: E402
import croco.utils.misc as misc  # noqa: E402

pgs = split_pose_gru_param_groups(
    misc.get_parameter_groups(model, 0.05), model.pose_gru, 100.0
)
gru_ids = {id(p) for p in model.pose_gru.parameters()}
gru_groups = [g for g in pgs if any(id(p) in gru_ids for p in g["params"])]
check("L3", "split_pose_gru_param_groups isolates the 14-d GRU with lr_scale=100",
      len(gru_groups) > 0 and all(g.get("lr_scale") == 100.0 for g in gru_groups)
      and sum(len(g["params"]) for g in gru_groups) == len(list(model.pose_gru.parameters())))

# =============================================================================
banner("SUMMARY")
# =============================================================================
n_fail = sum(1 for _, _, ok, _ in CHECKS if not ok)
for gid, desc, ok, _ in CHECKS:
    print(f"  {'PASS' if ok else 'FAIL':4s}  {gid:4s}  {desc}")
print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
sys.exit(1 if n_fail else 0)
