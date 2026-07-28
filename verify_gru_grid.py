#!/usr/bin/env python
"""
verify_gru_grid.py — GPU verification of the A1-A4 x G0-G2 lever grid
(pose_delta input, pose_gru_bptt, pose_gru_e2e) on the REAL pretrained
checkpoint, REAL wrist_test data, and the REAL loss_of_one_batch_tbptt path.

Stages:
  CFG   the 12 grid configs compose; lever keys per arm; twins differ only in levers
  UNIT  PoseGRU input widths; zero-init residual identity; pose_delta_encoding math
  ROLL  integration rollouts: delta wiring (captured GRU inputs vs recomputed),
        A4 zero-init parity, coherent PREV_PRED_RAY_SHUFFLE, direct+delta sanity
  GRAD  real tbptt: G0 seal, G1 within-chunk tape + boundary detach + no
        freed-graph crash + grad difference, G2 main-loss-only grads reach GRU
  EQ    forward equivalence: G flags never change forward math
  LOAD  load_model auto-enable of input_mode from weights; param-group split

Submit via:  sbatch verify_gru_grid.sbatch   (1x L40S)
"""
import os
import sys

WORKTREE = "/scratch/bdursun25/cuteanything/captain_gru_v2"
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
    "a3_g2": ("direct", "pose_delta", False, True),
    "a4_g0": ("residual", "pose_delta", False, False),
    "a4_g1": ("residual", "pose_delta", True, False),
    "a4_g2": ("residual", "pose_delta", False, True),
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
    """Wrap pose_gru.forward: record (gru_in, hidden_arg, raw stashes) per call."""

    def __init__(self, model):
        self.model, self.calls = model, []
        self.orig = model.pose_gru.forward

    def __enter__(self):
        m = self.model

        def spy(gru_in, hidden):
            self.calls.append(
                dict(
                    gru_in=gru_in.detach().clone(),
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
            )
            return self.orig(gru_in, hidden)

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

s_g0b, _, _ = run_tbptt(full_crit, bptt=False, e2e=False)
floor = max(
    (maxdiff(s_g0[i][n], s_g0b[i][n])
     for i in range(len(s_g0)) for n in s_g0[i]
     if s_g0[i][n] is not None and s_g0b[i][n] is not None),
    default=0.0,
)
check("T3", "grad determinism baseline (G0 twice -> near-identical grads)", floor < 1e-6,
      f"floor {floor:.2e}")

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
s_e2e, _, err2 = run_tbptt(main_crit, bptt=False, e2e=True)
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

# =============================================================================
banner("STAGE EQ — G flags change gradients only, never forward math")
# =============================================================================
with torch.no_grad():
    model.pose_gru.load_state_dict(gru_sd0)  # nudged-head state, same for all arms
model.eval()
outs = {}
for name, (bptt, e2e) in {"g0": (False, False), "g1": (True, False), "g2": (False, True)}.items():
    model.pose_gru_bptt, model.pose_gru_e2e = bptt, e2e
    outs[name] = rollout(model, batch)
model.pose_gru_bptt = model.pose_gru_e2e = False
eq = all(
    torch.equal(outs["g0"][v][k], outs[g][v][k])
    for g in ("g1", "g2") for v in range(NV) for k in ("camera_pose", "pts3d_in_self_view")
)
check("E1", "no-grad rollouts bit-identical across G0/G1/G2 (eval-time equivalence)", eq)

# =============================================================================
banner("STAGE LOAD — load_model auto-enable of input_mode; param-group split")
# =============================================================================
mini = {
    "model": {f"pose_gru.{k}": v.cpu() for k, v in model.pose_gru.state_dict().items()},
    "args": cfgs["a4_g0"],  # residual + pose_delta
}
mini_path = os.path.join(
    os.environ.get("SCRATCH_DIR", "/scratch/bdursun25/cuteanything/captain_gru_v2"),
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
