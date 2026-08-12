#!/usr/bin/env python
"""verify_gru_ray_gate.py — falsifier battery for P2 (pose_gru_ray_gate).

CPU only, no data, no checkpoint, seconds to run:

    python verify_gru_ray_gate.py

The gate is a 1-unit head off the PoseGRU hidden emitting a per-sample scalar
`a`, blended into the RAY ADD rather than into the GRU's correction:

    feat + a * ray_token + (1 - a) * masked_ray_map_token

Three properties have to hold before it can ride a run, each one a way this
could be silently wrong rather than loudly broken:

  1. ABSENT KEY => BYTE-IDENTICAL. No submodule, no state_dict keys, the same
     ray add. A gate that quietly defaulted to "built but open" would add two
     state_dict keys every pre-P2 checkpoint lacks, and the loader hard-fails
     on that -- so "off" has to mean absent, not neutral.
  2. THE GATE IS ACTUALLY ON THE RAY-ADD PATH, and its two limits are the two
     arms the directive names:
       a -> 1  reproduces the current conditioned arm BIT-FOR-BIT;
       a -> 0  reproduces the POSE-FREE construction view 0 already takes
               (feat + masked_ray_map_token) BIT-FOR-BIT -- i.e. the floor of
               this arm is arm A, not arm C. That is the entire point of
               gating the ray instead of the correction, and it is the half a
               gate that is built, serialized and logged but never multiplied
               in would still pass;
       intermediate a CHANGES the output, per sample, by the value the module
               computes from the hidden it just produced.
  3. IT SURVIVES A CHECKPOINT ROUND TRIP. The head is presence-gated, so
     _sniff_pose_gru_config has to rebuild it from key presence or a gated
     checkpoint fails to load with "unexpected key".

Property 2 is tested against the REAL call site. The blend lives ~400 lines
inside _forward_decoder_group_step, a method that otherwise wants a full CUT3R
(24 encoder blocks, 12 decoder blocks, a DPT head); rather than re-implement
the arithmetic -- which would pass just as happily if the call site never ran
it -- this file borrows the shipping method onto a duck-typed host and stubs
only what surrounds the blend (ray encoder, recurrent rollout, downstream
head). Everything from the GRU call to the ray add is executed verbatim.

Exit 0 = all checks pass.
"""
import os
import re
import sys
from types import SimpleNamespace

WORKTREE = os.path.dirname(os.path.abspath(__file__))
assert os.path.isfile(
    os.path.join(WORKTREE, "src", "CUT3R", "src", "train_cut3r_baseline.py")
), f"not a my-da3 checkout: {WORKTREE}"
for p in [
    os.path.join(WORKTREE, "src"),
    os.path.join(WORKTREE, "src/CUT3R"),
    os.path.join(WORKTREE, "src/CUT3R/src"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# The falsifier env vars are read inside the method under test; a stray one in
# the caller's environment would quietly change every number below.
for _fv in (
    "PREV_PRED_RAY_SHUFFLE",
    "POSE_GRU_HIDDEN_SHUFFLE",
    "POSE_GRU_HIDDEN_ZERO",
    "POSE_GRU_IMG_FEAT_SHUFFLE",
    "POSE_GRU_IMG_FEAT_ZERO",
    "POSE_GRU_FORCE_ITERS",
):
    os.environ.pop(_fv, None)

import torch

import dust3r.heads  # noqa: F401  MUST precede dust3r.utils.camera (circular import)
from dust3r.model import (
    RAY_GATE_BIAS_INIT,
    ARCroco3DStereo,
    PoseGRU,
    _sniff_pose_gru_config,
)

ENC_DIM = 64  # token width; the real arm's 1024 costs time and proves nothing
HID = 32
B, NTOK = 4, 6
H, W = 16, 16
FAILURES = []
MODEL_SRC = open(
    os.path.join(WORKTREE, "src/CUT3R/src/dust3r/model.py"), encoding="utf-8"
).read()
TRAIN_SRC = open(
    os.path.join(WORKTREE, "src/CUT3R/src/train_cut3r_baseline.py"), encoding="utf-8"
).read()


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


A4 = dict(mode="residual", input_mode="pose_delta", hidden_dim=HID)


def build(seed=0, **kw):
    torch.manual_seed(seed)
    return PoseGRU(**dict(A4, **kw)).eval()


def unlock(gru, seed=11):
    """Give the residual module a non-zero head and a non-degenerate cell.

    mode='residual' zero-inits the head, which is irrelevant to the gate (it
    reads the HIDDEN, not the head) but makes the emitted pose constant. Kept
    so the two arms compared below differ in the gate and nothing else.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        gru.head.weight.normal_(0, 0.05)
        gru.head.bias.normal_(0, 0.05)
    return gru


def force_gate(gru, logit=None, weight_std=0.0, seed=5):
    """Drive `a` to a known value. logit=+-1000 saturates sigmoid to EXACTLY
    1.0 / 0.0 in fp32 (exp overflows), which is what makes the two limits
    bit-identity checks rather than tolerance checks."""
    torch.manual_seed(seed)
    with torch.no_grad():
        if weight_std:
            gru.ray_gate.weight.normal_(0, weight_std)
        else:
            gru.ray_gate.weight.zero_()
        if logit is not None:
            gru.ray_gate.bias.fill_(float(logit))
    return gru


# ---------------------------------------------------------------------------
# The harness: the REAL _forward_decoder_group_step on a duck-typed host.
# ---------------------------------------------------------------------------
torch.manual_seed(3)
FEAT = torch.randn(B, NTOK, ENC_DIM)  # pre-ray encoder tokens of view 1
RAY_TOKEN = torch.randn(B, NTOK, ENC_DIM)  # what _encode_ray_map returns
MASKED_TOKEN = torch.randn(1, ENC_DIM) * 0.02  # shaped like the real Parameter
POS = torch.zeros(B, NTOK, 2, dtype=torch.long)
SHAPE = torch.tensor([[H, W]] * B)
STATE = torch.randn(B, 3, ENC_DIM)
MEM = torch.randn(B, 3, ENC_DIM)
STATE_POS = torch.zeros(B, 3, 2, dtype=torch.long)


def _pose(seed):
    torch.manual_seed(seed)
    t = torch.randn(B, 3) * 0.1
    q = torch.nn.functional.normalize(torch.randn(B, 4), dim=-1)
    return torch.cat([t, q], dim=-1)


PREV, PREV2, NEXT = _pose(21), _pose(22), _pose(23)
K = torch.tensor([[[W * 1.2, 0.0, W / 2], [0.0, W * 1.2, H / 2], [0.0, 0.0, 1.0]]])
VIEWS = [
    {
        "img": torch.zeros(B, 3, H, W),
        "camera_intrinsics": K.expand(B, 3, 3).contiguous(),
        "img_mask": torch.ones(B, dtype=torch.bool),
        "reset": torch.zeros(B, dtype=torch.bool),
    }
    for _ in range(4)  # enough to step views 1..3 through ONE host
]


class RayAddHarness:
    # The shipping code under test, borrowed verbatim.
    _forward_decoder_group_step = ARCroco3DStereo._forward_decoder_group_step
    _concat_group_feat_pos = ARCroco3DStereo._concat_group_feat_pos
    _record_ray_gate = ARCroco3DStereo._record_ray_gate
    pop_ray_gate_stats = ARCroco3DStereo.pop_ray_gate_stats

    pose_head_flag = False
    dec_depth = 4
    feed_prev_pred = True
    pose_gru_oracle = "off"
    pose_gru_iter_detach = True
    pose_gru_bptt = False

    def __init__(self, pose_gru, e2e=False, dtype=torch.float32):
        self.pose_gru = pose_gru
        self.pose_gru_e2e = e2e
        self._ray_token = RAY_TOKEN.to(dtype)
        self.masked_ray_map_token = MASKED_TOKEN.to(dtype)
        self._prev_pred_pose_enc = PREV
        self._prev_prev_pred_pose_enc = PREV2
        self._pose_gru_hidden = None
        self._prev_img_feat = None
        self._ray_gate_stats = {}
        self.blended = None
        self._head_calls = 0

    # --- stubs: everything AROUND the blend, none of it between GRU and ray add
    def _encode_ray_map(self, rmap, shape):
        # Held CONSTANT so a difference between two runs can only come from
        # the gate, never from the ray encoder or the pose that fed it.
        return [self._ray_token], None, None

    def _recurrent_rollout(self, state_feat, state_pos, current_feat, current_pos,
                           pose_feat, pose_pos, init_state_feat):
        # current_feat IS the blended token grid — the observable under test.
        self.blended = current_feat
        return state_feat, [current_feat] * (self.dec_depth + 1)

    def _downstream_head(self, decout, img_shape, **kw):
        # Call 0 returns NEXT exactly (so every single-step check is unchanged);
        # later calls move the translation so a multi-view run does not feed the
        # GRU a constant and go degenerate.
        k = self._head_calls
        self._head_calls += 1
        if k == 0:
            return {"camera_pose": NEXT}
        return {"camera_pose": torch.cat([NEXT[:, :3] * (1.0 + 0.1 * k),
                                          NEXT[:, 3:]], dim=-1)}


def run(gru, e2e=False, grad=False, hidden0=None, dtype=torch.float32):
    """One decode step for view 1. Returns (harness, res_group).

    hidden0=None is the sequence-start regime (the cell runs from zeros, which
    is what view 1 actually gets); pass a tensor to stand in for a mid-sequence
    step, where the recurrent weights are no longer multiplied by a zero state.
    dtype stands in for the ambient AMP regime: under training the tokens are
    half and only the GRU island is fp32.
    """
    host = RayAddHarness(gru, e2e=e2e, dtype=dtype)
    host._pose_gru_hidden = hidden0
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        res_group, _ = host._forward_decoder_group_step(
            views=VIEWS,
            view_indices=[1],
            feat_group=[FEAT.to(dtype)],
            pos_group=[POS],
            shape_group=[SHAPE],
            init_state_feat=STATE,
            init_mem=MEM,
            state_feat=STATE,
            state_pos=STATE_POS,
            mem=MEM,
        )
    return host, res_group


print("\n=== 1. absent key => byte-identical previous behavior ===")
off = unlock(build(**A4))
check("no ray_gate attribute value", off.ray_gate is None)
check(
    "no state_dict keys",
    not [k for k in off.state_dict() if k.startswith("ray_gate")],
    f"keys: {sorted(off.state_dict())}",
)
host_off, res_off = run(off)
UNGATED = FEAT + RAY_TOKEN  # the pre-P2 ray add, model.py `+ ray_out[-1]`
UNCONDITIONED = FEAT + MASKED_TOKEN  # the pose-free path view 0 takes
check(
    "ungated call site is exactly feat + ray_token",
    torch.equal(host_off.blended, UNGATED),
    f"max|d| {(host_off.blended - UNGATED).abs().max():.3e}",
)
check("ungated step records no gate telemetry", host_off.pop_ray_gate_stats() == {})
check("ungated step writes no gru_ray_gate key", "gru_ray_gate" not in res_off[-1])
check(
    "the two reference arms are actually different tensors",
    not torch.allclose(UNGATED, UNCONDITIONED, atol=1e-6),
)

print("\n=== 2. the gate is built, presence-gated, and open at init ===")
gated_init = build(**A4, ray_gate=True)
check("ray_gate head exists", isinstance(gated_init.ray_gate, torch.nn.Linear))
check(
    "one logit off the hidden",
    tuple(gated_init.ray_gate.weight.shape) == (1, HID),
    f"got {tuple(gated_init.ray_gate.weight.shape)}",
)
check(
    "state_dict now carries both keys",
    {"ray_gate.weight", "ray_gate.bias"} <= set(gated_init.state_dict()),
)
check("weight zero-init", torch.equal(gated_init.ray_gate.weight,
                                      torch.zeros(1, HID)))
check(
    f"bias init {RAY_GATE_BIAS_INIT} => a is the same for every sample",
    torch.equal(gated_init.ray_gate.bias, torch.full((1,), RAY_GATE_BIAS_INIT)),
)
with torch.no_grad():
    a_init = gated_init.ray_gate_value(torch.randn(B, HID))
check("ray_gate_value shape broadcasts over (B, N, C)", tuple(a_init.shape) == (B, 1, 1),
      f"got {tuple(a_init.shape)}")
check(
    "init-equivalence holds to within 2%",
    float((1.0 - a_init).abs().max()) <= 0.02,
    f"a_init = {float(a_init.flatten()[0]):.5f}, 1-a = {float((1 - a_init).max()):.5f}",
)
gi = unlock(build(**A4, ray_gate=True))
gi.load_state_dict({**{k: v for k, v in off.state_dict().items()},
                    "ray_gate.weight": gated_init.ray_gate.weight,
                    "ray_gate.bias": gated_init.ray_gate.bias}, strict=True)
host_init, _ = run(gi)
rel = float((host_init.blended - UNGATED).norm() / UNGATED.norm())
check(
    "an init-gated arm is the ungated arm to ~2% at the call site",
    rel <= 0.02 and rel > 0.0,
    f"relative difference {rel:.4f}",
)

print("\n=== 3. FALSIFIER: a -> 1 reproduces the CURRENT arm bit-for-bit ===")
one = force_gate(unlock(build(**A4, ray_gate=True)), logit=+1000.0)
with torch.no_grad():
    check("sigmoid saturates to exactly 1.0",
          float(one.ray_gate_value(torch.randn(B, HID)).min()) == 1.0)
host_one, _ = run(one)
check(
    "a=1 blend == feat + ray_token, bit for bit",
    torch.equal(host_one.blended, UNGATED),
    f"max|d| {(host_one.blended - UNGATED).abs().max():.3e}",
)

print("\n=== 4. FALSIFIER: a -> 0 reproduces the UNCONDITIONED arm bit-for-bit ===")
# The reference is not a guess: view 0 gets the pose-free token by
# `full_out[i][:batch_size] += self.masked_ray_map_token` in _encode_views, so
# feat + masked_ray_map_token IS arm A's token construction. Anchor it to the
# source so a refactor there cannot silently invalidate this comparison.
check(
    "view-0 pose-free construction is still `+= self.masked_ray_map_token`",
    len(re.findall(r"full_out\[i\]\[:batch_size\] \+= self\.masked_ray_map_token",
                   MODEL_SRC)) == 1,
)
zero = force_gate(unlock(build(**A4, ray_gate=True)), logit=-1000.0)
with torch.no_grad():
    check("sigmoid saturates to exactly 0.0",
          float(zero.ray_gate_value(torch.randn(B, HID)).max()) == 0.0)
host_zero, _ = run(zero)
check(
    "a=0 blend == feat + masked_ray_map_token, bit for bit",
    torch.equal(host_zero.blended, UNCONDITIONED),
    f"max|d| {(host_zero.blended - UNCONDITIONED).abs().max():.3e}",
)
check(
    "so the a->0 floor is arm A, NOT the conditioned arm",
    not torch.allclose(host_zero.blended, UNGATED, atol=1e-6),
    f"max|d| vs conditioned {(host_zero.blended - UNGATED).abs().max():.4f}",
)

print("\n=== 5. FALSIFIER: an intermediate a CHANGES the output, per sample ===")
half = force_gate(unlock(build(**A4, ray_gate=True)), logit=0.0)
host_half, res_half = run(half)
check("a=0.5 differs from the conditioned arm",
      not torch.allclose(host_half.blended, UNGATED, atol=1e-6),
      f"max|d| {(host_half.blended - UNGATED).abs().max():.4f}")
check("a=0.5 differs from the unconditioned arm",
      not torch.allclose(host_half.blended, UNCONDITIONED, atol=1e-6),
      f"max|d| {(host_half.blended - UNCONDITIONED).abs().max():.4f}")
check("a=0.5 is the exact midpoint of the two arms",
      torch.allclose(host_half.blended, 0.5 * UNGATED + 0.5 * UNCONDITIONED,
                     atol=1e-6))

# Now the version that cannot be faked by a constant: a nonzero gate weight
# makes `a` a function of the hidden the GRU JUST produced, so it differs per
# sample, and the blend must match the per-sample hand computation.
percase = force_gate(unlock(build(**A4, ray_gate=True)), logit=0.3, weight_std=1.5)
host_ps, res_ps = run(percase)
with torch.no_grad():
    # host._pose_gru_hidden is this step's post-update hidden (detached at the
    # view boundary under pose_gru_bptt=False) — the exact input the gate read.
    a_ps = percase.ray_gate_value(host_ps._pose_gru_hidden.float())
check("a varies per sample (not a broadcast constant)",
      float(a_ps.max() - a_ps.min()) > 1e-3,
      f"a in [{float(a_ps.min()):.4f}, {float(a_ps.max()):.4f}]")
expect_ps = FEAT + a_ps * RAY_TOKEN + (1 - a_ps) * MASKED_TOKEN
check("blend matches the per-sample hand computation",
      torch.allclose(host_ps.blended, expect_ps, atol=1e-6),
      f"max|d| {(host_ps.blended - expect_ps).abs().max():.3e}")
check("the gate is driven by the GRU hidden, not by a constant",
      not torch.allclose(host_ps.blended,
                         FEAT + a_ps.mean() * RAY_TOKEN
                         + (1 - a_ps.mean()) * MASKED_TOKEN, atol=1e-6))

print("\n=== 6. the main loss can train the gate (live tape at the ray add) ===")
# pose_gru_e2e=False on purpose: the ray build runs under no_grad there, so the
# POSE path is a constant — but `a` multiplies the ray token outside that
# block, so the reconstruction loss still reaches the gate. If this ever stops
# holding, the gate is decorative on every non-e2e arm.
trainable = force_gate(unlock(build(**A4, ray_gate=True)), logit=0.3, weight_std=1.0)
# Mid-sequence hidden: at view 1 the cell runs from zeros, so d/d(weight_hh) is
# structurally zero there and a check on it would be vacuous rather than
# informative.
torch.manual_seed(31)
HIDDEN_MID = torch.randn(B, HID)
host_g, _ = run(trainable, e2e=False, grad=True, hidden0=HIDDEN_MID)
check("blended tokens carry a graph without pose_gru_e2e",
      host_g.blended.requires_grad)
host_g.blended.sum().backward()
wg = trainable.ray_gate.weight.grad
bg = trainable.ray_gate.bias.grad
check("gradient reaches ray_gate.weight",
      wg is not None and float(wg.abs().max()) > 0,
      f"max|grad| {float(wg.abs().max()):.4e}" if wg is not None else "grad is None")
check("gradient reaches ray_gate.bias",
      bg is not None and float(bg.abs().max()) > 0,
      f"max|grad| {float(bg.abs().max()):.4e}" if bg is not None else "grad is None")
for pname in ("weight_ih", "weight_hh"):
    pg = getattr(trainable.cell, pname).grad
    check(f"and through the gate into the GRU cell ({pname})",
          pg is not None and float(pg.abs().max()) > 0,
          f"max|grad| {float(pg.abs().max()):.4e}" if pg is not None else "grad is None")
# Contrast: the ungated non-e2e arm has no path at all from the ray add back.
host_ng, _ = run(unlock(build(**A4)), e2e=False, grad=True)
check("the UNGATED non-e2e arm has no such path (the contrast)",
      not host_ng.blended.requires_grad)

print("\n=== 6b. the blend does not widen the token dtype (the AMP trap) ===")
# Under training the tokens are half and the GRU island is fp32. If `a` were
# left fp32 at the blend, type promotion would silently hand the decoder fp32
# tokens from view 1 on — no error, just a different (slower, higher-memory)
# model than the ungated arm it is being compared against.
for dt in (torch.float16, torch.bfloat16):
    host_h, _ = run(force_gate(unlock(build(**A4, ray_gate=True)), logit=0.3,
                               weight_std=1.0), dtype=dt)
    host_u, _ = run(unlock(build(**A4)), dtype=dt)
    check(f"gated blend stays {dt}", host_h.blended.dtype == dt,
          f"got {host_h.blended.dtype}")
    check(f"ungated arm stays {dt} too (the control)", host_u.blended.dtype == dt)
    check(f"and a=1 still reproduces the ungated arm in {dt}",
          torch.equal(run(force_gate(unlock(build(**A4, ray_gate=True)),
                                     logit=+1000.0), dtype=dt)[0].blended,
                      host_u.blended))

print("\n=== 7. telemetry: per-view mean of a, drained once ===")
host_t, res_t = run(force_gate(unlock(build(**A4, ray_gate=True)),
                               logit=0.3, weight_std=1.5))
with torch.no_grad():
    a_t = host_t.pose_gru.ray_gate_value(host_t._pose_gru_hidden.float())
stats = host_t.pop_ray_gate_stats()
check("per-view key is present for the view that ran",
      set(stats) == {"gru_ray_gate", "gru_ray_gate_v1"}, f"got {sorted(stats)}")
check("per-view mean matches the emitted a",
      abs(stats["gru_ray_gate_v1"] - float(a_t.mean())) < 1e-6,
      f"logged {stats['gru_ray_gate_v1']:.6f} vs {float(a_t.mean()):.6f}")
check("headline mean matches too",
      abs(stats["gru_ray_gate"] - float(a_t.mean())) < 1e-6)
check("draining resets the accumulator", host_t.pop_ray_gate_stats() == {})
check("the raw per-sample value is stashed on the data path",
      "gru_ray_gate" in res_t[-1]
      and tuple(res_t[-1]["gru_ray_gate"].shape) == (B, 1)
      and torch.allclose(res_t[-1]["gru_ray_gate"], a_t.reshape(B, 1), atol=1e-6))
check("the stash is detached (it is telemetry, not a second gradient path)",
      not res_t[-1]["gru_ray_gate"].requires_grad)

print("\n=== 7b. MULTI-VIEW: a is recomputed per view and keyed per view ===")
# A single decode step cannot falsify the per-view half of R4: an accumulator
# that OVERWRITES instead of accumulating, or one that hardcodes the key, is
# indistinguishable from the real thing when only view 1 ever runs. Step three
# consecutive views through ONE host, exactly as the sequential driver does.
seq_gru = force_gate(unlock(build(**A4, ray_gate=True)), logit=0.3, weight_std=1.5)
seq_host = RayAddHarness(seq_gru)
seq_a, seq_blend = {}, {}
with torch.no_grad():
    for vi in (1, 2, 3):
        seq_host._forward_decoder_group_step(
            views=VIEWS, view_indices=[vi], feat_group=[FEAT], pos_group=[POS],
            shape_group=[SHAPE], init_state_feat=STATE, init_mem=MEM,
            state_feat=STATE, state_pos=STATE_POS, mem=MEM,
        )
        # The post-step stash IS the hidden the gate read at this view.
        seq_a[vi] = seq_gru.ray_gate_value(seq_host._pose_gru_hidden.float())
        seq_blend[vi] = seq_host.blended
check("a is recomputed every view (not cached from view 1)",
      all(float((seq_a[v] - seq_a[1]).abs().max()) > 1e-3 for v in (2, 3)),
      "max|a(v2)-a(v1)| "
      f"{float((seq_a[2] - seq_a[1]).abs().max()):.4f}, "
      f"max|a(v3)-a(v2)| {float((seq_a[3] - seq_a[2]).abs().max()):.4f}")
for vi in (1, 2, 3):
    exp = FEAT + seq_a[vi] * RAY_TOKEN + (1 - seq_a[vi]) * MASKED_TOKEN
    check(f"view {vi} blend uses view {vi}'s own a",
          torch.allclose(seq_blend[vi], exp, atol=1e-6),
          f"max|d| {(seq_blend[vi] - exp).abs().max():.3e}")
seq_stats = seq_host.pop_ray_gate_stats()
check("one key per GLOBAL view index, no collisions",
      set(seq_stats) == {"gru_ray_gate", "gru_ray_gate_v1", "gru_ray_gate_v2",
                         "gru_ray_gate_v3"},
      f"got {sorted(seq_stats)}")
check("each per-view mean is that view's own a (keys are not shuffled)",
      all(abs(seq_stats[f"gru_ray_gate_v{v}"] - float(seq_a[v].mean())) < 1e-6
          for v in (1, 2, 3)),
      ", ".join(f"v{v}={seq_stats[f'gru_ray_gate_v{v}']:.6f}" for v in (1, 2, 3)))
check("headline is the mean over ALL views and samples",
      abs(seq_stats["gru_ray_gate"]
          - float(sum(float(seq_a[v].sum()) for v in (1, 2, 3)) / (3 * B))) < 1e-6,
      f"{seq_stats['gru_ray_gate']:.6f}")
# The accumulator itself: two records for the SAME view must average, not
# overwrite. This is the mutation a single-view battery cannot see.
acc = RayAddHarness(seq_gru)
acc._record_ray_gate(7, torch.full((B, 1, 1), 0.25))
acc._record_ray_gate(7, torch.full((B, 1, 1), 0.75))
acc_stats = acc.pop_ray_gate_stats()
check("repeat records of one view AVERAGE (not overwrite)",
      abs(acc_stats["gru_ray_gate_v7"] - 0.5) < 1e-6,
      f"got {acc_stats.get('gru_ray_gate_v7')}")

print("\n=== 8. checkpoint round trip through the sniff ===")
args = SimpleNamespace(pose_gru_mode="residual", pose_gru_input="pose_delta",
                       pose_gru_hidden_dim=HID, pose_gru_iters=1)
state = {f"pose_gru.{k}": v for k, v in percase.state_dict().items()}
cfg = _sniff_pose_gru_config(state, args, ENC_DIM, where="synthetic")
check("sniff detects the gate from key presence", cfg.get("ray_gate") is True,
      f"got {cfg!r}")
rebuilt = PoseGRU(**cfg, img_feat_src_dim=ENC_DIM)
missing, unexpected = rebuilt.load_state_dict(
    {k[len("pose_gru."):]: v for k, v in state.items()}, strict=True)
check("strict load has no missing/unexpected keys", not missing and not unexpected,
      f"missing={list(missing)} unexpected={list(unexpected)}")
check("gate weights survive the round trip",
      torch.equal(rebuilt.ray_gate.weight, percase.ray_gate.weight)
      and torch.equal(rebuilt.ray_gate.bias, percase.ray_gate.bias))
host_rb, _ = run(rebuilt.eval())
check("the rebuilt module reproduces the original blend",
      torch.equal(host_rb.blended, host_ps.blended))

state_off = {f"pose_gru.{k}": v for k, v in off.state_dict().items()}
cfg_off = _sniff_pose_gru_config(state_off, args, ENC_DIM, where="synthetic")
check("sniff reports off for a pre-P2 checkpoint", cfg_off.get("ray_gate") is False,
      f"got {cfg_off.get('ray_gate')!r}")

print("\n=== 9. mis-sniffed / mislabelled modules hard-fail ===")
try:
    PoseGRU(**A4).load_state_dict(
        {k[len("pose_gru."):]: v for k, v in state.items()}, strict=True)
    check("gated ckpt into an ungated module raises", False, "it loaded silently")
except RuntimeError as e:
    check("gated ckpt into an ungated module raises", "ray_gate" in str(e),
          str(e).splitlines()[0][:70])
try:
    PoseGRU(**A4, ray_gate=True).load_state_dict(
        {k[len("pose_gru."):]: v for k, v in state_off.items()}, strict=True)
    check("ungated ckpt into a gated module raises", False, "it loaded silently")
except RuntimeError as e:
    check("ungated ckpt into a gated module raises", "ray_gate" in str(e),
          str(e).splitlines()[0][:70])
for sd, flag, why in [
    (state, False, "gated weights + args say ray_gate=False"),
    (state_off, True, "ungated weights + args say ray_gate=True"),
]:
    bad_args = SimpleNamespace(**vars(args), pose_gru_ray_gate=flag)
    try:
        _sniff_pose_gru_config(sd, bad_args, ENC_DIM, where="synthetic")
        check(f"HARD-FAIL on {why}", False, "sniff guessed instead of raising")
    except RuntimeError as e:
        check(f"HARD-FAIL on {why}", "mislabelled" in str(e), str(e)[:60])
ok_args = SimpleNamespace(**vars(args), pose_gru_ray_gate=True)
check("matching args + weights sniff cleanly",
      _sniff_pose_gru_config(state, ok_args, ENC_DIM, where="synthetic")["ray_gate"]
      is True)

print("\n=== 10. config plumbing: default off end to end ===")


class GruHost:
    enc_embed_dim = ENC_DIM
    enable_pose_gru = ARCroco3DStereo.enable_pose_gru


h_def = GruHost()
h_def.enable_pose_gru(hidden_dim=HID, mode="residual", input_mode="pose_delta")
check("enable_pose_gru() default builds NO gate", h_def.pose_gru.ray_gate is None)
h_on = GruHost()
h_on.enable_pose_gru(hidden_dim=HID, mode="residual", input_mode="pose_delta",
                     ray_gate=True)
check("enable_pose_gru(ray_gate=True) builds one",
      isinstance(h_on.pose_gru.ray_gate, torch.nn.Linear))
h_sniff = GruHost()
h_sniff.enable_pose_gru(img_encoder_pretrained=False, **cfg)
check("the load_model path rebuilds a gated ckpt gated",
      isinstance(h_sniff.pose_gru.ray_gate, torch.nn.Linear))
check(
    "the trainer reads pose_gru_ray_gate with an absent-key default of False",
    len(re.findall(r'ray_gate=bool\(getattr\(args, "pose_gru_ray_gate", False\)\)',
                   TRAIN_SRC)) == 1,
)
check(
    "the trainer refuses pose_gru_ray_gate without pose_gru",
    'pose_gru_ray_gate=True needs pose_gru=True' in TRAIN_SRC,
)
check(
    "both train and test loops drain the gate telemetry",
    len(re.findall(r"pop_ray_gate_stats\(\)\)", TRAIN_SRC)) == 2,
)

print("\n" + "=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("all checks passed — pose_gru_ray_gate is default-off, on the ray-add "
      "compute path, floors at the pose-free arm, trainable by the main loss, "
      "logged per view, and checkpoint-round-trippable")
sys.exit(0)
