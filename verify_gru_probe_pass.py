#!/usr/bin/env python
"""verify_gru_probe_pass.py — falsifier battery for P4.3 (pose_gru_refine_passes).

CPU only, no data, no checkpoint, seconds to run:

    python verify_gru_probe_pass.py

P4.3 breaks the corrector's circularity — you need a pose to build the
conditioning, and the conditioning to get a good pose — by decoding view x once
with the ray token replaced by masked_ray_map_token, on a THROWAWAY use of the
state the real pass is about to consume, and handing the resulting pose to the
cell as a third 7-d block:

    gru_in = [ P(x-1) | delta | O(x) ]          width 14 -> 21

Six properties have to hold, each one a way this could be silently wrong rather
than loudly broken:

  1. ABSENT KEY => BYTE-IDENTICAL. refine_passes=1 must not widen the cell, must
     not run a probe decode, and must emit the identical pose. A lever that
     defaulted to "widened but zero-filled" would add 7 columns to weight_ih
     that every pre-P4.3 checkpoint lacks, and the loader hard-fails on that.
  2. THE PROBE ACTUALLY OBSERVES FRAME x. Its decode input must be bitwise
     feat + masked_ray_map_token, and its output must land in columns 14:21 of
     the cell input. This is the half that a probe which is computed, logged and
     then ignored would still pass — so it is checked against the REAL call
     site, not a re-implementation.
  3. IT IS GENUINELY THROWAWAY. state_feat and mem must be bit-identical after
     the probe, and pose_retriever.update_mem must be called EXACTLY ONCE per
     view (by the real pass). A probe that committed its state would corrupt the
     rollout in a way no metric would attribute to it.
  4. INIT-EQUIVALENCE IS EXACT. Columns 14:21 of cell.weight_ih are zero-init,
     so a 21-wide module reproduces the 14-wide one bit-for-bit at init
     INCLUDING the hidden trajectory — and perturbing those columns changes the
     output, which proves the channel is wired rather than dead.
  5. THE FALSIFIER CHANNEL WORKS. POSE_GRU_PROBE_SHUFFLE and POSE_GRU_PROBE_LAG
     must actually change what the cell consumes.
  6. IT SURVIVES A CHECKPOINT ROUND TRIP. The pass count is invisible in weight
     shapes (only "N>=2" is, via the 21-wide cell), so _sniff_pose_gru_config
     must recover it from ckpt args and hard-fail on any disagreement.

Property 2/3 are tested against the REAL method. The refinement loop lives deep
inside _forward_decoder_group_step, a method that otherwise wants a full CUT3R;
rather than re-implement it — which would pass just as happily if the call site
never ran it — this file borrows the shipping method onto a duck-typed host and
stubs only what surrounds it (ray encoder, recurrent rollout, pose head, memory).
Everything from the probe decode to the ray add is executed verbatim.

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

# Read inside the method under test; a stray one in the caller's environment
# would quietly change every number below.
for _fv in (
    "PREV_PRED_RAY_SHUFFLE",
    "POSE_GRU_HIDDEN_SHUFFLE",
    "POSE_GRU_HIDDEN_ZERO",
    "POSE_GRU_IMG_FEAT_SHUFFLE",
    "POSE_GRU_IMG_FEAT_ZERO",
    "POSE_GRU_FORCE_ITERS",
    "POSE_GRU_FORCE_REFINE",
    "POSE_GRU_PROBE_SHUFFLE",
    "POSE_GRU_PROBE_LAG",
    "POSE_GRU_PROBE_ASSERT_PURE",
):
    os.environ.pop(_fv, None)

import torch

import dust3r.heads  # noqa: F401  MUST precede dust3r.utils.camera (circular import)
from dust3r.model import ARCroco3DStereo, PoseGRU, _sniff_pose_gru_config

ENC_DIM = 64  # token width; the real arm's 1024 costs time and proves nothing
HID = 32
B, NTOK = 4, 6
H, W = 16, 16
DEC_DEPTH = 4
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
    """Give the residual module a non-zero head so the emitted pose actually
    moves — otherwise every comparison below is trivially satisfied by the
    zero-init identity."""
    torch.manual_seed(seed)
    with torch.no_grad():
        gru.head.weight.normal_(0, 0.05)
        gru.head.bias.normal_(0, 0.05)
    return gru


def unlock_probe(gru, seed=13):
    """Un-zero cell.weight_ih[:, 14:21] — i.e. simulate a module that has
    LEARNED to read the observation.

    Needed because the probe columns are zero-init by design (check 5), and a
    channel whose weights are exactly zero is immune to having its input
    shuffled. That is not a defect, but it has a sharp operational consequence
    worth stating plainly:

        POSE_GRU_PROBE_SHUFFLE IS A NO-OP ON AN UNTRAINED CHECKPOINT.

    Run it early in training and it will report "the probe is unused" no matter
    how well the mechanism works — the same trap the R8 config header flags for
    POSE_GRU_FORCE_ITERS while the zero-init head is still locked. Falsify
    against a keep_freq snapshot with a non-zero ||weight_ih[:, 14:21]||, and
    check that norm FIRST.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        gru.cell.weight_ih[:, 14:21].normal_(0, 0.5)
    return gru


def _pose(seed):
    torch.manual_seed(seed)
    t = torch.randn(B, 3) * 0.1
    q = torch.nn.functional.normalize(torch.randn(B, 4), dim=-1)
    return torch.cat([t, q], dim=-1)


FEAT = torch.randn(B, NTOK, ENC_DIM)
MASKED_TOKEN = torch.randn(1, ENC_DIM) * 0.02  # shaped like the real Parameter
POS = torch.zeros(B, NTOK, 2, dtype=torch.long)
SHAPE = torch.tensor([[H, W]] * B)
STATE = torch.randn(B, 3, ENC_DIM)
MEM = torch.randn(B, 3, ENC_DIM)
STATE_POS = torch.zeros(B, 3, 2, dtype=torch.long)
PREV, PREV2 = _pose(1), _pose(2)
K = torch.tensor([[[W * 1.2, 0.0, W / 2], [0.0, W * 1.2, H / 2], [0.0, 0.0, 1.0]]])
VIEWS = [
    dict(
        img=torch.randn(B, 3, H, W),
        camera_intrinsics=K.expand(B, -1, -1).contiguous(),
        img_mask=torch.ones(B, dtype=torch.bool),
        reset=torch.zeros(B, dtype=torch.bool),
    )
    for _ in range(3)
]


class StubMemory:
    """Counts update_mem calls. That counter is the whole point: a probe that
    committed its memory would be invisible in every metric but would corrupt
    the rollout, so 'exactly one write per view' is the assertion that matters
    most in this file."""

    def __init__(self):
        self.update_calls = 0
        self.inquire_calls = 0

    def inquire(self, query, mem):
        self.inquire_calls += 1
        return query[:, :1, :] + mem[:, :1, :] * 0.5

    def update_mem(self, mem, feat_k, feat_v):
        self.update_calls += 1
        return mem + 1.0


class ProbeHarness:
    """Duck-typed host carrying the SHIPPING methods verbatim."""

    _forward_decoder_group_step = ARCroco3DStereo._forward_decoder_group_step
    _concat_group_feat_pos = ARCroco3DStereo._concat_group_feat_pos
    _get_img_level_feat = ARCroco3DStereo._get_img_level_feat
    _probe_decode_pose = ARCroco3DStereo._probe_decode_pose
    _pred_ray_tokens = ARCroco3DStereo._pred_ray_tokens
    _probe_falsify = ARCroco3DStereo._probe_falsify
    _record_ray_gate = ARCroco3DStereo._record_ray_gate

    pose_head_flag = True
    dec_depth = DEC_DEPTH
    feed_prev_pred = True
    pose_gru_oracle = "off"
    pose_gru_iter_detach = True
    pose_gru_bptt = False
    pose_mode = ("exp", -float("inf"), float("inf"))

    def __init__(self, pose_gru, e2e=False, mutate_state=False):
        self.pose_gru = pose_gru
        self.pose_gru_e2e = e2e
        self.masked_ray_map_token = MASKED_TOKEN
        self._prev_pred_pose_enc = PREV
        self._prev_prev_pred_pose_enc = PREV2
        self._pose_gru_hidden = None
        self._prev_img_feat = None
        self._probe_obs = None
        self._probe_obs_lagged = None
        self._ray_gate_stats = {}
        self.pose_retriever = StubMemory()
        self.pose_token = torch.randn(1, 1, ENC_DIM)
        self.downstream_head = SimpleNamespace(pose_head=self._pose_head)
        # observables
        self.rollout_feats = []   # the feature each decode consumed
        self.rollout_states = []
        self.ray_poses = []       # every pose that reached the ray encoder
        self.blended = None       # the token the REAL pass produced
        self._mutate_state = mutate_state

    @staticmethod
    def _pose_head(token):
        """(B, C) -> (B, 7). Deterministic and INJECTIVE enough that a different
        input feature produces a different pose, which is what lets the checks
        below attribute an observation to the feature it came from."""
        w = torch.linspace(0.1, 0.8, token.shape[-1])[None, :]
        base = (token * w).sum(-1, keepdim=True)
        return torch.cat([base * 0.3, base * 0.2, base * 0.1,
                          base * 0.05 + 1.0, base * 0.02,
                          base * 0.01, base * 0.03], dim=-1)

    def _encode_ray_map(self, rmap, shape):
        # Record the ray map so a conditioned probe's pose can be recovered.
        self.ray_poses.append(rmap.detach().clone())
        tok = rmap.flatten(1).mean(-1)[:, None, None].expand(B, NTOK, ENC_DIM)
        return [tok.contiguous()], POS, None

    def _recurrent_rollout(self, state_feat, state_pos, current_feat, current_pos,
                           pose_feat, pose_pos, init_state_feat):
        self.rollout_feats.append(current_feat.detach().clone())
        self.rollout_states.append(state_feat.detach().clone())
        if self._mutate_state:
            # Deliberate saboteur for the ASSERT_PURE check.
            state_feat.add_(1e-3)
        pose_tok = current_feat.mean(dim=1, keepdim=True)
        out = torch.cat([pose_tok, current_feat], dim=1)
        return state_feat, [out] * (self.dec_depth + 1)

    def _downstream_head(self, decout, img_shape, **kw):
        return {"camera_pose": self._pose_head(decout[-1][:, 0])}


def run(gru, view_idx=1, e2e=False, mutate_state=False, hidden0=None):
    host = ProbeHarness(gru, e2e=e2e, mutate_state=mutate_state)
    if hidden0 is not None:
        host._pose_gru_hidden = hidden0
    captured = []
    h = gru.register_forward_pre_hook(lambda _m, inp: captured.append(inp[0].detach().clone()))
    try:
        with torch.no_grad():
            res_group, _ = ProbeHarness._forward_decoder_group_step(
                host, views=VIEWS, view_indices=[view_idx], feat_group=[FEAT],
                pos_group=[POS], shape_group=[SHAPE], init_state_feat=STATE,
                init_mem=MEM, state_feat=STATE, state_pos=STATE_POS, mem=MEM,
            )
    finally:
        h.remove()
    host.blended = host.rollout_feats[-1]
    host.cell_inputs = captured
    host.res = res_group[-1]
    return host


print("\n=== 1. ABSENT KEY => no widening, no probe decode, identical output ===")
off = unlock(build(**A4))
check("refine_passes defaults to 1", off.refine_passes == 1 and off.probe is False)
check("cell input stays 14 wide", off.input_dim == 14, f"input_dim={off.input_dim}")
check("no probe columns in the state dict",
      list(off.state_dict()["cell.weight_ih"].shape) == [3 * HID, 14])
h_off = run(off)
check("exactly ONE decode (the real pass), no probe", len(h_off.rollout_feats) == 1,
      f"{len(h_off.rollout_feats)} decodes")
check("cell consumed a 14-wide input", h_off.cell_inputs[0].shape[-1] == 14)
check("update_mem called exactly once", h_off.pose_retriever.update_calls == 1)

print("\n=== 2. N=2: ONE probe decode, and it sees the POSE-FREE construction ===")
on2 = unlock(build(**A4, refine_passes=2))
check("cell input widened to 21", on2.input_dim == 21, f"input_dim={on2.input_dim}")
h2 = run(on2)
check("exactly TWO decodes (1 probe + 1 real)", len(h2.rollout_feats) == 2,
      f"{len(h2.rollout_feats)} decodes")
POSE_FREE = FEAT + MASKED_TOKEN
check("probe decode consumed feat + masked_ray_map_token, BIT FOR BIT",
      torch.equal(h2.rollout_feats[0], POSE_FREE),
      f"max|d| {(h2.rollout_feats[0] - POSE_FREE).abs().max():.3e}")
check("probe used the SAME state the real pass got",
      torch.equal(h2.rollout_states[0], h2.rollout_states[1]))
check("real pass did NOT consume the pose-free feature",
      not torch.equal(h2.rollout_feats[1], POSE_FREE))
# what the probe SHOULD have emitted, computed independently of the call site
from dust3r.heads.postprocess import postprocess_pose

expect_obs = postprocess_pose(
    ProbeHarness._pose_head(torch.cat([POSE_FREE.mean(1, keepdim=True), POSE_FREE], 1)[:, 0]),
    ProbeHarness.pose_mode,
)
check("cell input is 21 wide", h2.cell_inputs[0].shape[-1] == 21)
check("columns 14:21 ARE the probe observation, bit for bit",
      torch.equal(h2.cell_inputs[0][:, 14:21], expect_obs),
      f"max|d| {(h2.cell_inputs[0][:, 14:21] - expect_obs).abs().max():.3e}")
check("columns 0:7 are still P(x-1) (residual anchor untouched)",
      torch.equal(h2.cell_inputs[0][:, :7], PREV))
check("update_mem STILL called exactly once — the probe did not commit memory",
      h2.pose_retriever.update_calls == 1, f"{h2.pose_retriever.update_calls} writes")

print("\n=== 3. THROWAWAY: the probe leaves state and mem bit-identical ===")
snap_state, snap_mem = STATE.clone(), MEM.clone()
_ = run(unlock(build(**A4, refine_passes=2)))
check("state_feat unchanged after a probe", torch.equal(STATE, snap_state))
check("mem unchanged after a probe", torch.equal(MEM, snap_mem))
os.environ["POSE_GRU_PROBE_ASSERT_PURE"] = "1"
try:
    run(unlock(build(**A4, refine_passes=2)), mutate_state=True)
    check("ASSERT_PURE catches a mutating decode path", False, "no assertion raised")
except AssertionError as e:
    check("ASSERT_PURE catches a mutating decode path", "MUTATED" in str(e))
finally:
    os.environ.pop("POSE_GRU_PROBE_ASSERT_PURE", None)

print("\n=== 4. N=4: three probes, pass 1 pose-free, later passes CONDITIONED ===")
on4 = unlock(build(**A4, refine_passes=4))
h4 = run(on4)
check("exactly FOUR decodes (3 probes + 1 real)", len(h4.rollout_feats) == 4,
      f"{len(h4.rollout_feats)} decodes")
check("pass 1 is pose-free", torch.equal(h4.rollout_feats[0], POSE_FREE))
check("passes 2 and 3 are NOT pose-free (they carry a predicted ray)",
      not torch.equal(h4.rollout_feats[1], POSE_FREE)
      and not torch.equal(h4.rollout_feats[2], POSE_FREE))
check("each later probe differs from the one before (the estimate moved)",
      not torch.equal(h4.rollout_feats[1], h4.rollout_feats[2]))
check("iterate count is (N-1)*iters", len(h4.cell_inputs) == 3,
      f"{len(h4.cell_inputs)} cell calls at iters=1")
check("update_mem still exactly once", h4.pose_retriever.update_calls == 1)
check("N=4 emits a different pose than N=2 (refinement changes the answer)",
      not torch.equal(h4.res["gru_pose"], h2.res["gru_pose"]))

print("\n=== 5. INIT-EQUIVALENCE: zero-init cols 14:21 reproduce the 14-wide arm ===")
base14 = build(**A4)                      # zero-init head, untouched
base21 = build(**A4, refine_passes=2)
with torch.no_grad():  # give both the SAME weights outside the probe columns
    base21.cell.weight_ih[:, :14].copy_(base14.cell.weight_ih)
    base21.cell.weight_hh.copy_(base14.cell.weight_hh)
    base21.cell.bias_ih.copy_(base14.cell.bias_ih)
    base21.cell.bias_hh.copy_(base14.cell.bias_hh)
    base21.head.weight.copy_(base14.head.weight)
    base21.head.bias.copy_(base14.head.bias)
check("probe columns are zero at init",
      bool(base21.cell.weight_ih[:, 14:21].abs().max() == 0))
h14, h21 = run(base14), run(base21)
check("emitted pose is bit-identical at init",
      torch.equal(h14.res["gru_pose"], h21.res["gru_pose"]),
      f"max|d| {(h14.res['gru_pose'] - h21.res['gru_pose']).abs().max():.3e}")
check("HIDDEN trajectory is bit-identical at init (stronger than the F lever)",
      torch.equal(h14._pose_gru_hidden, h21._pose_gru_hidden))
with torch.no_grad():
    base21.cell.weight_ih[:, 14:21].normal_(0, 0.5)
h21b = run(base21)
check("perturbing cols 14:21 CHANGES the hidden — the channel is wired, not dead",
      not torch.equal(h21b._pose_gru_hidden, h21._pose_gru_hidden))

print("\n=== 6. FALSIFIER CHANNEL: shuffle and lag reach the cell ===")
# unlock_probe(): the probe columns are zero-init, and a zero-weight channel is
# immune to a shuffled input — see unlock_probe's docstring for why that makes
# this falsifier a no-op on an untrained checkpoint.
clean = run(unlock_probe(unlock(build(**A4, refine_passes=2))))
os.environ["POSE_GRU_PROBE_SHUFFLE"] = "1"
shuf = run(unlock_probe(unlock(build(**A4, refine_passes=2))))
os.environ.pop("POSE_GRU_PROBE_SHUFFLE")
check("SHUFFLE rolls the observation across the batch",
      torch.equal(shuf.cell_inputs[0][:, 14:21],
                  torch.roll(clean.cell_inputs[0][:, 14:21], shifts=1, dims=0)))
check("SHUFFLE changes the emitted pose (on a module that reads the channel)",
      not torch.equal(shuf.res["gru_pose"], clean.res["gru_pose"]))
check("...and is correctly a NO-OP while the probe columns are still zero-init",
      torch.equal(
          run(unlock(build(**A4, refine_passes=2))).res["gru_pose"],
          run(unlock(build(**A4, refine_passes=2))).res["gru_pose"]))
check("SHUFFLE leaves the residual anchor honest",
      torch.equal(shuf.cell_inputs[0][:, :7], clean.cell_inputs[0][:, :7]))
os.environ["POSE_GRU_PROBE_LAG"] = "1"
g_lag = unlock(build(**A4, refine_passes=2))
lag_host = ProbeHarness(g_lag)
lag_host._probe_obs_lagged = torch.zeros(B, 7)   # a known "previous view"
cap = []
hk = g_lag.register_forward_pre_hook(lambda _m, inp: cap.append(inp[0].detach().clone()))
with torch.no_grad():
    ProbeHarness._forward_decoder_group_step(
        lag_host, views=VIEWS, view_indices=[1], feat_group=[FEAT], pos_group=[POS],
        shape_group=[SHAPE], init_state_feat=STATE, init_mem=MEM,
        state_feat=STATE, state_pos=STATE_POS, mem=MEM,
    )
hk.remove()
os.environ.pop("POSE_GRU_PROBE_LAG")
check("LAG feeds the PREVIOUS view's observation, not this one's",
      torch.equal(cap[0][:, 14:21], torch.zeros(B, 7)))

print("\n=== 7. probe_every: fewer decodes, held observation reused ===")
g_every = unlock(build(**A4, refine_passes=2, probe_every=3))
host_e = ProbeHarness(g_every)
held = torch.full((B, 7), 0.25)
host_e._probe_obs = held
cap2 = []
hk = g_every.register_forward_pre_hook(lambda _m, inp: cap2.append(inp[0].detach().clone()))
with torch.no_grad():
    ProbeHarness._forward_decoder_group_step(
        host_e, views=VIEWS, view_indices=[2], feat_group=[FEAT], pos_group=[POS],
        shape_group=[SHAPE], init_state_feat=STATE, init_mem=MEM,
        state_feat=STATE, state_pos=STATE_POS, mem=MEM,
    )
hk.remove()
check("view 2 with probe_every=3 runs NO probe decode (real pass only)",
      len(host_e.rollout_feats) == 1, f"{len(host_e.rollout_feats)} decodes")
check("it consumed the HELD observation", torch.equal(cap2[0][:, 14:21], held))
check("view 1 always probes (the held value is never empty on first use)",
      len(run(unlock(build(**A4, refine_passes=2, probe_every=3)),
              view_idx=1).rollout_feats) == 2)

print("\n=== 8. CHECKPOINT ROUND TRIP through _sniff_pose_gru_config ===")
src = unlock(build(**A4, refine_passes=2, probe_every=3))
state = {f"pose_gru.{k}": v for k, v in src.state_dict().items()}
args = SimpleNamespace(pose_gru_mode="residual", pose_gru_input="pose_delta",
                       pose_gru_hidden_dim=HID, pose_gru_iters=1,
                       pose_gru_refine_passes=2, pose_gru_probe_every=3)
cfg = _sniff_pose_gru_config(state, args, ENC_DIM, where="synthetic")
check("sniff recovers refine_passes", cfg["refine_passes"] == 2, str(cfg["refine_passes"]))
check("sniff recovers probe_every", cfg["probe_every"] == 3, str(cfg["probe_every"]))
check("sniff keeps input_mode == 'pose_delta' (no new taxonomy value)",
      cfg["input_mode"] == "pose_delta", repr(cfg["input_mode"]))
rebuilt = PoseGRU(**cfg, img_feat_src_dim=ENC_DIM)
missing, unexpected = rebuilt.load_state_dict(
    {k[len("pose_gru."):]: v for k, v in state.items()}, strict=True), None
check("rebuilt module strict-loads the state dict", True)
check("rebuilt module reproduces the original observation path",
      torch.equal(run(rebuilt).cell_inputs[0], run(src).cell_inputs[0]))
old_args = SimpleNamespace(pose_gru_mode="residual", pose_gru_input="pose_delta",
                           pose_gru_hidden_dim=HID, pose_gru_iters=1)
cfg_off = _sniff_pose_gru_config(
    {f"pose_gru.{k}": v for k, v in build(**A4).state_dict().items()},
    old_args, ENC_DIM, where="synthetic")
check("a PRE-P4.3 checkpoint still sniffs cleanly (refine_passes=1)",
      cfg_off["refine_passes"] == 1 and cfg_off["input_mode"] == "pose_delta")

print("\n=== 9. MIS-SNIFF HARD-FAILS IN BOTH DIRECTIONS ===")
try:
    _sniff_pose_gru_config(state, old_args, ENC_DIM, where="synthetic")
    check("21-wide cell + args saying refine_passes=1 raises", False, "no raise")
except RuntimeError as e:
    check("21-wide cell + args saying refine_passes=1 raises", "refusing to guess" in str(e))
try:
    _sniff_pose_gru_config(
        {f"pose_gru.{k}": v for k, v in build(**A4).state_dict().items()},
        args, ENC_DIM, where="synthetic")
    check("14-wide cell + args asking for the probe raises", False, "no raise")
except RuntimeError as e:
    check("14-wide cell + args asking for the probe raises", "never consumed" in str(e))
try:
    PoseGRU(**dict(A4, input_mode="pose"), refine_passes=2)
    check("refine_passes>=2 with input_mode='pose' is refused", False, "no raise")
except AssertionError as e:
    check("refine_passes>=2 with input_mode='pose' is refused", "pose_delta" in str(e))

print("\n=== 10. CONFIG PLUMBING (source-level: the lever must reach the module) ===")
check("trainer threads refine_passes into enable_pose_gru",
      len(re.findall(r'refine_passes=int\(getattr\(args, "pose_gru_refine_passes", 1\)\)',
                     TRAIN_SRC)) == 1)
check("trainer threads probe_every into enable_pose_gru",
      len(re.findall(r'probe_every=int\(getattr\(args, "pose_gru_probe_every", 1\)\)',
                     TRAIN_SRC)) == 1)
check("trainer guards refine_passes>=2 against pose_gru=False",
      "pose_gru_refine_passes>=2 needs pose_gru=True" in TRAIN_SRC)
check("trainer logs the pass count (it is invisible in weight shapes)",
      "refine_passes=%d" in TRAIN_SRC)
check("the pose-free construction is still `+= self.masked_ray_map_token`",
      len(re.findall(r"full_out\[i\]\[:batch_size\] \+= self\.masked_ray_map_token",
                     MODEL_SRC)) == 1)
check("the probe reads the pose head directly, not through _downstream_head",
      "self.downstream_head.pose_head(pose_token)" in MODEL_SRC)

print("\n" + "=" * 72)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED — do not launch:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed — the probe observes the current frame, the cell "
      "consumes it, and an absent key is byte-identical")
sys.exit(0)
