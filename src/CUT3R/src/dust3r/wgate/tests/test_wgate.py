"""Tests for dust3r.wgate (plain asserts; CPU only; run as a script).

    cd src/CUT3R/src && OMP_NUM_THREADS=1 timeout 280 python dust3r/wgate/tests/test_wgate.py

Covers, per WRITE_GATE_VARIANTS.md "Implementation contract":
  (1) the pure functions: frame_features layout / shapes / float32 / detached / bit-identical across
      calls and under bf16 autocast; sigma_to_weight, combine_pose_sigma, pose_sigma_to_weight,
      gaussian_nll against hand-computed values;
  (2) the head modules on CPU with fixed seeds: FrameConfHead (shape, near-zero init, fp32 under
      autocast, state-dict round trip) and PoseSigmaHead on a real PoseDecoder (penultimate ==
      act(fc1(x)) of the decoder, out layer only in state_dict / parameters / gradients, both
      state-dict key formats);
  (3) a source grep of dust3r/model.py: the state commit line and the update_mem line each occur
      exactly once, the mem-gate wrapper is the line right after update_mem, the state-only frame-gate
      multiply (state_mask) sits once between the token gate and the commit, and the joint-mode multiply
      (update_mask, "frame_gate_joint") sits once between the update_alpha block and the token gate;
  (4) the model hook methods exercised on a stub (no checkpoint): identity without keys, a_0 = b_0 = 1,
      wmin clipping, frame_gate_const, mem_gate_const / mem_gate_freeze_after precedence, unconditional
      trace reset at t == 0, trace format, record dicts, attach_* wmin / log_ref precedence.
The end-to-end bit-identity on the real checkpoint is parity_cpu.py (same directory).
"""

import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))  # .../CUT3R/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dust3r.heads  # noqa: F401,E402  (resolves the camera<->heads circular import)
from dust3r.utils.camera import PoseDecoder  # noqa: E402
from dust3r.wgate.heads import (  # noqa: E402
    FRAME_FEAT_DIM,
    FrameConfHead,
    PoseSigmaHead,
    combine_pose_sigma,
    frame_features,
    gaussian_nll,
    pose_sigma_to_weight,
    sigma_to_weight,
)

torch.manual_seed(0)
B, N, C, G = 2, 768, 768, 1024
MODEL_PY = os.path.join(_SRC, "dust3r", "model.py")


def _autocast_cpu_bf16():
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True)


# ----------------------------------------------------------------------------- (1) pure functions
def test_frame_features_layout():
    torch.manual_seed(1)
    new = torch.randn(B, N, C, requires_grad=True)
    old = torch.randn(B, N, C)
    g = torch.randn(B, 1, G)
    x = frame_features(new, old, g)
    assert x.shape == (B, FRAME_FEAT_DIM) == (B, 1796), x.shape
    assert x.dtype == torch.float32 and not x.requires_grad
    delta = (new - old).detach()
    tn = delta.norm(dim=-1)
    assert torch.equal(x[:, :768], delta.mean(1))
    assert torch.equal(x[:, 768:769], tn.mean(1, keepdim=True))
    assert torch.equal(x[:, 769], tn.mean(1)) and torch.equal(x[:, 770], tn.std(1, unbiased=False))
    assert torch.equal(x[:, 771], tn.max(1).values)
    assert torch.equal(x[:, 772:], g[:, 0])
    # (B,1024) and (B,1,1024) global features are the same thing
    assert torch.equal(frame_features(new, old, g[:, 0]), x)
    # bit-identical across calls and under bf16 autocast (the function disables autocast itself)
    assert torch.equal(frame_features(new, old, g), x)
    with _autocast_cpu_bf16():
        xa = frame_features(new, old, g)
    assert xa.dtype == torch.float32 and torch.equal(xa, x)
    # bf16 inputs (a bf16-autocast rollout) are upcast, not rejected
    xb = frame_features(new.bfloat16(), old.bfloat16(), g.bfloat16())
    assert xb.shape == x.shape and xb.dtype == torch.float32


def test_sigma_to_weight():
    lv = torch.tensor([0.0, 2.0, -2.0, 50.0, -50.0])
    w = sigma_to_weight(lv, 0.0, 0.5)
    assert w.shape == (5,) and w.dtype == torch.float32
    assert w[0].item() == 1.0  # at the reference sigma the weight is exactly 1
    assert abs(w[1].item() - max(math.exp(-1.0), 0.5)) < 1e-6  # sigma = e^1 > ref -> exp(-1) < .5 -> clipped
    assert w[1].item() == 0.5
    assert w[2].item() == 1.0  # more confident than the reference -> clipped to 1
    assert w[3].item() == 0.5 and w[4].item() == 1.0  # extreme values are finite and clipped
    lv2 = torch.tensor([0.5])
    assert abs(sigma_to_weight(lv2, 0.0, 0.0).item() - math.exp(-0.25)) < 1e-6
    # (B,1) input -> (B,), monotone in log_var
    lv3 = torch.linspace(-3, 3, 7)[:, None]
    w3 = sigma_to_weight(lv3, 0.0, 0.1)
    assert w3.shape == (7,) and bool((w3[1:] <= w3[:-1]).all())
    # tensor log_ref is accepted
    assert torch.equal(sigma_to_weight(lv, torch.tensor(0.0), 0.5), w)


def test_combine_pose_sigma():
    refs = (0.0, 0.0)
    s = torch.tensor([[0.0, 0.0], [2 * math.log(4.0), 0.0], [2 * math.log(4.0), 2 * math.log(9.0)]])
    sig = combine_pose_sigma(s, refs)
    assert sig.shape == (3,) and sig.dtype == torch.float32
    assert abs(sig[0].item() - 1.0) < 1e-6
    assert abs(sig[1].item() - 2.0) < 1e-5  # sqrt(4 * 1)
    assert abs(sig[2].item() - 6.0) < 1e-4  # sqrt(4 * 9)
    # references shift the combination: with ref_t = 2 log 4 row 1 is back at 1
    sig2 = combine_pose_sigma(s, torch.tensor([2 * math.log(4.0), 0.0]))
    assert abs(sig2[1].item() - 1.0) < 1e-6
    b = pose_sigma_to_weight(s, refs, 0.5)
    assert b.shape == (3,)
    assert b[0].item() == 1.0 and b[1].item() == 0.5 and b[2].item() == 0.5
    assert abs(pose_sigma_to_weight(s, refs, 0.1)[1].item() - 0.5) < 1e-6
    assert abs(pose_sigma_to_weight(s, refs, 0.1)[2].item() - 1.0 / 6.0) < 1e-5
    # 1-d (single sample) input is accepted
    assert combine_pose_sigma(s[1], refs).shape == (1,)


def test_gaussian_nll():
    r = torch.tensor([0.3, 2.0])
    s_star = torch.log(r * r)
    l0 = gaussian_nll(s_star, r)
    assert torch.allclose(l0, s_star + 1.0)
    assert bool((gaussian_nll(s_star + 1.0, r) > l0).all()) and bool((gaussian_nll(s_star - 1.0, r) > l0).all())
    assert gaussian_nll(torch.zeros(4, 1), torch.ones(4, 1)).shape == (4, 1)


# ----------------------------------------------------------------------------- (2) heads
def test_frame_conf_head():
    torch.manual_seed(2)
    head = FrameConfHead()
    assert head.norm.normalized_shape == (1796,) and head.fc1.out_features == 256 and head.out.out_features == 1
    x = torch.randn(B, 1796)
    s = head(x)
    assert s.shape == (B,) and s.dtype == torch.float32 and torch.isfinite(s).all()
    assert s.abs().max().item() < 0.1, s  # near-zero init: sigma ~ 1
    with _autocast_cpu_bf16():
        sa = head(x)
    assert sa.dtype == torch.float32 and torch.equal(sa, s)
    assert set(head.state_dict().keys()) == {"norm.weight", "norm.bias", "fc1.weight", "fc1.bias", "out.weight", "out.bias"}
    head2 = FrameConfHead.from_state_dict(head.state_dict())
    assert torch.equal(head2(x), s)
    # a head trained at other sizes is rebuilt from its state dict
    small = FrameConfHead(in_dim=10, hidden=4)
    assert FrameConfHead.from_state_dict(small.state_dict()).fc1.out_features == 4
    # gradient reaches every layer (the trainer optimises the whole head)
    head(x).sum().backward()
    assert all(p.grad is not None for p in head.parameters())


def test_pose_sigma_head():
    torch.manual_seed(3)
    dec = PoseDecoder(hidden_size=768)
    for p in dec.parameters():
        p.requires_grad_(False)
    head = PoseSigmaHead(dec)
    assert head.fc1 is dec.mlp.fc1 and head.act is dec.mlp.act
    assert head.out.in_features == 3072 and head.out.out_features == 2
    x = torch.randn(B, 768)
    h = head.penultimate(x)
    assert h.shape == (B, 3072) and h.dtype == torch.float32 and not h.requires_grad
    assert torch.allclose(h, dec.mlp.act(dec.mlp.fc1(x)), atol=1e-6)
    s2 = head(x)
    assert s2.shape == (B, 2) and s2.dtype == torch.float32 and s2.abs().max().item() < 0.1
    with _autocast_cpu_bf16():
        sa = head(x)
    assert sa.dtype == torch.float32 and torch.equal(sa, s2)
    assert torch.equal(head(x.bfloat16()).shape and head(x.bfloat16()), head(x.bfloat16().float()))
    # out layer only: state dict, parameters, gradients
    assert set(head.state_dict().keys()) == {"out.weight", "out.bias"}, head.state_dict().keys()
    assert len(list(head.parameters())) == 2
    dec.mlp.fc1.weight.requires_grad_(True)
    head(x).sum().backward()
    assert dec.mlp.fc1.weight.grad is None and head.out.weight.grad is not None
    dec.mlp.fc1.weight.requires_grad_(False)
    # both state-dict formats load, and .to()/.float() leave the decoder untouched
    head2 = PoseSigmaHead(dec).load_head_state_dict(head.state_dict())
    head3 = PoseSigmaHead(dec).load_head_state_dict(head.out.state_dict())
    assert torch.equal(head2(x), s2) and torch.equal(head3(x), s2)
    head2.to("cpu").float()
    assert dec.mlp.fc1.weight.dtype == torch.float32
    # a PoseDecoder-like object exposing fc1/act directly is accepted too
    assert PoseSigmaHead(dec.mlp).fc1 is dec.mlp.fc1


# ----------------------------------------------------------------------------- (3) source grep
def test_model_hooks_wrapped_once():
    src = open(MODEL_PY).read()
    commit = "state_feat = new_state_feat * state_mask + state_feat * (1 - state_mask)"
    upd = "new_mem = self.pose_retriever.update_mem(mem, global_img_feat_group, pooled_pose_feat)"
    memgate = "new_mem = self._wgate_mem_gate(views, view_indices, new_mem, mem, out_pose_feat_group[:, 0])"
    fgate = "state_mask = state_mask * wg_frame_w.to(state_mask.dtype)[:, None, None]"  # state-only (default)
    fjoint = "update_mask = update_mask * wg_frame_w.to(update_mask.dtype)[:, None, None]"  # joint mode
    alpha = "update_mask = update_mask * alpha[:, None, None]"
    pre = "wg_frame_w, wg_frame_feat = self._wgate_frame_pre("
    # the group step's mem commit goes through mem_mask, which IS update_mask unless the
    # mg_conf hook (_wgate_mem_conf) returned a weight -- see test_mem_conf_gate.py
    memmask = "mem_mask = update_mask"
    memcommit = "mem = new_mem * mem_mask + mem * (1 - mem_mask)"
    assert src.count(commit) == 1, src.count(commit)
    assert src.count(upd) == 1, src.count(upd)
    assert src.count(memgate) == 1, src.count(memgate)
    assert src.count(fgate) == 1, src.count(fgate)
    assert src.count(fjoint) == 1, src.count(fjoint)
    assert src.count(alpha) == 1, src.count(alpha)
    assert src.count(pre) == 1, src.count(pre)
    assert src.count(memcommit) == 1, src.count(memcommit)
    # the mem-gate wrapper is the very next statement after update_mem
    lines = src.splitlines()
    i_upd = next(i for i, l in enumerate(lines) if upd in l)
    assert memgate in lines[i_upd + 1], lines[i_upd + 1]
    # ordering inside _forward_decoder_group_step: rollout -> frame pre-hook -> update_mem/mem gate ->
    # ... -> update_alpha -> joint-mode multiply -> state_mask = update_mask -> token gate ->
    # state-only multiply -> state commit -> mem commit (update_mask, untouched by the state-only gate)
    i_step = src.index("def _forward_decoder_group_step(")
    i_next = src.index("def attach_conf_branch(")
    body = src[i_step:i_next]
    order = [
        body.index("new_state_feat, dec = self._recurrent_rollout("),
        body.index(pre),
        body.index(upd),
        body.index(memgate),
        body.index(alpha),
        body.index(fjoint),
        body.index("state_mask = update_mask\n"),
        body.index('tg_q = [views[i].get("token_gate_q"'),
        body.index(fgate),
        body.index(commit),
        body.index(memmask),
        body.index(memcommit),
    ]
    assert order == sorted(order), order
    # both multiplies are guarded by the None check (absent keys => no arithmetic at all) and are
    # mutually exclusive on the joint flag
    assert "if wg_frame_w is not None and not wg_frame_joint:\n            " + fgate in body
    assert "if wg_frame_w is not None and wg_frame_joint:\n            " + fjoint in body
    # the state-only multiply is the statement right before the commit
    lines = body.splitlines()
    i_fg = next(i for i, l in enumerate(lines) if fgate in l)
    assert commit in lines[i_fg + 1], lines[i_fg + 1]
    # and the commit line itself is untouched (not wrapped a second time)
    assert not re.search(r"_wgate\w*\(.*state_mask", body)


# ----------------------------------------------------------------------------- (4) hooks on a stub
def _stub():
    from dust3r.model import ARCroco3DStereo

    class _DH:  # stands in for model.downstream_head
        pose_head = PoseDecoder(hidden_size=768)

    class Stub:
        _wgate_key = staticmethod(ARCroco3DStereo._wgate_key)
        _wgate_scalar = staticmethod(ARCroco3DStereo._wgate_scalar)
        _wgate_trace_append = ARCroco3DStereo._wgate_trace_append
        _wgate_frame_pre = ARCroco3DStereo._wgate_frame_pre
        _wgate_mem_gate = ARCroco3DStereo._wgate_mem_gate
        _wgate_record_step = ARCroco3DStereo._wgate_record_step
        attach_frame_gate = ARCroco3DStereo.attach_frame_gate
        attach_mem_gate = ARCroco3DStereo.attach_mem_gate
        downstream_head = _DH()

        def parameters(self):  # attach_* read the model's device from here
            return iter([torch.zeros(1)])

    return Stub()


def _forced_frame_head(bias):
    h = FrameConfHead()
    nn.init.zeros_(h.out.weight)
    nn.init.constant_(h.out.bias, bias)
    return h.eval()


def _forced_pose_head(dec, bias):
    h = PoseSigmaHead(dec)
    nn.init.zeros_(h.out.weight)
    nn.init.constant_(h.out.bias, bias)
    return h.eval()


def _views(T, **keys):
    vs = [{"img": torch.zeros(1)} for _ in range(T)]
    for v in vs:
        for k, val in keys.items():
            v[k] = torch.tensor(float(val)).unsqueeze(0)
    return vs


def test_hooks_on_stub():
    torch.manual_seed(4)
    m = _stub()
    Nn, Bb = 16, 1
    new, old, g = torch.randn(Bb, Nn, C), torch.randn(Bb, Nn, C), torch.randn(Bb, 1, G)
    new_mem, mem = torch.randn(Bb, 4, 8), torch.randn(Bb, 4, 8)
    pose_feat = torch.randn(Bb, 768)
    # --- identity without keys: no features, no weight, the very same new_mem object, no trace
    vs = _views(3)
    assert m._wgate_frame_pre(vs, [1], new, old, g) == (None, None)
    assert m._wgate_mem_gate(vs, [1], new_mem, mem, pose_feat) is new_mem
    assert not hasattr(m, "_wgate_trace")
    # t == 0 without keys: still identity, but the trace lists are (re)set to empty (unconditional reset)
    assert m._wgate_frame_pre(vs, [0], new, old, g) == (None, None)
    assert m._wgate_mem_gate(vs, [0], new_mem, mem, pose_feat) is new_mem
    assert m._wgate_trace == [] and m._wgate_trace_frame == [] and m._wgate_trace_mem == [] and m._wgate_trace_logvar == []
    # --- recording only: features but no weight; dicts with squeezed shapes
    m.wgate_record = []
    w, f = m._wgate_frame_pre(vs, [0], new, old, g)
    assert w is None and f.shape == (Bb, 1796)
    m._wgate_record_step([0], f, pose_feat, {"camera_pose": torch.randn(Bb, 7)})
    r = m.wgate_record[0]
    assert r["t"] == 0 and r["frame_feat"].shape == (1796,) and r["pose_feat"].shape == (768,) and r["camera_pose"].shape == (7,)
    assert r["frame_feat"].dtype == torch.float32 and torch.equal(r["frame_feat"], f[0])
    m.wgate_record = None
    # --- frame gate: a_0 = 1 (None => no multiply), wmin clipping, wmin key override, trace
    m.frame_gate, m.frame_gate_log_ref, m.frame_gate_wmin = _forced_frame_head(+40.0), 0.0, 0.5
    vs = _views(3, frame_gate_on=1.0)
    assert m._wgate_frame_pre(vs, [0], new, old, g)[0] is None and m._wgate_trace == [(0, 1.0)]
    w, f = m._wgate_frame_pre(vs, [1], new, old, g)
    assert w.shape == (Bb,) and w.dtype == torch.float32 and w.item() == 0.5 and f.shape == (Bb, 1796)
    assert m._wgate_trace == [(0, 1.0), (1, 0.5)] and m._wgate_trace_frame == m._wgate_trace
    assert m._wgate_trace_logvar == [(1, "frame", [40.0])]
    vs = _views(3, frame_gate_on=1.0, frame_gate_wmin=0.25)
    assert m._wgate_frame_pre(vs, [2], new, old, g)[0].item() == 0.25
    m.frame_gate = _forced_frame_head(-40.0)  # far more confident than the reference -> 1.0
    assert m._wgate_frame_pre(vs, [2], new, old, g)[0].item() == 1.0
    # trace resets at t == 0
    m._wgate_frame_pre(vs, [0], new, old, g)
    assert m._wgate_trace == [(0, 1.0)]
    # frame_gate_on = 0 is "off"
    assert m._wgate_frame_pre(_views(2, frame_gate_on=0.0), [1], new, old, g) == (None, None)
    # ... and a following scene keyed off (or not keyed at all) at t == 0 does NOT inherit the trace
    m._wgate_frame_pre(vs, [1], new, old, g)
    assert len(m._wgate_trace) == 2
    assert m._wgate_frame_pre(_views(2, frame_gate_on=0.0), [0], new, old, g) == (None, None)
    assert m._wgate_trace == [] and m._wgate_trace_frame == [] and m._wgate_trace_logvar == []
    m._wgate_frame_pre(vs, [0], new, old, g)
    m._wgate_frame_pre(vs, [1], new, old, g)
    assert len(m._wgate_trace) == 2
    assert m._wgate_frame_pre(_views(2), [0], new, old, g) == (None, None)
    assert m._wgate_trace == []
    # --- frame_gate_const: fires without frame_gate_on (no head, no features), a_0 = 1, traced, no logvar
    m.frame_gate = None
    vs = _views(3, frame_gate_const=0.7)
    assert m._wgate_frame_pre(vs, [0], new, old, g) == (None, None) and m._wgate_trace == [(0, 1.0)]
    w, f = m._wgate_frame_pre(vs, [1], new, old, g)
    assert f is None and w.shape == (Bb,) and w.dtype == torch.float32 and abs(w.item() - 0.7) < 1e-7
    assert m._wgate_trace_frame == [(0, 1.0), (1, w.item())] and m._wgate_trace_logvar == []
    # const overrides the head (head still traced as logvar); wmin does not clip a constant
    m.frame_gate = _forced_frame_head(+40.0)
    vs = _views(3, frame_gate_on=1.0, frame_gate_const=0.3)
    m._wgate_frame_pre(vs, [0], new, old, g)
    w, f = m._wgate_frame_pre(vs, [1], new, old, g)
    assert f.shape == (Bb, 1796) and abs(w.item() - 0.3) < 1e-7
    assert m._wgate_trace_frame[-1][1] == w.item() and m._wgate_trace_logvar == [(1, "frame", [40.0])]
    # per-view schedule: a different const on each view
    vs = _views(3, frame_gate_const=0.9)
    vs[2]["frame_gate_const"] = torch.tensor(0.6).unsqueeze(0)
    m._wgate_frame_pre(vs, [0], new, old, g)
    assert abs(m._wgate_frame_pre(vs, [1], new, old, g)[0].item() - 0.9) < 1e-7
    assert abs(m._wgate_frame_pre(vs, [2], new, old, g)[0].item() - 0.6) < 1e-7
    # frame_gate_joint is read by the decoder step, not here: the pre-hook output is unchanged by it
    vs = _views(2, frame_gate_const=0.6, frame_gate_joint=1.0)
    m._wgate_frame_pre(vs, [0], new, old, g)
    assert abs(m._wgate_frame_pre(vs, [1], new, old, g)[0].item() - 0.6) < 1e-7
    m.frame_gate = _forced_frame_head(-40.0)
    vs = _views(3, frame_gate_on=1.0, frame_gate_wmin=0.25)
    m.frame_gate = None
    try:
        m._wgate_frame_pre(vs, [1], new, old, g)
        raise AssertionError("expected RuntimeError without an attached head")
    except RuntimeError:
        pass
    # --- mem gate: b_0 = 1 (same object), const, freeze_after, head, precedence, blend
    dec = PoseDecoder(hidden_size=768)
    vs = _views(4, mem_gate_const=0.0)
    m._wgate_frame_pre(vs, [0], new, old, g)  # the step runs the pre-hook first: it resets the trace at t == 0
    assert m._wgate_mem_gate(vs, [0], new_mem, mem, pose_feat) is new_mem
    out = m._wgate_mem_gate(vs, [1], new_mem, mem, pose_feat)
    assert torch.equal(out, mem) and m._wgate_trace == [(0, 1.0), (1, 0.0)] and m._wgate_trace_mem == m._wgate_trace
    assert m._wgate_mem_gate(_views(2, mem_gate_const=1.0), [1], new_mem, mem, pose_feat) is new_mem
    out = m._wgate_mem_gate(_views(2, mem_gate_const=0.25), [1], new_mem, mem, pose_feat)
    assert torch.allclose(out, 0.25 * new_mem + 0.75 * mem)
    vs = _views(4, mem_gate_freeze_after=2)
    assert m._wgate_mem_gate(vs, [2], new_mem, mem, pose_feat) is new_mem
    assert torch.equal(m._wgate_mem_gate(vs, [3], new_mem, mem, pose_feat), mem)
    m.mem_gate, m.mem_gate_log_ref, m.mem_gate_wmin = _forced_pose_head(dec, -40.0), (0.0, 0.0), 0.5
    vs = _views(3, mem_gate_on=1.0)
    assert m._wgate_mem_gate(vs, [1], new_mem, mem, pose_feat) is new_mem  # b = 1 exactly -> untouched
    m.mem_gate = _forced_pose_head(dec, +40.0)
    out = m._wgate_mem_gate(vs, [1], new_mem, mem, pose_feat)
    assert torch.allclose(out, 0.5 * new_mem + 0.5 * mem) and m._wgate_trace_mem[-1] == (1, 0.5)
    assert m._wgate_trace_logvar[-1] == (1, "mem", [40.0, 40.0])
    assert m._wgate_mem_gate(_views(3, mem_gate_on=1.0, mem_gate_wmin=0.2), [1], new_mem, mem, pose_feat)[0, 0, 0].item() \
        == (0.2 * new_mem + 0.8 * mem)[0, 0, 0].item()
    # const overrides the head; freeze_after zeroes both
    assert m._wgate_mem_gate(_views(3, mem_gate_on=1.0, mem_gate_const=1.0), [1], new_mem, mem, pose_feat) is new_mem
    assert torch.equal(m._wgate_mem_gate(_views(3, mem_gate_on=1.0, mem_gate_freeze_after=0), [1], new_mem, mem, pose_feat), mem)
    m.mem_gate = None
    try:
        m._wgate_mem_gate(vs, [1], new_mem, mem, pose_feat)
        raise AssertionError("expected RuntimeError without an attached head")
    except RuntimeError:
        pass
    # --- attach_*: explicit wmin / log_ref win over the whole-dict checkpoint, which wins over the 0.5 default
    fh = FrameConfHead()
    ck = {"state_dict": fh.state_dict(), "log_ref": 0.1, "wmin": 0.4, "args": {}, "metrics": {}}
    m.attach_frame_gate(ck)
    assert m.frame_gate_wmin == 0.4 and m.frame_gate_log_ref == 0.1 and not m.frame_gate.training
    assert all(not p.requires_grad for p in m.frame_gate.parameters())
    m.attach_frame_gate(ck, wmin=0.3)
    assert m.frame_gate_wmin == 0.3
    m.attach_frame_gate(ck, log_ref=0.7, wmin=0.3)
    assert m.frame_gate_log_ref == 0.7 and m.frame_gate_wmin == 0.3
    m.attach_frame_gate({"state_dict": fh.state_dict(), "log_ref": 0.1})  # no stored wmin -> default
    assert m.frame_gate_wmin == 0.5
    m.attach_frame_gate(fh.state_dict(), log_ref=0.2)  # bare state dict -> default wmin
    assert m.frame_gate_wmin == 0.5 and m.frame_gate_log_ref == 0.2
    m.attach_frame_gate(fh, log_ref=0.2, wmin=1.0)
    assert m.frame_gate is fh and m.frame_gate_wmin == 1.0
    try:
        m.attach_frame_gate(fh.state_dict())
        raise AssertionError("expected AssertionError without log_ref")
    except AssertionError as e:
        assert "log_ref" in str(e)
    assert m.attach_frame_gate(None) is None and m.frame_gate is None
    ph = PoseSigmaHead(m.downstream_head.pose_head)
    ck2 = {"state_dict": ph.state_dict(), "log_ref_t": 0.1, "log_ref_R": 0.2, "wmin": 0.4}
    m.attach_mem_gate(ck2)
    assert m.mem_gate_wmin == 0.4 and m.mem_gate_log_ref == (0.1, 0.2) and not m.mem_gate.training
    m.attach_mem_gate(ck2, wmin=0.3)
    assert m.mem_gate_wmin == 0.3
    m.attach_mem_gate(ck2, log_ref_R=0.9)
    assert m.mem_gate_log_ref == (0.1, 0.9) and m.mem_gate_wmin == 0.4
    m.attach_mem_gate({"state_dict": ph.state_dict(), "log_ref_t": 0.1, "log_ref_R": 0.2})
    assert m.mem_gate_wmin == 0.5
    m.attach_mem_gate(ph.out.state_dict(), log_ref_t=0.0, log_ref_R=0.0)  # {"weight","bias"} format
    assert m.mem_gate_wmin == 0.5 and torch.equal(m.mem_gate.out.weight, ph.out.weight)
    assert m.attach_mem_gate(None) is None and m.mem_gate is None
    # --- both gates on the same step: merged (t, a, b) entries, per-gate lists intact
    m.frame_gate, m.mem_gate = _forced_frame_head(+40.0), _forced_pose_head(dec, +40.0)
    m.frame_gate_log_ref, m.frame_gate_wmin = 0.0, 0.5  # (the attach tests above left other refs / wmin behind)
    m.mem_gate_log_ref, m.mem_gate_wmin = (0.0, 0.0), 0.5
    vs = _views(2, frame_gate_on=1.0, mem_gate_on=1.0)
    for t in range(2):
        m._wgate_frame_pre(vs, [t], new, old, g)
        m._wgate_mem_gate(vs, [t], new_mem, mem, pose_feat)
    assert m._wgate_trace == [(0, 1.0, 1.0), (1, 0.5, 0.5)], m._wgate_trace
    assert m._wgate_trace_frame == [(0, 1.0), (1, 0.5)] and m._wgate_trace_mem == [(0, 1.0), (1, 0.5)]
    # --- batched weights are stored as lists
    w = m._wgate_frame_pre(vs, [1], new.repeat(3, 1, 1), old.repeat(3, 1, 1), g.repeat(3, 1, 1))[0]
    assert w.shape == (3,) and m._wgate_trace_frame[-1] == (1, [0.5, 0.5, 0.5])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}", flush=True)
    print(f"all {len(tests)} tests passed")
