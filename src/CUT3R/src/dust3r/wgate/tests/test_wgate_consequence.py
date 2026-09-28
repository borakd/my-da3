"""Tests for variant C -- the CONSEQUENCE-trained gate (plain asserts, CPU only, fixed seeds).

    cd src/CUT3R/src && OMP_NUM_THREADS=1 timeout 280 python dust3r/wgate/tests/test_wgate_consequence.py
    (or: python -m pytest -q dust3r/wgate/tests/test_wgate_consequence.py)

Contract: WRITE_GATE_VARIANTS.md, section "Variant C: the CONSEQUENCE-trained gate". Covered here:
  (1) GateWeightHead, both kinds: the init weight is EXACTLY wmin + (1-wmin)*sigmoid(init_logit) for
      random inputs (zeroed last layer), the wmin override, wmin=1 -> exactly 1, fp32 under autocast,
      state_dict keys (the pose kind must not save the borrowed fc1), checkpoint round trip;
  (2) a synthetic end-to-end gradient test that needs no checkpoint: w = head(x), y = w*a + (1-w)*b,
      loss = y.sum() -> every head parameter gets a finite, non-zero grad, the borrowed frozen fc1 gets
      none, and at the zeroed init only the last layer moves;
  (3) the attach paths on a stub model: sigma vs weight checkpoints, GateWeightHead instances,
      train=True/False (requires_grad + module mode + whether the hook's weight carries gradient),
      log_ref required for sigma and absent for weight, kind mismatches rejected;
  (4) the hooks with a weight-mode head: a_0 = b_0 = 1, const / freeze_after still override, the
      "frame_u" / "mem_u" trace tags, gradient reaching the head through the state commit and through
      the mem blend, and the bit-exact b == 1 fast path (taken without grad, skipped with grad).
The sigma-mode behaviour of the same code paths is test_wgate.py, which must stay green.
"""

import math
import os
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
    GateWeightHead,
    PoseSigmaHead,
)

torch.manual_seed(0)
B, N, C, G = 2, 16, 768, 1024


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _autocast_cpu_bf16():
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True)


def _pose_decoder():
    dec = PoseDecoder(hidden_size=768)
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


# ----------------------------------------------------------------- (1) the head: init, map, state dict
def test_gate_weight_head_init_is_exact():
    torch.manual_seed(1)
    dec = _pose_decoder()
    for kind, x, kw in (
        ("frame", torch.randn(5, FRAME_FEAT_DIM) * 7.0, {}),
        ("pose", torch.randn(5, 768) * 7.0, {"pose_decoder": dec}),
    ):
        for wmin, logit in ((0.5, 2.0), (0.5, 0.0), (0.0, 3.0), (0.25, -1.5), (0.9, 2.0)):
            head = GateWeightHead(kind, wmin=wmin, init_logit=logit, **kw)
            want = wmin + (1.0 - wmin) * _sigmoid(logit)
            assert abs(head.weight_at_init - want) < 1e-12, (kind, wmin, logit, head.weight_at_init)
            w = head(x)
            assert w.shape == (5,) and w.dtype == torch.float32, (w.shape, w.dtype)
            # every input gives the SAME weight before the first step (last layer weight == 0)
            w = w.detach()
            assert float(w.max() - w.min()) == 0.0, w
            assert abs(float(w[0]) - want) < 1e-6, (kind, wmin, logit, float(w[0]), want)
            assert abs(float(head.logit(x).detach()[0]) - logit) < 1e-6  # u == the init bias
            assert torch.equal(head.out.weight, torch.zeros_like(head.out.weight))
            assert wmin <= float(w[0]) <= 1.0, (wmin, float(w[0]))
    # defaults of the contract: kind='frame', in 1796, hidden 256, wmin .5, init_logit 2.0 -> ~.94
    h = GateWeightHead("frame")
    assert h.in_dim == 1796 and h.hidden == 256 and h.wmin == 0.5 and h.init_logit == 2.0
    assert abs(h.weight_at_init - 0.9403985) < 1e-6
    assert h.norm.normalized_shape == (1796,) and h.fc1.out_features == 256 and h.out.out_features == 1
    hp = GateWeightHead("pose", pose_decoder=dec)
    assert hp.in_dim == 768 and hp.hidden == 3072 and hp.out.in_features == 3072 and hp.out.out_features == 1
    assert hp.borrowed_fc1 is dec.mlp.fc1 and hp.borrowed_act is dec.mlp.act
    assert h.borrowed_fc1 is None and h.borrowed_act is None


def test_gate_weight_head_map_and_precision():
    torch.manual_seed(2)
    head = GateWeightHead("frame", wmin=0.5, init_logit=2.0)
    nn.init.normal_(head.out.weight, std=0.05)  # leave the degenerate init
    x = torch.randn(7, FRAME_FEAT_DIM)
    u = head.logit(x)
    w = head(x)
    assert torch.allclose(w, 0.5 + 0.5 * torch.sigmoid(u), atol=0, rtol=0)
    assert bool(((w > 0.5) & (w < 1.0)).all()), w
    # wmin override (the model hook passes the per-view key through); wmin = 1 -> exactly 1 everywhere
    w25 = head(x, wmin=0.25)
    assert torch.allclose(w25, 0.25 + 0.75 * torch.sigmoid(u))
    w1 = head(x, wmin=1.0)
    assert bool((w1 == 1.0).all()), w1
    assert torch.allclose(head.weight_from_logit(u), w)
    # monotone in the logit
    us = torch.linspace(-5, 5, 11)
    ws = head.weight_from_logit(us)
    assert bool((ws[1:] > ws[:-1]).all())
    # float32 with autocast disabled, and bit-identical across calls
    with _autocast_cpu_bf16():
        wa = head(x)
    assert wa.dtype == torch.float32 and torch.equal(wa, w)
    assert torch.equal(head(x), w)
    assert head(x.bfloat16()).dtype == torch.float32


def test_gate_weight_head_state_dict_and_checkpoint():
    torch.manual_seed(3)
    dec = _pose_decoder()
    hf = GateWeightHead("frame")
    nn.init.normal_(hf.out.weight, std=0.05)
    assert set(hf.state_dict().keys()) == {
        "norm.weight", "norm.bias", "fc1.weight", "fc1.bias", "out.weight", "out.bias",
    }, hf.state_dict().keys()
    hp = GateWeightHead("pose", pose_decoder=dec)
    nn.init.normal_(hp.out.weight, std=0.05)
    # the borrowed fc1 is NOT saved and NOT a parameter of the head
    assert set(hp.state_dict().keys()) == {"out.weight", "out.bias"}, hp.state_dict().keys()
    assert len(list(hp.parameters())) == 2
    assert hp.out.weight.shape == (1, 3072)
    # .to()/.float() on the head never touch the model's decoder
    hp.to("cpu").float()
    assert dec.mlp.fc1.weight.dtype == torch.float32
    # round trip through from_state_dict (kind inferred) and through the checkpoint dict
    xf, xp = torch.randn(4, FRAME_FEAT_DIM), torch.randn(4, 768)
    assert torch.equal(GateWeightHead.from_state_dict(hf.state_dict())(xf), hf(xf))
    assert torch.equal(
        GateWeightHead.from_state_dict(hp.state_dict(), pose_decoder=dec)(xp), hp(xp)
    )
    assert GateWeightHead.from_state_dict(hf.state_dict()).kind == "frame"
    assert GateWeightHead.from_state_dict(hp.state_dict(), pose_decoder=dec).kind == "pose"
    # the bare {"weight","bias"} format loads too (mirrors PoseSigmaHead)
    assert torch.equal(
        GateWeightHead("pose", pose_decoder=dec).load_head_state_dict(hp.out.state_dict())(xp), hp(xp)
    )
    ck = GateWeightHead("frame", wmin=0.3, init_logit=1.25).checkpoint(args={"lr": 1e-4}, metrics={"ate": 1.0})
    assert ck["mode"] == "weight" and ck["wmin"] == 0.3 and ck["init_logit"] == 1.25
    assert set(ck["state_dict"].keys()) == set(hf.state_dict().keys())
    assert ck["head_cfg"]["kind"] == "frame" and ck["args"]["lr"] == 1e-4 and ck["metrics"]["ate"] == 1.0
    # a non-square head (other sizes) is rebuilt from its own state dict
    small = GateWeightHead("frame", in_dim=10, hidden=4)
    assert GateWeightHead.from_state_dict(small.state_dict()).fc1.out_features == 4
    # kind validation: an unknown kind, a pose kind without the decoder, a frame kind with one
    for kind_, kw_ in (("bogus", {}), ("pose", {}), ("frame", {"pose_decoder": dec})):
        raised = False
        try:
            GateWeightHead(kind_, **kw_)
        except AssertionError:
            raised = True
        assert raised, (kind_, kw_)


# ----------------------------------------------------------------- (2) synthetic gradient test
def _grad_probe(head, x):
    """w = head(x); y = w*a + (1-w)*b; loss = y.sum() -- the write blend in miniature."""
    torch.manual_seed(11)
    a = torch.randn(x.shape[0], 5)
    b = torch.randn(x.shape[0], 5)
    w = head(x)
    assert w.requires_grad, "the head's own forward must carry gradient"
    y = w[:, None] * a + (1.0 - w[:, None]) * b
    y.sum().backward()
    return w


def test_gradient_through_the_weight_synthetic():
    """No checkpoint, no rollout: the blend that both write sites use, differentiated in the weight."""
    torch.manual_seed(4)
    dec = _pose_decoder()
    for kind, x, kw in (
        ("frame", torch.randn(6, FRAME_FEAT_DIM), {}),
        ("pose", torch.randn(6, 768), {"pose_decoder": dec}),
    ):
        head = GateWeightHead(kind, **kw)
        nn.init.normal_(head.out.weight, std=0.05)  # a head that has taken a step already
        nn.init.normal_(head.out.bias, mean=2.0, std=0.05)
        _grad_probe(head, x)
        for n_, p_ in head.named_parameters():
            assert p_.grad is not None, (kind, n_)
            assert torch.isfinite(p_.grad).all(), (kind, n_)
            assert float(p_.grad.abs().max()) > 0, (kind, n_)
        if kind == "pose":
            # the borrowed fc1 never receives a gradient -- even when the model's own copy of it is
            # (temporarily) left trainable, because the head detaches its weights and its input
            assert dec.mlp.fc1.weight.grad is None and dec.mlp.fc1.bias.grad is None
            dec.mlp.fc1.weight.requires_grad_(True)
            head.out.weight.grad = None
            _grad_probe(head, x)
            assert dec.mlp.fc1.weight.grad is None and float(head.out.weight.grad.abs().max()) > 0
            dec.mlp.fc1.weight.requires_grad_(False)
    # the input features are NOT a gradient path for the pose kind (they are detached inside)
    xp = torch.randn(3, 768, requires_grad=True)
    hp = GateWeightHead("pose", pose_decoder=dec)
    nn.init.normal_(hp.out.weight, std=0.05)
    hp(xp).sum().backward()
    assert xp.grad is None
    # at the contract's init (out.weight == 0) only the LAST layer moves; the MLP grad is exactly 0
    hf = GateWeightHead("frame")
    _grad_probe(hf, torch.randn(6, FRAME_FEAT_DIM))
    assert float(hf.out.weight.grad.abs().max()) > 0 and float(hf.out.bias.grad.abs().max()) > 0
    assert float(hf.fc1.weight.grad.abs().max()) == 0.0 and float(hf.norm.weight.grad.abs().max()) == 0.0
    # one optimiser step moves the weight away from the init value
    hf2 = GateWeightHead("frame")
    w0 = float(hf2(torch.randn(1, FRAME_FEAT_DIM)).detach()[0])
    opt = torch.optim.SGD(hf2.parameters(), lr=1.0)
    _grad_probe(hf2, torch.randn(6, FRAME_FEAT_DIM))
    opt.step()
    assert abs(float(hf2(torch.randn(1, FRAME_FEAT_DIM)).detach()[0]) - w0) > 1e-4


# ----------------------------------------------------------------- (3)+(4) stub model: attach + hooks
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
        attach_frame_gate = ARCroco3DStereo.attach_frame_gate
        attach_mem_gate = ARCroco3DStereo.attach_mem_gate
        downstream_head = _DH()

        def parameters(self):  # attach_* read the model's device from here
            return iter([torch.zeros(1)])

    s = Stub()
    for p_ in s.downstream_head.pose_head.parameters():
        p_.requires_grad_(False)
    return s


def _views(T, **keys):
    vs = [{"img": torch.zeros(1)} for _ in range(T)]
    for v in vs:
        for k, val in keys.items():
            v[k] = torch.tensor(float(val)).unsqueeze(0)
    return vs


def _tensors(Bb=1):
    torch.manual_seed(7)
    return (
        torch.randn(Bb, N, C),      # new_state_feat
        torch.randn(Bb, N, C),      # state_feat
        torch.randn(Bb, 1, G),      # global_img_feat
        torch.randn(Bb, 4, 8),      # new_mem
        torch.randn(Bb, 4, 8),      # mem
        torch.randn(Bb, 768),       # pose_feat
    )


def test_attach_dispatches_on_mode():
    torch.manual_seed(5)
    m = _stub()
    dec = m.downstream_head.pose_head

    # --- frame: sigma checkpoints keep the historical behaviour ---------------------------------
    fh = FrameConfHead()
    ck_sig = {"state_dict": fh.state_dict(), "log_ref": 0.1, "wmin": 0.4}
    m.attach_frame_gate(ck_sig)
    assert m.frame_gate_mode == "sigma" and m.frame_gate_log_ref == 0.1 and m.frame_gate_wmin == 0.4
    assert isinstance(m.frame_gate, FrameConfHead) and not m.frame_gate.training
    assert m.frame_gate_train is False and all(not p.requires_grad for p in m.frame_gate.parameters())
    m.attach_frame_gate({"state_dict": fh.state_dict(), "log_ref": 0.1, "mode": "sigma"})
    assert m.frame_gate_mode == "sigma" and isinstance(m.frame_gate, FrameConfHead)
    try:  # sigma mode still demands a log_ref
        m.attach_frame_gate(fh.state_dict())
        raise AssertionError("expected AssertionError without log_ref")
    except AssertionError as e:
        assert "log_ref" in str(e), e

    # --- frame: weight checkpoints -----------------------------------------------------------
    gh = GateWeightHead("frame", wmin=0.5, init_logit=2.0)
    nn.init.normal_(gh.out.weight, std=0.05)
    ck_w = gh.checkpoint()
    head = m.attach_frame_gate(ck_w)
    assert m.frame_gate_mode == "weight" and m.frame_gate_log_ref is None and m.frame_gate_wmin == 0.5
    assert isinstance(head, GateWeightHead) and head.kind == "frame" and head.init_logit == 2.0
    x = torch.randn(3, FRAME_FEAT_DIM)
    assert torch.equal(head(x), gh(x))  # the stored weights round-tripped
    assert not head.training and all(not p.requires_grad for p in head.parameters())
    # an explicit wmin wins over the checkpoint's; the head instance is accepted directly
    m.attach_frame_gate(ck_w, wmin=0.25)
    assert m.frame_gate_wmin == 0.25 and m.frame_gate_mode == "weight"
    m.attach_frame_gate(gh)
    assert m.frame_gate is gh and m.frame_gate_mode == "weight" and m.frame_gate_wmin == 0.5
    m.attach_frame_gate({"state_dict": gh.state_dict(), "mode": "weight"})  # no wmin stored -> 0.5
    assert m.frame_gate_wmin == 0.5 and m.frame_gate_mode == "weight"
    # a log_ref passed by an old caller is ignored in weight mode (there is none)
    m.attach_frame_gate(ck_w, 0.9)
    assert m.frame_gate_log_ref is None
    # train=True / False
    m.attach_frame_gate(ck_w, train=True)
    assert m.frame_gate_train is True and m.frame_gate.training
    assert all(p.requires_grad for p in m.frame_gate.parameters())
    m.attach_frame_gate(ck_w, train=False)
    assert m.frame_gate_train is False and not m.frame_gate.training
    assert all(not p.requires_grad for p in m.frame_gate.parameters())
    # detaching clears the mode / train flags
    assert m.attach_frame_gate(None) is None
    assert m.frame_gate is None and m.frame_gate_mode == "sigma" and m.frame_gate_train is False
    # kind mismatch is rejected
    try:
        m.attach_frame_gate(GateWeightHead("pose", pose_decoder=dec))
        raise AssertionError("expected AssertionError on the kind mismatch")
    except AssertionError as e:
        assert "kind" in str(e), e

    # --- mem: sigma vs weight -----------------------------------------------------------------
    ph = PoseSigmaHead(dec)
    m.attach_mem_gate({"state_dict": ph.state_dict(), "log_ref_t": 0.1, "log_ref_R": 0.2})
    assert m.mem_gate_mode == "sigma" and m.mem_gate_log_ref == (0.1, 0.2) and m.mem_gate_train is False
    assert isinstance(m.mem_gate, PoseSigmaHead)
    gp = GateWeightHead("pose", pose_decoder=dec, wmin=0.5, init_logit=2.0)
    nn.init.normal_(gp.out.weight, std=0.05)
    ck_wp = gp.checkpoint()
    assert set(ck_wp["state_dict"].keys()) == {"out.weight", "out.bias"}
    head = m.attach_mem_gate(ck_wp)
    assert m.mem_gate_mode == "weight" and m.mem_gate_log_ref is None and m.mem_gate_wmin == 0.5
    assert isinstance(head, GateWeightHead) and head.kind == "pose"
    xp = torch.randn(3, 768)
    assert torch.equal(head(xp), gp(xp))
    assert head.borrowed_fc1 is dec.mlp.fc1  # borrowed from THIS model's frozen pose decoder
    m.attach_mem_gate(ck_wp, train=True)
    assert m.mem_gate_train is True and m.mem_gate.training
    assert all(p.requires_grad for p in m.mem_gate.parameters())
    assert all(not p.requires_grad for p in dec.parameters())  # the backbone stays frozen
    m.attach_mem_gate(gp)
    assert m.mem_gate is gp and m.mem_gate_mode == "weight"
    try:
        m.attach_mem_gate(GateWeightHead("frame"))
        raise AssertionError("expected AssertionError on the kind mismatch")
    except AssertionError as e:
        assert "kind" in str(e), e
    assert m.attach_mem_gate(None) is None
    assert m.mem_gate is None and m.mem_gate_mode == "sigma" and m.mem_gate_train is False
    # an unknown mode is refused rather than silently treated as sigma
    try:
        m.attach_frame_gate({"state_dict": gh.state_dict(), "mode": "bogus"})
        raise AssertionError("expected AssertionError on an unknown mode")
    except AssertionError as e:
        assert "mode" in str(e), e


def test_weight_mode_hooks():
    torch.manual_seed(6)
    m = _stub()
    dec = m.downstream_head.pose_head
    new, old, g, new_mem, mem, pose_feat = _tensors()
    gh = GateWeightHead("frame", wmin=0.5, init_logit=2.0)
    nn.init.normal_(gh.out.weight, std=0.05)
    gp = GateWeightHead("pose", pose_decoder=dec, wmin=0.5, init_logit=2.0)
    nn.init.normal_(gp.out.weight, std=0.05)

    # --- eval (train=False): no grad anywhere, a_0 = b_0 = 1, the "*_u" trace tags -------------
    m.attach_frame_gate(gh.checkpoint(), wmin=0.5)
    m.attach_mem_gate(gp.checkpoint(), wmin=0.5)
    vs = _views(3, frame_gate_on=1.0, mem_gate_on=1.0)
    assert m._wgate_frame_pre(vs, [0], new, old, g)[0] is None      # a_0 = 1: no multiply at all
    assert m._wgate_mem_gate(vs, [0], new_mem, mem, pose_feat) is new_mem  # b_0 = 1: same object
    w, feat = m._wgate_frame_pre(vs, [1], new, old, g)
    assert w.shape == (1,) and w.dtype == torch.float32 and not w.requires_grad
    assert feat.shape == (1, FRAME_FEAT_DIM)
    want_a = float(gh(feat).detach()[0])
    assert abs(float(w[0]) - want_a) < 1e-7 and 0.5 < float(w[0]) < 1.0
    out = m._wgate_mem_gate(vs, [1], new_mem, mem, pose_feat)
    b_traced = m._wgate_trace_mem[-1][1]
    want_b = float(gp(pose_feat).detach()[0])
    assert abs(b_traced - want_b) < 1e-7 and 0.5 < b_traced < 1.0
    assert torch.allclose(out, want_b * new_mem + (1 - want_b) * mem) and not out.requires_grad
    # the trace carries the raw LOGIT under the weight-mode tags, so another wmin can be replayed
    tags = [e[1] for e in m._wgate_trace_logvar]
    assert tags == ["frame_u", "mem_u"], tags
    u_a = m._wgate_trace_logvar[0][2][0]
    assert abs((0.5 + 0.5 / (1.0 + math.exp(-u_a))) - float(w[0])) < 1e-6
    assert abs(0.25 + 0.75 / (1.0 + math.exp(-u_a)) - float(gh(feat, wmin=0.25).detach()[0])) < 1e-6
    assert m._wgate_trace == [(0, 1.0, 1.0), (1, float(w[0]), b_traced)], m._wgate_trace

    # --- the per-view wmin key reaches the head; wmin = 1 is plain CUT3R, bit-exactly ----------
    vs1 = _views(3, frame_gate_on=1.0, mem_gate_on=1.0, frame_gate_wmin=1.0, mem_gate_wmin=1.0)
    m._wgate_frame_pre(vs1, [0], new, old, g)
    w1 = m._wgate_frame_pre(vs1, [1], new, old, g)[0]
    assert float(w1[0]) == 1.0
    assert m._wgate_mem_gate(vs1, [1], new_mem, mem, pose_feat) is new_mem  # fast path (no grad)
    vs2 = _views(3, frame_gate_on=1.0, frame_gate_wmin=0.25)
    m._wgate_frame_pre(vs2, [0], new, old, g)
    w2, f2 = m._wgate_frame_pre(vs2, [1], new, old, g)
    assert abs(float(w2[0]) - float(gh(f2, wmin=0.25).detach()[0])) < 1e-7 and float(w2[0]) < float(w[0])

    # --- the controls still override a weight-mode head ----------------------------------------
    vsc = _views(3, frame_gate_on=1.0, mem_gate_on=1.0, frame_gate_const=0.3, mem_gate_const=1.0)
    m._wgate_frame_pre(vsc, [0], new, old, g)
    wc = m._wgate_frame_pre(vsc, [1], new, old, g)[0]
    assert abs(float(wc[0]) - 0.3) < 1e-7 and not wc.requires_grad
    assert m._wgate_mem_gate(vsc, [1], new_mem, mem, pose_feat) is new_mem  # const 1 -> fast path
    vsf = _views(4, mem_gate_on=1.0, mem_gate_freeze_after=1)
    m._wgate_frame_pre(vsf, [0], new, old, g)
    assert m._wgate_mem_gate(vsf, [1], new_mem, mem, pose_feat).requires_grad is False
    assert torch.equal(m._wgate_mem_gate(vsf, [2], new_mem, mem, pose_feat), mem)  # frozen after k

    # --- train=True: the weight carries gradient, the FEATURES do not --------------------------
    m.attach_frame_gate(gh.checkpoint(), wmin=0.5, train=True)
    m.attach_mem_gate(gp.checkpoint(), wmin=0.5, train=True)
    fhead, mhead = m.frame_gate, m.mem_gate
    m._wgate_frame_pre(vs, [0], new, old, g)
    w, feat = m._wgate_frame_pre(vs, [1], new, old, g)
    assert w.requires_grad and not feat.requires_grad
    # the state commit, as _forward_decoder_group_step writes it (state-only frame gate)
    state_mask = torch.ones(1, 1, 1)
    state_mask = state_mask * w.to(state_mask.dtype)[:, None, None]
    state = new * state_mask + old * (1 - state_mask)
    state.sum().backward()
    for n_, p_ in fhead.named_parameters():
        assert p_.grad is not None and torch.isfinite(p_.grad).all(), n_
    assert float(fhead.out.weight.grad.abs().max()) > 0 and float(fhead.out.bias.grad.abs().max()) > 0
    # the mem blend
    out = m._wgate_mem_gate(vs, [1], new_mem, mem, pose_feat)
    assert out.requires_grad
    out.sum().backward()
    assert float(mhead.out.weight.grad.abs().max()) > 0 and torch.isfinite(mhead.out.bias.grad).all()
    assert dec.mlp.fc1.weight.grad is None  # the frozen borrowed layer never accumulates

    # --- the b == 1 fast path must NOT cut the gradient while training -------------------------
    m._wgate_frame_pre(vs1, [0], new, old, g)
    out1 = m._wgate_mem_gate(vs1, [1], new_mem, mem, pose_feat)  # wmin = 1 -> b == 1 exactly
    assert out1 is not new_mem and out1.requires_grad  # blend kept for the gradient
    assert torch.allclose(out1, new_mem, atol=0, rtol=0)  # ... and is the identity in value
    mhead.out.weight.grad = None
    mhead.out.bias.grad = None
    out1.sum().backward()
    assert mhead.out.bias.grad is not None and torch.isfinite(mhead.out.bias.grad).all()

    # --- a trainable head attached inside torch.no_grad() never builds a graph ------------------
    with torch.no_grad():
        m._wgate_frame_pre(vs, [0], new, old, g)
        w_ng = m._wgate_frame_pre(vs, [1], new, old, g)[0]
        out_ng = m._wgate_mem_gate(vs, [1], new_mem, mem, pose_feat)
    assert not w_ng.requires_grad and not out_ng.requires_grad
    assert abs(float(w_ng[0]) - float(w.detach()[0])) < 1e-7  # same value, no graph


def test_sigma_mode_unchanged_by_the_variant_c_additions():
    """A sigma head attached the old way must give exactly the old weights and no gradient."""
    torch.manual_seed(8)
    m = _stub()
    dec = m.downstream_head.pose_head
    new, old, g, new_mem, mem, pose_feat = _tensors()
    fh, ph = FrameConfHead(), PoseSigmaHead(dec)
    nn.init.normal_(fh.out.weight, std=0.05)
    nn.init.normal_(ph.out.weight, std=0.05)
    m.attach_frame_gate({"state_dict": fh.state_dict(), "log_ref": 0.1, "wmin": 0.5})
    m.attach_mem_gate({"state_dict": ph.state_dict(), "log_ref_t": 0.1, "log_ref_R": 0.2, "wmin": 0.5})
    from dust3r.wgate.heads import pose_sigma_to_weight, sigma_to_weight

    vs = _views(2, frame_gate_on=1.0, mem_gate_on=1.0)
    m._wgate_frame_pre(vs, [0], new, old, g)
    w, feat = m._wgate_frame_pre(vs, [1], new, old, g)
    assert torch.equal(w, sigma_to_weight(fh(feat), 0.1, 0.5)) and not w.requires_grad
    out = m._wgate_mem_gate(vs, [1], new_mem, mem, pose_feat)
    b = pose_sigma_to_weight(ph(pose_feat.float()), (0.1, 0.2), 0.5)
    assert torch.allclose(out, b[:, None, None] * new_mem + (1 - b[:, None, None]) * mem)
    assert [e[1] for e in m._wgate_trace_logvar] == ["frame", "mem"]
    # even a sigma head attached with train=True stays a valid sigma path (grad on, same map)
    m.attach_frame_gate({"state_dict": fh.state_dict(), "log_ref": 0.1, "wmin": 0.5}, train=True)
    m._wgate_frame_pre(vs, [0], new, old, g)
    w2 = m._wgate_frame_pre(vs, [1], new, old, g)[0]
    assert torch.allclose(w2, w)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}", flush=True)
    print(f"all {len(tests)} tests passed")
