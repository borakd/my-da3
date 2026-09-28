"""Tests for the PER-TOKEN consequence gate (GateWeightHead kind="token"), synthetic, CPU, seconds.

    cd src/CUT3R/src && OMP_NUM_THREADS=1 timeout 280 python dust3r/wgate/tests/test_token_gate.py

  (1) token_features: shape (B, N, 2565), the first 1796 dims are frame_features for every token, then
      the token's own delta and its norm; detached;
  (2) GateWeightHead("token"): (B, N) weights, EXACTLY weight_at_init for every token at the contract init,
      equal (float32) to the per-frame head's init weight, gradient reaches every parameter and differs
      per token once the last layer is non-degenerate; checkpoint round trip through from_state_dict;
  (3) model source: the per-token weights are split off right after the frame pre-hook, multiplied into
      state_mask after the token gate / fg_conf block and before the per-frame multiply, never into
      the mem commit, and the commit lines are unchanged.
"""

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dust3r.heads  # noqa: F401,E402
from dust3r.wgate.heads import (  # noqa: E402
    FRAME_FEAT_DIM, TOKEN_FEAT_DIM, GateWeightHead, frame_features, token_features)


def _inputs(B=2, N=768, C=768, seed=0):
    g = torch.Generator().manual_seed(seed)
    new = torch.randn(B, N, C, generator=g)
    old = torch.randn(B, N, C, generator=g)
    img = torch.randn(B, 1, 1024, generator=g)
    return new, old, img


def test_token_features():
    new, old, img = _inputs()
    x = token_features(new, old, img)
    f = frame_features(new, old, img)
    assert x.shape == (2, 768, TOKEN_FEAT_DIM) and TOKEN_FEAT_DIM == FRAME_FEAT_DIM + 768 + 1
    assert torch.equal(x[:, :, :FRAME_FEAT_DIM], f[:, None, :].expand(2, 768, FRAME_FEAT_DIM))
    d = new - old
    assert torch.allclose(x[:, :, FRAME_FEAT_DIM:FRAME_FEAT_DIM + 768], d)
    assert torch.allclose(x[:, :, -1], d.norm(dim=-1))
    assert not x.requires_grad
    new.requires_grad_(True)
    assert not token_features(new, old, img).requires_grad  # detached even when the input has a graph


def test_init_is_exact_and_equals_frame_head():
    new, old, img = _inputs()
    ht = GateWeightHead("token", wmin=0.5, init_logit=2.0)
    hf = GateWeightHead("frame", wmin=0.5, init_logit=2.0)
    wt = ht(token_features(new, old, img))
    wf = hf(frame_features(new, old, img))
    assert wt.shape == (2, 768) and wf.shape == (2,)
    assert torch.all(wt == wf[:, None]), (wt.unique(), wf)          # bitwise: same float32 value
    assert abs(float(wt[0, 0]) - ht.weight_at_init) < 1e-6
    assert ht.in_dim == TOKEN_FEAT_DIM and ht.head_cfg()["kind"] == "token"
    w1 = ht(token_features(new, old, img), wmin=1.0)
    assert torch.all(w1 == 1.0)


def test_gradient_per_token():
    new, old, img = _inputs()
    ht = GateWeightHead("token", wmin=0.5, init_logit=2.0)
    nn.init.normal_(ht.out.weight, std=0.02)
    w = ht(token_features(new, old, img))
    assert float(w.std()) > 0 and bool(((w > 0.5) & (w < 1.0)).all())
    coef = torch.randn(w.shape, generator=torch.Generator().manual_seed(1))
    (coef * w).sum().backward()
    for n, p in ht.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().max()) > 0, n


def test_checkpoint_round_trip():
    ht = GateWeightHead("token", wmin=0.5, init_logit=2.0)
    nn.init.normal_(ht.out.weight, std=0.02)
    ck = ht.checkpoint()
    h2 = GateWeightHead.from_state_dict(ck["state_dict"], kind="token", wmin=ck["wmin"], init_logit=ck["init_logit"])
    assert h2.kind == "token" and h2.in_dim == TOKEN_FEAT_DIM
    new, old, img = _inputs()
    x = token_features(new, old, img)
    assert torch.equal(ht(x), h2(x))
    from dust3r.model import _wgate_ck_kind
    assert _wgate_ck_kind(ck) == "token"
    assert _wgate_ck_kind({"kind": "token", "site": "state"}) == "token"


def test_model_source_placement():
    src = open(os.path.join(_SRC, "dust3r", "model.py")).read()
    body = src[src.index("def _forward_decoder_group_step("):src.index("def attach_conf_branch(")]
    pre = "wg_frame_w, wg_frame_feat = self._wgate_frame_pre("
    split = "wg_tok_w, wg_frame_w = wg_frame_w, None"
    tokmul = "state_mask = state_mask * wg_tok_w.to(state_mask.dtype)[:, :, None]"
    fconf = "wg_fconf_a = self._wgate_frame_conf(views, view_indices, res_group)"
    fgate = "state_mask = state_mask * wg_frame_w.to(state_mask.dtype)[:, None, None]"
    commit = "state_feat = new_state_feat * state_mask + state_feat * (1 - state_mask)"
    memcommit = "mem = new_mem * mem_mask + mem * (1 - mem_mask)"
    for s_ in (split, tokmul):
        assert body.count(s_) == 1, s_
    order = [body.index(pre), body.index(split), body.index('tg_q = [views[i].get("token_gate_q"'),
             body.index(fconf), body.index(tokmul), body.index(fgate), body.index(commit), body.index(memcommit)]
    assert order == sorted(order), order
    lines = body.splitlines()
    i_tm = next(i for i, l in enumerate(lines) if tokmul in l)
    assert "if wg_tok_w is not None" in lines[i_tm - 1], lines[i_tm - 1]
    assert not any("wg_tok_w" in l and "mem" in l.split("#")[0] for l in lines), "per-token weights must never touch the mem commit"
    assert not re.search(r"_wgate\w*\(.*state_mask", body)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"ok {name}")
    print(f"{n} tests passed")
