"""Tests for the `fg_conf` arm: the STATE write keyed per frame by the model's OWN self-view confidence.

    cd src/CUT3R/src && OMP_NUM_THREADS=1 timeout 280 python dust3r/wgate/tests/test_frame_conf_gate.py

Synthetic inputs only (no checkpoint, CPU, seconds):
  (1) the hook `ARCroco3DStereo._wgate_frame_conf` on a stub: inert without "frame_gate_src", a_0 = 1
      (returns None so the commit stays bit-exact), the CAUSAL median (never sees the current frame),
      the soft map (== conf_mem_weight), the hard rule (soft = 0: 1 at/above the median, wmin below),
      the (t, a) and (t, "frame_conf", [c]) trace entries, refusal to combine with the other frame-gate
      sources, and the history reset at view index 0;
  (2) a source grep of dust3r/model.py: the hook is called once, before the state-only frame-gate
      multiply, and the state / mem commit lines are unchanged.
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

import dust3r.heads  # noqa: F401,E402  (resolves the camera<->heads circular import)
from dust3r.wgate.heads import conf_mem_weight  # noqa: E402


def _stub():
    from dust3r.model import ARCroco3DStereo

    class Stub:
        _wgate_key = staticmethod(ARCroco3DStereo._wgate_key)
        _wgate_trace_append = ARCroco3DStereo._wgate_trace_append
        _wgate_frame_conf = ARCroco3DStereo._wgate_frame_conf

    m = Stub()
    m._wgate_trace, m._wgate_trace_frame, m._wgate_trace_mem, m._wgate_trace_logvar = [], [], [], []
    return m


def _views(T, **keys):
    vs = [{"img": torch.zeros(1)} for _ in range(T)]
    for v in vs:
        for k, val in keys.items():
            v[k] = torch.tensor(float(val)).unsqueeze(0)
    return vs


def _res(c):
    """res_group whose last entry has a (1, 2, 2) conf_self with mean log conf == c (c >= 0)."""
    return [{"conf_self": torch.full((1, 2, 2), math.exp(c))}]


def _lower_median(h):
    h = sorted(h)
    return h[(len(h) - 1) // 2]


def _run(m, cs, **keys):
    vs = _views(len(cs), frame_gate_src=1.0, **keys)
    out = []
    for t, c in enumerate(cs):
        a = m._wgate_frame_conf(vs, [t], _res(c))
        assert a is None or (a.shape == (1,) and a.dtype == torch.float32)
        out.append(None if a is None else float(a.reshape(-1)[0]))
    return out


def test_inert_without_key():
    from dust3r.model import ARCroco3DStereo

    class Bare:
        _wgate_key = staticmethod(ARCroco3DStereo._wgate_key)
        _wgate_trace_append = ARCroco3DStereo._wgate_trace_append
        _wgate_frame_conf = ARCroco3DStereo._wgate_frame_conf

    m = Bare()
    vs = _views(3)
    for t in range(3):
        assert m._wgate_frame_conf(vs, [t], _res(1.0)) is None
    assert not hasattr(m, "_frameconf_hist") and not hasattr(m, "_wgate_trace")


def test_soft_map_causal_and_a0():
    m = _stub()
    cs = [2.0, 1.0, 3.0, 0.5, 4.0]
    got = _run(m, cs)  # defaults soft .5, wmin .5
    assert got[0] is None  # a_0 = 1 -> identity, bit-exact commit
    for t in range(1, len(cs)):
        want = float(conf_mem_weight(cs[t], cs[:t], 0.5, 0.5))
        w2 = 0.5 + 0.5 / (1 + math.exp(-(cs[t] - _lower_median(cs[:t])) / 0.5))
        assert abs(want - w2) < 1e-6
        assert got[t] is not None and abs(got[t] - want) < 1e-6, (t, got[t], want)
    # trace: (t, a) for every t incl. (0, 1.0), and (t, "frame_conf", [c_t])
    assert [e[0] for e in m._wgate_trace_frame] == list(range(len(cs)))
    assert m._wgate_trace_frame[0][1] == 1.0
    assert [(e[0], e[1]) for e in m._wgate_trace_logvar] == [(t, "frame_conf") for t in range(len(cs))]
    for (t, _, v), c in zip(m._wgate_trace_logvar, cs):
        assert abs(v[0] - c) < 1e-5
    assert m._wgate_trace_mem == []  # the state gate never writes a mem entry


def test_hard_rule():
    m = _stub()
    cs = [1.0, 2.0, 0.5, 1.0, 3.0, 0.9]
    got = _run(m, cs, frame_gate_conf_soft=0.0, frame_gate_conf_wmin=0.3)
    want = [None]
    for t in range(1, len(cs)):
        want.append(None if cs[t] >= _lower_median(cs[:t]) else 0.3)  # 1.0 -> None (identity)
    assert [g is None for g in got] == [w is None for w in want], (got, want)
    assert all(g is None or abs(g - w) < 1e-6 for g, w in zip(got, want)), (got, want)


def test_overrides_and_bounds():
    m = _stub()
    cs = [0.0, 5.0, -0.0]
    got = _run(m, cs, frame_gate_conf_soft=2.0, frame_gate_conf_wmin=0.8)
    for t in range(1, len(cs)):
        a = 1.0 if got[t] is None else got[t]
        assert 0.8 - 1e-6 <= a <= 1.0 + 1e-6
        assert abs(a - float(conf_mem_weight(cs[t], cs[:t], 2.0, 0.8))) < 1e-6


def test_refuses_other_sources():
    for extra in ({"frame_gate_on": 1.0}, {"frame_gate_const": 0.5}, {"frame_gate_joint": 1.0}):
        m = _stub()
        vs = _views(2, frame_gate_src=1.0, **extra)
        try:
            m._wgate_frame_conf(vs, [0], _res(1.0))
        except RuntimeError:
            continue
        raise AssertionError(f"combination {extra} was not refused")


def test_history_resets_at_view_zero():
    m = _stub()
    _run(m, [5.0, 5.0, 5.0])
    assert m._frameconf_hist == [5.0, 5.0, 5.0] or all(abs(x - 5.0) < 1e-5 for x in m._frameconf_hist)
    got = _run(m, [1.0, 1.0])  # a new scene: frame 0 resets, frame 1 compares only to c_0 = 1.0
    assert got[0] is None and abs(got[1] - 0.75) < 1e-6, got
    assert len(m._frameconf_hist) == 2


def test_source_placement():
    src = open(os.path.join(_SRC, "dust3r", "model.py")).read()
    body = src[src.index("def _forward_decoder_group_step"):src.index("def _wgate_mem_conf")]
    lines = body.splitlines()
    calls = [i for i, l in enumerate(lines) if "self._wgate_frame_conf(" in l]
    assert len(calls) == 1, calls
    fmul = [i for i, l in enumerate(lines) if "state_mask = state_mask * wg_frame_w" in l]
    commit = [i for i, l in enumerate(lines) if l.strip() == "state_feat = new_state_feat * state_mask + state_feat * (1 - state_mask)"]
    memc = [i for i, l in enumerate(lines) if l.strip() == "mem = new_mem * mem_mask + mem * (1 - mem_mask)"]
    assert len(fmul) == 1 and len(commit) == 1 and len(memc) == 1, (fmul, commit, memc)
    assert calls[0] < fmul[0] < commit[0] < memc[0]
    assert not any(re.search(r"_wgate\w*\(.*state_mask", l) for l in lines)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"ok {name}")
    print(f"{n} tests passed")
