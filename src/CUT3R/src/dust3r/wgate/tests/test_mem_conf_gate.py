"""Tests for the `mg_conf` arm: the pose-memory write keyed by the model's OWN self-view confidence.

    cd src/CUT3R/src && OMP_NUM_THREADS=1 timeout 280 python dust3r/wgate/tests/test_mem_conf_gate.py

Covers, on synthetic inputs only (no checkpoint, CPU, seconds):
  (1) the pure map `dust3r.wgate.heads.conf_mem_weight`:
        b_t = wmin + (1 - wmin) * sigmoid((c_t - median(history)) / soft)
      -- hand-computed values, the range [wmin, 1], monotonicity in c_t, the median convention,
      the temperature, and the b_0 == 1 rule (empty history -> exactly 1.0, not wmin + (1-wmin)/2);
  (2) the model hook `ARCroco3DStereo._wgate_mem_conf` on a stub: bit-exact identity without the
      "mem_gate_src" key, b_0 = 1 (returns None so the commit stays bit-exact), the CAUSAL history
      (the median never sees the current frame), the (t, b) trace entries, the soft / wmin defaults
      and their per-view overrides, and the reset of the history at view index 0;
  (3) a source grep of dust3r/model.py: the mem commit is wrapped exactly once, through `mem_mask`,
      and the state commit line is untouched.
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


def _ref(c, hist, soft, wmin):
    """The formula, written out independently of the implementation."""
    if not hist:
        return 1.0
    h = sorted(hist)
    n = len(h)
    med = h[(n - 1) // 2]  # torch.median = LOWER median on an even-length history
    return wmin + (1.0 - wmin) / (1.0 + math.exp(-(c - med) / soft))


# ----------------------------------------------------------------------------- (1) the pure map
def test_conf_mem_weight_b0_is_exactly_one():
    for wmin in (0.0, 0.5, 0.9):
        for soft in (0.1, 0.5, 3.0):
            for hist in (None, [], ()):
                b = conf_mem_weight(1.234, hist, soft, wmin)
                assert b.dtype == torch.float32 and b.shape == ()
                assert float(b) == 1.0, (float(b), wmin, soft, hist)
    # ... and NOT the "c equals the median" value, which is what an empty history would give
    # if the caller had seeded it with c_0 before the call.
    assert float(conf_mem_weight(1.234, [1.234], 0.5, 0.5)) == 0.75


def test_conf_mem_weight_formula():
    cases = [
        # (c_t, history, soft, wmin)
        (1.0, [1.0], 0.5, 0.5),                    # at the median -> (1 + wmin) / 2
        (1.5, [1.0], 0.5, 0.5),                    # +1 sigma of the temperature
        (0.5, [1.0], 0.5, 0.5),
        (0.3, [0.1, 0.2, 0.9, 1.7], 0.5, 0.5),     # even history -> lower median (0.2)
        (0.3, [0.1, 0.2, 0.9], 0.25, 0.2),         # odd history, other soft / wmin
        (-2.0, [0.0, 1.0, 2.0, 3.0, 4.0], 1.0, 0.0),
        (7.0, [0.0, 1.0], 2.0, 0.75),
    ]
    for c, hist, soft, wmin in cases:
        got = float(conf_mem_weight(c, hist, soft, wmin))
        want = _ref(c, hist, soft, wmin)
        assert abs(got - want) < 1e-6, (c, hist, soft, wmin, got, want)
        assert wmin - 1e-6 <= got <= 1.0 + 1e-6, (got, wmin)
    # the exact hand-computed anchors
    assert abs(float(conf_mem_weight(1.0, [1.0], 0.5, 0.5)) - 0.75) < 1e-7
    assert abs(float(conf_mem_weight(1.5, [1.0], 0.5, 0.5)) - (0.5 + 0.5 * 0.7310585786)) < 1e-6
    assert abs(float(conf_mem_weight(0.5, [1.0], 0.5, 0.5)) - (0.5 + 0.5 * 0.2689414214)) < 1e-6


def test_conf_mem_weight_monotone_bounded_and_temperature():
    hist = [0.0, 1.0, 2.0]  # median 1.0
    prev = -1.0
    for c in [-50.0, -3.0, -1.0, 0.0, 0.5, 1.0, 1.5, 3.0, 50.0]:
        b = float(conf_mem_weight(c, hist, 0.5, 0.5))
        assert b >= prev - 1e-9, (c, b, prev)  # rises with confidence
        assert 0.5 - 1e-6 <= b <= 1.0 + 1e-6
        prev = b
    # extremes saturate at the floor / the ceiling, never outside
    assert abs(float(conf_mem_weight(-1e4, hist, 0.5, 0.5)) - 0.5) < 1e-6
    assert abs(float(conf_mem_weight(+1e4, hist, 0.5, 0.5)) - 1.0) < 1e-6
    # a colder temperature is sharper about the same deviation
    soft_small = float(conf_mem_weight(1.5, hist, 0.05, 0.5))
    soft_big = float(conf_mem_weight(1.5, hist, 5.0, 0.5))
    assert soft_small > soft_big > 0.5
    assert abs(soft_small - 1.0) < 1e-4 and abs(soft_big - 0.75) < 0.02
    # wmin = 1 pins the gate to the identity, wmin = 0 gives the raw sigmoid
    assert float(conf_mem_weight(-3.0, hist, 0.5, 1.0)) == 1.0
    assert abs(float(conf_mem_weight(1.0, hist, 0.5, 0.0)) - 0.5) < 1e-7


def test_conf_mem_weight_accepts_tensor_and_rejects_bad_args():
    assert abs(float(conf_mem_weight(torch.tensor([1.5]), [1.0], 0.5, 0.5))
               - float(conf_mem_weight(1.5, [1.0], 0.5, 0.5))) < 1e-7
    assert abs(float(conf_mem_weight(torch.tensor(1.5), (1.0,), 0.5, 0.5))
               - float(conf_mem_weight(1.5, [1.0], 0.5, 0.5))) < 1e-7
    for bad in (dict(soft=0.0), dict(soft=-1.0), dict(wmin=-0.1), dict(wmin=1.5)):
        kw = dict(soft=0.5, wmin=0.5)
        kw.update(bad)
        try:
            conf_mem_weight(1.0, [1.0], **kw)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"conf_mem_weight accepted {bad}")
    try:
        conf_mem_weight(torch.zeros(3), [1.0], 0.5, 0.5)
    except AssertionError:
        pass
    else:
        raise AssertionError("conf_mem_weight accepted a non-scalar c_t")


# ----------------------------------------------------------------------------- (2) the model hook
def _stub():
    from dust3r.model import ARCroco3DStereo

    class Stub:
        _wgate_key = staticmethod(ARCroco3DStereo._wgate_key)
        _wgate_trace_append = ARCroco3DStereo._wgate_trace_append
        _wgate_mem_conf = ARCroco3DStereo._wgate_mem_conf

    return Stub()


def _views(T, **keys):
    vs = [{"img": torch.zeros(1)} for _ in range(T)]
    for v in vs:
        for k, val in keys.items():
            v[k] = torch.tensor(float(val)).unsqueeze(0)
    return vs


def _res(c):
    """res_group whose last entry has a (1, 2, 2) conf_self with mean log conf == c (c >= 0)."""
    return [{"conf_self": torch.full((1, 2, 2), math.exp(c))}]


def test_hook_inert_without_key():
    m = _stub()
    vs = _views(3)
    for t in range(3):
        assert m._wgate_mem_conf(vs, [t], _res(1.0)) is None
    assert not hasattr(m, "_memconf_hist")  # no state touched at all
    assert not hasattr(m, "_wgate_trace")   # no trace written


def test_hook_b0_is_one_and_history_is_causal():
    m = _stub()
    m._wgate_trace = m._wgate_trace_frame = m._wgate_trace_mem = m._wgate_trace_logvar = []
    m._wgate_trace, m._wgate_trace_frame = [], []
    m._wgate_trace_mem, m._wgate_trace_logvar = [], []
    cs = [2.0, 1.0, 3.0, 0.5, 4.0]
    vs = _views(len(cs), mem_gate_src=1.0)
    got = []
    for t, c in enumerate(cs):
        b = m._wgate_mem_conf(vs, [t], _res(c))
        got.append(None if b is None else float(b.reshape(-1)[0]))
        assert b is None or (b.shape == (1,) and b.dtype == torch.float32)
    # frame 0: exactly 1 AND the hook returns None so the commit stays bit-exact
    assert got[0] is None, got
    # frames t > 0: the CAUSAL median over c_0..c_{t-1} (current frame excluded)
    for t in range(1, len(cs)):
        want = _ref(cs[t], cs[:t], 0.5, 0.5)
        assert abs(got[t] - want) < 1e-5, (t, got[t], want)
    # the history holds every frame seen so far INCLUDING the current one, appended after use
    assert [round(x, 5) for x in m._memconf_hist] == [round(c, 5) for c in cs]
    # the trace carries (t, b) for every frame, b_0 = 1.0
    assert [t for t, _ in m._wgate_trace_mem] == list(range(len(cs)))
    assert m._wgate_trace_mem[0][1] == 1.0
    for t in range(1, len(cs)):
        assert abs(m._wgate_trace_mem[t][1] - got[t]) < 1e-6
    assert m._wgate_trace == m._wgate_trace_mem and m._wgate_trace_logvar == []


def test_hook_soft_wmin_defaults_and_overrides():
    c = [0.0, 5.0]  # frame 1 is much MORE confident than the frame-0 median
    for keys, soft, wmin in [
        (dict(mem_gate_src=1.0), 0.5, 0.5),                                            # defaults
        (dict(mem_gate_src=1.0, mem_gate_conf_soft=2.0), 2.0, 0.5),
        (dict(mem_gate_src=1.0, mem_gate_conf_wmin=0.9), 0.5, 0.9),
        (dict(mem_gate_src=1.0, mem_gate_conf_soft=0.25, mem_gate_conf_wmin=0.2), 0.25, 0.2),
    ]:
        m = _stub()
        m._wgate_trace, m._wgate_trace_frame = [], []
        m._wgate_trace_mem, m._wgate_trace_logvar = [], []
        vs = _views(2, **keys)
        assert m._wgate_mem_conf(vs, [0], _res(c[0])) is None
        b = m._wgate_mem_conf(vs, [1], _res(c[1]))
        want = _ref(c[1], c[:1], soft, wmin)
        # b == 1 in float32 is returned as None (the bit-exact identity fast path)
        got = 1.0 if b is None else float(b.reshape(-1)[0])
        assert abs(got - want) < 1e-5, (keys, got, want)
    # a LESS confident frame is damped towards the floor, a MORE confident one towards 1
    m = _stub()
    m._wgate_trace, m._wgate_trace_frame = [], []
    m._wgate_trace_mem, m._wgate_trace_logvar = [], []
    vs = _views(3, mem_gate_src=1.0, mem_gate_conf_soft=0.5, mem_gate_conf_wmin=0.5)
    m._wgate_mem_conf(vs, [0], _res(2.0))
    down = float(m._wgate_mem_conf(vs, [1], _res(0.0)).reshape(-1)[0])
    m2 = _stub()
    m2._wgate_trace, m2._wgate_trace_frame = [], []
    m2._wgate_trace_mem, m2._wgate_trace_logvar = [], []
    m2._wgate_mem_conf(vs, [0], _res(2.0))
    up = float(m2._wgate_mem_conf(vs, [1], _res(4.0)).reshape(-1)[0])
    assert 0.5 <= down < 0.55 < 0.95 < up <= 1.0, (down, up)


def test_hook_history_resets_at_view_zero():
    m = _stub()
    m._wgate_trace, m._wgate_trace_frame = [], []
    m._wgate_trace_mem, m._wgate_trace_logvar = [], []
    vs = _views(2, mem_gate_src=1.0)
    m._wgate_mem_conf(vs, [0], _res(9.0))
    m._wgate_mem_conf(vs, [1], _res(1.0))
    assert len(m._memconf_hist) == 2
    # a new scene starts at t = 0: the previous scene's confidences must not be in the median
    m._wgate_mem_conf(vs, [0], _res(0.0))
    assert m._memconf_hist == [0.0]
    b = m._wgate_mem_conf(vs, [1], _res(1.0))
    assert abs(float(b.reshape(-1)[0]) - _ref(1.0, [0.0], 0.5, 0.5)) < 1e-5
    # an undefined source code is refused
    bad = _views(2, mem_gate_src=2.0)
    try:
        m._wgate_mem_conf(bad, [1], _res(1.0))
    except AssertionError:
        pass
    else:
        raise AssertionError("mem_gate_src = 2 was accepted")


def test_hook_batched_shape():
    m = _stub()
    m._wgate_trace, m._wgate_trace_frame = [], []
    m._wgate_trace_mem, m._wgate_trace_logvar = [], []
    vs = _views(2, mem_gate_src=1.0)
    res0 = [{"conf_self": torch.full((4, 2, 2), math.exp(2.0))}]
    res1 = [{"conf_self": torch.full((4, 2, 2), math.exp(0.0))}]
    assert m._wgate_mem_conf(vs, [0], res0) is None
    b = m._wgate_mem_conf(vs, [1], res1)
    assert b.shape == (4,) and len(set(b.tolist())) == 1  # one scalar c per step, broadcast over B


# ----------------------------------------------------------------------------- (3) source grep
def test_model_mem_commit_wrapped_once():
    src = os.path.join(_SRC, "dust3r", "model.py")
    body = open(src).read()
    lines = body.splitlines()
    grouped = body[body.index("def _forward_decoder_group_step"):body.index("def _wgate_mem_conf")]
    g = grouped.splitlines()
    # the group step's mem commit now goes through mem_mask, exactly once, and mem_mask is
    # update_mask itself unless the hook returned a weight
    commit = [l for l in g if l.strip() == "mem = new_mem * mem_mask + mem * (1 - mem_mask)"]
    assert len(commit) == 1, commit
    assert "mem = new_mem * update_mask + mem * (1 - update_mask)" not in grouped
    assert len([l for l in g if l.strip() == "mem_mask = update_mask"]) == 1
    assert len([l for l in g if "_wgate_mem_conf(" in l]) == 1
    # the state commit is untouched by this path
    st = [l for l in g if l.strip() == "state_feat = new_state_feat * state_mask + state_feat * (1 - state_mask)"]
    assert len(st) == 1, st
    assert not re.search(r"_wgate_mem_conf\(.*state", grouped)
    # the OTHER two rollout paths (non-grouped) keep the original commit line
    assert body.count("mem = new_mem * update_mask + mem * (1 - update_mask)  # then update local state") == 2
    # the hook call sits between the state commit and the commit line it guards
    i_st = next(i for i, l in enumerate(lines) if l.strip() == st[0].strip())
    i_hook = next(i for i, l in enumerate(lines) if "_wgate_mem_conf(" in l and "self." in l)
    i_commit = next(i for i, l in enumerate(lines) if l.strip() == commit[0].strip())
    assert i_st < i_hook < i_commit, (i_st, i_hook, i_commit)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}", flush=True)
    print(f"all {len(tests)} tests passed")
