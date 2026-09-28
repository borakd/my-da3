"""CPU check of VARIANT C on the REAL checkpoint: identity at wmin=1, and gradient through the rollout.

    cd src/CUT3R/src && OMP_NUM_THREADS=1 timeout 280 python dust3r/wgate/tests/parity_cpu_consequence.py

Companion of parity_cpu.py (whose model loading, CPU RoPE patch, views and reference arm are reused
verbatim -- see its docstring for the cuRoPE2D-on-CPU note). WRITE_GATE_VARIANTS.md, "Variant C".

    A   plain, no keys                                            -- reference (parity_cpu.run_arm)
    H   weight-mode GateWeightHead on BOTH gates, per-view
        frame_gate_wmin = mem_gate_wmin = 1, train=False          -- w == 1 exactly: bit-identical to A
    I   weight-mode GateWeightHead on BOTH gates, wmin .5,
        train=True, rollout WITH grad, loss = ||camera_pose||^2
        of the LAST frame only                                    -- the CONSEQUENCE signal: the only way
                                                                     the loss can reach the heads is through
                                                                     the writes of the EARLIER frames

Arm I is what the consequence trainer rests on and what no synthetic test can prove: the frozen
decoder is differentiable end to end in the gate weight, the gradient of a later frame's pose reaches
the head that gated an earlier write, and it is finite and non-zero for every head parameter. The
backbone is frozen (requires_grad False) exactly as in training, so only the two heads accumulate.

Budget measured on a login node: ~10 s load, ~12 s encode, ~13 s per no-grad arm, ~45 s for arm I
(forward with graph + backward), ~95 s wall / CPU in total -- well under the 300 s login-node cap.
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))  # .../CUT3R/src
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dust3r.heads  # noqa: F401,E402  (camera<->heads circular import)
import parity_cpu as P  # noqa: E402  (same directory: model loading + reference arm)
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.wgate.heads import GateWeightHead  # noqa: E402

T = 3


def make_heads(model, seed=0):
    """Two variant-C heads with a NON-degenerate last layer (a head that has already taken a step;
    at the contract's zeroed init only the last layer would receive a gradient)."""
    torch.manual_seed(seed)
    hf = GateWeightHead("frame", wmin=0.5, init_logit=2.0)
    hm = GateWeightHead("pose", pose_decoder=model.downstream_head.pose_head, wmin=0.5, init_logit=2.0)
    for h in (hf, hm):
        nn.init.normal_(h.out.weight, std=0.02)
    return hf, hm


def run_arm_grad(model, views, enc, keys):
    """parity_cpu.run_arm WITHOUT torch.no_grad(): returns the per-step result dicts with their graph."""
    (feat, pos, shape), (init_state_feat, init_mem, state_feat, state_pos, mem) = enc
    vs = [dict(v) for v in views]
    for v in vs:
        for k, val in keys.items():
            v[k] = torch.tensor(float(val)).unsqueeze(0)
    outs = []
    for t in range(T):
        res_group, (state_feat, mem) = model._forward_decoder_group_step(
            views=vs, view_indices=[t], feat_group=[feat[t]], pos_group=[pos[t]], shape_group=[shape[t]],
            init_state_feat=init_state_feat, init_mem=init_mem, state_feat=state_feat, state_pos=state_pos, mem=mem,
        )
        outs.append(res_group[0])
    return outs


def main():
    t0 = time.time()
    c0 = time.process_time()
    torch.manual_seed(0)
    model = ARCroco3DStereo.from_pretrained(P.CKPT_DEFAULT).float().eval()
    if getattr(model, "pose_gru", None) is not None:
        model.pose_gru = None
    P.patch_rope_cpu(model)
    for p_ in model.parameters():
        p_.requires_grad_(False)  # frozen backbone, exactly as the consequence trainer runs it
    views = P.make_views(T)
    with torch.no_grad():
        enc = model._forward_encoder(views)
    print(f"loaded + encoded in {time.time() - t0:.1f}s (cpu {time.process_time() - c0:.1f}s)", flush=True)

    failures = []

    def check(cond, msg):
        print(f"[{'ok' if cond else 'FAIL'}] {msg}", flush=True)
        if not cond:
            failures.append(msg)

    # --- A: plain reference -------------------------------------------------------------------
    A = P.run_arm(model, views, enc, T)
    print(f"  A done at {time.time() - t0:.1f}s", flush=True)

    # --- H: weight mode at wmin = 1 is bit-identical to plain ---------------------------------
    hf, hm = make_heads(model)
    model.attach_frame_gate(hf.checkpoint(), wmin=1.0, train=False)
    model.attach_mem_gate(hm.checkpoint(), wmin=1.0, train=False)
    check(model.frame_gate_mode == "weight" and model.mem_gate_mode == "weight", "H: both gates attached in weight mode")
    check(model.frame_gate_log_ref is None and model.mem_gate_log_ref is None, "H: no log_ref in weight mode")
    H = P.run_arm(model, views, enc, T, keys={
        "frame_gate_on": 1.0, "mem_gate_on": 1.0, "frame_gate_wmin": 1.0, "mem_gate_wmin": 1.0,
    })
    check(P.compare("H weight-mode heads at wmin=1", A, H, T), "H: bit-identical to plain")
    check(
        [e[1] for e in model._wgate_trace_logvar] == ["frame_u", "mem_u"] * (T - 1),
        f"H: trace tags are the weight-mode ones: {[e[1] for e in model._wgate_trace_logvar]}",
    )
    check(
        all(abs(a - 1.0) < 1e-7 and abs(b - 1.0) < 1e-7 for _, a, b in model._wgate_trace),
        f"H: every traced weight is 1.0: {model._wgate_trace}",
    )
    print(f"  H done at {time.time() - t0:.1f}s", flush=True)

    # --- I: train=True, the loss of the LAST frame reaches the EARLIER writes ------------------
    hf, hm = make_heads(model)
    model.attach_frame_gate(hf, wmin=0.5, train=True)
    model.attach_mem_gate(hm, wmin=0.5, train=True)
    check(hf.training and hm.training and all(p.requires_grad for p in list(hf.parameters()) + list(hm.parameters())),
          "I: heads are trainable (train mode, requires_grad)")
    outs = run_arm_grad(model, views, enc, {"frame_gate_on": 1.0, "mem_gate_on": 1.0})
    trace = list(model._wgate_trace)
    check(abs(trace[0][1] - 1.0) < 1e-7 and abs(trace[0][2] - 1.0) < 1e-7, f"I: a_0 = b_0 = 1 ({trace[0]})")
    check(all(0.5 < w < 1.0 for _, a_, b_ in trace[1:] for w in (a_, b_)), f"I: gated weights strictly inside (wmin, 1): {trace[1:]}")
    last = outs[-1]["camera_pose"]
    check(last.requires_grad, "I: the last frame's camera_pose carries a graph")
    loss = last.float().pow(2).sum()
    loss.backward()
    for name, head in (("frame", hf), ("mem", hm)):
        for n_, p_ in head.named_parameters():
            g = p_.grad
            check(g is not None and torch.isfinite(g).all() and float(g.abs().max()) > 0,
                  f"I: {name} head {n_} grad finite and non-zero "
                  f"({'None' if g is None else f'max|g|={float(g.abs().max()):.3e}'})")
    check(all(p.grad is None for p in model.parameters() if p is not None and not p.requires_grad),
          "I: the frozen backbone accumulated no gradient")
    print(f"  I done at {time.time() - t0:.1f}s", flush=True)

    # --- detaching restores the plain path ----------------------------------------------------
    model.attach_frame_gate(None)
    model.attach_mem_gate(None)
    A2 = P.run_arm(model, views, enc, T)
    check(P.compare("A2 after detaching the variant-C heads", A, A2, T), "detach: back to the plain path")

    print(f"total wall {time.time() - t0:.1f}s cpu {time.process_time() - c0:.1f}s", flush=True)
    if failures:
        print(f"VARIANT-C PARITY FAILED ({len(failures)}): " + "; ".join(failures), flush=True)
        return 1
    print("VARIANT-C PARITY OK: wmin=1 weight-mode is bit-identical to plain; the consequence "
          "gradient reaches both heads through the frozen rollout", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
