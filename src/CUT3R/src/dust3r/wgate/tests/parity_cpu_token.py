"""CPU check of the PER-TOKEN consequence gate (GateWeightHead kind="token") on the REAL checkpoint.

    cd src/CUT3R/src && OMP_NUM_THREADS=1 timeout 280 python dust3r/wgate/tests/parity_cpu_token.py

Reuses parity_cpu.py (model loading, CPU RoPE patch, views, reference arm) like parity_cpu_consequence.py.

    A   plain, no keys                                              -- reference
    J   token head attached from a trainer-style checkpoint dict,
        frame_gate_wmin = 1                                         -- w == 1 on every token: bit-identical to A
    K   token head at the CONTRACT init (last layer zero, bias 2),
        wmin .5                                                     -- every token .94
    Kf  per-frame head (arm 3) at the same init                     -- one .94 per frame: K must equal Kf
                                                                       bit-exactly (the per-token arm starts
                                                                       exactly where arm 3 starts)
    L   token head with a non-degenerate last layer, train=True,
        loss = ||camera_pose||^2 of the LAST frame                  -- the consequence gradient reaches the
                                                                       per-token head through the earlier
                                                                       writes; the weights differ across tokens
Also: the mem gate refuses a token checkpoint; detaching restores the plain path.
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

import dust3r.heads  # noqa: F401,E402
import parity_cpu as P  # noqa: E402
from parity_cpu_consequence import run_arm_grad, T  # noqa: E402
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.wgate.heads import GateWeightHead  # noqa: E402


def trainer_ck(head):
    """The dict train_wgate_consequence.save_ckpt writes for --token (kind + site metadata)."""
    return {"state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()}, "mode": "weight",
            "kind": head.kind, "site": "state", "wmin": head.wmin, "init_logit": head.init_logit}


def main():
    t0 = time.time()
    torch.manual_seed(0)
    model = ARCroco3DStereo.from_pretrained(P.CKPT_DEFAULT).float().eval()
    if getattr(model, "pose_gru", None) is not None:
        model.pose_gru = None
    P.patch_rope_cpu(model)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    views = P.make_views(T)
    with torch.no_grad():
        enc = model._forward_encoder(views)
    print(f"loaded + encoded in {time.time() - t0:.1f}s", flush=True)
    failures = []

    def check(cond, msg):
        print(f"[{'ok' if cond else 'FAIL'}] {msg}", flush=True)
        if not cond:
            failures.append(msg)

    A = P.run_arm(model, views, enc, T)

    # --- J: identity at wmin = 1 -------------------------------------------------------------
    ht = GateWeightHead("token", wmin=0.5, init_logit=2.0)
    nn.init.normal_(ht.out.weight, std=0.02)
    model.attach_frame_gate(trainer_ck(ht), wmin=1.0, train=False)
    check(model.frame_gate_mode == "weight" and model.frame_gate.kind == "token", "J: token checkpoint attached as kind='token', weight mode")
    J = P.run_arm(model, views, enc, T, keys={"frame_gate_on": 1.0, "frame_gate_wmin": 1.0})
    check(P.compare("J token head at wmin=1", A, J, T), "J: bit-identical to plain")
    check(all(abs(e[1] - 1.0) < 1e-7 for e in model._wgate_trace_frame), f"J: traced frame means all 1: {model._wgate_trace_frame}")
    check(len(model._wgate_trace_token) == T - 1 and all(abs(r[1][0] - 1) < 1e-7 and abs(r[1][1] - 1) < 1e-7
                                                          for r in model._wgate_trace_token), "J: token trace present, all 1")

    # --- K vs Kf: the per-token arm starts exactly where arm 3 starts --------------------------
    model.attach_frame_gate(GateWeightHead("token", wmin=0.5, init_logit=2.0), train=False)
    K = P.run_arm(model, views, enc, T, keys={"frame_gate_on": 1.0, "frame_gate_wmin": 0.5})
    tokK = list(model._wgate_trace_token)
    model.attach_frame_gate(GateWeightHead("frame", wmin=0.5, init_logit=2.0), train=False)
    Kf = P.run_arm(model, views, enc, T, keys={"frame_gate_on": 1.0, "frame_gate_wmin": 0.5})
    check(P.compare("K (token init) vs Kf (frame init)", Kf, K, T), "K == Kf bit-exactly at init")
    w0 = 0.5 + 0.5 * torch.sigmoid(torch.tensor(2.0)).item()
    check(all(abs(r[1][0] - w0) < 1e-6 and abs(r[1][1] - w0) < 1e-6 and r[1][2] < 1e-7 for r in tokK),
          f"K: every token at {w0:.6f} (sd 0): {tokK}")
    dK = max(float((A[t][k] - K[t][k]).abs().max()) for t in range(T) for k in P.KEYS)
    check(dK > 0, f"K differs from plain (gating is active): max|diff| {dK:.3e}")

    # --- L: gradient through the rollout into the per-token head -------------------------------
    ht = GateWeightHead("token", wmin=0.5, init_logit=2.0)
    nn.init.normal_(ht.out.weight, std=0.02)
    model.attach_frame_gate(ht, wmin=0.5, train=True)
    outs = run_arm_grad(model, views, enc, {"frame_gate_on": 1.0})
    check(abs(model._wgate_trace_frame[0][1] - 1.0) < 1e-7, f"L: a_0 = 1 ({model._wgate_trace_frame[0]})")
    sds = [r[1][2] for r in model._wgate_trace_token]
    check(all(s > 1e-5 for s in sds), f"L: per-token weights differ across tokens (sd per frame {sds})")
    check(all(0.5 < r[1][3] and r[1][4] < 1.0 for r in model._wgate_trace_token), "L: weights strictly inside (wmin, 1)")
    loss = outs[-1]["camera_pose"].float().pow(2).sum()
    loss.backward()
    for n_, p_ in ht.named_parameters():
        g = p_.grad
        check(g is not None and torch.isfinite(g).all() and float(g.abs().max()) > 0,
              f"L: token head {n_} grad finite and non-zero ({'None' if g is None else f'max|g|={float(g.abs().max()):.3e}'})")
    check(all(p.grad is None for p in model.parameters() if not p.requires_grad), "L: frozen backbone has no grad")

    # --- mem gate refuses a token checkpoint; detach restores plain -----------------------------
    try:
        model.attach_mem_gate(trainer_ck(GateWeightHead("token")))
        check(False, "attach_mem_gate refused a token checkpoint")
    except AssertionError:
        check(True, "attach_mem_gate refused a token checkpoint")
    model.attach_frame_gate(None)
    model.attach_mem_gate(None)
    A2 = P.run_arm(model, views, enc, T)
    check(P.compare("A2 after detach", A, A2, T), "detach: back to the plain path")
    print(f"total wall {time.time() - t0:.1f}s", flush=True)
    if failures:
        print(f"TOKEN-GATE PARITY FAILED ({len(failures)}): " + "; ".join(failures), flush=True)
        return 1
    print("TOKEN-GATE PARITY OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
