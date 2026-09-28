"""CPU bit-identity check of the wgate model patch on the REAL checkpoint (WRITE_GATE_VARIANTS.md, "Parity").

    cd src/CUT3R/src && OMP_NUM_THREADS=1 timeout 280 python dust3r/wgate/tests/parity_cpu.py [--ckpt PATH] [--T 3]

Loads ARCroco3DStereo.from_pretrained(CKPT_DEFAULT) in float32 on CPU, encodes T random 320x192
frames once (fixed seed) and rolls the decoder step (_forward_decoder_group_step, views_per_step=1)
over them under several arms that all start from the same encoder features / initial state:

    A   plain, no keys, model.wgate_record = [] (recording on)          -- reference
    A2  plain, no keys, no recording                                    -- determinism + "recording is inert"
    A3  plain, no keys, heads ATTACHED (forced to weight 1)             -- attaching alone changes nothing
    B   frame_gate_on=1 + mem_gate_on=1 with heads forced to weight 1   -- weight 1 == no gate, bit for bit
    C   mem_gate_const=1.0 + mem_gate_freeze_after=10                   -- constant 1 == no gate, bit for bit
    D   both heads forced to weight 0.5, keys on                        -- NEGATIVE control: gates are live;
                                                                           frame gate is STATE-ONLY (mem sees b only)
    E   mem_gate_const=0.0                                              -- NEGATIVE control: memory frozen
    F   frame_gate_const=0.5 (no head keys)                             -- V1 control path: state .5/.5, mem == A bit for bit
    G   D's keys + frame_gate_joint=1                                   -- joint mode: mem write a*b = .25 (pre-fix behaviour)
    then one plain t=0 step after the gated arms                        -- trace lists reset without keys

Assertions: torch.equal on camera_pose, pts3d_in_self_view, the returned state_feat and mem at
every step for A vs A2/A3/B/C; for D/E/F/G the analytically expected blends of A's tensors (allclose)
and a changed t=2 pose (the gates reach the outputs). Budget: ~9 s load + ~4.4 s per decoder step
single-threaded (28 steps at T=3 ~ 135 s CPU), under the 300 s login-node cap.

CPU note: the compiled curope kernel's CPU path takes float32 tokens only, but croco's Attention
casts q/k to float16 before RoPE (fine on CUDA). Every cuRoPE2D module is therefore swapped for the
pure-PyTorch RoPE2D of croco/models/pos_embed.py (its ImportError fallback) for ALL arms alike, so
the comparison stays like-for-like. The GPU numerics are untouched by this script.
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))  # .../CUT3R/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dust3r.heads  # noqa: F401,E402  (camera<->heads circular import)
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.wgate.heads import FrameConfHead, PoseSigmaHead  # noqa: E402

try:
    from train_unc_gate import CKPT_DEFAULT  # the finetuned augfull_lr1e5 checkpoint
except Exception:  # pragma: no cover - same path, spelled out
    CKPT_DEFAULT = (
        "/gpfs/scratch/etur59/koc821022/checkpoints_projects/cut3r_finetune_baselines/"
        "cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth"
    )

KEYS = ("camera_pose", "pts3d_in_self_view", "state_feat", "mem")


class TorchRoPE2D(nn.Module):
    """Pure-PyTorch RoPE2D (verbatim logic of croco/models/pos_embed.py's fallback class)."""

    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            self.cache[D, seq_len, device, dtype] = (freqs.cos(), freqs.sin())
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        cos = nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        D = tokens.size(3) // 2
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        return torch.cat((y, x), dim=-1)


def patch_rope_cpu(model):
    n = 0
    for m in model.modules():
        r = getattr(m, "rope", None)
        if r is not None and type(r).__name__ == "cuRoPE2D":
            m.rope = TorchRoPE2D(freq=r.base, F0=r.F0)
            n += 1
    return n


def make_views(T, H=192, W=320, seed=0):
    g = torch.Generator().manual_seed(seed)
    views = []
    for i in range(T):
        views.append(
            {
                "img": torch.randn(1, 3, H, W, generator=g),
                "ray_map": torch.full((1, 6, H, W), float("nan")),
                "true_shape": torch.tensor([[H, W]], dtype=torch.int32),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.eye(4)[None],
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
        )
    return views


def forced_frame_head(bias):
    """FrameConfHead with a constant output: bias=-40 -> weight 1.0, bias=+40 -> weight wmin (log_ref 0)."""
    h = FrameConfHead()
    nn.init.zeros_(h.out.weight)
    nn.init.constant_(h.out.bias, bias)
    return h


def forced_pose_head(model, bias):
    h = PoseSigmaHead(model.downstream_head.pose_head)
    nn.init.zeros_(h.out.weight)
    nn.init.constant_(h.out.bias, bias)
    return h


def run_arm(model, views, enc, T, keys=None):
    (feat, pos, shape), (init_state_feat, init_mem, state_feat, state_pos, mem) = enc
    vs = [dict(v) for v in views]  # shallow copies: keys never leak between arms
    for v in vs:
        for k, val in (keys or {}).items():
            v[k] = torch.tensor(float(val)).unsqueeze(0)
    outs = []
    with torch.no_grad():
        for t in range(T):
            res_group, (state_feat, mem) = model._forward_decoder_group_step(
                views=vs, view_indices=[t], feat_group=[feat[t]], pos_group=[pos[t]], shape_group=[shape[t]],
                init_state_feat=init_state_feat, init_mem=init_mem, state_feat=state_feat, state_pos=state_pos, mem=mem,
            )
            r = res_group[0]
            outs.append(
                {
                    "camera_pose": r["camera_pose"].detach().clone(),
                    "pts3d_in_self_view": r["pts3d_in_self_view"].detach().clone(),
                    "state_feat": state_feat.detach().clone(),
                    "mem": mem.detach().clone(),
                }
            )
    return outs


def compare(name, ref, arm, T):
    ok = True
    for t in range(T):
        for k in KEYS:
            a, b = ref[t][k], arm[t][k]
            eq = torch.equal(a, b)
            ok &= eq
            if not eq:
                print(f"  [{name}] t={t} {k}: NOT bit-identical, max|diff|={(a - b).abs().max().item():.3e}")
    print(f"[{'ok' if ok else 'FAIL'}] {name}: bit-identical on {KEYS} for t=0..{T - 1}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--T", type=int, default=3)
    args = ap.parse_args()
    T = args.T
    t0 = time.time()
    c0 = time.process_time()
    torch.manual_seed(0)
    model = ARCroco3DStereo.from_pretrained(args.ckpt).float().eval()
    if getattr(model, "pose_gru", None) is not None:
        model.pose_gru = None
    n = patch_rope_cpu(model)
    print(f"loaded ckpt in {time.time() - t0:.1f}s (cpu {time.process_time() - c0:.1f}s); cuRoPE2D->torch on {n} modules", flush=True)
    views = make_views(T)
    with torch.no_grad():
        enc = model._forward_encoder(views)
    print(f"encoded {T} views in {time.time() - t0:.1f}s", flush=True)

    failures = []

    def check(cond, msg):
        print(f"[{'ok' if cond else 'FAIL'}] {msg}", flush=True)
        if not cond:
            failures.append(msg)

    # --- A: plain reference with the recording hook on
    model.wgate_record = []
    A = run_arm(model, views, enc, T)
    rec = model.wgate_record
    model.wgate_record = None
    check(len(rec) == T and [r["t"] for r in rec] == list(range(T)), f"wgate_record: {len(rec)} entries, t = 0..{T - 1}")
    check(
        all(r["frame_feat"].shape == (1796,) and r["pose_feat"].shape == (768,) and r["camera_pose"].shape == (7,) for r in rec)
        and all(r["frame_feat"].dtype == torch.float32 and torch.isfinite(r["frame_feat"]).all() for r in rec),
        "wgate_record shapes (1796,), (768,), (7,), finite float32",
    )
    check(all(torch.equal(rec[t]["camera_pose"], A[t]["camera_pose"][0]) for t in range(T)), "wgate_record camera_pose == step output")
    print(f"  A done at {time.time() - t0:.1f}s", flush=True)

    # --- A2: plain again (determinism; recording inert)
    A2 = run_arm(model, views, enc, T)
    if not compare("A2 plain-vs-plain (determinism; recording inert)", A, A2, T):
        failures.append("A2")
    check(not hasattr(model, "_wgate_trace") or model._wgate_trace == [], "no trace written without keys")

    # --- heads forced to weight 1, attached
    model.attach_frame_gate(forced_frame_head(-40.0), log_ref=0.0, wmin=0.5)
    model.attach_mem_gate(forced_pose_head(model, -40.0), log_ref_t=0.0, log_ref_R=0.0, wmin=0.5)
    A3 = run_arm(model, views, enc, T)
    if not compare("A3 heads attached, no keys", A, A3, T):
        failures.append("A3")

    # --- B: gates on, weight 1
    B = run_arm(model, views, enc, T, keys={"frame_gate_on": 1.0, "mem_gate_on": 1.0, "frame_gate_wmin": 0.5, "mem_gate_wmin": 0.5})
    if not compare("B frame_gate_on + mem_gate_on, heads at weight 1", A, B, T):
        failures.append("B")
    check(model._wgate_trace == [(t, 1.0, 1.0) for t in range(T)], f"B trace == {[(t, 1.0, 1.0) for t in range(T)]}: {model._wgate_trace}")
    print(f"  B done at {time.time() - t0:.1f}s", flush=True)

    # --- C: constant 1 / freeze far away
    Cc = run_arm(model, views, enc, T, keys={"mem_gate_const": 1.0, "mem_gate_freeze_after": 10})
    if not compare("C mem_gate_const=1 + freeze_after=10", A, Cc, T):
        failures.append("C")
    check(model._wgate_trace == [(t, 1.0) for t in range(T)], f"C trace == {[(t, 1.0) for t in range(T)]}: {model._wgate_trace}")

    # --- D: negative control, both gates at 0.5
    model.attach_frame_gate(forced_frame_head(+40.0), log_ref=0.0, wmin=0.5)
    model.attach_mem_gate(forced_pose_head(model, +40.0), log_ref_t=0.0, log_ref_R=0.0, wmin=0.5)
    D = run_arm(model, views, enc, T, keys={"frame_gate_on": 1.0, "mem_gate_on": 1.0})
    check(model._wgate_trace == [(0, 1.0, 1.0)] + [(t, 0.5, 0.5) for t in range(1, T)], f"D trace: {model._wgate_trace}")
    check(all(torch.equal(D[0][k], A[0][k]) for k in KEYS), "D t=0 identical to A (a_0 = b_0 = 1)")
    check(
        torch.allclose(D[1]["state_feat"], 0.5 * (A[1]["state_feat"] + A[0]["state_feat"]), atol=1e-5),
        "D state after t=1 == 0.5*new + 0.5*old (frame gate a_1 = .5 on all 768 tokens)",
    )
    check(
        torch.allclose(D[1]["mem"], 0.5 * A[1]["mem"] + 0.5 * A[0]["mem"], atol=1e-5),
        "D mem after t=1 == b*update + (1-b)*old with b=.5 ONLY (frame gate is state-only: a does not touch the mem commit)",
    )
    if T > 2:
        check(not torch.equal(D[2]["camera_pose"], A[2]["camera_pose"]), "D camera_pose at t=2 differs from A (gates reach the outputs)")
    print(f"  D done at {time.time() - t0:.1f}s", flush=True)

    # --- F: the V1 control path (frame_gate_const, no head, no mem keys): state blended, mem untouched
    F = run_arm(model, views, enc, T, keys={"frame_gate_const": 0.5})
    check(model._wgate_trace == [(0, 1.0)] + [(t, 0.5) for t in range(1, T)] and model._wgate_trace_logvar == [], f"F trace: {model._wgate_trace}")
    check(all(torch.equal(F[0][k], A[0][k]) for k in KEYS), "F t=0 identical to A (a_0 = 1)")
    check(
        torch.allclose(F[1]["state_feat"], 0.5 * (A[1]["state_feat"] + A[0]["state_feat"]), atol=1e-5),
        "F state after t=1 == 0.5*new + 0.5*old (frame_gate_const = .5 on all 768 tokens)",
    )
    check(torch.equal(F[1]["mem"], A[1]["mem"]), "F mem after t=1 bit-identical to A (state-only gate never touches the retriever memory)")
    check(torch.equal(F[1]["camera_pose"], A[1]["camera_pose"]), "F t=1 pose untouched (state written after the pose is read)")
    if T > 2:
        check(not torch.equal(F[2]["camera_pose"], A[2]["camera_pose"]), "F camera_pose at t=2 differs from A (state gate reaches the outputs)")
    print(f"  F done at {time.time() - t0:.1f}s", flush=True)

    # --- G: joint mode (frame_gate_joint=1) reproduces the pre-2026-09-21 behaviour: a scales the mem commit too
    G = run_arm(model, views, enc, T, keys={"frame_gate_on": 1.0, "mem_gate_on": 1.0, "frame_gate_joint": 1.0})
    check(model._wgate_trace == [(0, 1.0, 1.0)] + [(t, 0.5, 0.5) for t in range(1, T)], f"G trace: {model._wgate_trace}")
    check(torch.allclose(G[1]["state_feat"], D[1]["state_feat"], atol=1e-5), "G state after t=1 == D (joint flag does not change the state write)")
    check(
        torch.allclose(G[1]["mem"], 0.25 * A[1]["mem"] + 0.75 * A[0]["mem"], atol=1e-5),
        "G mem after t=1 == a*b*update + (1-a*b)*old with a=b=.5 (joint mode: a_t multiplies update_mask like update_alpha)",
    )
    print(f"  G done at {time.time() - t0:.1f}s", flush=True)

    # --- E: memory frozen at its post-frame-0 value
    E = run_arm(model, views, enc, T, keys={"mem_gate_const": 0.0})
    check(all(torch.equal(E[t]["mem"], A[0]["mem"]) for t in range(T)), "E mem_gate_const=0: mem stays at its t=0 value, bit for bit")
    check(torch.equal(E[1]["state_feat"], A[1]["state_feat"]) and torch.equal(E[1]["camera_pose"], A[1]["camera_pose"]), "E t=1 state / pose untouched (mem written after the pose read)")
    if T > 2:
        check(not torch.equal(E[2]["camera_pose"], A[2]["camera_pose"]), "E camera_pose at t=2 differs from A (frozen retriever memory)")

    # --- one plain t=0 step after the gated arms: the trace lists are reset without any key
    run_arm(model, views[:1], enc, 1)
    check(model._wgate_trace == [] and model._wgate_trace_frame == [] and model._wgate_trace_mem == [] and model._wgate_trace_logvar == [],
          "plain t=0 step after gated arms resets the trace lists (unconditional reset)")

    model.attach_frame_gate(None)
    model.attach_mem_gate(None)
    print(f"total wall {time.time() - t0:.1f}s cpu {time.process_time() - c0:.1f}s", flush=True)
    if failures:
        print(f"PARITY FAILED: {failures}")
        sys.exit(1)
    print("PARITY OK: keys absent / present-with-weight-1 are bit-identical; negative controls behave as derived")


if __name__ == "__main__":
    main()
