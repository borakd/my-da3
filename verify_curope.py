#!/usr/bin/env python
"""Verify the compiled CroCo CUDA RoPE2D kernel is present, correct, and usable
by the layouts CUT3R's attention blocks actually produce.

Run on a GPU node (see build_curope.sbatch), from the repo root:

    python verify_curope.py

Why this is not just "does it import": the kernel does NOT accept an arbitrary
contiguous tensor. `cuRoPE2D.forward` (curope2d.py:39) passes
`tokens.transpose(1, 2)` to the extension, and kernels.cu:91 asserts

    tokens.stride(3) == 1 && tokens.stride(2) == D

on that TRANSPOSED (B,N,H,D) view. A plain contiguous (B,H,N,D) tensor has
stride(2) == N*D there and is REJECTED with "tokens are not contiguous".

So whether enabling this kernel helps or crashes training depends entirely on
the strides the call sites hand it — and blocks.py casts with `.to(torch.float16)`
immediately before calling rope, which can silently re-lay-out the tensor.
STAGE C exercises the real modules to settle that empirically.

Exits non-zero on any failure.
"""

import os
import sys

import torch

# Self-locating: resolve the checkout from THIS file, never a hardcoded path, and
# fail loudly if the layout is wrong (a bad sys.path.insert would silently no-op).
WORKTREE = os.path.dirname(os.path.abspath(__file__))
CROCO = os.path.join(WORKTREE, "src", "CUT3R", "src", "croco")
if not os.path.isdir(CROCO):
    sys.exit(f"FAIL: not a my-da3 checkout (missing {CROCO})")
sys.path.insert(0, CROCO)

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def banner(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# --------------------------------------------------------------------------
# Reference implementation, copied verbatim from the pure-PyTorch fallback in
# croco/models/pos_embed.py so we can diff the kernel against it. We cannot
# import that class: it lives inside an `except ImportError` block, and once the
# kernel builds, the import succeeds and the fallback is never defined.
# --------------------------------------------------------------------------
class RefRoPE2D(torch.nn.Module):
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
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        D = tokens.size(3) // 2
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        return torch.cat((y, x), dim=-1)


def make_positions(B, grid, device):
    """(B, grid*grid, 2) int64 y/x positions, the layout croco's encoder uses."""
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    pos = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)  # (N, 2)
    return pos.unsqueeze(0).repeat(B, 1, 1).contiguous().to(device)


def main():
    banner("STAGE A -- the kernel is built and is what croco resolves to")

    if not torch.cuda.is_available():
        sys.exit("FAIL: no CUDA device visible -- run this on a GPU node")
    print(f"device: {torch.cuda.get_device_name(0)}")

    try:
        from models.curope import cuRoPE2D
    except Exception as e:  # noqa: BLE001
        check("models.curope imports", False, repr(e))
        return 1
    check("models.curope imports", True)

    import models.pos_embed as pe

    check(
        "pos_embed.RoPE2D IS cuRoPE2D (pure-python fallback no longer used)",
        pe.RoPE2D is cuRoPE2D,
        f"RoPE2D={pe.RoPE2D.__name__}",
    )

    banner("STAGE B -- numerical equivalence against the pure-PyTorch reference")

    B, H, N, Dh, grid = 2, 16, 64, 64, 8  # CUT3R enc: 1024 dim / 16 heads = 64
    dev = "cuda"
    pos = make_positions(B, grid, dev)

    # Build a tensor the kernel accepts: contiguous as (B,N,H,D), viewed as
    # (B,H,N,D). Then the internal transpose(1,2) hands back contiguous memory.
    # NB: do NOT .clone() the transposed view -- clone() materialises contiguous
    # in the NEW shape, which is exactly the layout the kernel rejects.
    base = torch.randn(B, N, H, Dh, device=dev, dtype=torch.float32)
    tok_cu = base.transpose(1, 2)  # (B,H,N,Dh), kernel-compatible strides
    tok_ref = base.clone().transpose(1, 2)
    assert tok_cu.shape == (B, H, N, Dh), tok_cu.shape
    assert tok_cu.transpose(1, 2).is_contiguous(), "test tensor has the wrong layout"

    ref = RefRoPE2D(freq=100.0)
    expected = ref(tok_ref, pos)

    rope = cuRoPE2D(freq=100.0)
    got = rope(tok_cu, pos)  # NOTE: mutates in place

    err = (got.float() - expected.float()).abs().max().item()
    check("cuRoPE2D matches pure-PyTorch RoPE2D (fp32)", err < 1e-4, f"max|diff| = {err:.3e}")

    # The kernel mutates in place and returns the same storage.
    check("cuRoPE2D is in-place (returns the input tensor)", got.data_ptr() == tok_cu.data_ptr())

    banner("STAGE C -- the REAL call sites (this is the crash test)")

    from models.blocks import Attention, CrossAttention

    dim = H * Dh
    x = torch.randn(B, N, dim, device=dev, dtype=torch.float32)

    # Self-attention: qkv is reshape(B,N,3,H,Dh).transpose(1,3), so q carries a
    # stride-3 gap. If the `.to(float16)` at blocks.py:125 re-lays it out to plain
    # contiguous, the kernel rejects it and training dies inside every encoder block.
    attn = Attention(dim, rope=cuRoPE2D(freq=100.0), num_heads=H).to(dev)
    try:
        out = attn(x, pos)
        ok, detail = torch.isfinite(out).all().item(), f"out {tuple(out.shape)}"
    except Exception as e:  # noqa: BLE001
        ok, detail = False, repr(e)
    check("blocks.Attention forward with cuRoPE2D (self-attn qkv layout)", ok, detail)

    # Cross-attention: reshape(B,N,H,Dh).permute(0,2,1,3) -- dense, so `.to()`
    # should preserve the stride permutation.
    xattn = CrossAttention(dim, rope=cuRoPE2D(freq=100.0), num_heads=H).to(dev)
    try:
        out = xattn(x, x, x, pos, pos)
        ok, detail = torch.isfinite(out).all().item(), f"out {tuple(out.shape)}"
    except Exception as e:  # noqa: BLE001
        ok, detail = False, repr(e)
    check("blocks.CrossAttention forward with cuRoPE2D (cross-attn qkv layout)", ok, detail)

    banner("STAGE D -- kernel vs fallback agree through a whole attention block")

    # Same weights, same input, rope swapped: the block output must not depend on
    # which RoPE implementation ran. This is what actually protects the science.
    torch.manual_seed(0)
    a_cu = Attention(dim, rope=cuRoPE2D(freq=100.0), num_heads=H).to(dev).eval()
    a_ref = Attention(dim, rope=RefRoPE2D(freq=100.0), num_heads=H).to(dev).eval()
    a_ref.load_state_dict(a_cu.state_dict())
    try:
        with torch.no_grad():
            o_cu = a_cu(x, pos)
            o_ref = a_ref(x, pos)
        d = (o_cu.float() - o_ref.float()).abs().max().item()
        # blocks.py casts q/k to fp16 before rope in BOTH paths, so the tolerance
        # is set by fp16 rounding, not by the kernel.
        ok, detail = d < 5e-2, f"max|diff| = {d:.3e}"
    except Exception as e:  # noqa: BLE001
        ok, detail = False, repr(e)
    check("Attention output identical with kernel vs fallback RoPE", ok, detail)

    banner("SUMMARY")
    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    for name, ok, detail in CHECKS:
        if not ok:
            print(f"  FAILED: {name}  --  {detail}")
    print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
