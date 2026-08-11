#!/usr/bin/env python
"""Falsifier for the ddp_grad_sync fix (croco/utils/misc.py:all_reduce_grads_).

Runs 2 gloo ranks on CPU — no GPUs, no allocation, runs on a login node in
seconds. Feeds each rank DIFFERENT data so the local gradients genuinely
disagree, then checks three things:

  1. NEGATIVE CONTROL — the bug is real. Forwarding through the UNWRAPPED
     module (what dust3r/inference.py:118 does) leaves each rank's gradient
     purely local: rank r sees its own gradient, not the mean. This is the
     baseline every existing checkpoint was trained under.
  2. POSITIVE CONTROL — the wrapper works. Forwarding through the DDP wrapper
     all-reduces, proving the collective itself is healthy.
  3. THE FIX — all_reduce_grads_ on the unwrapped path reproduces (2) exactly,
     including across mixed shapes/dtypes and across a bucket boundary.

Usage:
    python verify_ddp_grad_sync.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/CUT3R/src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/CUT3R/src/croco"))

from croco.utils.misc import all_reduce_grads_  # noqa: E402

WORLD = 2
FAIL = []


def check(rank, name, cond, detail=""):
    if rank != 0:
        return
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def run(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29531"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)

    model = nn.Linear(4, 2, bias=False)
    ddp = nn.parallel.DistributedDataParallel(model)
    base = ddp.module  # the accelerator.unwrap_model(model) pattern
    x = torch.full((1, 4), float(rank + 1))  # rank 0 -> 1.0, rank 1 -> 2.0
    # d(sum(Wx))/dW = x broadcast over rows, so the local grad IS rank+1 and the
    # correct all-reduced mean is 1.5. Fully analytic — no tolerance games.
    local, mean = float(rank + 1), 1.5

    if rank == 0:
        print("\n1. NEGATIVE CONTROL — unwrapped forward (the current baseline)")
    ddp.zero_grad()
    base(x).sum().backward()
    g = model.weight.grad.clone()
    check(rank, "unwrapped forward does NOT all-reduce",
          torch.allclose(g, torch.full_like(g, local)),
          f"rank0 grad={g[0, 0]:.3f}, expected local {local:.3f}")

    if rank == 0:
        print("\n2. POSITIVE CONTROL — DDP wrapper forward")
    ddp.zero_grad()
    ddp(x).sum().backward()
    g_ddp = model.weight.grad.clone()
    check(rank, "wrapper forward DOES all-reduce",
          torch.allclose(g_ddp, torch.full_like(g_ddp, mean)),
          f"rank0 grad={g_ddp[0, 0]:.3f}, expected mean {mean:.3f}")

    if rank == 0:
        print("\n3. THE FIX — all_reduce_grads_ on the unwrapped path")
    ddp.zero_grad()
    base(x).sum().backward()
    n = all_reduce_grads_(list(model.parameters()))
    g_fix = model.weight.grad.clone()
    check(rank, "fix reproduces the wrapper's gradient exactly",
          torch.equal(g_fix, g_ddp),
          f"fix={g_fix[0, 0]:.6f} vs wrapper={g_ddp[0, 0]:.6f}")
    check(rank, "fix reports the expected tensor count", n == 1, f"synced {n} tensor(s)")

    if rank == 0:
        print("\n4. BUCKETING — many tensors, mixed shapes, forced bucket splits")
    big = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 32), nn.Linear(32, 8))
    for p in big.parameters():
        p.grad = torch.full_like(p, float(rank + 1))
    expected = [torch.full_like(p, mean) for p in big.parameters()]
    # 1 KiB buckets force several flush cycles over these tensors.
    n = all_reduce_grads_(list(big.parameters()), bucket_bytes=1024)
    ok = all(torch.allclose(p.grad, e) for p, e in zip(big.parameters(), expected))
    check(rank, "all tensors averaged across bucket boundaries", ok,
          f"synced {n} tensors in 1 KiB buckets")

    if rank == 0:
        print("\n5. DIVERGENT UNUSED PARAMS — the deadlock case")
    # The graded TBPTT chunks are data-dependent and the Accelerator is built
    # with find_unused_parameters=True, so one rank can have a gradient where
    # another has None. A filter on `p.grad is not None` buckets differently
    # per rank and MISMATCHES the collective — a 32-rank hang. Reproduce that
    # asymmetry deliberately: only rank 0 gives `skewed` a gradient.
    skewed = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    for i, p in enumerate(skewed.parameters()):
        if rank == 0 or i != 0:
            p.grad = torch.full_like(p, float(rank + 1))
    n = all_reduce_grads_(list(skewed.parameters()))
    first = list(skewed.parameters())[0]
    # rank0 contributed 1.0, rank1 contributed a materialized 0.0 -> mean 0.5
    check(rank, "asymmetric unused params do not deadlock or corrupt",
          torch.allclose(first.grad, torch.full_like(first, 0.5)),
          f"grad={first.grad.flatten()[0]:.3f}, expected 0.5 (1.0 and 0.0 averaged)")
    check(rank, "all params participate regardless of local grad presence",
          n == len(list(skewed.parameters())), f"synced {n} of 4")

    if rank == 0:
        print("\n6. ORDER DETERMINISM — bucket sequence must match across ranks")
    # Ranks agree on the collective sequence only if bucketing is order-stable.
    mixed = [nn.Parameter(torch.randn(s)) for s in (300, 5, 700, 2, 900)]
    for p in mixed:
        p.grad = torch.full_like(p, float(rank + 1))
    n = all_reduce_grads_(mixed, bucket_bytes=4096)
    ok = all(torch.allclose(p.grad, torch.full_like(p, mean)) for p in mixed)
    check(rank, "ragged tensor sizes bucket identically on every rank", ok,
          f"synced {n} tensors")

    dist.barrier()
    if rank == 0:
        print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"FAILED: {FAIL}"))
    dist.destroy_process_group()
    if rank == 0 and FAIL:
        sys.exit(1)


if __name__ == "__main__":
    mp.spawn(run, args=(WORLD,), nprocs=WORLD, join=True)
