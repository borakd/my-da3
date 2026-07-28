"""Step-4 verification: trainer wiring for the PoseGRU (param groups + lr).

Imports the REAL split_pose_gru_param_groups from train_cut3r_baseline and the
REAL get_parameter_groups / adjust_learning_rate from croco misc, and checks
the whole lr path on a dummy model carrying a real PoseGRU. CPU-only, tiny.

Run from the worktree root:
    PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src" \
        conda run -n cuteanything python verify_pose_gru_trainer.py
"""
from types import SimpleNamespace

import torch
import torch.nn as nn

from accelerate import PartialState

PartialState()  # croco's accelerate logger needs initialized state

import dust3r.heads  # noqa: F401  import-order gotcha
from dust3r.model import PoseGRU
from train_cut3r_baseline import split_pose_gru_param_groups
import croco.utils.misc as misc

PASS = []


def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name)
    print(f"  PASS  {name}")


class DummyModel(nn.Module):
    """Minimal stand-in with decay + no-decay params outside and inside the GRU."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(8, 8)          # weight -> decay, bias -> no_decay
        self.norm = nn.LayerNorm(8)              # 1-d params -> no_decay
        self.pose_gru = PoseGRU(hidden_dim=16, mode="residual")


model = DummyModel()
WD, LR_SCALE, BASE_LR = 0.05, 100.0, 1e-6

groups = misc.get_parameter_groups(model, WD)
split = split_pose_gru_param_groups(groups, model.pose_gru, LR_SCALE)

gru_ids = {id(p) for p in model.pose_gru.parameters()}
all_before = {id(p) for g in groups for p in g["params"]}
all_after = [id(p) for g in split for p in g["params"]]

print("== param accounting ==")
check("no param lost or duplicated", set(all_after) == all_before and len(all_after) == len(set(all_after)))

print("== group assignment ==")
for g in split:
    ids = {id(p) for p in g["params"]}
    check(
        f"group pure (wd={g['weight_decay']}, lr_scale={g['lr_scale']:g}, n={len(ids)})",
        ids <= gru_ids or not (ids & gru_ids),
    )
gru_groups = [g for g in split if {id(p) for p in g["params"]} <= gru_ids]
other_groups = [g for g in split if g not in gru_groups]
check("every gru group has lr_scale=100", all(g["lr_scale"] == LR_SCALE for g in gru_groups))
check("non-gru groups keep lr_scale=1", all(g["lr_scale"] == 1.0 for g in other_groups))
check(
    "gru covered by gru groups",
    {id(p) for g in gru_groups for p in g["params"]} == gru_ids,
)
gru_wds = {g["weight_decay"] for g in gru_groups}
check("gru wd split preserved (biases wd=0, weights wd)", gru_wds == {0.0, WD})
check("first group is non-gru (lr logging reads groups[0])", split[0] in other_groups)

print("== adjust_learning_rate honors the scale ==")
optimizer = torch.optim.AdamW(split, lr=BASE_LR, betas=(0.9, 0.95))
args = SimpleNamespace(lr=BASE_LR, min_lr=0.0, warmup_epochs=2, epochs=10)
lr_now = misc.adjust_learning_rate(optimizer, epoch=1, args=args)  # mid-warmup
for g in optimizer.param_groups:
    is_gru = {id(p) for p in g["params"]} <= gru_ids
    expect = lr_now * (LR_SCALE if is_gru else 1.0)
    check(
        f"group lr {'gru' if is_gru else 'base'} = {g['lr']:.2e}",
        abs(g["lr"] - expect) < 1e-15,
    )

print("== one optimizer step actually moves the zero-init head ==")
p_in = torch.nn.functional.normalize(torch.randn(4, 7), dim=-1)
out, _ = model.pose_gru(p_in, None)
loss = (out - torch.randn_like(out)).pow(2).mean()
loss.backward()
head_before = model.pose_gru.head.weight.clone()
misc.adjust_learning_rate(optimizer, epoch=5, args=args)  # post-warmup lr
optimizer.step()
check(
    "zero-init head weight moved after one step",
    not torch.equal(head_before, model.pose_gru.head.weight),
)

print(f"\nALL {len(PASS)} CHECKS PASSED")
