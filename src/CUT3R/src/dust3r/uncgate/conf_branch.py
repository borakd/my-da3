"""Trainable confidence branch for the self-view DPT head.

The DPT output stack is Conv3x3(256->128) -> Interpolate -> Conv3x3(128->128) -> ReLU -> Conv1x1(128->4),
channel 3 being the confidence pre-activation x (conf = 1 + exp(x) downstream). ConfBranchHead keeps the
original stack frozen for xyz and adds a copy of it, initialised from the same weights (a full copy of the stack), whose channel 3 replaces channel 3. At initialisation the module is
byte-identical to the original head; only the copy trains.
"""
import copy
import torch
import torch.nn as nn


class ConfBranchHead(nn.Module):
    """conf = a full deep copy of the frozen output stack (same ops, same weights at init, so channel 3 of the
    copy is bit-identical to channel 3 of the original until training moves it); only the copy trains."""
    def __init__(self, base_head: nn.Sequential, conf_channel: int = 3):
        super().__init__()
        self.base = base_head
        self.conf = copy.deepcopy(base_head)
        self.conf_channel = conf_channel

    def forward(self, x):
        out = self.base(x)
        cc = self.conf_channel
        c = self.conf(x)[:, cc:cc + 1]
        return torch.cat([out[:, :cc], c, out[:, cc + 1:]], dim=1)
