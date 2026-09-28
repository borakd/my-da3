"""wgate heads: frame-level Kendall & Gal confidences that weight CUT3R's two memory writes.

Design + implementation contract: ``WRITE_GATE_VARIANTS.md`` (repo root). Everything here is a
pure function of features the decoder step already produced; the backbone stays frozen.

Variant 1 (state write, "frame gate")
    ``FrameConfHead``: LayerNorm(1796) -> Linear(1796, 256) -> GELU -> Linear(256, 1), output
    ``s = log sigma^2`` for the frame. Its input is ``frame_features(new_state_feat, state_feat,
    global_img_feat)`` = concat( mean over tokens of delta [768], mean token-norm of delta [1],
    (mean, std, max) of the token norms [3], global encoder feature [1024] ), delta = proposal - old.
    Weight ``a_t = clip(sigma_ref / sigma_t, wmin, 1)`` (``sigma_to_weight``).

Variant 2 (pose-retriever memory write, "mem gate")
    ``PoseSigmaHead``: the frozen pose decoder's own penultimate activation ``h = GELU(fc1(pose_feat))``
    (3072-d, borrowed from ``PoseDecoder.mlp``) followed by ONE new ``Linear(3072, 2)`` giving
    ``(s_t, s_R) = log sigma^2`` of the translation / rotation residuals. Only the new layer trains.
    Weight ``b_t = clip(1 / sigma_comb, wmin, 1)`` with
    ``sigma_comb = sqrt((sigma_t / ref_t) * (sigma_R / ref_R))`` (``combine_pose_sigma``,
    ``pose_sigma_to_weight``).

Loss (both heads, paper eq. 8 on a scalar residual r): ``exp(-s) * r^2 + s`` (``gaussian_nll``).

Variant C (CONSEQUENCE-trained gate, no sigma)
    ``GateWeightHead(kind, ...)``: the SAME two inputs, but the output IS the weight, trained by the
    gradient of a causal trajectory error through the memory (``train_wgate_consequence.py``), not by a
    Kendall & Gal NLL. ``kind="frame"`` is the 1796-d MLP of variant 1; ``kind="pose"`` borrows the frozen
    pose-decoder ``mlp.fc1``/``act`` exactly as ``PoseSigmaHead`` does. Both end in ``Linear(*, 1) -> u``
    and ``w = wmin + (1 - wmin) * sigmoid(u)``; the last layer starts at ``weight = 0, bias = init_logit``
    so EVERY frame starts at the same weight ``weight_at_init`` (near 1) and training can only move it.
    Checkpoints carry ``{"state_dict", "mode": "weight", "wmin", "init_logit", ...}``; the model's
    ``attach_frame_gate`` / ``attach_mem_gate`` dispatch on that ``"mode"``.

Precision: every function here upcasts to float32 and runs with autocast DISABLED, so the heads
behave identically under a bf16-autocast rollout and the fp32 eval worker.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FRAME_FEAT_DIM",
    "POSE_FEAT_DIM",
    "POSE_HIDDEN_DIM",
    "FrameConfHead",
    "PoseSigmaHead",
    "GateWeightHead",
    "frame_features",
    "sigma_to_weight",
    "combine_pose_sigma",
    "pose_sigma_to_weight",
    "gaussian_nll",
    "conf_mem_weight",
]

FRAME_FEAT_DIM = 768 + 1 + 3 + 1024  # = 1796, see frame_features()
POSE_FEAT_DIM = 768  # decoder width = pose token width
POSE_HIDDEN_DIM = 3072  # PoseDecoder.mlp.fc1 out_features (mlp_ratio 4)

# |s| is clamped inside exp() only -- numerical stability, s itself is unbounded.
_S_CLAMP = 30.0


def _no_autocast(t):
    """Context manager: autocast OFF for the device of tensor ``t`` (works for cpu and cuda)."""
    return torch.autocast(device_type=t.device.type, enabled=False)


# ----------------------------------------------------------------------------- features
def frame_features(new_state_feat, state_feat, global_img_feat):
    """Frame-level input vector of ``FrameConfHead``: (B, 1796) float32, detached.

    Args:
        new_state_feat: (B, N, C) the decoder's proposed state (output of ``_recurrent_rollout``).
        state_feat:     (B, N, C) the state BEFORE the commit (old memory).
        global_img_feat: (B, 1, 1024) or (B, 1024) the frame's mean encoder feature
                        (``global_img_feat_group`` in ``_forward_decoder_group_step``).

    Layout (exactly as in WRITE_GATE_VARIANTS.md, "Variant 1 / Confidence"):
        [0:768]      f_delta_mean = delta.mean(1)                       (mean over tokens, per channel)
        [768:769]    f_delta_norm = delta.norm(dim=-1).mean(1)          (mean token norm)
        [769:772]    f_delta_tok  = (mean, std, max) over tokens of delta.norm(dim=-1)
        [772:1796]   f_img        = global_img_feat[:, 0]               (1024)
    where ``delta = new_state_feat - state_feat`` in float32. Detached and computed with autocast
    disabled, so the result is identical under bf16 autocast and fp32.
    """
    with torch.no_grad(), _no_autocast(new_state_feat):
        new = new_state_feat.detach().float()
        old = state_feat.detach().float()
        assert new.shape == old.shape and new.dim() == 3, (new.shape, old.shape)
        delta = new - old  # (B, N, C)
        f_delta_mean = delta.mean(dim=1)  # (B, C)
        tok_norm = delta.norm(dim=-1)  # (B, N)
        f_delta_norm = tok_norm.mean(dim=1, keepdim=True)  # (B, 1)
        f_delta_tok = torch.stack(
            [tok_norm.mean(dim=1), tok_norm.std(dim=1, unbiased=False), tok_norm.max(dim=1).values],
            dim=-1,
        )  # (B, 3)
        g = global_img_feat.detach().float()
        if g.dim() == 3:
            g = g[:, 0]
        assert g.dim() == 2 and g.shape[0] == new.shape[0], g.shape
        x = torch.cat([f_delta_mean, f_delta_norm, f_delta_tok, g], dim=-1)
    assert x.shape[-1] == FRAME_FEAT_DIM, x.shape
    return x


STATE_CH = 768  # channels of one CUT3R state token (state_feat is (B, 768 tokens, 768 channels))
TOKEN_FEAT_DIM = FRAME_FEAT_DIM + STATE_CH + 1  # = 2565, see token_features()


def token_features(new_state_feat, state_feat, global_img_feat):
    """Per-TOKEN input of the kind="token" GateWeightHead: (B, N, 2565) float32, detached.

    Token i of frame t gets the frame's own descriptor (``frame_features``, the exact input of the
    per-frame head, repeated for every token) followed by that token's proposed change:
        [0:1796]     frame_features(new_state_feat, state_feat, global_img_feat)   (same for all i)
        [1796:2564]  delta_i = new_state_feat[:, i] - state_feat[:, i]              (768)
        [2564:2565]  ||delta_i||                                                    (1)
    so the per-token head sees everything the per-frame head sees plus what is about to be written
    into its own token. Detached, autocast off (like frame_features).
    """
    x = frame_features(new_state_feat, state_feat, global_img_feat)  # (B, 1796), detached fp32
    with torch.no_grad(), _no_autocast(new_state_feat):
        delta = new_state_feat.detach().float() - state_feat.detach().float()  # (B, N, C)
        assert delta.shape[-1] == STATE_CH, delta.shape
        B, N, _ = delta.shape
        xt = torch.cat([x[:, None, :].expand(B, N, x.shape[-1]), delta, delta.norm(dim=-1, keepdim=True)], dim=-1)
    assert xt.shape[-1] == TOKEN_FEAT_DIM, xt.shape
    return xt


# ----------------------------------------------------------------------------- sigma -> weight
def sigma_to_weight(log_var, log_ref, wmin):
    """``a = clip(sigma_ref / sigma, wmin, 1)`` from log-variances, returned as (B,) float32.

    ``sigma_ref / sigma = exp(0.5 * (log_ref - log_var))``. ``log_var`` is (B,) or (B, 1);
    ``log_ref`` a float or 0-d/1-elem tensor (the training-table median of ``log sigma^2``);
    ``wmin`` a float in [0, 1]. A frame at the reference confidence gets weight exactly 1.0.
    """
    lv = torch.as_tensor(log_var).float().reshape(-1)
    lr = torch.as_tensor(log_ref, dtype=torch.float32, device=lv.device).reshape(-1)
    z = (0.5 * (lr - lv)).clamp(min=-_S_CLAMP, max=_S_CLAMP)
    return torch.exp(z).clamp(min=float(wmin), max=1.0)


def combine_pose_sigma(log_var2, log_ref2):
    """``sigma_comb = sqrt((sigma_t / ref_t) * (sigma_R / ref_R))`` from log-variances -> (B,) float32.

    ``log_var2`` is (B, 2) = ``(s_t, s_R)`` (translation, rotation log sigma^2); ``log_ref2`` a
    2-vector (or 2-tuple) of the training medians. In log space:
    ``log sigma_comb = 0.25 * ((s_t - ref_t) + (s_R - ref_R))``.
    """
    lv = torch.as_tensor(log_var2).float()
    if lv.dim() == 1:
        lv = lv[None]
    assert lv.shape[-1] == 2, lv.shape
    lr = torch.as_tensor(log_ref2, dtype=torch.float32, device=lv.device).reshape(-1)
    assert lr.numel() == 2, lr.shape
    z = (0.25 * (lv - lr[None]).sum(dim=-1)).clamp(min=-_S_CLAMP, max=_S_CLAMP)
    return torch.exp(z)


def pose_sigma_to_weight(log_var2, log_ref2, wmin):
    """``b = clip(1 / sigma_comb, wmin, 1)`` -> (B,) float32 (the mem-gate weight)."""
    sig = combine_pose_sigma(log_var2, log_ref2)
    return (1.0 / sig).clamp(min=float(wmin), max=1.0)


def gaussian_nll(log_var, residual):
    """Kendall & Gal Gaussian NLL on a scalar residual, per element: ``exp(-s) * r^2 + s``.

    Shapes broadcast; returns the un-reduced loss (reduce with .mean() in the trainer).
    """
    s = torch.as_tensor(log_var).float()
    r = torch.as_tensor(residual).float()
    return torch.exp(-s.clamp(min=-_S_CLAMP, max=_S_CLAMP)) * r * r + s


def conf_mem_weight(c_t, hist, soft=0.5, wmin=0.5):
    """Causal self-view confidence -> pose-memory write weight ``b_t`` (arm ``mg_conf``).

    ``b_t = wmin + (1 - wmin) * sigmoid((c_t - median(hist)) / soft)``.

    ``c_t`` is this frame's mean ``log(conf_self.clamp(min=1))`` (a float or 0-d/1-element
    tensor) and ``hist`` the CAUSAL history of that quantity over frames ``0..t-1`` -- the
    current frame is NOT in it (the caller appends ``c_t`` after this call). An empty history
    returns exactly ``1.0``, which is the ``b_0 = 1`` rule: frame 0 is always written in full.
    A frame exactly at the running median gets ``(1 + wmin) / 2``; a much more confident frame
    tends to 1 and a much less confident one to ``wmin``. ``torch.median`` is the lower median
    on an even-length history (the convention the existing conf trigger uses).

    Returns a 0-d float32 CPU/`c_t`-device tensor. Pure function: no state, no grad.
    """
    c = torch.as_tensor(c_t, dtype=torch.float32).reshape(-1)
    assert c.numel() == 1, f"conf_mem_weight: c_t must be a scalar, got {tuple(c.shape)}"
    c = c.reshape(())
    if hist is None or len(hist) == 0:
        return torch.ones((), dtype=torch.float32, device=c.device)  # b_0 = 1 exactly
    soft = float(soft)
    wmin = float(wmin)
    assert soft > 0, f"conf_mem_weight: soft must be > 0, got {soft}"
    assert 0.0 <= wmin <= 1.0, f"conf_mem_weight: wmin must be in [0, 1], got {wmin}"
    h = torch.as_tensor(list(hist), dtype=torch.float32, device=c.device).reshape(-1)
    z = ((c - h.median()) / soft).clamp(min=-_S_CLAMP, max=_S_CLAMP)
    return wmin + (1.0 - wmin) * torch.sigmoid(z)


# ----------------------------------------------------------------------------- heads
class FrameConfHead(nn.Module):
    """Variant 1 confidence: LayerNorm(in) -> Linear(in, hidden) -> GELU -> Linear(hidden, 1).

    ``forward(x)`` takes (B, in_dim) float32 (see ``frame_features``) and returns ``s = log sigma^2``
    as (B,). The output layer starts near zero (weights std 1e-3, zero bias) so ``sigma ~ 1``
    initially. Runs in float32 with autocast disabled whatever the ambient context.
    """

    def __init__(self, in_dim=FRAME_FEAT_DIM, hidden=256):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.norm = nn.LayerNorm(self.in_dim)
        self.fc1 = nn.Linear(self.in_dim, self.hidden)
        self.act = nn.GELU()
        self.out = nn.Linear(self.hidden, 1)
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        with _no_autocast(x):
            x = x.float()
            assert x.dim() == 2 and x.shape[-1] == self.in_dim, (x.shape, self.in_dim)
            h = self.act(self.fc1(self.norm(x)))
            return self.out(h)[:, 0]

    @classmethod
    def from_state_dict(cls, sd):
        """Build a head whose sizes are read off the state dict, then load it."""
        head = cls(in_dim=sd["norm.weight"].shape[0], hidden=sd["fc1.weight"].shape[0])
        head.load_state_dict(sd)
        return head


class PoseSigmaHead(nn.Module):
    """Variant 2 confidence: extra output rows on the frozen pose head.

    ``h = act(fc1(pose_feat))`` reuses the pose decoder's OWN ``mlp.fc1`` / ``mlp.act`` (frozen,
    shared with the model: the same 3072-d activation the pose regressor reads), then a NEW
    ``self.out = nn.Linear(3072, 2)`` gives ``(s_t, s_R) = log sigma^2``.

    The borrowed ``fc1``/``act`` are deliberately NOT registered as submodules: ``state_dict()``
    holds the out layer only (contract: ``"state_dict" (out layer only)``), ``parameters()`` are
    the trainable ones only, and ``.to()`` / ``.float()`` on this head never touch the model. The
    pose decoder must therefore already live on the device the head is called on. fc1 is applied
    in float32 (weights upcast on the fly) with autocast disabled.

    ``forward(pose_feat)``: (B, 768) -> (B, 2). ``penultimate(pose_feat)``: (B, 768) -> (B, 3072)
    (no grad; recompute from the recorded ``pose_feat`` instead of storing h).
    """

    def __init__(self, pose_decoder):
        super().__init__()
        mlp = pose_decoder.mlp if hasattr(pose_decoder, "mlp") else pose_decoder
        fc1, act = mlp.fc1, mlp.act
        assert isinstance(fc1, nn.Linear), type(fc1)
        self._borrowed = (fc1, act)  # tuple => not registered by nn.Module
        self.hidden = int(fc1.out_features)
        self.in_dim = int(fc1.in_features)
        self.out = nn.Linear(self.hidden, 2)
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out.bias)

    @property
    def fc1(self):
        return self._borrowed[0]

    @property
    def act(self):
        return self._borrowed[1]

    def penultimate(self, pose_feat):
        """(B, 768) -> (B, 3072) = act(fc1(pose_feat)), float32, no grad (fc1 is frozen)."""
        with torch.no_grad(), _no_autocast(pose_feat):
            x = pose_feat.detach().float()
            assert x.dim() == 2 and x.shape[-1] == self.in_dim, (x.shape, self.in_dim)
            fc1 = self.fc1
            w = fc1.weight.detach().float()
            b = fc1.bias.detach().float() if fc1.bias is not None else None
            if x.device != w.device:
                x = x.to(w.device)
            return self.act(F.linear(x, w, b))

    def forward(self, pose_feat):
        h = self.penultimate(pose_feat)
        with _no_autocast(h):
            return self.out(h)

    def load_head_state_dict(self, sd):
        """Load the out layer from either ``{"weight","bias"}`` or ``{"out.weight","out.bias", ...}``
        (any borrowed ``fc1.*`` entries are ignored)."""
        if "out.weight" in sd:
            sd = {k[len("out."):]: v for k, v in sd.items() if k.startswith("out.")}
        self.out.load_state_dict({"weight": sd["weight"], "bias": sd["bias"]})
        return self


# ----------------------------------------------------------------------------- variant C head
class GateWeightHead(nn.Module):
    """Variant C: the write weight itself, trained by the CONSEQUENCE of the write.

    Same inputs as the two sigma heads, but the output is the gate weight, not a log-variance --
    there is no ``sigma_ref / sigma`` map and no Kendall & Gal term anywhere in this class
    (WRITE_GATE_VARIANTS.md, "Variant C", DEPARTURE 6).

    ``kind="frame"`` (C1, the 768-token state write)
        LayerNorm(in_dim=1796) -> Linear(1796, hidden=256) -> GELU -> Linear(256, 1) -> u,
        on ``frame_features(...)``. The whole head trains.
    ``kind="pose"`` (C2, the pose-retriever memory write)
        ``h = act(fc1(pose_feat))`` borrowed FROZEN from the model's ``PoseDecoder.mlp`` (exactly as
        ``PoseSigmaHead``: kept in a tuple, so it is not registered, ``state_dict()`` / ``parameters()``
        hold the new layer only and ``.to()`` / ``.float()`` never touch the model), then a NEW
        ``Linear(3072, 1) -> u``. Only that layer trains; the input is detached.

    Weight map (both kinds): ``w = wmin + (1 - wmin) * sigmoid(u)``, so ``w in (wmin, 1)``.
    Init: the last ``Linear`` starts at ``weight = 0``, ``bias = init_logit``, hence
    ``w = weight_at_init = wmin + (1 - wmin) * sigmoid(init_logit)`` for EVERY input (exactly, not
    approximately) before the first optimiser step -- the run starts as near-plain CUT3R and training
    can only move it.

    Precision: float32 with autocast disabled, like the sigma heads. ``forward`` / ``weight_from_logit``
    take an optional ``wmin`` override (the model hook passes the per-view ``*_gate_wmin`` key through,
    so one trained head can be evaluated at another floor without re-training; ``wmin=1`` reproduces
    plain CUT3R bit-exactly).
    """

    KINDS = ("frame", "pose", "token")

    def __init__(self, kind, in_dim=None, hidden=256, wmin=0.5, init_logit=2.0, pose_decoder=None):
        super().__init__()
        kind = str(kind)
        assert kind in self.KINDS, f"kind must be one of {self.KINDS}, got {kind!r}"
        self.kind = kind
        self.wmin = float(wmin)
        self.init_logit = float(init_logit)
        if kind in ("frame", "token"):
            assert pose_decoder is None, f"kind={kind!r} takes no pose_decoder"
            self.in_dim = int((FRAME_FEAT_DIM if kind == "frame" else TOKEN_FEAT_DIM) if in_dim is None else in_dim)
            self.hidden = int(hidden)
            self.norm = nn.LayerNorm(self.in_dim)
            self.fc1 = nn.Linear(self.in_dim, self.hidden)
            self.act = nn.GELU()
            self.out = nn.Linear(self.hidden, 1)
            self._borrowed = None
        else:
            assert pose_decoder is not None, "kind='pose' needs the model's frozen PoseDecoder"
            mlp = pose_decoder.mlp if hasattr(pose_decoder, "mlp") else pose_decoder
            fc1, act = mlp.fc1, mlp.act
            assert isinstance(fc1, nn.Linear), type(fc1)
            self._borrowed = (fc1, act)  # tuple => not registered by nn.Module
            self.in_dim = int(fc1.in_features)
            self.hidden = int(fc1.out_features)
            if in_dim is not None:
                assert int(in_dim) == self.in_dim, (in_dim, self.in_dim)
            self.out = nn.Linear(self.hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, self.init_logit)

    # --- borrowed (pose kind only); named so they cannot collide with the frame kind's submodules
    @property
    def borrowed_fc1(self):
        return None if self._borrowed is None else self._borrowed[0]

    @property
    def borrowed_act(self):
        return None if self._borrowed is None else self._borrowed[1]

    @property
    def weight_at_init(self):
        """The weight every frame gets before the first optimiser step: wmin + (1-wmin)*sigmoid(init_logit)."""
        return self.wmin + (1.0 - self.wmin) / (1.0 + math.exp(-self.init_logit))

    def head_cfg(self):
        return {
            "kind": self.kind,
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "wmin": self.wmin,
            "init_logit": self.init_logit,
        }

    def checkpoint(self, args=None, metrics=None):
        """The contract's variant-C checkpoint dict (what attach_* dispatches on)."""
        ck = {
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
            "mode": "weight",
            "wmin": self.wmin,
            "init_logit": self.init_logit,
            "head_cfg": self.head_cfg(),
        }
        if args is not None:
            ck["args"] = args
        if metrics is not None:
            ck["metrics"] = metrics
        return ck

    # --- forward
    def logit(self, x):
        """(B, in_dim) -> (B,) raw logit u, float32, autocast off. Gradient reaches the head's own
        parameters; for kind='pose' the borrowed fc1 and its input are detached (frozen backbone)."""
        with _no_autocast(x):
            if self.kind == "frame":
                x = x.float()
                assert x.dim() == 2 and x.shape[-1] == self.in_dim, (x.shape, self.in_dim)
                h = self.act(self.fc1(self.norm(x)))
            elif self.kind == "token":
                # (B, N, in_dim) -> (B, N): the same MLP applied to every state token (shared weights)
                x = x.float()
                assert x.dim() == 3 and x.shape[-1] == self.in_dim, (x.shape, self.in_dim)
                return self.out(self.act(self.fc1(self.norm(x))))[..., 0]
            else:
                x = x.detach().float()
                assert x.dim() == 2 and x.shape[-1] == self.in_dim, (x.shape, self.in_dim)
                fc1 = self.borrowed_fc1
                w = fc1.weight.detach().float()
                b = fc1.bias.detach().float() if fc1.bias is not None else None
                if x.device != w.device:
                    x = x.to(w.device)
                h = self.borrowed_act(F.linear(x, w, b))
            return self.out(h)[:, 0]

    def weight_from_logit(self, u, wmin=None):
        """``w = wmin + (1 - wmin) * sigmoid(u)`` -> same shape as u ((B,), or (B, N) for kind="token"),
        float32 (``wmin`` overrides ``self.wmin``)."""
        wm = self.wmin if wmin is None else float(wmin)
        u = u.float()
        with _no_autocast(u):
            return wm + (1.0 - wm) * torch.sigmoid(u)

    def forward(self, x, wmin=None):
        """(B, in_dim) -> (B,) weight in (wmin, 1); kind="token": (B, N, in_dim) -> (B, N)."""
        return self.weight_from_logit(self.logit(x), wmin=wmin)

    # --- (de)serialisation
    def load_head_state_dict(self, sd):
        """kind='pose': load the out layer from ``{"weight","bias"}`` or ``{"out.weight","out.bias"}``."""
        assert self.kind == "pose", "load_head_state_dict is the pose kind's out-layer loader"
        if "out.weight" in sd:
            sd = {k[len("out."):]: v for k, v in sd.items() if k.startswith("out.")}
        self.out.load_state_dict({"weight": sd["weight"], "bias": sd["bias"]})
        return self

    @classmethod
    def from_state_dict(cls, sd, kind=None, wmin=0.5, init_logit=2.0, pose_decoder=None):
        """Build a head whose sizes are read off the state dict, then load it.

        ``kind`` defaults to "frame" when the state dict has the MLP ("norm.weight"), else "pose"
        (which needs ``pose_decoder``). ``wmin`` / ``init_logit`` are metadata of the map, not weights:
        pass the checkpoint's values.
        """
        sd = dict(sd)
        if kind is None:
            kind = "frame" if "norm.weight" in sd else "pose"
        if kind in ("frame", "token"):
            head = cls(
                kind,
                in_dim=int(sd["norm.weight"].shape[0]),
                hidden=int(sd["fc1.weight"].shape[0]),
                wmin=wmin,
                init_logit=init_logit,
            )
            head.load_state_dict(sd)
        else:
            head = cls("pose", wmin=wmin, init_logit=init_logit, pose_decoder=pose_decoder)
            head.load_head_state_dict(sd)
        return head
