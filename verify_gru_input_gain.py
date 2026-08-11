#!/usr/bin/env python
"""verify_gru_input_gain.py — falsifier battery for P3.4 (pose_gru_input_gain).

CPU only, no data, no checkpoint, seconds to run:

    python verify_gru_input_gain.py

The gain is a per-dimension multiplier on the PoseGRU cell input, added under a
default-off config key. Three properties have to hold before it can ride a run,
and each one is a way this could silently be wrong rather than loudly broken:

  1. ABSENT KEY => BYTE-IDENTICAL. No buffer, no state_dict key, the same
     arithmetic. A gain that quietly defaulted to all-ones would still add a
     state_dict key every pre-P3.4 checkpoint lacks, and the loader hard-fails
     on that -- so "off" has to mean absent, not neutral.
  2. THE BUFFER IS ACTUALLY ON THE PATH. All-ones must reproduce the off arm
     BIT-for-bit, and a non-uniform gain must CHANGE the output. The second half
     is the falsifier: a gain that is built, serialized and logged but never
     multiplied in would pass every other check here.
  3. IT SURVIVES A CHECKPOINT ROUND TRIP. The buffer is presence-gated, so
     _sniff_pose_gru_config has to rebuild it from key presence or a
     gain-trained checkpoint fails to load with "unexpected key".

Plus the two width traps the directive calls out: the buffer must be
self.input_dim long (46 on the F1 pooled arms, NOT 14), and a base-width config
vector must pad with ones over the appended image-feature block rather than
broadcast-fail or shift the columns.

Exit 0 = all checks pass.
"""
import os
import sys
from types import SimpleNamespace

WORKTREE = os.path.dirname(os.path.abspath(__file__))
assert os.path.isfile(
    os.path.join(WORKTREE, "src", "CUT3R", "src", "train_cut3r_baseline.py")
), f"not a my-da3 checkout: {WORKTREE}"
for p in [
    os.path.join(WORKTREE, "src"),
    os.path.join(WORKTREE, "src/CUT3R"),
    os.path.join(WORKTREE, "src/CUT3R/src"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import dust3r.heads  # noqa: F401  MUST precede dust3r.utils.camera (circular import)
from dust3r.model import PoseGRU, _sniff_pose_gru_config

ENC_DIM = 1024
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def build(seed=0, **kw):
    torch.manual_seed(seed)
    return PoseGRU(**kw).eval()


def clone_weights_into(dst, src):
    """Give dst src's parameters, leaving dst's own input_gain buffer alone."""
    sd = {k: v for k, v in src.state_dict().items() if k != "input_gain"}
    if dst.input_gain is not None:
        sd["input_gain"] = dst.input_gain
    dst.load_state_dict(sd, strict=True)


def ref_forward(gru, gru_in, hidden, gain=None, img_feat=None):
    """Independent re-implementation of PoseGRU.forward, gain applied by hand."""
    cell_in = gru_in
    if gru.img_feat == "input":
        f = img_feat.reshape(img_feat.shape[0], gru.img_feat_blocks, gru.img_feat_src_dim)
        f = gru.img_norm(f).reshape(img_feat.shape[0], -1)
        cell_in = torch.cat(
            [gru_in, gru.img_proj(f) if gru.img_feat_proj else f], dim=-1
        )
    if gain is not None:
        cell_in = cell_in * gain
    h = gru.cell(cell_in, hidden)
    raw = gru.head(h)
    if gru.mode == "residual":
        t, q = gru_in[:, :3] + raw[:, :3], gru_in[:, 3:7] + raw[:, 3:7]
    elif gru.mode == "direct":
        t, q = raw[:, :3], raw[:, 3:7]
    else:
        t, q = raw[:, :3], gru_in[:, 3:7] + raw[:, 3:7]
    return torch.cat([t, torch.nn.functional.normalize(q, dim=-1)], dim=-1), h


A4 = dict(mode="residual", input_mode="pose_delta", hidden_dim=128)
F1 = dict(A4, img_feat="input", img_feat_dim=32, img_feat_frames=2,
          img_feat_proj=True, img_feat_src="pooled", img_feat_src_dim=ENC_DIM)

torch.manual_seed(7)
B = 5
X = torch.randn(B, 14)
H0 = torch.randn(B, 128)
IMG = torch.randn(B, 2 * ENC_DIM)
# A non-uniform gain in the shape P3.4 will actually use: shrink the absT
# columns, blow up the delta_t ones, quaternion dims left at 1.0.
GAIN14 = [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 1.0]

def untrain_head(gru):
    """Give a residual-mode module a NON-ZERO head, i.e. make it look trained.

    mode='residual' zero-inits the head, so raw == 0 and the emitted pose is the
    fed-back anchor for ANY hidden state — which means an untrained module's
    OUTPUT cannot respond to the gain even when the gain is working perfectly
    (check 4 asserts exactly that, deliberately). Every check that needs to
    observe the gain downstream of the cell therefore has to unlock the head
    first, or its falsifier is vacuous: it would "pass" on a no-op gain too.
    """
    with torch.no_grad():
        gru.head.weight.normal_(0, 0.05)
        gru.head.bias.normal_(0, 0.05)
    return gru


print("\n=== 1. absent key => byte-identical previous behavior ===")
off = untrain_head(build(**A4))
check("input_dim is 14 on A4xF0", off.input_dim == 14, f"got {off.input_dim}")
check("no input_gain attribute value", off.input_gain is None)
check("no state_dict key", "input_gain" not in off.state_dict(),
      f"keys: {sorted(off.state_dict())}")
with torch.no_grad():
    got, goth = off(X, H0)
    exp, exph = ref_forward(off, X, H0, gain=None)
check("forward matches the ungained reference exactly",
      torch.equal(got, exp) and torch.equal(goth, exph))

print("\n=== 2. all-ones gain is BIT-identical to off (init-equivalence) ===")
ones = build(**A4, input_gain=[1.0] * 14)
clone_weights_into(ones, off)
check("buffer built at input_dim", tuple(ones.input_gain.shape) == (14,),
      f"got {tuple(ones.input_gain.shape)}")
check("state_dict now carries the key", "input_gain" in ones.state_dict())
with torch.no_grad():
    o, oh = off(X, H0)
    g, gh = ones(X, H0)
check("output bit-identical", torch.equal(o, g), f"max|d| {(o - g).abs().max():.3e}")
check("hidden bit-identical", torch.equal(oh, gh))

print("\n=== 3. FALSIFIER: a non-uniform gain must CHANGE the output ===")
gained = build(**A4, input_gain=GAIN14)
clone_weights_into(gained, off)
with torch.no_grad():
    o, oh = off(X, H0)
    g, gh = gained(X, H0)
    r, rh = ref_forward(off, X, H0, gain=torch.tensor(GAIN14))
check("hidden state differs from the off arm", not torch.allclose(oh, gh, atol=1e-6),
      f"max|d| {(oh - gh).abs().max():.4f}")
check("output differs from the off arm", not torch.allclose(o, g, atol=1e-6),
      f"max|d| {(o - g).abs().max():.4f}")
check("matches the hand-computed reference", torch.allclose(g, r, atol=1e-6)
      and torch.allclose(gh, rh, atol=1e-6))

print("\n=== 4. residual anchor is NOT rescaled (zero-init head equivalence) ===")
# In residual mode the head is zero-init, so raw == 0 and the emitted pose is
# the anchor gru_in itself -- for ANY gain. If the gain ever leaked into the
# anchor algebra (t = gru_in[:, :3] + raw[:, :3]) this would break, and every
# existing checkpoint's "an untrained GRU reproduces plain feed_prev_pred
# bit-for-bit" guarantee would break with it.
fresh_off, fresh_gain = build(1, **A4), build(1, **A4, input_gain=GAIN14)
with torch.no_grad():
    po, _ = fresh_off(X, H0)
    pg, _ = fresh_gain(X, H0)
    anchor = torch.cat([X[:, :3], torch.nn.functional.normalize(X[:, 3:7], dim=-1)], -1)
check("untrained gained module == untrained off module", torch.equal(po, pg))
check("both == the plain fed-back anchor", torch.allclose(po, anchor, atol=1e-6))

print("\n=== 5. F1 arm: input_dim 46, base-width config pads with ones ===")
f1_off = untrain_head(build(2, **F1))
check("input_dim is 46 on F1 pooled/proj", f1_off.input_dim == 46,
      f"got {f1_off.input_dim}")
# Un-zero the projector: it is zero-init, so with the default weights the
# appended columns are all 0 and a wrong tail gain would be invisible.
with torch.no_grad():
    f1_off.img_proj.weight.normal_(0, 0.05)
    f1_off.img_proj.bias.normal_(0, 0.05)
f1_short = build(2, **F1, input_gain=GAIN14)           # 14 given, 46 expected
f1_full = build(2, **F1, input_gain=GAIN14 + [1.0] * 32)
for m in (f1_short, f1_full):
    clone_weights_into(m, f1_off)
check("short form padded to input_dim", tuple(f1_short.input_gain.shape) == (46,),
      f"got {tuple(f1_short.input_gain.shape)}")
check("padding is ones over the image block",
      torch.equal(f1_short.input_gain[14:], torch.ones(32)))
with torch.no_grad():
    s, sh = f1_short(X, H0, img_feat=IMG)
    fu, fuh = f1_full(X, H0, img_feat=IMG)
    ro, rh2 = ref_forward(f1_off, X, H0, gain=torch.tensor(GAIN14 + [1.0] * 32),
                          img_feat=IMG)
    base, _ = f1_off(X, H0, img_feat=IMG)
check("short form == explicit full form", torch.equal(s, fu) and torch.equal(sh, fuh))
check("matches the hand-computed reference", torch.allclose(s, ro, atol=1e-6)
      and torch.allclose(sh, rh2, atol=1e-6))
check("and still differs from the F1 off arm", not torch.allclose(base, s, atol=1e-6),
      f"max|d| {(base - s).abs().max():.4f}")
# The image columns must be untouched: scaling them too would change the result
# even when the pose block gain is all-ones.
f1_ones = build(2, **F1, input_gain=[1.0] * 14)
clone_weights_into(f1_ones, f1_off)
with torch.no_grad():
    z, _ = f1_ones(X, H0, img_feat=IMG)
check("all-ones pose block leaves LayerNormed image features unscaled",
      torch.equal(base, z))

print("\n=== 6. checkpoint round trip through the sniff ===")
args = SimpleNamespace(pose_gru_mode="residual", pose_gru_input="pose_delta",
                       pose_gru_hidden_dim=128, pose_gru_iters=1)
state = {f"pose_gru.{k}": v for k, v in gained.state_dict().items()}
cfg = _sniff_pose_gru_config(state, args, ENC_DIM, where="synthetic")
check("sniff detects the buffer", cfg.get("input_gain") is True, f"got {cfg!r}")
rebuilt = PoseGRU(**{k: v for k, v in cfg.items()}, img_feat_src_dim=ENC_DIM)
missing, unexpected = rebuilt.load_state_dict(
    {k[len("pose_gru."):]: v for k, v in state.items()}, strict=True)
check("strict load has no missing/unexpected keys", not missing and not unexpected,
      f"missing={list(missing)} unexpected={list(unexpected)}")
check("gain values survive the round trip",
      torch.equal(rebuilt.input_gain, gained.input_gain))
with torch.no_grad():
    rr, _ = rebuilt(X, H0)
    gg, _ = gained(X, H0)
check("rebuilt module reproduces the original forward", torch.equal(rr, gg))

state_off = {f"pose_gru.{k}": v for k, v in off.state_dict().items()}
cfg_off = _sniff_pose_gru_config(state_off, args, ENC_DIM, where="synthetic")
check("sniff reports off for a pre-P3.4 checkpoint", cfg_off.get("input_gain") is None,
      f"got {cfg_off.get('input_gain')!r}")

print("\n=== 7. mis-sniffed modules hard-fail rather than load silently ===")
try:
    PoseGRU(**A4).load_state_dict(
        {k[len("pose_gru."):]: v for k, v in state.items()}, strict=True)
    check("gain ckpt into an off module raises", False, "it loaded silently")
except RuntimeError as e:
    check("gain ckpt into an off module raises", "input_gain" in str(e),
          str(e).splitlines()[0][:70])

print("\n=== 8. bad config values are refused, not silently coerced ===")
for bad, why in [
    ([1.0] * 13, "wrong length"),
    ([1.0] * 46, "F-width vector on an F-off arm"),
    ([1.0] * 6 + [0.0] + [1.0] * 7, "zero gain"),
    ([1.0] * 6 + [-2.0] + [1.0] * 7, "negative gain"),
    ([1.0] * 6 + [float("nan")] + [1.0] * 7, "NaN gain"),
]:
    try:
        PoseGRU(**A4, input_gain=bad)
        check(f"refuses {why}", False, "constructed anyway")
    except (ValueError, AssertionError):
        check(f"refuses {why}", True)

print("\n" + "=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("all checks passed — pose_gru_input_gain is default-off, on the cell "
      "input path, width-correct on F arms, and checkpoint-round-trippable")
sys.exit(0)
