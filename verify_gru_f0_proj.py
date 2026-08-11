#!/usr/bin/env python
"""
verify_gru_f0_proj.py — falsifier for the two new F-lever knobs:

  pose_gru_img_feat_frames: 1   ("F0") — the GRU sees ONLY the current view's
                                pooled pre-ray features (frames: 2 = "F1" =
                                current + previous, the pre-existing arm)
  pose_gru_img_feat_proj: False — the zero-init Linear(src*frames -> 32) is
                                REMOVED and the LayerNormed features enter the
                                GRUCell raw, widening the cell input by the
                                full 1024 (F0) / 2048 (F1)

Runs entirely on CPU — no GPU, no pretrained checkpoint, no dataset. Every
stage is a falsifier: it is written so that the WRONG implementation fails.

  UNIT   ctor: widths, appended dims, param counts, which submodules exist,
         and the ctor asserts that must still fire
  IDENT  residual init-equivalence survives BOTH projector modes (the head,
         not the projector, is what guarantees it when proj=False)
  GATE   proj=True is image-BLIND at init (hidden invariant to the features);
         proj=False is image-SENSITIVE at init (hidden moves with them).
         This is the load-bearing behavioural difference between the arms.
  GRAD   the zero head locks the cell in both modes; once unlocked, gradient
         reaches img_proj (proj=True) / the raw feature columns of weight_ih
         and img_norm (proj=False)
  SNIFF  save -> load_model round-trip restores frames/proj/appended width for
         all five arms, INCLUDING the projector-less ones (which carry no
         img_proj tensor to sniff), and legacy arms are unchanged

Run:  python verify_gru_f0_proj.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace

WORKTREE = os.path.dirname(os.path.abspath(__file__))
assert os.path.isfile(
    os.path.join(WORKTREE, "src", "CUT3R", "src", "train_cut3r_baseline.py")
), f"not a my-da3 checkout: {WORKTREE}"
for p in [
    WORKTREE,
    os.path.join(WORKTREE, "src"),
    os.path.join(WORKTREE, "src/CUT3R"),
    os.path.join(WORKTREE, "src/CUT3R/src"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import dust3r.heads  # noqa: F401,E402  circular-import order
from dust3r.model import PoseGRU, load_model  # noqa: E402

MODEL_STR = (
    "ARCroco3DStereo(ARCroco3DStereoConfig(freeze='encoder', state_size=768, "
    "state_pe='2d', pos_embed='RoPE100', rgb_head=True, pose_head=True, "
    "patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', "
    "output_mode='pts3d+pose', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), "
    "pose_mode=('exp', -inf, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, "
    "dec_embed_dim=768, dec_depth=12, dec_num_heads=12, landscape_only=False))"
)
SRC = 1024   # encoder width
H = 128      # hidden_dim
B = 3
FAILS = []


def check(tag, desc, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}  {desc}" + (f"   ({extra})" if extra else ""))
    if not ok:
        FAILS.append(tag)


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def mk(frames, proj, mode="residual"):
    """A4 (pose_delta) PoseGRU with the requested F arm, fixed init."""
    torch.manual_seed(GRU_SEED)
    return PoseGRU(
        hidden_dim=H, mode=mode, input_mode="pose_delta",
        img_feat="input", img_feat_dim=32, img_feat_frames=frames,
        img_feat_proj=proj, img_feat_src_dim=SRC, iters=1,
    )


GRU_SEED = 4242
ARMS = {  # label -> (frames, proj, expected appended width)
    "F0 proj  (frames=1, proj=True )": (1, True, 32),
    "F1 proj  (frames=2, proj=True )": (2, True, 32),
    "F0 raw   (frames=1, proj=False)": (1, False, 1024),
    "F1 raw   (frames=2, proj=False)": (2, False, 2048),
}

# =============================================================================
banner("STAGE UNIT — ctor widths, submodules, param counts, asserts")
# =============================================================================
for label, (frames, proj, appended) in ARMS.items():
    g = mk(frames, proj)
    ok = (
        g.input_dim == 14 + appended
        and g.img_feat_dim == appended
        and g.img_feat_frames == frames
        and g.img_feat_proj is proj
        and g.cell.weight_ih.shape == (3 * H, 14 + appended)
        and hasattr(g, "img_norm")                 # LayerNorm in BOTH modes
        and hasattr(g, "img_proj") == proj         # projector only when asked
        and tuple(g.img_norm.weight.shape) == (SRC,)
    )
    check("U1", f"{label}: cell input 14+{appended}, img_norm present, "
                f"img_proj {'present' if proj else 'ABSENT'}",
          ok, f"input_dim={g.input_dim}, params={sum(p.numel() for p in g.parameters())}")

g_off = PoseGRU(hidden_dim=H, mode="residual", input_mode="pose_delta")
check("U2", "F-off arm untouched: 14-wide, no img_norm, no img_proj",
      g_off.input_dim == 14 and not hasattr(g_off, "img_norm")
      and not hasattr(g_off, "img_proj") and g_off.img_feat_frames == 0
      and g_off.img_feat_proj is False)

check("U3", "proj=True param count is frames-dependent (projector cols) and "
            "proj=False is FAR larger (no bottleneck)",
      sum(p.numel() for p in mk(1, True).parameters())
      < sum(p.numel() for p in mk(2, True).parameters())
      < sum(p.numel() for p in mk(1, False).parameters())
      < sum(p.numel() for p in mk(2, False).parameters()),
      " < ".join(str(sum(p.numel() for p in mk(f, p_).parameters()))
                 for f, p_ in [(1, True), (2, True), (1, False), (2, False)]))

# proj=False must NOT reintroduce the zero-init projector by another name:
# the appended columns of weight_ih are default-init, hence nonzero.
g = mk(2, False)
check("U4", "proj=False: the 2048 appended weight_ih columns are default-init "
            "(nonzero) — there is no zero gate",
      float(g.cell.weight_ih[:, 14:].abs().sum()) > 0.0)
check("U5", "proj=True: img_proj is still zero-init (gate intact)",
      float(mk(2, True).img_proj.weight.abs().sum()) == 0.0
      and float(mk(2, True).img_proj.bias.abs().sum()) == 0.0)

for bad, why in [
    (dict(img_feat_frames=3), "frames=3 rejected"),
    (dict(img_feat_frames=0), "frames=0 rejected"),
]:
    try:
        PoseGRU(hidden_dim=H, img_feat="input", img_feat_src_dim=SRC, **bad)
        raised = False
    except AssertionError:
        raised = True
    check("U6", f"ctor assert: {why}", raised)

try:  # img_feat_dim is only meaningful (and only validated) when proj=True
    PoseGRU(hidden_dim=H, img_feat="input", img_feat_dim=0, img_feat_src_dim=SRC)
    raised = False
except AssertionError:
    raised = True
check("U7", "ctor assert: img_feat_dim=0 rejected when proj=True", raised)
g = PoseGRU(hidden_dim=H, img_feat="input", img_feat_dim=0, img_feat_proj=False,
            img_feat_src_dim=SRC, img_feat_frames=1)
check("U8", "img_feat_dim is IGNORED when proj=False (appended width comes "
            "from the encoder), not silently used as 0",
      g.input_dim == 7 + SRC and g.img_feat_dim == SRC)

# =============================================================================
banner("STAGE IDENT — residual init-equivalence in both projector modes")
# =============================================================================
torch.manual_seed(0)
# The quaternion block MUST be unit-norm: forward() ends in F.normalize(q), so
# an unnormalized test pose would fail the identity check for reasons that have
# nothing to do with the F lever. The real fed-back pose is always unit-quat.
pose = torch.cat(
    [torch.randn(B, 3), torch.nn.functional.normalize(torch.randn(B, 4), dim=-1)], dim=-1
)
delta = torch.cat(
    [torch.randn(B, 3), torch.nn.functional.normalize(torch.randn(B, 4), dim=-1)], dim=-1
)
gru_in = torch.cat([pose, delta], dim=-1)

for label, (frames, proj, _) in ARMS.items():
    g = mk(frames, proj)
    feats = torch.randn(B, frames * SRC) * 5.0   # deliberately large
    out, hid = g(gru_in, None, img_feat=feats)
    # residual + zero head => output IS the input pose, regardless of what the
    # image branch did to the hidden state. Translation is a pure passthrough
    # and must be BITWISE identical. The quaternion goes through a final
    # F.normalize, which is not a fp32 fixed point even on an already-unit
    # quat, so it is held to 1 ULP rather than bit-equality — demanding
    # bit-equality there would flag float rounding as a lever bug.
    dt = float((out[:, :3] - pose[:, :3]).abs().max())
    dq = float((out[:, 3:] - pose[:, 3:]).abs().max())
    check("I1", f"{label}: output == fed-back pose at init "
                f"(t bitwise, q to renormalize noise)",
          torch.equal(out[:, :3], pose[:, :3]) and dq < 1e-6,
          f"max|dt|={dt:.3e}, max|dq|={dq:.3e}")

# The direct-mode contrast: proj=True is still init-blind to features because
# the projector zeroes them; proj=False is NOT. This is what "init-equivalence
# holds for a different reason" means, made falsifiable.
for proj in (True, False):
    g = mk(2, proj, mode="direct")
    f1 = torch.randn(B, 2 * SRC)
    f2 = torch.randn(B, 2 * SRC)
    o1, _ = g(gru_in, None, img_feat=f1)
    o2, _ = g(gru_in, None, img_feat=f2)
    same = torch.equal(o1, o2)
    check("I2", f"mode=direct, proj={proj}: output {'INVARIANT' if proj else 'VARIES'} "
                f"with the image features at init",
          same is proj, f"max|d|={float((o1 - o2).abs().max()):.3e}")

# =============================================================================
banner("STAGE GATE — does image content actually reach the cell?")
# =============================================================================
for label, (frames, proj, _) in ARMS.items():
    g = mk(frames, proj)
    fa = torch.randn(B, frames * SRC)
    fb = torch.randn(B, frames * SRC)
    _, ha = g(gru_in, None, img_feat=fa)
    _, hb = g(gru_in, None, img_feat=fb)
    moved = float((ha - hb).abs().max())
    # proj=True: zero projector => the hidden cannot see the features yet.
    # proj=False: no gate => it must.
    ok = (moved == 0.0) if proj else (moved > 1e-6)
    check("G1", f"{label}: hidden {'INVARIANT to' if proj else 'MOVES with'} "
                f"the features at init", ok, f"max|dh|={moved:.3e}")

# F0 must ignore the second frame entirely: it is not merely down-weighted,
# it is never read. Feed a 1-frame module and confirm the width contract.
g = mk(1, False)
try:
    g(gru_in, None, img_feat=torch.randn(B, 2 * SRC))   # F1-shaped features
    wrong_width_ok = True
except Exception:
    wrong_width_ok = False
check("G2", "F0 module REJECTS 2-frame features (width contract enforced by "
            "the reshape, not silently truncated)", not wrong_width_ok)

# Same seed, same pose input, F0 vs F1 with the current frame IDENTICAL:
# the outputs must differ only because F1 additionally consumed frame t-1.
cur = torch.randn(B, SRC)
prev = torch.randn(B, SRC)
g1 = mk(1, False)
g2 = mk(2, False)
_, h_f0 = g1(gru_in, None, img_feat=cur)
_, h_f1 = g2(gru_in, None, img_feat=torch.cat([cur, prev], dim=-1))
check("G3", "F0 and F1 are genuinely different functions of the same current "
            "frame (F1 mixes in t-1)", not torch.equal(h_f0, h_f1[:, : h_f0.shape[1]]))

# =============================================================================
banner("STAGE GRAD — the head is the gate; features are reachable once unlocked")
# =============================================================================
for label, (frames, proj, _) in ARMS.items():
    g = mk(frames, proj)
    feats = torch.randn(B, frames * SRC)
    out, _ = g(gru_in, None, img_feat=feats)
    out.sum().backward()
    locked = (
        float(g.cell.weight_ih.grad.abs().sum()) == 0.0
        and float(g.img_norm.weight.grad.abs().sum()) == 0.0
        and (not proj or float(g.img_proj.weight.grad.abs().sum()) == 0.0)
        and float(g.head.weight.grad.abs().sum()) > 0.0   # head itself DOES learn
    )
    check("D1", f"{label}: zero head locks cell/img_norm/img_proj at init while "
                f"the head accumulates gradient", locked)

for label, (frames, proj, _) in ARMS.items():
    g = mk(frames, proj)
    with torch.no_grad():                        # unlock the head
        g.head.weight.normal_(0, 0.05)
    feats = torch.randn(B, frames * SRC)
    out, _ = g(gru_in, None, img_feat=feats)
    out.sum().backward()
    img_path = (
        float(g.img_proj.weight.grad.abs().sum()) if proj
        else float(g.cell.weight_ih.grad[:, 14:].abs().sum())
    )
    check("D2", f"{label}: with the head unlocked, gradient reaches the image "
                f"path ({'img_proj' if proj else 'weight_ih[:, 14:]'})",
          img_path > 0.0, f"|g|={img_path:.3e}")
    # SECOND gate, proj=True only: img_norm sits BEHIND the zero-init
    # projector, so unlocking the head is not enough — d(loss)/d(img_norm)
    # routes through W_proj = 0. proj=False has no such stage: one unlock and
    # the whole image path is live. This is a real behavioural difference in
    # how quickly each arm can start adapting to appearance.
    norm_g = float(g.img_norm.weight.grad.abs().sum())
    state = (
        "STILL LOCKED behind the zero projector" if proj
        else "LIVE immediately (single gate)"
    )
    check("D3", f"{label}: img_norm is {state}",
          (norm_g == 0.0) if proj else (norm_g > 0.0), f"|g_norm|={norm_g:.3e}")

# ...and the projector arm does unlock img_norm once img_proj leaves zero.
g = mk(2, True)
with torch.no_grad():
    g.head.weight.normal_(0, 0.05)
    g.img_proj.weight.normal_(0, 1e-3)
out, _ = g(gru_in, None, img_feat=torch.randn(B, 2 * SRC))
out.sum().backward()
check("D4", "proj=True: img_norm becomes reachable once img_proj leaves zero "
            "(the gate is temporary, not structural)",
      float(g.img_norm.weight.grad.abs().sum()) > 0.0)

# =============================================================================
banner("STAGE SNIFF — load_model round-trip for projector-less checkpoints")
# =============================================================================
inf = float("inf")  # noqa: F841  used by MODEL_STR


def roundtrip(gru, iters=1, mode="residual", input_mode="pose_delta"):
    """Save a pose_gru-only ckpt and let load_model rebuild it from scratch."""
    args = SimpleNamespace(
        model=MODEL_STR, pose_gru_mode=mode, pose_gru_hidden_dim=H,
        pose_gru_input=input_mode, pose_gru_iters=iters,
    )
    fd, path = tempfile.mkstemp(suffix=".pth", dir=os.environ.get("TMPDIR", "/tmp"))
    os.close(fd)
    torch.save({"model": {f"pose_gru.{k}": v for k, v in gru.state_dict().items()},
                "args": args}, path)
    try:
        net = load_model(path, device="cpu", verbose=False)
        return net.pose_gru, {k: v for k, v in gru.state_dict().items()}
    finally:
        os.remove(path)


for label, (frames, proj, appended) in ARMS.items():
    src_gru = mk(frames, proj)
    got, ref = roundtrip(src_gru, iters=8)
    ok = (
        got.img_feat == "input"
        and got.img_feat_frames == frames
        and got.img_feat_proj is proj
        and got.img_feat_dim == appended
        and got.iters == 8
        and got.input_mode == "pose_delta"
        and got.cell.weight_ih.shape[1] == 14 + appended
        and torch.equal(got.cell.weight_ih, ref["cell.weight_ih"])
        and torch.equal(got.img_norm.weight, ref["img_norm.weight"])
        and (not proj or torch.equal(got.img_proj.weight, ref["img_proj.weight"]))
    )
    check("N1", f"{label}: round-trip restores frames/proj/appended width AND "
                f"the trained weights", ok,
          f"sniffed frames={got.img_feat_frames}, proj={got.img_feat_proj}, "
          f"appended={got.img_feat_dim}")
    del got

got, _ = roundtrip(PoseGRU(hidden_dim=H, input_mode="pose_delta"))
check("N2", "legacy F-off ckpt sniffs unchanged (no img_norm -> img_feat=none)",
      got.img_feat == "none" and got.img_feat_proj is False
      and got.cell.weight_ih.shape[1] == 14 and got.iters == 1)
del got

got, _ = roundtrip(PoseGRU(hidden_dim=H, input_mode="pose"))
check("N3", "legacy 7-wide pose-only ckpt still sniffs input_mode=pose",
      got.img_feat == "none" and got.input_mode == "pose"
      and got.cell.weight_ih.shape[1] == 7)
del got

# A projector-less F ckpt loaded into a plain module must HARD-FAIL rather than
# silently evaluate a random GRU — the guarantee the sniff fix protects.
from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig  # noqa: F401,E402

clash = eval(MODEL_STR)
clash.enable_pose_gru(hidden_dim=H, mode="residual", input_mode="pose_delta")
raw_sd = {f"pose_gru.{k}": v for k, v in mk(1, False).state_dict().items()}
try:
    clash.load_state_dict(raw_sd, strict=False)
    hard_fail = False
except RuntimeError as e:
    hard_fail = "pose_gru" in str(e)
check("N4", "loading a proj=False GRU into a 14-wide module HARD-FAILS",
      bool(hard_fail))
del clash

print("\n" + "=" * 78)
print(f"RESULT: {'ALL CHECKS PASSED' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
print("=" * 78)
sys.exit(1 if FAILS else 0)
