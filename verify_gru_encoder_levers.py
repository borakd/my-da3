#!/usr/bin/env python
"""
verify_gru_encoder_levers.py — CPU verification of the F-SOURCE lever
(pose_gru_img_feat_src): where the per-step image feature entering the
PoseGRU comes from — "pooled" (CUT3R pre-ray token mean, the legacy path),
a frozen "resnet18" / "dinov2_vits14" encoder on the raw image, or "corr"
(hand-crafted correlation/flow statistics between consecutive token sets).

Runs WITHOUT a GPU and WITHOUT building the full ARCroco3DStereo net, so it
fits on a login node (every tensor is tiny: hidden 16, batch 2, grid 12x20).

Groups:
  INIT   init-equivalence per source arm (residual mode, proj=True): module
         output == quat-normalized input pose exactly, and the HIDDEN
         trajectory bitwise equals an img_feat="none" module's hidden with
         the same base weights (the zero-init projector feeds exactly-zero
         appended columns); ctor width/assert checks
  CORR   corr_motion_stats: identical tokens -> dx=dy=0, m=1; a circular
         one-column shift -> per-cell mean dx == +-1/W' with the cur->prev
         sign convention; shape (B, 54), finite
  SNIFF  _sniff_pose_gru_config round-trip on SYNTHETIC state dicts for
         every arm (pooled F1, resnet18 F0/F1, dinov2 F0/F1, corr F1c,
         proj=False pooled F1) + the hard-fail cases (contradictory args
         src, img_norm width matching nothing)
  ENC    FrozenImageEncoder(pretrained=False) smoke: (2,3,192,320) in [-1,1]
         -> (2,512)/(2,384) fp32 detached, params frozen, eval-locked even
         after .train(True)

Run on a login node:  python verify_gru_encoder_levers.py
"""
import os
import sys

# Self-locating: derive the checkout from THIS file, never a hardcoded path.
# sys.path.insert(0, <nonexistent>) SILENTLY SUCCEEDS, so a stale hardcoded
# worktree makes imports fall through to PYTHONPATH -- letting you verify a
# DIFFERENT checkout than the file you are editing, with no warning. Assert.
WORKTREE = os.path.dirname(os.path.abspath(__file__))
assert os.path.isfile(
    os.path.join(WORKTREE, "src", "CUT3R", "src", "train_cut3r_baseline.py")
), f"not a my-da3 checkout: {WORKTREE}"

HID = 16          # tiny hidden: keeps every GRU tensor login-node cheap
B = 2             # batch
FDIM = 8          # appended (projected) feature width
ENC = 32          # synthetic enc_embed_dim for the "pooled" arms (sniff is
                  # a pure fn of (state_dict, args, enc_embed_dim) -- tiny ok)
GRID = (12, 20)   # (H', W') token grid at 192x320 / patch 16
SEED = 20260811


def setup_paths(worktree):
    for p in [
        worktree,
        os.path.join(worktree, "src"),
        os.path.join(worktree, "src/CUT3R"),
        os.path.join(worktree, "src/CUT3R/src"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)


def main():
    setup_paths(WORKTREE)
    import types

    import torch
    import torch.nn.functional as F
    import dust3r.heads  # noqa: F401  MUST precede dust3r.utils.camera
    from dust3r.model import PoseGRU, _sniff_pose_gru_config
    from dust3r.img_encoders import (
        CORR_FEAT_DIM,
        ENCODER_DIMS,
        FrozenImageEncoder,
        corr_motion_stats,
    )

    torch.manual_seed(SEED)
    torch.set_num_threads(4)  # login node: bound the CPU-time burn rate

    CHECKS = []

    def banner(t):
        print("\n" + "=" * 96 + f"\n {t}\n" + "=" * 96, flush=True)

    def check(gid, desc, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        print(
            f"  [{tag}] {gid}: {desc}" + (f"\n         -> {detail}" if detail else ""),
            flush=True,
        )
        CHECKS.append((gid, desc, bool(ok), detail))

    def maxdiff(a, b):
        return float((a.double() - b.double()).abs().max())

    # Arm table: (tag, src, frames, src_dim, n_blocks). n_blocks is what the
    # appended feature reshapes over: per-frame sources use one block per
    # frame; corr is ONE 54-wide block describing the (cur, prev) PAIR.
    ARMS = [
        ("pooled-F1", "pooled", 2, ENC, 2),
        ("resnet18-F0r", "resnet18", 1, ENCODER_DIMS["resnet18"], 1),
        ("dinov2-F1d", "dinov2_vits14", 2, ENCODER_DIMS["dinov2_vits14"], 2),
        ("corr-F1c", "corr", 2, CORR_FEAT_DIM, 1),
    ]

    def build_arm(src, frames, proj=True):
        kw = dict(
            hidden_dim=HID,
            mode="residual",
            input_mode="pose_delta",
            img_feat="input",
            img_feat_dim=FDIM,
            img_feat_frames=frames,
            img_feat_proj=proj,
            img_feat_src_dim=ENC,
            img_feat_src=src,
        )
        if src in ENCODER_DIMS:
            kw["img_encoder_pretrained"] = False  # random init: no weight file
        return PoseGRU(**kw)

    # =========================================================================
    banner("GROUP INIT — per-source init-equivalence (residual, proj=True, zero head)")
    # =========================================================================
    torch.manual_seed(SEED)
    g_off = PoseGRU(hidden_dim=HID, mode="residual", input_mode="pose_delta")
    base_w = g_off.cell.weight_ih.shape[1]
    check("N0", "F-off reference module: cell input width 14 (pose_delta), zero head",
          base_w == 14
          and float(g_off.head.weight.abs().sum()) == 0.0
          and float(g_off.head.bias.abs().sum()) == 0.0)

    for tag, src, frames, src_dim, n_blocks in ARMS:
        torch.manual_seed(SEED + 1)  # deliberately != g_off's seed: the graft
        g_f = build_arm(src, frames)  # below must carry the equivalence alone
        wants_enc = src in ENCODER_DIMS
        ok_w = (
            g_f.cell.weight_ih.shape[1] == base_w + FDIM
            and tuple(g_f.img_norm.weight.shape) == (src_dim,)
            and tuple(g_f.img_proj.weight.shape) == (FDIM, src_dim * n_blocks)
            and float(g_f.img_proj.weight.abs().sum()) == 0.0
            and float(g_f.img_proj.bias.abs().sum()) == 0.0
            and (getattr(g_f, "img_encoder", None) is not None) == wants_enc
            and float(g_f.head.weight.abs().sum()) == 0.0
        )
        check(
            f"N1-{tag}",
            f"widths: cell {base_w}+{FDIM}, img_norm({src_dim}), "
            f"img_proj({FDIM},{src_dim}*{n_blocks}) zero-init, "
            f"img_encoder {'present' if wants_enc else 'absent'}",
            ok_w,
            f"cell_in={g_f.cell.weight_ih.shape[1]}, "
            f"norm={tuple(g_f.img_norm.weight.shape)}, "
            f"proj={tuple(g_f.img_proj.weight.shape)}, "
            f"enc={getattr(g_f, 'img_encoder', None) is not None}",
        )
        # Graft the F-off base weights: base input columns + hidden path +
        # biases. The appended columns keep their own (default) init -- they
        # must be silenced purely by the zero-init projector.
        with torch.no_grad():
            g_f.cell.weight_ih[:, :base_w].copy_(g_off.cell.weight_ih)
            g_f.cell.weight_hh.copy_(g_off.cell.weight_hh)
            g_f.cell.bias_ih.copy_(g_off.cell.bias_ih)
            g_f.cell.bias_hh.copy_(g_off.cell.bias_hh)

        torch.manual_seed(SEED + 2)
        h_f, h_off = None, None
        ok_out, ok_hid, worst_out, worst_hid = True, True, 0.0, 0.0
        with torch.no_grad():
            for _ in range(3):
                gin = torch.randn(B, 14)
                feat = torch.randn(B, src_dim * n_blocks)
                exp = torch.cat([gin[:, :3], F.normalize(gin[:, 3:7], dim=-1)], -1)
                out_f, h_f = g_f(gin, h_f, img_feat=feat)
                out_off, h_off = g_off(gin, h_off)
                del out_off  # same identity by N0; the F arm is under test
                ok_out &= torch.equal(out_f, exp)
                ok_hid &= torch.equal(h_f, h_off)
                worst_out = max(worst_out, maxdiff(out_f, exp))
                worst_hid = max(worst_hid, maxdiff(h_f, h_off))
        check(f"N2-{tag}", "output == quat-normalized input pose EXACTLY "
              "(zero-init residual head)", ok_out, f"maxdiff {worst_out:.1e}")
        check(f"N3-{tag}", "hidden trajectory bitwise == F-off module's hidden "
              "(zero features through the zero projector)", ok_hid,
              f"maxdiff {worst_hid:.1e}")
        del g_f

    raised = {}
    for name, kw in (
        ("corr_frames1", dict(img_feat="input", img_feat_src="corr",
                              img_feat_frames=1, img_feat_src_dim=ENC)),
        ("bogus_src", dict(img_feat="input", img_feat_src="bogus",
                           img_feat_src_dim=ENC)),
    ):
        try:
            PoseGRU(hidden_dim=HID, input_mode="pose_delta", **kw)
            raised[name] = False
        except (AssertionError, ValueError, KeyError):
            raised[name] = True
    check("N4", "ctor asserts: corr with frames=1 refused, unknown src refused",
          all(raised.values()), str(raised))

    # =========================================================================
    banner("GROUP CORR — corr_motion_stats on the 12x20 token grid")
    # =========================================================================
    Hp, Wp = GRID
    N = Hp * Wp
    C = 32  # token dim: wide enough that no random off-match beats the exact one
    torch.manual_seed(SEED + 3)
    toks = torch.randn(B, N, C)

    # (a) prev == cur: every token matches itself -> dx = dy = 0, m = 1.
    out_same = corr_motion_stats(toks, toks, GRID)
    ok_shape = tuple(out_same.shape) == (B, CORR_FEAT_DIM)
    cells = out_same[:, :48].reshape(B, 16, 3)  # row-major cells, (dx,dy,m) fastest
    ok_dx0 = bool((cells[:, :, 0] == 0).all())
    ok_dy0 = bool((cells[:, :, 1] == 0).all())
    ok_m1 = bool(torch.allclose(cells[:, :, 2], torch.ones(B, 16), atol=1e-4))
    # The 6 global dims are mean/std of dx, dy, m; for a self-match that is
    # five (near-)zeros and one 1.0 (mean m). Sorting makes the check
    # independent of the exact interleaving order of the six.
    g_sorted, _ = out_same[:, 48:].sort(dim=1)
    ok_glob = bool(torch.allclose(
        g_sorted, torch.tensor([0.0, 0, 0, 0, 0, 1]).expand(B, 6), atol=1e-4
    ))
    check("C1", "prev == cur: shape (B,54), per-cell dx = dy = 0 exactly, m = 1, "
          "globals = five zeros + mean-m 1",
          ok_shape and ok_dx0 and ok_dy0 and ok_m1 and ok_glob,
          f"shape={tuple(out_same.shape)}, dx0={ok_dx0}, dy0={ok_dy0}, "
          f"m1={ok_m1}, glob={ok_glob}")

    # (b) circular one-column shift. prev(y,x) = cur(y, x-1) puts cur(y,x)'s
    # exact match at prev column x+1, so dx = (x_j - x_i)/W' = +1/W' -- the
    # cur->prev convention (~ negative optical flow of content moving right).
    # The wrap column's match is x=0 (dx = -(W'-1)/W'), which lands in ONE
    # cell column of the 4x4 partition (5-col cells at W'=20); every cell
    # NOT containing the wrap column must read mean dx == +-1/W' exactly-ish.
    grid = toks.reshape(B, Hp, Wp, C)
    inv_w = 1.0 / Wp
    ok_shift = True
    det = []
    for shift, sign, wrap_cx in ((1, +1.0, 3), (-1, -1.0, 0)):
        prev = torch.roll(grid, shifts=shift, dims=2).reshape(B, N, C)
        out = corr_motion_stats(toks, prev, GRID)
        c = out[:, :48].reshape(B, 4, 4, 3)  # (B, cell_y, cell_x, [dx,dy,m])
        keep = [cx for cx in range(4) if cx != wrap_cx]
        dx_in = c[:, :, keep, 0]
        ok_dx = bool(torch.allclose(
            dx_in, torch.full_like(dx_in, sign * inv_w), atol=1e-5
        ))
        ok_dy = bool((c[:, :, :, 1] == 0).all())  # same-row matches everywhere
        ok_m = bool(torch.allclose(c[:, :, :, 2], torch.ones(B, 4, 4), atol=1e-4))
        ok_shift &= ok_dx and ok_dy and ok_m
        det.append(
            f"shift{shift:+d}: dx(interior)={float(dx_in.mean()):+.5f} "
            f"want {sign * inv_w:+.5f} ok={ok_dx}, dy0={ok_dy}, m1={ok_m}"
        )
    check("C2", "one-column circular shift: non-wrap cells mean dx == +-1/W' "
          "with the cur->prev sign, dy = 0, m = 1", ok_shift, "; ".join(det))

    # (c) independent random token sets: shape + finiteness.
    out_rand = corr_motion_stats(toks, torch.randn(B, N, C), GRID)
    check("C3", "random pair: output (B, 54), float32, all finite",
          tuple(out_rand.shape) == (B, CORR_FEAT_DIM)
          and out_rand.dtype == torch.float32
          and bool(torch.isfinite(out_rand).all()),
          f"shape={tuple(out_rand.shape)}, dtype={out_rand.dtype}")

    # =========================================================================
    banner("GROUP SNIFF — _sniff_pose_gru_config round-trip on synthetic state dicts")
    # =========================================================================
    def make_args(src, frames, proj):
        return types.SimpleNamespace(
            pose_gru=True,
            pose_gru_mode="residual",
            pose_gru_input="pose_delta",
            pose_gru_hidden_dim=HID,
            pose_gru_iters=8,
            pose_gru_img_feat="input",
            pose_gru_img_feat_dim=FDIM,
            pose_gru_img_feat_frames=frames,
            pose_gru_img_feat_proj=proj,
            pose_gru_img_feat_src=src,
            pose_gru_img_encoder_weights=None,
        )

    SNIFF_ARMS = [
        # (tag, src, frames, proj, appended width the sniff must report)
        ("pooled-F1", "pooled", 2, True, FDIM),
        ("resnet18-F0r", "resnet18", 1, True, FDIM),
        ("resnet18-F1r", "resnet18", 2, True, FDIM),
        ("dinov2-F0d", "dinov2_vits14", 1, True, FDIM),
        ("dinov2-F1d", "dinov2_vits14", 2, True, FDIM),
        ("corr-F1c", "corr", 2, True, FDIM),
        ("pooled-F1-noproj", "pooled", 2, False, 2 * ENC),
    ]

    arm_sds = {}  # tag -> (prefixed sd, args); reused by the hard-fail cases
    for tag, src, frames, proj, app_dim in SNIFF_ARMS:
        torch.manual_seed(SEED + 4)
        g = build_arm(src, frames, proj=proj)
        sd = {f"pose_gru.{k}": v for k, v in g.state_dict().items()}
        args = make_args(src, frames, proj)
        arm_sds[tag] = (sd, args)
        try:
            cfg = _sniff_pose_gru_config(sd, args, ENC)
        except Exception as e:
            check(f"S1-{tag}", "sniff returns the constructor config", False,
                  f"raised {type(e).__name__}: {e}")
            del g
            continue
        expected = dict(
            hidden_dim=HID, mode="residual", input_mode="pose_delta",
            img_feat="input", img_feat_dim=app_dim, img_feat_frames=frames,
            img_feat_proj=proj, iters=8,
        )
        bad = [
            f"{k}: got {cfg.get(k, '<missing>')!r} want {v!r}"
            for k, v in expected.items()
            if cfg.get(k) != v
        ]
        # img_feat_src may legitimately be omitted for the legacy pooled arms.
        if cfg.get("img_feat_src", "pooled") != src:
            bad.append(f"img_feat_src: got {cfg.get('img_feat_src')!r} want {src!r}")
        # Eval must NEVER touch the pretrained file: ckpt["model"] overwrites.
        if cfg.get("img_encoder_pretrained", False):
            bad.append("img_encoder_pretrained: got True, load path must pass False")
        # Functional round-trip: a module rebuilt FROM the sniffed config must
        # reproduce the exact state_dict skeleton (keys + shapes) it was
        # sniffed from -- i.e. load_state_dict of the ckpt would succeed.
        skel_ok, skel_msg = False, ""
        try:
            kw2 = dict(cfg)
            kw2["img_encoder_pretrained"] = False  # never read weight files here
            kw2.setdefault("img_feat_src_dim", ENC)  # enable_pose_gru supplies this
            g2 = PoseGRU(**kw2)
            got = {k: tuple(v.shape) for k, v in g2.state_dict().items()}
            want = {k: tuple(v.shape) for k, v in g.state_dict().items()}
            skel_ok = got == want
            if not skel_ok:
                miss = sorted(set(want) - set(got))[:3]
                extra = sorted(set(got) - set(want))[:3]
                shp = [k for k in set(got) & set(want) if got[k] != want[k]][:3]
                skel_msg = f"skeleton mismatch: miss={miss} extra={extra} shapes={shp}"
            del g2
        except Exception as e:
            skel_msg = f"rebuild from sniffed cfg raised {type(e).__name__}: {e}"
        check(
            f"S1-{tag}",
            "sniff == constructor config AND rebuilt module matches the "
            "state-dict skeleton",
            not bad and skel_ok,
            "; ".join(bad + ([skel_msg] if skel_msg else [])) or f"cfg={cfg}",
        )
        del g

    # Hard-fail cases: the sniff must RAISE, never guess.
    def expect_raise(gid, desc, sd, args):
        try:
            cfg = _sniff_pose_gru_config(sd, args, ENC)
            check(gid, desc, False, f"did NOT raise; returned {cfg}")
        except Exception as e:
            check(gid, desc, True, f"raised {type(e).__name__}")

    sd_r, _ = arm_sds["resnet18-F1r"]
    expect_raise(
        "S2", "resnet18 encoder keys + args src='dinov2_vits14' HARD-FAILS",
        dict(sd_r), make_args("dinov2_vits14", 2, True),
    )
    sd_p, args_p = arm_sds["pooled-F1"]
    sd_bad = dict(sd_p)
    sd_bad["pose_gru.img_norm.weight"] = torch.ones(999)
    sd_bad["pose_gru.img_norm.bias"] = torch.zeros(999)
    sd_bad["pose_gru.img_proj.weight"] = torch.zeros(FDIM, 2 * 999)
    expect_raise(
        "S3", "img_norm width 999 (matches no source: enc/512/384/54) HARD-FAILS",
        sd_bad, args_p,
    )
    expect_raise(
        "S4", "pooled-shaped weights + args src='resnet18' HARD-FAILS "
        "(no encoder keys, img_norm width != 512)",
        dict(sd_p), make_args("resnet18", 2, True),
    )

    # =========================================================================
    banner("GROUP ENC — FrozenImageEncoder(pretrained=False) smoke")
    # =========================================================================
    for name in ("resnet18", "dinov2_vits14"):
        dim = ENCODER_DIMS[name]
        enc = FrozenImageEncoder(name, pretrained=False)
        enc.train(True)  # must be a no-op: the module is eval-locked
        frozen = all(not p.requires_grad for p in enc.parameters())
        locked = not enc.training and all(not m.training for m in enc.modules())
        check(f"E1-{name}", "params frozen, module eval-locked even after "
              ".train(True)", frozen and locked,
              f"requires_grad any={not frozen}, training={enc.training}, "
              f"out_dim={enc.out_dim}")
        torch.manual_seed(SEED + 5)
        img = torch.rand(B, 3, 192, 320) * 2 - 1  # CUT3R's [-1,1] normalization
        out = enc(img)
        check(
            f"E2-{name}",
            f"forward (2,3,192,320) -> (2,{dim}) fp32, detached, finite",
            tuple(out.shape) == (B, dim)
            and out.dtype == torch.float32
            and not out.requires_grad
            and bool(torch.isfinite(out).all())
            and enc.out_dim == dim,
            f"shape={tuple(out.shape)}, dtype={out.dtype}, "
            f"requires_grad={out.requires_grad}",
        )
        # The ImageNet renorm buffers must be non-persistent: encoder weights
        # ride every ckpt, but constants must not enter the state_dict. No
        # genuine resnet18/vit-s tensor has 3 elements, so any 3-element
        # entry can only be a leaked mean/std buffer.
        leaked = [k for k, v in enc.state_dict().items() if v.numel() == 3]
        check(f"E3-{name}", "ImageNet mean/std buffers are NOT in the state_dict "
              "(persistent=False)", not leaked, f"3-element keys: {leaked}")
        del enc

    # =========================================================================
    banner("SUMMARY")
    # =========================================================================
    n_fail = sum(1 for _, _, ok, _ in CHECKS if not ok)
    for gid, desc, ok, _ in CHECKS:
        print(f"  {'PASS' if ok else 'FAIL':4s}  {gid:18s}  {desc}")
    print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
