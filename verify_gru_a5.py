#!/usr/bin/env python
"""verify_gru_a5.py — correctness suite for the A5 "anchored A3" PoseGRU mode.

A5 (`pose_gru_mode: split_anchor`) regresses TRANSLATION directly (as A3) and
ANCHORS ROTATION on the fed-back quaternion (as A4). It adds NO parameters and
NO config keys — the head is the same shared nn.Linear(hidden, 7), with only
its rotation rows (3:7) zero-init.

Stages
  P  parity   — 'residual' and 'direct' are BITWISE unchanged vs the v2 worktree
                code (the A5 edit must not touch the existing arms)
  U  unit     — construction, parameter-set identity, zero-init placement
  A  algebra  — forward is exactly (direct translation, anchored rotation), and
                each half is BITWISE equal to the corresponding A3/A4 half
  I  init     — an untrained A5 emits q == normalize(anchor q) exactly, while t
                is a free regression (NOT the anchor) — quantified
  G  grad     — both halves of the shared head receive gradient
  E  errors   — unknown modes hard-fail at ctor AND in forward (no silent A3)
  C  ckpt     — an A4 checkpoint loads into an A5 module (identical param set),
                load_model restores mode from args, and a ckpt whose args lack
                pose_gru_mode is REFUSED instead of silently defaulting
  Y  config   — the two A5 configs differ from their A4 twins only in
                pose_gru_mode + exp_name

Usage:  python verify_gru_a5.py            (CPU is fine; ckpt stage needs RAM)
Submit: sbatch verify_gru_a5.sbatch
"""
import argparse
import os
import subprocess
import sys

WORKTREE = "/scratch/bdursun25/cuteanything/captain_gru_v3"
V2_WORKTREE = "/scratch/bdursun25/cuteanything/captain_gru_v2"
# The 16 grid runs were reorganised out of captain_cut3r_sim3rmse/ into
# captain_gru_overfit/ on 2026-07-30; fall back to the old path for safety.
CKPT_CANDIDATES = [
    "/scratch/bdursun25/cuteanything/checkpoints/captain_gru_overfit/"
    "captain_gru_v2_a4_g1/checkpoint-final.pth",
    "/scratch/bdursun25/cuteanything/checkpoints/captain_cut3r_sim3rmse/"
    "captain_gru_v2_a4_g1/checkpoint-final.pth",
]
CKPT = next((p for p in CKPT_CANDIDATES if os.path.isfile(p)), CKPT_CANDIDATES[0])
CFG_DIR = f"{WORKTREE}/src/CUT3R/config"

H, B = 128, 5          # hidden dim, batch
SEED = 1234


def setup_paths(wt):
    for p in (wt, f"{wt}/src", f"{wt}/src/CUT3R", f"{wt}/src/CUT3R/src"):
        while p in sys.path:
            sys.path.remove(p)
    for p in (f"{wt}/src/CUT3R/src", f"{wt}/src/CUT3R", f"{wt}/src", wt):
        sys.path.insert(0, p)


def fixed_inputs(torch):
    """Deterministic (gru_in 14-d pose_delta, hidden) pair."""
    g = torch.Generator().manual_seed(SEED)
    pose = torch.randn(B, 7, generator=g)
    pose[:, 3:7] = torch.nn.functional.normalize(pose[:, 3:7], dim=-1)
    delta = torch.randn(B, 7, generator=g)
    delta[:, 3:7] = torch.nn.functional.normalize(delta[:, 3:7], dim=-1)
    hidden = torch.randn(B, H, generator=g) * 0.05
    return torch.cat([pose, delta], dim=-1), hidden


def build(torch, PoseGRU, mode):
    torch.manual_seed(SEED)
    return PoseGRU(hidden_dim=H, mode=mode, input_mode="pose_delta")


def dump_forward(out_path, wt, mode):
    """Subprocess entry: run one PoseGRU forward under worktree `wt`'s code."""
    setup_paths(wt)
    import torch
    import dust3r.heads  # noqa: F401  MUST precede dust3r.utils.camera
    from dust3r.model import PoseGRU

    gru = build(torch, PoseGRU, mode)
    gru_in, hidden = fixed_inputs(torch)
    with torch.no_grad():
        out, new_h = gru(gru_in, hidden)
    torch.save(
        {"out": out, "hidden": new_h, "sd": {k: v.clone() for k, v in gru.state_dict().items()}},
        out_path,
    )


def main():
    setup_paths(WORKTREE)
    import torch
    import dust3r.heads  # noqa: F401
    from dust3r.model import PoseGRU

    CHECKS = []

    def banner(t):
        print("\n" + "=" * 96 + f"\n {t}\n" + "=" * 96, flush=True)

    def check(gid, desc, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {gid}: {desc}"
              + (f"\n         -> {detail}" if detail else ""), flush=True)
        CHECKS.append((gid, desc, bool(ok), detail))

    def maxdiff(a, b):
        return float((a.double() - b.double()).abs().max())

    scratch = os.path.join(WORKTREE, "logs")
    os.makedirs(scratch, exist_ok=True)
    gru_in, hidden = fixed_inputs(torch)
    anchor_t, anchor_q = gru_in[:, :3], gru_in[:, 3:7]

    # ---------------------------------------------------------------- P ----
    banner("STAGE P — existing modes bitwise unchanged vs the v2 worktree code")
    dumps = {}
    for wt_name, wt in (("v2", V2_WORKTREE), ("v3", WORKTREE)):
        for mode in ("residual", "direct"):
            out = os.path.join(scratch, f"a5_parity_{wt_name}_{mode}.pt")
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            r = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--dump-forward", out,
                 "--worktree", wt, "--mode", mode],
                env=env, capture_output=True, text=True)
            if r.returncode != 0:
                check("P0", f"dump {wt_name}/{mode}", False, r.stderr[-600:])
            else:
                dumps[(wt_name, mode)] = torch.load(out, map_location="cpu", weights_only=False)
    for mode, gid in (("residual", "P1"), ("direct", "P2")):
        if ("v2", mode) in dumps and ("v3", mode) in dumps:
            a, b = dumps[("v2", mode)], dumps[("v3", mode)]
            same_sd = (set(a["sd"]) == set(b["sd"])
                       and all(torch.equal(a["sd"][k], b["sd"][k]) for k in a["sd"]))
            bit = torch.equal(a["out"], b["out"]) and torch.equal(a["hidden"], b["hidden"])
            check(gid, f"mode={mode!r}: v2 code == v3 code, BITWISE (params + output)",
                  same_sd and bit,
                  f"same_state_dict={same_sd} out_maxdiff={maxdiff(a['out'], b['out']):.3e}")

    # ---------------------------------------------------------------- U ----
    banner("STAGE U — construction, parameter set, zero-init placement")
    g5 = build(torch, PoseGRU, "split_anchor")
    g4 = build(torch, PoseGRU, "residual")
    g3 = build(torch, PoseGRU, "direct")
    check("U1", "mode='split_anchor' constructs and records its mode",
          g5.mode == "split_anchor", f"mode={g5.mode!r} input_dim={g5.input_dim}")
    names5, names4 = set(g5.state_dict()), set(g4.state_dict())
    shapes_same = all(g5.state_dict()[k].shape == g4.state_dict()[k].shape for k in names5 & names4)
    check("U2", "parameter SET and SHAPES identical to residual/direct "
                "(no new tensors -> state_dict-compatible with A3/A4 ckpts)",
          names5 == names4 == set(g3.state_dict()) and shapes_same,
          f"{sorted(names5)}")
    check("U3", "rotation rows head.weight[3:7]/bias[3:7] are EXACTLY zero",
          bool(g5.head.weight[3:7].eq(0).all() and g5.head.bias[3:7].eq(0).all()),
          f"|w[3:7]|max={float(g5.head.weight[3:7].abs().max()):.3e}")
    check("U4", "translation rows head.weight[0:3] are NOT zero (default init, as A3)",
          bool(g5.head.weight[:3].abs().max() > 0),
          f"|w[0:3]|max={float(g5.head.weight[:3].abs().max()):.5f}")
    check("U5", "translation rows are BITWISE the A3 (direct) init — same seed, "
                "so A5 == A3 on the translation half at init",
          torch.equal(g5.head.weight[:3], g3.head.weight[:3])
          and torch.equal(g5.head.bias[:3], g3.head.bias[:3]))
    check("U6", "residual arm still fully zero-init (unchanged by the A5 edit)",
          bool(g4.head.weight.eq(0).all() and g4.head.bias.eq(0).all()))

    # ---------------------------------------------------------------- A ----
    banner("STAGE A — forward algebra: direct translation, anchored rotation")
    # give all three modes the SAME trained-like weights so the halves are comparable
    torch.manual_seed(99)
    trained = {k: torch.randn_like(v) * 0.1 for k, v in g5.state_dict().items()}
    for g in (g3, g4, g5):
        g.load_state_dict(trained)
    with torch.no_grad():
        o5, h5 = g5(gru_in, hidden)
        o3, h3 = g3(gru_in, hidden)
        o4, h4 = g4(gru_in, hidden)
        raw = g5.head(g5.cell(gru_in, hidden))
        exp_t = raw[:, :3]
        exp_q = torch.nn.functional.normalize(anchor_q + raw[:, 3:7], dim=-1)
    check("A1", "translation == raw head output (DIRECT, no anchor term)",
          torch.equal(o5[:, :3], exp_t), f"maxdiff={maxdiff(o5[:, :3], exp_t):.3e}")
    check("A2", "rotation == normalize(anchor_q + raw_q) (ANCHORED)",
          torch.equal(o5[:, 3:7], exp_q), f"maxdiff={maxdiff(o5[:, 3:7], exp_q):.3e}")
    check("A3", "A5 translation is BITWISE A3's translation (same weights)",
          torch.equal(o5[:, :3], o3[:, :3]), f"maxdiff={maxdiff(o5[:, :3], o3[:, :3]):.3e}")
    check("A4", "A5 rotation is BITWISE A4's rotation (same weights)",
          torch.equal(o5[:, 3:7], o4[:, 3:7]), f"maxdiff={maxdiff(o5[:, 3:7], o4[:, 3:7]):.3e}")
    check("A5", "hidden update identical across modes (mode touches only the head)",
          torch.equal(h5, h3) and torch.equal(h5, h4))
    check("A6", "emitted quaternion is unit-norm",
          bool((o5[:, 3:7].norm(dim=-1) - 1).abs().max() < 1e-6),
          f"max|‖q‖-1|={float((o5[:, 3:7].norm(dim=-1) - 1).abs().max()):.2e}")

    # ---------------------------------------------------------------- I ----
    banner("STAGE I — behaviour of an UNTRAINED A5 module")
    g5i = build(torch, PoseGRU, "split_anchor")
    with torch.no_grad():
        o = g5i(gru_in, hidden)[0]
    q_anchor = torch.nn.functional.normalize(anchor_q, dim=-1)
    check("I1", "at init, rotation is EXACTLY the anchor quaternion "
                "(zero-init rotation rows) — A4's identity property, preserved",
          torch.equal(o[:, 3:7], q_anchor), f"maxdiff={maxdiff(o[:, 3:7], q_anchor):.3e}")
    dt = float((o[:, :3] - anchor_t).norm(dim=-1).mean())
    at = float(anchor_t.norm(dim=-1).mean())
    check("I2", "at init, translation is a FREE regression, NOT the anchor "
                "(this is intended — A5 keeps A3's translation path)",
          not torch.allclose(o[:, :3], anchor_t),
          f"mean‖t_out - t_anchor‖={dt:.4f} vs mean‖t_anchor‖={at:.4f} "
          f"(so an untrained A5 is NOT equivalent to plain prev_pred — same as A3)")

    # ---------------------------------------------------------------- G ----
    banner("STAGE G — gradient reaches BOTH halves of the shared head")
    g5g = build(torch, PoseGRU, "split_anchor")
    out, _ = g5g(gru_in, hidden)
    out.sum().backward()
    gw = g5g.head.weight.grad
    check("G1", "translation rows receive gradient", bool(gw[:3].abs().sum() > 0),
          f"|grad[0:3]|sum={float(gw[:3].abs().sum()):.5f}")
    check("G2", "rotation rows receive gradient DESPITE zero init "
                "(so the anchored half can actually learn)",
          bool(gw[3:7].abs().sum() > 0), f"|grad[3:7]|sum={float(gw[3:7].abs().sum()):.5f}")
    check("G3", "cell receives gradient", bool(g5g.cell.weight_ih.grad.abs().sum() > 0))

    # ---------------------------------------------------------------- E ----
    banner("STAGE E — unknown modes hard-fail (no silent fallback to A3)")
    try:
        PoseGRU(hidden_dim=H, mode="anchored", input_mode="pose_delta")
        check("E1", "ctor rejects an unknown mode", False, "no exception raised")
    except AssertionError as e:
        check("E1", "ctor rejects an unknown mode", True, str(e)[:120])
    gbad = build(torch, PoseGRU, "split_anchor")
    gbad.mode = "typo_mode"           # post-hoc assignment bypasses the ctor assert
    try:
        gbad(gru_in, hidden)
        check("E2", "forward raises on an unrecognised mode instead of silently "
                    "running the DIRECT algebra", False, "no exception raised")
    except ValueError as e:
        check("E2", "forward raises on an unrecognised mode instead of silently "
                    "running the DIRECT algebra", True, str(e)[:120])

    # ---------------------------------------------------------------- Y ----
    banner("STAGE Y — configs differ from their A4 twins only in mode + exp_name")
    import re
    for g in ("1", "2"):
        a4 = open(f"{CFG_DIR}/captain_gru_v2_a4_g{g}.yaml").read().splitlines()
        a5 = open(f"{CFG_DIR}/captain_gru_v3_a5_g{g}.yaml").read().splitlines()
        strip = lambda L: [x for x in L if not x.lstrip().startswith("#")]  # noqa: E731
        d4, d5 = strip(a4), strip(a5)
        diffs = [(x, y) for x, y in zip(d4, d5) if x != y]
        keys = {re.split(r":", x, 1)[0].strip() for x, _ in diffs}
        # save_dir differs because the 16 grid runs were reorganised into
        # checkpoints/captain_gru_overfit/ on 2026-07-30 — an OUTPUT location,
        # not part of the training recipe.
        check(f"Y{g}", f"a5_g{g}.yaml vs a4_g{g}.yaml: only pose_gru_mode + exp_name "
                       f"+ save_dir differ (recipe otherwise byte-identical)",
              len(d4) == len(d5) and keys == {"pose_gru_mode", "exp_name", "save_dir"},
              f"lines {len(d4)}/{len(d5)}, changed keys={sorted(keys)}")
        check(f"Y{g}m", f"a5_g{g}.yaml sets pose_gru_mode: split_anchor",
              any(x.strip() == "pose_gru_mode: split_anchor" for x in d5))
        check(f"Y{g}r", f"a5_g{g}.yaml keeps the R/F levers OFF (keys absent -> 1 / none)",
              not any(x.strip().startswith(("pose_gru_iters", "pose_gru_img_feat")) for x in d5))

    # ---------------------------------------------------------------- C ----
    banner("STAGE C — checkpoint round-trip and the args hard-fail")
    if not os.path.isfile(CKPT):
        check("C0", f"checkpoint available ({CKPT})", False, "missing — stage skipped")
    else:
        from dust3r.model import load_model
        import copy
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        gru_keys = [k for k in ck["model"] if k.startswith(("pose_gru.", "module.pose_gru."))]
        check("C1", "reference A4 checkpoint carries pose_gru weights", len(gru_keys) > 0,
              f"{len(gru_keys)} tensors: {sorted(k.split('pose_gru.')[-1] for k in gru_keys)}")
        # C2: an A5 module accepts the A4 weight set verbatim (identical param set)
        g5c = build(torch, PoseGRU, "split_anchor")
        sd = {k.split("pose_gru.")[-1]: v for k, v in ck["model"].items()
              if k.startswith(("pose_gru.", "module.pose_gru."))}
        try:
            g5c.load_state_dict(sd, strict=True)
            tmag = float(g5c.head.weight[:3].abs().mean())
            check("C2", "A4 checkpoint weights load STRICTLY into an A5 module "
                        "(identical param set — no surgery script exists or is needed)", True,
                  f"NOTE: loading is possible but WARM-STARTING A5 FROM A4 IS A TRAP — A4's "
                  f"translation rows are a zero-init CORRECTION (|w[0:3]|mean={tmag:.2e}), so "
                  f"under A5's direct-translation algebra they emit t~0, i.e. a camera pinned at "
                  f"the view-0 origin. The A5 runs train from cut3r_512_dpt_4_64.pth (no pose_gru) "
                  f"like the whole grid, so this path is not used.")
        except Exception as e:
            check("C2", "A4 checkpoint weights load STRICTLY into an A5 module", False, str(e)[:200])
        # C3: load_model honours pose_gru_mode from args
        tmp = os.path.join(scratch, "a5_roundtrip.pth")
        ck2 = {"args": copy.deepcopy(ck["args"]), "model": ck["model"]}
        ck2["args"].pose_gru_mode = "split_anchor"
        torch.save(ck2, tmp)
        try:
            net = load_model(tmp, device="cpu", verbose=False)
            got = getattr(net.pose_gru, "mode", None)
            check("C3", "load_model restores mode='split_anchor' from ckpt args",
                  got == "split_anchor", f"got mode={got!r}")
        except Exception as e:
            check("C3", "load_model restores mode='split_anchor' from ckpt args", False, str(e)[:300])
        # C4: a ckpt with pose_gru weights but NO pose_gru_mode is refused
        ck3 = {"args": copy.deepcopy(ck["args"]), "model": ck["model"]}
        try:
            del ck3["args"].pose_gru_mode
        except Exception:
            import omegaconf
            with omegaconf.open_dict(ck3["args"]):
                del ck3["args"]["pose_gru_mode"]
        torch.save(ck3, tmp)
        try:
            load_model(tmp, device="cpu", verbose=False)
            check("C4", "ckpt with pose_gru weights but no pose_gru_mode is REFUSED "
                        "(mode is unrecoverable from weights -> never guess)", False,
                  "loaded silently — the hard-fail did not fire")
        except RuntimeError as e:
            check("C4", "ckpt with pose_gru weights but no pose_gru_mode is REFUSED "
                        "(mode is unrecoverable from weights -> never guess)", True, str(e)[:160])
        except Exception as e:
            check("C4", "ckpt without pose_gru_mode raises RuntimeError specifically",
                  False, f"{type(e).__name__}: {str(e)[:160]}")
        for f in (tmp,):
            if os.path.exists(f):
                os.remove(f)

    # ---------------------------------------------------------------------
    banner("SUMMARY")
    npass = sum(1 for _, _, ok, _ in CHECKS if ok)
    for gid, desc, ok, _ in CHECKS:
        print(f"  {'PASS' if ok else 'FAIL':4s}  {gid:5s}  {desc}")
    print(f"\n{npass}/{len(CHECKS)} checks passed", flush=True)
    return 0 if npass == len(CHECKS) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-forward")
    ap.add_argument("--worktree")
    ap.add_argument("--mode")
    a, _ = ap.parse_known_args()
    if a.dump_forward:
        dump_forward(a.dump_forward, a.worktree, a.mode)
    else:
        sys.exit(main())
