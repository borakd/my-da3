#!/usr/bin/env python
"""verify_backward_compat.py — enforcement for GOVERNING CONSTRAINT 2.

Every change in the v4 program must stay ISOLATABLE and NEGATABLE: if a run
goes wrong you must be able to name the single change that caused it and turn
that one off, by config, without editing code. The half of that promise which
rots silently is checkpoint compatibility — a new parameter or buffer that is
present-and-neutral instead of absent still adds a state_dict key that every
earlier checkpoint lacks, and then old checkpoints stop loading.

That failure is invisible until someone tries to score a months-old arm, so it
gets its own gate:

    conda activate cuteanything
    python verify_backward_compat.py            # all eras
    python verify_backward_compat.py --quick    # one ckpt per era

For each reference checkpoint it asserts:
  1. it LOADS under today's code,
  2. with a STRICT key match — no missing keys, no unexpected keys, never
     strict=False and never a silent drop,
  3. and every lever invisible in weight shapes (iters, mode, input_mode,
     gains, gates) comes back at its EXPECTED value, so an eval can never
     silently score a differently-configured module.

Add a row to REFERENCE_CKPTS whenever a new era of checkpoint appears. A row
whose file is missing is SKIPPED, not failed — checkpoints get pruned, and a
gate that fails on housekeeping gets ignored, which defeats the point.

Exit 0 = compatible. Exit 1 = a real break; do not commit.
"""

import argparse
import os
import sys
import warnings

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src/CUT3R/src"))
sys.path.insert(0, os.path.join(REPO, "src/CUT3R/src/croco"))

CKPT_ROOT = "/gpfs/projects/etur59/koc821022/checkpoints"
AUG = f"{CKPT_ROOT}/captain_cut3r_finetune_aug_full"
BASE = f"{CKPT_ROOT}/cut3r_finetune_baselines"

# (label, path, expected)  — expected=None for "no pose_gru at all".
# `expected` lists ONLY the shape-invisible levers; shape-visible ones already
# hard-fail at load.
REFERENCE_CKPTS = [
    ("baseline A (no GRU, pre-program)",
     f"{BASE}/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth", None),
    ("baseline B (GT ray, no GRU)",
     f"{BASE}/cut3r_finetune_aug_full_gtray_32gpu_lr1e5/checkpoint-final.pth", None),
    ("baseline C (prev-pred, no GRU)",
     f"{BASE}/cut3r_finetune_aug_full_prevpred_32gpu_lr1e5/checkpoint-final.pth", None),
    ("GRU R1 (pre-program)",
     f"{AUG}/captain_gru_v3_a4_g3_finetune/checkpoint-final.pth",
     dict(iters=1, mode="residual", input_mode="pose_delta",
          img_feat="none", input_gain=None)),
    ("GRU R8 (pre-program, best arm)",
     f"{AUG}/captain_gru_v3_a4_g3_r8_finetune/checkpoint-final.pth",
     dict(iters=8, mode="residual", input_mode="pose_delta",
          img_feat="none", input_gain=None)),
    ("GRU F1 pooled (pre-program)",
     f"{AUG}/captain_gru_v3_a4_g3_f1_finetune/checkpoint-final.pth",
     dict(iters=1, mode="residual", input_mode="pose_delta",
          img_feat="input", input_gain=None)),
    ("GRU F1 no-proj (widest cell input)",
     f"{AUG}/captain_gru_v3_a4_g3_f1_noproj_finetune/checkpoint-final.pth",
     dict(iters=1, img_feat="input", input_gain=None)),
]

FAIL, SKIP = [], []


def check(label, path, expected, quick=False):
    if not os.path.isfile(path):
        SKIP.append((label, "checkpoint not on disk"))
        print(f"[SKIP] {label}\n       {path} (absent)")
        return

    import torch  # noqa: F401  (imported late so --help works without the env)
    from dust3r.model import ARCroco3DStereo

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = ARCroco3DStereo.from_pretrained(path)
    except Exception as e:
        FAIL.append((label, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {label}\n       load raised {type(e).__name__}: {str(e)[:200]}")
        return

    gru = getattr(net, "pose_gru", None)

    # --- 2. strict key match, re-asserted explicitly -----------------------
    # from_pretrained prints its own load report, but a printed report is not
    # an assertion — re-load the state dict strictly so a regression fails the
    # process rather than scrolling past in a log.
    try:
        import torch
        sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
        sd = {k[len("module."):] if k.startswith("module.") else k: v
              for k, v in sd.items()}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            FAIL.append((label, f"missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}"))
            print(f"[FAIL] {label}\n       NOT a strict match: "
                  f"missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
            del net
            return
    except Exception as e:
        FAIL.append((label, f"strict re-check raised {type(e).__name__}: {e}"))
        print(f"[FAIL] {label}\n       strict re-check raised {type(e).__name__}: {str(e)[:160]}")
        del net
        return

    # --- 3. shape-invisible levers came back correctly ---------------------
    if expected is None:
        if gru is not None:
            FAIL.append((label, "pose_gru materialized on a non-GRU checkpoint"))
            print(f"[FAIL] {label}\n       expected NO pose_gru, got one")
            del net
            return
        print(f"[ok  ] {label}  — no pose_gru, strict match")
    else:
        if gru is None:
            FAIL.append((label, "pose_gru missing on a GRU checkpoint"))
            print(f"[FAIL] {label}\n       expected a pose_gru, got none")
            del net
            return
        bad = []
        for k, want in expected.items():
            got = getattr(gru, k, "<absent>")
            if k == "input_gain":
                ok = (got is None) if want is None else (got is not None)
            else:
                ok = got == want
            if not ok:
                bad.append(f"{k}: expected {want!r}, got {got!r}")
        if bad:
            FAIL.append((label, "; ".join(bad)))
            print(f"[FAIL] {label}\n       " + "\n       ".join(bad))
        else:
            print(f"[ok  ] {label}  — strict match, levers restored "
                  f"({', '.join(f'{k}={getattr(gru,k,None)}' for k in expected)})")
    del net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="one checkpoint per era instead of all")
    args = ap.parse_args()

    rows = REFERENCE_CKPTS
    if args.quick:
        rows = [REFERENCE_CKPTS[0], REFERENCE_CKPTS[3], REFERENCE_CKPTS[4]]

    print(f"GOVERNING CONSTRAINT 2 — backward compatibility ({len(rows)} reference checkpoints)\n")
    for label, path, expected in rows:
        check(label, path, expected, args.quick)

    print()
    if SKIP:
        print(f"{len(SKIP)} skipped (not on disk): " + ", ".join(l for l, _ in SKIP))
    if FAIL:
        print(f"\nBACKWARD COMPATIBILITY BROKEN — {len(FAIL)} checkpoint(s). DO NOT COMMIT.")
        for l, why in FAIL:
            print(f"  {l}: {why}")
        return 1
    print(f"all {len(rows) - len(SKIP)} checked checkpoints load strictly with levers intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
