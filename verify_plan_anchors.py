#!/usr/bin/env python3
"""Preflight for GRU_GAP_CLOSURE_DIRECTIVE.md — are its code anchors still true?

The directive cites file:line for every edit site. Line numbers drift the
moment anyone touches these files (model.py already shifted +9 in the
call-site region on 2026-08-11 mid-session), and a stale anchor sends an
implementer to the wrong place. This script resolves every anchor by SYMBOL
(a regex that matches the actual code line) and reports where it lives now.

Run it FIRST in any session that is about to implement a proposal:

    python verify_plan_anchors.py            # report
    python verify_plan_anchors.py --fix-doc  # rewrite stale numbers in the doc

Exit code 0 = every anchor resolves uniquely and matches the doc.
Exit code 1 = an anchor is missing/ambiguous (the code moved substantively —
              re-read that region before editing) or a number is stale.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOC = ROOT / "GRU_GAP_CLOSURE_DIRECTIVE.md"

# (file, expected_line_in_doc, regex, label). expected=None -> report only.
ANCHORS = [
    # ---- §1 GRU-only gradient sync -------------------------------------
    ("src/CUT3R/src/dust3r/inference.py", 118,
     r"base_model = accelerator\.unwrap_model", "unwrapped TBPTT forward (the defect)"),
    ("src/CUT3R/src/dust3r/inference.py", 162,
     r"if chunk_id < num_chunks - 4", "graded-chunk window -> GRADED_CHUNKS"),
    ("src/CUT3R/src/dust3r/inference.py", 226,
     r"^\s*loss_scaler\($", "per-chunk loss_scaler call"),
    ("src/CUT3R/src/dust3r/inference.py", 239,
     r"already_backprop=True", "end of chunk loop -> pad_gru_sync just above this dict"),
    ("src/CUT3R/src/dust3r/inference.py", 156,
     r"_gru_hidden = getattr\(base_model", "chunk-boundary hidden detach"),
    ("src/CUT3R/src/croco/utils/misc.py", 320,
     r"^def all_reduce_grads_", "all_reduce_grads_ (add `contributing`)"),
    ("src/CUT3R/src/croco/utils/misc.py", 374,
     r"^\s*scale = 1\.0 / world_size", "divisor -> 1/n_contributing"),
    ("src/CUT3R/src/croco/utils/misc.py", 402,
     r"^class NativeScalerWithGradNormCount", "scaler class"),
    ("src/CUT3R/src/croco/utils/misc.py", 405,
     r"def __init__\(self, enabled=True, accelerator: Accelerator = None, ddp_grad_sync",
     "scaler __init__ (add sync_params)"),
    ("src/CUT3R/src/croco/utils/misc.py", 426,
     r"if self\.ddp_grad_sync", "existing full-sync branch"),
    ("src/CUT3R/src/croco/utils/misc.py", 431,
     r"^\s*if clip_grad is not None", "clip site (sync must precede it)"),
    ("src/CUT3R/src/train_cut3r_baseline.py", 356,
     r"ddp_grad_sync = bool\(getattr", "flag read (add pose_gru_grad_sync)"),
    ("src/CUT3R/src/train_cut3r_baseline.py", 363,
     r"loss_scaler = NativeScaler\(", "scaler construction"),
    ("src/CUT3R/src/train_cut3r_baseline.py", 372,
     r"base_model = accelerator\.unwrap_model", "unwrap -> attach sync_params after"),
    ("src/CUT3R/src/train_cut3r_baseline.py", 110,
     r"^def split_pose_gru_param_groups", "GRU param set (same set as sync_params)"),
    ("src/CUT3R/src/train_cut3r_baseline.py", 344,
     r"param_groups = split_pose_gru_param_groups", "GRU param-group split call"),
    ("src/CUT3R/src/dust3r/datasets/base/batched_sampler.py", 69,
     r"_view_idxs = rng\.integers", "random per-batch view count (why padding)"),
    # ---- §2 abstention gate --------------------------------------------
    ("src/CUT3R/src/dust3r/model.py", 641,
     r"^\s*pose_dim=7,$", "PoseGRU.__init__ signature (add gate kwarg)"),
    ("src/CUT3R/src/dust3r/model.py", 751,
     r"self\.head = nn\.Linear\(hidden_dim, pose_dim\)", "head (gate built next to it)"),
    ("src/CUT3R/src/dust3r/model.py", 796,
     r"^\s*raw = self\.head\(hidden\)", "head output (gate multiplies here)"),
    ("src/CUT3R/src/dust3r/model.py", 77,
     r"^def _sniff_pose_gru_config", "checkpoint sniff (detect gate)"),
    ("src/CUT3R/src/dust3r/model.py", 120,
     r"img_norm\.weight exists iff", "sniff key-presence table"),
    ("src/CUT3R/src/dust3r/model.py", 1856,
     r"res_group\[-1\]\[\"gru_pose\"\] = gru_pose_pred", "res write (add gru_gate)"),
    ("src/CUT3R/src/train_cut3r_baseline.py", 250,
     r"enable_pose_gru\(", "enable_pose_gru kwargs assembly"),
    # ---- §0b P0 retarget the loss --------------------------------------
    ("src/CUT3R/src/dust3r/losses.py", 1132,
     r"gru_idx = \[i for i, pred in enumerate", "PoseGRULoss.compute_loss body (retarget here)"),
    ("src/CUT3R/src/dust3r/losses.py", 1158,
     r"gru_pose = preds\[i\]\[\"gru_pose\"\]\.float\(\)", "per-view GRU pose (target pairing)"),
    ("src/CUT3R/src/dust3r/model.py", 459,
     r"^def pose_delta_encoding", "pose_delta_encoding (import into losses.py)"),
    ("src/CUT3R/src/dust3r/model.py", 1758,
     r"feat_group = \[feat_group\[0\] \+ ray_out", "ray add -> P2 abstention blend"),
    # ---- §3 training-signal repair -------------------------------------
    ("src/CUT3R/src/dust3r/losses.py", 1084,
     r"^class PoseGRULoss", "PoseGRULoss"),
    ("src/CUT3R/src/dust3r/losses.py", 1103,
     r"def __init__\(self, norm_mode=\"\?avg_dis\", iter_gamma", "loss __init__ (rot_weight)"),
    ("src/CUT3R/src/dust3r/losses.py", 1164,
     r"q_err = torch\.norm\(gru_pose", "chordal q_err -> geodesic"),
    ("src/CUT3R/src/dust3r/losses.py", 1177,
     r"^\s*loss = t_loss \+ q_loss", "loss sum (apply rot_weight)"),
    ("src/CUT3R/src/dust3r/losses.py", 1211,
     r"term = w_k \* \(t_k\.mean", "iterate term (apply rot_weight)"),
    ("src/CUT3R/src/dust3r/model.py", 1529,
     r"# Refine the fed-back pose with the recurrent filter",
     "GRU island START, 2 lines below (full-window wrap)"),
    ("src/CUT3R/src/dust3r/model.py", 1729,
     r"prev_pose_enc = gru_pose_pred if pose_gru_e2e", "GRU island END"),
    ("src/CUT3R/src/dust3r/model.py", 1685,
     r"^\s*gru_in = \($", "gru_in construction (input_gain / probe widen)"),
    ("src/CUT3R/src/dust3r/model.py", 783,
     r"^\s*cell_in = gru_in$", "cell_in (input_gain multiply)"),
    # ---- §4 observability ----------------------------------------------
    ("src/CUT3R/src/dust3r/model.py", 1559,
     r"cur_img_feat = feat_group\[0\]\.detach\(\)\.mean", "pooled F feature (the lossy one)"),
    ("src/CUT3R/src/dust3r/model.py", 1564,
     r"cur_img_feat = feat_group\[0\]\.detach\(\)\.float\(\)", "corr full-token stash"),
    ("src/CUT3R/src/dust3r/model.py", 1577,
     r"from dust3r\.img_encoders import corr_motion_stats",
     "corr pair statistic (xattn slots beside it)"),
    ("src/CUT3R/src/dust3r/model.py", 1318,
     r"full_out\[i\]\[:batch_size\] \+= self\.masked_ray_map_token", "view-0 unconditioned path"),
    ("src/CUT3R/src/dust3r/model.py", 1443,
     r"def _forward_decoder_group_step", "decoder group step (probe pass)"),
    ("src/CUT3R/src/dust3r/model.py", 232,
     r"base_width = w_ih\.shape", "sniff base_width {7,14} -> add 21"),
]

REQUIRED_FILES = [
    "src/CUT3R/config/captain_gru_v3_a4_g3_finetune.yaml",
    "src/CUT3R/config/captain_gru_v3_a4_g3_ddpsync_finetune.yaml",
    "src/CUT3R/config/captain_gru_v3_a4_g3_f1c_r8_finetune.yaml",
    "train_gru_v3_finetune_64node.sbatch",
    "smoke_train_gru_v3.sbatch",
    "run_gru_window64_eval.sbatch",
    "verify_ddp_grad_sync.py",
    "probe_gru_hidden_falsifier.py",
    "src/CUT3R/src/dust3r/img_encoders.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-doc", action="store_true",
                    help="rewrite stale line numbers in the directive")
    args = ap.parse_args()

    cache, bad, stale = {}, [], []
    for path, expected, pattern, label in ANCHORS:
        f = ROOT / path
        if not f.exists():
            bad.append((path, label, "FILE MISSING"))
            continue
        lines = cache.setdefault(path, f.read_text().split("\n"))
        hits = [i + 1 for i, l in enumerate(lines) if re.search(pattern, l)]
        if len(hits) != 1:
            bad.append((path, label, f"{len(hits)} matches {hits[:5]} — code moved, re-read"))
            continue
        actual = hits[0]
        flag = "ok  "
        if expected is not None and actual != expected:
            flag = "DRIFT"
            stale.append((path, expected, actual, label))
        print(f"[{flag}] {path}:{actual:<5d} {label}"
              + ("" if flag == "ok  " else f"   (doc says {expected})"))

    print("\n--- required files ---")
    for rel in REQUIRED_FILES:
        print(f"[{'ok  ' if (ROOT / rel).exists() else 'MISS'}] {rel}")

    if stale and args.fix_doc and DOC.exists():
        text = DOC.read_text()
        n = 0
        for path, expected, actual, _ in stale:
            base = Path(path).name
            for pat in (rf"({re.escape(base)}:){expected}\b", rf"(line )({expected})\b"):
                text, k = re.subn(pat, lambda m: m.group(1) + str(actual), text)
                n += k
        DOC.write_text(text)
        print(f"\n--fix-doc: rewrote {n} line references in {DOC.name} "
              "(review the diff — the `line NNN` form is heuristic)")

    if bad:
        print("\nUNRESOLVED ANCHORS (the code changed — do not edit blind):")
        for path, label, why in bad:
            print(f"  {path}: {label} — {why}")
    print(f"\n{len(ANCHORS) - len(bad) - len(stale)}/{len(ANCHORS)} anchors exact, "
          f"{len(stale)} drifted, {len(bad)} unresolved")
    return 1 if (bad or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
