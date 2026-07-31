#!/usr/bin/env python
"""probe_gru_grid_all.py — one uniform falsifier sweep over the whole lever grid.

Runs the probe_gru_hidden_falsifier machinery (same data, same metrics) on
every trained arm at the SAME epoch, plus the v3-only arms:

  normal            trained model as designed
  hidden_zero       POSE_GRU_HIDDEN_ZERO=1     — no recurrent memory
  hidden_shuffle    POSE_GRU_HIDDEN_SHUFFLE=1  — another sample's memory
  feedback_shuffle  PREV_PRED_RAY_SHUFFLE=1    — another sample's fed-back pose
  no_gru            pose_gru stripped (plain prev_pred closed loop)
  img_feat_zero     POSE_GRU_IMG_FEAT_ZERO=1     (F arms only)
  img_feat_shuffle  POSE_GRU_IMG_FEAT_SHUFFLE=1  (F arms only)
  iters_<n>         POSE_GRU_FORCE_ITERS=<n>     (R arms only; anytime-N curve)

Per arm: GRU pose error vs GT (PoseGRULoss metric), head pose error, scale-aligned
pts3d loss, and the lag baseline (fed-back pose vs current GT) the GRU must beat.

Usage:  python probe_gru_grid_all.py --out <dir> [--shard i --num_shards n]
"""
import argparse
import json
import os
import sys
import time

WORKTREE = "/scratch/bdursun25/cuteanything/captain_gru_v3"
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import probe_gru_hidden_falsifier as P  # noqa: E402
from dust3r.datasets.dl3dv import DL3DV_Multi  # noqa: E402
from dust3r.datasets.utils.transforms import ImgNorm  # noqa: E402
from dust3r.losses import L21, Regr3DPose  # noqa: E402
from dust3r.model import load_model  # noqa: E402
from torch.utils.data._utils.collate import default_collate  # noqa: E402

CKPT_ROOT = "/scratch/bdursun25/cuteanything/checkpoints/captain_cut3r_sim3rmse"

# label -> run dir; every arm probed at checkpoint-final (epoch 50) so the
# comparison is epoch-uniform (the earlier one-off probes were ep20/30/40).
RUNS = [f"v2_a{a}_g{g}" for g in (0, 1, 2) for a in (1, 2, 3, 4)] + [
    "v3_a4_g1_r8", "v3_a4_g1_f1_r8", "v3_a4_g2_r8", "v3_a4_g2_f1_r8",
]


def ckpt_of(label):
    return os.path.join(CKPT_ROOT, "captain_gru_" + label, "checkpoint-final.pth")


def build_batches(device):
    ds = DL3DV_Multi(
        split="test", ROOT=P.DATA_ROOT, resolution=[P.RES], transform=ImgNorm,
        num_views=P.NV, n_corres=0, aug_crop=0, allow_repeat=True,
        force_consecutive_frame_sampling=True, seed=P.SEED,
    )
    batches = []
    for idxs in P.BATCH_SETS:
        vls = [ds[i] for i in idxs]
        batches.append([
            {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in default_collate([vl[v] for vl in vls]).items()}
            for v in range(P.NV)
        ])
    return batches


def probe_one(label, batches):
    path = ckpt_of(label)
    model = load_model(path, device=P.DEVICE, verbose=False).eval()
    model.views_per_step = 1
    model.feed_prev_pred = True
    gru = getattr(model, "pose_gru", None)
    assert gru is not None, f"{label}: checkpoint has no pose_gru"
    iters = int(getattr(gru, "iters", 1))
    has_f = getattr(gru, "img_feat", "none") == "input"
    cfg = {
        "mode": gru.mode, "input_mode": getattr(gru, "input_mode", "?"),
        "iters": iters, "img_feat": getattr(gru, "img_feat", "none"),
        "img_feat_dim": int(getattr(gru, "img_feat_dim", 0)),
        "img_feat_frames": int(getattr(gru, "img_feat_frames", 0)),
        "hidden_dim": gru.hidden_dim, "ckpt": path,
    }
    print(f"[{label}] {cfg}", flush=True)

    arms = {}
    arms["normal"] = P.eval_arm(model, batches)
    arms["hidden_zero"] = P.eval_arm(model, batches, {"POSE_GRU_HIDDEN_ZERO": "1"})
    arms["hidden_shuffle"] = P.eval_arm(model, batches, {"POSE_GRU_HIDDEN_SHUFFLE": "1"})
    arms["feedback_shuffle"] = P.eval_arm(model, batches, {"PREV_PRED_RAY_SHUFFLE": "1"})
    if has_f:
        arms["img_feat_zero"] = P.eval_arm(model, batches, {"POSE_GRU_IMG_FEAT_ZERO": "1"})
        arms["img_feat_shuffle"] = P.eval_arm(model, batches, {"POSE_GRU_IMG_FEAT_SHUFFLE": "1"})
    # anytime-N curve: force every N, INCLUDING the trained one (a sanity check
    # that iters_<trained> reproduces the normal arm bit-for-bit).
    for n in (1, 2, 4, 8):
        arms[f"iters_{n}"] = P.eval_arm(model, batches, {"POSE_GRU_FORCE_ITERS": str(n)})
    saved = model.pose_gru
    model.pose_gru = None
    arms["no_gru"] = P.eval_arm(model, batches)
    model.pose_gru = saved
    del model
    torch.cuda.empty_cache()
    return {"config": cfg, "arms": arms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    assert torch.cuda.is_available(), "GPU required"
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    P.PTS_METRIC = Regr3DPose(L21, norm_mode="?avg_dis", gt_scale=True, sky_loss_value=0)
    batches = build_batches(P.DEVICE)
    print(f"data: {len(batches)} batches x {len(P.BATCH_SETS[0])} scenes x {P.NV} views",
          flush=True)

    mine = [r for i, r in enumerate(RUNS) if i % args.num_shards == args.shard]
    print(f"shard {args.shard}/{args.num_shards}: {mine}", flush=True)
    for label in mine:
        dst = os.path.join(args.out, f"{label}.json")
        if os.path.exists(dst):
            print(f"[{label}] exists, skip", flush=True)
            continue
        t0 = time.time()
        try:
            res = probe_one(label, batches)
        except Exception as e:  # keep the sweep alive; record the failure
            print(f"[{label}] FAILED: {type(e).__name__}: {e}", flush=True)
            json.dump({"error": f"{type(e).__name__}: {e}"}, open(dst, "w"))
            continue
        res["seconds"] = time.time() - t0
        json.dump(res, open(dst, "w"), indent=1)
        n = res["arms"]["normal"]
        print(f"[{label}] done in {res['seconds']:.0f}s  gru={n['gru_pose_loss']:.5f} "
              f"lag={n['lag_pose_loss']:.5f} head={n['head_pose_loss']:.5f} "
              f"pts={n['pts3d_orig']:.4f}", flush=True)


if __name__ == "__main__":
    main()
