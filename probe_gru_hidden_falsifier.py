#!/usr/bin/env python
"""probe_gru_hidden_falsifier.py — is the trained PoseGRU's recurrence used?

Batched no-grad rollouts of a TRAINED GRU checkpoint on real wrist_test scenes,
four arms on identical data:
  normal          trained GRU, hidden as designed
  hidden_shuffle  POSE_GRU_HIDDEN_SHUFFLE=1 — every sample gets another
                  sample's memory (wrong memory; needs batch>1)
  hidden_zero     POSE_GRU_HIDDEN_ZERO=1 — hidden reset every step
                  (no memory; the stateless-filter control that killed v1)
  no_gru          pose_gru stripped — plain prev_pred closed loop

Per arm: GRU-pose error vs GT (PoseGRULoss metric), head-pose error, and the
eval-style scale-aligned pts3d loss. Plus the lag baseline (fed-back pose vs
current GT) the GRU is supposed to beat.

Verdicts:
  recurrence used   <=> hidden_zero (and hidden_shuffle) degrade gru error
  GRU learned       <=> normal gru error < lag-baseline error
  GRU helps loop    <=> normal head/pts metrics better than no_gru

Usage:  python probe_gru_hidden_falsifier.py [ckpt]   (default: checkpoint-20)
Submit: sbatch probe_gru_hidden.sbatch [ckpt]
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
for p in [
    os.path.join(WORKTREE, "src"),
    os.path.join(WORKTREE, "src/CUT3R"),
    os.path.join(WORKTREE, "src/CUT3R/src"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

import dust3r.heads  # noqa: F401  MUST precede dust3r.utils.camera (circular import)
from dust3r.datasets.dl3dv import DL3DV_Multi
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.losses import L21, PoseGRULoss, Regr3DPose
from dust3r.model import load_model
from torch.utils.data._utils.collate import default_collate

# MN5: the DROID store moved to gpfs_scratch and the splits tree replaced the
# old wrist_test layout. DL3DV_Multi walks ROOT two levels
# (<scene>/<subscene>/dense), which both layouts satisfy. Same value the
# captain_gru_v3 finetune configs train against.
DATA_ROOT = os.environ.get(
    "DL3DV_TEST_ROOT",
    "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi",
)
DEFAULT_CKPT = os.environ.get(
    "PROBE_CKPT",
    os.path.join(
        os.environ.get("CKPT_ROOT", "/gpfs/projects/etur59/koc821022/checkpoints"),
        "captain_cut3r_sim3rmse", "captain_gru_v2", "checkpoint-20.pth",
    ),
)
NV = 32                       # consecutive views per scene (real motion depth)
BATCH_SETS = [                # 2 batches x 6 scenes; shuffle mixes across 6
    [3, 47, 91, 135, 179, 223],
    [267, 311, 355, 399, 443, 487],
]
RES = (320, 192)
SEED = 777
DEVICE = "cuda"

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
np.set_printoptions(precision=5, suppress=True)


def to_device(view):
    return {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in view.items()}


@torch.no_grad()
def rollout(model, batch):
    """Sequential per-view decode, mirroring loss_of_one_batch_tbptt (no grad)."""
    (feat, pos, shape), (isf, imem, sf, sp, mem) = model._forward_encoder(batch)
    ress = []
    for v in range(len(batch)):
        res_group, (sf, mem) = model._forward_decoder_group_step(
            views=batch, view_indices=[v],
            feat_group=[feat[v]], pos_group=[pos[v]], shape_group=[shape[v]],
            init_state_feat=isf, init_mem=imem,
            state_feat=sf, state_pos=sp, mem=mem,
        )
        ress.append({k: (t.detach() if isinstance(t, torch.Tensor) else t)
                     for k, t in res_group[0].items()})
    return ress


GRU_METRIC = PoseGRULoss()
PTS_METRIC = None  # built after imports resolve inf


def gru_err(batch, ress):
    loss, det = GRU_METRIC(batch, ress)
    return {k.replace("gru_", ""): v for k, v in det.items()} if det else None


def pose_err_of(batch, ress, poses_per_view):
    """Head/lag pose error via the same normalized metric: swap gru_pose."""
    preds = [dict(r) for r in ress]
    for v, pose in poses_per_view.items():
        q = torch.nn.functional.normalize(pose[:, 3:7], dim=-1)
        preds[v] = dict(preds[v])
        preds[v]["gru_pose"] = torch.cat([pose[:, :3], q], dim=-1)
    for v in range(len(preds)):
        if v not in poses_per_view and "gru_pose" in preds[v]:
            del preds[v]["gru_pose"]
    loss, det = GRU_METRIC(batch, preds)
    return {k.replace("gru_", ""): v for k, v in det.items()}


def eval_arm(model, batches, env=None):
    env = env or {}
    for k, v in env.items():
        os.environ[k] = v
    try:
        agg = {}
        for batch in batches:
            ress = rollout(model, batch)
            row = {}
            g = gru_err(batch, ress)
            if g:
                row.update({f"gru_{k}": v for k, v in g.items()})
            # lag baseline: fed-back pose (head pose of view v-1) vs GT of view v
            lag = pose_err_of(batch, ress,
                              {v: ress[v - 1]["camera_pose"] for v in range(1, len(ress))})
            row.update({f"lag_{k}": v for k, v in lag.items()})
            # head accuracy: head pose of view v vs GT of view v
            head = pose_err_of(batch, ress,
                               {v: ress[v]["camera_pose"] for v in range(1, len(ress))})
            row.update({f"head_{k}": v for k, v in head.items()})
            try:
                pts, _ = PTS_METRIC(batch, ress, camera1=batch[0]["camera_pose"])
            except TypeError:
                pts, _ = PTS_METRIC(batch, ress)
            row["pts3d_orig"] = float(pts)
            for k, v in row.items():
                agg.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in agg.items()}
    finally:
        for k in env:
            os.environ.pop(k, None)


def main():
    global PTS_METRIC
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    assert torch.cuda.is_available(), "GPU required"
    print(f"device: {torch.cuda.get_device_name(0)} | ckpt: {ckpt_path}", flush=True)

    PTS_METRIC = Regr3DPose(L21, norm_mode="?avg_dis", gt_scale=True, sky_loss_value=0)

    model = load_model(ckpt_path, device=DEVICE, verbose=True).eval()
    model.views_per_step = 1
    model.feed_prev_pred = True
    has_gru = getattr(model, "pose_gru", None) is not None
    assert has_gru, "checkpoint has no pose_gru — nothing to probe"
    print(f"pose_gru: mode={model.pose_gru.mode}, hidden_dim={model.pose_gru.hidden_dim}",
          flush=True)

    ds = DL3DV_Multi(
        split="test", ROOT=DATA_ROOT, resolution=[RES], transform=ImgNorm,
        num_views=NV, n_corres=0, aug_crop=0, allow_repeat=True,
        force_consecutive_frame_sampling=True, seed=SEED,
    )
    batches = []
    for idxs in BATCH_SETS:
        vls = [ds[i] for i in idxs]
        batches.append([to_device(default_collate([vl[v] for vl in vls]))
                        for v in range(NV)])
    print(f"data: {len(BATCH_SETS)} batches x {len(BATCH_SETS[0])} scenes x {NV} views",
          flush=True)

    arms = {}
    arms["normal"] = eval_arm(model, batches)
    arms["hidden_shuffle"] = eval_arm(model, batches, {"POSE_GRU_HIDDEN_SHUFFLE": "1"})
    arms["hidden_zero"] = eval_arm(model, batches, {"POSE_GRU_HIDDEN_ZERO": "1"})
    saved_gru = model.pose_gru
    model.pose_gru = None
    arms["no_gru"] = eval_arm(model, batches)
    model.pose_gru = saved_gru

    cols = ["gru_pose_loss", "gru_trans_loss", "gru_quat_loss",
            "head_pose_loss", "pts3d_orig"]
    print("\n" + "=" * 100)
    print(f"{'arm':>16} | " + " | ".join(f"{c:>14}" for c in cols))
    print("-" * 100)
    for name, row in arms.items():
        print(f"{name:>16} | " + " | ".join(
            f"{row[c]:>14.5f}" if c in row else f"{'—':>14}" for c in cols))
    lag = arms["normal"]["lag_pose_loss"]
    print("-" * 100)
    print(f"{'lag baseline':>16} | {lag:>14.5f}   (fed-back head pose vs current GT — "
          "what an identity GRU would score)")

    print("\nVERDICTS")
    n, hz, hs = arms["normal"], arms["hidden_zero"], arms["hidden_shuffle"]

    def pct(a, b):
        return (a - b) / max(abs(b), 1e-9) * 100.0

    d_zero = pct(hz["gru_pose_loss"], n["gru_pose_loss"])
    d_shuf = pct(hs["gru_pose_loss"], n["gru_pose_loss"])
    d_lag = pct(lag, n["gru_pose_loss"])
    print(f"  GRU learned (beats lag baseline): {'YES' if d_lag > 5 else 'NO '} "
          f"— normal {n['gru_pose_loss']:.5f} vs lag {lag:.5f} ({d_lag:+.1f}%)")
    print(f"  recurrence used (zeroing hurts):  {'YES' if d_zero > 5 else 'NO '} "
          f"— hidden_zero degrades gru error by {d_zero:+.1f}%")
    print(f"  wrong memory hurts (shuffle):     {'YES' if d_shuf > 5 else 'NO '} "
          f"— hidden_shuffle degrades gru error by {d_shuf:+.1f}%")
    print(f"  GRU helps the loop downstream:    head_pose {pct(arms['no_gru']['head_pose_loss'], n['head_pose_loss']):+.1f}%, "
          f"pts3d {pct(arms['no_gru']['pts3d_orig'], n['pts3d_orig']):+.1f}% "
          "(positive = no_gru worse = GRU helping)")
    print("  (thresholds are 5% — treat near-zero deltas as 'not used / not helping')")


if __name__ == "__main__":
    main()
