#!/bin/bash
# Submit the 2-arm captain_gru_v3 DROID eval: 32 GPUs (8 nodes x 4 H100) PER ARM.
#
# Arm derived from .hydra/config.yaml and confirmed against ckpt["args"] +
# the pose_gru.* tensors in the state_dict:
#
#   run dir                            feed_prev_pred  pose_gru  img_feat  -> --conditioning
#   captain_gru_v3_a4_g3_finetune      True            True      none (F0)    prev_pred_gru
#   captain_gru_v3_a4_g3_f1_finetune   True            True      input (F1)   prev_pred_gru
#
# Both: mode=residual, input=pose_delta, hidden=128, iters=1 (R1),
# bptt=e2e=True (G3, inert at eval), lr 1e-5 -> 1e-6, 50 epochs.
#
# checkpoint-final.pth ONLY -- checkpoint-last.pth is rewritten every epoch and
# mid-epoch (train_cut3r_baseline.py:740-746), so torch.load races the writer.
#
# Same two MN5 constraints as submit_lr1e5_3arm_eval.sh: OUT_ROOT forced onto
# gpfs_scratch (~332 GB of predictions per arm), and --cpus-per-task=80.
#
# Usage:  bash eval_pipeline/submit_gru_v3_eval.sh [ARM ...]
set -o pipefail

WT=/gpfs/home/koc/koc821022/my-da3
export OUT_ROOT=/gpfs/scratch/etur59/koc821022/outputs
OUT=$OUT_ROOT/cut3r_eval
# Every checkpoint named below predates the 2026-08-14 storage move, so it is
# under checkpoints_projects/. Runs started after that date save to
# .../checkpoints/ instead -- override CR when adding one.
CR=${CR:-/gpfs/scratch/etur59/koc821022/checkpoints_projects/captain_cut3r_finetune_aug_full}

# Falsifier env vars silently contaminate a scoring run and nothing logs it.
# POSE_GRU_FORCE_ITERS in particular would override the restored R lever.
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE \
      POSE_GRU_HIDDEN_ZERO POSE_GRU_HIDDEN_SHUFFLE \
      POSE_GRU_IMG_FEAT_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_FORCE_ITERS

declare -A CKPTS=(
  [gru_a4g3]=$CR/captain_gru_v3_a4_g3_finetune/checkpoint-final.pth
  [gru_a4g3f1]=$CR/captain_gru_v3_a4_g3_f1_finetune/checkpoint-final.pth
)

ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(gru_a4g3 gru_a4g3f1)

mkdir -p "$OUT/logs" || exit 1
[ -s "$OUT/scene_list.txt" ] || { echo "ERROR: missing $OUT/scene_list.txt" >&2; exit 1; }
cd "$WT" || exit 1

for LABEL in "${ARMS[@]}"; do
  CKPT=${CKPTS[$LABEL]}
  [ -f "$CKPT" ] || { echo "ERROR: missing $CKPT (has the run finished?)" >&2; exit 1; }
  # Refuse to score a checkpoint that is still being written.
  age=$(( $(date +%s) - $(stat -c %Y "$CKPT") ))
  [ "$age" -gt 60 ] || { echo "ERROR: $CKPT was modified ${age}s ago -- still being written" >&2; exit 1; }
  echo "=== $LABEL  conditioning=prev_pred_gru"
  echo "    $CKPT"
  for B in 0 4 8 12 16 20 24 28; do
    sbatch \
      --job-name="ev_${LABEL}_b${B}" \
      --output="$OUT/logs/slurm_${LABEL}_b${B}_%j.out" \
      --error="$OUT/logs/slurm_${LABEL}_b${B}_%j.err" \
      --export=ALL,OUT_ROOT="$OUT_ROOT",LABEL="$LABEL",CONDITIONING=prev_pred_gru,CKPT="$CKPT",BASE_SHARD="$B",NUM_SHARDS=32,LIMIT=0 \
      eval_pipeline/run_captain_ray_eval_node.sh
  done
done
