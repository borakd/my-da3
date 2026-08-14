#!/bin/bash
# Submit the 3-arm lr1e-5 DROID eval: 32 GPUs (8 nodes x 4 H100) PER ARM.
#
# The arm is NOT stored in the state_dict -- it is read off ckpt["args"] and
# supplied as --conditioning. Verified directly from each checkpoint-final.pth:
#
#   run dir                                    ckpt["args"]                    -> --conditioning
#   cut3r_finetune_aug_full_32gpu_lr1e5        no feed_* keys at all           -> none
#   cut3r_finetune_aug_full_gtray_32gpu_lr1e5  feed_gt_ray_map=True            -> gt
#   cut3r_finetune_aug_full_prevpred_32gpu_..  feed_prev_pred=True             -> prev_pred
#
# All three: epoch 50, 1248 tensors (no pose_gru), resolution [[320,192]],
# lr 1e-5 -> min_lr 1e-6.
#
# Two MN5 facts this script encodes:
#   * OUT_ROOT is forced onto gpfs_scratch. Predictions are ~332 GB PER ARM
#     (1,260,994 frames x 192x320 f32 depth); gpfs_projects has ~146 GB of
#     group-quota headroom, so the default OUT_ROOT would hit quota mid-run.
#   * --cpus-per-task must be exactly 80 for a 4-GPU acc job (lua submit plugin
#     enforces cpus >= nodes*gpus*20; 57 is hard-rejected). Fixed in the launcher.
#
# Usage:  bash eval_pipeline/submit_lr1e5_3arm_eval.sh [ARM ...]
#         (no args = all three arms)
set -o pipefail

WT=/gpfs/home/koc/koc821022/my-da3
export OUT_ROOT=/gpfs/scratch/etur59/koc821022/outputs
OUT=$OUT_ROOT/cut3r_eval
# The three lr1e-5 baselines moved twice: out of the DELETED cut3r_multinode/
# into cut3r_finetune_baselines/, then off /gpfs/projects entirely in the
# 2026-08-14 storage move. The three dir names below survived both moves.
CKPT_ROOT=${CKPT_ROOT:-/gpfs/scratch/etur59/koc821022/checkpoints_projects/cut3r_finetune_baselines}

# Falsifier env vars silently contaminate a scoring run and nothing logs it.
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE \
      POSE_GRU_HIDDEN_ZERO POSE_GRU_HIDDEN_SHUFFLE \
      POSE_GRU_IMG_FEAT_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_FORCE_ITERS

declare -A CKPTS=(
  [augfull_lr1e5]=$CKPT_ROOT/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth
  [gtray_lr1e5]=$CKPT_ROOT/cut3r_finetune_aug_full_gtray_32gpu_lr1e5/checkpoint-final.pth
  [prevpred_lr1e5]=$CKPT_ROOT/cut3r_finetune_aug_full_prevpred_32gpu_lr1e5/checkpoint-final.pth
)
declare -A CONDS=(
  [augfull_lr1e5]=none
  [gtray_lr1e5]=gt
  [prevpred_lr1e5]=prev_pred
)

ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(augfull_lr1e5 gtray_lr1e5 prevpred_lr1e5)

mkdir -p "$OUT/logs" || exit 1
[ -s "$OUT/scene_list.txt" ] || { echo "ERROR: missing $OUT/scene_list.txt" >&2; exit 1; }

cd "$WT" || exit 1   # WT is taken from SLURM_SUBMIT_DIR by the launcher

for LABEL in "${ARMS[@]}"; do
  CKPT=${CKPTS[$LABEL]}
  COND=${CONDS[$LABEL]}
  [ -f "$CKPT" ] || { echo "ERROR: missing $CKPT" >&2; exit 1; }
  echo "=== $LABEL  conditioning=$COND"
  echo "    $CKPT"
  for B in 0 4 8 12 16 20 24 28; do
    sbatch \
      --job-name="ev_${LABEL}_b${B}" \
      --output="$OUT/logs/slurm_${LABEL}_b${B}_%j.out" \
      --error="$OUT/logs/slurm_${LABEL}_b${B}_%j.err" \
      --export=ALL,OUT_ROOT="$OUT_ROOT",LABEL="$LABEL",CONDITIONING="$COND",CKPT="$CKPT",BASE_SHARD="$B",NUM_SHARDS=32,LIMIT=0 \
      eval_pipeline/run_captain_ray_eval_node.sh
  done
done
