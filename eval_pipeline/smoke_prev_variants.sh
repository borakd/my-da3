#!/bin/bash
# Smoke test for the two NEW captain_ray variants (prev_gt / prev_pred) on ONE
# GPU before committing the full 12-GPU run. For each checkpoint the worker
# processes 2 scenes (--limit 2) into the REAL output dirs (they count toward
# the full run, which skips them via the eval-CSV resume check), then we print
# the eval-CSV MEAN rows so we can confirm the metrics are finite/sane.
#
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --job-name=prev_smoke
#SBATCH --output=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_prev_smoke_%j.out
#SBATCH --error=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_prev_smoke_%j.err

set -o pipefail
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything

ROOT=/scratch/bdursun25/cuteanything
CRAY=$ROOT/captain_gru_v2
CUT3R_DIR=$CRAY/src/CUT3R
EVAL_SCRIPT=/scratch/bdursun25/streaming-3d/eval_depth_poses.py
SCENES_ROOT=$ROOT/scenes/pointworld_droid_splits/test/dl3dv_multi/wrist
OUT=$ROOT/outputs/cut3r_eval
SCENE_LIST=$OUT/scene_list.txt

CKPT_GT=$ROOT/checkpoints/captain_cut3r_sim3rmse/prev_gt_finetune_aug_full/checkpoint-final.pth
CKPT_PP=$ROOT/checkpoints/captain_cut3r_sim3rmse/prev_pred_finetune_aug_full/checkpoint-final.pth

export PYTHONPATH="$CRAY/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs"

run_smoke () {
  local LABEL=$1 CKPT=$2 COND=$3
  echo "=========================================================="
  echo "=== SMOKE $LABEL  cond=$COND  ckpt=$CKPT"
  echo "=========================================================="
  python "$CRAY/eval_pipeline/infer_and_eval_worker_ray.py" \
    --ckpt "$CKPT" --label "$LABEL" --size 320 --conditioning "$COND" \
    --scenes_root "$SCENES_ROOT" --scene_list "$SCENE_LIST" \
    --pred_base "$OUT/$LABEL/preds" --eval_base "$OUT/$LABEL/eval" \
    --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
    --shard_id 0 --num_shards 12 --device cuda --limit 2
  local rc=$?
  echo "[$LABEL] worker rc=$rc"
  echo "--- eval MEAN rows for $LABEL ---"
  for d in "$OUT/$LABEL/eval/"*/; do
    csv="$d/eval_depth_pose_metrics.csv"
    [ -f "$csv" ] && echo "scene=$(basename "$d")" && head -1 "$csv" && grep ",MEAN," "$csv"
  done
  return $rc
}

run_smoke captain_ray_prev_gt   "$CKPT_GT" prev_gt
rc1=$?
run_smoke captain_ray_prev_pred "$CKPT_PP" prev_pred
rc2=$?

echo "=========================================================="
echo "SMOKE SUMMARY: prev_gt rc=$rc1  prev_pred rc=$rc2"
echo "Done: $(date)"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
