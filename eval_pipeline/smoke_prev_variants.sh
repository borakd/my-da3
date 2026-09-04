#!/bin/bash
# Smoke test for the two NEW captain_ray variants (prev_gt / prev_pred) on ONE
# GPU before committing the full 12-GPU run. For each checkpoint the worker
# processes 2 scenes (--limit 2) into the REAL output dirs (they count toward
# the full run, which skips them via the eval-CSV resume check), then we print
# the eval-CSV MEAN rows so we can confirm the metrics are finite/sane.
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15   # was 12; MN5 derives mem from cores (8G/core) -> 120G ~= old --mem=120G
#SBATCH --time=01:30:00
#SBATCH --job-name=prev_smoke
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_prev_smoke_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_prev_smoke_%j.err

set -o pipefail
# --- self-locating worktree -------------------------------------------------
# Run the checkout this job was SUBMITTED from, never a hardcoded path, and fail
# loudly if that is not a my-da3 tree. (Do not use ${BASH_SOURCE[0]} here: SLURM
# copies the batch script to its spool dir, so it would not point at the repo.)
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything

# --- MN5 storage anchors (shared; every value overridable) -------------------
source "$WT/eval_pipeline/mn5_paths.sh"
CRAY=$WT

CKPT_GT=${CKPT_GT:-}
CKPT_PP=${CKPT_PP:-}

[ -n "$CKPT_GT" ] || { echo "ERROR: CKPT_GT is required (no default exists on MN5); see eval_bundle/MN5_EVAL_README.md" >&2; exit 1; }
[ -n "$CKPT_PP" ] || { echo "ERROR: CKPT_PP is required (no default exists on MN5); see eval_bundle/MN5_EVAL_README.md" >&2; exit 1; }
mn5_require f "$CKPT_GT" f "$CKPT_PP" f "$EVAL_SCRIPT" d "$SCENES_ROOT" s "$SCENE_LIST"
mn5_require_slurm

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
