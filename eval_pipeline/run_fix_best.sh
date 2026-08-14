#!/bin/bash
# Tiny recovery pass: re-run the OOM-hardened worker for 'best' over the full
# scene list on ONE GPU. It skips the 4291 scenes that already have a CSV and
# only (re)processes the 1 giant scene that OOM'd during best's pre-hardening
# pass, so best reaches 4292/4292 like final/regular.
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15   # was 12; MN5 derives mem from cores (8G/core) -> 120G ~= old --mem=120G
#SBATCH --time=2:00:00
#SBATCH --job-name=fixbest
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_fixbest_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_fixbest_%j.err

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
MYDA3=$WT
BEST_CKPT=${BEST_CKPT:-$CKPT_ROOT/cut3r_multinode/multinode_finetune_aug_mini/checkpoint-best.pth}
mn5_require f "$BEST_CKPT" f "$EVAL_SCRIPT" d "$SCENES_ROOT" s "$SCENE_LIST"
mn5_require_slurm

export PYTHONPATH="$MYDA3/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python "$MYDA3/eval_pipeline/infer_and_eval_worker.py" \
  --ckpt "$BEST_CKPT" --label best --size 320 \
  --scenes_root "$SCENES_ROOT" --scene_list "$OUT/scene_list.txt" \
  --pred_base "$OUT/best/preds" --eval_base "$OUT/best/eval" \
  --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
  --shard_id 0 --num_shards 1 --device cuda --ignore_skip_sentinel \
  > "$OUT/logs/worker_FIXg0_best.log" 2>&1
echo "fixbest done: $(date)"
