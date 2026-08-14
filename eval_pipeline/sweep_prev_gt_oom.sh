#!/bin/bash
# End-sweep for the 2 giant scenes that OOM'd on L40S (44GB) for captain_ray_prev_gt.
# Runs the SAME ray worker (conditioning=prev_gt) on an A6000 (48GB) over only the
# missing_scenes.txt list, into the REAL output dirs (resume-safe).
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=25   # was 12; MN5 derives mem from cores (8G/core) -> 200G ~= old --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=pgt_sweep
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_pgt_sweep_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_pgt_sweep_%j.err

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
LABEL=captain_ray_prev_gt
CKPT=${CKPT:-}
MISS=$OUT/$LABEL/missing_scenes.txt

[ -n "$CKPT" ] || { echo "ERROR: CKPT is required (no default exists on MN5); see eval_bundle/MN5_EVAL_README.md" >&2; exit 1; }
mn5_require f "$CKPT" f "$EVAL_SCRIPT" d "$SCENES_ROOT"
mn5_require_slurm

export PYTHONPATH="$CRAY/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Node: $(hostname)  GPU: $(nvidia-smi -L)  Scenes: $(wc -l < "$MISS")  Start: $(date)"
python "$CRAY/eval_pipeline/infer_and_eval_worker_ray.py" \
  --ckpt "$CKPT" --label "$LABEL" --size 320 --conditioning prev_gt \
  --scenes_root "$SCENES_ROOT" --scene_list "$MISS" \
  --pred_base "$OUT/$LABEL/preds" --eval_base "$OUT/$LABEL/eval" \
  --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
  --shard_id 0 --num_shards 1 --device cuda
echo "Sweep rc=$?  Done: $(date)"
