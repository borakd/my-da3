#!/bin/bash
# CPU-only dual-aggregation (mean+rmse) sim3 pose recompute from saved preds.
# Pose-only (camera npzs, no depth) -> ~100x cheaper than a full re-eval.
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=6:00:00
#SBATCH --job-name=sim3_both
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_sim3both_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_sim3both_%j.err

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
# 48 workers each spinning BLAS threads would thrash the node.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

LABELS=${LABELS:-"regular final augfull_last augfull_last_ep14 infinitevggt augfull_final"}

# Pose-only recompute reads the camera npzs a previous inference pass wrote, so
# a missing label is a silently shorter table rather than an error.
mn5_require d "$SCENES_ROOT" s "$SCENE_LIST"

echo "Node: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK}  Labels: $LABELS  Start: $(date)"
python "$WT/eval_pipeline/pose_sim3_both.py" \
  --out_root "$OUT" --scenes_root "$SCENES_ROOT" \
  --scene_list "$OUT/scene_list.txt" \
  --labels $LABELS \
  --workers "${SLURM_CPUS_PER_TASK:-48}"
echo "Done: $(date)"
