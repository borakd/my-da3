#!/bin/bash
#SBATCH --job-name=verify_gru_a5
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15   # was 12; MN5 derives mem from cores (8G/core) -> 120G ~= old --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=logs/verify_gru_a5_%A.out
#SBATCH --error=logs/verify_gru_a5_%A.err
# Correctness suite for the A5 ("anchored A3", pose_gru_mode: split_anchor) arm.
# CPU-only maths, but the ckpt stage instantiates the full model and reads a
# 3.2 GB checkpoint, so it runs on a compute node (never the login node).
#   sbatch verify_gru_a5.sbatch
set -o pipefail
mkdir -p logs
# --- self-locating worktree -------------------------------------------------
# Run the checkout this job was SUBMITTED from, never a hardcoded path, and fail
# loudly if that is not a my-da3 tree. (Do not use ${BASH_SOURCE[0]} here: SLURM
# copies the batch script to its spool dir, so it would not point at the repo.)
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything

export DL3DV_CACHE_DIR=/gpfs/scratch/etur59/koc821022/.dl3dv_cache
export PYTHONPATH=$WT:$WT/src:$WT/src/CUT3R:$WT/src/CUT3R/src:${PYTHONPATH:-}

python "$WT/verify_gru_a5.py"
rc=$?
echo "=== verify_gru_a5 rc=$rc ==="
exit $rc
