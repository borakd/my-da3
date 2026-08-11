#!/bin/bash
#SBATCH --job-name=verify_pose_gru_e2e
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_pose_gru_e2e_%A.out
#SBATCH --error=logs/verify_pose_gru_e2e_%A.err
# GPU end-to-end verification of the PoseGRU line (steps 1-5) on the real
# checkpoint + real wrist_test data. Submit from the worktree root:
#   sbatch verify_pose_gru_e2e.sbatch
# NOTE: source the conda hook directly so the job never depends on an interactive shell rc.
set -eo pipefail
# --- self-locating worktree -------------------------------------------------
# Run the checkout this job was SUBMITTED from, never a hardcoded path, and fail
# loudly if that is not a my-da3 tree. (Do not use ${BASH_SOURCE[0]} here: SLURM
# copies the batch script to its spool dir, so it would not point at the repo.)
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
cd "$WT"
export PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:$PYTHONPATH"
export DL3DV_CACHE_DIR=/gpfs/scratch/etur59/koc821022/.dl3dv_cache
python verify_pose_gru_e2e.py
