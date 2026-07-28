#!/bin/bash
#SBATCH --job-name=verify_pose_gru_e2e
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_pose_gru_e2e_%A.out
#SBATCH --error=logs/verify_pose_gru_e2e_%A.err
# GPU end-to-end verification of the PoseGRU line (steps 1-5) on the real
# checkpoint + real wrist_test data. Submit from the worktree root:
#   sbatch verify_pose_gru_e2e.sbatch
# NOTE: source conda directly — ~/.bashrc line 1 is corrupted (fatal under set -e).
set -eo pipefail
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything
cd /scratch/bdursun25/cuteanything/captain_gru_v2
export PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:$PYTHONPATH"
export DL3DV_CACHE_DIR=/scratch/bdursun25/cuteanything/.dl3dv_cache
python verify_pose_gru_e2e.py
