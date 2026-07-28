#!/bin/bash
#SBATCH --job-name=verify_prev_flags
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/verify_prev_flags_%A.out
#SBATCH --error=logs/verify_prev_flags_%A.err
# CPU-only verification of feed_prev_gt_ray_map / feed_prev_pred (no GPU needed).
# Submit from the worktree root:  sbatch verify_prev_flags.sbatch
# NOTE: source conda directly — ~/.bashrc line 1 is corrupted (fatal under set -e).
set -eo pipefail
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything
cd /scratch/bdursun25/cuteanything/captain_gru_v2
export PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:$PYTHONPATH"
export DL3DV_CACHE_DIR=/scratch/bdursun25/cuteanything/.dl3dv_cache
python verify_prev_flags.py
