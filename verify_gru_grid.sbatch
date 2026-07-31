#!/bin/bash
#SBATCH --job-name=verify_gru_grid
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:50:00
#SBATCH --output=logs/verify_gru_grid_%A.out
#SBATCH --error=logs/verify_gru_grid_%A.err
# GPU verification of the A1-A4 x G0-G3 PoseGRU lever grid:
#   sbatch verify_gru_grid.sbatch
set -eo pipefail
mkdir -p logs

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything

WT=/scratch/bdursun25/cuteanything/captain_gru_v3
export DL3DV_CACHE_DIR=/scratch/bdursun25/cuteanything/.dl3dv_cache
export PYTHONPATH=$WT:$WT/src:$WT/src/CUT3R:$WT/src/CUT3R/src:${PYTHONPATH:-}

python "$WT/verify_gru_grid.py"
