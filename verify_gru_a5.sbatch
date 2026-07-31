#!/bin/bash
#SBATCH --job-name=verify_gru_a5
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=logs/verify_gru_a5_%A.out
#SBATCH --error=logs/verify_gru_a5_%A.err
# Correctness suite for the A5 ("anchored A3", pose_gru_mode: split_anchor) arm.
# CPU-only maths, but the ckpt stage instantiates the full model and reads a
# 3.2 GB checkpoint, so it runs on a compute node (never the login node).
#   sbatch verify_gru_a5.sbatch
set -o pipefail
mkdir -p logs
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything

WT=/scratch/bdursun25/cuteanything/captain_gru_v3
export DL3DV_CACHE_DIR=/scratch/bdursun25/cuteanything/.dl3dv_cache
export PYTHONPATH=$WT:$WT/src:$WT/src/CUT3R:$WT/src/CUT3R/src:${PYTHONPATH:-}

python "$WT/verify_gru_a5.py"
rc=$?
echo "=== verify_gru_a5 rc=$rc ==="
exit $rc
