#!/bin/bash
#SBATCH --job-name=captain_gru_overfit
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:lovelace_l40s:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --output=logs/captain_gru_overfit_%A.out
#SBATCH --error=logs/captain_gru_overfit_%A.err
# Match of the captain_ray_prev_pred overfit run (wandb q4dbmz7e, 2026-07-20:
# single node ai29, 2x L40S, accelerate multi_gpu num_processes=2, no config
# overrides, ~20h wall for 50 epochs), pointed at the captain_gru_v2 worktree
# with default config captain_gru_v2. Submit from the worktree root:
#   sbatch train_captain_gru_overfit.sbatch                # -> captain_gru_v2
#   sbatch train_captain_gru_overfit.sbatch <config_name>  # any other config
# NOTE: source conda directly — ~/.bashrc line 1 is corrupted (fatal under set -e).
set -eo pipefail
mkdir -p logs

CONFIG_NAME=${1:-captain_gru_v2}

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything
cd /scratch/bdursun25/cuteanything/captain_gru_v2/src/CUT3R/src

export DL3DV_CACHE_DIR=/scratch/bdursun25/cuteanything/.dl3dv_cache
export PYTHONPATH=/scratch/bdursun25/cuteanything/captain_gru_v2:${PYTHONPATH:-}
export HYDRA_FULL_ERROR=1
# PCIe P2P is broken on some avg nodes — without this, DDP init can hang with
# all GPUs pinned at 100% util.
export NCCL_P2P_DISABLE=1

accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  train_cut3r_baseline.py --config-name "$CONFIG_NAME"
