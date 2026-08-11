#!/bin/bash
#SBATCH --job-name=captain_gru_overfit
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --output=logs/captain_gru_overfit_%A.out
#SBATCH --error=logs/captain_gru_overfit_%A.err
# Match of the captain_ray_prev_pred overfit run (wandb q4dbmz7e, 2026-07-20:
# single node ai29, 2x L40S, accelerate multi_gpu num_processes=2, no config
# overrides, ~20h wall for 50 epochs), pointed at the captain_gru_v2 worktree
# with default config captain_gru_v2. Submit from the worktree root:
#   sbatch train_captain_gru_overfit.sbatch                # -> captain_gru_v2
#   sbatch train_captain_gru_overfit.sbatch <config_name>  # any other config
# NOTE: source the conda hook directly so the job never depends on an interactive
# shell rc. (On the old avg cluster ~/.bashrc line 1 was corrupted and fatal under
# set -e; MN5's ~/.bashrc is clean, but sourcing the hook is still the right thing.)
set -eo pipefail
mkdir -p logs

CONFIG_NAME=${1:-captain_gru_v2}

# --- self-locating worktree -------------------------------------------------
# Run the checkout this job was SUBMITTED from, never a hardcoded path, and fail
# loudly if that is not a my-da3 tree. (Do not use ${BASH_SOURCE[0]} here: SLURM
# copies the batch script to its spool dir, so it would not point at the repo.)
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
cd "$WT/src/CUT3R/src"

export DL3DV_CACHE_DIR=/gpfs/scratch/etur59/koc821022/.dl3dv_cache
# Full three-tree path (worktree root + src + src/CUT3R): src/CUT3R is REQUIRED for
# the optional `eval.monodepth`/`eval.video_depth` import in the test loop — without
# it, absrel/a1 depth metrics are silently skipped ("No module named 'eval'").
export PYTHONPATH=$WT:$WT/src:$WT/src/CUT3R:${PYTHONPATH:-}
export HYDRA_FULL_ERROR=1
# Kept from the avg cluster, where PCIe P2P was broken and DDP init hung with all
# GPUs pinned at 100% util. UNVERIFIED on MN5 (NVLink H100s), where it may cost
# throughput — flip it as a deliberate experiment, not as part of a path migration.
export NCCL_P2P_DISABLE=1

# Unique rendezvous port per job (default 29500 collides when two jobs share
# a node — e.g. two 2-GPU runs packed onto one 4-GPU node).
MAIN_PORT=$((29000 + SLURM_JOB_ID % 1000))

accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  --main_process_port "$MAIN_PORT" \
  train_cut3r_baseline.py --config-name "$CONFIG_NAME"
