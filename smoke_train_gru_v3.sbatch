#!/bin/bash
#SBATCH --job-name=smoke_gru_v3
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_gru_v3_%A.out
#SBATCH --error=logs/smoke_gru_v3_%A.err
#
# MareNostrum5 single-GPU SMOKE TEST of train_cut3r_baseline.py with the newest
# lever set (A4 x G3 x F1 x R8). Proves the trainer runs cleanly end to end here;
# the numbers are meaningless (2 epochs over 16 samples).
#
#   sbatch smoke_train_gru_v3.sbatch [config_name]
#
# W&B: MN5 compute nodes have NO outbound internet, so this runs OFFLINE and
# writes a replayable transaction log. Port it afterwards from the LOGIN node:
#   ./wandb_port_offline_runs.sh <output_dir>
# That reproduces the online run exactly -- see the script's header for why.
#
set -eo pipefail

# --- self-locating worktree -------------------------------------------------
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
cd "$WT"
mkdir -p logs

CONFIG_NAME=${1:-captain_gru_v3_a4_g3_f1_r8_smoke}

source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything

# Three-tree path + cd into src/CUT3R/src (matches train_captain_gru_overfit.sbatch
# exactly). src/CUT3R is REQUIRED or the optional eval.monodepth import fails in a
# try/except and the absrel/a1 depth metrics silently vanish -- one of the things
# this smoke test is meant to confirm.
export PYTHONPATH=$WT:$WT/src:$WT/src/CUT3R:${PYTHONPATH:-}
export DL3DV_CACHE_DIR=/gpfs/scratch/etur59/koc821022/.dl3dv_cache
export HYDRA_FULL_ERROR=1

# --- W&B offline ------------------------------------------------------------
# Compute nodes cannot reach api.wandb.ai. Offline mode needs no credentials and
# records the SAME transaction log an online run would stream; `wandb sync` on the
# login node replays it. Do NOT unset the proxy vars -- they are harmless here.
export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-pointworld-droid-v2}"
# Pin the run id so the offline dir, wandb_run_id.txt and the synced online run
# all agree. Without this the trainer generates a fresh id per launch (by design).
export WANDB_RUN_ID="${WANDB_RUN_ID:-smoke$(date +%Y%m%d%H%M%S)}"

echo "=================================================================="
echo "node       : $(hostname)"
echo "worktree   : $WT"
echo "config     : $CONFIG_NAME"
echo "scene      : $(cat /gpfs/scratch/etur59/koc821022/smoke/SCENE.txt 2>/dev/null)"
echo "WANDB_MODE : $WANDB_MODE   id=$WANDB_RUN_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=================================================================="

# No --mixed_precision flag: the trainer constructs Accelerator(mixed_precision="bf16")
# itself (train_cut3r_baseline.py:135-137), so passing one here only risks a conflict.
# No --multi_gpu either -- this is a single-process run.
cd "$WT/src/CUT3R/src"
accelerate launch --num_processes 1 \
  train_cut3r_baseline.py --config-name "$CONFIG_NAME"

echo "SMOKE RUN FINISHED (exit 0)"
