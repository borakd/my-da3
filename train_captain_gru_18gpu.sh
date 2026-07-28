#!/bin/bash
#SBATCH --job-name=captain_gru_mn
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --nodes=6                       # 6 nodes x 3 GPUs = 18 GPUs (20 not claimable: avg is fragmented)
#SBATCH --ntasks-per-node=1             # ONE srun task per node; accelerate spawns the 3 per-GPU procs
#SBATCH --gres=gpu:3                    # untyped: mixes lovelace_l40s (ai29/30/32/33) + rtx_a6000 (ai15/17), both 48GB
#SBATCH --nodelist=ai15,ai17,ai29,ai30,ai32,ai33
#SBATCH --cpus-per-task=24              # capped by ai17 (24 cores total); pair with num_workers=7 (3*7+3 <= 24)
#SBATCH --mem=250G                      # ai15 has ~368G free (partially occupied) — stay well under
#SBATCH --time=168:00:00
#SBATCH --output=logs/captain_gru_mn_%A.out
#SBATCH --error=logs/captain_gru_mn_%A.err
# Identical launch recipe to captain_ray's train_captain_ray_18gpu.sh (the
# captain_ray_prev_pred run), repointed at the captain_gru_v2 worktree with
# default config captain_gru_v2. Submit from the worktree root:
#   sbatch train_captain_gru_18gpu.sh                # -> captain_gru_v2
#   sbatch train_captain_gru_18gpu.sh <config_name>  # any other config

set -euo pipefail
mkdir -p logs

# ---- knobs ----
CONDA_ENV=cuteanything
CONFIG_NAME=${1:-captain_gru_v2}
GPUS_PER_NODE=3

# ---- topology ----
NNODES=${SLURM_JOB_NUM_NODES}
NUM_PROCESSES=$(( NNODES * GPUS_PER_NODE ))   # 6 * 3 = 18

# ---- rendezvous head node (first node in the allocation) ----
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))   # stable per-job port, avoids collisions

echo "NNODES=$NNODES  NUM_PROCESSES=$NUM_PROCESSES  MASTER=$MASTER_ADDR:$MASTER_PORT  CONFIG=$CONFIG_NAME"

# srun launches the body ONCE PER NODE (ntasks-per-node=1) on every allocated node at the same
# time. $SLURM_NODEID is set per task by srun, so it must be expanded ON THE NODE -> keep it
# inside the single-quoted body. Outer vars are injected by concatenation.
srun --ntasks="$NNODES" --ntasks-per-node=1 bash -c '
  # NOTE: no "-u" (nounset) here — conda env activation scripts (e.g. the MKL
  # libblas_mkl_activate.sh) reference unset vars and would abort under nounset.
  set -eo pipefail
  # Source conda directly (NOT ~/.bashrc: its line 1 is corrupted ("\# .bashrc") and
  # returns 127, which under set -e aborts the task before training starts).
  source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
  conda activate '"$CONDA_ENV"'
  cd /scratch/bdursun25/cuteanything/captain_gru_v2/src/CUT3R/src

  export DL3DV_CACHE_DIR=/scratch/bdursun25/cuteanything/.dl3dv_cache
  # Full three-tree path: src/CUT3R is required for the optional eval.monodepth
  # import (absrel/a1 depth metrics) — silently skipped without it.
  WT=/scratch/bdursun25/cuteanything/captain_gru_v2
  export PYTHONPATH=$WT:$WT/src:$WT/src/CUT3R:${PYTHONPATH:-}
  export HYDRA_FULL_ERROR=1
  # PCIe P2P is broken on some avg nodes (ai15 confirmed, kernel 4.18) — without this,
  # DDP init hangs with all GPUs pinned at 100% util.
  export NCCL_P2P_DISABLE=1
  # If rendezvous hangs at startup, pin the NCCL interface (find it via `ip addr` on a node):
  # export NCCL_SOCKET_IFNAME=ib0
  # export NCCL_DEBUG=INFO

  echo "[node $SLURM_NODEID @ $(hostname)] launching machine_rank=$SLURM_NODEID"

  accelerate launch \
    --multi_gpu \
    --num_machines '"$NNODES"' \
    --num_processes '"$NUM_PROCESSES"' \
    --machine_rank "$SLURM_NODEID" \
    --main_process_ip '"$MASTER_ADDR"' \
    --main_process_port '"$MASTER_PORT"' \
    train_cut3r_baseline.py --config-name '"$CONFIG_NAME"' num_workers=7
'
