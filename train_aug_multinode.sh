#!/bin/bash
#SBATCH --job-name=cut3r_aug_mn
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --nodes=5                       # 5 nodes
#SBATCH --ntasks-per-node=1             # ONE srun task per node; accelerate spawns the 4 per-GPU procs
#SBATCH --gres=gpu:lovelace_l40s:4      # 4 L40S GPUs per node  -> 20 GPUs total
#SBATCH --cpus-per-task=48              # full node: 4 GPUs * 10 num_workers (40) + main procs/pin threads
#SBATCH --mem=450G
#SBATCH --time=168:00:00
#SBATCH --output=logs/aug_mn_%A.out
#SBATCH --error=logs/aug_mn_%A.err

set -euo pipefail
mkdir -p logs

# ---- knobs ----
CONDA_ENV=cuteanything
CONFIG_NAME=cut3r_pointworld_droid_aug_multinode
GPUS_PER_NODE=4

# ---- topology ----
NNODES=${SLURM_JOB_NUM_NODES}
NUM_PROCESSES=$(( NNODES * GPUS_PER_NODE ))   # 5 * 4 = 20

# ---- rendezvous head node (first node in the allocation) ----
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))   # stable per-job port, avoids collisions

echo "NNODES=$NNODES  NUM_PROCESSES=$NUM_PROCESSES  MASTER=$MASTER_ADDR:$MASTER_PORT  CONFIG=$CONFIG_NAME"

# srun launches the body ONCE PER NODE (ntasks-per-node=1) on every allocated node at the same
# time. $SLURM_NODEID is set per task by srun, so it must be expanded ON THE NODE -> keep it
# inside the single-quoted body. Outer vars are injected by concatenation.
# This mirrors the single-node command exactly (cd into src, same env vars, same accelerate
# launch + --config-name); only the multinode rendezvous flags are added.
srun --ntasks="$NNODES" --ntasks-per-node=1 bash -c '
  # NOTE: no "-u" (nounset) here — conda env activation scripts (e.g. the MKL
  # libblas_mkl_activate.sh) reference unset vars and would abort under nounset.
  set -eo pipefail
  # Source conda directly (NOT ~/.bashrc: its line 1 is corrupted ("\# .bashrc") and
  # returns 127, which under set -e aborts the task before training starts).
  source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
  conda activate '"$CONDA_ENV"'
  cd /scratch/bdursun25/cuteanything/my-da3/src/CUT3R/src

  export DL3DV_CACHE_DIR=/scratch/bdursun25/cuteanything/.dl3dv_cache
  export PYTHONPATH=/scratch/bdursun25/cuteanything/my-da3:${PYTHONPATH:-}
  export HYDRA_FULL_ERROR=1
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
    train_cut3r_baseline.py --config-name '"$CONFIG_NAME"'
'
