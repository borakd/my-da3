#!/bin/bash
#SBATCH --job-name=cut3r_multinode
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --nodes=5                       # <-- number of nodes
#SBATCH --ntasks-per-node=1             # ONE srun task per node; accelerate spawns the per-GPU procs
#SBATCH --gres=gpu:lovelace_l40s:4      # <-- GPUs per node
#SBATCH --cpus-per-task=48              # ~12 cpus/gpu * 4 gpus
#SBATCH --mem=480G
#SBATCH --time=168:00:00
#SBATCH --output=logs/mn_%A.out
#SBATCH --error=logs/mn_%A.err

set -euo pipefail
mkdir -p logs

# ---- topology ----
GPUS_PER_NODE=4
NNODES=${SLURM_JOB_NUM_NODES}
NUM_PROCESSES=$(( NNODES * GPUS_PER_NODE ))

# ---- rendezvous head node (first node in the allocation) ----
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# derive a stable per-job port to avoid collisions with other jobs on shared nodes
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))

echo "NNODES=$NNODES  NUM_PROCESSES=$NUM_PROCESSES  MASTER=$MASTER_ADDR:$MASTER_PORT"

CONFIG_NAME=${1:-cut3r_pointworld_droid}

# srun launches the body ONCE PER NODE (ntasks-per-node=1) on every allocated node at once.
# $SLURM_NODEID is set per task by srun, so it must be expanded ON THE NODE -> keep it inside
# the single-quoted body. Outer vars (NNODES, MASTER_*, CONFIG_NAME) are injected by concatenation.
srun --ntasks="$NNODES" --ntasks-per-node=1 bash -c '
  set -euo pipefail
  source ~/.bashrc
  conda activate cuteanything
  cd "'"$SLURM_SUBMIT_DIR"'"
  export PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:${PYTHONPATH:-}"

  echo "[node $SLURM_NODEID / host $(hostname)] launching $((1))-machine slice"

  accelerate launch \
    --multi_gpu \
    --num_machines '"$NNODES"' \
    --num_processes '"$NUM_PROCESSES"' \
    --machine_rank "$SLURM_NODEID" \
    --main_process_ip '"$MASTER_ADDR"' \
    --main_process_port '"$MASTER_PORT"' \
    --mixed_precision bf16 \
    src/CUT3R/src/train_cut3r_baseline.py --config-name '"$CONFIG_NAME"'
'
