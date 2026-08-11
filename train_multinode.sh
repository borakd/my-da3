#!/bin/bash
#SBATCH --job-name=cut3r_multinode
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=5                       # <-- number of nodes
#SBATCH --ntasks-per-node=1             # ONE srun task per node; accelerate spawns the per-GPU procs
#SBATCH --gres=gpu:4      # <-- GPUs per node
#SBATCH --cpus-per-task=60              # ~12 cpus/gpu * 4 gpus   # was 48; MN5 derives mem from cores (8G/core) -> 480G ~= old --mem=480G
#SBATCH --time=72:00:00   # was 168:00:00; acc_ehpc MaxWall is 3-00:00:00
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
  # Source the conda hook directly rather than ~/.bashrc: an interactive rc is not
  # guaranteed to define `conda` in a non-interactive srun shell, and under set -e a
  # non-zero return from it aborts the task before training starts.
  source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
  conda activate cuteanything
  # This is the one launcher that already honoured $SLURM_SUBMIT_DIR -- i.e. it always
  # ran the checkout you submitted from. Kept, with a loud check added.
  cd "'"$SLURM_SUBMIT_DIR"'"
  [ -f "src/CUT3R/src/train_cut3r_baseline.py" ] || {
    echo "ERROR: submit dir is not a my-da3 checkout: $PWD" >&2; exit 1; }
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
