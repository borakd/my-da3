#!/bin/bash
#SBATCH --job-name=gru_oracle_ft
# (override per arm on the command line: sbatch --job-name=... <script> <config>)
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=8                       # DEFAULT = 8 nodes x 4 H100 = 32 GPUs.
                                        # For the 64-GPU arm pass --nodes=16 on
                                        # the sbatch command line (see below).
#SBATCH --ntasks-per-node=1             # ONE srun task per node; accelerate spawns the 4 per-GPU procs
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80              # whole node (80 physical / 160 logical). NO --mem: MN5
                                        # rejects it and derives memory from cores.
#SBATCH --time=72:00:00                 # acc_ehpc ceiling.
#SBATCH --output=logs/gru_oracle_ft_%A.out
#SBATCH --error=logs/gru_oracle_ft_%A.err
#
# FULL-DATASET ORACLE finetune of the captain_gru_v3 A4 x G3 x F0 x R1 arm on the
# complete DROID wrist splits (38633 train / 4292 test), at 32 or 64 GPUs.
#
#   32 GPUs:  sbatch --nodes=8  train_gru_oracle_finetune.sbatch \
#                    captain_gru_v3_a4_g3_oracle_finetune_32gpu
#   64 GPUs:  sbatch --nodes=16 train_gru_oracle_finetune.sbatch \
#                    captain_gru_v3_a4_g3_oracle_finetune_64gpu
#
# The GPU count comes from --nodes, NOT from the config -- the two configs differ
# only in their header and exp_name, exactly like the
# cut3r_pointworld_droid_aug_mn5_prevpred_{32,64}gpu pair. Because that makes a
# mismatch invisible in the logs (a 64-GPU run writing into the _32gpu exp_name
# directory would look completely normal), this script HARD-FAILS when --nodes
# and the config's _NNgpu suffix disagree. See the guard below.
#
# These are DIAGNOSTIC runs: the PoseGRU is fed the current view's GROUND-TRUTH
# pose. Their checkpoints must never be resumed as a training arm or entered in a
# cross-arm table. See the config header for what to read and when.
#
# PREREQUISITES (both verified by build_dataset_symlinks.py --check):
#   1. the two-hop symlink layout exists and has ZERO dangling links
#   2. the DL3DV scan cache is warm -- otherwise all ranks contend on a
#      first-time walk of 42925 scenes while burning allocation
#
# W&B runs OFFLINE (compute nodes have no internet). Port it afterwards from the
# login node with ./wandb_port_offline_runs.sh <output_dir>.
#
set -eo pipefail

# --- self-locating worktree -------------------------------------------------
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
cd "$WT"
mkdir -p logs

CONDA_ENV=cuteanything
CONFIG_NAME=${1:-captain_gru_v3_a4_g3_oracle_finetune_32gpu}
GPUS_PER_NODE=4

# --- topology ---------------------------------------------------------------
NNODES=${SLURM_JOB_NUM_NODES}
NUM_PROCESSES=$(( NNODES * GPUS_PER_NODE ))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))

# --- guard: --nodes must match the config's _NNgpu suffix -------------------
# Nothing downstream can catch this. exp_name (hence output_dir, the W&B run name
# and the checkpoint directory) comes from the config, while the world size comes
# from --nodes, so a mismatched pair produces a perfectly healthy-looking run
# whose name lies about its global batch. Refuse instead.
CFG_GPUS=$(echo "$CONFIG_NAME" | sed -n 's/.*_\([0-9][0-9]*\)gpu$/\1/p')
if [ -n "$CFG_GPUS" ] && [ "$CFG_GPUS" -ne "$NUM_PROCESSES" ]; then
  echo "ERROR: config '$CONFIG_NAME' declares ${CFG_GPUS} GPUs but this allocation" >&2
  echo "       is ${NNODES} nodes x ${GPUS_PER_NODE} = ${NUM_PROCESSES} processes." >&2
  echo "       Use --nodes=$(( CFG_GPUS / GPUS_PER_NODE )) for this config, or pick" >&2
  echo "       the config matching ${NUM_PROCESSES} GPUs. Refusing to start." >&2
  exit 1
fi
if [ -z "$CFG_GPUS" ]; then
  echo "WARNING: config '$CONFIG_NAME' has no _NNgpu suffix -- the GPU-count guard" >&2
  echo "         cannot check it. Running at ${NUM_PROCESSES} processes." >&2
fi

# --- fail fast on the dataset, BEFORE burning the allocation ----------------
# A dangling link is silently skipped by dl3dv.py and would train on a short
# dataset without any warning. Check on the head node first.
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
python "$WT/build_dataset_symlinks.py" --check || {
  echo "ERROR: dataset symlink layout is not sound -- refusing to start." >&2; exit 1; }

# Pin the W&B id so the offline dir, wandb_run_id.txt and the synced run agree.
# Derived from the CONFIG NAME so concurrent arms can never collide on one id
# (the run name itself comes from output_dir's basename, i.e. exp_name).
ARM=$(echo "$CONFIG_NAME" | sed 's/^captain_gru_v3_//; s/_finetune$//; s/_//g')
export WANDB_RUN_ID="${WANDB_RUN_ID:-ft${ARM}$(date +%Y%m%d%H%M%S)}"

echo "=================================================================="
echo "config      : $CONFIG_NAME   (ORACLE DIAGNOSTIC RUN)"
echo "nodes       : $NNODES x $GPUS_PER_NODE GPU = $NUM_PROCESSES processes"
echo "global batch: $NUM_PROCESSES x batch_size(4) = $(( NUM_PROCESSES * 4 ))"
echo "master      : $MASTER_ADDR:$MASTER_PORT"
echo "wandb id    : $WANDB_RUN_ID (offline)"
echo "=================================================================="

srun --ntasks="$NNODES" --ntasks-per-node=1 bash -c '
  # No "-u": conda activation scripts read unset vars and would abort under nounset.
  set -eo pipefail
  source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
  conda activate '"$CONDA_ENV"'

  WT="${SLURM_SUBMIT_DIR:-$PWD}"
  cd "$WT/src/CUT3R/src"

  # Three-tree path; src/CUT3R is REQUIRED or the optional eval.monodepth import
  # fails in a try/except and the absrel/a1 depth metrics silently vanish.
  export PYTHONPATH=$WT:$WT/src:$WT/src/CUT3R:${PYTHONPATH:-}
  export HYDRA_FULL_ERROR=1
  export DL3DV_CACHE_DIR=/gpfs/scratch/etur59/koc821022/.dl3dv_cache_wrist_all

  # W&B offline: MN5 compute nodes have no outbound internet.
  export WANDB_MODE=offline
  export WANDB_PROJECT="${WANDB_PROJECT:-pointworld-droid-v2}"
  export WANDB_RUN_ID='"$WANDB_RUN_ID"'

  # NOTE: NCCL_P2P_DISABLE is deliberately NOT set. MN5 acc nodes are NVLinked
  # H100s where disabling P2P would cripple intra-node allreduce. If rendezvous
  # hangs at startup, try instead:
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

echo "ORACLE FINETUNE JOB FINISHED (exit 0)"
