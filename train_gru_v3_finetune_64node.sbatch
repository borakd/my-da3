#!/bin/bash
#SBATCH --job-name=gru_v3_ft
# (override per arm on the command line: sbatch --job-name=... <script> <config>)
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=64                      # 64 nodes x 4 H100 = 256 GPUs
#SBATCH --ntasks-per-node=1             # ONE srun task per node; accelerate spawns the 4 per-GPU procs
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80              # whole node (80 physical / 160 logical). NO --mem: MN5
                                        # rejects it and derives memory from cores.
#SBATCH --time=72:00:00                 # acc_ehpc ceiling. Expected runtime ~4-6h (256 H100).
#SBATCH --output=logs/gru_v3_ft_%A.out
#SBATCH --error=logs/gru_v3_ft_%A.err
#
# FULL-DATASET finetune of captain_gru_v3_a4_g0_f1_r8_finetune (A4 x G0 x F1 x R8)
# on the complete DROID wrist splits (38633 train / 4292 test).
#
#   sbatch train_gru_v3_finetune_64node.sbatch [config_name]
#
# Global batch = 64 nodes x 4 GPUs x batch_size 4 = 1024, matching the 256-GPU
# Leonardo reference run exactly, so the loss curves stay comparable.
# num_workers is left at the config's value (10) -- 4 x 10 + main procs fits in 80 cores.
#
# PREREQUISITES (both verified by build_dataset_symlinks.py --check):
#   1. the two-hop symlink layout exists and has ZERO dangling links
#   2. the DL3DV scan cache is warm (see PRE-WARM below) -- otherwise all 256
#      ranks contend on a first-time walk of 42925 scenes while burning allocation
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
CONFIG_NAME=${1:-captain_gru_v3_a4_g0_f1_r8_finetune}
GPUS_PER_NODE=4

# --- fail fast on the dataset, BEFORE burning 64 nodes ----------------------
# A dangling link is silently skipped by dl3dv.py and would train on a short
# dataset without any warning. Check on the head node first.
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
python "$WT/build_dataset_symlinks.py" --check || {
  echo "ERROR: dataset symlink layout is not sound -- refusing to start." >&2; exit 1; }

# --- topology ---------------------------------------------------------------
NNODES=${SLURM_JOB_NUM_NODES}
NUM_PROCESSES=$(( NNODES * GPUS_PER_NODE ))          # 64 * 4 = 256
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))

# Pin the W&B id so the offline dir, wandb_run_id.txt and the synced run agree.
# Derived from the CONFIG NAME so concurrent arms can never collide on one id
# (the run name itself comes from output_dir's basename, i.e. exp_name).
#
# RESUME KEEPS THE SAME RUN: if this arm has run before, its id is persisted in
# <output_dir>/wandb_run_id.txt (written by train_cut3r_baseline.py:168-178).
# Reuse it so a crash-resume / requeue / 72h-continuation appends to the SAME
# wandb run (the trainer inits with resume="allow") instead of minting a new id
# per launch — which is what fragmented f0_noproj into 3 runs on 2026-08-10.
# Minting happens only on the arm's very first launch.
ARM=$(echo "$CONFIG_NAME" | sed 's/^captain_gru_v3_//; s/_finetune$//; s/_//g')
CFG_YAML="$WT/src/CUT3R/config/$CONFIG_NAME.yaml"
SAVE_DIR=$(sed -n "s/^save_dir:[[:space:]]*'\?\([^']*\)'\?.*/\1/p" "$CFG_YAML" | head -n1)
RUN_ID_FILE="$SAVE_DIR/$CONFIG_NAME/wandb_run_id.txt"
if [ -z "$WANDB_RUN_ID" ] && [ -s "$RUN_ID_FILE" ]; then
  export WANDB_RUN_ID="$(tr -d '[:space:]' < "$RUN_ID_FILE")"
  echo "reusing persisted wandb id from $RUN_ID_FILE"
fi
export WANDB_RUN_ID="${WANDB_RUN_ID:-ft${ARM}$(date +%Y%m%d%H%M%S)}"

echo "=================================================================="
echo "config      : $CONFIG_NAME"
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

  # NOTE: NCCL_P2P_DISABLE is deliberately NOT set here. On the old avg cluster it
  # worked around broken PCIe P2P (ai15, kernel 4.18). MN5 acc nodes are NVLinked
  # H100s where disabling P2P would cripple intra-node allreduce -- which matters a
  # great deal at 256 ranks. If rendezvous hangs at startup, try these instead:
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

echo "FINETUNE JOB FINISHED (exit 0)"
