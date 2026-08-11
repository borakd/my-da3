#!/bin/bash
#SBATCH --job-name=captain_gru_mn
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=6                       # 6 nodes x 3 GPUs = 18 GPUs (world size KEPT at 18 so
#SBATCH --ntasks-per-node=1             # ONE srun task per node; accelerate spawns the 3 per-GPU procs
#SBATCH --gres=gpu:3                    # results stay comparable to the avg-cluster runs; MN5 acc
                                        # nodes have 4 GPUs, so 5x4 would also give 20 -- see NOTE)
#SBATCH --cpus-per-task=32              # was 24 (capped by the old ai17's 24 cores); raised because MN5
                                        # derives memory from cores (8G/core) -> 256G ~= the old --mem=250G.
                                        # num_workers=7 below still fits (3*7+3 <= 32).
#SBATCH --time=72:00:00   # was 168:00:00; acc_ehpc MaxWall is 3-00:00:00
#SBATCH --output=logs/captain_gru_mn_%A.out
#SBATCH --error=logs/captain_gru_mn_%A.err
# Identical launch recipe to captain_ray's train_captain_ray_18gpu.sh (the
# captain_ray_prev_pred run), repointed at the captain_gru_v2 worktree with
# default config captain_gru_v2. Submit from the worktree root:
#   sbatch train_captain_gru_18gpu.sh                # -> captain_gru_v2
#   sbatch train_captain_gru_18gpu.sh <config_name>  # any other config
#
# NOTE(mn5): the old --nodelist=ai15,ai17,ai29,ai30,ai32,ai33 was REMOVED (those
# nodes do not exist on MareNostrum5 and the job would never schedule). The 6x3
# topology is otherwise kept verbatim: MN5 acc nodes are a uniform 4xH100, so
# 5 nodes x 4 GPUs would be the natural shape, but that changes the world size
# from 18 to 20 and with it the global batch size and effective LR schedule.
# Retuning that is a science decision, not a migration one -- do it deliberately.

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
  # Source the conda hook directly so the job never depends on an interactive rc.
  source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
  conda activate '"$CONDA_ENV"'

  # --- self-locating worktree -----------------------------------------------
  # Run the checkout this job was SUBMITTED from, never a hardcoded path, and
  # fail loudly if that is not a my-da3 tree. $SLURM_SUBMIT_DIR is propagated to
  # every node by srun. (Not ${BASH_SOURCE[0]}: SLURM runs a spool copy.)
  WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
  [ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
    echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
  cd "$WT/src/CUT3R/src"

  export DL3DV_CACHE_DIR=/gpfs/scratch/etur59/koc821022/.dl3dv_cache
  # Full three-tree path: src/CUT3R is required for the optional eval.monodepth
  # import (absrel/a1 depth metrics) — silently skipped without it.
  export PYTHONPATH=$WT:$WT/src:$WT/src/CUT3R:${PYTHONPATH:-}
  export HYDRA_FULL_ERROR=1
  # Kept from the avg cluster, where PCIe P2P was broken (ai15, kernel 4.18) and
  # DDP init hung with all GPUs at 100% util. UNVERIFIED on MN5 (NVLink H100s),
  # where it may cost throughput — change it as a deliberate experiment, not as
  # part of the path migration.
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
