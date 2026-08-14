#!/bin/bash
# Per-node cooperative-queue eval launcher for the aug_full (checkpoint-last)
# model over all 4292 DROID test scenes.
#
# Designed to be submitted once per node with --gres/--nodelist/--cpus-per-task/
# --mem and BASE_SHARD passed on the sbatch command line. Every worker on every
# node points at the SAME claim_dir + output dirs and atomically claims scenes
# via mkdir, so any subset of nodes that start will cooperatively drain one
# shared queue (no double-processing, no stall if a node never starts).
#
# Env (passed via --export):
#   BASE_SHARD   global shard offset for this node's GPUs (default 0)
#   NUM_SHARDS   total planned workers, only affects starting offset (default 16)
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=augfull_eval
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_augfull_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_augfull_%j.err

# Source the conda hook directly so the job never depends on an interactive shell rc.
# No 'set -u' (MKL activation scripts read unset vars).
set -o pipefail
# --- self-locating worktree -------------------------------------------------
# Run the checkout this job was SUBMITTED from, never a hardcoded path, and fail
# loudly if that is not a my-da3 tree. (Do not use ${BASH_SOURCE[0]} here: SLURM
# copies the batch script to its spool dir, so it would not point at the repo.)
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything

# --- MN5 storage anchors (shared; every value overridable) -------------------
source "$WT/eval_pipeline/mn5_paths.sh"
MYDA3=$WT
LABEL=${LABEL:-augfull_last}
CKPT=${CKPT:-$OUT/$LABEL/checkpoint-augfull-last-snapshot.pth}
CLAIM_DIR=$OUT/$LABEL/claims

BASE_SHARD=${BASE_SHARD:-0}
NUM_SHARDS=${NUM_SHARDS:-16}

mn5_require f "$CKPT" f "$EVAL_SCRIPT" d "$SCENES_ROOT" s "$SCENE_LIST"
mn5_require_slurm

export PYTHONPATH="$MYDA3/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs" "$CLAIM_DIR"

NUM_GPUS=$(mn5_require_gpus) || exit 1
echo "Node: $(hostname)  GPUs visible: $NUM_GPUS  BASE_SHARD=$BASE_SHARD  NUM_SHARDS=$NUM_SHARDS"
echo "Ckpt: $CKPT"
echo "Scenes: $(wc -l < "$SCENE_LIST")   Start: $(date)"

# One worker process per GPU; all share the cooperative claim queue.
for g in $(seq 0 $((NUM_GPUS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=$g
    SHARD=$((BASE_SHARD + g))
    echo "[$(hostname) g$g shard$SHARD] starting at $(date)"
    python "$MYDA3/eval_pipeline/infer_and_eval_worker.py" \
      --ckpt "$CKPT" --label "$LABEL" --size 320 \
      --scenes_root "$SCENES_ROOT" --scene_list "$SCENE_LIST" \
      --pred_base "$OUT/$LABEL/preds" --eval_base "$OUT/$LABEL/eval" \
      --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
      --claim_dir "$CLAIM_DIR" \
      --shard_id "$SHARD" --num_shards "$NUM_SHARDS" --device cuda \
      --ignore_skip_sentinel \
      > "$OUT/logs/worker_$(hostname)_g${g}_${LABEL}.log" 2>&1
    echo "[$(hostname) g$g shard$SHARD] finished rc=$? at $(date)"
  ) &
done
wait
echo "Node $(hostname) workers done: $(date)"
