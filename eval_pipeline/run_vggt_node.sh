#!/bin/bash
# Per-node cooperative-queue InfiniteVGGT eval launcher over the DROID test set.
# Same structure as run_augfull_node.sh but uses the infinitevggt conda env and
# the VGGT worker. Submit once per node with --gres/--nodelist/--cpus/--mem and
# LABEL/CKPT/VGGT_ROOT/BASE_SHARD/NUM_SHARDS/LIMIT via --export.
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=vggt_eval
#SBATCH --output=/gpfs/projects/etur59/koc821022/outputs/cut3r_eval/logs/slurm_vggt_%j.out
#SBATCH --error=/gpfs/projects/etur59/koc821022/outputs/cut3r_eval/logs/slurm_vggt_%j.err

set -o pipefail
# --- self-locating worktree -------------------------------------------------
# Run the checkout this job was SUBMITTED from, never a hardcoded path, and fail
# loudly if that is not a my-da3 tree. (Do not use ${BASH_SOURCE[0]} here: SLURM
# copies the batch script to its spool dir, so it would not point at the repo.)
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
# NOT AVAILABLE ON MN5: the `infinitevggt` env does not exist here and neither
# does the InfiniteVGGT checkout it wraps. Without `set -e` a failed activate
# used to fall through and run the whole loop under the base interpreter, so
# check explicitly. (InfiniteVGGT is not part of the CUT3R eval -- it only
# supplies the zero-shot comparison row.)
VGGT_ENV=${VGGT_ENV:-infinitevggt}
conda activate "$VGGT_ENV" || {
  echo "ERROR: conda env '$VGGT_ENV' not found. InfiniteVGGT is not installed on" >&2
  echo "       MareNostrum5; see eval_bundle/MN5_EVAL_README.md (Not available)." >&2
  exit 1; }

# --- MN5 storage anchors (shared; every value overridable) -------------------
source "$WT/eval_pipeline/mn5_paths.sh"
MYDA3=$WT

LABEL=${LABEL:-infinitevggt}
# TODO(mn5): streaming-3d/InfiniteVGGT is not on MareNostrum5.
VGGT_ROOT=${VGGT_ROOT:-/scratch/bdursun25/streaming-3d/InfiniteVGGT}
CKPT=${CKPT:-$VGGT_ROOT/.ckpt/model.safetensors}
CLAIM_DIR=$OUT/$LABEL/claims
BASE_SHARD=${BASE_SHARD:-0}
NUM_SHARDS=${NUM_SHARDS:-16}
LIMIT=${LIMIT:-0}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mn5_require d "$VGGT_ROOT" f "$CKPT" f "$EVAL_SCRIPT" d "$SCENES_ROOT" s "$SCENE_LIST"
mn5_require_slurm

mkdir -p "$OUT/logs" "$CLAIM_DIR"

NUM_GPUS=$(mn5_require_gpus) || exit 1
echo "Node: $(hostname)  GPUs: $NUM_GPUS  LABEL=$LABEL  BASE_SHARD=$BASE_SHARD NUM_SHARDS=$NUM_SHARDS LIMIT=$LIMIT"
echo "Ckpt: $CKPT"
echo "Scenes: $(wc -l < "$SCENE_LIST")  Start: $(date)"

for g in $(seq 0 $((NUM_GPUS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=$g
    SHARD=$((BASE_SHARD + g))
    echo "[$(hostname) g$g shard$SHARD] start $(date)"
    python "$MYDA3/eval_pipeline/infer_and_eval_worker_vggt.py" \
      --ckpt "$CKPT" --label "$LABEL" --vggt_root "$VGGT_ROOT" \
      --scenes_root "$SCENES_ROOT" --scene_list "$SCENE_LIST" \
      --pred_base "$OUT/$LABEL/preds" --eval_base "$OUT/$LABEL/eval" \
      --eval_script "$EVAL_SCRIPT" \
      --claim_dir "$CLAIM_DIR" --limit "$LIMIT" \
      --shard_id "$SHARD" --num_shards "$NUM_SHARDS" --device cuda \
      --ignore_skip_sentinel \
      > "$OUT/logs/worker_$(hostname)_g${g}_${LABEL}.log" 2>&1
    echo "[$(hostname) g$g shard$SHARD] done rc=$? $(date)"
  ) &
done
wait
echo "Node $(hostname) done: $(date)"
