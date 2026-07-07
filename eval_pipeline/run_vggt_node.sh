#!/bin/bash
# Per-node cooperative-queue InfiniteVGGT eval launcher over the DROID test set.
# Same structure as run_augfull_node.sh but uses the infinitevggt conda env and
# the VGGT worker. Submit once per node with --gres/--nodelist/--cpus/--mem and
# LABEL/CKPT/VGGT_ROOT/BASE_SHARD/NUM_SHARDS/LIMIT via --export.
#
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=vggt_eval
#SBATCH --output=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_vggt_%j.out
#SBATCH --error=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_vggt_%j.err

set -o pipefail
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate infinitevggt

ROOT=/scratch/bdursun25/cuteanything
MYDA3=$ROOT/my-da3
EVAL_SCRIPT=/scratch/bdursun25/streaming-3d/eval_depth_poses.py
SCENES_ROOT=$ROOT/scenes/pointworld_droid_splits/test/dl3dv_multi/wrist
OUT=$ROOT/outputs/cut3r_eval
SCENE_LIST=$OUT/scene_list.txt

LABEL=${LABEL:-infinitevggt}
VGGT_ROOT=${VGGT_ROOT:-/scratch/bdursun25/streaming-3d/InfiniteVGGT}
CKPT=${CKPT:-$VGGT_ROOT/.ckpt/model.safetensors}
CLAIM_DIR=$OUT/$LABEL/claims
BASE_SHARD=${BASE_SHARD:-0}
NUM_SHARDS=${NUM_SHARDS:-16}
LIMIT=${LIMIT:-0}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs" "$CLAIM_DIR"

NUM_GPUS=$(nvidia-smi -L | wc -l)
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
