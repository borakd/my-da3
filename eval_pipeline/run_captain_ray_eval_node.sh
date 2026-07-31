#!/bin/bash
# Per-node cooperative-queue eval launcher for the captain_ray (GT ray-map
# conditioned) checkpoint over all 4292 DROID test scenes.
#
# Submit once per node (default: full L40S node). Every worker on every node
# points at the SAME claim_dir + output dirs and atomically claims scenes via
# mkdir, so any subset of nodes that start will cooperatively drain one shared
# queue (no double-processing, no stall if a node never starts).
#
# Env (passed via --export):
#   LABEL        output label under $OUT      (default captain_ray_gt_last)
#   CKPT         checkpoint path              (default captain_ray_finetune_aug_full/checkpoint-last.pth)
#   CONDITIONING demo_ray conditioning mode   (default gt)
#   BASE_SHARD   global shard offset for this node's GPUs (default 0)
#   NUM_SHARDS   total planned workers, only affects starting offset (default 12)
#   LIMIT        max scenes per worker, 0 = all (default 0; for smoke tests)
#
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --gres=gpu:lovelace_l40s:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=450G
#SBATCH --time=24:00:00
#SBATCH --job-name=cray_eval
#SBATCH --output=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_cray_%j.out
#SBATCH --error=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_cray_%j.err

# Source conda directly (NOT ~/.bashrc, whose line 1 is corrupted -> fatal under
# set -e). No 'set -u' (MKL activation reads unset vars).
set -o pipefail
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything

ROOT=/scratch/bdursun25/cuteanything
CRAY=$ROOT/captain_gru_v3
CUT3R_DIR=$CRAY/src/CUT3R
EVAL_SCRIPT=/scratch/bdursun25/streaming-3d/eval_depth_poses.py
SCENES_ROOT=$ROOT/scenes/pointworld_droid_splits/test/dl3dv_multi/wrist
OUT=$ROOT/outputs/cut3r_eval
SCENE_LIST=$OUT/scene_list.txt
LABEL=${LABEL:-captain_ray_gt_last}
CKPT=${CKPT:-$ROOT/checkpoints/cut3r_multinode/captain_ray_finetune_aug_full/checkpoint-last.pth}
CONDITIONING=${CONDITIONING:-gt}
CLAIM_DIR=$OUT/$LABEL/claims

BASE_SHARD=${BASE_SHARD:-0}
NUM_SHARDS=${NUM_SHARDS:-12}
LIMIT=${LIMIT:-0}

export PYTHONPATH="$CRAY/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs" "$CLAIM_DIR"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Node: $(hostname)  GPUs visible: $NUM_GPUS  BASE_SHARD=$BASE_SHARD  NUM_SHARDS=$NUM_SHARDS  LIMIT=$LIMIT"
echo "Ckpt: $CKPT  conditioning=$CONDITIONING  label=$LABEL"
echo "Scenes: $(wc -l < "$SCENE_LIST")   Start: $(date)"

# One worker process per GPU; all share the cooperative claim queue.
for g in $(seq 0 $((NUM_GPUS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=$g
    SHARD=$((BASE_SHARD + g))
    echo "[$(hostname) g$g shard$SHARD] starting at $(date)"
    python "$CRAY/eval_pipeline/infer_and_eval_worker_ray.py" \
      --ckpt "$CKPT" --label "$LABEL" --size 320 \
      --conditioning "$CONDITIONING" \
      --scenes_root "$SCENES_ROOT" --scene_list "$SCENE_LIST" \
      --pred_base "$OUT/$LABEL/preds" --eval_base "$OUT/$LABEL/eval" \
      --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
      --shard_id "$SHARD" --num_shards "$NUM_SHARDS" --device cuda \
      --claim_dir "$CLAIM_DIR" --limit "$LIMIT" \
      > "$OUT/logs/worker_cray_$(hostname)_g${g}_${LABEL}.log" 2>&1
    echo "[$(hostname) g$g shard$SHARD] finished rc=$? at $(date)"
  ) &
done
wait
echo "All workers on $(hostname) done: $(date)"
