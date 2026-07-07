#!/bin/bash
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --gres=gpu:lovelace_l40s:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=450G
#SBATCH --time=72:00:00
#SBATCH --job-name=cut3r_eval
#SBATCH --output=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_%j.out
#SBATCH --error=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_%j.err

# NOTE: source conda directly (NOT ~/.bashrc, whose line 1 is corrupted and is
# fatal under set -e). No 'set -u' (MKL activation reads unset vars).
set -o pipefail
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything

ROOT=/scratch/bdursun25/cuteanything
MYDA3=$ROOT/my-da3
CUT3R_DIR=$MYDA3/src/CUT3R
EVAL_SCRIPT=/scratch/bdursun25/streaming-3d/eval_depth_poses.py
SCENES_ROOT=$ROOT/scenes/pointworld_droid_splits/test/dl3dv_multi/wrist
OUT=$ROOT/outputs/cut3r_eval
SCENE_LIST=$OUT/scene_list.txt
CKPT_DIR=$ROOT/checkpoints/cut3r_multinode/multinode_finetune_aug_mini
REG_CKPT=$CUT3R_DIR/src/cut3r_512_dpt_4_64.pth

export PYTHONPATH="$MYDA3/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
# Reduce CUDA fragmentation OOMs on very long sequences.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs"

declare -A CKPTS
CKPTS[best]=$CKPT_DIR/checkpoint-best.pth
CKPTS[final]=$CKPT_DIR/checkpoint-final.pth
CKPTS[regular]=$REG_CKPT

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Node: $(hostname)  GPUs visible: $NUM_GPUS  Scenes: $(wc -l < "$SCENE_LIST")"
echo "Start: $(date)"

# One worker process per GPU; each runs all 3 checkpoints sequentially over its
# round-robin scene shard (model loaded once per checkpoint -> 3 loads/GPU).
for g in $(seq 0 $((NUM_GPUS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=$g
    for label in best final regular; do
      echo "[g$g] starting $label at $(date)"
      python "$MYDA3/eval_pipeline/infer_and_eval_worker.py" \
        --ckpt "${CKPTS[$label]}" --label "$label" --size 320 \
        --scenes_root "$SCENES_ROOT" --scene_list "$SCENE_LIST" \
        --pred_base "$OUT/$label/preds" --eval_base "$OUT/$label/eval" \
        --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
        --shard_id "$g" --num_shards "$NUM_GPUS" --device cuda \
        > "$OUT/logs/worker_g${g}_${label}.log" 2>&1
      echo "[g$g] finished $label rc=$? at $(date)"
    done
  ) &
done
wait
echo "All workers done: $(date)"

# Aggregate everything into the comparison table.
python "$MYDA3/eval_pipeline/aggregate_results.py" \
  --out_root "$OUT" --scene_list "$SCENE_LIST" \
  > "$OUT/logs/aggregate.log" 2>&1
echo "Aggregation done: $(date)"
cat "$OUT/summary/averages_table.txt" || true
