#!/bin/bash
# Tiny recovery pass: re-run the OOM-hardened worker for 'best' over the full
# scene list on ONE GPU. It skips the 4291 scenes that already have a CSV and
# only (re)processes the 1 giant scene that OOM'd during best's pre-hardening
# pass, so best reaches 4292/4292 like final/regular.
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --job-name=fixbest
#SBATCH --output=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_fixbest_%j.out
#SBATCH --error=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_fixbest_%j.err

set -o pipefail
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything

ROOT=/scratch/bdursun25/cuteanything
MYDA3=$ROOT/my-da3
CUT3R_DIR=$MYDA3/src/CUT3R
EVAL_SCRIPT=/scratch/bdursun25/streaming-3d/eval_depth_poses.py
SCENES_ROOT=$ROOT/scenes/pointworld_droid_splits/test/dl3dv_multi/wrist
OUT=$ROOT/outputs/cut3r_eval
BEST_CKPT=$ROOT/checkpoints/cut3r_multinode/multinode_finetune_aug_mini/checkpoint-best.pth
export PYTHONPATH="$MYDA3/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python "$MYDA3/eval_pipeline/infer_and_eval_worker.py" \
  --ckpt "$BEST_CKPT" --label best --size 320 \
  --scenes_root "$SCENES_ROOT" --scene_list "$OUT/scene_list.txt" \
  --pred_base "$OUT/best/preds" --eval_base "$OUT/best/eval" \
  --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
  --shard_id 0 --num_shards 1 --device cuda --ignore_skip_sentinel \
  > "$OUT/logs/worker_FIXg0_best.log" 2>&1
echo "fixbest done: $(date)"
