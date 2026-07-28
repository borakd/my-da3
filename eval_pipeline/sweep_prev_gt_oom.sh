#!/bin/bash
# End-sweep for the 2 giant scenes that OOM'd on L40S (44GB) for captain_ray_prev_gt.
# Runs the SAME ray worker (conditioning=prev_gt) on an A6000 (48GB) over only the
# missing_scenes.txt list, into the REAL output dirs (resume-safe).
#
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=pgt_sweep
#SBATCH --output=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_pgt_sweep_%j.out
#SBATCH --error=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_pgt_sweep_%j.err

set -o pipefail
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything

ROOT=/scratch/bdursun25/cuteanything
CRAY=$ROOT/captain_gru_v2
CUT3R_DIR=$CRAY/src/CUT3R
EVAL_SCRIPT=/scratch/bdursun25/streaming-3d/eval_depth_poses.py
SCENES_ROOT=$ROOT/scenes/pointworld_droid_splits/test/dl3dv_multi/wrist
OUT=$ROOT/outputs/cut3r_eval
LABEL=captain_ray_prev_gt
CKPT=$ROOT/checkpoints/captain_cut3r_sim3rmse/prev_gt_finetune_aug_full/checkpoint-final.pth
MISS=$OUT/$LABEL/missing_scenes.txt

export PYTHONPATH="$CRAY/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Node: $(hostname)  GPU: $(nvidia-smi -L)  Scenes: $(wc -l < "$MISS")  Start: $(date)"
python "$CRAY/eval_pipeline/infer_and_eval_worker_ray.py" \
  --ckpt "$CKPT" --label "$LABEL" --size 320 --conditioning prev_gt \
  --scenes_root "$SCENES_ROOT" --scene_list "$MISS" \
  --pred_base "$OUT/$LABEL/preds" --eval_base "$OUT/$LABEL/eval" \
  --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
  --shard_id 0 --num_shards 1 --device cuda
echo "Sweep rc=$?  Done: $(date)"
