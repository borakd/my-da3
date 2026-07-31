#!/bin/bash
#SBATCH --job-name=prevpred_base
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prevpred_base_%A.out
#SBATCH --error=logs/prevpred_base_%A.err
# GRU-less control for the single-hardware companion table: the CURRENT prev_pred
# overfit run (captain_ray_prev_pred, 50 ep, same wrist_test scene, no pose_gru),
# evaluated on the SAME A6000 hardware and code path as the 16 GRU arms.
# The table's published baseline row instead comes from preds dated 10 Jul, which
# predate every surviving prev_pred overfit checkpoint.
set -o pipefail
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything

ROOT=/scratch/bdursun25/cuteanything
CRAY=$ROOT/captain_gru_v3
CUT3R_DIR=$CRAY/src/CUT3R
OUT=$ROOT/outputs/cut3r_eval/overfit_test_scene/gru_grid_final
export PYTHONPATH="$CRAY/src:$CUT3R_DIR:$CUT3R_DIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CRAY" || exit 1

python "$CRAY/eval_pipeline/infer_and_eval_worker_ray.py" \
  --ckpt "$ROOT/checkpoints/captain_cut3r_sim3rmse/captain_ray_prev_pred/checkpoint-final.pth" \
  --label prev_pred_base --size 320 --conditioning prev_pred \
  --scenes_root /frozen/avg/bora_data/droid_datasets/training_data/pointworld_droid_wrist_test/dl3dv_multi \
  --scene_list "$OUT/scene_list_wrist_test.txt" \
  --pred_base "$OUT/prev_pred_base/preds" --eval_base "$OUT/prev_pred_base/eval" \
  --eval_script /scratch/bdursun25/streaming-3d/eval_depth_poses.py \
  --cut3r_dir "$CUT3R_DIR" --shard_id 0 --num_shards 1 --device cuda
rc=$?
echo "50" > "$OUT/prev_pred_base/epochs.txt"
echo "DONE rc=$rc $(date)"
exit $rc
