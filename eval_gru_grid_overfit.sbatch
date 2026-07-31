#!/bin/bash
#SBATCH --job-name=gru_grid_eval
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gru_grid_eval_%A.out
#SBATCH --error=logs/gru_grid_eval_%A.err
# Single-scene (overfit wrist_test episode) inference + eval for PoseGRU grid
# checkpoints, via the same worker/eval path as the captain_ray table rows:
# infer_and_eval_worker_ray.py --conditioning prev_pred_gru, then
# eval_depth_poses.py with ALL DEFAULT args (sim3 align, median depth align,
# rmse pose reduce — the current single-test-scene table convention).
#   sbatch eval_gru_grid_overfit.sbatch <label>=<ckpt.pth> [<label>=<ckpt> ...]
# Outputs: $OUT/overfit_test_scene/gru_grid/<label>/{preds,eval}/<scene>/
set -o pipefail
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything

ROOT=/scratch/bdursun25/cuteanything
CRAY=$ROOT/captain_gru_v3
CUT3R_DIR=$CRAY/src/CUT3R
EVAL_SCRIPT=/scratch/bdursun25/streaming-3d/eval_depth_poses.py
SCENES_ROOT=/frozen/avg/bora_data/droid_datasets/training_data/pointworld_droid_wrist_test/dl3dv_multi
OUT=$ROOT/outputs/cut3r_eval/overfit_test_scene/gru_grid
SCENE_LIST=$OUT/scene_list_wrist_test.txt

export PYTHONPATH="$CRAY/src:$CUT3R_DIR:$CUT3R_DIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" logs
echo "RAIL+eh61f232+2023-10-26-17h-33m-59s/13062452+wrist" > "$SCENE_LIST"

rc_all=0
for spec in "$@"; do
  LABEL=${spec%%=*}
  CKPT=${spec#*=}
  # label=@<run_dir> resolves the newest SAFE checkpoint at runtime:
  # checkpoint-final if the run has completed, else the highest keep_freq
  # snapshot (never checkpoint-last/best — rewritten every epoch mid-run).
  # The resolved epoch count is recorded in $OUT/$LABEL/epochs.txt.
  if [ "${CKPT:0:1}" = "@" ]; then
    RUN_DIR=${CKPT:1}
    if [ -f "$RUN_DIR/checkpoint-final.pth" ]; then
      CKPT=$RUN_DIR/checkpoint-final.pth
      EPOCHS=50
    else
      LAST=$(ls "$RUN_DIR" | grep -oE 'checkpoint-[0-9]+\.pth' | sed 's/checkpoint-//;s/\.pth//' | sort -n | tail -1)
      CKPT=$RUN_DIR/checkpoint-$LAST.pth
      EPOCHS=$LAST
    fi
    mkdir -p "$OUT/$LABEL"
    echo "$EPOCHS" > "$OUT/$LABEL/epochs.txt"
    echo "=== $LABEL resolved: $CKPT (epochs=$EPOCHS)"
  fi
  echo "=== $LABEL  ckpt=$CKPT  $(date) ==="
  python "$CRAY/eval_pipeline/infer_and_eval_worker_ray.py" \
    --ckpt "$CKPT" --label "$LABEL" --size 320 \
    --conditioning prev_pred_gru \
    --scenes_root "$SCENES_ROOT" --scene_list "$SCENE_LIST" \
    --pred_base "$OUT/$LABEL/preds" --eval_base "$OUT/$LABEL/eval" \
    --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
    --shard_id 0 --num_shards 1 --device cuda
  rc=$?
  echo "=== $LABEL rc=$rc ==="
  [ $rc -ne 0 ] && rc_all=$rc
done
exit $rc_all
