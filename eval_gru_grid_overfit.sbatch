#!/bin/bash
#SBATCH --job-name=gru_grid_eval
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
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
CRAY=$WT
SCENES_ROOT=$OVERFIT_ROOT
OUT=$OUT_ROOT/cut3r_eval/overfit_test_scene/gru_grid
SCENE_LIST=$OUT/scene_list_wrist_test.txt

export PYTHONPATH="$CRAY/src:$CUT3R_DIR:$CUT3R_DIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ $# -gt 0 ] || { echo "usage: sbatch $0 <label>=<ckpt.pth> [<label>=@<run_dir> ...]" >&2
                  echo "  (with no arguments this used to exit 0 having done nothing)" >&2
                  exit 1; }
mn5_require f "$EVAL_SCRIPT" d "$OVERFIT_ROOT/$OVERFIT_SCENE/dense"
mn5_require_slurm

mkdir -p "$OUT" logs
echo "$OVERFIT_SCENE" > "$SCENE_LIST"

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
