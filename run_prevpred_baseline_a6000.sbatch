#!/bin/bash
#SBATCH --job-name=prevpred_base
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=13   # was 8; MN5 derives mem from cores (8G/core) -> 104G ~= old --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prevpred_base_%A.out
#SBATCH --error=logs/prevpred_base_%A.err
# GRU-less control for the single-hardware companion table: the CURRENT prev_pred
# overfit run (captain_ray_prev_pred, 50 ep, same wrist_test scene, no pose_gru),
# evaluated on the SAME A6000 hardware and code path as the 16 GRU arms.
# The table's published baseline row instead comes from preds dated 10 Jul, which
# predate every surviving prev_pred overfit checkpoint.
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
OUT=$OUT_ROOT/cut3r_eval/overfit_test_scene/gru_grid_final
SCENE_LIST=$OUT/scene_list_wrist_test.txt
# Was hardcoded to captain_cut3r_sim3rmse/captain_ray_prev_pred, which is not on MN5.
CKPT=${CKPT:-}
export PYTHONPATH="$CRAY/src:$CUT3R_DIR:$CUT3R_DIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CRAY" || exit 1

[ -n "$CKPT" ] || { echo "ERROR: CKPT is required (a prev_pred-trained checkpoint)." >&2; exit 1; }
mn5_require f "$CKPT" f "$EVAL_SCRIPT" d "$OVERFIT_ROOT/$OVERFIT_SCENE/dense"
mn5_require_slurm

# eval_gru_overfit_all.sbatch writes the scene list; this job only reads it.
mkdir -p "$OUT/prev_pred_base"
[ -s "$SCENE_LIST" ] || echo "$OVERFIT_SCENE" > "$SCENE_LIST"

python "$CRAY/eval_pipeline/infer_and_eval_worker_ray.py" \
  --ckpt "$CKPT" \
  --label prev_pred_base --size 320 --conditioning prev_pred \
  --scenes_root "$OVERFIT_ROOT" \
  --scene_list "$SCENE_LIST" \
  --pred_base "$OUT/prev_pred_base/preds" --eval_base "$OUT/prev_pred_base/eval" \
  --eval_script "$EVAL_SCRIPT" \
  --cut3r_dir "$CUT3R_DIR" --shard_id 0 --num_shards 1 --device cuda
rc=$?
echo "50" > "$OUT/prev_pred_base/epochs.txt"
echo "DONE rc=$rc $(date)"
exit $rc
