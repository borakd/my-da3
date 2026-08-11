#!/bin/bash
#SBATCH --job-name=gru_win64
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=57   # was 48; MN5 derives mem from cores (8G/core) -> 456G ~= old --mem=450G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gru_win64_%A.out
#SBATCH --error=logs/gru_win64_%A.err
# A3-vs-A4 reliability test: evaluate each checkpoint over 77 sliding 64-frame
# windows (stride 8) of the overfit episode, so ATE/RPE come as a DISTRIBUTION
# instead of the single 674-frame rollout (n=1) the grid table reports.
# 64 = the training num_views, long enough to exercise the recurrence.
#   sbatch run_gru_window64_eval.sbatch
# Results: outputs/.../gru_windows64/<label>/eval/win_XXXX/.../eval_depth_pose_metrics.csv
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
CKPTS=${CKPTS:-$CKPT_ROOT/captain_cut3r_sim3rmse}
OUT=$OUT_ROOT/cut3r_eval/overfit_test_scene/gru_windows64

export PYTHONPATH="$CRAY/src:$CUT3R_DIR:$CUT3R_DIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

# The window tree is NOT built here. Nothing in the repo invokes
# build_windows_strided.py, so without this guard all 8 labels x 4 shards used
# to run against an empty scene list and the job still exited 0.
mn5_require d "$CKPTS" f "$EVAL_SCRIPT" d "$OUT/windows_scenes" s "$OUT/scene_list.txt"
mn5_require_slurm

# A3 (direct head) and A4 (residual head) across all three G arms = the
# comparison in question; the two v3 G1 arms ride along for free.
LABELS="a3_g0 a3_g1 a3_g2 a4_g0 a4_g1 a4_g2"
V3="a4_g1_r8 a4_g1_f1_r8"

run_label () {  # $1=label $2=ckpt
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g python "$CRAY/eval_pipeline/infer_and_eval_worker_ray.py" \
      --ckpt "$2" --label "$1" --size 320 \
      --conditioning prev_pred_gru \
      --scenes_root "$OUT/windows_scenes" --scene_list "$OUT/scene_list.txt" \
      --pred_base "$OUT/$1/preds" --eval_base "$OUT/$1/eval" \
      --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
      --shard_id "$g" --num_shards 4 --device cuda \
      > "logs/win64_${SLURM_JOB_ID}_${1}_g${g}.log" 2>&1 &
  done
  wait
  echo "=== $1 done $(date) ==="
}

for l in $LABELS; do run_label "$l" "$CKPTS/captain_gru_v2_${l}/checkpoint-final.pth"; done
for l in $V3;    do run_label "$l" "$CKPTS/captain_gru_v3_${l}/checkpoint-final.pth"; done
echo "ALL DONE $(date)"
