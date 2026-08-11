#!/bin/bash
#SBATCH --job-name=gru_overfit_all
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=50   # was 24; MN5 derives mem from cores (8G/core) -> 400G ~= old --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gru_overfit_all_%A.out
#SBATCH --error=logs/gru_overfit_all_%A.err
#
# Single-scene (overfit wrist_test episode) inference + eval for EVERY training run
# under checkpoints/captain_gru_overfit, using each run's LATEST checkpoint.
#
#   Step 1  verify_gru_ckpt_matrix.py — every run's PoseGRU levers are recoverable
#           from its checkpoint and actually restored by demo_ray.py's load path,
#           every pose_gru tensor is bit-identical, and the GRU is demonstrably
#           live (stripping it moves the predicted poses). Aborts the sweep on
#           failure so a mis-restored arm can never produce a table row.
#   Step 2  infer_and_eval_worker_ray.py --conditioning prev_pred_gru (the demo_ray
#           closed-loop arm with the checkpoint's PoseGRU refiner active — the ONLY
#           conditioning any of these runs takes, since every lever rides the ckpt),
#           then eval_depth_poses.py with ALL DEFAULT arguments against the same
#           overfit scene's dense/ GT. 4 GPUs, run dirs round-robined across them.
#
#   sbatch eval_gru_overfit_all.sbatch            # all runs with a checkpoint
#   VERIFY_ONLY=1 sbatch eval_gru_overfit_all.sbatch
#   ONLY="v3_a4_g1_r8 v3_a4_g2_r8" sbatch eval_gru_overfit_all.sbatch
#
# Outputs: $OUT/<label>/{preds,eval,epochs.txt,ckpt.txt}
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
CKPTS=${CKPTS:-$CKPT_ROOT/captain_gru_overfit}
SCENES_ROOT=$OVERFIT_ROOT
SCENE=$OVERFIT_SCENE
OUT=$OUT_ROOT/cut3r_eval/overfit_test_scene/gru_grid_final
SCENE_LIST=$OUT/scene_list_wrist_test.txt

export PYTHONPATH="$CRAY/src:$CUT3R_DIR:$CUT3R_DIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CRAY" || exit 1
mn5_require d "$CKPTS" f "$EVAL_SCRIPT" d "$OVERFIT_ROOT/$OVERFIT_SCENE/dense"
mn5_require_slurm
mkdir -p "$OUT" logs
echo "$SCENE" > "$SCENE_LIST"

echo "########## STEP 1: checkpoint/lever verification $(date) ##########"
python "$CRAY/verify_gru_ckpt_matrix.py" --ckpt_root "$CKPTS"
vrc=$?
echo "########## verification rc=$vrc ##########"
if [ $vrc -ne 0 ]; then
  echo "ABORT: verification failed — not running the sweep."
  exit $vrc
fi
[ -n "$VERIFY_ONLY" ] && { echo "VERIFY_ONLY set — stopping after verification."; exit 0; }

# ---- build the work list: <label>=<latest ckpt> ----------------------------
# label = run dir name minus the shared 'captain_gru_' prefix (v2_a1_g0, v3_a4_g1_r8, ...).
# Latest = checkpoint-final.pth when the run finished, else the highest keep_freq
# snapshot; checkpoint-last/best are deliberately never used (rewritten mid-run).
WORK=()
for RUN_DIR in "$CKPTS"/*/; do
  RUN_DIR=${RUN_DIR%/}
  NAME=$(basename "$RUN_DIR")
  LABEL=${NAME#captain_gru_}
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $LABEL "* ]]; then continue; fi
  if [ -f "$RUN_DIR/checkpoint-final.pth" ]; then
    CKPT=$RUN_DIR/checkpoint-final.pth
    EPOCHS=50
  else
    LAST=$(ls "$RUN_DIR" 2>/dev/null | grep -oE '^checkpoint-[0-9]+\.pth$' \
           | sed 's/checkpoint-//;s/\.pth//' | sort -n | tail -1)
    if [ -z "$LAST" ]; then
      echo "SKIP $LABEL: no checkpoint yet (run still training)"
      continue
    fi
    CKPT=$RUN_DIR/checkpoint-$LAST.pth
    EPOCHS=$LAST
  fi
  mkdir -p "$OUT/$LABEL"
  echo "$EPOCHS" > "$OUT/$LABEL/epochs.txt"
  echo "$CKPT"   > "$OUT/$LABEL/ckpt.txt"
  WORK+=("$LABEL=$CKPT")
  echo "QUEUE $LABEL  epochs=$EPOCHS  $CKPT"
done
echo "########## STEP 2: ${#WORK[@]} runs over 4 GPUs $(date) ##########"

run_gpu () {  # $1 = gpu index; processes WORK[i] for i % 4 == gpu
  local g=$1 i spec LABEL CKPT
  for ((i = g; i < ${#WORK[@]}; i += 4)); do
    spec=${WORK[$i]}
    LABEL=${spec%%=*}
    CKPT=${spec#*=}
    echo "[gpu$g] START $LABEL $(date)"
    CUDA_VISIBLE_DEVICES=$g python "$CRAY/eval_pipeline/infer_and_eval_worker_ray.py" \
      --ckpt "$CKPT" --label "$LABEL" --size 320 \
      --conditioning prev_pred_gru \
      --scenes_root "$SCENES_ROOT" --scene_list "$SCENE_LIST" \
      --pred_base "$OUT/$LABEL/preds" --eval_base "$OUT/$LABEL/eval" \
      --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
      --shard_id 0 --num_shards 1 --device cuda \
      > "logs/gru_overfit_all_${SLURM_JOB_ID}_${LABEL}.log" 2>&1
    local rc=$?
    echo "[gpu$g] END   $LABEL rc=$rc $(date)"
  done
}

for g in 0 1 2 3; do run_gpu "$g" & done
wait

echo "########## STEP 3: results $(date) ##########"
rc_all=0
for spec in "${WORK[@]}"; do
  LABEL=${spec%%=*}
  CSV="$OUT/$LABEL/eval/$SCENE/eval_depth_pose_metrics.csv"
  if [ -s "$CSV" ]; then
    echo "OK   $LABEL"
  else
    echo "MISS $LABEL  (no eval CSV at $CSV)"
    rc_all=1
  fi
done
echo "ALL DONE $(date) rc=$rc_all"
exit $rc_all
