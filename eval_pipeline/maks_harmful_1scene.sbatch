#!/bin/bash
# Harmful-frame skip pilot on ONE scene with the pretrained CUT3R checkpoint.
# See HARMFUL_FRAME_SKIP_PLAN.md. Runs, in order, on one GPU:
#   P1  plain pass                     -> $ROOT/p1/{preds,eval}
#   flag harmful frames from P1's rows -> $ROOT/masks/skip_<rule>.json
#   per rule:
#     V0  P1 preds rescored on kept K  -> $ROOT/p1/eval_kept_<rule>
#     V1  block (update=False)         -> $ROOT/v1_<rule>/{preds,eval,eval_kept}
#     V2  drop (frames removed)        -> $ROOT/v2_<rule>/{preds,eval}
#   report                             -> $ROOT/report_<scene>.md
#
#   sbatch eval_pipeline/maks_harmful_1scene.sbatch
# Env overrides: SCENE_LIST_OVERRIDE (default eval_pipeline/maks_scene_1.txt), PILOT
# (default maks_harmful_1scene), RULES, CKPT, plus the mn5_paths.sh anchors.
# FORCE=1 wipes an existing $ROOT first (the worker resumes on an existing CSV).
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20   # MN5 lua plugin: cpus >= gpus*20
#SBATCH --time=01:00:00
#SBATCH --job-name=maks_harm1
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_maks_harm1_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_maks_harm1_%j.err

set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything

source "$WT/eval_pipeline/mn5_paths.sh"
SCENE_LIST=${SCENE_LIST_OVERRIDE:-$WT/eval_pipeline/maks_scene_1.txt}
CKPT=${CKPT:-$CUT3R_DIR/src/cut3r_512_dpt_4_64.pth}
PILOT=${PILOT:-maks_harmful_1scene}
RULES=${RULES:-"ate15 rpe2x absrel15 any"}
ROOT=$OUT/$PILOT
WORKER=$WT/eval_pipeline/infer_and_eval_worker.py

mn5_require f "$CKPT" f "$EVAL_SCRIPT" d "$SCENES_ROOT" s "$SCENE_LIST"
mn5_require_gpus >/dev/null || exit 1
# Falsifier env vars must never leak into a scoring run (mirrors arm_eval_for_run.sh).
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE POSE_GRU_HIDDEN_ZERO POSE_GRU_HIDDEN_SHUFFLE \
      POSE_GRU_IMG_FEAT_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_FORCE_ITERS \
      SKIP_FRAMES_FILE STATE_GATE_MODE REVERSE REVISIT

export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

SCENE=$(head -1 "$SCENE_LIST")
[ "$(wc -l < "$SCENE_LIST")" -eq 1 ] || echo "WARNING: scene list has more than one scene; report covers only $SCENE"
GT=$SCENES_ROOT/$SCENE/dense
if [ "${FORCE:-0}" = "1" ] && [ -d "$ROOT" ]; then echo "FORCE=1: wiping $ROOT"; rm -rf "$ROOT"; fi
mkdir -p "$ROOT" "$OUT/logs"
echo "Node: $(hostname)  scene=$SCENE  frames=$(ls "$GT/rgb" | wc -l)  ckpt=$CKPT  root=$ROOT  Start: $(date)"

run_worker () {  # label [extra worker args...]
  local label=$1; shift
  python "$WORKER" --ckpt "$CKPT" --label "$label" --size 320 \
    --scenes_root "$SCENES_ROOT" --scene_list "$SCENE_LIST" \
    --pred_base "$ROOT/$label/preds" --eval_base "$ROOT/$label/eval" \
    --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
    --shard_id 0 --num_shards 1 --device cuda --ignore_skip_sentinel "$@" \
    2>&1 | tee "$ROOT/worker_$label.log"
  [ -s "$ROOT/$label/eval/$SCENE/eval_depth_pose_metrics.csv" ] || {
    echo "ERROR: $label produced no eval CSV" >&2; exit 1; }
}

echo "=== P1 plain ($(date))"
run_worker p1

echo "=== flag ($(date))"
python "$WT/eval_pipeline/maks_flag_harmful_frames.py" \
  --csv "$ROOT/p1/eval/$SCENE/eval_depth_pose_metrics.csv" --scene "$SCENE" \
  --out_dir "$ROOT/masks" --rules $RULES || exit 1

for rule in $RULES; do
  SK=$ROOT/masks/skip_$rule.json
  echo "=== $rule: V0 rescore P1 on kept ($(date))"
  python "$WT/eval_pipeline/maks_rescore_subset.py" --pred_root "$ROOT/p1/preds/$SCENE" \
    --gt_root "$GT" --skip_json "$SK" --scene "$SCENE" \
    --out_dir "$ROOT/p1/eval_kept_$rule/$SCENE" --eval_script "$EVAL_SCRIPT" || exit 1

  echo "=== $rule: V1 block ($(date))"
  run_worker "v1_$rule" --skip_frames_file "$SK" --skip_mode block
  python "$WT/eval_pipeline/maks_rescore_subset.py" --pred_root "$ROOT/v1_$rule/preds/$SCENE" \
    --gt_root "$GT" --skip_json "$SK" --scene "$SCENE" \
    --out_dir "$ROOT/v1_$rule/eval_kept/$SCENE" --eval_script "$EVAL_SCRIPT" || exit 1

  echo "=== $rule: V2 drop ($(date))"
  run_worker "v2_$rule" --skip_frames_file "$SK" --skip_mode drop
done

echo "=== report ($(date))"
python "$WT/eval_pipeline/maks_report_1scene.py" --root "$ROOT" --scene "$SCENE" --rules $RULES
echo "Done: $(date)"
