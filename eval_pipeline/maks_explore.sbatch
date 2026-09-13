#!/bin/bash
# Run the six memory-write explorations (maks_explore_build.py) on one scene:
# every run = inference with a control spec (block / soft-commit alpha / reset /
# per-token gate), table-standard eval, metrics-over-frames plots, decline
# analysis and overlay video with the affected frames marked, then a
# baseline-vs-controls comparison per exploration and one overall summary.
#
#   sbatch eval_pipeline/maks_explore.sbatch
# Env: SCENE, CKPT, BASE_LABEL (baseline label; its eval/<SCENE> is the scene dir
#      that receives the exp*/ subfolders), EXPS (subset of exp names), FORCE=1
#      to re-run runs whose eval CSV already exists.
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:30:00
#SBATCH --job-name=maks_explore
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_maks_explore_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_maks_explore_%j.err

set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"

SCENE=${SCENE:-RAIL+80edfcb1+2023-07-14-14h-28m-45s}
CKPT=${CKPT:-$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth}
BASE_LABEL=${BASE_LABEL:-maks_plots_augfull_lr1e5}
BASE=$OUT/$BASE_LABEL
SD=$BASE/eval/$SCENE                 # scene dir: receives exp*/ subfolders
GT=$SCENES_ROOT/$SCENE/dense
EP=$WT/eval_pipeline
LIST=$SD/scene_list_$SCENE.txt

mn5_require f "$CKPT" f "$EVAL_SCRIPT" d "$GT/rgb" s "$SD/eval_depth_pose_metrics.csv" f "$SD/decline_windows.json" \
            d "$BASE/preds/$SCENE"
mn5_require_gpus >/dev/null || exit 1
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE POSE_GRU_HIDDEN_ZERO POSE_GRU_HIDDEN_SHUFFLE \
      POSE_GRU_IMG_FEAT_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_FORCE_ITERS \
      SKIP_FRAMES_FILE STATE_GATE_MODE REVERSE REVISIT
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUT/logs"
echo "$SCENE" > "$LIST"
NF=$(ls "$GT/rgb" | wc -l)
echo "Node: $(hostname)  scene=$SCENE  frames=$NF  ckpt=$CKPT  scene_dir=$SD  Start: $(date)"

python "$EP/maks_explore_build.py" --decline_json "$SD/decline_windows.json" --scene "$SCENE" \
  --n_frames "$NF" --scene_dir "$SD" || exit 1

EXPS=${EXPS:-$(python -c "import json;print(' '.join(json.load(open('$SD/explore_manifest.json'))))")}
for EXP in $EXPS; do
  RUNS=$(python -c "import json;print(' '.join(json.load(open('$SD/explore_manifest.json'))['$EXP']))")
  CTRL_ARGS=()
  for RUN in $RUNS; do
    CJ=$SD/$EXP/masks/$RUN.control.json; AJ=$SD/$EXP/masks/$RUN.affected.json
    RD=$SD/$EXP/$RUN; E=$RD/eval/$SCENE
    echo "=== $EXP / $RUN ($(date))"
    if [ "${FORCE:-0}" = "1" ] || [ ! -s "$E/eval_depth_pose_metrics.csv" ]; then
      rm -rf "$RD"; mkdir -p "$RD"
      python "$EP/infer_and_eval_worker.py" \
        --ckpt "$CKPT" --label "$RUN" --size 320 \
        --scenes_root "$SCENES_ROOT" --scene_list "$LIST" \
        --pred_base "$RD/preds" --eval_base "$RD/eval" \
        --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
        --shard_id 0 --num_shards 1 --device cuda --ignore_skip_sentinel \
        --control_json "$CJ" --skip_mode block \
        2>&1 | grep -v "FutureWarning\|autocast" | tee "$RD/worker.log" | grep -E "control|token_gate|frames|FAIL|Error" | head -5
      [ -s "$E/eval_depth_pose_metrics.csv" ] || { echo "ERROR: $EXP/$RUN produced no eval CSV" >&2; exit 1; }
    fi
    grep -A2 "Camera 0" "$E/eval_metrics_readable.txt" | tail -n 2
    python "$EP/find_decline_windows.py" --csv "$E/eval_depth_pose_metrics.csv" --blocked "$AJ" \
      --title "$SCENE ($EXP: $RUN)" >/dev/null 2>&1
    python "$EP/render_metrics_video.py" --frames "$GT/rgb" --csv "$E/eval_depth_pose_metrics.csv" \
      --windows "$E/decline_windows.json" --blocked "$AJ" --out "$E/metrics_overlay.mp4" --fps 6 \
      --title "$SCENE ($EXP: $RUN)" --snap 20 64 120 >/dev/null
    CTRL_ARGS+=(--control "$RUN=$E/eval_depth_pose_metrics.csv:$RD/preds/$SCENE:$AJ")
  done
  echo "=== $EXP: comparison"
  python "$EP/maks_compare_runs.py" --baseline_csv "$SD/eval_depth_pose_metrics.csv" \
    --baseline_preds "$BASE/preds/$SCENE" "${CTRL_ARGS[@]}" --out_dir "$SD/$EXP/compare" \
    --title "$SCENE (augfull_lr1e5) $EXP" | grep -E "^\| "
done

# overall summary
{
  echo "# Memory-write explorations: $SCENE (augfull_lr1e5)"; echo
  echo "Baseline = plain run. Affected frames per run are listed in each exp's masks/*.affected.json."; echo
  for EXP in $EXPS; do echo "## $EXP"; echo; sed -n '/^| run/,/^$/p' "$SD/$EXP/compare/compare_summary.md" | grep -E "^\|"; echo; done
} > "$SD/explore_summary.md"
cat "$SD/explore_summary.md"
echo "Done: $(date)"
