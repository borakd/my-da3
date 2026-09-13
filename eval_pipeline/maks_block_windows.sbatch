#!/bin/bash
# Memory-write blocking controls for one scene: for EACH decline window found by
# find_decline_windows.py on the baseline run, re-run inference with that
# window's frames blocked from writing state/memory (views[i]['update']=False:
# the frames are still decoded and scored; the state that frames after the
# window see is the state as of the frame right before the window). One run per
# window, all other frames update normally. Then per run: metrics-over-frames
# plots (auto), decline-window analysis, overlay video (blocked span marked),
# and finally a baseline-vs-control comparison.
#
#   sbatch eval_pipeline/maks_block_windows.sbatch
# Env: SCENE, CKPT, BASE_LABEL (baseline run label whose eval/<SCENE> holds
#      decline_windows.json), PILOT (output label prefix), WINDOWS ("D1 D2 D3";
#      "ALL" = the union of every decline window in one run)
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:40:00
#SBATCH --job-name=maks_blockw
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_maks_blockw_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_maks_blockw_%j.err

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
PILOT=${PILOT:-maks_block_windows}
WINDOWS=${WINDOWS:-"D1 D2 D3"}
ROOT=$OUT/$PILOT
BASE=$OUT/$BASE_LABEL
GT=$SCENES_ROOT/$SCENE/dense
LIST=$ROOT/scene_list_$SCENE.txt
EP=$WT/eval_pipeline

mn5_require f "$CKPT" f "$EVAL_SCRIPT" d "$GT/rgb" s "$BASE/eval/$SCENE/eval_depth_pose_metrics.csv" \
            f "$BASE/eval/$SCENE/decline_windows.json"
mn5_require_gpus >/dev/null || exit 1
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE POSE_GRU_HIDDEN_ZERO POSE_GRU_HIDDEN_SHUFFLE \
      POSE_GRU_IMG_FEAT_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_FORCE_ITERS \
      SKIP_FRAMES_FILE STATE_GATE_MODE REVERSE REVISIT
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$ROOT/masks" "$OUT/logs"
echo "$SCENE" > "$LIST"
echo "Node: $(hostname)  scene=$SCENE  ckpt=$CKPT  baseline=$BASE_LABEL  windows=[$WINDOWS]  Start: $(date)"

# masks from the PRECISE window boundaries in the baseline's decline_windows.json
python - "$BASE/eval/$SCENE/decline_windows.json" "$ROOT/masks" "$SCENE" <<'PY'
import json, os, sys
d = json.load(open(sys.argv[1])); out = sys.argv[2]; scene = sys.argv[3]
union = set()
for k, r in enumerate(d["declines"], 1):
    a, b = r["window"]
    json.dump({scene: list(range(a, b + 1))}, open(f"{out}/skip_D{k}.json", "w"))
    union.update(range(a, b + 1))
    print(f"D{k}: block frames {a}-{b} ({b - a + 1} frames), Z={r['z']}")
json.dump({scene: sorted(union)}, open(f"{out}/skip_ALL.json", "w"))
print(f"ALL: block the union of every decline window ({len(union)} frames)")
PY

CTRL_ARGS=()
for W in $WINDOWS; do
  SK=$ROOT/masks/skip_$W.json
  [ -f "$SK" ] || { echo "ERROR: no mask $SK (fewer windows than requested?)" >&2; exit 1; }
  LABEL=${PILOT}_$W
  echo "=== $W: inference with memory write blocked on $(python -c "import json;v=list(json.load(open('$SK')).values())[0];print(f'{v[0]}-{v[-1]}')") ($(date))"
  rm -rf "$ROOT/$LABEL"
  python "$EP/infer_and_eval_worker.py" \
    --ckpt "$CKPT" --label "$LABEL" --size 320 \
    --scenes_root "$SCENES_ROOT" --scene_list "$LIST" \
    --pred_base "$ROOT/$LABEL/preds" --eval_base "$ROOT/$LABEL/eval" \
    --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
    --shard_id 0 --num_shards 1 --device cuda --ignore_skip_sentinel \
    --skip_frames_file "$SK" --skip_mode block \
    2>&1 | tee "$ROOT/worker_$W.log"
  E=$ROOT/$LABEL/eval/$SCENE
  [ -s "$E/eval_depth_pose_metrics.csv" ] || { echo "ERROR: $W produced no eval CSV" >&2; exit 1; }
  cat "$E/eval_metrics_readable.txt"
  echo "--- $W: decline windows"
  python "$EP/find_decline_windows.py" --csv "$E/eval_depth_pose_metrics.csv" --blocked "$SK" \
    --title "$SCENE (block $W)" 2>/dev/null | sed -n '5,12p'
  echo "--- $W: overlay video"
  python "$EP/render_metrics_video.py" --frames "$GT/rgb" --csv "$E/eval_depth_pose_metrics.csv" \
    --windows "$E/decline_windows.json" --blocked "$SK" --out "$E/metrics_overlay.mp4" --fps 6 \
    --title "$SCENE (augfull_lr1e5, memory write blocked: $W)" --snap 20 64 120
  CTRL_ARGS+=(--control "$W=$E/eval_depth_pose_metrics.csv:$ROOT/$LABEL/preds/$SCENE:$SK")
done

echo "=== comparison over every control present under $ROOT ($(date))"
CTRL_ARGS=()
for SK in "$ROOT"/masks/skip_*.json; do
  W=$(basename "$SK" .json); W=${W#skip_}
  E=$ROOT/${PILOT}_$W/eval/$SCENE
  [ -s "$E/eval_depth_pose_metrics.csv" ] && CTRL_ARGS+=(--control "$W=$E/eval_depth_pose_metrics.csv:$ROOT/${PILOT}_$W/preds/$SCENE:$SK")
done
python "$EP/maks_compare_runs.py" --baseline_csv "$BASE/eval/$SCENE/eval_depth_pose_metrics.csv" \
  --baseline_preds "$BASE/preds/$SCENE" "${CTRL_ARGS[@]}" --out_dir "$ROOT/compare" \
  --title "$SCENE (augfull_lr1e5)"
echo "Done: $(date)"
