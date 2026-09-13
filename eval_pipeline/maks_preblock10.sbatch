#!/bin/bash
# Fixed-length pre-window blocking for one scene: for EACH decline window, block
# the 10 frames IMMEDIATELY BEFORE the window from writing state/memory, leaving
# the window itself (and everything else) writing normally. One run per window.
#
# This is the length-standardised control for exp1_preblock, which used a
# pre-span equal to each window's own length (12/20/20 here). Same operation,
# same masks format, same UI; only the span length changes, so the two are
# directly comparable.
#
# Per run: metrics-over-frames plots, decline-window analysis with the blocked
# frames greyed, and the overlay video with the blocked span marked.
#
#   sbatch eval_pipeline/maks_preblock10.sbatch
# Env: SCENE, CKPT, BASE_LABEL, PRELEN (default 10), RUNS (default "D1 D2 D3")
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --job-name=preblock10
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_preblock10_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_preblock10_%j.err

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
PRELEN=${PRELEN:-10}
RUNS=${RUNS:-"D1 D2 D3"}
BASE=$OUT/$BASE_LABEL
SD=$BASE/eval/$SCENE                      # scene dir, holds exp1..exp6
EXP=$SD/exp7_preblock${PRELEN}
GT=$SCENES_ROOT/$SCENE/dense
EP=$WT/eval_pipeline
LIST=$EXP/scene_list.txt

mn5_require f "$CKPT" f "$EVAL_SCRIPT" d "$GT/rgb" f "$SD/decline_windows.json" \
            s "$SD/eval_depth_pose_metrics.csv" d "$BASE/preds/$SCENE"
mn5_require_gpus >/dev/null || exit 1
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE POSE_GRU_HIDDEN_ZERO POSE_GRU_HIDDEN_SHUFFLE \
      POSE_GRU_IMG_FEAT_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_FORCE_ITERS \
      SKIP_FRAMES_FILE STATE_GATE_MODE REVERSE REVISIT
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$EXP/masks" "$OUT/logs"
echo "$SCENE" > "$LIST"
echo "Node: $(hostname)  scene=$SCENE  prelen=$PRELEN  runs=[$RUNS]  Start: $(date)"

python - "$SD/decline_windows.json" "$EXP/masks" "$SCENE" "$PRELEN" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); out, scene, L = sys.argv[2], sys.argv[3], int(sys.argv[4])
n = d["n_frames"]
for k, r in enumerate(d["declines"], 1):
    a, b = r["window"]
    fr = [t for t in range(max(0, a - L), a) if 0 <= t < n]   # the L frames before the window
    json.dump({scene: {"block": fr}}, open(f"{out}/pre{L}_D{k}.control.json", "w"))
    json.dump({scene: fr}, open(f"{out}/pre{L}_D{k}.affected.json", "w"))
    print(f"D{k}: window {a}-{b} (z={r['z']}) -> pre-block {fr[0]}-{fr[-1]} ({len(fr)} frames)")
PY

CTRL_ARGS=()
for W in $RUNS; do
  CJ=$EXP/masks/pre${PRELEN}_$W.control.json
  AJ=$EXP/masks/pre${PRELEN}_$W.affected.json
  [ -f "$CJ" ] || { echo "ERROR: no mask $CJ" >&2; exit 1; }
  RD=$EXP/pre${PRELEN}_$W; E=$RD/eval/$SCENE
  echo "=== $W ($(date))"
  rm -rf "$RD"; mkdir -p "$RD"
  python "$EP/infer_and_eval_worker.py" \
    --ckpt "$CKPT" --label "pre${PRELEN}_$W" --size 320 \
    --scenes_root "$SCENES_ROOT" --scene_list "$LIST" \
    --pred_base "$RD/preds" --eval_base "$RD/eval" \
    --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
    --shard_id 0 --num_shards 1 --device cuda --ignore_skip_sentinel \
    --control_json "$CJ" --skip_mode block \
    2>&1 | grep -v "FutureWarning\|autocast" | tee "$RD/worker.log" | grep -E "control|frames|FAIL|Error" | head -5
  [ -s "$E/eval_depth_pose_metrics.csv" ] || { echo "ERROR: $W produced no eval CSV" >&2; exit 1; }
  grep -A2 "Camera 0" "$E/eval_metrics_readable.txt" | tail -n 2
  python "$EP/find_decline_windows.py" --csv "$E/eval_depth_pose_metrics.csv" --blocked "$AJ" \
    --title "$SCENE (pre-block ${PRELEN}f before $W)" >/dev/null 2>&1
  python "$EP/render_metrics_video.py" --frames "$GT/rgb" --csv "$E/eval_depth_pose_metrics.csv" \
    --windows "$E/decline_windows.json" --blocked "$AJ" --out "$E/metrics_overlay.mp4" --fps 6 \
    --title "$SCENE (augfull_lr1e5, ${PRELEN} frames before $W blocked from memory)" \
    --snap 20 64 120 >/dev/null
  CTRL_ARGS+=(--control "pre${PRELEN}_$W=$E/eval_depth_pose_metrics.csv:$RD/preds/$SCENE:$AJ")
done

echo "=== comparison ($(date))"
python "$EP/maks_compare_runs.py" --baseline_csv "$SD/eval_depth_pose_metrics.csv" \
  --baseline_preds "$BASE/preds/$SCENE" "${CTRL_ARGS[@]}" --out_dir "$EXP/compare" \
  --title "$SCENE (augfull_lr1e5) pre-block ${PRELEN} frames" | grep -E "^\| "
echo "Done: $(date)"
ls -la "$EXP"/pre${PRELEN}_*/eval/"$SCENE"/metrics_overlay.mp4
