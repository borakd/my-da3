#!/bin/bash
# One-scene inference + eval + metrics-over-frames plots with the image-only
# (regular CUT3R / demo.py-path) worker. Plots land next to the eval CSV:
#   $OUT/$LABEL/eval/$SCENE/metrics_over_time.png            (5 panels)
#   $OUT/$LABEL/eval/$SCENE/metrics_over_time_<metric>.png   (one per metric)
#
#   SCENE=<scene> CKPT=<ckpt> LABEL=<label> sbatch eval_pipeline/maks_plot_scene.sbatch
# Defaults: SCENE=RAIL+80edfcb1+2023-07-14-14h-28m-45s, CKPT = augfull_lr1e5
# checkpoint-final (the finetuned 50-epoch lr1e-5 run behind the table's
# "Regular CUT3R" row), LABEL=maks_plots_augfull_lr1e5. REF_LABEL (default
# augfull_lr1e5) names an existing full-run label whose per-frame CSV for the
# scene is compared against the fresh one as a parity check.
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:30:00
#SBATCH --job-name=maks_plot1
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_maks_plot1_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_maks_plot1_%j.err

set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"

SCENE=${SCENE:-RAIL+80edfcb1+2023-07-14-14h-28m-45s}
CKPT=${CKPT:-$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth}
LABEL=${LABEL:-maks_plots_augfull_lr1e5}
REF_LABEL=${REF_LABEL:-augfull_lr1e5}
ROOT=$OUT/$LABEL
LIST=$ROOT/scene_list_$SCENE.txt

mn5_require f "$CKPT" f "$EVAL_SCRIPT" d "$SCENES_ROOT/$SCENE/dense/rgb"
mn5_require_gpus >/dev/null || exit 1
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE POSE_GRU_HIDDEN_ZERO POSE_GRU_HIDDEN_SHUFFLE \
      POSE_GRU_IMG_FEAT_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_FORCE_ITERS \
      SKIP_FRAMES_FILE STATE_GATE_MODE REVERSE REVISIT
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$ROOT" "$OUT/logs"
echo "$SCENE" > "$LIST"
if [ "${FORCE:-1}" = "1" ]; then rm -rf "$ROOT/eval/$SCENE" "$ROOT/preds/$SCENE"; fi
echo "Node: $(hostname)  scene=$SCENE  frames=$(ls "$SCENES_ROOT/$SCENE/dense/rgb" | wc -l)  ckpt=$CKPT  label=$LABEL  Start: $(date)"

python "$WT/eval_pipeline/infer_and_eval_worker.py" \
  --ckpt "$CKPT" --label "$LABEL" --size 320 \
  --scenes_root "$SCENES_ROOT" --scene_list "$LIST" \
  --pred_base "$ROOT/preds" --eval_base "$ROOT/eval" \
  --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
  --shard_id 0 --num_shards 1 --device cuda --ignore_skip_sentinel \
  2>&1 | tee "$ROOT/worker_$SCENE.log"

CSV=$ROOT/eval/$SCENE/eval_depth_pose_metrics.csv
[ -s "$CSV" ] || { echo "ERROR: no eval CSV at $CSV" >&2; exit 1; }
echo "=== scene metrics"; cat "$ROOT/eval/$SCENE/eval_metrics_readable.txt"
echo "=== plots"; ls -1 "$ROOT/eval/$SCENE"/metrics_over_time*.png

REF=$OUT/$REF_LABEL/eval/$SCENE/eval_depth_pose_metrics.csv
if [ -s "$REF" ]; then
  echo "=== parity vs $REF_LABEL full-run CSV (max |diff| per metric over per-frame rows)"
  python - "$CSV" "$REF" <<'PY'
import csv, sys, math
def rows(p):
    d = {}
    for r in csv.DictReader(open(p)):
        if r["camera_id"] == "0" and r["local_timestep"].isdigit():
            d[int(r["local_timestep"])] = r
    return d
a, b = rows(sys.argv[1]), rows(sys.argv[2])
print(f"rows: fresh={len(a)} ref={len(b)}")
for m in ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]:
    diffs = [abs(float(a[t][m]) - float(b[t][m])) for t in a if t in b
             and math.isfinite(float(a[t][m])) and math.isfinite(float(b[t][m]))]
    print(f"  {m:10s} max|diff|={max(diffs) if diffs else float('nan'):.3e}  n={len(diffs)}")
PY
fi
echo "Done: $(date)"
