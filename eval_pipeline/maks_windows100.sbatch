#!/bin/bash
# Window / pre-window blocking at scale: run every arm built by
# maks_windows100_build.py (w100_W<k> = k-th window blocked, w100_P<k> = its
# pre-window span blocked) sequentially on one 4-GPU node, 4 sharded workers per
# arm (model loaded once per worker), eval-only outputs, then aggregate.
#
#   sbatch eval_pipeline/maks_windows100.sbatch
# Env: PILOT (default maks_windows100; its dir holds scenes_<arm>.txt + manifest.json),
#      CKPT, ARMS (override the arm list)
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=06:00:00
#SBATCH --job-name=maks_w100
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_maks_w100_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_maks_w100_%j.err

set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"

PILOT=${PILOT:-maks_windows100}
ROOT=$OUT/$PILOT
CKPT=${CKPT:-$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth}
EP=$WT/eval_pipeline
ARMS=${ARMS:-$(ls "$ROOT"/scenes_w100_*.txt | sed 's|.*/scenes_||; s|\.txt$||' | tr '\n' ' ')}

mn5_require f "$CKPT" f "$EVAL_SCRIPT" d "$SCENES_ROOT" f "$ROOT/manifest.json"
NUM_GPUS=$(mn5_require_gpus) || exit 1
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE POSE_GRU_HIDDEN_ZERO POSE_GRU_HIDDEN_SHUFFLE \
      POSE_GRU_IMG_FEAT_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_FORCE_ITERS \
      SKIP_FRAMES_FILE STATE_GATE_MODE REVERSE REVISIT
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs"
echo "Node: $(hostname)  GPUs=$NUM_GPUS  arms=[$ARMS]  ckpt=$CKPT  root=$ROOT  Start: $(date)"

for ARM in $ARMS; do
  CJ=$EP/maks_arm_$ARM.json; LIST=$ROOT/scenes_$ARM.txt
  [ -f "$CJ" ] && [ -s "$LIST" ] || { echo "ERROR: missing $CJ or $LIST" >&2; exit 1; }
  mkdir -p "$ROOT/$ARM"; cp "$CJ" "$ROOT/$ARM/control.json"
  echo "=== $ARM: $(wc -l < "$LIST") scenes ($(date))"
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    (
      export CUDA_VISIBLE_DEVICES=$g
      python "$EP/infer_and_eval_worker.py" \
        --ckpt "$CKPT" --label "$ARM" --size 320 \
        --scenes_root "$SCENES_ROOT" --scene_list "$LIST" \
        --pred_base "$ROOT/$ARM/preds" --eval_base "$ROOT/$ARM/eval" \
        --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
        --shard_id "$g" --num_shards "$NUM_GPUS" --device cuda --ignore_skip_sentinel \
        --control_json "$CJ" --skip_mode block --no_plots --no_keep_preds \
        > "$ROOT/$ARM/worker_g$g.log" 2>&1
    ) &
  done
  wait
  echo "[$ARM] scored $(find "$ROOT/$ARM/eval" -mindepth 2 -name eval_depth_pose_metrics.csv | wc -l) / $(wc -l < "$LIST") at $(date)"
done

echo "=== aggregate ($(date))"
export PATH=/apps/GPP/LATEX/20240430/bin/x86_64-linux:$PATH
python "$EP/maks_windows100_aggregate.py" --root "$ROOT" --baseline_eval "$OUT/augfull_lr1e5/eval" --out_dir "$ROOT/tables"
echo "Done: $(date)"
