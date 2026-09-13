#!/bin/bash
# One ARM of the honest per-token-gate sweep on the 430-scene subset: 4 workers
# (one per GPU, round-robin scene shards) with the arm's control json; eval-only
# outputs (predictions deleted after scoring, no per-scene plots).
#
#   ARM=<name> sbatch eval_pipeline/maks_sweep430.sbatch
# Env: ARM (required; needs eval_pipeline/maks_arm_$ARM.json), SCENE_LIST_OVERRIDE
#      (default eval_pipeline/maks_subset430.txt), CKPT, PILOT (default maks_sweep430)
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=04:00:00
#SBATCH --job-name=maks_sw430
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_maks_sw430_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_maks_sw430_%j.err

set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"

ARM=${ARM:?ARM is required}
LIST=${SCENE_LIST_OVERRIDE:-$WT/eval_pipeline/maks_subset430.txt}
CKPT=${CKPT:-$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth}
PILOT=${PILOT:-maks_sweep430}
ROOT=$OUT/$PILOT
EP=$WT/eval_pipeline
CJ=$EP/maks_arm_$ARM.json

mn5_require f "$CKPT" f "$EVAL_SCRIPT" d "$SCENES_ROOT" s "$LIST" f "$CJ"
NUM_GPUS=$(mn5_require_gpus) || exit 1
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE POSE_GRU_HIDDEN_ZERO POSE_GRU_HIDDEN_SHUFFLE \
      POSE_GRU_IMG_FEAT_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_FORCE_ITERS \
      SKIP_FRAMES_FILE STATE_GATE_MODE REVERSE REVISIT
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KEEP_FLAG=$([ "${KEEP_PREDS:-0}" = "1" ] && echo "" || echo "--no_keep_preds")
mkdir -p "$ROOT/$ARM" "$OUT/logs"
cp "$CJ" "$ROOT/$ARM/control.json"; cp "$LIST" "$ROOT/scene_list.txt"
echo "Node: $(hostname)  GPUs=$NUM_GPUS  arm=$ARM  scenes=$(wc -l < "$LIST")  ckpt=$CKPT  Start: $(date)"

for g in $(seq 0 $((NUM_GPUS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=$g
    python "$EP/infer_and_eval_worker.py" \
      --ckpt "$CKPT" --label "$ARM" --size 320 \
      --scenes_root "$SCENES_ROOT" --scene_list "$LIST" \
      --pred_base "$ROOT/$ARM/preds" --eval_base "$ROOT/$ARM/eval" \
      --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
      --shard_id "$g" --num_shards "$NUM_GPUS" --device cuda --ignore_skip_sentinel \
      --control_json "$CJ" --skip_mode block --no_plots $KEEP_FLAG \
      > "$ROOT/$ARM/worker_g$g.log" 2>&1
    echo "[$ARM g$g] done rc=$? at $(date)"
  ) &
done
wait
echo "[$ARM] scenes scored: $(find "$ROOT/$ARM/eval" -mindepth 2 -name eval_depth_pose_metrics.csv | wc -l) / $(wc -l < "$LIST")   Done: $(date)"
