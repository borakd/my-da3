#!/bin/bash
# Honest (no-GT) memory-write controls on a seeded scene subset of the 4292-scene
# test list, one arm per GPU in parallel, then a paired comparison against the
# baseline's existing per-scene results for exactly the same scenes.
#
#   sbatch eval_pipeline/maks_subset.sbatch
# Env: SCENE_LIST_OVERRIDE (default eval_pipeline/maks_subset40_seed1.txt),
#      CKPT (default augfull_lr1e5 final), PILOT (default maks_subset40),
#      ARMS (space list of arm names; each needs $WT/eval_pipeline/maks_arm_<name>.json)
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=03:00:00
#SBATCH --job-name=maks_subset
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_maks_subset_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_maks_subset_%j.err

set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"

LIST=${SCENE_LIST_OVERRIDE:-$WT/eval_pipeline/maks_subset40_seed1.txt}
CKPT=${CKPT:-$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth}
PILOT=${PILOT:-maks_subset40}
ARMS=${ARMS:-"tok_all_q50_g50 tok_trig_q50_g0 tok_trig_q50_g50 alpha_trig_050"}
ROOT=$OUT/$PILOT
EP=$WT/eval_pipeline

mn5_require f "$CKPT" f "$EVAL_SCRIPT" d "$SCENES_ROOT" s "$LIST"
NUM_GPUS=$(mn5_require_gpus) || exit 1
# every listed scene must be in the 4292-scene test list
python - "$LIST" "$OUT/scene_list.txt" <<'PY' || exit 1
import sys
sub=[l.strip() for l in open(sys.argv[1]) if l.strip()]; full=set(l.strip() for l in open(sys.argv[2]))
bad=[s for s in sub if s not in full]
print(f"subset: {len(sub)} scenes, {len(sub)-len(bad)} in the 4292 test list")
sys.exit(1 if bad else 0)
PY
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE POSE_GRU_HIDDEN_ZERO POSE_GRU_HIDDEN_SHUFFLE \
      POSE_GRU_IMG_FEAT_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_FORCE_ITERS \
      SKIP_FRAMES_FILE STATE_GATE_MODE REVERSE REVISIT
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT" "$OUT/logs"
cp "$LIST" "$ROOT/scene_list.txt"
echo "Node: $(hostname)  GPUs=$NUM_GPUS  arms=[$ARMS]  ckpt=$CKPT  root=$ROOT  Start: $(date)"

g=0
for ARM in $ARMS; do
  CJ=$EP/maks_arm_$ARM.json
  [ -f "$CJ" ] || { echo "ERROR: no control json $CJ" >&2; exit 1; }
  cp "$CJ" "$ROOT/control_$ARM.json"
  (
    export CUDA_VISIBLE_DEVICES=$(( g % NUM_GPUS ))
    echo "[$ARM] start on GPU $CUDA_VISIBLE_DEVICES at $(date)"
    python "$EP/infer_and_eval_worker.py" \
      --ckpt "$CKPT" --label "$ARM" --size 320 \
      --scenes_root "$SCENES_ROOT" --scene_list "$LIST" \
      --pred_base "$ROOT/$ARM/preds" --eval_base "$ROOT/$ARM/eval" \
      --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
      --shard_id 0 --num_shards 1 --device cuda --ignore_skip_sentinel \
      --control_json "$CJ" --skip_mode block \
      > "$ROOT/worker_$ARM.log" 2>&1
    echo "[$ARM] done rc=$? at $(date)  scenes=$(ls $ROOT/$ARM/eval 2>/dev/null | wc -l)"
  ) &
  g=$((g + 1))
  # more arms than GPUs: wait for a slot
  if [ $(( g % NUM_GPUS )) -eq 0 ]; then wait; fi
done
wait
echo "All arms done: $(date)"

ARM_ARGS=(); for ARM in $ARMS; do ARM_ARGS+=(--arm "$ARM=$ROOT/$ARM/eval"); done
python "$EP/maks_subset_compare.py" --scene_list "$LIST" \
  --baseline "augfull_lr1e5=$OUT/augfull_lr1e5/eval" "${ARM_ARGS[@]}" \
  --ref "augfull_cg_g7ema=$OUT/augfull_cg_g7ema/eval" --ref "augfull_cg_fuse_g7=$OUT/augfull_cg_fuse_g7/eval" \
  --out_dir "$ROOT/compare" --title "$PILOT ($(wc -l < "$LIST") scenes, augfull_lr1e5 ckpt)"
# triggered-frame statistics for the triggered arms
python - "$ROOT" "$LIST" $ARMS <<'PY'
import json, os, sys
root, lst, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
scenes = [l.strip() for l in open(lst) if l.strip()]
for a in arms:
    fr, tot = 0, 0
    for s in scenes:
        p = f"{root}/{a}/eval/{s}/triggered_frames.json"
        if os.path.isfile(p):
            fr += len(json.load(open(p))[s]); tot += len(os.listdir(f"{root}/{a}/preds/{s}/depth"))
    if tot:
        print(f"[{a}] triggered frames: {fr}/{tot} = {100*fr/tot:.1f}%")
PY
echo "Done: $(date)"
