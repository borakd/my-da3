#!/bin/bash
# INTEGRATION SMOKE, job 2 (WRITE_GATE_VARIANTS.md): the eval worker on ONE scene
# (RAIL+80edfcb1+2023-07-14-14h-28m-45s, 128 frames) for the write-gate arms, run SERIALLY on one GPU
# with the maks_sweep430.sbatch conventions (WT, CKPT default, --skip_mode block, --no_plots,
# --ignore_skip_sentinel, control.json copied next to the arm), PILOT=wgate_smoke.
# Arms (label:json-suffix): plain / plain2 (same-job determinism control), mg_const1 and fg_wmin1
# (must equal plain), fg_head / mg_head (smoke heads, wmin .5), mg_const0, mg_freeze8.
# Predictions are KEPT (KEEP_PREDS=1) so camera poses can be diffed against $OUT/augfull_lr1e5/preds.
#   sbatch -A etur59 -p acc -q acc_debug --gres=gpu:1 -c 20 --time=00:40:00 eval_pipeline/wgate_smoke_worker.sbatch
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:40:00
#SBATCH --job-name=wg_smoke2
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_wg_smoke2_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_wg_smoke2_%j.err
set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || { echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"
CKPT=${CKPT:-$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth}
PILOT=${PILOT:-wgate_smoke}
ROOT=$OUT/$PILOT
EP=$WT/eval_pipeline
SCENE=${SCENE:-RAIL+80edfcb1+2023-07-14-14h-28m-45s}
ARMS=${ARMS:-"plain:wgsmoke_plain plain2:wgsmoke_plain mg_const1:wgsmoke_mg_const1 fg_wmin1:wgsmoke_fg_wmin1 fg_head:wgsmoke_fg_head mg_head:wgsmoke_mg_head mg_const0:wgsmoke_mg_const0 mg_freeze8:wgsmoke_mg_freeze8"}
mkdir -p "$ROOT" "$OUT/logs"
LIST=$ROOT/scene_list.txt; echo "$SCENE" > "$LIST"
mn5_require f "$CKPT" f "$EVAL_SCRIPT" d "$SCENES_ROOT" s "$LIST" d "$SCENES_ROOT/$SCENE"
NUM_GPUS=$(mn5_require_gpus) || exit 1
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE POSE_GRU_HIDDEN_ZERO POSE_GRU_HIDDEN_SHUFFLE \
      POSE_GRU_IMG_FEAT_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_FORCE_ITERS \
      SKIP_FRAMES_FILE STATE_GATE_MODE REVERSE REVISIT
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
KEEP_FLAG=$([ "${KEEP_PREDS:-1}" = "1" ] && echo "" || echo "--no_keep_preds")
echo "Node: $(hostname)  GPUs=$NUM_GPUS  scene=$SCENE  ckpt=$CKPT  arms=$ARMS  Start: $(date)"
FAIL=0
for pair in $ARMS; do
  ARM=${pair%%:*}; CJ=$EP/maks_arm_${pair#*:}.json
  [ -f "$CJ" ] || { echo "[$ARM] MISSING control json $CJ"; FAIL=1; continue; }
  mkdir -p "$ROOT/$ARM"; cp "$CJ" "$ROOT/$ARM/control.json"
  rm -rf "$ROOT/$ARM/eval/$SCENE" "$ROOT/$ARM/preds/$SCENE"
  T0=$(date +%s)
  python "$EP/infer_and_eval_worker.py" \
    --ckpt "$CKPT" --label "$ARM" --size 320 \
    --scenes_root "$SCENES_ROOT" --scene_list "$LIST" \
    --pred_base "$ROOT/$ARM/preds" --eval_base "$ROOT/$ARM/eval" \
    --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
    --shard_id 0 --num_shards 1 --device cuda --ignore_skip_sentinel \
    --control_json "$CJ" --skip_mode block --no_plots $KEEP_FLAG \
    > "$ROOT/$ARM/worker_g0.log" 2>&1
  rc=$?
  CSV=$ROOT/$ARM/eval/$SCENE/eval_depth_pose_metrics.csv; TR=$ROOT/$ARM/eval/$SCENE/wgate_trace.json
  echo "[$ARM] rc=$rc in $(( $(date +%s) - T0 )) s; csv=$([ -f "$CSV" ] && echo yes || echo NO); trace=$([ -f "$TR" ] && echo yes || echo no); $(grep -c "Traceback" "$ROOT/$ARM/worker_g0.log") tracebacks; $(date)"
  [ "$rc" = 0 ] && [ -f "$CSV" ] || FAIL=1
done
echo "FAIL=$FAIL  Done: $(date)"
exit $FAIL
