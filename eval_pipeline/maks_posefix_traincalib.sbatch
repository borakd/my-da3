#!/bin/bash
# Train-split calibration of the causal pose recalibration bias.
#
# Closes the only leakage objection to the causal_posefix rows: the published
# bias was fitted on 100 scenes drawn from the TEST split. Here the same
# quantity is fitted on 100 scenes drawn from the TRAIN split (disjoint by
# construction: zero scene-name overlap with scene_list.txt), and the resulting
# constant is applied to all 4292 test scenes. No test label is touched by the fit.
#
# Stage 1  CUT3R inference (augfull_lr1e5) on the 100 train calibration scenes,
#          4-way GPU shard. Needed because calibration reads the model's own
#          predicted increments, which do not exist for train episodes.
# Stage 2  the sweep, with --calib_root/--calib_pred_base pointed at that tree,
#          scoring all 4292 test scenes with the official evaluator.
#
#   sbatch eval_pipeline/maks_posefix_traincalib.sbatch
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --job-name=posefix_tc
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_posefixtc_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_posefixtc_%j.err

set -o pipefail
WT=${WT:-/gpfs/home/koc/koc821022/maks_idea}
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"

TRAIN_ROOT=/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/train/dl3dv_multi/wrist
CAL=$OUT/traincalib100
CKPT=$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth
EP=$WT/eval_pipeline

mn5_require f "$CKPT" f "$EVAL_SCRIPT" s "$CAL/scenes_100.txt" d "$TRAIN_ROOT"
mn5_require_gpus >/dev/null || exit 1
unset SKIP_FRAMES_FILE STATE_GATE_MODE REVERSE REVISIT
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Node $(hostname)  start $(date)"

# ---- stage 1: inference on the train calibration scenes -------------------
NS=4
for i in $(seq 0 $((NS-1))); do
  CUDA_VISIBLE_DEVICES=$i python "$EP/infer_and_eval_worker.py" \
    --ckpt "$CKPT" --label traincalib100 --size 320 \
    --scenes_root "$TRAIN_ROOT" --scene_list "$CAL/scenes_100.txt" \
    --pred_base "$CAL/preds" --eval_base "$CAL/eval" \
    --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
    --shard_id "$i" --num_shards "$NS" --device cuda \
    --ignore_skip_sentinel --no_plots \
    > "$OUT/logs/traincalib100_shard$i.log" 2>&1 &
done
wait
NP=$(ls "$CAL/preds" 2>/dev/null | wc -l)
echo "stage 1 done $(date): $NP/100 calibration scenes have predictions"
[ "$NP" -ge 90 ] || { echo "ERROR: only $NP calibration scenes predicted" >&2; exit 1; }

# ---- stage 2: fit the bias there, score all 4292 test scenes --------------
for ARM in "causal_posefix_tc" "causal_posefix_tc_ema --ema"; do
  set -- $ARM
  LBL=$1; shift
  python "$EP/maks_posefix_sweep.py" --label "$LBL" "$@" \
    --calib_scenes "$CAL/scenes_100.txt" \
    --calib_root "$TRAIN_ROOT" --calib_pred_base "$CAL/preds" \
    --procs 72 || exit 1
done
echo "done $(date)"
