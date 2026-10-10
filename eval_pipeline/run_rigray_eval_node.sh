#!/bin/bash
# Per-node launcher for infer_and_eval_worker_rigray.py (CUT3R conditioned on the exterior rig's same-timestep pose).
# Env via --export: LABEL CKPT CONDITIONING POSE_SOURCE(rig|gt) GAP(hold|mask) RIG_DIR SCENE_LIST BASE_SHARD NUM_SHARDS LIMIT
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=06:00:00
#SBATCH --job-name=rigray_eval
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_rigray_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_rigray_%j.err
set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || { echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"
for v in LABEL CKPT CONDITIONING POSE_SOURCE RIG_DIR SCENE_LIST; do [ -n "${!v}" ] || { echo "ERROR: $v is required" >&2; exit 1; }; done
GAP=${GAP:-hold}; BASE_SHARD=${BASE_SHARD:-0}; NUM_SHARDS=${NUM_SHARDS:-16}; LIMIT=${LIMIT:-0}
GT_WIN_ROOT=${GT_WIN_ROOT:-$OUT/rig_windows_gt}
mn5_require f "$CKPT" f "$EVAL_SCRIPT" d "$SCENES_ROOT" s "$SCENE_LIST" d "$RIG_DIR"
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$WT/eval_pipeline:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs" "$GT_WIN_ROOT"
NUM_GPUS=$(mn5_require_gpus) || exit 1
echo "Node: $(hostname) GPUs=$NUM_GPUS label=$LABEL cond=$CONDITIONING pose=$POSE_SOURCE gap=$GAP shards=$BASE_SHARD+/$NUM_SHARDS ckpt=$CKPT"
for g in $(seq 0 $((NUM_GPUS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=$g; SHARD=$((BASE_SHARD + g))
    python "$WT/eval_pipeline/infer_and_eval_worker_rigray.py" \
      --ckpt "$CKPT" --label "$LABEL" --conditioning "$CONDITIONING" --pose_source "$POSE_SOURCE" --gap "$GAP" \
      --rig_dir "$RIG_DIR" --scenes_root "$SCENES_ROOT" --scene_list "$SCENE_LIST" \
      --pred_base "$OUT/$LABEL/preds" --eval_base "$OUT/$LABEL/eval" --gt_win_root "$GT_WIN_ROOT" \
      --eval_script "$EVAL_SCRIPT" --size 320 --shard_id "$SHARD" --num_shards "$NUM_SHARDS" --limit "$LIMIT" \
      > "$OUT/logs/worker_rigray_$(hostname)_g${g}_${LABEL}.log" 2>&1
    echo "[g$g shard$SHARD] rc=$?"
  ) &
done
wait; echo "node done $(date)"
