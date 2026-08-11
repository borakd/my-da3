#!/bin/bash
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=57   # was 48; MN5 derives mem from cores (8G/core) -> 456G ~= old --mem=450G
#SBATCH --time=72:00:00
#SBATCH --job-name=cut3r_eval
#SBATCH --output=/gpfs/projects/etur59/koc821022/outputs/cut3r_eval/logs/slurm_%j.out
#SBATCH --error=/gpfs/projects/etur59/koc821022/outputs/cut3r_eval/logs/slurm_%j.err

# NOTE: source the conda hook directly so the job never depends on an interactive
# fatal under set -e). No 'set -u' (MKL activation reads unset vars).
set -o pipefail
# --- self-locating worktree -------------------------------------------------
# Run the checkout this job was SUBMITTED from, never a hardcoded path, and fail
# loudly if that is not a my-da3 tree. (Do not use ${BASH_SOURCE[0]} here: SLURM
# copies the batch script to its spool dir, so it would not point at the repo.)
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything

# --- MN5 storage anchors (shared; every value overridable) -------------------
source "$WT/eval_pipeline/mn5_paths.sh"
MYDA3=$WT
CKPT_DIR=${CKPT_DIR:-$CKPT_ROOT/cut3r_multinode/multinode_finetune_aug_mini}
REG_CKPT=$CUT3R_DIR/src/cut3r_512_dpt_4_64.pth

mn5_require f "$CKPT_DIR/checkpoint-best.pth" f "$CKPT_DIR/checkpoint-final.pth" \
mn5_require_slurm
            f "$REG_CKPT" f "$EVAL_SCRIPT" d "$SCENES_ROOT" s "$SCENE_LIST"

export PYTHONPATH="$MYDA3/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
# Reduce CUDA fragmentation OOMs on very long sequences.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs"

declare -A CKPTS
CKPTS[best]=$CKPT_DIR/checkpoint-best.pth
CKPTS[final]=$CKPT_DIR/checkpoint-final.pth
CKPTS[regular]=$REG_CKPT

NUM_GPUS=$(mn5_require_gpus) || exit 1
echo "Node: $(hostname)  GPUs visible: $NUM_GPUS  Scenes: $(wc -l < "$SCENE_LIST")"
echo "Start: $(date)"

# One worker process per GPU; each runs all 3 checkpoints sequentially over its
# round-robin scene shard (model loaded once per checkpoint -> 3 loads/GPU).
for g in $(seq 0 $((NUM_GPUS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=$g
    for label in best final regular; do
      echo "[g$g] starting $label at $(date)"
      python "$MYDA3/eval_pipeline/infer_and_eval_worker.py" \
        --ckpt "${CKPTS[$label]}" --label "$label" --size 320 \
        --scenes_root "$SCENES_ROOT" --scene_list "$SCENE_LIST" \
        --pred_base "$OUT/$label/preds" --eval_base "$OUT/$label/eval" \
        --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
        --shard_id "$g" --num_shards "$NUM_GPUS" --device cuda \
        > "$OUT/logs/worker_g${g}_${label}.log" 2>&1
      echo "[g$g] finished $label rc=$? at $(date)"
    done
  ) &
done
wait
echo "All workers done: $(date)"

# Aggregate everything into the comparison table.
python "$MYDA3/eval_pipeline/aggregate_results.py" \
  --out_root "$OUT" --scene_list "$SCENE_LIST" \
  > "$OUT/logs/aggregate.log" 2>&1
echo "Aggregation done: $(date)"
cat "$OUT/summary/averages_table.txt" || true
