#!/bin/bash
# Parallel "regular" (pretrained CUT3R) eval over ALL test scenes, on a separate
# node, concurrent with the main job's "final" pass. The main job's own regular
# pass is disabled via the SKIP_regular sentinel, so this job owns regular.
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=57   # was 24; MN5 derives mem from cores (8G/core) -> 456G ~= old --mem=450G
#SBATCH --time=72:00:00
#SBATCH --job-name=cut3r_reg
#SBATCH --output=/gpfs/projects/etur59/koc821022/outputs/cut3r_eval/logs/slurm_reg_%j.out
#SBATCH --error=/gpfs/projects/etur59/koc821022/outputs/cut3r_eval/logs/slurm_reg_%j.err

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
REG_CKPT=$CUT3R_DIR/src/cut3r_512_dpt_4_64.pth

mn5_require f "$REG_CKPT" f "$EVAL_SCRIPT" d "$SCENES_ROOT" s "$SCENE_LIST"
mn5_require_slurm

export PYTHONPATH="$MYDA3/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs"

NUM_GPUS=$(mn5_require_gpus) || exit 1
echo "Node: $(hostname)  GPUs: $NUM_GPUS  regular-only over $(wc -l < "$SCENE_LIST") scenes"
echo "Start: $(date)"

# One worker per GPU; round-robin scene shard. --ignore_skip_sentinel so this
# job runs regular even though SKIP_regular is set (that sentinel only disables
# the OTHER job's regular pass). Logs to worker_NEWg*_regular.log to avoid
# clobbering the main job's (no-op) worker_g*_regular.log.
for g in $(seq 0 $((NUM_GPUS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=$g
    echo "[NEWg$g] starting regular at $(date)"
    python "$MYDA3/eval_pipeline/infer_and_eval_worker.py" \
      --ckpt "$REG_CKPT" --label regular --size 320 \
      --scenes_root "$SCENES_ROOT" --scene_list "$SCENE_LIST" \
      --pred_base "$OUT/regular/preds" --eval_base "$OUT/regular/eval" \
      --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
      --shard_id "$g" --num_shards "$NUM_GPUS" --device cuda \
      --ignore_skip_sentinel \
      > "$OUT/logs/worker_NEWg${g}_regular.log" 2>&1
    echo "[NEWg$g] finished regular rc=$? at $(date)"
  ) &
done
wait
echo "Parallel regular done: $(date)"
