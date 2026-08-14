#!/bin/bash
# Per-node cooperative-queue eval launcher for the captain_ray (GT ray-map
# conditioned) checkpoint over all 4292 DROID test scenes.
#
# Submit once per node (default: full L40S node). Every worker on every node
# points at the SAME claim_dir + output dirs and atomically claims scenes via
# mkdir, so any subset of nodes that start will cooperatively drain one shared
# queue (no double-processing, no stall if a node never starts).
#
# Env (passed via --export):
#   LABEL        output label under $OUT      (default captain_ray_gt_last)
#   CKPT         checkpoint path              (REQUIRED -- see note below)
#   CONDITIONING demo_ray conditioning mode   (default gt)
#   BASE_SHARD   global shard offset for this node's GPUs (default 0)
#   NUM_SHARDS   total planned workers, only affects starting offset (default 12)
#   LIMIT        max scenes per worker, 0 = all (default 0; for smoke tests)
# Storage anchors (CKPT_ROOT/OUT_ROOT/DATA_ROOT/SCENES_ROOT/EVAL_SCRIPT) come
# from eval_pipeline/mn5_paths.sh and are all overridable the same way.
#
# CKPT has no default: the old one (cut3r_multinode/captain_ray_finetune_aug_full)
# was never copied to MN5, and a dead default that silently fails four workers
# deep is worse than an explicit error.
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80   # MN5 lua submit plugin: cpus >= nodes*gpus*20, so 4 GPUs => exactly 80
                             # (57 is HARD REJECTED at submit: "Required cpus: 80"; 160 => node config
                             # unavailable). Memory is derived at 8G/core -> 640G. Never add --mem.
#SBATCH --time=24:00:00
#SBATCH --job-name=cray_eval
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_cray_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_cray_%j.err

# Source the conda hook directly so the job never depends on an interactive shell rc.
# No 'set -u' (MKL activation scripts read unset vars).
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

# --- MN5 storage anchors ----------------------------------------------------
source "$WT/eval_pipeline/mn5_paths.sh"
CRAY=$WT
LABEL=${LABEL:-captain_ray_gt_last}
# No default: a hand submission that forgot CONDITIONING used to silently
# score under GT ray-map conditioning (the privileged arm) with the GRU
# stripped. Like CKPT below, an explicit error beats a dead default.
[ -n "$CONDITIONING" ] || { echo "ERROR: CONDITIONING is required (preflight derives it for real arms)" >&2; exit 1; }
CLAIM_DIR=$OUT/$LABEL/claims

BASE_SHARD=${BASE_SHARD:-0}
NUM_SHARDS=${NUM_SHARDS:-12}
LIMIT=${LIMIT:-0}
# GT-injection DIAGNOSTIC passthrough (infer_and_eval_worker_ray.py --oracle).
# 'off' (the default) is byte-identical to every previous invocation of this
# script. 'gt' feeds the GRU the CURRENT view's GT pose at its input only; the
# worker asserts CONDITIONING=prev_pred_gru and prints a loud banner.
# NOT AN ARM: a run with ORACLE=gt must never be aggregated into
# summary/averages_table.csv alongside honest arms -- see
# GRU_GAP_CLOSURE_DIRECTIVE.md rung 0 for why (the GRU emits a translation
# ~3 GT-scales wrong even when handed an exact pose).
ORACLE=${ORACLE:-off}

[ -n "$CKPT" ] || { echo "ERROR: CKPT is required, e.g. CKPT=\$CKPT_ROOT/captain_cut3r_finetune_aug_full/<run>/checkpoint-final.pth" >&2; exit 1; }
mn5_require f "$CKPT" f "$EVAL_SCRIPT" d "$SCENES_ROOT" s "$SCENE_LIST"
mn5_require_slurm

export PYTHONPATH="$CRAY/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs" "$CLAIM_DIR"

NUM_GPUS=$(mn5_require_gpus) || exit 1
echo "Node: $(hostname)  GPUs visible: $NUM_GPUS  BASE_SHARD=$BASE_SHARD  NUM_SHARDS=$NUM_SHARDS  LIMIT=$LIMIT"
echo "Ckpt: $CKPT  conditioning=$CONDITIONING  oracle=$ORACLE  label=$LABEL"
echo "Scenes: $(wc -l < "$SCENE_LIST")   Start: $(date)"

# One worker process per GPU; all share the cooperative claim queue.
for g in $(seq 0 $((NUM_GPUS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=$g
    SHARD=$((BASE_SHARD + g))
    echo "[$(hostname) g$g shard$SHARD] starting at $(date)"
    python "$CRAY/eval_pipeline/infer_and_eval_worker_ray.py" \
      --ckpt "$CKPT" --label "$LABEL" --size 320 \
      --conditioning "$CONDITIONING" --oracle "$ORACLE" \
      --scenes_root "$SCENES_ROOT" --scene_list "$SCENE_LIST" \
      --pred_base "$OUT/$LABEL/preds" --eval_base "$OUT/$LABEL/eval" \
      --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
      --shard_id "$SHARD" --num_shards "$NUM_SHARDS" --device cuda \
      --claim_dir "$CLAIM_DIR" --limit "$LIMIT" \
      > "$OUT/logs/worker_cray_$(hostname)_g${g}_${LABEL}.log" 2>&1
    echo "[$(hostname) g$g shard$SHARD] finished rc=$? at $(date)"
  ) &
done
wait
echo "All workers on $(hostname) done: $(date)"
