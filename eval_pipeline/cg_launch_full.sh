#!/bin/bash
# Full 4292-scene eval of a STATE_GATE inference variant of the augfull
# baseline checkpoint. arm_eval_for_run.sh CANNOT be used for gate arms: it
# unsets every STATE_GATE_* var by design (protecting honest arms). This
# launcher passes the gate config EXPLICITLY in --export so sacct records it.
#
#   bash eval_pipeline/cg_launch_full.sh <LABEL> KEY=VAL [KEY=VAL ...]
# e.g.
#   bash eval_pipeline/cg_launch_full.sh augfull_cg_g7ema \
#     STATE_GATE_MODE=soft STATE_GATE_SIGNAL=conf_mean STATE_GATE_TAU=-11.99 \
#     STATE_GATE_TEMP=1.5 STATE_GATE_GMIN=0.70 STATE_GATE_EMA=0.7
#
# Submits 8 nodes x 4 GPU (BASE_SHARD 0..28, NUM_SHARDS=32) over the full
# scene list, mirroring arm_eval_for_run.sh:111-118. ~332 GB of preds.
set -o pipefail
WT=${WT:-/gpfs/home/koc/koc821022/my-da3}
OUT_ROOT=${OUT_ROOT:-/gpfs/scratch/etur59/koc821022/outputs}
OUT=$OUT_ROOT/cut3r_eval
CKPT=${CKPT:-/gpfs/scratch/etur59/koc821022/checkpoints_projects/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth}

LABEL=${1:?usage: cg_launch_full.sh <LABEL> STATE_GATE_MODE=... [more KEY=VAL]}
shift
GATE_ENV=""
for kv in "$@"; do
  case "$kv" in
    STATE_GATE_*=*|REVERSE=*|REVISIT=*) GATE_ENV="$GATE_ENV,$kv" ;;
    *) echo "REFUSING: '$kv' is not a STATE_GATE_*/REVERSE/REVISIT assignment" >&2; exit 1 ;;
  esac
done
[ -n "$GATE_ENV" ] || { echo "REFUSING: no STATE_GATE_* vars given" >&2; exit 1; }

[ -f "$CKPT" ] || { echo "REFUSING: no ckpt $CKPT" >&2; exit 1; }
if [ -d "$OUT/$LABEL/eval" ] && [ "${FORCE:-0}" != "1" ]; then
  n=$(find "$OUT/$LABEL/eval" -mindepth 2 -name eval_depth_pose_metrics.csv -printf . 2>/dev/null | wc -c)
  [ "$n" -eq 0 ] || { echo "REFUSING: $LABEL already holds $n scene results (FORCE=1 to override)" >&2; exit 1; }
fi

# ~332 GB of predictions land on gpfs_scratch; refuse without headroom.
free_gb=$(bsc_quota 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
          | awk '/gpfs_scratch/ {u=$3; q=$5; print int(q-u)*1024}')
if [ -n "$free_gb" ] && [ "$free_gb" -lt "${NEED_GB:-400}" ]; then
  echo "REFUSING: only ${free_gb} GB under quota (< ${NEED_GB:-400})" >&2; exit 1
fi

echo "Submitting 8-node full eval: LABEL=$LABEL gate:${GATE_ENV#,}"
for B in 0 4 8 12 16 20 24 28; do
  sbatch --job-name="ev_${LABEL}_b${B}" \
    --output="$OUT/logs/slurm_${LABEL}_b${B}_%j.out" \
    --error="$OUT/logs/slurm_${LABEL}_b${B}_%j.err" \
    --export="ALL,OUT_ROOT=$OUT_ROOT,LABEL=$LABEL,CONDITIONING=none,CKPT=$CKPT,BASE_SHARD=$B,NUM_SHARDS=32,LIMIT=0,ORACLE=off$GATE_ENV" \
    "$WT/eval_pipeline/run_captain_ray_eval_node.sh"
done
