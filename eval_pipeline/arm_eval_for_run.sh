#!/bin/bash
# Validate one finished training run and, only if it is sound, submit its full
# 4292-scene evaluation across 32 GPUs (8 nodes x 4 H100).
#
#   bash eval_pipeline/arm_eval_for_run.sh <RUN_DIR> <LABEL> [--dry-run]
#
# RUN_DIR is the trainer's output_dir (the one holding .hydra/ and
# checkpoint-final.pth). LABEL is the short name the eval outputs and the table
# row are keyed by.
#
# The --conditioning arm is NOT passed in: it is derived by preflight_ckpt.py
# from <RUN_DIR>/.hydra/config.yaml and cross-checked against the checkpoint's
# own ckpt["args"] and weight shapes. Passing it by hand is how you end up
# scoring an R8 GRU at R1, or a prev_pred_gru checkpoint as plain prev_pred.
#
# Gates, in order (any failure => nothing is submitted):
#   1. checkpoint-final.pth exists and is not still being written
#      (mtime older than SETTLE_S, and size unchanged across a second stat)
#   2. preflight_ckpt.py passes -- config/ckpt/load-path agreement, see there
#   3. this LABEL has no prior eval output (a re-arm would double-count against
#      a claim queue that is already drained). Override with FORCE=1.
#   4. gpfs_scratch has room for ~332 GB of predictions
#
# Env overrides: OUT_ROOT, SETTLE_S, FORCE, NEED_GB
set -o pipefail

WT=${WT:-/gpfs/home/koc/koc821022/vggt_features}
export OUT_ROOT=${OUT_ROOT:-/gpfs/scratch/etur59/koc821022/outputs}
OUT=$OUT_ROOT/cut3r_eval
SETTLE_S=${SETTLE_S:-120}
NEED_GB=${NEED_GB:-400}

RUN_DIR=${1:?usage: arm_eval_for_run.sh <RUN_DIR> <LABEL> [--dry-run]}
LABEL=${2:?usage: arm_eval_for_run.sh <RUN_DIR> <LABEL> [--dry-run]}
DRY=${3:-}
RUN_DIR=${RUN_DIR%/}
CKPT=$RUN_DIR/checkpoint-final.pth

say () { echo "[arm $LABEL] $*"; }
# Two distinct failure kinds, because the caller must react differently:
#   die      exit 1  -- PERMANENT. The checkpoint is wrong/unscoreable, or the
#                       label is already populated. Retrying changes nothing.
#   tempfail exit 75 -- TRANSIENT (EX_TEMPFAIL). Nothing is wrong yet; the
#                       condition just is not true *yet*. The caller should
#                       come back on its next poll.
# Conflating these retires an arm forever over a race that resolves in seconds
# (e.g. catching checkpoint-final.pth at 111s old against a 120s settle gate).
die      () { echo "[arm $LABEL] REFUSING: $*" >&2; exit 1; }
tempfail () { echo "[arm $LABEL] NOT YET: $*" >&2; exit 75; }

# Falsifier env vars silently contaminate a scoring run and nothing logs it.
# POSE_GRU_FORCE_ITERS in particular would override the restored R lever.
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE \
      POSE_GRU_HIDDEN_ZERO POSE_GRU_HIDDEN_SHUFFLE \
      POSE_GRU_IMG_FEAT_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_FORCE_ITERS

[ -d "$RUN_DIR" ] || die "no such run dir: $RUN_DIR"
[ -f "$CKPT" ]    || die "no checkpoint-final.pth in $RUN_DIR"
[ -s "$OUT/scene_list.txt" ] || die "missing $OUT/scene_list.txt"

# --- gate 1: the writer is done ---------------------------------------------
age=$(( $(date +%s) - $(stat -c %Y "$CKPT") ))
[ "$age" -ge "$SETTLE_S" ] || tempfail "$CKPT was modified ${age}s ago (< ${SETTLE_S}s) -- still being written"
s1=$(stat -c %s "$CKPT"); sleep 10; s2=$(stat -c %s "$CKPT")
[ "$s1" = "$s2" ] || tempfail "checkpoint size changed under us ($s1 -> $s2) -- still being written"
say "checkpoint settled: $(( s2 / 1000000 )) MB, mtime ${age}s ago"

# --- gate 3 (cheap, do it before the 3 GB torch.load) ------------------------
# Checks eval_gru too: a stale GRU-pose tree left behind by a manual wipe of
# only eval/preds/claims would otherwise pair an OLD checkpoint's <label>_gru
# row with the new regular row in the averages table, silently. A re-arm wipe
# must therefore cover eval/, eval_gru/, preds/, preds_gru/ and claims/.
if [ "${FORCE:-0}" != "1" ]; then
  for d in eval eval_gru; do
    [ -d "$OUT/$LABEL/$d" ] || continue
    n=$(find "$OUT/$LABEL/$d" -mindepth 2 -name eval_depth_pose_metrics.csv -printf . 2>/dev/null | wc -c)
    [ "$n" -eq 0 ] || die "$OUT/$LABEL/$d already holds $n scene results -- set FORCE=1 to re-arm"
  done
fi

# --- gate 4 ------------------------------------------------------------------
free_gb=$(bsc_quota 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
          | awk '/gpfs_scratch/ {u=$3; uq=$4; q=$5; qq=$6;
                 if (uq=="TB") u*=1024; if (qq=="TB") q*=1024; printf "%d", q-u}')
if [ -n "$free_gb" ]; then
  say "gpfs_scratch headroom: ${free_gb} GB (need ~${NEED_GB})"
  # Transient by nature: another arm's predictions may be cleaned up, or a
  # concurrent eval may finish. Retiring the arm over a full disk would be
  # the wrong permanent answer to a temporary condition.
  [ "$free_gb" -ge "$NEED_GB" ] || tempfail "only ${free_gb} GB left on gpfs_scratch (need ${NEED_GB})"
else
  say "WARNING: could not parse bsc_quota -- skipping the disk gate"
fi

# --- gate 2: the real soundness check ---------------------------------------
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything || die "conda activate cuteanything failed"
cd "$WT" || die "cannot cd $WT"

PF=$(python eval_pipeline/preflight_ckpt.py --run_dir "$RUN_DIR" 2>&1)
echo "$PF" | sed "s/^/[arm $LABEL] /"
ARM=$(echo "$PF" | sed -n 's/^PREFLIGHT_OK ARM=\([^ ]*\).*/\1/p')
[ -n "$ARM" ] || die "preflight did not pass -- see the output above"
say "preflight passed: --conditioning $ARM"

if [ "$DRY" = "--dry-run" ]; then
  say "DRY RUN -- would submit 8 nodes x 4 GPUs with LABEL=$LABEL CONDITIONING=$ARM"
  exit 0
fi

# --- submit: 8 nodes x 4 GPUs = 32 workers on one cooperative claim queue ----
mkdir -p "$OUT/logs" || die "cannot create $OUT/logs"
say "submitting 8 nodes (32 GPUs), ckpt=$CKPT"
for B in 0 4 8 12 16 20 24 28; do
  sbatch \
    --job-name="ev_${LABEL}_b${B}" \
    --output="$OUT/logs/slurm_${LABEL}_b${B}_%j.out" \
    --error="$OUT/logs/slurm_${LABEL}_b${B}_%j.err" \
    --export=ALL,OUT_ROOT="$OUT_ROOT",LABEL="$LABEL",CONDITIONING="$ARM",CKPT="$CKPT",BASE_SHARD="$B",NUM_SHARDS=32,LIMIT=0 \
    eval_pipeline/run_captain_ray_eval_node.sh || die "sbatch failed at BASE_SHARD=$B"
done
say "armed and submitted"
