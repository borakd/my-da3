#!/bin/bash
# Periodically push a still-running OFFLINE W&B run up to wandb.ai until its SLURM
# job finishes, then do one final sync.
#
#   ./sync_offline_run.sh <jobid> <run_output_dir> [interval_s]
#   setsid nohup ./sync_offline_run.sh 44241270 /path/to/run >/dev/null 2>&1 &
#
# WHY --append IS SAFE HERE. Verified empirically on wandb 0.25.1 by syncing a
# 5-step offline log and then re-sending a 10-step superset of the SAME run id:
#     sync   5-step log           -> 5 rows,  steps 0-4
#     --append 10-step superset   -> 10 rows, steps 0-9   (deduped, NOT shifted)
#     --append the same log again -> 10 rows, steps 0-9   (no-op)
# Each cycle simply brings the online run up to the offline log's current state.
#
# NO --sync-tensorboard: the trainer calls wandb.tensorboard.patch() before
# wandb.init(sync_tensorboard=True), so tb scalars are already inside the .wandb
# log. wandb says so itself ("Found .wandb file, not streaming tensorboard
# metrics"). Passing the flag would double-count.
#
# LESSON FROM THE FIRST RUN (job 44200762): a bare `wandb sync` hit
# "Network error (TransientError), entering retry loop" and blocked for 2h24m --
# wandb's retry loop has no ceiling. The loop never noticed the job finish and
# never ran the final sync, leaving ~2h of training unsynced. Hence SYNC_TIMEOUT
# below, plus consecutive-failure reporting. A killed sync is harmless: syncing
# only READS the offline log, and --append makes the retry idempotent.
set -o pipefail

JOB=${1:?usage: sync_offline_run.sh <jobid> <run_output_dir> [interval_s]}
D=${2:?usage: sync_offline_run.sh <jobid> <run_output_dir> [interval_s]}
INTERVAL=${3:-600}
# A sync uploads the .wandb log (tens of MB) PLUS accumulated media, which grows
# without bound -- ~240MB of print_img_freq visualisations by epoch 40. Two arms
# syncing concurrently through one proxy needs far more than 600s.
SYNC_TIMEOUT=${SYNC_TIMEOUT:-1800}
# Never retry back-to-back. Without this floor a sync that consumes the FULL
# timeout leaves rest=INTERVAL-INTERVAL=0, so the loop retries instantly; two
# loops doing that saturate the shared proxy and keep each other timing out --
# a self-sustaining failure observed on jobs 44241270/44241271 at 07:11.
MIN_REST=${MIN_REST:-120}
# Optional startup delay, so concurrent arms do not collide every cycle.
STAGGER=${STAGGER:-0}
LOG=$D/sync_loop.log

source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

fails=0
do_sync() {
  local R t0 rc dt
  R=$(ls -d "$D"/wandb/offline-run-* 2>/dev/null | head -1)
  [ -n "$R" ] || { say "no offline run dir yet"; return 1; }
  t0=$SECONDS
  timeout "$SYNC_TIMEOUT" wandb sync --append "$R" >>"$LOG" 2>&1
  rc=$?; dt=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then
    fails=0
    say "sync OK in ${dt}s (offline log $(du -sh "$R" 2>/dev/null | cut -f1))"
  elif [ $rc -eq 124 ]; then
    fails=$((fails + 1))
    say "sync TIMED OUT after ${SYNC_TIMEOUT}s (consecutive failures: $fails) -- retrying next cycle"
  else
    fails=$((fails + 1))
    say "sync FAILED rc=$rc after ${dt}s (consecutive failures: $fails) -- retrying next cycle"
  fi
  [ $fails -ge 5 ] && say "WARNING: $fails consecutive sync failures -- check connectivity/credentials"
  return 0
}

say "sync loop started; job=$JOB dir=$D interval=${INTERVAL}s timeout=${SYNC_TIMEOUT}s min_rest=${MIN_REST}s stagger=${STAGGER}s"
[ "$STAGGER" -gt 0 ] && { say "staggering ${STAGGER}s to avoid colliding with a concurrent arm"; sleep "$STAGGER"; }
while true; do
  start=$SECONDS
  do_sync
  if ! squeue -h -j "$JOB" 2>/dev/null | grep -q .; then
    say "job $JOB left the queue -- waiting 90s for final records to flush"
    sleep 90
    do_sync
    say "FINAL SYNC DONE: $(sacct -j "$JOB" --format=State,ExitCode,Elapsed -n 2>/dev/null | head -1 | tr -s ' ')"
    say "sync loop exiting"
    exit 0
  fi
  # Keep a true cadence when syncs are fast, but ALWAYS pause at least MIN_REST,
  # and back off further while failures persist so a congested proxy can recover.
  rest=$((INTERVAL - (SECONDS - start)))
  [ $rest -lt $MIN_REST ] && rest=$MIN_REST
  if [ $fails -gt 0 ]; then
    backoff=$((MIN_REST * fails)); [ $backoff -gt 1800 ] && backoff=1800
    [ $rest -lt $backoff ] && rest=$backoff
    say "backing off ${rest}s after $fails consecutive failure(s)"
  fi
  sleep $rest
done
