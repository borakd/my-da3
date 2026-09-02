#!/bin/bash
# ONE syncer for MANY concurrent offline W&B runs, syncing each SEQUENTIALLY.
#
#   ./sync_all_offline_runs.sh <interval_s> <jobid>:<run_output_dir> [more...]
#   setsid nohup ./sync_all_offline_runs.sh 900 44298249:/path/a 44298250:/path/b &
#
# WHY ONE SYNCER INSTEAD OF ONE LOOP PER RUN.
# Per-run loops were tried first and failed. Each `wandb sync --append` re-sends
# the whole transaction log PLUS accumulated files, so its duration grows without
# bound (16s early -> 25min late, as media and the tb events file pile up). Any
# FIXED stagger is therefore eventually smaller than a sync takes, the loops
# overlap on the shared MN5 proxy, each slows the other, and they can wedge into
# permanent mutual timeout (observed on jobs 44241270/44241271 at 07:11).
# Syncing sequentially from a single process makes overlap impossible by
# construction, whatever the durations become.
#
# --append is safe and idempotent here; verified on wandb 0.25.1:
#   sync 5-step log -> 5 rows; --append 10-step superset -> 10 rows (deduped,
#   not shifted); --append the same log again -> unchanged.
#
# No --sync-tensorboard: wandb.tensorboard.patch() already folded the tb scalars
# into the .wandb log at write time; passing it would double-count.
set -o pipefail

INTERVAL=${1:?usage: sync_all_offline_runs.sh <interval_s> <jobid>:<dir> ...}
shift
SYNC_TIMEOUT=${SYNC_TIMEOUT:-1800}
LOG=${SYNC_ALL_LOG:-/gpfs/scratch/etur59/koc821022/checkpoints/sync_all.log}

source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

declare -A DIR FINAL
ORDER=()
for spec in "$@"; do
  j=${spec%%:*}; d=${spec#*:}
  DIR[$j]=$d; FINAL[$j]=0; ORDER+=("$j")
done
say "syncer started: ${#ORDER[@]} runs, interval=${INTERVAL}s timeout=${SYNC_TIMEOUT}s (SEQUENTIAL)"

sync_one() {                      # $1=jobid -> 0 ok, 1 problem
  local j=$1 d=${DIR[$1]} R t0 rc dt
  R=$(ls -d "$d"/wandb/offline-run-* 2>/dev/null | head -1)
  [ -n "$R" ] || { say "  $(basename "$d"): no offline dir yet"; return 0; }
  t0=$SECONDS
  timeout "$SYNC_TIMEOUT" wandb sync --append "$R" >>"$LOG" 2>&1
  rc=$?; dt=$((SECONDS-t0))
  case $rc in
    0)   say "  $(basename "$d"): OK in ${dt}s ($(du -sh "$R" 2>/dev/null | cut -f1))" ;;
    124) say "  $(basename "$d"): TIMEOUT after ${dt}s" ; return 1 ;;
    *)   say "  $(basename "$d"): FAILED rc=$rc after ${dt}s" ; return 1 ;;
  esac
}

while true; do
  remaining=0
  for j in "${ORDER[@]}"; do
    [ "${FINAL[$j]}" = "2" ] && continue          # already closed out
    if squeue -h -j "$j" 2>/dev/null | grep -q .; then
      remaining=1
      sync_one "$j"
    else
      if [ "${FINAL[$j]}" = "0" ]; then
        say "job $j left the queue -- 90s flush then final sync"
        sleep 90
        FINAL[$j]=1
      fi
      sync_one "$j" && FINAL[$j]=2 && say "job $j FINAL SYNC DONE: $(sacct -j "$j" --format=State,ExitCode,Elapsed -n 2>/dev/null | head -1 | tr -s ' ')"
      [ "${FINAL[$j]}" != "2" ] && remaining=1     # retry the final sync next cycle
    fi
  done
  [ $remaining -eq 0 ] && { say "all runs closed out; syncer exiting"; exit 0; }
  sleep "$INTERVAL"
done
