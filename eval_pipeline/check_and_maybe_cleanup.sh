#!/bin/bash
# Deterministic state machine for the autonomous finalize of the CUT3R eval run.
# Two concurrent jobs:
#   - main (jobid.txt):           best + final  (its regular pass is a no-op via
#                                 the SKIP_regular sentinel)
#   - parallel-regular (regular_jobid.txt): regular, on a separate node
# Prints a final line "VERDICT: <STATE>":
#   RUNNING             -> a job is still in queue; reschedule (no ping)
#   RESUBMITTED_MAIN    -> main died before best/final done; resubmitted; reschedule
#   RESUBMITTED_REGULAR -> regular job died before done; resubmitted; reschedule
#   CLEANUP_RESUBMITTED -> all finished but coverage incomplete; one cleanup pass; reschedule
#   COMPLETE            -> done; caller should aggregate, present table, and ping
set -o pipefail
OUT=/scratch/bdursun25/cuteanything/outputs/cut3r_eval
MYDA3=/scratch/bdursun25/cuteanything/my-da3
JOBID=$(cat "$OUT/logs/jobid.txt" 2>/dev/null)
RJOB=$(cat "$OUT/logs/regular_jobid.txt" 2>/dev/null)
echo "main_jobid=$JOBID regular_jobid=$RJOB"

# --- Intermediate per-checkpoint averages (idempotent) ---------------------
# A checkpoint is "finished" once all 4 shards logged DONE (globs cover both the
# main job's worker_g*_ logs and the parallel job's worker_NEWg*_ logs). When a
# checkpoint finishes, snapshot its averages to averages_after_<label>.{txt,md}
# (eval CSVs already exist from inline per-scene eval), plus a cumulative table.
AGG="conda run -n cuteanything python /scratch/bdursun25/cuteanything/my-da3/eval_pipeline/aggregate_results.py"
SL="$OUT/scene_list.txt"
done_labels=""
for lbl in best final regular; do
  n=$(grep -h DONE "$OUT"/logs/worker_*_${lbl}.log 2>/dev/null | wc -l)
  if [ "$n" -ge 4 ]; then
    done_labels="$done_labels $lbl"
    if [ ! -f "$OUT/summary/averages_after_${lbl}.txt" ]; then
      echo "NEW_SNAPSHOT: $lbl (computing averages for finished checkpoint)"
      $AGG --out_root "$OUT" --scene_list "$SL" --labels "$lbl" >/dev/null 2>&1
      cp "$OUT/summary/averages_table.txt" "$OUT/summary/averages_after_${lbl}.txt" 2>/dev/null
      cp "$OUT/summary/averages_table.md"  "$OUT/summary/averages_after_${lbl}.md"  2>/dev/null
    fi
  fi
done
if [ -n "$done_labels" ]; then
  $AGG --out_root "$OUT" --scene_list "$SL" --labels $done_labels >/dev/null 2>&1
fi
echo "finished_checkpoints:$done_labels"
# ---------------------------------------------------------------------------

# Are EITHER of the jobs still in the queue (R or PD)?
in_queue=""
for j in "$JOBID" "$RJOB"; do
  [ -z "$j" ] && continue
  st=$(squeue -j "$j" -h -o "%t" 2>/dev/null | head -1)
  [ -n "$st" ] && in_queue="$in_queue $j:$st"
done
if [ -n "$in_queue" ]; then
  echo "in_queue:$in_queue"
  for lbl in best final regular; do
    c=$(find "$OUT/$lbl/eval" -name eval_depth_pose_metrics.csv 2>/dev/null | wc -l)
    echo "  $lbl: $c / 4292"
  done
  echo "VERDICT: RUNNING"
  exit 0
fi

# Both jobs gone from the queue. Per-label completion = all 4 shards logged DONE.
best_done=$(grep -h DONE "$OUT"/logs/worker_*_best.log 2>/dev/null | wc -l)
final_done=$(grep -h DONE "$OUT"/logs/worker_*_final.log 2>/dev/null | wc -l)
reg_done=$(grep -h DONE "$OUT"/logs/worker_*_regular.log 2>/dev/null | wc -l)
echo "shard_done: best=$best_done final=$final_done regular=$reg_done (>=4 each = finished)"
for lbl in best final regular; do
  c=$(find "$OUT/$lbl/eval" -name eval_depth_pose_metrics.csv 2>/dev/null | wc -l)
  echo "  $lbl: $c / 4292"
  eval "cnt_$lbl=$c"
done
nfail=$(cat "$OUT"/{best,final,regular}/eval/_failures_shard*.txt 2>/dev/null | awk -F'\t' '{print $1}' | sort -u | wc -l)
echo "unique_failed_scenes=$nfail"

# If main (best/final) didn't finish, resume it (resumable; regular stays no-op).
if [ "$best_done" -lt 4 ] || [ "$final_done" -lt 4 ]; then
  cd "$MYDA3" || exit 1
  NEW=$(sbatch --parsable eval_pipeline/run_cut3r_eval.sh)
  echo "$NEW" > "$OUT/logs/jobid.txt"
  echo "resubmitted_main=$NEW"
  echo "VERDICT: RESUBMITTED_MAIN"
  exit 0
fi
# If the parallel regular job didn't finish, resume just it.
if [ "$reg_done" -lt 4 ]; then
  cd "$MYDA3" || exit 1
  NEW=$(sbatch --parsable eval_pipeline/run_regular_parallel.sh)
  echo "$NEW" > "$OUT/logs/regular_jobid.txt"
  echo "resubmitted_regular=$NEW"
  echo "VERDICT: RESUBMITTED_REGULAR"
  exit 0
fi

# All three finished. If coverage disagrees or there are failures, run ONE cleanup
# pass (OOM-hardened worker retries missing scenes), unless already done.
need_cleanup=0
if [ "$cnt_best" -ne "$cnt_final" ] || [ "$cnt_best" -ne "$cnt_regular" ] || [ "$nfail" -gt 0 ]; then
  need_cleanup=1
fi
if [ "$need_cleanup" -eq 1 ] && [ ! -f "$OUT/logs/cleanup_submitted.txt" ]; then
  cd "$MYDA3" || exit 1
  rm -f "$OUT/logs/SKIP_regular"   # let the cleanup pass also backfill regular
  NEW=$(sbatch --parsable eval_pipeline/run_cut3r_eval.sh)
  echo "$NEW" > "$OUT/logs/jobid.txt"
  echo "$NEW" > "$OUT/logs/cleanup_submitted.txt"
  echo "cleanup_resubmitted=$NEW"
  echo "VERDICT: CLEANUP_RESUBMITTED"
  exit 0
fi

echo "VERDICT: COMPLETE"
exit 0
