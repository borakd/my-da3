#!/bin/bash
# Armed watcher: sit on the in-flight finetunes and fire each arm's full
# 4292-scene / 32-GPU evaluation the moment its training job lands COMPLETED
# with a sound checkpoint-final.pth.
#
#   nohup bash eval_pipeline/armed_eval_watcher.sh > <log> 2>&1 &
#
# Each arm is independent: one job finishing early gets its eval submitted
# immediately, one job dying does not hold up the others.
#
# Per-arm sequence, all gates hard:
#   a. the SLURM job leaves the queue
#   b. sacct reports COMPLETED with ExitCode 0:0   (TIMEOUT / FAILED / CANCELLED
#      => mark FAILED, never submit -- a walltime kill leaves a valid-looking
#      checkpoint-last.pth and no final)
#   c. checkpoint-final.pth appears within CKPT_GRACE, and its mtime is NEWER
#      than the job's start time (so a leftover from an earlier attempt at the
#      same exp_name can never be scored by mistake)
#   d. eval_pipeline/arm_eval_for_run.sh passes its own gates (writer settled,
#      preflight_ckpt.py config/ckpt/load-path agreement, no prior output for
#      the label, disk headroom) and then submits 8 nodes x 4 GPUs
#
# The --conditioning arm is never hardcoded here -- preflight_ckpt.py derives
# it from each run's own .hydra/config.yaml. See that file for why.
#
# State lives in $OUT/armed/<label>.{submitted,failed} so a restarted watcher
# picks up where it left off instead of re-submitting.
set -o pipefail

WT=${WT:-/gpfs/home/koc/koc821022/my-da3}
export OUT_ROOT=${OUT_ROOT:-/gpfs/scratch/etur59/koc821022/outputs}
OUT=$OUT_ROOT/cut3r_eval
# Run dirs live under TWO roots since the 2026-08-14 storage move: new
# finetunes save to checkpoints/, everything older was mv'd to
# checkpoints_projects/, and the old /gpfs/projects root is gone. A watcher
# pinned to one root does not error -- it logs "no <path> yet" forever and
# silently never arms the run. So CR_ROOTS is a search path, newest first, and
# every run dir is resolved through cr_run().
CR_ROOTS=(
  "${CR:-/gpfs/scratch/etur59/koc821022/checkpoints}/captain_cut3r_finetune_aug_full"
  /gpfs/scratch/etur59/koc821022/checkpoints_projects/captain_cut3r_finetune_aug_full
)
# cr_run <run-dir-name> -> absolute path of the first root that has it.
# Prints the FIRST root's candidate when the run exists nowhere, so the
# "not there yet" log line names the place a new run is expected to appear.
cr_run () {
  local r
  for r in "${CR_ROOTS[@]}"; do [ -d "$r/$1" ] && { echo "$r/$1"; return 0; }; done
  echo "${CR_ROOTS[0]}/$1"; return 1
}
STATE=$OUT/armed
POLL=${POLL:-300}
CKPT_GRACE=${CKPT_GRACE:-1800}   # seconds to wait for the final ckpt after job exit
DEADLINE=$(( $(date +%s) + ${MAX_HOURS:-720} * 3600 ))   # standing order: 30d, not 3

# The registry is a FILE, not a constant: a crashed finetune gets resubmitted
# under a new jobid, and re-arming should be "edit one line + rm the .failed
# marker + restart the watcher", not "edit this script".
#
#   jobid|run dir under $CR|eval label[|jobname]
#
# (jobid -> config confirmed with `sacct -o SubmitLine`; config -> output_dir is
#  ${save_dir}/${exp_name}, and exp_name == the config's own name for all four)
#
# The optional 4th field is a SLURM job NAME, and it exists because a finetune
# may be submitted as an auto-retry ladder:
#
#   sbatch --dependency=afternotok:<prev> --kill-on-invalid-dep=yes ...
#
# Each link only runs if the previous one failed, and the chain self-cancels
# once one succeeds. Watching a single jobid would declare the arm FAILED the
# moment the first link crashes -- while the ladder goes on to finish the run.
# With a jobname set, the arm is "still training" while ANY job of that name is
# queued or running, and it is judged on whichever link ENDED LAST.
mkdir -p "$STATE" || exit 1
REGISTRY=${REGISTRY:-$STATE/registry.txt}
touch "$REGISTRY"
log () { echo "$(date '+%F %T') $*"; }

# --- standing mode ------------------------------------------------------------
# The watcher used to break out of its loop once every registered arm resolved.
# That is wrong for a standing order to cover whatever gets submitted next: it
# exited at 05:18 and three finetunes launched at 10:35 and 14:48 went entirely
# unwatched. It now runs until $STOP appears (or MAX_HOURS elapses), and each
# poll re-reads the registry and looks for training jobs it has not seen.
STOP=${STOP:-$STATE/STOP}
[ -e "$STOP" ] && { log "refusing to start: $STOP exists (rm it to re-arm)"; exit 1; }

# Auto-discovery. Every poll, any of MY queued/running jobs whose SubmitLine
# names a trainer sbatch gets registered, so a run submitted while this is
# already up is still covered. Two deliberate limits:
#   * only jobs SEEN IN THE QUEUE are auto-armed. A finished-but-unevaluated
#     run dir found lying on disk is REPORTED, never armed: without a job to
#     judge, "did it finish cleanly" is unanswerable, and one of those dirs is
#     the oracle diagnostic that must never reach the table.
#   * the label is derived, then checked for collisions. Two arms sharing a
#     label would write into one output tree and silently blend two models.
TRAIN_SBATCH_RE=${TRAIN_SBATCH_RE:-train_.*\.sbatch}

derive_label () {  # $1 = exp/config name -> the short table key
  local n=$1
  n=${n#captain_}          # captain_gru_v3_a4_g3_f1r_r8_finetune
  n=${n/_v3_/_}            # gru_a4_g3_f1r_r8_finetune
  n=${n%_finetune}
  n=${n/noproj/np}
  local head=${n%%_*} rest=${n#*_}
  echo "${head}_${rest//_/}"
}

discover () {
  local jid jname sl cfg run run_dir label
  while IFS='|' read -r jid jname; do
    [ -n "$jid" ] || continue
    grep -q "^$jid|" "$REGISTRY" && continue
    grep -q "|$jname\$" "$REGISTRY" && continue     # another link of a retry ladder
    sl=$(sacct -n -X -j "$jid" -o SubmitLine --parsable2 2>/dev/null | head -n1)
    echo "$sl" | grep -qE "$TRAIN_SBATCH_RE" || continue
    cfg=$(echo "$sl" | awk '{print $NF}')           # trainer takes the config name last
    run=$cfg
    run_dir=$(cr_run "$run") || { log "discover: job $jid ($jname) -> config '$cfg' but no $run_dir yet"; continue; }
    label=$(derive_label "$cfg")
    # Deliberately-ignored runs must not come back on the next poll just
    # because someone tidied their line out of the registry.
    [ -f "$STATE/$label.ignored" ] && continue
    if grep -q "|$label|" "$REGISTRY"; then
      log "discover: REFUSING job $jid -- label '$label' already registered (would collide)"
      continue
    fi
    if [ -d "$OUT/$label/eval" ]; then
      log "discover: REFUSING job $jid -- $OUT/$label/eval already exists"
      continue
    fi
    echo "$jid|$run|$label|$jname" >> "$REGISTRY"
    log "discover: ARMED $label  job $jid ($jname)  $run_dir"
  done < <(squeue -u "$USER" -h -o '%i|%j' -t PENDING,RUNNING 2>/dev/null)
}

# Finished run dirs nobody has scored. Reported once per poll-cycle change, and
# never acted on -- see the note above.
report_orphans () {
  local r d n label
  # Both roots, because a run finished before the 2026-08-14 move and one
  # started after it are equally unscored.
  for r in "${CR_ROOTS[@]}"; do
    [ -d "$r" ] || continue
    for d in "$r"/*/; do
      [ -f "$d/checkpoint-final.pth" ] || continue
      n=$(basename "$d")
      label=$(derive_label "$n")
      [ -d "$OUT/$label/eval" ] && continue
      grep -q "|$n|" "$REGISTRY" && continue
      log "orphan: $n has checkpoint-final.pth but no eval and no arm (not auto-armed)"
    done
  done
}

log "armed watcher up (standing): poll ${POLL}s, OUT=$OUT, stop with: touch $STOP"

# Job start times back the stale-checkpoint gate. They are looked up LAZILY,
# not once up front: a re-armed job that is still PENDING has Start=Unknown,
# and caching that as 0 would silently disable the gate for exactly the arm
# that needs it most (its run dir already holds the crashed attempt's
# checkpoints). sacct keeps Start after the job ends, so resolving it at gate
# (c) is both correct and always available by then.
#
# WAITED must be declared associative too -- an undeclared name would index
# arithmetically and every label would collide on slot 0.
declare -A JSTART
declare -A WAITED

job_start_epoch () {  # $1 = jobid; echoes epoch seconds, or 0 if not started yet
  local st
  st=$(sacct -n -X -j "$1" -o Start --parsable2 2>/dev/null | head -n1)
  case "$st" in
    ""|Unknown|None) echo 0 ;;
    *) date -d "$st" +%s 2>/dev/null || echo 0 ;;
  esac
}

# Is any job of this arm still queued or running? With a jobname the whole
# retry ladder counts; without one, just the anchor jobid.
arm_in_queue () {  # $1 = jobid, $2 = jobname (may be empty)
  if [ -n "$2" ]; then
    squeue -u "$USER" -h -n "$2" -o '%i' 2>/dev/null | grep -q .
  else
    squeue -h -j "$1" -o '%i' 2>/dev/null | grep -q .
  fi
}

# The jobid that actually decides the arm's fate: the last link of the ladder
# to END. sacct is queried from the anchor job's submit time so that earlier,
# unrelated runs with the same name (e.g. a crashed pre-ladder attempt) can
# never be mistaken for this arm's outcome.
arm_final_jobid () {  # $1 = anchor jobid, $2 = jobname (may be empty)
  if [ -z "$2" ]; then echo "$1"; return; fi
  local since
  since=$(sacct -n -X -j "$1" -o Submit --parsable2 2>/dev/null | head -n1)
  [ -n "$since" ] || since=now-3days
  sacct -n -X --name "$2" -S "$since" -o JobID,End,State --parsable2 2>/dev/null \
    | grep -vE '\|(PENDING|RUNNING)$' \
    | grep -vE '\|(Unknown|None)\|' \
    | sort -t'|' -k2,2 \
    | tail -n1 | cut -d'|' -f1
}

while :; do
  discover
  # Re-read every poll: discover appends to it, and a human re-arming a crashed
  # finetune is meant to be "edit one line, rm the marker" with no restart.
  mapfile -t ARMS < <(grep -vE '^\s*(#|$)' "$REGISTRY")
  pending=0; line=""
  for a in "${ARMS[@]}"; do
    IFS='|' read -r jid run label jobname <<< "$a"
    RUN_DIR=$(cr_run "$run")

    # .ignored is a THIRD terminal state, distinct from .failed on purpose: a
    # run that is deliberately not a table arm (a short probe, a debug run) is
    # not a failure, and parking it under .failed would make a healthy watcher
    # permanently report FAILED for something nobody ever intended to score.
    if [ -f "$STATE/$label.ignored" ]; then
      line="$line ${label}=ignored"
      continue
    fi
    if [ -f "$STATE/$label.submitted" ] || [ -f "$STATE/$label.failed" ]; then
      line="$line ${label}=$( [ -f "$STATE/$label.submitted" ] && echo SUBMITTED || echo FAILED)"
      continue
    fi

    # (a) still queued or running? (the whole retry ladder, if there is one)
    if arm_in_queue "$jid" "$jobname"; then
      ep=$(wc -l < "$RUN_DIR/log.txt" 2>/dev/null || echo 0)
      line="$line ${label}=train(ep=$ep)"
      pending=$((pending + 1))
      continue
    fi

    # (b) how did the deciding link end?
    decider=$(arm_final_jobid "$jid" "$jobname")
    [ -n "$decider" ] || decider=$jid
    [ "$decider" = "$jid" ] || log "arm $label: ladder resolved to job $decider (anchor $jid)"
    read -r state exitcode < <(sacct -n -X -j "$decider" -o State,ExitCode --parsable2 2>/dev/null \
                               | head -n1 | tr '|' ' ')
    if [ "$state" != "COMPLETED" ]; then
      log "arm $label: job $decider ended $state ($exitcode) -- NOT submitting eval"
      echo "$state $exitcode (job $decider)" > "$STATE/$label.failed"
      line="$line ${label}=FAILED($state)"
      continue
    fi
    jid=$decider   # the stale-checkpoint gate must compare against THIS job's start

    # (c) the final checkpoint, and it must belong to THIS job
    CKPT=$RUN_DIR/checkpoint-final.pth
    if [ ! -f "$CKPT" ]; then
      waited=${WAITED[$label]:-0}
      if [ "$waited" -ge "$CKPT_GRACE" ]; then
        log "arm $label: job COMPLETED but no checkpoint-final.pth after ${CKPT_GRACE}s"
        echo "no checkpoint-final.pth" > "$STATE/$label.failed"
        line="$line ${label}=FAILED(no-final)"
        continue
      fi
      WAITED[$label]=$((waited + POLL))
      line="$line ${label}=await-final"
      pending=$((pending + 1))
      continue
    fi
    mt=$(stat -c %Y "$CKPT")
    JSTART[$label]=$(job_start_epoch "$jid")
    log "arm $label: ckpt mtime $mt vs job start ${JSTART[$label]}"
    if [ "${JSTART[$label]}" -gt 0 ] && [ "$mt" -lt "${JSTART[$label]}" ]; then
      log "arm $label: checkpoint-final.pth predates job start -- stale leftover, refusing"
      echo "stale checkpoint-final.pth (mtime $mt < job start ${JSTART[$label]})" > "$STATE/$label.failed"
      line="$line ${label}=FAILED(stale-ckpt)"
      continue
    fi

    # (d) validate + submit
    log "arm $label: job $jid COMPLETED, checkpoint present -- running gates"
    bash "$WT/eval_pipeline/arm_eval_for_run.sh" "$RUN_DIR" "$label"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      date > "$STATE/$label.submitted"
      log "arm $label: EVAL SUBMITTED (8 nodes x 4 GPUs)"
      line="$line ${label}=SUBMITTED"
    elif [ "$rc" -eq 75 ]; then
      # EX_TEMPFAIL: the gate is not satisfied YET (settle window, disk).
      # Stay pending and try again next poll -- do NOT retire the arm.
      log "arm $label: gate not satisfied yet (rc=75) -- will retry next poll"
      line="$line ${label}=retry"
      pending=$((pending + 1))
    else
      log "arm $label: gates REFUSED the checkpoint (rc=$rc) -- no eval submitted"
      echo "arm_eval_for_run.sh refused (rc=$rc)" > "$STATE/$label.failed"
      line="$line ${label}=FAILED(gates)"
    fi
  done

  log "pending=$pending |$line"
  [ "$pending" -eq 0 ] && report_orphans
  # pending==0 is NOT a reason to exit any more: it just means everything armed
  # so far has resolved, and the whole point of the standing order is to still
  # be here when the next finetune is submitted.
  [ -e "$STOP" ] && { log "STOP file present -- standing down"; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "STATUS=TIMEOUT pending=$pending"; exit 3; }
  sleep "$POLL"
done

log "=== summary ==="
for a in "${ARMS[@]}"; do
  IFS='|' read -r jid run label <<< "$a"
  if [ -f "$STATE/$label.submitted" ]; then
    log "  $label  SUBMITTED  $(cat "$STATE/$label.submitted")"
  else
    log "  $label  FAILED     $(cat "$STATE/$label.failed" 2>/dev/null)"
  fi
done
