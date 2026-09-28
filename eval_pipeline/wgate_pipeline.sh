#!/bin/bash
# Write-gate variants supervisor (WRITE_GATE_VARIANTS.md). Runs DETACHED on a login node
# (setsid nohup bash eval_pipeline/wgate_pipeline.sh >> $OUT/wgate_pipeline/supervisor.out 2>&1 < /dev/null &)
# and drives everything after the head arms exist on the 430 subset:
#
#   S1  wait for the 430 head arms (fg_head, mg_head; jobs given in FG_JID/MG_JID or looked up in wgate430.jids)
#   S2  wgate_make_controls.py -> maks_arm_{fg_const,mg_const,fg_oracle,mg_oracle}.json (login-node CPU, ~1 min)
#   S3  submit + wait the four control arms on 430 (maks_sweep430.sbatch, PILOT=wgate430), one resubmit on failure
#   S4  table job for 430 (Slurm, 1 GPU: wgate_table.py --subset 430)
#   S5  4292: 8 shards x {fg_head, mg_head, fg_const, mg_const, mg_const0} (PILOT=wgate4292, shard lists of
#       unc_full4292), one resubmit per failed shard; oracles stay on 430 (they are bounds, not methods)
#   S6  table job for 4292
#
# State: $ROOT/stage (S1..S6, DONE, FAILED), $ROOT/pipeline.log, $ROOT/heartbeat, $ROOT/<arm>.<stage>.jid.
# Re-running is safe: finished stages are skipped; scored arms are not resubmitted.
set -uo pipefail
WT=${WT:-/gpfs/home/koc/koc821022/maks_idea}
OUT=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval
ROOT=$OUT/wgate_pipeline; mkdir -p "$ROOT" "$OUT/logs"
LOG=$ROOT/pipeline.log
EP=$WT/eval_pipeline
POLL=${POLL:-300}
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
beat(){ date '+%F %T' > "$ROOT/heartbeat"; }
stage(){ cat "$ROOT/stage" 2>/dev/null || echo S1; }
setstage(){ echo "$1" > "$ROOT/stage"; log "stage -> $1"; }
jstate(){ sacct -j "$1" -n -X --format=State 2>/dev/null | head -1 | tr -d ' '; }
n_scored(){ find "$1/eval" -mindepth 2 -name eval_depth_pose_metrics.csv 2>/dev/null | wc -l; }
wait_job(){ # jid -> 0 on COMPLETED, 1 otherwise
  local j=$1 s
  while true; do beat; s=$(jstate "$j")
    case "$s" in COMPLETED) return 0;; FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) log "job $j ended $s"; return 1;; esac
    sleep "$POLL"; done
}
submit430(){ # arm -> jid
  (cd "$WT" && WT=$WT ARM=$1 PILOT=wgate430 sbatch --parsable --job-name="wg430_$1" eval_pipeline/maks_sweep430.sbatch 2>>"$LOG")
}
submit4292(){ # arm shard -> jid
  (cd "$WT" && WT=$WT ARM=$1 PILOT=wgate4292 SCENE_LIST_OVERRIDE=$OUT/unc_full4292/shards/shard_$2.txt \
     sbatch --parsable --time=03:00:00 --job-name="wg4292_$1_$2" eval_pipeline/maks_sweep430.sbatch 2>>"$LOG")
}
run_arm430(){ # arm: submit (unless scored) and wait; one retry
  local arm=$1; local f="$ROOT/$arm.430.jid"; local j att=1   # two statements: `local a=$1 b=$a` expands $a before assigning (set -u trap)
  if [ "$(n_scored "$OUT/wgate430/$arm")" -ge 427 ]; then log "430 $arm already scored"; return 0; fi
  while [ $att -le 2 ]; do
    if [ -s "$f" ] && [ "$att" = 1 ]; then j=$(cat "$f"); else j=$(submit430 "$arm"); echo "$j" > "$f"; log "430 $arm -> job $j (attempt $att)"; fi
    if wait_job "$j"; then log "430 $arm COMPLETED ($(n_scored "$OUT/wgate430/$arm")/430)"; return 0; fi
    att=$((att+1))
  done
  log "430 $arm FAILED twice"; return 1
}
table_job(){ # subset pilot_dir scene_list -> waits; returns job rc
  local j
  j=$(sbatch --parsable -A etur59 -p acc -q acc_ehpc -N1 --gres=gpu:1 -c 20 -t 00:40:00 -J wgate_table_$1 -o "$OUT/logs/wgate_table_$1_%j.out" \
      --wrap="source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything && cd $WT && export OMP_NUM_THREADS=8 && export PYTHONPATH=$WT/src:$WT/src/CUT3R:$WT/src/CUT3R/src && PATH=/apps/GPP/LATEX/20240430/bin/x86_64-linux:\$PATH python eval_pipeline/wgate_table.py --subset $1 --pilot_dir $2 --scene_list $3 --ref g7ema=$OUT/augfull_cg_g7ema/eval='Conf-Gate frame gate (g7ema), joint state+mem' --ref tok_al=$OUT/unc_full4292/tok_al_q50_g50/eval='Aligned token gate, self-view conf'" 2>>"$LOG")
  log "table $1 -> job $j"; wait_job "$j"
}

log "supervisor start pid=$$ host=$(hostname) stage=$(stage)"
while true; do
  beat
  case "$(stage)" in
    S1)
      FG=${FG_JID:-$(grep " fg_head job=" "$OUT/wgate430.jids" | tail -n1 | sed 's/.*job=\([0-9]*\).*/\1/')}
      MG=${MG_JID:-$(grep " mg_head job=" "$OUT/wgate430.jids" | tail -n1 | sed 's/.*job=\([0-9]*\).*/\1/')}
      log "S1 waiting for head arms fg_head=$FG mg_head=$MG"
      ok=1; for j in $FG $MG; do wait_job "$j" || ok=0; done
      if [ $ok = 0 ]; then
        # the head arms depend on the training job; if they failed because it did, stop here loudly
        log "S1 a head arm failed; check $OUT/logs and $OUT/wgate430/{fg_head,mg_head}/worker_g*.log"; setstage FAILED; exit 1
      fi
      for a in fg_head mg_head; do log "430 $a scored $(n_scored "$OUT/wgate430/$a")/430"; done
      setstage S2 ;;
    S2)
      # NOTE: no `conda activate` here -- under `set -u` conda's activate scripts abort on unset variables
      # (that is why the first run died with rc=1 and no output file); call the env's python directly.
      (cd "$WT" && export PYTHONPATH=$WT/src:$WT/src/CUT3R:$WT/src/CUT3R/src && OMP_NUM_THREADS=1 timeout 280 \
        /home/koc/koc821022/.conda/envs/cuteanything/bin/python eval_pipeline/wgate_make_controls.py \
        --pilot_dir "$OUT/wgate430" --scene_list "$EP/maks_subset430.txt" --out_dir "$EP" >> "$ROOT/make_controls.out" 2>&1)
      rc=$?
      if [ $rc != 0 ] || [ ! -s "$EP/maks_arm_fg_const.json" ] || [ ! -s "$EP/maks_arm_mg_const.json" ]; then log "S2 make_controls rc=$rc; see $ROOT/make_controls.out"; setstage FAILED; exit 1; fi
      log "S2 controls: fg_const $(python3 -c "import json;print(json.load(open('$EP/maks_arm_fg_const.json'))['*'])") mg_const $(python3 -c "import json;print(json.load(open('$EP/maks_arm_mg_const.json'))['*'])")"
      setstage S3 ;;
    S3)
      ok=1; pids=()
      for a in fg_const mg_const fg_oracle mg_oracle; do run_arm430 "$a" & pids+=($!); done
      for p in "${pids[@]}"; do wait "$p" || ok=0; done
      [ $ok = 1 ] || { log "S3 a control arm failed twice"; setstage FAILED; exit 1; }
      setstage S4 ;;
    S4)
      table_job 430 "$OUT/wgate430" "$EP/maks_subset430.txt" || log "S4 table job failed (continuing; rebuild by hand)"
      setstage S5 ;;
    S5)
      for arm in fg_head mg_head fg_const mg_const mg_const0; do
        for sh in 0 1 2 3 4 5 6 7; do
          f="$ROOT/$arm.4292.$sh.jid"; [ -s "$f" ] && continue
          j=$(submit4292 "$arm" "$sh"); echo "$j" > "$f"; log "4292 $arm shard $sh -> job $j"
        done
      done
      # wait, resubmitting a failed shard once
      for arm in fg_head mg_head fg_const mg_const mg_const0; do
        for sh in 0 1 2 3 4 5 6 7; do
          f="$ROOT/$arm.4292.$sh.jid"; j=$(cat "$f")
          if ! wait_job "$j"; then
            if [ ! -f "$f.retry" ]; then j=$(submit4292 "$arm" "$sh"); echo "$j" > "$f"; touch "$f.retry"; log "4292 $arm shard $sh resubmitted -> $j"
              wait_job "$j" || log "4292 $arm shard $sh failed twice"
            fi
          fi
        done
        log "4292 $arm scored $(n_scored "$OUT/wgate4292/$arm")/4292"
      done
      setstage S6 ;;
    S6)
      table_job 4292 "$OUT/wgate4292" "$OUT/scene_list.txt" || log "S6 table job failed (rebuild by hand)"
      setstage DONE ;;
    DONE) log "all stages done; supervisor exiting"; exit 0 ;;
    FAILED) log "stage FAILED; fix and reset $ROOT/stage"; exit 1 ;;
  esac
done
