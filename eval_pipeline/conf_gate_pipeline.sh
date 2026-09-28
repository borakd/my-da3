#!/bin/bash
# Disconnection-proof supervisor for the conf-branch (train-through-the-gate) campaign. The account has no CPU partition, so it runs on a
# LOGIN node detached from any session (setsid + nohup; its own CPU use is negligible, every heavy step is a Slurm
# job). All state lives on disk under $ROOT so it can be killed and restarted at any time (idempotent):
#
#   cd /gpfs/home/koc/koc821022/maks_idea && setsid nohup bash eval_pipeline/unc_gate_pipeline.sh \
#       >> /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/unc_gate_pipeline/supervisor.out 2>&1 < /dev/null &
#   status: tail /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/unc_gate_pipeline/pipeline.log ; cat .../heartbeat
#
# Stages per run (A/B/C = the three v1 training variants):
#   TRAIN  train_unc_gate.sbatch (4 GPUs, one node, full pass over the wrist train split); on FAILED/TIMEOUT/
#          NODE_FAIL/OOM/CANCELLED -> resubmit with RESUME=1 (max $MAX_ATTEMPTS); a RUNNING job whose log is silent
#          for > $STALL_MIN minutes is cancelled and resumed.
#   EVAL   the 4292-scene test set in 8 shards x 4 GPUs (maks_sweep430.sbatch) for two arms per run
#          (head-keyed aligned token gate q50/g50, without and with per-token EMA .7) + the self-conf comparator
#          tok_al_q50_g50 once; failed/incomplete shards are resubmitted.
#   TABLE  unc_gate_table.py (CPU job): paired vs augfull_lr1e5, LaTeX -> PDF -> PNG. Rebuilt whenever a run's
#          evaluation completes, so the table always shows every finished run.
set -uo pipefail
WT=${WT:-/gpfs/home/koc/koc821022/maks_idea}
OUT=${OUT:-/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval}
CKROOT=/gpfs/scratch/etur59/koc821022/checkpoints/unc_gate
ROOT=$OUT/conf_gate_pipeline; mkdir -p "$ROOT"
LOG=$ROOT/pipeline.log
RUNS=${RUNS:-"G1 G10"}
EPISODES=${EPISODES:-38633}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-6}
STALL_MIN=${STALL_MIN:-150}
POLL=${POLL:-300}
NSHARD=8
declare -A RUN_EXTRA=( [G1]="--gscale 1 --chunk 16" [G10]="--gscale 10 --chunk 16" )
declare -A RUN_NAME=( [G1]="cb_g1" [G10]="cb_g10" )
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
state(){ sacct -j "$1" --format=State -n -X 2>/dev/null | head -1 | awk '{print $1}'; }
is_done_state(){ case "$1" in COMPLETED|FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|CANCELLED*|PREEMPTED|BOOT_FAIL|DEADLINE) return 0;; *) return 1;; esac; }
submit_train(){ # run resume_flag -> job id
  local r=$1 resume=$2 name=${RUN_NAME[$r]}
  local jid
  jid=$(cd "$WT" && WT=$WT NAME=$name EPISODES=$EPISODES BS=1 EXTRA="${RUN_EXTRA[$r]} --eval_every 2000 --eval_episodes 16" RESUME=$resume \
        sbatch --parsable --time=2-12:00:00 --gres=gpu:4 --cpus-per-task=80 eval_pipeline/train_conf_gate.sbatch 2>>"$LOG")
  echo "$jid"
}
submit_eval_shard(){ # arm shard -> job id
  local arm=$1 sh=$2
  (cd "$WT" && WT=$WT ARM=$arm PILOT=unc_full4292 SCENE_LIST_OVERRIDE=$OUT/unc_full4292/shards/shard_$sh.txt \
     sbatch --parsable --time=03:00:00 --job-name=unc4292_${arm}_$sh eval_pipeline/maks_sweep430.sbatch 2>>"$LOG")
}
write_arm_json(){ # run -> eval_pipeline/maks_arm_tok_cb<run>_q50_g50[_soft05].json
  local r=$1 name=${RUN_NAME[$r]}
  python3 - "$WT" "$CKROOT/$name/conf_branch.pth" "$r" <<'PY'
import json, sys
wt, cb, r = sys.argv[1:4]
for tag, soft in (("", None), ("_soft05", 0.5)):
    d = {"frames": "all", "q": 0.5, "gmin": 0.5, "align": "patch", "conf_branch": cb}
    if soft: d["soft"] = soft
    json.dump({"*": {"token_gate": d}}, open(f"{wt}/eval_pipeline/maks_arm_tok_cb{r}_q50_g50{tag}.json", "w"), indent=1)
PY
}
arms_of(){ local r=$1; echo "tok_cb${r}_q50_g50 tok_cb${r}_q50_g50_soft05"; }
n_scored(){ find "$OUT/unc_full4292/$1/eval" -mindepth 2 -name eval_depth_pose_metrics.csv 2>/dev/null | wc -l; }
shard_scored(){ # arm shard -> count of scored scenes of that shard
  local arm=$1 sh=$2 n=0
  while read -r s; do [ -f "$OUT/unc_full4292/$arm/eval/$s/eval_depth_pose_metrics.csv" ] && n=$((n+1)); done < "$OUT/unc_full4292/shards/shard_$sh.txt"
  echo $n
}
shard_size(){ wc -l < "$OUT/unc_full4292/shards/shard_$1.txt"; }
submit_table(){
  # no CPU partition in this account: the table job takes one GPU slot on acc_ehpc for a few minutes
  (cd "$WT" && sbatch --parsable --account=etur59 --partition=acc --qos=acc_ehpc --gres=gpu:1 --cpus-per-task=20 --time=01:00:00 --job-name=unc_table \
     --output="$ROOT/table_%j.out" --error="$ROOT/table_%j.err" \
     --wrap="source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything && cd $WT && export OMP_NUM_THREADS=4 && PATH=/apps/GPP/LATEX/20240430/bin/x86_64-linux:\$PATH python eval_pipeline/unc_gate_table.py" 2>>"$LOG")
}
# ---------------------------------------------------------------- main loop
log "supervisor start pid=$$ host=$(hostname) runs=[$RUNS] episodes=$EPISODES"
# comparator arm (self-conf aligned gate) on 4292: submitted once
COMP=tok_al_q50_g50
while true; do
  echo "$(date '+%F %T')" > "$ROOT/heartbeat"
  all_done=1
  # ---- comparator shards
  for sh in $(seq 0 $((NSHARD-1))); do
    f=$ROOT/eval_${COMP}_$sh.jid
    if [ "$(shard_scored $COMP $sh)" -ge "$(shard_size $sh)" ]; then continue; fi
    all_done=0
    if [ ! -f "$f" ] || is_done_state "$(state "$(cat $f)")"; then
      [ -f "$f" ] && log "comparator shard $sh job $(cat $f) ended $(state "$(cat $f)") with $(shard_scored $COMP $sh)/$(shard_size $sh) scored -> resubmit"
      att=$(cat "$ROOT/eval_${COMP}_$sh.att" 2>/dev/null || echo 0)
      if [ "$att" -ge 4 ]; then log "comparator shard $sh: giving up after $att attempts"; continue; fi
      jid=$(submit_eval_shard $COMP $sh); echo "$jid" > "$f"; echo $((att+1)) > "$ROOT/eval_${COMP}_$sh.att"; log "comparator $COMP shard $sh -> job $jid"
    fi
  done
  # ---- runs
  for r in $RUNS; do
    name=${RUN_NAME[$r]}; stage=$(cat "$ROOT/$r.stage" 2>/dev/null || echo TRAIN)
    if [ "$stage" = TRAIN ]; then
      all_done=0
      f=$ROOT/$r.train.jid; att=$(cat "$ROOT/$r.train.att" 2>/dev/null || echo 0)
      if [ ! -f "$f" ]; then
        jid=$(submit_train $r ""); echo "$jid" > "$f"; echo 1 > "$ROOT/$r.train.att"; log "run $r ($name) TRAIN -> job $jid attempt 1"; continue
      fi
      jid=$(cat "$f"); st=$(state "$jid")
      if [ "$st" = COMPLETED ] && grep -q "\[conf_gate\] done" "$OUT/logs/slurm_conf_gate_$jid.out" 2>/dev/null; then
        log "run $r TRAIN job $jid COMPLETED"; echo EVAL > "$ROOT/$r.stage"; write_arm_json $r; continue
      fi
      if is_done_state "$st" || [ "$st" = COMPLETED ]; then
        log "run $r TRAIN job $jid ended with state $st (attempt $att)"
        if [ "$att" -ge "$MAX_ATTEMPTS" ]; then log "run $r: giving up after $att attempts"; echo FAILED > "$ROOT/$r.stage"; continue; fi
        jid=$(submit_train $r 1); echo "$jid" > "$f"; echo $((att+1)) > "$ROOT/$r.train.att"; log "run $r TRAIN resumed -> job $jid attempt $((att+1))"; continue
      fi
      if [ "$st" = RUNNING ]; then
        lf=$CKROOT/$name/log.txt
        if [ -f "$lf" ]; then
          age=$(( ( $(date +%s) - $(stat -c %Y "$lf") ) / 60 ))
          if [ "$age" -gt "$STALL_MIN" ]; then log "run $r job $jid stalled ${age} min -> scancel + resume"; scancel "$jid"; sleep 30; fi
        fi
      fi
    elif [ "$stage" = EVAL ]; then
      all_done=0; complete=1
      for arm in $(arms_of $r); do
        for sh in $(seq 0 $((NSHARD-1))); do
          f=$ROOT/eval_${arm}_$sh.jid
          if [ "$(shard_scored $arm $sh)" -ge "$(shard_size $sh)" ]; then continue; fi
          complete=0
          if [ ! -f "$f" ] || is_done_state "$(state "$(cat $f)")"; then
            att=$(cat "$ROOT/eval_${arm}_$sh.att" 2>/dev/null || echo 0)
            if [ "$att" -ge 4 ]; then continue; fi
            jid=$(submit_eval_shard $arm $sh); echo "$jid" > "$f"; echo $((att+1)) > "$ROOT/eval_${arm}_$sh.att"; log "run $r EVAL $arm shard $sh -> job $jid attempt $((att+1))"
          fi
        done
      done
      if [ "$complete" = 1 ]; then log "run $r EVAL complete ($(n_scored tok_cb${r}_q50_g50)/4292)"; echo TABLE > "$ROOT/$r.stage"; fi
    elif [ "$stage" = TABLE ]; then
      f=$ROOT/$r.table.jid
      if [ ! -f "$f" ]; then jid=$(submit_table); echo "$jid" > "$f"; log "run $r TABLE -> job $jid"; all_done=0
      else
        st=$(state "$(cat $f)")
        if [ "$st" = COMPLETED ]; then echo DONE > "$ROOT/$r.stage"; log "run $r DONE (table job $(cat $f))"
        elif is_done_state "$st"; then log "run $r table job $(cat $f) $st -> resubmit"; rm -f "$f"; all_done=0
        else all_done=0; fi
      fi
    fi
  done
  if [ "$all_done" = 1 ]; then log "all runs DONE; supervisor exiting"; break; fi
  sleep "$POLL"
done
