#!/bin/bash
# Variant-C (CONSEQUENCE-trained gate) supervisor -- WRITE_GATE_VARIANTS.md, "Variant C".
# Same shape as eval_pipeline/wgate_pipeline.sh. Runs DETACHED on a login node:
#
#   mkdir -p $OUT/wgate_cons_pipeline
#   setsid nohup bash eval_pipeline/wgate_cons_pipeline.sh \
#       >> /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/wgate_cons_pipeline/supervisor.out 2>&1 < /dev/null &
#
# Stages (state file $ROOT/stage; re-running is safe, finished stages are skipped):
#   W      wait for the four consequence-training jobs listed in $OUT/wgate_cons.jids, one per line
#          "NAME JOBID [ARM]" (comments with # and blank lines ignored). NAME is the trainer's --name
#          (normally the arm itself: cg1_head, cg2_head, cg1_head_lr5, cg2_head_lr5); the eval ARM
#          defaults to name_to_arm(NAME), which is the identity for those and also maps the older
#          cg1 -> cg1_head / cg1_lr5 -> cg1_head_lr5 spelling, and can be given explicitly in a
#          third field. SEVERAL lines with the same NAME = the resubmits, waited in file order and
#          the first COMPLETED one wins; a training job that fails (twice) is LOGGED AND SKIPPED,
#          not fatal -- its arms simply have no gate_head_best.pth and drop out of every later stage.
#   E430   one maks_sweep430.sbatch per head arm whose checkpoint exists (PILOT=wgate430), in parallel,
#          one resubmit on failure. The checkpoint an arm needs is the "head" path in its
#          eval_pipeline/maks_arm_<arm>.json, which is the consequence trainer's own default layout
#          $CKROOT/consequence/<ARM>/gate_head_best.pth (--out default, NAME = the arm name), so an
#          arm also scores standalone. resolve_ckpt only verifies+logs that file; if a training job
#          used a different --out/--name it symlinks the alternative layouts into the json's path,
#          but only when that path is itself under $CKROOT.
#   C430   wgate_make_controls.py --only const for the two dose-matched constants of the lr 1e-4 arms
#          (cg1_head -> maks_arm_cg1_const.json, cg2_head -> maks_arm_cg2_const.json, summary suffix
#          _cg), then their two arms on 430 (same submit/wait/retry path)
#   T430   table job (wgate_table.py --subset $SUBSET430, pilot $OUT/wgate430)
#   E4292  8 shards per surviving arm (head arms + the constants), PILOT=wgate4292, one resubmit per
#          failed shard
#   T4292  table job for the 4292 list
#
# State: $ROOT/{stage,pipeline.log,heartbeat,<arm>.430.jid,<arm>.4292.<shard>.jid,make_controls.out}.
# NOTE: no `conda activate` in this script -- under `set -u` conda's activate scripts abort on unset
# variables; the env's python is called directly ($PY). Inside an sbatch --wrap the job gets a fresh
# shell without set -u, so conda activate is fine there (as in wgate_pipeline.sh).
# Env: WT, POLL (300 s), JIDS, SUBSET430/SUBSET4292 (table names), ARMS, N430_OK (427),
#      WGATE_CONS_LIB=1 (source the file for its functions only -- used by the dry run; no stages run).
set -uo pipefail
WT=${WT:-/gpfs/home/koc/koc821022/using_ext_cams}
OUT=${OUT:-/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval}
CKROOT=${CKROOT:-/gpfs/scratch/etur59/koc821022/checkpoints/wgate}
ROOT=$OUT/wgate_cons_pipeline; mkdir -p "$ROOT" "$OUT/logs"
LOG=$ROOT/pipeline.log
EP=$WT/eval_pipeline
PY=${PY:-/home/koc/koc821022/.conda/envs/cuteanything/bin/python}
POLL=${POLL:-300}
JIDS=${JIDS:-$OUT/wgate_cons.jids}
LIST430=${LIST430:-$EP/maks_subset430.txt}
LIST4292=${LIST4292:-$OUT/scene_list.txt}
SUBSET430=${SUBSET430:-430cons}
SUBSET4292=${SUBSET4292:-4292cons}
N430_OK=${N430_OK:-427}
ARMS=${ARMS:-"cg1_head cg2_head cg1_head_lr5 cg2_head_lr5"}
CONST_ARMS=${CONST_ARMS:-"cg1_const cg2_const"}

log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
# same, but on stderr: for functions whose STDOUT is captured by a command substitution
# (live_arms / resolve_ckpt -- a log line on stdout would be read back as an arm name)
log2(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }
beat(){ date '+%F %T' > "$ROOT/heartbeat"; }
stage(){ cat "$ROOT/stage" 2>/dev/null || echo W; }
setstage(){ echo "$1" > "$ROOT/stage"; log "stage -> $1"; }
jstate(){ sacct -j "$1" -n -X --format=State 2>/dev/null | head -1 | tr -d ' '; }
n_scored(){ find "$1/eval" -mindepth 2 -name eval_depth_pose_metrics.csv 2>/dev/null | wc -l; }

# NAME (checkpoint dir) -> eval arm name: cg1 -> cg1_head, cg1_lr5 -> cg1_head_lr5, cg1_head -> itself.
name_to_arm(){
  local nm=$1
  case "$nm" in
    *_head|*_head_lr*) echo "$nm" ;;
    *_lr5) echo "${nm%_lr5}_head_lr5" ;;
    *) echo "${nm}_head" ;;
  esac
}
# eval arm name -> checkpoint dir: cg1_head -> cg1, cg1_head_lr5 -> cg1_lr5.
arm_to_name(){
  local arm=$1
  case "$arm" in
    *_head_lr5) echo "${arm%_head_lr5}_lr5" ;;
    *_head) echo "${arm%_head}" ;;
    *) echo "$arm" ;;
  esac
}
# the head checkpoint an arm json points at ("" when the json has no head key)
arm_head_path(){
  local cj=$EP/maks_arm_$1.json
  [ -f "$cj" ] || { echo ""; return 0; }
  "$PY" -c "import json,sys; c=json.load(open(sys.argv[1])).get('*',{}); print(((c.get('frame_gate') or {}).get('head') or (c.get('mem_gate') or {}).get('head') or ''))" "$cj" 2>/dev/null
}

wait_job(){ # jid -> 0 on COMPLETED, 1 otherwise (also 1 for an id neither sacct nor squeue knows)
  local j=$1
  local s=""
  local miss=0
  while true; do
    beat
    s=$(jstate "$j")
    case "$s" in
      COMPLETED) return 0 ;;
      FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED) log "job $j ended $s"; return 1 ;;
      "")
        if [ -z "$(squeue -h -j "$j" -o %i 2>/dev/null)" ]; then
          miss=$((miss+1))
          if [ $miss -ge 3 ]; then log "job $j unknown to sacct and squeue -- giving up on it"; return 1; fi
        else
          miss=0
        fi ;;
      *) miss=0 ;;
    esac
    sleep "$POLL"
  done
}

submit430(){ # arm -> jid on stdout
  (cd "$WT" && WT=$WT ARM=$1 PILOT=wgate430 SCENE_LIST_OVERRIDE=$LIST430 \
     sbatch --parsable --job-name="wg430_$1" eval_pipeline/maks_sweep430.sbatch 2>>"$LOG")
}
submit4292(){ # arm shard -> jid on stdout
  (cd "$WT" && WT=$WT ARM=$1 PILOT=wgate4292 SCENE_LIST_OVERRIDE=$OUT/unc_full4292/shards/shard_$2.txt \
     sbatch --parsable --time=03:00:00 --job-name="wg4292_$1_$2" eval_pipeline/maks_sweep430.sbatch 2>>"$LOG")
}
run_arm430(){ # arm: submit (unless already scored) and wait; one retry. 0 = scored, 1 = failed twice
  local arm=$1
  local f="$ROOT/$arm.430.jid"
  local j=""
  local att=1
  if [ ! -f "$EP/maks_arm_$arm.json" ]; then log "430 $arm: no maks_arm_$arm.json -- skipped"; return 1; fi
  if [ "$(n_scored "$OUT/wgate430/$arm")" -ge "$N430_OK" ]; then log "430 $arm already scored"; return 0; fi
  while [ $att -le 2 ]; do
    if [ -s "$f" ] && [ "$att" = 1 ]; then j=$(cat "$f"); else j=$(submit430 "$arm"); echo "$j" > "$f"; log "430 $arm -> job $j (attempt $att)"; fi
    if wait_job "$j"; then log "430 $arm COMPLETED ($(n_scored "$OUT/wgate430/$arm")/430)"; return 0; fi
    att=$((att+1))
  done
  log "430 $arm FAILED twice"; return 1
}
table_job(){ # subset pilot_dir scene_list -> waits for the job; rc of the job
  local j=""
  j=$(sbatch --parsable -A etur59 -p acc -q acc_ehpc -N1 --gres=gpu:1 -c 20 -t 00:40:00 -J wgate_table_$1 \
      -o "$OUT/logs/wgate_table_$1_%j.out" \
      --wrap="source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything && cd $WT && export OMP_NUM_THREADS=8 && export PYTHONPATH=$WT/src:$WT/src/CUT3R:$WT/src/CUT3R/src && PATH=/apps/GPP/LATEX/20240430/bin/x86_64-linux:\$PATH python eval_pipeline/wgate_table.py --subset $1 --pilot_dir $2 --scene_list $3" 2>>"$LOG")
  log "table $1 -> job $j"; wait_job "$j"
}
# The arm json's head path is the single source of truth for the worker. It now names the TRAINER's real
# output ($CKROOT/consequence/<ARM>/gate_head_best.pth, i.e. `--out` left at its default and NAME = the arm
# name), so in the normal case resolve_ckpt only has to CHECK that the file is there -- `ARM=cg1_head sbatch
# eval_pipeline/maks_sweep430.sbatch` works standalone, with no symlink. Linking is a FALLBACK for a training
# job launched with a different --out/--name layout, and it is deliberately fenced:
#   * the link is only ever created when the json's head path is itself under $CKROOT (so a test run with
#     CKROOT=<tmp> cannot write into the production checkpoint tree);
#   * an existing path is logged with its size/mtime (and link target) EVERY time, short-circuit included,
#     so the log always records which file an arm scored;
#   * a symlink whose target is outside $CKROOT, or which dangles, is REPLACED rather than accepted -- a
#     stale link from an earlier/smoke run must never silently feed the wrong checkpoint to an arm.
ck_desc(){ # path -> "size=.. mtime=.. [-> target]" (follows links; empty when the file is unreadable)
  local p=$1
  local d=""
  d=$(stat -Lc 'size=%s mtime=%y' "$p" 2>/dev/null)
  if [ -L "$p" ]; then d="$d -> $(readlink "$p" 2>/dev/null)"; fi
  echo "$d"
}
under_ckroot(){ case "$1" in "$CKROOT"/*) return 0 ;; *) return 1 ;; esac; }
resolve_ckpt(){ # arm -> 0 when the json's head path exists (possibly after linking), 1 otherwise
  local arm=$1
  local want=""
  want=$(arm_head_path "$arm")
  [ -n "$want" ] || return 1
  if ! under_ckroot "$want"; then
    # never link into a path we do not own; only accept what is already there, and say so loudly
    if [ -f "$want" ]; then
      log2 "resolve_ckpt $arm: head path $want is OUTSIDE CKROOT ($CKROOT) -- using the existing file [$(ck_desc "$want")]"
      return 0
    fi
    log2 "resolve_ckpt $arm: head path $want is OUTSIDE CKROOT ($CKROOT) and does not exist -- refusing to link there"
    return 1
  fi
  if [ -L "$want" ]; then
    local tgt=""
    tgt=$(readlink -f "$want" 2>/dev/null)
    if [ -f "$want" ] && [ -n "$tgt" ] && under_ckroot "$tgt"; then
      log2 "resolve_ckpt $arm: $want [$(ck_desc "$want")]"
      return 0
    fi
    log2 "resolve_ckpt $arm: replacing symlink $want (target '${tgt:-<dangling>}' is outside $CKROOT or missing)"
    rm -f "$want"
  elif [ -f "$want" ]; then
    log2 "resolve_ckpt $arm: $want [$(ck_desc "$want")]"
    return 0
  fi
  local nm=""
  nm=$(arm_to_name "$arm")
  local c=""
  for c in "$CKROOT/consequence/$arm/gate_head_best.pth" "$CKROOT/consequence/$nm/gate_head_best.pth" \
           "$CKROOT/$arm/gate_head_best.pth" "$CKROOT/$nm/gate_head_best.pth"; do
    [ "$c" = "$want" ] && continue
    if [ -f "$c" ]; then
      mkdir -p "$(dirname "$want")" && ln -sfn "$c" "$want" \
        && log2 "resolve_ckpt $arm: linked $want -> $c [$(ck_desc "$c")]"
      return 0
    fi
  done
  log2 "resolve_ckpt $arm: no checkpoint at $want (nor in the fallback layouts under $CKROOT)"
  return 1
}
# arms whose consequence-trained checkpoint exists (E430 / E4292 input)
live_arms(){
  local arm=""
  for arm in $ARMS; do
    if resolve_ckpt "$arm"; then echo "$arm"; fi
  done
}

if [ "${WGATE_CONS_LIB:-0}" = "1" ]; then return 0 2>/dev/null || exit 0; fi   # functions only (dry run)

log "supervisor start pid=$$ host=$(hostname) stage=$(stage) jids=$JIDS"
while true; do
  beat
  case "$(stage)" in
    W)
      if [ ! -s "$JIDS" ]; then
        log "W: $JIDS missing/empty -- not waiting for any training job (checkpoints are checked in E430)"
      else
        for nm in $(awk '$1 !~ /^#/ && NF >= 2 {print $1}' "$JIDS" | awk '!seen[$0]++'); do
          arm=$(awk -v n="$nm" '$1 == n && NF >= 3 {print $3}' "$JIDS" | tail -n1)
          [ -n "$arm" ] || arm=$(name_to_arm "$nm")
          good=0
          for j in $(awk -v n="$nm" '$1 == n && NF >= 2 {print $2}' "$JIDS"); do
            log "W waiting for training $nm (arm $arm) job $j"
            if wait_job "$j"; then good=1; log "W training $nm job $j COMPLETED"; break; fi
          done
          ck=$(arm_head_path "$arm"); [ -n "$ck" ] || ck=$CKROOT/consequence/$arm/gate_head_best.pth
          if [ $good = 0 ]; then
            log "W training $nm FAILED (all listed jobs) -- arm $arm will be skipped unless $ck exists"
          fi
          [ -f "$ck" ] && log "W $nm: $ck present" || log "W $nm: NO checkpoint at $ck"
        done
      fi
      setstage E430 ;;
    E430)
      arms=$(live_arms | tr "\n" " ")
      if [ -z "$arms" ]; then log "E430: no arm has a gate_head_best.pth -- nothing to evaluate"; setstage FAILED; exit 1; fi
      log "E430 arms: $arms"
      pids=()
      for a in $arms; do run_arm430 "$a" & pids+=($!); done
      for p in "${pids[@]}"; do wait "$p" || true; done      # a failed arm is logged, not fatal
      for a in $arms; do log "430 $a scored $(n_scored "$OUT/wgate430/$a")/430"; done
      setstage C430 ;;
    C430)
      (cd "$WT" && export PYTHONPATH=$WT/src:$WT/src/CUT3R:$WT/src/CUT3R/src && OMP_NUM_THREADS=1 \
        "$PY" eval_pipeline/wgate_make_controls.py --only const --pilot_dir "$OUT/wgate430" \
        --scene_list "$LIST430" --out_dir "$EP" --suffix _cg \
        --fg_arm cg1_head --mg_arm cg2_head --fg_const_name cg1_const --mg_const_name cg2_const \
        >> "$ROOT/make_controls.out" 2>&1)
      rc=$?
      log "C430 make_controls rc=$rc (log $ROOT/make_controls.out)"
      cok=""
      for a in $CONST_ARMS; do
        if [ -s "$EP/maks_arm_$a.json" ]; then cok="$cok $a"; else log "C430 $a: no maks_arm_$a.json (its head arm has no traces) -- skipped"; fi
      done
      if [ -n "${cok// /}" ]; then
        log "C430 constants:$cok"
        pids=()
        for a in $cok; do run_arm430 "$a" & pids+=($!); done
        for p in "${pids[@]}"; do wait "$p" || true; done
      fi
      setstage T430 ;;
    T430)
      table_job "$SUBSET430" "$OUT/wgate430" "$LIST430" || log "T430 table job failed (continuing; rebuild by hand)"
      setstage E4292 ;;
    E4292)
      arms=""
      for a in $(live_arms) $CONST_ARMS; do
        [ -f "$EP/maks_arm_$a.json" ] || continue
        if [ "$(n_scored "$OUT/wgate430/$a")" -ge "$N430_OK" ]; then arms="$arms $a"; else log "E4292 $a: not scored on 430 -- skipped"; fi
      done
      if [ -z "${arms// /}" ]; then log "E4292: no arm scored on 430"; setstage FAILED; exit 1; fi
      log "E4292 arms:$arms"
      for arm in $arms; do
        for sh in 0 1 2 3 4 5 6 7; do
          f="$ROOT/$arm.4292.$sh.jid"; [ -s "$f" ] && continue
          j=$(submit4292 "$arm" "$sh"); echo "$j" > "$f"; log "4292 $arm shard $sh -> job $j"
        done
      done
      for arm in $arms; do
        for sh in 0 1 2 3 4 5 6 7; do
          f="$ROOT/$arm.4292.$sh.jid"; j=$(cat "$f" 2>/dev/null)
          if [ -n "$j" ] && ! wait_job "$j"; then
            if [ ! -f "$f.retry" ]; then
              j=$(submit4292 "$arm" "$sh"); echo "$j" > "$f"; touch "$f.retry"; log "4292 $arm shard $sh resubmitted -> $j"
              wait_job "$j" || log "4292 $arm shard $sh failed twice"
            fi
          fi
        done
        log "4292 $arm scored $(n_scored "$OUT/wgate4292/$arm")/4292"
      done
      setstage T4292 ;;
    T4292)
      table_job "$SUBSET4292" "$OUT/wgate4292" "$LIST4292" || log "T4292 table job failed (rebuild by hand)"
      setstage DONE ;;
    DONE) log "all stages done; supervisor exiting"; exit 0 ;;
    FAILED) log "stage FAILED; fix and reset $ROOT/stage"; exit 1 ;;
    *) log "unknown stage '$(stage)' in $ROOT/stage"; exit 1 ;;
  esac
done
