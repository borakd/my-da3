#!/bin/bash
# Repeat-eval campaign (revised scope 2026-08-13): rerun the requested captain
# GRU rows under fresh labels so the GRU-refined pose (eval_gru) is scored
# alongside the regular CUT3R poses, WITHOUT touching any original result tree.
#
# Rows (user-requested; "_full" = the wandb identity of a killed+resumed arm —
# the on-disk checkpoint-final.pth of the plain-named run dir IS the end of
# that full training):
#   gru_a4g3f1np_rpt  <- captain_gru_v3_a4_g3_f1_noproj_finetune
#   gru_a4g3f0np_rpt  <- captain_gru_v3_a4_g3_f0_noproj_finetune      (_full)
#   gru_a4g3f1_rpt    <- captain_gru_v3_a4_g3_f1_finetune             (_full)
#   gru_a4g3otr       <- captain_gru_v3_a4_g3_oracle_finetune_32gpu
#   (captain_gru_v3_a4_g3_finetune (_full) was already rerun today as
#    gru_a4g3_rpt — complete with eval + eval_gru; not resubmitted.)
#
# gru_a4g3otr is an oracle-TRAINED checkpoint evaluated HONESTLY: normal
# prev_pred_gru closed loop, ORACLE stays off (the worker's warning that the
# ckpt was oracle-trained is expected). No GT enters the eval.
#
# Two labels at a time (2 x 8 nodes x 4 GPUs = 64 GPUs). Every label goes
# through arm_eval_for_run.sh: preflight re-derives the conditioning arm from
# the run's own .hydra config and cross-checks ckpt["args"] + weight shapes at
# submit time, so the eval wiring cannot drift from the checkpoint.
#
# Disk bounded: each campaign label's preds/preds_gru (~332 GB) are DELETED as
# soon as its eval CSVs are complete (CSVs and original labels never touched).
#
# Stall recovery: when a label's jobs have all left the queue but coverage is
# short, stale claims (claim dirs whose scene lacks its required CSVs — safe
# because no worker is alive) are cleared and the label's 8 nodes resubmitted,
# at most twice per label.
set -o pipefail

WT=${WT:-/gpfs/home/koc/koc821022/vggt_features}
export OUT_ROOT=/gpfs/scratch/etur59/koc821022/outputs
OUT=$OUT_ROOT/cut3r_eval
CR=/gpfs/scratch/etur59/koc821022/checkpoints_projects/captain_cut3r_finetune_aug_full
NSCENES=4292

# Falsifier env vars silently contaminate a scoring run and nothing logs it.
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE \
      POSE_GRU_HIDDEN_ZERO POSE_GRU_HIDDEN_SHUFFLE \
      POSE_GRU_IMG_FEAT_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_FORCE_ITERS

say () { echo "[rpt-driver $(date +%m-%d\ %H:%M)] $*"; }

# label|run_dir|mode — every row is a prev_pred_gru arm scored honestly.
# mode=arm: preflight-gated via arm_eval_for_run.sh.
# mode=direct: gru_a4g3otr only. preflight HARD-REFUSES oracle-TRAINED ckpts
# (FAIL pose_gru_oracle == off) because it assumes they can only be scored as
# GT-injection diagnostics; this row is a deliberate honest oracle-OFF eval
# (out-of-distribution probe). Its ckpt/arm/lever agreement was verified by
# the certification pass (ckpt args == .hydra == prev_pred_gru, epoch 50/50),
# and the worker prints the oracle-trained-but-oracle-off WARNING in every log.
ROWS=(
  "gru_a4g3f1np_rpt|$CR/captain_gru_v3_a4_g3_f1_noproj_finetune|arm"
  "gru_a4g3f0np_rpt|$CR/captain_gru_v3_a4_g3_f0_noproj_finetune|arm"
  "gru_a4g3f1_rpt|$CR/captain_gru_v3_a4_g3_f1_finetune|arm"
  "gru_a4g3otr|$CR/captain_gru_v3_a4_g3_oracle_finetune_32gpu|direct"
)
# The only labels this script is ever allowed to delete preds for.
CLEANUP_OK="gru_a4g3f1np_rpt gru_a4g3f0np_rpt gru_a4g3f1_rpt gru_a4g3otr gru_a4g3_rpt"

headroom_gb () {
  bsc_quota 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
    | awk '/gpfs_scratch/ {u=$3; uq=$4; q=$5; qq=$6;
           if (uq=="TB") u*=1024; if (qq=="TB") q*=1024; printf "%d", q-u}'
}

count_csvs () { find "$1" -name eval_depth_pose_metrics.csv 2>/dev/null | wc -l; }

label_done () {  # $1=label (all rows are gru arms: both trees required)
  [ "$(count_csvs "$OUT/$1/eval")" -ge "$NSCENES" ] || return 1
  [ "$(count_csvs "$OUT/$1/eval_gru")" -ge "$NSCENES" ] || return 1
  return 0
}

jobs_in_queue () { squeue -u "$USER" -h -o %j 2>/dev/null | grep -c "^ev_$1"; }

clear_stale_claims () {  # $1=label (only called when no jobs alive)
  local cdir=$OUT/$1/claims n=0 scene base
  [ -d "$cdir" ] || return 0
  for c in "$cdir"/*/; do
    [ -d "$c" ] || continue
    base=$(basename "$c"); scene=${base%.gru}
    if [ "$base" != "$scene" ]; then
      [ -s "$OUT/$1/eval_gru/$scene/eval_depth_pose_metrics.csv" ] && continue
    else
      [ -s "$OUT/$1/eval/$scene/eval_depth_pose_metrics.csv" ] && \
      [ -s "$OUT/$1/eval_gru/$scene/eval_depth_pose_metrics.csv" ] && continue
    fi
    rmdir "$c" 2>/dev/null && n=$((n+1))
  done
  say "$1: cleared $n stale claims"
}

submit_label () {  # $1=label $2=run_dir $3=mode  (arm: preflight-gated; direct: see ROWS)
  local rc
  if [ "$3" = "direct" ]; then
    [ -f "$2/checkpoint-final.pth" ] || { say "FATAL: missing $2/checkpoint-final.pth"; return 1; }
    resubmit_label "$1" "$2"
    return 0
  fi
  for try in $(seq 1 36); do
    ( cd "$WT" && bash eval_pipeline/arm_eval_for_run.sh "$2" "$1" ) \
      >> "$OUT/logs/rpt_driver_arm_${1}.log" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] && { say "$1 armed+submitted"; return 0; }
    [ "$rc" -eq 75 ] || { say "FATAL: arm failed rc=$rc for $1 (see rpt_driver_arm_${1}.log)"; return 1; }
    say "$1 arm tempfail (transient), retry $try/36 in 10 min"
    sleep 600
  done
  say "FATAL: $1 arm still tempfailing after 6h"; return 1
}

resubmit_label () {  # $1=label $2=run_dir  (resume; claims already cleared)
  local CKPT=$2/checkpoint-final.pth
  for B in 0 4 8 12 16 20 24 28; do
    sbatch \
      --job-name="ev_${1}_b${B}" \
      --output="$OUT/logs/slurm_${1}_b${B}_%j.out" \
      --error="$OUT/logs/slurm_${1}_b${B}_%j.err" \
      --export=ALL,OUT_ROOT="$OUT_ROOT",LABEL="$1",CONDITIONING=prev_pred_gru,ORACLE=off,CKPT="$CKPT",BASE_SHARD="$B",NUM_SHARDS=32,LIMIT=0 \
      "$WT/eval_pipeline/run_captain_ray_eval_node.sh" || say "WARN: resubmit sbatch failed for $1 b$B"
  done
  say "$1 resubmitted (resume)"
}

cleanup_preds () {  # $1=label — allowlist-guarded
  case " $CLEANUP_OK " in *" $1 "*) ;; *) say "REFUSING preds cleanup of $1"; return ;; esac
  local sz
  sz=$(du -sh "$OUT/$1/preds" 2>/dev/null | cut -f1)
  rm -rf "$OUT/$1/preds" "$OUT/$1/preds_gru"
  say "$1: preds cleaned (${sz:-0})"
}

mkdir -p "$OUT/logs"
say "campaign start: ${#ROWS[@]} rows, 2 at a time"

# gru_a4g3_rpt finished earlier today — reclaim its ~332 GB now.
if label_done gru_a4g3_rpt; then cleanup_preds gru_a4g3_rpt; fi

INCOMPLETE=()
i=0
while [ $i -lt ${#ROWS[@]} ]; do
  wave=("${ROWS[@]:$i:2}"); i=$((i+2))
  for try in $(seq 1 36); do
    free=$(headroom_gb)
    [ -z "$free" ] && { say "WARN: cannot parse bsc_quota, proceeding"; break; }
    [ "$free" -ge 900 ] && break
    say "disk gate: ${free} GB free (< 900), waiting 10 min ($try/36)"
    sleep 600
    [ "$try" -eq 36 ] && { say "FATAL: disk never freed"; exit 1; }
  done
  labels=(); runs=(); resubs=(); state=()
  for row in "${wave[@]}"; do
    IFS='|' read -r lbl run mode <<< "$row"
    say "=== wave: submitting $lbl (mode=$mode)"
    submit_label "$lbl" "$run" "$mode" || exit 1
    labels+=("$lbl"); runs+=("$run"); resubs+=(0); state+=("running")
  done
  for tick in $(seq 1 120); do   # poll 5 min, cap 10 h per wave
    sleep 300
    alldone=1
    for k in "${!labels[@]}"; do
      lbl=${labels[$k]}
      [ "${state[$k]}" = "givenup" ] && continue
      if label_done "$lbl"; then state[$k]="done"; continue; fi
      alldone=0
      nq=$(jobs_in_queue "$lbl")
      if [ "$nq" -eq 0 ]; then
        if [ "${resubs[$k]}" -lt 2 ]; then
          say "$lbl: queue drained but incomplete (reg=$(count_csvs "$OUT/$lbl/eval") gru=$(count_csvs "$OUT/$lbl/eval_gru")) — clearing stale claims, resubmitting"
          clear_stale_claims "$lbl"
          resubmit_label "$lbl" "${runs[$k]}"
          resubs[$k]=$((resubs[$k]+1))
        else
          say "$lbl: still incomplete after 2 resubmits — marking INCOMPLETE"
          INCOMPLETE+=("$lbl"); state[$k]="givenup"
        fi
      fi
    done
    [ "$alldone" -eq 1 ] && break
    [ $((tick % 6)) -eq 0 ] && for k in "${!labels[@]}"; do
      lbl=${labels[$k]}
      say "  $lbl: jobs=$(jobs_in_queue "$lbl") reg=$(count_csvs "$OUT/$lbl/eval") gru=$(count_csvs "$OUT/$lbl/eval_gru")"
    done
  done
  for k in "${!labels[@]}"; do
    label_done "${labels[$k]}" && cleanup_preds "${labels[$k]}"
  done
done

say "all waves finished. incomplete: ${INCOMPLETE[*]:-none}"
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
python "$WT/eval_pipeline/aggregate_results.py" \
  --out_root "$OUT" --scene_list "$OUT/scene_list.txt" \
  --labels augfull_lr1e5 gtray_lr1e5 prevpred_lr1e5 gru_a4g3 gru_a4g3f1 \
           gru_a4g3r8 gru_a4g3f1r8 gru_a4g3f0np gru_a4g3f1np gru_a4g3f1rr8 \
           gru_a4g3f1dr8 gru_a4g3f1cr8 gru_a4g3f1rr8np gru_a4g3f1dr8np \
           gru_a4g3oraclegt \
           gru_a4g3_rpt gru_a4g3f1np_rpt gru_a4g3f0np_rpt gru_a4g3f1_rpt \
           gru_a4g3otr \
  2>&1 | tail -n 40
say "campaign COMPLETE (incomplete labels: ${INCOMPLETE[*]:-none})"
