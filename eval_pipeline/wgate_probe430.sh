#!/bin/bash
# wgate_probe430.sh -- submit the write-gate probe arms on the 430-scene subset
# (WRITE_GATE_VARIANTS.md, "Leverage probe" + the two head arms). One
# maks_sweep430.sbatch job per arm (4 GPUs, acc_ehpc, ~1-2 h each), PILOT=wgate430,
# outputs under $OUT/wgate430/<arm>/{eval,control.json,worker_g*.log}.
#
#   V2 leverage probes (no head needed):  mg_const0   mg_freeze8   mg_const50
#   head arms (need checkpoints/wgate/{frame_conf_head,pose_sigma_head}.pth,
#              written by train_wgate_heads.py):  fg_head   mg_head
#   RELATIVE-target head arms (checkpoints/wgate/{frame_conf_head,pose_sigma_head}_rel.pth from
#              train_wgate_heads.py --target rel / TARGET=rel):  fg_head_rel   mg_head_rel
#   control arms generated afterwards (wgate_make_controls.py [--suffix _rel]):
#              fg_const fg_oracle mg_const mg_oracle [ + the _rel four ]
#   The script is generic: any ARM with an eval_pipeline/maks_arm_<ARM>.json works; a head arm's
#   checkpoint path is read from the json (frame_gate.head / mem_gate.head).
#
# Usage (from the worktree, on a login node -- sbatch only, nothing runs here):
#   bash eval_pipeline/wgate_probe430.sh                       # all five arms
#   ARMS="mg_const0 mg_freeze8 mg_const50" bash eval_pipeline/wgate_probe430.sh
#   ARMS="fg_head_rel mg_head_rel" bash eval_pipeline/wgate_probe430.sh         # the rel heads
# Env: WT (worktree), PILOT (wgate430), SCENE_LIST_OVERRIDE (eval_pipeline/maks_subset430.txt),
#      FORCE=1 (submit a head arm even if its .pth is missing -- the job would fail on load).
# Idempotent: an arm whose eval tree already holds every scene of the list is not resubmitted,
# and an arm whose job (name wg<pilot-suffix>_<arm>) is still queued/running is not resubmitted
# either (two jobs of the same arm shard identically and race on the same preds/eval tree);
# a partially scored arm with no live job is resubmitted and the worker skips its scored scenes.
# Afterwards: python eval_pipeline/wgate_make_controls.py   (constants + oracles from the head arms)
#             python eval_pipeline/wgate_table.py --scene_list eval_pipeline/maks_subset430.txt
set -uo pipefail
WT=${WT:-/gpfs/home/koc/koc821022/maks_idea}
OUT=${OUT:-/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval}
PILOT=${PILOT:-wgate430}
LIST=${SCENE_LIST_OVERRIDE:-$WT/eval_pipeline/maks_subset430.txt}
ARMS=${ARMS:-"mg_const0 mg_freeze8 mg_const50 fg_head mg_head"}
JIDS=$OUT/$PILOT.jids
[ -f "$WT/eval_pipeline/maks_sweep430.sbatch" ] || { echo "ERROR: not a worktree: $WT" >&2; exit 1; }
[ -s "$LIST" ] || { echo "ERROR: scene list missing/empty: $LIST" >&2; exit 1; }
mkdir -p "$OUT/logs" "$OUT/$PILOT"
NLIST=$(wc -l < "$LIST")
for ARM in $ARMS; do
  CJ=$WT/eval_pipeline/maks_arm_$ARM.json
  [ -f "$CJ" ] || { echo "SKIP $ARM: control json missing: $CJ"; continue; }
  # head arms name their checkpoint in the json; do not burn a 4-GPU job on a load failure
  head=$(python3 -c "import json,sys; c=json.load(open(sys.argv[1])).get('*',{}); print(((c.get('frame_gate') or {}).get('head') or (c.get('mem_gate') or {}).get('head') or ''))" "$CJ")
  if [ -n "$head" ] && [ ! -f "$head" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "SKIP $ARM: head checkpoint not found: $head   (train_wgate_heads.py first, or FORCE=1)"; continue
  fi
  done_n=$(find "$OUT/$PILOT/$ARM/eval" -mindepth 2 -name eval_depth_pose_metrics.csv 2>/dev/null | wc -l)
  if [ "$done_n" -ge "$NLIST" ]; then echo "DONE $ARM: $done_n/$NLIST scenes scored -- not resubmitting"; continue; fi
  JN="wg${PILOT#wgate}_$ARM"   # wgate430 -> wg430_<arm> (unchanged), wgate4292 -> wg4292_<arm>
  live=$(squeue -h -u "$USER" -n "$JN" -o %i 2>/dev/null | tr '\n' ' ')
  if [ -n "${live// /}" ]; then echo "QUEUED/RUNNING $ARM (job $live) -- not resubmitting"; continue; fi
  jid=$(cd "$WT" && WT=$WT ARM=$ARM PILOT=$PILOT SCENE_LIST_OVERRIDE=$LIST \
        sbatch --parsable --job-name="$JN" eval_pipeline/maks_sweep430.sbatch) || { echo "FAILED to submit $ARM"; continue; }
  echo "$(date '+%F %T') $PILOT $ARM job=$jid scored_before=$done_n/$NLIST" | tee -a "$JIDS"
done
echo "queue:  squeue -u \$USER -o '%.10i %.18j %.9T %.10M %R'"
echo "logs:   $OUT/logs/slurm_maks_sw430_<job>.out ; $OUT/$PILOT/<arm>/worker_g*.log"
