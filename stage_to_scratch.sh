#!/bin/bash
#SBATCH --job-name=stage_pw
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4   # batch equivalent of your l40s4 alias: claims a full l40s node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=57           # all 48 CPUs on the l40s node (CPUTot=48)   # was 48; MN5 derives mem from cores (8G/core) -> 456G ~= old --mem=450G
#SBATCH --time=24:00:00
# NOTE: the 4 GPUs sit idle during the copy (it needs none) — the deliberate tradeoff for
# grabbing one full dedicated node so all 48 CPUs + the node's NIC are yours for the transfer.
#SBATCH --output=logs/stage_%A.out
#SBATCH --error=logs/stage_%A.err

# Copy the pointworld droid store from NFS /frozen -> BeeGFS /scratch.
# COPY (never deletes source) + atomic rsync writes => safe to scancel at any time.
# RESUMABLE: a manifest of completed scenes lets a re-run skip finished work; rsync -a also
# re-verifies/repairs any scene that was mid-flight when the previous run was killed.
# Just re-run `sbatch stage_to_scratch.sh` to resume.

set -uo pipefail
mkdir -p logs

SRCROOT=/frozen/avg/bora_data/droid_datasets/pointworld_droid_wrist_all/dl3dv_multi
DSTROOT=/scratch/bdursun25/pointworld_droid_wrist_all/dl3dv_multi
SRC="$SRCROOT/wrist"
DST="$DSTROOT/wrist"
PAR=48
MANIFEST="$DSTROOT/.copied_scenes.txt"   # scenes confirmed fully synced (one per line)
WORKLIST="$DSTROOT/.worklist.txt"

mkdir -p "$DST"
[ -d "$SRCROOT/splits" ] && rsync -a "$SRCROOT/splits" "$DSTROOT/" || true
touch "$MANIFEST"

# Canonical target = union of scenes present in frozen and already in dst (robust even if the
# earlier mv had pre-moved some scenes out of frozen).
TOTAL=$(cat <(ls "$SRC" 2>/dev/null) <(ls "$DST" 2>/dev/null) | sort -u | wc -l)

# Worklist = scenes still in frozen that aren't already recorded done.
comm -23 <(ls "$SRC" 2>/dev/null | sort -u) <(sort -u "$MANIFEST") > "$WORKLIST"
echo "[$(date '+%F %T')] target=$TOTAL  already-done(manifest)=$(wc -l < "$MANIFEST")  to-process=$(wc -l < "$WORKLIST")  streams=$PAR"

# ---------- background progress monitor: one tail-friendly line every 20s ----------
monitor() {
  local total=$1
  local HASH DOT; HASH=$(printf '%50s' '' | tr ' ' '#'); DOT=$(printf '%50s' '' | tr ' ' '.')
  local last_done last_t now_done now_t rate pct fill eta
  last_done=$(ls "$DST" 2>/dev/null | wc -l); last_t=$(date +%s)
  while :; do
    sleep 20
    now_done=$(ls "$DST" 2>/dev/null | wc -l); now_t=$(date +%s)
    rate=$(awk -v d="$((now_done-last_done))" -v dt="$((now_t-last_t))" 'BEGIN{printf "%.2f",(dt>0)?d/dt:0}')
    pct=$(( total>0 ? now_done*100/total : 0 )); (( pct>100 )) && pct=100
    fill=$(( pct/2 ))
    eta=$(awk -v r="$rate" -v rem="$((total-now_done))" 'BEGIN{if(r>0){s=int(rem/r);printf "%d:%02d:%02d",int(s/3600),int((s%3600)/60),s%60}else printf "??:??:??"}')
    printf '[%s] |%s%s| %d/%d (%d%%)  %s sc/s  ETA %s\n' \
      "$(date '+%T')" "${HASH:0:fill}" "${DOT:0:$((50-fill))}" "$now_done" "$total" "$pct" "$rate" "$eta"
    last_done=$now_done; last_t=$now_t
  done
}
monitor "$TOTAL" &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT

# ---------- the copy: one rsync per scene, PAR at a time, record on success ----------
copy_one() {
  local sc="$1"
  if [ -d "$SRC/$sc" ]; then
    # rsync -a is atomic per file and idempotent: repairs partials, skips finished files.
    rsync -a "$SRC/$sc" "$DST/" && printf '%s\n' "$sc" >> "$MANIFEST"
  else
    # not in frozen (e.g. pre-moved by the earlier mv) -> already in dst; record as done
    [ -d "$DST/$sc" ] && printf '%s\n' "$sc" >> "$MANIFEST"
  fi
}
export -f copy_one
export SRC DST MANIFEST

xargs -a "$WORKLIST" -P "$PAR" -I{} bash -c 'copy_one "$@"' _ {} || true

kill "$MON" 2>/dev/null || true
DONE=$(ls "$DST" 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] pass finished: dst scenes=$DONE / target=$TOTAL"
if [ "$DONE" -ge "$TOTAL" ]; then
  echo "[$(date '+%F %T')] COPY COMPLETE ✅"
else
  echo "[$(date '+%F %T')] INCOMPLETE — re-run 'sbatch stage_to_scratch.sh' to resume the remaining $((TOTAL-DONE))."
fi
