#!/bin/bash
# sigma=0 byte-identity check: diag_noise_sig0check (hook enabled, sigma=0)
# must produce byte-identical predictions to diag_noise_clean (hook disabled).
# Full camera npz sweep (the pose-affected artifact) + depth spot-check on the
# first 20 scenes. Any diff blocks the curve readout (rung-0 precedent).
set -uo pipefail
OUT_ROOT=${OUT_ROOT:-/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval}
A=$OUT_ROOT/diag_noise_clean/preds
B=$OUT_ROOT/diag_noise_sig0check/preds
SUBSET=${SUBSET:-eval_pipeline/noise_oracle_subset_430.txt}
bad=0; n=0; ndepth=0
i=0
while IFS= read -r s; do
  [ -n "$s" ] || continue
  i=$((i+1))
  for f in "$A/$s"/camera/*.npz; do
    [ -f "$f" ] || continue
    g="$B/$s/camera/$(basename "$f")"
    n=$((n+1))
    cmp -s "$f" "$g" || { echo "DIFF: $s/camera/$(basename "$f")"; bad=$((bad+1)); }
  done
  if [ "$i" -le 20 ]; then
    for f in "$A/$s"/depth/*; do
      [ -f "$f" ] || continue
      g="$B/$s/depth/$(basename "$f")"
      ndepth=$((ndepth+1))
      cmp -s "$f" "$g" || { echo "DIFF-DEPTH: $s/depth/$(basename "$f")"; bad=$((bad+1)); }
    done
  fi
done < "$SUBSET"
echo "compared: $n camera files + $ndepth depth files, mismatches: $bad"
[ "$bad" -eq 0 ] && echo "BYTE-IDENTITY: PASS" || { echo "BYTE-IDENTITY: FAIL"; exit 1; }
