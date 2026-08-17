#!/bin/bash
# Byte-compare predictions of two eval labels over a scene list.
#
#   bash eval_pipeline/cg_bytecheck.sh <LABEL_A> <LABEL_B> <SCENE_LIST> [N_DEPTH_SCENES]
#
# Compares camera/*.npz for EVERY scene in the list and depth/*.npy for the
# first N_DEPTH_SCENES (default 5). Exit 0 = byte-identical, 1 = any diff or
# missing file. Same contract as noise_oracle_bytecheck.sh: identity certifies
# an OFF/no-op hook; a gated arm is EXPECTED to fail this against clean.
set -o pipefail
OUT=${OUT_ROOT:-/gpfs/scratch/etur59/koc821022/outputs}/cut3r_eval
A=${1:?usage: cg_bytecheck.sh <LABEL_A> <LABEL_B> <SCENE_LIST> [N_DEPTH]}
B=${2:?}
LIST=${3:?}
NDEPTH=${4:-5}

ndiff=0; nmiss=0; nsame=0; i=0
while read -r scene; do
  [ -n "$scene" ] || continue
  i=$((i+1))
  for f in "$OUT/$A/preds/$scene/camera"/*.npz; do
    [ -e "$f" ] || { echo "MISS $A $scene camera"; nmiss=$((nmiss+1)); continue; }
    g="$OUT/$B/preds/$scene/camera/$(basename "$f")"
    if [ ! -e "$g" ]; then echo "MISS $B $scene $(basename "$f")"; nmiss=$((nmiss+1))
    elif cmp -s "$f" "$g"; then nsame=$((nsame+1))
    else echo "DIFF $scene camera/$(basename "$f")"; ndiff=$((ndiff+1)); fi
  done
  if [ "$i" -le "$NDEPTH" ]; then
    for f in "$OUT/$A/preds/$scene/depth"/*.npy; do
      [ -e "$f" ] || continue
      g="$OUT/$B/preds/$scene/depth/$(basename "$f")"
      if [ ! -e "$g" ]; then echo "MISS $B $scene depth/$(basename "$f")"; nmiss=$((nmiss+1))
      elif cmp -s "$f" "$g"; then nsame=$((nsame+1))
      else echo "DIFF $scene depth/$(basename "$f")"; ndiff=$((ndiff+1)); fi
    done
  fi
done < "$LIST"
echo "SUMMARY same=$nsame diff=$ndiff missing=$nmiss"
[ "$ndiff" -eq 0 ] && [ "$nmiss" -eq 0 ]
