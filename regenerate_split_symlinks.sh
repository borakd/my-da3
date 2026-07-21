#!/bin/bash
# Regenerate the pointworld_droid train/test split symlink trees on a new machine.
#
# The canonical split definition lives WITH the dataset, in
#   <DATA>/splits/{train.txt,test.txt}   (one scene-dir name per line; see splits.json
#   for provenance: per-lab stratified 90/10, seed=42, over 42925 scenes).
# This script builds the symlink trees the training configs point their ROOT at:
#   <SPLITS>/{train,test}/dl3dv_multi/wrist/<scene> -> <DATA>/wrist/<scene>
#
# Usage:
#   ./regenerate_split_symlinks.sh <DATA> <SPLITS>
#
#   DATA    path to the dataset's dl3dv_multi dir (must contain wrist/ and splits/)
#           e.g. /some/path/pointworld_droid_wrist_all/dl3dv_multi
#   SPLITS  where to create the split trees
#           e.g. /some/path/scenes/pointworld_droid_splits
#
# After running, point the config's dataset ROOTs at:
#   <SPLITS>/train/dl3dv_multi   and   <SPLITS>/test/dl3dv_multi
# and remember the DL3DV scan cache is keyed on abspath(ROOT): first load on the new
# machine does a fresh walk — set DL3DV_CACHE_DIR to a scratch filesystem beforehand.
#
# Idempotent: ln -sfn replaces existing links, so it is safe to re-run.

set -euo pipefail

if [ $# -ne 2 ]; then
  sed -n '2,24p' "$0"   # print the header comment as usage
  exit 1
fi

DATA=$1
SPL=$2

[ -d "$DATA/wrist" ]  || { echo "ERROR: $DATA/wrist not found (DATA must be the dl3dv_multi dir)"; exit 1; }
[ -d "$DATA/splits" ] || { echo "ERROR: $DATA/splits not found (copy splits/ from the original store)"; exit 1; }

for split in train test; do
  list="$DATA/splits/$split.txt"
  [ -f "$list" ] || { echo "ERROR: $list not found"; exit 1; }
  W="$SPL/$split/dl3dv_multi/wrist"
  mkdir -p "$W"
  n=$(wc -l < "$list")
  echo "linking $split ($n scenes) -> $W ..."
  i=0
  while IFS= read -r sc; do
    [ -n "$sc" ] || continue
    ln -sfn "$DATA/wrist/$sc" "$W/$sc"
    i=$((i + 1))
    if [ $((i % 5000)) -eq 0 ]; then echo "  $i/$n"; fi
  done < "$list"
done

echo
echo "sanity check:"
echo "  train links: $(ls "$SPL/train/dl3dv_multi/wrist" | wc -l) (expect $(wc -l < "$DATA/splits/train.txt"))"
echo "  test  links: $(ls "$SPL/test/dl3dv_multi/wrist" | wc -l) (expect $(wc -l < "$DATA/splits/test.txt"))"
sample=$(head -1 "$DATA/splits/train.txt")
echo "  sample: $SPL/train/dl3dv_multi/wrist/$sample -> $(readlink "$SPL/train/dl3dv_multi/wrist/$sample")"
if [ ! -d "$SPL/train/dl3dv_multi/wrist/$sample" ]; then
  echo "  WARNING: sample symlink does not resolve to an existing directory — check DATA path"
fi
echo "done. Next: update the config ROOTs to $SPL/{train,test}/dl3dv_multi"
