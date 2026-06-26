#!/bin/bash
# Re-point the train/test split symlinks from /frozen to the /scratch copy.
# Run this ONLY AFTER stage_to_scratch.sh has completed AND the old training job is cancelled
# (the running job reads these symlinks; re-pointing them mid-run would break it).
# Instant: this only rewrites symlinks, it moves no data. Config ROOT does NOT change.

set -euo pipefail

DST=/scratch/bdursun25/pointworld_droid_wrist_all/dl3dv_multi
SPL=/scratch/bdursun25/cuteanything/scenes/pointworld_droid_splits

for split in train test; do
  W="$SPL/$split/dl3dv_multi/wrist"
  list="$DST/splits/$split.txt"
  echo "relinking $split ($(wc -l < "$list") scenes) -> $DST/wrist ..."
  while IFS= read -r sc; do
    [ -n "$sc" ] || continue
    ln -sfn "$DST/wrist/$sc" "$W/$sc"   # -f replace, -n don't deref existing symlink dir
  done < "$list"
done

echo "verifying a sample symlink now points to /scratch:"
sc=$(head -1 "$DST/splits/train.txt")
readlink "$SPL/train/dl3dv_multi/wrist/$sc"
echo "done. Next: invalidate the scan cache and resubmit (see notes)."
