#!/bin/bash
# Download metadata + exterior MP4s for the 13 smoke episodes (login node, curl only).
set -u
CACHE=/gpfs/scratch/etur59/koc821022/vggt_cache
RAW=$CACHE/raw
BASE=https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1
RAIL_LOCAL=/home/koc/koc821022/vggt-omega/outputs/droid_raw/RAIL+80edfcb1+2023-07-14-14h-28m-45s
python3 - <<'PY' > $CACHE/raw/_dl_list.txt
import json
d=json.load(open('/gpfs/scratch/etur59/koc821022/vggt_cache/smoke13_raw_paths.json'))
for ep,v in d.items():
    print(ep, v['path'], v['ext1'], v['ext2'])
PY
while read ep path e1 e2; do
  out=$RAW/$ep; mkdir -p $out/frames $out/recordings/MP4
  meta=$out/metadata_$ep.json
  if [ "$ep" = "RAIL+80edfcb1+2023-07-14-14h-28m-45s" ]; then
    [ -e $meta ] || cp $RAIL_LOCAL/metadata_$ep.json $meta
    for s in $e1 $e2; do [ -e $out/recordings/MP4/$s.mp4 ] || cp $RAIL_LOCAL/recordings/MP4/$s.mp4 $out/recordings/MP4/$s.mp4; done
    echo "$ep: copied local"; continue
  fi
  # curl each file as a separate short process
  if [ ! -s $meta ]; then
    curl -sfL --retry 3 -o $meta "$BASE/$path/metadata_$ep.json" || echo "FAIL meta $ep"
  fi
  for s in $e1 $e2; do
    f=$out/recordings/MP4/$s.mp4
    if [ ! -s $f ]; then
      curl -sfL --retry 3 -o $f "$BASE/$path/recordings/MP4/$s.mp4" || echo "FAIL mp4 $ep $s"
    fi
  done
  echo "$ep: $(du -sh $out | cut -f1)"
done < $CACHE/raw/_dl_list.txt
