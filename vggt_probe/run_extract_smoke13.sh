#!/bin/bash
# One python process per episode (login-node 300 s CPU cap).
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh; conda activate vggt-omega; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENCV_FFMPEG_THREADS=1
CACHE=/gpfs/scratch/etur59/koc821022/vggt_cache
LOG=$CACHE/raw/logs; mkdir -p $LOG
for ep in $(python -c "import json;print(' '.join(json.load(open('$CACHE/smoke13_raw_paths.json'))))"); do
  python /gpfs/home/koc/koc821022/vggt_features/vggt_probe/extract_ext_frames.py "$ep" 2>&1 | tee $LOG/extract_$ep.log
done
