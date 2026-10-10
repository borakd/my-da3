#!/bin/bash
#SBATCH --job-name=pwdepth_chain --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=05:00:00
#SBATCH --output=/gpfs/scratch/etur59/koc821022/pointworld_droid_depth_pkg/logs/chain_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/pointworld_droid_depth_pkg/logs/chain_%j.err
# Self-chaining restore + convert while pwdepth_download.sh is still fetching parts: each run restores every newly
# verified part (re-streaming from part 0; the package is one zstd frame) and converts the new episodes, then resubmits
# itself to start $GAP later until all 252 parts are verified AND every store episode has an ok status.
# Start once:  sbatch pwdepth_convert_chain.sbatch      Stop:  scancel -n pwdepth_chain
# CPU-only work; the acc partition needs a GPU in the request (no CPU partition for this account).
GAP=${GAP:-now+5hours}
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
PKG=/gpfs/scratch/etur59/koc821022/pointworld_droid_depth_pkg
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
/gpfs/scratch/etur59/koc821022/conda_envs/robotseg/bin/python pwdepth_extract_convert.py --workers 18
# Depth-gated RobotSeg (robotseg_depthgate.py) on every smoke scene whose exterior depth is now converted and that has not
# been run yet (RAIL+80edfcb1 is the decisive one, ~part 124), then re-score all of them (robotseg_depthgate_score.py).
PY=/gpfs/scratch/etur59/koc821022/conda_envs/robotseg/bin/python
RS=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/robotseg_depthgate
EXT=/gpfs/scratch/etur59/koc821022/pointworld_droid_ext_all/dl3dv_multi
pending=""; donelist=$RS/episodes_done.txt; : > $donelist.tmp
while read -r ep; do
  [ -z "$ep" ] && continue
  if [ -f "$RS/1280x720/$ep/run_info.json" ] && [ -f "$RS/320x180/$ep/run_info.json" ]; then echo "$ep" >> $donelist.tmp; continue; fi
  grep -q '"ok": true' "$EXT/_status/$ep.json" 2>/dev/null && pending="$pending $ep"
done < /gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt
if [ -n "$pending" ]; then
  echo "$(date '+%F %T') depth-gated RobotSeg on:$pending"
  OMP_NUM_THREADS=8 $PY robotseg_depthgate.py --episodes $pending
  for ep in $pending; do [ -f "$RS/1280x720/$ep/run_info.json" ] && echo "$ep" >> $donelist.tmp; done
  mv $donelist.tmp $donelist
  $PY robotseg_depthgate_score.py --episodes $donelist > $RS/logs/score_chain_${SLURM_JOB_ID}.md 2>&1
  echo "$(date '+%F %T') scored $(wc -l < $donelist) smoke scenes -> $RS/depthgate_vs_seedstudy.md"
else
  mv $donelist.tmp $donelist
fi
nparts=$(wc -l < $PKG/manifest.tsv)
nok=$(ls $PKG/droid/depth_320x180 | grep -c '\.ok$')
last=$(grep "CONVERT this run" $PKG/logs/chain_${SLURM_JOB_ID}.out | tail -n 1)
echo "$(date '+%F %T') parts verified $nok/$nparts | $last"
if [ "$nok" -lt "$nparts" ] || ! echo "$last" | grep -q "episodes done: 42925/42925"; then
  if [ "$(squeue -u $USER -h -n pwdepth_chain -t PENDING | wc -l)" -eq 0 ]; then
    sbatch --begin="$GAP" pwdepth_convert_chain.sbatch
  fi
else
  echo "ALL DONE: every part verified and every store episode converted"
fi
