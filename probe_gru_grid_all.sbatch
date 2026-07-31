#!/bin/bash
#SBATCH --job-name=gru_grid_probe
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:lovelace_l40s:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=450G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gru_grid_probe_%A.out
#SBATCH --error=logs/gru_grid_probe_%A.err
# Uniform falsifier sweep over the whole PoseGRU lever grid (16 arms at
# checkpoint-final), sharded one process per GPU on a full l40s node.
#   sbatch probe_gru_grid_all.sbatch
# Results: logs/gru_grid_probe/<label>.json  (re-runnable; existing labels skip)
set -eo pipefail
mkdir -p logs

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything

WT=/scratch/bdursun25/cuteanything/captain_gru_v3
OUT=$WT/logs/gru_grid_probe
export DL3DV_CACHE_DIR=/scratch/bdursun25/cuteanything/.dl3dv_cache
export PYTHONPATH=$WT:$WT/src:$WT/src/CUT3R:$WT/src/CUT3R/src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

N=4
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i python "$WT/probe_gru_grid_all.py" \
      --out "$OUT" --shard "$i" --num_shards "$N" \
      > "logs/gru_grid_probe_${SLURM_JOB_ID}_shard$i.log" 2>&1 &
done
wait
echo "=== all shards done ==="
ls -la "$OUT"
