#!/bin/bash
#SBATCH --job-name=gru_grid_probe
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=57   # was 48; MN5 derives mem from cores (8G/core) -> 456G ~= old --mem=450G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gru_grid_probe_%A.out
#SBATCH --error=logs/gru_grid_probe_%A.err
# Uniform falsifier sweep over the whole PoseGRU lever grid (16 arms at
# checkpoint-final), sharded one process per GPU on a full l40s node.
#   sbatch probe_gru_grid_all.sbatch
# Results: logs/gru_grid_probe/<label>.json  (re-runnable; existing labels skip)
set -eo pipefail
mkdir -p logs

# --- self-locating worktree -------------------------------------------------
# Run the checkout this job was SUBMITTED from, never a hardcoded path, and fail
# loudly if that is not a my-da3 tree. (Do not use ${BASH_SOURCE[0]} here: SLURM
# copies the batch script to its spool dir, so it would not point at the repo.)
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything

OUT=$WT/logs/gru_grid_probe
export DL3DV_CACHE_DIR=/gpfs/scratch/etur59/koc821022/.dl3dv_cache
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
