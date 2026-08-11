#!/bin/bash
#SBATCH --job-name=probe_gru_instats
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:55:00
#SBATCH --output=logs/probe_gru_instats_%A.out
#SBATCH --error=logs/probe_gru_instats_%A.err
# P3.4 input-scale probe: measure the real gru_in distribution of a trained
# PoseGRU so pose_gru_input_gain can be filled from 1/std.
#   sbatch probe_gru_input_stats.sbatch                     # -> D arm, checkpoint-final
#   sbatch probe_gru_input_stats.sbatch <path/to/ckpt.pth>
# NOTE: pass a keep_freq/final snapshot, NOT checkpoint-last/best while the
# training job is alive (they are rewritten every epoch; reads race the writer).
# acc_debug caps at 2h and one job at a time; the probe writes its JSON every
# 25 batches, so even a wall-clock kill leaves a usable (if shorter) sample.
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

# The probe hard-fails if any of these is set (they change what the GRU
# consumes, silently). Clear them here so an inherited --export=ALL environment
# cannot contaminate the measurement.
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE POSE_GRU_HIDDEN_SHUFFLE \
      POSE_GRU_HIDDEN_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_IMG_FEAT_ZERO \
      POSE_GRU_FORCE_ITERS

export DL3DV_CACHE_DIR=/gpfs/scratch/etur59/koc821022/.dl3dv_cache
export PYTHONPATH=$WT:$WT/src:$WT/src/CUT3R:$WT/src/CUT3R/src:${PYTHONPATH:-}
export PROBE_WORKERS="${PROBE_WORKERS:-16}"
export PROBE_OUT="${PROBE_OUT:-$WT/gru_input_stats.json}"

python "$WT/probe_gru_input_stats.py" "$@"
