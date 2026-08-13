#!/bin/bash
#SBATCH --job-name=p43_premise
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:40:00
#SBATCH --output=logs/p43_premise_%A.out
#SBATCH --error=logs/p43_premise_%A.err
# P4.3 PREMISE TEST — is a pose-free readout of the current view a better
# estimate of that view's pose than the fed-back trajectory is?
#
#   sbatch probe_p43_premise.sbatch
#
# Reports in minutes. Gate the 8-NODE LAUNCH on it, not the build. Reads two
# checkpoints that already exist (the scored r8 arm and baseline A) over the
# same 100 test scenes; writes p43_premise.json plus a per-view table.
# Headline is e_probe/e_vel on the late views — NOT e_probe/e_lag, which a pure
# velocity extrapolator beats using no image evidence at all.
set -eo pipefail
mkdir -p logs

# --- self-locating worktree -------------------------------------------------
# Run the checkout this job was SUBMITTED from, never a hardcoded path. (Do not
# use ${BASH_SOURCE[0]}: SLURM copies the batch script to its spool dir.)
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || {
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything

# The script hard-fails if any of these is set, but clear them here too so an
# inherited --export=ALL environment cannot contaminate the rollout.
# P43_ALLOW_CONTAMINATION=1 marks a DELIBERATE falsifier run, where the point is
# to have one of these set. Everything else keeps the unconditional scrub.
if [ "${P43_ALLOW_CONTAMINATION:-0}" != "1" ]; then
  unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE POSE_GRU_HIDDEN_SHUFFLE \
        POSE_GRU_HIDDEN_ZERO POSE_GRU_IMG_FEAT_SHUFFLE POSE_GRU_IMG_FEAT_ZERO \
        POSE_GRU_FORCE_ITERS POSE_GRU_PROBE_SHUFFLE POSE_GRU_PROBE_LAG \
        POSE_GRU_FORCE_REFINE
fi

export DL3DV_CACHE_DIR=/gpfs/scratch/etur59/koc821022/.dl3dv_cache
export PYTHONPATH=$WT:$WT/src:$WT/src/CUT3R:$WT/src/CUT3R/src:${PYTHONPATH:-}
export P43_WORKERS="${P43_WORKERS:-16}"
export P43_OUT="${P43_OUT:-$WT/p43_premise.json}"

python "$WT/probe_p43_premise.py" "$@"
