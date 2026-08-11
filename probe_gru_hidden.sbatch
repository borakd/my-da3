#!/bin/bash
#SBATCH --job-name=probe_gru_hidden
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=logs/probe_gru_hidden_%A.out
#SBATCH --error=logs/probe_gru_hidden_%A.err
# Hidden-state falsifier probe for a trained PoseGRU checkpoint.
#   sbatch probe_gru_hidden.sbatch                       # -> checkpoint-20
#   sbatch probe_gru_hidden.sbatch <path/to/ckpt.pth>    # e.g. checkpoint-final
# NOTE: pass a keep_freq/final snapshot, NOT checkpoint-last/best while the
# training job is alive (they are rewritten every epoch; reads race the writer).
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

export DL3DV_CACHE_DIR=/gpfs/scratch/etur59/koc821022/.dl3dv_cache
export PYTHONPATH=$WT:$WT/src:$WT/src/CUT3R:$WT/src/CUT3R/src:${PYTHONPATH:-}

python "$WT/probe_gru_hidden_falsifier.py" "$@"
