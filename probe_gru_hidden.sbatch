#!/bin/bash
#SBATCH --job-name=probe_gru_hidden
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
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

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything

WT=/scratch/bdursun25/cuteanything/captain_gru_v2
export DL3DV_CACHE_DIR=/scratch/bdursun25/cuteanything/.dl3dv_cache
export PYTHONPATH=$WT:$WT/src:$WT/src/CUT3R:$WT/src/CUT3R/src:${PYTHONPATH:-}

python "$WT/probe_gru_hidden_falsifier.py" "$@"
