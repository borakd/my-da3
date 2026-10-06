#!/bin/bash
#SBATCH --job-name=trackon
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#
# Usage:
#   sbatch ~/vggt_features/vggt_probe/trackon_stage/track.sbatch --video in.mp4 --grid 10 --output-dir outputs/run1
#   (any extra args are passed straight to causal_track.py)

source ~/vggt_features/vggt_probe/trackon_stage/env.sh
cd "$TRACKON_ROOT"
echo "node $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python "$TRACKON_STAGE/causal_track.py" "$@"
