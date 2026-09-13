#!/bin/bash
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=posefix
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_posefix_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_posefix_%j.err
set -o pipefail
WT=/gpfs/home/koc/koc821022/maks_idea
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh; conda activate cuteanything
export PYTHONPATH="$WT/src:$WT/src/CUT3R:$WT/src/CUT3R/src:$PYTHONPATH"
echo "Node $(hostname)  start $(date)"
python "$WT/eval_pipeline/maks_posefix_sweep.py" --label causal_posefix --procs 72
python "$WT/eval_pipeline/maks_posefix_sweep.py" --label causal_posefix_ema --ema --procs 72
echo "done $(date)"
