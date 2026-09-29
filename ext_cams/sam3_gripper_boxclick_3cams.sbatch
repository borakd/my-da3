#!/bin/bash
#SBATCH --job-name=sam3_boxclick --account=etur59 --partition=acc --qos=acc_debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=00:30:00
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/gripper_sam3/logs/sam3_boxclick_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/gripper_sam3/logs/sam3_boxclick_%j.err
# usage: sbatch sam3_gripper_masks.sbatch [EPISODE] [extra args to the .py]
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate /gpfs/scratch/etur59/koc821022/conda_envs/sam3
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=/gpfs/scratch/etur59/koc821022/hf_cache
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
EP=${1:-RAIL+80edfcb1+2023-07-14-14h-28m-45s}; shift || true
nvidia-smi -L
python sam3_gripper_boxclick_3cams.py --episode "$EP" "$@"
