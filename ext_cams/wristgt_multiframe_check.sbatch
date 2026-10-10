#!/bin/bash
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --job-name=wristgt_mf
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam/logs/%x_%j.err
set -o pipefail
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate vggt-omega
export PYTHONNOUSERSITE=1
O=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam/wristgt_check
python -u /gpfs/home/koc/koc821022/using_ext_cams/ext_cams/wristgt_multiframe_check.py --scene_list $O/mf_scenes.txt --out $O/mf
