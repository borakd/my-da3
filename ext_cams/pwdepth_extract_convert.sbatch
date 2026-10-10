#!/bin/bash
#SBATCH --job-name=pwdepth_conv --account=etur59 --partition=acc --qos=acc_debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=02:00:00
#SBATCH --output=/gpfs/scratch/etur59/koc821022/pointworld_droid_depth_pkg/logs/extract_convert_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/pointworld_droid_depth_pkg/logs/extract_convert_%j.err
# CPU-only work; acc has no CPU partition for us, so this takes the minimum 1 GPU / 20 cores. Re-run whenever more parts
# are verified (each run re-streams from part 0, restores only new episodes, converts only episodes without a status).
# Long final run: sbatch --qos=acc_ehpc --time=12:00:00 pwdepth_extract_convert.sbatch
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
/gpfs/scratch/etur59/koc821022/conda_envs/robotseg/bin/python pwdepth_extract_convert.py --workers 18 "$@"
