#!/bin/bash
#SBATCH --job-name=rsdg --account=etur59 --partition=acc --qos=acc_debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=02:00:00
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/robotseg_depthgate/logs/rsdg_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/robotseg_depthgate/logs/rsdg_%j.err
# usage: sbatch robotseg_depthgate.sbatch [--convert_first] EP [EP ...]
#   --convert_first: run pwdepth_extract_convert.py (restore new parts + convert) before RobotSeg, in the same job
#   (acc_debug allows one job per user at a time).
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
PY=/gpfs/scratch/etur59/koc821022/conda_envs/robotseg/bin/python
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
if [ "${1:-}" = "--convert_first" ]; then
  shift
  OMP_NUM_THREADS=1 $PY pwdepth_extract_convert.py --workers 18 || exit 1
fi
nvidia-smi -L
$PY robotseg_depthgate.py --episodes "$@"
