#!/bin/bash
#SBATCH --job-name=robotseg_ext --account=etur59 --partition=acc --qos=acc_debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=00:40:00
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/robotseg/logs/robotseg_ext_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/robotseg/logs/robotseg_ext_%j.err
# usage: sbatch robotseg_gripper_ext.sbatch [EPISODE] [extra args to the .py]
# RobotSeg env = venv over recal3r (py3.11, torch 2.5.1+cu121); user site off so ~/.local (py3.12) cannot leak in.
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
PY=/gpfs/scratch/etur59/koc821022/conda_envs/robotseg/bin/python
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
EP=${1:-RAIL+80edfcb1+2023-07-14-14h-28m-45s}; shift || true
nvidia-smi -L
$PY robotseg_gripper_ext.py --episode "$EP" "$@"
