#!/bin/bash
#SBATCH --job-name=rs_birth_all
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/leonardo/home/userexternal/bdursun0/RobotSeg/logs/rs_birth_all_%A_%a.out
#SBATCH --error=/leonardo/home/userexternal/bdursun0/RobotSeg/logs/rs_birth_all_%A_%a.err
# RobotSeg single-frame gripper masks on the birth frames of all 4292 test scenes, 4 shards.
set -o pipefail
source /leonardo/home/userexternal/bdursun0/miniforge3/etc/profile.d/conda.sh
conda activate robotseg
cd /leonardo/home/userexternal/bdursun0/vggt_features/vggt_probe/robotseg_stage
echo "job $SLURM_JOB_ID shard $SLURM_ARRAY_TASK_ID on $(hostname) $(date)"
nvidia-smi -L
python segment_birth_frames.py --shard $SLURM_ARRAY_TASK_ID --nshards 4
echo "done $(date)"
