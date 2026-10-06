#!/bin/bash
#SBATCH --job-name=rs_refine
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo/home/userexternal/bdursun0/RobotSeg/logs/refine_%j.out
#SBATCH --error=/leonardo/home/userexternal/bdursun0/RobotSeg/logs/refine_%j.err
# Second pass over the segment_birth_frames.py results: final mask = automatic mask unless near-empty
# (see refine_birth_masks.py).
set -o pipefail
source /leonardo/home/userexternal/bdursun0/miniforge3/etc/profile.d/conda.sh
conda activate robotseg
cd /leonardo/home/userexternal/bdursun0/vggt_features/vggt_probe/robotseg_stage
echo "job $SLURM_JOB_ID on $(hostname) $(date)"
python refine_birth_masks.py
echo "done $(date)"
