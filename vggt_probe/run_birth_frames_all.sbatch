#!/bin/bash
#SBATCH --job-name=birth_frames
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/logs/birth_frames_%j.out
#SBATCH --error=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/logs/birth_frames_%j.err
# Kinematic gripper birth-frame detection over every scene of the DROID wrist test split.
# Needs the per-scene metadata + ZED calibrations fetched beforehand on a login node
# (fetch_droid_metadata.py); this job touches no network.
set -eo pipefail
source /leonardo/home/userexternal/bdursun0/miniforge3/etc/profile.d/conda.sh
conda activate cuteanything
cd /leonardo/home/userexternal/bdursun0/vggt_features/vggt_probe
echo "job $SLURM_JOB_ID on $(hostname) $(date)"
python birth_frames.py --all --workers ${SLURM_CPUS_PER_TASK:-4}
echo "done $(date)"
