#!/bin/bash
#SBATCH --job-name=rs_refine_v3
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/masks_run_v3_reentries/logs/refine_%j.out
#SBATCH --error=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/masks_run_v3_reentries/logs/refine_%j.err
# Second pass over masks_run_v3_reentries (final mask = automatic unless near-empty), PointWorld extrinsics.
set -o pipefail
source /leonardo/home/userexternal/bdursun0/miniforge3/etc/profile.d/conda.sh
conda activate robotseg
cd /leonardo/home/userexternal/bdursun0/vggt_features/vggt_probe/robotseg_stage
echo "job $SLURM_JOB_ID on $(hostname) $(date)"
python refine_birth_masks.py --run /leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/masks_run_v3_reentries --extrinsics pointworld
echo "done $(date)"
