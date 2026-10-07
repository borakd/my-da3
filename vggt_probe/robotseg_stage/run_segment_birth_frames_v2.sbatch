#!/bin/bash
#SBATCH --job-name=rs_birth_v2
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/masks_run_v2/logs/rs_birth_v2_%A_%a.out
#SBATCH --error=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/masks_run_v2/logs/rs_birth_v2_%A_%a.err
# v2 (PointWorld extrinsics): RobotSeg masks on the NEW birth frames only (scenes listed in
# birth_frames_v2/scenes_to_resegment.txt), written to masks_run_v2/. v1 is untouched.
set -o pipefail
source /leonardo/home/userexternal/bdursun0/miniforge3/etc/profile.d/conda.sh
conda activate robotseg
cd /leonardo/home/userexternal/bdursun0/vggt_features/vggt_probe/robotseg_stage
O=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames
echo "job $SLURM_JOB_ID shard $SLURM_ARRAY_TASK_ID on $(hostname) $(date)"
nvidia-smi -L
python segment_birth_frames.py --shard $SLURM_ARRAY_TASK_ID --nshards 4 --extrinsics pointworld \
  --birth_json $O/birth_frames_v2/birth_frames.json --scenes $O/birth_frames_v2/scenes_to_resegment.txt --out $O/masks_run_v2
echo "done $(date)"
