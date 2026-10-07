#!/bin/bash
#SBATCH --job-name=rs_birth_v3
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/masks_run_v3_reentries/logs/rs_birth_v3_%A_%a.out
#SBATCH --error=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/masks_run_v3_reentries/logs/rs_birth_v3_%A_%a.err
# v3: RobotSeg masks on every RE-ENTRY frame (events >= 1 in events_v3/events.csv), both cameras,
# same settings as v2 (automatic prompt, box fallback, PointWorld spill flag), written to masks_run_v3_reentries/. v1 and v2 untouched.
set -o pipefail
source /leonardo/home/userexternal/bdursun0/miniforge3/etc/profile.d/conda.sh
conda activate robotseg
cd /leonardo/home/userexternal/bdursun0/vggt_features/vggt_probe/robotseg_stage
O=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames
echo "job $SLURM_JOB_ID shard $SLURM_ARRAY_TASK_ID on $(hostname) $(date)"
nvidia-smi -L
python segment_birth_frames.py --shard $SLURM_ARRAY_TASK_ID --nshards 4 --extrinsics pointworld \
  --birth_json $O/birth_frames_v2/birth_frames.json --events $O/events_v3/events.csv --out $O/masks_run_v3_reentries
echo "done $(date)"
