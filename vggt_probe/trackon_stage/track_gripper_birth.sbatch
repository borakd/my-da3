#!/bin/bash
#SBATCH --job-name=trk_gripper
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-31
#SBATCH --output=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/trackon_run/logs/slurm_%A_%a.out
#SBATCH --error=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/trackon_run/logs/slurm_%A_%a.err
# Track-On gripper tracks from the birth frame for all DROID test scenes, 32 shards x 1 GPU.
#   NSHARDS must equal the array size. EXTRA passes flags to the script (e.g. --render_every 200).
source ~/vggt_features/vggt_probe/trackon_stage/env.sh
cd "$TRACKON_ROOT"
NSHARDS=${NSHARDS:-32}
echo "job $SLURM_JOB_ID shard $SLURM_ARRAY_TASK_ID/$NSHARDS on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$TRACKON_STAGE/track_gripper_birth.py" --shard $SLURM_ARRAY_TASK_ID --nshards $NSHARDS ${EXTRA:-}
echo "done $(date)"
