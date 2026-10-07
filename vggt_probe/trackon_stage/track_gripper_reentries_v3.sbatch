#!/bin/bash
#SBATCH --job-name=trk_gripper_v3
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/trackon_run_v3_reentries/logs/slurm_%A_%a.out
#SBATCH --error=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/trackon_run_v3_reentries/logs/slurm_%A_%a.err
# Track-On-R gripper tracks from every RE-ENTRY frame (v3), identical settings to v1/v2
# (track_on_r.pt, 20x20 support grid, delta_v 0.8, same query sampling), entry frame to episode end.
# Work list from build_events_run.py --phase worklist; npz named <cam>_f<entry>.npz next to the v2 files.
source ~/vggt_features/vggt_probe/trackon_stage/env.sh
cd "$TRACKON_ROOT"
NSHARDS=${NSHARDS:-8}
O=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames
echo "job $SLURM_JOB_ID shard $SLURM_ARRAY_TASK_ID/$NSHARDS on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$TRACKON_STAGE/track_gripper_birth.py" --shard $SLURM_ARRAY_TASK_ID --nshards $NSHARDS \
  --worklist $O/trackon_run_v3_reentries/worklist.json --out $O/trackon_run_v3_reentries ${EXTRA:-}
echo "done $(date)"
