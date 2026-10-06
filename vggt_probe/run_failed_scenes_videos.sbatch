#!/bin/bash
#SBATCH --job-name=failed_vids
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/logs/failed_vids_%j.out
#SBATCH --error=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/logs/failed_vids_%j.err
# Review videos (3 x 10 scenes, wrist|ext1|ext2 side by side) for the never-visible scenes.
set -eo pipefail
source /leonardo/home/userexternal/bdursun0/miniforge3/etc/profile.d/conda.sh
conda activate robotseg
cd /leonardo/home/userexternal/bdursun0/vggt_features/vggt_probe
O=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames
echo "job $SLURM_JOB_ID on $(hostname) $(date)"
python make_failed_scenes_videos.py --list $O/failed_scenes.txt --out_dir $O --tile_w 384
echo "done $(date)"
