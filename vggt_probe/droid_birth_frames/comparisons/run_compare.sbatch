#!/bin/bash
#SBATCH --job-name=trk_compare
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --time=00:30:00
#SBATCH --output=/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/comparisons/track_%j.out
source ~/track_on/leonardo/env.sh
cd $TRACKON_ROOT
python leonardo/compare_checkpoints_track.py --items AUTOLab+5d05c5aa+2023-08-12-20h-31m-55s:ext1 AUTOLab+5d05c5aa+2023-08-12-20h-31m-55s:ext2 AUTOLab+5d05c5aa+2023-09-02-09h-55m-32s:ext1 AUTOLab+5d05c5aa+2023-09-02-09h-55m-32s:ext2 --ckpts track_on_r=/leonardo_work/AIFAC_S07_110/bora/checkpoints/track_on/track_on_r.pt trackon2=/leonardo_work/AIFAC_S07_110/bora/checkpoints/track_on/trackon2_dinov3.pt
