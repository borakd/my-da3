#!/bin/bash
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=01:00:00
#SBATCH --job-name=trainflip
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam/logs/%x_%j.err
OMP_NUM_THREADS=1 /home/koc/koc821022/.conda/envs/cuteanything/bin/python -u /gpfs/home/koc/koc821022/using_ext_cams/ext_cams/train_flip_classify.py \
  --compact_root /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam_train --scene_list /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam_train/train_scenes.txt \
  --store /gpfs/scratch/etur59/koc821022/pointworld_droid_splits/train/dl3dv_multi/wrist \
  --template /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam/wristgt_check/updown_template.npy --out_csv /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam_train/train_flipscore.csv --threads 38
