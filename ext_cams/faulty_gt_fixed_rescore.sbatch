#!/bin/bash
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --job-name=fixedgt
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam/logs/%x_%j.err
W=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam/wristgt_check
OMP_NUM_THREADS=1 /home/koc/koc821022/.conda/envs/cuteanything/bin/python -u /gpfs/home/koc/koc821022/using_ext_cams/ext_cams/faulty_gt_fixed_rescore.py \
  --root $W/fixedgt --faulty_csv $W/faulty_wrist_gt.csv --procs 38
