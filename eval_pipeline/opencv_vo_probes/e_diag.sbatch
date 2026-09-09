#!/bin/bash
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:45:00
#SBATCH --job-name=e_diag
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_e_diag_%j.out
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
G=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/e_diag
R=/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist
for GAP in 4 8 16; do
  for S in $(cat /gpfs/home/koc/koc821022/my-da3/eval_pipeline/cg_smoke_scenes_12.txt); do
    python $G/e_diag.py $R/$S $G/out/$S.gap$GAP.json $GAP > $G/out/$S.gap$GAP.log 2>&1 &
  done
done
wait
echo ALLDONE
