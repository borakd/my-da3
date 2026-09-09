#!/bin/bash
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:30:00
#SBATCH --job-name=corr_bench
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_corr_bench_%j.out
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SC=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/corr_bench
R=/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist
for S in $(cat /gpfs/home/koc/koc821022/my-da3/eval_pipeline/cg_smoke_scenes_12.txt); do
  python $SC/corr_bench.py $R/$S $SC/corr/$S.json 1 4 > $SC/corr/$S.log 2>&1 &
done
wait
echo ALLDONE
