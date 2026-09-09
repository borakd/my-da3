#!/bin/bash
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:50:00
#SBATCH --job-name=tri_kf
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_tri_kf_%j.out
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KOFF=4 LEVER=0 E_THR=0.5
G=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/tri_bench
R=/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist
run () {  # mode val tag
  for S in $(cat /gpfs/home/koc/koc821022/my-da3/eval_pipeline/cg_smoke_scenes_12.txt); do
    KF_MODE=$1 KF_VAL=$2 python $G/tri_bench.py $R/$S $G/out_kf/$S.$3.json 4 > $G/out_kf/$S.$3.log 2>&1 &
  done
  wait
}
run parallax 5 par5
run parallax 10 par10
run parallax 20 par20
run ratio 0.9 ratio0.9
run ratio 0.8 ratio0.8
run ratio 0.7 ratio0.7
echo ALLDONE
