#!/bin/bash
#SBATCH --job-name=robotseg_seeds --account=etur59 --partition=acc --qos=acc_debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=00:45:00
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/robotseg/logs/robotseg_seeds_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/robotseg/logs/robotseg_seeds_%j.err
# Seeded repeats (seeds 0 1 2) of every variant incl. the visible-start sensitivity rows, each seed in its own out_root:
#   .../ext_cams/robotseg_seed<k>/<res>/<EP>/ ; then CPU scoring of each (no videos).
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
PY=/gpfs/scratch/etur59/koc821022/conda_envs/robotseg/bin/python
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
EP=${1:-RAIL+80edfcb1+2023-07-14-14h-28m-45s}
R=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams
nvidia-smi -L
for k in 0 1 2; do
  $PY robotseg_gripper_ext.py --episode "$EP" --seed $k --out_root $R/robotseg_seed$k || exit 1
done
for k in 0 1 2; do for r in 1280x720 320x180; do
  OMP_NUM_THREADS=1 $PY robotseg_score.py --episode "$EP" --res $r --out_root $R/robotseg_seed$k --no_video &
done; done; wait
