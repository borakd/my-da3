#!/bin/bash
#SBATCH --job-name=seed_motion --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=02:00:00
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/motion_only/logs/sbatch_%A_%a.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/motion_only/logs/sbatch_%A_%a.err
# CPU-only seeding at scale (the account has no CPU QoS, so one GPU is requested and 20 CPU workers run in parallel).
# usage: sbatch --array=0-N seed_motion_only.sbatch SCENE_LIST [CHUNK] [extra args]
#   task i processes lines [i*CHUNK, (i+1)*CHUNK) of SCENE_LIST with 20 episodes at a time; CHUNK default 200.
#   ~20 ms/frame/camera per process at 640x360 -> a 300-frame episode takes ~15 s (+ overlay if --video is passed).
LIST=${1:?scene list}; CHUNK=${2:-200}; shift 2 || shift $#
OUT=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/motion_only
PY=/home/koc/koc821022/.conda/envs/cuteanything/bin/python
mkdir -p $OUT/logs
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
I=${SLURM_ARRAY_TASK_ID:-0}
sed -n "$((I*CHUNK+1)),$(((I+1)*CHUNK))p" "$LIST" | grep -v '^$' | \
  OMP_NUM_THREADS=1 xargs -P ${SLURM_CPUS_PER_TASK:-20} -I{} sh -c "$PY seed_motion_only.py --episode {} --out_root $OUT $* > $OUT/logs/{}.log 2>&1; echo {} rc=\$?"
echo TASK_DONE $I
