#!/bin/bash
#SBATCH --job-name=seed_vmotion --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=02:00:00
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy2/verified_motion/logs/sbatch_%A_%a.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy2/verified_motion/logs/sbatch_%A_%a.err
# CPU-only seeding (seed_verified_motion.py) at scale: the account has no CPU QoS, so one GPU is requested and
# SLURM_CPUS_PER_TASK episodes run in parallel under OMP_NUM_THREADS=1 (the script pins cv2 to one thread too).
# usage: sbatch --array=0-N seed_verified_motion.sbatch SCENE_LIST [CHUNK] [extra seed_verified_motion.py args]
#   task i processes lines [i*CHUNK, (i+1)*CHUNK) of SCENE_LIST, 20 episodes at a time; CHUNK default 200.
#   ~90-200 ms/frame for both cameras at 640x360 (700-frame episode: ~2-3 min; + overlay when --video is passed).
#   Smoke-13: sbatch --array=0 seed_verified_motion.sbatch /gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt 13 --video
# Check PRODUCTS afterwards (never exit codes): seedstudy_check_products.py --method verified_motion --root $OUT
LIST=${1:?scene list}; CHUNK=${2:-200}; shift 2 || shift $#
OUT=${OUT:-/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy2/verified_motion}  # env OUT overrides (variants)
PY=/home/koc/koc821022/.conda/envs/cuteanything/bin/python
mkdir -p $OUT/logs
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
I=${SLURM_ARRAY_TASK_ID:-0}
sed -n "$((I*CHUNK+1)),$(((I+1)*CHUNK))p" "$LIST" | grep -v '^$' | \
  OMP_NUM_THREADS=1 xargs -P ${SLURM_CPUS_PER_TASK:-20} -I{} sh -c "$PY seed_verified_motion.py --episode {} --out_root $OUT $* > $OUT/logs/{}.log 2>&1; echo {} rc=\$?"
echo TASK_DONE $I
