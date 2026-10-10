#!/bin/bash
#SBATCH --job-name=rig_track --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=02:00:00
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/logs/rig_track_%A_%a.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/logs/rig_track_%A_%a.err
# Array driver for rig_track.py (CPU only; the GPU is requested because the account has no CPU partition).
#   sbatch --array=0-N%8 rig_track_array.sbatch MANIFEST [extra rig_track.py args, e.g. --video]
# MANIFEST: one episode per line, four whitespace-separated fields:  EP  MASK_NPZ_EXT1  MASK_NPZ_EXT2  OUT_DIR
# Array task k runs lines [k*CHUNK, (k+1)*CHUNK) with CHUNK=${CHUNK:-100}, SLURM_CPUS_PER_TASK episodes at a time,
# each under OMP_NUM_THREADS=1. Check PRODUCTS afterwards: every OUT_DIR must hold record.json (status ok|failed).
set -u
MANIFEST=$1; shift || true
CHUNK=${CHUNK:-100}
PY=/home/koc/koc821022/.conda/envs/cuteanything/bin/python
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
mkdir -p /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/logs
START=$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 )); END=$(( START + CHUNK - 1 ))
export OMP_NUM_THREADS=1 EXTRA="$*"
sed -n "${START},${END}p" "$MANIFEST" | xargs -P "${SLURM_CPUS_PER_TASK:-4}" -L 1 bash -c \
  'timeout 1500 '"$PY"' rig_track.py --episode "$0" --mask_npz_ext1 "$1" --mask_npz_ext2 "$2" --out "$3" $EXTRA \
   || echo "TIMEOUT/ERROR $0 rc=$?"'
echo "array task $SLURM_ARRAY_TASK_ID done: lines $START-$END of $MANIFEST"
