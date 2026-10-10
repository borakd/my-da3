#!/bin/bash
#SBATCH --job-name=silh_model --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=02:00:00
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas/silhouette_model/logs/silh_%A.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas/silhouette_model/logs/silh_%A.err
# Slurm fallback for seed_silhouette_model.py (CPU only; the GPU is requested because the account has no CPU partition).
#   sbatch seed_silhouette_model.sbatch MANIFEST [extra args]     (manifest lines: EP MASK1 MASK2 OUT ANCHORS|NONE)
set -u
MANIFEST=$1; shift || true
PY=/home/koc/koc821022/.conda/envs/cuteanything/bin/python
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
export OMP_NUM_THREADS=1 EXTRA="$*"
xargs -P "${SLURM_CPUS_PER_TASK:-4}" -L 1 bash -c '
  AA=""; [ "${4:-NONE}" != "NONE" ] && AA="--anchors $4"; mkdir -p "$3"
  timeout 3000 '"$PY"' seed_silhouette_model.py --episode "$0" --mask_npz_ext1 "$1" --mask_npz_ext2 "$2" --out "$3" $AA --diag $EXTRA > "$3/run.log" 2>&1 || echo "TIMEOUT/ERROR $0 rc=$?"' < "$MANIFEST"
echo "done: $(date)"
