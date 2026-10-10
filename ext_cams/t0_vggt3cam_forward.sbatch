#!/bin/bash
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --job-name=t0v3cam
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam/logs/%x_%A_%a.err
# VGGT-Omega t=0 [wrist, ext1, ext2] forwards x 3 reference orderings + standard eval.
# SCENE_LIST / NUM_SHARDS / LIMIT from the environment; shard = array task id.
set -o pipefail
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate vggt-omega
export PYTHONNOUSERSITE=1
OUT=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_t0_3cam
python -u /gpfs/home/koc/koc821022/using_ext_cams/ext_cams/t0_vggt3cam_forward.py \
  --scene_list "${SCENE_LIST:?}" --out_root "${OUT_ROOT:-$OUT}" \
  --shard_id "${SLURM_ARRAY_TASK_ID:-0}" --num_shards "${NUM_SHARDS:-1}" --limit "${LIMIT:-0}" ${EXTRA_ARGS:-}
