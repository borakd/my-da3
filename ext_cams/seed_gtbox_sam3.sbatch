#!/bin/bash
#SBATCH --job-name=gtbox_sam3 --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=01:00:00
#SBATCH --array=0-12
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/gtbox_sam3/logs/gtbox_sam3_%A_%a.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/gtbox_sam3/logs/gtbox_sam3_%A_%a.err
# REFERENCE / CEILING gripper masks (gtbox_sam3) for the exterior cameras: array over the 13 smoke scenes.
# usage: sbatch seed_gtbox_sam3.sbatch [scene_list] [extra args to seed_gtbox_sam3.py]
#        sbatch --array=0 seed_gtbox_sam3.sbatch          # one scene
# Based on sam3_gripper_boxclick_3cams.sbatch (SAM3 env, offline HF cache); qos acc_ehpc so the array runs in parallel.
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate /gpfs/scratch/etur59/koc821022/conda_envs/sam3
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=/gpfs/scratch/etur59/koc821022/hf_cache
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
LIST=${1:-/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt}; shift || true
EP=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
[ -n "$EP" ] || { echo "no scene at index $SLURM_ARRAY_TASK_ID in $LIST"; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID episode $EP host $(hostname) start $(date)"
nvidia-smi -L
python seed_gtbox_sam3.py --episode "$EP" --out_root /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/gtbox_sam3 "$@"
echo "end $(date)"
