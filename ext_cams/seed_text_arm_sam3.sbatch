#!/bin/bash
#SBATCH --job-name=seed_text_arm --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=02:00:00
#SBATCH --array=0-12
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/text_arm_sam3/logs/seed_text_arm_%A_%a.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/text_arm_sam3/logs/seed_text_arm_%A_%a.err
# METHOD text_arm_sam3 on the 13 smoke scenes (one array task per scene). Based on sam3_gripper_boxclick_3cams.sbatch.
# usage: sbatch seed_text_arm_sam3.sbatch [SCENES_FILE] [extra args to seed_text_arm_sam3.py]
#        sbatch --array=0 seed_text_arm_sam3.sbatch               # RAIL only
#        sbatch --array=0 --qos=acc_debug --time=00:30:00 seed_text_arm_sam3.sbatch
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate /gpfs/scratch/etur59/koc821022/conda_envs/sam3
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=/gpfs/scratch/etur59/koc821022/hf_cache
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
SCENES=${1:-/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt}; shift || true
EP=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SCENES")
[ -z "$EP" ] && { echo "no scene at index $SLURM_ARRAY_TASK_ID in $SCENES"; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID episode $EP host $(hostname) start $(date)"
nvidia-smi -L
python seed_text_arm_sam3.py --episode "$EP" --out_root /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/text_arm_sam3 "$@"
echo "end $(date) rc $?"
