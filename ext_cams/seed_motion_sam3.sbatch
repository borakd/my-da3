#!/bin/bash
#SBATCH --job-name=motion_sam3 --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=01:30:00
#SBATCH --array=0-12
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/motion_sam3/logs/%A_%a.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/motion_sam3/logs/%A_%a.err
# motion_sam3 seeding on the 13 smoke scenes (one array task per scene; based on sam3_gripper_boxclick_3cams.sbatch).
# usage: sbatch seed_motion_sam3.sbatch [extra args to seed_motion_sam3.py]
#        SCENES=/path/to/list.txt sbatch --array=0-N seed_motion_sam3.sbatch
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate /gpfs/scratch/etur59/koc821022/conda_envs/sam3
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=/gpfs/scratch/etur59/koc821022/hf_cache
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
SCENES=${SCENES:-/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt}
EP=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SCENES")
[ -n "$EP" ] || { echo "no scene at index $SLURM_ARRAY_TASK_ID in $SCENES"; exit 1; }
nvidia-smi -L
echo "task $SLURM_ARRAY_TASK_ID episode $EP start $(date)"
python seed_motion_sam3.py --episode "$EP" --out_root /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/motion_sam3 "$@"
echo "end $(date) rc=$?"
