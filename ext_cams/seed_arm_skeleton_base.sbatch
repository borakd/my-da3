#!/bin/bash
#SBATCH --job-name=seed_armskel --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=02:00:00
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas/arm_skeleton_base/logs/seed_%A_%a.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas/arm_skeleton_base/logs/seed_%A_%a.err
# IDEA 1 arm_skeleton_base, full GPU pipeline per scene: --stage detect (SAM3 text detector, all arm components) ->
# --stage cpu --arm_source detect (skeleton, chain walk, cross-view verification, CPU masks, verified_points) ->
# --stage sam3 (SAM3 box+click tracker at the verified leaf, forward only) [--video].
#   sbatch seed_arm_skeleton_base.sbatch [SCENES_FILE]                 # array: one scene per task (set --array=0-12)
#   sbatch --array=0 seed_arm_skeleton_base.sbatch                     # RAIL only
#   MODE=all sbatch --array=0 --qos=acc_debug seed_arm_skeleton_base.sbatch   # ONE job, all scenes sequentially (models built once per stage)
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate /gpfs/scratch/etur59/koc821022/conda_envs/sam3
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=/gpfs/scratch/etur59/koc821022/hf_cache OMP_NUM_THREADS=4
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
SCENES=${1:-/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt}; shift || true
OUT=${OUT_ROOT:-/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas/arm_skeleton_base}
echo "task ${SLURM_ARRAY_TASK_ID:-x} mode ${MODE:-array} host $(hostname) start $(date)"; nvidia-smi -L
if [ "${MODE:-array}" = "all" ]; then
  python seed_arm_skeleton_base.py --episodes "$SCENES" --stage detect --out_root "$OUT"
  python seed_arm_skeleton_base.py --episodes "$SCENES" --stage cpu --arm_source detect --out_root "$OUT"
  python seed_arm_skeleton_base.py --episodes "$SCENES" --stage sam3 --out_root "$OUT" "$@"
else
  EP=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SCENES")
  [ -z "$EP" ] && { echo "no scene at index $SLURM_ARRAY_TASK_ID in $SCENES"; exit 1; }
  echo "episode $EP"
  python seed_arm_skeleton_base.py --episode "$EP" --stage detect --out_root "$OUT" && \
  python seed_arm_skeleton_base.py --episode "$EP" --stage cpu --arm_source detect --out_root "$OUT" && \
  python seed_arm_skeleton_base.py --episode "$EP" --stage sam3 --out_root "$OUT" "$@"
fi
echo "end $(date) rc $?"
