#!/bin/bash
# CPU-only dual-aggregation (mean+rmse) sim3 pose recompute from saved preds.
# Pose-only (camera npzs, no depth) -> ~100x cheaper than a full re-eval.
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --job-name=sim3_both
#SBATCH --output=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_sim3both_%j.out
#SBATCH --error=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_sim3both_%j.err

set -o pipefail
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything

ROOT=/scratch/bdursun25/cuteanything
OUT=$ROOT/outputs/cut3r_eval
SCENES_ROOT=$ROOT/scenes/pointworld_droid_splits/test/dl3dv_multi/wrist
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

LABELS=${LABELS:-"regular final augfull_last augfull_last_ep14 infinitevggt augfull_final"}

echo "Node: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK}  Labels: $LABELS  Start: $(date)"
python "$ROOT/captain_gru/eval_pipeline/pose_sim3_both.py" \
  --out_root "$OUT" --scenes_root "$SCENES_ROOT" \
  --scene_list "$OUT/scene_list.txt" \
  --labels $LABELS \
  --workers "${SLURM_CPUS_PER_TASK:-48}"
echo "Done: $(date)"
