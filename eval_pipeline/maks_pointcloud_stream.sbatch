#!/bin/bash
# Render the 3D point-cloud stream comparison (GT | CUT3R | CUT3R + causal pose
# recalibration) for a handful of representative scenes.
#
# Must run under Slurm: the login nodes enforce a hard 300s RLIMIT_CPU per
# process, which SIGKILLs the renderer mid-sequence and leaves a truncated mp4
# with no moov atom.
#
#   sbatch eval_pipeline/maks_pointcloud_stream.sbatch
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=pcstream
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_pcstream_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_pcstream_%j.err

set -o pipefail
WT=${WT:-/gpfs/home/koc/koc821022/maks_idea}
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
OUT=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval
VD=$OUT/pointcloud_streams
mkdir -p "$VD"
echo "Node $(hostname)  start $(date)"

# name:scene  -- one clear win, one median-behaviour scene, one where it hurts.
ARMS=(
  "win:IRIS+938130c4+2023-07-10-16h-29m-44s"
  "median:TRI+52ca9b6a+2023-11-14-13h-30m-20s"
  "regression:AUTOLab+5d05c5aa+2023-07-31-18h-51m-02s"
  "familiar:RAIL+80edfcb1+2023-07-14-14h-28m-45s"
)
for A in "${ARMS[@]}"; do
  TAG=${A%%:*}; SCENE=${A#*:}
  echo "=== $TAG  $SCENE  ($(date))"
  python "$WT/eval_pipeline/maks_render_pointcloud_stream.py" \
    --scene "$SCENE" --out "$VD/${TAG}_${SCENE}.mp4" --stride 4 --fps 12 --fill 0.42 --depth_pct 88 || exit 1
done
echo "done $(date)"
ls -la "$VD"
