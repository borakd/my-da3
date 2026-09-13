#!/bin/bash
# Input-property analysis over the 100-scene / 367-window set.
# CPU-heavy (two dense optical-flow fields plus a reprojection per frame over
# ~31k frames), so it must run under Slurm -- the login nodes SIGKILL at 300s CPU.
#
#   sbatch eval_pipeline/maks_window_properties.sbatch
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=winprops
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_winprops_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_winprops_%j.err

set -o pipefail
WT=${WT:-/gpfs/home/koc/koc821022/maks_idea}
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
export OPENCV_FOR_THREADS_NUM=1 OMP_NUM_THREADS=1
B=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/maks_windows100
echo "Node $(hostname)  start $(date)"
python "$WT/eval_pipeline/maks_window_properties.py" \
  --manifest "$B/manifest.json" --out_dir "$B/properties" --procs 72
echo "done $(date)"
