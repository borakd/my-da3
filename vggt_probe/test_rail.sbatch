#!/bin/bash
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=vggt_rail_test
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/logs/slurm_%x_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/logs/slurm_%x_%j.err
set -o pipefail
WT=/gpfs/home/koc/koc821022/vggt_features
export WT OUT_ROOT=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe BATCH=8
EP_LIST=$WT/vggt_probe/episodes_rail_test.txt VARIANTS="anchored_t0 wrist_pair ext_first offline_all offline_all_ext" \
  bash $WT/vggt_probe/run_vggt_variants.sbatch
EP_LIST=$WT/vggt_probe/episodes_rail_test_rawwrist.txt VARIANTS="anchored_t0=anchored_t0_rawwrist" \
  bash $WT/vggt_probe/run_vggt_variants.sbatch
echo "=== MEAN rows"
for v in anchored_t0 anchored_t0_rawwrist wrist_pair ext_first offline_all offline_all_ext; do
  c=$OUT_ROOT/$v/eval/RAIL+80edfcb1+2023-07-14-14h-28m-45s/eval_depth_pose_metrics.csv
  [ -f "$c" ] && echo "$v $(grep '^ALL,MEAN' $c)" || echo "$v MISSING"
done
