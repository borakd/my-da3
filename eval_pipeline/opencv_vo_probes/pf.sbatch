#!/bin/bash
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --job-name=vo_pf
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_vo_pf_%j.out
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WT=/gpfs/home/koc/koc821022/my-da3
OUT=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval
R=/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist
L=$WT/eval_pipeline/cg_smoke_scenes_12.txt
declare -A V
B="--cull outlier --reboot_after_fails 3 --min_inliers 30 --scale_handoff"
V[vo6_ft_perframe]="$B --focal perframe:$OUT/augfull_lr1e5/preds"
V[vo6_ft_causal]="$B --focal causal:$OUT/augfull_lr1e5/preds"
V[vo6_zs_perframe]="$B --focal perframe:$OUT/cut3r_zeroshot/preds"
V[vo6_zs_causal]="$B --focal causal:$OUT/cut3r_zeroshot/preds"
for lbl in "${!V[@]}"; do
  rm -rf $OUT/$lbl
  for sh in $(seq 0 11); do
    python $WT/eval_pipeline/opencv_vo.py --scenes_root $R --scene_list $L --preds_root $OUT/augfull_lr1e5/preds \
      --out_root $OUT --label $lbl --shard_id $sh --num_shards 12 ${V[$lbl]} > $OUT/logs/vo_pf_${lbl}_$sh.log 2>&1 &
  done
done
wait
python $WT/eval_pipeline/pose_sim3_both.py --out_root $OUT --scenes_root $R --scene_list $L --workers 12 \
  --labels "${!V[@]}" vo3_best_ho vo4_fnominal cut3r_zeroshot augfull_lr1e5 augfull_cg_fuse_g7
echo ALLDONE
