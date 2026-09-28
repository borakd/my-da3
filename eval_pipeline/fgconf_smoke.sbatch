#!/bin/bash
# Smoke for the fg_conf per-frame confidence state gate and the uniform-token control, zero-shot backbone.
# One arm per GPU on the scenes in $OUT/summary_tmp/fgconf_smoke_scenes.txt, preds kept, gate dumps on.
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --time=00:45:00
#SBATCH --job-name=fgconf_smoke
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/fgconf_smoke_%j.out
set -o pipefail
WT=/gpfs/home/koc/koc821022/using_ext_cams
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 DUMP_GATE=1
CKPT=$WT/src/CUT3R/src/cut3r_512_dpt_4_64.pth
ROOT=$OUT/fgconf_smoke; LIST=$OUT/summary_tmp/fgconf_smoke_scenes.txt
g=0
for ARM in zsfc_soft zsfc_hard zsfc_w1 zstok_aluni; do
  mkdir -p "$ROOT/$ARM"
  CUDA_VISIBLE_DEVICES=$g python "$WT/eval_pipeline/infer_and_eval_worker.py" \
    --ckpt "$CKPT" --label "$ARM" --size 320 --scenes_root "$SCENES_ROOT" --scene_list "$LIST" \
    --pred_base "$ROOT/$ARM/preds" --eval_base "$ROOT/$ARM/eval" --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
    --shard_id 0 --num_shards 1 --device cuda --ignore_skip_sentinel \
    --control_json "$WT/eval_pipeline/maks_arm_$ARM.json" --skip_mode block --no_plots > "$ROOT/$ARM/worker.log" 2>&1 &
  g=$((g + 1))
done
wait
echo "done $(date)"; ls "$ROOT"/*/eval/*/eval_depth_pose_metrics.csv | wc -l
