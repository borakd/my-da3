#!/bin/bash
# Per-frame write-dose oracle on ONE scene, both write sites (perframe_dose_search.py), then every resulting
# schedule is re-run and scored by the STANDARD worker + evaluator (plots on, preds kept).
#
#   BACKBONE=zs|ft [SCENE=...] sbatch eval_pipeline/perframe_dose.sbatch
#
# GPU g0: state site, init 1.0 (plain)       g1: state site, init = the scene's best constant
# GPU g2: mem site,   init 1.0 (plain)       g3: mem site,   init = the scene's best constant
# + placebo searches (grid 1 - k*1e-6: the chaos control) for both sites on g0 / g2
# Outputs: $OUT/perframe_dose/$BACKBONE/{search_*.json, maks_arm_*.json, <arm>/{eval,preds}, search_*.log}
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=04:00:00
#SBATCH --job-name=pfdose
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/pfdose_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/pfdose_%j.out
set -o pipefail
WT=/gpfs/home/koc/koc821022/maks_idea
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BACKBONE=${BACKBONE:?zs or ft}
SCENE=${SCENE:-RAIL+80edfcb1+2023-07-14-14h-28m-45s}
case $BACKBONE in
  zs) CKPT=$WT/src/CUT3R/src/cut3r_512_dpt_4_64.pth; PLAIN=$OUT/cut3r_zeroshot ;;
  ft) CKPT=$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth; PLAIN=$OUT/augfull_lr1e5 ;;
esac
ROOT=$OUT/perframe_dose/$BACKBONE; mkdir -p "$ROOT"
echo "$SCENE" > "$ROOT/scene.txt"
echo "Node $(hostname) backbone=$BACKBONE ckpt=$CKPT scene=$SCENE start $(date)"

run_search(){ # gpu site init tag [--placebo]
  CUDA_VISIBLE_DEVICES=$1 python "$WT/eval_pipeline/perframe_dose_search.py" --scene "$SCENE" --site "$2" --init "$3" \
    --tag "$4" --passes 2 --ckpt "$CKPT" --plain_preds "$PLAIN/preds/$SCENE" --out_dir "$ROOT" $5 2>&1 | grep -v '^DEBUG' > "$ROOT/search_$2_$4.log"
  echo "search $2/$4 rc=$? $(date)"
}
# batch-1 searches are launch-bound, so two share a GPU (g0, g2) at little cost
run_search 0 state 1.0 init1 &
run_search 1 state best best &
run_search 2 mem 1.0 init1 &
run_search 3 mem best best &
run_search 0 state 1.0 placebo --placebo &
run_search 2 mem 1.0 placebo --placebo &
wait

# score every schedule with the standard worker (model hooks, not the search harness)
ARMS=$(cd "$ROOT" && ls maks_arm_pf_*.json | sed 's/^maks_arm_//; s/\.json$//')
g=0
for ARM in $ARMS; do
  (
    CUDA_VISIBLE_DEVICES=$g python "$WT/eval_pipeline/infer_and_eval_worker.py" \
      --ckpt "$CKPT" --label "$ARM" --size 320 --scenes_root "$SCENES_ROOT" --scene_list "$ROOT/scene.txt" \
      --pred_base "$ROOT/$ARM/preds" --eval_base "$ROOT/$ARM/eval" --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
      --shard_id 0 --num_shards 1 --device cuda --ignore_skip_sentinel \
      --control_json "$ROOT/maks_arm_$ARM.json" --skip_mode block > "$ROOT/$ARM.worker.log" 2>&1
    echo "worker $ARM rc=$? $(date)"
  ) &
  g=$(( (g + 1) % 4 )); [ $g = 0 ] && wait
done
wait
echo "scored: $(ls "$ROOT"/*/eval/*/eval_depth_pose_metrics.csv 2>/dev/null | wc -l)  done $(date)"
