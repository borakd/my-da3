#!/bin/bash
# Parallel "regular" (pretrained CUT3R) eval over ALL test scenes, on a separate
# node, concurrent with the main job's "final" pass. The main job's own regular
# pass is disabled via the SKIP_regular sentinel, so this job owns regular.
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --gres=gpu:rtx_a6000:4
#SBATCH --nodelist=ai17
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=450G
#SBATCH --time=72:00:00
#SBATCH --job-name=cut3r_reg
#SBATCH --output=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_reg_%j.out
#SBATCH --error=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_reg_%j.err

set -o pipefail
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything

ROOT=/scratch/bdursun25/cuteanything
MYDA3=$ROOT/my-da3
CUT3R_DIR=$MYDA3/src/CUT3R
EVAL_SCRIPT=/scratch/bdursun25/streaming-3d/eval_depth_poses.py
SCENES_ROOT=$ROOT/scenes/pointworld_droid_splits/test/dl3dv_multi/wrist
OUT=$ROOT/outputs/cut3r_eval
SCENE_LIST=$OUT/scene_list.txt
REG_CKPT=$CUT3R_DIR/src/cut3r_512_dpt_4_64.pth

export PYTHONPATH="$MYDA3/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Node: $(hostname)  GPUs: $NUM_GPUS  regular-only over $(wc -l < "$SCENE_LIST") scenes"
echo "Start: $(date)"

# One worker per GPU; round-robin scene shard. --ignore_skip_sentinel so this
# job runs regular even though SKIP_regular is set (that sentinel only disables
# the OTHER job's regular pass). Logs to worker_NEWg*_regular.log to avoid
# clobbering the main job's (no-op) worker_g*_regular.log.
for g in $(seq 0 $((NUM_GPUS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=$g
    echo "[NEWg$g] starting regular at $(date)"
    python "$MYDA3/eval_pipeline/infer_and_eval_worker.py" \
      --ckpt "$REG_CKPT" --label regular --size 320 \
      --scenes_root "$SCENES_ROOT" --scene_list "$SCENE_LIST" \
      --pred_base "$OUT/regular/preds" --eval_base "$OUT/regular/eval" \
      --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
      --shard_id "$g" --num_shards "$NUM_GPUS" --device cuda \
      --ignore_skip_sentinel \
      > "$OUT/logs/worker_NEWg${g}_regular.log" 2>&1
    echo "[NEWg$g] finished regular rc=$? at $(date)"
  ) &
done
wait
echo "Parallel regular done: $(date)"
