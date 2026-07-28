#!/bin/bash
# Smoke test for infer_and_eval_worker_ray.py on ONE GPU:
#   1. Worker processes 2 scenes (shard 0/12, --limit 2) into the REAL
#      captain_ray_gt_last output dirs (they count toward the full run, which
#      skips them via the eval-CSV resume check).
#   2. Parity check: run demo_ray.py --conditioning gt directly on the first
#      scene and compare per-frame depth + camera pose against the worker's
#      saved preds (the worker must reproduce the user-facing script exactly).
#
#SBATCH --account=avg
#SBATCH --partition=avg
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --job-name=cray_smoke
#SBATCH --output=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_cray_smoke_%j.out
#SBATCH --error=/scratch/bdursun25/cuteanything/outputs/cut3r_eval/logs/slurm_cray_smoke_%j.err

set -o pipefail
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything

ROOT=/scratch/bdursun25/cuteanything
CRAY=$ROOT/captain_gru_v2
CUT3R_DIR=$CRAY/src/CUT3R
EVAL_SCRIPT=/scratch/bdursun25/streaming-3d/eval_depth_poses.py
SCENES_ROOT=$ROOT/scenes/pointworld_droid_splits/test/dl3dv_multi/wrist
OUT=$ROOT/outputs/cut3r_eval
SCENE_LIST=$OUT/scene_list.txt
LABEL=captain_ray_gt_last
CKPT=$ROOT/checkpoints/cut3r_multinode/captain_ray_finetune_aug_full/checkpoint-last.pth
PARITY_DIR=$OUT/$LABEL/parity_demo_ray

export PYTHONPATH="$CRAY/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs" "$PARITY_DIR"

echo "=== 1/3 worker smoke: 2 scenes, shard 0/12, conditioning=gt ==="
python "$CRAY/eval_pipeline/infer_and_eval_worker_ray.py" \
  --ckpt "$CKPT" --label "$LABEL" --size 320 --conditioning gt \
  --scenes_root "$SCENES_ROOT" --scene_list "$SCENE_LIST" \
  --pred_base "$OUT/$LABEL/preds" --eval_base "$OUT/$LABEL/eval" \
  --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
  --shard_id 0 --num_shards 12 --device cuda --limit 2
rc=$?
echo "worker rc=$rc"
[ $rc -ne 0 ] && exit $rc

SCENE=$(sed -n 1p "$SCENE_LIST")
echo "=== 2/3 direct demo_ray.py on $SCENE ==="
python "$CUT3R_DIR/demo_ray.py" \
  --model_path "$CKPT" \
  --seq_path "$SCENES_ROOT/$SCENE/dense/rgb" \
  --conditioning gt --size 320 --device cuda \
  --output_dir "$PARITY_DIR" --disable_viewer
rc=$?
echo "demo_ray rc=$rc"
[ $rc -ne 0 ] && exit $rc

echo "=== 3/3 parity compare worker preds vs demo_ray outputs ==="
python - "$OUT/$LABEL/preds/$SCENE" "$PARITY_DIR" <<'EOF'
import glob, os, sys
import numpy as np

worker_dir, demo_dir = sys.argv[1], sys.argv[2]
wd = sorted(glob.glob(os.path.join(worker_dir, "depth", "*.npy")))
dd = sorted(glob.glob(os.path.join(demo_dir, "depth", "*.npy")))
assert len(wd) == len(dd) and len(wd) > 0, f"frame count mismatch: {len(wd)} vs {len(dd)}"
max_depth = 0.0
for a, b in zip(wd, dd):
    da, db = np.load(a), np.load(b)
    assert da.shape == db.shape, f"{a}: {da.shape} vs {db.shape}"
    max_depth = max(max_depth, float(np.abs(da - db).max()))
max_pose = 0.0
for a, b in zip(sorted(glob.glob(os.path.join(worker_dir, "camera", "*.npz"))),
                sorted(glob.glob(os.path.join(demo_dir, "camera", "*.npz")))):
    pa, pb = np.load(a), np.load(b)
    max_pose = max(max_pose, float(np.abs(pa["pose"] - pb["pose"]).max()),
                   float(np.abs(pa["intrinsics"] - pb["intrinsics"]).max()))
print(f"frames={len(wd)}  max|depth diff|={max_depth:.3e}  max|pose/intrinsics diff|={max_pose:.3e}")
print("PARITY " + ("PASS" if (max_depth < 1e-3 and max_pose < 1e-3) else "FAIL (check nondeterminism vs bug)"))
EOF

echo "=== eval CSV MEAN rows for smoke scenes ==="
for d in "$OUT/$LABEL/eval/"*/; do
  csv="$d/eval_depth_pose_metrics.csv"
  [ -f "$csv" ] && echo "--- $(basename "$d")" && head -1 "$csv" && grep ",MEAN," "$csv"
done
echo "Smoke done: $(date)"
