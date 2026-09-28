#!/bin/bash
# Train the two write-gate heads (FrameConfHead, PoseSigmaHead) on the recorded feature table
# (WRITE_GATE_VARIANTS.md, Component B). One GPU; the table (4000 x 63 frames) fits in memory.
#   sbatch eval_pipeline/train_wgate.sbatch                                  # both heads, 20 epochs
#   HEADS=frame EXTRA="--epochs 40 --r_clip 20" sbatch eval_pipeline/train_wgate.sbatch
#   SMOKE=1 sbatch -q acc_debug --gres=gpu:1 -c 20 --time=00:20:00 eval_pipeline/train_wgate.sbatch   # 2 episodes, 1 epoch
#   TARGET=rel sbatch eval_pipeline/train_wgate.sbatch   # --target rel: heads on the consecutive-frame residual (npz need
#                                                        # res_rel_*: record_wgate.sbatch EXTRA="--fill_rel" on an older table)
# Output: $WGATE_ROOT/frame_conf_head.pth, $WGATE_ROOT/pose_sigma_head.pth, train_wgate_metrics.json, train_wgate_log.txt
#         (TARGET=rel: frame_conf_head_rel.pth, pose_sigma_head_rel.pth, train_wgate_metrics_rel.json, train_wgate_log_rel.txt).
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --job-name=wgate_train
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_wgate_train_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_wgate_train_%j.err
set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export NUMEXPR_MAX_THREADS=64
WGATE_ROOT=${WGATE_ROOT:-/gpfs/scratch/etur59/koc821022/checkpoints/wgate}
CKPT=${CKPT:-$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth}
HEADS=${HEADS:-both}
TARGET=${TARGET:-abs}   # abs (default, frame-0-relative loss residual) | rel (consecutive-frame residual, _rel outputs)
case "$TARGET" in abs|rel) ;; *) echo "TARGET must be abs|rel, got $TARGET"; exit 2;; esac
mn5_require f "$CUT3R_DIR/src/train_wgate_heads.py" f "$CUT3R_DIR/src/dust3r/wgate/heads.py" d "$WGATE_ROOT/features/train"
echo "Node $(hostname) heads=$HEADS target=$TARGET smoke=${SMOKE:-0} root=$WGATE_ROOT train npz: $(ls "$WGATE_ROOT/features/train" | grep -c npz) start $(date)"
python "$CUT3R_DIR/src/train_wgate_heads.py" --features "$WGATE_ROOT/features" --out "$WGATE_ROOT" \
  --pose_decoder "$WGATE_ROOT/pose_decoder.pth" --ckpt "$CKPT" --heads "$HEADS" --target "$TARGET" ${SMOKE:+--smoke} ${EXTRA:-} \
  || { echo "TRAIN FAILED rc=$?"; exit 1; }
echo "done $(date)"
