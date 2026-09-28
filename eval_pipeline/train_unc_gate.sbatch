#!/bin/bash
# Train the uncgate registration-uncertainty head (v1 pixel / v2 token) on the FROZEN augfull checkpoint.
#   MODE=v2 NAME=v2_flow EPISODES=2000 sbatch eval_pipeline/train_unc_gate.sbatch
#   EXTRA="--next_frame_channels" for the v2 next-frame ablation; EXTRA="--loss laplace --clip 16" etc.
#   RESUME=1 continues from <out>/<NAME>/head.pth. fp32 throughout (the eval worker is fp32); --amp is opt-in via EXTRA.
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=08:00:00
#SBATCH --job-name=unc_gate
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_unc_gate_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_unc_gate_%j.err
set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMEXPR_MAX_THREADS=64
MODE=${MODE:-v2}; NAME=${NAME:-${MODE}_flow}; EPISODES=${EPISODES:-2000}; BS=${BS:-2}
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
echo "Node $(hostname) gpus=$NGPU mode=$MODE name=$NAME episodes=$EPISODES start $(date)"
if [ "$NGPU" -gt 1 ]; then
  accelerate launch --multi_gpu --num_processes "$NGPU" --num_machines 1 --mixed_precision no --main_process_port $((20000 + SLURM_JOB_ID % 20000)) \
    "$CUT3R_DIR/src/train_unc_gate.py" --mode "$MODE" --name "$NAME" --episodes "$EPISODES" --batch_size "$BS" ${RESUME:+--resume} ${EXTRA:-} || { echo "TRAIN FAILED rc=$?"; exit 1; }
else
  python "$CUT3R_DIR/src/train_unc_gate.py" --mode "$MODE" --name "$NAME" --episodes "$EPISODES" --batch_size "$BS" ${RESUME:+--resume} ${EXTRA:-} || { echo "TRAIN FAILED rc=$?"; exit 1; }
fi
echo "done $(date)"
