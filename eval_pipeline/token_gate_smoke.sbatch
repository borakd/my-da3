#!/bin/bash
# Per-token consequence gate: (1) real-checkpoint CPU parity (parity_cpu_token.py), (2) trainer --smoke,
# (3) a 4-GPU timing run at arm 3's exact config (batch 8/GPU, finetuned backbone) on 512 episodes.
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --time=01:30:00
#SBATCH --job-name=tokgate_smoke
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/tokgate_smoke_%j.out
set -o pipefail
WT=/gpfs/home/koc/koc821022/maks_idea
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NUMEXPR_MAX_THREADS=64
cd "$CUT3R_DIR/src"
echo "== (1) parity_cpu_token $(date)"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= python dust3r/wgate/tests/parity_cpu_token.py 2>&1 | grep -v "Warning\|autocast"
echo "== (2) trainer smoke --token $(date)"
OUTS=/gpfs/scratch/etur59/koc821022/checkpoints/wgate/cons_smoke_token
CUDA_VISIBLE_DEVICES=0 python train_wgate_consequence.py --name smoke --site state --token --smoke --out "$OUTS" 2>&1 \
  | grep -v "Warning\|autocast" | grep "wgcons" | tail -n 12
echo "== (3) 4-GPU timing, arm-3 config + --token, 512 episodes $(date)"
FT=$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth
rm -rf "$OUTS/timing"
accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision no \
  --main_process_port $((20000 + SLURM_JOB_ID % 20000)) train_wgate_consequence.py \
  --name timing --site state --token --out "$OUTS" --ckpt "$FT" --lr 1e-4 --episodes 512 --chunk 16 \
  --wmin 0.5 --init_logit 2.0 --loss ate --batch_size 8 --num_workers 8 --precision fp32 --eval_every 100000 \
  2>&1 | grep -v "Warning\|autocast" | grep "wgcons" | tail -n 25
nvidia-smi --query-gpu=memory.used --format=csv
echo "done $(date)"
