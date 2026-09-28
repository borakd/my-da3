#!/bin/bash
# Horizon probe (src/CUT3R/src/horizon_probe.py) on both backbones, one GPU each, in parallel.
#   sbatch eval_pipeline/horizon_probe.sbatch            full run (acc_ehpc)
#   SMOKE=1 sbatch -q acc_debug --gres=gpu:2 -c 40 -t 00:30:00 eval_pipeline/horizon_probe.sbatch
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=06:00:00
#SBATCH --job-name=hprobe
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/hprobe_%j.out
set -o pipefail
WT=/gpfs/home/koc/koc821022/using_ext_cams
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$OUT/horizon_probe; mkdir -p "$ROOT"
SZ="--n_train 64 --n_test 48"
if [ -n "$SMOKE" ]; then SZ="--n_train 1 --n_test 1 --test_len 64 --horizons 16,32,64"; SFX=_smoke; fi
ZS=$WT/src/CUT3R/src/cut3r_512_dpt_4_64.pth
FT=$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth
cd "$CUT3R_DIR/src"
CUDA_VISIBLE_DEVICES=0 python horizon_probe.py --ckpt "$ZS" --tag zs --out "$ROOT/zs$SFX.json" $SZ 2>&1 | grep -v "Warning\|autocast" > "$ROOT/zs$SFX.log" &
CUDA_VISIBLE_DEVICES=1 python horizon_probe.py --ckpt "$FT" --tag ft --out "$ROOT/ft$SFX.json" $SZ 2>&1 | grep -v "Warning\|autocast" > "$ROOT/ft$SFX.log" &
wait
echo "done $(date)"; tail -n 12 "$ROOT"/*$SFX.log
