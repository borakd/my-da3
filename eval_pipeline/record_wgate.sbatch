#!/bin/bash
# Record write-gate features + GT residuals from the FROZEN augfull checkpoint (WRITE_GATE_VARIANTS.md, Component B).
#   sbatch eval_pipeline/record_wgate.sbatch                       # 4000 train + 64 heldout episodes, 4 GPUs, fp32
#   EPISODES=500 HELDOUT=16 sbatch eval_pipeline/record_wgate.sbatch
#   SMOKE=1 sbatch -q acc_debug --gres=gpu:1 -c 20 --time=00:30:00 eval_pipeline/record_wgate.sbatch   # 2 + 2 episodes
#     (acc_debug = ONE 1-GPU job per user; the 4-GPU / 80-CPU defaults below are for acc_ehpc)
#   PRECISION=bf16 sbatch ...   # rollout under bf16 autocast (faster; default fp32 matches the eval worker exactly)
#   EXTRA="--overwrite" / "--skip_heldout" / "--allow_overlap" / "--seed 1" pass through; WGATE_ROOT relocates the root.
#   EXTRA="--fill_rel"   re-record ONLY the episodes whose npz lacks res_rel_t / res_rel_R (a table recorded before the
#                        relative target existed); TARGET is a training-side choice (train_wgate.sbatch) -- the recorder
#                        always stores both the absolute and the consecutive-frame relative residuals, no flag needed.
# Output: $WGATE_ROOT/features/{train,heldout}/<episode_idx>.npz (+ episodes.json, manifest.json),
#         $WGATE_ROOT/pose_decoder.pth, record_log.txt.
# Episodes = first N windows of the seeded order that do not overlap an already kept window of the same scene
# (record_wgate_features.select_episodes); episode numbering is fixed by (seed, N-independent order), so
# re-running is safe and cheap: episodes whose npz exists are neither loaded nor recorded.
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=06:00:00
#SBATCH --job-name=wgate_rec
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_wgate_rec_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_wgate_rec_%j.err
set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMEXPR_MAX_THREADS=64
WGATE_ROOT=${WGATE_ROOT:-/gpfs/scratch/etur59/koc821022/checkpoints/wgate}
CKPT=${CKPT:-$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth}
EPISODES=${EPISODES:-4000}; HELDOUT=${HELDOUT:-64}; WORKERS=${WORKERS:-8}
PRECISION=${PRECISION:-fp32}   # fp32 = the eval worker's inference() path (autocast off); bf16 = the contract's autocast
case "$PRECISION" in fp32) PREC_FLAG=--fp32;; bf16) PREC_FLAG=--bf16;; *) echo "PRECISION must be fp32|bf16, got $PRECISION"; exit 2;; esac
mn5_require f "$CKPT" f "$CUT3R_DIR/src/record_wgate_features.py" f "$CUT3R_DIR/src/dust3r/wgate/heads.py"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
mkdir -p "$WGATE_ROOT/features"
ARGS=(--ckpt "$CKPT" --root "$WGATE_ROOT" --episodes "$EPISODES" --heldout "$HELDOUT" --num_workers "$WORKERS" "$PREC_FLAG" ${SMOKE:+--smoke} ${EXTRA:-})
echo "Node $(hostname) gpus=$NGPU episodes=$EPISODES heldout=$HELDOUT precision=$PRECISION smoke=${SMOKE:-0} root=$WGATE_ROOT start $(date)"
if [ "$NGPU" -gt 1 ]; then
  accelerate launch --multi_gpu --num_processes "$NGPU" --num_machines 1 --mixed_precision no --main_process_port $((20000 + SLURM_JOB_ID % 20000)) \
    "$CUT3R_DIR/src/record_wgate_features.py" "${ARGS[@]}" || { echo "RECORD FAILED rc=$?"; exit 1; }
else
  python "$CUT3R_DIR/src/record_wgate_features.py" "${ARGS[@]}" || { echo "RECORD FAILED rc=$?"; exit 1; }
fi
echo "train npz: $(ls "$WGATE_ROOT/features/train" 2>/dev/null | grep -c npz)  heldout npz: $(ls "$WGATE_ROOT/features/heldout" 2>/dev/null | grep -c npz)"
echo "done $(date)"
