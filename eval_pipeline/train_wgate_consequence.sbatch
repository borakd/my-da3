#!/bin/bash
# Variant C of WRITE_GATE_VARIANTS.md: train a write gate by its CONSEQUENCE (causal trajectory
# error backpropagated through the memory into a per-frame write weight). Frozen augfull backbone,
# only the GateWeightHead trains; TBPTT over CHUNK-frame chunks, 4 GPUs x batch 1 via accelerate.
#
#   NAME=cg1_head SITE=state              sbatch eval_pipeline/train_wgate_consequence.sbatch   # C1
#   NAME=cg2_head SITE=mem                sbatch eval_pipeline/train_wgate_consequence.sbatch   # C2
#   NAME=cg1_head_lr5 SITE=state LR=1e-5  sbatch eval_pipeline/train_wgate_consequence.sbatch
#   NAME=cg1_rpe SITE=state EXTRA="--loss ate_rpe --rpe_w 0.1" sbatch ...
#   RESUME=1 NAME=cg1_head SITE=state     sbatch ...          # continue from <OUT>/<NAME>/gate_head.pth
#   SMOKE=1 NAME=cgsmoke SITE=state sbatch -q acc_debug --gres=gpu:1 -c 20 --time=00:40:00 \
#       eval_pipeline/train_wgate_consequence.sbatch          # 2 episodes, 1 chunk each, 1 held-out
#     (acc_debug = ONE 1-GPU job per user; the 4-GPU / 80-CPU defaults below are for acc_ehpc)
#   PRECISION=bf16 ...   # rollout under bf16 autocast (faster); default fp32 = the eval worker's path
#
# Env: NAME (required unless SMOKE) SITE={state,mem} LR EPISODES CHUNK WMIN INIT_LOGIT LOSS RPE_W
#      WGATE_OUT PRECISION WORKERS EXTRA RESUME SMOKE CKPT.
# Output: $WGATE_OUT/$NAME/{gate_head.pth,gate_head_best.pth,gate_head_opt.pth,log.txt,eval.json}
#         (gate_head.pth carries mode="weight" -> the eval worker's frame_gate/mem_gate "head" key).
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=2-00:00:00
#SBATCH --job-name=wgcons
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_wgcons_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_wgcons_%j.err
set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything || { echo "conda activate cuteanything FAILED"; exit 1; }
source "$WT/eval_pipeline/mn5_paths.sh" || exit 1
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMEXPR_MAX_THREADS=64

SITE=${SITE:-state}
case "$SITE" in state|mem) ;; *) echo "SITE must be state|mem, got $SITE"; exit 2;; esac
NAME=${NAME:-${SMOKE:+cgsmoke}}
[ -n "$NAME" ] || { echo "NAME is required (outputs go to \$WGATE_OUT/\$NAME)"; exit 2; }
WGATE_OUT=${WGATE_OUT:-/gpfs/scratch/etur59/koc821022/checkpoints/wgate/consequence}
case "$WGATE_OUT" in
  /gpfs/projects/*) echo "refusing to write to /gpfs/projects (see CLAUDE.md)"; exit 2;;
  /scratch/*|/tmp/*) echo "refusing to write to $WGATE_OUT: /scratch and /tmp are NODE-LOCAL NVMe" \
                          "on MN5 (invisible to every other node) -- use /gpfs/scratch/..."; exit 2;;
esac
CKPT=${CKPT:-$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth}
LR=${LR:-1e-4}; EPISODES=${EPISODES:-38633}; CHUNK=${CHUNK:-16}
WMIN=${WMIN:-0.5}; INIT_LOGIT=${INIT_LOGIT:-2.0}
LOSS=${LOSS:-ate}; case "$LOSS" in ate|ate_rpe) ;; *) echo "LOSS must be ate|ate_rpe"; exit 2;; esac
RPE_W=${RPE_W:-0.1}; WORKERS=${WORKERS:-8}
PRECISION=${PRECISION:-fp32}
case "$PRECISION" in fp32|bf16) ;; *) echo "PRECISION must be fp32|bf16, got $PRECISION"; exit 2;; esac

mn5_require f "$CKPT" f "$CUT3R_DIR/src/train_wgate_consequence.py" \
  f "$CUT3R_DIR/src/dust3r/wgate/heads.py" f "$CUT3R_DIR/src/dust3r/wgate/traj_loss.py"
mkdir -p "$WGATE_OUT/$NAME" || exit 1

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
[ "$NGPU" -ge 1 ] || { echo "no GPUs visible -- pass --gres=gpu:N"; exit 1; }
ARGS=(--name "$NAME" --site "$SITE" --out "$WGATE_OUT" --ckpt "$CKPT" --lr "$LR" \
      --episodes "$EPISODES" --chunk "$CHUNK" --wmin "$WMIN" --init_logit "$INIT_LOGIT" \
      --loss "$LOSS" --rpe_w "$RPE_W" --num_workers "$WORKERS" --precision "$PRECISION" \
      ${RESUME:+--resume} ${SMOKE:+--smoke} ${EXTRA:-})
echo "Node $(hostname) gpus=$NGPU name=$NAME site=$SITE lr=$LR episodes=$EPISODES chunk=$CHUNK" \
     "loss=$LOSS precision=$PRECISION resume=${RESUME:-0} smoke=${SMOKE:-0} out=$WGATE_OUT start $(date)"

# CPU self-test first: it asserts that Component A's GateWeightHead / attach_*(train=True) and
# Component B's traj_loss names are present before 4 GPUs are spent finding out they are not.
python "$CUT3R_DIR/src/train_wgate_consequence.py" --selftest \
  || { echo "SELFTEST FAILED rc=$? -- not launching"; exit 1; }

if [ "$NGPU" -gt 1 ]; then
  accelerate launch --multi_gpu --num_processes "$NGPU" --num_machines 1 --mixed_precision no \
    --main_process_port $((20000 + SLURM_JOB_ID % 20000)) \
    "$CUT3R_DIR/src/train_wgate_consequence.py" "${ARGS[@]}" \
    || { echo "TRAIN FAILED rc=$?"; exit 1; }
else
  python "$CUT3R_DIR/src/train_wgate_consequence.py" "${ARGS[@]}" \
    || { echo "TRAIN FAILED rc=$?"; exit 1; }
fi
echo "done $(date)"
ls -l "$WGATE_OUT/$NAME"
