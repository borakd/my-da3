#!/bin/bash
# CUT3R feed probe: run up to 4 infer_and_eval_worker_ray.py configs in parallel
# (one per GPU of ONE node) over the 13 smoke scenes.
#
#   sbatch [--qos=acc_debug --time=02:00:00] $WT/vggt_probe/run_cut3r_feed.sbatch <config.txt>
#
# Config file: one worker per line, '#' comments allowed, fields separated by '|':
#   label|ckpt|conditioning|scenes_root|gt_scenes_root|context_dir_or_-
#     label            output subdir: $OUT/vggt_probe/<label>/{preds,eval}
#     ckpt             checkpoint-final.pth (a run DIRECTORY is accepted: checkpoint-final.pth is appended)
#     conditioning     gt | prev_gt | prev_pred | none
#     scenes_root      frames + conditioning cam npz (store root or a pseudo_scenes/<variant>_<scale>)
#     gt_scenes_root   evaluator GT root, or '-' (= scenes_root)
#     context_dir      <root>/<scene>/*.png prepended as unscored context, or '-'
# More than 4 lines are processed in waves of 4. Env overrides: OUT_ROOT, SCENE_LIST,
# WT (defaults to the submit dir), QOS is a header default — override with sbatch --qos.
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=06:00:00
#SBATCH --job-name=cut3r_feed
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/logs/slurm_cut3r_feed_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/logs/slurm_cut3r_feed_%j.err

set -o pipefail
CONFIG=${1:?usage: sbatch run_cut3r_feed.sbatch <config.txt>}
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ] || { echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2; exit 1; }
[ -s "$CONFIG" ] || { echo "ERROR: config file missing/empty: $CONFIG" >&2; exit 1; }

source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
export PYTHONNOUSERSITE=1
# Resolve the scene list BEFORE mn5_paths.sh: it defaults SCENE_LIST to the full
# 4292-scene $OUT/scene_list.txt when unset (bit us on job 45534412).
SCENE_LIST=${SCENE_LIST:-$WT/vggt_probe/smoke13_scenes.txt}
source "$WT/eval_pipeline/mn5_paths.sh"          # OUT, EVAL_SCRIPT, CUT3R_DIR (all overridable)
PROBE=$OUT/vggt_probe
LOGS=$PROBE/logs
WORKER=$WT/eval_pipeline/infer_and_eval_worker_ray.py
mn5_require f "$WORKER" f "$EVAL_SCRIPT" s "$SCENE_LIST" d "$CUT3R_DIR"
NGPU=$(mn5_require_gpus)
mkdir -p "$LOGS"
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job=$SLURM_JOB_ID node=$(hostname) gpus=$NGPU config=$CONFIG scenes=$(grep -c . "$SCENE_LIST") start=$(date)"

# --- parse config -------------------------------------------------------------
LABELS=(); CKPTS=(); CONDS=(); SROOTS=(); GROOTS=(); CTXS=()
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%%#*}"; line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [ -z "$line" ] && continue
  IFS='|' read -r label ckpt cond sroot groot ctx <<<"$line"
  for v in label ckpt cond sroot groot ctx; do eval "$v=\"\$(echo \"\$$v\" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')\""; done
  [ -d "$ckpt" ] && ckpt="$ckpt/checkpoint-final.pth"
  [ -f "$ckpt" ] || { echo "ERROR: [$label] ckpt not found: $ckpt" >&2; exit 1; }
  case "$cond" in gt|prev_gt|prev_pred|none) ;; *) echo "ERROR: [$label] bad conditioning '$cond'" >&2; exit 1;; esac
  [ -d "$sroot" ] || { echo "ERROR: [$label] scenes_root missing: $sroot" >&2; exit 1; }
  if [ -n "$groot" ] && [ "$groot" != "-" ]; then [ -d "$groot" ] || { echo "ERROR: [$label] gt_scenes_root missing: $groot" >&2; exit 1; }; else groot=""; fi
  if [ -n "$ctx" ] && [ "$ctx" != "-" ]; then [ -d "$ctx" ] || { echo "ERROR: [$label] context_dir missing: $ctx" >&2; exit 1; }; else ctx=""; fi
  LABELS+=("$label"); CKPTS+=("$ckpt"); CONDS+=("$cond"); SROOTS+=("$sroot"); GROOTS+=("$groot"); CTXS+=("$ctx")
done < "$CONFIG"
N=${#LABELS[@]}
[ "$N" -gt 0 ] || { echo "ERROR: no worker lines in $CONFIG" >&2; exit 1; }
echo "$N worker config(s); running $NGPU at a time"

# --- run in waves of NGPU -------------------------------------------------------
declare -A RC
i=0
while [ $i -lt $N ]; do
  PIDS=(); WLAB=()
  for g in $(seq 0 $((NGPU-1))); do
    [ $i -ge $N ] && break
    label=${LABELS[$i]}
    extra=()
    [ -n "${GROOTS[$i]}" ] && extra+=(--gt_scenes_root "${GROOTS[$i]}")
    [ -n "${CTXS[$i]}" ] && extra+=(--context_dir "${CTXS[$i]}")
    log=$LOGS/${label}_${SLURM_JOB_ID}.log
    echo "[$label] gpu=$g cond=${CONDS[$i]} ckpt=${CKPTS[$i]} scenes_root=${SROOTS[$i]} gt=${GROOTS[$i]:-<scenes_root>} ctx=${CTXS[$i]:--} log=$log"
    CUDA_VISIBLE_DEVICES=$g python "$WORKER" \
      --ckpt "${CKPTS[$i]}" --label "$label" --size 320 --conditioning "${CONDS[$i]}" \
      --scenes_root "${SROOTS[$i]}" --scene_list "$SCENE_LIST" \
      --pred_base "$PROBE/$label/preds" --eval_base "$PROBE/$label/eval" \
      --eval_script "$EVAL_SCRIPT" --cut3r_dir "$CUT3R_DIR" \
      --shard_id 0 --num_shards 1 --device cuda "${extra[@]}" > "$log" 2>&1 &
    PIDS+=($!); WLAB+=("$label")
    i=$((i+1))
  done
  for k in "${!PIDS[@]}"; do
    wait "${PIDS[$k]}"; RC[${WLAB[$k]}]=$?
    echo "[${WLAB[$k]}] worker exited rc=${RC[${WLAB[$k]}]} at $(date)"
  done
done

# --- completion counts ----------------------------------------------------------
TOTAL=$(grep -c . "$SCENE_LIST")
echo "=== completion (eval CSVs present / $TOTAL scenes) ==="
for label in "${LABELS[@]}"; do
  n=0
  while IFS= read -r sc; do
    [ -z "$sc" ] && continue
    [ -s "$PROBE/$label/eval/$sc/eval_depth_pose_metrics.csv" ] && n=$((n+1))
  done < "$SCENE_LIST"
  nf=0; for f in "$PROBE/$label/eval"/_failures_shard*.txt; do [ -f "$f" ] && nf=$((nf + $(grep -c . "$f"))); done
  echo "[$label] done=$n/$TOTAL failures_logged=$nf rc=${RC[$label]:-?}"
done
echo "end=$(date)"
