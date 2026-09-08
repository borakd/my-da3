#!/bin/bash
# VGGT-Omega probe runner: ONE GPU, loops over an episode list x a variant list,
# calling extract_vggt_omega.py once per (episode, variant). One failure never
# stops the loop (logged to $LOG_DIR/FAILURES_<jobid>.txt, then continue).
#
# Submit through submit_vggt_variants.sh (picks --qos/--time from $QOS) or directly:
#   sbatch --qos=acc_debug --time=02:00:00 run_vggt_variants.sbatch
#   sbatch --qos=acc_ehpc  --time=1-00:00:00 run_vggt_variants.sbatch
#
# Environment (all optional except EP_LIST):
#   EP_LIST    file, one episode per line:  EP  EXT1_SPEC  EXT2_SPEC  [WRIST_SRC]
#              EXT*_SPEC = PNG path or raw:<mp4>[:frame]; WRIST_SRC = store | raw:<mp4>
#              (lines starting with # are skipped)
#   VARIANTS   space-separated: "anchored_t0 wrist_pair offline_all" ... each item may be
#              "variant=out_name" to write under a different name (default: all six)
#   OUT_ROOT   default /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe
#   BATCH      forwards per call for per-t variants (default 8)
#   EXTRA_ARGS extra CLI args for extract_vggt_omega.py (e.g. "--force" or "--no_eval")
#
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=vggt_probe
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/logs/slurm_%x_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/logs/slurm_%x_%j.err
set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# allow submission from the vggt_probe dir itself
[ -f "$WT/vggt_probe/extract_vggt_omega.py" ] || WT="$(dirname "$WT")"
[ -f "$WT/vggt_probe/extract_vggt_omega.py" ] || { echo "ERROR: cannot locate vggt_probe/extract_vggt_omega.py from WT=$WT" >&2; exit 1; }
[ -n "$SLURM_JOB_ID" ] || { echo "ERROR: run under Slurm (sbatch), never on a login node" >&2; exit 1; }
: "${EP_LIST:?EP_LIST (episode list file) is required}"
[ -s "$EP_LIST" ] || { echo "ERROR: EP_LIST missing/empty: $EP_LIST" >&2; exit 1; }

source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate vggt-omega
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_ROOT=${OUT_ROOT:-/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe}
VARIANTS=${VARIANTS:-"anchored_t0 anchored_sliding wrist_pair ext_first offline_all offline_all_ext"}
BATCH=${BATCH:-8}
LOG_DIR=$OUT_ROOT/logs
mkdir -p "$LOG_DIR"
FAIL_LOG=$LOG_DIR/FAILURES_$SLURM_JOB_ID.txt
SCRIPT=$WT/vggt_probe/extract_vggt_omega.py

echo "job $SLURM_JOB_ID on $(hostname) $(date) | WT=$WT | EP_LIST=$EP_LIST | VARIANTS=$VARIANTS | OUT_ROOT=$OUT_ROOT | BATCH=$BATCH"
nvidia-smi -L
n_ok=0; n_fail=0
while read -r ep ext1 ext2 wsrc _rest; do
  [ -z "$ep" ] && continue
  case "$ep" in \#*) continue;; esac
  wsrc=${wsrc:-store}
  for v in $VARIANTS; do
    variant=${v%%=*}; out_name=${v#*=}
    [ "$out_name" = "$v" ] && out_name=$variant
    log=$LOG_DIR/${out_name}__${ep}.log
    ext_args=()
    [ -n "$ext1" ] && [ "$ext1" != "-" ] && ext_args+=(--ext1 "$ext1")
    [ -n "$ext2" ] && [ "$ext2" != "-" ] && ext_args+=(--ext2 "$ext2")
    echo "=== $(date +%H:%M:%S) $out_name ($variant) $ep wrist=$wsrc"
    if python "$SCRIPT" --ep "$ep" --variant "$variant" --out_name "$out_name" --out_root "$OUT_ROOT" \
         --wrist_src "$wsrc" "${ext_args[@]}" --batch "$BATCH" $EXTRA_ARGS >"$log" 2>&1; then
      n_ok=$((n_ok+1)); grep -h "MEAN\|inference done\|SKIP" "$log" | tail -4
    else
      n_fail=$((n_fail+1)); echo "FAIL $out_name $ep (see $log)"; tail -5 "$log"
      echo -e "$out_name\t$ep\t$(date +%FT%T)\t$log" >> "$FAIL_LOG"
    fi
  done
done < "$EP_LIST"
echo "=== DONE ok=$n_ok fail=$n_fail $(date)"
[ $n_fail -eq 0 ]
