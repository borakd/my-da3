#!/bin/bash
# Submit run_vggt_variants.sbatch with the right QoS/time.
#   QOS=acc_debug (default; 2h, one job per user)  or  QOS=acc_ehpc (TIME default 1-00:00:00)
#   EP_LIST=<file> [VARIANTS=...] [OUT_ROOT=...] [BATCH=...] [EXTRA_ARGS=...] [TIME=...] [JOB_NAME=...]
# Example:
#   EP_LIST=$WT/vggt_probe/episodes_smoke13.txt QOS=acc_ehpc $WT/vggt_probe/submit_vggt_variants.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
: "${EP_LIST:?EP_LIST is required}"
QOS=${QOS:-acc_debug}
case "$QOS" in
  acc_debug) TIME=${TIME:-02:00:00};;
  acc_ehpc)  TIME=${TIME:-1-00:00:00};;
  *) echo "unknown QOS $QOS" >&2; exit 1;;
esac
mkdir -p "${OUT_ROOT:-/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe}/logs"
export EP_LIST VARIANTS OUT_ROOT BATCH EXTRA_ARGS WT="$(dirname "$HERE")"
exec sbatch --qos="$QOS" --time="$TIME" --job-name="${JOB_NAME:-vggt_probe}" \
  ${OUT_ROOT:+--output="$OUT_ROOT/logs/slurm_%x_%j.out" --error="$OUT_ROOT/logs/slurm_%x_%j.err"} \
  "$HERE/run_vggt_variants.sbatch"
