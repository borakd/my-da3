#!/bin/bash
# Noise-tolerance oracle curve launcher (Step 1, GRU v4 campaign).
#
# Submits the pre-registered grid as 1-node diag evals of checkpoint B
# (gtray_lr1e5, --conditioning gt) on the fixed 430-scene subset:
#   - diag_noise_clean          sigma=0, hook DISABLED (exact pairing base)
#   - diag_noise_sig0check      sigma=0, hook ENABLED (byte-identity sanity)
#   - diag_noise_{white,walk}_m{0125,0250,0500,1000}
# Nominal per-axis sigmas are grid_multiplier * sigma_ref / 1.5382 (chi-3
# median), so the REALIZED median injected error equals the grid point, which
# is calibrated as a median (noise_oracle_calibration.json).
# diag_ labels are never aggregated into averages_table.csv.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1

CKPT=${CKPT:-/gpfs/scratch/etur59/koc821022/checkpoints_projects/cut3r_finetune_baselines/cut3r_finetune_aug_full_gtray_32gpu_lr1e5/checkpoint-final.pth}
SUBSET=$PWD/eval_pipeline/noise_oracle_subset_430.txt
OUT_ROOT=/gpfs/scratch/etur59/koc821022/outputs
SIGMA_T=0.07753853660694085
SIGMA_R=36.67527478716412
CHI3=1.5382
SEED=1000

[ -f "$CKPT" ]   || { echo "missing ckpt: $CKPT" >&2; exit 1; }
[ -s "$SUBSET" ] || { echo "missing subset list: $SUBSET" >&2; exit 1; }

submit () { # label mode sig_t sig_r
  local label=$1 mode=$2 st=$3 sr=$4 exports
  exports="ALL,OUT_ROOT=$OUT_ROOT,LABEL=$label,CONDITIONING=gt,CKPT=$CKPT"
  exports="$exports,SCENE_LIST=$SUBSET,BASE_SHARD=0,NUM_SHARDS=4,LIMIT=0"
  if [ -n "$mode" ]; then
    exports="$exports,GT_RAY_NOISE_MODE=$mode,GT_RAY_NOISE_T=$st,GT_RAY_NOISE_R_DEG=$sr,GT_RAY_NOISE_SEED=$SEED"
  fi
  sbatch --time=04:00:00 --job-name="nz_$label" \
    --output="$OUT_ROOT/cut3r_eval/logs/slurm_${label}_%j.out" \
    --error="$OUT_ROOT/cut3r_eval/logs/slurm_${label}_%j.err" \
    --export="$exports" \
    eval_pipeline/run_captain_ray_eval_node.sh || exit 1
}

submit diag_noise_clean "" "" ""
submit diag_noise_sig0check white 0 0

for m in 0.125 0.25 0.5 1.0; do
  tag=$(python3 -c "print(f'{int(float('$m')*1000):04d}')")
  st=$(python3 -c "print($SIGMA_T*$m/$CHI3)")
  sr=$(python3 -c "print($SIGMA_R*$m/$CHI3)")
  submit "diag_noise_white_m$tag" white "$st" "$sr"
  submit "diag_noise_walk_m$tag"  walk  "$st" "$sr"
done
echo "submitted 10 noise-oracle jobs"
