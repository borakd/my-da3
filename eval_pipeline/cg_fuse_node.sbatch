#!/bin/bash
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=06:00:00
#SBATCH --job-name=cg_fuse_full
# Full-harness fwd/bwd fusion: CPU-bound (npz + per-scene eval subprocess);
# runs 16 parallel shards of cg_fuse_fwd_bwd.py on one node. GPUs unused but
# MN5 acc partition allocates by GPU. Env (via --export): FWD_LABEL,
# BWD_LABEL, FUSED_LABEL, SCENE_LIST, plus optional MAX_ANGLE/MAX_RESID.
set -eo pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
cd "$WT"
for s in $(seq 0 15); do
  python eval_pipeline/cg_fuse_fwd_bwd.py \
    --fwd_label "$FWD_LABEL" --bwd_label "$BWD_LABEL" \
    --fused_label "$FUSED_LABEL" --scene_list "$SCENE_LIST" \
    --shard_id "$s" --num_shards 16 \
    --max_angle_deg "${MAX_ANGLE:-75}" --max_resid_frac "${MAX_RESID:-1.5}" &
done
wait
echo "FUSION NODE DONE"
