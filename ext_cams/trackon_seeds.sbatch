#!/bin/bash
# Track-On-R on the guided birth-frame seeds (trackon_seeds.py). env: LIST (episode list), OUT (raw tracks dir), NSH (shards per GPU task, default 1)
#SBATCH --job-name=to_seeds --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=08:00:00
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/rig_anchor/logs/%x_%A_%a.out
S=/gpfs/scratch/etur59/koc821022
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate $S/conda_envs/track_on_r
export HF_HUB_OFFLINE=1 TRACKON_ROOT=$HOME/track_on PYTHONNOUSERSITE=1
NT=${SLURM_ARRAY_TASK_COUNT:-1}; TI=${SLURM_ARRAY_TASK_ID:-0}; NSH=${NSH:-1}
cd $HOME/track_on
for j in $(seq 0 $((NSH-1))); do
  python $HOME/using_ext_cams/ext_cams/trackon_seeds.py --episodes $LIST --seeds $S/outputs/cut3r_eval/ext_cams/rig_trackon/match_ct/tracks \
    --out $OUT --shard $((TI*NSH+j)) --nshards $((NT*NSH)) &
done
wait
