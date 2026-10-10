#!/bin/bash
#SBATCH --job-name=rig_diag --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/rig_trackon/logs/%x_%A_%a.out
S=/gpfs/scratch/etur59/koc821022; R=$S/outputs/cut3r_eval/ext_cams/rig_trackon; PY=$S/conda_envs/robotseg/bin/python
cd $HOME/using_ext_cams/ext_cams
awk -v i=$SLURM_ARRAY_TASK_ID 'NR % 4 == i' $R/eps_v2.txt | xargs -P 20 -I{} sh -c '[ -s "$0/diag/{}.csv" ] || OMP_NUM_THREADS=1 $1 rig_trackon_diag.py --episode {} --tracks $2/outputs/droid_birth_frames/trackon_run_v2/tracks --events $2/outputs/droid_birth_frames/entries/entry_events_gap3.csv --out $0/diag/{}.csv > /dev/null 2> $0/diag/{}.err || echo "FAIL {}"' $R $PY $S
echo "task $SLURM_ARRAY_TASK_ID done"
