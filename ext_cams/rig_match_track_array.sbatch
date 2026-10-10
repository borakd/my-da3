#!/bin/bash
# rig_match_track.py on every scene with a v2 birth frame (eps_v2.txt), then rig_trackon_eval.py --paired. Resumable.
# env: TAG (run name), FLAGS (rig_match_track options, e.g. "--matcher disk_epi --tracker cotracker"), PROCS (per task, default 6)
#SBATCH --job-name=rig_match --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/rig_trackon/logs/%x_%A_%a.out
S=/gpfs/scratch/etur59/koc821022; R=$S/outputs/cut3r_eval/ext_cams/rig_trackon; PY=$S/conda_envs/robotseg/bin/python
export TORCH_HOME=$S/torch_hub OMP_NUM_THREADS=2
EV=$S/outputs/droid_birth_frames/entries/entry_events_gap3.csv; V2=$S/outputs/droid_birth_frames/trackon_run_v2/tracks
TR=$R/match_$TAG/tracks; EVAL=$R/match_$TAG/eval; mkdir -p $TR $EVAL; PROCS=${PROCS:-6}
cd $HOME/using_ext_cams/ext_cams
awk -v i=$SLURM_ARRAY_TASK_ID 'NR % 8 == i' $R/eps_v2.txt | xargs -P $PROCS -I{} sh -c '
  [ -f "$0/{}/match_record.json" ] || $1 rig_match_track.py --episode {} --tracks_v2 $2 --events $3 --out $0 $4 > /dev/null 2> $0/{}.err
  [ -s "$5/{}/record.json" ] || $1 rig_trackon_eval.py --episode {} --tracks $0 --events $3 --out $5/{} --tag match_'$TAG' --paired > /dev/null 2> $5/{}.err || echo "FAIL {}"' $TR $PY $V2 $EV "$FLAGS" $EVAL
echo "task $SLURM_ARRAY_TASK_ID done"
