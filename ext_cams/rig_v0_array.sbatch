#!/bin/bash
# rig_v0.py over a scene list. Resumable. env: TAG (run name), FLAGS (rig_v0 options, e.g. "--source labels --links base"),
# LIST (scene list, default rig_v0/sample100.txt), PROCS (parallel scenes per task, default 20).
#SBATCH --job-name=rig_v0 --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/rig_v0/logs/%x_%A_%a.out
S=/gpfs/scratch/etur59/koc821022; V=$S/outputs/cut3r_eval/ext_cams/rig_v0; PY=$S/conda_envs/robotseg/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1; LIST=${LIST:-$V/sample100.txt}; OUT=$V/$TAG; mkdir -p $OUT; PROCS=${PROCS:-20}
cd $HOME/using_ext_cams/ext_cams
awk -v i=$SLURM_ARRAY_TASK_ID -v n=$SLURM_ARRAY_TASK_COUNT 'NR % n == i' $LIST | xargs -P $PROCS -I{} sh -c '
  [ -s "$0/{}/record.json" ] || $1 rig_v0.py --episode {} --out $0/{} --tag '$TAG' $2 > /dev/null 2> $0/{}.err || echo "FAIL {}"' $OUT $PY "$FLAGS"
echo "task $SLURM_ARRAY_TASK_ID done"
