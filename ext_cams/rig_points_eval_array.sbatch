#!/bin/bash
# rig_points_eval.py (label windows, >= 3 shared points, all links) on every evaluable test scene; resumable per scene.
#SBATCH --job-name=rig_points --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/gpfs/scratch/etur59/koc821022/gripper_points/logs/rig_points_%A_%a.out
source /gpfs/scratch/etur59/koc821022/gripper_points/env.sh
cd $HOME/using_ext_cams/ext_cams
EPS=/gpfs/scratch/etur59/koc821022/gripper_points/${EPS_FILE:-rig_eval_test_eps.txt}; OUT=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/rig_points/test_label_windows; POS=/gpfs/scratch/etur59/koc821022/gripper_points/positions_v1/positions
awk -v i=$SLURM_ARRAY_TASK_ID 'NR % 8 == i' $EPS | xargs -P 20 -I{} sh -c '[ -s "$0/{}/record.json" ] || OMP_NUM_THREADS=1 $1 rig_points_eval.py --episode {} --positions $2 --out $0/{} --tag label_windows > /dev/null 2> $0/{}.err || echo "FAIL {}"' $OUT $PY $POS
echo "task $SLURM_ARRAY_TASK_ID done"
