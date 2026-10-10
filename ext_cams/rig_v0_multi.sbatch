#!/bin/bash
# Several rig_v0 configurations over one scene list in ONE job (acc_debug limits array jobs). Resumable.
# env: CFGS = "TAG1:FLAGS1;TAG2:FLAGS2;..." , LIST (default rig_v0/sample100.txt), PROCS (default 80)
#SBATCH --job-name=rig_v0m --account=etur59 --partition=acc --qos=acc_debug --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=80 --time=02:00:00
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/rig_v0/logs/%x_%j.out
S=/gpfs/scratch/etur59/koc821022; V=$S/outputs/cut3r_eval/ext_cams/rig_v0; PY=$S/conda_envs/robotseg/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1; LIST=${LIST:-$V/sample100.txt}; PROCS=${PROCS:-80}
cd $HOME/using_ext_cams/ext_cams
IFS=';' read -ra C <<< "$CFGS"
for cfg in "${C[@]}"; do TAG=${cfg%%:*}; FL=${cfg#*:}; mkdir -p $V/$TAG; while read -r ep; do echo "$TAG|$FL|$ep"; done < $LIST; done |
  xargs -P $PROCS -I{} sh -c 'IFS="|"; set -- $1; IFS=" "; [ -s "'$V'/$1/$3/record.json" ] || '$PY' rig_v0.py --episode $3 --out "'$V'/$1/$3" --tag $1 $2 > /dev/null 2> "'$V'/$1/$3.err" || echo "FAIL $1 $3"' _ {}
echo done
