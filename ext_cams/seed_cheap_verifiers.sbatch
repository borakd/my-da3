#!/bin/bash
#SBATCH --job-name=seed_cv --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=02:00:00
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas/cheap_verifiers/logs/sbatch_%A.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas/cheap_verifiers/logs/sbatch_%A.err
# CPU-only seeding (seed_cheap_verifiers.py, idea 4) for a JOBS file: one run per line, "EP [extra seed_cheap_verifiers.py args]",
# e.g.  RAIL+...  --method cv_twotone --twotone 1 --size2 0 --terminal 0 --out_root .../ideas/cv_twotone
# The account has no CPU QoS, so one GPU is requested and SLURM_CPUS_PER_TASK runs go in parallel under OMP_NUM_THREADS=1
# (the script pins cv2 to one thread too). Logs: $LOGDIR/<EP>__<method>.log. Check PRODUCTS afterwards, never exit codes:
#   seedstudy_check_products.py --method cheap_verifiers --root $OUT_ROOT/ideas/cheap_verifiers
#   usage: sbatch seed_cheap_verifiers.sbatch JOBS_FILE
set -u
JOBS=${1:?jobs file}
PY=/home/koc/koc821022/.conda/envs/cuteanything/bin/python
LOGDIR=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas/cheap_verifiers/logs
mkdir -p $LOGDIR
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
export OMP_NUM_THREADS=1 PY LOGDIR
grep -v '^$' "$JOBS" | xargs -P ${SLURM_CPUS_PER_TASK:-20} -L 1 bash -c '
  EP=$0; M=cheap_verifiers; A=("$@"); for ((i=0;i<${#A[@]};i++)); do [ "${A[$i]}" = "--method" ] && M=${A[$((i+1))]}; done
  t0=$(date +%s); "$PY" seed_cheap_verifiers.py --episode "$EP" "$@" > "$LOGDIR/${EP}__${M}.log" 2>&1; rc=$?
  echo "$EP $M rc=$rc $(( $(date +%s) - t0 ))s :: $(tail -n 1 "$LOGDIR/${EP}__${M}.log" | cut -c1-200)"'
echo TASK_DONE
