#!/bin/bash
# Exterior-rig ray conditioning: 5 arms on the identical rig windows (scenes the rig solved, frames t_ref..FK exit).
#   bash eval_pipeline/submit_rigray_eval.sh [ARM ...]      (no args = all five; NODES=4 per arm by default)
set -o pipefail
WT=/gpfs/home/koc/koc821022/my-da3; export OUT_ROOT=/gpfs/scratch/etur59/koc821022/outputs; OUT=$OUT_ROOT/cut3r_eval
CK=/gpfs/scratch/etur59/koc821022/checkpoints_projects/cut3r_finetune_baselines
PRE=$WT/src/CUT3R/src/cut3r_512_dpt_4_64.pth
RIG_DIR=${RIG_DIR:-$OUT/ext_cams/rig_trackon/v2_cut}
SCENE_LIST=${SCENE_LIST:-$OUT/ext_cams/rig_trackon/eps_rig_solved.txt}
NODES=${NODES:-4}; LIMIT=${LIMIT:-0}; QOS=${QOS:-acc_ehpc}; GPUS=${GPUS:-4}
unset PREV_PRED_RAY_SHUFFLE GT_RAY_MAP_SHUFFLE GT_RAY_NOISE REVERSE REVISIT SKIP_FRAMES_FILE STATE_GATE_MODE STATE_GATE_TAU STATE_GATE_TAU_FILE
declare -A CKPT=( [rigray_ft]=$CK/cut3r_finetune_aug_full_gtray_32gpu_lr1e5/checkpoint-final.pth
                  [rigray_zs]=$PRE
                  [win_gtray_ft]=$CK/cut3r_finetune_aug_full_gtray_32gpu_lr1e5/checkpoint-final.pth
                  [win_augfull_ft]=$CK/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth
                  [win_zs_none]=$PRE )
declare -A COND=( [rigray_ft]=gt [rigray_zs]=gt_all [win_gtray_ft]=gt [win_augfull_ft]=none [win_zs_none]=none_ts )
declare -A POSE=( [rigray_ft]=rig [rigray_zs]=rig [win_gtray_ft]=gt [win_augfull_ft]=gt [win_zs_none]=gt )
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(rigray_ft rigray_zs win_gtray_ft win_augfull_ft win_zs_none)
[ -s "$SCENE_LIST" ] || { for d in $RIG_DIR/*/; do [ -f "$d/anchors.npz" ] && basename "$d"; done | sort > "$SCENE_LIST"; echo "scene list: $(wc -l < $SCENE_LIST) scenes"; }
mkdir -p $OUT/logs; cd $WT || exit 1
for L in "${ARMS[@]}"; do
  [ -n "${CKPT[$L]}" ] || { echo "unknown arm $L" >&2; exit 1; }
  LABEL=${LABEL_PREFIX:-}$L${LABEL_SUFFIX:-}
  echo "=== $LABEL cond=${COND[$L]} pose=${POSE[$L]} ckpt=${CKPT[$L]}"
  for ((n = 0; n < NODES; n++)); do
    sbatch --qos=$QOS --gres=gpu:$GPUS --cpus-per-task=$((GPUS * 20)) ${TIME:+--time=$TIME} --job-name="rigray_${L}_n$n" \
      --output="$OUT/logs/slurm_${LABEL}_n${n}_%j.out" --error="$OUT/logs/slurm_${LABEL}_n${n}_%j.err" \
      --export=ALL,OUT_ROOT="$OUT_ROOT",LABEL="$LABEL",CKPT="${CKPT[$L]}",CONDITIONING="${COND[$L]}",POSE_SOURCE="${POSE[$L]}",GAP="${GAP:-hold}",RIG_DIR="$RIG_DIR",SCENE_LIST="$SCENE_LIST",BASE_SHARD=$((n * GPUS)),NUM_SHARDS=$((NODES * GPUS)),LIMIT="$LIMIT" \
      eval_pipeline/run_rigray_eval_node.sh
  done
done
