#!/bin/bash
# Shared MareNostrum5 storage anchors + fail-fast guards for the eval pipeline.
#
# The old cluster kept worktree/checkpoints/outputs/scenes under one $ROOT. On
# MN5 they sit on three filesystems, and every launcher used to carry its own
# copy of the anchor block -- nine copies that drifted and all defaulted to
# /scratch/bdursun25/..., which does not exist here. This is the single copy.
#
# Source it AFTER setting $WT (the my-da3 checkout). Every value is overridable
# from the environment, so a relocation is one export, not a sed over nine files.
#
#   WT           the my-da3 checkout (caller sets it; validated here)
#   CKPT_ROOT    training checkpoints
#   OUT_ROOT     outputs root; $OUT = $OUT_ROOT/cut3r_eval
#   DATA_ROOT    DROID episode store
#   SCENES_ROOT  4292-scene test split (the full-benchmark scene root)
#   OVERFIT_ROOT/OVERFIT_SCENE   single-episode overfit scene (GRU grid drivers)
#   EVAL_SCRIPT  eval_depth_poses.py -- vendored in-repo since the upstream
#                streaming-3d checkout is not on MN5
#
# Helpers: mn5_require (fail loudly on missing inputs) and mn5_require_gpus.
# These matter because the runners deliberately use `set -o pipefail` WITHOUT
# `set -e` (the login shell rc is corrupt, and MKL activation reads unset vars),
# so an unguarded missing path used to sail past and produce an empty run.

: "${WT:?mn5_paths.sh: \$WT must be set before sourcing}"
if [ ! -f "$WT/src/CUT3R/src/train_cut3r_baseline.py" ]; then
  echo "ERROR: \$WT is not a my-da3 checkout: $WT" >&2
  return 1 2>/dev/null || exit 1
fi

CKPT_ROOT=${CKPT_ROOT:-/gpfs/scratch/etur59/koc821022/checkpoints_projects}
OUT_ROOT=${OUT_ROOT:-/gpfs/scratch/etur59/koc821022/outputs}
DATA_ROOT=${DATA_ROOT:-/gpfs/scratch/etur59/koc821022}

# NOTE: no `scenes/` component on MN5 -- the splits tree sits directly under
# DATA_ROOT. SCENES_ROOT is overridable in its own right, so a differently
# shaped store needs no DATA_ROOT gymnastics.
SCENES_ROOT=${SCENES_ROOT:-$DATA_ROOT/pointworld_droid_splits/test/dl3dv_multi/wrist}

# The overfit episode the PoseGRU grid drivers evaluate. The old /frozen layout
# nested it one level deeper (<episode>/13062452+wrist/dense); on MN5 the VALAR
# store puts dense/ directly under the episode, so OVERFIT_SCENE has no cam
# component. Anything joining these must use "$OVERFIT_ROOT/$OVERFIT_SCENE".
OVERFIT_ROOT=${OVERFIT_ROOT:-$DATA_ROOT/pointworld_droid_wrist_VALAR}
OVERFIT_SCENE=${OVERFIT_SCENE:-RAIL+eh61f232+2023-10-26-17h-33m-59s}

EVAL_SCRIPT=${EVAL_SCRIPT:-$WT/eval_bundle/bin/eval_depth_poses.py}
OUT=${OUT:-$OUT_ROOT/cut3r_eval}
SCENE_LIST=${SCENE_LIST:-$OUT/scene_list.txt}
CUT3R_DIR=$WT/src/CUT3R

export DL3DV_CACHE_DIR=${DL3DV_CACHE_DIR:-$DATA_ROOT/.dl3dv_cache}

# mn5_require <kind> <path> [<kind> <path> ...]
# kind: f = regular file, d = directory, s = non-empty file.
# Reports EVERY missing input before exiting, so one submit surfaces the whole
# list instead of one path per failed job.
mn5_require () {
  local missing=0 kind path
  while [ $# -gt 0 ]; do
    kind=$1; path=$2; shift 2
    case "$kind" in
      f) [ -f "$path" ] || { echo "MISSING FILE:      $path" >&2; missing=1; } ;;
      d) [ -d "$path" ] || { echo "MISSING DIRECTORY: $path" >&2; missing=1; } ;;
      s) [ -s "$path" ] || { echo "MISSING/EMPTY:     $path" >&2; missing=1; } ;;
      *) echo "mn5_require: unknown kind '$kind'" >&2; missing=1 ;;
    esac
  done
  if [ "$missing" -ne 0 ]; then
    echo "ERROR: required eval inputs are absent -- see eval_bundle/MN5_EVAL_README.md" >&2
    exit 1
  fi
}

# Refuse to run inference outside SLURM. The MN5 login nodes DO expose GPUs, so
# a launcher started on one does not fail the GPU check -- it launches real
# workers that the cgroup kills ~20s in, *after* they have created claim/pred/
# eval dirs. A claim is released only on a caught Python exception, so those
# scenes are then silently skipped by every later run. Call this BEFORE any
# mkdir. Set ALLOW_NO_SLURM=1 to override deliberately.
mn5_require_slurm () {
  if [ -z "$SLURM_JOB_ID" ] && [ "$ALLOW_NO_SLURM" != "1" ]; then
    echo "ERROR: not running under SLURM. Never run inference on a login node --" >&2
    echo "       MN5 login nodes have GPUs, so the workers start and are killed" >&2
    echo "       mid-scene, leaving stale claim dirs that permanently skip those" >&2
    echo "       scenes. Use sbatch (or ALLOW_NO_SLURM=1 under salloc/srun)." >&2
    exit 1
  fi
}

# Guard the silent no-op: with no --gres, nvidia-smi reports 0 GPUs, the
# per-GPU `seq 0 -1` loop body never runs, and the job exits 0 having done
# nothing. Echoes the count so the log records it.
mn5_require_gpus () {
  local n
  mn5_require_slurm
  n=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [ "${n:-0}" -lt 1 ]; then
    echo "ERROR: no GPUs visible. Pass --gres=gpu:N on the sbatch command line." >&2
    exit 1
  fi
  echo "$n"
}
