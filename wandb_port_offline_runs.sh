#!/bin/bash
# LEGACY PATH — prefer /gpfs/scratch/etur59/koc821022/checkpoints/sync_nomedia.sh
# for anything but small/young runs: glogin1's hard 300s RLIMIT_CPU SIGKILLs
# (rc=137) this script's `wandb sync` while it replays large .wandb logs, a
# failure --no-media does NOT avoid. sync_nomedia.sh (rsync snapshot excluding
# files/media -> `wandb beta sync`, Go core) is the measured fix. See its header.
#
# Port W&B runs recorded OFFLINE on MareNostrum5 compute nodes up to wandb.ai,
# from the LOGIN node (the only place with outbound internet).
#
#   ./wandb_port_offline_runs.sh <run_output_dir> [more dirs ...]
#   ./wandb_port_offline_runs.sh --all            # every offline run under CKPT_ROOT
#   ./wandb_port_offline_runs.sh --dry-run <dir>  # list what would be synced
#   ./wandb_port_offline_runs.sh --no-media <dir> # skip files/media/** (the
#                                 # print_img_freq PNGs, 100s of MB on long runs);
#                                 # scalars/history/config still sync in full
#
# ---------------------------------------------------------------------------
# WHY THIS IS LOSSLESS (and why you must NOT pass --sync-tensorboard)
# ---------------------------------------------------------------------------
# train_cut3r_baseline.py:179-190 calls
#     wandb.tensorboard.patch(root_logdir=<output_dir>/tb)
#     wandb.init(..., sync_tensorboard=True, id=..., config=...)
# `tensorboard.patch` MONKEYPATCHES SummaryWriter IN-PROCESS, so every add_scalar
# is folded into the wandb run's own transaction log AS IT IS WRITTEN -- not
# scraped from tfevents afterwards. Verified empirically on this machine with
# wandb 0.25.1: an offline run using exactly this pattern ends with the
# tensorboard scalars already in its run summary and history.
#
# Consequence: the offline directory `wandb/offline-run-<ts>-<id>/run-<id>.wandb`
# is a COMPLETE record. Plain `wandb sync <dir>` replays it verbatim -- same run
# id, name, config, history, summary, step numbering, console log, system
# metrics and timestamps.
#
# Passing `--sync-tensorboard` (or additionally syncing the tb/ directory) would
# import the SAME scalars a SECOND time, duplicating history and corrupting the
# step axis. This script deliberately never does that.
#
# What offline genuinely cannot capture: nothing that the trainer produces. The
# only differences from a live run are that the wall-clock "created" timestamp is
# the original run's (correct) while the upload happens later, and that a live
# run would have been visible in the UI during training.
# ---------------------------------------------------------------------------
set -eo pipefail

CKPT_ROOT=${CKPT_ROOT:-/gpfs/scratch/etur59/koc821022/checkpoints}
DRY=0
NO_MEDIA=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --dry-run)  DRY=1 ;;
    --no-media) NO_MEDIA=1 ;;
    --all)      ARGS+=("$CKPT_ROOT") ;;
    *)          ARGS+=("$a") ;;
  esac
done
[ ${#ARGS[@]} -gt 0 ] || { sed -n '2,8p' "$0"; exit 1; }

if [ "$(hostname)" != "${HOSTNAME_OVERRIDE:-$(hostname)}" ]; then :; fi
case "$(hostname)" in
  glogin*|alogin*|transfer*) ;;
  *) echo "WARNING: '$(hostname)' does not look like an MN5 login node." >&2
     echo "         Compute nodes have no internet; this will fail." >&2 ;;
esac

source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything

# --- collect offline run dirs ----------------------------------------------
# Discovery runs BEFORE the auth/connectivity gates so --dry-run works with no
# credentials (inspecting what you have is always safe).
mapfile -t RUNS < <(
  for d in "${ARGS[@]}"; do
    [ -e "$d" ] || { echo "WARNING: no such path: $d" >&2; continue; }
    find "$d" -type d -name 'offline-run-*' -print
  done | sort -u
)

if [ ${#RUNS[@]} -eq 0 ]; then
  echo "No offline-run-* directories found under: ${ARGS[*]}"
  exit 0
fi

echo "found ${#RUNS[@]} offline run(s):"
for r in "${RUNS[@]}"; do
  id=$(basename "$r" | sed -E 's/^offline-run-[0-9_]+-//')
  n=$(du -sh "$r" 2>/dev/null | cut -f1)
  echo "  [$id] $n  $r"
done

if [ "$DRY" = "1" ]; then
  echo; echo "(dry run -- nothing uploaded)"; exit 0
fi

# --- credentials ------------------------------------------------------------
# Offline recording needs no auth, but uploading does.
if [ -z "${WANDB_API_KEY:-}" ] && ! /usr/bin/grep -qs "api.wandb.ai" "${NETRC:-$HOME/.netrc}"; then
  cat >&2 <<'MSG'
ERROR: W&B is not authenticated on this machine (no WANDB_API_KEY and no
       api.wandb.ai entry in ~/.netrc).

       Run ONE of these on the login node first, then re-run this script:
           wandb login                     # paste the key from wandb.ai/authorize
           export WANDB_API_KEY=<key>

       Nothing is lost by fixing this later: the offline run directories are
       complete and can be synced at any time.
MSG
  exit 2
fi

# --- connectivity -----------------------------------------------------------
code=$(curl -sS -m 20 -o /dev/null -w "%{http_code}" https://api.wandb.ai 2>/dev/null || echo 000)
if [ "$code" = "000" ]; then
  echo "ERROR: cannot reach https://api.wandb.ai from $(hostname)." >&2
  echo "       Check HTTP_PROXY/HTTPS_PROXY (currently: ${HTTPS_PROXY:-unset})." >&2
  exit 3
fi
echo "api.wandb.ai reachable (HTTP $code)"

# --- sync -------------------------------------------------------------------
# NOTE: no --sync-tensorboard, on purpose (see the header). wandb itself confirms
# this is right, printing "Found .wandb file, not streaming tensorboard metrics".
#
# --append is used deliberately. Verified empirically on wandb 0.25.1 by syncing a
# 5-step offline log, then re-sending a 10-step superset of the SAME run id:
#     sync   5-step log           -> 5 rows,  steps 0-4
#     --append 10-step superset   -> 10 rows, steps 0-9   (deduped, NOT shifted)
#     --append the same log again -> 10 rows, steps 0-9   (no-op)
# So it is safe to re-run this script at any time, and safe to run it against a
# run that is still being written -- which is what sync_loop.sh does every 10 min.
# --no-media: skip the files/media/** payload (print_img_freq visualizations —
# routinely 100s of MB per long run). Scalars/history/config are untouched; the
# UI's image panels for those steps just stay empty. wandb's glob filter is
# fnmatch over paths relative to files/, and fnmatch's '*' crosses '/', so
# 'media/*' covers media/images/... too.
SYNC_FLAGS=(--append)
MEDIA_NOTE=""
if [ "$NO_MEDIA" = "1" ]; then
  SYNC_FLAGS+=(--exclude-globs 'media/*')
  MEDIA_NOTE=" (media excluded)"
fi
rc=0
for r in "${RUNS[@]}"; do
  id=$(basename "$r" | sed -E 's/^offline-run-[0-9_]+-//')
  echo "== syncing [$id] $r$MEDIA_NOTE"
  if wandb sync "${SYNC_FLAGS[@]}" "$r"; then
    echo "   OK"
  else
    echo "   FAILED for $r" >&2
    rc=1
  fi
done

echo
if [ $rc -eq 0 ]; then
  echo "All runs synced. Verify in the UI: history, summary and the tb-derived"
  echo "scalars (train/*, test/*, absrel, a1, gru_pose_loss*) should each appear"
  echo "EXACTLY ONCE per step."
else
  echo "Some runs failed to sync (see above); re-run to retry -- it is idempotent." >&2
fi
exit $rc
