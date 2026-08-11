# my-da3 — notes for Claude

## W&B syncing (MareNostrum5): always exclude media, always use the beta path

Compute nodes have no internet, so W&B records OFFLINE under
`<save_dir>/<exp_name>/wandb/offline-run-*`. To port runs to wandb.ai, run from
a LOGIN node:

    /gpfs/projects/etur59/koc821022/checkpoints/sync_nomedia.sh [path-substring ...]

Non-negotiables, all learned empirically (full record in that script's header
and `checkpoints/sync_nomedia.log`):

- **Exclude media.** `files/media/**` (the `print_img_freq` PNGs) is hundreds
  of MB per long run and we only want scalars/history/config on wandb.ai. The
  script excludes it in its rsync snapshot (`--exclude files/media`).
- **`wandb beta sync`, never legacy `wandb sync`, for anything but tiny runs.**
  glogin1 enforces a hard, non-raisable 300s RLIMIT_CPU per process. Legacy
  sync replays the `.wandb` transaction log in Python at O(steps × ~377 metric
  keys) and gets SIGKILLed (rc=137) at ~345s on big runs; the Go-core beta path
  synced the same runs in ~150–200s, 20/20 OK. Excluding media does NOT fix
  this (it cuts upload payload, not replay CPU) — the beta path is the fix.
- **Snapshot before syncing a live run.** The script rsyncs a point-in-time
  copy to node-local NVMe (`/scratch/tmp/wandb_sync.$USER` — NOT /gpfs/projects,
  which filled to 100% on 2026-08-09 and silently produced empty snapshots) so
  the sync terminates instead of chasing a still-writing run. Re-running is
  always safe: wandb dedupes by run id.
- `wandb_port_offline_runs.sh` (repo root) is the LEGACY path. Its `--no-media`
  flag excludes media, but the 300s CPU cap still kills large replays — prefer
  `sync_nomedia.sh` unless the run is small/young.
