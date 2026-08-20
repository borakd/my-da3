# my-da3 — notes for Claude

## Confidence-gate campaign (2026-08-17→20): CLOSED, ALL-5 SWEEP — see CONF_GATE_CAMPAIGN.md

GRAND CHAMPION augfull_cg_fuse_g7: confidence-gated (STATE_GATE g7ema)
forward × backward passes fused per scene (cg_fuse_fwd_bwd.py) beats
augfull_lr1e5 on ALL FIVE metrics significantly on the full 4292 harness
(ATE −15.5%, rot −12.8%); zero training, 2× inference. Training-side gate
variants (constant/scheduled/revisit) are all measured-dead. Gate/REVERSE/
REVISIT arms MUST launch via eval_pipeline/cg_launch_full.sh
(arm_eval_for_run.sh strips those envs by design). NEVER purge
augfull_cg_g7ema_bwd/preds — augfull_cg_fuse_g7 depth symlinks into it.

## GRU v4 campaign (2026-08-13→15): read before any GRU/conditioning work

`GRU_V4_CAMPAIGN_STATE.md` (repo root) is the authoritative chronicle and has
an ORIENTATION section for new sessions: verdicts, evidence map
(`eval_pipeline/evidence/`), verified traps, and the paused decision state.
All four conditioning design families are measured-dead; do NOT launch
training arms before the user's pending scope decision recorded there.
Plain-language history: `REPORT_2026-08-13.md` (addendum has the ending).

## Checkpoint locations (changed 2026-08-14)

- **All new finetuning runs must save checkpoints under
  `/gpfs/scratch/etur59/koc821022/checkpoints`.**
- All pre-2026-08-14 runs were moved to
  `/gpfs/scratch/etur59/koc821022/checkpoints_projects`
  (formerly `/gpfs/projects/etur59/koc821022/checkpoints`, which no longer
  exists — do not write there; anything landing on /gpfs/projects is a bug).
- YAML configs and sbatch/eval scripts that still hardcode the old
  `/gpfs/projects/.../checkpoints` root must be repointed before use.

## W&B syncing (MareNostrum5): always exclude media, always use the beta path

Compute nodes have no internet, so W&B records OFFLINE under
`<save_dir>/<exp_name>/wandb/offline-run-*`. To port runs to wandb.ai, run from
a LOGIN node:

    /gpfs/scratch/etur59/koc821022/checkpoints/sync_nomedia.sh [path-substring ...]

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
