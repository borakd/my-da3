# DROID 4292-scene eval bundle — the out-of-repo pieces (OLD CLUSTER)

> **On MareNostrum5, read [`MN5_EVAL_README.md`](MN5_EVAL_README.md) instead.**
> Two claims below are now wrong: `bin/eval_depth_poses.py` is the *default* `EVAL_SCRIPT`
> on MN5 (no install step), and the note at the end of this file asserting that
> `EVAL_SCRIPT` is "a plain assignment, **not** `${EVAL_SCRIPT:-...}`" is false — every
> runner uses the overridable form, so the prescribed `sed -i` is unnecessary and would
> destroy the override.

Everything else the eval pipeline needs is **already tracked** on branch `captain_gru_v3`
(worktree `/scratch/bdursun25/cuteanything/captain_gru_v3`, clean at `aab1778`):
`eval_pipeline/*` (all 23 files), `src/CUT3R/demo_ray.py`, `src/CUT3R/demo.py`,
`src/CUT3R/add_ckpt_path.py`, the whole `src/CUT3R/src/` tree, and the curope sources.

This bundle carries only what a fresh clone of that branch does **not** give you.

## Contents

| Path | Why it's not in the repo | Install to |
|---|---|---|
| `bin/eval_depth_poses.py` | Lives in a different project (`/scratch/bdursun25/streaming-3d/`). Hardcoded as `EVAL_SCRIPT=` in every `run_*_node.sh`. **Strictly necessary** — it computes absrel/a1/ate/rpe_trans/rpe_rot. | anywhere; point `EVAL_SCRIPT` at it |
| `data/scene_list.txt` | Generated artifact (4292 lines), not committed. | `$OUT/scene_list.txt` |

`eval_depth_poses.py` is fully standalone: `argparse, csv, re, pathlib, typing, numpy`.
No sibling imports, never imported as a module — the worker shells out to it via
`subprocess.run([sys.executable, eval_script, ...])`.

`scene_list.txt` is regenerable and deterministic (sorted `os.listdir`, keep if
`dense/rgb` has >= 2 frames):

```bash
python eval_pipeline/make_scene_list.py \
  --scenes_root /scratch/bdursun25/cuteanything/scenes/pointworld_droid_splits/test/dl3dv_multi/wrist \
  --out         /scratch/bdursun25/cuteanything/outputs/cut3r_eval/scene_list.txt
```

It is included so a rerun is byte-identical to the published numbers without
depending on the dataset symlink set being unchanged.
sha256 `45ac49be…3059ad`.

## Deliberately NOT bundled

- **Dataset** `scenes/pointworld_droid_splits/test/dl3dv_multi/wrist/<scene>/dense/{rgb,cam,depth}`
  — hundreds of GB, staged on BeeGFS scratch.
- **Checkpoints** under `checkpoints/` — e.g.
  `cut3r_multinode/captain_ray_finetune_aug_full/checkpoint-last.pth`.
- **`InfiniteVGGT`** (`/scratch/bdursun25/streaming-3d/InfiniteVGGT`) — the only other
  out-of-repo path, used *only* by `infer_and_eval_worker_vggt.py` / `run_vggt_node.sh`.
  Not part of the CUT3R eval, so not strictly necessary.
- **Built `curope`** — sources are tracked; `build/` is gitignored. It must be compiled in
  the target env or **all** inference crashes.

## Bring-up on a fresh clone

```bash
ROOT=/scratch/bdursun25/cuteanything
REPO=$ROOT/captain_gru_v3          # git worktree/clone of branch captain_gru_v3
OUT=$ROOT/outputs/cut3r_eval

# 1. bundle pieces
mkdir -p "$OUT/logs"
cp "$REPO/eval_bundle/data/scene_list.txt" "$OUT/"

# 2. env + curope (once)
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate cuteanything
cd "$REPO/src/CUT3R/src/croco/models/curope" && python setup.py build_ext --inplace

# 3. point the launchers at the bundled evaluator (EVAL_SCRIPT is hardcoded, not env-overridable)
sed -i "s|^EVAL_SCRIPT=.*|EVAL_SCRIPT=$REPO/eval_bundle/bin/eval_depth_poses.py|" \
  "$REPO/eval_pipeline"/run_*.sh

# 4. launch (per node, cooperative claim queue; distinct BASE_SHARD per node)
cd "$REPO"
LABEL=captain_ray_gt_last CONDITIONING=gt BASE_SHARD=0 NUM_SHARDS=12 \
  sbatch eval_pipeline/run_captain_ray_eval_node.sh

# 5. aggregate (BOTH scripts require arguments; bare invocation fails)
python eval_pipeline/aggregate_results.py \
  --out_root "$OUT" --scene_list "$OUT/scene_list.txt" \
  --labels captain_ray_gt_last regular final     # -> $OUT/summary/averages_table.{csv,md,txt}
sbatch eval_pipeline/run_pose_sim3_both.sh       # optional sim3-aligned ATE/RPE (CPU-only)
```

Full operational detail — conditioning modes, sharding, resume/OOM behavior, output layout,
the single-scene contract — is in **`EVAL_README.md`**.

**Note:** on `captain_gru_v3` the launchers already set `CRAY=$ROOT/captain_gru_v3`, so the
only stale path is `EVAL_SCRIPT` (line 39 of `run_captain_ray_eval_node.sh`), pointing at
`/scratch/bdursun25/streaming-3d/`. It is a plain assignment, **not** `${EVAL_SCRIPT:-...}`,
so exporting the var does nothing — step 3 rewrites the line instead. `ROOT`, `SCENES_ROOT`,
`OUT` are likewise hardcoded to `/scratch/bdursun25/cuteanything`; adjust if relocating.
