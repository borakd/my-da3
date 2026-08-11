# DROID 4292-scene eval — complete runbook (OLD CLUSTER)

> **On MareNostrum5, read [`MN5_EVAL_README.md`](MN5_EVAL_README.md) instead.**
> This file documents the cluster the published numbers were produced on. Its paths
> (`/scratch/bdursun25/...`), its conda hook (`/opt/ohpc/...`) and its `sed`-the-launchers
> bring-up step do not apply on MN5 — the launchers now source
> `eval_pipeline/mn5_paths.sh` and take `EVAL_SCRIPT` from the environment, so the sed
> would replace an overridable line with a hard path.
> Kept for the provenance of the published numbers: the evaluator contract in the second
> half ("Evaluating one scene — the exact contract") is unchanged and still authoritative.

Read this first if you are picking up the eval pipeline on a new machine. It assumes the
`captain_gru_v3` branch is checked out and the data/checkpoints are already staged; it
covers everything else needed to reproduce our numbers **identically**.

Companion file `README.md` explains why this bundle exists (what is *not* in the repo).

---

## 0. TL;DR — the whole pipeline

```bash
ROOT=/scratch/bdursun25/cuteanything
REPO=$ROOT/captain_gru_v3
OUT=$ROOT/outputs/cut3r_eval

# once per machine
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh && conda activate cuteanything
cd "$REPO/src/CUT3R/src/croco/models/curope" && python setup.py build_ext --inplace
mkdir -p "$OUT/logs" && cp "$REPO/eval_bundle/data/scene_list.txt" "$OUT/"
sed -i "s|^EVAL_SCRIPT=.*|EVAL_SCRIPT=$REPO/eval_bundle/bin/eval_depth_poses.py|" \
  "$REPO/eval_pipeline"/run_*.sh

# per checkpoint (submit once per node; all nodes share one claim queue)
cd "$REPO"
LABEL=captain_ray_gt_last CONDITIONING=gt BASE_SHARD=0 NUM_SHARDS=12 \
  CKPT=$ROOT/checkpoints/cut3r_multinode/captain_ray_finetune_aug_full/checkpoint-last.pth \
  sbatch eval_pipeline/run_captain_ray_eval_node.sh

# after all scenes land
python eval_pipeline/aggregate_results.py \
  --out_root "$OUT" --scene_list "$OUT/scene_list.txt" --labels captain_ray_gt_last regular final
```

---

## 1. Environment

- conda env **`cuteanything`**. Source conda **directly**, never `~/.bashrc` — its line 1 is
  corrupted (`#: command not found`, exit 127), which is harmless interactively but **fatal
  under `set -e`** in batch scripts. The runners use `set -o pipefail` (no `-u`, since MKL
  activation scripts read unset vars).
- `PYTHONPATH` must carry the three src trees; the runners export it:
  `$REPO/src:$REPO/src/CUT3R:$REPO/src/CUT3R/src`.
- **`curope` must be compiled** or every inference crashes. Sources are tracked, `build/` is
  gitignored, so a fresh clone always needs the `build_ext --inplace` above.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — set by the runner and, defensively, by
  the worker before torch is imported (`infer_and_eval_worker_ray.py:22`).
- **Never run inference on the login node.** Always `sbatch`.

## 2. Scene list

`$OUT/scene_list.txt`, 4292 lines, shipped in `data/`. Regenerable and deterministic
(sorted `os.listdir`, keep scenes whose `dense/rgb` holds >= 2 frames):

```bash
python eval_pipeline/make_scene_list.py \
  --scenes_root $ROOT/scenes/pointworld_droid_splits/test/dl3dv_multi/wrist \
  --out         $OUT/scene_list.txt
```

Use the shipped copy to stay byte-identical with published runs regardless of the current
symlink set.

## 3. Inference + per-scene eval

One launcher, `eval_pipeline/run_captain_ray_eval_node.sh`, parameterized by env vars:

| Var | Default | Notes |
|---|---|---|
| `LABEL` | `captain_ray_gt_last` | names `$OUT/$LABEL/{preds,eval,claims}` |
| `CKPT` | `.../captain_ray_finetune_aug_full/checkpoint-last.pth` | |
| `CONDITIONING` | `gt` | `gt \| prev_gt \| prev_pred \| prev_pred_gru \| none` |
| `BASE_SHARD` | `0` | global shard offset **for this node's GPUs** |
| `NUM_SHARDS` | `12` | total planned workers; only sets starting offsets |
| `LIMIT` | `0` | max scenes per worker; `0` = all (use for smoke tests) |

`--size 320` is hardcoded in the launcher and must stay 320 — it is the training resolution
(via `load_images_cover`). The stock demo loader yields **320x176** instead of **320x192**,
which silently makes numbers non-comparable.

Submit **once per node**. Every worker (4 per node, one per GPU) walks the *whole* list and
atomically claims each scene with `mkdir` under the shared `claims/` dir, so any subset of
nodes that actually start will cooperatively drain one queue — no double-processing, no stall
if a node never starts. Give each node a distinct `BASE_SHARD` (0, 4, 8, …) so workers begin
in different parts of the list.

The run is **resumable**: a scene whose eval CSV exists and is non-empty is skipped. Failures
release their claim so a later pass retries them. Per-shard failures land in
`$OUT/$LABEL/eval/_failures_shard<N>.txt`.

`prev_pred_gru` is the gru_v3-only mode (this worker adds it; the `captain_ray` worktree's copy
has only the other four). It sets `feed_prev_pred` and activates the PoseGRU refiner.

Two escape hatches: a `$OUT/logs/SKIP_<label>` sentinel makes a worker exit immediately
(disable one job's pass without editing a running launcher); `--ignore_skip_sentinel` overrides
it.

### Memory

`inference()` is **O(N) in sequence length**. Two very long scenes needed an A6000 rather than
an L40S. The worker retries once after an OOM with a cache clear; persistent OOMs go to the
failure log for a targeted sweep (`sweep_prev_gt_oom.sh` is the worked example).

## 4. Aggregation

**Both scripts require arguments** — bare invocations fail.

```bash
# depth+pose averages -> $OUT/summary/{averages_table.{csv,md,txt}, per_scene_<label>.csv}
python eval_pipeline/aggregate_results.py \
  --out_root "$OUT" --scene_list "$OUT/scene_list.txt" \
  --labels captain_ray_gt_last regular final augfull_last

# optional: sim3-aligned pose recompute from saved camera npzs (CPU-only, ~100x cheaper
# than a full re-eval) -> $OUT/<label>/sim3_pose_both.csv
sbatch eval_pipeline/run_pose_sim3_both.sh          # LABELS="a b c" to override
python eval_pipeline/build_sim3_table.py            # -> LaTeX table
```

`aggregate_results.py` reads `$out_root/<label>/eval/<scene>/eval_depth_pose_metrics.csv`,
takes the `camera_id == "ALL"` / `local_timestep == "MEAN"` row (falling back to any `MEAN` row
for single-camera data), and **nanmean**s across scenes. Metrics: `absrel, a1, ate, rpe_trans,
rpe_rot`. Scenes with no CSV are silently skipped — always check the printed
`<label>: aggregated N scenes` equals 4292 before trusting a table.

Row labels for the paper table live in `build_sim3_table.py:20-27`, e.g.
`regular` = CUT3R zero-shot, `augfull_final` = finetuned full/50ep,
`captain_ray_gt_last` = Captain Cut3r, `infinitevggt` = InfiniteVGGT zero-shot.

## 5. Output layout

```
$OUT/scene_list.txt
$OUT/logs/                         slurm + per-worker logs, SKIP_<label> sentinels
$OUT/<label>/claims/<scene>/       cooperative claim markers (mkdir)
$OUT/<label>/preds/<scene>/depth/000000.npy
$OUT/<label>/preds/<scene>/camera/000000.npz
$OUT/<label>/eval/<scene>/eval_depth_pose_metrics.csv
$OUT/<label>/eval/_failures_shard<N>.txt
$OUT/<label>/sim3_pose_both.csv    (only if pose_sim3_both.py was run)
$OUT/summary/averages_table.{csv,md,txt}
```

---

# Evaluating one scene — the exact contract

Given predictions already on disk in the layout below, **`eval_depth_poses.py` is the only
thing you need**. It never imports the model, CUT3R, or torch on the default path — just
`argparse, csv, re, pathlib, typing, numpy`. Every number in the 4292-scene tables came out of
exactly this call, with **all optional arguments left at their defaults**:

```bash
python eval_bundle/bin/eval_depth_poses.py \
  --pred_root  <pred_dir> \
  --gt_root    <scene>/dense \
  --output_csv <out_dir>/eval_depth_pose_metrics.csv
```

That is verbatim what the worker shells out
(`eval_pipeline/infer_and_eval_worker_ray.py:258-264`, via
`subprocess.run([sys.executable, eval_script, ...])`) — three flags, nothing else.
Matching those numbers means **passing nothing else**.

## File layout

```
<pred_root>/depth/000000.npy      float (H,W)      pred depth
<pred_root>/camera/000000.npz     key "pose" (4,4 or 3,4 c2w), "intrinsics"
<gt_root>/depth/000000.npy        float (H,W)      GT depth
<gt_root>/cam/000000.npz          key "pose"       ("camera/" also accepted)
```

- Filenames must be **zero-padded integers** (`^\d+$` stem) — matched by index, not by name,
  and sorted numerically.
- The camera folder resolves `camera/` first, then `cam/`. DROID GT uses `cam/`.
- Only the `pose` key is read; `intrinsics` is written but unused for these metrics. A missing
  `pose` key raises `KeyError`.
- `(3,4)` poses are promoted to `(4,4)`.

Predictions are written by `save_depth_camera()` (`infer_and_eval_worker_ray.py:37-69`),
**not** by `demo_ray.prepare_output` — depth is `pts3d_in_self_view[..., 2]`, pose is
`pose_encoding_to_camera(camera_pose)` as c2w, focal via
`estimate_focal_knowing_depth(..., "weiszfeld")`. Any producer writing those two folders is
equally valid input.

## The defaults that define "identically to how we have done it"

| Flag | Default in use | Meaning |
|---|---|---|
| `--align` | `sim3` | per-camera trajectory alignment for ATE/RPE |
| `--depth_scale_align` | `median` | `median(gt)/median(pred)` before absrel/a1 |
| `--pose_reduce` | `rmse` | how MEAN rows summarize pose metrics |
| `--pred_pose_type` / `--gt_pose_type` | `c2w` / `c2w` | pose convention |
| `--num_cameras` | `1` | single stream; frames are one sequence in index order |
| `--depth_eps` | `1e-9` | min positive depth for a valid pixel |
| `--eval_like_cut3r` | `False` | **off** |

Two footguns:

1. **`--pose_reduce` defaults to `rmse`.** It was switched from `mean` partway through and the
   published tables were redone under `rmse`. A CSV produced with `mean` is not comparable.
2. **`--gt_root` has a hardcoded default** pointing at one `/frozen/...` DROID scene. Omitting
   it silently evaluates against the wrong ground truth instead of erroring. Always pass it.

`--eval_like_cut3r` swaps in CUT3R's own metric math (evo APE/RPE, Weiszfeld scale-only depth
alignment) and **ignores** `--align`, `--depth_scale_align`, `--pose_reduce`. It also needs
`evo`, `torch`, `cv2`, `scipy`. It was **not** used for our tables — leave it off.

## Outputs

`eval_depth_pose_metrics.csv` (per-camera, per-local-timestep rows plus MEAN summaries) and
`eval_metrics_readable.txt`, written alongside it automatically.

## Verified

Re-running the bundled script on an already-computed scene reproduces the published CSV
**bit-for-bit**:

```
scene AUTOLab+0d4edc83+2023-10-21-19h-11m-38s   (label captain_ray_gt_last)
ALL,MEAN,0.1594611203042993,0.8275433421028047,0.008681230522003314,0.004261850564465868,0.2271613936440336
```
identical in both the fresh CSV and
`outputs/cut3r_eval/captain_ray_gt_last/eval/<scene>/eval_depth_pose_metrics.csv`
(columns: `absrel, a1, ate, rpe_trans, rpe_rot`).

## Caveat on "identical"

This gets you an identical **evaluation**. Identical *numbers* also require the predictions to
have been generated the same way — notably `--size 320` and the same conditioning mode. The
evaluator is deterministic and stateless; the inference side is where comparability is won or
lost. One known exception: the 672-frame closed-loop single-scene setting is **GPU-architecture
sensitive** (ATE shifts up to 18%), so single-scene closed-loop rankings are noise-dominated.
The 4292-scene aggregate is not affected at that magnitude.
