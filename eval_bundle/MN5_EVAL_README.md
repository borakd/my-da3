# DROID eval on MareNostrum5 — runbook

The runbook for evaluating a CUT3R-based checkpoint (including captain_gru / PoseGRU
variants) **on MN5**. `EVAL_README.md` next to this file describes the *old* cluster and is
kept for the provenance of the published numbers — its paths, its conda hook, and its
`sed`-the-launchers step no longer apply here. Read this one.

---

## 0. TL;DR

```bash
REPO=/gpfs/home/koc/koc821022/vggt_features          # your checkout of branch captain_gru_v3
cd "$REPO"

# once per machine
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything
cd src/CUT3R/src/croco/models/curope && python setup.py build_ext --inplace && cd "$REPO"
OUT=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval
mkdir -p "$OUT/logs" && cp -n eval_bundle/data/scene_list.txt "$OUT/"

# per checkpoint — submit once per node, distinct BASE_SHARD per node (0, 4, 8, …)
LABEL=gru_a4g3f1r4 CONDITIONING=prev_pred_gru \
CKPT=/gpfs/scratch/etur59/koc821022/checkpoints_projects/captain_cut3r_finetune_aug_full/captain_gru_v3_a4_g3_f1_r4_finetune/checkpoint-final.pth \
BASE_SHARD=0 NUM_SHARDS=12 \
  sbatch eval_pipeline/run_captain_ray_eval_node.sh

# after all scenes land
python eval_pipeline/aggregate_results.py \
  --out_root "$OUT" --scene_list "$OUT/scene_list.txt" --labels gru_a4g3f1r4
```

There is no `sed` step and no symlink hack. Every path is a variable with a working MN5
default.

---

## 1. Storage anchors

All of them live in one place — `eval_pipeline/mn5_paths.sh`, sourced by every launcher —
and every one is overridable from the environment.

| Var | MN5 default | Notes |
|---|---|---|
| `WT` | `$SLURM_SUBMIT_DIR`, else `$PWD` | the my-da3 checkout; validated, hard-fails otherwise |
| `CKPT_ROOT` | `/gpfs/scratch/etur59/koc821022/checkpoints_projects` | pre-2026-08-14 runs; new runs save to `/gpfs/scratch/etur59/koc821022/checkpoints` |
| `OUT_ROOT` | `/gpfs/scratch/etur59/koc821022/outputs` | `$OUT = $OUT_ROOT/cut3r_eval` |
| `DATA_ROOT` | `/gpfs/scratch/etur59/koc821022` | DROID episode store |
| `SCENES_ROOT` | `$DATA_ROOT/pointworld_droid_splits/test/dl3dv_multi/wrist` | the 4292-scene test split |
| `OVERFIT_ROOT` | `$DATA_ROOT/pointworld_droid_wrist_VALAR` | single-episode store for the GRU grid |
| `OVERFIT_SCENE` | `RAIL+eh61f232+2023-10-26-17h-33m-59s` | 672 frames |
| `EVAL_SCRIPT` | `$WT/eval_bundle/bin/eval_depth_poses.py` | vendored; the upstream streaming-3d checkout is not on MN5 (see caveat below) |
| `SCENE_LIST` | `$OUT/scene_list.txt` | copy of `eval_bundle/data/scene_list.txt` |
| `DL3DV_CACHE_DIR` | `$DATA_ROOT/.dl3dv_cache` | dataset scan cache, used by the probes |

> ⚠️ **`eval_bundle/` is untracked in git** (not ignored — just never `git add`ed), and so
> is `eval_pipeline/mn5_paths.sh`. Since `EVAL_SCRIPT` now defaults into `eval_bundle/`, a
> fresh clone of `captain_gru_v3` would come up **without the evaluator and without the
> shared anchors**. Commit both before anyone clones this branch:
> `git add eval_bundle eval_pipeline/mn5_paths.sh`.

**Three shape changes from the old cluster**, worth knowing because they are why the old
paths cannot simply be re-pointed:

1. There is **no `scenes/` component**. The splits tree sits directly under `DATA_ROOT`, so
   no value of `DATA_ROOT` could reach the data through the old
   `$DATA_ROOT/scenes/pointworld_droid_splits/...` expression. `SCENES_ROOT` is now its own
   overridable variable.
2. The **overfit episode lost a directory level**. Old:
   `<episode>/13062452+wrist/dense`. MN5: `<episode>/dense`. So `OVERFIT_SCENE` has no
   camera component, and eval output paths under `overfit_test_scene/<label>/eval/<scene>/`
   are one level shallower than in the published tables.
3. The conda hook is `/apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh`, not the ohpc path
   in the old README. Source it directly, never `~/.bashrc` — its line 1 is corrupt (exit
   127), harmless interactively but fatal under `set -e`.

---

## 2. What is and is not on MN5

**Available.** The full 4292-scene test split (verified: every scene in `scene_list.txt`
resolves, `dense/{rgb,cam,depth}` present, cam npz carries `pose` + `intrinsic`). The
overfit episode. The `cuteanything` conda env, with `curope` already compiled. Checkpoints:

```
$CKPT_ROOT/cut3r_multinode/cut3r_finetune_aug_full_{,gtray_,prevpred_}{32,64}gpu*/
$CKPT_ROOT/captain_cut3r_finetune_aug_full/captain_gru_v3_a4_g{0,3}*_finetune/
```

**Not available.** These are guarded now — you get a `MISSING FILE:` list and exit 1 up
front rather than four workers failing an hour in:

- `captain_ray_finetune_aug_full`, `captain_cut3r_sim3rmse`, `captain_gru_overfit`,
  `multinode_finetune_aug_mini` — old checkpoint roots, never copied. Scripts that used
  them as hardcoded defaults now require `CKPT` (or `CKPTS`/`CKPT_DIR`/`BEST_CKPT`)
  explicitly. **No substitute default was invented** — silently scoring a differently
  trained experiment is worse than an error.
- **InfiniteVGGT** — neither the `infinitevggt` conda env nor the repo exists.
  `run_vggt_node.sh` now fails on the activate instead of running the whole loop under the
  base interpreter. It only supplies a zero-shot comparison row; it is not part of the
  CUT3R eval.
- The `demo_ray_smoketest/prev_pred` baseline CSV the GRU tables cite. It came from a
  Jul-10 pass predating every surviving prev_pred checkpoint, so it can be located
  (`PREV_PRED_CSV=`) but not regenerated.

---

## 3. Step by step

### 3.1 Environment (once)

```bash
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything
cd "$REPO/src/CUT3R/src/croco/models/curope" && python setup.py build_ext --inplace
```
`curope` must be compiled or **every** inference crashes; sources are tracked, `build/` is
gitignored. It is already built in the current checkout.

**Never run a launcher on a login node — this is now enforced.** MN5 login nodes expose
GPUs, so the old scripts happily started real workers there; the cgroup killed them ~20s in,
*after* they had claimed scenes, and a claim is released only on a caught Python exception.
Those scenes were then silently skipped by every later run. Every GPU launcher now calls
`mn5_require_slurm` before any `mkdir`. Under `salloc`/`srun`, set `ALLOW_NO_SLURM=1`.

### 3.2 Scene list (once)

```bash
cp -n eval_bundle/data/scene_list.txt "$OUT/scene_list.txt"    # 4292 lines
```
Use the shipped copy to stay byte-identical with published runs (sha256 `45ac49be…3059ad`).
`make_scene_list.py --scenes_root "$SCENES_ROOT" --out "$OUT/scene_list.txt"` regenerates it
deterministically, but only for the flat split layout.

### 3.3 Pick the checkpoint, then pick the arm

Never `checkpoint-last.pth`/`checkpoint-best.pth` from a live job — they are rewritten every
epoch and `torch.load` races the writer. Use `checkpoint-final.pth`, else the highest
`checkpoint-<N>.pth`.

**The checkpoint does not select the arm.** `feed_prev_pred` and `pose_gru` are runtime
attributes, never stored in the state_dict. `--conditioning` selects it:

| checkpoint | worker | `--conditioning` |
|---|---|---|
| vanilla / pretrained CUT3R | `infer_and_eval_worker.py` | *(has no such flag)* |
| GT ray-map (captain_ray, `*_gtray_*`) | `..._ray.py` | `gt` |
| prev-GT / prev-pred ablations | `..._ray.py` | `prev_gt` / `prev_pred` |
| **captain_gru (PoseGRU)** | `..._ray.py` | **`prev_pred_gru`** |

### 3.4 Pre-flight, captain_gru only

```bash
python verify_gru_ckpt_matrix.py --ckpt_root "$CKPT_ROOT/captain_cut3r_finetune_aug_full"
```
Derives the A/G/F/R levers three independent ways (run-dir name, the run's
`.hydra/config.yaml`, the live restored module), asserts they agree and that every
`pose_gru.*` tensor is bit-identical, then strips the GRU and asserts the predicted poses
actually move. Non-zero exit means do not publish that arm.
`eval_gru_overfit_all.sbatch` runs it as step 1 and aborts the sweep on failure.

Cheap CPU-only companions: `verify_gru_eval_and_direct.py`,
`sbatch verify_gru_v3_levers.sbatch`.

### 3.5 Smoke test — one GPU, two scenes

```bash
CKPT=<...>/checkpoint-final.pth sbatch eval_pipeline/smoke_captain_ray_eval.sh
```
Worth the 90 minutes: besides running the worker it re-runs `demo_ray.py` directly on the
first scene and asserts depth and pose match to 1e-3. The two scenes write into the real
output dirs and count toward the full run.

### 3.6 Full run

```bash
LABEL=<label> CONDITIONING=prev_pred_gru CKPT=<...> BASE_SHARD=0 NUM_SHARDS=12 \
  sbatch eval_pipeline/run_captain_ray_eval_node.sh
```
Submit **once per node**, `BASE_SHARD` in steps of 4 (0, 4, 8, …), `NUM_SHARDS` = total
planned workers. Every worker walks the whole list and claims each scene atomically with
`mkdir` under a shared `claims/` dir, so any subset of nodes cooperatively drains one queue
— no double-processing, no stall if a node never starts. Resumable: a non-empty eval CSV is
skipped.

`--size 320` is hardcoded and must stay 320 — it is the 320×192 training resolution via
`load_frames_training_style`. The stock demo loader yields 320×176 and silently makes
numbers non-comparable.

Multi-node-safe launchers pass `--claim_dir`: `run_captain_ray_eval_node.sh`,
`run_augfull_node.sh`, `run_vggt_node.sh`. `run_cut3r_eval.sh` and
`run_regular_parallel.sh` do **not** — they shard round-robin, so submitting either to two
nodes duplicates 100% of the work.

### 3.7 Reconcile before aggregating

Failures land in `$OUT/<label>/eval/_failures_shard<N>.txt`. `inference()` is O(N) in
sequence length; two very long scenes need more memory than an L40S had
(`sweep_prev_gt_oom.sh` is the worked example of a targeted re-run).

A worker killed by SLURM (timeout, OOM-killer, node failure) leaves its claim dir behind
forever and that scene is **never retried** — only Python-level exceptions release claims.
Before a resume pass, delete claim dirs with no matching non-empty eval CSV:

```bash
for c in "$OUT/$LABEL/claims"/*/; do
  s=$(basename "$c")
  [ -s "$OUT/$LABEL/eval/$s/eval_depth_pose_metrics.csv" ] || rmdir "$c"
done
```

### 3.8 Aggregate

```bash
python eval_pipeline/aggregate_results.py \
  --out_root "$OUT" --scene_list "$OUT/scene_list.txt" --labels <label> …
```
Reads the `camera_id=="ALL" / local_timestep=="MEAN"` row per scene and nanmeans
`absrel, a1, ate, rpe_trans, rpe_rot` into `$OUT/summary/averages_table.{csv,md,txt}`.
**Check the printed `<label>: aggregated N scenes` equals 4292** — missing CSVs are silently
skipped and you still get a table.

Optional pose-only sim3 recompute from the saved camera npzs — CPU, ~100× cheaper than a
re-eval, and recovers both the `mean` and `rmse` aggregations:

```bash
LABELS="a b c" sbatch eval_pipeline/run_pose_sim3_both.sh
python eval_pipeline/build_sim3_table.py --out_root "$OUT" --agg rmse --out_tex "$OUT/tables/x.tex"
```
(`build_sim3_table.py` requires all three flags; the old README showed a bare invocation,
which exits 2.)

### 3.9 GRU-specific follow-ups

- **Falsifiers** — the go/no-go that the GRU contributes anything. `POSE_GRU_HIDDEN_ZERO=1`
  kills recurrent memory; `POSE_GRU_FORCE_ITERS={1,2,4,8}` is the anytime-N curve on an R8
  checkpoint. Batched probes: `sbatch probe_gru_grid_all.sbatch`.
- **Window eval** — `run_gru_window64_eval.sbatch` gives ATE/RPE as a distribution over 77
  sliding 64-frame windows instead of one n=1 672-frame rollout. Build the windows first
  (`build_windows_strided.py --src_dense "$OVERFIT_ROOT/$OVERFIT_SCENE/dense" --out_root
  <OUT> --window 64 --stride 8` — pass an **absolute** `--src_dense` or every symlink
  breaks); nothing else invokes it, and the job now refuses to start without them.

---

## 4. The evaluator contract

Given predictions on disk, `eval_bundle/bin/eval_depth_poses.py` is the whole of stage 2. It
never imports torch or CUT3R on the default path. Every published number came from exactly:

```bash
python eval_bundle/bin/eval_depth_poses.py \
  --pred_root <pred_dir> --gt_root <scene>/dense --output_csv <out>/eval_depth_pose_metrics.csv
```

Three flags, nothing else — matching those numbers means passing nothing else.

```
<pred_root>/depth/000000.npy      float (H,W)
<pred_root>/camera/000000.npz     "pose" (4,4 or 3,4 c2w), "intrinsics"
<gt_root>/depth/000000.npy
<gt_root>/cam/000000.npz          "pose"     (DROID GT uses cam/)
```

Filename stems must be **pure digits** — matched by integer index, not by name. A prefixed
name (`frame_000000.npy`) is invisible with no warning.

**The `camera/` vs `cam/` fallback applies to `--gt_root` only.** `--pred_root` is read as a
hard `camera/` (`eval_depth_poses.py:709`). A pred tree using `cam/` silently yields
all-`nan` pose metrics with perfectly good depth metrics. The worker writes `camera/`, so
this only bites hand-built pred trees.

Defaults that define comparability: `--align sim3`, `--depth_scale_align median`,
`--pose_reduce rmse`, `--num_cameras 1`, `--pred_pose_type/--gt_pose_type c2w`,
`--depth_eps 1e-9`, `--eval_like_cut3r` off.

- `--pose_reduce` was switched from `mean` to `rmse` partway through and the published
  tables were redone. A `mean` CSV is not comparable to anything published.
- `--gt_root` has a hardcoded default pointing at one old `/frozen` scene. On MN5 that path
  does not resolve, so omitting it fails loudly rather than silently scoring the wrong GT —
  but always pass it.
- `--eval_like_cut3r` swaps in CUT3R's own metric math and **ignores** `--align`,
  `--depth_scale_align`, `--pose_reduce`. It was not used for any published table.

Sim3/SE3 alignment is fitted on **camera centres only** (Umeyama over `T[:3,3]`); rotations
never enter the objective and the recovered scale is applied to translations only. The
default depth path applies **no maximum-depth cap** — the 70 m cap exists only on the
`--eval_like_cut3r` branch.

---

## 5. Output layout

```
$OUT/scene_list.txt
$OUT/logs/                                     slurm + per-worker logs, SKIP_<label> sentinels
$OUT/<label>/claims/<scene>/                   cooperative claim markers
$OUT/<label>/preds/<scene>/depth/000000.npy
$OUT/<label>/preds/<scene>/camera/000000.npz
$OUT/<label>/eval/<scene>/eval_depth_pose_metrics.csv
$OUT/<label>/eval/_failures_shard<N>.txt
$OUT/<label>/sim3_pose_both.csv                (only if pose_sim3_both.py was run)
$OUT/summary/averages_table.{csv,md,txt}
```

---

## 6. Footguns

- **`f0`/`f1` in a config name means two different things, so never label a table row from
  the filename.** In the older family (`a4_g3`, `a4_g3_r8`, `a4_g3_f1`, `a4_g3_f1_r8`) the F
  lever is on/off: no suffix ⇒ `pose_gru_img_feat: none` (NO image pathway at all), `_f1` ⇒
  `input`. In the `_noproj` family BOTH are on and the digit is `pose_gru_img_feat_frames`:
  `f0` = 1 frame (current view only), `f1` = 2 (current + previous). So `f1` means "F on" in
  one family and "2 frames" in the other, and `f0` never meant "F off".
  The ground truth is the checkpoint, not the name. Since the SOURCE sub-lever
  (`pose_gru_img_feat_src`: suffix `r`/`d`/`c` on the F token), the cell input width
  ALONE no longer identifies the arm — width 46 (14+32) is produced by pooled F1,
  resnet18 F0r/F1r, dinov2 F0d/F1d AND corr F1c alike, because every projected arm
  appends the same 32 columns. Decode in two steps:

      sd = ckpt["model"]
      # Step 1 — SOURCE, from the img_norm width (or encoder key fingerprints):
      src_dim = sd["pose_gru.img_norm.weight"].shape[0]
      #   1024 => pooled CUT3R tokens; 512 => resnet18; 384 => dinov2_vits14; 54 => corr
      #   (equivalently: any pose_gru.img_encoder.* key present => frozen encoder;
      #    ...net.conv1.weight => resnet18, ...net.cls_token => dinov2_vits14)
      # Step 2 — FRAMES, from the projector width (or the cell width if noproj):
      blocks = sd["pose_gru.img_proj.weight"].shape[1] // src_dim   # if img_proj exists
      # else solve: sd["pose_gru.cell.weight_ih"].shape[1] == base(7|14) + blocks*src_dim
      #   blocks 1 => F0, 2 => F1 — EXCEPT corr, always 1 block but semantically
      #   2-frame (label it F1c; there is no F0c).

  `img_norm` exists iff features are fed; `img_proj` exists iff they are also projected;
  no `pose_gru.img_norm.weight` at all ⇒ F off (14-D cell). Write `F0`/`F1` (+ source
  suffix) in a row label ONLY when features actually enter the GRU. This is exactly the
  logic `_sniff_pose_gru_config` in dust3r/model.py runs — when in doubt, just
  `from_pretrained` the ckpt and read the printed `pose_gru enabled from ckpt:` line.
- **`--conditioning prev_pred_gru` on a checkpoint with no GRU does not fail.** It enables
  an identity-init residual GRU (≡ plain prev_pred) and prints a NOTE. Grep the worker log
  for `identity-init residual GRU` before publishing a GRU row.
- **`pose_gru_iters` mis-restores silently** (defaults to 1) — an R8 checkpoint whose args
  lost the key evaluates as R1 with no error. `pose_gru_mode` is the opposite: a hard
  RuntimeError, because residual/direct/split_anchor are shape-identical.
- **`strict=False` for all non-GRU weights**, with size mismatches downgraded to INFO log
  lines. A partly-random model runs and produces plausible metrics.
- **Stale falsifier env vars contaminate a run** and nothing logs it. Unset
  `PREV_PRED_RAY_SHUFFLE`, `POSE_GRU_HIDDEN_{ZERO,SHUFFLE}`,
  `POSE_GRU_IMG_FEAT_{ZERO,SHUFFLE}`, `POSE_GRU_FORCE_ITERS`, `GT_RAY_MAP_SHUFFLE` before a
  scoring run. The `*_SHUFFLE` ones are batch-rolls and no-ops at batch size 1 (the eval
  worker runs one sequence at a time).
- **The SKIP sentinel is silent and sticky.** `$OUT/logs/SKIP_<label>` makes every worker
  for that label print one line and exit 0. `run_captain_ray_eval_node.sh` and the smokes do
  not pass `--ignore_skip_sentinel`, so a stray sentinel turns the main launcher into a
  no-op.
- **G levers are inert at eval.** `pose_gru_bptt`/`pose_gru_e2e` are trainer-set and never
  restored; a `_g3` and a `_g1` checkpoint run identical forward math under inference.
- **Scripts use `set -o pipefail` only** — no `-e` (corrupt login rc), no `-u` (MKL
  activation reads unset vars). The `mn5_require` guards exist because of this. Per-GPU
  worker output is redirected inside a subshell and the bare `wait` discards exit codes, so
  the job can still report success with all four workers dead. Read
  `$OUT/logs/worker_*.log`, not just the slurm `.out`.
- **`.gitignore` has `*.sh` with `!eval_pipeline/*.sh`**, and the shell's `grep` is a ugrep
  wrapper that respects it — a plain `grep -r` silently skips every shell script. Use
  `git ls-files -z | xargs -0 /usr/bin/grep`. A new launcher outside `eval_pipeline/` is
  invisible to git; `.sbatch` files are fine.
- **MN5 SLURM:** `--account=etur59 --partition=acc`; `--mem` is rejected (memory is derived
  at 8G/core, hence the raised `--cpus-per-task` with inline `was N` comments — do not
  re-add `--mem`); `acc_debug` MaxWall is 02:00:00, `acc_ehpc` is 3 days. `--gres=gpu:4` on
  acc nodes means 4×H100, so the `a6000`/`l40s` names in some filenames are stale.
  `#SBATCH --output=` paths are absolute and not derived from `OUT_ROOT`; the directory must
  exist at submit time.

---

## 7. What was changed to make this work

- **New `eval_pipeline/mn5_paths.sh`** — one copy of the anchors, replacing nine drifted
  copies, plus three guards: `mn5_require` (reports every missing input, then exits 1),
  `mn5_require_gpus` (a missing `--gres` used to make the job exit 0 having done nothing),
  and `mn5_require_slurm` (login-node runs used to claim scenes and then get killed,
  permanently skipping them).
- `DATA_ROOT` → `/gpfs/scratch/etur59/koc821022`; `SCENES_ROOT` promoted to its own
  overridable variable with the `scenes/` component dropped.
- `EVAL_SCRIPT` → the vendored `eval_bundle/bin/eval_depth_poses.py`. **The old
  `sed -i` bring-up step is obsolete and actively harmful** — all nine runners already used
  `${EVAL_SCRIPT:-…}`, so the sed replaced an overridable line with a hard path.
  `eval_bundle/README.md:81-85` claims otherwise; that claim was already wrong.
- Overfit-episode paths → `OVERFIT_ROOT`/`OVERFIT_SCENE`, with the vanished
  `13062452+wrist` level removed. Same substitution in `verify_gru_ckpt_matrix.py`, and
  `DL3DV_TEST_ROOT` in `verify_gru_grid.py`, `verify_gru_v3_levers.py`,
  `verify_prev_flags.py`, `verify_pose_gru_e2e.py`, `probe_gru_hidden_falsifier.py`.
- Hardcoded checkpoints made overridable and guarded: `CKPT` in
  `run_captain_ray_eval_node.sh` / `smoke_captain_ray_eval.sh` / `sweep_prev_gt_oom.sh` /
  `run_prevpred_baseline_a6000.sbatch`, `CKPT_GT`/`CKPT_PP` in `smoke_prev_variants.sh`,
  `CKPT_DIR` in `run_cut3r_eval.sh`, `BEST_CKPT` in `run_fix_best.sh`, `CKPTS` in
  `eval_gru_overfit_all.sbatch` / `run_gru_window64_eval.sbatch`,
  `GRID_CKPT_ROOT` in `probe_gru_grid_all.py` (which ignored `$CKPT_ROOT` entirely).
- `run_prevpred_baseline_a6000.sbatch` had `--scenes_root` and `--eval_script` inline on the
  python call, immune to any env override — now wired to the anchors.
- Removed `#SBATCH --nodelist=ai17` from `run_regular_parallel.sh` (pre-MareNostrum node;
  pends forever on `partition=acc`).
- `eval_gru_grid_overfit.sbatch` with no arguments used to exit 0 silently — now a usage
  error. `run_gru_window64_eval.sbatch` now refuses to start without its window tree.
- `aggregate_results.py` captioned its own tables `pose SE3-aligned` while the default is
  `--align sim3`; now `sim3-aligned, RMSE-reduced`.
- Table builders: eval root and scene id made overridable, and they now create their own
  `tables/` output directory (all four used to crash on a missing parent).

### Verified on MN5, not assumed

- All 4292 scenes in `scene_list.txt` resolve under `SCENES_ROOT`; the copy at
  `$OUT/scene_list.txt` matches the documented sha256.
- Every anchor in `mn5_paths.sh` resolves; `mn5_require` exits 1 with a full missing list;
  a login-node launch is refused and leaves `$OUT` untouched (this was found the hard way —
  an earlier dry-run did start workers on `glogin1` before the guard existed).
- Evaluator end-to-end on real MN5 data (GT vs itself through a correctly shaped pred tree):
  `absrel 0.0, a1 1.0, ate 7.9e-17, rpe_trans 1.1e-16, rpe_rot 2.4e-7`.
- `aggregate_results.py` consumes that CSV and emits the summary tables with the corrected
  caption.
- `bash -n` on every runner and `.sbatch`; `py_compile` on every touched Python file.

**Not verified:** no GPU inference was run — the worker → preds → eval → aggregate chain has
not been exercised end-to-end on this cluster. Run §3.5 first; it is the cheapest thing that
would catch a real breakage.

### Out of scope, still stale

Training-side paths still point at `/frozen` or `/scratch/bdursun25`: 37 files under
`src/CUT3R/config/` (the v3 `*_finetune` configs are already MN5-correct; the v2 grid
configs are not), plus `stage_to_scratch.sh`, `relink_splits_to_scratch.sh`,
`run-all-inference.sh` and `overfit_all_tasks_demo_0_commands.txt`. `run_vggt_node.sh`
keeps its dead `VGGT_ROOT` default because the dependency genuinely is not here.
