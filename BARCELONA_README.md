# BARCELONA_README — orientation for agents working in this repo on MareNostrum5

**Read this first.** It is the compressed map of (a) what this repo is doing scientifically,
(b) what the MareNostrum5 (BSC) environment looks like, and (c) the traps that will waste
your first hour. Written 2026-08-04 against `captain_gru_v3` @ `aab1778`.

Companion docs: `TRAINING_SETUP.md` (original-cluster setup), `REPORT.md` (full experimental
log), `captain_gru_grid_ANALYSIS.md` (lever-by-lever results),
`src/CUT3R/config/captain_gru_v{2,3}_grid_README.md` (grid protocol).

---

## 0. TL;DR — the state you are walking into

We are **finishing the migration from the old "avg"/valar cluster to MareNostrum5**. There
has been **no new science since `69d62f6`**; the port is the work in flight.

**The MN5 port has now largely landed** (83 files changed). Paths, SLURM directives, the
conda hook and the worktree anchors are migrated; the pretrained checkpoint is present and
the CUDA RoPE kernel is compiled. **The one deliberate exception is the dataset** — the copy
was still running, so every `train_data`/`test_data` `ROOT=` and every scene-store path was
left pointing at the OLD location on purpose, so it stays visibly broken rather than
silently wrong. See §3.

| | |
|---|---|
| Repo | `/gpfs/home/koc/koc821022/my-da3`, branch `captain_gru_v3` |
| Machine | MareNostrum5 GPP login node `glogin1` (BSC), user `koc821022`, groups `koc` + `etur59` |
| Conda env | `~/.conda/envs/cuteanything` — complete and working (py3.11.14, torch 2.8.0+cu128) |
| Dataset | staged at `/gpfs/scratch/etur59/koc821022` — see §3. **Paths deliberately NOT repointed yet.** |
| Pretrained ckpt | ✅ present, 3.17 GB, at `src/CUT3R/src/cut3r_512_dpt_4_64.pth` |
| CUDA RoPE kernel | ✅ compiled and verified (7/7 checks) — see §5 |
| Checkpoint output | `/gpfs/scratch/etur59/koc821022/checkpoints` — new runs (since 2026-08-14). Older runs moved to `/gpfs/scratch/etur59/koc821022/checkpoints_projects` |
| `git push` | **broken** — origin is an SSH tunnel to a laptop that isn't up |

**Verified working right now:** `verify_pose_gru_loss.py` (14), `verify_pose_gru_trainer.py`
(15) and `verify_pose_gru_config.py` (22) all pass on the login node; all 42 hydra configs
compose; all 36 shell scripts parse; a real `sbatch verify_prev_flags.sbatch` schedules on
`acc`, activates conda, self-locates the worktree and loads the pretrained checkpoint —
failing only at the deferred dataset path, exactly as intended.

---

## 1. What the repo is

Three layers, only one of which is live:

- **`src/depth_anything_3/`** — upstream ByteDance Depth Anything 3, essentially untouched.
  DINOv2 ViT backbone with alternating local/global attention, one `DualDPT` head emitting
  depth + camera-ray maps. Not part of the active research line. (Its CLI is partly broken —
  see §8.)
- **`with_cut3r_v{0..3}.py`** — a **dormant** DA3↔CUT3R bridge: monkeypatch
  `ARCroco3DStereo._encode_views` so CUT3R's per-frame encoder is replaced by DA3's
  cross-view tokens. Mostly dead code; only `get_cut3r_encoder_outputs_from_da3` and
  `save_geometry_outputs_cuteanything` are live.
- **`src/CUT3R/` + the `captain_*` configs + the root `verify_*`/`probe_*` harnesses** —
  **this is the work.** Everything on this branch is here.

### The research question

CUT3R reconstructs 3D online from streaming video. **How much of its error on DROID
robot-wrist video is actually _camera-localization_ error?** If you feed camera geometry
back in at each timestep, do depth and pointmaps improve?

**The channel is free.** `cut3r_512_dpt_4_64.pth` already contains a fully pretrained
ray-map encoder branch (29 tensors: `patch_embed_ray_map.*`, `enc_blocks_ray_map.{0,1}.*`,
`enc_norm_ray_map.*`, `masked_ray_map_token`) that upstream never activates for
DL3DV-format data. **Zero new parameters, strict load.** This is the whole reason the route
was chosen over the rejected "captain adapter" (see §9).

### The ladder of flags

Each defaults to `False`, i.e. **byte-identical to upstream CUT3R when off**:

1. `feed_gt_ray_map` — GT camera of frame *t*. The oracle ceiling.
2. `feed_prev_gt_ray_map` — GT camera of frame *t−1*.
3. `feed_prev_pred` — the model's **own predicted pose** at *t−1*. A real, deployable closed loop.
4. `pose_gru_*` — a 128-d GRU refiner sitting on top of `feed_prev_pred`.

---

## 2. Findings so far — three regimes that disagree

**You must quote all three.** Cherry-picking any one of them misrepresents the result.

| Regime | Verdict | Numbers |
|---|---|---|
| In-training 4-view test (last-10-epoch mean) | conditioning **loses** | no-conditioning absrel **0.0376** / ate 0.00054 · `captain_ray_prev_pred` 0.0395 · best GRU `a4_g1` 0.0422 · `a4_g1_r8` 0.0502 |
| Epoch-50 32-view pose probe (GRU's own pose error vs a "reuse P(x−1)" lag baseline) | GRU **wins big** | A3 **+24.2/+21.5/+22.1 %** (G0/G1/G2) · A4 +16.2/+17.9/+17.2 % · A2 +7.4/+11.8/+4.8 % · A1 −10.2/−8.1/−12.5 % (never learns) |
| 672-frame closed-loop rollout (the regime the filter exists for) | ATE **sign flips** | plain CUT3R AbsRel 0.0981 / ATE 0.0780 · prev-GT oracle 0.1314/**0.0183** · prev-pred no GRU 0.1244/0.0765 · best GRU `a3_g0` 0.1213/**0.0486** |

The GRU closes **~48 %** of the pred-ray→GT-ray ATE gap. Depth never recovers to the
no-conditioning baseline, and `rpe_rot` is worse than the no-GRU control for **every** arm.

**The uncomfortable headline:** stripping the GRU entirely at eval changes head pose by
**≤3 %** and scale-aligned pts3d by **≤0.1 %** for every v2 arm. The filter improves its own
pose estimate and barely touches the reconstruction.

### Lever-level

- **A lever (information):** the win is the **`pose_delta` input**, not the head algebra.
  A1→A3 = +32 pp, A2→A4 = +7 pp. Component split (error as a multiple of that arm's own lag
  baseline; 1.00 = learned nothing): A3 translation **0.80–0.81** / rotation **0.99–1.08**
  (worthless on rotation); A4 translation 0.85–0.86 / rotation **0.89–0.90**. Frame-to-frame
  rotation is near-identity, so `q(x−1)` is a prior that a zero-init residual exploits and a
  from-scratch regression cannot match.
- **77-window reliability eval** (64-frame windows, stride 8, job 1421570): ATE/absrel are a
  **statistical tie** across A3/A4/G. The one reproducible A4 advantage is `rpe_rot`
  **0.1784** (A4, all three G) vs 0.1943–0.1997 (A3), ~11 %, 3-vs-3 no overlap.
- **G lever (gradient flow only — forward math never changes):** accuracy differences sit
  inside the epoch-to-epoch sd. The real effect is memory reliance: `POSE_GRU_HIDDEN_ZERO`
  degradation 13.5 % (G0) → **25.9 %** (G1) → 11.9 % (G2). **Verdict: keep G1, drop G2.**
- **R lever (`pose_gru_iters`, RAFT-style):** refinement is real but **non-monotone** —
  anytime-N on `a4_g1_r8` gives 0.15644 / 0.15180 / **0.14567** / 0.15316 at N=1/2/4/8.
  **N=4 is the optimum; N=8 over-corrects.** R8 also **destroyed the recurrence**:
  HIDDEN_ZERO degradation collapses 25.9 % → −0.4 %.
- **F lever (pooled pre-ray image features into the cell):** unambiguously read —
  `POSE_GRU_IMG_FEAT_ZERO` degrades gru pose error by **+93.0 %** (`a4_g1_f1_r8`). It is the
  **only lever with a downstream effect** (stripping the GRU costs head pose +2.0 % vs
  ~0.0–0.6 % for every v2 arm).
- **A5 (`split_anchor`) is the current frontier arm.** Motivation: rotation is only 3–4 % of
  the unweighted `PoseGRULoss` (trans 0.141 vs quat 0.0047), so the aux gradient can never
  teach rotation — the anchor must be **structural**. Success criterion: translation ≤0.81×
  **and** rotation ≤0.90× of lag simultaneously, plus `rpe_rot` matching A4's 0.178 on the
  77-window eval. Jobs 1422852/1422853 were launched; **no results exist in any document.**

### Work built but never reported

`a4_g3_f1` (the missing F1×R1 control — without it F cannot be separated from R),
`a4_g3_f1_r4`, `a5_g1`, `a5_g3`, `a3_g3`/`a4_g3`/`a4_g3_r8`/`a4_g3_f1_r8`, and
`captain_gru_v2_a4_g1_finetune.yaml` (full-DROID, never run).

**Recommended next arm, verbatim from the analysis:** A4 (+ an A3 twin) + F1 + R4, γ-sum
renormalised, G1, trained well past 50 epochs — plus the F1×R1 control.

---

## 3. The dataset on MareNostrum5 — `/gpfs/scratch/etur59/koc821022`

This is **the** data location on this machine. Everything the configs say about
`/frozen/avg/...` is from the old cluster and is dead.

```
/gpfs/scratch/etur59/koc821022/
├── pointworld_droid_wrist_VALAR/      # the scenes (DROID wrist episodes)
│   └── <lab>+<hash>+<timestamp>/
│       └── dense/{rgb,depth,cam,sky_mask,outlier_mask}
└── dl3dv_multi/
    └── splits/                        # the canonical split definition
        ├── train.txt                  # 38633 scene names
        ├── test.txt                   #  4292 scene names
        └── splits.json                # per-lab stratified 90/10, seed=42, 42925 scenes
```

**Copy is INCOMPLETE and still running.** 39,172 of 42,925 scene dirs present at last check
(alphabetical order, ~25/min, from another session). **Do not start a run or a dataset scan
until it finishes** — the DL3DV disk cache is keyed on `abspath(ROOT)` and would bake in a
truncated scene list. Check with:

```bash
ls /gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_VALAR | wc -l   # want 42925
```

### The layout does not match what the code expects — one symlink fixes it

`regenerate_split_symlinks.sh:35-36` requires `<DATA>/wrist/` **and** `<DATA>/splits/` under
one directory. On disk, `splits/` is under `dl3dv_multi/` and the scenes are its **sibling**.

```bash
cd /gpfs/scratch/etur59/koc821022
ln -s ../pointworld_droid_wrist_VALAR dl3dv_multi/wrist          # <- currently MISSING
cd /gpfs/home/koc/koc821022/my-da3
./regenerate_split_symlinks.sh /gpfs/scratch/etur59/koc821022/dl3dv_multi \
                               /gpfs/scratch/etur59/koc821022/pointworld_droid_splits
```

**This is not optional and you cannot skip it by pointing `ROOT` at the scene dir.**
`dl3dv.py:_build_raw_scan` (:202-227) hardcodes a three-level walk
`ROOT/<l1>/<l2>/dense/rgb`; pointing `ROOT` straight at `pointworld_droid_wrist_VALAR` makes
`l1=<scene>`, `l2='dense'` and it then looks for `<scene>/dense/dense/rgb`.

After the script, point config ROOTs at `<SPLITS>/{train,test}/dl3dv_multi` and set
`DL3DV_CACHE_DIR` to a scratch path before the first scan.

### Quota — thin

Group `gpfs_scratch` is at **16.15 TB of a 16.65 TB soft quota** (17.48 hard) with the copy
still running. `gpfs_projects/etur59` has ~1 TB free and is essentially empty — use it if you
need room. Check with `bsc_quota`.

---

## 4. The MareNostrum5 environment

| | old avg/valar cluster | MareNostrum5 | status |
|---|---|---|---|
| SLURM account | `--account=avg` | `--account=etur59` | ✅ migrated |
| Partition | `--partition=avg` | `--partition=acc` | ✅ migrated |
| QOS | *(none set)* | `acc_debug` ≤2h · `acc_ehpc` ≤3d | ✅ added per job |
| GPU request | `--gres=gpu:lovelace_l40s:N` | `--gres=gpu:N` (4×H100 per node) | ✅ migrated |
| **Memory** | `--mem=NNNG` | **`--mem` is REJECTED at submit** | ✅ removed everywhere |
| Walltime | up to `168:00:00` | `acc_ehpc` MaxWall = **72h** | ✅ clamped |
| Conda hook | `/opt/ohpc/pub/compiler/conda3/...` | `/apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh` | ✅ migrated |
| Worktree | `/scratch/bdursun25/cuteanything/...` | derived from `$SLURM_SUBMIT_DIR` | ✅ self-locating |
| Checkpoints | `.../cuteanything/checkpoints` | `/gpfs/scratch/etur59/koc821022/checkpoints` (old runs: `.../checkpoints_projects`) | ✅ migrated |
| Data | `/frozen/avg/bora_data/...` | `/gpfs/scratch/etur59/koc821022/...` (§3) | ⏸️ **deferred on purpose** |

Three MN5 constraints worth internalising, all found the hard way:

1. **`--mem` is rejected at submit time.** `sbatch` errors with *"You cannot submit a job
   requesting memory parameters, memory is automatically set for each asked cpu … ACC:
   8G/core"*. Memory now derives from `--cpus-per-task`, so several jobs had their core
   counts **raised** to preserve the original memory guarantee (each is annotated inline).
2. **`acc_ehpc` MaxWall is 3 days.** The four `--time=168:00:00` scripts would have been
   rejected outright; they are clamped to `72:00:00`.
3. **This account has NO `gpp` (CPU-partition) QOS** — only `acc_debug`, `acc_ehpc`,
   `acc_interactive`. CPU-only jobs (`verify_prev_flags.sbatch`, `stage_to_scratch.sh`)
   therefore stay on the `acc` partition; there is no CPU-partition option to move them to.

Notes:
- `sinfo` is **permission-denied** for regular users here; use `scontrol show partition`.
- `~/.bashrc` is **clean** on MN5. The "line 1 is corrupted, do not source it" warning in the
  launchers is an avg-cluster problem only.
- `/scratch` on MN5 is a tiny local disk — **not** the old cluster's `/scratch`. Use `/gpfs/scratch`.
- The env `~/.conda/envs/cuteanything` already exists and works: py 3.11.14, torch 2.8.0+cu128,
  torchvision 0.23.0, xformers 0.0.32.post2, accelerate 1.10.1, hydra-core 1.3.2, wandb 0.25.1,
  evo 1.31.1, gsplat 1.5.3, lpips, pycolmap 3.13.0, open3d 0.18.0, cv2 4.11.0, gdown 5.2.0.

### PYTHONPATH — the one thing you always need

```bash
cd /gpfs/home/koc/koc821022/my-da3
export PYTHONPATH="$PWD:$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src"
```

`pyproject.toml:60` sets `packages = ["src/depth_anything_3"]` **only** — `pip install -e .`
never installs dust3r/croco/CUT3R. That is *why* PYTHONPATH is mandatory rather than a
convenience.

⚠️ The launchers export only **three** entries (`$WT:$WT/src:$WT/src/CUT3R`, commented "Full
three-tree path" at `train_captain_gru_overfit.sbatch:30-34`) and get the fourth implicitly by
`cd`-ing into `src/CUT3R/src`. `TRAINING_SETUP.md:19` documents four. **If you port and change
one without the other, you silently change which tree `dust3r` resolves from.**

⚠️ `src/CUT3R` must be on the path or the trainer's `from eval.monodepth.tools import
depth_evaluation` fails inside a try/except (`train_cut3r_baseline.py:760-768`) and the
absrel/a1/video-depth metrics **silently vanish** from the logs. This bit the project once
(fixed in `6e033be`).

---

## 5. Getting to a first run on MN5 — what is done, what is left

**Done** (do not redo):

1. ✅ **Pretrained checkpoint** present — 3.17 GB at `src/CUT3R/src/cut3r_512_dpt_4_64.pth`.
   (If it ever goes missing: `gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view`,
   documented at `src/CUT3R/README.md:82-99`.)
2. ✅ **CUDA RoPE kernel compiled and verified.** `sbatch build_curope.sbatch` (1 GPU,
   `acc_debug`, ~15 s) builds it and then runs `verify_curope.py`, which checks far more than
   "does it import": numerical agreement with the pure-PyTorch RoPE (max|diff| **2.7e-06**),
   both real `blocks.Attention` / `blocks.CrossAttention` call sites, and whole-block output
   equivalence kernel-vs-fallback (**5.7e-05**). 7/7 pass on H100.
   - Gotcha for anyone rebuilding: do **not** `module load nvidia-hpc-sdk/25.3`. It setenv's
     `CC`/`CXX` to `nvc`/`nvc++`, which fails on Python's `-fwrapv` and makes nvcc abort with
     *"Unsupported NVHPC compiler found"*. Point `CUDA_HOME` at the SDK's CUDA tree and keep
     GCC — that is what `build_curope.sbatch` does.
   - Gotcha for anyone testing it: the kernel rejects plain-contiguous tensors. `cuRoPE2D`
     passes `tokens.transpose(1,2)` and `kernels.cu:91` asserts `stride(3)==1 && stride(2)==D`
     on that view. Build test tensors as contiguous `(B,N,H,D)` viewed as `(B,H,N,D)`.
3. ✅ **Paths, SLURM directives, conda hook, worktree anchors migrated** (§4).
4. ✅ **Checkpoint/output dirs created** at `/gpfs/projects/etur59/koc821022/{checkpoints,outputs}`.

**Left**:

5. ⏸️ **The dataset.** Wait for the copy, then the symlink + `regenerate_split_symlinks.sh`
   (§3), then repoint the deferred paths. Find every one of them with:
   ```bash
   # NB: /usr/bin/grep, not `grep` -- see the warning below
   git ls-files -z | xargs -0 /usr/bin/grep -nI '/frozen/avg\|/scratch/bdursun25\|/leonardo_scratch'
   ```
   The deferred set is: every `train_data:`/`test_data:` `ROOT=` in `src/CUT3R/config/*.yaml`;
   `DATA_ROOT` in `verify_prev_flags.py`, `verify_gru_v3_levers.py`, `verify_gru_ckpt_matrix.py`,
   `probe_gru_hidden_falsifier.py`; the `DATA_ROOT=${DATA_ROOT:-...}` default in the 12
   decomposed launchers; `SCENES_ROOT` in `eval_gru_grid_overfit.sbatch`; and the scene paths
   in `run-all-inference.sh`, `relink_splits_to_scratch.sh`, `stage_to_scratch.sh`.
6. ⏸️ **`streaming-3d` is not on MN5.** `eval_depth_poses.py` and `InfiniteVGGT` are in a
   separate, un-vendored repo, so nothing in `eval_pipeline/` can run. Each site is now
   `EVAL_SCRIPT=${EVAL_SCRIPT:-...}` with a `TODO(mn5)`.
7. ⏸️ **Restore the git tunnel** to push — origin is `ssh://git@localhost:2222/borakd/my-da3.git`.

> ⚠️ **The `grep` in this shell is a ugrep wrapper that respects `.gitignore`.** Since
> `.gitignore` contains `*.sh`, a plain `grep -r` **silently skips every shell script** — it
> hid 55 hits across 7 tracked files during this migration. Use `/usr/bin/grep`, and prefer
> `git ls-files -z | xargs -0 /usr/bin/grep` for repo-wide sweeps.

---

## 6. How to run things

### Verify — climb this ladder in order

```bash
python verify_pose_gru_loss.py      # 14 checks — CPU, no ckpt, no data, no hardcoded paths
python verify_pose_gru_trainer.py   # 15 checks — adds accelerate + croco + trainer imports
python verify_pose_gru_config.py    # 22 checks — adds hydra; MUST run from repo root (:13 is CWD-relative)
```

✅ **All three pass on the MN5 login node today** with the existing env and the §4 PYTHONPATH.
They are the only harnesses runnable here without edits.

The GPU tier (all need a GPU + the 3.2 GB ckpt + the dataset, all still hardcoded to the old
cluster): `verify_gru_grid.py` (34 checks, A/G levers — the documented "re-verify on a new
machine" step), `verify_gru_v3_levers.py` (28, F/R), `verify_gru_a5.py` (31, split_anchor —
**currently broken, see §8**), `verify_pose_gru_e2e.py` (22), `verify_prev_flags.py` (CPU-only,
730 lines), `verify_gru_ckpt_matrix.py` (proves every run dir is `demo_ray.py`-runnable).

Exit contract: the big harnesses accumulate a `CHECKS` list and `sys.exit(1 if n_fail else 0)`;
the three small ones raise a bare `AssertionError`.

### Train

```bash
sbatch smoke_train_gru_v3.sbatch                               # 1 GPU, ~2 min end-to-end smoke test
sbatch train_captain_gru_overfit.sbatch captain_gru_v3_a5_g1   # 1 node, 2 GPU (default: captain_gru_v2)
sbatch train_captain_gru_18gpu.sh                              # 6 nodes x 3 GPU
```

**Start with the smoke test on any new machine or after touching the trainer.**
`smoke_train_gru_v3.sbatch` + `src/CUT3R/config/captain_gru_v3_a4_g3_f1_r8_smoke.yaml` run the
newest lever set (A4×G3×F1×R8) for 2 epochs over 16 samples of ONE real staged DROID episode.
Every lever field is byte-identical to the parent `captain_gru_v3_a4_g3_f1_r8.yaml`; only run
size, `save_dir`/`exp_name` and the data ROOT differ. Verified working: **COMPLETED, exit 0,
2m03s on one H100**.

Checking it is *clean*, not merely non-crashing — the four things to grep for:

| Lever | Proof it actually engaged |
|---|---|
| **A4** (pose_delta → residual) | ckpt `pose_gru.cell.weight_ih` has input dim **46** = 14 (pose 7 + delta 7) + 32 (img feat). A2 would be 7. |
| **G3** (bptt ∧ e2e) | `ckpt['args'].pose_gru_bptt/​_e2e == True`. These are TRAINING-time gradient switches set by `train_cut3r_baseline.py:346-347`; `load_model` does **not** restore them and should not — at eval `e2e` is gated on `torch.is_grad_enabled()` (`model.py:1281`). |
| **F1** (image features) | `pose_gru.img_proj.weight` is `(32, 2048)` = `Linear(1024×2 frames → 32)`, and its **‖W‖₁ ≠ 0** — the zero-init projector received gradient. Zero would mean F was inert. |
| **R8** (RAFT iterates) | `gru_pose_loss_iters` appears in the logs (the γ=0.8 sequence loss, train criterion only), and `ckpt['args'].pose_gru_iters == 8`. |

Also confirm **`absrel` and `a1` appear** — if they don't, `src/CUT3R` fell off PYTHONPATH and
the depth metrics were silently dropped (§4).

### W&B: compute nodes have NO internet — record offline, port from the login node

```bash
# training: already handled by smoke_train_gru_v3.sbatch
export WANDB_MODE=offline
export WANDB_RUN_ID=<pin an id>      # else the trainer generates a fresh one per launch

# afterwards, ON THE LOGIN NODE:
wandb login                                            # one-time; not currently configured
./wandb_port_offline_runs.sh --dry-run <run_output_dir>   # list, works without auth
./wandb_port_offline_runs.sh <run_output_dir>             # upload
./wandb_port_offline_runs.sh --all                        # everything under $CKPT_ROOT
```

**Why this is lossless, and the one thing you must not do.** `train_cut3r_baseline.py:179-190`
calls `wandb.tensorboard.patch(root_logdir=<output_dir>/tb)` then `wandb.init(sync_tensorboard=True)`.
`tensorboard.patch` **monkeypatches `SummaryWriter` in-process**, so every `add_scalar` is folded
into the wandb run's own transaction log *as it is written* — not scraped from tfevents later.
Verified empirically here on wandb 0.25.1: the smoke run's offline
`run-<id>.wandb` contains **202 train/test scalar keys** including `absrel`, `a1`,
`gru_pose_loss`, `gru_pose_loss_iters`, `gru_trans_loss`, `gru_quat_loss`, plus the 8
visualization images.

So `wandb sync <offline-run-dir>` replays everything verbatim — same run id, name, config,
history, summary, step axis, console log and media. ⚠️ **Do NOT also pass `--sync-tensorboard`
or sync the `tb/` directory**: that imports the same scalars a second time and corrupts the step
axis. `wandb_port_offline_runs.sh` deliberately never does, and is idempotent (skips `.synced`).

Distributed is always HuggingFace `accelerate launch`, never `torchrun`, with no
`accelerate config` file — every flag on the command line. Multi-node rendezvous:
`MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)`,
`MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))`, `srun --ntasks-per-node=1` →
`accelerate launch --multi_gpu --num_machines N --num_processes N*G --machine_rank $SLURM_NODEID`.

### Probe (measurement, not pass/fail — these always exit 0)

```bash
python probe_gru_hidden_falsifier.py <ckpt>          # 4 arms: normal / hidden_shuffle / hidden_zero / no_gru
python probe_gru_grid_all.py --out <dir> [--shard i --num_shards 4]
```

⚠️ **Only ever point a probe at a `keep_freq` snapshot (`checkpoint-<N>.pth`) or
`checkpoint-final.pth` — never `checkpoint-last/best.pth` while a job is alive.** Those are
rewritten mid-epoch and `torch.load` races the writer.

### Eval

```bash
python eval_pipeline/make_scene_list.py --scenes_root <...> --out scene_list.txt
sbatch eval_pipeline/run_captain_ray_eval_node.sh      # 4 GPU, cooperative mkdir claim queue
python eval_pipeline/aggregate_results.py --out_root <...> --scene_list <...>
sbatch run_gru_window64_eval.sbatch                    # the 77-window reliability eval
```

⚠️ `eval_pipeline` additionally needs `eval_depth_poses.py` from a **different, un-vendored
repo** (`streaming-3d`). It cannot run until that repo is present.

Inference arms: `demo_ray.py` and `eval_pipeline/infer_and_eval_worker_ray.py` both take
`--conditioning gt|prev_gt|prev_pred|prev_pred_gru|none`, deliberately kept identical.
`prev_pred` sets `feed_prev_pred=True` **and `model.pose_gru = None`** — the true GRU-less
ablation.

---

## 7. The architecture — how PoseGRU actually works

Everything lives in **`src/CUT3R/src/dust3r/model.py`**. Key anchors:
`pose_delta_encoding` :306 · `PoseGRU` :332 / `.forward` :450 · `load_model` :76 ·
`load_state_dict` override :645 · `enable_pose_gru` :703 · `_encode_views` :888 ·
**`_forward_decoder_group_step` :1101 (the whole closed loop)** · `_forward_impl` :1477.
Losses in `src/CUT3R/src/dust3r/losses.py`, `PoseGRULoss` at :1077.

### The closed loop (`model.py:1101`)

Under `feed_prev_pred`, view *x* receives **no dataset-side ray map** (asserted :968-971):

- **View 0** (:1129-1149) resets `_prev_pred_pose_enc`, `_prev_prev_pred_pose_enc`,
  `_pose_gru_hidden`, `_prev_img_feat`; keeps the pretrained `masked_ray_map_token`.
- **View x ≥ 1** (:1150-1313): read stashed `P(x−1)` [and `P(x−2)`] → env falsifiers →
  **PoseGRU refine** (fp32, autocast disabled, *not* under `no_grad`) →
  `pose_encoding_to_camera` → `get_ray_map_torch(...)` → `_encode_ray_map` →
  `feat_group[0] += ray_out[-1]`.
- **After the head runs** (:1397-1417): the stash takes `res_group[-1]['camera_pose'].detach()`
  — **always the RAW head pose, never the GRU output**. `res_group[-1]['gru_pose']` carries the
  grad for the aux loss.

`feed_prev_pred` requires `views_per_step == 1` (asserts `model.py:1128`).

### The module

`nn.GRUCell(input_dim, hidden_dim=128)` + **one shared** `nn.Linear(hidden, 7)`.
Consumes a detached `(B,7)` `absT_quaR` encoding of `P(x−1)`, or `(B,14)` with
`pose_delta_encoding(P(x−2), P(x−1))` appended. Three head modes (:471-482), with a
**terminal `raise`** on anything unrecognised:

- `residual` — `t = in[:3] + raw[:3]`, `q = in[3:7] + raw[3:7]`, **fully zero-init head**
  (an untrained model is output-identical to plain prev_pred)
- `direct` — `t = raw[:3]`, `q = raw[3:7]`
- `split_anchor` (A5) — `t = raw[:3]` but `q = in[3:7] + raw[3:7]`, with **only head rows/bias
  3:7** zeroed

A5 keeps a **byte-identical parameter set** to A3/A4, so checkpoints interchange — the mode is
recoverable **only** from `ckpt['args']`, never from weight shapes.

### The levers

| Lever | Keys | Values |
|---|---|---|
| **A** (what the GRU consumes/emits) | `pose_gru_input` + `pose_gru_mode` | A1 = pose/direct · A2 = pose/residual · A3 = pose_delta/direct · **A4 = pose_delta/residual** · A5 = pose_delta/split_anchor |
| **G** (where gradients flow — **forward math never changes**) | `pose_gru_bptt`, `pose_gru_e2e` | G0 = F/F · G1 = T/F (hidden tape spans one TBPTT chunk) · G2 = F/T · **G3 = T/T, the real superset** |
| **F** (image features into the cell) | `pose_gru_img_feat: input\|none`, `_dim: 32`, `_frames: 2` | Zero-init `Linear(1024*frames → D)` appended **after** the pose block, so `gru_in` keeps its pose-first 7/14 layout and the residual anchor is untouched |
| **R** (RAFT iterations) | `pose_gru_iters`, `pose_gru_iter_detach: True`, `pose_gru_iter_gamma` | The **call site** owns the loop (:1220-1256); the module stays a pure single-step cell. Zero new params. Ray map built **once**, from the final iterate. |

**Enable order is non-negotiable** (`train_cut3r_baseline.py:240-263`): `enable_pose_gru()`
runs **before** the pretrained/resume load, **before** `.to(device)`, and **before** optimizer
construction (`pose_gru_lr_scale` default **100.0**).

### The six env-var falsifiers (read inside the per-view forward)

| Var | Line | Effect |
|---|---|---|
| `PREV_PRED_RAY_SHUFFLE=1` | :1157 | rolls `prev_pose_enc` **and** `prev_prev` together so the (pose, delta) pair stays coherent. **No-op at batch size 1.** |
| `POSE_GRU_HIDDEN_SHUFFLE=1` | :1174 | rolls the recurrent memory across samples. No-op at B=1. |
| `POSE_GRU_HIDDEN_ZERO=1` | :1182 | `hidden = None` → stateless control. **Works at B=1.** Evaluated *after* shuffle, so ZERO wins if both set. |
| `POSE_GRU_IMG_FEAT_SHUFFLE=1` | :1207 | rolls the (current, previous) image pair. F + B>1 only. |
| `POSE_GRU_IMG_FEAT_ZERO=1` | :1217 | zeroes **before** `img_norm`/`img_proj`. Works at B=1. |
| `POSE_GRU_FORCE_ITERS=<n>` | :1230 | `max(1, int(v))` — the anytime-N probe. |

Dataset-side sibling: `GT_RAY_MAP_SHUFFLE=1` (`base_multiview_dataset.py:411`).
Falsified hidden is **written back into the stash**, so hidden effects compound across a rollout.

---

## 8. Traps — read before touching anything

### Will bite you in the first hour

- 🔴 **`.gitignore:38-39` is `*.sh` then `!eval_pipeline/*.sh`.** Any NEW root launcher shell
  script is **silently invisible to git**. Verify with `git check-ignore -v <file>`. **New
  `.sbatch` files are fine** (which is why `build_curope.sbatch` uses that extension), and
  `.md` is fine. This also breaks plain `grep -r` — see the warning in §5.
- ✅ ~~The `# FULL DATASET` comment block is a lie~~ — **FIXED.** In all 25 affected configs the
  redundant byte-identical commented duplicate was deleted, the inverted `# FULL DATASET`
  label corrected to say it is one 672-frame episode, and the misleading `# 38633 4292`
  annotated as the full-split counts *not used by that arm*. `captain_gru_v2_a4_g1_finetune.yaml`
  was **deliberately not touched** — its commented block genuinely differs, so it carries real
  information. It remains the only true full-dataset arm.
- ✅ ~~Launchers ignore where you submit from~~ — **FIXED.** All 26 launchers now derive the
  worktree from `$SLURM_SUBMIT_DIR` and hard-fail if it is not a my-da3 checkout; the 9 Python
  harnesses derive `WORKTREE` from `__file__` and `assert`. The underlying hazard is worth
  remembering: **`sys.path.insert(0, "<nonexistent>")` silently succeeds**, so a stale anchor
  makes imports fall through to `PYTHONPATH` and you verify a *different* checkout than the
  file you are editing, with no warning at all.
  - When adding a launcher, use `${SLURM_SUBMIT_DIR}` — **not** `${BASH_SOURCE[0]}`. SLURM
    copies the batch script into its spool dir, so `BASH_SOURCE` points at the spool copy.
    (`eval_pipeline/check_and_maybe_cleanup.sh` is the exception: it is a plain driver script,
    never an sbatch body, so `BASH_SOURCE` is correct there.)
- `src/CUT3R/config/train.yaml` **does not exist** while `train_cut3r_baseline.py:1307` sets
  `config_name='train.yaml'` — running without `--config-name` fails. Every launcher passes one.
- **PYTHONPATH arity was deliberately left alone everywhere.** Which trees `dust3r` resolves
  from is behaviour, not a path. In particular `train_aug_multinode.sh` still has its
  **one-entry** PYTHONPATH (the documented bug that silently drops the absrel/a1 depth
  metrics) — now flagged in-line. Fix it as its own change, not as part of a migration.

### Science / correctness hazards

- 🔴 **G2 IS NOT A SUPERSET OF G1** — the project's single biggest correction (`REPORT.md` §8b).
  `bptt` and `e2e` are independent booleans; every `*_g2` config set `bptt: False`, so G2 was
  G0+e2e. Every published g1-vs-g2 comparison traded multi-step credit **away** instead of
  adding the main-loss path. A3/A4/A5 `_g2` configs were renamed in place to `_g3`; A1/A2 keep
  genuine G2 configs; **already-trained `*_g2` checkpoints remain on disk under their old names
  as valid G0+e2e runs.** `_g2` means two different things depending on whether you are reading
  a config or a checkpoint directory.
- **A methodology trap explicitly marked "do not re-litigate"** (`REPORT.md` §8b): checks T13/T14
  originally gated gradient-tensor differences at 10× the G0 nondeterminism floor (2.9e-07).
  Invalid for any e2e arm — a G2-vs-G2 *repeat* differs by 2.11e-05, **larger** than the
  1.92e-05 G3-vs-G2 difference once called proof of a superset. Two repair attempts failed
  (jobs 1428195, 1428210). Both checks were rebuilt as exact all-or-nothing tests.
- **n=1 evals lie.** The "A3 is unstable" finding came from one 672-frame rollout per arm and
  did **not** reproduce over 77 windows. Any new arm comparison must go through
  `run_gru_window64_eval.sbatch`.
- **50 epochs is not converged.** `test_gru` minima land at epoch 46-49 for nearly every arm,
  and the A ranking *flipped* between epoch 30 and 50. **Never compare arms at mixed epochs** —
  that is why the uniform epoch-50 `probe_gru_grid_all.py` replaced the earlier probes.
- **Unresolved R8 confound:** `PoseGRULoss(iter_gamma=0.8)` at N=8 adds
  `1 + Σ_{k=1..7} 0.8^k = 4.16×` the aux gradient mass of R1. Renormalise the γ sum (or set
  `pose_gru_loss_weight ≈ 0.25`) before any final R verdict.
- **F is only ever measured on top of R8.** At forced N=1 the F1 arm is **worse** than F0
  (0.16301 vs 0.15644). F's isolated effect is inferred. `a4_g3_f1` exists to fix this and has
  no result.
- **`best_ckpt_agg` selects on a test loss that includes the PoseGRU aux term** — "best"
  checkpoints across arms are not on a common scale. Compare at fixed epochs or subtract
  `gru_pose_loss_avg`.
- **Do not "clean up" the zero-init asymmetry:** `img_proj` weight *and* bias are zero-init, but
  the GRUCell's new feature input columns keep their **default** init. Zeroing both sides
  permanently deadlocks the feature channel's gradient.
- **Do not rename any `pose_gru` tensor.** Every loader uses `strict=False` and the hard-fail
  only catches **size** mismatches. This is why A5 reuses one shared head.
- **A4→A5 warm start is wrong** even though the checkpoint loads strictly: A4's translation rows
  are a zero-init *correction*, so under A5's direct-translation algebra they emit t≈0, pinning
  the camera at the view-0 origin. (`verify_gru_a5.py:275-289` documents this in its own PASS message.)
- **`mode` and `iters` are invisible in weight shapes** — they ride `ckpt['args']`. `load_model`
  **raises** rather than defaulting when `pose_gru_mode` is missing (:109-115); mis-restoring
  `iters` is **silent**, and `verify_gru_v3_levers.py` is the only guard.
- **A stale exported falsifier var silently contaminates a training run** — nothing logs that
  one is active.

### Stale code that will crash or lie

- `verify_gru_a5.py` STAGE Y loops `for g in ("1","2")` (:244) opening `*_g2.yaml` files that no
  longer exist → uncaught `FileNotFoundError`. Should be `("1","3")`. On this machine it dies on
  the *first* iteration anyway — `:44` sets `CFG_DIR` under a nonexistent worktree.
- `probe_gru_grid_all.py` — **the tool that produced the headline numbers** — has a stale
  `CKPT_ROOT` (:41) and stale arm labels (:45-48 builds `v2_a{a}_g2` / `v3_a4_g2*` names that no
  longer exist). Its per-arm try/except (:135-138) means a re-run emits `{"error": ...}` JSONs
  for all 16 arms **without failing**.
- `run_gru_window64_eval.sbatch:36` still lists the renamed `a3_g2 a4_g2`.
- `MODEL_STR` is **hand-duplicated byte-identically** across `verify_gru_grid.py:76-83`,
  `verify_gru_v3_levers.py:44-51`, `verify_pose_gru_e2e.py:57-64`. It matches
  `captain_gru_v2_a4_g1.yaml:12` today but nothing enforces it — any change to the config's
  `ARCroco3DStereoConfig` args leaves all three "trainer-identical build" verifiers silently
  checking a different architecture.
- **No test suite, no CI.** `.pre-commit-config.yaml` exists (black 99, flake8 100,
  `check-added-large-files --maxkb=125`) but `pre-commit` is not installed and a 536 KB PNG is
  committed — the hook has demonstrably never run. The `verify_*.py` harnesses are the only
  regression net, and 7 of 10 cannot execute here.

### Latent bugs in the training stack (not currently triggered)

- `misc.load_model` (resume) is called **after** `accelerator.prepare` with the DDP-wrapped
  model (`train_cut3r_baseline.py:405`) while `pretrained` is loaded **before** with
  `strip_module`. Resuming a single-GPU checkpoint into a multi-GPU DDP run does a
  `strict=False` load where every key is unexpected — **weights load silently as random init.**
- Auto-resume: if `args.resume` is falsy and `<output_dir>/checkpoint-last.pth` exists it is
  picked up automatically (:194-197), which **skips** the `pretrained` load.
- `DL3DV_Multi` accepts `split=` and **never uses it** — train/test separation is purely which
  ROOT you point at. Several captain configs train and test on identical data.
- The DL3DV disk cache signature only fingerprints ROOT's level-1 dir mtimes/entry counts
  (`dl3dv.py:70-93`) — editing scene contents **through symlinks does not invalidate it**. Bump
  `CACHE_VERSION` (:68) or set `DL3DV_CACHE_DISABLE=1`.
- `forward_recurrent` (`model.py:1699`) and `inference_step` (:1639) do **not** route through
  `_forward_decoder_group_step` — no `feed_prev_pred`, no PoseGRU. `viser_utils.py:542` uses
  `inference_step`, so the interactive viewer silently runs an **unconditioned** model even with
  a GRU checkpoint loaded.
- `NCCL_P2P_DISABLE=1` was required on the old cluster. If rendezvous hangs on MN5, the
  documented next knobs are `NCCL_SOCKET_IFNAME=ib0` and `NCCL_DEBUG=INFO`
  (commented out at `train_captain_gru_18gpu.sh:60-62`).

### DA3-side bugs (only if you touch that half)

- `cli.py` is broken for 4 of 5 inference commands: `image` (:383), `images` (:462), `colmap`
  (:549), `video` (:626) reference `reference_view_strategy` while the actual Typer option is
  `ref_view_strategy` → `NameError` at call time. Only `auto` works.
- `da3_inference.py` is dead code. `feat_layer_39` hard-codes vitg (depth 40) — any non-giant
  DA3 KeyErrors. `process_sequence_multiview.py:222` passes inverted percentile bounds to
  `visualize_depth`.

---

## 9. Git state and history

- Branch **`captain_gru_v3` @ `aab1778`**; `origin` is unreachable (SSH tunnel down).
- **Uncommitted: the MN5 port** — 83 tracked files changed (+900/−562), plus three new
  untracked files: `BARCELONA_README.md`, `build_curope.sbatch`, `verify_curope.py`.
- `cuteanything_mn5.yml` carries two deletions: `depth-anything-3==0.0.0` (this repo's own
  local editable package; no PyPI presence, hard-fails `conda env create`) and
  `prefix: /home/bdursun25/...`. **The working tree is the installable version; HEAD is not.**
  The same `prefix:` defect was removed from `cuteanything.yml`, `cuteanything_valar.yml` and
  `cut3r_8_Mar_2026.yml` during the port.
- **31 commits ahead of `temp_clean_main`** (+16015/−467 over 107 files) but **NOT a descendant
  of it.** merge-base `9ec1c39`; main's tip has 5 commits the branch never took — including
  `9045208` "mem_same_step" with a **+94-line change to `src/CUT3R/src/dust3r/model.py`**, the
  exact file this branch rewrote (+795). **Any eventual merge to main will conflict there.**
- Arc: `temp_clean_main` → `captain_ray` (GT-ray oracle, `demo_ray.py`, prev-pred loop,
  `eval_pipeline`) → `captain_gru` (path repoint only) → `captain_gru_v2` (PoseGRU, the 12-config
  grid, `TRAINING_SETUP.md`) → `captain_gru_v3` (`69d62f6`: A5 + F/R levers + all harnesses +
  `REPORT.md`; `aab1778`: env port to MN5).

### Rejected route — preserved so it is not re-attempted

The **"captain adapter"** (GT pose → 768-d feature fused into `LocalMemory`), on
`temp_clean_main` at `9045208`. It fails because **no pretrained pose→feature encoder exists** —
`PoseEncoder` is defined and imported but has **no weights in the checkpoint** — so it must learn
a 761-dimensional-coset inverse of `PoseDecoder` from scratch through a truncated recurrence at
lr 1e-6. Its failure is uninterpretable, which is exactly what the zero-new-parameter ray-map
route avoids.

### Junk that is real repo content

`=2` (7.4 KB) is a **pip stdout log** captured by an unquoted `pip install ... torch>=2`
redirect, tracked since `7d5ddfb`. `source_zero_bytes.txt` (0 bytes) is the same class of
accident. `METRICS.md` describes a pipeline that no longer exists. `README.md`, `docs/*`,
`notebooks/da3.ipynb`, `src/CUT3R/docs/*` are verbatim upstream.
