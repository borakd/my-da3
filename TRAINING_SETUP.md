# captain_gru_v2 — training setup on a fresh machine

Branch `captain_gru_v2` of `git@github.com:borakd/my-da3.git`: CUT3R `feed_prev_pred`
closed loop + PoseGRU refiner + the full A1-A4 × G0-G2 lever grid
(see `src/CUT3R/config/captain_gru_v2_grid_README.md` for the grid map and protocol).

## 1. Clone + environment

```bash
git clone git@github.com:borakd/my-da3.git && cd my-da3
git checkout captain_gru_v2
conda env create -f cuteanything.yml        # env name: cuteanything
```

Every CUT3R-involving command needs the four source trees on `PYTHONPATH`
(the launchers below set this themselves):

```bash
export PYTHONPATH="$PWD:$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:$PYTHONPATH"
```

`src/CUT3R` is REQUIRED in the path — without it the trainer's optional
`eval.monodepth` import fails silently and the absrel/a1 depth metrics vanish.

## 2. External assets (NOT in git)

The configs/scripts reference these absolute paths; provide the files and either
mirror the paths or edit them (section 4):

| asset | expected path |
|---|---|
| pretrained CUT3R ckpt | `/scratch/bdursun25/cuteanything/my-da3/src/CUT3R/src/cut3r_512_dpt_4_64.pth` |
| dataset (single wrist_test episode, DL3DV_Multi format) | `/frozen/avg/bora_data/droid_datasets/training_data/pointworld_droid_wrist_test/dl3dv_multi` |
| checkpoint output root | `/scratch/bdursun25/cuteanything/checkpoints/captain_cut3r_sim3rmse/` |
| DL3DV scan cache (any scratch dir) | `DL3DV_CACHE_DIR=/scratch/bdursun25/cuteanything/.dl3dv_cache` |

Training logs to Weights & Biases — `wandb login` once, or `WANDB_MODE=offline`.

## 3. Launch (SLURM)

From the repo root:

```bash
# any grid arm (a1..a4 x g0..g2); no arg -> captain_gru_v2 (== a2_g0)
sbatch train_captain_gru_overfit.sbatch captain_gru_v2_a4_g1

# falsifier probe against a trained ckpt (keep_freq snapshots only,
# never checkpoint-last/best.pth while the training job is alive)
sbatch probe_gru_hidden.sbatch <ckpt_dir>/checkpoint-40.pth

# re-verify the implementation on the new machine (29 checks, 1 GPU)
sbatch verify_gru_grid.sbatch
```

Don't retrain a2_g0 / a1_g0 if `captain_gru_v2` / `captain_gru_v2_direct` runs
already exist — they are the same experiments under different names.

SLURM headers assume `--account=avg --partition=avg --gres=gpu:lovelace_l40s:2`
(1 GPU for probe/verify) — adjust to the new cluster's account/partition/GPU names.
The launchers source conda via
`source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh` — point this
at the new machine's conda (do NOT rely on `~/.bashrc`; on the original cluster
it is corrupted and kills batch scripts under `set -e`).

## 4. If paths can't be mirrored, edit exactly these

- `src/CUT3R/config/captain_gru_v2*.yaml`: `pretrained:`, `save_dir:`, and the
  `ROOT=` inside the `train_data:` / `test_data:` strings.
- `train_captain_gru_overfit.sbatch`, `probe_gru_hidden.sbatch`,
  `verify_gru_grid.sbatch`: the `WT=` worktree path, `DL3DV_CACHE_DIR`, conda source line.
- `probe_gru_hidden_falsifier.py`, `verify_gru_grid.py`: the `WORKTREE`,
  `DATA_ROOT`, `CKPT`/`DEFAULT_CKPT` constants at the top.

Useful env knobs: `NCCL_P2P_DISABLE=1` (multi-GPU hangs on some nodes),
`DL3DV_CACHE_DISABLE=1` (force a fresh dataset scan),
falsifiers `PREV_PRED_RAY_SHUFFLE=1` / `POSE_GRU_HIDDEN_SHUFFLE=1` / `POSE_GRU_HIDDEN_ZERO=1`.
