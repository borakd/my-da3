# Handoff: exterior-camera anchoring for CUT3R on the DROID wrist benchmark

Written 2026-09-28 at the cut of branch `using_ext_cams`. Read this first in a new session, then the documents it points to.
Nothing described under "the plan" is implemented yet; the first session on this branch is a design discussion.

## Where you are

- Worktree `/gpfs/home/koc/koc821022/using_ext_cams` (same directory as `/home/koc/koc821022/using_ext_cams`), branch
  `using_ext_cams`, cut from `maks_idea` at `b742f69` plus one path-repointing commit `7744068`. Working tree clean, nothing
  pushed. Do not commit or push unless asked. Run everything from this worktree; read the other worktrees, do not run from them.
- `CLAUDE.md` in the worktree root has the non-negotiables: new checkpoints go under
  `/gpfs/scratch/etur59/koc821022/checkpoints`; W&B records offline on compute nodes and is synced with `sync_nomedia.sh`.
- Cluster gotchas (MareNostrum5): login nodes enforce a hard 300 s CPU cap per process (use `OMP_NUM_THREADS=1` and `timeout`
  for probes, `sbatch` for anything real); compute nodes have no internet; `acc_debug` is 2 h and one job per user, `acc_ehpc`
  runs 3 days in parallel; the numpy python is `/home/koc/koc821022/.conda/envs/cuteanything/bin/python`; pdflatex is
  `/gpfs/apps/MN5/GPP/LATEX/20240430/bin/x86_64-linux/pdflatex`. Bash tool calls cap at 10 minutes, so wait on Slurm jobs with
  a background monitor rather than a foreground sleep.

## The goal

Beat the best rows on the 4292-scene DROID wrist test split for depth and pose, under these rules:

- fully causal and closed loop at inference: only frames `<= t`, everything computed online;
- no precomputed per-checkpoint correction fitted against ground-truth error statistics (the causal pose recalibration's
  calibrated bias is therefore OUT), and no hand-set post-processing constants chosen to move a metric (its EMA smoother is
  out too; smoothing, if wanted, is learned inside the model under the causal ATE + RPE criterion);
- training on the DROID train split is fine;
- the two static exterior cameras MAY be used at inference, as long as that use is causal and online, in the same sense that
  CUT3R, TTT3R, RayMap3R and ReCal3R are.

## Current leaders (4292 scenes, causal, non-oracle)

`$OUT = /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval`.

- Master causal table: `$OUT/master/master_4292_causal_vggt_norecalib.{tex,png}`; write-gate master table:
  `$OUT/wgate/table/learned_dose_4292_master.{tex,png,md}`; CSV of every scored row: `$OUT/master/master_4292.csv`.
- Finetuned backbone (`augfull_lr1e5`, encoder frozen, 64-frame chunks, 50 epochs, lr 1e-5), columns AbsRel / d<1.25 / ATE /
  RPE_trans / RPE_rot: plain 0.1794 / 0.7863 / 0.0759 / 0.0079 / 1.101. Best per column: AbsRel 0.1734 (ReCal3R), d<1.25
  0.7926 (consequence gate lr 1e-4), ATE 0.0611 (ReCal3R, at +30% rotation), RPE_trans 0.0066 (RayMap3R), RPE_rot 1.089
  (conf branch G1L). No row holds more than two columns. Every state-write method, ours and the three published rules, sits on
  one dose curve: ATE bought with rotation. That family is saturated.
- Zero-shot backbone: plain ATE 0.1197, which is the random-walk floor; best ATE 0.0898 (Scal3R relative-pose query on frozen
  zero-shot CUT3R), best rotation 1.351 (OpenCV VO + WAFT mask).
- Oracles: CUT3R fed the current-frame GT ray map reaches ATE 0.0134; fed the previous-frame GT pose, 0.0151. The pose head is
  not the bottleneck; the missing ingredient is an anchor. That is the whole motivation for this branch.

## The plan we agreed to discuss

### Main track: the exterior cameras as a live position anchor

The robot arm and the wrist camera are visible in both static exterior cameras every frame. The previous campaign used those
cameras only weakly, as two extra views at t=0 inside a VGGT-Omega window, and its winning trained feed was wrist-only. The
proposal is to use them as a motion-capture system:

1. **Labels for free.** Project the kinematic wrist-camera pose (store `cam/*.npz`, robot base frame) into each exterior image
   with the PointWorld `optimized_extrinsics` (a 4x4 WORLD-TO-CAMERA matrix, invert it for camera-to-base; 100% coverage of
   train and test) and the exterior intrinsics (KarlP/droid `intrinsics.json`), as the camera centre plus a few virtual keypoints
   fixed in the wrist frame. The kinematic chain and the Euler convention were audited to 0.014 degrees:
   `$OUT/vggt_probe/gt_audit_2026-09-08/GT_CHAIN_AUDIT.md`.
2. **Train a keypoint detector** on the train split. Two views with about a metre of baseline at about 1.5 m give roughly 1 cm
   of position per frame with independent, non-drifting error. Orientation comes from PnP on the virtual points at lower
   weight. Detections carry a confidence; occluded or out-of-view frames yield no anchor.
3. **First 4292 row with no CUT3R training.** A causal complementary filter: CUT3R's increments carry the high frequencies, the
   anchor pulls the low frequencies. Expect ATE toward 0.015 to 0.02 with RPE unchanged.
4. **Closed-loop version.** Feed the anchor through CUT3R's ray-map channel with the Arm P recipe (`feed_vggt_ray_map
   current|prev`, trained from augfull checkpoint-final). That rung proved the network fuses a noisy exogenous pose rather than
   copying it (input-to-output ATE correlation 0.98 untrained, 0.88 trained). Because a detector feed costs milliseconds per
   frame instead of 0.8 s, train on all 38,633 episodes; the 754-episode subset overfit (its control got worse on every metric).
5. **Depth in the same recipe.** Unfreeze the encoder. The only unfrozen-encoder runs in the project moved AbsRel from 0.179 to
   0.156; their pose loss came from the VGGT state init, not from the unfreezing (the frozen-encoder VGGT-state arms already had
   ATE 0.125).

Projected against the finetuned causal leaders (projection, not measurement):

| | ATE | RPE trans | RPE rot | AbsRel |
|---|---|---|---|---|
| current best causal row | 0.0611 | 0.0066 | 1.089 | 0.1734 |
| step 3, filter only | about 0.015 to 0.02 | about 0.008 | 1.101 | 0.1794 |
| step 4 + 5, trained channel | about 0.02 (13-scene analogue 0.033) | below 0.0079 | below 1.10 | about 0.16 |

Why the detector route rather than the VGGT window route: it does not depend on wrist-view content, so it is robust exactly
where the window failed (close-ups, the content-limited session); its feed is affordable for the full train split; it anchors
every frame instead of a t=0 registration that drifts.

### Wrist-only track, in parallel

- Train away the drift instead of calibrating it: finetune the decoder and pose head with the ReCal3R or TTT3R write rule active,
  on full-episode truncated-BPTT rollouts under the causal ATE + RPE criterion. `src/CUT3R/src/train_wgate_consequence.py` and
  `dust3r/wgate/traj_loss.py` have the machinery. Bounded prize: a rotation-neutral row near ATE 0.058.
- Unfrozen encoder on the plain recipe, needed for a wrist-only depth row.
- Test-time self-supervision on photometric consistency between consecutive frames, with the online WAFT static mask hiding the
  gripper. Higher risk; after the two above.

### Zero-shot backbone

A DROID-trained detector breaks the "no DROID training" label. The rig row there is the training-free VGGT-Omega window feed
with exterior frames (`stab32_ext` / `stab64_ext` plus the per-t rescale) fused through the same filter. On 13 scenes that feed
sat at ATE 0.027 to 0.032 against the zero-shot leader's 0.090, at about 290 GPU-hours per 4292 pass. Zero-shot depth is not
worth chasing.

## Data and infrastructure that already exist

- Exterior MP4s for 42,925 of 42,929 episodes, i.e. all 4292 test and all 38,633 train episodes, 1280x720, 641 GB in total:
  `/gpfs/scratch/etur59/koc821022/vggt_cache/raw/<EP>/recordings/MP4/<serial>.mp4`, with `metadata_<EP>.json` beside them
  giving the wrist / ext1 / ext2 serials. MP4 frame index == store frame index (verified pixel-exact). The public bucket
  `gs://gresearch/robotics/droid_raw/1.0.1` also holds ZED SVO stereo recordings per exterior camera (not downloaded; there is
  no `-stereo.mp4`). Login nodes only for any download.
- Store: `/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all` (wrist PNGs, `cam/*.npz` kinematic GT); splits in
  `/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/{train,test}`.
- PointWorld camera sidecars: local copy in the audit directory above; source dataset `nvidia/PointWorld-DROID` on Hugging Face.
  Raw `metadata_*.json` exterior extrinsics are wrong on about 30% of episodes; never read those.
- Checkpoints: `augfull_lr1e5` = `/gpfs/scratch/etur59/koc821022/checkpoints_projects/cut3r_finetune_baselines/`
  `cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth`; zero-shot = `src/CUT3R/src/cut3r_512_dpt_4_64.pth` (symlink into
  the `known_good_fsrc_noproj` worktree, as is `src/CUT3R/src/pretrained_encoders`).
- Eval harness: `eval_pipeline/arm_eval_for_run.sh` scores a label on 4292 scenes with `eval_bundle/bin/eval_depth_poses.py` at
  published defaults (depth median-scaled per frame, poses Sim3-aligned on camera centres, unweighted mean over scenes); paired
  scene-bootstrap CIs via `eval_pipeline/maks_subset_compare.py` (`load_label`, `boot_ci`). `load_label` takes about 75 s per
  arm on GPFS, so cache per-scene metrics.
- Arm P machinery lives in the `vggt_features` worktree (`/gpfs/home/koc/koc821022/vggt_features`, branch `vggt_features`),
  NOT in this tree: the dataset flag `feed_vggt_ray_map` in `src/CUT3R/src/dust3r/datasets/dl3dv.py`, the eval worker
  `--conditioning vggt|vggt_prev` with the pseudo-scene path, `pack_vggt_poses.py`, `build_pseudo_scenes.py`, the
  `vggtray/vggtctx/vggtctl` configs, `verify_vggt_arms.py`, the online exterior-MP4 preprocessing (`vggt_online_preproc`).
  Read `/gpfs/home/koc/koc821022/vggt_features/VGGT_FEATURES_DESIGN.md` first: Arm P results, the calibration audit, measured
  costs, and the five silent-failure modes (each one produced something that looked like a result because a step's exit status
  was checked and its output was not). Porting that machinery here is part of the design discussion.
- Branch documents to read: `MAKS_IDEA_STATE.md` (what is dead), `CAUSAL_POSE_RECALIBRATION.md` (the progressive step-inflation
  mechanism and why the bias is out), `WRITE_GATE_VARIANTS.md`, `UNCGATE_CAMPAIGN.md`.

## Dead. Do not reopen without new evidence

New write-gate heads at either site; Kendall & Gal targets for the write; frame skipping or blocking; gripper removal on the
finetuned model; OpenCV VO as a backbone; PoseGRU; Arm L (t=0 latent as decoder context); hold-last-good gating of an anchored
feed (chained dead reckoning loses more than the outliers cost); online estimation of the step-inflation bias (nothing in the
emitted stream predicts it past r = 0.36).

## Open decisions to settle before any code

1. **Calibration source at test time.** Declare PointWorld static extrinsics as sensor calibration, or register the exterior
   cameras online at t=0 with VGGT-Omega (2.6 degrees against PointWorld, no failures), or self-calibrate from the two 2D tracks
   or the ZED stereo.
2. **Detector design.** Keypoint set (camera centre plus how many virtual points), backbone, input resolution, occlusion
   handling, label noise from the calibration, held-out-lab validation, and how to measure anchor accuracy against kinematic GT
   on the test split as a diagnostic.
3. **Fusion.** The filter for step 3, and how the anchor enters the ray channel for step 4 (position from triangulation, rotation
   from CUT3R with a slow correction?), plus context dropout so one checkpoint is honest with and without anchors.
4. **Reporting.** Rig rows get their own table group, with the calibration source in the footnote.
5. **Budget.** Label pass (CPU, Slurm), detector training, the full-split ray-channel finetune at 32 GPUs, and a 4292 eval with
   the detector running online inside the eval worker.

## What to do first

Read the documents above and the two master tables, then give a view on the open decisions and a concrete first-week plan.
Do not implement anything yet. Answer with text only unless a file read is needed.
