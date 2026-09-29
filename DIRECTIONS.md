# Directions ledger: beating the DROID wrist leaders under the closed-loop rules

Opened 2026-09-28 on branch `using_ext_cams`. One entry per direction, so any of them can be picked up cold.
Update the **Status** line of an entry when work starts or ends; never delete an entry, move it to the "Dead" section instead.
Companion documents: `HANDOFF_using_ext_cams.md` (context and paths), `MAKS_IDEA_STATE.md` (what the previous branch killed),
`/gpfs/home/koc/koc821022/vggt_features/VGGT_FEATURES_DESIGN.md` (exterior-camera campaign record).

## The rules every direction must satisfy

- Fully causal and closed loop at inference: only frames `<= t`, everything computed online during the episode.
- No precomputed per-checkpoint correction fitted against ground-truth error statistics (the recalibration bias is out), and
  no hand-set post-processing constants chosen to move a metric (the EMA smoother is out).
- Training on the DROID train split is allowed. Exterior cameras are allowed at inference if used causally and online.
- Every number is scored on all 4292 test scenes with `eval_depth_poses.py` at published defaults and paired against the plain
  run of the same backbone with a 2000-resample scene bootstrap. No subset numbers are quoted as results.

Baselines to beat (finetuned backbone, causal, non-oracle, best per column): AbsRel 0.1734, d<1.25 0.7926, ATE 0.0611,
RPE_trans 0.0066, RPE_rot 1.089. Plain finetuned: 0.1794 / 0.7863 / 0.0759 / 0.0079 / 1.101. Oracle with the previous-frame
GT pose: ATE 0.0151. Zero-shot leaders: ATE 0.0898, RPE_rot 1.351.

---

## D1. Exterior cameras as a live position anchor (main track)

**Status:** proposed, nothing implemented. Design discussion first.

**Idea.** The robot arm and the wrist camera are visible in both static exterior cameras every frame. Detect the wrist camera
in both views, triangulate its position each frame, and give CUT3R that anchor. This is the quantity the oracle rows show is
worth five times the current ATE, obtained online from sensors the rig already has.

**Evidence.** Oracle rows: GT ray map ATE 0.0134, previous-frame GT pose 0.0151, against 0.0611 for the best method. The
trained Arm P rung proved CUT3R's ray channel can fuse a noisy exogenous pose instead of copying it (input-output ATE
correlation 0.98 untrained, 0.88 trained; feed-attributable gains ATE -41%, RPE_trans -14%, RPE_rot -13% on 13 scenes). The
VGGT window feed that produced those gains costs 0.8 s per frame and fails on content-limited scenes; a detector does not look
at wrist content at all.

**Steps.**
1. Label pass (CPU, Slurm): for every train episode and frame, project the kinematic wrist-camera pose from `cam/*.npz` into
   ext1 and ext2 using PointWorld `optimized_extrinsics` (world-to-camera, invert) and the KarlP intrinsics. Labels = the camera
   centre plus a few virtual keypoints fixed in the wrist frame (so PnP can recover orientation). Store pixel targets, depth
   and a visibility flag per frame and camera. Validate on a handful of episodes by overlaying on the exterior frames.
2. Detector: heatmap keypoint regressor on exterior frames (decide backbone and resolution in the design discussion),
   trained on a subsample of the train split, held-out-lab validation. Output keypoints plus confidence.
3. Anchor per frame: two-view triangulation of the centre, PnP per camera for orientation at lower weight, confidence gating,
   no anchor when occluded or out of view. Diagnostic only: anchor error against kinematic GT on the test split.
4. First 4292 row, no CUT3R training: causal complementary filter over the plain finetuned pose stream already on disk,
   CUT3R increments for the high frequencies, the anchor for the low frequencies. Label it as a rig row.
5. Closed-loop row: port the Arm P machinery from the `vggt_features` worktree (`feed_vggt_ray_map`, `--conditioning
   vggt|vggt_prev`, pseudo-scene path, `verify_vggt_arms.py`), build the ray map from the anchor pose, finetune from augfull
   checkpoint-final on all 38,633 episodes with context dropout, evaluate with the detector running online in the eval worker.
   Combine with D2 in the same recipe.

**Expected.** Filter row: ATE toward 0.015 to 0.02, RPE unchanged. Trained row: ATE about 0.02, RPE below plain, depth from D2.

**Cost.** Label pass: CPU hours, decode on the fly from the 641 GB of exterior MP4s already on disk. Detector: order 100 to 200
GPU-hours. Channel finetune: a multi-day 32-GPU run like augfull. Eval: one 4292 pass with the detector in the loop.

**Risks and open questions.** Calibration source at test time (declared PointWorld extrinsics, online t=0 registration with
VGGT-Omega at 2.6 degrees, or self-calibration from the two tracks or the ZED SVO stereo). Gripper occlusion rate. Label noise
from calibration error, a 1.5-degree rotation error at 1.5 m is about 4 cm of reprojection. Detector generalisation to unseen
rigs (test shares labs with train, so a held-out-lab check is the honest test). Rotation from PnP will be worse than CUT3R's
local rotation, so the anchor should carry position with high weight and orientation with low weight.

**Done when.** A rig-group row on the 4292 table beats the finetuned leaders on ATE with RPE no worse than plain, with the
anchor computed online inside the eval worker and the calibration source declared in the footnote.

---

## D2. Unfrozen encoder for depth

**Status:** proposed, not started. Independent of D1, but meant to share its finetune.

**Idea.** The augfull recipe freezes the encoder (`freeze='encoder'` in the finetune configs). Unfreeze it, with a lower encoder
learning rate and layer decay.

**Evidence.** The only unfrozen-encoder runs in the project (VGGT-state finetunes F2/F4) moved AbsRel from 0.179 to 0.156 and
d<1.25 from 0.786 to 0.832, the largest depth movement in any table. Their pose was bad (ATE 0.11), but the frozen-encoder
VGGT-state arms already had ATE 0.125, so the pose damage came from the VGGT state init, not from the unfreezing.

**Steps.** Plain augfull recipe with the encoder unfrozen at 1e-6 (decoder 1e-5), otherwise identical; 4292 eval; check pose
does not degrade. If pose holds, fold into the D1 channel finetune so one model takes depth and pose.

**Expected.** AbsRel about 0.16, d<1.25 about 0.82. Pose: unknown, that is the experiment.

**Risks.** Pose regression from a moving encoder. Fallback of pairing depth from one checkpoint with pose from another is a
two-model system and should not be shipped as a row.

**Done when.** A single causal row holds AbsRel and d<1.25 with pose no worse than plain.

---

## D3. Train away the drift: decaying write rule on episode-length rollouts (wrist-only)

**Status:** proposed, not started.

**Idea.** CUT3R over-predicts its step size progressively through a sequence; the finetune saw 64-frame chunks while test
episodes average about 270 frames. The published training-free rules (TTT3R, ReCal3R) and the recalibration both attack this
length-generalisation symptom from outside. Attack it in training: finetune the decoder and pose head with the ReCal3R or TTT3R
write rule active, on full-episode truncated-BPTT rollouts, under the causal ATE + RPE criterion.

**Evidence.** ReCal3R reaches ATE 0.0611 (-19%) but pays +30% rotation; TTT3R 0.0624 at +3%; our consequence gate 0.0645 at
+2%. All sit on one dose curve. The recalibration paper-level write-up shows the drift is monotone and checkpoint-specific. The
consequence-gate trainer (`src/CUT3R/src/train_wgate_consequence.py`, `dust3r/wgate/traj_loss.py`) already runs 38,633
episodes in 16-frame chunks with state carried across chunks; this direction unfreezes the decoder and pose head under it and
switches the write rule on.

**Steps.** (1) Add the decaying write rule as a training-time option in the model. (2) Rollout length sweep: 64, 128, full
episode. (3) Train from augfull checkpoint-final, 32 GPUs. (4) 4292 eval, paired.

**Expected.** A rotation-neutral row near ATE 0.058. Bounded prize; this cannot reach the anchor rows.

**Risks.** Memory for long rollouts (the DDP no-sync note in memory: long-context runs train at effective batch 4). The rule's
gain may be entirely a test-time artefact that training removes rather than absorbs.

**Done when.** A causal wrist-only row beats ReCal3R on ATE with RPE_rot at or below plain.

---

## D4. Test-time self-supervision on the wrist stream (wrist-only, later)

**Status:** idea only. After D1 to D3.

**Idea.** The one wrist-only lever with unbounded headroom that stays closed loop: at each frame, take a few gradient steps on
a small adapter using photometric or flow consistency between frames t-1 and t under the predicted depth and pose, with the
online WAFT static mask hiding the gripper.

**Evidence.** None local yet. The WAFT mask tooling exists (`waft_sequence.py`, masks under `outputs/cut3r_eval/waft_masks4292`)
and the flow mask was validated as strictly causal. Gripper-in-view and low texture make the self-supervised signal weak; that
is the risk.

**Steps.** Smoke on the 12 smoke scenes with a LoRA-style adapter on the decoder; measure per-frame cost; only then 430 and 4292.

**Done when.** Any 4292 pose gain at a per-frame cost comparable to the VGGT window, or a documented negative result.

---

## D5. Zero-shot backbone: training-free rig feed

**Status:** feed exists for 13 scenes in the `vggt_features` worktree; 4292 not run.

**Idea.** A DROID-trained detector breaks the "no DROID training" label, so the rig row for the pretrained backbone uses the
training-free VGGT-Omega window with the two exterior frames (`stab32_ext` or `stab64_ext` plus the per-t rescale), fused
through the same causal filter as D1 step 4 over the zero-shot pose stream.

**Evidence.** 13 scenes: `stab64_ext + rescale` ATE 0.0271, RPE_rot 1.192 (plain zero-shot on the same scenes is far worse;
the 4292 zero-shot leader is ATE 0.0898). Cost 0.77 to 0.82 s per frame, about 290 GPU-hours per 4292 pass.

**Steps.** Decide whether one 290 GPU-hour pass is worth a zero-shot rig row; if yes, generate the feed for the test split,
filter, score. Zero-shot depth is not worth chasing; nothing training-free has moved it past ReCal3R's 0.435.

**Done when.** A zero-shot rig row on the 4292 table, or a decision not to spend the pass.

---

## Decisions still open (owner: you)

1. Calibration source at test time for D1 and D5: declared PointWorld extrinsics, online t=0 VGGT-Omega registration, or
   self-calibration from tracks or ZED stereo.
2. Rig rows as their own table group, with the calibration source in the footnote.

## Dropped by the rules

- Causal pose recalibration with a calibrated bias, and its stacking on TTT3R / RayMap3R / ReCal3R streams.
- Online estimation of that bias from the emitted stream: nothing predicts it past r = 0.36, per-scene scatter exceeds the mean.
- The causal EMA smoother on the pose increments (alphas 0.25 / 0.40, hardcoded, provenance unrecorded, never scored alone on
  4292): hand-set constants that exist only to move RPE. Smoothing, if wanted, is learned inside the model (D1, D3) or as a
  tiny causal filter trained on the train split; never a hand-tuned post-process.

## Dead. Do not reopen without new evidence

New write-gate heads at either site; Kendall & Gal targets for the write; frame skipping or blocking; gripper removal on the
finetuned model; OpenCV VO as a backbone; PoseGRU; Arm L (t=0 latent as decoder context); hold-last-good gating of an anchored
feed; fixed-K window geometry beyond K=64 for the VGGT feed. Reasons and numbers: `MAKS_IDEA_STATE.md` section 4 and
`VGGT_FEATURES_DESIGN.md`.
