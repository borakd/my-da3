# OpenCV visual-odometry control arm: design decisions (2026-09-02)

Purpose: a bare-bones classical pose baseline for the 4292-scene DROID wrist
harness, with as few free parameters as possible, so it can serve as a control
next to the CUT3R arms (champion: augfull_cg_fuse_g7, see CONF_GATE_CAMPAIGN.md).

## Dataset facts the decisions rest on (verified 2026-09-02)

- Scenes: `$SCENES_ROOT/<scene>/dense/{rgb,cam,depth,outlier_mask,sky_mask}`
  (SCENES_ROOT from eval_pipeline/mn5_paths.sh).
- Images 320x180 PNG. GT intrinsics per frame in `cam/*.npz` (key `intrinsic`,
  fx ~ 192 px, no distortion). GT c2w pose in the same npz (key `pose`).
- GT depth `depth/*.npy` float32, 0 = invalid, ~88% valid, 0.3-1.4 m typical.
- Sequence length 57 to ~1700 frames. Per-step motion median 9 mm / 1.3 deg.
- `outlier_mask` is mostly the invalid-depth region. Its constant part is a
  rectification band on the left edge plus a gripper blob bottom-left
  (roughly the left ~110 columns). The gripper is IMAGE-STATIC: its corners
  vote for zero motion in PnP, so it must be masked, not "filtered by RANSAC".
- Scoring: Sim(3)-aligned ATE / RPE (eval_pipeline/pose_sim3_both.py). A global
  scale is free; scale DRIFT is not. eval_depth_poses.py numbers frames by
  position, so every frame must get a pose.

## Decision list, in the order they must be made

Status: PENDING until the walkthrough fixes each one. "Pick" = recommendation.

### Tier 0: experimental frame (constrain everything below)

| # | Decision | Options | Pick | Why | Status |
|---|---|---|---|---|---|
| 0.1 | Depth source / arms | GT depth (oracle); model depth (e.g. champion preds); triangulated monocular map | **DECIDED: triangulated monocular map.** User rule: closed loop, plain image inputs only, no privileged info; scale treated as a separate issue, pose accuracy is the target. (Note: model depth is NOT privileged, it is inference output; option kept for a later hybrid arm.) | Self-contained classical control that owes nothing to the model. | DECIDED 2026-09-02 |
| 0.1b | Intrinsics source | calibrated K from cam/*.npz; one nominal dataset K; model-estimated focal from preds; self-calibration | **DECIDED: model-estimated focal** (augfull_lr1e5 preds camera/*.npz), per-scene MEDIAN of per-frame values, pp = image center. Run the classical arm on the model's own preprocessed 320x192 frames (same loader) so K and pixels agree. Facts: model focal ~211 px @320x192 vs calibrated ~190 @320x180 (ratio 1.11, tight); within-scene jitter 0.7%. Calibrated-K copy kept as a one-off diagnostic. | Closed-loop rule (model output is not privileged); classical and model arms then share identical K. | **REVISED 2026-09-06: inference-time per-frame focal.** Frame t uses the focal CUT3R produced for frame t (`--focal perframe:<preds_root>`); every geometric step carries a per-frame K (bootstrap E in per-view normalised coordinates, KF-to-KF triangulation with each keyframe's K, PnP with frame t's K). The per-scene median is retroactive and is no longer the reported configuration. DECIDED 2026-09-02; **re-checked 2026-09-05 in the full pipeline: focal source is immaterial** (finetuned-model focal 111.9/6.93/0.871, fixed nominal 203 px 109.1/7.04/0.869, calibrated 116.0/6.91/0.885). The zero-shot checkpoint's focal is NOT usable (229-570 px, 1.5x median, scene-inconsistent). `--focal 203` gives a strictly model-free arm at no cost. |
| 0.2 | Reference frame | frame-to-frame; keyframe-to-frame; local map (PnP against persistent map) | local map | Forced by 0.1: every frame registers to map points via PnP; keyframes are where the map grows. | DECIDED (forced) |
| 0.3 | Eval scope | 12-scene smoke; 430 subset; full 4292 | **DECIDED: smoke on eval_pipeline/cg_smoke_scenes_12.txt** (all 12 are inside the 430 subset; augfull_lr1e5 and augfull_cg_fuse_g7 both have preds+eval on all 12, so the finetuned checkpoint and the OpenCV arm are compared on identical scenes). Then 430, then 4292 once frozen. | User: smoke must overlap an existing smoke set so model and classical arms share scenes. | DECIDED 2026-09-02 |
| 0.4 | Failure reporting | ATE/RPE only; + failed-frame count; + failed-frame count and median inliers | **DECIDED: harness scores ATE / RPE-trans / RPE-rot only** (AbsRel, d1 are INHERITED from augfull_lr1e5 depth via symlink; mark as inherited in tables). **Plus diagnostics** in a side CSV (not the harness CSV): per-frame inlier count + failed flag, aggregated to per-scene failed-frame count and median inliers. | Separates lost-tracking from drift; median inliers moves before ATE does. | DECIDED 2026-09-04 |

### Tier 1: frontend, in pipeline order

| # | Decision | Options | Pick | Why | Status |
|---|---|---|---|---|---|
| 1.1 | Preprocessing | gray only; 2x upsample; CLAHE; undistort | **DECIDED: grayscale conversion only**, on the eval loader's 320x192 frames (load_images_cover: uniform 1.067x scale + center crop). All pixel thresholds are in these units. | Gray is mandatory for the detector/LK; everything else is a v1 single-variable experiment (CLAHE first if median inliers are low). | DECIDED 2026-09-04 |
| 1.2 | Masking | none; fixed dataset polygon; per-scene temporal-variance mask (Otsu); flow-based rejection; GT outlier mask | **DECIDED: NO MASK for v0** (user). Log per-frame zero-motion track fraction (disp<0.3 px while median>2 px) as the gripper diagnostic. Probe on RAIL+80edfcb1+2023-07-14-14h-28m-45s: gripper attracts few corners (median 12% zero-motion tracks on moving frames, max 42%); GT outlier_mask does NOT cover the gripper; Otsu temporal-variance mask OVER-masks (58%) because uniform regions are low-variance regardless of motion. Fixed polygon fails: finger geometry differs per lab. **Configurable lever `reject_static_tracks` (DEFAULT OFF):** when ON, and only on frames where the median track displacement >= 2 px (parallax present, so a non-moving track is provably not scene geometry), drop tracks with displacement < 1 px before any geometry (E / PnP / triangulation). Never active on low-parallax frames, so a genuinely paused camera is unaffected. Benchmark: cuts parallax-gated E-rotation error 2.14 -> 1.70 deg (LK). Static share of LK tracks across the 12 smoke scenes: 16-53%, median ~25%. | Data-driven: gripper is dark/low-texture here; masking needs a better design than any of the cheap options. | DECIDED 2026-09-04; lever added 2026-09-04; **sweeps 1-3: lever ON is better on every metric** (v0 final: 111.9/6.93/0.871 ON vs 115.6/8.40/0.971 OFF). **Default flipped to ON on 2026-09-05 (user decision); `--no_lever` disables.** |
| 1.3 | Detector | Shi-Tomasi; FAST; ORB; SIFT/AKAZE; fixed grid | **DECIDED: Shi-Tomasi goodFeaturesToTrack(maxCorners=1000, qualityLevel=0.01, minDistance=5)** (probe values; cap never reached, ~140-300 corners/frame). | Designed for LK; relative quality threshold adapts across exposure; no wasted descriptor. | DECIDED 2026-09-04 |
| 1.4 | Spatial spread | none; grid bucketing; top-up in empty regions | **DECIDED: none.** minDistance=5 already spreads corners at this resolution (probe). Bucketing = first v1 frontend experiment if long-scene drift dominates. | Zero parameters; no probe evidence of a coverage problem. | DECIDED 2026-09-04 |
| 1.5 | Correspondence | sparse LK (+/- fwd-bwd check); ORB/SIFT + ratio; DIS/Farneback dense flow at corners or on a grid | **DECIDED: sparse LK at OpenCV defaults (21 px window, 3 levels) + forward-backward check at 1 px.** DIS-grid recorded as measured runner-up = first v1 frontend swap. | Benchmark (section below): ORB/SIFT/Farneback rejected; DIS-grid best E-rotation via 5x points; LK+FB best per-point precision at keyframe-scale gaps and gives corner-anchored persistent tracks the map needs. | DECIDED 2026-09-04 |
| 1.6 | Track lifetime | re-detect every frame; persistent tracks + top-up on count floor; persistent tracks + top-up only at keyframes | **DECIDED: persistent tracks, top-up only at keyframes** (new corners detected with existing track positions excluded via mask). No floor parameter: the keyframe trigger (2.12) governs refill. | Every map point is born at a keyframe and triangulated at the next; 95% per-frame survival means the count decays slowly between keyframes. | DECIDED 2026-09-04 |

### Tier 2: solver

| # | Decision | Options | Pick | Why | Status |
|---|---|---|---|---|---|
| 2.1 | Geometry | 2D-2D essential; 2D-3D PnP; 3D-3D Procrustes | 2D-3D PnP against the map (E only inside the bootstrap, 2.10) | Forced by 0.1. | DECIDED (forced) |
| 2.2 | Direction | (depth-anchored frame-pair question) | dropped | Meaningless with a map: every frame registers to map points. | DROPPED by 0.1 |
| 2.3 | Depth read | (n/a with monocular map: 3D comes from triangulation) | dropped | | DROPPED by 0.1 |
| 2.4 | PnP variant | ITERATIVE; EPnP; P3P/AP3P; SQPnP; each +/- solvePnPRefineLM | **DECIDED: solvePnPRansac(ITERATIVE) + solvePnPRefineLM on inliers** (standard v0; LM refine is a no-op after ITERATIVE-RANSAC but kept so the refine step exists for the P3P/SQPnP swaps). | Benchmark (section below): all 10 variants within 0.05 deg / 2 mm of each other; differences are noise. SQPnP nominally best by 0.04 deg; P3P/AP3P need +LM to match on translation. | DECIDED 2026-09-04 |
| 2.5 | Initial guess | none; identity; previous motion | **DECIDED: none** (useExtrinsicGuess=False). NO automatic fallback solver in v0: a failed PnP is logged as a failed frame (3.1/3.2), not retried with SQPnP. SQPnP is a manual single-variable swap only if the diagnostics show solver failures. | Benchmark: zero failures over 2118 pairs with no guess; a guess or a fallback would hide instability the diagnostics must expose. | DECIDED 2026-09-04 |
| 2.6 | Robust estimator | RANSAC; USAC/MAGSAC++; none | **DECIDED: plain RANSAC** (solvePnPRansac default). | Handled identity-voting outliers with a 0.04 deg penalty in the PnP benchmark; three stateable numbers. | DECIDED 2026-09-04 |
| 2.7 | RANSAC threshold | OpenCV default 8 px; 1-3 px | **DECIDED: 2 px.** | 8 px = 2.2 deg at f~210, admits gripper tracks; 1 px is at the LK noise floor; ~65% of consecutive-frame tracks fall within 2 px of GT (corr benchmark); same unit as the benchmarks' "correct" definition. | DECIDED 2026-09-04 |
| 2.8 | Confidence / iterations | 0.99/100 (default); 0.999/1000 | **DECIDED: 0.999 / 1000.** | 1-2 ms per frame; removes the iteration cap as a variable on high-outlier frames. | DECIDED 2026-09-04 |

### Tier 2b: map (added 2026-09-02 after 0.1 chose the monocular map)

| # | Decision | Options | Pick | Why | Status |
|---|---|---|---|---|---|
| 2.9 | Bootstrap pair | frames 0/1; first frame with median track displacement > P px; H-vs-E model selection (ORB-SLAM); tracked-ratio trigger | **DECIDED: frame 0 vs first frame b whose median LK displacement > 5 px (was 10; revised 2026-09-05 with 2.12)**, guarded by recoverPose cheirality count >= 20 (same floor as 3.1; rejects pure-rotation pairs, then keep tracking and retry at the next frame). **Frames 1..b-1 are filled retroactively by PnP against the bootstrap map** so every timestep gets an OpenCV pose. Scene where no pair ever passes = total failure, flagged. | Probe: RAIL scene stationary for frames 0-15; 9 mm steps make frame 0/1 degenerate. 10 px ~ 3 typical frames ~ 25 mm baseline at 0.5 m. H-vs-E selection = v1 if bootstrap fails on planar scenes. | DECIDED 2026-09-04 |
| 2.10 | Bootstrap geometry | findEssentialMat + recoverPose; homography; H-vs-E model selection; RANSAC threshold 2 / 1 / 0.5 px | **DECIDED (revised 2026-09-05): essential matrix only, RANSAC threshold 0.5 px** (0.999 conf) + recoverPose; cheirality guard from 2.9. Unit translation sets map scale. | E-diag: 2 -> 0.5 px cuts two-view rot error 2.06 -> 1.23 deg and t-dir 31.6 -> 18.1 deg at gap 4; map-level PnP-at-+4 improves 4.19 -> 3.02 deg (gap 4), 3.58 -> 2.62 (gap 2). Only the sub-pixel minority of chained LK tracks is geometrically clean. H-vs-E = v1 if bootstrap rejections cluster on planar scenes. | DECIDED 2026-09-05 |
| 2.11 | Triangulation acceptance | none; cheirality only; + reprojection < 2 px; + parallax >= 1 deg; all three | **DECIDED: cheirality in both cameras + reprojection error < 2 px in both keyframes.** No new parameter (2 px = PnP threshold). Parallax filter NOT in v0: at keyframe-scale baselines (~36 mm) a 1 deg floor discards half the points and raises PnP failures 2x. | Benchmark (section below): with GT pose, cheir+rep is best on every metric (PnP rot 0.56 vs 0.63 cheir-only vs 0.73 none; bad-depth 22% vs 32% vs 46%). With the E pose all filters tie (~4 deg) because the two-view pose dominates map error. | DECIDED 2026-09-05 |
| 2.12 | Keyframe trigger | fixed stride (2/4/8); median parallax since last KF >= P px (5/10/20); tracked-inlier ratio < r (0.9/0.8/0.7) | **DECIDED: median track displacement since last keyframe >= 5 px** (shares P with 2.9; 2.9's P drops 10 -> 5). | Sweep (section below): tighter = better for every family; stride 2 is best on gated pairs (2.62 deg) but 25% of 2-frame windows have < 2 px motion, where a stride keyframe would triangulate at ~zero baseline and pass cheir+rep. Parallax 5 gives median gap 2 (2.96 deg) and never fires on a paused camera. | DECIDED 2026-09-05 |
| 2.13 | New map points | triangulate KF-to-KF only; every frame | **DECIDED: KF-to-KF only.** | Sweep 1: every-frame 126.2/10.40/1.402 vs KF-only 122.9/9.65/1.129 (ATE mm / RPE-t mm / RPE-r deg). | DECIDED 2026-09-05 |
| 2.14 | Map culling | none; drop points unseen for N KFs; drop points after k PnP-outlier hits | **DECIDED: drop a map point after 3 consecutive PnP-outlier hits** (dead tracks are dropped implicitly: no re-association without descriptors). | Sweep 1: PnP failures 26% -> 5.6% of frames, RPE-t 9.65 -> 8.18, 2x faster; ATE unchanged. | DECIDED 2026-09-05 |
| 2.15 | Scale propagation | implicit via map; explicit hand-off across re-bootstraps | **DECIDED: implicit within a segment + scale hand-off at re-bootstrap** (keep tracks alive, match new segment's scale to the old points' depth through >= 10 shared tracks, 0.2 < s < 5). | Sweep 3: ATE 116.8 -> 111.9 mean, 117 -> 98 median; fired on 12 of 46 reboots. | DECIDED 2026-09-05 |

### Tier 3: trajectory

| # | Decision | Options | Pick | Why | Status |
|---|---|---|---|---|---|
| 3.1 | Failure floor | inlier count 10 / 20 / 30; ratio | **DECIDED: inlier count 30.** | Sweeps 1-2: 10 -> RPE-r 1.80 (bad PnP accepted); 30 vs 20: 0.861 vs 0.923 RPE-r, 7.07 vs 7.34 RPE-t, ATE 116.8 vs 120.0. | DECIDED 2026-09-05 |
| 3.2 | Failure policy | hold last pose; constant velocity; drop frame | **DECIDED: hold last pose + flag.** Plus recovery rule (3.2b): re-bootstrap after 3 consecutive PnP failures (not only at map starvation). | Sweep 1: const-velocity 121.9/8.47/1.375 vs hold 122.9/9.65/1.129 (mixed; rotation worse). Sweep 2: rb3 cuts failed frames 18 -> 14% and RPE-r 1.11 -> 1.01 vs waiting for starvation; rb10 no better. | DECIDED 2026-09-05 |
| 3.3 | Scale | as given by the map; per-frame normalisation | **DECIDED: as given** (Sim(3) scoring absorbs one global scale; segment scale continuity handled by 2.15). | No alternative worth a run. | DECIDED 2026-09-05 |
| 3.4 | Refinement | none; sliding-window BA (scipy); pose graph | **DECIDED: none for v0.** | Sweep 1: local BA window 5 / 10: ATE 128.8 / 130.7 vs 122.9, RPE-r 1.07 / 1.18 vs 1.13, 5-9x runtime. | DECIDED 2026-09-05 |

## Probe evidence (2026-09-04, scene RAIL+80edfcb1+2023-07-14-14h-28m-45s, 128 frames, 320x192 cover frames, model focal 212)

- goodFeaturesToTrack(maxCorners=1000, quality=0.01, minDist=5): only ~215 corners/frame (min 139). Low-texture scene.
- Consecutive-frame LK with forward-backward check (<1 px): median survival 0.95. Over a 15-frame gap in fast segments survival drops to ~15%: tracking must be frame-to-frame.
- Frames 0-15: camera stationary (GT rot 0.03 deg, |t|=0.2 mm). Essential matrix degenerate there: bootstrap MUST be parallax-gated (2.9).
- Pair 0-15 (1.6 deg, 6 mm): E rotation error 0.3 deg, translation-direction error 40 deg. Translation direction is poorly observable at this baseline; rotation dominates the flow.
- Fast pairs (60-75, 64-79; 25 px median disp): E rotation error 4-5 deg, t-direction 40-80 deg, inliers ~0.5.

## Correspondence benchmark (2026-09-04, decision 1.5; 12 smoke scenes, 4243 frames; Slurm job 45403560)

Script + per-scene JSON: /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/corr_bench/.
Scoring uses GT depth+pose ONLY as reference (native 320x180, calibrated K). Metrics per pair:
ncorr = correspondences; ncorrect = within 2 px of GT flow; rotErr = essential-matrix rotation
error (deg) on pairs with GT rot > 0.5 deg. Medians pooled over ~2100 pairs per gap.

| method | gap1 ncorr/ncorrect | gap1 rotErr | gap4 ncorr/ncorrect | gap4 rotErr (p90) | ms |
|---|---|---|---|---|---|
| LK + fwd-bwd check | 166 / 88 | 0.88 | 103 / 27 | 2.55 (8.4) | 4 |
| LK no check | 183 / 91 | 0.85 | 167 / 31 | 2.46 (8.1) | 2 |
| ORB + ratio 0.75 | 138 / 77 | 1.11 | 57 / 12 | 3.53 (12.5) | 3 |
| SIFT + ratio 0.75 | 94 / 48 | 0.95 | 54 / 12 | 2.54 (8.7) | 19 |
| DIS flow at corners | 187 / 95 | 0.82 | 187 / 38 | 2.27 (8.3) | 6 |
| DIS flow on 8-px grid | 880 / 478 | 0.75 | 880 / 192 | 1.88 (6.9) | 5 |
| Farneback at corners | 187 / 92 | 0.87 | 187 / 12 | 3.66 (10.3) | 10 |

Paired gap-4 rotErr: DIS-grid better on 848 pairs, LK better on 523.
Static (gripper) share of LK tracks: 16-53% per scene, median ~25%.
Parallax-gated gap-4 pairs (median flow >= 2 px, n=814), rotErr median / p90:
LK all 2.14 / 7.9; LK minus zero-motion tracks 1.70 / 7.0; DIS-grid all 1.55 / 6.0; DIS-grid minus zero-motion 1.15 / 4.3.
Verdict: descriptor matching rejected by data; Farneback collapses at gap 4; DIS-grid wins rotation via point count;
LK+FB has the best per-correspondence precision at gap 4 (in2 0.35 vs 0.29). Removing static tracks helps as much as changing method.

## PnP-variant benchmark (2026-09-04, decision 2.4; 12 smoke scenes; Slurm job 45407881)

Script + JSON: /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/pnp_bench/. Same LK+FB frontend;
3D points from GT depth on the source frame (oracle 3D); scored vs GT relative pose. "outliers" condition adds
static no-depth tracks with plausible wrong 3D points (identity-voting outliers). RANSAC 2 px / 0.999 / 1000.

| variant | gap1 rot (p90) | gap1 trans mm | gap4 rot (p90) | gap4 trans mm | ms |
|---|---|---|---|---|---|
| ITERATIVE (+LM identical) | 0.44 (1.72) | 3.9 | 1.10 (6.86) | 8.3 | 1.0 / 2.3 |
| EPnP / EPnP+LM | 0.41 / 0.44 | 4.0 / 3.9 | 1.18 / 1.10 | 10.1 / 8.3 | 0.8 / 2.2 |
| P3P, AP3P / +LM | 0.40 / 0.41 | 3.9 / 3.6 | 1.14 / 1.08 | 10.2 / 8.7 | 0.3 / 0.7 |
| SQPnP / +LM | 0.37 / 0.44 | 3.3 / 3.9 | 1.06 / 1.10 | 8.4 / 8.3 | 0.7 / 2.2 |

Identity-voting outliers cost only 1.10 -> 1.14 deg at gap 4 (RANSAC copes when the map points are right).
Zero failures on 2118 pairs. Paired diffs vs ITERATIVE+LM: median |diff| < 0.01 deg for every variant.
NOTE the absolute floor: even with ORACLE 3D points, per-frame PnP rotation error is 0.44 deg median
(GT median step 1.3 deg) and translation error 3.9 mm (GT median step 9 mm). That floor comes from LK noise at
320x180 plus GT depth/pose inconsistency, not from the solver; the map arm cannot beat it.
Calibration vs the model on the SAME 12 scenes (per-frame medians from eval_depth_pose_metrics.csv): augfull_lr1e5
RPE-rot 0.88 deg / RPE-trans 5.7 mm / ATE 90 mm; champion augfull_cg_fuse_g7 0.82 deg / 5.5 mm / 81 mm.
Oracle-3D PnP per-frame (0.44 deg / 3.9 mm) is ~2x better than the model on relative pose; the map arm sits between.

## Triangulation-acceptance benchmark (2026-09-05, decision 2.11; 12 smoke scenes; Slurm job 45443315)

Script + JSON: /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/tri_bench/. Triplets (i, i+4, i+8) with
persistent LK tracks; parallax-gated (median flow >= 2 px): 1373 of 2077 triplets. Triangulate i<->j with either the
E-matrix pose (pipeline reality) or the GT pose (isolates the filter); apply filter; score admitted map by depth error
vs GT (>20% = bad) and by PnP of frame k against it (rotation error deg; fail = <20 inliers). Static tracks kept (lever OFF).

| filter | E: n_acc | E: bad | E: PnP rot (p90) | E: fails | GT: n_acc | GT: bad | GT: PnP rot (p90) | GT: fails |
|---|---|---|---|---|---|---|---|---|
| none | 87 | 0.64 | 4.18 (13.5) | 96 | 87 | 0.46 | 0.73 (2.77) | 117 |
| cheirality | 72 | 0.54 | 3.96 (13.6) | 159 | 72 | 0.32 | 0.63 (2.31) | 170 |
| cheir + reproj<2px | 67 | 0.53 | 3.98 (14.0) | 163 | 62 | 0.22 | 0.56 (1.83) | 183 |
| cheir + parallax>=1deg | 47 | 0.42 | 4.64 (15.1) | 329 | 43 | 0.29 | 0.87 (3.21) | 398 |
| all three | 43 | 0.40 | 4.60 (15.1) | 331 | 34 | 0.15 | 0.74 (2.33) | 377 |

KEY FINDING (bigger than 2.11): with the E-matrix pose the admitted map is ~53% bad and PnP against it lands at ~4 deg
rotation error at a 4-frame gap, vs 0.56 deg with the GT pose on the SAME tracks and filter, vs 1.10 deg for oracle-3D
PnP, vs 0.88 deg model RPE-rot. The two-view bootstrap pose (translation direction ~40 deg off at 36 mm baselines, see
probe evidence) is the bottleneck of the map arm, not the acceptance filter. Baseline sweep launched to quantify (2.12).

## Two-view (essential matrix) diagnostics and model calibration (2026-09-05; Slurm 45443683 / 45443704)

Scripts + JSON: /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/e_diag/. Chained LK tracks, lever ON,
parallax-gated, ~1780 pairs per gap over the 12 smoke scenes. Rotation error / translation-direction error vs GT (deg, medians).

| estimate | gap 4 rot / tdir | gap 8 rot / tdir | gap 16 rot / tdir |
|---|---|---|---|
| E RANSAC 2 px (pipeline) | 2.06 / 31.6 | 3.80 / 31.0 | 8.22 / 41.0 |
| E RANSAC 1 px | 1.64 / 24.3 | 3.05 / 25.2 | 7.32 / 35.9 |
| E RANSAC 0.5 px | 1.23 / 18.1 | 2.88 / 23.5 | 8.11 / 36.8 |
| E MAGSAC 2 px | 1.34 / 22.9 | 2.87 / 24.2 | 7.57 / 37.0 |
| E from GT flow + 1 px noise (same points) | 1.43 / 13.8 | 1.67 / 7.4 | 1.95 / 4.2 |
| E from GT flow, no noise (convention check) | 0.01 / 0.0 | 0.01 / 0.0 | 0.00 / 0.0 |
| 2-view Sampson refinement of E(2px) | 1.70 / 29.0 | 3.38 / 30.0 | 7.80 / 39.8 |
| relative static lever (drop < 20% of median disp) | 1.96 / 32.5 | 3.73 / 32.8 | 7.75 / 39.1 |
| **finetuned CUT3R (augfull_lr1e5) relative pose** | **2.22 / 31.7** | **3.76 / 29.0** | **6.25 / 25.8** |
| **champion (augfull_cg_fuse_g7) relative pose** | **1.99 / 30.5** | **3.33 / 27.1** | **5.37 / 23.5** |

(Model rows: all pairs, no gating; gap 1: 0.88 / 40.2 (finetuned), 0.82 / 39.4 (champion).)

Findings:
- Conventions verified (synth0 = 0). GT poses are consistent with the images to ~1-2 px/frame on gripper-free tracks
  (photometric/flow check on 3 scenes); a ~20% depth-vs-pose scale mismatch exists in some scenes (irrelevant to E).
- The ~30 deg translation-direction error is NOT specific to classical VO: the model has the same 30-40 deg error vs GT.
  Rotation: E at 0.5 px BEATS the model at gaps 4 and 8 (1.23 vs 2.22; 2.88 vs 3.76) and loses at gap 16.
- Static-track removal (absolute 1 px or relative 20%) does not change E. Tighter RANSAC threshold does (2 -> 0.5 px:
  rot 2.06 -> 1.23, tdir 31.6 -> 18.1 at gap 4): only a sub-pixel minority of chained LK tracks is geometrically clean.
- Geometry is well conditioned (synthetic 1 px noise: tdir 14 -> 4 deg with baseline); real tracks get WORSE with gap
  because chained LK drift grows with chain length. Keyframes must be close (gap 2-4), not far.
- GT depth is defined ON the gripper in many scenes, so "valid GT depth" is not a gripper oracle; the corr-benchmark
  in2/EPE numbers on high-gripper scenes are contaminated by that.

Third diagnostic (Slurm 45443759): non-chained keyframe-pair correspondences and noise calibration (rot / tdir, deg):

| estimate | gap 4 | gap 8 | gap 16 |
|---|---|---|---|
| E(2px) from chained LK (pipeline) | 2.06 / 31.6 | 3.80 / 31.0 | 8.22 / 41.0 |
| E(2px) from single-hop LK | 2.33 / 38.3 | 4.85 / 49.1 | 9.20 / 60.4 (934 pairs) |
| E(2px) from SIFT + ratio matches | 2.16 / 38.9 (1601) | 3.48 / 38.5 (1329) | 4.94 / 38.4 (764 pairs) |
| E(2px) from DIS dense flow, 8-px grid | 1.70 / 26.4 | 4.15 / 40.4 | 12.05 / 66.1 |
| E from GT flow + 3 px noise | 2.20 / 29.0 | 2.82 / 14.7 | 3.55 / 9.1 |

Reading: at gap 4 real chained-LK tracks behave like ~3 px iid noise for two-view geometry; beyond gap 8 they are worse
than 3 px noise (chain drift). No alternative correspondence source fixes it (SIFT only helps at gap 16, and finds
enough matches on half the pairs). The gtR_t rows in the JSON are INVALID (sign-ambiguous linear solver); ignore them.

Map quality vs bootstrap-E RANSAC threshold (Slurm 45443823; PnP test at keyframe+4, cheir+rep, lever OFF):

| E thr | gap 2: bad / PnP rot | gap 4: bad / PnP rot | gap 8: bad / PnP rot |
|---|---|---|---|
| 2.0 px | 0.61 / 3.58 | 0.55 / 4.19 | 0.52 / 5.83 |
| 1.0 px | 0.56 / 3.29 | 0.52 / 3.66 | 0.49 / 4.84 |
| 0.5 px | 0.54 / 2.62 | 0.51 / 3.02 | 0.48 / 4.49 |

For comparison at the same horizon (keyframe+4 => 6-12 frames from the first keyframe): model relative-rotation error is
2.2 deg (gap 4) to 3.8 deg (gap 8). The bootstrap map with E at 0.5 px and gap 2 (2.62 deg) is at rough parity with the
model over that horizon; the GT-pose map (0.59) and oracle-3D PnP (1.10) show the remaining headroom is in the map's 3D.

Keyframe-trigger sweep (Slurm 45443937; E 0.5 px, PnP test at keyframe+4, cheir+rep, lever OFF; pooled medians):

| trigger | fired/starts | KF gap median (p10-p90) | n_acc | bad | PnP rot (p90) | fail% |
|---|---|---|---|---|---|---|
| stride 2 | 1555/2089 (gated) | 2 | 97 | 0.54 | 2.62 (9.9) | 5.9 |
| stride 4 | 1677/2077 | 4 | 88 | 0.51 | 3.02 (12.1) | 7.2 |
| stride 8 | 1651/2053 | 8 | 65 | 0.48 | 4.49 (19.7) | 13.2 |
| parallax 5 px | 1828/1942 | 2 (1-8) | 91 | 0.56 | 2.96 (10.6) | 11.9 |
| parallax 10 px | 1783/1858 | 4 (2-13) | 83 | 0.50 | 3.23 (11.9) | 11.9 |
| parallax 20 px | 1623/1695 | 7 (3-20) | 66 | 0.44 | 4.25 (16.0) | 14.8 |
| ratio 0.9 | 1667/2066 | 2 (1-8) | 93 | 0.52 | 2.91 (10.4) | 7.4 |
| ratio 0.8 | 1713/2047 | 5 (1-14) | 78 | 0.48 | 3.68 (14.4) | 9.0 |
| ratio 0.7 | 1695/1996 | 7 (2-19) | 65 | 0.47 | 4.78 (19.1) | 12.0 |

Every family improves monotonically as keyframes get closer; all adaptive triggers at their tightest setting converge to a
median gap of 2. Stride rows are conditioned on the 2 px motion gate (25% of 2-frame windows excluded); parallax/ratio
rows fire on their own rule, so their populations differ slightly.

## v0 pipeline sweep 1 (2026-09-05; eval_pipeline/opencv_vo.py; 12 smoke scenes; Slurm 45453490; pose_sim3_both.py)

Every decided choice hard-coded; each open decision as a switch. Means over 12 scenes (ATE mm / RPE-trans mm / RPE-rot deg),
plus diagnostics: failed frames %, re-bootstraps, keyframes per 100 frames, median PnP inliers, seconds per scene.

| variant | ATE | RPE-t | RPE-r | failed% | reboots | KF/100f | med inl | s/scene |
|---|---|---|---|---|---|---|---|---|
| vo_v0 (defaults) | 122.9 | 9.65 | 1.129 | 37.1 | 23 | 13.0 | 63 | 23 |
| 2.13 new points every frame | 126.2 | 10.40 | 1.402 | 33.7 | 21 | 11.5 | 59 | 23 |
| 2.14 cull outlier points (3 hits) | 123.4 | 8.18 | 1.113 | 18.1 | 30 | 13.4 | 85 | 13 |
| 3.1 min inliers 10 | 123.5 | 10.24 | 1.795 | 9.5 | 11 | 21.1 | 52 | 23 |
| 3.1 min inliers 30 | 123.3 | 9.09 | 1.063 | 42.7 | 23 | 10.8 | 91 | 22 |
| 3.2 constant-velocity on failure | 121.9 | 8.47 | 1.375 | 31.9 | 21 | 13.4 | 62 | 22 |
| 3.4 local BA, window 5 | 128.8 | 8.49 | 1.068 | 32.5 | 20 | 10.9 | 78 | 104 |
| 3.4 local BA, window 10 | 130.7 | 9.04 | 1.175 | 27.7 | 20 | 13.0 | 81 | 189 |
| 1.2 static-track lever ON | 116.4 | 7.93 | 1.009 | 26.8 | 32 | 16.5 | 52 | 20 |
| finetuned CUT3R (augfull_lr1e5) | 72.3 | 6.54 | 0.960 | | | | | |
| champion (augfull_cg_fuse_g7) | 62.8 | 6.02 | 0.860 | | | | | |

Failure anatomy (v0): 37% failed frames = 26% PnP < 20 inliers with a live map (map inconsistent with tracks),
11% waiting for (re)bootstrap parallax, 0.5% map starved. Death spiral: after a PnP failure no keyframe is declared,
so no new points are triangulated, the map decays for 100+ frames until it starves, then a re-bootstrap starts a new
scale segment (23 reboots / 12 scenes). Culling outlier points cuts PnP failures 26% -> 5.6%. Per-scene ATE is worse
than the finetuned model on every scene, including the one scene with zero failures (74 vs 56 mm).

## v0 pipeline sweep 2 (2026-09-05; Slurm 45453768) and the zero-shot reference

cut3r_zeroshot = pretrained cut3r_512_dpt_4_64.pth run fresh on the 12 smoke scenes (Slurm 45453660).

| variant | ATE | RPE-t | RPE-r | failed% | reboots | med inl |
|---|---|---|---|---|---|---|
| vo_v0 | 122.9 | 9.65 | 1.129 | 37.1 | 23 | 63 |
| cull | 123.4 | 8.18 | 1.113 | 18.1 | 30 | 85 |
| cull + lever | 118.3 | 7.38 | 0.976 | 22.4 | 37 | 70 |
| cull + reboot after 3 fails | 119.9 | 7.99 | 1.009 | 14.3 | 33 | 87 |
| cull + reboot after 10 fails | 122.6 | 7.92 | 1.020 | 14.1 | 33 | 85 |
| cull + lever + rb3 | 120.0 | 7.34 | 0.923 | 15.7 | 42 | 69 |
| cull + lever + rb3 + min inliers 30 | 116.8 | 7.07 | 0.861 | 27.4 | 41 | 81 |
| cull + lever + rb3 + PnP 3 px | 124.6 | 8.07 | 1.084 | 13.1 | 31 | 92 |
| cull + lever + rb3 + PnP 1 px | 120.4 | 7.71 | 0.866 | 21.4 | 58 | 50 |
| **CUT3R zero-shot (pretrained)** | 118.0 | 10.12 | 1.506 | | | |
| CUT3R finetuned | 72.3 | 6.54 | 0.960 | | | |
| champion | 62.8 | 6.02 | 0.860 | | | |

Reading: RPE (local accuracy) of the best classical config matches the champion on rotation (0.861 vs 0.860) and
beats the finetuned model (0.960); RPE-trans sits between finetuned (6.5) and zero-shot (10.1). ATE is stuck at
~117-125 mm for every variant = parity with zero-shot, 60% worse than finetuned: it is set by the 30-58 re-bootstraps
per 12 scenes, each starting a new arbitrary-scale segment that Sim(3) cannot repair. Decisions from this sweep:
2.7 stays 2 px (1 px: more reboots; 3 px: worse everything); 3.1 -> 30; 3.2b recovery -> reboot after 3 consecutive
PnP failures. Next: 2.15 scale hand-off across re-bootstraps (sweep 3).

## v0 pipeline sweep 3 (2026-09-05; Slurm 45453807): scale hand-off and the lever default

| variant | ATE | RPE-t | RPE-r | ATE median | failed% | reboots (hand-offs) |
|---|---|---|---|---|---|---|
| cull+lever+rb3+mi30 (sweep-2 best) | 116.8 | 7.07 | 0.861 | 117.4 | 27.4 | 41 (0) |
| **+ scale hand-off = v0 FINAL (lever ON)** | **111.9** | **6.93** | **0.871** | **97.7** | 21.5 | 46 (12) |
| v0 FINAL, lever OFF | 115.6 | 8.40 | 0.971 | 104.7 | 20.1 | 36 (8) |
| v0 FINAL with min inliers 20 | 116.3 | 7.28 | 0.927 | 102.1 | 9.2 | 44 (9) |
| CUT3R zero-shot | 118.0 | 10.12 | 1.506 | 118.4 | | |
| CUT3R finetuned | 72.3 | 6.54 | 0.960 | 69.8 | | |
| champion | 62.8 | 6.02 | 0.860 | 55.1 | | |

Per-scene: hand-off helps most on short scenes with several reboots (124.6 -> 64.0, 88.9 -> 71.5); the 400-700-frame
scenes stay at 130-195 mm (drift + reboots the hand-off could not bridge). The classical arm beats the finetuned model
on ATE on no scene; on RPE-rot it beats it on the mean.

## v0 FINAL configuration (frozen 2026-09-05)

    python eval_pipeline/opencv_vo.py --scenes_root $SCENES_ROOT --scene_list <list> \
        --preds_root $OUT/augfull_lr1e5/preds --out_root $OUT --label <label> \
        --cull outlier --reboot_after_fails 3 --min_inliers 30 --scale_handoff \\
        --focal perframe:$OUT/augfull_lr1e5/preds   # inference-time focal; lever ON by default (--no_lever disables)

Then: python eval_pipeline/pose_sim3_both.py --out_root $OUT --scenes_root $SCENES_ROOT --scene_list <list> --labels <label> ...
Depth columns (AbsRel, d1) are inherited from augfull_lr1e5 (pass --depth_link $OUT/augfull_lr1e5/preds for the full harness).
Diagnostics: <label>/preds/<scene>/diag.csv (per frame) and summary.json (failed_frames, reboots, keyframes, median_inliers).

## Focal-source check in the full pipeline (2026-09-05; Slurm 45453975)

| config (v0 final otherwise) | ATE | RPE-t | RPE-r | ATE median | failed% | reboots |
|---|---|---|---|---|---|---|
| finetuned-model focal (0.1b as decided) | 111.9 | 6.93 | 0.871 | 97.7 | 21.5 | 46 |
| fixed nominal focal 203 px (strictly model-free) | 109.1 | 7.04 | 0.869 | 92.7 | 16.4 | 54 |
| calibrated GT focal (diagnostic) | 116.0 | 6.91 | 0.885 | 107.9 | 17.1 | 54 |
| finetuned focal, lever off | 115.6 | 8.40 | 0.971 | 104.7 | 20.1 | 36 |
| nominal focal, lever off | 122.3 | 7.96 | 0.944 | 103.4 | 16.0 | 40 |

Zero-shot CUT3R focal (229-570 px across the 12 scenes, median 1.5x the finetuned value) was not run: it is not a
usable camera model. Conclusion: the OpenCV rows do not depend on which checkpoint supplied the focal; the strict
zero-shot classical row is `--focal 203` and it is within noise of the v0 final.

## Focal sensitivity and OpenCV self-calibration (2026-09-05; Slurm 45456412; v0 final, lever ON)

What CUT3R's focal is: the model has NO focal/intrinsics head. Its per-frame outputs are point maps (self-view and
world-view), confidence and a camera-pose encoding. The focal in camera/*.npz is derived AFTER inference by
estimate_focal_knowing_depth(pts3d_self, pp=centre, "weiszfeld"): each pixel's predicted 3D point implies
f = (u-cx)*z/x and the Weiszfeld fit is a robust aggregate over the frame. This is upstream demo.py's convention,
reproduced unchanged by infer_and_eval_worker.py since the first run. Nothing in eval_depth_poses.py or the pose
scorer reads it; the OpenCV arm is its first consumer. The zero-shot checkpoint's 229-570 px values therefore mean
its point-map geometry is inconsistent on this footage, not that a focal predictor failed.

| focal fed to OpenCV v0 | ATE | RPE-t | RPE-r | failed% | reboots |
|---|---|---|---|---|---|
| nominal 203 px (calibrated value in the 320x192 frame) | 109.1 | 7.04 | 0.869 | 16.4 | 54 |
| finetuned-model focal (~213 px) | 111.9 | 6.93 | 0.871 | 21.5 | 46 |
| 150 px (0.74x) | 109.8 | 7.23 | 0.991 | 14.0 | 62 |
| 305 px (1.5x) | 117.9 | 7.54 | 0.905 | 17.1 | 52 |
| 406 px (2x) | 119.2 | 7.45 | 0.975 | 15.7 | 53 |
| OpenCV self-calibration (essential-matrix consistency sweep over 30 parallax-gated pairs) | 116.6 | 7.86 | 0.935 | 17.4 | 50 |

Self-calibrated focal per scene: 132, 280, 196, 162, 358, 448, 174, 180, 210, 278, 298, 510 (true ~203).
Reading: (1) the pipeline is far more focal-tolerant than the pinhole argument predicts: a 2x focal error costs
~10 mm ATE and ~0.1 deg RPE-r, because map and PnP share the same (wrong) K and Sim(3) absorbs the scale.
(2) Classical self-calibration from this footage is unreliable (0.65x-2.5x scatter), failing the same way the
zero-shot model's implicit focal does; the finetuned model's stable 211-217 px is learned knowledge of the DROID
camera that images alone at this resolution/baseline do not pin down. (3) Decision 0.1b stands; `--focal 203`
remains the model-free row; `--focal selfcal` exists but is not recommended.

## Inference-time (causal) focal (2026-09-06; Slurm 45485470; v0 final, lever ON)

The user's constraint: OpenCV at frame t may only use what CUT3R has produced up to t; retroactive per-scene medians
are not an inference-time method. Pipeline changed to a per-frame camera matrix everywhere.

| focal fed to OpenCV v0 | focal range (px) | ATE | RPE-t | RPE-r | ATE median | failed% | reboots |
|---|---|---|---|---|---|---|---|
| finetuned CUT3R, focal of frame t | 205-221 | 110.4 | 7.53 | 0.858 | 91.5 | 19.1 | 51 |
| finetuned CUT3R, running median to t | 207-217 | 113.8 | 6.85 | 0.844 | 98.9 | 19.6 | 51 |
| zero-shot CUT3R, focal of frame t | 207-819 | 124.8 | 10.33 | 1.059 | 109.4 | 14.8 | 63 |
| zero-shot CUT3R, running median to t | 213-599 | 114.7 | 7.92 | 0.891 | 97.8 | 17.8 | 55 |
| finetuned, per-scene median (retroactive; previous rows) | 212-217 | 111.9 | 6.93 | 0.871 | 97.7 | 21.5 | 46 |
| fixed 203 px (model-free) | 203 | 109.1 | 7.04 | 0.869 | 92.7 | 16.4 | 54 |
| CUT3R zero-shot | | 118.0 | 10.12 | 1.506 | 118.4 | | |
| CUT3R finetuned | | 72.3 | 6.54 | 0.960 | 69.8 | | |
| champion | | 62.8 | 6.02 | 0.860 | 55.1 | | |

Reading: with the finetuned checkpoint the causal per-frame focal reproduces the retroactive numbers (110.4/7.53/0.858
vs 111.9/6.93/0.871). Paired rows: zero-shot CUT3R + OpenCV (its own per-frame focal, 207-819 px) = 124.8/10.33/1.06
vs zero-shot CUT3R 118/10.1/1.51; finetuned CUT3R + OpenCV = 110.4/7.53/0.86 vs finetuned 72.3/6.54/0.96. The
OpenCV arm beats each checkpoint on RPE-rot, ties/loses on ATE, and never beats the finetuned model on ATE.
The table build (eval_pipeline/build_opencv_vo_table.py) now reports the inference-time rows.

## Honesty audit (2026-09-06/07) and the user's rulings

Listed for the user on 2026-09-06 (numbers as in that message): (1) retroactive PnP fill of pre-bootstrap frames
[non-causal]; (2) retro-fill accepts >= 4 inliers vs 30 live and is marked not-failed; (3) whole sequence in
memory; (4) the 203 px "model-free" focal came from GT calibration; (5) lever thresholds designed with GT-aided
inspection; (6) a `--focal gt` code path exists; (7) all thresholds tuned on the 12 reported (test) scenes;
(8) design benchmarks GT-scored on test scenes; (9) finetuned row causal vs champion row fwd x bwd;
(10) depth columns would be the paired model's; (11) zero-shot got no configuration search; (12) failed-frame
count excludes retro-filled frames; (13) stale per-scene-median rows on disk.
Rulings so far: item 1 ACCEPTED 2026-09-07 with a footnote clause in the full-4292 table
(eval_pipeline/build_opencv_full_table.py). Item 7 is addressed by reporting on all 4292 scenes with the
configuration frozen. Others pending the user's decision.

## Retro-fill ablation (2026-09-07; 12 smoke scenes; v0 final with per-frame finetuned focal)

`--no_retro_fill` = strictly causal (pre-bootstrap frames keep the held anchor pose and stay flagged failed).

| | ATE | RPE-t | RPE-r |
|---|---|---|---|
| with retro-fill (audit item 1, accepted) | 110.4 | 7.5 | 0.858 |
| without (strictly causal) | 110.6 | 7.7 | 0.881 |
| CUT3R finetuned | 72.3 | 6.5 | 0.960 |

Effect of the non-causal step: +0.2 mm ATE, +0.2 mm RPE-t, +0.02 deg RPE-r. It does not change any ordering.

## FULL HARNESS (all 4292 scenes; 2026-09-07): CUT3R with and without the OpenCV pose backbone

Table: /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/tables/opencv_backbone_full4292.{tex,pdf,png}
(builder: eval_pipeline/build_opencv_full_table.py; scoring: eval_depth_poses.py per scene via
eval_pipeline/opencv_vo_eval.py, nanmean over scenes = aggregate_results.py convention, pose RMSE-reduced per scene).
Zero-shot inference: Slurm 45527499-502 (16 GPU workers, ~90 min). OpenCV rows: Slurm 45527508 / 45529336
(48 cores, ~16 min per row + scoring). Per-frame focal from the paired checkpoint (inference-time); lever ON;
retro-fill retained (accepted, footnoted).

| Method | AbsRel | d<1.25 | ATE (m) | RPE-t (m) | RPE-r (deg) |
|---|---|---|---|---|---|
| CUT3R zero-shot | 0.479 | 0.558 | 0.120 | 0.012 | 1.713 |
| Regular CUT3R (finetuned) | 0.179 | 0.786 | 0.076 | 0.008 | 1.101 |
| CUT3R zero-shot + OpenCV backbone | 0.479 | 0.558 | 0.120 | 0.012 | 1.301 |
| Regular CUT3R + OpenCV backbone | 0.179 | 0.786 | 0.104 | 0.008 | 1.114 |
| Regular CUT3R + OpenCV backbone, GT intrinsics | 0.179 | 0.786 | 0.103 | 0.008 | 1.102 |

Reading: on the zero-shot checkpoint the OpenCV backbone cuts RPE-rot 1.71 -> 1.30 deg at equal ATE / RPE-t.
On the finetuned checkpoint it ties on RPE (1.114 vs 1.101; 0.0084 vs 0.0079) and loses 37% on ATE (0.104 vs
0.076). GT intrinsics are worth ~2 mm ATE. Runtime: OpenCV 30 ms/frame on one CPU core vs ~70 ms/frame CUT3R
on one H100 (incl. eval subprocess); OpenCV is additive (needs the model's focal/depth) but GPU-free.

## Why ATE / RPE-trans look alike across arms: the metric floors (2026-09-07)

Per-scene check on all 4292 scenes: 0 scenes have identical pose metrics between a model row and its OpenCV row
(per-scene correlation 0.44-0.68; median |ATE diff| 20-32 mm; OpenCV better on 2013/4292 vs zero-shot, 1030/4292 vs
finetuned). Depth columns identical by construction (symlinked). So the similarity of the MEANS is not a plumbing bug.

Trivial trajectories pushed through the SAME Sim(3) scorer (144 sampled scenes, harness convention):

| trajectory | ATE (m) | RPE-t (m) | RPE-r (deg) |
|---|---|---|---|
| constant pose (no motion at all) | 0.171 | 0.0079 | 1.21 |
| constant velocity from GT step 1 | 0.125 | 0.0082 | 1.23 |
| CUT3R zero-shot | 0.122 | 0.0123 | 1.71 |
| zero-shot + OpenCV | 0.123 | 0.0123 | 1.29 |
| CUT3R finetuned | 0.076 | 0.0081 | 1.08 |
| finetuned + OpenCV | 0.109 | 0.0083 | 1.07 |
| champion fusion | 0.068 | 0.0072 | 0.98 |

GT per-frame step: translation median 3.7 mm / per-scene RMSE 7.9 mm; rotation median 0.56 deg.
Findings: (1) RPE-trans is SATURATED at the no-motion floor (7.9 mm = the RMS step itself) for every arm except the
champion (7.2); no method resolves per-frame translation at this frame rate / resolution, so RPE-trans cannot
separate arms. (2) Zero-shot CUT3R's RPE-rot (1.71) is WORSE than predicting no rotation (1.21); OpenCV on top
brings it to 1.29, near the floor. (3) ATE of zero-shot and of the OpenCV rows sits at the constant-velocity floor
(0.12); only the finetuned model and the champion beat it clearly. (4) RPE-rot and ATE are the informative columns.

## Verification of the full-4292 table + why cells look unchanged (2026-09-09)

Independent re-derivation: all 4292 per-scene CSVs re-aggregated outside the table builder -> all
five rows reproduce to 4 dp, n=4292 each, no dropped scenes. One scene's ATE recomputed by hand
from the raw camera/*.npz through Sim(3) alignment matches the harness to 5 dp for all five rows
(and confirms the harness MEAN row = RMSE over frames). OpenCV rows contain OpenCV poses: distinct
md5 over concatenated pose bytes, frame-10 translations differ 0.01-1.44 m, and 0/4292 scenes share
a pose metric with their base row. Depth dirs are symlinks to the paired model's depth.

Paired statistics over 4292 scenes (base -> base+OpenCV):

| pairing | metric | mean diff | median abs per-scene diff | better/worse | paired t | reading |
|---|---|---|---|---|---|---|
| zero-shot | ATE | +0.00053 | 0.0204 (39x) | 2013/2279 | 0.9 | statistical tie |
| zero-shot | RPE-t | +0.00040 | 0.0024 (6x) | 2186/2106 | 4.8 | significant, 0.4 mm = 5% of floor |
| zero-shot | RPE-rot | -0.41215 | 0.4897 (1x) | 3407/885 | -28.4 | real improvement |
| finetuned | ATE | +0.02842 | 0.0319 (1x) | 1030/3262 | 44.8 | real regression |
| finetuned | RPE-t | +0.00046 | 0.0016 (3x) | 2034/2258 | 7.5 | significant, 0.5 mm |
| finetuned | RPE-rot | +0.01314 | 0.2256 (17x) | 2183/2109 | 1.5 | statistical tie |

Three mechanisms explain the near-identical cells: (1) depth is architecturally identical (symlink);
(2) cancellation - per-scene differences are 6-39x the mean difference and wins/losses nearly
offset (zero-shot ATE: -57.49 m of wins vs +59.75 m of losses); (3) RPE-trans is saturated at the
no-motion floor (0.0079 m) since GT moves 3.7 mm/frame median. Backbone is not degenerating: it
holds pose on 21.4% of frames, re-bootstraps 15071 times over 4292 scenes (median 2/scene), median
85 PnP inliers; a degenerate constant-pose trajectory would score 0.1710 m ATE, not 0.1043 m.

## Trivial-baseline floors on ALL 4292 scenes (2026-09-09; supersedes the 144-scene sample)

Same scorer, per-scene RMSE then mean over scenes:

| trajectory | ATE (m) | RPE-t (m) | RPE-rot (deg) |
|---|---|---|---|
| constant pose (camera never moves) | 0.1685 | 0.0079 | 1.2068 |
| constant velocity from GT step 1 | 0.1248 | 0.0083 | 1.2150 |

ATE headroom below the constant-velocity floor: zero-shot 4.1%, zs+OpenCV 3.6%, finetuned 39.2%,
ft+OpenCV 16.4%, ft+OpenCV GT-K 17.8%. This is the cleanest statement of the whole evaluation:
on ATE the entire zero-shot pairing lives in a 4%-wide band above a trivial baseline (so a tie
there means "both nearly uninformative", not "both good"), whereas the finetuned model has 39%
headroom and the OpenCV backbone gives back well over half of it.

Per-scene ATE difference distributions (OpenCV minus model, mm, n=4292):
  zero-shot pairing: mean +0.53, std 36.90, SE 0.56 (0.9 SE from zero), p25/p50/p75 = -19.0/+1.8/+21.0
                     -> wide, near-symmetric, centred on zero = no systematic difference
  finetuned pairing: mean +28.42, std 41.57, SE 0.63 (44.8 SE from zero), p25/p50/p75 = +1.1/+26.6/+54.1
                     -> same spread but shifted right = a real, systematic regression
The spread is NOT small: the two arms disagree by ~37 mm (1 sigma) per scene in both pairings.
The zero-shot means coincide because the disagreements are symmetric, not because the trajectories
are alike.

## DECISIVE CONTROL (2026-09-11): ATE ~0.12 is the random-walk value on this dataset

537-scene sample, same Sim(3)/RMSE scorer, deliberately-constructed trajectories:
constant pose 0.1678 | GT positions permuted in time 0.1673 | RANDOM WALK with GT step sizes and
random directions 0.1206 | GT+noise at 1.0/0.5/0.25x extent 0.1406/0.1023/0.0603.

Paired per-scene against the random walk (3 seeds/scene, 537 scenes):
  CUT3R zero-shot      0.1203  diff -0.0004  t=-0.3   INDISTINGUISHABLE FROM A RANDOM WALK
  zero-shot + OpenCV   0.1199  diff -0.0009  t=-0.6   INDISTINGUISHABLE FROM A RANDOM WALK
  finetuned + OpenCV   0.1045  diff -0.0162  t=-10.5  better than random
  CUT3R finetuned      0.0765  diff -0.0443  t=-32.8  better than random

Implication for every table in this campaign: an ATE near 0.12 on the DROID wrist set means "no
usable global directional information", not "a competitive result". The zero-shot row and both
zero-shot-focal OpenCV rows sit there. This is why the classical arm appeared to match zero-shot
CUT3R to within a millimetre: both are on the random-walk attractor, as is an actual random walk.
The finetuned model (0.0759) and the finetuned+OpenCV arm (0.1043) are the only rows that carry
real global information.

## Why "different poses must give different metrics" fails (2026-09-11)

ATE is bounded, many-to-one, and then averaged over 4292 scenes:
- BOUNDED: on a 131-frame scene, GT+noise at 2/10/100/10000x the trajectory extent scores
  0.1281/0.1326/0.1315/0.1324, and a collapse to a point scores 0.1334. You cannot score worse than
  the GT trajectory's own spread, because Sim(3) shrinks garbage to a point. GT scaled 1000x scores
  0.0000 - the alignment absorbs scale entirely.
- MANY-TO-ONE: ten independent random walks on one scene score 0.083-0.114 while being 0.1406 m RMS
  apart from each other (comparable to the whole GT path extent).
- AVERAGING: per-scene zero-shot vs zero-shot+OpenCV differ by 36.9 mm (1 sigma); SE of the mean
  over 4292 scenes is 0.56 mm.
PROOF it needs no explanation: two INDEPENDENT random-walk ensembles differ by 0.67 mm in mean ATE
(537 scenes); CUT3R zero-shot vs zero-shot+OpenCV differ by 0.60 mm (4292 scenes). Millimetre
agreement between mean ATEs is the metric's behaviour, not evidence of a relationship.
Probe: eval_pipeline/opencv_vo_probes/ate_two_random_walks.py

## Frozen v0 parameters (if all picks accepted)

| Parameter | Value |
|---|---|
| Max corners (goodFeaturesToTrack) | 1000 |
| LK window / pyramid levels | 21 px / 3 (OpenCV defaults) |
| PnP RANSAC reprojection threshold | 2 px |
| Bootstrap essential-matrix RANSAC threshold | 0.5 px |
| RANSAC confidence / iterations | 0.999 / 1000 |
| Min inliers before failure | 30 |
| Keyframe / bootstrap parallax trigger | 5 px median track displacement |
| Gripper handling | no mask. Lever `reject_static_tracks` ON by default since 2026-09-05 (drop tracks < 1 px on frames whose median flow >= 2 px); `--no_lever` disables |

## Plumbing and day-one checks

- Output per scene: `camera/%06d.npz` with `pose` (c2w, 4x4) and `intrinsics`
  (GT K). Depth dir symlinked from the run whose depth was used (or GT for the
  oracle arm). Score with eval_pipeline/pose_sim3_both.py.
- solvePnP returns world-to-camera (rvec, tvec): invert before writing c2w.
- Pose-convention check before any run: on one scene, PnP with GT depth between
  two frames must reproduce the GT relative pose composed from cam/*.npz.
- Run as a CPU job (login node has a 300 s per-process CPU cap).
