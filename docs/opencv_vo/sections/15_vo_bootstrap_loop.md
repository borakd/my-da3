## `eval_pipeline/opencv_vo.py` lines 338-423: frame 0, the per-frame loop head, and the bootstrap branch

This block is the start of the per-scene main loop inside `run_scene`. It seeds the trajectory at frame 0
(identity pose, first corner detection), then for every later frame it advances the persistent Lucas-Kanade
tracks by one frame, applies the static-track lever (decision 1.2), and, while the map does not yet exist
(`booted == False`), waits for enough parallax to bootstrap: essential matrix in per-view normalised
coordinates (decisions 2.9, 2.10), `recoverPose` cheirality guard, two-view triangulation with the cheirality
+ reprojection filter (decision 2.11), the scale hand-off across re-bootstraps (decision 2.15, gated by
`--scale_handoff`, which the frozen v0 configuration passes), the non-causal retro-fill of the frames that
were waiting (honesty-audit item 1, `--no_retro_fill`), and finally the promotion of the anchor and the
current frame to the first two keyframes. Frames that cannot bootstrap yet hold the anchor pose and are
flagged failed; if the tracks run out (or the wait exceeds a hard cap) the anchor is re-detected. Everything
after line 423 (PnP against the map, culling, the keyframe trigger) is the steady-state tracking branch and is
documented in the next section. Coordinate conventions used throughout: `T_w2c` is a 4x4 world-to-camera
transform (`X_c = R X_w + t`), the world frame is camera 0, pixel units are the eval loader's 320x192 cover
frame (decision 1.1), and every frame `f` has its own camera matrix `Ks[f]` (decision 0.1b, revised
2026-09-06). OpenCV behaviour quoted below was checked on cv2 4.11.0 in the `cuteanything` env, which is the
env the eval scripts activate.

### Lines 338-341: frame 0

```
338:     # ---- frame 0
339:     poses_c2w[0] = np.eye(4)
340:     new_bootstrap(0)
341:     diag.append(dict(frame=0, n_tracks=len(tracks), n_mp=0, inliers=-1, failed=0, keyframe=1, booted=0))
```

**What it does.** Line 339 pins the world frame to camera 0: the camera-to-world pose of frame 0 is the
identity, so every later pose is expressed relative to it (`poses_c2w` is a list of `n` 4x4 c2w matrices,
one per frame, pre-filled with `None`). Line 340 calls the `new_bootstrap` closure (defined earlier in
`run_scene`, lines 300-313): it detects Shi-Tomasi corners on frame 0 with `goodFeaturesToTrack(1000, 0.01,
5)` (decision 1.3) into the empty `Tracks` container, sets `booted = False`, records the anchor
world-to-camera pose `boot_anchor_T = last_T` (identity here), sets `anchor_frame = 0`, and starts
`pre_hist` with the `(ids, pos)` snapshot of frame 0 — `pre_hist[k - anchor_frame]` is the track snapshot
of frame `k`, used later by the retro-fill. Because `tracks` is empty at this point, the `--scale_handoff`
branch inside `new_bootstrap` produces an empty `old_mp` (no previous scale segment). Line 341 writes the
first diagnostics record: `inliers=-1` is the sentinel meaning "no PnP ran on this frame" (the
`summary.json` median is computed over records with `inliers >= 0`, so the sentinel is excluded),
`keyframe=1` because frame 0 is the anchor of the first map, `booted=0` because there is no map yet.

**Alternatives considered.** Fixing the world frame to the first frame is the universal monocular-VO
convention; the only alternative would be to anchor the world to GT, which the closed-loop rule of
decision 0.1 forbids (plain image inputs only). Scoring is Sim(3)-aligned (dataset facts in the design
doc), so the choice of origin is free anyway.

**Why this choice.** Decision 0.1 (DECIDED 2026-09-02): triangulated monocular map, no privileged inputs.
Decision 0.4: diagnostics go to a side CSV (`diag.csv`), never to the harness CSV.

### Lines 342-345: loop head and the LK step

```
342: 
343:     for f in range(1, n):
344:         prev_pos = tracks.pos.copy()
345:         ok = lk_step(G[f - 1], G[f], tracks)
```

**What it does.** After the blank line 342, line 343 iterates over frames 1..n-1 of the whole scene, all of
whose grayscale cover frames are already in memory in the list `G` (loaded before this block). Line 344
snapshots the `(N,2)` float32 track positions before they are moved, because the lever below needs the
per-frame displacement. Line 345 calls `lk_step`, which runs `cv2.calcOpticalFlowPyrLK` with
`winSize=(21,21)`, `maxLevel=3` (OpenCV defaults; `maxLevel` is 0-based, so four pyramid levels — the
design doc's "3 levels" is this `maxLevel` value) from `G[f-1]` to `G[f]`, then backwards, keeps a track
only if both status flags are 1, the forward-backward round-trip error is below `FB_THRESH = 1.0` px and the
new position lies inside the image, **overwrites `tracks.pos` with the forward result for every track**
(including the ones that failed), and returns the boolean `ok` mask of length N. With zero live tracks it
returns an empty mask.

**Alternatives considered.** Decision 1.5 options: sparse LK with/without the forward-backward check,
ORB or SIFT + ratio test, DIS or Farneback dense flow at corners or on a grid. Decision 1.6 options: re-detect
every frame, persistent tracks with a count-floor top-up, persistent tracks with top-up only at keyframes.

**Why this choice.** Decision 1.5 (DECIDED 2026-09-04), correspondence benchmark on the 12 smoke scenes,
4243 frames: LK + forward-backward check gives 166/88 correspondences/correct at gap 1 with 0.88 deg
essential-matrix rotation error, 103/27 at gap 4 (2.55 deg, p90 8.4), 4 ms; ORB (3.53 deg at gap 4), SIFT
(2.54 deg, 19 ms) and Farneback (3.66 deg) were rejected; DIS on an 8-px grid (1.88 deg) is the recorded
runner-up but does not give corner-anchored persistent tracks. LK+FB has the best per-correspondence
precision at gap 4 (in2 0.35 vs 0.29 for DIS-grid). Frame-to-frame tracking is forced by the probe:
consecutive-frame survival is 0.95, but over a 15-frame gap in fast segments it drops to ~15%. Decision
1.6 (DECIDED 2026-09-04): persistent tracks, top-up only at keyframes, no floor parameter. The window and
pyramid depth are OpenCV defaults, recorded in the frozen-parameter table as "21 px / 3 (OpenCV
defaults)"; no benchmark varied them.

### Lines 346-351: the static-track lever (decision 1.2)

```
346:         if args.lever and ok.any():
347:             # 1.2 lever: on frames with clear parallax (median per-frame displacement >= 2 px) a track that did not
348:             # move (< 1 px) is provably not scene geometry (image-static gripper): kill it.
349:             step = np.linalg.norm(tracks.pos - prev_pos, axis=1)
350:             if np.median(step[ok]) >= 2.0:
351:                 ok &= step >= 1.0
```

**What it does.** `args.lever` is `True` by default (`--no_lever` disables). The `ok.any()` guard avoids a
median over an empty set. Line 349 computes each track's displacement between frames `f-1` and `f` in
cover-frame pixels. Line 350 is the parallax condition: only if the median displacement over the
successfully tracked corners is at least 2.0 px is the camera considered to be moving. In that case line
351 additionally kills every track that moved less than 1.0 px — such a track is image-static while the
scene flows, which is the signature of the wrist-mounted gripper (dataset facts: "the gripper is
IMAGE-STATIC: its corners vote for zero motion in PnP"). On low-parallax frames the lever never fires, so a
genuinely paused camera is unaffected. The two thresholds (2 px / 1 px) are absolute pixel values in the
320x192 frame; the lever is applied per consecutive-frame step, before any geometry (E / PnP /
triangulation) sees the tracks. The comment lines 347-348 state the rule.

**Alternatives considered.** Decision 1.2 options: no mask; fixed dataset polygon; per-scene temporal-variance
mask (Otsu); flow-based rejection; GT `outlier_mask`. In the two-view diagnostics a relative variant (drop
tracks below 20% of the median displacement) was also measured.

**Why this choice.** Decision 1.2 (DECIDED 2026-09-04, lever added 2026-09-04, default flipped to ON on
2026-09-05 by user decision). The probe on RAIL+80edfcb1+2023-07-14-14h-28m-45s showed that the GT
`outlier_mask` does NOT cover the gripper, the Otsu temporal-variance mask over-masks (58%), and a fixed
polygon fails because finger geometry differs per lab; the gripper attracts few corners (median 12%
zero-motion tracks on moving frames, max 42%). Across the 12 smoke scenes the static share of LK tracks is
16-53% (median ~25%). On parallax-gated gap-4 pairs removing zero-motion tracks cuts E-rotation error from
2.14 to 1.70 deg (correspondence benchmark: "removing static tracks helps as much as changing method"). In
the full pipeline the lever is better on every metric: v0 final 111.9 / 6.93 / 0.871 (ATE mm / RPE-t mm /
RPE-r deg) ON vs 115.6 / 8.40 / 0.971 OFF (sweep 3), and 116.4 / 7.93 / 1.009 ON vs 122.9 / 9.65 / 1.129
for plain v0 (sweep 1). Honesty note: the two-view diagnostics (run with lever ON) state that static-track
removal, absolute 1 px or relative 20%, "does not change E" (the relative variant scores 1.96 / 32.5 vs
2.06 / 31.6 for the pipeline row at gap 4), whereas the correspondence benchmark measured 2.14 -> 1.70 deg
from dropping zero-motion tracks; the design doc does not reconcile the two, and the pipeline-level gain
(sweep 3) is the decisive measurement. Audit item 5 records that the thresholds were designed with
GT-aided inspection.

### Lines 352-355: prune the tracks and open the per-frame record

```
352:         tracks.keep(ok)
353:         # dead map points can never be re-observed (no descriptors): drop them
354:         rec = dict(frame=f, n_tracks=len(tracks), n_mp=int((tracks.mp >= 0).sum()), inliers=-1, failed=0, keyframe=0, booted=int(booted))
355: 
```

**What it does.** Line 352 applies the mask to all four structure-of-arrays fields of `Tracks` (`pos`,
`kf_pos`, `mp`, `ids`), removing tracks that failed LK, the forward-backward check, the bounds check, or
the lever. As the comment on line 353 says, a map point whose track died is thereby lost for good: the
pipeline has no descriptors, so it cannot re-associate a point with a new corner (its `mp_xyz` entry stays
in the dictionary but is unreachable from any track). Line 354 starts the diagnostics record for frame
`f`: number of surviving tracks, number of tracks that currently carry a map point (`mp >= 0`), the `-1`
inlier sentinel, `failed=0`, `keyframe=0`, and the `booted` state *before* this frame is processed. Line
355 is blank.

**Alternatives considered.** Decision 2.14 lists the culling options (none; drop points unseen for N
keyframes; drop after k PnP-outlier hits); implicit loss of dead tracks is common to all of them. A
descriptor-based re-observation (ORB-SLAM-style relocalisation) would be the alternative and was ruled out
with descriptors in decision 1.5.

**Why this choice.** Decision 2.14 (DECIDED 2026-09-05): "dead tracks are dropped implicitly: no
re-association without descriptors". Decision 0.4: per-frame inlier count and failed flag are the two
mandated diagnostics ("median inliers moves before ATE does").

### Lines 356-360: the not-yet-bootstrapped branch and the parallax gate (decision 2.9)

```
356:         if not booted:
357:             pre_hist.append((tracks.ids.copy(), tracks.pos.copy()))
358:             disp = np.linalg.norm(tracks.pos - tracks.kf_pos, axis=1) if len(tracks) else np.zeros(0)
359:             T_w2c = None
360:             if len(tracks) >= args.min_inliers and np.median(disp) >= PARALLAX_PX:
```

**What it does.** Line 356 enters the waiting state (no map). Line 357 appends this frame's `(ids, pos)`
snapshot to `pre_hist`, so that if the bootstrap succeeds later the intermediate frames can be re-posed.
Line 358 measures, per track, the displacement from `kf_pos` — the position the track had at the anchor
frame (set by `new_bootstrap`, or the detection position for tracks born there) — to its current
position, i.e. the accumulated parallax proxy since the anchor, in pixels. Line 359 initialises the
sentinel `T_w2c = None` ("no pose from bootstrap this frame"). Line 360 is the bootstrap gate: at least
`args.min_inliers` live tracks (30 in the frozen v0 configuration; the argparse default is 20) **and** a
median displacement of at least `PARALLAX_PX = 5.0` px. Note that the gate uses the median of track
displacement, not a baseline in metres — the baseline is unknown before the map exists.

**Alternatives considered.** Decision 2.9 options: frames 0/1; first frame with median track displacement
above P px (P = 10 originally, then 5); H-vs-E model selection (ORB-SLAM style); tracked-ratio trigger.

**Why this choice.** Decision 2.9 (DECIDED 2026-09-04, P revised 10 -> 5 on 2026-09-05 together with decision
2.12 so the two share one parameter). Probe evidence: the RAIL scene is stationary for frames 0-15 (GT
rotation 0.03 deg, |t| = 0.2 mm), so the essential matrix is degenerate there and the bootstrap MUST be
parallax-gated; 9 mm per-step motion makes frame 0/1 degenerate. The keyframe-trigger sweep gives the number
for 5 px: median gap 2 frames (p10-p90 1-8), PnP rotation error at keyframe+4 of 2.96 deg, versus 3.23 deg for
10 px and 4.25 deg for 20 px; "every family improves monotonically as keyframes get closer" (lever OFF in that
sweep, unlike the shipped frontend). H-vs-E selection is recorded as the v1 fallback if bootstrap rejections
cluster on planar scenes. The track-count floor reuses decision 3.1's 30 (sweeps 1-2: 30 vs 20 gives RPE-r
0.861 vs 0.923).

### Lines 361-364: per-view normalised coordinates

```
361:                 a = tracks.kf_pos.astype(np.float64); b = tracks.pos.astype(np.float64)
362:                 Ka, Kb = Ks[anchor_frame], Ks[f]
363:                 an = (a - Ka[:2, 2]) / Ka[0, 0]; bn = (b - Kb[:2, 2]) / Kb[0, 0]     # each view normalised by its own K
364:                 I3 = np.eye(3)
```

**What it does.** `a` are the anchor-frame pixel positions and `b` the current-frame positions of the same
tracks, both cast to `(N,2)` float64 (OpenCV accepts single or double precision; the cast simply avoids
float32 rounding in the hand normalisation on line 363). Line 362 fetches the camera matrix of each view
separately: `Ks[anchor_frame]` and `Ks[f]` may differ because the focal is CUT3R's per-frame estimate for
that frame. Line 363 converts both point sets to normalised image coordinates, `(u - cx)/f, (v - cy)/f`;
since every `K` in this pipeline has `fx = fy = f` and the principal point at the image centre (`K_of` /
`model_K` build them that way), this is exactly `K^-1 [u v 1]^T`. Line 364 builds the identity camera
matrix that will be handed to OpenCV so that it treats the inputs as already normalised. The single-K
overloads of `findEssentialMat` / `recoverPose` assume both views share `K`; OpenCV's docstring offers two
routes for differing intrinsics — the two-camera overloads (`cameraMatrix1 / distCoeffs1 / cameraMatrix2 /
distCoeffs2`, which re-estimate E from the points) or pre-normalising the points and passing the identity
matrix. The code takes the second route, which keeps the RANSAC `E` and its inlier mask available for lines
373 / 387.

**Alternatives considered.** Decision 0.1b options: calibrated K from `cam/*.npz`; one nominal dataset K;
model-estimated focal (per-scene median, running median, or per-frame); self-calibration. For the two-view
call itself: pass one shared K to OpenCV in pixel units (only valid when both views share K); or the
two-camera overloads `findEssentialMat(points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2,
distCoeffs2, ...)` and `recoverPose(points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2,
distCoeffs2, ...)`, which exist in cv2 4.11.0 and take the two intrinsics directly.

**Why this choice.** Decision 0.1b, REVISED 2026-09-06 to the inference-time per-frame focal: "frame t uses
the focal CUT3R produced for frame t; every geometric step carries a per-frame K (bootstrap E in per-view
normalised coordinates, KF-to-KF triangulation with each keyframe's K, PnP with frame t's K)". The user's
constraint was that OpenCV at frame t may only use what CUT3R has produced up to t. The per-frame focal
reproduces the retroactive per-scene-median numbers (110.4 / 7.53 / 0.858 vs 111.9 / 6.93 / 0.871), and the
focal source is immaterial overall (nominal 203 px: 109.1 / 7.04 / 0.869; calibrated: 116.0 / 6.91 /
0.885). The finetuned focal is 205-221 px within the smoke scenes. With the zero-shot checkpoint (207-819 px
per frame) the choice of per-frame vs running-median focal matters: 124.8 / 10.33 / 1.059 vs 114.7 / 7.92 /
0.891 (inference-time focal table).

### Lines 365-368: essential matrix and pose recovery (decision 2.10)

```
365:                 E, inl = cv2.findEssentialMat(an, bn, I3, method=cv2.RANSAC, prob=E_CONF,
366:                                               threshold=E_THRESH / (0.5 * (Ka[0, 0] + Kb[0, 0])))
367:                 if E is not None and E.shape == (3, 3) and inl is not None:
368:                     npass, R, t, _ = cv2.recoverPose(E, an, bn, I3, mask=inl.copy())
```

**What it does.** Lines 365-366 run Nister's five-point solver inside RANSAC with confidence `E_CONF = 0.999`.
OpenCV's `threshold` argument is a distance to the epipolar line expressed in the units of the input
coordinates (when a real K is passed OpenCV itself divides the pixel threshold by the mean focal after
normalising the points internally); because the points are already normalised and `I3` is passed, the code
performs that division explicitly: `E_THRESH = 0.5` px divided by the mean of the two views' focals. The
return values are the 3x3 essential matrix `E` (or `None` on failure; the solver can also return several
candidate matrices stacked as `(3k,3)`) and `inl`, a `(N,1)` uint8 inlier mask. Line 367 accepts only a single
well-formed `E`; anything else falls through to the waiting path (`T_w2c` stays `None`). Line 368 decomposes
`E` into its four `(R, t)` candidates and picks the one with the most inliers in front of both cameras:
`npass` is that cheirality count, `R` (3x3) and `t` (3x1, unit norm) map anchor-camera coordinates to
current-camera coordinates (`x_b = R x_a + t`), and the fourth return value is the mask updated to contain
only points passing cheirality. The mask is passed as a **copy** because the Python binding writes into the
array it is given (on cv2 4.11.0 the returned mask *is* the passed-in object) and the code needs the untouched
RANSAC mask `inl` later (lines 373 / 387). OpenCV pitfall: this overload of `recoverPose` applies its default
`distanceThresh = 50`, so a point whose triangulated depth exceeds 50 baseline-units in either camera does not
count as passing (checked on cv2 4.11.0 with a clean synthetic pair: points at 52-60 baseline units give
`npass = 0` with the default, the same pair at 40-48 units passes all of them, and `distanceThresh = 1000`
restores the 52-60 set). The four-return call used here cannot override the default: passing `distanceThresh`
selects the other overload, which returns the triangulated points as a fifth value (also checked on 4.11.0).
In the frozen configuration this cut is expected to bite on most of the scene, not just its far edge. Decision
2.9's scaling ("10 px ~ 3 typical frames ~ 25 mm baseline at 0.5 m") was written for the original 10 px
trigger; 2.9 was revised on 2026-09-05 and the reported configuration bootstraps at `PARALLAX_PX = 5.0`, i.e.
about half the displacement — ~1.5 typical frames by that same scaling, and the keyframe-trigger sweep (which
shares the parameter, decision 2.12) measures a median gap of 2 frames (p10-p90 1-8) at 5 px. At the dataset's
median per-step motion of 9 mm that is a bootstrap baseline on the order of 13-18 mm, shorter than 2.9's 25 mm
estimate (the ~36 mm of decision 2.11 is the later keyframe-to-keyframe scale, not this pair). Against the
doc's depths (0.3-1.4 m, dataset facts) a scene then spans roughly 17-110 baseline units — derived from those
doc numbers, not measured — and the 50-unit cut falls at about 0.65-0.9 m depth, i.e. across the middle of the
typical depth range. So the far half of a typical scene does not count towards `npass` at line 369 — a
systematic truncation of the count, not the marginal "near the cut-off" effect that a 25 mm baseline would
imply: the shorter the bootstrap baseline, the more of the scene falls past the cut. Those points are still
triangulated at line 372, which does not use the `recoverPose` mask, so the truncation acts on the cheirality
guard only, making it stricter than the nominal 30.

**Alternatives considered.** Decision 2.10 options: `findEssentialMat` + `recoverPose` (RANSAC threshold 2,
1 or 0.5 px), homography, H-vs-E model selection. The two-view diagnostics additionally measured MAGSAC++
at 2 px and a two-view Sampson refinement of E. Decision 2.6 (robust estimator) options: RANSAC, USAC /
MAGSAC++, none.

**Why this choice.** Decision 2.10 (DECIDED 2026-09-05, revised from 2 px): tightening the RANSAC threshold
2 -> 0.5 px cuts the two-view rotation error from 2.06 to 1.23 deg and the translation-direction error from
31.6 to 18.1 deg at gap 4 (1 px: 1.64 / 24.3; MAGSAC 2 px: 1.34 / 22.9; Sampson refinement of E(2px): 1.70 /
29.0); at the map level, PnP at keyframe+4 improves 4.19 -> 3.02 deg (gap 4) and 3.58 -> 2.62 deg (gap 2)
(map-quality-vs-E-threshold sweep, lever OFF in that sweep).
The reading in the design doc: "only the sub-pixel minority of chained LK tracks is geometrically clean".
Decision 2.6: plain RANSAC ("three stateable numbers"). `E_CONF = 0.999` is fixed by decision 2.10 ("0.999
conf"); it happens to equal the PnP confidence of decision 2.8. The unit translation from `recoverPose`
sets the map scale of this segment (decision 2.10: "Unit translation sets map scale"; decision 3.3: scale
left as given, Sim(3) scoring absorbs one global scale). Known limitation (two-view diagnostics): the ~30
deg translation-direction error at these baselines is not specific to the classical arm — the finetuned
model has 2.22 / 31.7 and the champion 1.99 / 30.5 at gap 4 (model rows: all pairs, no parallax gating,
per the design doc) — and "the two-view bootstrap pose ... is the bottleneck of the map arm, not the
acceptance filter".

### Lines 369-373: cheirality guard, relative pose, triangulation (decisions 2.9, 2.11)

```
369:                     if npass >= args.min_inliers:
370:                         T_rel = rt_to_T(R, t)                       # anchor cam -> this cam (unit translation)
371:                         T_w2c = T_rel @ boot_anchor_T
372:                         X, acc = triangulate_pair(Ka, Kb, boot_anchor_T, T_w2c, a, b)
373:                         acc &= inl[:, 0].astype(bool)
```

**What it does.** Line 369 is the cheirality guard: the pair is accepted only if at least
`args.min_inliers` (30 in v0 final) RANSAC inliers lie in front of both cameras (and within the
`distanceThresh` noted above). A near-pure-rotation pair produces an E whose decomposition puts points at
inconsistent depths, so `npass` collapses and the pipeline keeps tracking and retries on the next frame.
Line 370 packs `(R, t)` into a 4x4 rigid transform from the anchor camera to the current camera; because
`t` is a unit vector, the baseline of this pair is 1 in map units. Line 371 composes it with the anchor's
world-to-camera pose to get the current frame's world-to-camera pose (`T_w2c = T_rel @ boot_anchor_T`:
world -> anchor camera -> current camera; for the first bootstrap `boot_anchor_T` is the identity). Line
372 calls `triangulate_pair`, which forms `P = K [R | t]` for each view with its own `K`, runs
`cv2.triangulatePoints` (DLT on 3x4 projection matrices and 2xN points, returning 4xN homogeneous points),
dehomogenises, and returns the `(N,3)` world points `X` plus an acceptance mask requiring positive depth in
**both** cameras, reprojection error below `TRI_REPROJ = 2.0` px in **both** views, and finite
coordinates. Line 373 additionally restricts acceptance to the RANSAC inliers of E (column 0 of the `(N,1)`
mask, cast to bool).

**Alternatives considered.** Decision 2.9: cheirality floor 20 (as originally written) vs the 3.1 floor;
decision 2.11 options: no filter; cheirality only; cheirality + reprojection < 2 px; cheirality + parallax
>= 1 deg; all three. Decision 3.1 options for the floor: inlier count 10 / 20 / 30, or a ratio.

**Why this choice.** Decision 2.9 ties the guard to "the same floor as 3.1"; decision 3.1 (DECIDED
2026-09-05) settled on 30 (sweeps 1-2: 10 -> RPE-r 1.80 with bad PnP accepted; 30 vs 20: RPE-r 0.861 vs
0.923, RPE-t 7.07 vs 7.34, ATE 116.8 vs 120.0). The design-doc text of 2.9 still says ">= 20"; the code
uses `args.min_inliers`, so in the frozen configuration the guard is 30. Decision 2.11 (DECIDED
2026-09-05), triangulation-acceptance benchmark with the GT pose (isolating the filter): cheirality +
reprojection gives PnP rotation 0.56 deg vs 0.63 (cheirality only) vs 0.73 (none) and 22% bad-depth points
vs 32% vs 46%; the parallax >= 1 deg filter was left out of v0 because at keyframe-scale baselines (~36 mm)
it discards half the points and doubles PnP failures (fails 329 vs 163). The 2 px value is not a new
parameter: it is the PnP RANSAC threshold of decision 2.7. With the E pose all filters tie at ~4 deg
because the two-view pose dominates the map error (key finding of that benchmark).

### Lines 374-378: scale hand-off, part 1 — eligibility (decision 2.15)

```
374:                         if old_mp and acc.sum() >= 10:
375:                             # 2.15 scale hand-off: match the new segment's scale to the previous one through tracks
376:                             # that carried a map point before the re-bootstrap (depth ratio in the anchor camera)
377:                             ratios = []
378:                             for row in np.where(acc)[0]:
```

**What it does.** `old_mp` is a dictionary `track id -> world xyz` filled by `new_bootstrap` when
`--scale_handoff` is set (line 303): at a re-bootstrap the live tracks are kept alive, their map-point links
are forgotten (`tracks.mp[:] = -1`, line 304) but the points they used to carry are remembered here, in the
*previous* segment's scale (points already culled out of `mp_xyz` are skipped). Without the flag `old_mp` is
always empty (line 307) and this whole block is skipped — but the argparse default being off does not make the
block dead in the reported numbers: the frozen v0 FINAL command line passes `--scale_handoff`, so this path is
live in every reported row (the 12-scene sweeps and the full 4292). Line 374 also requires at least 10
accepted triangulated points before attempting a hand-off. Lines 375-376 are the explanatory comment; line 377
opens the list of per-track depth ratios and line 378 iterates over the rows of accepted points.

**Alternatives considered.** Decision 2.15 options: implicit scale via the map only (each re-bootstrap
starts a new arbitrary-scale segment); explicit hand-off across re-bootstraps. Decision 3.3 also lists
per-frame normalisation (rejected: "No alternative worth a run").

**Why this choice.** Decision 2.15 (DECIDED 2026-09-05). Sweep 2's reading: ATE was "stuck at ~117-125 mm
for every variant = parity with zero-shot ... set by the 30-58 re-bootstraps per 12 scenes, each starting a
new arbitrary-scale segment that Sim(3) cannot repair". Sweep 3: the hand-off cuts mean ATE 116.8 -> 111.9
and median ATE 117.4 -> 97.7 (RPE-t 7.07 -> 6.93, RPE-r 0.861 -> 0.871); it fired on 12 of 46 re-bootstraps.

### Lines 379-383: scale hand-off, part 2 — depth ratios in the anchor camera

```
379:                                 tid = int(tracks.ids[row])
380:                                 if tid in old_mp:
381:                                     zo = (boot_anchor_T @ np.r_[old_mp[tid], 1.0])[2]; zn = (boot_anchor_T @ np.r_[X[row], 1.0])[2]
382:                                     if zo > 0 and zn > 0: ratios.append(zo / zn)
383:                             if len(ratios) >= 10:
```

**What it does.** For each accepted point, line 379 reads the persistent track id and line 380 checks
whether that same track carried a map point in the previous segment. If so, line 381 transforms both the
old point (old scale) and the newly triangulated point (new, unit-baseline scale) into the anchor camera
with the anchor's world-to-camera pose and takes their depths (`z` component, index 2 of the transformed
homogeneous vector). Because the anchor pose is common to both segments (the re-bootstrap anchor keeps the
last held pose), the ratio of depths of the *same physical point* is the scale factor old/new. Line 382
keeps the ratio only when both depths are positive. Line 383 requires at least 10 such shared tracks
before trusting the estimate.

**Alternatives considered.** A single-point ratio, a least-squares fit of old vs new coordinates, or a full
Sim(3) Umeyama alignment of the shared points would be the standard alternatives; the design doc records
only the median-of-depth-ratios rule with the 10-track floor.

**Why this choice.** Decision 2.15 as written: "match new segment's scale to the old points' depth
through >= 10 shared tracks". The 10-track floor and the median (next group) are the only two free choices
and neither has a separate benchmark; they are part of the single measured variant of sweep 3.

### Lines 384-389: scale hand-off, part 3 — apply the scale and re-triangulate

```
384:                                 sc = float(np.median(ratios))
385:                                 if 0.2 < sc < 5.0:
386:                                     T_rel = rt_to_T(R, sc * np.asarray(t).ravel()); T_w2c = T_rel @ boot_anchor_T
387:                                     X, acc = triangulate_pair(Ka, Kb, boot_anchor_T, T_w2c, a, b); acc &= inl[:, 0].astype(bool)
388:                                     rec["event"] = f"handoff:{sc:.2f}:{len(ratios)}"
389:                             old_mp = {}
```

**What it does.** Line 384 takes the median of the ratios (robust to a few wrong old or new points). Line 385
is a sanity range: a scale factor outside (0.2, 5.0) is treated as unreliable and ignored. Line 386 rescales
the unit translation of the relative pose by `sc` (rotation unchanged) and recomposes the current frame's
world-to-camera pose; line 387 re-triangulates all tracks with the rescaled baseline — the 3D points now live
in the *old* segment's scale — and re-applies the RANSAC-inlier restriction, so `X` and `acc` are consistent
with the new pose. Line 388 logs the event in `diag.csv` as `handoff:<scale>:<n_shared>`. The tag is written
into `rec`, this frame's diagnostics dict, which is appended on either outcome: if the bootstrap is then
rejected at line 390, line 412 nulls `T_w2c` and line 419 appends that same `rec` with `failed=1`, so a
`handoff:` row can sit on a frame that produced no pose and no map. `handoff:` rows in `diag.csv` are
therefore hand-off *attempts*, not accepted hand-offs: any per-scene count read off the event column includes
attempts whose bootstrap was then rejected. Decision 2.15 and the sweep-3 table report "fired on 12 of 46
reboots" for the v0 FINAL row, but neither the doc nor any committed script records how those 12 were counted,
so whether that figure counts attempts or accepted hand-offs cannot be settled from the record. Line 389 sits
inside the `if old_mp and acc.sum() >= 10:` block of line 374, so it clears `old_mp` whenever the hand-off
block is entered (cheirality passed and >= 10 accepted points), regardless of whether the ratio test (line
383) or the range test (line 385) fired and regardless of whether the bootstrap is then accepted at line 390 —
a pair that reaches this block but fails the 30-point floor forfeits the hand-off for the segment. If fewer
than 10 points are accepted, `old_mp` survives to the next frame and the hand-off can still be tried there.

**Alternatives considered.** No range check (accept any median); a tighter or looser window; rescaling the
triangulated points directly instead of re-triangulating; keeping `old_mp` until a bootstrap is actually
accepted. The design doc records only the (0.2, 5) window.

**Why this choice.** Decision 2.15: "0.2 < s < 5". The re-triangulation (rather than multiplying `X` by
`sc`) keeps the acceptance test evaluated against the actual pose that will be stored. Per-scene reading
of sweep 3: the hand-off helps most on short scenes with several re-bootstraps (124.6 -> 64.0 mm, 88.9 ->
71.5 mm), while the 400-700-frame scenes stay at 130-195 mm ("drift + reboots the hand-off could not
bridge"). The placement of line 389 has no benchmark; it is the code as measured in sweep 3.

### Lines 390-394: accept the bootstrap and register the map

```
390:                         if acc.sum() >= args.min_inliers:
391:                             for row in np.where(acc)[0]:
392:                                 tracks.mp[row] = next_mp; mp_xyz[next_mp] = X[row]; mp_outl[next_mp] = 0; next_mp += 1
393:                             booted = True
394:                             # retro-fill frames anchor+1 .. f-1 by PnP against the fresh map (non-causal; --no_retro_fill disables)
```

**What it does.** Line 390 is the final acceptance test: at least `args.min_inliers` (30) triangulated
points must have survived cheirality, reprojection and the E-inlier restriction; otherwise the `else` at
line 411 keeps waiting. Lines 391-392 create the map: each accepted track gets a fresh map-point id
(`next_mp`, monotonically increasing), the world coordinate is stored in `mp_xyz`, and the point's
consecutive-PnP-outlier counter (used by decision 2.14's culling in the tracking branch) starts at 0. Line
393 flips the state to bootstrapped. Line 394 is the comment announcing the retro-fill.

**Alternatives considered.** The floor could be a separate parameter from the PnP floor (decision 3.1's
options 10 / 20 / 30 / ratio); the code instead reuses one number for every floor in this block: the
live-track gate at line 360 (which is not an inlier floor at all — it counts live tracks before any geometry
runs), the cheirality count at line 369, the map size here, the pre-boot re-detection trigger at line 417,
and, past this range, the PnP inlier floor at line 427 and the live-map-point floor that triggers a
re-bootstrap at line 437. The decision list covers only two of those uses — decision 2.9's cheirality floor
("same floor as 3.1") and decision 3.1's PnP floor; the line-360 track gate, the map-size floor here and the
line-417 re-detection trigger are code rules with no entry and no benchmark of their own.

**Why this choice.** Decision 2.9: "Scene where no pair ever passes = total failure, flagged" — the
flagging happens through the per-frame `failed=1` records of the waiting path (lines 414-419). The 30 floor
is decision 3.1's value (numbers above); its use as a map-size floor is a code rule without a benchmark of
its own.

### Lines 395-398: retro-fill, part 1 — which frames and which points

```
395:                             id2mp = {int(i): int(m) for i, m in zip(tracks.ids, tracks.mp) if m >= 0}
396:                             for k, (ids_k, pos_k) in enumerate(pre_hist[1:-1] if not args.no_retro_fill else [], start=anchor_frame + 1):
397:                                 sel = [j for j, i in enumerate(ids_k) if int(i) in id2mp]
398:                                 if len(sel) >= 6:
```

**What it does.** Line 395 builds a lookup from persistent track id to map-point id for every track that
just received a point. Line 396 walks the snapshots of the intermediate frames: `pre_hist[0]` is the anchor
frame and `pre_hist[-1]` is the current frame `f` (appended at line 357), so `pre_hist[1:-1]` are frames
`anchor+1 .. f-1`, and `start=anchor_frame + 1` makes `k` the absolute frame index. With `--no_retro_fill`
the iterable is empty and nothing below runs (strictly causal mode). For each such frame, line 397 selects
the tracks that were alive at frame `k` **and** now carry a map point (track ids are persistent, so a track
that died between `k` and `f` never got a point and is skipped), and line 398 requires at least 6 of them —
the same minimum `pnp()` enforces internally (`if len(X) < 6: return None, None`) for `solvePnPRansac`.

**Alternatives considered.** Strictly causal (waiting frames keep the held anchor pose and stay flagged
failed, i.e. `--no_retro_fill`); the retro-fill as implemented.

**Why this choice.** Decision 2.9: "Frames 1..b-1 are filled retroactively by PnP against the bootstrap map
so every timestep gets an OpenCV pose" — required because `eval_depth_poses.py` numbers frames by position
and every frame must get a pose (dataset facts). **This step is non-causal**: it is honesty-audit item 1,
listed 2026-09-06 and ACCEPTED by the user on 2026-09-07 with a footnote clause in the full-4292 table
(`eval_pipeline/build_opencv_full_table.py`). The retro-fill ablation on the 12 smoke scenes: with
retro-fill 110.4 / 7.5 / 0.858, without 110.6 / 7.7 / 0.881 (ATE mm / RPE-t mm / RPE-r deg) — "+0.2 mm ATE,
+0.2 mm RPE-t, +0.02 deg RPE-r. It does not change any ordering."

### Lines 399-402: retro-fill, part 2 — PnP of each waiting frame

```
399:                                     Xk = np.array([mp_xyz[id2mp[int(ids_k[j])]] for j in sel]); xk = pos_k[sel]
400:                                     Tk, inl_k = pnp(Ks[k], Xk, xk, 4)
401:                                     if Tk is not None:
402:                                         poses_c2w[k] = np.linalg.inv(Tk)
```

**What it does.** Line 399 gathers the `(M,3)` world points (in the just-created segment's scale) and the
`(M,2)` pixel positions those tracks had at frame `k`. Line 400 calls `pnp`, which runs
`cv2.solvePnPRansac(SOLVEPNP_ITERATIVE, reprojectionError=2 px, confidence=0.999, iterationsCount=1000)`
with no initial guess, followed by `solvePnPRefineLM` on the inliers, with frame `k`'s own camera matrix
`Ks[k]`; it returns the world-to-camera 4x4 (or `None`) and the inlier index array. The last argument is
the inlier floor for **this** call: 4, far below the live floor of 30. Line 402 inverts the world-to-camera
result to camera-to-world before storing it — `solvePnP` returns `(rvec, tvec)` mapping world to camera
(plumbing note in the design doc: "invert before writing c2w").

**Alternatives considered.** Use the live floor (30) for retro-filled frames; interpolate poses between the
anchor and frame `f` instead of solving PnP; leave the frames at the anchor pose (`--no_retro_fill`).

**Why this choice.** Decisions 2.4-2.8 fix the PnP call (ITERATIVE + LM, no guess, plain RANSAC, 2 px,
0.999 / 1000; PnP-variant benchmark: all 10 variants within 0.05 deg / 2 mm of each other, zero failures on
2118 pairs). The floor of 4 has no benchmark: it is honesty-audit item 2 ("retro-fill accepts >= 4 inliers
vs 30 live and is marked not-failed"), whose ruling is still pending per the design doc.

### Lines 403-406: retro-fill, part 3 — bookkeeping and the hold fallback

```
403:                                         if k < len(diag):
404:                                             diag[k]["failed"] = 0; diag[k]["inliers"] = int(len(inl_k)); diag[k]["event"] = "retro"
405:                                         n_fail -= 1; continue
406:                                 poses_c2w[k] = poses_c2w[k - 1] if poses_c2w[k - 1] is not None else np.linalg.inv(boot_anchor_T)
```

**What it does.** When the retro PnP succeeded, lines 403-404 rewrite frame `k`'s existing diagnostics
record (frame `k` was already recorded as `failed=1` while waiting; the `k < len(diag)` guard is defensive
since all frames before `f` have records): clear the failed flag, store the PnP inlier count, and tag the
event as `retro`. `diag[k]` is frame `k`'s record only because every frame appends exactly one record; a
`ba_error` event record from `declare_keyframe` (line 334, `--refine ba` only) would shift the indices of
every later frame, which this guard does not detect — irrelevant for v0 (`--refine none`). Line 405
decrements the scene's failed-frame counter accordingly and moves to the next waiting frame. If the retro
PnP was not attempted (fewer than 6 shared tracks) or failed, line 406 holds the previous frame's
camera-to-world pose (which may itself be a retro-filled pose), falling back to the anchor's pose if there
is none; in that case the frame keeps its `failed=1` flag.

**Alternatives considered.** Keep counting retro-filled frames as failed; report retro-filled frames in a
separate counter; interpolate instead of holding.

**Why this choice.** The hold-last rule is decision 3.2 (hold last pose + flag; sweep 1: constant velocity
121.9 / 8.47 / 1.375 vs hold 122.9 / 9.65 / 1.129, "mixed; rotation worse"). The decrement of `n_fail` is
honesty-audit item 12: "failed-frame count excludes retro-filled frames" — pending the user's ruling. In
sweep 1's failure anatomy, 11% of all frames were "waiting for (re)bootstrap parallax", which is the
population this block re-poses.

### Lines 407-410: seed the keyframe list and declare the bootstrap frame a keyframe

```
407:                             kf_T.clear(); kf_obs.clear(); kf_frames.clear()
408:                             kf_T.append(boot_anchor_T.copy()); kf_frames.append(anchor_frame)
409:                             kf_obs.append({int(m): (float(u), float(v)) for m, (u, v) in zip(tracks.mp, tracks.kf_pos) if m >= 0})
410:                             declare_keyframe(f, T_w2c); rec["keyframe"] = 1; rec["booted"] = 1
```

**What it does.** Line 407 discards the keyframe history of any previous segment (poses, observations,
frame indices): after a re-bootstrap the new map has a new scale and, unless the hand-off fired, is not
metrically consistent with the old keyframes. One reporting consequence: `summary.json` writes
`keyframes=len(kf_frames)` at the end of the scene, so on a scene with re-bootstraps that field counts the
keyframes of the last segment only, not the scene total. Line 408 installs the anchor as keyframe 0 with its
world-to-camera pose; line 409 records, for that keyframe, the observations `map-point id -> (u, v)` using
`kf_pos` (the tracks' positions *at the anchor frame*). Line 410 calls `declare_keyframe` (lines 315-336)
for the current frame, which (with a previous keyframe now present) triangulates the tracks that do
**not** yet carry a map point between the anchor keyframe and frame `f` using each view's own `K` and the
same cheirality + 2 px filter, appends `f` as keyframe 1 with its observations, tops up the corners with
`goodFeaturesToTrack` masked at the existing track positions (decision 1.6), and resets every track's
`kf_pos` to its current position so that the keyframe-trigger parallax (decision 2.12) restarts from
here. The record for frame `f` is marked `keyframe=1, booted=1`. Note a subtle consequence of the second
triangulation pass inside `declare_keyframe`: it does not see the E-inlier mask, so a track that was an
E-RANSAC outlier at 0.5 px but reprojects within 2 px in both views can still be admitted there.

**Alternatives considered.** Decision 2.13 (new map points KF-to-KF only vs every frame); decision 3.4
(no refinement vs sliding-window BA vs pose graph) — `declare_keyframe` contains the `--refine ba` hook,
which cannot fire at this point because it requires at least 3 keyframes.

**Why this choice.** Decision 2.13 (DECIDED 2026-09-05): every-frame triangulation was worse (126.2 / 10.40
/ 1.402 vs 122.9 / 9.65 / 1.129). Decision 3.4 (DECIDED 2026-09-05): local BA with window 5 / 10 gave ATE
128.8 / 130.7 vs 122.9 at 5-9x runtime, so v0 has none. Decision 1.6: new corners only at keyframes.

### Lines 411-413: not enough points, or no bootstrap this frame

```
411:                         else:
412:                             T_w2c = None
413:             if T_w2c is None:
```

**What it does.** Lines 411-412 undo the provisional pose computed at line 371 (or 386) when fewer than
`min_inliers` points were accepted at line 390: the frame is treated as if the bootstrap had not been
attempted, and the pipeline keeps tracking and retries at the next frame with the same anchor (but, as
noted at line 389, with `old_mp` already cleared if the hand-off block had been entered). Line 413
collects every way of *not* bootstrapping on this frame — gate not met (line 360), `E` invalid (line 367),
cheirality count too low (line 369), map too small (line 390) — into the single waiting path below.

**Alternatives considered.** Re-anchor immediately on a rejected pair (would discard the accumulated
parallax); accept a smaller map and rely on the keyframe pass to grow it.

**Why this choice.** Decision 2.9: "rejects pure-rotation pairs, then keep tracking and retry at the next
frame". Keeping the anchor lets the parallax keep growing (each further frame adds ~9 mm / 1.3 deg of
median motion, dataset facts), which is what the degenerate pair needs.

### Lines 414-419: the waiting path — hold, flag, re-detect

```
414:                 # not yet bootstrapped: hold the anchor pose, keep tracking; top up corners if tracks run low
415:                 poses_c2w[f] = np.linalg.inv(last_T)
416:                 rec["failed"] = 1; n_fail += 1
417:                 if len(tracks) < args.min_inliers or len(pre_hist) > MAX_TRACK_LEN_BEFORE_BOOT:
418:                     new_bootstrap(f)
419:                 diag.append(rec); continue
```

**What it does.** Line 415 gives the frame the camera-to-world inverse of `last_T`, which during the waiting
state is unchanged since the anchor was set (`new_bootstrap` copies `last_T` into `boot_anchor_T`; `last_T` is
written only at the bootstrap commit, line 420, and in the tracking branch, lines 435 and 456 — never in this
waiting path), i.e. the held anchor pose. Line 416 flags the frame failed and counts it (a later retro-fill
may undo both). Line 417 decides whether to re-anchor: if fewer than `min_inliers` (30) tracks are alive, or
if the wait has exceeded `MAX_TRACK_LEN_BEFORE_BOOT = 400` frames (`pre_hist` holds one snapshot per frame
since the anchor). Line 418 then calls `new_bootstrap(f)`: frame `f` becomes the new anchor with the same held
pose, corners are re-detected (masked at the surviving track positions if `--scale_handoff` keeps them,
otherwise all tracks are dropped first), `pre_hist` is restarted, and the accumulated parallax is reset.
Because the retro-fill (line 396) walks `pre_hist`, restarting it also forfeits the re-posing of every frame
waited so far: those frames keep the held anchor pose and stay flagged failed even if the next anchor
bootstraps immediately. Line 419 stores the record and skips the tracking branch.

**Alternatives considered.** Decision 3.2 options for the pose of an un-posed frame: hold last pose;
constant velocity; drop frame (impossible here: every frame must get a pose). Decision 1.6 options for
refilling tracks: count-floor top-up vs top-up only at (re)detection events.

**Why this choice.** Decision 3.2 (DECIDED 2026-09-05): hold + flag. Decision 1.6 rules out a count-floor
top-up in the booted state ("No floor parameter: the keyframe trigger (2.12) governs refill"). The pre-boot
re-detection floor on line 417 is a code rule outside the decision list; it reuses decision 3.1's number
rather than adding a parameter, and has no benchmark of its own. `MAX_TRACK_LEN_BEFORE_BOOT = 400` is
likewise a code constant with no entry in the decision list and no benchmark; it bounds `pre_hist` memory
and the retro-fill span on a camera that never moves. Two consequences worth knowing: (a) `new_bootstrap`
recomputes `old_mp` from the tracks' *current* map links (line 303), which are all `-1` while waiting, so a
re-anchor during the wait empties `old_mp` and the scale hand-off (decision 2.15) can no longer fire for
that segment; (b) a scene in which no pair ever passes ends with every frame holding the identity/anchor
pose and flagged failed (decision 2.9: "total failure, flagged").

### Lines 420-423: successful bootstrap — commit the pose

```
420:             prev_T, last_T = last_T, T_w2c
421:             poses_c2w[f] = np.linalg.inv(T_w2c)
422:             rec["inliers"] = int((tracks.mp >= 0).sum()); diag.append(rec); continue
423: 
```

**What it does.** Reached only when the bootstrap succeeded on this frame. Line 420 shifts the pose
history: `prev_T` (used by the `--fail_policy cv` constant-velocity option of decision 3.2 in the tracking
branch) takes the old `last_T`, and `last_T` becomes the new world-to-camera pose. Line 421 stores the
camera-to-world pose for the harness. Line 422 replaces the `-1` sentinel with the number of tracks that
now carry a map point — for this frame the pose came from the essential matrix, not from PnP, so this is
a map-size proxy rather than a PnP inlier count, but it makes the frame count towards `median_inliers` in
`summary.json`. The `continue` skips the PnP branch for this frame (the map was built *from* this frame, so
PnP against it would be circular). Line 423 is the blank line that separates the bootstrap branch from the
normal-tracking branch.

**Alternatives considered.** Run PnP on the bootstrap frame anyway as a consistency check; record `-1`
(exclude the frame from the inlier median).

**Why this choice.** Decision 0.4: per-frame inlier count + failed flag as the diagnostics; the c2w
inversion is the plumbing rule ("solvePnP returns world-to-camera (rvec, tvec): invert before writing
c2w"), which applies equally to `recoverPose`.

### Known limitations / honesty notes for this block

- **Retro-fill is non-causal** (lines 394-406): frames `anchor+1 .. f-1` are posed against a map that did not
  exist when they were seen. Honesty-audit item 1, accepted by the user 2026-09-07 with a table footnote;
  `--no_retro_fill` gives the strictly causal variant (110.6 / 7.7 / 0.881 vs 110.4 / 7.5 / 0.858 on the 12
  smoke scenes; no ordering changes).
- **Retro-fill floor is 4 inliers vs 30 live** (line 400) and the re-posed frames are marked not-failed and
  removed from the failed-frame count (lines 404-405): audit items 2 and 12, rulings pending.
- **Whole sequence in memory** (line 343 iterates over the pre-loaded `G`): audit item 3, pending.
- **The bootstrap pose is the bottleneck, not the filters**: with the E pose the admitted map is ~53% bad and
  PnP at a 4-frame gap lands at ~4 deg rotation error, vs 0.56 deg with the GT pose on the same tracks
  (triangulation benchmark); translation direction is ~18-32 deg off at gap 4 depending on threshold, and
  the model arms share that error (two-view diagnostics).
- **Every re-bootstrap starts a new scale segment** that Sim(3) cannot repair; the hand-off (decision 2.15)
  fired on 12 of 46 reboots in sweep 3. Two code paths forfeit it for a segment: a re-anchor while waiting
  (line 418) empties `old_mp`, and line 389 clears `old_mp` as soon as the hand-off block is entered (>= 10
  accepted points), even if the ratio or range test does not fire or the bootstrap is then rejected at
  line 390. The `handoff:` event tag (line 388) is written before that acceptance test, so `diag.csv` records
  attempts, not accepted hand-offs; the provenance of the doc's "(12)" hand-off count is not recorded.
- **`recoverPose` default `distanceThresh = 50`** (line 368): points whose triangulated depth exceeds 50
  baseline units in either camera do not count towards `npass`. Decision 2.9's 25 mm baseline figure was
  written for the original 10 px trigger; at the frozen 5 px trigger (median keyframe gap 2 frames, 9 mm
  median per-step motion) the bootstrap baseline is nearer 13-18 mm, so with the doc's depths (0.3-1.4 m) a
  scene spans roughly 17-110 baseline units (derived, not measured) and everything past ~0.65-0.9 m is
  dropped from the line-369 count — the far half of a typical scene, not just its far edge. Line 372 still
  triangulates those points, so the effect is on the guard (it is stricter than the nominal 30), not on the
  map. The default cannot be raised in the four-return call the code uses: `distanceThresh` belongs to the
  five-return overload (verified on cv2 4.11.0).
- **Lever thresholds** (2 px / 1 px, lines 350-351) were designed with GT-aided inspection (audit item 5)
  and, like every threshold in this file, tuned on the 12 reported test scenes (audit item 7, addressed by
  reporting on all 4292 scenes with the configuration frozen).
- **Decision-doc text vs code**: decision 2.9 still reads "cheirality count >= 20"; the code uses
  `args.min_inliers`, which is 30 in the frozen v0 configuration (argparse default 20).
- **Code constants without a dedicated benchmark**: `MAX_TRACK_LEN_BEFORE_BOOT = 400`, the `(0.2, 5)`
  hand-off window, and three undocumented reuses of `args.min_inliers` — the live-track gate at line 360
  (decision 2.9 specifies only the parallax trigger and the cheirality floor, not a track count), the
  map-size floor at line 390, and the pre-boot re-detection trigger at line 417 (decision 1.6's "no floor
  parameter" refers to the booted state). None of the three appears in the decision list.
- **`summary.json`'s `keyframes` field is per-segment** (line 407 clears `kf_frames` at every
  (re-)bootstrap), so it under-reports the scene's keyframe count whenever `reboots > 0`.
- **`diag[k]` indexing in the retro-fill** (line 403) assumes one record per frame; the `ba_error` record
  of `--refine ba` (line 334) would break that, but v0 runs `--refine none`.
- **Second triangulation pass in `declare_keyframe`** (line 410) does not apply the E-inlier mask, so tracks
  rejected by the 0.5 px E-RANSAC can be admitted to the map if they pass the 2 px reprojection test.
