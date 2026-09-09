## `eval_pipeline/opencv_vo.py` lines 424-492: normal tracking, failure handling, culling, keyframes, and output writing

This block is the steady-state body of the per-frame loop in `run_scene` plus the scene's output stage. Every frame `f` that reaches line 424 has already been LK-tracked from frame `f-1` (persistent tracks, forward-backward checked, static-track lever applied) and the map is live (`booted == True`; the un-booted branch above `continue`s before this point). What happens here, in order: (1) the 2D tracks that carry a map-point id are registered against their 3D points with RANSAC PnP using the per-frame camera matrix `Ks[f]` (decisions 2.1, 2.4-2.8, 0.1b); (2) if PnP fails the frame is given a substitute pose (hold or constant velocity, decision 3.2) and, after enough consecutive failures or when the map has starved, the pipeline re-bootstraps into a new scale segment (3.2b); (3) on success, map points that were PnP outliers three times in a row are culled (2.14); (4) optionally, new points are triangulated every frame (2.13, off in v0); (5) a keyframe is declared when the median track displacement since the last keyframe reaches 5 px (2.12), which is where the map normally grows. After the loop, the scene's poses are written as `camera/%06d.npz` (c2w 4x4 + the K that was used), a per-frame `diag.csv`, and a `summary.json` (decision 0.4). Coordinate convention throughout: `T_w2c` is world-to-camera (OpenCV's `rvec`/`tvec` convention), `poses_c2w` is its inverse, which is what the harness scorer expects.

### Lines 424-427: PnP against the map with the current frame's K

```
424:         # ---- normal tracking: PnP against the map
425:         has = tracks.mp >= 0
426:         X = np.array([mp_xyz[int(m)] for m in tracks.mp[has]]) if has.any() else np.zeros((0, 3))
427:         T_w2c, inl = pnp(Ks[f], X, tracks.pos[has], args.min_inliers)
```

**What it does.** `tracks.mp` is an `(N,)` int64 array holding a map-point id per live track or `-1`; `has` is the boolean mask of tracks that currently have a 3D point. `X` gathers their world coordinates from the `mp_xyz` dict into an `(M,3)` float array in map units (the bootstrap's unit translation fixes the scale, decision 2.10/3.3, or, after a re-bootstrap where the 2.15 hand-off fired (line 386, `0.2 < sc < 5` from a median depth ratio over >= 10 shared tracks), the previous segment's scale), or an empty `(0,3)` array when no track has a point. `tracks.pos[has]` are the matching `(M,2)` pixel positions in the 320x192 cover frame (x right, y down, origin top-left, un-normalised). `pnp` (defined at 181-199) wraps `cv2.solvePnPRansac(SOLVEPNP_ITERATIVE, reprojectionError=2 px, confidence=0.999, iterationsCount=1000, no distortion, no extrinsic guess)` followed by `cv2.solvePnPRefineLM` on the inliers and returns either `(T_w2c 4x4, inlier indices into X)` or `(None, None)` when fewer than 6 points are available, `solvePnPRansac` raises `cv2.error`, RANSAC reports failure (`not ok` or `inl is None`), or the inlier count is below `min_inliers` (a `cv2.error` from the LM refine is swallowed, lines 196-197, and the un-refined RANSAC pose is returned as a success). The camera matrix passed is `Ks[f]`, the per-frame K (with `--focal perframe:<preds_root>` it is the focal CUT3R produced for this very frame, pp at the image centre). `inl` is a 1-D array of row indices into `X` (i.e. into the `has`-selected subset, not into the full track array), which matters for the culling code below.

**Alternatives considered.** Reference frame: frame-to-frame 2D-2D essential matrix, keyframe-to-frame, or a persistent local map (decision 0.2); geometry: 2D-2D E, 2D-3D PnP, or 3D-3D Procrustes (2.1). Solver: ITERATIVE, EPnP, P3P/AP3P, SQPnP, each with or without LM refinement (2.4). Initial guess: none, identity, or previous motion (2.5). Robust estimator: RANSAC, USAC/MAGSAC++, none (2.6). Threshold: OpenCV's default 8 px vs 1-3 px (2.7). Confidence/iterations: 0.99/100 (OpenCV default) vs 0.999/1000 (2.8). Intrinsics: calibrated K, one nominal K, model focal per scene, model focal per frame, self-calibration (0.1b).

**Why this choice.** The map/PnP structure is forced by decision 0.1 (closed loop, triangulated monocular map, no privileged inputs) via 0.2 and 2.1. ITERATIVE+LM (2.4): the PnP-variant benchmark put all 10 variants within 0.05 deg / 2 mm of each other (ITERATIVE gap-1 rot 0.44 deg, trans 3.9 mm; gap-4 1.10 deg, 8.3 mm), so differences are noise. No initial guess and no fallback solver (2.5): zero failures over 2118 pairs in that benchmark, and a guess or fallback would hide instability that the diagnostics must expose. Plain RANSAC (2.6) handled identity-voting outliers at a 0.04 deg penalty (1.10 -> 1.14 deg at gap 4). 2 px (2.7) rather than OpenCV's 8 px default: 8 px is ~2.2 deg at f~210 and admits gripper tracks, 1 px sits at the LK noise floor, and ~65% of consecutive-frame tracks fall within 2 px of GT; sweep 2 re-confirmed it (1 px: more reboots, 58; 3 px: 124.6/8.07/1.084, worse on everything). 0.999/1000 (2.8) costs 1-2 ms per frame and removes the iteration cap as a variable. Per-frame K (0.1b, revised 2026-09-06) is the user's inference-time constraint: frame t may use only what CUT3R produced up to t; with the finetuned checkpoint it reproduces the retroactive per-scene-median numbers (110.4/7.53/0.858 vs 111.9/6.93/0.871). `min_inliers` defaults to 20 in argparse but the frozen v0 uses `--min_inliers 30` (decision 3.1): 10 accepted bad PnP solutions (RPE-r 1.795 in sweep 1), and 30 vs 20 gave 0.861 vs 0.923 deg RPE-r, 7.07 vs 7.34 mm RPE-t, ATE 116.8 vs 120.0 in sweep 2.

### Lines 428-433: PnP failure, substitute pose (hold or constant velocity)

```
428:         if T_w2c is None:
429:             n_fail += 1; rec["failed"] = 1
430:             if args.fail_policy == "cv" and prev_T is not None:
431:                 T_w2c = (last_T @ np.linalg.inv(prev_T)) @ last_T
432:             else:
433:                 T_w2c = last_T.copy()
```

**What it does.** A failed frame increments the scene's failed-frame counter and sets `failed=1` in this frame's diagnostic record `rec` (its `inliers` field stays at the initial `-1` from line 354; a PnP-failed frame is never retro-filled, because the retro-fill only covers frames between a bootstrap anchor and the bootstrap frame, so failed frames drop out of the median-inlier statistic at line 485). `last_T` and `prev_T` are the w2c poses of the previous frame and the one before it. Under `--fail_policy cv`, the w2c motion between them, `M = last_T @ inv(prev_T)` (so that `last_T = M @ prev_T`), is re-applied once: `T_w2c = M @ last_T`, i.e. a constant-velocity extrapolation in SE(3) done consistently in the w2c convention (left-multiplication). Under the default `hold` policy the previous w2c pose is copied verbatim (the camera is assumed not to have moved). `prev_T is None` only before the first bootstrap has succeeded (line 420 sets it on bootstrap), and this branch is reachable only after a bootstrap, so the `prev_T is not None` guard is never the deciding condition here.

**Alternatives considered.** Decision 3.2 lists: hold last pose; constant velocity; drop the frame. Dropping is not viable here because `eval_depth_poses.py` numbers frames by position, so every frame must receive a pose (design doc, dataset facts). A further standard option, retrying with a different solver (SQPnP) or with an extrinsic guess, was explicitly rejected under 2.5.

**Why this choice.** Decision 3.2: hold last pose + flag. Sweep 1: constant velocity scored 121.9 / 8.47 / 1.375 (ATE mm / RPE-t mm / RPE-r deg, failed 31.9%) against hold at 122.9 / 9.65 / 1.129 (failed 37.1%), a mixed result with rotation clearly worse, so hold stays the default and `cv` remains a switch. Note from the metric-floor analysis (144 sampled scenes): a constant-pose trajectory pushed through the same Sim(3) scorer gives RPE-t 0.0079 m / RPE-r 1.21 deg, so held frames land near the no-motion floor rather than being scored as gross errors.

### Lines 434-436: commit the substitute pose and count the failure

```
434:             poses_c2w[f] = np.linalg.inv(T_w2c)
435:             prev_T, last_T = last_T, T_w2c
436:             consec_fail += 1
```

**What it does.** The substitute w2c is inverted to c2w and stored for frame `f` (the scorer wants c2w, and `solvePnP` returns w2c: "invert before writing c2w", design doc plumbing section). The two-frame pose history is shifted, so a held pose feeds forward as "zero velocity" (under `hold`, `prev_T` and `last_T` become equal). `consec_fail` counts PnP failures in a row; it is reset by any success (line 441) or by a reboot (line 439).

**Alternatives considered.** Keeping the pose history frozen during failures (so a later `cv` extrapolation would reuse the last good velocity) is the other standard option; the code instead lets the substitute pose become the new reference, which is the simplest consistent state update.

**Why this choice.** Straightforward consequence of 3.2 ("hold last pose + flag"); no benchmark distinguishes the two bookkeeping variants and the design doc records none.

### Lines 437-440: recovery, re-bootstrap on map starvation or consecutive failures

```
437:             if has.sum() < args.min_inliers or (args.reboot_after_fails and consec_fail >= args.reboot_after_fails):
438:                 # map is dead or persistently inconsistent with the tracks: re-bootstrap from here (new scale segment)
439:                 n_reboot += 1; new_bootstrap(f); rec["event"] = "reboot"; consec_fail = 0
440:             diag.append(rec); continue
```

**What it does.** Two triggers, either of which starts a new bootstrap from the current frame: map starvation, i.e. fewer tracks with a map point than the inlier floor (`has.sum()` is counted before this frame's PnP, and with fewer than `min_inliers` candidate points PnP can never pass, so waiting is pointless); or `reboot_after_fails` consecutive failures (argparse default 0, which disables this trigger because `0 and ...` is falsy; the frozen v0 uses `--reboot_after_fails 3`). `new_bootstrap(f)` (lines 300-313) either drops all tracks or, with `--scale_handoff`, keeps them alive while remembering their old-scale 3D points in `old_mp` (setting `tracks.mp[:] = -1` and resetting `kf_pos`, lines 304-305; the orphaned points stay in `mp_xyz`), then detects fresh corners, marks the pipeline un-booted, and anchors the next bootstrap at `last_T` (the substitute pose just committed). The event is logged on this frame's row, the reboot counter incremented, and the loop moves to the next frame without culling, triangulating, or declaring a keyframe. Each reboot starts a new arbitrary-scale segment unless the 2.15 hand-off later fires on the bootstrap frame; the Sim(3) scorer can absorb one global scale, not per-segment scales (design doc, dataset facts).

**Alternatives considered.** Decision 3.2b: re-bootstrap only at map starvation (the original v0 behaviour, `reboot_after_fails 0`), after 3 consecutive failures, or after 10.

**Why this choice.** Sweep 1's failure anatomy: 37% failed frames = 26% PnP below the inlier floor with a live map (map inconsistent with the tracks), 11% waiting for bootstrap parallax, 0.5% map starved; and a death spiral: after a PnP failure no keyframe is declared, so no new points are triangulated, the map decays for 100+ frames until it starves, then a re-bootstrap starts a new scale segment (23 reboots / 12 scenes). Sweep 2: cull + reboot-after-3 cut failed frames 18.1% -> 14.3% and RPE-r 1.113 -> 1.009 against waiting for starvation; reboot-after-10 was no better (14.1%, 1.020). Hence 3.2b = 3. The cost, recorded in the sweep-2 reading, is that ATE stays at ~117-125 mm for every variant, set by the 30-58 re-bootstraps per 12 scenes that Sim(3) cannot repair; the 2.15 scale hand-off inside `new_bootstrap` is the mitigation (sweep 3: ATE 116.8 -> 111.9 mean, 117 -> 98 median, fired on 12 of 46 reboots).

### Lines 441-442: success bookkeeping

```
441:         consec_fail = 0
442:         rec["inliers"] = int(len(inl))
```

**What it does.** A successful PnP resets the consecutive-failure counter and records the RANSAC inlier count (after the `min_inliers` gate, so it is always >= `min_inliers` here) in the frame's diagnostic record. This is the number aggregated into `median_inliers` in `summary.json`.

**Alternatives considered.** Decision 0.4 lists ATE/RPE only; + failed-frame count; + failed-frame count and median inliers.

**Why this choice.** Decision 0.4: the harness scores ATE / RPE-trans / RPE-rot only; per-frame inlier count and failed flag go to a side CSV and are aggregated per scene, because that separates lost-tracking from drift and "median inliers moves before ATE does".

### Lines 443-448: outlier culling, map the PnP inlier set back to track rows

```
443:         if args.cull == "outlier":
444:             hi = np.where(has)[0]; inl_set = set(hi[inl].tolist())
445:             drop = np.zeros(len(tracks), bool)
446:             for row in hi:
447:                 m = int(tracks.mp[row])
448:                 if row in inl_set:
```

**What it does.** Active only with `--cull outlier` (argparse default is `none`; the frozen v0 passes `outlier`). `hi` is the array of track rows that entered PnP; since `inl` indexes into the `has`-subset (OpenCV returns inlier indices relative to the arrays it was given), `hi[inl]` converts them back to track rows, collected into a set for O(1) membership tests. `drop` is a per-track boolean mask that will mark tracks to delete. The loop then visits every track that carried a map point and reads its map-point id `m`, branching on whether this track was a RANSAC inlier in this frame.

**Alternatives considered.** Decision 2.14: no culling; drop points unseen for N keyframes; drop points after k PnP-outlier hits. (Dead tracks are dropped implicitly anyway: without descriptors a lost track cannot be re-associated, so the "unseen for N keyframes" option has little purchase in this design.)

**Why this choice.** Decision 2.14, from sweep 1: culling after 3 consecutive outlier hits cut PnP failures from 26% to 5.6% of frames, RPE-t 9.65 -> 8.18 mm, median inliers 63 -> 85, and halved runtime (23 -> 13 s/scene), with ATE unchanged (122.9 -> 123.4 mm). Failed-frame share fell 37.1% -> 18.1% while reboots rose 23 -> 30.

### Lines 449-454: outlier culling, hit counter and deletion

```
449:                     mp_outl[m] = 0
450:                 else:
451:                     mp_outl[m] += 1
452:                     if mp_outl[m] >= args.cull_hits:
453:                         drop[row] = True; mp_xyz.pop(m, None)
454:             tracks.keep(~drop)
```

**What it does.** `mp_outl` maps map-point id to its count of consecutive PnP-outlier hits; an inlier resets it to zero, an outlier increments it. On reaching `--cull_hits` (default 3) the point's 3D coordinates are removed from `mp_xyz` and the track row is flagged; `tracks.keep(~drop)` (line 142) then deletes those rows from all track arrays (`pos`, `kf_pos`, `mp`, `ids`). Note that this removes the 2D track as well as its 3D point, so a culled track is not re-triangulated at the next keyframe; it is simply gone (a fresh corner may be detected at that location at the next keyframe, since `detect` (lines 145-150) only masks a `MIN_DIST` disc around live track positions). Ids of culled points can linger in older `kf_obs` dicts; `local_ba` filters on `m in mp_xyz` (line 215), so this is harmless, and BA is off in v0 anyway.

**Alternatives considered.** Same as the previous group (2.14). A cull that keeps the 2D track and only forgets its 3D point would be the milder variant; the design doc records no run of it.

**Why this choice.** The k=3 value is the decided setting of 2.14 (sweep 1 numbers above); `--cull_hits` is exposed as a switch but the design doc records no sweep over it, so 3 is the single tested value.

### Lines 455-457: commit the PnP pose

```
455:         poses_c2w[f] = np.linalg.inv(T_w2c)
456:         prev_T, last_T = last_T, T_w2c
457: 
```

**What it does.** The refined w2c from PnP is inverted to c2w and stored for frame `f`; the two-frame pose history shifts so `last_T` is this frame's w2c (used as the hold/velocity source on a future failure and as the anchor of a future re-bootstrap). Line 457 is a blank separator. The poses are stored in memory and only written to disk after the loop (lines 472-479).

**Alternatives considered.** Decision 3.4 (refinement): none, sliding-window BA (scipy), or a pose graph; the stored pose would be post-refinement in those variants. Decision 3.3 (scale): as given by the map vs per-frame normalisation.

**Why this choice.** 3.4 = none for v0: sweep 1 measured local BA with windows 5 / 10 at ATE 128.8 / 130.7 vs 122.9, RPE-r 1.07 / 1.18 vs 1.13, at 5-9x runtime. 3.3 = as given: Sim(3) scoring absorbs one global scale and segment continuity is 2.15's job; "no alternative worth a run".

### Lines 458-461: every-frame triangulation option (2.13), candidate selection

```
458:         # ---- every-frame triangulation option (2.13)
459:         if args.new_points == "every" and kf_T:
460:             cand = (tracks.mp < 0) & (np.linalg.norm(tracks.pos - tracks.kf_pos, axis=1) >= PARALLAX_PX)
461:             if cand.sum() >= 1:
```

**What it does.** Only with `--new_points every` (argparse default `kf`, which is also the frozen v0) and only once at least one keyframe exists. Candidates are tracks that have no map point yet and whose displacement since their reference position `kf_pos` (position at the last keyframe or at detection, in pixels) is at least `PARALLAX_PX = 5.0` px. This is a per-track parallax gate, stricter than the keyframe path, which triangulates every point-less track once the median displacement passes 5 px. The `cand.sum() >= 1` guard avoids calling the triangulator on an empty set.

**Alternatives considered.** Decision 2.13: triangulate KF-to-KF only, or every frame.

**Why this choice.** KF-to-KF only (2.13). Sweep 1: every-frame scored 126.2 / 10.40 / 1.402 vs KF-only 122.9 / 9.65 / 1.129 (ATE mm / RPE-t mm / RPE-r deg); it lowered failed frames slightly (33.7% vs 37.1%) but worsened all three pose metrics. The switch is retained so the option remains scorable head to head.

### Lines 462-465: every-frame triangulation, admit points

```
462:                 Xn, acc = triangulate_pair(Ks[kf_frames[-1]], Ks[f], kf_T[-1], T_w2c, tracks.kf_pos[cand].astype(np.float64), tracks.pos[cand].astype(np.float64))
463:                 for row, Xi in zip(np.where(cand)[0][acc], Xn[acc]):
464:                     tracks.mp[row] = next_mp; mp_xyz[next_mp] = Xi; mp_outl[next_mp] = 0; next_mp += 1
465: 
```

**What it does.** `triangulate_pair` (lines 166-178) builds `P = K @ T_w2c[:3]` for the last keyframe (its own K, `Ks[kf_frames[-1]]`, and its w2c `kf_T[-1]`) and for the current frame (`Ks[f]`, `T_w2c`), runs `cv2.triangulatePoints` on the pixel correspondences (`kf_pos` at the keyframe, `pos` now), dehomogenises, and returns world points `Xn (C,3)` plus an acceptance mask `acc`: positive depth in both cameras and reprojection error < 2 px (`TRI_REPROJ`) in both views, finite. Each accepted candidate gets a fresh map-point id, its 3D point is stored, and its outlier counter initialised to zero. `kf_pos` is deliberately not reset here, so the baseline for later frames keeps growing until the next keyframe. Line 465 is blank.

**Alternatives considered.** Decision 2.11 acceptance filters: none; cheirality only; + reprojection < 2 px; + parallax >= 1 deg; all three.

**Why this choice.** 2.11: cheirality + reprojection < 2 px, no new parameter (2 px = the PnP threshold). Triangulation benchmark with the GT pose: PnP rot 0.56 deg (cheir+rep) vs 0.63 (cheirality only) vs 0.73 (none); bad-depth 22% vs 32% vs 46%. The 1 deg parallax filter was dropped because at keyframe-scale baselines (~36 mm) it discards half the points and doubles PnP failures. With the E-matrix pose all filters tie at ~4 deg because the two-view pose dominates map error. The per-frame K on each view follows the 0.1b revision.

### Lines 466-471: keyframe trigger (2.12) and diagnostic record

```
466:         # ---- keyframe trigger (2.12): parallax since last keyframe
467:         disp = np.linalg.norm(tracks.pos - tracks.kf_pos, axis=1) if len(tracks) else np.zeros(0)
468:         if len(tracks) and np.median(disp) >= PARALLAX_PX:
469:             declare_keyframe(f, T_w2c); rec["keyframe"] = 1
470:         diag.append(rec)
471: 
```

**What it does.** `disp` is the per-track Euclidean displacement in pixels between the current position and `kf_pos`. Because new tracks are only added at keyframes or bootstraps (decision 1.6) and both reset `kf_pos` (`declare_keyframe` line 336; `new_bootstrap` line 305 or via `Tracks.add`), every live track shares the same reference frame, so the median is a clean "parallax since the last keyframe" statistic. If it reaches `PARALLAX_PX = 5.0` px, `declare_keyframe` (lines 315-336) triangulates every point-less track KF-to-KF with the two keyframes' own Ks and the 2.11 filter, records the keyframe pose and its observations, runs local BA if `--refine ba` (with the single `K = Ks[0]`, line 280, and appends a `ba_error` diagnostic row on an exception, line 334), detects new corners away from live tracks, and resets `kf_pos`; the frame's record gets `keyframe=1`. The guards on `len(tracks)` avoid `np.median` of an empty array. The record is appended to `diag` (one row per frame, in frame order) and line 471 is blank. This is also where the map normally grows, since 2.13 is off; a failed frame never reaches this point (line 440 `continue`s), which is the mechanism behind sweep 1's death spiral.

**Alternatives considered.** Decision 2.12: fixed stride (2/4/8); median parallax since last KF >= P px (5/10/20); tracked-inlier ratio < r (0.9/0.8/0.7). (ORB-SLAM-style H-vs-E model selection is noted under 2.9/2.10 as a v1 option, not for the keyframe rule.)

**Why this choice.** Keyframe-trigger sweep: every family improves monotonically as keyframes get closer, and all adaptive triggers converge to a median gap of 2 at their tightest setting. Parallax 5 px: fired 1828/1942, KF gap median 2 (p10-p90 1-8), n_acc 91, bad 0.56, PnP rot 2.96 deg (p90 10.6), fail 11.9%. Stride 2 was nominally best on gated pairs (2.62 deg, p90 9.9) but 25% of 2-frame windows have < 2 px motion, where a stride keyframe would triangulate at ~zero baseline and still pass cheir+rep; the parallax rule never fires on a paused camera. Sharing P = 5 px with the bootstrap gate (2.9) keeps a single parameter.

### Lines 472-475: output stage, start of the pose-writing loop

```
472:     # ---- write outputs
473:     cam_dir = os.path.join(out_scene_dir, "camera"); os.makedirs(cam_dir, exist_ok=True)
474:     last = np.eye(4)
475:     for f in range(n):
```

**What it does.** After the frame loop, the per-scene `camera/` directory is created under `<out_root>/<label>/preds/<scene>/` (path assembled in `main`, line 525). `last` is initialised to identity, the same pose frame 0 always receives (line 339), and the loop walks all `n` frames in order so that every position gets a file.

**Alternatives considered.** Streaming each pose to disk as it is produced would be the causal-output variant; this code writes everything after the loop.

**Why this choice.** The harness convention (`eval_depth_poses.py` numbers frames by position, so every frame must get a pose) dictates one file per frame. Writing after the loop is a consequence of the whole sequence being processed in memory (honesty-audit item 3, pending the user's ruling).

### Lines 476-479: fill any missing pose and write `camera/%06d.npz`

```
476:         if poses_c2w[f] is None:
477:             poses_c2w[f] = last
478:         last = poses_c2w[f]
479:         np.savez(os.path.join(cam_dir, f"{f:06d}.npz"), pose=poses_c2w[f].astype(np.float32), intrinsics=Ks[f].astype(np.float32))
```

**What it does.** A safety net: every loop path assigns `poses_c2w[f]` (PnP, hold/cv, un-booted hold, or retro-fill), but any frame still `None` inherits the previous frame's c2w (hold semantics). Each file stores `pose` = 4x4 float32 c2w in the map's arbitrary scale and `intrinsics` = the 3x3 float32 camera matrix `Ks[f]` actually used for that frame (fx = fy = the per-frame focal, pp = (W/2, H/2) of the 320x192 cover frame, `K_of` at lines 260-261). The filename `%06d` is the frame's position index. This is the format `pose_sim3_both.py` / `eval_depth_poses.py` read.

**Alternatives considered.** Decision 3.3: writing the map scale as is, or normalising per frame. The design doc's plumbing note describes the `intrinsics` key as "GT K"; the code writes the K the VO used, which under 0.1b (revised) is the model's per-frame focal, not the calibrated one.

**Why this choice.** 3.3 = as given (Sim(3) scoring absorbs one global scale). Writing the K that was used rather than GT K is consistent with the closed-loop rule of 0.1 / 0.1b (no privileged inputs); nothing in the pose scorer reads the intrinsics key (design doc, focal-sensitivity section: "Nothing in eval_depth_poses.py or the pose scorer reads it").

### Lines 480-484: `diag.csv`

```
480:     with open(os.path.join(out_scene_dir, "diag.csv"), "w", newline="") as fh:
481:         keys = ["frame", "n_tracks", "n_mp", "inliers", "failed", "keyframe", "booted", "event"]
482:         w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader()
483:         for r in diag:
484:             w.writerow({k: r.get(k, "") for k in keys})
```

**What it does.** One CSV row per entry of `diag`, in frame order, with fixed columns: `frame`; `n_tracks` (live tracks after LK + lever, sampled at line 354 before any corners a bootstrap or keyframe adds later in the frame); `n_mp` (tracks holding a map point before this frame's PnP); `inliers` = PnP RANSAC inlier count on successful frames; `-1` on failed frames and on un-booted frames unless the retro-fill later overwrote it (line 404, retro PnP accepted at >= 4 inliers, audit item 2); on the bootstrap frame it is the number of live tracks carrying a map point (`(tracks.mp >= 0).sum()`, line 422), which is not `len(mp_xyz)` once a re-bootstrap has orphaned old-segment points; `failed` (0/1); `keyframe` (0/1); `booted` (0/1: the un-booted/booted state at the start of the frame, line 354, except that the frame on which a bootstrap succeeds is written as 1, line 410, so a 0->1 transition in this column marks each (re-)bootstrap frame; frame 0's row is written with `booted=0, keyframe=1`, line 341); `event` (free text: `reboot` on the frame that triggered a re-bootstrap (line 439), `handoff:<scale>:<n>` on the later bootstrap frame where the 2.15 hand-off fired (line 388), `retro` on rows re-posed by the retro-fill (line 404), `ba_error:<type>` (line 334), or empty). Keys not in the list are ignored (`extrasaction="ignore"`); missing keys are written as empty strings. With `--refine ba` a `ba_error` row can be appended out of the per-frame sequence (line 334, before the frame's own row), which is off in v0.

**Alternatives considered.** Decision 0.4: ATE/RPE only; add a failed-frame count; add failed-frame count and median inliers. Putting diagnostics into the harness CSV was rejected in favour of a side file.

**Why this choice.** Decision 0.4 (DECIDED 2026-09-04): diagnostics in a side CSV, not the harness CSV; the failed flag and inlier count are the two columns that separate lost-tracking from drift. Reboots are aggregated from `summary.json` (`reboots` = `n_reboot`, line 486; `build_opencv_vo_table.py` line 45 sums it), not from this CSV. The `handoff:<scale>:<n>` entry in the `event` column is the only place a scale hand-off is recorded, so a hand-off count such as sweep 3's "46 (12)" can only come from this CSV; the design doc does not record the counting script.

### Lines 485-488: per-scene summary

```
485:     inl = [r["inliers"] for r in diag if r.get("inliers", -1) >= 0]
486:     summary = dict(scene=os.path.basename(scene_dir), frames=n, failed_frames=n_fail, reboots=n_reboot,
487:                    keyframes=len(kf_frames), median_inliers=float(np.median(inl)) if inl else 0.0,
488:                    map_points=len(mp_xyz), seconds=round(time.time() - t_start, 1), focal=float(np.median([k[0, 0] for k in Ks])), focal_min=float(min(k[0, 0] for k in Ks)), focal_max=float(max(k[0, 0] for k in Ks)), selfcal_pairs=selfcal_pairs)
```

**What it does.** `inl` collects inlier counts from all rows with `inliers >= 0`: successful PnP frames (>= `min_inliers`), retro-filled frames (retro PnP accepted at >= 4 inliers, line 400), and each bootstrap frame (its count of live tracks carrying a map point, line 422); `ba_error` rows have no `inliers` key and are excluded via the `-1` default. The summary holds: scene name; frame count; `failed_frames` = `n_fail` (PnP failures plus un-booted waiting frames, minus frames later retro-filled); `reboots` = `n_reboot`; `keyframes` = `len(kf_frames)`, which is the number of keyframes in the current scale segment only, because every successful bootstrap clears the keyframe lists (line 407); `median_inliers`; `map_points` = size of `mp_xyz` at the end (all points ever created minus culled ones; points orphaned by a re-bootstrap are not removed); wall-clock `seconds` for the scene including image loading; `focal` = median of the per-frame focals with their min and max; `selfcal_pairs` (None unless `--focal selfcal`).

**Alternatives considered.** Decision 0.4's three reporting levels; the doc's v0 FINAL section lists `failed_frames, reboots, keyframes, median_inliers` as the summary fields the sweeps consumed.

**Why this choice.** Decision 0.4: per-scene failed-frame count and median inliers are the two aggregates that "separate lost-tracking from drift". `build_opencv_vo_table.py` (lines 40-45) reads `frames`, `failed_frames` and `reboots` from these files to fill the failed-% and reboot columns of the sweep tables. The focal fields were added for the focal-source checks (0.1b re-check: focal ranges 205-221 px for the finetuned per-frame focal, 207-819 px zero-shot).

### Lines 489-492: persist and return

```
489:     json.dump(summary, open(os.path.join(out_scene_dir, "summary.json"), "w"))
490:     return summary
491: 
492: 
```

**What it does.** The summary is written as `summary.json` next to `camera/` and `diag.csv`, and returned to `main`, which prints it as one JSON line per scene (line 534). The existence of `summary.json` is also `main`'s resume marker (line 526: a scene with one is skipped), so it is written last, after all poses are on disk. Lines 491-492 are the blank lines separating `run_scene` from `main`.

**Alternatives considered.** None recorded; the file-per-scene layout mirrors the model preds layout (`<label>/preds/<scene>/`) that the table builders (`build_opencv_vo_table.py`, `build_opencv_full_table.py`) read.

**Why this choice.** Plumbing per the v0 FINAL section: "Diagnostics: <label>/preds/<scene>/diag.csv (per frame) and summary.json (failed_frames, reboots, keyframes, median_inliers)".

### Known limitations / honesty notes for this block

- **Held frames are scored.** Under the `hold` policy a failed frame is written with the previous pose and enters ATE/RPE like any other; the diagnostics (0.4) are the only place the failure is visible. The metric-floor analysis (144 sampled scenes) shows a no-motion trajectory scores RPE-t 0.0079 m / RPE-r 1.21 deg, so held frames sit near the floor, which is one reason RPE-trans cannot separate arms.
- **Failed-frame count excludes retro-filled frames** (audit item 12): `n_fail` is incremented here and in the un-booted branch, then decremented by the retro-fill (line 405, outside this range). The retro-fill itself is non-causal (audit item 1, ACCEPTED 2026-09-07 with a footnote); its ablation cost is +0.2 mm ATE, +0.2 mm RPE-t, +0.02 deg RPE-r.
- **No keyframe, cull, or triangulation on a failed frame.** The `continue` at line 440 skips all of them, which is the death-spiral mechanism documented in sweep 1; 3.2b (reboot after 3) and 2.14 (cull) are the measured mitigations, not a fix. Every reboot starts a new scale segment that Sim(3) cannot repair (unless the 2.15 hand-off bridges it, 12 of 46 reboots in sweep 3); ATE stays at ~117-125 mm for every variant (parity with zero-shot CUT3R at 118.0 on the 12 smoke scenes; on the 144-scene metric-floor sample the constant-velocity trajectory scores 0.125 m ATE and the OpenCV rows sit at that floor (0.123 zero-shot+OpenCV, 0.109 finetuned+OpenCV), while the full-4292 OpenCV rows are 0.120 / 0.104 m).
- **Culling removes the 2D track, not just the 3D point** (line 454), so a wrongly culled track cannot be re-triangulated at the next keyframe.
- **`summary.keyframes` counts only the last scale segment** (keyframe lists are cleared at every bootstrap success, line 407); the per-frame `keyframe` column in `diag.csv` is the whole-scene record.
- **`inliers` on a bootstrap frame is not a PnP inlier count**: it is the number of live tracks carrying a map point (`(tracks.mp >= 0).sum()`, line 422, outside this range), which differs from `len(mp_xyz)` once a re-bootstrap has orphaned old-segment points, and it enters `median_inliers`. Retro-filled rows also enter `median_inliers` with counts accepted at >= 4 inliers (audit item 2), below the live floor of 30.
- **`intrinsics` in the npz is the VO's K, not GT K**, contrary to the wording of the doc's plumbing note; nothing in the pose scorer reads it.
- **Whole scene in memory, outputs written at the end** (audit item 3, ruling pending); the pose estimates themselves are causal (per-frame K from frame t only, decision 0.1b revised), apart from the accepted retro-fill.
- **argparse defaults are not the frozen v0.** `--cull none`, `--min_inliers 20`, `--reboot_after_fails 0` are the defaults; the frozen configuration passes `--cull outlier --reboot_after_fails 3 --min_inliers 30 --scale_handoff --focal perframe:<preds_root>`.
- **All thresholds in this block (2 px, 5 px, 3 hits, 30 inliers) were tuned on the 12 reported smoke scenes** (audit item 7), addressed by reporting on all 4292 scenes with the configuration frozen.
