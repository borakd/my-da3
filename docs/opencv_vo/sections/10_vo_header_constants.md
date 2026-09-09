## `eval_pipeline/opencv_vo.py`, lines 1-45: module docstring, imports, thread cap, decided constants

This block is the front matter of the OpenCV monocular visual-odometry control arm ("v0", frozen 2026-09-05). It declares what the file is (a classical, GPU-free pose baseline that runs next to the CUT3R arms on the 4292-scene DROID wrist harness), pulls in the seven stdlib modules plus NumPy / OpenCV / PIL, pins OpenCV to one thread, and then hard-codes every design decision from `OPENCV_VO_DESIGN.md` that is *closed* as a module-level constant. Nothing here executes geometry; the constants are consumed later by `detect`, `lk_step`, `triangulate_pair`, `pnp`, `selfcal_focal`, the optional `local_ba` (which reuses `PNP_THRESH`) and the per-scene loop in `run_scene`. The rule the docstring states, and the constants enforce, is the doc's own: "Every decided choice is hard-coded; every still-open decision is a CLI switch so the options can be scored head to head." Every pixel threshold below is expressed in the eval loader's 320x192 cover frame (decision 1.1), not in the native 320x180 PNGs.

### Lines 1-2: shebang and docstring opener

```text
1: #!/usr/bin/env python
2: """OpenCV monocular visual-odometry control arm (v0) for the DROID wrist harness.
```

**What it does.** Makes the file directly executable under whatever `python` is on `PATH`. The design doc's "Plumbing" section says to run it as a Slurm job rather than on the login node, which enforces a 300 s per-process CPU cap; the checked-in smoke launchers (`eval_pipeline/opencv_vo_probes/pf.sbatch`, `sweep.sbatch`) request a 48-core `acc` node for it. The whole docstring (lines 2-21) becomes the `--help` description via `description=__doc__` with `RawDescriptionHelpFormatter` (line 494); this first line is its one-line summary.

**Alternatives considered.** Decision 0.1 lists three arms: GT-depth oracle, model-depth-anchored, and a triangulated monocular map. Decision 0.2 lists frame-to-frame, keyframe-to-frame, and local-map (PnP against a persistent map) as reference-frame choices.

**Why this choice.** Decision 0.1 (DECIDED 2026-09-02, user rule): "closed loop, plain image inputs only, no privileged info; scale treated as a separate issue, pose accuracy is the target" -> triangulated monocular map. Decision 0.2 is then forced: every frame registers to map points by PnP. The word "monocular" in the title is that decision; "control arm" is the doc's stated purpose (a baseline "that owes nothing to the model").

### Lines 3-5: design of record

```text
3: 
4: Design of record: OPENCV_VO_DESIGN.md (repo root). Every decided choice is hard-coded;
5: every still-open decision is a CLI switch so the options can be scored head to head.
```

**What it does.** Line 3 is a blank separator. Lines 4-5 state the contract that shapes the rest of this block: anything with status DECIDED in the design doc becomes a bare constant (lines 37-43), anything that was still PENDING when the file was written becomes an `argparse` switch (line 20 and `main`). This is why the constants have no CLI override except `PNP_THRESH` (see line 41).

**Alternatives considered.** A single YAML/JSON config for all parameters; or all parameters as CLI flags. Neither is in the doc.

**Why this choice.** The doc's purpose statement: "a bare-bones classical pose baseline ... with as few free parameters as possible, so it can serve as a control". Hard-coding decided values keeps the sweep surface equal to the open-decision set: sweep 1 in the doc ("v0 pipeline sweep 1") varies those switches one at a time from the defaults; sweeps 2-3 stack the winners cumulatively (cull + lever + rb3 + min inliers 30, then + scale hand-off), and sweep 2 also re-checks the decided `PNP_THRESH` via `--pnp_thresh` (1 px / 3 px rows).

### Lines 6-9: pipeline summary, inputs and camera matrix

```text
6: 
7: Pipeline (per scene, offline, images only + the model's own predicted focal):
8:   frames  : the eval loader's 320x192 cover frames (load_images_cover), grayscale
9:   K       : fx=fy = per-scene median of the model's predicted focal (preds camera/*.npz), pp = image centre
```

**What it does.** Line 6 blank. Line 7 names the operating regime: per scene, offline (the whole frame list is loaded before tracking starts, line 257), no depth input, and the only non-image input is the focal that CUT3R itself produced. Line 8: frames are not the native 320x180 PNGs but the geometry of `load_images_cover` (uniform 1.067x scale + centre crop to 320x192, reproduced by `load_gray_cover`) then converted to grayscale, so the classical arm and the model arm see the same pixels and K. Line 9: the pinhole matrix is square-pixel (`fx=fy`), zero skew, no distortion, with the principal point at `(W/2, H/2)` of the cover frame; the focal is read from the model's `preds/<scene>/camera/*.npz` (`intrinsics[0,0]`) rather than from the GT `cam/*.npz`.

**Alternatives considered.** Decision 1.1 (preprocessing): gray only; 2x upsample; CLAHE; undistort. Decision 0.1b (intrinsics source): calibrated K from `cam/*.npz`; one nominal dataset K; model-estimated focal; self-calibration. Within the model-focal option: per-scene median (retroactive), per-frame focal of frame t, running median to t.

**Why this choice.** Decision 1.1 (DECIDED 2026-09-04): "grayscale conversion only ... All pixel thresholds are in these units"; CLAHE etc. are deferred v1 single-variable experiments. Decision 0.1b (DECIDED 2026-09-02): model output is not privileged under the closed-loop rule, and "classical and model arms then share identical K"; facts recorded: model focal ~211 px at 320x192 vs calibrated ~190 px at 320x180 (ratio 1.11), within-scene jitter 0.7%. The doc then shows the focal source is immaterial in the full pipeline (finetuned-model focal 111.9/6.93/0.871, fixed nominal 203 px 109.1/7.04/0.869, calibrated 116.0/6.91/0.885 ATE mm / RPE-t mm / RPE-r deg on the 12 smoke scenes) and that a 2x focal error costs only ~10 mm ATE and ~0.1 deg RPE-r "because map and PnP share the same (wrong) K and Sim(3) absorbs the scale". **Line 9 is stale:** 0.1b was REVISED 2026-09-06 to the inference-time per-frame focal (`--focal perframe:<preds_root>`, frame t uses only the focal CUT3R produced for frame t; every geometric step carries a per-frame K). The per-scene median described on line 9 "is no longer the reported configuration"; the causal per-frame variant reproduces it (110.4/7.53/0.858 vs 111.9/6.93/0.871). The zero-shot checkpoint's focal (229-570 px, scene-inconsistent) and OpenCV self-calibration (per-scene 132-510 px against a true ~203) were both measured unusable.

### Lines 10-11: frontend

```text
10:   frontend: Shi-Tomasi (1000, 0.01, 5) + pyramidal LK (OpenCV defaults) + forward-backward check < 1 px;
11:             persistent tracks, new corners only at keyframes (existing tracks masked out)
```

**What it does.** Summarises decisions 1.3, 1.5 and 1.6. Corners come from `cv2.goodFeaturesToTrack` (Shi-Tomasi min-eigenvalue score) with the three constants of line 37; correspondences from `cv2.calcOpticalFlowPyrLK` at its default window/pyramid, accepted only if the backward flow lands within `FB_THRESH` of the origin (line 38, strict `<`); tracks persist frame to frame and are only replenished at keyframes, with a mask that blanks a `MIN_DIST`-radius disc around every live track (`cv2.circle`, line 148) so new corners do not duplicate old ones.

**Alternatives considered.** Detector (1.3): FAST, ORB, SIFT/AKAZE, fixed grid. Spread (1.4): grid bucketing, top-up in empty regions. Correspondence (1.5): LK without FB check, ORB / SIFT + ratio test, DIS or Farneback dense flow at corners or on a grid. Lifetime (1.6): re-detect every frame; persistent + top-up on a count floor.

**Why this choice.** Correspondence benchmark (decision 1.5; 12 smoke scenes, 4243 frames, Slurm 45403560), pooled medians: LK+FB 166 correspondences / 88 within 2 px of GT flow at gap 1 with 0.88 deg E-rotation error, vs ORB+ratio 138/77 and 1.11 deg, SIFT+ratio 94/48 and 0.95 deg (19 ms), Farneback 187/92 but collapsing to 187/12 at gap 4 (3.66 deg). DIS on an 8-px grid was the best rotation estimator (0.75 deg gap 1, 1.88 deg gap 4) via 5x more points and is recorded as the v1 runner-up; LK+FB was kept because it "has the best per-correspondence precision at gap 4 (in2 0.35 vs 0.29)" and yields corner-anchored persistent tracks the map needs. Decision 1.6: "Every map point is born at a keyframe and triangulated at the next; 95% per-frame survival means the count decays slowly between keyframes" (probe: median LK survival 0.95 per frame, ~15% over a 15-frame gap in fast segments, so tracking must be frame-to-frame).

### Lines 12-14: bootstrap

```text
12:   bootstrap: frame 0 vs first frame whose median track displacement >= 5 px; findEssentialMat(RANSAC 0.5 px,
13:             0.999) + recoverPose; cheirality count >= min_inliers else keep waiting; triangulate (cheirality +
14:             reprojection < 2 px); frames 1..b-1 filled retroactively by PnP against the bootstrap map
```

**What it does.** Decisions 2.9, 2.10, 2.11. The map is seeded from a two-view pair: the anchor frame (frame 0, or the frame at which a re-bootstrap started) against the first later frame `b` whose *median* track displacement since the anchor is `>= PARALLAX_PX` (line 39). The attempt is only made if at least `--min_inliers` tracks are alive (line 360, ANDed with the parallax test). `cv2.findEssentialMat` with RANSAC at `E_THRESH`/`E_CONF` (line 40) gives `E` and an inlier mask; `cv2.recoverPose` decomposes `E` into the four `(R, t)` candidates and returns the one with the most points in front of both cameras, together with that count (`npass`) and a unit-norm `t` (this unit translation is what fixes the map's arbitrary scale, decision 3.3). If `npass < --min_inliers` (line 369) the pair is rejected as degenerate (pure rotation / no parallax) and the loop keeps tracking and retries at the next frame. Accepted pairs are triangulated with the cheirality + `TRI_REPROJ` filter (line 42); after triangulation the pair is also rejected, and the frame treated as not booted, unless at least `--min_inliers` points pass both that filter and the E inlier mask (lines 373, 390). Frames strictly between the anchor and `b` had no map when they were visited; they are re-posed afterwards by PnP against the fresh map ("retro-fill").

**Alternatives considered.** Pair (2.9): frames 0/1; H-vs-E model selection (ORB-SLAM style); tracked-ratio trigger. Geometry (2.10): homography; H-vs-E selection; RANSAC threshold 2 / 1 / 0.5 px. Acceptance (2.11): none; cheirality only; + reprojection; + parallax >= 1 deg; all three. Retro-fill: `--no_retro_fill` (strictly causal) exists.

**Why this choice.** 2.9: probe scene stationary for frames 0-15 (GT rot 0.03 deg, |t| = 0.2 mm) and 9 mm steps make frame 0/1 degenerate, so the bootstrap must be parallax-gated; the threshold dropped from 10 to 5 px on 2026-09-05 to share `P` with the keyframe trigger 2.12. 2.10: numbers in the line-40 group. 2.11: numbers in the line-42 group. Retro-fill is **non-causal** and is honesty-audit item 1; it was ACCEPTED by the user on 2026-09-07 with a footnote clause in the full-4292 table, and the ablation shows it changes nothing material (with 110.4 / 7.5 / 0.858 vs without 110.6 / 7.7 / 0.881).

### Line 15: tracking

```text
15:   tracking: solvePnPRansac(ITERATIVE, 2 px, 0.999, 1000) + solvePnPRefineLM; no initial guess; no fallback solver
```

**What it does.** Decisions 2.4-2.8. Every post-bootstrap frame is registered by `cv2.solvePnPRansac` with `flags=SOLVEPNP_ITERATIVE` (OpenCV's closed-form initialisation followed by Levenberg-Marquardt on the reprojection error), a `reprojectionError` of `PNP_THRESH` px, `confidence=PNP_CONF`, `iterationsCount=PNP_ITERS`, then `cv2.solvePnPRefineLM` on the inlier set; the wrapper `pnp` (line 181) refuses to solve with fewer than 6 map points and catches `cv2.error` from either call. `useExtrinsicGuess` is left at its default `False`, and a failed solve is recorded as a failed frame rather than retried with another solver. The returned `rvec, tvec` are **world-to-camera** (the doc's plumbing note: "solvePnP returns world-to-camera (rvec, tvec): invert before writing c2w"); `run_scene` keeps w2c (`last_T`, `prev_T`, `kf_T`) for all geometry and stores `np.linalg.inv(T_w2c)` into `poses_c2w[f]` at each frame (lines 402, 415, 421, 434, 455); the output loop (lines 475-479) only forward-fills gaps.

**Alternatives considered.** Variant (2.4): EPnP, P3P/AP3P, SQPnP, each with/without LM. Guess (2.5): identity; previous motion. Estimator (2.6): USAC/MAGSAC++; none. Threshold (2.7) and confidence/iterations (2.8): see line 41.

**Why this choice.** PnP-variant benchmark (decision 2.4; Slurm 45407881; oracle 3D from GT depth): all ten variants within 0.05 deg / 2 mm of each other; ITERATIVE 0.44 deg (p90 1.72) / 3.9 mm at gap 1, 1.10 deg / 8.3 mm at gap 4; SQPnP nominally best by 0.04 deg (0.37 at gap 1); P3P/AP3P need +LM to match on translation (10.2 -> 8.7 mm at gap 4). The doc notes LM refine "is a no-op after ITERATIVE-RANSAC but kept so the refine step exists for the P3P/SQPnP swaps". 2.5: zero failures over 2118 pairs with no guess; "a guess or a fallback would hide instability the diagnostics must expose". 2.6: plain RANSAC absorbed identity-voting outliers at a 0.04 deg cost (1.10 -> 1.14 deg at gap 4).

### Lines 16-17: keyframes

```text
16:   keyframe: median track displacement since last keyframe >= 5 px -> triangulate new points (KF-to-KF),
17:             detect new corners
```

**What it does.** Decisions 2.12, 2.13, 1.6. After a successful PnP, if the median displacement of live tracks relative to their position at the last keyframe (`Tracks.kf_pos`) is `>= PARALLAX_PX`, the frame becomes a keyframe (`declare_keyframe`, line 315): tracks that have no map point yet are triangulated between the previous keyframe's pose and this one (each keyframe with its own per-frame K, line 324), the keyframe's observations are registered in `kf_obs`, new corners are detected with existing tracks masked out, and `kf_pos` is reset.

**Alternatives considered.** Trigger (2.12): fixed stride 2/4/8; parallax 5/10/20 px; tracked-inlier ratio 0.9/0.8/0.7. New points (2.13): triangulate every frame.

**Why this choice.** Keyframe-trigger sweep (Slurm 45443937): every family improves monotonically as keyframes get closer; stride 2 is nominally best (2.62 deg PnP rotation at keyframe+4) but 25% of 2-frame windows have < 2 px motion, where a stride keyframe "would triangulate at ~zero baseline and pass cheir+rep"; parallax 5 px gives median gap 2 (p10-p90 1-8), 2.96 deg (p90 10.6), and "never fires on a paused camera"; 10 px 3.23, 20 px 4.25. 2.13 sweep 1: every-frame 126.2 / 10.40 / 1.402 vs KF-only 122.9 / 9.65 / 1.129 (ATE mm / RPE-t mm / RPE-r deg).

### Lines 18-21: outputs, open switches, docstring close

```text
18:   output  : camera/%06d.npz with pose (c2w, 4x4) + intrinsics, for every frame; diag.csv per scene
19: 
20: Open-decision switches: --new_points, --cull, --min_inliers, --fail_policy, --refine, --lever (see argparse).
21: """
```

**What it does.** Line 18: per scene the arm writes `<out_root>/<label>/preds/<scene>/camera/%06d.npz` with keys `pose` (camera-to-world 4x4, float32) and `intrinsics` (the 3x3 K actually used for that frame, float32), one file per frame without exception (line 479), plus a `diag.csv` (per-frame `frame, n_tracks, n_mp, inliers, failed, keyframe, booted, event`) and a `summary.json`. Line 19 blank. Line 20 lists the `argparse` switches for decisions that were open when the file was written (2.13, 2.14, 3.1, 3.2, 3.4, 1.2). Line 21 closes the docstring.

**Alternatives considered.** Decision 0.4 (failure reporting): ATE/RPE only; + failed-frame count; + failed-frame count and median inliers. Output plumbing was dictated by the harness.

**Why this choice.** "eval_depth_poses.py numbers frames by position, so every frame must get a pose" (dataset facts) is why the docstring stresses "for every frame". Decision 0.4 (DECIDED 2026-09-04): the harness scores ATE / RPE-trans / RPE-rot only, with AbsRel and d1 inherited from `augfull_lr1e5` depth via symlink; diagnostics go in a side CSV "not the harness CSV" because "median inliers moves before ATE does". **Line 20 is incomplete and its defaults are pre-sweep:** `main` also exposes `--pnp_thresh` (2.7), `--focal` (0.1b), `--no_retro_fill`, `--scale_handoff` (2.15), `--reboot_after_fails` (3.2b), `--cull_hits`, `--ba_window`, `--no_lever`, `--depth_link` and sharding (`--shard_id/--num_shards`), and every decision named on line 20 has since been DECIDED (2.13 KF-only, 2.14 cull after 3 hits, 3.1 = 30, 3.2 hold + reboot after 3 fails, 3.4 none, 1.2 lever ON). The v0 FINAL command in the doc therefore passes `--cull outlier --reboot_after_fails 3 --min_inliers 30 --scale_handoff --focal perframe:...` explicitly; running with argparse defaults does *not* reproduce the frozen configuration.

### Lines 22-25: stdlib imports, part 1

```text
22: import argparse
23: import csv
24: import glob
25: import json
```

**What it does.** `argparse` builds the CLI in `main` (with `RawDescriptionHelpFormatter` so the docstring above prints verbatim). `csv.DictWriter` writes `diag.csv` with `extrasaction="ignore"`. `glob` enumerates `dense/rgb/*.png` per scene and `camera/*.npz` in the model preds (sorted, so frame order is lexical `%06d` order). `json` dumps the per-scene `summary.json` and prints one JSON line per scene to stdout.

**Alternatives considered.** None recorded; these are plain stdlib choices.

**Why this choice.** Decision 0.4 requires a side CSV plus per-scene aggregates (failed-frame count, median inliers), which is what `csv` and `json` serve.

### Lines 26-29: stdlib imports, part 2

```text
26: import os
27: import sys
28: import time
29: 
```

**What it does.** `os` for path joins, `makedirs`, `symlink` (the `--depth_link` inherited-depth symlink) and the `summary.json` existence check (line 526) that makes re-runs skip finished scenes. `sys` for `sys.stderr` in the per-scene `ERROR` line (a scene exception never aborts the shard). `time` for the `seconds` field of `summary.json`. Line 29 is the blank between stdlib and third-party imports.

**Alternatives considered.** None recorded.

**Why this choice.** The `seconds` field feeds the sweep tables' s/scene column (23 s for v0 defaults, 13 s with culling in sweep 1); the doc's full-harness "OpenCV 30 ms/frame on one CPU core" figure is stated without a derivation. The skip-if-summary-exists behaviour is what makes a killed shard resumable.

### Lines 30-33: third-party imports

```text
30: import numpy as np
31: import cv2
32: import PIL.Image
33: 
```

**What it does.** NumPy holds all state (`Tracks` is a structure of arrays; map points are `dict[int, ndarray(3)]`; poses are 4x4 `float64`). `cv2` supplies the geometric primitives used later (`goodFeaturesToTrack`, `calcOpticalFlowPyrLK`, `findEssentialMat`, `recoverPose`, `triangulatePoints`, `solvePnPRansac`, `solvePnPRefineLM`, `Rodrigues`) plus `circle` for the detection mask (line 148) and `cvtColor` for grayscale (line 58). `PIL.Image` is used for image *loading and resizing* (`load_gray_cover`) rather than `cv2.imread` / `cv2.resize`, because `load_images_cover` in the eval loader (`src/CUT3R/src/dust3r/utils/image.py`) resizes with PIL and centre-crops; `load_gray_cover` reproduces the same resize/crop arithmetic and interpolation (BICUBIC when upscaling, LANCZOS when downscaling; image.py line 185), so pixels and K agree. The only difference is that the eval loader also applies `exif_transpose` (image.py line 175), a no-op on these PNGs. The grayscale conversion is then `cv2.cvtColor(..., COLOR_RGB2GRAY)` on the RGB array (note PIL gives RGB, not OpenCV's BGR default; the flag matches). Line 33 blank.

**Alternatives considered.** `cv2.imread` + `cv2.resize(INTER_AREA/INTER_CUBIC)`; loading the model's own preprocessed tensors. Neither is in the doc.

**Why this choice.** Decision 1.1: "on the eval loader's 320x192 frames (load_images_cover: uniform 1.067x scale + center crop). All pixel thresholds are in these units." Decision 0.1b: "Run the classical arm on the model's own preprocessed 320x192 frames (same loader) so K and pixels agree."

### Lines 34-35: OpenCV thread cap

```text
34: cv2.setNumThreads(1)
35: 
```

**What it does.** Sets OpenCV's process-global worker-thread count for its `parallel_for_` backend to 1, so `goodFeaturesToTrack`, `calcOpticalFlowPyrLK` and the RANSAC loops run single-threaded in this process. It does not affect NumPy/BLAS threads, and it must be called in every process (it is set at import time here, before any work). Line 35 blank.

**Alternatives considered.** Leave OpenCV at its default (all cores); `cv2.setNumThreads(0)`; environment-variable control (whether `OMP_NUM_THREADS` reaches OpenCV's pool depends on the parallel backend the wheel was built with; the launchers export `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for BLAS regardless, `opencv_vo_probes/pf.sbatch` line 13, and the in-code call makes the cap independent of the build).

**Why this choice.** This is **not a numbered decision** in the design doc. The recorded facts it serves: the doc records the full-4292 rows on 48-core allocations (Slurm 45527508 / 45529336, ~16 min per row + scoring) and quotes the per-core cost as "OpenCV 30 ms/frame on one CPU core"; the checked-in smoke launchers (`eval_pipeline/opencv_vo_probes/pf.sbatch` line 28, `sweep.sbatch` line 32) run 12 backgrounded shards per label on a 48-core node with `OMP_NUM_THREADS=1` exported (the full-4292 launcher is not in the repo). Parallelism is therefore at the process level (`--shard_id/--num_shards`, one scene at a time per shard), and per-process OpenCV threads would only oversubscribe; the single thread also makes the `seconds` diagnostic a clean per-core number. The doc's "Plumbing" note that the login node enforces a 300 s per-process CPU cap is a further reason to keep every process's CPU accounting predictable (the arm must be run as a Slurm job either way).

### Lines 36-37: corner detector parameters

```text
36: # ----------------------------------------------------------------------------- decided constants
37: MAX_CORNERS, QUALITY, MIN_DIST = 1000, 0.01, 5
```

**What it does.** Line 36 opens the decided-constants block. Line 37 sets the three required positional arguments of `cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance, mask=...)`: return at most 1000 corners; reject any corner whose Shi-Tomasi score (minimum eigenvalue of the structure tensor) is below `0.01` x the best score *in that image* (a relative threshold, so it self-adapts to exposure and texture); enforce a Euclidean separation of at least 5 px between kept corners (greedy non-max suppression in quality order). The function returns an `(N,1,2) float32` array of `(x, y)` corner locations at integer pixel positions (no `cornerSubPix` refinement is applied anywhere in the file; sub-pixel positions arise only after LK), sorted by decreasing quality, or `None` when nothing passes (both call sites, lines 69-70 and 149-150, guard for `None`). `MIN_DIST` is also reused as the radius of the exclusion disc drawn around every live track in `detect`'s mask (mask value 0 = do not detect here), which is what implements "existing tracks masked out" from line 11.

**Alternatives considered.** Decision 1.3: FAST; ORB; SIFT/AKAZE; fixed grid. Decision 1.4 (spread): grid bucketing; top-up in empty regions.

**Why this choice.** Decision 1.3 (DECIDED 2026-09-04): "(probe values; cap never reached, ~140-300 corners/frame)"; "Designed for LK; relative quality threshold adapts across exposure; no wasted descriptor." Probe (scene RAIL+80edfcb1+2023-07-14-14h-28m-45s, 128 frames): only ~215 corners/frame (min 139), i.e. the 1000 cap is never binding, and the scenes are low-texture. Decision 1.4 (DECIDED 2026-09-04): no bucketing because "minDistance=5 already spreads corners at this resolution (probe)"; "Zero parameters; no probe evidence of a coverage problem"; bucketing is the first v1 frontend experiment if long-scene drift dominates. The three values are probe picks, not swept (decision 1.3: "probe values"); `0.01` is the value used in OpenCV's own documentation example, not a library default (`qualityLevel` is a required argument).

### Line 38: forward-backward threshold

```text
38: FB_THRESH = 1.0
```

**What it does.** Pixel bound for the forward-backward consistency check in `lk_step` (and `selfcal_focal`): after `calcOpticalFlowPyrLK(g0, g1, p0)` gives `p1` and status `st`, the code runs LK again from `g1` back to `g0` on `p1` to get `p0b`, and keeps a track only if both status flags are 1 and `||p0 - p0b||_2 < 1.0` px (strict). `calcOpticalFlowPyrLK` here runs at its defaults, which the doc records as a 21 px window and 3 pyramid levels; its `status` output is 1 when the flow was found, and its `err` output is ignored. `lk_step` additionally drops tracks whose new position lies outside `[0, W) x [0, H)` (line 161).

**Alternatives considered.** Decision 1.5: LK with no check (benchmarked); descriptor matching with a ratio test; dense flow. The value 1.0 px itself was not swept.

**Why this choice.** Decision 1.5 (DECIDED 2026-09-04): "sparse LK at OpenCV defaults (21 px window, 3 levels) + forward-backward check at 1 px." Benchmark: LK+FB 166/88 vs LK-no-check 183/91 at gap 1 (0.88 vs 0.85 deg E-rotation), 103/27 vs 167/31 at gap 4 (2.55 vs 2.46 deg); the check is retained for per-correspondence precision ("in2 0.35 vs 0.29" at gap 4), not for pair-level rotation error, which it does not improve. Probe: consecutive-frame LK with FB < 1 px has median survival 0.95. Decision 2.7 characterises 1 px as "the LK noise floor", which is the doc's only justification of the magnitude.

### Line 39: parallax threshold

```text
39: PARALLAX_PX = 5.0
```

**What it does.** One number, three consumers. (a) Bootstrap gate (2.9): a not-yet-booted scene attempts `findEssentialMat` only when `len(tracks) >= --min_inliers` **and** `median(||pos - kf_pos||) >= 5.0` (line 360), where `kf_pos` is each track's position at the anchor frame. (b) Keyframe trigger (2.12): after a successful PnP, `median(||pos - kf_pos||) >= 5.0` (positions relative to the last keyframe, line 468) declares a keyframe. (c) Per-track candidate gate for the `--new_points every` option (2.13, decided OFF; line 460, each track's own displacement, not the median) and the pair gate in `selfcal_focal` (line 76, median). The statistic in (a) and (b) is a median over *all* live tracks, so on a frame where the camera is paused it stays near zero regardless of a few moving tracks; units are pixels in the 320x192 cover frame.

**Alternatives considered.** 2.9: 10 px (the original value); H-vs-E model selection; tracked-ratio trigger. 2.12: fixed stride 2/4/8; parallax 10/20 px; tracked-inlier ratio 0.9/0.8/0.7.

**Why this choice.** Decision 2.12 (DECIDED 2026-09-05) sets the value and 2.9 inherits it ("shares P with 2.9; 2.9's P drops 10 -> 5"). Sweep numbers (Slurm 45443937, PnP test at keyframe+4): parallax 5 px -> median KF gap 2 (1-8), PnP rot 2.96 deg (p90 10.6), 11.9% fails; 10 px -> gap 4, 3.23 deg; 20 px -> gap 7, 4.25 deg; stride 2 -> 2.62 deg but 25% of 2-frame windows have < 2 px motion. The E-diagnostics finding that motivates tight keyframes: "real tracks get WORSE with gap because chained LK drift grows with chain length. Keyframes must be close (gap 2-4), not far." For the old bootstrap value the doc notes "10 px ~ 3 typical frames ~ 25 mm baseline at 0.5 m".

### Line 40: essential-matrix RANSAC settings

```text
40: E_THRESH, E_CONF = 0.5, 0.999
```

**What it does.** `E_THRESH` (px) and `E_CONF` are passed to `cv2.findEssentialMat(points1, points2, cameraMatrix, method=cv2.RANSAC, prob=E_CONF, threshold=...)` in the bootstrap (lines 365-366). `threshold` is the maximum point-to-epipolar-line distance for a RANSAC inlier, *in the units of the point coordinates*; `prob` is the confidence used for adaptive termination. Because the two views may have different K (per-frame focal, 0.1b revised), `run_scene` normalises each view's points by its own K (`(x - pp) / f`, line 363), passes `I3` as the camera matrix, and rescales the threshold to normalised units as `E_THRESH / (0.5 * (f_a + f_b))`. The function returns `E` (3x3; the guard `E.shape == (3, 3)` on line 367 matters because OpenCV can return a vertically stacked `3k x 3` array when the 5-point solver yields several candidates) and an `(N,1) uint8` inlier mask, which is then handed to `cv2.recoverPose(E, an, bn, I3, mask=...)`. `recoverPose` returns `(npass, R, t, mask)`: `R, t` satisfy `x2 = R x1 + t`, i.e. they map the *anchor* camera frame into the *current* camera frame, so the code composes `T_w2c = [R|t] @ T_anchor_w2c` (line 371); `t` has unit norm; `npass` is the number of inliers passing the cheirality test.

**Alternatives considered.** Decision 2.10: RANSAC threshold 2 / 1 / 0.5 px; homography; H-vs-E model selection. Also measured: MAGSAC 2 px; 2-view Sampson refinement of E(2 px); relative static lever.

**Why this choice.** Decision 2.10 (revised 2026-09-05): "2 -> 0.5 px cuts two-view rot error 2.06 -> 1.23 deg and t-dir 31.6 -> 18.1 deg at gap 4; map-level PnP-at-+4 improves 4.19 -> 3.02 deg (gap 4), 3.58 -> 2.62 (gap 2). Only the sub-pixel minority of chained LK tracks is geometrically clean." Two-view diagnostics table (Slurm 45443683 / 45443704), rot / tdir in deg: 2 px 2.06/31.6, 1 px 1.64/24.3, 0.5 px 1.23/18.1, MAGSAC 2 px 1.34/22.9 at gap 4; at gap 8, 0.5 px 2.88/23.5 vs MAGSAC 2.87/24.2; at gap 16, 0.5 px 8.11/36.8 is *worse* than 1 px (7.32) and MAGSAC (7.57). Map-quality table (Slurm 45443823): bad-depth fraction / PnP rot at gap 2 is 0.61/3.58 (2 px), 0.56/3.29 (1 px), 0.54/2.62 (0.5 px). E at 0.5 px beats the finetuned CUT3R relative rotation at gaps 4 and 8 (1.23 vs 2.22; 2.88 vs 3.76) and loses at gap 16. `E_CONF = 0.999` is the same confidence as 2.8 chooses for PnP; the doc gives no separate sweep of it. H-vs-E selection is deferred to v1 "if bootstrap rejections cluster on planar scenes".

### Line 41: PnP RANSAC settings

```text
41: PNP_THRESH, PNP_CONF, PNP_ITERS = 2.0, 0.999, 1000   # PNP_THRESH may be overridden by --pnp_thresh
```

**What it does.** The three RANSAC arguments of `cv2.solvePnPRansac(objectPoints, imagePoints, K, None, flags=SOLVEPNP_ITERATIVE, reprojectionError=PNP_THRESH, confidence=PNP_CONF, iterationsCount=PNP_ITERS)` (lines 186-188): a point is an inlier if its reprojection error is within `2.0` px; the sampler stops when the confidence that an outlier-free minimal sample has been drawn reaches `0.999`, or after `1000` iterations, whichever first (so 1000 is a cap, not a fixed count). `distCoeffs=None` because the cover frames are treated as undistorted (dataset facts: "no distortion"). Return is `(ok, rvec, tvec, inliers)` with `inliers` an `(M,1) int32` index array (or `None`) and `rvec, tvec` world-to-camera. The trailing comment is the one exception to "decided = hard-coded": `main` does `global PNP_THRESH; PNP_THRESH = args.pnp_thresh` (lines 518-519, default 2.0), which is how the 1 px / 3 px rows of sweep 2 were run. `PNP_THRESH` is also reused as the Huber `f_scale` of the optional local BA (line 244).

**Alternatives considered.** Decision 2.7: OpenCV default 8 px; 1-3 px. Decision 2.8: OpenCV default 0.99 / 100. Decision 2.6: USAC / MAGSAC++.

**Why this choice.** Decision 2.7 (DECIDED 2026-09-04): "8 px = 2.2 deg at f~210, admits gripper tracks; 1 px is at the LK noise floor; ~65% of consecutive-frame tracks fall within 2 px of GT (corr benchmark); same unit as the benchmarks' 'correct' definition." Re-checked in sweep 2 (Slurm 45453768) on the cull+lever+rb3 configuration: 2 px 120.0 / 7.34 / 0.923 (42 reboots), 3 px 124.6 / 8.07 / 1.084, 1 px 120.4 / 7.71 / 0.866 (58 reboots); doc verdict "2.7 stays 2 px (1 px: more reboots; 3 px: worse everything)". Note that the 1 px row had the *better* RPE-rot; it lost on ATE, RPE-trans and reboot count. Decision 2.8 (DECIDED 2026-09-04): "0.999 / 1000 ... 1-2 ms per frame; removes the iteration cap as a variable on high-outlier frames" (PnP benchmark timing: ITERATIVE 1.0 ms, +LM 2.3 ms). Decision 2.6: plain RANSAC because it "handled identity-voting outliers with a 0.04 deg penalty in the PnP benchmark; three stateable numbers".

### Line 42: triangulation reprojection cap

```text
42: TRI_REPROJ = 2.0
```

**What it does.** Acceptance bound used by `triangulate_pair` (line 166): after `cv2.triangulatePoints(P0, P1, a.T, b.T)` (DLT on `P = K [R|t]`, returning homogeneous `4 x N`, dehomogenised with a `1e-12` guard), a point is kept only if its depth is positive in *both* cameras (cheirality), its reprojection error is `< 2.0` px in *both* views (strict), and its coordinates are finite (line 177). `a, b` are pixel coordinates in the cover frame; `P0, P1` are built from each view's own K and w2c pose, so the check is in pixels of each view. It is a separate constant from `PNP_THRESH` even though the two share the value 2 px, so `--pnp_thresh` does not move it.

**Alternatives considered.** Decision 2.11: no filter; cheirality only; cheirality + reprojection < 2 px; cheirality + parallax >= 1 deg; all three.

**Why this choice.** Decision 2.11 (DECIDED 2026-09-05): "cheirality in both cameras + reprojection error < 2 px in both keyframes. No new parameter (2 px = PnP threshold)." Triangulation-acceptance benchmark (Slurm 45443315; triplets i, i+4, i+8; 1373 parallax-gated triplets), with the GT pose isolating the filter: PnP rotation 0.56 deg (p90 1.83) for cheir+reproj vs 0.63 cheirality-only vs 0.73 none; bad-depth fraction 0.22 vs 0.32 vs 0.46. With the E-matrix pose (pipeline reality) all filters tie at ~4 deg (3.98 / 3.96 / 4.18) because "the two-view pose dominates map error". The parallax >= 1 deg filter was excluded because at keyframe-scale baselines (~36 mm) it "discards half the points and raises PnP failures 2x" (E column: 47 accepted / 329 fails vs 67 / 163). The doc's KEY FINDING attached to this benchmark: the bootstrap two-view pose, not the acceptance filter, is the map arm's bottleneck (admitted map ~53% bad under the E pose).

### Lines 43-45: pre-bootstrap track-length cap and trailing blanks

```text
43: MAX_TRACK_LEN_BEFORE_BOOT = 400
44: 
45: 
```

**What it does.** In the not-yet-booted branch of `run_scene`, every frame appends `(ids, pos)` to `pre_hist` (one entry per frame since the anchor, anchor included; lines 313 and 357). If a frame fails to bootstrap and either the live-track count has dropped below `--min_inliers` or `len(pre_hist) > 400` (line 417), `new_bootstrap(f)` re-anchors: without `--scale_handoff` the track set is wiped and re-detected at frame `f` (line 308); with it (the frozen v0 config) live tracks are kept, their map ids cleared and `kf_pos` reset (lines 302-305), and new corners are topped up with existing tracks masked out (line 309). Either way the held pose becomes the anchor pose and `pre_hist` restarts (lines 311-313). So a scene that never reaches 5 px of median parallax within ~400 frames of an anchor is re-anchored rather than left tracking one decaying corner set forever; all frames in the interval keep the held anchor pose and stay flagged failed. Because retro-fill (line 14) re-poses frames from `pre_hist`, this constant also bounds how many frames one bootstrap can retro-fill. Lines 44-45 are trailing blank lines before `load_gray_cover`.

**Alternatives considered.** No cap (track until starvation); a cap tied to sequence length. None recorded in the doc.

**Why this choice.** This constant has **no numbered decision** and does not appear in `OPENCV_VO_DESIGN.md`; the only doc statement on the case is decision 2.9's "Scene where no pair ever passes = total failure, flagged" and the dataset fact that sequences run from 57 to ~1700 frames. Treat 400 as an unswept implementation safeguard.

### Known limitations / honesty notes for this block

- **Docstring line 9 is stale.** It describes the retroactive per-scene-median focal; decision 0.1b was revised on 2026-09-06 to the inference-time per-frame focal (`--focal perframe:<preds_root>`), which is the reported configuration. The doc's honesty-audit item 13 also records "stale per-scene-median rows on disk".
- **Docstring line 20 is incomplete, and argparse defaults are not the frozen v0.** `--cull none`, `--min_inliers 20`, `--reboot_after_fails 0`, no `--scale_handoff` are the pre-sweep defaults; the v0 FINAL requires the explicit flags in the doc's "v0 FINAL configuration" block. Only the lever default was flipped to ON (user decision, 2026-09-05).
- **Retro-fill (line 14) is non-causal** (honesty-audit item 1; accepted 2026-09-07 with a footnote). Related item 2: retro-filled frames are accepted at >= 4 PnP inliers (line 400) vs 30 live and are marked not-failed, so the failed-frame count excludes them (item 12). Ablation: +0.2 mm ATE, +0.2 mm RPE-t, +0.02 deg RPE-r without it.
- **Whole sequence in memory** (audit item 3): line 7's "offline" means all grayscale frames and `pre_hist` are held for the scene.
- **`MAX_TRACK_LEN_BEFORE_BOOT = 400` is undocumented** in the design doc and was never swept.
- **All thresholds on lines 37-42 were tuned on the 12 reported smoke scenes** (audit item 7), which are test scenes; the doc addresses this by reporting on all 4292 scenes with the configuration frozen. The design benchmarks that justify them are GT-scored on those scenes (item 8); the lever thresholds, though outside this block, were designed with GT-aided inspection (item 5).
- **`E_THRESH = 0.5 px` wins only at short gaps.** At gap 16 it is worse than 1 px and MAGSAC (8.11 vs 7.32 / 7.57 deg); the choice is tied to the tight keyframe policy (median gap 2).
- **`FB_THRESH = 1.0` and `E_CONF = 0.999` were not swept**; the doc's justification of FB is per-point precision, not pair-level rotation error, which the no-check variant matched.
- **Decision 2.9's text says the cheirality guard is 20 "(same floor as 3.1)"**, but 3.1 later moved to 30 and the code ties all three bootstrap gates (live-track count, `npass`, accepted-point count; lines 360, 369, 390) to `--min_inliers`, so the frozen v0 bootstrap floor is 30, not 20.
- **`TRI_REPROJ` is not coupled to `--pnp_thresh`** even though 2.11 calls it "no new parameter (2 px = PnP threshold)"; the 1 px / 3 px PnP rows in sweep 2 leave triangulation at 2 px under the current code.
- **The doc's "Plumbing" section says the output `intrinsics` key is "(GT K)"**; the code writes the K actually used (model / per-frame / nominal focal, pp at centre), which is the correct closed-loop behaviour but not what that sentence says.
- **`cv2.setNumThreads(1)` has no decision record**; its rationale above is reconstructed from the doc's runtime facts (30 ms/frame on one core; 48-core rows) and the checked-in smoke launchers (12 shards per label, `OMP_NUM_THREADS=1`), not from a stated ruling; the full-4292 launcher and its shard count are not in the repo.
