## `eval_pipeline/opencv_vo_probes/pnp_bench.py`, lines 1-95 (whole file): PnP-variant benchmark for decision 2.4

This 95-line script is the offline probe that settled decision 2.4 (which PnP solver the OpenCV visual-odometry control arm registers frames with) and supplied the evidence quoted under 2.5, 2.6 and 2.8 in `OPENCV_VO_DESIGN.md`. It is not part of the inference pipeline (`eval_pipeline/opencv_vo.py`); it is a one-shot experiment that isolates the *solver* from every other pipeline stage by giving PnP oracle 3D points. For each frame pair (i, i+gap) in one scene it runs the same Shi-Tomasi + pyramidal-LK + forward-backward frontend the arm uses (decisions 1.3, 1.5), back-projects the tracked corners of frame i through GT depth and the calibrated K into metric 3D points, and then asks ten `solvePnPRansac` variants (ITERATIVE, EPnP, P3P, AP3P, SQPnP, each with and without a `solvePnPRefineLM` pass on the inlier set) to recover the pose of frame j. Each estimate is scored against the GT relative pose, under a `clean` condition and an `outliers` condition that injects identity-voting outliers of the kind an image-static gripper produces. The Slurm driver `pnp_bench.sbatch` (job 45407881, 2026-09-04) runs the script once per scene on the 12 smoke scenes of decision 0.3 with gaps 1 and 4; the per-scene JSONs under `/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/pnp_bench/` were pooled into the "PnP-variant benchmark" table of the design doc. Result of the whole exercise: all ten variants are within 0.05 deg / 2 mm of each other, zero failures on 2118 pairs, so the arm keeps the standard `ITERATIVE` + LM choice (decision 2.4) and, more importantly, the benchmark exposes the *absolute floor* of per-frame PnP on this footage (0.44 deg / 3.9 mm median even with oracle 3D) that the map-based arm cannot beat.

Conventions used throughout the file, stated once: images are the native 320x180 PNGs with the calibrated K, as the design doc's benchmark headers state for both this probe and the correspondence benchmark ("native 320x180, calibrated K"); decision 1.1 fixed the *pipeline*, not the probes, on the eval loader's 320x192 cover frames, so the pixel thresholds here are in native-pixel units. `c2w` matrices are camera-to-world 4x4. All `solvePnP*` calls return the *object-to-camera* transform (`rvec`, `tvec`), which here means camera-i-to-camera-j because the "object" points are expressed in camera i; the design doc's plumbing section records the general pitfall ("solvePnP returns world-to-camera: invert before writing c2w"), but this script never writes c2w, so no inversion is needed.

### Lines 1-2: shebang and docstring opening

```text
1: #!/usr/bin/env python
2: """PnP-variant benchmark for decision 2.4.
```

**What it does.** Standard `env` shebang so the script runs under whatever `python` the activated conda env provides (the driver activates `cuteanything`). Line 2 opens the module docstring and names the single decision the probe exists for.

**Alternatives considered.** None; a docstring is the only documentation this probe carries, so its correctness matters (see honesty notes on the "clean" claim in lines 9-13).

**Why this choice.** Decision 2.4's option list is "ITERATIVE; EPnP; P3P/AP3P; SQPnP; each +/- solvePnPRefineLM", exactly what the file enumerates.

### Lines 3-7: docstring, frontend and the oracle-3D setup

```text
3: 
4: Same LK+FB frontend as decision 1.5. For each pair (i, i+gap): 3D points come from
5: GT depth on frame i (native 320x180, calibrated K; GT used for the 3D points and
6: the reference pose only). Each solver estimates frame j's pose from the 2D-3D
7: correspondences and is scored against the GT relative pose.
```

**What it does.** States the experimental frame: correspondences are produced by the same sparse-LK + forward-backward frontend that decision 1.5 selected (implemented in `lk()`, lines 38-44); 3D points are *oracle* points from GT depth on the source frame i, back-projected with the calibrated per-scene K at native 320x180 resolution; the only other use of GT is the reference relative pose used for scoring. The blank line 3 separates title from body.

**Alternatives considered.** The 3D source could have been (a) the arm's own triangulated map (decision 0.1's actual choice for the arm), (b) model-predicted depth, or (c) GT depth. For a *solver* benchmark only (c) isolates the solver: with (a) or (b) the map error would dominate, which the later triangulation benchmark (decision 2.11) confirms (E-pose map: ~4 deg PnP rotation error vs 0.56 deg with a GT-pose map on the same tracks).

**Why this choice.** Decision 0.1 rules out GT depth for the *arm* (closed loop, plain image inputs only) but explicitly keeps GT as a reference for the design benchmarks; the design doc's benchmark header says "3D points from GT depth on the source frame (oracle 3D); scored vs GT relative pose". This makes the probe an upper bound on PnP: the doc's floor statement ("even with ORACLE 3D points, per-frame PnP rotation error is 0.44 deg median ... translation error 3.9 mm") rests on exactly this setup. Honesty-audit item 8 ("design benchmarks GT-scored on test scenes") applies to this whole file.

### Lines 8-13: docstring, the two conditions

```text
8: 
9: Two conditions:
10:   clean    : only tracks with valid GT depth (no identity-voting outliers)
11:   outliers : additionally, static tracks (disp < 1 px) with NO valid depth are
12:              assigned a plausible wrong 3D point (median scene depth), so they act
13:              as the identity-voting outliers a real map will contain.
```

**What it does.** Defines the two evaluation conditions realised at lines 76-81. `clean` keeps only tracks whose source pixel has a valid GT depth and is not in the outlier mask. `outliers` additionally admits tracks that moved less than 1 px between i and j *and* have no valid depth, assigning them the median valid depth of the frame as a deliberately wrong 3D point. Because those tracks did not move in the image, their 2D-3D pair is consistent with the identity motion, so they "vote" for zero camera motion inside RANSAC; that is the failure mode the design doc identifies for the image-static gripper ("its corners vote for zero motion in PnP, so it must be masked, not 'filtered by RANSAC'", dataset-facts section and decision 1.2).

**Alternatives considered.** Injecting synthetic random outliers (uniform wrong 3D or wrong 2D) is the textbook robustness test but does not model the *structured* outlier the DROID wrist camera produces; a real gripper mask (GT `outlier_mask`, Otsu variance mask, fixed polygon) was evaluated for decision 1.2 and none was adopted; using only real gripper tracks with their real (wrong-for-motion) GT depth is what happens implicitly in `clean` wherever the gripper has depth (see honesty notes).

**Why this choice.** Decision 2.6 needed "three stateable numbers" on how plain RANSAC copes with identity-voting outliers; the outcome, quoted in the doc, is that the outliers cost only 1.10 -> 1.14 deg at gap 4 "when the map points are right". The 1 px static threshold is the same value later frozen as the `reject_static_tracks` lever's displacement floor (decision 1.2: "drop tracks with displacement < 1 px"); honesty-audit item 5 records that the lever thresholds were designed with GT-aided inspection.

### Lines 14-17: docstring, variants and fixed RANSAC parameters

```text
14: 
15: Variants: solvePnPRansac with flag in {ITERATIVE, EPNP, P3P, AP3P, SQPNP},
16: each with and without solvePnPRefineLM on the inlier set.
17: Fixed: reprojectionError=2 px, confidence=0.999, iterationsCount=1000.
```

**What it does.** Lists the 5 x 2 = 10 variants built at lines 46-48 and the three RANSAC parameters hard-wired at lines 53-54. `reprojectionError` is the inlier threshold in pixels (OpenCV default 8 px per decision 2.7), `confidence` the RANSAC stopping confidence and `iterationsCount` the iteration cap (OpenCV defaults 0.99 / 100 per decision 2.8).

**Alternatives considered.** Decision 2.7 lists "OpenCV default 8 px; 1-3 px"; decision 2.8 lists "0.99/100 (default); 0.999/1000". The solver options are those of decision 2.4; USAC/MAGSAC++ (`cv2.USAC_*` flags for `solvePnPRansac` via `UsacParams`) were the alternatives for decision 2.6.

**Why this choice.** 2 px is decision 2.7: "8 px = 2.2 deg at f~210, admits gripper tracks; 1 px is at the LK noise floor; ~65% of consecutive-frame tracks fall within 2 px of GT (corr benchmark); same unit as the benchmarks' 'correct' definition." 0.999 / 1000 is decision 2.8: "1-2 ms per frame; removes the iteration cap as a variable on high-outlier frames." Note these are fixed *inputs* to this benchmark, not swept here. 2.7 rests on the correspondence benchmark (~65% of consecutive-frame tracks within 2 px) and was re-confirmed by pipeline sweep 2 (1 px: more reboots; 3 px: worse everything). 2.8 is a stated policy whose only quoted cost ("1-2 ms per frame") is this probe's `ms` column (lines 85, 83-86 block); the three RANSAC arguments are fixed inputs here, not variables. Plain RANSAC rather than MAGSAC++ is decision 2.6, for which this probe supplies the evidence.

### Lines 18-20: docstring, usage and close

```text
18: 
19: Usage: pnp_bench.py <scene_dir> <out_json> [gap ...]
20: """
```

**What it does.** Command line: a scene directory, an output JSON path, and optional frame gaps. Parsed at lines 25-26.

**Alternatives considered.** An `argparse` interface with switches for the RANSAC parameters would have made 2.7/2.8 sweepable from here; the probe deliberately hard-codes them.

**Why this choice.** Single-variable discipline (the design doc's stated approach for every v1 experiment): the probe varies only the solver flag and the LM toggle, so nothing else is exposed on the command line.

### Lines 21-23: imports and single-threaded OpenCV

```text
21: import sys, os, glob, json, time
22: import numpy as np, cv2
23: cv2.setNumThreads(1)
```

**What it does.** Standard library, NumPy and OpenCV (OpenCV 4.11.0 as installed in the `cuteanything` env the driver activates, checked 2026-09-08; the version is not recorded in the design doc). `cv2.setNumThreads(1)` disables OpenCV's internal parallelism for this process, so the wall-clock `ms` numbers recorded at line 85 are single-core solver times and the 12 per-scene processes the driver launches concurrently do not oversubscribe the node (the driver also exports `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, `pnp_bench.sbatch` line 13).

**Alternatives considered.** Default threading (OpenCV picks the core count) gives faster but non-reproducible per-call timings under 12-way concurrency.

**Why this choice.** The design doc's `ms` column (1.0 / 2.3 ms for ITERATIVE / +LM, 0.3 / 0.7 for P3P, etc.) is meaningful only as a single-thread cost; the repo-wide practice for the OpenCV arm is "single-thread cv2" (the arm's full-4292 runtime is quoted as "30 ms/frame on one CPU core").

### Lines 24-26: argument parsing

```text
24: 
25: scene, out = sys.argv[1], sys.argv[2]
26: gaps = [int(g) for g in sys.argv[3:]] or [1, 4]
```

**What it does.** Positional arguments: scene directory and output JSON. Remaining arguments are integer gaps; if none are given the default is `[1, 4]`. The driver passes `1 4` explicitly (`pnp_bench.sbatch` line 17), matching the two gap columns of the doc's table.

**Alternatives considered.** Larger gaps (8, 16) are what the later two-view diagnostics use; for PnP they are less relevant because the arm registers every frame against a map, and the keyframe trigger of decision 2.12 keeps keyframe gaps at a median of 2.

**Why this choice.** Gap 1 is the arm's per-frame registration step; gap 4 approximates a keyframe-scale baseline (decision 2.11 describes keyframe-scale baselines as ~36 mm; the doc's dataset facts give the per-step median as 9 mm / 1.3 deg). Both are user-visible in the table as "gap1" and "gap4".

### Lines 27-29: image loading

```text
27: d = os.path.join(scene, "dense")
28: fs = sorted(glob.glob(os.path.join(d, "rgb", "*.png"))); n = len(fs)
29: G = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in fs]; H, W = G[0].shape
```

**What it does.** Scene layout is `<scene>/dense/{rgb,cam,depth,outlier_mask,sky_mask}` (dataset-facts section). All RGB PNGs are globbed and sorted by filename, then loaded straight to 8-bit grayscale (`IMREAD_GRAYSCALE` applies OpenCV's BGR-to-gray weights on decode). The sibling `cam`, `depth` and `outlier_mask` files are addressed by a zero-padded `%06d` index (lines 30, 33, 34), so the script relies on the rgb filenames sorting into the same frame order. `H, W = 180, 320` for the native frames; `n` is the sequence length (57 to ~1700 frames per the doc). The whole scene is held in memory.

**Alternatives considered.** Decision 1.1 lists "gray only; 2x upsample; CLAHE; undistort".

**Why this choice.** Decision 1.1: grayscale only ("Gray is mandatory for the detector/LK; everything else is a v1 single-variable experiment"). Note the difference from the frozen pipeline: this probe reads the *native* 320x180 PNGs, whereas the arm runs on the eval loader's 320x192 cover frames (uniform 1.067x scale + centre crop). Holding the whole sequence in memory is honesty-audit item 3 for the arm; for an offline probe it is immaterial.

### Lines 30-32: calibrated intrinsics and GT poses

```text
30: cams = [np.load(os.path.join(d, "cam", "%06d.npz" % i)) for i in range(n)]
31: K = cams[0]["intrinsic"].astype(np.float64); Kinv = np.linalg.inv(K)
32: c2w = [c["pose"] for c in cams]
```

**What it does.** Loads every per-frame `cam/%06d.npz`. `K` is the 3x3 calibrated pinhole matrix (`intrinsic` key; fx ~192 px at 320x180, no distortion per the dataset facts) taken from **frame 0 only** and used for every frame of the scene; `Kinv` is precomputed for back-projection at line 79. `c2w` is the list of GT 4x4 camera-to-world poses (`pose` key). Pixel convention: OpenCV's `goodFeaturesToTrack`/`calcOpticalFlowPyrLK` report coordinates with (0,0) at the centre of the top-left pixel; the script assumes (without checking) that the npz K uses the same convention and applies no half-pixel offset.

**Alternatives considered.** Decision 0.1b lists "calibrated K from cam/*.npz; one nominal dataset K; model-estimated focal from preds; self-calibration", and its 2026-09-06 revision makes the *pipeline* carry a per-frame K.

**Why this choice.** For an oracle benchmark the calibrated K is the right reference (the doc's corr and PnP benchmark headers both say "native 320x180, calibrated K"); decision 0.1b's closed-loop rule applies to the arm, not to GT-scored probes. Using frame 0's K for the whole scene is a simplification the file makes silently; the doc records that the intrinsics are stored per frame, and that in the full pipeline the "focal source is immaterial" (finetuned focal 111.9/6.93/0.871 vs calibrated 116.0/6.91/0.885 ATE mm / RPE-t mm / RPE-r deg).

### Lines 33-34: GT depth and outlier mask

```text
33: D = [np.load(os.path.join(d, "depth", "%06d.npy" % i)) for i in range(n)]
34: M = [cv2.imread(os.path.join(d, "outlier_mask", "%06d.png" % i), cv2.IMREAD_GRAYSCALE) > 0 for i in range(n)]
```

**What it does.** `D[i]` is the GT depth map of frame i (float32; 0 = invalid; ~88% valid; 0.3-1.4 m typical per the dataset facts). Whether the stored value is z-depth along the optical axis or ray length is not recorded in the design doc; line 79 back-projects it as z-depth (see that block). `M[i]` is a boolean (H, W) array, True where the dataset's `outlier_mask` PNG is nonzero, i.e. True = *masked out*. The doc characterises this mask as "mostly the invalid-depth region" plus a constant left-edge rectification band and a bottom-left gripper blob, and separately notes that it "does NOT cover the gripper" in the probe scene.

**Alternatives considered.** Decision 1.2's mask options: none; fixed dataset polygon; per-scene temporal-variance (Otsu) mask; flow-based rejection; GT outlier mask.

**Why this choice.** Here the mask is used only to define which GT depths are trusted as *oracle 3D* (line 75), not to mask the detector; that is consistent with decision 1.2 ("NO MASK for v0") for the arm while still excluding depth pixels the dataset itself flags. Because the mask does not reliably cover the gripper and "GT depth is defined ON the gripper in many scenes" (two-view diagnostics findings), gripper corners with valid depth pass into the `clean` set with a 3D point that is geometrically correct in frame i but votes for identity in frame j; see the honesty notes.

### Lines 35-36: rotation-error helper

```text
35: 
36: def rot_angle(R): return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
```

**What it does.** Geodesic angle of a rotation matrix, in degrees: `theta = arccos((trace(R) - 1) / 2)`. The clip to [-1, 1] guards against floating-point trace values slightly outside the valid range, which would otherwise produce NaN for near-identity rotations (the common case at gap 1, where GT steps are ~1.3 deg). Used at line 88 on the error rotation.

**Alternatives considered.** Quaternion or log-map based angle; Frobenius-norm proxies. For scoring, the geodesic angle is the standard choice.

**Why this choice.** Reports rotation error in the same unit (deg) as the harness RPE-rot column, so the doc can place the oracle floor (0.44 deg) directly against the model's per-frame RPE-rot medians (0.88 deg finetuned, 0.82 champion on the same 12 scenes).

### Lines 37-40: LK frontend, corner detection

```text
37: 
38: def lk(gi, gj):
39:     p0 = cv2.goodFeaturesToTrack(gi, 1000, 0.01, 5)
40:     if p0 is None: return np.zeros((0, 2)), np.zeros((0, 2))
```

**What it does.** `lk(gi, gj)` builds corner correspondences from gray frame i to gray frame j. `goodFeaturesToTrack(image, maxCorners=1000, qualityLevel=0.01, minDistance=5)` is Shi-Tomasi: corners whose minimum eigenvalue is at least 1% of the strongest corner's, non-maximum-suppressed to 5 px spacing, at most 1000 returned (the cap is never reached on this data: ~140-300 corners/frame, ~215 on the probe scene). Return shape is `(N, 1, 2)` float32 in pixel coordinates, or `None` if nothing qualifies, in which case two empty `(0, 2)` arrays are returned so the caller's `len(a) < 8` test at line 73 skips the pair.

**Alternatives considered.** Decision 1.3 lists "Shi-Tomasi; FAST; ORB; SIFT/AKAZE; fixed grid"; decision 1.4 lists grid bucketing / top-up for spatial spread.

**Why this choice.** Decision 1.3: Shi-Tomasi with exactly these values ("Designed for LK; relative quality threshold adapts across exposure; no wasted descriptor"); decision 1.4: no bucketing because "minDistance=5 already spreads corners at this resolution". The values match the frozen v0 parameters table (max corners 1000). No mask argument is passed, consistent with decision 1.2.

### Lines 41-44: LK frontend, forward-backward tracking

```text
41:     p1, st, _ = cv2.calcOpticalFlowPyrLK(gi, gj, p0, None)
42:     p0b, stb, _ = cv2.calcOpticalFlowPyrLK(gj, gi, p1, None)
43:     ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm((p0 - p0b)[:, 0], axis=1) < 1.0)
44:     return p0[ok, 0].astype(np.float64), p1[ok, 0].astype(np.float64)
```

**What it does.** Pyramidal Lucas-Kanade at OpenCV defaults (`winSize=(21,21)`, `maxLevel=3`, default termination criteria; `nextPts=None` means no initial guess). Line 41 tracks the corners forward i -> j (`p1`, `(N,1,2)`), line 42 tracks the results back j -> i (`p0b`). `st`/`stb` are `(N,1)` uint8 status flags (1 = a flow was found); the third return, the per-point error, is discarded. A track survives if both directions succeed and the round-trip lands within 1.0 px of the original corner. Returns two `(M, 2)` float64 arrays of matched pixel coordinates in i and in j. Note this is *direct* LK from i to j across the whole gap, not chained frame-to-frame tracking (the pipeline uses persistent frame-to-frame tracks, decision 1.6; the two-view diagnostics measure chained vs single-hop separately).

**Alternatives considered.** Decision 1.5's options: sparse LK with/without the forward-backward check; ORB/SIFT + ratio test; DIS or Farneback dense flow at corners or on a grid.

**Why this choice.** Decision 1.5, settled by the correspondence benchmark: "sparse LK at OpenCV defaults (21 px window, 3 levels) + forward-backward check at 1 px". From that table (12 smoke scenes, 4243 frames): LK+FB 166 correspondences / 88 within 2 px of GT at gap 1 and 103 / 27 at gap 4, at 4 ms; ORB, SIFT and Farneback were rejected by the data; DIS-grid wins two-view rotation via point count (880 / 478) but LK+FB has the best per-correspondence precision at gap 4 and yields the corner-anchored persistent tracks the map needs. The docstring's "Same LK+FB frontend as decision 1.5" is this function.

### Lines 45-48: solver flags and the 10 variants

```text
45: 
46: FLAGS = {"iterative": cv2.SOLVEPNP_ITERATIVE, "epnp": cv2.SOLVEPNP_EPNP, "p3p": cv2.SOLVEPNP_P3P,
47:          "ap3p": cv2.SOLVEPNP_AP3P, "sqpnp": cv2.SOLVEPNP_SQPNP}
48: VARIANTS = [(f, r) for f in FLAGS for r in (False, True)]
```

**What it does.** Maps short names to OpenCV `solvePnP` method flags: `ITERATIVE` (Levenberg-Marquardt on reprojection error, DLT-initialised when no seed is given; OpenCV's default method), `EPNP` (Lepetit et al., closed-form via 4 control points), `P3P` (Gao et al. minimal 3-point solver), `AP3P` (Ke & Roumeliotis algebraic P3P), `SQPNP` (Terzakis & Lourakis, globally optimal least-squares via sequential quadratic programming). `VARIANTS` is the cross product with the LM-refine toggle, giving 10 `(flag, refine)` pairs in dictionary insertion order, named `iterative`, `iterative+lm`, `epnp`, ... at line 84.

Relevant OpenCV `solvePnPRansac` semantics (from the OpenCV 4.x implementation, checked in the `cuteanything` env on 2026-09-08, not from the design doc). For ITERATIVE/EPNP/SQPNP the RANSAC hypothesis kernel is EPnP on 5-point samples; P3P/AP3P sample 4 points and use their own solver. After consensus one non-robust solve is run on the whole inlier set with the requested flag, EXCEPT that P3P/AP3P are replaced by EPnP in that final step (P3P needs exactly 4 points; `solvePnP(P3P)` on the full inlier set raises `cv2.error`, and the `p3p`/`ap3p` RANSAC results coincide with an EPnP solve on their inlier set to ~1e-8). For ITERATIVE that final step is an LM minimisation of reprojection error seeded with the best hypothesis, so a subsequent `solvePnPRefineLM` on the same inliers (line 61) is a no-op (doc: "+LM identical"; checked: rvec changes by ~1e-9). For SQPNP the final step is an SQP iteration, not a closed form. Consequently the `epnp`, `p3p` and `ap3p` rows without LM all end in an EPnP solve and differ only through the inlier set their kernel found, which is why the doc calls their differences noise.

**Alternatives considered.** Decision 2.4's full option list is exactly these five plus the LM toggle. `SOLVEPNP_DLS` and `SOLVEPNP_UPNP` (both documented by OpenCV as broken implementations that fall back to EPnP) and `SOLVEPNP_IPPE*` (planar-only) exist but are not candidates here.

**Why this choice.** Decision 2.4: "**DECIDED: solvePnPRansac(ITERATIVE) + solvePnPRefineLM on inliers** (standard v0; LM refine is a no-op after ITERATIVE-RANSAC but kept so the refine step exists for the P3P/SQPnP swaps)." The table that settled it (gap1 rot deg (p90) / gap1 trans mm / gap4 rot (p90) / gap4 trans mm / ms): ITERATIVE (+LM identical) 0.44 (1.72) / 3.9 / 1.10 (6.86) / 8.3 / 1.0 or 2.3; EPnP / +LM 0.41 / 0.44, 4.0 / 3.9, 1.18 / 1.10, 10.1 / 8.3, 0.8 / 2.2; P3P, AP3P / +LM 0.40 / 0.41, 3.9 / 3.6, 1.14 / 1.08, 10.2 / 8.7, 0.3 / 0.7; SQPnP / +LM 0.37 / 0.44, 3.3 / 3.9, 1.06 / 1.10, 8.4 / 8.3, 0.7 / 2.2. Verdict in the doc: "all 10 variants within 0.05 deg / 2 mm of each other; differences are noise. SQPnP nominally best by 0.04 deg; P3P/AP3P need +LM to match on translation." Paired differences vs ITERATIVE+LM have median |diff| < 0.01 deg for every variant. Decision 2.5 adds that SQPnP is retained only as "a manual single-variable swap ... if the diagnostics show solver failures", never as an automatic fallback.

### Lines 49-51: `solve()` signature and contract

```text
49: 
50: def solve(X, x, flag, refine):
51:     """X: (N,3) in frame i camera coords; x: (N,2) pixels in frame j. Returns T_ji (4x4) or None, n_inliers."""
```

**What it does.** One PnP solve. Inputs: `X`, `(N,3)` float64 metric 3D points expressed in the camera-i frame; `x`, `(N,2)` float64 pixel coordinates of the same points observed in frame j; the method flag name and the LM toggle. Output: the 4x4 rigid transform `T_ji` that maps camera-i coordinates into camera-j coordinates (i.e. the object-to-camera pose `solvePnP` natively returns, because the "object" frame *is* camera i), or `None` on failure, plus the inlier count.

**Alternatives considered.** Returning `(rvec, tvec)` directly, or the inverse (camera-j-to-camera-i, the "motion" of the camera). Either works if the scorer is consistent.

**Why this choice.** The convention matches `T_gt` at line 71 (`inv(c2w[j]) @ c2w[i]`, also i -> j), so no inversion is needed before scoring. The design doc's day-one check ("PnP with GT depth between two frames must reproduce the GT relative pose composed from cam/*.npz") is, in effect, what this function does across 2118 pairs; the two-view diagnostics separately verified conventions on synthetic data ("E from GT flow, no noise: 0.01 / 0.0").

### Lines 52-56: robust PnP call

```text
52:     try:
53:         ok, rvec, tvec, inl = cv2.solvePnPRansac(X, x, K, None, flags=FLAGS[flag], reprojectionError=2.0,
54:                                                  confidence=0.999, iterationsCount=1000)
55:     except cv2.error:
56:         return None, 0
```

**What it does.** `cv2.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, ...)`: `K` is the calibrated 3x3 matrix (line 31); `distCoeffs=None` means no lens distortion (the dataset facts say the calibration has none, so image points are treated as ideal pinhole pixels); `flags` selects the method; `reprojectionError=2.0` px is the RANSAC inlier threshold; `confidence=0.999` and `iterationsCount=1000` bound the sampling. `useExtrinsicGuess` is left at its default `False`, and no `rvec`/`tvec` seed is passed. Returns `ok` (bool), `rvec` (3x1 Rodrigues axis-angle, radians), `tvec` (3x1, same metric unit as `X`, here metres), and `inl`, an `(m,1)` int32 array of inlier row indices (or `None`/empty on failure). OpenCV raises `cv2.error` rather than returning `False` on invalid input, in particular fewer than 4 points (it asserts `npoints >= 4`; checked in the `cuteanything` env), so the call is wrapped and a raised error is counted as a failed solve.

**Alternatives considered.** Decision 2.5: "none; identity; previous motion" for the initial guess. Decision 2.6: "RANSAC; USAC/MAGSAC++; none". Decision 2.7: 8 px default vs 1-3 px. Decision 2.8: 0.99/100 default vs 0.999/1000.

**Why this choice.** Decision 2.5 (no guess, `useExtrinsicGuess=False`): "zero failures over 2118 pairs with no guess; a guess or a fallback would hide instability the diagnostics must expose". Decision 2.6 (plain RANSAC): it "handled identity-voting outliers with a 0.04 deg penalty in the PnP benchmark" (the 1.10 -> 1.14 deg gap-4 number). Decisions 2.7 and 2.8 fix the three numeric arguments as described under lines 14-17; they are inputs here: the 2 px threshold is the correspondence benchmark's "correct" definition reused unchanged, and 0.999 / 1000 is decision 2.8's stated policy. The `distCoeffs=None` choice is forced by the calibration having no distortion; the pipeline likewise never undistorts (decision 1.1).

### Lines 57-58: success test and inlier index flattening

```text
57:     if not ok or inl is None or len(inl) < 4: return None, 0
58:     inl = inl[:, 0]
```

**What it does.** A solve counts as failed if OpenCV reports failure, returns no inlier array, or found fewer than 4 inliers. 4 is the hard floor `solvePnPRansac` asserts on its input (`npoints >= 4`); in the RANSAC path the final ITERATIVE solve is always seeded from the best hypothesis, so the 6-point DLT requirement of an unseeded ITERATIVE solve does not apply. `inl` is then squeezed from `(m,1)` to `(m,)` so it can index `X` and `x` at line 61.

**Alternatives considered.** Decision 3.1 lists inlier floors of 10 / 20 / 30 (or a ratio) for the *arm*; the benchmark uses the weakest possible floor.

**Why this choice.** The probe's purpose is to count solver failures, not to enforce a quality gate, so the threshold is "solvable at all" (4). The arm's own floor is decision 3.1 = 30 inliers (sweeps 1-2: floor 10 gave RPE-r 1.80 deg because bad PnP was accepted; 30 vs 20 gave 0.861 vs 0.923 RPE-r). Interpreting the doc's "zero failures on 2118 pairs" therefore means zero solves with fewer than 4 inliers, not zero solves below the arm's 30-inlier bar; the JSON's `inl` list (line 92) is what carries the count distribution.

### Lines 59-63: optional LM refinement

```text
59:     if refine:
60:         try:
61:             rvec, tvec = cv2.solvePnPRefineLM(X[inl], x[inl], K, None, rvec, tvec)
62:         except cv2.error:
63:             pass
```

**What it does.** When the variant's `refine` flag is set, `cv2.solvePnPRefineLM(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec)` runs a Levenberg-Marquardt minimisation of the summed squared reprojection error over the RANSAC inlier subset only, starting from the RANSAC pose; it returns the refined `(rvec, tvec)`. It is non-robust (every passed point is trusted), which is why it is fed only inliers. A `cv2.error` (e.g. a degenerate inlier configuration) leaves the unrefined pose in place instead of failing the pair.

**Alternatives considered.** `solvePnPRefineVVS` (virtual visual servoing, the other OpenCV refiner); re-running RANSAC; a windowed bundle adjustment (decision 3.4, rejected: "local BA window 5 / 10: ATE 128.8 / 130.7 vs 122.9").

**Why this choice.** Decision 2.4 keeps the LM refine step even though it is measured to be a no-op after ITERATIVE ("+LM identical" in the table; see the lines 45-48 block for why), because the refine is what makes the P3P/AP3P/SQPnP swaps comparable: with +LM, P3P/AP3P translation error at gap 4 drops from 10.2 to 8.7 mm and EPnP from 10.1 to 8.3 mm, matching ITERATIVE's 8.3 mm. Since the `epnp`, `p3p` and `ap3p` variants all end their RANSAC in an EPnP solve on the inliers, this LM pass is the only step that gives those three variants a reprojection-error-minimising pose at all. Cost, from the `ms` column: roughly +1.3 ms per solve (1.0 -> 2.3 ms for ITERATIVE).

### Lines 64-65: assemble the 4x4 transform

```text
64:     R, _ = cv2.Rodrigues(rvec); T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.ravel()
65:     return T, len(inl)
```

**What it does.** `cv2.Rodrigues` converts the 3x1 axis-angle vector to a 3x3 rotation (the second return is the Jacobian, discarded). The rigid transform is packed as `T = [R t; 0 1]`, so `X_j = R X_i + t` for a point `X_i` in camera-i coordinates; this is `T_ji` as promised by the docstring, in metres because the oracle 3D points are metric. Returns the transform and the inlier count.

**Alternatives considered.** Keeping `(rvec, tvec)` and composing errors in that form; returning c2w by inverting.

**Why this choice.** A homogeneous matrix makes the error composition at line 87 a single matrix product and keeps the i -> j direction identical to `T_gt`. The doc's general warning that `solvePnP` returns world-to-camera and must be inverted before writing c2w applies to the *arm's* output writer, not here.

### Lines 66-68: result accumulator

```text
66: 
67: res = {f"{f}{'+lm' if r else ''}": {g: {c: dict(pairs=0, fail=0, rot=[], trans_mm=[], tdir=[], inl=[], ms=[]) for c in ("clean", "outliers")}
68:                                      for g in gaps} for f, r in VARIANTS}
```

**What it does.** Nested dictionary `res[variant][gap][condition]` with per-cell counters `pairs` and `fail` and per-pair lists `rot` (deg), `trans_mm` (mm), `tdir` (deg), `inl` (count), `ms` (solver wall time). Variant keys are `iterative`, `iterative+lm`, `epnp`, `epnp+lm`, `p3p`, `p3p+lm`, `ap3p`, `ap3p+lm`, `sqpnp`, `sqpnp+lm`. Gap keys are Python ints here; `json.dump` at line 94 turns them into strings.

**Alternatives considered.** Streaming per-pair rows to CSV (as the arm's `diag.csv` does, decision 0.4) would allow per-scene inspection at pair granularity; the probe stores raw lists so medians and p90s can be pooled across scenes afterwards.

**Why this choice.** The doc's table is built from these lists: medians (and p90 for rotation) pooled over all pairs of the 12 scenes per gap, plus the failure count (`fail`) and timing (`ms`). Keeping raw lists rather than per-scene aggregates is what allows the "paired diffs vs ITERATIVE+LM: median |diff| < 0.01 deg" statement, since every variant sees the identical `(X, x)` for every pair.

### Lines 69-71: pair loop and GT relative pose

```text
69: for g in gaps:
70:     for i in range(0, n - g, 2):
71:         j = i + g; T_gt = np.linalg.inv(c2w[j]) @ c2w[i]
```

**What it does.** For each gap, source frames `i` are taken with stride 2 (`0, 2, 4, ...`) up to `n - g - 1`, so pairs are `(i, i+g)` and consecutive pairs overlap for gap 4. `T_gt = w2c_j @ c2w_i` maps camera-i coordinates to camera-j coordinates: the GT counterpart of the `T_ji` that `solve()` returns, in the dataset's metric units.

**Alternatives considered.** Stride 1 (every source frame, twice the pairs and runtime); random pair sampling; a parallax gate as used by the later triangulation and two-view benchmarks (median flow >= 2 px). No gate is applied here.

**Why this choice.** Stride 2 halves runtime while still yielding on the order of ~2100 pairs per gap over the 12 scenes (the corr benchmark's wording; the PnP benchmark reports 2118 pairs). Not gating on parallax is deliberate for a per-frame solver test: the arm must register *every* frame including stationary ones (eval_depth_poses.py numbers frames by position, so every frame must get a pose), so near-zero-motion pairs belong in the population; the `tdir` guard at line 90 handles the metric that is undefined there.

### Lines 72-73: correspondences and the minimum-track floor

```text
72:         a, b = lk(G[i], G[j])
73:         if len(a) < 8: continue
```

**What it does.** Runs the LK+FB frontend from gray frame i to gray frame j, yielding matched pixel arrays `a` (in i) and `b` (in j). Pairs with fewer than 8 surviving tracks are skipped entirely (not counted as pairs or failures for any variant).

**Alternatives considered.** A floor of 4 (PnP minimum) or of the arm's 30; skipping nothing and letting the solver fail.

**Why this choice.** 8 is not a design-doc parameter. Skipped pairs are not counted anywhere (`pairs` is only incremented at line 84), so the doc's 2118 pairs is the post-skip population; the doc's medians (LK+FB 166 correspondences at gap 1, 103 at gap 4) suggest the floor is rarely reached, but the JSON cannot confirm how often.

### Lines 74-75: oracle depth lookup and validity

```text
74:         u = np.clip(np.round(a[:, 0]).astype(int), 0, W - 1); v = np.clip(np.round(a[:, 1]).astype(int), 0, H - 1)
75:         z = D[i][v, u]; valid = (z > 0) & (~M[i][v, u])
```

**What it does.** Each sub-pixel corner `(x, y)` in frame i is rounded to the nearest integer pixel `(u, v)` and clipped into the image, then the GT depth at that pixel is read (`D[i][v, u]`, row = y, column = x). A track is `valid` when its depth is nonzero (0 = invalid in this dataset) and its pixel is not in the outlier mask. `z` is in metres.

**Alternatives considered.** Bilinear depth interpolation (biased across depth discontinuities); nearest-valid-neighbour search; rejecting corners within some radius of a depth edge.

**Why this choice.** Nearest-pixel lookup is the simplest oracle and is standard for a solver probe. The doc attributes the oracle floor to "LK noise at 320x180 plus GT depth/pose inconsistency, not ... the solver" and separately records a "~20% depth-vs-pose scale mismatch ... in some scenes" (two-view findings), which enters here through `z` and is a reason the oracle floor (3.9 mm translation) is not zero. A plausible, unmeasured contributor is nearest-pixel lookup at depth edges (corners favour edges, and rounding can land on the far or near surface); the doc itself names only LK noise and the depth-vs-pose mismatch.

### Lines 76-77: static no-depth tracks and the valid-count floor

```text
76:         disp = np.linalg.norm(b - a, axis=1); static_nodepth = (disp < 1.0) & (~valid)
77:         if valid.sum() < 8: continue
```

**What it does.** `disp` is the per-track image displacement in pixels between i and j. `static_nodepth` marks tracks that moved less than 1.0 px *and* have no usable GT depth: the candidates the `outliers` condition will equip with a wrong 3D point. Pairs with fewer than 8 valid-depth tracks are skipped for all variants (again uncounted, like line 73).

**Alternatives considered.** The arm's lever (decision 1.2) uses the same 1 px static threshold but only on frames whose *median* displacement is >= 2 px (so that a paused camera is not affected); no such gate exists here, so on a stationary pair every no-depth track becomes an "outlier" whose identity vote is in fact correct. Alternatively one could inject outliers from *all* no-depth tracks regardless of motion, which would test random rather than identity-voting outliers.

**Why this choice.** The docstring's aim is to reproduce "the identity-voting outliers a real map will contain": in the arm, gripper corners get triangulated (or at least survive) and then, being image-static, agree with the identity pose. The 1 px cut is the same value decision 1.2 later froze for the lever ("drop tracks with displacement < 1 px"), whose static share across the 12 smoke scenes is 16-53% (median ~25%) of LK tracks. The absence of the 2 px parallax gate here is a benchmark simplification (see honesty notes).

### Lines 78-79: fill wrong depths and back-project

```text
78:         zfill = z.copy(); zfill[~valid] = np.median(z[valid])
79:         Xall = ((Kinv @ np.c_[a, np.ones(len(a))].T) * zfill).T  # N,3 in cam i
```

**What it does.** `zfill` replaces every invalid depth by the median of the valid depths in this frame, the "plausible wrong 3D point (median scene depth)" of the docstring. Line 79 back-projects every track: homogeneous pixel `[x, y, 1]` is multiplied by `K^-1` to get a normalised ray `[X/Z, Y/Z, 1]`, then scaled by the depth so the third coordinate equals `z`. This treats `D` as z-depth along the optical axis, not ray length; the design doc does not record which the dataset stores, so this is an assumption the script makes (had `D` been ray length, the ray would need normalising to unit length before scaling). The result `Xall` is `(N,3)` metric points in the camera-i frame, for valid and filled tracks alike; the condition selection at line 81 decides which rows are used.

**Alternatives considered.** For the wrong point: a random depth in the scene range; the true depth of a neighbouring pixel; the GT depth where it exists on the gripper. For the back-projection: OpenCV's `undistortPoints` (identical here since there is no distortion).

**Why this choice.** The median depth is a "plausible" outlier: it lies inside the observed 0.3-1.4 m range, so it is not trivially rejectable by a depth prior, yet its image-static observation contradicts the true motion. The `# N,3 in cam i` comment pins the coordinate frame the whole `solve()` contract depends on.

### Lines 80-82: condition selection

```text
80:         for cond in ("clean", "outliers"):
81:             sel = valid if cond == "clean" else (valid | static_nodepth)
82:             X = np.ascontiguousarray(Xall[sel]); x = np.ascontiguousarray(b[sel])
```

**What it does.** `clean` uses only valid-depth tracks; `outliers` adds the static no-depth tracks with their filled (wrong) depth. `X` is the selected 3D set in camera i, `x` the selected frame-j pixel observations (`b`, not `a`: the observation is in the *target* frame, the 3D point in the *source* frame, which is what makes the solved pose i -> j). `np.ascontiguousarray` guarantees the C-contiguous float64 layout OpenCV's Python bindings expect.

**Alternatives considered.** A third condition with *all* no-depth tracks filled (random-motion outliers), or graded outlier fractions; neither was needed to answer decision 2.6.

**Why this choice.** Two conditions give the paired comparison the doc quotes: "Identity-voting outliers cost only 1.10 -> 1.14 deg at gap 4 (RANSAC copes when the map points are right)". The parenthetical is the important caveat: this probe's outlier condition is easy for RANSAC because the *inlier* 3D is oracle; the later triangulation benchmark shows that with the arm's own E-matrix map (~53% bad points) PnP lands at ~4 deg, i.e. map quality, not solver robustness, is the arm's bottleneck.

### Lines 83-86: run every variant, time it, count failures

```text
83:             for f, r in VARIANTS:
84:                 name = f"{f}{'+lm' if r else ''}"; rr = res[name][g][cond]; rr["pairs"] += 1
85:                 t0 = time.time(); T, ninl = solve(X, x, f, r); rr["ms"].append(1000 * (time.time() - t0))
86:                 if T is None: rr["fail"] += 1; continue
```

**What it does.** All 10 variants are run on the identical `(X, x)`. Each cell's `pairs` counter is incremented, the solve is wall-clock timed (milliseconds, `time.time()` resolution), and a `None` result increments `fail` and skips scoring.

**Alternatives considered.** `time.perf_counter()` for higher-resolution timing; repeating each solve to average out RANSAC's random iteration count.

**Why this choice.** Identical inputs per variant are what make the paired-difference statement in the doc valid. The `fail` counter is the source of decision 2.5's "zero failures over 2118 pairs" (a `None` from the line 55 or line 57 exits). Timing is informational (the `ms` column: 0.3-2.3 ms per solve) and is the only cost evidence the doc quotes for decision 2.8's 0.999 / 1000 setting ("1-2 ms per frame"); the per-scene process runs single-threaded (line 23), but 12 scene processes share one node, so absolute timings are approximate.

### Lines 87-88: rotation and translation error

```text
87:                 E = np.linalg.inv(T_gt) @ T
88:                 rr["rot"].append(float(rot_angle(E[:3, :3]))); rr["trans_mm"].append(float(1000 * np.linalg.norm(E[:3, 3])))
```

**What it does.** The error transform `E = T_gt^-1 T` is the residual motion that would turn the estimate into the truth; its rotation angle (line 36) is the rotation error in degrees, and the norm of its translation part, converted to millimetres, is the translation error. Because the 3D points are metric, this is an absolute metric translation error, not a direction-only error as for the essential-matrix probes.

**Alternatives considered.** Frobenius/chordal distances; scale-invariant translation error (unnecessary here since scale is metric); the harness's own Sim(3)-aligned ATE/RPE (a trajectory-level metric, meaningless for isolated pairs).

**Why this choice.** These are the two numbers in the doc's table (per-pair medians, p90 for rotation). They establish the oracle floor the doc emphasises: "even with ORACLE 3D points, per-frame PnP rotation error is 0.44 deg median (GT median step 1.3 deg) and translation error 3.9 mm (GT median step 9 mm)", and the calibration "Oracle-3D PnP per-frame (0.44 deg / 3.9 mm) is ~2x better than the model on relative pose; the map arm sits between" (model per-frame medians on the same 12 scenes: augfull_lr1e5 0.88 deg / 5.7 mm / ATE 90 mm; champion 0.82 deg / 5.5 mm / 81 mm).

### Lines 89-92: translation direction error and inlier count

```text
89:                 tg = T_gt[:3, 3]; tp = T[:3, 3]
90:                 if np.linalg.norm(tg) > 2e-3 and np.linalg.norm(tp) > 1e-9:
91:                     rr["tdir"].append(float(np.degrees(np.arccos(np.clip(np.dot(tg, tp) / np.linalg.norm(tg) / np.linalg.norm(tp), -1, 1)))))
92:                 rr["inl"].append(int(ninl))
```

**What it does.** Angle in degrees between the GT and estimated translation vectors (both i -> j, in camera-j coordinates), recorded only when the GT step exceeds 2 mm (`2e-3` m, so that direction is defined; the per-step median is 9 mm) and the estimate is non-degenerate. The clip guards `arccos`. The RANSAC inlier count is appended unconditionally for scored pairs.

**Alternatives considered.** Reporting `tdir` for all pairs (dominated by noise on near-stationary steps); a fixed fraction of the median step instead of an absolute 2 mm.

**Why this choice.** `tdir` is the diagnostic the essential-matrix probes are scored on (the doc's two-view tables list "rot / tdir"), so recording it here allows a like-for-like reading against those. `inl` and `tdir` are recorded but not reported in the doc's PnP table, which reports only rotation, metric translation and ms. The decision 2.5 evidence ("zero failures over 2118 pairs") is the `fail` counter of line 86; `inl` only lets a later reader inspect the inlier-count distribution behind those solves. The arm's own median-inlier diagnostic (decision 0.4) comes from a separate `diag.csv` written by `opencv_vo.py`, not from this list.

### Lines 93-95: write results

```text
93: 
94: json.dump(dict(scene=os.path.basename(scene), n=n, res=res), open(out, "w"))
95: print(os.path.basename(scene), n, "done")
```

**What it does.** Serialises the scene name, frame count and the full `res` tree to the output JSON (integer gap keys become strings; the file is never explicitly closed, which CPython's refcounting makes harmless at exit). Prints a one-line completion marker that the driver redirects to a per-scene log (`$G/out/$S.log`, `pnp_bench.sbatch` line 17).

**Alternatives considered.** NumPy `.npz` or CSV outputs; a pooled multi-scene run in one process.

**Why this choice.** One JSON per scene under `.../opencv_vo_probe/pnp_bench/out/` (design doc: "Script + JSON: .../pnp_bench/"; driver: `$G/out/$S.json`) lets the 12 scenes run as independent parallel processes on one debug-QoS node and be pooled afterwards; the doc's table is that pooled result.

### Known limitations / honesty notes for this block

- **GT-scored on the reported test scenes.** Every number this probe produced comes from the 12 smoke scenes that are also the reported scenes; honesty-audit items 7 and 8 ("all thresholds tuned on the 12 reported (test) scenes", "design benchmarks GT-scored on test scenes"). Item 7 is addressed by the frozen-configuration full-4292 run; item 8 is inherent to an oracle probe.
- **Oracle 3D is privileged and non-causal by construction.** GT depth and GT pose are used (lines 33, 71, 75, 79). This is allowed only because the probe is a design experiment; decision 0.1 forbids it in the arm. Its results bound the arm from above, they do not describe it: the arm's own map gives ~4 deg PnP rotation at gap 4 vs 1.10 deg here (triangulation benchmark).
- **"clean" is not free of identity-voting outliers.** Line 10's claim holds only where the gripper has no GT depth. The doc's two-view findings state "GT depth is defined ON the gripper in many scenes, so 'valid GT depth' is not a gripper oracle", and the dataset facts note the `outlier_mask` "does NOT cover the gripper". Such tracks enter `clean` with a correct source-frame 3D point but an image-static observation. The reported 1.10 -> 1.14 deg outlier penalty is therefore the *marginal* cost of the synthetic no-depth static tracks on top of whatever real gripper contamination `clean` already carries.
- **No parallax gate on the outlier injection** (line 76). On stationary pairs the injected "outliers" vote for the correct (identity) pose; the arm's lever (decision 1.2) only fires when the median flow is >= 2 px. This makes the outlier condition slightly easier than the arm's reality.
- **Resolution and K differ from the frozen pipeline.** The probe uses native 320x180 frames with the calibrated K of frame 0 (lines 28-31); the arm uses 320x192 cover frames with a per-frame model focal (decisions 0.1b, 1.1). The 2 px RANSAC threshold is thus in slightly different pixel units in the two settings (1.067x scale). The doc's full-pipeline focal check found the focal source immaterial, and decision 2.7 re-confirmed 2 px in pipeline sweep 2, so the conclusion transfers, but the numbers here are not pipeline numbers.
- **Depth semantics are assumed, not verified** (line 79). The back-projection treats GT depth as z-depth along the optical axis; the design doc does not say whether the `.npy` files store z-depth or ray length.
- **Skipped pairs are invisible** (lines 73, 77). Pairs with fewer than 8 tracks or fewer than 8 valid-depth tracks are dropped before any counter is touched, so the JSON cannot say how often that happened or whether the dropped pairs were the hard ones.
- **Failure means "< 4 inliers", not the arm's 30-inlier floor** (line 57). "Zero failures on 2118 pairs" (decision 2.5) is a statement about solvability with no initial guess, not about meeting decision 3.1's quality bar.
- **The solver differences are declared noise by the doc itself.** All 10 variants lie within 0.05 deg / 2 mm; SQPnP's nominal 0.04 deg edge and the P3P/AP3P translation deficit without LM are the only structure, and the `epnp`/`p3p`/`ap3p` rows without LM are in fact the same EPnP final solve on differently found inlier sets (lines 45-48 block). Decision 2.4 is therefore a convention ("standard v0"), not a measured win, and the LM refine is kept although measured to be a no-op after ITERATIVE.
- **Timings are approximate.** Single-threaded per process (line 23) but 12 processes share one node; the `ms` column supports decision 2.8's "1-2 ms per frame" only at that level of precision.
- **Direct i -> j LK, not the arm's chained tracks** (lines 41-42). The two-view diagnostics later found chained-LK drift grows with chain length; this probe's gap-4 correspondences are single-hop and so somewhat cleaner than what the arm's persistent tracks provide at the same gap.
- **The metric floor is the lasting result, not the solver ranking.** The doc's reading of this benchmark is that per-frame PnP on this footage bottoms out at 0.44 deg / 3.9 mm because of "LK noise at 320x180 plus GT depth/pose inconsistency, not ... the solver; the map arm cannot beat it". Any future solver swap should be judged against that floor, not against ITERATIVE.
