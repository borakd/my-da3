## `eval_pipeline/opencv_vo_probes/corr_bench.py`, lines 1-127 — correspondence benchmark (decision 1.5)

This file is the frontend-only probe that settled **decision 1.5 (correspondence method)** of the OpenCV visual-odometry control arm. It takes one DROID wrist scene, forms frame pairs `(i, i+gap)` for `gap` in `{1, 4}` (every second start frame), and runs seven correspondence methods on each pair: sparse Lucas-Kanade with and without a forward-backward check, ORB and SIFT with Lowe's ratio test, DIS dense flow sampled at Shi-Tomasi corners or on an 8-px grid, and Farneback dense flow at corners. Each method's 2D-to-2D correspondences are scored against the flow *induced* by GT depth and GT relative pose (end-point error, 1 px / 2 px inlier rates), against the GT essential geometry (Sampson distance), for zero-motion share (the gripper diagnostic), and, on pairs with more than 0.5 deg of GT rotation, by the rotation error of an essential matrix fitted to the correspondences. It writes one JSON per scene; the sbatch launcher next to it (`corr_bench.sbatch`) fans the 12 smoke scenes (`eval_pipeline/cg_smoke_scenes_12.txt`) out as parallel single-thread processes on one node. The design doc's table under "Correspondence benchmark (2026-09-04, decision 1.5; 12 smoke scenes, 4243 frames; Slurm job 45403560)" was pooled from those JSONs ("Medians pooled over ~2100 pairs per gap"). Nothing in this file is used at inference time: it is a design-time measurement that runs on the *native* 320x180 frames with the *calibrated* K, whereas the frozen pipeline (`eval_pipeline/opencv_vo.py`) runs on the eval loader's 320x192 frames with the model's focal (decisions 1.1 and 0.1b). GT depth and pose enter only as the scoring reference.

### Lines 1-3: shebang and docstring opening

```
1: #!/usr/bin/env python
2: """Correspondence-method benchmark for decision 1.5 (frontend only).
3: 
```

**What it does.** Marks the script as directly executable under whichever `python` is on `PATH` (the launcher activates the `cuteanything` conda env first) and opens the module docstring. "Frontend only" is the scope statement: the probe measures correspondences, not poses or maps.

**Alternatives considered.** A benchmark embedded in the pipeline script behind a flag; a notebook. Neither was used.

**Why this choice.** The design doc's tiering (Tier 1 frontend, Tier 2 solver, Tier 2b map) demands that each decision be settled by a single-variable measurement; a standalone probe per tier-1 decision is the simplest way to keep the variable single. This benchmark (2026-09-04) predates the user's 2026-09-05 standing rule to benchmark every proposed option empirically; the project-memory note recording that rule cites this benchmark as one of the results that motivated it ("DIS-grid beat LK on rotation").

### Lines 4-8: docstring, scoring protocol

```
4: For each frame pair (i, i+gap) of one scene, every method produces a set of
5: 2D->2D correspondences. Each correspondence is scored against the GT flow
6: induced by GT depth + GT relative pose (native 320x180 frame, calibrated K;
7: GT is used ONLY for scoring). Also: essential-matrix rotation error from the
8: method's correspondences on pairs with GT rotation > 0.5 deg.
```

**What it does.** States the protocol implemented below: the unit of measurement is a frame pair; the reference is GT-induced flow (a 2D point in frame `i` is back-projected with GT depth, moved by the GT relative pose, re-projected into frame `j`); the frame is the dataset's native 320x180 PNG with the calibrated intrinsics of frame 0 from `cam/000000.npz`, applied to the whole scene (line 24; the dataset stores K per frame, `fx ~ 192 px`, no distortion per the doc's dataset facts, but the probe reads only the first). The second metric, essential-matrix rotation error, is only computed on pairs whose GT rotation exceeds 0.5 deg. The docstring gives no reason for the gate; the natural reading is the doc's probe evidence that on near-stationary pairs the essential matrix is degenerate (frames 0-15 of the RAIL scene: GT rot 0.03 deg, |t| = 0.2 mm, "Essential matrix degenerate there").

**Alternatives considered.** Scoring by epipolar (Sampson) distance only, which needs no depth; scoring by downstream pose error only; scoring on the pipeline's 320x192 cover frames with the model focal.

**Why this choice.** GT-induced flow gives a per-correspondence "correct / incorrect" label at a stated pixel tolerance, which is what decision 2.7 later reuses ("~65% of consecutive-frame tracks fall within 2 px of GT (corr benchmark); same unit as the benchmarks' 'correct' definition"). Sampson is also computed (line 111) but was not the headline. Native frames + calibrated K make the GT flow exact; the 1.067x scale and crop of the pipeline frames (decision 1.1) would only add a resampling step to the reference. Honesty audit item 8 ("design benchmarks GT-scored on test scenes") applies to this whole file.

### Lines 9-11: usage line and docstring close

```
9: 
10: Usage: corr_bench.py <scene_dir> <out_json> [gap ...]
11: """
```

**What it does.** Documents the positional CLI: a scene directory (the one containing `dense/`), an output JSON path, and zero or more integer gaps. Blank line 9 separates the paragraphs.

**Alternatives considered.** `argparse` with named flags. Not needed for a two-argument probe launched from a fixed sbatch loop.

**Why this choice.** The launcher (`corr_bench.sbatch`) calls `corr_bench.py $R/$S $SC/corr/$S.json 1 4`, so gaps 1 and 4 are what produced every number in the doc's table.

### Lines 12-15: imports and OpenCV threading

```
12: import sys, os, glob, json, time
13: import numpy as np, cv2
14: cv2.setNumThreads(1)
15: 
```

**What it does.** Standard library for CLI/IO/timing, NumPy, OpenCV. `cv2.setNumThreads(1)` disables OpenCV's internal parallelism (the thread pool used by DIS, Farneback, pyramid building and the matchers) for this process. Line 15 is blank.

**Alternatives considered.** Leave OpenCV multi-threaded and run scenes serially; set threads via `OPENCV_FOR_THREADS_NUM` in the environment.

**Why this choice.** The launcher runs 12 scene processes concurrently on a 20-core allocation with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`; one OpenCV thread per process keeps the 12 processes from oversubscribing the node and makes the per-method `ms` column (doc table, last column) a single-core number, the same basis as the "OpenCV 30 ms/frame on one CPU core" figure later reported for the full pipeline. The doc's plumbing notes also require probes to run as CPU jobs (login node 300 s per-process CPU cap). The single-thread `cv2.setNumThreads(1)` practice was later codified in the project-memory note of 2026-09-05; it is a fact of this file, not a consequence of that note.

### Lines 16-20: arguments, gap list, frame list

```
16: scene, out = sys.argv[1], sys.argv[2]
17: gaps = [int(g) for g in sys.argv[3:]] or [1, 4]
18: d = os.path.join(scene, "dense")
19: fs = sorted(glob.glob(os.path.join(d, "rgb", "*.png")))
20: n = len(fs)
```

**What it does.** Reads the two mandatory paths; parses any remaining arguments as integer gaps, defaulting to `[1, 4]` when none are given. `d` is the `dense/` subfolder that holds `rgb, cam, depth, outlier_mask, sky_mask` (doc dataset facts). `fs` is the lexically sorted list of RGB PNGs, which for zero-padded `%06d` names is frame order; `n` is the sequence length (57 to ~1700 frames across the dataset).

**Alternatives considered.** Gaps 1 only (consecutive frames, the LK tracking regime); larger gaps such as 8 or 16 (later used by the two-view diagnostics).

**Why this choice.** Gap 1 is the frame-to-frame tracking regime (decision 1.6: persistent tracks). Gap 4 approximates a keyframe-scale baseline: the probe evidence in the doc shows survival "drops to ~15%" over a 15-frame gap in fast segments, so "tracking must be frame-to-frame", and the later keyframe sweep (decision 2.12) confirms that keyframes should be 2-4 frames apart ("Keyframes must be close (gap 2-4), not far"). The default `[1, 4]` equals what the launcher passes explicitly.

### Lines 21-22: grayscale frames

```
21: G = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in fs]
22: H, W = G[0].shape
```

**What it does.** Loads every frame as a single-channel `uint8` array of shape `(H, W)` = `(180, 320)`; `cv2.imread(..., IMREAD_GRAYSCALE)` converts BGR to luma at load time. The whole sequence is held in memory. `H, W` are read from the first frame and used for clipping and the grid below.

**Alternatives considered.** Decision 1.1 lists gray only; 2x upsample; CLAHE; undistort.

**Why this choice.** Decision 1.1: "grayscale conversion only ... Gray is mandatory for the detector/LK; everything else is a v1 single-variable experiment". The probe therefore feeds every method the same plain grayscale input, so the comparison is between correspondence methods only. (Holding the whole sequence in memory is audit item 3 for the pipeline; for a probe it is immaterial.)

### Lines 23-25: intrinsics and GT poses

```
23: cams = [np.load(os.path.join(d, "cam", "%06d.npz" % i)) for i in range(n)]
24: K = cams[0]["intrinsic"]; Kinv = np.linalg.inv(K)
25: c2w = [c["pose"] for c in cams]
```

**What it does.** Loads the per-frame `cam/%06d.npz` archives. `K` is the 3x3 pinhole matrix of frame 0 (`fx ~ 192 px` at 320x180, principal point inside the matrix, no distortion) and `Kinv` its inverse; **the same `K` is applied to every frame of the scene**, although the doc notes intrinsics are stored per frame. `c2w[i]` is the 4x4 **camera-to-world** GT pose of frame `i` (key `pose`), i.e. it maps points expressed in camera `i` into the world frame. Nothing in this block inverts it yet; the world-to-camera direction is formed where a relative pose is needed (lines 34, 99).

**Alternatives considered.** Decision 0.1b lists calibrated K from `cam/*.npz`; one nominal dataset K; model-estimated focal; self-calibration. Per-frame calibrated K vs frame-0 K.

**Why this choice.** For a *scoring reference* the calibrated K is the right choice (the doc: "Scoring uses GT depth+pose ONLY as reference (native 320x180, calibrated K)"). The pipeline itself uses the model's per-frame focal (0.1b, revised 2026-09-06), and the doc's focal check shows the focal source is "immaterial" to the pipeline's results. Using frame 0's K for the whole scene is a simplification of this probe, not a documented decision; the doc gives no within-scene variation figure for the calibrated K (only 0.7% jitter for the model focal), so its effect is unmeasured.

### Lines 26-28: GT depth and outlier mask

```
26: D = [np.load(os.path.join(d, "depth", "%06d.npy" % i)) for i in range(n)]
27: M = [cv2.imread(os.path.join(d, "outlier_mask", "%06d.png" % i), cv2.IMREAD_GRAYSCALE) > 0 for i in range(n)]
28: 
```

**What it does.** `D[i]` is the float32 `(H, W)` GT depth in metres along the optical axis (`z`), with `0 = invalid`; the doc reports ~88% valid and 0.3-1.4 m typical. `M[i]` is a boolean `(H, W)` mask, `True` where the dataset's `outlier_mask` PNG is non-zero; the doc says this is "mostly the invalid-depth region" plus a constant rectification band on the left edge. Line 28 is blank.

**Alternatives considered.** Skip the mask (rely on `z > 0` only); use `sky_mask` too; use a gripper mask.

**Why this choice.** Both are used purely to declare which reference points are trustworthy (line 32). The mask is *not* a gripper oracle: decision 1.2's probe found "GT outlier_mask does NOT cover the gripper", and the two-view diagnostics add "GT depth is defined ON the gripper in many scenes, so 'valid GT depth' is not a gripper oracle". See the honesty notes at the end.

### Lines 29-32: `gt_flow`, sampling depth at the query points

```
29: def gt_flow(i, j, pts):
30:     """pts: (N,2) float pixel coords in frame i -> (N,2) GT location in frame j, valid mask."""
31:     u = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1); v = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
32:     z = D[i][v, u]; valid = (z > 0) & (~M[i][v, u])
```

**What it does.** Defines the reference-flow function. `pts` are `(N, 2)` pixel coordinates in frame `i` in OpenCV's convention (`x = column`, `y = row`), origin at the top-left, `x` rightward, `y` downward; OpenCV and NumPy differ only in index order (`[row, col]` = `[v, u]`). Line 31 rounds each point to the nearest integer pixel and clips into the image so the depth lookup cannot go out of bounds. Line 32 reads the GT depth at that pixel and marks a point valid only if the depth is non-zero and the outlier mask is clear there.

**Alternatives considered.** Bilinear depth interpolation at the sub-pixel location; rejecting points whose 3x3 depth neighbourhood is discontinuous; scoring against a dense GT flow field instead of per-point lookup.

**Why this choice.** `gt_flow` is called on the frame-`i` query points `a` (line 104). Rounding is exact for the five corner/grid-based methods: `goodFeaturesToTrack` returns integer positions (lines 51, 59, 81) and the grid (line 90) is integer by construction. Only `orb_ratio` and `sift_ratio` supply sub-pixel query points (`KeyPoint.pt`, line 74), so for them the depth is read at the nearest pixel. LK's sub-pixel output `p1` is on the `b` side and is never rounded. No design-doc decision covers this; it is an implementation choice of the probe.

### Lines 33-35: back-project, move by GT relative pose

```
33:     X = (Kinv @ np.c_[pts, np.ones(len(pts))].T) * z  # 3,N
34:     T = np.linalg.inv(c2w[j]) @ c2w[i]
35:     Xj = T[:3, :3] @ X + T[:3, 3:4]
```

**What it does.** Line 33 lifts each pixel to a homogeneous ray `Kinv @ [u, v, 1]^T` (normalised image coordinates, `z = 1`) and scales by the GT depth, giving 3D points `X` of shape `(3, N)` in camera `i` (metres). Note that `X` uses the *unrounded* `pts` for the ray and the *rounded* pixel's depth `z`. Line 34 builds the 4x4 relative transform `T = w2c_j @ c2w_i`, which maps camera-`i` coordinates to camera-`j` coordinates (GT pose "i to j"; its rotation block is what `recoverPose` estimates at line 117). Line 35 applies it: `Xj = R X + t`, still `(3, N)`.

**Alternatives considered.** Composing `c2w_i^-1 @ c2w_j` (the opposite direction, a classic c2w/w2c sign error); using a pre-inverted `w2c` list.

**Why this choice.** Direction is fixed by the scoring need: we want where a frame-`i` point lands in frame `j`. The doc's plumbing check ("PnP with GT depth between two frames must reproduce the GT relative pose composed from cam/*.npz") and the later two-view diagnostics' "Conventions verified (synth0 = 0)" (E from GT flow without noise gives 0.01 deg / 0.0 deg error) are the recorded evidence that this composition is the right way round; `e_diag.py` line 97 forms `T` with the identical expression.

### Lines 36-40: cheirality, projection, in-image test

```
36:     valid &= Xj[2] > 1e-6
37:     pj = (K @ Xj); pj = (pj[:2] / np.maximum(pj[2], 1e-6)).T
38:     inside = (pj[:, 0] >= 0) & (pj[:, 0] < W) & (pj[:, 1] >= 0) & (pj[:, 1] < H)
39:     return pj, valid & inside
40: 
```

**What it does.** Line 36 drops points that end up behind camera `j` (`z <= 1e-6 m`, a cheirality guard). Line 37 projects with `K` and perspective-divides, clamping the divisor so a rejected point cannot produce `inf`; `pj` becomes `(N, 2)` pixel coordinates in frame `j`. Line 38 requires the reference location to lie inside the 320x180 image (half-open on the right/bottom). Line 39 returns the reference points and the combined validity mask; the caller (line 104) only scores valid entries. Line 40 is blank.

**Alternatives considered.** Keeping out-of-image references (they are legitimate "the point left the frame" answers, and any method reporting a match there is wrong); a small margin instead of the exact border.

**Why this choice.** Scoring only where the reference is well-defined keeps the "correct" label unambiguous; methods are not penalised for points whose truth is unobservable. This is a probe convention, not a numbered decision.

### Lines 41-42: skew-symmetric helper and `sampson_gt` signature

```
41: def skew(t): return np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
42: def sampson_gt(T, a, b):
```

**What it does.** `skew(t)` builds the 3x3 cross-product matrix `[t]_x` so that `[t]_x v = t x v`. `sampson_gt(T, a, b)` will compute the first-order (Sampson) epipolar distance of correspondences `a` (frame `i`) and `b` (frame `j`) with respect to the GT relative pose `T`.

**Alternatives considered.** `cv2.sampsonDistance` (one pair at a time, slow in Python); algebraic error `b^T F a` (not in pixel units); symmetric epipolar distance.

**Why this choice.** A vectorised NumPy Sampson distance in pixel^2 gives a depth-free "on the GT epipolar line" test; no decision number attaches to it.

### Lines 43-46: fundamental matrix from GT pose and the Sampson distance

```
43:     F = Kinv.T @ skew(T[:3, 3]) @ T[:3, :3] @ Kinv
44:     a1 = np.c_[a, np.ones(len(a))]; b1 = np.c_[b, np.ones(len(b))]
45:     Fa = (F @ a1.T).T; Ftb = (F.T @ b1.T).T; num = np.sum(b1 * Fa, 1) ** 2
46:     return num / (Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2)
```

**What it does.** Line 43 forms the essential matrix `E = [t]_x R` for the i-to-j transform (so that `x_j^T E x_i = 0` in normalised coordinates) and converts it to the pixel-space fundamental matrix `F = K^-T E K^-1` (same `K` on both sides, see line 24). Line 44 homogenises the two point sets to `(N, 3)`. Line 45 computes the epipolar lines `F a` in frame `j` and `F^T b` in frame `i`, and the squared algebraic residual `(b^T F a)^2`. Line 46 returns the Sampson distance: residual divided by the sum of the squared first two components of both lines, which is a first-order approximation to the squared geometric distance, in pixel^2. The caller thresholds it at `< 1` (line 111), i.e. roughly "within 1 px of the GT epipolar line".

**Alternatives considered.** Thresholding at 2 px^2 to match the flow tolerance; normalising by only one line (one-sided distance).

**Why this choice.** Only the fraction below 1 px^2 (`samp1`) is recorded, and it is a secondary diagnostic: it is not one of the columns in the doc's benchmark table (which reports `ncorr/ncorrect`, `rotErr (p90)`, `ms`), so no decision hinges on it. Because `F` is undefined for zero translation, the caller only evaluates it when `|t| > 1e-3 m` (line 110).

### Lines 47-49: rotation angle helper and the methods banner

```
47: def rot_angle(R): return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
48: 
49: # ---------------- methods: each returns (ptsA (N,2), ptsB (N,2)) ----------------
```

**What it does.** `rot_angle` returns the geodesic angle of a rotation matrix in degrees via `trace(R) = 1 + 2 cos(theta)`, clipping the cosine so floating-point round-off cannot push `arccos` out of domain. Line 48 is blank; line 49 is a comment fixing the contract every method below obeys: return two `(N, 2)` float arrays of matched pixel coordinates in frames `i` and `j`, row-aligned.

**Alternatives considered.** `cv2.Rodrigues` and the norm of the rotation vector (same quantity, one more call).

**Why this choice.** This is the metric behind every "rot" number in the doc's benchmark tables; the `rotErr` column of the 1.5 table and the `gt_rot > 0.5` gate (line 114) both use it.

### Lines 50-52: `m_lk` (LK + forward-backward): corner detection

```
50: def m_lk(gi, gj, win=21, lvl=3, fb=1.0):
51:     p0 = cv2.goodFeaturesToTrack(gi, 1000, 0.01, 5)
52:     if p0 is None: return np.zeros((0, 2)), np.zeros((0, 2))
```

**What it does.** Method 1 of 7, name `lk_fb`. Defaults: a 21-px LK window, 3 pyramid levels, a 1 px forward-backward tolerance. Line 51 runs Shi-Tomasi corner detection with `maxCorners=1000`, `qualityLevel=0.01` (a corner is kept if its minimum eigenvalue is at least 1% of the strongest corner's), `minDistance=5` px (non-maximum suppression radius). The return is `(N, 1, 2)` float32 with integer-valued positions, or `None` when no corner passes, which line 52 turns into two empty `(0, 2)` arrays so the caller's `len(a) == 0` branch fires.

**Alternatives considered.** Decision 1.3 lists Shi-Tomasi; FAST; ORB; SIFT/AKAZE; fixed grid. Decision 1.4 lists grid bucketing / top-up for spatial spread.

**Why this choice.** Decision 1.3: "Shi-Tomasi goodFeaturesToTrack(maxCorners=1000, qualityLevel=0.01, minDistance=5) (probe values; cap never reached, ~140-300 corners/frame)"; the RAIL probe measured ~215 corners/frame (min 139). Decision 1.4: no spreading, "minDistance=5 already spreads corners at this resolution". The same three values are frozen in the v0 parameter table ("Max corners 1000"). ORB/SIFT detectors are benchmarked as full methods below (lines 65-77) rather than as LK seeds.

### Lines 53-56: `m_lk`: forward and backward tracking, FB gate

```
53:     p1, st, _ = cv2.calcOpticalFlowPyrLK(gi, gj, p0, None, winSize=(win, win), maxLevel=lvl)
54:     p0b, stb, _ = cv2.calcOpticalFlowPyrLK(gj, gi, p1, None, winSize=(win, win), maxLevel=lvl)
55:     ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm((p0 - p0b)[:, 0], axis=1) < fb)
56:     return p0[ok, 0], p1[ok, 0]
```

**What it does.** Line 53 tracks the corners from `gi` to `gj` with pyramidal Lucas-Kanade: `winSize=(21, 21)`, `maxLevel=3` (both are OpenCV's documented defaults, so the explicit keywords change nothing), `nextPts=None` so the initial guess is the same position (no motion prior), default termination criteria and no `minEigThreshold`. It returns `p1` `(N, 1, 2)` float32 sub-pixel positions in `gj`, `st` `(N, 1)` uint8 status (1 = found), and the per-point error which is discarded. Line 54 tracks the found positions *back* from `gj` to `gi`. Line 55 keeps a correspondence only if both passes succeeded and the round-trip landed within `fb = 1.0` px of the original corner. Line 56 squeezes the middle axis and returns `(N_ok, 2)` pairs.

**Alternatives considered.** Decision 1.5 lists sparse LK with vs without the forward-backward check (the next method), a different window or level count, and a looser FB tolerance. The LK error output (`err`) could gate instead of FB.

**Why this choice.** This is the method decision 1.5 selected: "sparse LK at OpenCV defaults (21 px window, 3 levels) + forward-backward check at 1 px", frozen in the v0 table as "LK window / pyramid levels 21 px / 3 (OpenCV defaults)". Benchmark numbers (doc table): gap 1 `166 / 88` correspondences/correct, rotErr 0.88 deg; gap 4 `103 / 27`, rotErr 2.55 deg (p90 8.4), 4 ms. The doc's verdict: "LK+FB has the best per-correspondence precision at gap 4 (in2 0.35 vs 0.29)" and it "gives corner-anchored persistent tracks the map needs" (decision 1.6 relies on persistent tracks). On consecutive frames the FB check passes 95% of tracks (probe evidence: "median survival 0.95"). Window/levels were never swept: the OpenCV defaults were accepted as-is.

### Lines 57-60: `m_lk_nofb` (LK without the check): detection

```
57: 
58: def m_lk_nofb(gi, gj):
59:     p0 = cv2.goodFeaturesToTrack(gi, 1000, 0.01, 5)
60:     if p0 is None: return np.zeros((0, 2)), np.zeros((0, 2))
```

**What it does.** Line 57 is blank. Method 2, name `lk_nofb`: identical corner detection to lines 51-52 (same three Shi-Tomasi parameters, same `None` guard).

**Alternatives considered.** Sharing the detection through a helper (as `_dense_at_corners` does at line 81). Duplicated so each method is self-contained.

**Why this choice.** Same detector as decision 1.3, so the only variable between `lk_fb` and `lk_nofb` is the backward pass.

### Lines 61-64: `m_lk_nofb`: single forward pass

```
61:     p1, st, _ = cv2.calcOpticalFlowPyrLK(gi, gj, p0, None)
62:     ok = st[:, 0] == 1
63:     return p0[ok, 0], p1[ok, 0]
64: 
```

**What it does.** One forward LK pass at pure OpenCV defaults (`winSize=(21, 21)`, `maxLevel=3`, the same values `m_lk` spells out), keeping every point whose status is 1. No round-trip test, so drifted or occluded points that LK still "finds" are returned. Line 64 is blank.

**Alternatives considered.** The FB-checked variant above; gating on LK's `err` output.

**Why this choice.** The control that measures what the 1 px FB check buys. Doc table: gap 1 `183 / 91`, rotErr 0.85, 2 ms; gap 4 `167 / 31`, rotErr 2.46 (p90 8.1). It returns more correspondences and, on the pooled median, a marginally lower rotation error, but a smaller fraction of them is correct at gap 4 (`31 / 167` vs `27 / 103`); decision 1.5 kept the check because per-point precision matters more than count for map points that will be triangulated and then reused for PnP (decisions 1.6, 2.11).

### Lines 65-68: descriptor detectors and brute-force matchers

```
65: _orb = cv2.ORB_create(nfeatures=1000)
66: _sift = cv2.SIFT_create(nfeatures=1000)
67: _bf_h = cv2.BFMatcher(cv2.NORM_HAMMING)
68: _bf_l2 = cv2.BFMatcher(cv2.NORM_L2)
```

**What it does.** Module-level singletons. `ORB_create(nfeatures=1000)`: FAST keypoints with binary (rBRIEF) descriptors, capped at 1000, other parameters at OpenCV defaults. `SIFT_create(nfeatures=1000)`: DoG keypoints with float descriptors, the 1000 strongest kept. Two brute-force matchers: Hamming distance for the binary ORB descriptors, L2 for SIFT; `crossCheck` is left at its default `False`, which is required for the `k=2` kNN query used by the ratio test.

**Alternatives considered.** Decision 1.5 lists "ORB/SIFT + ratio". Standard further options: AKAZE/BRISK, FLANN matching, cross-check instead of ratio, a different feature cap.

**Why this choice.** These are the two canonical descriptor-based families. `nfeatures=1000` mirrors `maxCorners=1000` of the corner detector; that is the reviewer's reading of the code, the doc records no rationale for the cap. Doc verdict: "descriptor matching rejected by data".

### Lines 69-71: `_match`: detect and describe, empty guard

```
69: def _match(det, bf, gi, gj, ratio=0.75):
70:     k0, d0 = det.detectAndCompute(gi, None); k1, d1 = det.detectAndCompute(gj, None)
71:     if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2: return np.zeros((0, 2)), np.zeros((0, 2))
```

**What it does.** Shared matcher for both descriptor methods, Lowe ratio 0.75. Line 70 runs detection and description on both frames (no mask); `k0, k1` are lists of `cv2.KeyPoint` (sub-pixel `.pt`), `d0, d1` are the descriptor matrices (one row per keypoint), and `None` when nothing was detected. Line 71 returns empty arrays if either side has no descriptors or fewer than two keypoints (a `k=2` query needs at least two train descriptors).

**Alternatives considered.** Ratio 0.7 or 0.8; mutual nearest-neighbour instead of ratio; masking the gripper region at detection time (decision 1.2 rejected masks).

**Why this choice.** 0.75 is Lowe's published value and the one the doc's table rows are labelled with ("ORB + ratio 0.75", "SIFT + ratio 0.75"). Not swept.

### Lines 72-75: `_match`: kNN, ratio test, coordinate extraction

```
72:     mm = bf.knnMatch(d0, d1, k=2)
73:     good = [m[0] for m in mm if len(m) == 2 and m[0].distance < ratio * m[1].distance]
74:     a = np.array([k0[m.queryIdx].pt for m in good]); b = np.array([k1[m.trainIdx].pt for m in good])
75:     return a.reshape(-1, 2), b.reshape(-1, 2)
```

**What it does.** Line 72 finds, for each query descriptor in `d0`, its two nearest train descriptors in `d1`; the result is a list of lists of `DMatch` (a list may be shorter than 2 when the train set is tiny, hence the `len(m) == 2` guard). Line 73 keeps a match only if the best distance is below 0.75x the second best (Lowe's ratio test). Line 74 pulls the keypoint coordinates: `queryIdx` indexes frame `i`'s keypoints, `trainIdx` frame `j`'s; these `.pt` values are sub-pixel, the only sub-pixel query points any method hands to `gt_flow`. Line 75 reshapes so that an empty `good` still yields `(0, 2)` arrays.

**Alternatives considered.** RANSAC-filtered matches (deferred: the essential-matrix RANSAC at line 115 does that for the rotation metric only, so the `ncorrect` count measures raw matcher precision).

**Why this choice.** Raw ratio-test output is what a descriptor frontend would hand to the solver, so that is what is scored. Doc numbers: ORB gap 1 `138 / 77`, rotErr 1.11; gap 4 `57 / 12`, rotErr 3.53 (p90 12.5), 3 ms. SIFT gap 1 `94 / 48`, rotErr 0.95; gap 4 `54 / 12`, rotErr 2.54 (p90 8.7), 19 ms. Both find far fewer correct correspondences than LK on this 320x180 low-texture footage; SIFT is also the slowest method (19 ms vs 4 ms for LK+FB). Decision 1.5: "ORB/SIFT/Farneback rejected". The later two-view diagnostics (third diagnostic table) add that SIFT "only helps at gap 16, and finds enough matches on half the pairs".

### Lines 76-78: the two descriptor methods

```
76: def m_orb(gi, gj): return _match(_orb, _bf_h, gi, gj)
77: def m_sift(gi, gj): return _match(_sift, _bf_l2, gi, gj)
78: 
```

**What it does.** Methods 3 and 4 (`orb_ratio`, `sift_ratio`) bind the singletons to `_match` with the default ratio. Line 78 is blank.

**Alternatives considered.** See lines 65-68.

**Why this choice.** Pairs each descriptor with its correct metric (Hamming for binary, L2 for float); using L2 on ORB or Hamming on SIFT would be a bug.

### Lines 79-82: DIS optical flow and `_dense_at_corners`: detection

```
79: _dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
80: def _dense_at_corners(flow, gi):
81:     p0 = cv2.goodFeaturesToTrack(gi, 1000, 0.01, 5)
82:     if p0 is None: return np.zeros((0, 2)), np.zeros((0, 2))
```

**What it does.** Line 79 creates a Dense Inverse Search flow estimator with the `MEDIUM` preset (OpenCV offers `ULTRAFAST`, `FAST`, `MEDIUM`; the presets fix patch size, stride, pyramid levels and variational-refinement iterations). Lines 80-82 begin a helper that converts any dense flow field into the `(ptsA, ptsB)` contract by sampling it at Shi-Tomasi corners (same detector and parameters as decision 1.3, same `None` guard).

**Alternatives considered.** `ULTRAFAST`/`FAST` presets; RAFT or other learned flow (excluded by the "plain classical" frame, decision 0.1 and the doc's purpose statement); sampling at every pixel.

**Why this choice.** Sampling at the same corners as LK isolates the flow estimator as the single variable (`dis_corners` vs `lk_fb`); `MEDIUM` is the default-quality preset. Not swept.

### Lines 83-85: `_dense_at_corners`: sampling; `m_dis`

```
83:     a = p0[:, 0]; u = np.clip(np.round(a[:, 0]).astype(int), 0, W - 1); v = np.clip(np.round(a[:, 1]).astype(int), 0, H - 1)
84:     return a, a + flow[v, u]
85: def m_dis(gi, gj): return _dense_at_corners(_dis.calc(gi, gj, None), gi)
```

**What it does.** Line 83 squeezes the corners to `(N, 2)` and rounds/clips them to integer pixel indices (row `v`, column `u`), as in `gt_flow`. Line 84 returns the corner and the corner displaced by the flow vector stored at that pixel: OpenCV dense flow is an `(H, W, 2)` float32 array with `flow[v, u] = (dx, dy)` meaning pixel `(u, v)` in the first image corresponds to `(u + dx, v + dy)` in the second, so the addition is in the right order. Nearest-pixel lookup (no bilinear interpolation) is exact because `goodFeaturesToTrack` returns integer positions. Line 85 defines method 5, `dis_corners`: `_dis.calc(prev, next, flow=None)` computes the dense forward flow from `gi` to `gj`.

**Alternatives considered.** Bilinear flow interpolation; forward-backward consistency on the dense field (DIS has no built-in check, so every corner yields a correspondence, valid or not).

**Why this choice.** `dis_corners` isolates the flow estimator (same corners as LK). Doc numbers: gap 1 `187 / 95`, rotErr 0.82, 6 ms; gap 4 `187 / 38`, rotErr 2.27 (p90 8.3). It returns 187 correspondences at both gaps (a dense method never loses a point, so its count is the corner count) and a lower correct fraction at gap 4 than LK+FB (`38 / 187` vs `27 / 103`). The doc records DIS-grid, not DIS-at-corners, as the runner-up (decision 1.5); no separate ruling is recorded for this variant.

### Lines 86-87: `m_farneback`

```
86: def m_farneback(gi, gj):
87:     return _dense_at_corners(cv2.calcOpticalFlowFarneback(gi, gj, None, 0.5, 3, 15, 3, 5, 1.2, 0), gi)
```

**What it does.** Method 6, `farneback_corners`: Farneback polynomial-expansion dense flow sampled at corners. Positional arguments after `flow=None`: `pyr_scale=0.5` (each pyramid level is half the previous), `levels=3`, `winsize=15` px averaging window, `iterations=3` per level, `poly_n=5` pixel neighbourhood for the polynomial fit, `poly_sigma=1.2` Gaussian for that fit, `flags=0` (no initial-flow reuse, no Gaussian window). These are the values used in OpenCV's own optical-flow tutorial.

**Alternatives considered.** The other `poly_n` / `poly_sigma` pairing documented by OpenCV; more levels for large motion; `OPTFLOW_FARNEBACK_GAUSSIAN`.

**Why this choice.** Included as the classic dense-flow baseline at tutorial settings, not tuned. Doc numbers: gap 1 `187 / 92`, rotErr 0.87, 10 ms; gap 4 `187 / 12`, rotErr 3.66 (p90 10.3). The doc's verdict: "Farneback collapses at gap 4" (only 12 of 187 corners within 2 px of GT flow). Rejected by decision 1.5.

### Lines 88-92: `m_dis_grid`

```
88: def m_dis_grid(gi, gj):
89:     flow = _dis.calc(gi, gj, None)
90:     vv, uu = np.mgrid[4:H:8, 4:W:8]; a = np.c_[uu.ravel(), vv.ravel()].astype(np.float32)
91:     return a, a + flow[vv.ravel(), uu.ravel()]
92: 
```

**What it does.** Method 7, `dis_grid`: the same DIS flow as `m_dis`, but sampled on a fixed lattice instead of at corners. Line 90 builds integer grid coordinates every 8 px starting at 4 (rows `4, 12, ..., 172`, columns `4, 12, ..., 316` at 180x320), i.e. 22 x 40 = 880 points, and stacks them as `(880, 2)` float32 `(x, y)`. Line 91 returns each grid point and its flow-displaced location. There is no texture or validity test: uniform regions and the gripper contribute correspondences too. Line 92 is blank.

**Alternatives considered.** A finer grid (4 px); grid points restricted by a gradient threshold; decision 1.4's "fixed grid" detector option.

**Why this choice.** Tests whether *point count* rather than *point quality* drives the two-view rotation estimate. Doc numbers: gap 1 `880 / 478`, rotErr 0.75, 5 ms; gap 4 `880 / 192`, rotErr 1.88 (p90 6.9), the best rotation error of the seven at both gaps; paired gap-4 comparison "DIS-grid better on 848 pairs, LK better on 523". Decision 1.5's reading: "DIS-grid best E-rotation via 5x points" (880 vs 166 / 103 correspondences for LK+FB at gaps 1 / 4), while LK+FB has the better per-point precision (`192 / 880` vs `27 / 103` at gap 4). Ruling: LK+FB is the pick; "DIS-grid recorded as measured runner-up = first v1 frontend swap". The later two-view diagnostics temper the runner-up: DIS 8-px grid gives E(2px) rot/tdir `1.70 / 26.4` at gap 4 but `4.15 / 40.4` at gap 8 and `12.05 / 66.1` at gap 16, worse than chained LK beyond gap 4.

### Lines 93-95: the method registry

```
93: METHODS = {"lk_fb": m_lk, "lk_nofb": m_lk_nofb, "orb_ratio": m_orb, "sift_ratio": m_sift,
94:            "dis_corners": m_dis, "dis_grid": m_dis_grid, "farneback_corners": m_farneback}
95: 
```

**What it does.** Ordered name-to-function map; the names are the keys of the output JSON and the row order of the doc's table (LK + fwd-bwd, LK no check, ORB, SIFT, DIS at corners, DIS on grid, Farneback). Line 95 is blank.

**Alternatives considered.** Selecting methods from the CLI. All seven always run, so every method is scored on the identical pair set.

**Why this choice.** Decision 1.5's option list is exactly "sparse LK (+/- fwd-bwd check); ORB/SIFT + ratio; DIS/Farneback dense flow at corners or on a grid"; the registry is that list.

### Line 96: per-method, per-gap accumulators

```
96: res = {m: {g: dict(pairs=0, ncorr=[], ncorrect=[], epe=[], in1=[], in2=[], static=[], samp1=[], survival=[], time=[], rot_err=[], rot_pairs=0, e_inl=[]) for g in gaps} for m in METHODS}
```

**What it does.** Nested dict `res[method][gap]` holding, per evaluated pair, one appended scalar per metric: `ncorr` (correspondences returned), `ncorrect` (within 2 px of GT flow), `epe` (median end-point error, px), `in1` / `in2` (fraction within 1 / 2 px), `static` (fraction with displacement < 1 px), `samp1` (fraction with Sampson distance < 1 px^2 vs GT geometry), `survival` (LK only: correspondences / detected corners), `time` (wall seconds), `rot_err` (E-matrix rotation error, deg), `e_inl` (E RANSAC inlier fraction); plus counters `pairs` and `rot_pairs`.

**Alternatives considered.** A pandas frame; writing per-pair rows to CSV.

**Why this choice.** Plain lists keep the probe dependency-free and let the summary (lines 121-125) emit both medians and the raw arrays needed for the doc's cross-scene pooling and paired comparisons.

### Lines 97-99: pair enumeration and GT relative pose

```
97: for g in gaps:
98:     for i in range(0, n - g, 2):
99:         j = i + g; T = np.linalg.inv(c2w[j]) @ c2w[i]; gt_rot = rot_angle(T[:3, :3])
```

**What it does.** For each gap, walks start frames `i = 0, 2, 4, ...` with `i + g < n` (stride 2 halves the pair count), sets `j = i + g`, forms the GT i-to-j transform exactly as in `gt_flow` (line 34), and its rotation magnitude in degrees. Over the 12 smoke scenes (4243 frames) this yields the "~2100 pairs per gap" the doc pools over.

**Alternatives considered.** Every start frame (2x the cost, near-duplicate pairs at gap 1); random pair sampling.

**Why this choice.** Stride 2 is an implementation choice of the probe with no recorded rationale in the design doc; it halves the pair count (adjacent start frames are strongly correlated at the doc's per-step motion median of 9 mm / 1.3 deg) and is what produced the ~2100 pairs per gap the doc pools over. As a fact of the launcher, `corr_bench.sbatch` requests `--time=00:30:00` on `acc_debug`; the doc does not connect the stride to that budget.

### Lines 100-103: run each method, time it, count

```
100:         for name, fn in METHODS.items():
101:             r = res[name][g]; t0 = time.time(); a, b = fn(G[i], G[j]); r["time"].append(time.time() - t0)
102:             r["pairs"] += 1; r["ncorr"].append(len(a))
103:             if len(a) == 0: continue
```

**What it does.** Calls every method on the grayscale pair, timing the full call (detection + matching/tracking, wall clock, single-threaded per line 14). Records the pair and the number of correspondences, and skips all scoring when the method returned none (the pair still counts in `pairs` and contributes a 0 to `ncorr`).

**Alternatives considered.** `time.perf_counter()`; excluding detection from the timing.

**Why this choice.** Wall time including detection is what a frontend costs per frame; the doc's `ms` column (4 / 2 / 3 / 19 / 6 / 5 / 10 ms for the seven rows) comes from this measurement.

### Lines 104-108: flow-based scoring

```
104:             pj, valid = gt_flow(i, j, a)
105:             if valid.sum() > 0:
106:                 e = np.linalg.norm(b[valid] - pj[valid], axis=1)
107:                 r["epe"].append(float(np.median(e))); r["in1"].append(float((e < 1).mean())); r["in2"].append(float((e < 2).mean()))
108:                 r["ncorrect"].append(int((e < 2).sum()))
```

**What it does.** Computes the reference location of every returned frame-`i` point (line 104) and, where the reference is valid, the Euclidean end-point error in pixels between the method's frame-`j` point and the GT one (line 106). Line 107 records the per-pair median EPE and the fractions below 1 px and 2 px *among valid points*; line 108 records the absolute count below 2 px, the `ncorrect` of the doc table. Pairs with no valid reference point contribute nothing to these four lists (but still to `ncorr`, `static`, timing).

**Alternatives considered.** A 1 px "correct" threshold (`in1` is also stored); scoring with `ncorrect` over *all* returned points rather than valid-reference ones.

**Why this choice.** 2 px is the "correct" tolerance that the doc's table reports (`ncorrect = within 2 px of GT flow`) and the same unit that decision 2.7 adopts for the PnP RANSAC threshold: "~65% of consecutive-frame tracks fall within 2 px of GT (corr benchmark); same unit as the benchmarks' 'correct' definition". At gap 4 the pooled `in2` is the basis of decision 1.5's precision statement ("in2 0.35 vs 0.29"). Because validity depends on GT depth being defined, and GT depth "is defined ON the gripper in many scenes", the two-view diagnostics warn that "the corr-benchmark in2/EPE numbers on high-gripper scenes are contaminated by that".

### Lines 109-111: zero-motion share and Sampson inlier fraction

```
109:             disp = np.linalg.norm(b - a, axis=1); r["static"].append(float((disp < 1).mean()))
110:             if np.linalg.norm(T[:3, 3]) > 1e-3:
111:                 r["samp1"].append(float((sampson_gt(T, a.astype(np.float64), b.astype(np.float64)) < 1).mean()))
```

**What it does.** Line 109 measures the per-pair fraction of correspondences whose own displacement is under 1 px, regardless of GT: on a moving camera this is the share of correspondences anchored to something image-static (the wrist-mounted gripper, or a detector artefact). Lines 110-111 evaluate the GT-epipolar Sampson test only when the GT translation exceeds `1e-3` m (`F` is undefined at zero baseline) and record the fraction below 1 px^2; casting to float64 avoids float32 round-off in the `F` products.

**Alternatives considered.** Decision 1.2's masking options (none; fixed polygon; per-scene temporal-variance/Otsu mask; flow-based rejection; GT outlier mask) all target the same quantity. A relative static definition (< 20% of the median displacement) was tried in the later two-view diagnostics.

**Why this choice.** `static` is the gripper diagnostic the doc reports from this benchmark: "Static (gripper) share of LK tracks: 16-53% per scene, median ~25%". It motivated the `reject_static_tracks` lever of decision 1.2 (drop tracks < 1 px on frames whose median flow >= 2 px), whose default was flipped to ON on 2026-09-05 after the full-pipeline sweeps (v0 final `111.9 / 6.93 / 0.871` ON vs `115.6 / 8.40 / 0.971` OFF). `samp1` is a diagnostic only; no reported number depends on it.

### Lines 112-113: LK survival

```
112:             if name in ("lk_fb", "lk_nofb"):
113:                 p0 = cv2.goodFeaturesToTrack(G[i], 1000, 0.01, 5); r["survival"].append(len(a) / max(1, len(p0)))
```

**What it does.** For the two LK methods only, re-runs the same Shi-Tomasi detection on frame `i` to recover the denominator (the methods return only surviving points) and records the fraction of detected corners that produced a correspondence. `max(1, ...)` is a divide-by-zero guard that is effectively dead: `goodFeaturesToTrack` returns `None` (never an empty array) when no corner passes, and the same detection on the same frame inside the method would then have returned empty arrays and hit `continue` at line 103, so `len(p0) >= 1` whenever this line runs.

**Alternatives considered.** Returning the detected count from the method itself (would break the uniform `(ptsA, ptsB)` contract).

**Why this choice.** Survival is what decision 1.6 (persistent tracks, top-up only at keyframes) rests on: the doc's probe evidence gives "median survival 0.95" for consecutive-frame LK+FB and "~15%" over a 15-frame gap in fast segments, hence "tracking must be frame-to-frame". Only meaningful for sparse trackers; descriptor and dense methods do not have a survival notion in this sense.

### Lines 114-118: essential matrix and rotation error

```
114:             if gt_rot > 0.5 and len(a) >= 8:
115:                 E, inl = cv2.findEssentialMat(a.astype(np.float64), b.astype(np.float64), K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
116:                 if E is not None and E.shape == (3, 3) and inl is not None:
117:                     _, R, t, _ = cv2.recoverPose(E, a.astype(np.float64), b.astype(np.float64), K, mask=inl.copy())
118:                     r["rot_err"].append(float(rot_angle(R.T @ T[:3, :3]))); r["rot_pairs"] += 1; r["e_inl"].append(float(inl.mean()))
```

**What it does.** Gate (line 114): GT rotation above 0.5 deg (so the essential matrix is not degenerate) and at least 8 correspondences (a conservative floor; OpenCV's 5-point solver needs 5). Line 115 fits `E` with RANSAC from pixel coordinates and the calibrated `K` (OpenCV normalises internally, so `threshold=1.0` is a 1 px epipolar-distance threshold on the pixel-space points), confidence 0.999, default max iterations. Returns `E` (3x3, or `None`, or a vertically stacked array when the solver yields several candidates, hence the shape check on line 116) and `inl`, an `(N, 1)` uint8 inlier mask. Line 117 disambiguates the four `(R, t)` decompositions by cheirality on the inliers; `recoverPose` *modifies* its `mask` argument in place, so a copy is passed to keep `inl` intact for the inlier-fraction statistic. The recovered `R` maps camera-`i` coordinates to camera-`j` coordinates, the same direction as the GT `T` (line 99), so line 118's `rot_angle(R.T @ R_gt)` is the residual rotation between estimate and truth in degrees; `t` (unit norm) is discarded, and the RANSAC inlier fraction is recorded.

**Alternatives considered.** Decision 2.10 lists: essential matrix; homography; H-vs-E model selection; RANSAC threshold 2 / 1 / 0.5 px; the two-view diagnostics also tried MAGSAC and a 2-view Sampson refinement. Scoring translation direction too (done in the e_diag probe, not here).

**Why this choice.** Rotation error of a two-view fit is the most solver-independent way to rank *correspondence sets* for pose, and the doc's `rotErr (p90)` columns (e.g. LK+FB 0.88 / 2.55 (8.4); DIS-grid 0.75 / 1.88 (6.9)) come from these lines. The 1 px threshold used here is *not* the pipeline's: decision 2.10 later settled the bootstrap threshold at 0.5 px from the separate two-view diagnostics (`2 -> 0.5 px cuts two-view rot error 2.06 -> 1.23 deg` at gap 4 on chained tracks); this file's rotation numbers are therefore method-comparison numbers at 1 px, not the pipeline's absolute performance. The 0.5 deg gate is not the same as the doc's later "parallax-gated" filter (median flow >= 2 px, n = 814 at gap 4); that filter and the "LK minus zero-motion tracks 1.70 / 7.0" and "DIS-grid minus zero-motion 1.15 / 4.3" rows quoted under the benchmark heading were produced by follow-up analysis, not by this file as committed (the copy in the Slurm output directory, `.../opencv_vo_probe/corr_bench/corr_bench.py`, is byte-identical to the repo copy and contains no static-track removal).

### Lines 119-120: median helper

```
119: 
120: def agg(v, f=np.median): return float(f(v)) if len(v) else None
```

**What it does.** Line 119 is blank. `agg` applies a reducer (median by default) to a per-pair list and returns a Python float, or `None` when the list is empty (e.g. `survival` for non-LK methods, `rot_err` for a scene with no rotating pair), so the JSON stays valid.

**Alternatives considered.** Mean; NaN instead of `None`.

**Why this choice.** Medians are the doc's reporting statistic throughout ("Medians pooled over ~2100 pairs per gap"); they are robust to the heavy-tailed rotation errors seen in the p90 column (up to 12.5 deg for ORB at gap 4 against a 3.53 median).

### Lines 121-125: per-scene summary

```
121: summary = {m: {g: dict(pairs=r["pairs"], ncorr_med=agg(r["ncorr"]), ncorrect_med=agg(r["ncorrect"]), static_med=agg(r["static"]), samp1_med=agg(r["samp1"]), epe_med=agg(r["epe"]), in1_med=agg(r["in1"]), in2_med=agg(r["in2"]),
122:                        survival_med=agg(r["survival"]), ms_med=1000 * (agg(r["time"]) or 0), rot_pairs=r["rot_pairs"],
123:                        rot_err_med=agg(r["rot_err"]), rot_err_p90=agg(r["rot_err"], lambda x: np.percentile(x, 90)), e_inl_med=agg(r["e_inl"]),
124:                        raw=dict(ncorr=r["ncorr"], ncorrect=r["ncorrect"], epe=r["epe"], in2=r["in2"], static=r["static"], samp1=r["samp1"], rot_err=r["rot_err"]))
125:                for g, r in rg.items()} for m, rg in res.items()}
```

**What it does.** Builds `summary[method][gap]` with: pair count; per-scene medians of every per-pair metric (`ncorr_med`, `ncorrect_med`, `static_med`, `samp1_med`, `epe_med`, `in1_med`, `in2_med`, `survival_med`); median runtime converted to milliseconds (with `or 0` so an empty list gives 0 rather than a `None` multiplication error); the number of pairs that produced a rotation error, its median and its 90th percentile; the median E inlier fraction; and a `raw` sub-dict with the full per-pair arrays for `ncorr, ncorrect, epe, in2, static, samp1, rot_err` (not `time`, `in1`, `survival`, `e_inl`).

**Alternatives considered.** Per-scene means; emitting only medians (would prevent cross-scene pooling and the paired DIS-vs-LK test); emitting every raw list; storing a pair index alongside each raw value.

**Why this choice.** The doc's table is pooled across the 12 scene JSONs at the pair level, and its "Paired gap-4 rotErr: DIS-grid better on 848 pairs, LK better on 523" needs per-pair `rot_err` arrays aligned across methods. All methods see the same pair sequence, so the raw lists align except where a method returned no correspondences (line 103) or fewer than 8 (line 114), or where `findEssentialMat` returned `None` / a non-3x3 stack (line 116); the `gt_rot > 0.5` gate drops the same pairs for every method, so it does not break alignment by itself. Because `rot_err` carries no pair index in the JSON, the paired comparison requires re-deriving the alignment from the per-method drop cases. The p90 column of the table (`rot_err_p90`) is computed here.

### Lines 126-127: write JSON, print

```
126: json.dump(dict(scene=os.path.basename(scene), n=n, summary=summary), open(out, "w"))
127: print(os.path.basename(scene), n, "done")
```

**What it does.** Serialises `{scene, n, summary}` to the output path given on the command line (the launcher uses `<probe_dir>/corr/<scene>.json`) and prints a one-line completion marker that the launcher redirects into `<scene>.log`. The file handle is left to the interpreter to close at exit.

**Alternatives considered.** A `with` block; NPZ output; appending to a shared results file (would need locking across the 12 parallel processes).

**Why this choice.** One JSON per scene per process is the simplest lock-free layout for the parallel launcher; the design doc records the location: `/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/corr_bench/`.

### Known limitations / honesty notes

- **GT-scored on the reported test scenes.** The 12 smoke scenes are the same scenes on which the OpenCV arm is later reported; honesty audit items 7 ("all thresholds tuned on the 12 reported (test) scenes") and 8 ("design benchmarks GT-scored on test scenes") cover this file. The doc's mitigation for item 7 is the frozen-configuration run on all 4292 scenes; item 8 is pending the user's ruling.
- **Reference validity is not a gripper oracle.** `gt_flow` trusts any pixel with non-zero GT depth outside `outlier_mask`, but per the doc "GT outlier_mask does NOT cover the gripper" and "GT depth is defined ON the gripper in many scenes", so on high-gripper scenes the `in2` / EPE / `ncorrect` numbers count gripper correspondences as scored points; the doc explicitly flags those numbers as "contaminated by that".
- **Different frame and intrinsics from the pipeline.** Scoring runs on native 320x180 frames with calibrated K (fx ~ 192 px); the frozen pipeline runs on 320x192 cover frames with the model's per-frame focal (205-221 px in the inference-time table). Pixel-unit metrics here are therefore in a frame 1.067x smaller than the pipeline's, and one `K` (frame 0's) is applied to the whole scene although the dataset stores intrinsics per frame.
- **The rotation metric is at RANSAC 1 px, not the pipeline's 0.5 px.** Decision 2.10's threshold came from the separate two-view diagnostics; `rotErr` here ranks methods and should not be read as the bootstrap's absolute accuracy. The "parallax-gated" and "minus zero-motion tracks" rows under the doc's corr-benchmark heading are not produced by this file.
- **Only gaps 1 and 4 were run.** The later diagnostics show the ranking changes with gap (DIS-grid degrades sharply by gap 8-16; SIFT only helps at gap 16), so the runner-up status of DIS-grid holds at the keyframe-scale gap only.
- **Timing is single-core wall clock including detection**, with 12 processes sharing one node; it ranks methods but is not a clean per-method latency.
- **Dense methods never lose points**: `ncorr` for `dis_*` and `farneback_corners` equals the corner (187) or grid (880) count at both gaps, so only their `ncorrect` / `in2` and `rotErr` are comparable with the sparse methods.
- **Sub-pixel query points are rounded for ORB/SIFT only.** The depth lookup in `gt_flow` rounds to the nearest pixel; this is exact for the corner- and grid-based methods and an approximation only for the two descriptor methods, whose `KeyPoint.pt` values are sub-pixel.
- **No translation-direction scoring** in this file (recovered `t` is discarded); the doc's ~30-40 deg translation-direction errors come from the two-view diagnostics probe.
- **Not causal, by design.** This is a design-time measurement; nothing here runs at inference, so the causality rulings (audit item 1, retro-fill) do not apply to it.
