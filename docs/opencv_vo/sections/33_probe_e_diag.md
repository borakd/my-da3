## `eval_pipeline/opencv_vo_probes/e_diag.py`, lines 1-138: two-view essential-matrix diagnostic

This is the whole file: a self-contained probe, not part of the runtime pipeline. It exists to answer one question raised by the triangulation-acceptance benchmark (decision 2.11): the map arm's bottleneck was the two-view bootstrap pose, whose translation direction came out "~40 deg off at 36 mm baselines" (design doc, "KEY FINDING" under the 2.11 benchmark; the doc's findings under the e_diag table call it "the ~30 deg translation-direction error"; the file's own docstring says "~35 deg", a figure that appears nowhere in the design doc). The probe takes one DROID scene, forms frame pairs `(i, i+GAP)`, tracks Shi-Tomasi corners across the gap with chained Lucas-Kanade (the pipeline's frontend, decisions 1.3/1.5, with the static-track lever of 1.2 ON and the 2 px parallax gate), and then scores many *different* two-view pose estimates on the *same* correspondences against the ground-truth relative pose: RANSAC at 2 / 1 / 0.5 px, MAGSAC, a Sampson-distance refinement, a relative-lever variant, two "oracle" masks, a GT-rotation solve, and — the key control — the same solver fed synthetic correspondences generated from GT depth + GT pose with 0 or 1 px Gaussian noise, which isolates the geometric conditioning of the motion from the quality of the tracks. Its output JSONs (one per scene per gap; `e_diag.sbatch` runs gaps 4, 8, 16 over the 12 smoke scenes) are the source of the "Two-view (essential matrix) diagnostics" table in the design doc (Slurm 45443683 / 45443704, ~1780 gated pairs per gap), and that table is what changed decision 2.10 from a 2 px to a 0.5 px bootstrap RANSAC threshold. In the pipeline it sits *upstream* of everything: the E-matrix bootstrap (2.9/2.10) sets the map's first 3D points, and every later PnP registers to that map.

Note on units and frames: unlike the pipeline (decision 1.1: 320x192 cover frames, model focal per 0.1b), this probe reads the native 320x180 PNGs and the *calibrated* K from `cam/*.npz` (fx ~ 192 px; dataset facts in the design doc). Decision 0.1b's re-check found the focal source immaterial in the full pipeline, so the numbers transfer; but pixel thresholds here are native-frame pixels.

---

### Lines 1-2: shebang and docstring title

```
1: #!/usr/bin/env python
2: """Why is the essential-matrix translation direction ~35 deg off?  Diagnostic for 2.12 / 2.10.
```

**What it does.** Interpreter line and the opening of the module docstring. The docstring states the question and ties the file to decisions 2.12 (keyframe trigger) and 2.10 (bootstrap geometry). The docstring's "~35 deg" has no counterpart in the design doc: the doc reports ~40 deg on pair 0-15 and 40-80 deg on fast pairs in the RAIL probe (2026-09-04), and this script's pooled medians for the pipeline configuration are 31.6 / 31.0 / 41.0 deg at gaps 4 / 8 / 16.

**Alternatives considered.** None — this is documentation.

**Why this choice.** The tie to 2.10/2.12 is the reason the probe was written: the 2.11 benchmark showed the admitted map is ~53% bad and PnP against it lands at ~4 deg at a 4-frame gap versus 0.56 deg with the GT pose on the same tracks, so the two-view pose, not the filter, had to be diagnosed.

### Lines 3-8: docstring, setup and the first two estimators

```
3: 
4: Pairs (i, j=i+GAP), chained LK tracks with the static lever ON (disp>=1 px), parallax gated.
5: For each pair, several two-view pose estimates are scored vs the GT relative pose
6: (rotation error deg, translation-direction error deg):
7:   e2     : findEssentialMat RANSAC 2 px (pipeline)            + recoverPose
8:   e1     : RANSAC 1 px
```

**What it does.** Blank line 3, then the experimental frame: pairs are `(i, i+GAP)`, correspondences are chained frame-to-frame LK (line 47-51), the static lever is ON (absolute 1 px floor, line 94), and pairs are parallax-gated (median displacement >= 2 px, line 93). Every estimate is scored by two numbers: rotation error in degrees and translation-*direction* error in degrees (the essential matrix gives `t` only up to scale, so magnitude cannot be scored). `e2` is labelled "(pipeline)" because at the time of writing decision 2.10 used 2 px; `e1` is the 1 px variant.

**Alternatives considered.** Scoring translation as a metric length (impossible for E), or scoring only rotation as the correspondence benchmark did (decision 1.5 table reports `rotErr` only). Non-gated pairs (the model rows of the design doc table are un-gated).

**Why this choice.** Rotation + t-direction is the complete observable content of an essential matrix. Gating is required because on a stationary camera E is degenerate ("Frames 0-15: camera stationary ... Essential matrix degenerate there", probe evidence) and t-direction is undefined when `|t_gt| ~ 0`. Lever ON matches the pipeline default flipped on 2026-09-05 (decision 1.2). The "(pipeline)" label is historical: decision 2.10 was revised to 0.5 px on 2026-09-05 *because of this script* (2.06 -> 1.23 deg rotation, 31.6 -> 18.1 deg t-dir at gap 4).

### Lines 9-13: docstring, the threshold sweep, MAGSAC and the synthetic controls

```
9:   e05    : RANSAC 0.5 px
10:   magsac : cv2.USAC_MAGSAC, 2 px
11:   synth1 : SAME points, but correspondences replaced by GT flow (GT depth + GT pose) + 1 px Gaussian noise
12:            -> geometric conditioning of this motion, independent of the frontend
13:   synth0 : GT flow with no noise (sanity: should be ~0)
```

**What it does.** Names three more arms. `e05` completes the 2 / 1 / 0.5 px threshold ladder. `magsac` swaps the robust estimator for OpenCV's USAC MAGSAC++ at the same 2 px. `synth1` / `synth0` keep the *same pixel locations* in frame `i` but replace the tracked location in frame `j` by the GT-flow location (unproject with GT depth, transform with GT relative pose, reproject), with 1 px or 0 px isotropic Gaussian noise. `synth0` is a convention check (it must give ~0 error if the pose composition in line 97 and OpenCV's `recoverPose` agree); `synth1` measures how well *this motion* can be resolved at all with 1 px-accurate correspondences.

**Alternatives considered.** Design-doc options for 2.10: essential matrix vs homography vs H-vs-E model selection, and RANSAC threshold 2 / 1 / 0.5 px. For the estimator: RANSAC vs USAC/MAGSAC++ vs none (decision 2.6's option list for PnP, reused here). For the control: one could also perturb with larger noise — the third diagnostic in the design doc adds a 3 px noise row (2.20 / 29.0, 2.82 / 14.7, 3.55 / 9.1 at gaps 4 / 8 / 16).

**Why this choice.** Results settled 2.10: RANSAC 0.5 px = 1.23 / 18.1 (gap 4), 2.88 / 23.5 (gap 8), 8.11 / 36.8 (gap 16) versus 2 px = 2.06 / 31.6, 3.80 / 31.0, 8.22 / 41.0. MAGSAC 2 px = 1.34 / 22.9, 2.87 / 24.2, 7.57 / 37.0 — better than RANSAC 2 px but not better than RANSAC 0.5 px at gap 4, so the doc kept plain RANSAC with the tighter threshold ("Only the sub-pixel minority of chained LK tracks is geometrically clean"). synth0 = 0.01 / 0.0 at every gap ("Conventions verified (synth0 = 0)"). synth1 = 1.43 / 13.8, 1.67 / 7.4, 1.95 / 4.2: t-direction improves with baseline, so "Geometry is well conditioned"; real tracks get *worse* with gap because chained LK drift grows with chain length — the finding behind "Keyframes must be close (gap 2-4), not far" (decision 2.12).

### Lines 14-18: docstring, refinement, GT-rotation solve and the recorded motion statistics

```
14:   refine : e2 inliers, then scipy least_squares on (rotvec, t-direction) minimizing Sampson distance
15:   gtR_t  : rotation fixed to GT, translation direction by least squares on the epipolar constraint
16:            (isolates whether t is bad because R is bad)
17: Also records GT motion: rotation deg, translation mm, and the ratio of median translational parallax to
18: median rotational flow (rotation-dominance).
```

**What it does.** `refine` takes the RANSAC-2 px inlier set and polishes the 5-DOF pose (3 rotation-vector components + 2 spherical angles for a unit `t`) by nonlinear least squares on the first-order geometric (Sampson) error. `gtR_t` fixes `R` to ground truth and solves for `t` linearly from the epipolar constraint, intended to test whether the bad `t` is a consequence of a bad `R`. Lines 17-18 announce three per-pair GT statistics: rotation angle (deg), translation magnitude (mm), and the "rotation-dominance" ratio. Note the docstring's wording is inverted relative to the code: line 110 stores `rf / tf`, i.e. rotational flow over translational parallax, so larger = more rotation-dominated.

**Alternatives considered.** Refinement: none (OpenCV's `recoverPose` output as-is), Sampson-distance LM (chosen here), full two-view bundle adjustment with 3D points, or an 8-point/algebraic re-fit on inliers. Isolating `t`: fixing `R` (chosen), or comparing t-direction error conditioned on rotation-error bins.

**Why this choice.** Refinement is a standard textbook step; it was tried and measured: 1.70 / 29.0, 3.38 / 30.0, 7.80 / 39.8 versus 2.06 / 31.6, 3.80 / 31.0, 8.22 / 41.0 unrefined — a small rotation gain and essentially no t-direction gain, so decision 2.10 stayed with findEssentialMat + recoverPose (with a tighter threshold) rather than adding a refinement step. (Decision 3.4, "none for v0", concerns trajectory-level BA / pose-graph refinement and is decided on the local-BA sweep — ATE 128.8 / 130.7 vs 122.9 — not on this row.) **`gtR_t` is INVALID**: the design doc states "The gtR_t rows in the JSON are INVALID (sign-ambiguous linear solver); ignore them" — see line 76-83. The rotation-dominance ratio is recorded but not tabulated in the design doc; the qualitative statement it supports is in the probe evidence: "Translation direction is poorly observable at this baseline; rotation dominates the flow."

### Lines 19-20: usage line and docstring close

```
19: Usage: e_diag.py <scene_dir> <out_json> [GAP]
20: """
```

**What it does.** Three positional CLI arguments: the scene directory (the one containing `dense/`), the output JSON path, and an optional integer gap (default 8, line 28). `e_diag.sbatch` invokes it with gaps 4, 8 and 16 for every scene in `eval_pipeline/cg_smoke_scenes_12.txt`, each (scene, gap) as its own background process (`&` ... `wait`), i.e. 36 processes at once. The sbatch runs the copy of this file under `/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/e_diag/` (`$G/e_diag.py`, byte-identical to the repo file) as an `acc_debug` job on partition `acc` (`--gres=gpu:1`, `--cpus-per-task=48`, `--time 00:45:00`), even though the probe is CPU-only, and writes JSON and log to `$G/out/`.

**Alternatives considered.** argparse; a single multi-gap run per scene. Not worth it for a one-shot probe.

**Why this choice.** Per-(scene, gap) processes are embarrassingly parallel and each writes its own JSON, so a crash loses one cell of the table. The 12 smoke scenes are decision 0.3 ("smoke on eval_pipeline/cg_smoke_scenes_12.txt").

### Lines 21-25: imports

```
21: import sys, os, glob, json
22: import numpy as np, cv2
23: cv2.setNumThreads(1)
24: from scipy.optimize import least_squares
25: from scipy.spatial.transform import Rotation as Rot
```

**What it does.** Standard library, NumPy, OpenCV (4.11.0 in the `cuteanything` env), SciPy's Levenberg-Marquardt/TRF least squares and its rotation-vector <-> matrix conversion. `cv2.setNumThreads(1)` disables OpenCV's internal thread pool so that the 36 concurrent processes (3 gaps x 12 scenes) on one 48-core node do not oversubscribe cores; the sbatch also sets `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

**Alternatives considered.** Letting OpenCV pick its thread count; running scenes sequentially.

**Why this choice.** Single-thread cv2 with many processes is the standing rule for these probes (memory: "Slurm acc_debug, GPFS staging, single-thread cv2"); the login node has a 300 s per-process CPU cap (design doc, "Plumbing"), so the probe must run as a Slurm job.

### Lines 26-28: CLI parsing and default gap

```
26: 
27: scene, out = sys.argv[1], sys.argv[2]
28: GAP = int(sys.argv[3]) if len(sys.argv) > 3 else 8
```

**What it does.** Blank line 26; positional args as documented. `GAP` is the frame offset `j - i`; default 8.

**Alternatives considered.** Gaps 2 / 4 / 8 / 16 are the ones studied across the design doc (keyframe-trigger sweep uses stride 2 / 4 / 8; this probe uses 4 / 8 / 16).

**Why this choice.** Gap 4 is the keyframe-scale baseline (~36 mm, 2.11 benchmark) where the problem was first seen; 8 and 16 test whether a longer baseline helps (it would if the limitation were geometric) or hurts (it does if the limitation is track drift). The answer — real tracks get worse with gap while synthetic ones get better — is the core finding.

### Lines 29-31: scene paths, images

```
29: d = os.path.join(scene, "dense")
30: fs = sorted(glob.glob(os.path.join(d, "rgb", "*.png"))); n = len(fs)
31: G = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in fs]; H, W = G[0].shape
```

**What it does.** Points at `<scene>/dense/` (layout `dense/{rgb,cam,depth,outlier_mask,sky_mask}`, dataset facts). Loads every RGB PNG as an 8-bit grayscale `uint8` array; `n` is the sequence length (57 to ~1700 frames). `H, W` = 180, 320 in native frames. `sorted()` on zero-padded names gives frame order.

**Alternatives considered.** Decision 1.1 options: gray only; 2x upsample; CLAHE; undistort. The pipeline additionally uses the eval loader's 320x192 cover frames.

**Why this choice.** Decision 1.1: "grayscale conversion only" — gray is mandatory for Shi-Tomasi/LK and nothing else was shown to help. The probe deliberately reads native frames because it needs pixel-aligned GT depth and calibrated K for the synthetic correspondences (lines 101-104), which exist only in the native 320x180 frame.

### Lines 32-34: intrinsics and GT poses

```
32: cams = [np.load(os.path.join(d, "cam", "%06d.npz" % i)) for i in range(n)]
33: K = cams[0]["intrinsic"].astype(np.float64); Kinv = np.linalg.inv(K)
34: c2w = [c["pose"] for c in cams]
```

**What it does.** Loads every `cam/%06d.npz`. `K` is the 3x3 pinhole matrix of frame 0 (fx ~ 192 px, no distortion per the dataset facts); the code uses frame 0's K for the whole scene even though the npz files carry one per frame, `Kinv` its inverse for pixel -> normalised-ray conversion. `c2w` is the list of 4x4 **camera-to-world** GT poses (key `pose`), i.e. columns are the camera axes expressed in world coordinates and the last column is the camera centre.

**Alternatives considered.** Decision 0.1b options: calibrated K (used here), one nominal dataset K, model-estimated focal, self-calibration.

**Why this choice.** For a *diagnostic* the calibrated K is the right reference (0.1b keeps "Calibrated-K copy ... as a one-off diagnostic"); the pipeline itself may not use it (closed-loop rule). The full-pipeline re-check found the focal source immaterial (finetuned focal 111.9 / 6.93 / 0.871, nominal 203 px 109.1 / 7.04 / 0.869, calibrated 116.0 / 6.91 / 0.885), so nothing here hinges on it. c2w is the storage convention; the world-to-camera inverse is taken explicitly at line 97.

### Lines 35-37: GT depth, outlier mask, RNG

```
35: D = [np.load(os.path.join(d, "depth", "%06d.npy" % i)) for i in range(n)]
36: M = [cv2.imread(os.path.join(d, "outlier_mask", "%06d.png" % i), cv2.IMREAD_GRAYSCALE) > 0 for i in range(n)]
37: rng = np.random.default_rng(0)
```

**What it does.** `D[i]` is float32 metric depth (metres) in frame `i`, 0 = invalid (~88% valid, 0.3-1.4 m typical). `M[i]` is a boolean mask, True where the dataset's `outlier_mask` PNG is non-zero (mostly the invalid-depth region: a left-edge rectification band plus a gripper blob bottom-left). `rng` is a seeded NumPy generator so the 1 px noise of `synth1` (line 130) is reproducible.

**Alternatives considered.** Decision 1.2 masking options: none; fixed polygon; per-scene temporal-variance (Otsu) mask; flow-based rejection; GT outlier mask. In the pipeline all are rejected in favour of the static-track lever; here depth and mask are loaded **only** to build synthetic correspondences and the "oracle" subset, not to filter the real estimate.

**Why this choice.** GT depth/pose are the reference for the control arms (design doc: benchmarks use "GT depth+pose ONLY as reference"). The mask is used to exclude undefined-depth pixels from the synthetic set. The design doc's finding that "GT outlier_mask does NOT cover the gripper" and that "GT depth is defined ON the gripper in many scenes" is exactly why the `e2_oracle` arm (line 122) is not a gripper oracle — see honesty notes.

### Lines 38-39: rotation angle helper

```
38: 
39: def rot_angle(R): return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
```

**What it does.** Blank line 38. Geodesic angle of a rotation matrix in degrees via the trace identity `cos(theta) = (tr R - 1) / 2`, clipped to `[-1, 1]` against round-off. Used both for the GT rotation magnitude (line 99) and the error `rot_angle(R_est^T R_gt)` (line 114).

**Alternatives considered.** `Rotation.from_matrix(R).magnitude()`; Frobenius-norm-based angle. Equivalent.

**Why this choice.** Standard SO(3) metric, matches the "rotation error deg" used across every benchmark table in the design doc.

### Lines 40-43: translation-direction error and the skew operator

```
40: def tdir(t, tg):
41:     if np.linalg.norm(tg) < 2e-3 or np.linalg.norm(t) < 1e-12: return None
42:     return float(np.degrees(np.arccos(np.clip(np.dot(t.ravel(), tg) / np.linalg.norm(t) / np.linalg.norm(tg), -1, 1))))
43: def skew(t): return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
```

**What it does.** `tdir` returns the angle in degrees between an estimated translation `t` (any scale) and the GT translation `tg` (metres), or `None` if the GT translation is below 2 mm (direction meaningless, and the 2 mm figure is the same floor used to skip the pair at line 98) or `t` is numerically zero. It is **signed-aware**: it does not fold `theta` to `[0, 90]`, so an estimate with the correct axis but flipped sign scores ~180 deg. `skew(t)` builds the 3x3 cross-product matrix `[t]x` so that `[t]x v = t x v`; used to assemble `E = [t]x R` in line 59.

**Alternatives considered.** Folding the angle (`min(theta, 180 - theta)`) would hide sign errors; for `recoverPose` output the sign is resolved by cheirality, so a flip is a genuine error and the unfolded angle is the honest score. For the SVD-based `gtR_t` (line 81) the sign is arbitrary and the unfolded angle makes that arm's rows meaningless.

**Why this choice.** The unfolded angle is correct for every `recoverPose`-derived arm. The design doc's verdict on the one arm it breaks: "The gtR_t rows in the JSON are INVALID (sign-ambiguous linear solver); ignore them." The 2 mm floor is well below the median per-step translation of 9 mm (dataset facts), so at gaps >= 4 almost every moving pair passes.

### Lines 44-46: one LK hop with forward-backward check

```
44: def lk_hop(gi, gj, p):
45:     p1, st, _ = cv2.calcOpticalFlowPyrLK(gi, gj, p, None); p0b, stb, _ = cv2.calcOpticalFlowPyrLK(gj, gi, p1, None)
46:     return p1, (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm((p - p0b)[:, 0], axis=1) < 1.0)
```

**What it does.** Sparse pyramidal Lucas-Kanade from image `gi` to `gj` for points `p` of shape `(N, 1, 2)` float32 (pixel coordinates), then back from `gj` to `gi` starting at the forward result. `calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts=None)` returns `(nextPts, status, err)`; `status` is `(N, 1)` uint8 with 1 where a flow was found. All other parameters are OpenCV defaults: `winSize` 21x21, `maxLevel` 3, default termination criteria. A track survives only if both directions succeeded and the round-trip landed within 1.0 px of where it started.

**Alternatives considered.** Decision 1.5 options: LK with/without fwd-bwd check, ORB / SIFT + ratio test, DIS or Farneback dense flow at corners or on a grid.

**Why this choice.** Decision 1.5: "sparse LK at OpenCV defaults (21 px window, 3 levels) + forward-backward check at 1 px", from the correspondence benchmark: LK+FB 166 / 88 correct at gap 1 (rotErr 0.88 deg), 103 / 27 at gap 4 (2.55 deg), 4 ms; ORB/SIFT/Farneback rejected; DIS-grid better rotation (1.88 deg at gap 4) but only via 5x more points, and LK+FB had the best per-correspondence precision at gap 4 ("in2 0.35 vs 0.29"). The window / level values are OpenCV defaults, not tuned.

### Lines 47-52: chaining hops across the gap

```
47: def lk_chain(i, j, p):
48:     ok = np.ones(len(p), bool); cur = p.copy()
49:     for f in range(i, j):
50:         nxt, okf = lk_hop(G[f], G[f + 1], cur); ok &= okf; cur = nxt
51:     return cur, ok
52: 
```

**What it does.** Tracks the corners from frame `i` to frame `j` one consecutive frame at a time, ANDing the per-hop survival flags. Points that fail a hop keep being propagated (their coordinates are garbage) but are flagged dead. Returns the final positions `(N, 1, 2)` and the survival mask. Blank line 52 closes the frontend helpers.

**Alternatives considered.** A single direct LK hop `i -> j`; descriptor matching `i <-> j`; dense flow. The design doc's third diagnostic measured exactly these at 2 px RANSAC: chained LK 2.06 / 31.6, single-hop LK 2.33 / 38.3, SIFT + ratio 2.16 / 38.9 (1601 pairs), DIS 8-px grid 1.70 / 26.4 at gap 4; at gap 16 chained 8.22 / 41.0, single-hop 9.20 / 60.4 (934 pairs), SIFT 4.94 / 38.4 (764 pairs), DIS 12.05 / 66.1.

**Why this choice.** The probe evidence says "Over a 15-frame gap in fast segments survival drops to ~15%: tracking must be frame-to-frame", and consecutive-frame survival is 0.95 (median). Chaining is what the pipeline does (decision 1.6: persistent tracks). The cost is the finding of this probe: "real tracks get WORSE with gap because chained LK drift grows with chain length", and "No alternative correspondence source fixes it (SIFT only helps at gap 16, and finds enough matches on half the pairs)".

### Lines 53-57: essential matrix + pose recovery

```
53: def est(a, b, method, thr):
54:     E, inl = cv2.findEssentialMat(a, b, K, method=method, prob=0.999, threshold=thr)
55:     if E is None or E.shape != (3, 3) or inl is None: return None
56:     _, R, t, _ = cv2.recoverPose(E, a, b, K, mask=inl.copy()); return R, t.reshape(3), inl[:, 0].astype(bool)
57: 
```

**What it does.** `a`, `b` are `(N, 2)` float64 **pixel** coordinates in frames `i` and `j`. `findEssentialMat(points1, points2, cameraMatrix, method, prob, threshold)` runs Nister's 5-point solver inside the chosen robust loop (`cv2.RANSAC` = 8, or `cv2.USAC_MAGSAC` = 38), with confidence `prob` 0.999 and `threshold` = maximum point-to-epipolar-line distance **in pixels** (OpenCV divides by the focal internally because `cameraMatrix` is given); `maxIters` is left at OpenCV's default 1000. It returns `(E, mask)`; the mask is `(N, 1)` uint8, 1 for inliers. Pitfall handled on line 55: `findEssentialMat` returns a stacked `(3k, 3)` array of all real 5-point solutions only when exactly 5 correspondences are passed (verified with cv2 4.11.0: 5 points -> `(18, 3)`, >= 6 points -> `(3, 3)`, because inside the RANSAC loop the solver's multiple solutions are scored and one is kept); with the >= 20-track floors in this file (lines 88 / 91 / 95 / 120 / 122 / 123 / 127) the shape guard is defensive. On degenerate input the function can return `None`, which is the case that matters here — it is treated as failure. `recoverPose(E, points1, points2, cameraMatrix, mask)` decomposes `E` into the four `(R, t)` candidates, triangulates the masked inliers and keeps the candidate with the most points in front of both cameras (cheirality); it returns `(n_pass, R, t, mask)` where `t` is a **unit vector** and `R, t` perform the change of basis from camera 1 to camera 2 (`x_2 = R x_1 + t`), i.e. **world-to-camera-2 with camera 1 as the world** — the same convention as the GT composition in line 97. The mask argument is input/output: OpenCV overwrites it with the cheirality-passing subset, hence `inl.copy()` so the returned inlier mask is the RANSAC one, not the cheirality one. The cheirality count `n_pass` is discarded here (the pipeline uses it as the >= 20 bootstrap guard, decision 2.9). Blank line 57.

**Alternatives considered.** Decision 2.10 options: findEssentialMat + recoverPose (chosen), homography, H-vs-E model selection (ORB-SLAM style), and thresholds 2 / 1 / 0.5 px. Estimators: RANSAC, LMedS, USAC variants (MAGSAC++ tested via this function). Working in normalised coordinates with `K = I` (the pipeline's per-frame-K bootstrap does this, decision 0.1b revised) is equivalent for a single K.

**Why this choice.** This *is* the object under test; `prob=0.999` is the bootstrap confidence stated in decision 2.10 ("RANSAC threshold 0.5 px (0.999 conf)") and coincides with the PnP value of decision 2.8 ("0.999 / 1000"); `maxIters` is OpenCV's default 1000 (the 4.11.0 signature is `findEssentialMat(points1, points2, cameraMatrix[, method[, prob[, threshold[, maxIters[, mask]]]]])`) rather than a passed argument. The design doc keeps E-only and defers H-vs-E to v1 "if bootstrap rejections cluster on planar scenes". The threshold is deliberately a parameter because the threshold ladder was the decisive experiment (2.06 -> 1.64 -> 1.23 deg rotation and 31.6 -> 24.3 -> 18.1 deg t-dir at gap 4 for 2 / 1 / 0.5 px), which set decision 2.10 to 0.5 px.

### Lines 58-63: Sampson distance

```
58: def sampson(R, t, a, b):
59:     F = Kinv.T @ skew(t) @ R @ Kinv
60:     a1 = np.c_[a, np.ones(len(a))]; b1 = np.c_[b, np.ones(len(b))]
61:     Fa = (F @ a1.T).T; Ftb = (F.T @ b1.T).T; num = np.sum(b1 * Fa, 1)
62:     return num / np.sqrt(Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2)
63: 
```

**What it does.** Builds the fundamental matrix `F = K^-T [t]x R K^-1` from a candidate `(R, t)` (so `b^T F a = 0` for a perfect correspondence in pixel homogeneous coordinates `a = [u_i, v_i, 1]`, `b = [u_j, v_j, 1]`; this is `x_j^T E x_i = 0` with `E = [t]x R`, consistent with `recoverPose`'s `x_2 = R x_1 + t`). Returns the **signed** Sampson distance per correspondence: the algebraic residual `b^T F a` divided by the norm of its gradient with respect to the four image coordinates, `sqrt((Fa)_1^2 + (Fa)_2^2 + (F^T b)_1^2 + (F^T b)_2^2)`. It is the first-order approximation of the geometric reprojection error and is in **pixels** — the natural residual for a least-squares refinement of the epipolar geometry. Blank line 63.

**Alternatives considered.** Symmetric epipolar distance (point-to-line in both images), the plain algebraic error `b^T F a` (scale-dependent, biased), or the gold-standard error with explicit 3D points (needs triangulation, 3 extra unknowns per point).

**Why this choice.** Sampson is the standard compromise (Hartley-Zisserman): geometric-quality residual with no extra unknowns. It is only used by the `refine` arm, whose result (1.70 / 29.0 at gap 4) did not move the t-direction problem, so nothing downstream depends on this function.

### Lines 64-67: refinement parameterisation

```
64: def refine(R0, t0, a, b):
65:     """least-squares on (rotvec, spherical t) minimizing Sampson distance."""
66:     th0 = np.array([np.arctan2(t0[1], t0[0]), np.arccos(np.clip(t0[2] / np.linalg.norm(t0), -1, 1))])
67:     x0 = np.r_[Rot.from_matrix(R0).as_rotvec(), th0]
```

**What it does.** Starts from the `recoverPose` estimate `(R0, t0)`. The 5-DOF state `x0` is the rotation vector (axis * angle, 3 numbers) concatenated with the spherical angles of `t0`: azimuth `phi = atan2(t_y, t_x)` and polar angle `theta = acos(t_z / |t|)`. Parameterising `t` on the unit sphere removes the unobservable scale from the optimisation instead of adding a norm constraint.

**Alternatives considered.** Optimising `t` in R^3 with a normalisation penalty; a local tangent-plane update; quaternions for `R`. Or the one the pipeline actually uses: no refinement of the bootstrap E at all (decision 2.10 is findEssentialMat + recoverPose only).

**Why this choice.** Minimal (5 unknowns for a 5-DOF problem), directly initialised from the closed-form solution. A singularity exists at `theta = 0 / pi` (t along the optical axis), which is uncommon for a wrist camera but not impossible — this is a diagnostic, not the pipeline. The pipeline's only refinement question, decision 3.4 (trajectory-level local BA / pose graph), was rejected for v0 on the local-BA sweep (windows 5 / 10: ATE 128.8 / 130.7 vs 122.9, 5-9x runtime); two-view refinement of the bootstrap E was never one of its options.

### Lines 68-70: unpacking the state

```
68:     def unpack(x):
69:         R = Rot.from_rotvec(x[:3]).as_matrix(); ph, th = x[3], x[4]
70:         return R, np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
```

**What it does.** Inverse of line 66-67: rotation vector -> 3x3 matrix via SciPy, spherical angles -> unit translation `(sin th cos ph, sin th sin ph, cos th)`. The result feeds `sampson` and is what `refine` returns.

**Alternatives considered.** As for line 64-67.

**Why this choice.** Keeps `t` exactly unit-norm at every iterate, so the Sampson residual (which is homogeneous in `t`) is never fooled by scale drift.

### Lines 71-75: residual, solver call, return

```
71:     def resid(x):
72:         R, t = unpack(x); return sampson(R, t, a, b)
73:     sol = least_squares(resid, x0, loss="huber", f_scale=1.0, max_nfev=200)
74:     return unpack(sol.x)
75: 
```

**What it does.** The residual vector is the per-correspondence signed Sampson distance in pixels. `scipy.optimize.least_squares` with the default trust-region-reflective method, a Huber loss with `f_scale = 1.0` px (residuals beyond ~1 px are down-weighted linearly rather than squared, a soft second layer of outlier handling on top of the RANSAC inlier set), and at most 200 function evaluations (finite-difference Jacobian over the 5 parameters). Returns the refined `(R, t_unit)`. Blank line 75.

**Alternatives considered.** Plain squared loss (would let residual outliers among the 2 px inliers dominate); Cauchy/arctan loss; an analytic Jacobian.

**Why this choice.** Huber at 1 px matches the intuition later confirmed by the threshold ladder that only sub-pixel correspondences are geometrically clean. Measured outcome (design doc table, "2-view Sampson refinement of E(2px)"): 1.70 / 29.0, 3.38 / 30.0, 7.80 / 39.8 — the t-direction error is unchanged within a few degrees, which is the evidence that the E-matrix estimate is not stuck in a poor local optimum: the correspondences themselves do not constrain `t` better. `f_scale` and `max_nfev` are probe choices, not decisions.

### Lines 76-78: translation given rotation — normalised coordinates

```
76: def t_given_R(R, a, b):
77:     """with R known, epipolar constraint b^T [t]x R a = 0 is linear in t: solve by SVD."""
78:     an = (Kinv @ np.c_[a, np.ones(len(a))].T).T; bn = (Kinv @ np.c_[b, np.ones(len(b))].T).T
```

**What it does.** Converts both pixel sets to normalised homogeneous camera coordinates `(x/z, y/z, 1)` by left-multiplying with `K^-1` (the essential-matrix constraint holds in normalised, not pixel, coordinates). Shapes `(N, 3)`.

**Alternatives considered.** Using `cv2.undistortPoints` with `P = None` (same result for a distortion-free K).

**Why this choice.** Straightforward; nothing decided here. This arm was written to test the hypothesis on line 15-16 ("isolates whether t is bad because R is bad").

### Lines 79-83: linear solve for `t` and its sign ambiguity

```
79:     Ra = (R @ an.T).T
80:     A = np.cross(bn, Ra)            # b x (R a) . t = 0  <=>  b^T [t]x (R a) = 0 up to sign
81:     _, _, Vt = np.linalg.svd(A); t = Vt[-1]
82:     return t
83: 
```

**What it does.** With `R` fixed to GT, the epipolar constraint `b^T [t]x (R a) = b . (t x Ra) = t . (Ra x b) = 0` is linear in `t`; each correspondence contributes the row `b x (R a)` (equal to `-(Ra x b)`, hence "up to sign" in the comment). `t` is taken as the right-singular vector of the smallest singular value of the `(N, 3)` matrix `A`, i.e. the unit null vector in the least-squares sense. **The null vector of an SVD has an arbitrary sign**, and no cheirality test is applied afterwards; `tdir` (line 40-42) does not fold the angle, so about half of these estimates score near 180 deg regardless of the axis quality. There is also no robustness (all inliers of the 2 px RANSAC are used with equal algebraic weight). Blank line 83 closes the helper block.

**Alternatives considered.** Resolve the sign by triangulating a few points and checking positive depth (what `recoverPose` does); use a cheirality vote; fold the angle in `tdir` for this arm only; or a robust (IRLS) linear solve.

**Why this choice.** None of those fixes was applied. The design doc's ruling is explicit: "The gtR_t rows in the JSON are INVALID (sign-ambiguous linear solver); ignore them." The `gtR_t` row is consequently absent from the doc's summary table, and no decision rests on it. The question it was meant to answer (is `t` bad because `R` is bad?) was answered instead by the synthetic arms: with GT-flow + 1 px noise both `R` and `t` are fine (1.43 / 13.8 at gap 4), so neither is intrinsically unobservable; the tracks are the problem.

### Line 84: result container

```
84: res = dict(pairs=0, gated=0, gt_rot=[], gt_t_mm=[], rot_dom=[], n_tracks=[], methods={m: dict(rot=[], tdir=[], inl=[]) for m in ("e2", "e1", "e05", "magsac", "synth1", "synth0", "refine", "gtR_t", "e2_rel", "e2_oracle", "e2_oracle_rel")})
```

**What it does.** Per-scene accumulator serialised to JSON at line 137. `pairs` counts all `(i, i+GAP)` pairs visited; `gated` those that survive every gate (lines 88-98). Per gated pair: GT rotation (deg), GT translation (mm), rotation-dominance ratio (when computable), number of surviving tracks. `methods` holds, for each of eleven arms, lists of rotation error (deg), translation-direction error (deg; only when defined) and inlier fraction. Three arms — `e2_rel`, `e2_oracle`, `e2_oracle_rel` — are not in the docstring; they were added later (relative lever and oracle-mask variants, lines 118-123).

**Alternatives considered.** Per-pair records (one dict per pair) would allow paired comparisons and conditioning on GT motion; the design doc's third diagnostic reports pair counts per arm, which this flat layout supports only via list lengths.

**Why this choice.** The consumers are pooled medians across scenes (design doc: "Medians pooled over ~2100 pairs per gap" for the corr benchmark; "~1780 pairs per gap" here), which the flat lists give directly. Note `tdir` lists can be shorter than `rot` lists for the same arm (skipped when `None`), so they are not index-aligned.

### Lines 85-88: pair loop and corner detection

```
85: for i in range(0, n - GAP, 2):
86:     j = i + GAP; res["pairs"] += 1
87:     p0 = cv2.goodFeaturesToTrack(G[i], 1000, 0.01, 5)
88:     if p0 is None or len(p0) < 20: continue
```

**What it does.** Pairs start at every second frame (`range(0, n - GAP, 2)`); no rationale for the stride is recorded in the code or the design doc, and the number of pairs visited, `(n - GAP) / 2`, does depend on `GAP`. Over the 12 scenes this yields ~1780 gated pairs per gap (design doc). `goodFeaturesToTrack(image, maxCorners=1000, qualityLevel=0.01, minDistance=5)`: Shi-Tomasi minimum-eigenvalue corners, at most 1000, rejecting corners weaker than 1% of the strongest, with 5 px non-maximum suppression; returns `(N, 1, 2)` float32 or `None`. Pairs with fewer than 20 corners are skipped (not counted as gated).

**Alternatives considered.** Decision 1.3 options: Shi-Tomasi (chosen), FAST, ORB, SIFT/AKAZE, fixed grid. Decision 1.4: grid bucketing / top-up (rejected: "minDistance=5 already spreads corners at this resolution").

**Why this choice.** Decision 1.3: exactly these three values ("probe values; cap never reached, ~140-300 corners/frame"; the RAIL probe saw ~215 corners/frame, min 139). The 20-point floor is the same "cheirality count >= 20" floor as decision 2.9's bootstrap guard and decision 3.1's initial failure floor (later raised to 30 for PnP).

### Lines 89-91: chained tracking and survivor extraction

```
89:     p1, ok = lk_chain(i, j, p0)
90:     a = p0[ok, 0].astype(np.float64); b = p1[ok, 0].astype(np.float64)
91:     if len(a) < 20: continue
```

**What it does.** Tracks the corners through the `GAP` consecutive hops. `a`, `b` become `(N_ok, 2)` float64 pixel coordinates of the survivors in frames `i` and `j` (float64 is accepted by `findEssentialMat`; LK itself works in float32). Fewer than 20 survivors -> skip.

**Alternatives considered.** See lines 47-52 (single hop / descriptors / dense flow).

**Why this choice.** Same frontend as the pipeline, so the diagnostic measures what the bootstrap actually sees. The corr benchmark reports median 103 LK+FB correspondences per pair at gap 4 (166 at gap 1), well above the 20 floor on textured frames; per-hop survival is 0.95 (probe evidence).

### Lines 92-96: parallax gate and the static-track lever

```
92:     disp = np.linalg.norm(b - a, axis=1)
93:     if np.median(disp) < 2.0: continue
94:     keep = disp >= 1.0
95:     if keep.sum() < 20: continue
96:     a, b = a[keep], b[keep]
```

**What it does.** `disp` is the per-track displacement in pixels across the gap. Gate 1 (line 93): the pair is dropped unless the *median* displacement is >= 2.0 px — the parallax gate that makes the two-view problem well-posed and, per decision 1.2, the only condition under which a zero-motion track is provably not scene geometry. Gate 2 (line 94-96): the static-track lever, absolute form — tracks that moved less than 1.0 px while the median moved >= 2 px are removed before any geometry. The pair must still have >= 20 tracks.

**Alternatives considered.** Decision 1.2 options: no mask; fixed polygon; Otsu temporal-variance mask (over-masks 58%); flow-based rejection; GT outlier mask (does not cover the gripper). Lever variants: absolute 1 px (here and in the pipeline) vs relative 20% of median (line 119). Lever OFF (design doc: 12-scene v0 final 115.6 / 8.40 / 0.971 OFF vs 111.9 / 6.93 / 0.871 ON).

**Why this choice.** Decision 1.2's lever with its exact thresholds ("drop tracks with displacement < 1 px" only on frames with "median track displacement >= 2 px"). It was chosen because the gripper is image-static and its corners "vote for zero motion" (dataset facts), a quarter of LK tracks are static (16-53% per scene, median ~25%), and removing them "cuts parallax-gated E-rotation error 2.14 -> 1.70 deg (LK)" in the corr benchmark. The user flipped the default to ON on 2026-09-05. Finding *from this probe*: for the two-view estimate specifically, "Static-track removal (absolute 1 px or relative 20%) does not change E" — the lever's pipeline benefit comes through PnP/map hygiene, not through E.

### Lines 97-99: GT relative pose and per-pair statistics

```
97:     T = np.linalg.inv(c2w[j]) @ c2w[i]; Rg, tg = T[:3, :3], T[:3, 3]
98:     if np.linalg.norm(tg) < 2e-3: continue
99:     res["gated"] += 1; res["gt_rot"].append(float(rot_angle(Rg))); res["gt_t_mm"].append(float(1000 * np.linalg.norm(tg))); res["n_tracks"].append(int(len(a)))
```

**What it does.** `T = w2c_j @ c2w_i` maps camera-`i` coordinates into camera-`j` coordinates: `X_j = Rg X_i + tg`. This is exactly `recoverPose`'s `(R, t)` convention (change of basis from the first to the second camera), so `Rg`, `tg` can be compared directly with the estimates; `tg` is in **metres**. Pairs with less than 2 mm of GT translation are skipped (direction undefined — same floor as `tdir`). Then the pair is counted as gated and its GT rotation (deg), translation (mm) and surviving track count are stored.

**Alternatives considered.** The opposite composition `inv(c2w[i]) @ c2w[j]` (camera j -> camera i) would give `R_g^T` and `-R_g^T t_g`; a correct estimate `R_est = R_g` would then score `rot_angle(R_g^T R_g^T) = rot_angle((R_g R_g)^T)` = twice the GT rotation angle on line 114, and a wildly wrong t-direction on lines 40-42 — the classic c2w/w2c trap, which `synth0` = 0.01 / 0.0 rules out.

**Why this choice.** The convention is verified empirically rather than argued: the noise-free synthetic arm (`synth0`) returns 0.01 / 0.0 deg at every gap ("Conventions verified (synth0 = 0)"). The design doc's "Plumbing" section demands such a pose-convention check "before any run". The 2 mm floor is stated on line 41 too.

### Lines 100-104: synthetic GT-flow correspondences

```
100:     # synthetic GT correspondences at the same points (need GT depth)
101:     u = np.clip(np.round(a[:, 0]).astype(int), 0, W - 1); v = np.clip(np.round(a[:, 1]).astype(int), 0, H - 1)
102:     z = D[i][v, u]; zv = (z > 0) & (~M[i][v, u])
103:     X = (Kinv @ np.c_[a, np.ones(len(a))].T) * z; Xj = Rg @ X + tg[:, None]
104:     pj = (K @ Xj); pj = (pj[:2] / np.maximum(pj[2], 1e-9)).T
```

**What it does.** Comment line 100. For every surviving track position `a` in frame `i` (sub-pixel), round to the nearest integer pixel `(u, v)`, clipped into the image, and read GT depth `z` (metres) at that pixel; `zv` marks tracks with valid depth (`z > 0`) that are also outside the outlier mask. Unproject: `X = z * K^-1 [u, v, 1]^T` gives `(3, N)` metric points in camera `i` (nearest-neighbour depth lookup, no interpolation). Transform with the GT relative pose to camera `j`, project with `K`, and dehomogenise with a `1e-9` floor on `z_j` to avoid division by zero. `pj` is `(N, 2)`: where the tracked point *should* be in frame `j` if depth and pose were exact. Tracks with `z = 0` give `X = 0`, `Xj = tg`, a meaningless `pj` — those rows are excluded by `zv` wherever `pj` is used (lines 108, 127-130).

**Alternatives considered.** Bilinear depth interpolation; using the model's depth instead of GT (allowed by 0.1's note that model depth is not privileged, but the point here is an *oracle* control); generating fresh random pixels instead of re-using the tracked corners.

**Why this choice.** Re-using the same points is the whole design: it makes `synth*` differ from `e2` *only* in the correspondence quality, so the comparison factorises the error into "motion conditioning" and "frontend". Nearest-neighbour depth at 320x180 with 0.3-1.4 m depth is sufficient for a 1 px-noise control. Caveat from the design doc: a "~20% depth-vs-pose scale mismatch exists in some scenes (irrelevant to E)" — E is scale-invariant, so the synthetic flow's direction is right even where GT depth scale is off.

### Lines 105-110: rotation-dominance ratio

```
105:     # rotation-dominance: flow under pure rotation vs under pure translation at these points
106:     Xr = Rg @ X; pr = (K @ Xr); pr = (pr[:2] / np.maximum(pr[2], 1e-9)).T
107:     Xt = X + tg[:, None]; pt = (K @ Xt); pt = (pt[:2] / np.maximum(pt[2], 1e-9)).T
108:     if zv.sum() >= 10:
109:         rf = np.median(np.linalg.norm(pr[zv] - a[zv], axis=1)); tf = np.median(np.linalg.norm(pt[zv] - a[zv], axis=1))
110:         res["rot_dom"].append(float(rf / max(tf, 1e-6)))
```

**What it does.** Comment line 105. Decomposes the GT motion's image effect at the tracked points: `pr` is where the points would land under rotation only (`Rg X`), `pt` under translation only (`X + tg`). With at least 10 valid-depth tracks, `rf` = median rotational flow magnitude (px), `tf` = median translational parallax (px), and the stored ratio is `rf / tf` (floored denominator). A ratio >> 1 means the observed flow is almost entirely rotation, and the translation direction — which is encoded only in the parallax component — is weakly observable.

**Alternatives considered.** The standard SfM conditioning proxies: baseline-to-depth ratio, or the median triangulation angle (which decision 2.11 tested as an *acceptance filter* — "parallax >= 1 deg" — and rejected for v0 because "at keyframe-scale baselines (~36 mm) a 1 deg floor discards half the points").

**Why this choice.** A pixel-space ratio is directly comparable to the LK noise floor: if the parallax component is comparable to the ~1-3 px track error, `t` cannot be resolved. The design doc does not tabulate `rot_dom`; the qualitative conclusion it underpins is in the probe evidence ("rotation dominates the flow") and, quantitatively, in the synthetic rows: with exact geometry and 1 px noise t-direction error is still 13.8 deg at gap 4 and only reaches 4.2 deg at gap 16 — that is the conditioning cost by itself.

### Lines 111-116: the scoring closure

```
111:     def rec(name, r):
112:         if r is None: return
113:         R, t, inl = r; m = res["methods"][name]
114:         m["rot"].append(float(rot_angle(R.T @ Rg))); td = tdir(t, tg)
115:         if td is not None: m["tdir"].append(td)
116:         m["inl"].append(float(inl.mean()) if inl is not None else 1.0)
```

**What it does.** Records one estimate under `name`. A `None` estimate (E failed, line 55) is silently skipped — so per-arm list lengths differ, and a failed pair does not count against the arm. Rotation error = angle of `R_est^T R_gt` (deg). Translation-direction error via `tdir`, appended only when defined. Inlier fraction = mean of the boolean inlier mask (1.0 if an arm has no mask — never the case in this file, every arm passes a mask).

**Alternatives considered.** Recording failures explicitly as a failure rate per arm (the PnP and triangulation benchmarks do this: "Zero failures on 2118 pairs", `fails` columns); recording the cheirality count from `recoverPose`.

**Why this choice.** Adequate for pooled medians on ~1780 pairs. The doc does not report a per-arm failure count for this probe; per-arm list lengths in the JSON are the only record. The same kind of inlier fraction appears in the earlier RAIL probe evidence ("inliers ~0.5" on fast pairs, 2026-09-04), which predates this script; the doc does not tabulate this script's `inl` lists.

### Lines 117-120: pipeline estimate and the relative lever

```
117:     r2 = est(a, b, cv2.RANSAC, 2.0); rec("e2", r2)
118:     # relative static lever: drop tracks moving < 20% of the median (chained drift of static points exceeds 1 px)
119:     rel = np.linalg.norm(b - a, axis=1) >= 0.2 * np.median(np.linalg.norm(b - a, axis=1))
120:     if rel.sum() >= 20: rec("e2_rel", est(a[rel], b[rel], cv2.RANSAC, 2.0))
```

**What it does.** `e2`: RANSAC at 2.0 px on the lever-filtered tracks — the pipeline configuration at the time of the probe; its result `r2` is kept for the `refine` and `gtR_t` arms. Then a second, *relative* lever on top of the absolute one: keep tracks whose displacement is at least 20% of the pair's median displacement. The comment gives the motivation: over a long chain a gripper track's accumulated LK drift can exceed the 1 px absolute floor, so a threshold that scales with the true motion should catch more static tracks at large gaps. `e2_rel` is recorded if >= 20 tracks remain.

**Alternatives considered.** Decision 1.2: absolute 1 px lever (pipeline) vs this relative 20% lever vs no lever; or a real mask.

**Why this choice.** Measured and rejected: "relative static lever (drop < 20% of median disp)" = 1.96 / 32.5, 3.73 / 32.8, 7.75 / 39.1 versus `e2` 2.06 / 31.6, 3.80 / 31.0, 8.22 / 41.0 — within noise, and the doc concludes "Static-track removal (absolute 1 px or relative 20%) does not change E." The pipeline keeps the absolute 1 px form (decision 1.2 / frozen parameters table).

### Lines 121-123: "oracle" gripper mask arms

```
121:     # ORACLE gripper mask (diagnostic only): keep only tracks with valid GT depth (gripper region has none)
122:     if zv.sum() >= 20: rec("e2_oracle", est(a[zv], b[zv], cv2.RANSAC, 2.0))
123:     if (zv & rel).sum() >= 20: rec("e2_oracle_rel", est(a[zv & rel], b[zv & rel], cv2.RANSAC, 2.0))
```

**What it does.** Comment line 121 states the assumption: the gripper region has no valid GT depth, so restricting to `zv` (valid depth and not in `outlier_mask`) should be an oracle gripper mask. `e2_oracle` runs the 2 px estimate on that subset; `e2_oracle_rel` additionally applies the relative lever. Both are privileged (use GT depth) and labelled diagnostic-only.

**Alternatives considered.** Any of decision 1.2's non-privileged masks; the GT `outlier_mask` alone.

**Why this choice.** The assumption turned out to be false, which is itself a finding: "GT depth is defined ON the gripper in many scenes, so 'valid GT depth' is not a gripper oracle; the corr-benchmark in2/EPE numbers on high-gripper scenes are contaminated by that." Correspondingly, the design doc's summary table carries no oracle rows, and no decision cites them. Dataset facts already say the `outlier_mask` "does NOT cover the gripper".

### Lines 124-126: threshold ladder and MAGSAC

```
124:     rec("e1", est(a, b, cv2.RANSAC, 1.0)); rec("e05", est(a, b, cv2.RANSAC, 0.5))
125:     try: rec("magsac", est(a, b, cv2.USAC_MAGSAC, 2.0))
126:     except cv2.error: pass
```

**What it does.** Same tracks, RANSAC thresholds 1.0 and 0.5 px, and USAC MAGSAC++ (`cv2.USAC_MAGSAC`) at 2.0 px. With MAGSAC the threshold is an upper bound on its sigma-marginalised inlier scoring rather than a hard inlier cut, and `prob` still applies. The `try/except cv2.error` guards OpenCV's USAC path, which can throw on some inputs where the RANSAC path merely returns `None`; the arm is then simply not recorded for that pair.

**Alternatives considered.** Decision 2.10's threshold options 2 / 1 / 0.5 px; USAC variants (MAGSAC++ tested; other USAC flags not); LMedS (`cv2.LMEDS`, not tested — it needs > 50% inliers, which the RAIL probe's fast pairs at "inliers ~0.5" cannot guarantee).

**Why this choice.** This ladder settled decision 2.10: "essential matrix only, RANSAC threshold 0.5 px" — gap 4 rotation 2.06 (2 px) -> 1.64 (1 px) -> 1.23 (0.5 px), t-dir 31.6 -> 24.3 -> 18.1; gap 8: 3.80 -> 3.05 -> 2.88 and 31.0 -> 25.2 -> 23.5; at gap 16 the 1 px row (7.32 / 35.9) edges 0.5 px (8.11 / 36.8), but the pipeline's keyframe gaps are 2-4 (decision 2.12), so 0.5 px wins where it matters. The map-level confirmation (Slurm 45443823): PnP-at-+4 rotation 4.19 -> 3.66 -> 3.02 deg at gap 4 and 3.58 -> 3.29 -> 2.62 at gap 2 for 2 / 1 / 0.5 px. MAGSAC at 2 px (1.34 / 22.9 at gap 4) was better than RANSAC 2 px but not than RANSAC 0.5 px, and it has fewer "stateable numbers" (the logic behind decision 2.6's "plain RANSAC" for PnP), so it was not adopted. Note that the 0.5 px value contrasts with the 2 px PnP threshold of decision 2.7, which sweep 2 re-confirmed ("2.7 stays 2 px (1 px: more reboots; 3 px: worse everything)") — different solvers, different noise regimes.

### Lines 127-130: the synthetic controls

```
127:     if zv.sum() >= 20:
128:         as_, bs = a[zv], pj[zv]
129:         rec("synth0", est(as_, bs, cv2.RANSAC, 2.0))
130:         rec("synth1", est(as_, bs + rng.normal(0, 1.0, bs.shape), cv2.RANSAC, 2.0))
```

**What it does.** On the valid-depth subset: frame-`i` points `as_` are the real corner positions, frame-`j` points `bs` are the GT-flow positions from line 104. `synth0` runs the pipeline estimator on exact correspondences; `synth1` adds iid `N(0, 1 px^2)` noise to both coordinates of the frame-`j` points only (frame-`i` points are kept exact, so the total correspondence noise is 1 px, not sqrt(2) px). Both use RANSAC 2 px so that they differ from `e2` *only* in correspondence quality. The RNG seed (line 37) makes the noise reproducible.

**Alternatives considered.** Larger noise (the third diagnostic added 3 px: 2.20 / 29.0, 2.82 / 14.7, 3.55 / 9.1); noise on both frames; anisotropic or outlier-contaminated noise to mimic gripper tracks.

**Why this choice.** Two clean anchors. `synth0` = 0.01 / 0.0 at all gaps is the convention proof (any c2w/w2c or `E = [t]x R` vs `R [t]x` slip would show up here as a large constant error). `synth1` = 1.43 / 13.8, 1.67 / 7.4, 1.95 / 4.2: t-direction error *falls* with baseline as SfM theory predicts, while the real-track `e2` row stays at ~31-41 deg — so the ~30 deg is not a property of the motion but of the correspondences. The follow-up 3 px row matches `e2` at gap 4 (2.20 / 29.0 vs 2.06 / 31.6), giving the calibration "at gap 4 real chained-LK tracks behave like ~3 px iid noise for two-view geometry; beyond gap 8 they are worse than 3 px noise (chain drift)". That, in turn, is the reason decision 2.12 pushes keyframes close (median gap 2) rather than far.

### Lines 131-136: refinement and GT-rotation arms on the e2 inliers

```
131:     if r2 is not None:
132:         R0, t0, inl = r2
133:         try:
134:             Rr, tr = refine(R0, t0, a[inl], b[inl]); rec("refine", (Rr, tr, inl))
135:         except Exception: pass
136:         rec("gtR_t", (Rg, t_given_R(Rg, a[inl], b[inl]), inl))
```

**What it does.** Only when the 2 px estimate exists: `refine` polishes `(R0, t0)` on the RANSAC inliers (Sampson + Huber, lines 64-74) and is recorded with the *same* inlier mask (so its `inl` fraction equals `e2`'s by construction); any SciPy exception (e.g. a degenerate spherical parameterisation) drops the arm for that pair. `gtR_t` records the GT rotation together with the SVD translation from `t_given_R` on the same inliers — its rotation error is therefore identically 0 and only its `tdir` list carries information, which is sign-ambiguous (line 81).

**Alternatives considered.** Refining on the 0.5 px inliers instead of the 2 px ones; a cheirality-fixed `gtR_t`. Neither was run.

**Why this choice.** Measured: refinement gives 1.70 / 29.0, 3.38 / 30.0, 7.80 / 39.8 — a tighter RANSAC threshold (1.23 / 18.1 at gap 4) helps far more than nonlinear refinement of the loose-threshold inlier set, which is why decision 2.10 changed the threshold and stayed with findEssentialMat + recoverPose rather than adding a refinement step. (Decision 3.4, "none for v0", concerns trajectory-level BA / pose-graph refinement and rests on the local-BA sweep, not on this row.) `gtR_t`: INVALID per the design doc, ignore.

### Lines 137-138: output

```
137: json.dump(dict(scene=os.path.basename(scene), n=n, gap=GAP, res=res), open(out, "w"))
138: print(os.path.basename(scene), n, "gated", res["gated"], "of", res["pairs"], "done")
```

**What it does.** Writes one JSON with the scene name, frame count, gap and the full `res` accumulator; prints a one-line summary (gated / total pairs). The sbatch redirects stdout to `out/<scene>.gap<GAP>.log` next to the JSON, under `/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/e_diag/` (the path the design doc cites as "Scripts + JSON").

**Alternatives considered.** CSV per pair; aggregating across scenes inside the script.

**Why this choice.** The aggregation into the design-doc table (pooled medians per arm per gap, plus the model rows — finetuned / champion relative pose at gaps 4 / 8 / 16; the doc does not record which script produced them, and neither `e_diag.py` nor `e_diag.sbatch` reads any preds or CSV — only the gap-1 figures are tied to `eval_depth_pose_metrics.csv`, in the PnP-variant section) was done separately; keeping raw per-scene lists lets the same JSONs be re-pooled under different gates.

---

### Known limitations / honesty notes

- **`gtR_t` rows are INVALID.** The linear solve in lines 79-82 returns an SVD null vector of arbitrary sign with no cheirality fix, and `tdir` does not fold the angle; the design doc says "The gtR_t rows in the JSON are INVALID (sign-ambiguous linear solver); ignore them." No decision uses them.
- **`e2_oracle` / `e2_oracle_rel` are not oracles.** They assume the gripper has no valid GT depth (line 121); the design doc found "GT depth is defined ON the gripper in many scenes", so the "valid GT depth" subset does not isolate scene geometry. These rows are absent from the doc's tables. They also use GT depth, i.e. privileged information, and are diagnostic-only by construction (decision 0.1 forbids privileged inputs in the arm itself).
- **Privileged inputs everywhere in the controls.** `synth0` / `synth1` / `rot_dom` use GT depth and GT pose; scoring uses GT pose. This is a benchmark "GT-scored on test scenes" (honesty-audit item 8) run on the 12 reported smoke scenes (item 7: "all thresholds tuned on the 12 reported (test) scenes" — the 0.5 px threshold of decision 2.10 is one of them; item 7 is addressed by reporting the frozen configuration on all 4292 scenes).
- **Native frame / calibrated K, not the pipeline's frame.** The probe runs at 320x180 with the calibrated K; the pipeline runs on 320x192 cover frames with the model's per-frame focal (decisions 1.1, 0.1b). Pixel thresholds are therefore in slightly different units (fx ~ 192 here vs ~211-213 in the pipeline frame). The focal-source re-check showed the pipeline is insensitive to this.
- **Selection effects.** Only parallax-gated pairs (median flow >= 2 px) with >= 20 tracks and >= 2 mm GT translation are scored; the model rows in the design doc table are "all pairs, no gating", so the classical and model rows are on different populations. Arms whose estimate fails on a pair are silently skipped (line 112), so per-arm pair counts differ.
- **Rotation-dominance is stored inverted relative to the docstring** (line 110 stores rotational flow / translational parallax, the docstring says the reverse), and the quantity is never tabulated in the design doc.
- **What this probe does not answer.** It shows the ~30 deg t-direction error is a correspondence-quality problem (real tracks ~ 3 px iid noise at gap 4, worse beyond gap 8) and that "the model has the same 30-40 deg error vs GT" (finetuned 2.22 / 31.7, champion 1.99 / 30.5 at gap 4), so it is not specific to classical VO. It does not provide a fix: no correspondence source tried (single-hop LK, SIFT, DIS grid) removes it, and the eventual pipeline remedy is indirect — tighter bootstrap threshold (2.10), close keyframes (2.12) and cheir+reproj triangulation gating (2.11).
