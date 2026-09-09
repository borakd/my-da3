## `eval_pipeline/opencv_vo_probes/tri_bench.py`, lines 1-147: triangulation-acceptance / keyframe-trigger / E-threshold benchmark

This is the offline probe that settled three map-side decisions of the OpenCV control arm: which triangulated points to admit into the map (decision 2.11), the RANSAC threshold of the bootstrap essential matrix (the threshold part of decision 2.10, revised 2026-09-05), and when to declare the next keyframe (decision 2.12). It also carries a `LEVER` switch for decision 1.2's `reject_static_tracks` lever, but `OPENCV_VO_DESIGN.md` records no `tri_bench` numbers with the lever ON: all three tables it attributes to this probe are labelled lever OFF, and the lever ruling came from the full-pipeline sweeps 1-3 and the E-matrix diagnostics, not from here. It is not part of the inference pipeline (`eval_pipeline/opencv_vo.py`); it replays the pipeline's frontend on one scene, builds a tiny two-keyframe "map" per triplet of frames `(i, j, k)`, and scores that map against ground truth in two ways: how many of the admitted points have the right depth, and how well a PnP of the third frame `k` against the map recovers the GT relative pose. The essential-matrix pose is swapped for the GT relative pose in a paired arm so the acceptance filter can be judged in isolation from the two-view bootstrap error. One process handles one scene and writes one JSON. The only launcher in the repo, `tri_kf.sbatch`, runs the six adaptive-trigger configurations of the keyframe-trigger sweep (`KF_MODE=parallax` at 5 / 10 / 20 px and `KF_MODE=ratio` at 0.9 / 0.8 / 0.7; Slurm 45443937) over the 12 smoke scenes (`eval_pipeline/cg_smoke_scenes_12.txt`). Every other table the doc attributes to this probe came from launchers that sit only in the GPFS staging directory the doc names (`/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/tri_bench/`): the "Triangulation-acceptance benchmark" table (Slurm 45443315) from `tri_bench.sbatch` (outputs in `out/`), and the "Map quality vs bootstrap-E RANSAC threshold" table from *two* jobs, not the one the doc labels it with — its 2.0 px row from `tri_k4.sbatch` (Slurm 45443579, `out_k4/*.lever0`, `E_THR` left at the in-code default) and its 1.0 / 0.5 px rows from `tri_ethr.sbatch` (Slurm 45443823, `out_ethr/`, which loops `ETHR` over 0.5 and 1.0 only). The three stride rows of the trigger sweep (0.54 / 2.62, 0.51 / 3.02, 0.48 / 4.49 bad / PnP rot at stride 2 / 4 / 8) are numerically the 0.5 px row of the E-threshold table, i.e. re-used from Slurm 45443823, not re-run by `tri_kf.sbatch`. The pooling of the per-scene JSONs into the doc's medians is done by a script that is not in the repo either, but the two pooled columns that can be checked against the JSONs both reproduce exactly and are stated below: `fired/starts` is summed `triplets_gated` / `triplets_total`, and `fail%` is summed `(pnp_fail + skipped) / triplets` of the E-arm `cheir+rep` cell — not `pnp_fail` alone, which misses six of the nine rows. One provenance fact established while re-checking this block and recorded nowhere in the design doc: the 2.11 table alone predates the frame-by-frame chaining this file's docstring advertises, so the current script does not reproduce it (lines 54-59 and the limitations list).

### Lines 1-6: shebang and the docstring's statement of the triplet frontend

```
1: #!/usr/bin/env python
2: """Triangulation-acceptance benchmark for decision 2.11.
3: 
4: Triplets (i, j=i+G, k=j+G) with persistent LK tracks i->j->k, CHAINED frame-by-frame (same frontend as
5: 1.3/1.5: Shi-Tomasi 1000/0.01/5, LK defaults, fwd-bwd < 1 px at each hop).
6: Relative pose i->j is either
```

**What it does.** Module docstring. Declares the unit of measurement: a triplet of frames `(i, j, k)` in which the same corners are tracked `i -> j -> k`. "CHAINED frame-by-frame" is the important qualifier: tracks are propagated one consecutive frame at a time (as the pipeline does), not by a single LK hop from `i` straight to `j`. The docstring on line 4 says `k = j + G`, but the code default on line 34 makes `k = j + 2G` unless `KOFF` is set (see that group). The frontend named on line 5 is the one frozen in decisions 1.3 (Shi-Tomasi `goodFeaturesToTrack(1000, 0.01, 5)`) and 1.5 (sparse LK at OpenCV defaults, 21 px window / 3 levels, forward-backward check at 1 px), so the map is built from exactly the correspondences the pipeline would have. Line 3 is blank.

**Alternatives considered.** The doc's decision 1.5 options were sparse LK (with or without the fwd-bwd check), ORB/SIFT + ratio test, and DIS/Farneback dense flow at corners or on a grid. For the correspondence *chaining* specifically, the "Third diagnostic" (Slurm 45443759) compared chained LK against single-hop LK, SIFT + ratio, and DIS-grid for two-view geometry.

**Why this choice.** Decision 1.5 was settled by the correspondence benchmark (LK+FB at gap 4: 103 / 27 correct, rotErr 2.55 deg, 4 ms; DIS-grid 880 / 192, 1.88 deg, recorded as runner-up); LK gives corner-anchored persistent tracks the map needs. Chaining rather than single-hop is the pipeline's reality and is also measured better at every gap (E(2px) from chained LK 2.06 / 31.6 deg rot / tdir at gap 4 vs 2.33 / 38.3 from single-hop LK). This probe deliberately reproduces that frontend so the filter question is not confounded by a different correspondence source. The file as it stands does; the 2.11 run that actually answered the filter question did not, because it predates `lk_chain` (lines 54-59).

### Lines 7-12: the two pose sources and the first two metrics

```
7:   E  : findEssentialMat(RANSAC 2 px, 0.999) + recoverPose   (what the pipeline has)
8:   GT : the GT relative pose                                  (isolates the filter)
9: Triangulate every i<->j correspondence (cv2.triangulatePoints, P_i=K[I|0], P_j=K[R|t]),
10: apply each acceptance filter, then score the admitted "map":
11:   n_acc      : points admitted
12:   bad_frac   : admitted points whose depth (in cam i) is off from GT depth by >20%
```

**What it does.** Two relative poses `i -> j` are used for triangulation, in a paired design: `E`, the essential matrix from RANSAC (threshold "2 px" in the docstring; the actual value is the `E_THR` env, line 36) followed by `recoverPose`, i.e. what the pipeline's bootstrap does; and `GT`, the ground-truth relative pose from the `cam/*.npz` files. Line 9 fixes the triangulation convention: camera `i` is the world frame (`P_i = K[I|0]`), camera `j` is `P_j = K[R|t]` with `(R, t)` the cam-`i`-to-cam-`j` transform, so every triangulated point lives in cam-`i` coordinates. `n_acc` is the count of points that survive a filter; `bad_frac` is the fraction of those *that have valid GT depth* (line 114) whose cam-`i` depth disagrees with GT depth by more than 20 %.

**Alternatives considered.** Scoring the filter only under the E pose (the pipeline condition) would have been the obvious single-arm design. The GT-pose arm is the extra control. For triangulation itself, the standard alternatives are the linear DLT (`cv2.triangulatePoints`, used here), the optimal two-view method (`cv2.correctMatches` + DLT), or midpoint triangulation; the doc records no benchmark among these.

**Why this choice.** The paired E/GT design is what made the benchmark's key finding visible: under the E pose all filters tie at ~4 deg PnP rotation error and ~53 % bad points, under the GT pose the filters separate (0.73 / 0.63 / 0.56 deg for none / cheir / cheir+rep), so "the two-view bootstrap pose ... is the bottleneck of the map arm, not the acceptance filter" (design doc, "Triangulation-acceptance benchmark" section, KEY FINDING). `cv2.triangulatePoints` is what the pipeline uses (`opencv_vo.py`, `triangulate_pair`); no alternative was benchmarked.

### Lines 13-18: the scale alignment and the PnP metrics

```
13:                after a single median scale alignment (E has arbitrary scale; GT is metric
14:                but we align the same way for comparability)
15:   pnp rot/tdir: solvePnPRansac(ITERATIVE, 2 px, 0.999, 1000) + RefineLM of frame k
16:                against the admitted map using the k-observations of the same tracks;
17:                rotation error (deg) and translation-direction error (deg) vs GT.
18:   pnp_fail   : PnP returned < 20 inliers (the 3.1 floor)
```

**What it does.** `bad_frac` needs a scale: the E-pose map has unit-norm `t` (`recoverPose` returns a unit translation), so its depths are in an arbitrary unit and a single median scale factor maps them onto GT metres. "Single" means one scale per admitted map, i.e. per (pose source x filter) cell and so ten per triplet, not one per triplet: line 129 recomputes it inside the filter loop over that cell's own admitted, GT-depth-valid points (lines 126-130). The GT-pose map is metric already, but the same alignment is applied so the two arms are scored identically. The PnP metrics register frame `k` to the admitted map using the *same tracks'* positions in frame `k` (persistent tracks give the 2D-3D association for free), with the pipeline's solver: `solvePnPRansac(ITERATIVE)`, 2 px reprojection threshold, 0.999 confidence, 1000 iterations (decisions 2.4, 2.7, 2.8) followed by `solvePnPRefineLM`. Rotation error is the geodesic angle vs GT; translation-direction error is the angle between the PnP translation and the GT translation (direction only, since the map scale is arbitrary). `pnp_fail` counts PnPs that returned fewer than 20 RANSAC inliers, and equally those that reported failure, returned no inlier array, or raised `cv2.error` (lines 136-138).

**Alternatives considered.** Decision 3.1's options for the failure floor were inlier counts 10 / 20 / 30 or a ratio. Decision 2.4's PnP options were ITERATIVE, EPnP, P3P/AP3P, SQPnP, each with or without LM.

**Why this choice.** 20 was the 3.1 floor at the time of this probe; decision 3.1 was later revised to 30 in pipeline sweep 2 (30 vs 20: 0.861 vs 0.923 RPE-r, 7.07 vs 7.34 RPE-t, ATE 116.8 vs 120.0), so the `pnp_fail` column of the benchmark tables is at the older, looser floor. The PnP configuration is the decided one: all 10 variants were within 0.05 deg / 2 mm in the PnP-variant benchmark and LM after ITERATIVE-RANSAC is a measured no-op ("ITERATIVE (+LM identical)"), kept so the refine step exists.

### Lines 19-24: the five acceptance filters

```
19: Filters:
20:   none        : accept all
21:   cheir       : depth > 0 in both cameras
22:   cheir+rep   : + reprojection error < 2 px in both views
23:   cheir+par   : + parallax angle >= 1 deg
24:   all         : cheir + rep + par
```

**What it does.** Names the five filters that lines 74-80 implement: no filter; cheirality (positive depth in both cameras); cheirality plus a 2 px reprojection bound in both views; cheirality plus a 1 deg parallax floor; and all three. They form a small lattice, not a single chain: `cheir` extends `none`; `cheir+rep` and `cheir+par` each extend `cheir` independently of one another (neither is a subset of the other, since `cheir+par` applies no reprojection test and `cheir+rep` no parallax test); `all` is the conjunction of both. So `none ⊃ cheir ⊃ {cheir+rep, cheir+par} ⊃ all` in terms of admitted sets, but rows 3 and 4 of the result table are siblings.

**Alternatives considered.** These are exactly the options listed for decision 2.11 ("none; cheirality only; + reprojection < 2 px; + parallax >= 1 deg; all three"). Standard SLAM systems (ORB-SLAM) use all three plus a scale-consistency check across the two views' pyramid levels; the latter has no analogue with LK tracks and was not tried.

**Why this choice.** Decision 2.11 picked `cheir+rep`: with the GT pose it is best on every metric (PnP rot 0.56 deg vs 0.63 cheir-only vs 0.73 none; bad-depth 22 % vs 32 % vs 46 %), and it introduces no new parameter because 2 px is already the PnP threshold. The parallax floor was rejected for v0: at keyframe-scale baselines (~36 mm) a 1 deg floor "discards half the points and raises PnP failures 2x" (GT-pose `n_acc` 43 vs 62, fails 398 vs 183).

### Lines 25-30: what GT is used for, usage line, imports, single-threaded OpenCV

```
25: GT depth / pose are used ONLY for scoring. Static tracks are kept (lever OFF).
26: Usage: tri_bench.py <scene_dir> <out_json> [G]
27: """
28: import sys, os, glob, json
29: import numpy as np, cv2
30: cv2.setNumThreads(1)
```

**What it does.** Line 25 is the honesty clause of the probe: the GT pose enters the estimation path only in the explicit `GT` arm, and GT depth / outlier mask appear only in the `bad_frac` scorer. "Static tracks are kept" records that the lever (decision 1.2) was off in every run the design doc reports from this file; the `LEVER` env (line 35) can turn it on, but no lever-ON `tri_bench` numbers appear in the doc. Line 26 gives the CLI: scene directory, output JSON path, optional gap `G` (default 4, line 33). Lines 28-29 import the only dependencies (numpy, OpenCV). Line 30 pins OpenCV to one thread, because `tri_kf.sbatch` runs 12 scene processes in parallel on one 48-core node with `OMP_NUM_THREADS=1`; letting each OpenCV process spawn its own thread pool would oversubscribe the node.

**Alternatives considered.** Running the lever ON (the pipeline's eventual default) in this probe; multi-threaded OpenCV with fewer parallel scenes.

**Why this choice.** Lever OFF matches the pipeline default *at the time of the probe* (1.2 was "DEFAULT OFF" until 2026-09-05; flipped ON by user decision after sweeps 1-3 showed ON better on every metric, 111.9/6.93/0.871 vs 115.6/8.40/0.971). The E-matrix diagnostics separately found "Static-track removal (absolute 1 px or relative 20%) does not change E". The doc gives no reason for not reporting a lever-ON run of this benchmark. Single-threading is the standing rule for these probes (memory note: "single-thread cv2"; MN5's login nodes also impose a 300 s per-process CPU cap, so the probes must run as Slurm CPU jobs).

### Lines 31-35: CLI arguments, `GAP`, `KOFF`, `LEVER`

```
31: 
32: scene, out = sys.argv[1], sys.argv[2]
33: GAP = int(sys.argv[3]) if len(sys.argv) > 3 else 4
34: KOFF = int(os.environ.get("KOFF", "0")) or 2 * GAP      # PnP test frame k = j + KOFF (default: k = j + GAP)
35: LEVER = os.environ.get("LEVER", "0") == "1"
```

**What it does.** Line 31 is blank. `scene` is the scene directory (`.../<scene>`; the code appends `dense`), `out` the JSON path. `GAP` is the `i -> j` stride (default 4). `KOFF` is the offset of the PnP test frame `k` after `j`: read from the env, and if unset or `0` it falls back, via Python's `or`, to `2 * GAP`. Note the discrepancy: the inline comment says "default: k = j + GAP" and the doc's 2.11 table header says triplets `(i, i+4, i+8)` (i.e. `KOFF = GAP = 4`), but the code as it stands defaults to `KOFF = 8`. `tri_kf.sbatch` exports `KOFF=4` explicitly, and the doc's E-threshold and keyframe-trigger tables are labelled "PnP test at keyframe+4", so for those two tables the explicit env value, not the fallback, is what produced the numbers; anyone reproducing them must pass `KOFF=4`. The 2.11 table (Slurm 45443315) does not echo its configuration — its per-scene JSONs in the staging directory (`tri_bench/out/*.json`) carry only the keys `gap, n, res, scene, triplets_gated, triplets_total` (no `koff`, `lever`, `e_thr`, `kf_*`), because they were written by an earlier revision that predates lines 34-38 — but its `k` horizon is pinned all the same, by `triplets_total` alone. The number of start frames is fixed by the line-85 loop bound, `len(range(0, n - GAP - KOFF, 2))`, and `total` can only be less than or equal to it (line 87 drops corner-starved starts before line 100 increments `total`). On every one of the 12 scenes the recorded total equals the `KOFF=4` bound exactly and *exceeds* the `KOFF=8` bound, which no amount of skipping could produce: on the n = 131 scene, total 62 vs bounds 62 (`KOFF=4`) and 60 (`KOFF=8`); pooled, 2077 vs 2077 and 2053 (verified 2026-09-09 against `tri_bench/out/*.json`). So that run used `k = j + 4`, exactly as its table header `(i, i+4, i+8)` says, and it lost no start frame to corner starvation. `LEVER` is the boolean for decision 1.2's `reject_static_tracks` lever (lines 107-110), off unless the env is exactly `"1"`.

**Alternatives considered.** The doc's decision 2.12 sweep varied the `i -> j` stride (2 / 4 / 8) but kept the PnP horizon at keyframe+4 throughout; no alternative `KOFF` is recorded.

**Why this choice.** `GAP = 4` and `KOFF = 4` are probe constants that match the `(i, i+4, i+8)` triplet definition in the doc's 2.11 table header and the "PnP test at keyframe+4" labels of the E-threshold and trigger tables. The doc records no rationale for the 4-frame choice beyond the gap 2 / 4 / 8 sweep that followed it (where stride 4 is the middle row); it does not state a pipeline keyframe stride before 2.12 (decision 2.9's original bootstrap trigger was "10 px ~ 3 typical frames"). The two-frame `i` stride on line 85 and `GAP` / `KOFF` together bound how many triplets a scene yields (2077 over the 12 scenes at gap 4 with `KOFF=4`, 2053 with `KOFF=8`) — which is what makes the 2.11 run's horizon recoverable from its output. Whether `KOFF` should be `GAP` or `2*GAP` is not a design decision the doc records; it is a code/comment inconsistency and is listed under known limitations below.

### Lines 36-38: `E_THR`, `KF_MODE`, `KF_VAL` (and a mangled comment)

```
36: E_THR = float(os.environ.get("E_THR", "2.0"))
37: KF_MODE = os.environ.get("KF_MODE", "stride")           # stride | parallax | ratio : how the second keyframe j is chosen
38: KF_VAL = float(os.environ.get("KF_VAL", "0"))            # parallax: median track displacement (px) since i; ratio: surviving-track fraction            # RANSAC threshold for the bootstrap essential matrix             # reject_static_tracks lever: drop tracks with disp(i->j) < 1 px
```

**What it does.** `E_THR` is the RANSAC inlier threshold, in pixels, for `findEssentialMat` on line 116; the in-code default is 2.0 px (the pipeline's value before the 2026-09-05 revision of decision 2.10), while `tri_kf.sbatch` exports `E_THR=0.5` (the value 2.10 settled on). `KF_MODE` selects how the second keyframe `j` is chosen: `stride` (fixed `j = i + GAP`), `parallax` (first frame at which the median track displacement since `i` reaches `KF_VAL` px) or `ratio` (first frame at which the fraction of surviving tracks drops below `KF_VAL`). `KF_VAL` is that trigger's threshold; it is unused in `stride` mode. The tail of line 38 is three comments concatenated onto one line: the first belongs to `KF_VAL`, the second ("RANSAC threshold for the bootstrap essential matrix") is the stray comment for `E_THR` on line 36, and the third ("reject_static_tracks lever: drop tracks with disp(i->j) < 1 px") is the stray comment for `LEVER` on line 35. This is a formatting accident from an edit, not semantics; the variables behave as described.

**Alternatives considered.** Decision 2.10's options for the bootstrap: findEssentialMat + recoverPose, homography, H-vs-E model selection (ORB-SLAM), and RANSAC threshold 2 / 1 / 0.5 px. Decision 2.12's options for the keyframe trigger: fixed stride (2/4/8), median parallax since last KF >= P px (5/10/20), tracked-inlier ratio < r (0.9/0.8/0.7). The `KF_MODE` / `KF_VAL` pair is what implements the second and third families here; the stride family is `KF_MODE=stride` with `GAP` on the command line.

**Why this choice.** `E_THR`: decision 2.10 (revised 2026-09-05) chose 0.5 px because in this very benchmark "map-level PnP-at-+4 improves 4.19 -> 3.02 deg (gap 4), 3.58 -> 2.62 (gap 2)" going from 2.0 to 0.5 px (full table: 2.0 px 0.61 / 3.58, 0.55 / 4.19, 0.52 / 5.83 bad / PnP rot at gaps 2 / 4 / 8; 1.0 px 0.56 / 3.29, 0.52 / 3.66, 0.49 / 4.84; 0.5 px 0.54 / 2.62, 0.51 / 3.02, 0.48 / 4.49), consistent with the two-view E-diag (2 -> 0.5 px: rot 2.06 -> 1.23, tdir 31.6 -> 18.1 deg at gap 4). The doc's reading: "Only the sub-pixel minority of chained LK tracks is geometrically clean." `KF_MODE`/`KF_VAL`: decision 2.12 chose parallax 5 px, which in the sweep gives median KF gap 2 (p10-p90 1-8), PnP rot 2.96 deg (p90 10.6), and "never fires on a paused camera"; stride 2 is nominally better on gated pairs (2.62 deg) but "25% of 2-frame windows have < 2 px motion, where a stride keyframe would triangulate at ~zero baseline and pass cheir+rep" (the ratio family is discussed under lines 96-99). The 2.0 px in-code default is a leftover of the earlier pipeline value; the env override is the operative configuration.

### Lines 39-42: loading the scene: images and GT camera files

```
39: d = os.path.join(scene, "dense")
40: fs = sorted(glob.glob(os.path.join(d, "rgb", "*.png"))); n = len(fs)
41: G = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in fs]; H, W = G[0].shape
42: cams = [np.load(os.path.join(d, "cam", "%06d.npz" % i)) for i in range(n)]
```

**What it does.** Resolves the `dense/` subdirectory of the DROID wrist scene, lists the RGB PNGs in sorted (frame) order, and loads every frame as an 8-bit grayscale `uint8` array of shape `(H, W)`. `n` is the frame count; the whole scene is held in memory (dataset sequences run 57 to ~1700 frames; these 12 smoke scenes are 67 to 742 frames, 4243 in all). Name collision worth flagging: this `G` is the list of grayscale images, not the docstring's `G` (line 4, echoed in the usage line's `[G]`), which is the frame gap — the gap variable is `GAP`, line 33. These are the *native* 320x180 frames, not the 320x192 "cover" frames (uniform 1.067x scale + centre crop) the pipeline's eval loader produces per decision 1.1, so all pixel thresholds in this probe are in native-pixel units. Line 42 loads the per-frame `cam/%06d.npz` (keys `intrinsic` and `pose`) for every frame.

**Alternatives considered.** Running on the pipeline's 320x192 cover frames with the model-estimated focal (decision 0.1b) would have made the probe's pixel units identical to the pipeline's. Streaming frames instead of holding all in memory (audit item 3 concerns the pipeline, not this probe).

**Why this choice.** This probe, like the correspondence and PnP benchmarks, scores against GT "at native 320x180, calibrated K" so that the depth/pose references are read directly from the dataset without a resampling step. The focal-source check in the full pipeline later showed the focal *value* is immaterial (finetuned-model focal 111.9/6.93/0.871 vs calibrated 116.0/6.91/0.885 ATE/RPE-t/RPE-r); that check varied only the focal, not the native-vs-cover framing, so a framing effect on this probe's conclusions is unlikely but untested. Grayscale is mandatory for `goodFeaturesToTrack` / LK (decision 1.1).

### Lines 43-47: intrinsics, GT poses, GT depth, outlier mask

```
43: K = cams[0]["intrinsic"].astype(np.float64); Kinv = np.linalg.inv(K)
44: c2w = [c["pose"] for c in cams]
45: D = [np.load(os.path.join(d, "depth", "%06d.npy" % i)) for i in range(n)]
46: M = [cv2.imread(os.path.join(d, "outlier_mask", "%06d.png" % i), cv2.IMREAD_GRAYSCALE) > 0 for i in range(n)]
47: 
```

**What it does.** `K` is the 3x3 calibrated pinhole matrix of frame 0 (fx ~ 192 px at 320x180, no distortion), cast to `float64` as OpenCV's geometry functions require; one `K` is used for the whole scene. `Kinv` is computed but never used in this file. `c2w` is the list of 4x4 camera-to-world GT poses (the `pose` key is c2w, as the doc's dataset-facts section states; the relative transforms are built on line 112 by inverting). `D` is the list of GT depth maps (`float32`, `0` = invalid, ~88 % valid, 0.3-1.4 m typical). `M` is the boolean outlier mask per frame (`True` where the PNG is non-zero). What the mask covers is stated twice in the design doc and the two statements disagree: the dataset-facts section says it "is mostly the invalid-depth region. Its constant part is a rectification band on the left edge plus a gripper blob bottom-left (roughly the left ~110 columns)", while decision 1.2's probe finding says "GT outlier_mask does NOT cover the gripper". The doc is internally inconsistent on this point; the operative fact for this probe is that `zvalid` (line 114) excludes masked pixels but is not a gripper oracle. Line 47 is blank.

**Alternatives considered.** Decision 0.1b's intrinsics options: calibrated K from `cam/*.npz` (used here), one nominal dataset K, model-estimated focal from preds, self-calibration.

**Why this choice.** The calibrated K is the diagnostic choice the doc explicitly keeps for probes ("Calibrated-K copy kept as a one-off diagnostic"); the pipeline itself uses the model's per-frame focal (0.1b, revised 2026-09-06). GT depth and the outlier mask are consumed only in the `zvalid` scorer on line 114, i.e. only to decide which admitted points can be depth-checked, in line with the docstring's "GT depth / pose are used ONLY for scoring".

### Lines 48-53: rotation angle and the single LK hop with forward-backward check

```
48: def rot_angle(R): return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
49: def lk_hop(gi, gj, p):
50:     p1, st, _ = cv2.calcOpticalFlowPyrLK(gi, gj, p, None)
51:     p0b, stb, _ = cv2.calcOpticalFlowPyrLK(gj, gi, p1, None)
52:     ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm((p - p0b)[:, 0], axis=1) < 1.0)
53:     return p1, ok
```

**What it does.** `rot_angle` is the geodesic angle of a rotation matrix in degrees, `arccos((tr R - 1)/2)` with the argument clipped to `[-1, 1]` to survive floating-point rounding. `lk_hop` tracks a point array `p` of shape `(N, 1, 2)` `float32` (the layout `goodFeaturesToTrack` returns) from gray image `gi` to `gj` with `cv2.calcOpticalFlowPyrLK` at OpenCV defaults (21 px window, 3 pyramid levels, decision 1.5), which returns the new positions `p1` `(N,1,2)`, a status column `st` `(N,1)` `uint8` (1 = found) and per-point error (discarded). It then tracks `p1` back to `gi`, and a point is kept (`ok`) only if both hops succeeded and the round-trip landed within 1 px of the start. `p1` is returned for every point, including those with `ok == False`, so callers must mask.

**Alternatives considered.** Decision 1.5's correspondence benchmark (`corr_bench.py`) compared LK + forward-backward check, LK without the check ("LK no check": gap 1 183 / 91 correct, gap 4 167 / 31, 2 ms), ORB + ratio, SIFT + ratio, DIS flow at corners, DIS flow on an 8-px grid, and Farneback at corners; LK ran at its defaults in every case (`m_lk` in `corr_bench.py` takes `win` / `lvl` / `fb` keyword arguments but is registered only at their defaults, so no window / level sweep exists in the code or the doc).

**Why this choice.** Decision 1.5: "sparse LK at OpenCV defaults (21 px window, 3 levels) + forward-backward check at 1 px". The check costs a second LK call (4 vs 2 ms) and returns fewer gap-4 correspondences (103 vs 167) but gives "the best per-correspondence precision at gap 4 (in2 0.35 vs 0.29)"; per-point precision, not count, is what a PnP map needs. The 1 px round-trip tolerance is the probe value that the pipeline froze; it sits at the LK noise floor (the doc rejects a 1 px *PnP* threshold for the same reason in decision 2.7).

### Lines 54-59: chaining hops frame by frame

```
54: def lk_chain(i, j, p):
55:     """Track p from frame i to frame j one frame at a time (pipeline reality), fwd-bwd check per step."""
56:     ok = np.ones(len(p), bool); cur = p.copy()
57:     for f in range(i, j):
58:         nxt, okf = lk_hop(G[f], G[f + 1], cur); ok &= okf; cur = nxt
59:     return cur, ok
```

**What it does.** Propagates the point array from frame `i` to frame `j` (`j > i`) through every intermediate consecutive pair, applying `lk_hop` (with its forward-backward check) at each step and AND-ing the survival flags, so a track survives only if it passed the check at every hop. Positions of dead tracks keep being propagated (they are just garbage that the caller masks out). This is what the pipeline's persistent tracks (decision 1.6) do between keyframes.

**Alternatives considered.** A single direct LK hop `i -> j` (evaluated in the third E-diagnostic as "E(2px) from single-hop LK"); re-detecting corners each frame instead of persistent tracks (decision 1.6's first option).

**Why this choice.** Chained tracking is "pipeline reality" (decision 1.6: persistent tracks, top-up only at keyframes) and, as measured, better than single-hop for two-view geometry at every gap (2.06 / 31.6 vs 2.33 / 38.3 rot / tdir at gap 4, 8.22 / 41.0 vs 9.20 / 60.4 at gap 16). The known cost, also from that diagnostic, is chain drift: "beyond gap 8 they are worse than 3 px noise", which is why keyframes must be close (decision 2.12) and why this probe's `KF_MODE` sweep exists. The probe's "over a 15-frame gap in fast segments survival drops to ~15%" observation is the reason tracking must be frame-to-frame at all. One honesty note belongs here, because the design doc records none of it: the chaining in this function is not what produced the doc's 2.11 table. On identical start frames and the same `k = j + 4` horizon, the current file gates 1677 of 2077 triplets, while the 2.11 run gates 1373 of 2077 — and the doc carries both sides of the discrepancy itself: its 2.11 table reports E-pose `cheir+rep` `n_acc` 67, bad 0.53, PnP rot 3.98, while its E-threshold table's 2.0 px / gap-4 cell reports 0.55 / 4.19 for what is nominally the same configuration (gap 4, `k = j + 4`, `cheir+rep`, lever OFF, E RANSAC at 2 px). The chronology on scratch explains it: `out/` was written at 09:28 by `tri_bench.sbatch` (Slurm 45443315), `out_gap/` at 09:30 (gaps 8/16/32), and then `out_chain/` at 09:31 by a launcher literally named `tri_chain.sbatch` (Slurm 45443494) that re-ran gap 4 — and `out_chain/*.gap4` agrees cell for cell with `out_k4/*.gap4.lever0`, i.e. with the current script. The two pre-`out_chain` runs lose tracks with gap exactly as single LK hops would (gap 32: 11 gated of 1741 in `out_gap/` against 523 in `out_chain/`), while the chained runs degrade gently. Only the final revision of the script is on disk, so the exact diff is unrecoverable and this section does not assert more than the outputs show: the file as it stands does not reproduce the 2.11 table. What does survive is the decision — re-pooling the chained re-run gives the same ordering under the GT pose (PnP rot 0.78 none / 0.67 cheir / 0.59 cheir+rep / 0.94 cheir+par / 0.77 all; verified 2026-09-09 from `out_chain/*.gap4.json`, which the design doc never tabulates), with `cheir+rep` still best and the parallax floor still worse.

### Lines 60-64: two-view triangulation in cam-`i` coordinates

```
60: 
61: def triangulate(R, t, a, b):
62:     P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))]); P1 = K @ np.hstack([R, t.reshape(3, 1)])
63:     Xh = cv2.triangulatePoints(P0, P1, a.T, b.T); X = (Xh[:3] / Xh[3]).T  # in cam i
64:     Xj = (R @ X.T + t.reshape(3, 1)).T
```

**What it does.** Line 60 is blank. `triangulate` takes the relative pose `(R, t)` mapping cam-`i` coordinates into cam-`j` coordinates (`x_j = R x_i + t`, the convention `recoverPose` returns and the one line 112 builds from GT), and the matched pixel arrays `a` (in frame `i`) and `b` (in frame `j`), each `(N, 2)` `float64`. It forms the two 3x4 projection matrices `P0 = K[I|0]` and `P1 = K[R|t]` and calls `cv2.triangulatePoints`, which wants points as `2xN` (hence the transposes) and returns `4xN` homogeneous coordinates from a linear DLT solve. Dividing by the fourth row gives `X` `(N, 3)` in cam-`i` coordinates; a point at infinity (`Xh[3] == 0`) becomes `inf`/`nan`, which line 123 removes. `Xj` is the same point expressed in cam-`j` coordinates, needed for the cam-`j` cheirality test.

**Alternatives considered.** Optimal (Hartley-Sturm) triangulation via `cv2.correctMatches`, midpoint triangulation, or a non-linear refine of each point; none are benchmarked in the doc.

**Why this choice.** DLT triangulation is the OpenCV standard and what the pipeline does (`triangulate_pair` in `opencv_vo.py`); under the E pose the map error is dominated by the pose itself (~53 % bad points regardless of filter), so a better triangulator would not have moved the 2.11 result. The `x_j = R x_i + t` convention is verified end-to-end by the E-diag row "E from GT flow, no noise (convention check)": 0.01 / 0.0 deg.

### Lines 65-67: reprojection errors in both views

```
65:     def proj(P, X3):
66:         x = (P @ np.c_[X3, np.ones(len(X3))].T); return (x[:2] / x[2]).T
67:     ra = np.linalg.norm(proj(P0, X) - a, axis=1); rb = np.linalg.norm(proj(P1, X) - b, axis=1)
```

**What it does.** `proj` applies a 3x4 projection matrix to `(N, 3)` points (homogenised with a column of ones) and dehomogenises to `(N, 2)` pixels. `ra` and `rb` are the Euclidean reprojection residuals, in pixels, of each triangulated point against its own observation in frame `i` and frame `j`. A point behind a camera (`x[2] < 0`) still gets a finite residual here; cheirality is tested separately in the filters.

**Alternatives considered.** Reprojection bound in one view only; a symmetric or Sampson-distance bound. The doc records only the two-view 2 px bound.

**Why this choice.** Decision 2.11: the reprojection test is applied "in both keyframes" with the threshold reused from the PnP RANSAC (2.7), 2 px, so it adds no new parameter. The measured gain of adding it to cheirality (GT pose) is bad-depth 32 % -> 22 % and PnP rot 0.63 -> 0.56 deg.

### Lines 68-72: parallax angle and the return tuple

```
68:     # parallax angle between rays from the two centers (cam i at 0, cam j at -R^T t)
69:     Cj = -R.T @ t.reshape(3)
70:     r1 = X / np.linalg.norm(X, axis=1, keepdims=True); r2 = (X - Cj); r2 /= np.linalg.norm(r2, axis=1, keepdims=True)
71:     par = np.degrees(np.arccos(np.clip(np.sum(r1 * r2, 1), -1, 1)))
72:     return X, Xj, ra, rb, par
```

**What it does.** The optical centre of cam `j` in cam-`i` coordinates is `C_j = -R^T t` (inverse of `x_j = R x_i + t` applied to the origin). `r1` is the unit ray from cam `i`'s origin to each point, `r2` the unit ray from `C_j`; `par` is the angle between them in degrees, i.e. the triangulation (parallax) angle. Under the E pose `|t| = 1`, so `C_j` is at unit distance; the angle itself is scale-invariant, so this is fine. The function returns the cam-`i` points, cam-`j` points, both residuals and the parallax vector for the filters on lines 74-80.

**Alternatives considered.** Parallax floor values other than 1 deg; a ratio-of-baseline-to-depth test (equivalent up to a small-angle approximation). The doc only records 1 deg.

**Why this choice.** The 1 deg floor is the standard SLAM heuristic tested as decision 2.11's fourth and fifth options. It lost: with keyframe-scale baselines (~36 mm at 0.3-1.4 m depth) it "discards half the points and raises PnP failures 2x" (GT pose: `n_acc` 62 -> 43, fails 183 -> 398; E pose: 67 -> 47, 163 -> 329), and PnP rotation gets *worse* (GT 0.56 -> 0.87 deg). Parallax is therefore computed and reported but not used by the v0 pipeline.

### Lines 73-76: the filter table, first half

```
73: 
74: FILTERS = {
75:     "none":      lambda X, Xj, ra, rb, par: np.ones(len(X), bool),
76:     "cheir":     lambda X, Xj, ra, rb, par: (X[:, 2] > 0) & (Xj[:, 2] > 0),
```

**What it does.** Line 73 is blank. `FILTERS` maps each filter name to a boolean-mask function over the tuple `triangulate` returns. `none` admits everything (the "accept all" control). `cheir` requires positive `z` in both cam-`i` (`X[:, 2]`) and cam-`j` (`Xj[:, 2]`) coordinates, i.e. the point is in front of both cameras.

**Alternatives considered.** Cheirality in one camera only. Note that `recoverPose` (line 118) already applies a both-camera cheirality *count* to choose among the four `(R, t)` decompositions of `E`; this filter re-applies the same both-camera test per point after triangulation with the chosen pose, which the GT arm needs too (there `recoverPose` never runs). The doc lists only the both-camera version.

**Why this choice.** Decision 2.11: cheirality in both cameras is the first component of the accepted filter. Even alone it helps clearly under the GT pose (bad-depth 46 % -> 32 %, PnP rot 0.73 -> 0.63 deg) and modestly under the E pose (0.64 -> 0.54 bad, 4.18 -> 3.96 deg). `none` is retained purely as the control row.

### Lines 77-80: the filter table, second half

```
77:     "cheir+rep": lambda X, Xj, ra, rb, par: (X[:, 2] > 0) & (Xj[:, 2] > 0) & (ra < 2) & (rb < 2),
78:     "cheir+par": lambda X, Xj, ra, rb, par: (X[:, 2] > 0) & (Xj[:, 2] > 0) & (par >= 1.0),
79:     "all":       lambda X, Xj, ra, rb, par: (X[:, 2] > 0) & (Xj[:, 2] > 0) & (ra < 2) & (rb < 2) & (par >= 1.0),
80: }
```

**What it does.** `cheir+rep` adds `ra < 2` and `rb < 2` (both reprojection residuals under 2 px, strict inequality). `cheir+par` adds `par >= 1.0` deg instead. `all` applies both. The literals `2` and `1.0` are the values the docstring names; they are not env-configurable.

**Alternatives considered.** Per decision 2.11's option list; see lines 19-24.

**Why this choice.** `cheir+rep` is the decided filter (decision 2.11, 2026-09-05), best on every GT-pose metric (0.56 deg / 22 % bad / 62 points) while keeping 183 fails vs 377-398 for the parallax variants; the pipeline hard-codes the same test (`triangulate_pair` in `opencv_vo.py`: cheirality in both cameras + reprojection < 2 px in both views), applied KF-to-KF per decision 2.13. `all` is the "ORB-SLAM-style" full filter and is the best on GT-pose bad-depth (15 %) but loses on PnP rotation (0.74 deg) and fails (377); this is the evidence behind the doc's statement that the 1 deg floor "discards half the points and raises PnP failures 2x".

### Lines 81-84: result accumulators

```
81: res = {pc: {f: dict(triplets=0, n_acc=[], bad_frac=[], pnp_rot=[], pnp_tdir=[], pnp_inl=[], pnp_fail=0, skipped=0)
82:             for f in FILTERS} for pc in ("E", "GT")}
83: gated = 0; total = 0
84: kf_gaps = []
```

**What it does.** `res[pose_source][filter]` holds, per (E / GT) x (5 filters) cell, the triplet count, per-triplet lists of admitted-point count, bad-depth fraction, PnP rotation error, PnP translation-direction error and PnP inlier count, plus two integer counters: `pnp_fail` (PnP under 20 inliers) and `skipped` (filter left fewer than 4 points, line 125). `total` counts triplets that reached a second keyframe `j` (incremented on line 100, before the `j -> k` chaining, so a triplet that then loses its tracks still counts); `gated` counts those that passed the 20-track floor and the 2 px motion gate (and the lever's floor when on). `kf_gaps` records `j - i` per triplet, meaningful in the adaptive `KF_MODE`s.

**Alternatives considered.** None recorded; this is bookkeeping. The doc's tables are pooled medians of these lists across the 12 scenes.

**Why this choice.** The E / GT split is the paired design explained under lines 7-12. The JSON (line 146) exposes `total` and `gated` as `triplets_total` / `triplets_gated` and the gap list as `kf_gaps`. The 2.11 run's "1373 of 2077 triplets" is `gated` / `total` summed over the 12 per-scene JSONs (verified 2026-09-08 on `tri_bench/out/*.json`). The keyframe-trigger sweep's "fired/starts" column is the same pair of counters: summing `triplets_gated` / `triplets_total` over the 12 per-scene JSONs reproduces all nine rows exactly — stride 2 1555/2089, stride 4 1677/2077, stride 8 1651/2053, parallax 5 1828/1942, parallax 10 1783/1858, parallax 20 1623/1695, ratio 0.9 1667/2066, ratio 0.8 1713/2047, ratio 0.7 1695/1996 (verified 2026-09-09 against `out_ethr/` and `out_kf/`). That makes it directly comparable with the 2.11 run's 1373/2077, and the two share a denominator because they share the `k = j + 4` horizon (lines 31-35). So the gated gap — 1373 against 1677 on the very same start frames — is *not* a `k`-offset effect. Its cause lies in the earlier script revision that produced the 2.11 run (honesty note under lines 54-59), which keeps far fewer tracks alive across a triplet: its pooled `n_acc` median under the `none` filter is 87, against 124 for the current code on the same start frames. Both exits before `gated` — the 20-track floor on line 104 and the 2 px median-displacement gate on line 106 — are sensitive to that, and neither is counted in the JSON, so which of the two dropped the 304 missing triplets is not recoverable from the outputs. The design doc records neither the difference nor any cause. `kf_gaps` is what the "KF gap median (p10-p90)" column summarises (stride 2: 2; parallax 5 px: 2 (1-8); ratio 0.7: 7 (2-19)).

### Lines 85-87: the triplet loop and corner detection at frame `i`

```
85: for i in range(0, n - GAP - KOFF, 2):
86:     p0 = cv2.goodFeaturesToTrack(G[i], 1000, 0.01, 5)
87:     if p0 is None or len(p0) < 20: continue
```

**What it does.** Iterates over first-keyframe indices `i` with stride 2, leaving room for `j = i + GAP` and `k = j + KOFF` inside the sequence (in the adaptive modes `j` can exceed `i + GAP`, which the inner loop's own bound on line 93 handles). Consecutive triplets share frames, so the per-scene samples are correlated, not independent. `goodFeaturesToTrack(image, maxCorners=1000, qualityLevel=0.01, minDistance=5)` returns Shi-Tomasi corners as `(N, 1, 2)` `float32` or `None` if there are none; the triplet is dropped if fewer than 20 corners are found (the doc's probe shows ~140-300 corners per frame, so this is rare). No detection mask is passed: the gripper is not masked (decision 1.2, "NO MASK for v0").

**Alternatives considered.** Decision 1.3: FAST, ORB, SIFT/AKAZE, fixed grid. Decision 1.4: grid bucketing / top-up (rejected, "none"). Decision 1.2: fixed polygon, Otsu temporal-variance mask, flow-based rejection, GT outlier mask.

**Why this choice.** Decision 1.3 froze exactly these Shi-Tomasi parameters ("cap never reached, ~140-300 corners/frame"); 1.4 found `minDistance=5` already spreads corners at this resolution. No mask because the probe on the RAIL scene found the gripper "attracts few corners (median 12% zero-motion tracks on moving frames, max 42%)", decision 1.2 records the GT outlier mask as not covering it, and the Otsu mask over-masks (58 %). The stride-2 loop is a sampling density choice (2077 triplets / 12 scenes at gap 4); the doc does not record an alternative.

### Lines 88-90: stride mode and the adaptive branch

```
88:     if KF_MODE == "stride":
89:         j = i + GAP; p1, ok1 = lk_chain(i, j, p0)
90:     else:
```

**What it does.** In `stride` mode the second keyframe is fixed at `j = i + GAP` and the corners are chained there; `p1` are the frame-`j` positions and `ok1` the survival flags. Any other `KF_MODE` enters the adaptive branch on lines 91-99.

**Alternatives considered.** Decision 2.12's stride options 2 / 4 / 8 are all this branch with different `GAP` on the command line.

**Why this choice.** Stride rows in the sweep: 2 -> 2.62 deg (p90 9.9), 5.9 % fail; 4 -> 3.02 (12.1), 7.2 %; 8 -> 4.49 (19.7), 13.2 %. Monotonically better as keyframes get closer. These three rows are the 0.5 px row of the E-threshold table (Slurm 45443823) reproduced in the sweep table, not a separate run of `tri_kf.sbatch`. The stride family was nevertheless *not* chosen (decision 2.12) because it cannot avoid declaring a keyframe on a paused camera; a stride-2 keyframe on one of the 25 % of 2-frame windows with < 2 px motion "would triangulate at ~zero baseline and pass cheir+rep". In this probe the 2 px gate on line 106 hides that failure mode, which is why the doc notes "Stride rows are conditioned on the 2 px motion gate".

### Lines 91-95: adaptive trigger, walking forward one frame at a time

```
91:         # walk forward one frame at a time until the trigger fires (max 32 frames)
92:         cur = p0.copy(); ok1 = np.ones(len(p0), bool); j = None
93:         for f in range(i, min(i + 32, n - KOFF - 1)):
94:             nxt, okf = lk_hop(G[f], G[f + 1], cur); ok1 &= okf; cur = nxt
95:             if ok1.sum() < 20: break
```

**What it does.** Re-implements `lk_chain` inline so the trigger can be evaluated after every hop. Starting at `i`, it hops one frame at a time for at most 32 frames, AND-ing the survival flags; if fewer than 20 tracks survive it gives up (`j` stays `None`). The `range` upper bound is exclusive, so `f` runs no further than `n - KOFF - 2`; it is the candidate keyframe `j = f + 1` that can reach `n - KOFF - 1`, which is what keeps `k = j + KOFF <= n - 1` inside the sequence.

**Alternatives considered.** The 32-frame cap and the 20-track floor are probe constants; the doc records no alternatives. The pipeline's bootstrap guard is the analogous test but is not a fixed 20: `opencv_vo.py` line 369 reads `if npass >= args.min_inliers:`, so in the frozen v0 configuration (which passes `--min_inliers 30`) that guard is 30. Decision 2.9's prose still says "cheirality count >= 20 (same floor as 3.1)", but 3.1 was later DECIDED at 30 in sweep 2, so the literal 20 there is stale text, not the operative value.

**Why this choice.** 32 frames is well beyond any useful keyframe spacing (the sweep's widest p90 gap is 20 frames, at parallax 20 px; ratio 0.7 reaches 19; and the E-diag shows chained tracks are worse than 3 px noise beyond gap 8), so the cap only prevents runaway walks on a paused camera. The 20-track floor mirrors the 3.1 floor of the time.

### Lines 96-99: the parallax and ratio trigger rules

```
96:             fired = (np.median(np.linalg.norm((cur - p0)[ok1, 0], axis=1)) >= KF_VAL) if KF_MODE == "parallax" else (ok1.mean() < KF_VAL)
97:             if fired: j = f + 1; break
98:         if j is None: continue
99:         p1 = cur
```

**What it does.** After each hop the trigger is tested. `parallax`: the median, over surviving tracks, of the pixel displacement between the current position and the detection position at `i` reaches `KF_VAL` px (this is *total* displacement since the last keyframe, not per-frame flow, and it is a median so a static minority of tracks cannot suppress it). `ratio`: the surviving fraction of the originally detected corners (`ok1.mean()`) drops below `KF_VAL`. When it fires, `j` is the frame just reached; if the walk ends without firing the triplet is skipped. `p1` takes the frame-`j` positions.

**Alternatives considered.** Decision 2.12: parallax 5 / 10 / 20 px; ratio 0.9 / 0.8 / 0.7 (all six run by `tri_kf.sbatch` at `E_THR=0.5`, `KOFF=4`, `LEVER=0`, Slurm 45443937). H-vs-E model selection is a bootstrap alternative (2.9/2.10), not a keyframe trigger.

**Why this choice.** Decision 2.12 picked parallax >= 5 px (and lowered decision 2.9's bootstrap threshold from 10 to 5 px to share the parameter). Sweep numbers: parallax 5 px: gap 2 (1-8), 91 points, 0.56 bad, 2.96 deg (10.6), 11.9 % fail; 10 px: gap 4 (2-13), 3.23 (11.9); 20 px: gap 7 (3-20), 4.25 (16.0); ratio 0.9: gap 2 (1-8), 2.91 (10.4), 7.4 %; 0.8: 3.68 (14.4); 0.7: 4.78 (19.1). "Every family improves monotonically as keyframes get closer; all adaptive triggers at their tightest setting converge to a median gap of 2." On the sweep's own numbers ratio 0.9 is slightly better than parallax 5 px (2.91 vs 2.96 deg PnP rot; 7.4 % vs 11.9 % fail); the doc nevertheless rules for parallax on two stated grounds: a displacement trigger "never fires on a paused camera", and it shares its one parameter `P` with the bootstrap trigger of decision 2.9 ("shares P with 2.9; 2.9's P drops 10 -> 5"). The doc also notes that "parallax/ratio rows fire on their own rule, so their populations differ slightly" from the gated stride rows.

### Lines 100-104: frame `k`, chaining `j -> k`, assembling the triplet

```
100:     k = j + KOFF; total += 1; kf_gaps.append(j - i)
101:     p2, ok2 = lk_chain(j, k, p1)
102:     ok = ok1 & ok2
103:     a = p0[ok, 0].astype(np.float64); b = p1[ok, 0].astype(np.float64); c = p2[ok, 0].astype(np.float64)
104:     if len(a) < 20: continue
```

**What it does.** The PnP test frame is `k = j + KOFF`; the triplet is counted in `total` and its keyframe gap recorded (note `kf_gaps` is appended *before* the motion gate, so it includes triplets that the gate later drops). The tracks are chained on from `j` to `k` and a track is kept only if it survived both legs. `a`, `b`, `c` are the `(N, 2)` `float64` pixel positions of the surviving tracks in frames `i`, `j`, `k` (the `[ok, 0]` indexing strips the singleton axis of the `(N,1,2)` LK layout; `float64` is what `findEssentialMat` / `solvePnPRansac` accept without type mismatch). Fewer than 20 three-frame tracks aborts the triplet, after `total` but before `gated` has been counted.

**Alternatives considered.** Re-detecting corners at `j` and associating them to map points by descriptor (the pipeline has no descriptors; decision 2.14 notes "no re-association without descriptors").

**Why this choice.** Using the *same* tracks for the map (`a`, `b`) and the query (`c`) is what the pipeline does: every map point is born at a keyframe and observed in later frames by the same persistent LK track (decision 1.6). The 20 floor matches the 3.1 floor of the time.

### Lines 105-106: the 2 px motion gate

```
105:     disp_ij = np.linalg.norm(b - a, axis=1)
106:     if np.median(disp_ij) < 2.0: continue   # parallax gate: only pairs where the scene moved
```

**What it does.** `disp_ij` is the per-track pixel displacement between frames `i` and `j`. If the median is below 2 px the triplet is discarded: the essential matrix is degenerate without baseline (the RAIL scene is stationary for frames 0-15, GT rot 0.03 deg, |t| = 0.2 mm), and scoring it would measure noise. This gate runs in all `KF_MODE`s.

**Alternatives considered.** Scoring all pairs (the E-diag's model rows are "all pairs, no gating"); a stricter gate (the pipeline's bootstrap trigger is 5 px median displacement, decision 2.9).

**Why this choice.** The gate is the same 2 px rule used to define "parallax-gated" pairs throughout the doc's benchmarks (corr benchmark: "Parallax-gated gap-4 pairs (median flow >= 2 px, n=814)"; this run: 1373 of 2077 triplets gated). It is also the *arming* condition of the lever (decision 1.2): the lever may drop static tracks only "on frames where the median track displacement >= 2 px (parallax present, so a non-moving track is provably not scene geometry)". Note the honesty consequence spelled out in the doc: stride rows are conditioned on this gate whereas the pipeline's stride keyframes would not be.

### Lines 107-111: the `reject_static_tracks` lever (decision 1.2)

```
107:     if LEVER:
108:         keep = disp_ij >= 1.0
109:         if keep.sum() < 20: continue
110:         a, b, c = a[keep], b[keep], c[keep]
111:     gated += 1
```

**What it does.** When `LEVER=1`, tracks whose `i -> j` displacement is under 1 px are dropped from all three frames before any geometry (E, triangulation, PnP), and the triplet is abandoned if fewer than 20 remain. Because line 106 has already established a median displacement >= 2 px, a sub-pixel track on this triplet is provably not moving with the scene (it is the image-static gripper, which "votes for zero motion in PnP"). `gated` counts the triplets that reach the geometry stage.

**Alternatives considered.** Decision 1.2: no mask (used), fixed polygon, Otsu temporal-variance mask, flow-based rejection, GT outlier mask; and for the lever, a relative variant ("drop < 20% of median disp") tried in the E-diag.

**Why this choice.** The lever's design and its 1 px / 2 px thresholds are from decision 1.2, where it cut parallax-gated E-rotation error 2.14 -> 1.70 deg (LK) in the corr benchmark; static share of LK tracks is 16-53 % per scene, median ~25 %. In *this* probe every run the doc reports used lever OFF (docstring line 25; `tri_kf.sbatch` exports `LEVER=0`), consistent with the pipeline default at the time; the switch exists here, and a `LEVER=1` launcher does sit in the scratch staging directory (`tri_k4.sbatch`, not in the repo) together with its lever-ON per-scene JSONs (`out_k4/*.lever1.json`, gaps 2 / 4 / 8 / 16), but they were never pooled and the doc records no lever-ON `tri_bench` numbers. The lever ruling came from elsewhere: the E-diag (whose whole table is lever ON) found "Static-track removal (absolute 1 px or relative 20%) does not change E" (its "relative static lever (drop < 20% of median disp)" row reads 1.96 / 32.5 vs 2.06 / 31.6 for the 2 px pipeline row at gap 4; the absolute-1 px variant has no separate row), while pipeline sweeps 1-3 found it better on every end metric (111.9/6.93/0.871 ON vs 115.6/8.40/0.971 OFF), so the default was flipped to ON by user decision on 2026-09-05 (`--no_lever` disables). The doc also notes that the lever's thresholds were "designed with GT-aided inspection" (audit item 5, ruling pending).

### Lines 112-114: GT relative poses and the GT-depth reference for frame `i`

```
112:     T_ji = np.linalg.inv(c2w[j]) @ c2w[i]; T_ki = np.linalg.inv(c2w[k]) @ c2w[i]
113:     u = np.clip(np.round(a[:, 0]).astype(int), 0, W - 1); v = np.clip(np.round(a[:, 1]).astype(int), 0, H - 1)
114:     zgt = D[i][v, u]; zvalid = (zgt > 0) & (~M[i][v, u])
```

**What it does.** `T_ji = c2w_j^{-1} c2w_i` is the 4x4 transform taking cam-`i` coordinates to cam-`j` coordinates (`X_j = T_ji X_i`), i.e. exactly the `x_j = R x_i + t` convention of `recoverPose` and `triangulate`; `T_ki` likewise for `k`. This is the w2c-style relative pose, obtained from the c2w GT by inversion (the doc's plumbing note warns that `solvePnP` returns world-to-camera, so consistency here matters). Lines 113-114 look up GT depth at the rounded, clipped pixel of each track's frame-`i` position (`D[i][v, u]`: row = `y`, column = `x`), and mark it valid when depth is non-zero and the pixel is outside the outlier mask. Nearest-pixel lookup (no interpolation) is used.

**Alternatives considered.** Bilinear depth interpolation; excluding a hand-drawn gripper region from `zvalid`. Neither is in the doc.

**Why this choice.** GT pose and depth are used only to score (docstring line 25). The doc's dataset facts confirm `pose` is c2w and depth `0` means invalid. A caveat the doc records: "GT depth is defined ON the gripper in many scenes, so 'valid GT depth' is not a gripper oracle", so `bad_frac` on high-gripper scenes counts static-gripper points against the scene-depth reference (and, per lines 43-47, the doc is inconsistent about whether the outlier mask covers the gripper). A second caveat: "a ~20% depth-vs-pose scale mismatch exists in some scenes", which the single median scale on line 129 absorbs.

### Lines 115-118: the GT pose arm and the E-matrix pose arm

```
115:     poses = {"GT": (T_ji[:3, :3], T_ji[:3, 3])}
116:     E, inl = cv2.findEssentialMat(a, b, K, method=cv2.RANSAC, prob=0.999, threshold=E_THR)
117:     if E is not None and E.shape == (3, 3) and inl is not None:
118:         _, Re, te, _ = cv2.recoverPose(E, a, b, K, mask=inl.copy()); poses["E"] = (Re, te.reshape(3))
```

**What it does.** The `GT` arm uses the rotation and *metric* translation from `T_ji`. `cv2.findEssentialMat(points1, points2, cameraMatrix, method=RANSAC, prob=0.999, threshold=E_THR)` takes pixel coordinates and the calibrated `K`, normalises internally, and returns the 3x3 essential matrix plus an `(N, 1)` `uint8` inlier mask; `threshold` is the RANSAC inlier distance *in pixels* (OpenCV divides it by the focal internally to compare in normalised coordinates, which is why the pipeline, working in normalised coordinates itself with per-frame `K`, passes `E_THRESH / (0.5 * (Ka[0, 0] + Kb[0, 0]))`, `opencv_vo.py` line 366). Pitfalls guarded on line 117: the function can return `None`, and when the 5-point solver yields several candidate solutions it can return a stacked `(3k, 3)` matrix; both cases skip the E arm for that triplet (the GT arm still runs, so the two arms' populations can differ slightly). `cv2.recoverPose(E, points1, points2, K, mask=...)` decomposes `E` into the four `(R, t)` candidates and picks the one with the most points in front of *both* cameras, returning `(n_cheirality_inliers, R, t, mask)`; `t` is unit-norm (this is where the map's arbitrary scale comes from) and `(R, t)` maps points from the first image's camera to the second's, matching `T_ji`. `mask` is modified in place, hence `inl.copy()`. The inlier count is discarded (`_`): unlike the pipeline's bootstrap (decision 2.9), this probe applies no cheirality-count >= 20 guard, so pure-rotation pairs are not rejected here.

**Alternatives considered.** Decision 2.10: homography, H-vs-E model selection, RANSAC thresholds 2 / 1 / 0.5 px; the E-diag also ran MAGSAC++ at 2 px (1.34 / 22.9 at gap 4), a 2-view Sampson refinement of E (1.70 / 29.0), and alternative correspondence sources.

**Why this choice.** Essential matrix only, RANSAC, 0.999 confidence: decision 2.10. The threshold is the `E_THR` env, whose 0.5 px decided value comes from this probe's own E-threshold table (see lines 36-38) and the two-view diagnostics; MAGSAC++ at 2 px (1.34 deg) was beaten by plain RANSAC at 0.5 px (1.23 deg), so no USAC variant was adopted. The docstring's "RANSAC 2 px" and the code default `2.0` describe the pipeline *before* the 2026-09-05 revision. The doc's key reading of this arm: the E pose's ~30 deg translation-direction error at 36 mm baselines is the map arm's bottleneck, and "is NOT specific to classical VO: the model has the same 30-40 deg error vs GT" (finetuned CUT3R 2.22 / 31.7 at gap 4).

### Lines 119-122: triangulate under each pose, loop over filters

```
119:     for pc, (R, t) in poses.items():
120:         X, Xj, ra, rb, par = triangulate(R, t, a, b)
121:         for fname, ffn in FILTERS.items():
122:             r = res[pc][fname]; r["triplets"] += 1
```

**What it does.** For each available pose source (`GT` always; `E` when lines 116-118 succeeded), triangulates all surviving `i <-> j` correspondences once, then evaluates every filter on the same triangulation, incrementing that cell's triplet counter. Triangulation is done once per pose and shared across filters, so the filters are compared on identical 3D points.

**Alternatives considered.** None; this is the paired-comparison scaffold.

**Why this choice.** Same points, same pose, five filters: differences between the rows of the 2.11 table are attributable to the filter alone. The E / GT pairing across the outer loop attributes the residual to the pose.

### Lines 123-125: apply the filter, drop non-finite points, skip tiny maps

```
123:             sel = ffn(X, Xj, ra, rb, par) & np.isfinite(X).all(1)
124:             r["n_acc"].append(int(sel.sum()))
125:             if sel.sum() < 4: r["skipped"] += 1; continue
```

**What it does.** `sel` is the filter's boolean mask AND-ed with a finiteness test on all three coordinates, which removes points that `triangulatePoints` placed at infinity (`Xh[3] == 0`, giving `inf`/`nan` after dehomogenisation); this applies to the `none` filter too, so "accept all" really means "accept all finite". `n_acc` records the admitted count. Fewer than 4 points is below what `solvePnPRansac` accepts, so the triplet is counted as `skipped` for this cell and no further metrics are recorded.

**Alternatives considered.** None recorded.

**Why this choice.** The `n_acc` column of the 2.11 table (E: 87 / 72 / 67 / 47 / 43; GT: 87 / 72 / 62 / 43 / 34 for none / cheir / cheir+rep / cheir+par / all) is this list's pooled median; it is the direct evidence for the "parallax discards half the points" statement in decision 2.11. `skipped` is a counter separate from `pnp_fail` in the JSON, so a starved map is distinguishable there from a solver failure — but that separation is not preserved by the design doc's trigger-sweep `fail%` column, which reproduces only as `(pnp_fail + skipped) / triplets` (lines 136-139).

### Lines 126-130: depth accuracy of the admitted map (`bad_frac`)

```
126:             # depth accuracy of admitted points (single median scale)
127:             sv = sel & zvalid
128:             if sv.sum() >= 5:
129:                 s = np.median(zgt[sv] / np.maximum(X[sv, 2], 1e-9))
130:                 rel = np.abs(s * X[sv, 2] - zgt[sv]) / zgt[sv]; r["bad_frac"].append(float((rel > 0.2).mean()))
```

**What it does.** Restricts to admitted points with valid GT depth. With at least 5 such points, computes one scale factor `s` as the median ratio of GT depth to triangulated cam-`i` depth (`X[:, 2]`, clamped at `1e-9` so a zero or negative depth, possible under the `none` filter, does not divide by zero or flip the sign; a negative depth then maps to a huge ratio, which the median ignores) and reports the fraction of points whose scaled depth differs from GT by more than 20 % relative. Under the GT pose `s` should be ~1 (the doc notes a ~20 % depth-vs-pose scale mismatch in some scenes, which this absorbs); under the E pose `s` converts the unit-baseline map to metres.

**Alternatives considered.** A least-squares or Sim(3) scale, a per-point relative tolerance other than 20 %, or scoring by 3D distance instead of depth. The doc records only this definition.

**Why this choice.** This gives the "bad" column: E pose 0.64 / 0.54 / 0.53 / 0.42 / 0.40 and GT pose 0.46 / 0.32 / 0.22 / 0.29 / 0.15 for none / cheir / cheir+rep / cheir+par / all, and the E-threshold table's "bad" entries (0.61 -> 0.54 at gap 2 for 2.0 -> 0.5 px). The doc's headline reading rests on it: "with the E-matrix pose the admitted map is ~53% bad ... vs ... the GT pose on the SAME tracks and filter". The single median scale is justified by the docstring itself (lines 13-14): the E map has arbitrary scale, and the GT-pose map is aligned the same way "for comparability", so the two arms' `bad_frac` are computed by one rule.

### Lines 131-135: PnP of frame `k` against the admitted map

```
131:             # PnP of frame k against the admitted map
132:             Xm = np.ascontiguousarray(X[sel]); xk = np.ascontiguousarray(c[sel])
133:             try:
134:                 okp, rvec, tvec, inlp = cv2.solvePnPRansac(Xm, xk, K, None, flags=cv2.SOLVEPNP_ITERATIVE,
135:                                                            reprojectionError=2.0, confidence=0.999, iterationsCount=1000)
```

**What it does.** `Xm` `(M, 3)` are the admitted map points in cam-`i` coordinates (the "world" of this mini-map), `xk` `(M, 2)` their observations in frame `k`. Boolean-mask indexing in numpy already returns a fresh contiguous copy, so `ascontiguousarray` is a defensive no-op here that guarantees the C-contiguous `float64` layout OpenCV requires. `cv2.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs=None, flags=SOLVEPNP_ITERATIVE, reprojectionError=2.0, confidence=0.999, iterationsCount=1000)` returns `(retval, rvec, tvec, inliers)`: `rvec`/`tvec` are the *world-to-camera* (cam-`i`-to-cam-`k`) rotation vector and translation, and `inliers` is an `(n, 1)` `int32` index array (or `None`). `useExtrinsicGuess` is left at its default `False` (decision 2.5: no initial guess). `distCoeffs=None` because the dataset intrinsics carry no distortion. The `try` guards the `cv2.error` that OpenCV raises on degenerate input.

**Alternatives considered.** Decision 2.4: EPnP, P3P/AP3P, SQPnP (+/- LM). 2.5: identity or previous-motion guess. 2.6: USAC/MAGSAC++. 2.7: the OpenCV default 8 px, or 1-3 px. 2.8: the default 0.99 / 100.

**Why this choice.** All decided values: ITERATIVE (2.4; all variants within 0.05 deg / 2 mm, differences are noise), no guess (2.5; zero failures on 2118 pairs without one), plain RANSAC (2.6; identity-voting outliers cost only 1.10 -> 1.14 deg), 2 px (2.7; 8 px = 2.2 deg at f ~ 210 and admits gripper tracks, 1 px is at the LK noise floor, ~65 % of consecutive-frame tracks fall within 2 px of GT), 0.999 / 1000 (2.8; 1-2 ms per frame, removes the iteration cap as a variable). Note the 2 px threshold is the *same* 2 px that `cheir+rep` uses for triangulation acceptance, which is why 2.11 adds "no new parameter".

### Lines 136-139: PnP failure handling

```
136:             except cv2.error:
137:                 okp = False
138:             if not okp or inlp is None or len(inlp) < 20: r["pnp_fail"] += 1; continue
139:             inlp = inlp[:, 0]
```

**What it does.** An OpenCV exception is treated as a failed solve. A solve that reports failure, returns no inlier array, or returns fewer than 20 inliers is counted in `pnp_fail` and yields no pose metrics for this cell (so the rotation / direction medians are conditioned on success; the fail count is reported alongside to make that visible). Line 139 flattens the `(n, 1)` inlier index array to `(n,)` for indexing.

**Alternatives considered.** Decision 3.1: floors 10 / 20 / 30 or a ratio. 3.2: what to do on failure (hold / constant velocity / drop), which does not apply to a per-triplet probe.

**Why this choice.** 20 was the 3.1 floor when this benchmark ran (docstring line 18); the pipeline later moved to 30 (sweep 2: 0.861 vs 0.923 RPE-r). The `fails` column of the 2.11 table (E: 96 / 159 / 163 / 329 / 331; GT: 117 / 170 / 183 / 398 / 377) is this raw counter, verified cell for cell against `out/*.json` (that run's E-arm `skipped` counts are 0 / 0 / 0 / 60 / 66, so a summed column would have read 96 / 159 / 163 / 389 / 397). The trigger sweep's "fail%" column is a different quantity: it reproduces exactly as `(pnp_fail + skipped) / triplets` on all nine rows (5.9 / 7.2 / 13.2 for stride 2 / 4 / 8; 11.9 / 11.9 / 14.8 for parallax 5 / 10 / 20; 7.4 / 9.0 / 12.0 for ratio 0.9 / 0.8 / 0.7), whereas `pnp_fail / triplets` alone gives 5.4 / 7.1 / 13.0 / 11.7 / 11.9 / 14.8 / 7.2 / 8.9 / 12.0 and misses six of the nine — every row except parallax 10 / 20 and ratio 0.7, whose `skipped` counts in that cell are 1 / 0 / 0 (verified 2026-09-09 against `out_ethr/` and `out_kf/`; `skipped` is small everywhere here, 0-8 per row, so the two definitions differ by at most half a point, 5.40 vs 5.92 on the widest row). So in that column the starved-map cases are folded into the failure rate. Either way the counter is the evidence that the parallax filter "raises PnP failures 2x". Reporting failures separately from accuracy follows decision 0.4's principle ("Separates lost-tracking from drift").

### Lines 140-142: LM refinement and rotation error

```
140:             rvec, tvec = cv2.solvePnPRefineLM(Xm[inlp], xk[inlp], K, None, rvec, tvec)
141:             Rk, _ = cv2.Rodrigues(rvec); tk = tvec.reshape(3)
142:             r["pnp_rot"].append(float(rot_angle(Rk.T @ T_ki[:3, :3]))); r["pnp_inl"].append(int(len(inlp)))
```

**What it does.** `cv2.solvePnPRefineLM(objectPoints, imagePoints, K, distCoeffs=None, rvec, tvec)` runs Levenberg-Marquardt on the RANSAC inliers only, starting from the RANSAC pose, and returns refined `(rvec, tvec)`. `cv2.Rodrigues` converts the rotation vector to the 3x3 matrix `Rk` (cam-`i` -> cam-`k`). The rotation error is the geodesic angle of `Rk^T R_gt`, with `R_gt = T_ki[:3, :3]` in the same cam-`i` -> cam-`k` convention (line 112), so no inversion is needed. The inlier count is stored for the "med inl"-style diagnostics.

**Alternatives considered.** Decision 2.4's "+/- solvePnPRefineLM" for every solver; `solvePnPRefineVVS` as the other OpenCV refiner (not in the doc).

**Why this choice.** Decision 2.4 keeps RefineLM even though it is "a no-op after ITERATIVE-RANSAC" (the PnP benchmark row reads "ITERATIVE (+LM identical)"), so that the refine step exists for the P3P / SQPnP swaps where it does matter (P3P/AP3P gap-4 trans 10.2 -> 8.7 mm with LM). `pnp_rot` is the headline column of all three tables produced by this file: 2.11 (E ~4 deg vs GT 0.56 deg), E-threshold (4.19 -> 3.02 deg at gap 4), trigger sweep (2.62 deg stride 2 ... 4.78 deg ratio 0.7). For calibration, the doc pairs these against oracle-3D PnP (1.10 deg at gap 4) and the finetuned model's relative rotation (2.2 deg at gap 4).

### Lines 143-145: translation-direction error

```
143:             tg = T_ki[:3, 3]
144:             if np.linalg.norm(tg) > 2e-3 and np.linalg.norm(tk) > 1e-9:
145:                 r["pnp_tdir"].append(float(np.degrees(np.arccos(np.clip(np.dot(tg, tk) / np.linalg.norm(tg) / np.linalg.norm(tk), -1, 1)))))
```

**What it does.** `tg` is the GT cam-`i` -> cam-`k` translation in metres; `tk` the PnP translation in map units (metric under the GT arm, arbitrary under the E arm). Only the *direction* is compared, as the angle between the two vectors, and only when the GT motion exceeds 2 mm (below that the GT direction is itself ill-defined; the dataset's per-step median is 9 mm) and the PnP translation is non-zero. Triplets failing either guard simply contribute no `pnp_tdir` sample.

**Alternatives considered.** Scoring translation *magnitude* after the median scale (as the PnP-variant benchmark does with oracle 3D, in mm); the doc's map-level tables report only rotation and use t-direction in the two-view E-diag.

**Why this choice.** With an arbitrary-scale map, direction is the only scale-free translation quantity; the same t-dir definition is used in the E-diag, where it exposes the ~30 deg two-view direction error that the doc identifies as the bottleneck ("translation direction ~40 deg off at 36 mm baselines"). The 2 mm guard avoids the degenerate case that the probe evidence documents (frames 0-15 of RAIL: |t| = 0.2 mm). The `pnp_tdir` values are stored in the JSON but are not tabulated in the design doc's tables for this probe.

### Lines 146-147: output

```
146: json.dump(dict(scene=os.path.basename(scene), n=n, gap=GAP, koff=KOFF, lever=LEVER, e_thr=E_THR, triplets_total=total, triplets_gated=gated, kf_mode=KF_MODE, kf_val=KF_VAL, kf_gaps=kf_gaps, res=res), open(out, "w"))
147: print(os.path.basename(scene), n, "gated", gated, "of", total, "done")
```

**What it does.** Writes one JSON per scene with the full configuration echoed (`gap`, `koff`, `lever`, `e_thr`, `kf_mode`, `kf_val`), the triplet counts (`triplets_total`, `triplets_gated`), the per-triplet keyframe gaps (`kf_gaps`) and the entire `res` structure of raw per-triplet lists, so the pooling script (not in the repo) can compute medians / p90s across scenes rather than averaging per-scene aggregates. The final print is the one line that reaches the `.log` file `tri_kf.sbatch` redirects to.

**Alternatives considered.** Writing per-scene medians only; the doc's tables are all *pooled* medians ("Medians pooled over ~2100 pairs per gap"), which needs the raw lists.

**Why this choice.** Echoing the configuration into the JSON is what makes the E-threshold and trigger sweeps auditable after the fact (their JSONs on scratch carry `koff: 4`, `lever: false`, and `e_thr: 0.5` for the 0.5 px rows), given that the in-code defaults differ from the launched values. Two of the doc's rows predate parts of that echo. The 2.11 run's JSONs (`out/`) carry none of `koff` / `lever` / `e_thr` / `kf_*` (see lines 31-35). And the E-threshold table's 2.0 px row is not from Slurm 45443823 at all — `tri_ethr.sbatch` loops `ETHR` over 0.5 and 1.0 only; that row comes from `out_k4/*.lever0` (Slurm 45443579, `tri_k4.sbatch`), whose JSONs carry `gap`, `koff`, `lever`, `n`, `scene` and the triplet counts but no `e_thr` and no `kf_*`, i.e. it ran at the in-code 2.0 px default. Those files reproduce the doc's cells exactly (0.612 / 3.578 at gap 2, 0.553 / 4.186 at gap 4, 0.521 / 5.832 at gap 8, against the doc's 0.61 / 3.58, 0.55 / 4.19, 0.52 / 5.83; verified 2026-09-09). The output directory in the doc is `/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/opencv_vo_probe/tri_bench/` (`out_kf/` for the trigger sweep).

### Known limitations / honesty notes for this block

- **`KOFF` default vs documented triplets.** Line 34 defaults `KOFF` to `2 * GAP` while its own comment says `k = j + GAP` and the doc's 2.11 table describes triplets `(i, i+4, i+8)`. The launcher in the repo (`tri_kf.sbatch`) and the doc's E-threshold / trigger tables use an explicit `KOFF=4`. The 2.11 run's per-scene JSONs (Slurm 45443315, `tri_bench/out/`) were written by an earlier script revision that echoes no `koff` / `lever` / `e_thr` / `kf_*`, but its horizon is not in doubt: `triplets_total` is bounded by the line-85 loop alone and can only fall short of that bound, and on all 12 scenes it equals the `KOFF=4` bound while exceeding the `KOFF=8` one (2077 pooled, against bounds 2077 and 2053), so that run also used `k = j + 4`. Its lower gated count (1373 of 2077, against 1677 for the current code on the identical start frames) is therefore not a `k`-offset effect; it comes from the earlier revision's frontend (next bullet). Reproduce with `KOFF=4` set explicitly.
- **The 2.11 table is not reproducible from this file.** The current script, on the same scenes, gap, horizon, gate and threshold, gates 1677 of 2077 triplets where the 2.11 run gated 1373, and the doc's own E-threshold 2.0 px / gap-4 cell (0.55 / 4.19) disagrees with the 2.11 table's E `cheir+rep` cell (0.53 / 3.98) for what is nominally the same configuration. The scratch chronology (`out/` 09:28, `out_gap/` 09:30, then `out_chain/` 09:31 from a launcher named `tri_chain.sbatch`, Slurm 45443494, whose gap-4 output matches the current code cell for cell) and the way the two earlier runs' survival collapses with gap (gap 32: 11 gated of 1741 vs 523 chained) place the 2.11 run before the frame-by-frame chaining of lines 54-59. The earlier revision itself is not on disk, so the diff cannot be shown; the design doc records none of this. The 2.11 ordering is unaffected — the chained re-run keeps `cheir+rep` best under the GT pose — but the published absolute numbers belong to a frontend this file no longer implements.
- **Launcher coverage, and one mislabelled table.** `tri_kf.sbatch` is the only launcher in the repo and produces only the six adaptive-trigger rows of the keyframe-trigger sweep (Slurm 45443937); note that it runs `$G/tri_bench.py`, the copy in the scratch staging directory, not the repo copy — the two are byte-identical today (verified 2026-09-09), but the launcher does not pin the repo file. The 2.11 table (`tri_bench.sbatch`, Slurm 45443315) and the E-threshold table came from launchers that exist only in the scratch staging directory, and the E-threshold table is two jobs, not the single 45443823 the doc labels it with: its 2.0 px row is `tri_k4.sbatch` (Slurm 45443579) at the in-code default, its 1.0 / 0.5 px rows `tri_ethr.sbatch` (Slurm 45443823). The sweep's stride rows are the E-threshold table's 0.5 px row re-used. The script that pools per-scene JSONs into the doc's medians is not in the repo; the `fired/starts` and `fail%` columns were re-derived here instead (lines 81-84, 136-139).
- **No lever-ON numbers — though the runs exist.** The `LEVER` switch (line 35, lines 107-110) was exercised: `tri_k4.sbatch` (Slurm 45443579) ran gaps 2 / 4 / 8 / 16 with the lever both OFF and ON, and the lever-ON per-scene JSONs are on scratch as `out_k4/*.gap<G>.lever1.json` (gap 4: 1672 triplets gated of 2077, against 1677 with the lever OFF on the same start frames; verified 2026-09-09). They were simply never pooled into a table, so the design doc still reports no lever-ON `tri_bench` numbers, and the lever's ruling (default ON since 2026-09-05) rests on pipeline sweeps 1-3 and the E-diag, not on this probe.
- **In-code defaults are stale relative to the decisions.** `E_THR` defaults to 2.0 px (decided: 0.5 px, decision 2.10); `LEVER` defaults off (pipeline default ON since 2026-09-05, decision 1.2); the PnP failure floor is 20 (decided: 30, decision 3.1). The docstring lines 7 and 18 describe the pre-revision pipeline. The trigger sweep was produced with the env overrides shown in `tri_kf.sbatch` (`KOFF=4 LEVER=0 E_THR=0.5`); the 2.11 table is labelled lever OFF at the docstring's 2 px.
- **Line 38 carries three concatenated comments**; the second and third are the intended comments for `E_THR` (line 36) and `LEVER` (line 35).
- **No cheirality guard on the E arm.** `recoverPose`'s inlier count is discarded (line 118); the pipeline's bootstrap keeps the pose only when that count reaches `--min_inliers` (`opencv_vo.py` line 369), i.e. 30 in the frozen configuration — decision 2.9's prose still says 20 because it was written against the pre-revision 3.1 floor. Pure-rotation triplets that pass the 2 px gate are scored here but would be rejected in the pipeline.
- **Native frames and calibrated K.** The probe runs on native 320x180 frames with the frame-0 calibrated `K` (fx ~ 192 px), not the pipeline's 320x192 cover frames with the model's per-frame focal (decisions 0.1b, 1.1). The full-pipeline focal check showed the focal value is immaterial (111.9/6.93/0.871 vs 116.0/6.91/0.885) but did not test the framing; pixel thresholds here are in native units.
- **Outlier-mask semantics are contested in the doc.** The dataset-facts section says the mask's constant part includes "a gripper blob bottom-left"; decision 1.2 says it "does NOT cover the gripper". `zvalid` (line 114) excludes the mask either way, but must not be read as a gripper oracle.
- **Stride rows are gate-conditioned.** The 2 px motion gate (line 106) applies to every `KF_MODE`, so the stride rows of the trigger sweep exclude the 25 % of 2-frame windows with < 2 px motion that a real stride keyframe would have to handle; parallax / ratio rows fire on their own rule, so populations differ slightly (doc, keyframe-trigger sweep).
- **Correlated samples.** Start frames advance by 2 (line 85) and triplets share frames, so the pooled medians are over correlated samples, not independent trials; `kf_gaps` is appended before gating (line 100) and so includes gated-out triplets.
- **GT depth is not a gripper oracle.** GT depth is defined on the gripper in many scenes (doc, E-diag findings), so `bad_frac` on high-gripper scenes charges static-gripper points against scene depth; and a ~20 % depth-vs-pose scale mismatch in some scenes is absorbed by the single median scale.
- **Conditioned accuracy.** `pnp_rot` / `pnp_tdir` are medians over *successful* PnPs; the `fails` / `skipped` counters must be read alongside them (the parallax filters look competitive on `bad` only because they also fail 2x as often).
- **GT-scored design benchmark on test scenes.** Like the other probes, this benchmark uses GT pose / depth for scoring and was run on the 12 reported smoke scenes (audit items 7 and 8 in the design doc; item 7 addressed by reporting the frozen configuration on all 4292 scenes, others pending the user's ruling).
- **Non-causal by construction.** The `GT` arm and the depth scorer use privileged information; the probe is diagnostic only and nothing from it enters the inference path except the decided thresholds.
