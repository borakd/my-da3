## `eval_pipeline/opencv_vo.py` lines 46–120: frame loader, self-calibration sweep, camera matrix, and the `[R|t]` → 4×4 helper

This block is the input side of the classical arm: it turns each scene's PNGs into the grayscale frames the CUT3R eval loader sees, and it produces the pinhole camera matrix `K` (fx = fy, principal point at the image centre) that every later geometric step consumes. Four functions live here. `load_gray_cover` re-implements the geometry of `load_images_cover` (the eval loader in `src/CUT3R/src/dust3r/utils/image.py`) so that the OpenCV arm and the model arm operate on the same 320×192 "cover" frames (decision 1.1). `selfcal_focal` is the image-only focal estimator behind `--focal selfcal`, a measured option that the design doc records as "not recommended" (decision 0.1b, focal-sensitivity section). `model_K` builds `K` for the three whole-scene focal sources (`model` = retroactive per-scene median of the model's predicted focal, a fixed number such as `203`, or `gt` = calibrated focal, diagnostic only). `rt_to_T` packs an OpenCV `(R, t)` pair into a 4×4 homogeneous transform. In the pipeline, `run_scene` (outside this range) calls `load_gray_cover` once per frame for the whole sequence, then picks the focal path from `--focal`; the reported configuration since 2026-09-06 is `--focal perframe:<preds_root>` (frame *t* uses the focal CUT3R produced for frame *t*), which is resolved inside `run_scene` and does **not** pass through `model_K` — `model_K` serves the `model`, numeric and `gt` modes, and `selfcal_focal` the `selfcal` mode. All pixel-unit thresholds elsewhere in the file (0.5 px E-RANSAC, 2 px PnP/triangulation, 5 px parallax, 1 px forward–backward) are stated in the coordinate frame this block defines.

---

### Lines 46–47 — signature and contract

```text
46: def load_gray_cover(path, size=320, patch=16):
47:     """Exactly load_images_cover's geometry (scale to cover 320x192, centre crop), then grayscale."""
```

**What it does.** Declares a per-image loader with the same two geometry knobs as the eval loader: `size` is the target *width* (320) and `patch` is the ViT patch size (16) both dimensions must be multiples of. The docstring states the contract: reproduce `load_images_cover`'s resize-to-cover-then-centre-crop, then convert to a single grayscale channel. It returns a 2-D `uint8` array of shape `(H, W)` = `(192, 320)` for this dataset's 320×180 PNGs.

**Alternatives considered.** (a) Run OpenCV on the native 320×180 PNGs with the calibrated K from `cam/*.npz`; (b) import and call `load_images_cover` itself and undo its `ImgNorm` tensor normalisation; (c) the *other* eval loader `load_images`, which (per the `load_images_cover` docstring, image.py lines 139–140) "fits the LONG edge to `size` and then crops each side DOWN to a multiple of 16, so a 320x180 frame at size=320 becomes 320x176". Design doc decision 1.1 also lists 2× upsampling, CLAHE and undistortion as preprocessing options.

**Why this choice.** Decision 0.1b: "Run the classical arm on the model's own preprocessed 320x192 frames (same loader) so K and pixels agree" — the model's focal is expressed in the 320×192 cover frame, so the pixels must be too. Decision 1.1 fixes preprocessing to "grayscale conversion only, on the eval loader's 320x192 frames (load_images_cover: uniform 1.067x scale + center crop). All pixel thresholds are in these units." A local re-implementation (rather than importing the loader) avoids pulling the CUT3R package and its tensor normalisation into a CPU-only script; the arithmetic below is copied from `image.py` lines 176–189. CLAHE/upsampling are deferred as v1 single-variable experiments ("CLAHE first if median inliers are low").

---

### Lines 48–49 — open the image, read native size

```text
48:     img = PIL.Image.open(path).convert("RGB")
49:     W1, H1 = img.size
```

**What it does.** Opens the PNG through PIL and forces a 3-channel RGB image (drops alpha / palette modes). `PIL.Image.size` is `(width, height)`, so `W1, H1` = `(320, 180)` for the DROID wrist frames (design doc "Dataset facts": images 320×180 PNG).

**Alternatives considered.** `cv2.imread` (returns BGR, `(H, W, 3)`), which would be one line shorter but uses a different resampling implementation than PIL and would make the "exact replica" claim false; the eval loader additionally applies `exif_transpose` before `.convert("RGB")` (image.py line 175).

**Why this choice.** PIL is what the eval loader uses, and the resize interpolation kernels (line 54) are PIL's; using the same library is what makes the frames bit-comparable. The eval loader's `exif_transpose` is omitted; the code has no comment explaining why. It is a no-op if the PNGs carry no EXIF orientation tag (not recorded in the design doc; a one-scene spot check on AUTOLab+0d4edc83 found none), so the frames are expected to be identical in practice.

---

### Lines 50–51 — target box (tw, th)

```text
50:     tw = max(patch, (int(size) // patch) * patch)
51:     th = ((H1 * tw + W1 * patch - 1) // (W1 * patch)) * patch
```

**What it does.** `tw` is the requested width floored to a multiple of 16, but never below one patch: `320 // 16 * 16 = 320`. `th` is the native-aspect height at that width, rounded **up** to a multiple of 16 via integer ceiling division: `ceil(180·320 / (320·16))·16 = ceil(11.25)·16 = 12·16 = 192`. This is where 320×180 becomes a 320×192 box (the partial patch row is kept, not dropped).

**Alternatives considered.** Rounding `th` *down* (the `load_images` behaviour, 320×176, which discards content instead of stretching) or padding to 192 with a border instead of scaling.

**Why this choice.** Straight copy of `load_images_cover` (image.py lines 179–180). Its docstring documents the rounding in two places: the example code comment (image.py line 148) `# round the partial patch row UP`, and the prose (image.py lines 151–152) "A 320x180 frame at size=320 -> 320x192, matching the resolution the model trained at." Decision 1.1 requires these exact frame dimensions because the model's predicted focal (~211 px, decision 0.1b) is defined on them; the doc records the calibrated focal ratio 190 @320×180 → 211 @320×192 as 1.11, "tight", i.e. the loader's 1.067× scale accounts for most of the difference.

---

### Lines 52–53 — cover scale and resized dimensions

```text
52:     scale = max(tw / W1, th / H1)
53:     new_w = max(tw, int(round(W1 * scale))); new_h = max(th, int(round(H1 * scale)))
```

**What it does.** A single isotropic scale factor that makes the image *cover* the box: `max(320/320, 192/180) = 1.0667`. The resized size is `(341, 192)` (`round(320·1.0667) = 341`), clamped so neither side can fall below the box through rounding. Isotropic scaling preserves the pixel aspect ratio, so fx = fy remains true after the transform (relied on by every `K` built in this file).

**Alternatives considered.** Anisotropic resize straight to `(320, 192)` (would make fx ≠ fy by 6.7% and break the square-pixel assumption stated in `selfcal_focal`'s docstring); letterbox / padding.

**Why this choice.** Verbatim `load_images_cover` (image.py lines 182–184). The training path (`datasets.utils.cropping.rescale_image_depthmap`, cited in the loader docstring at image.py line 142) resizes to cover and centre-crops, and the eval loader reproduces that; decision 1.1 ties the classical arm to it. Note the design doc's "uniform 1.067x scale" is exactly this `scale` value.

---

### Lines 54–55 — interpolation kernel and resize

```text
54:     interp = PIL.Image.BICUBIC if scale >= 1 else PIL.Image.LANCZOS
55:     img = img.resize((new_w, new_h), interp)
```

**What it does.** Upscaling (`scale ≥ 1`, the case here) uses bicubic; downscaling would use Lanczos (an anti-aliasing kernel). `PIL.Image.resize` takes `(width, height)`, so the result is 341×192 RGB.

**Alternatives considered.** Bilinear or area interpolation (`cv2.INTER_AREA` is the usual OpenCV pick for downscaling); nearest-neighbour. The choice matters for corner detection and sub-pixel LK because it sets the frames' high-frequency content.

**Why this choice.** Copied from `load_images_cover` (image.py lines 185–186); the design doc does not benchmark the kernel, it only requires the frames to be the eval loader's. Any other kernel would make the OpenCV frames differ from the model's input by a resampling residual.

---

### Lines 56–58 — centre crop and grayscale conversion

```text
56:     left = (new_w - tw) // 2; top = (new_h - th) // 2
57:     img = img.crop((left, top, left + tw, top + th))
58:     return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
```

**What it does.** Crops the central `tw × th` window: `left = (341−320)//2 = 10`, `top = (192−192)//2 = 0`, so 10 columns are dropped on the left and 11 on the right (floor division puts the odd pixel on the right). `PIL.Image.crop` takes `(left, upper, right, lower)` in pixel coordinates. `np.asarray(img)` gives an `(H, W, 3)` `uint8` RGB array; `cv2.cvtColor(..., COLOR_RGB2GRAY)` applies OpenCV's luma weighting (the standard 0.299 R + 0.587 G + 0.114 B per the OpenCV documentation; a library constant, not a parameter of this file) and returns `(192, 320)` `uint8`. Note the flag is `RGB2GRAY`, not `BGR2GRAY`, because the array came from PIL (RGB order); using the BGR flag would silently swap the R/B weights.

**Alternatives considered.** Keeping colour (Shi-Tomasi and LK require a single channel, so this is not an option in OpenCV without converting anyway); PIL's `.convert("L")` (ITU-R 601 weights too, but a different rounding path); the "2x upsample / CLAHE / undistort" options in decision 1.1.

**Why this choice.** Crop arithmetic is verbatim `load_images_cover` (image.py lines 187–189). Grayscale-only is decision 1.1 ("Gray is mandatory for the detector/LK; everything else is a v1 single-variable experiment"). Undistortion is unnecessary: the dataset facts record "no distortion" for the GT intrinsics. The 10-px left offset means the crop shifts the true principal point by 10 px relative to the native frame — consistent with `model_K`'s pp = image centre only because the model's focal was itself fitted with pp = centre of this same cropped frame (see lines 106–110).

---

### Lines 59–65 — (blank lines) and the `selfcal_focal` signature / contract

```text
59: 
60: 
61: def selfcal_focal(G, W, H, n_pairs=40, gap=6):
62:     """Self-calibration from the images alone: sweep candidate focals and keep the one under which the two-view
63:     geometry across many parallax-gated frame pairs is most consistent (most inliers at 1 px in findEssentialMat,
64:     tie-break by mean Sampson error of the inliers). Assumes pp at the image centre and square pixels.
65:     Returns (focal_px, n_pairs_used)."""
```

**What it does.** Lines 59–60 are the PEP-8 two-blank-line separator. `selfcal_focal` takes the whole scene's grayscale frame list `G` (from `load_gray_cover`), the frame width/height `W, H` (320, 192), the number of candidate frame pairs to sample (`n_pairs=40`) and the frame gap inside each pair (`gap=6`). It returns `(focal_px, n_pairs_used)`, or `(None, n)` when too few usable pairs exist. The estimator is a 1-D brute-force search: for each candidate focal it builds `K`, runs `findEssentialMat` on every retained pair, and sums RANSAC inlier counts; the focal with the most inliers wins. **Docstring caveat:** the "tie-break by mean Sampson error" it mentions is *not implemented* in the code below (the score is inlier count only, line 86); `np.argmax` breaks ties by taking the lowest candidate focal.

**Alternatives considered.** Decision 0.1b's option list for the intrinsics source: calibrated K from `cam/*.npz`; one nominal dataset K; model-estimated focal from the preds; self-calibration. Within self-calibration the literature offers Kruppa-equation / absolute-quadric methods, focal-from-fundamental-matrix closed forms (Bougnoux / Sturm), and the ORB-SLAM-style "assume calibrated" stance; this implementation is the simplest possible consistency sweep.

**Why this choice.** The function exists because the user's closed-loop rule (decision 0.1) demanded the question "can the images alone supply K?" be answered empirically (standing rule: benchmark every option on the 12 smoke scenes). The answer is recorded in the focal-sensitivity section: self-calibration scores 116.6 mm ATE / 7.86 mm RPE-t / 0.935° RPE-r versus 109.1 / 7.04 / 0.869 for the fixed 203 px focal, with per-scene estimates "132, 280, 196, 162, 358, 448, 174, 180, 210, 278, 298, 510 (true ~203)", i.e. "0.65x–2.5x scatter". Conclusion (2): "Classical self-calibration from this footage is unreliable … `--focal selfcal` exists but is not recommended." Defaults `n_pairs=40` / `gap=6` are code values; the design doc describes the run as a sweep "over 30 parallax-gated pairs" — the code's current default is `n_pairs=40` candidate pairs with ≥ 30 tracks per pair; whether the doc's 30 refers to an earlier `n_pairs` or to pairs surviving the gates cannot be determined from the doc or the code.

---

### Lines 66–68 — sample candidate frame pairs

```text
66:     n = len(G); pairs = []
67:     for i in np.linspace(0, max(0, n - gap - 1), n_pairs).astype(int):
68:         j = i + gap
```

**What it does.** Picks `n_pairs` start indices evenly spread over `[0, n−gap−1]` (truncated to int) and pairs each with the frame `gap = 6` steps later. Truncation could produce duplicate start indices only when `n − 7 < 39`, i.e. below 46 frames, which never happens on this dataset (minimum sequence length 57, dataset facts); with 57–~1700-frame sequences `j` always stays in range. The pairs span the whole scene, so this estimator looks at future frames before any pose is produced — it is a batch, non-causal step by construction.

**Alternatives considered.** All consecutive pairs (dominated by 9 mm / 1.3° steps that make E degenerate, per the probe evidence); pairs chosen by a parallax trigger like the keyframe logic (decision 2.12); a single well-conditioned pair.

**Why this choice.** No benchmark in the design doc separates these; the code picks fixed, evenly spaced pairs plus the parallax gate on line 76 so the sweep sees several independent viewpoints. Gap 6 is a code value, not benchmarked; it sits between the gap-4 and gap-8 rows of the two-view diagnostics (E at 0.5 px: 1.23° / 2.88°) and beyond the doc's recommended keyframe spacing of 2–4 ("Keyframes must be close (gap 2-4), not far").

---

### Lines 69–70 — Shi-Tomasi corners on frame i

```text
69:         p0 = cv2.goodFeaturesToTrack(G[i], MAX_CORNERS, QUALITY, MIN_DIST)
70:         if p0 is None or len(p0) < 30: continue
```

**What it does.** `cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)` with the frozen constants `1000, 0.01, 5` returns an `(N, 1, 2)` `float32` array of `(x, y)` pixel coordinates (OpenCV's Shi-Tomasi minimum-eigenvalue detector; `qualityLevel` is relative to the strongest corner, `minDistance` in px). It returns `None` (not an empty array) when nothing passes, hence the explicit `None` check. Pairs with fewer than 30 corners are skipped.

**Alternatives considered.** Decision 1.3 lists FAST, ORB, SIFT/AKAZE and a fixed grid.

**Why this choice.** Decision 1.3: "Shi-Tomasi goodFeaturesToTrack(maxCorners=1000, qualityLevel=0.01, minDistance=5) (probe values; cap never reached, ~140–300 corners/frame)". The same constants are reused here so the self-calibration sees the same corner population as the main loop. The 30 floor coincides numerically with decision 3.1's inlier floor (30) — a code choice, not separately benchmarked for self-calibration.

---

### Lines 71–72 — forward–backward LK

```text
71:         p1, st, _ = cv2.calcOpticalFlowPyrLK(G[i], G[j], p0, None); p0b, stb, _ = cv2.calcOpticalFlowPyrLK(G[j], G[i], p1, None)
72:         ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm((p0 - p0b)[:, 0], axis=1) < FB_THRESH)
```

**What it does.** `cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts=None)` at OpenCV defaults (`winSize=(21,21)`, `maxLevel=3`; the iteration/epsilon stop criterion is also OpenCV's default `TermCriteria`, per the OpenCV documentation, not a parameter of this file) returns `(nextPts (N,1,2) float32, status (N,1) uint8, err (N,1))`; `status == 1` means the flow was found. The second call tracks the found points back from `j` to `i`. A track is kept only if both directions succeeded and the round-trip landing point lies within `FB_THRESH = 1.0` px of the original corner.

**Alternatives considered.** Decision 1.5: LK without the check, ORB/SIFT + ratio test, DIS or Farneback dense flow at corners or on a grid.

**Why this choice.** Decision 1.5: "sparse LK at OpenCV defaults (21 px window, 3 levels) + forward-backward check at 1 px." From the correspondence benchmark: LK+FB 166 / 88 correspondences / correct at gap 1 (0.88° E rotation error), and "the best per-correspondence precision at gap 4 (in2 0.35 vs 0.29)"; ORB/SIFT/Farneback rejected; DIS-grid the measured runner-up. Window/levels are OpenCV defaults, as the doc states.

---

### Lines 73–76 — inlier coordinates and parallax gate

```text
73:         a, b = p0[ok, 0].astype(np.float64), p1[ok, 0].astype(np.float64)
74:         if len(a) < 30: continue
75:         disp = np.linalg.norm(b - a, axis=1)
76:         if np.median(disp) < PARALLAX_PX: continue
```

**What it does.** Squeezes the surviving tracks to plain `(M, 2)` pixel arrays cast to `float64` (a code choice; OpenCV accepts `float32` too, but the bootstrap in `run_scene` also works in `float64`, so the two paths match), drops pairs with fewer than 30 survivors, and computes each track's displacement in px. The pair is discarded if the *median* displacement is below `PARALLAX_PX = 5.0`, i.e. the camera barely moved between `i` and `j`.

**Alternatives considered.** No gate (probe: frames 0–15 of the RAIL scene are stationary — "GT rot 0.03 deg, |t|=0.2 mm" — and the essential matrix is degenerate there); a tracked-ratio trigger; H-vs-E model selection (decisions 2.9, 2.12).

**Why this choice.** The gate reuses the bootstrap/keyframe parallax constant from decisions 2.9 and 2.12 (5 px median track displacement, revised from 10 on 2026-09-05; decision 2.12's ruling: "Parallax 5 gives median gap 2 (2.96 deg) and never fires on a paused camera"). Under pure rotation or zero motion the inlier count is uninformative about focal, so gating is what makes the score on line 86 meaningful.

---

### Lines 77–79 — static-track lever and pair retention

```text
77:         keep = disp >= 1.0                                     # static-track lever, same rule as the main loop
78:         if keep.sum() < 30: continue
79:         pairs.append((a[keep], b[keep]))
```

**What it does.** On a pair that has already passed the parallax gate (median ≥ 5 px, so a non-moving track cannot be scene geometry), tracks moving less than 1 px are dropped — these are the image-static gripper corners. If fewer than 30 tracks remain the pair is discarded; otherwise the `(a, b)` correspondence arrays are stored for the sweep.

**Alternatives considered.** Decision 1.2: no mask; a fixed dataset polygon; per-scene temporal-variance (Otsu) mask; flow-based rejection; the GT `outlier_mask`; a *relative* lever (drop < 20% of median displacement).

**Why this choice.** Decision 1.2's lever `reject_static_tracks` ("drop tracks with displacement < 1 px before any geometry", only on frames with median ≥ 2 px). The correspondence benchmark: removing zero-motion tracks cuts parallax-gated gap-4 E-rotation error 2.14 → 1.70° for LK; the static share of LK tracks is 16–53% per scene, median ~25%. Two differences from the main loop, both code-derived: here the lever is unconditional (there is no separate 2 px activation gate because the 5 px parallax gate on line 76 already implies it) and cannot be switched off by `--no_lever`. The lever's measured gain is in the full pipeline (sweeps 1–3: 111.9 / 6.93 / 0.871 ON vs 115.6 / 8.40 / 0.971 OFF), whereas the two-view diagnostics found it "does not change E", so here it is a consistency choice with no measured benefit for self-calibration. Note the 1 px rule is applied to the gap-6 displacement, not the per-frame step the main loop uses (opencv_vo.py lines 346–350), so the comment's "same rule as the main loop" holds for the threshold, not for the quantity it is applied to.

---

### Lines 80–81 — minimum pair count

```text
80:     if len(pairs) < 5:
81:         return None, len(pairs)
```

**What it does.** Fewer than 5 usable pairs means the sweep cannot vote reliably; the function returns `None` for the focal (the caller in `run_scene`, line 264, then falls back to `model_K(..., "model", ...)`, the per-scene median of the model focal) together with the pair count, which `run_scene` writes to `summary.json` as `selfcal_pairs` (line 488).

**Alternatives considered.** Falling back to the fixed 203 px nominal; failing the scene outright.

**Why this choice.** Code choice; not benchmarked. The threshold 5 is a hand-picked floor. Note the fallback to the *model* focal makes `--focal selfcal` not strictly model-free on scenes where self-calibration aborts (honesty note below).

---

### Lines 82–87 — the score function

```text
82:     def score(f):
83:         K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]]); tot = 0
84:         for a, b in pairs:
85:             E, inl = cv2.findEssentialMat(a, b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
86:             if E is not None and E.shape == (3, 3) and inl is not None: tot += int(inl.sum())
87:         return tot
```

**What it does.** For a candidate focal `f` it forms `K = [[f, 0, W/2], [0, f, H/2], [0, 0, 1]]` — square pixels, zero skew, principal point at `(160, 96)` in the 320×192 frame — and calls `cv2.findEssentialMat(points1, points2, cameraMatrix, method, prob, threshold)`. Because a `cameraMatrix` is passed, `a, b` are in **pixel** coordinates and `threshold=1.0` is a Sampson distance in **pixels** (OpenCV normalises internally); this is different from the bootstrap in `run_scene` (lines 361–365), which pre-normalises each view with its own per-frame K and passes `I3` with the threshold divided by the mean focal. `prob=0.999` is the RANSAC confidence. The call returns `(E, mask)`: `E` is `(3, 3)`, but Nistér's 5-point solver can return several stacked solutions as a `(3k, 3)` array, or `None` on failure — hence the explicit shape check, which simply drops such pairs from the tally. `mask` is `(M, 1)` `uint8` with 1 for inliers; the score is the total inlier count over all pairs. Larger totals mean more tracks are consistent with epipolar geometry under this `K`.

**Alternatives considered.** Scoring by mean Sampson error of the inliers (what the docstring promised as tie-break; not implemented), by the smallest singular-value ratio of `E` (the "essential-ness" test), or by a closed-form focal from the fundamental matrix; MAGSAC++ instead of RANSAC (E-diag: MAGSAC 2 px 1.34° vs RANSAC 2 px 2.06° at gap 4, but not tried for this sweep); a 0.5 px threshold like decision 2.10.

**Why this choice.** Inlier count at 1 px is the simplest monotone consistency measure and shares the RANSAC confidence of decisions 2.8/2.10 (0.999). The 1 px threshold is a code value between the pipeline's 0.5 px bootstrap threshold and the 2 px PnP threshold (E-diag: 1 px gives 1.64° / 24.3° at gap 4). None of this was tuned further because the outcome (0.65×–2.5× focal scatter) showed the objective itself is too flat on this footage: "the finetuned model's stable 211–217 px is learned knowledge of the DROID camera that images alone at this resolution/baseline do not pin down."

---

### Lines 88–91 — coarse-to-fine 1-D search and return

```text
88:     coarse = np.arange(100, 501, 10.0); sc = [score(f) for f in coarse]
89:     f0 = coarse[int(np.argmax(sc))]
90:     fine = np.arange(f0 - 10, f0 + 10.1, 2.0); sf = [score(f) for f in fine]
91:     return float(fine[int(np.argmax(sf))]), len(pairs)
```

**What it does.** Evaluates `score` at 41 focals from 100 to 500 px in 10 px steps, takes the best `f0`, then re-evaluates 11 focals from `f0−10` to `f0+10` in 2 px steps (the `10.1` upper bound makes `np.arange` include `f0+10`). Total cost: 52 `score` calls × up to 40 RANSAC essential-matrix fits. Returns the best fine focal as a Python float plus the number of pairs used. `np.argmax` returns the *first* maximum, so ties resolve toward the smaller focal. The fine pass can step outside the coarse grid by up to 10 px (90–510 px), which is how the reported per-scene value of 510 arose.

**Alternatives considered.** Golden-section / Brent search on the score (unsafe: the score is integer-valued and non-smooth); a wider or log-spaced grid; joint search over `(f, cx, cy)`.

**Why this choice.** The grid brackets the calibrated and finetuned-model values with a wide margin: the dataset facts give calibrated fx ~192 px @320×180 (≈203 in the cover frame), the model focal is ~211 px, and the focal-sensitivity table tested 150 / 305 / 406 px, all inside 100–500. It does not reach the upper end of the zero-shot checkpoint's implicit focals (229–570 px per scene, 207–819 px per frame), which the design doc says are "not a usable camera model" anyway. The 2 px fine step is comparable to (slightly above) the model's within-scene jitter (0.7% of ~212 px ≈ 1.5 px, decision 0.1b); finer resolution would be meaningless given the 132–510 px per-scene scatter of the method. Search-strategy details are code values, not benchmarked.

---

### Lines 92–97 — (blank lines) and `model_K` signature / contract

```text
92: 
93: 
94: def model_K(preds_scene_dir, W, H, focal_mode="model", scene_dir=None):
95:     """Camera matrix in the 320x192 cover frame. focal_mode: 'model' = per-scene median of the model's
96:     predicted focal (decision 0.1b); a float = fixed nominal focal (strict zero-shot); 'gt' = calibrated focal
97:     from the scene's cam npz rescaled by the cover factor (DIAGNOSTIC ONLY, reads GT intrinsics)."""
```

**What it does.** After the separator (92–93), `model_K` returns one 3×3 `K` for the whole scene in the 320×192 cover frame. Inputs: the scene's preds directory (`<preds_root>/<scene>`, containing `camera/*.npz` written by the CUT3R eval worker), the cover frame size, the focal mode string from `--focal`, and the raw scene directory (needed only for `gt`). The docstring enumerates three of the four handled strings; `selfcal` is also recognised but only to raise (lines 104–105). The per-frame `perframe:` / `causal:` modes never reach this function (they are parsed in `run_scene`, lines 265–277).

**Alternatives considered.** Decision 0.1b's four options: calibrated K from `cam/*.npz`; one nominal dataset K; model-estimated focal from the preds; self-calibration. A fifth, per-frame inference-time focal, was added by the 2026-09-06 revision.

**Why this choice.** Decision 0.1b as first decided (2026-09-02): "model-estimated focal (augfull_lr1e5 preds camera/*.npz), per-scene MEDIAN of per-frame values, pp = image center", with the calibrated-K copy "kept as a one-off diagnostic". The docstring's "(decision 0.1b)" refers to that original ruling; the decision was **revised on 2026-09-06** to the inference-time per-frame focal, so `model` is now explicitly "retroactive and … no longer the reported configuration". The design doc's re-check (2026-09-05) found the focal source immaterial in the full pipeline — model 111.9 / 6.93 / 0.871, fixed 203 px 109.1 / 7.04 / 0.869, calibrated 116.0 / 6.91 / 0.885 (ATE mm / RPE-t mm / RPE-r deg on the 12 smoke scenes) — which is why all three modes remain available.

---

### Lines 98–103 — `gt` mode: calibrated focal rescaled into the cover frame

```text
98:     if focal_mode == "gt":
99:         z = np.load(os.path.join(scene_dir, "dense", "cam", "000000.npz"))
100:         Kn = z["intrinsic"]
101:         from PIL import Image
102:         W1, H1 = Image.open(sorted(glob.glob(os.path.join(scene_dir, "dense", "rgb", "*.png")))[0]).size
103:         f = float(Kn[0, 0]) * max(W / W1, H / H1)
```

**What it does.** Loads the GT intrinsics of frame 0 only (`cam/000000.npz`, key `intrinsic`, singular — note the preds files use the plural `intrinsics`), takes `fx = K[0,0]` in native 320×180 pixels, reads the native size from the first PNG (a redundant local `from PIL import Image`; the module already imports `PIL.Image`), and multiplies by the cover scale `max(320/320, 192/180) = 1.0667` — the same `scale` as `load_gray_cover` line 52. With fx ~190–192 px native (dataset facts) this gives ≈203–205 px in the cover frame, which is the origin of the "nominal 203 px" number. The GT principal point `Kn[0,2], Kn[1,2]` is **ignored**; line 113 will substitute the image centre.

**Alternatives considered.** Reading every frame's `cam/*.npz` and taking the median (as the `model` branch does); carrying the GT principal point through the crop (`cx' = cx·scale − 10`, `cy' = cy·scale − 0`); using the full GT K including any skew.

**Why this choice.** Decision 0.1b: "Calibrated-K copy kept as a one-off diagnostic" — the docstring shouts DIAGNOSTIC ONLY because reading `cam/*.npz` violates the closed-loop rule (decision 0.1: "plain image inputs only, no privileged info"). Frame 0 is assumed representative; the design doc only says "GT intrinsics per frame in cam/*.npz" and does not state that per-frame K is constant (a one-scene spot check shows it is), so this is an assumption of the code, not a documented dataset fact. The measured value of the privilege is small: on the 12 scenes calibrated focal scores 116.0 / 6.91 / 0.885 vs 111.9 / 6.93 / 0.871 for the model focal; on the full 4292 harness "GT intrinsics are worth ~2 mm ATE" (0.103 vs 0.104 m, RPE-r 1.102 vs 1.114°). Its existence is honesty-audit item 6 (pending the user's ruling).

---

### Lines 104–105 — `selfcal` is not resolvable here

```text
104:     elif focal_mode == "selfcal":
105:         raise ValueError("selfcal is resolved in run_scene (needs the frames)")
```

**What it does.** Guards against calling `model_K` with the string `selfcal`: the self-calibration needs the grayscale frame list, which this function does not receive, so `run_scene` calls `selfcal_focal` directly (line 263) and only falls back to `model_K(..., "model", ...)` when it returns `None` (line 264).

**Alternatives considered.** Passing `G` into `model_K`; making `selfcal_focal` load the frames itself.

**Why this choice.** Code structure only; `run_scene` already holds the frames for tracking, so loading them twice would double the I/O for an option that is "not recommended" (focal-sensitivity section).

---

### Lines 106–110 — `model` mode: per-scene median of the model's predicted focal

```text
106:     elif focal_mode == "model":
107:         fs = sorted(glob.glob(os.path.join(preds_scene_dir, "camera", "*.npz")))
108:         if not fs:
109:             raise FileNotFoundError(f"no model camera npz under {preds_scene_dir}")
110:         f = float(np.median([np.load(x)["intrinsics"][0, 0] for x in fs]))
```

**What it does.** Globs every `camera/%06d.npz` the CUT3R eval worker wrote for the scene, reads `intrinsics[0, 0]` (fx in the 320×192 frame the model was run on) from each, and takes the median across all frames of the scene. An empty glob is a hard error (the preds must exist; the classical arm cannot run without them in this mode). Per the design doc, this focal is not a model head output: it is derived after inference by `estimate_focal_knowing_depth(pts3d_self, pp=centre, "weiszfeld")`, a robust fit of `f = (u−cx)·z/x` over the frame's predicted self-view point map, and the OpenCV arm is its first consumer.

**Alternatives considered.** Mean instead of median; the focal of frame 0 only; the per-frame value (`perframe:`) or the running median up to *t* (`causal:`), both implemented in `run_scene`; a nominal constant.

**Why this choice.** Original decision 0.1b: "per-scene MEDIAN of per-frame values … within-scene jitter 0.7%" — the median is robust to the occasional bad frame. Measured: finetuned per-scene medians span 212–217 px across the 12 scenes (inference-time focal table). This mode is **non-causal** (frame *t* sees focals from frames after *t*) and was superseded on 2026-09-06 by `perframe:` at the user's inference-time constraint; the causal per-frame focal "reproduces the retroactive numbers (110.4 / 7.53 / 0.858 vs 111.9 / 6.93 / 0.871)", so nothing is lost by the switch. The zero-shot checkpoint's focals (229–570 px per scene, 207–819 px per frame) are "NOT usable" through this path in the sense that they encode inconsistent point-map geometry, though the full-4292 zero-shot + OpenCV row does use them per-frame (RPE-r 1.71 → 1.30°).

---

### Lines 111–114 — numeric mode and the shared `K` assembly

```text
111:     else:
112:         f = float(focal_mode)
113:     K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
114:     return K
```

**What it does.** Any other `--focal` string is parsed as a focal length in pixels at 320×192 (e.g. `--focal 203`; a non-numeric string raises `ValueError` from `float`). All three surviving modes then share one assembly: `fx = fy = f`, zero skew, principal point `(W/2, H/2) = (160.0, 96.0)`, returned as a `float64` 3×3 — the format `cv2.findEssentialMat`, `cv2.solvePnPRansac`, `cv2.triangulatePoints` and the output `camera/*.npz` `intrinsics` field expect. The same assembly is duplicated as the local `K_of` in `run_scene` (line 260) for the `selfcal`, `perframe:` and `causal:` paths. The pp is the centre of the *cropped* frame, which is also the convention under which the model's focal was fitted (`pp=centre`), so `f` and `pp` are self-consistent for the `model` path; for `gt` and numeric modes the true pp is approximated by the centre.

**Alternatives considered.** Carrying the GT principal point; fitting `(cx, cy)` in self-calibration; distinct fx / fy.

**Why this choice.** Decision 0.1b: "pp = image center" and the dataset facts ("no distortion"). The numeric mode is the "strict zero-shot" / "strictly model-free" arm: `--focal 203` scored 109.1 / 7.04 / 0.869 vs 111.9 / 6.93 / 0.871 for the model focal, "within noise of the v0 final", and "gives a strictly model-free arm at no cost". The focal-sensitivity sweep shows why the exact number barely matters: 150 px → 109.8 / 7.23 / 0.991, 305 px → 117.9 / 7.54 / 0.905, 406 px → 119.2 / 7.45 / 0.975 — "a 2x focal error costs ~10 mm ATE and ~0.1 deg RPE-r, because map and PnP share the same (wrong) K and Sim(3) absorbs the scale." Honesty-audit item 4 records that 203 "came from GT calibration" (it is the rescaled calibrated value, lines 98–103), so "model-free" is not "calibration-free".

---

### Lines 115–120 — (blank lines) and `rt_to_T`

```text
115: 
116: 
117: def rt_to_T(R, t):
118:     T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(t).ravel(); return T
119: 
120: 
```

**What it does.** Lines 115–116 and 119–120 are blank separators (the latter precede the `Tracks` class). `rt_to_T` builds the homogeneous 4×4 `T = [[R, t], [0, 1]]` in `float64` from a 3×3 rotation and a translation given as `(3,)`, `(3,1)` or `(1,3)` (`ravel` accepts all of OpenCV's shapes; `cv2.solvePnP*` and `cv2.recoverPose` return `(3,1)` columns). It preserves whatever direction convention `(R, t)` carries: in this file both producers are **world-to-camera / source-to-destination** — `cv2.recoverPose` returns, per the OpenCV convention, the pose of the second camera relative to the first (`x2 = R·x1 + t`, `|t| = 1`, which is what sets the map's arbitrary scale, decision 2.10; the code comment at line 370 reads "anchor cam -> this cam (unit translation)"), and `cv2.solvePnP*` returns `(rvec, tvec)` mapping world points into the camera (`x_cam = R·X_world + t`). The design doc's day-one check states the pitfall: "solvePnP returns world-to-camera (rvec, tvec): invert before writing c2w" — that inversion happens at the call sites, not here.

**Alternatives considered.** `scipy.spatial.transform.Rotation` objects or `(rvec, tvec)` tuples carried around instead of 4×4 matrices; `cv2.Rodrigues` inside the helper (the caller does it: `R, _ = cv2.Rodrigues(rvec)` before calling `rt_to_T`).

**Why this choice.** Plain code hygiene: chaining transforms (`T_rel @ boot_anchor_T` at line 371, scale hand-off `rt_to_T(R, sc·t)` for decision 2.15) is matrix multiplication with 4×4s, and the output format is `camera/%06d.npz` with `pose` as a c2w 4×4 (plumbing section). The doc's pose-convention check ("PnP with GT depth between two frames must reproduce the GT relative pose composed from cam/*.npz") and the E-diagnostics' "E from GT flow, no noise (convention check)" row at 0.01° / 0.0° are the evidence that the conventions this helper preserves are right.

---

### Known limitations / honesty notes for this block

- **Whole sequence in memory, batch loading** (audit item 3). `load_gray_cover` is called for every frame before tracking starts; nothing in this block streams.
- **`model` mode is retroactive / non-causal** (decision 0.1b revision, audit item 13). It uses focals from every frame of the scene; the reported configuration is `perframe:` (resolved in `run_scene`), and stale per-scene-median rows remain on disk (audit item 13). The `model_K` docstring still cites decision 0.1b as if the median were current.
- **`selfcal_focal` is non-causal and not recommended.** It samples pairs across the entire scene before any pose is produced, its docstring promises a Sampson tie-break the code does not implement, the lever inside it cannot be disabled by `--no_lever`, and on failure (< 5 pairs) `--focal selfcal` falls back to the model focal; the only trace is the `selfcal_pairs` field (< 5) written to `summary.json`. Measured: 116.6 / 7.86 / 0.935 with per-scene focals scattered 132–510 px against a true ~203 (focal-sensitivity section).
- **"Model-free" 203 px is calibration-derived** (audit item 4): it is the GT fx ~190–192 px rescaled by the loader's 1.067× cover factor; the pipeline's focal tolerance (2× error ≈ 10 mm ATE, 0.1° RPE-r) is what makes the exact value immaterial, not independence from calibration.
- **`--focal gt` reads privileged GT intrinsics** (audit item 6); it is a diagnostic only, worth ~2 mm ATE on the full 4292 harness (0.103 vs 0.104 m). Ruling pending. It also assumes frame 0's K holds for the whole scene, which the design doc does not state.
- **Principal point is always the crop centre.** The 10-px left crop offset and the true GT pp are ignored in every mode; only the `model` / `perframe` paths are internally consistent with this, because the model's focal was fitted under the same `pp=centre` assumption.
- **Doc/code mismatch on the self-calibration pair count.** The design doc says "30 parallax-gated pairs"; the code's `n_pairs` default is 40 candidate pairs with ≥30 tracks required per pair. Which the doc's 30 refers to cannot be settled from the doc or the code.
- **Minor loader difference.** `load_gray_cover` omits the eval loader's `exif_transpose` (image.py line 175) without a comment; a no-op if the PNGs carry no EXIF orientation tag (not recorded in the design doc; one-scene spot check found none), but not a literal "exact" replica.
