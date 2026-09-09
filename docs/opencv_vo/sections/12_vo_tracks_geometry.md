## `eval_pipeline/opencv_vo.py` lines 121–201: track store, corner top-up, LK step, two-view triangulation, PnP

This block is the geometric core of the OpenCV monocular VO control arm (design of record: `OPENCV_VO_DESIGN.md`). It holds the five primitives that the per-scene loop (`run_scene`, further down the file) composes into a local-map odometry: a structure-of-arrays container for live 2D tracks (`Tracks`), a masked Shi-Tomasi top-up detector (`detect`), a frame-to-frame pyramidal Lucas–Kanade step with forward–backward verification (`lk_step`), a two-view DLT triangulator with cheirality + reprojection acceptance (`triangulate_pair`), and a RANSAC PnP registration with an inlier floor (`pnp`). Everything operates on the eval loader's 320×192 grayscale "cover" frames (decision 1.1), in pixel coordinates of that frame, with a per-frame camera matrix `K` (fx = fy = the model's predicted focal for that frame, principal point at the image centre; decision 0.1b, revised 2026-09-06). Poses passed into and out of these functions are **world-to-camera (w2c)** 4×4 matrices, which is OpenCV's native `solvePnP` convention; the caller inverts to c2w once per frame when it stores `poses_c2w[f] = np.linalg.inv(T_w2c)` (lines 402, 415, 421, 434, 455), and that c2w matrix is what is written to `camera/%06d.npz` (line 479). Nothing in this block reads GT depth, GT pose, or masks: per decision 0.1 (closed loop, plain image inputs only) the map is a triangulated monocular map at an arbitrary scale that the Sim(3) scorer later absorbs (decision 3.3).

### Lines 121–123: the `Tracks` container

```python
121: class Tracks:
122:     """Live 2D tracks (structure of arrays)."""
123: 
```

**What it does.** Declares the container for all currently-alive 2D feature tracks. "Structure of arrays" means that instead of a list of track objects, each attribute (position, keyframe position, map-point id, track id) is one NumPy array indexed by track row, so that survival masks, keyframe candidates and PnP input can be built with vectorised boolean indexing. Line 123 is a blank separator.

**Alternatives considered.** Array-of-structs (a Python list of per-track objects, as in many textbook VO implementations), OpenCV `KeyPoint` lists with a parallel `dict` keyed by id, or a full "map point with observations" graph as in ORB-SLAM. The design doc records no numbered decision on the data layout; it is an implementation choice.

**Why this choice.** Decision 1.6 fixes the semantic model (persistent tracks, topped up only at keyframes), and every downstream step (lever at the call site, PnP input, triangulation candidates, culling) is a row mask over the same N tracks, so parallel arrays are the simplest layout that keeps those operations O(N) NumPy calls. The per-scene loop runs the whole pipeline at ~30 ms/frame on one CPU core (full-4292 section of the design doc), which the SoA layout helps keep.

### Lines 124–129: per-track arrays

```python
124:     def __init__(self):
125:         self.pos = np.zeros((0, 2), np.float32)     # current position
126:         self.kf_pos = np.zeros((0, 2), np.float32)  # position at the last keyframe (or at detection)
127:         self.mp = np.zeros(0, np.int64)             # map point id or -1
128:         self.ids = np.zeros(0, np.int64)
129:         self._next = 0
```

**What it does.** Four parallel arrays of length N (initially 0) plus an id counter:

- `pos` (N,2) `float32`: the track's current pixel position `(x, y)` in the 320×192 cover frame, x along columns, y along rows, origin at the centre of the top-left pixel (OpenCV convention: integer coordinates are pixel centres, shared by `goodFeaturesToTrack`, `calcOpticalFlowPyrLK` and `projectPoints`). `float32` is what LK returns and consumes.
- `kf_pos` (N,2) `float32`: the position the track had at the last keyframe, or at detection if it was born after the last keyframe. This is the "first view" endpoint used for parallax measurement (keyframe trigger, decision 2.12; bootstrap trigger, decision 2.9) and for KF-to-KF triangulation (decision 2.13).
- `mp` (N,) `int64`: id of the world map point this track observes, or `-1` if it has not been triangulated yet. Only rows with `mp >= 0` feed PnP.
- `ids` (N,) `int64`: globally unique, monotone track ids (never reused). They let the caller relate a track across frames independently of its row index, which is needed for the retro-fill of pre-bootstrap frames (`pre_hist` / `id2mp`, lines 395–397) and the scale hand-off across re-bootstraps (`old_mp` keyed by id, lines 303 and 379–380; decision 2.15: matching the new segment's scale to old points "through >= 10 shared tracks").
- `_next`: the next id to hand out.

**Alternatives considered.** Storing the full position history per track (what a per-track sliding-window BA would need) or per-track descriptors for re-association (would permit re-observing culled points, decision 2.14). Neither is stored.

**Why this choice.** `Tracks` stores only two endpoints per track; the optional local-BA path (`--refine ba`, `local_ba` at line 203, called from `declare_keyframe` at line 332; rejected for v0 by decision 3.4: window 5 / 10 = ATE 128.8 / 130.7 vs 122.9 mm, 5–9× runtime) keeps its per-keyframe observations in `kf_obs` (line 329) at the call site, so no per-track history is needed in this container. Decision 2.14 notes that dead tracks are dropped implicitly because there is "no re-association without descriptors", so no descriptor storage is needed either. The two-endpoint (`pos`, `kf_pos`) representation is exactly what KF-to-KF triangulation consumes.

### Lines 130–132: length

```python
130: 
131:     def __len__(self): return len(self.pos)
132: 
```

**What it does.** `len(tracks)` is the number of live tracks N. Lines 130 and 132 are blank separators. The caller compares this against `args.min_inliers` to decide whether a bootstrap attempt is worth trying (line 360) and whether to re-seed while waiting for bootstrap (line 417); post-bootstrap map starvation is tested on the count of map-bearing tracks (`has.sum()`, line 437), not on `len(tracks)`. It also guards the median-parallax computations (lines 358, 467–468) against an empty array and feeds the `n_tracks` column of `diag.csv` (lines 341, 354).

**Alternatives considered.** None recorded; trivial accessor.

**Why this choice.** Convenience so that the `len(tracks) == 0` guard in `lk_step` (line 154) and the bootstrap / re-seed floors read naturally.

### Lines 133–136: `add`, positions

```python
133:     def add(self, pts):
134:         n = len(pts)
135:         self.pos = np.vstack([self.pos, pts.astype(np.float32)])
136:         self.kf_pos = np.vstack([self.kf_pos, pts.astype(np.float32)])
```

**What it does.** Appends `n` freshly detected corners `pts` (shape (n,2), pixel units) as new tracks. Both `pos` and `kf_pos` are set to the detection position, i.e. a new track's "keyframe endpoint" is where it was born; its parallax since the last keyframe starts at zero.

**Alternatives considered.** Re-detecting every frame (no persistent state to append to) or refilling to a count floor between keyframes; both are options listed under decision 1.6.

**Why this choice.** Decision 1.6: "persistent tracks, top-up only at keyframes … Every map point is born at a keyframe and triangulated at the next"; setting `kf_pos` at birth is what makes a track born at keyframe k triangulable at keyframe k+1 with the correct first-view coordinates. The probe measured 95% per-frame survival, so "the count decays slowly between keyframes" and no floor parameter is needed.

### Lines 137–139: `add`, map ids and track ids

```python
137:         self.mp = np.concatenate([self.mp, -np.ones(n, np.int64)])
138:         self.ids = np.concatenate([self.ids, np.arange(self._next, self._next + n)])
139:         self._next += n
```

**What it does.** New tracks get `mp = -1` (not yet triangulated) and consecutive fresh ids from the monotone counter; the counter advances so ids are never reused within a scene.

**Alternatives considered.** Reusing row indices as ids (breaks under `keep`), or assigning map-point ids at detection (impossible: a monocular corner has no 3D until a second view).

**Why this choice.** Forced by decision 0.1 (triangulated monocular map: 3D comes only from a later view) and by the need for stable identities across `keep` for retro-fill and scale hand-off (decision 2.15).

### Lines 140–144: `keep`

```python
140: 
141:     def keep(self, mask):
142:         self.pos, self.kf_pos, self.mp, self.ids = self.pos[mask], self.kf_pos[mask], self.mp[mask], self.ids[mask]
143: 
144: 
```

**What it does.** Applies one boolean row mask to all four arrays at once, dropping the tracks where `mask` is `False`. This is the single mutation path for track death: LK failures / forward–backward failures / out-of-bounds (the `ok` mask from `lk_step`, applied at line 352 after the static-track lever of decision 1.2 has optionally been ANDed in at lines 346–351: on frames whose median displacement ≥ 2 px, tracks that moved < 1 px are removed), PnP-outlier culling (decision 2.14, `keep(~drop)` at line 454), and the wipe of all tracks at a re-bootstrap when `--scale_handoff` is off (`keep` with an all-`False` mask, line 308). Lines 140, 143, 144 are blank separators.

**Alternatives considered.** Marking tracks dead but retaining them for later re-association (ORB-SLAM style relocalisation), or keeping tracks alive through a PnP failure with a constant-velocity prediction.

**Why this choice.** Decision 2.14: "drop a map point after 3 consecutive PnP-outlier hits (dead tracks are dropped implicitly: no re-association without descriptors)" — sweep 1 showed culling cut PnP failures 26% → 5.6% of frames and RPE-t 9.65 → 8.18 mm at unchanged ATE, 2× faster. One mask over parallel arrays is the cheapest way to implement that.

### Lines 145–146: `detect`, exclusion mask

```python
145: def detect(gray, tracks, W, H):
146:     mask = np.full((H, W), 255, np.uint8)
```

**What it does.** Starts the keyframe top-up detector. `gray` is the current 320×192 8-bit grayscale frame (`load_gray_cover`, line 257); `W, H` are its width and height (line 258). Builds an all-255 (all-allowed) `uint8` mask of shape (H, W) — OpenCV's `goodFeaturesToTrack` `mask` argument is an 8-bit single-channel image where non-zero pixels are candidate locations.

**Alternatives considered.** This mask is **not** a gripper / outlier mask. Decision 1.2 lists the gripper-masking options (none; fixed dataset polygon; per-scene temporal-variance Otsu mask; flow-based rejection; GT `outlier_mask`) and decides "NO MASK for v0": the GT `outlier_mask` does not cover the gripper, the Otsu temporal-variance mask over-masks (58%) and a fixed polygon fails because finger geometry differs per lab. The gripper is instead handled after tracking by the static-track lever at the call site.

**Why this choice.** The only masking in v0 is the *track-exclusion* mask that decision 1.6 prescribes ("new corners detected with existing track positions excluded via mask"), built in the next two lines.

### Lines 147–148: punch out existing tracks

```python
147:     for x, y in tracks.pos:
148:         cv2.circle(mask, (int(round(x)), int(round(y))), MIN_DIST, 0, -1)
```

**What it does.** For every live track, draws a filled (`thickness=-1`) disc of radius `MIN_DIST = 5` px, value 0, centred on the track's rounded integer position (`cv2.circle` takes an integer `(x, y)` centre in column/row order). Pixels inside those discs are excluded from detection, so a new corner cannot be born within 5 px of a track that is already being followed. The 5 px radius is the same value as the detector's `minDistance`, so new–new and new–old corner spacing follow the same rule. (A centre outside the image is not an error: `cv2.circle` simply clips the disc, so a stray out-of-image track would just exclude nothing.)

**Alternatives considered.** No exclusion (re-detect and dedupe by distance afterwards); grid bucketing with per-cell quotas; top-up only in empty regions (decision 1.4's options).

**Why this choice.** Decision 1.6 (exclusion mask) plus decision 1.4: spatial spread = "none. minDistance=5 already spreads corners at this resolution (probe)"; "Bucketing = first v1 frontend experiment if long-scene drift dominates." Radius = `MIN_DIST` introduces no new parameter.

### Lines 149–152: Shi-Tomasi and return

```python
149:     p = cv2.goodFeaturesToTrack(gray, MAX_CORNERS, QUALITY, MIN_DIST, mask=mask)
150:     return np.zeros((0, 2), np.float32) if p is None else p[:, 0]
151: 
152: 
```

**What it does.** `cv2.goodFeaturesToTrack(image, maxCorners=1000, qualityLevel=0.01, minDistance=5, mask=mask)` runs the Shi-Tomasi minimum-eigenvalue detector (`useHarrisDetector=False`, `blockSize=3`, OpenCV defaults). `qualityLevel` is *relative*: a corner is kept only if its min-eigenvalue response is ≥ 1% of the strongest response in the frame; `minDistance` is a greedy Euclidean non-maximum suppression radius in pixels; `maxCorners` caps the strongest-first list. The return value is `None` when nothing passes, otherwise an (n,1,2) `float32` array in pixel coordinates; line 150 normalises both cases to an (n,2) array. Lines 151–152 are blank separators.

**Alternatives considered.** Decision 1.3 lists FAST, ORB, SIFT/AKAZE and a fixed grid.

**Why this choice.** Decision 1.3: Shi-Tomasi `(maxCorners=1000, qualityLevel=0.01, minDistance=5)` — "Designed for LK; relative quality threshold adapts across exposure; no wasted descriptor." The values are probe values: on the probe scene the detector produced only ~215 corners/frame (min 139), and per decision 1.3 the "cap never reached, ~140-300 corners/frame", so `MAX_CORNERS = 1000` is inactive in practice. The relative `qualityLevel` is what lets the same constant work across DROID scenes of very different exposure without a per-scene threshold.

### Lines 153–155: `lk_step`, empty guard

```python
153: def lk_step(g0, g1, tracks):
154:     if len(tracks) == 0:
155:         return np.zeros(0, bool)
```

**What it does.** Frame-to-frame track propagation from grayscale frame `g0` (frame f−1) to `g1` (frame f). If there are no live tracks it returns an empty boolean survival mask. OpenCV returns `(None, None, None)` for an empty point set, so without this guard line 159 would fail with a `TypeError` on `st[:, 0]`; the guard returns an empty bool mask so the caller's `tracks.keep(ok)` and lever code (`ok.any()` is `False`, line 346) degrade gracefully.

**Alternatives considered.** Raising, or triggering a detection inside `lk_step`. Neither: refill is governed only by the keyframe/bootstrap logic (decision 1.6, "No floor parameter").

**Why this choice.** Keeps `lk_step` a pure tracking primitive; map starvation and re-seeding are diagnosed by the caller (lines 417 and 437; decision 3.2b re-bootstrap rule).

### Lines 156–158: forward and backward pyramidal LK

```python
156:     p0 = tracks.pos.reshape(-1, 1, 2)
157:     p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None)
158:     p0b, stb, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None)
```

**What it does.** OpenCV wants (N,1,2) `float32` points, hence the reshape. `cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts=None)` runs sparse iterative Lucas–Kanade on an image pyramid with all defaults: `winSize=(21,21)`, `maxLevel=3`, termination `(COUNT|EPS, 30 iters, 0.01 px)`, `minEigThreshold=1e-4`, no `OPTFLOW_USE_INITIAL_FLOW` (so `nextPts=None` means the search starts at the previous position, i.e. a zero-motion prior). It returns `nextPts` (N,1,2) `float32`, `status` (N,1) `uint8` with 1 where the flow was found, and a per-point error which is discarded. Pitfall: when `status == 0` the corresponding `nextPts` entry is not meaningful and must be masked, which line 159 does. Line 158 tracks the *found* points back from `g1` to `g0`, producing `p0b`, the round-trip estimate of the original positions. Pyramids are rebuilt on every call (no `buildOpticalFlowPyramid` caching); at 320×192 this is cheap (correspondence benchmark: 4 ms per pair for LK + FB).

**Alternatives considered.** Decision 1.5: ORB / SIFT + ratio test; DIS or Farneback dense flow at corners or on a grid; LK without the backward pass. Larger windows / more levels for fast segments are standard knobs but not benchmarked.

**Why this choice.** Decision 1.5 and the correspondence benchmark (12 smoke scenes, 4243 frames, Slurm 45403560): at gap 1, LK+FB gives 166 correspondences / 88 within 2 px of GT flow and 0.88° essential-matrix rotation error, vs ORB 138/77/1.11°, SIFT 94/48/0.95° at 19 ms, Farneback 187/92/0.87° but collapsing to 187/12 at gap 4 (3.66°). "Descriptor matching rejected by data; Farneback collapses at gap 4." DIS on an 8-px grid was the measured runner-up (0.75° / 1.88° at gap 4 via 5× the point count) and is recorded as the first v1 frontend swap, but LK gives "corner-anchored persistent tracks the map needs". Window/level values are OpenCV defaults (frozen-parameter table: "21 px / 3 (OpenCV defaults)"). The probe also fixed frame-to-frame (not keyframe-to-frame) tracking: over a 15-frame gap in fast segments survival drops to ~15%, vs a 0.95 median consecutive-frame survival.

### Line 159: forward–backward acceptance

```python
159:     ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm((p0 - p0b)[:, 0], axis=1) < FB_THRESH)
```

**What it does.** A track survives only if the forward LK converged, the backward LK converged, and the round-trip position error `|p0 − p0b|` (Euclidean, pixels) is below `FB_THRESH = 1.0` px. This is the standard forward–backward consistency check: a point whose reverse track does not return to its origin was drawn to a different structure (occlusion boundary, repetitive texture, aperture problem) and is rejected before it can corrupt geometry.

**Alternatives considered.** No check (decision 1.5 lists "sparse LK (+/- fwd-bwd check)"); thresholding OpenCV's returned `err` (patch SSD) instead; a looser FB threshold.

**Why this choice.** Decision 1.5: "sparse LK at OpenCV defaults … + forward-backward check at 1 px". The benchmark shows the trade: LK without the check keeps more points (183/91 at gap 1; 167/31 at gap 4) with near-identical rotation error (0.85 vs 0.88), but "LK+FB has the best per-correspondence precision at gap 4 (in2 0.35 vs 0.29)", which is what matters for a persistent map whose points are triangulated and re-used many frames later. The 1 px value is stated in the doc as the LK noise floor (decision 2.7: "1 px is at the LK noise floor"), so a tighter FB threshold would reject by noise.

### Lines 160–161: image-bounds check

```python
160:     H, W = g1.shape
161:     ok &= (p1[:, 0, 0] >= 0) & (p1[:, 0, 0] < W) & (p1[:, 0, 1] >= 0) & (p1[:, 0, 1] < H)
```

**What it does.** Reads the frame size from the target image (`shape` is (rows, cols) = (H, W)) and additionally rejects any track whose new position lies outside `[0, W) × [0, H)` in pixel units (x = column, y = row). LK can return a converged point slightly outside the image because the window is padded at the border; such points have no image support (and no valid LK window for the next step) and would corrupt parallax / PnP input. This is a geometric guard, not a crash guard.

**Alternatives considered.** A border margin (e.g. half the LK window) as in many VO frontends; no check.

**Why this choice.** No numbered decision; it is a correctness guard chosen in code. A wider margin would introduce a parameter, contrary to the "as few free parameters as possible" purpose stated at the top of the design doc.

### Lines 162–165: commit new positions and return the mask

```python
162:     tracks.pos = p1[:, 0]
163:     return ok
164: 
165: 
```

**What it does.** Overwrites `tracks.pos` for **all** tracks (including the ones that failed) with the LK output and returns the survival mask; the caller is responsible for `tracks.keep(ok)` (line 352) after optionally ANDing in the static-track lever. `kf_pos` is deliberately untouched, so `pos − kf_pos` remains the parallax since the last keyframe. Lines 164–165 are blank separators.

**Alternatives considered.** Calling `keep` inside `lk_step`.

**Why this choice.** Decision 1.2's lever needs the per-frame displacement of every track (`tracks.pos − prev_pos`, line 349) and the median over the survivors *before* deciding which rows to drop; returning the mask instead of mutating lets the caller do that in one place. The lever itself (drop tracks with < 1 px step on frames whose median step ≥ 2 px) was measured to cut parallax-gated E-rotation error 2.14 → 1.70° and, in the full pipeline, v0 final ON vs OFF = 111.9/6.93/0.871 vs 115.6/8.40/0.971 (ATE mm / RPE-t mm / RPE-r deg); its default was flipped to ON on 2026-09-05 by user decision.

### Lines 166–168: `triangulate_pair` signature

```python
166: def triangulate_pair(K0, K1, T0_w2c, T1_w2c, a, b):
167:     """Triangulate world points from two w2c poses (each view with its own K); returns X (N,3), accept mask
168:     (cheirality in both cameras + reprojection < 2 px in both views)."""
```

**What it does.** Two-view triangulation of N tracks observed at pixel positions `a` (N,2) in view 0 and `b` (N,2) in view 1, given each view's camera matrix `K0`, `K1` (3×3, pixel units of the 320×192 frame) and each view's **w2c** pose `T0_w2c`, `T1_w2c` (4×4, maps world points into that camera's frame). In the pipeline view 0 is the previous keyframe (endpoint `kf_pos`, camera `Ks[kf_frames[-1]]`) and view 1 the new keyframe (endpoint `pos`, camera `Ks[frame_idx]`) — line 324, and line 462 for the `--new_points every` option — or the bootstrap anchor (`Ks[anchor_frame]`) and the bootstrap frame (`Ks[f]`), lines 372 and 387.

**Alternatives considered.** A single shared `K` for both views (the retroactive per-scene-median focal of the original decision 0.1b).

**Why this choice.** Decision 0.1b as revised 2026-09-06: "every geometric step carries a per-frame K (… KF-to-KF triangulation with each keyframe's K, PnP with frame t's K)", because the user's constraint is that OpenCV at frame t may only use what CUT3R has produced up to t. The revision reproduced the retroactive numbers (110.4/7.53/0.858 vs 111.9/6.93/0.871). Design note: with a per-scene constant K the pipeline turned out to be very focal-tolerant (a 2× focal error costs ~10 mm ATE and ~0.1° RPE-r) "because map and PnP share the same (wrong) K and Sim(3) absorbs the scale" — which is why map and PnP must draw their focal from the same source (here: the model's per-frame focal), even though each view carries its own K.

### Lines 169–171: projection matrices, DLT, dehomogenisation

```python
169:     P0 = K0 @ T0_w2c[:3]; P1 = K1 @ T1_w2c[:3]
170:     Xh = cv2.triangulatePoints(P0, P1, a.T.astype(np.float64), b.T.astype(np.float64))
171:     X = (Xh[:3] / np.where(np.abs(Xh[3]) < 1e-12, 1e-12, Xh[3])).T
```

**What it does.** Builds the 3×4 pixel-unit projection matrices `P = K [R | t]` from the top three rows of each w2c pose. `cv2.triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2)` is OpenCV's linear DLT (homogeneous least squares per point, no iterative refinement); pitfalls handled here: it requires the points as **2×N** arrays (hence `.T`) in the same units as `P` (pixels, because `K` is folded in — not normalised coordinates), and it returns **4×N homogeneous** coordinates `Xh`. Line 171 divides by the homogeneous coordinate `w = Xh[3]`, clamping `|w| < 1e-12` to avoid division by zero for points at infinity; such points get huge coordinates and are eliminated by the reprojection / finiteness tests below. Result `X` is (N,3) world coordinates in the map's arbitrary scale (set by the unit bootstrap translation, decision 2.10).

**Alternatives considered.** Midpoint triangulation, the optimal (Hartley–Sturm) two-view method, or a nonlinear reprojection refinement per point; triangulating in normalised coordinates (`K⁻¹x`) for better conditioning.

**Why this choice.** The design doc records no decision on the triangulation *algorithm*, only on acceptance (2.11); the linear DLT is OpenCV's only built-in and the benchmark showed that with the E-matrix pose "the two-view pose dominates map error", not the triangulation method. The design doc records no conditioning test for the DLT itself; the only related evidence is that map error is dominated by the two-view pose (2.11 KEY FINDING).

### Lines 172–173: cheirality in both cameras

```python
172:     Xw1 = np.c_[X, np.ones(len(X))]
173:     z0 = (T0_w2c @ Xw1.T)[2]; z1 = (T1_w2c @ Xw1.T)[2]
```

**What it does.** Homogenises `X` to (N,4) and transforms it into each camera frame with the w2c poses; row 2 of the result is the depth `z` along that camera's optical axis (OpenCV camera convention: +z forward, +x right, +y down). Positive `z` in **both** cameras is the cheirality condition: the point lies in front of both cameras. Because `T` is w2c, this is a direct transform, not an inverse.

**Alternatives considered.** Decision 2.11's option list: no filter; cheirality only; cheirality + reprojection; cheirality + parallax ≥ 1°; all three.

**Why this choice.** Decision 2.11, triangulation-acceptance benchmark (Slurm 45443315, GT-pose column isolates the filter): with no filter 46% of admitted points have > 20% depth error and PnP of the third frame reaches 0.73° rotation error; cheirality alone brings that to 32% / 0.63°. Cheirality improves bad-depth share and PnP rotation at zero parameter cost, at the price of more PnP failures (117 → 170 with the GT pose, 96 → 159 with the E pose) because fewer points are admitted (87 → 72 in both columns).

### Lines 174–176: reprojection residuals in both views

```python
174:     def proj(P):
175:         x = P @ Xw1.T; return (x[:2] / np.where(np.abs(x[2]) < 1e-12, 1e-12, x[2])).T
176:     r0 = np.linalg.norm(proj(P0) - a, axis=1); r1 = np.linalg.norm(proj(P1) - b, axis=1)
```

**What it does.** `proj(P)` projects the homogeneous world points with a 3×4 projection matrix and perspective-divides (same zero-guard as line 171) to (N,2) pixel positions. `r0`, `r1` are the Euclidean reprojection errors in pixels of the triangulated point against the observed track endpoints `a` and `b` in each keyframe.

**Alternatives considered.** Sampson or angular (bearing) error instead of pixel error; a single symmetric residual; a parallax-angle test (decision 2.11's fourth option).

**Why this choice.** Decision 2.11: "+ reprojection error < 2 px in both keyframes. No new parameter (2 px = PnP threshold)." The benchmark (GT pose) improves bad-depth share 32% → 22% and PnP rotation 0.63° → 0.56° over cheirality alone. The parallax filter was explicitly rejected for v0: "at keyframe-scale baselines (~36 mm) a 1 deg floor discards half the points and raises PnP failures 2x" (E-pose row: 47 accepted / 329 fails vs 72 / 159 for cheirality; GT-pose: 43 / 398 vs 72 / 170).

### Lines 177–180: acceptance mask and return

```python
177:     acc = (z0 > 0) & (z1 > 0) & (r0 < TRI_REPROJ) & (r1 < TRI_REPROJ) & np.isfinite(X).all(1)
178:     return X, acc
179: 
180: 
```

**What it does.** A point is accepted iff it has positive depth in both cameras, reprojects within `TRI_REPROJ = 2.0` px in both views, and has finite coordinates (catches the clamped-`w` points and any NaN from degenerate DLT). Returns the full `X` and the boolean mask; the caller assigns map-point ids to `X[acc]` only (lines 325–327; at bootstrap the mask is further ANDed with the essential-matrix inlier mask, line 373). Lines 179–180 are blank separators. Note that with a ~zero baseline (both views nearly coincident) a point can still pass cheirality + reprojection while being geometrically meaningless — this is why the keyframe trigger (decision 2.12) is parallax-based rather than a fixed stride.

**Alternatives considered.** As under 2.11; also "all three" (cheir + reproj + parallax).

**Why this choice.** Decision 2.11: cheir+reproj is "best on every metric" with the GT pose (PnP rot 0.56 vs 0.63 cheir-only vs 0.73 none; bad-depth 22% vs 32% vs 46%). With the actual E-matrix bootstrap pose all filters tie at ~4° (4.18 / 3.96 / 3.98 / 4.64 / 4.60) because "the two-view bootstrap pose … is the bottleneck of the map arm, not the acceptance filter" — the KEY FINDING recorded under that benchmark, which motivated the 0.5 px bootstrap threshold (2.10) and 5 px keyframe trigger (2.12). Decision 2.12's rationale for parallax over stride is precisely the ~zero-baseline pass-through noted above: "25% of 2-frame windows have < 2 px motion, where a stride keyframe would triangulate at ~zero baseline and pass cheir+rep."

### Lines 181–184: `pnp` signature and minimum-count guard

```python
181: def pnp(K, X, x, min_inliers):
182:     """Returns (T_w2c or None, inlier index array or None)."""
183:     if len(X) < 6:
184:         return None, None
```

**What it does.** Registers the current frame against the map: `X` (N,3) world points, `x` (N,2) their observed pixel positions in the current frame, `K` the current frame's 3×3 camera matrix (`Ks[f]`, line 427), `min_inliers` the failure floor. Returns the frame's **w2c** pose as a 4×4 matrix and the inlier row indices, or `(None, None)` on failure. Fewer than 6 correspondences is treated as a failure before calling OpenCV: `solvePnP` accepts ≥ 4 points, but the ITERATIVE method's non-coplanar initialisation is a DLT that needs 6 (checked on the installed OpenCV: `SOLVEPNP_ITERATIVE` raises "DLT algorithm needs at least 6 points" at n = 4, 5 and succeeds at n = 6), so 6 is the guard that keeps the solver call well-posed.

**Alternatives considered.** Falling back to P3P/AP3P (which are exact at 4 points), or trusting `solvePnPRansac` to handle 4–5 points.

**Why this choice.** The guard value 6 is a code choice, not a numbered decision. Decision 2.5 forbids fallback solvers ("NO automatic fallback solver in v0: a failed PnP is logged as a failed frame"), so a tiny correspondence set must surface as a failure. Note that the retro-fill of pre-bootstrap frames at the call site also requires ≥ 6 candidate tracks before calling this function (line 398).

### Lines 185–190: RANSAC PnP

```python
185:     try:
186:         ok, rvec, tvec, inl = cv2.solvePnPRansac(X.astype(np.float64), x.astype(np.float64), K, None,
187:                                                  flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=PNP_THRESH,
188:                                                  confidence=PNP_CONF, iterationsCount=PNP_ITERS)
189:     except cv2.error:
190:         return None, None
```

**What it does.** `cv2.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs=None, …)` — `distCoeffs=None` means a pure pinhole model (the dataset has no distortion and the frames are the model's own preprocessed 320×192 crops, decision 1.1 / dataset facts). Arguments: `flags=SOLVEPNP_ITERATIVE` selects the solver used for the final fit over all inliers (DLT / homography initialisation followed by Levenberg–Marquardt reprojection minimisation; in OpenCV 4.x — the installed OpenCV in this environment reports 4.11.0 via `python -c "import cv2; print(cv2.__version__)"`, a fact not recorded in the design doc — the minimal-sample hypotheses inside RANSAC are generated by EPnP on 5-point samples regardless of this flag, unless P3P/AP3P is requested); `reprojectionError=PNP_THRESH` (2.0 px; module constant at line 41, overridable by `--pnp_thresh`, lines 511 and 518–519) is the inlier threshold in pixels; `confidence=0.999` and `iterationsCount=1000` set the RANSAC stopping rule and cap; `useExtrinsicGuess` is left at its default `False`, so no initial pose is supplied. Return values: `ok` (bool), `rvec` (3,1) Rodrigues rotation vector and `tvec` (3,1) translation of the **world-to-camera** transform (`x_cam = R X_world + t`), and `inl` — an (M,1) `int32` array of inlier row indices, or `None` if no consensus was found. OpenCV raises `cv2.error` on some degenerate inputs (e.g. collinear image points or a rank-deficient DLT system); that is caught and reported as a failure rather than crashing the scene.

**Alternatives considered.** Decision 2.4: EPnP, P3P/AP3P, SQPnP, each ± `solvePnPRefineLM`. Decision 2.6: USAC/MAGSAC++ or no robust estimator. Decision 2.7: OpenCV default 8 px, or 1–3 px. Decision 2.8: default 0.99 / 100. Decision 2.5: identity or previous-motion initial guess.

**Why this choice.**
- Solver (2.4): PnP-variant benchmark (Slurm 45407881, oracle 3D from GT depth, RANSAC 2 px / 0.999 / 1000): all 10 variants within 0.05° / 2 mm; ITERATIVE gap-1 rotation 0.44° (p90 1.72), translation 3.9 mm; gap-4 1.10° (6.86) / 8.3 mm at 1.0 ms. SQPnP is nominally best by 0.04°; "differences are noise". ITERATIVE + RANSAC is "standard v0".
- Robust estimator (2.6): plain RANSAC handled identity-voting outliers (the image-static gripper) at a cost of only 1.10 → 1.14° at gap 4, and has "three stateable numbers".
- Threshold (2.7): 2 px, because "8 px = 2.2 deg at f~210, admits gripper tracks; 1 px is at the LK noise floor; ~65% of consecutive-frame tracks fall within 2 px of GT". Re-checked in sweep 2: 3 px → 124.6/8.07/1.084 (worse everything), 1 px → 120.4/7.71/0.866 but 58 re-bootstraps vs 42, so "2.7 stays 2 px". OpenCV's default (8 px) is *not* used.
- Confidence / iterations (2.8): 0.999 / 1000 costs 1–2 ms per frame and "removes the iteration cap as a variable on high-outlier frames". OpenCV's defaults (0.99 / 100) are not used.
- Initial guess (2.5): none; the benchmark had zero failures over 2118 pairs without a guess, and "a guess or a fallback would hide instability the diagnostics must expose".
- Per-frame `K` (0.1b revised): PnP uses frame t's focal (`Ks[f]`, line 427; `Ks[k]` for retro-filled frame k, line 400).

### Lines 191–193: inlier floor

```python
191:     if not ok or inl is None or len(inl) < min_inliers:
192:         return None, None
193:     inl = inl[:, 0]
```

**What it does.** The frame is declared a PnP failure if RANSAC reported failure, found no inliers, or found fewer than `min_inliers`. Otherwise the (M,1) index array is flattened to (M,). The caller counts failures, records `len(inl)` in `diag.csv` as the per-frame inlier count (line 442; decision 0.4), holds the last pose on failure (line 433; decision 3.2) and re-bootstraps after 3 consecutive failures (decision 3.2b) — only when `--reboot_after_fails 3` is passed, as in the frozen v0 command line; the argparse default is `0` (line 515), meaning re-bootstrap only on map starvation (`has.sum() < args.min_inliers`, line 437).

**Alternatives considered.** Decision 3.1: inlier count 10 / 20 / 30, or an inlier *ratio*.

**Why this choice.** Decision 3.1: 30. Sweep 1: floor 10 admits bad PnP solutions (RPE-r 1.795°); sweeps 2–3: 30 vs 20 gives RPE-r 0.861 vs 0.923°, RPE-t 7.07 vs 7.34 mm, ATE 116.8 vs 120.0 mm, at the price of more flagged frames (v0 final 21.5% vs 9.2% with floor 20). Note the argparse default in this file is still `--min_inliers 20` (line 505); the frozen v0 FINAL command line passes `--min_inliers 30` explicitly. The retro-fill of pre-bootstrap frames calls this function with `min_inliers = 4` (line 400; honesty-audit item 2, see below).

### Lines 194–197: LM refinement on the inliers

```python
194:     try:
195:         rvec, tvec = cv2.solvePnPRefineLM(X[inl].astype(np.float64), x[inl].astype(np.float64), K, None, rvec, tvec)
196:     except cv2.error:
197:         pass
```

**What it does.** `cv2.solvePnPRefineLM(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec)` minimises the sum of squared reprojection errors over the given points by Levenberg–Marquardt, starting from (and, in C++, updating in place) the supplied `rvec`/`tvec`; default criteria `(COUNT|EPS, 20, FLT_EPSILON)`. It is run on the RANSAC inlier subset only. A `cv2.error` (e.g. too few points for LM) leaves the RANSAC pose untouched.

**Alternatives considered.** No refinement, or `solvePnPRefineVVS`; each of the 2.4 variants "± solvePnPRefineLM".

**Why this choice.** Decision 2.4: "+ solvePnPRefineLM on inliers … LM refine is a no-op after ITERATIVE-RANSAC but kept so the refine step exists for the P3P/SQPnP swaps." The benchmark row is literally "ITERATIVE (+LM identical)": 0.44° / 3.9 mm with or without, because `SOLVEPNP_ITERATIVE`'s final inlier fit already ends in the same LM minimisation. The benchmark did show that P3P/AP3P need +LM to match on translation (10.2 → 8.7 mm at gap 4), which is why the step is kept as a hook.

### Lines 198–201: rotation vector to pose matrix

```python
198:     R, _ = cv2.Rodrigues(rvec)
199:     return rt_to_T(R, tvec), inl
200: 
201: 
```

**What it does.** `cv2.Rodrigues` converts the (3,1) axis-angle vector into a 3×3 rotation matrix (the second return is the Jacobian, unused). `rt_to_T` (lines 117–118) packs `R` and `t` into a 4×4 homogeneous matrix. The result is still **world-to-camera**; the design doc's plumbing checklist says explicitly "solvePnP returns world-to-camera (rvec, tvec): invert before writing c2w", and the caller does `poses_c2w[f] = np.linalg.inv(T_w2c)` when it stores each frame's pose (lines 402, 421, 434, 455). Lines 200–201 are blank separators.

**Alternatives considered.** Returning `(rvec, tvec)` directly; returning c2w.

**Why this choice.** Every consumer in the loop (keyframe poses `kf_T`, bootstrap anchor `boot_anchor_T`, triangulation `T_w2c` inputs, the last-pose hold `last_T` on failure) works in w2c so that `K @ T[:3]` is directly a projection matrix; one inversion per frame at store time is the fewest convention changes. The E-diag synthetic row ("E from GT flow, no noise (convention check)": 0.01° / 0.0°) is the recorded convention verification ("Conventions verified (synth0 = 0)"); the plumbing section additionally prescribes a PnP-with-GT-depth check ("PnP with GT depth between two frames must reproduce the GT relative pose"), which is not recorded as having been run, and the oracle-3D PnP benchmark (0.44° / 3.9 mm vs GT relative pose) is the closest recorded evidence for the PnP side.

### Known limitations / honesty notes for this block

- **Absolute accuracy floor is upstream of this code.** Even with oracle 3D points, per-frame PnP through this exact RANSAC/LM path has a 0.44° / 3.9 mm median error (GT median step 1.3° / 9 mm); the doc's reading: "That floor comes from LK noise at 320x180 plus GT depth/pose inconsistency, not from the solver; the map arm cannot beat it." In the full 4292-scene harness RPE-trans is saturated at the no-motion floor (7.9 mm) for every arm except the champion, so PnP translation quality is not measurable by that column.
- **Triangulation quality is bounded by the two-view bootstrap pose, not by the acceptance filter.** With the E-matrix pose ~53% of admitted map points have > 20% depth error and third-frame PnP lands at ~4° at a 4-frame gap, vs 0.56° with the GT pose on the same tracks and filter (decision 2.11 KEY FINDING). Chained LK tracks behave like ~3 px iid noise for two-view geometry at gap 4 and worse beyond gap 8.
- **Zero-baseline pass-through.** Cheirality + reprojection admits points triangulated at ~zero baseline; the pipeline relies on the parallax-based keyframe trigger (decision 2.12) rather than this function to avoid that. No parallax-angle test exists in v0 by measured choice.
- **No gripper mask.** The exclusion mask in `detect` only prevents duplicate tracks; the image-static gripper is handled downstream by the static-track lever (decision 1.2) and by RANSAC, not here. The lever thresholds (2 px median gate, 1 px drop) were designed with GT-aided inspection (audit item 5).
- **`min_inliers` is call-site dependent.** Live tracking uses the decided 30, but the retro-fill of frames between the bootstrap anchor and the bootstrap frame calls `pnp` with a floor of 4 and marks those frames as not failed (audit items 1, 2 and 12). Retro-fill is non-causal; the user ACCEPTED it on 2026-09-07 with a footnote in the full-4292 table. Its measured effect (12 smoke scenes, `--no_retro_fill` ablation) is +0.2 mm ATE, +0.2 mm RPE-t, +0.02° RPE-r; it changes no ordering.
- **Argparse defaults do not equal the frozen v0 configuration.** `--min_inliers` defaults to 20 (line 505) and `--reboot_after_fails` to 0 (line 515); the frozen v0 command line (design doc, "v0 FINAL configuration") passes `--min_inliers 30 --reboot_after_fails 3` explicitly, and `--pnp_thresh` can override the decided 2 px.
- **Guard values 6 (PnP) and the 1e-12 division clamps are code choices** with no benchmark behind them; the LK window (21 px), pyramid depth (3 levels), `minEigThreshold` and `solvePnPRefineLM` criteria are OpenCV defaults, not tuned values.
- **All thresholds in this block (1 px FB, 2 px reprojection / PnP, 0.999 / 1000, floor 30) were tuned on the 12 reported smoke scenes** (audit item 7); the doc addresses this by reporting the frozen configuration on all 4292 scenes, where the finetuned-CUT3R + OpenCV row ties on RPE (1.114° vs 1.101°; 0.0084 vs 0.0079 m) and loses 37% on ATE (0.104 vs 0.076 m), while on the zero-shot checkpoint the same code cuts RPE-rot 1.713 → 1.301° at equal ATE / RPE-t.
