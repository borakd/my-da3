## `eval_pipeline/opencv_vo.py` lines 252-337: `run_scene` setup (frames, per-frame K, VO state, `new_bootstrap`, `declare_keyframe`)

This block is the head of `run_scene`, the per-scene driver of the OpenCV monocular VO control arm (design of record: `OPENCV_VO_DESIGN.md`). It runs once per scene, before the frame-0 handling (lines 338-341) and the per-frame loop (line 343 onward): it loads every frame of the scene as a grayscale 320x192 image, builds one 3x3 camera matrix **per frame** (decision 0.1b, revised 2026-09-06 to an inference-time per-frame focal), allocates all the mutable state the loop will share (pose list, diagnostics, live 2D tracks, sparse map, keyframe lists, bootstrap anchor, failure counters), and defines the two closures the loop calls to change segments: `new_bootstrap` (start a new scale segment from the current frame, optionally remembering old map points for the 2.15 scale hand-off) and `declare_keyframe` (grow the map by keyframe-to-keyframe triangulation, register observations, optionally run local BA, top up corners and reset the keyframe reference positions). Everything downstream, PnP tracking (2.1/2.4-2.8), the bootstrap (2.9/2.10), culling (2.14) and the output writer, reads and writes the variables introduced here. Coordinate conventions used throughout: all 2D coordinates are pixels in the 320x192 cover frame (u right, v down); all stored camera poses inside the loop are **world-to-camera** 4x4 (`T_w2c`), as returned by OpenCV's `recoverPose`/`solvePnP`; only `poses_c2w` holds camera-to-world matrices, which is what the harness reads (`camera/%06d.npz`, key `pose`).

### Lines 252-254: function entry and timer

```python
252: # ----------------------------------------------------------------------------- the pipeline
253: def run_scene(scene_dir, preds_scene_dir, out_scene_dir, args):
254:     t_start = time.time()
```

**What it does.** `run_scene` takes the scene directory (`$SCENES_ROOT/<scene>`, containing `dense/rgb`, `dense/cam`, ...), the paired model's prediction directory for that scene (`<preds_root>/<scene>`, used only for the focal), the output directory (`<out_root>/<label>/preds/<scene>`) and the parsed CLI `args`. The wall-clock start feeds the `seconds` field of `summary.json` (line 488), which is the "s/scene" column of the sweep tables in the design doc. The comment on line 252 is a section divider.

**Alternatives considered.** None for this line; the per-scene, single-process, whole-sequence-in-memory structure is the offline harness convention (honesty audit item 3: "whole sequence in memory").

**Why this choice.** The arm is scored by the same per-scene harness as the CUT3R arms (`eval_depth_poses.py` per scene via `opencv_vo_eval.py`, nanmean over scenes), so a per-scene function that returns a summary dict is the natural unit. `main()` (lines 493-536) processes one shard of the scene list sequentially (`scenes[args.shard_id::args.num_shards]`, line 522; the fan-out over CPU workers is the caller's job) and skips scenes that already have a `summary.json` (line 526).

### Lines 255-258: load every frame as a grayscale 320x192 cover image

```python
255:     fs = sorted(glob.glob(os.path.join(scene_dir, "dense", "rgb", "*.png")))
256:     n = len(fs)
257:     G = [load_gray_cover(f) for f in fs]
258:     H, W = G[0].shape
```

**What it does.** Lists the scene's PNGs in name order (`dense/rgb/%06d.png`, 320x180 native, 57 to ~1700 frames per scene), sets `n` = number of frames (every one of them must get a pose, because `eval_depth_poses.py` numbers frames by position), and decodes all of them up front into `G`, a Python list of `uint8` `(H, W)` grayscale arrays. `load_gray_cover` (lines 46-58) reproduces the eval loader's `load_images_cover` geometry exactly: uniform scale by `max(320/W1, 192/H1)` = 1.067 for 320x180 input (bicubic, since the scale is >= 1), then a centre crop to 320x192, then `cv2.COLOR_RGB2GRAY`. `H, W` are therefore 192 and 320, read from the first frame rather than hard-coded. Memory is `n * 320 * 192` bytes, which is why the whole sequence can be held.

**Alternatives considered.** Decision 1.1 options: gray only; 2x upsample; CLAHE; undistort. Working in the native 320x180 frame with the calibrated K was the other framing option in 0.1b.

**Why this choice.** Decision 1.1: **grayscale conversion only**, on the eval loader's 320x192 frames, "so K and pixels agree" with the model's focal (0.1b: the model focal is ~211 px @320x192 vs calibrated ~190 @320x180). Gray is mandatory for Shi-Tomasi/LK; CLAHE/upsampling are listed as v1 single-variable experiments, never run. All pixel thresholds in the file (LK forward-backward 1 px, PnP 2 px, E 0.5 px, parallax 5 px) are in these 320x192 units. The GT images have no distortion (dataset facts), so no undistortion is needed.

### Lines 259-261: `K_of` and the self-calibration bookkeeping slot

```python
259:     selfcal_pairs = None
260:     def K_of(f):
261:         return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
```

**What it does.** `selfcal_pairs` is only filled by the `selfcal` branch (number of frame pairs the self-calibration used) and is written verbatim to `summary.json` (line 488). `K_of(f)` builds a pinhole matrix with `fx = fy = f` pixels, zero skew and principal point at the image centre `(W/2, H/2)`, as a `float64` `(3, 3)` array. This is the only K constructor in the per-frame branches, so aspect ratio 1 and pp = centre are baked in: even when the model's `intrinsics` matrix is read (line 274), only its `[0, 0]` entry is used.

**Alternatives considered.** Reading the model's full intrinsics matrix (its own pp and fy); the calibrated K from `cam/*.npz` (0.1b option "calibrated K"), which also has pp near the centre and no distortion.

**Why this choice.** Decision 0.1b fixes "pp = image center" and square pixels. This matches how CUT3R's focal is produced in the first place: the model has no intrinsics head; `camera/*.npz` focal is derived after inference by `estimate_focal_knowing_depth(pts3d_self, pp=centre, "weiszfeld")` (design doc, "Focal sensitivity and OpenCV self-calibration"), so the only meaningful scalar in that file is the focal. The pipeline is focal-tolerant anyway (2x focal error costs ~10 mm ATE / ~0.1 deg RPE-r, same section).

### Lines 262-264: `--focal selfcal` (classical self-calibration, not recommended)

```python
262:     if args.focal == "selfcal":
263:         f_sc, selfcal_pairs = selfcal_focal(G, W, H)
264:         Ks = [K_of(f_sc) if f_sc else model_K(preds_scene_dir, W, H, "model", scene_dir)] * n
```

**What it does.** `selfcal_focal` (lines 61-91) samples 40 candidate frame pairs at gap 6 (`n_pairs=40, gap=6`), tracks Shi-Tomasi corners with LK + forward-backward check, keeps parallax-gated pairs (median displacement >= 5 px, static tracks < 1 px dropped, >= 30 survivors), then sweeps candidate focals 100..500 px in 10 px steps (fine pass +-10 px in 2 px steps) and keeps the focal under which `cv2.findEssentialMat(RANSAC, 1 px, 0.999)` finds the most inliers summed over the pairs. It returns `(focal_px or None, n_pairs_used)`; `None` when fewer than 5 usable pairs exist, in which case line 264 silently falls back to the per-scene median model focal. `Ks` becomes a list of `n` references to one shared K (so it is a per-scene constant, not per-frame). Note the design doc describes the run as a sweep "over 30 parallax-gated pairs" while the code samples 40 candidates and keeps those passing the gate; the number of pairs actually used is what `selfcal_pairs` records.

**Alternatives considered.** 0.1b options: calibrated K; one nominal dataset K; model-estimated focal; self-calibration. Within self-calibration, the score could have used the Sampson error instead of the inlier count (the docstring mentions Sampson as a tie-break; the code as written uses inlier count only).

**Why this choice.** Decision 0.1b, "Focal sensitivity and OpenCV self-calibration" section: self-calibration was run once (Slurm 45456412) and gave 116.6 / 7.86 / 0.935 (ATE mm / RPE-t mm / RPE-r deg) vs 111.9 / 6.93 / 0.871 for the model focal, with per-scene focals of 132, 280, 196, 162, 358, 448, 174, 180, 210, 278, 298, 510 px against a true ~203. Verdict: "Classical self-calibration from this footage is unreliable (0.65x-2.5x scatter) ... `--focal selfcal` exists but is not recommended." The branch is kept as a measured, rejected option.

### Lines 265-268: inference-time focal modes (branch head)

```python
265:     elif args.focal.startswith("perframe:") or args.focal.startswith("causal:"):
266:         # INFERENCE-TIME focal: frame t uses only what the model has produced up to t.
267:         #   perframe:<preds_root> -> the model's focal for frame t itself
268:         #   causal:<preds_root>   -> running median of the model's focals for frames 0..t
```

**What it does.** Selects the two causal focal modes. The argument carries its own preds root after the colon (e.g. `perframe:$OUT/augfull_lr1e5/preds`), which is independent of `--preds_root`; the comments define the two modes. This is the branch the frozen v0 configuration uses (`--focal perframe:$OUT/augfull_lr1e5/preds`, "v0 FINAL configuration").

**Alternatives considered.** The retroactive per-scene median (branch on line 279, mode `model`); a fixed nominal focal (`--focal 203`); the running median (`causal:`), which is the other causal variant.

**Why this choice.** Decision 0.1b, revised 2026-09-06 ("Inference-time (causal) focal"): user constraint that "OpenCV at frame t may only use what CUT3R has produced up to t; retroactive per-scene medians are not an inference-time method." With the finetuned checkpoint the per-frame focal reproduces the retroactive numbers: 110.4 / 7.53 / 0.858 (focal of frame t, range 205-221 px) vs 111.9 / 6.93 / 0.871 (per-scene median). `perframe:` is the reported configuration (frozen command line; the full-4292 rows use "per-frame focal from the paired checkpoint"). The design doc gives no ruling on `causal:` and only records its measurements: finetuned running median 113.8 / 6.85 / 0.844; zero-shot running median 114.7 / 7.92 / 0.891 vs zero-shot per-frame 124.8 / 10.33 / 1.059 (per-frame range 207-819 px, running-median range 213-599 px).

### Lines 269-274: read the model's per-frame focal

```python
269:         mode, root = args.focal.split(":", 1)
270:         scene_name = os.path.basename(scene_dir)
271:         fpf = []
272:         for f in range(n):
273:             z = np.load(os.path.join(root, scene_name, "camera", f"{f:06d}.npz"))
274:             fpf.append(float(z["intrinsics"][0, 0]))
```

**What it does.** Splits the mode from the root on the first colon (so a root containing colons still works), then reads `<root>/<scene>/camera/%06d.npz` for frames `0..n-1` and takes `intrinsics[0, 0]` = `fx` in the model's 320x192 frame. `fpf` is a list of `n` floats. The loop assumes the paired model wrote one npz per frame with the same numbering as `dense/rgb`; a missing file raises `FileNotFoundError`, which `main()` catches (lines 535-536) and reports as `ERROR <scene>` (the scene then has no output). `fy`, the model's pp and any other key are ignored (see `K_of`).

**Alternatives considered.** Reading the model's full K; deriving the focal from the model's point map inside this script (that is what `estimate_focal_knowing_depth` already did upstream, so it would be a re-implementation).

**Why this choice.** Decision 0.1b (revised): "Frame t uses the focal CUT3R produced for frame t". The design doc records that this focal is "upstream demo.py's convention, reproduced unchanged by infer_and_eval_worker.py since the first run" and that "the OpenCV arm is its first consumer". The model's focal is inference output, not privileged information (0.1: "model depth is NOT privileged, it is inference output"; the same reasoning covers the focal). Which checkpoint supplies it is immaterial for the finetuned family (focal check: finetuned 111.9 / 6.93 / 0.871, fixed 203 px 109.1 / 7.04 / 0.869, calibrated 116.0 / 6.91 / 0.885), but the zero-shot checkpoint's 229-570 px per-scene values (207-819 per frame) are "not a usable camera model". On the full 4292 harness the zero-shot checkpoint's own per-frame focal is nevertheless what the "CUT3R zero-shot + OpenCV backbone" row uses, and that row cuts RPE-rot 1.713 -> 1.301 deg at equal ATE / RPE-t.

### Lines 275-277: running median (causal mode) and the per-frame K list

```python
275:         if mode == "causal":
276:             fpf = [float(np.median(fpf[:f + 1])) for f in range(n)]
277:         Ks = [K_of(v) for v in fpf]
```

**What it does.** For `causal:`, replaces each frame's focal by the median of the focals of frames `0..t` inclusive (an expanding window, recomputed from scratch for every frame); for `perframe:` `fpf` is left as read. `Ks` is then a list of `n` distinct `(3, 3)` matrices, one per frame. Every geometric step downstream indexes it by frame: the bootstrap essential matrix normalises each view by its own K (`Ks[anchor_frame]`, `Ks[f]`, lines 362-363), `triangulate_pair` receives each keyframe's K (line 324), and PnP uses `Ks[f]` (line 427) (design doc 0.1b revision: "every geometric step carries a per-frame K"). The writer also stores `Ks[f]` as the `intrinsics` of each output npz (line 479), and `summary.json` records the median, min and max focal over the scene (line 488).

**Alternatives considered.** A single K for the scene (the other two branches); a fixed-length trailing window instead of an expanding median (not run).

**Why this choice.** Decision 0.1b (revised 2026-09-06): the per-frame K plumbing is what makes the inference-time focal possible. The running median is the causal analogue of the retroactive per-scene median; the design doc records its numbers (finetuned 113.8 / 6.85 / 0.844; zero-shot 114.7 / 7.92 / 0.891 vs per-frame 124.8 / 10.33 / 1.059) without a ruling, and `perframe:` remains the reported configuration.

### Lines 278-280: per-scene constant K (model median, fixed float, or GT) and the BA copy

```python
278:     else:
279:         Ks = [model_K(preds_scene_dir, W, H, args.focal, scene_dir)] * n
280:     K = Ks[0]                                   # kept for the (unused in v0) local BA
```

**What it does.** All remaining `--focal` values go through `model_K` (lines 94-114): `"model"` (the argparse default) = median of `intrinsics[0, 0]` over all `camera/*.npz` under `--preds_root/<scene>` (retroactive, whole-scene); a numeric string such as `"203"` = fixed focal in px at 320x192; `"gt"` = the calibrated focal from `dense/cam/000000.npz` (key `intrinsic`) rescaled by the cover factor `max(W/W1, H/H1)` = 1.067, which reads GT and is marked DIAGNOSTIC ONLY. Again `Ks` is `n` references to one matrix. Line 280 keeps `K = Ks[0]` for the single-K `local_ba` signature (line 332): with a per-frame focal the BA would reproject every keyframe with frame 0's K, but BA is off in v0.

**Alternatives considered.** 0.1b options as above. For the "model-free" row: nominal 203 px vs 150 / 305 / 406 px (focal-sensitivity table: 109.8 / 7.23 / 0.991, 117.9 / 7.54 / 0.905, 119.2 / 7.45 / 0.975).

**Why this choice.** `"model"` is the original 2026-09-02 decision 0.1b (per-scene median, within-scene jitter 0.7%); it is no longer the reported configuration after the 2026-09-06 revision, and rows produced with it are honesty-audit item 13 ("stale per-scene-median rows on disk"). `--focal 203` "gives a strictly model-free arm at no cost" (109.1 / 7.04 / 0.869), but audit item 4 notes the 203 px value itself "came from GT calibration". `--focal gt` is audit item 6 ("a `--focal gt` code path exists"); on the full 4292 harness GT intrinsics are worth ~2 mm ATE (0.103 vs 0.104 m). Rulings on items 4, 6 and 13 are recorded as pending the user's decision.

### Lines 281-284: pose list, diagnostics, live tracks

```python
281: 
282:     poses_c2w = [None] * n
283:     diag = []                                   # per-frame dicts
284:     tracks = Tracks()
```

**What it does.** (Line 281 is blank.) `poses_c2w` is the output: one `4x4` **camera-to-world** matrix per frame, `None` until set; the writer (lines 473-479) holds the previous pose forward for any remaining `None` so that every frame gets a pose. `diag` collects one dict per frame with keys `frame, n_tracks, n_mp, inliers, failed, keyframe, booted, event`, written to `diag.csv` (lines 480-484). `Tracks` (lines 121-142) is a structure-of-arrays for the live 2D tracks: `pos (N, 2) float32` current pixel position, `kf_pos (N, 2)` position at the last keyframe (or at detection), `mp (N,) int64` map-point id or `-1`, `ids (N,) int64` unique track ids from a monotone counter; `add` appends corners with `kf_pos = pos` and `mp = -1`, `keep(mask)` filters all four arrays together.

**Alternatives considered.** Decision 0.4 options for failure reporting: ATE/RPE only; + failed-frame count; + failed-frame count and median inliers. Decision 1.6 options: re-detect every frame (no persistent state); persistent tracks + top-up on a count floor; persistent tracks + top-up only at keyframes.

**Why this choice.** 0.4: the harness scores ATE / RPE-trans / RPE-rot only; diagnostics go to "a side CSV (not the harness CSV): per-frame inlier count + failed flag, aggregated to per-scene failed-frame count and median inliers" because they "separate lost-tracking from drift; median inliers moves before ATE does". 1.6: persistent tracks are required because "every map point is born at a keyframe and triangulated at the next"; the probe measured 95% per-frame LK survival, so counts decay slowly between keyframes and no floor parameter is needed. Poses are stored c2w because the harness `camera/%06d.npz` convention is c2w ("solvePnP returns world-to-camera (rvec, tvec): invert before writing c2w", plumbing checks).

### Lines 285-287: the sparse map

```python
285:     mp_xyz = {}                                 # map point id -> world xyz
286:     mp_outl = {}                                # map point id -> consecutive PnP-outlier count
287:     next_mp = 0
```

**What it does.** The map is a dict from integer point id to a `(3,)` `float64` world point (world = the frame-0 camera frame of the current scale segment chain, see `boot_anchor_T`). `mp_outl` is the per-point run length of consecutive PnP-outlier hits used by `--cull outlier` (a point is popped from `mp_xyz` and its track dropped in the same step, lines 452-454, after `--cull_hits` consecutive hits, default 3). `next_mp` is the id counter; it is never reset, and `mp_xyz` is never cleared at a re-bootstrap, so `map_points` in `summary.json` (`len(mp_xyz)`) counts surviving points over all segments of the scene. Points have no descriptors: a map point lives exactly as long as its track (2.14: "dead tracks are dropped implicitly: no re-association without descriptors").

**Alternatives considered.** Decision 2.14 options: none; drop points unseen for N keyframes; drop after k PnP-outlier hits. Decision 0.2 options: frame-to-frame; keyframe-to-frame; local map.

**Why this choice.** 0.2 is forced by 0.1 (triangulated monocular map): "every frame registers to map points via PnP; keyframes are where the map grows". 2.14: dropping a point after 3 consecutive PnP-outlier hits cut PnP failures 26% -> 5.6% of frames and RPE-t 9.65 -> 8.18 mm at unchanged ATE, and halved runtime (sweep 1: 123.4 / 8.18 / 1.113, 13 s/scene vs 23). The frozen command line passes `--cull outlier`; the argparse default is `none`, so the switch must be given.

### Lines 288-291: keyframe lists and the bootstrap flag

```python
288:     kf_T = []                                   # keyframe w2c poses
289:     kf_obs = []                                 # keyframe: mp_id -> (u,v)
290:     kf_frames = []
291:     booted = False
```

**What it does.** Three parallel lists, one entry per keyframe of the current segment: `kf_T` the keyframe's `4x4` w2c pose, `kf_obs` a dict `mp_id -> (u, v)` pixel observations at that keyframe (consumed only by `local_ba`), `kf_frames` the frame index (used to fetch that keyframe's K from `Ks`). They are cleared and re-seeded with the anchor at every successful bootstrap (lines 407-409, outside this range), so `kf_T[-1]` is always the previous keyframe of the live segment. `len(kf_frames)` at the end is the `keyframes` field of `summary.json` (line 487), the source of the "KF/100f" column; because the lists are cleared at every re-bootstrap (line 407), this counts keyframes of the final segment only, not of the whole scene. `booted` is `False` while the loop is waiting for a bootstrap pair and `True` once a map exists.

**Alternatives considered.** A pose-graph or sliding-window structure holding all keyframes across segments (3.4 option "pose graph", never run); keeping observations for every frame rather than keyframes only (needed only for a full BA).

**Why this choice.** 2.13 (KF-to-KF triangulation only) needs just the last keyframe's pose, K and the tracks' `kf_pos`; observations are kept per keyframe because 3.4's local BA consumed them, and 3.4 was decided "none for v0" (BA window 5 / 10: ATE 128.8 / 130.7 vs 122.9, 5-9x runtime).

### Lines 292-296: bootstrap anchor, retro-fill history, last poses, counters

```python
292:     boot_anchor_T = np.eye(4)                   # w2c of the frame the current bootstrap is anchored at
293:     pre_hist = []                               # (ids, pos) per frame since anchor, for retro PnP
294:     anchor_frame = 0
295:     last_T = np.eye(4); prev_T = None           # w2c of last / previous frame (for const-velocity)
296:     n_reboot = 0; n_fail = 0; consec_fail = 0
```

**What it does.** `boot_anchor_T` is the w2c pose of the frame the current bootstrap is anchored at; identity means frame 0 is the world origin. At a re-bootstrap it becomes the held `last_T`, so a new segment starts from the last pose rather than from identity (position continuity; scale continuity only through 2.15). `pre_hist` stores `(ids.copy(), pos.copy())` for every frame since the anchor, so that once the bootstrap succeeds, frames `anchor+1..b-1` can be re-posed by PnP against the fresh map (retro-fill); `len(pre_hist)` also drives the `MAX_TRACK_LEN_BEFORE_BOOT = 400` re-anchor at line 417. `anchor_frame` indexes `Ks` for the anchor view. `last_T` is the w2c of the most recent frame (accepted or held), `prev_T` the one before, used by `--fail_policy cv` for a constant-velocity extrapolation (line 431); `prev_T = None` disables that until a second pose exists (guard on line 430). `n_reboot`, `n_fail` go to `summary.json` (`reboots`, `failed_frames`); `consec_fail` is the 3.2b counter. Note that `n_reboot` is incremented only on the map-dead / persistent-failure reboot at line 439; the pre-bootstrap re-anchor at lines 417-418 calls `new_bootstrap` without counting, so `reboots` undercounts the number of `new_bootstrap` calls.

**Alternatives considered.** 2.9 options for the bootstrap pair: frames 0/1; first frame with median displacement > P px; H-vs-E model selection (ORB-SLAM); tracked-ratio trigger. Filling pre-bootstrap frames: retro-fill by PnP (as coded, `--no_retro_fill` disables) vs holding the anchor pose (strictly causal). 3.2 options: hold last pose; constant velocity; drop frame.

**Why this choice.** 2.9: the probe found the RAIL scene stationary for frames 0-15 (GT rot 0.03 deg, |t| = 0.2 mm), so frame 0/1 is degenerate and the bootstrap must wait for parallax; "Frames 1..b-1 are filled retroactively by PnP against the bootstrap map so every timestep gets an OpenCV pose", which is why `pre_hist` exists. That retro-fill is **non-causal**: honesty audit item 1, ACCEPTED 2026-09-07 with a footnote clause in the full-4292 table; the ablation shows it is worth +0.2 mm ATE, +0.2 mm RPE-t, +0.02 deg RPE-r (110.4 / 7.5 / 0.858 with vs 110.6 / 7.7 / 0.881 without). 3.2: hold last pose + flag (sweep 1: constant velocity 121.9 / 8.47 / 1.375 vs hold 122.9 / 9.65 / 1.129, "mixed; rotation worse"), so `prev_T` only matters under the rejected `cv` policy.

### Lines 297-299: previous-segment map points for the scale hand-off

```python
297: 
298:     old_mp = {}                                 # track id -> world point in the PREVIOUS scale segment (for scale hand-off)
299: 
```

**What it does.** (Lines 297 and 299 are blank.) `old_mp` maps a *track* id (not a map-point id) to the world point that track carried in the previous scale segment. It is filled by `new_bootstrap` under `--scale_handoff` and consumed at the first bootstrap attempt after the reboot that reaches line 374 with >= 10 accepted triangulations (`if old_mp and acc.sum() >= 10`), whether or not that attempt then passes the `min_inliers` (30) floor on line 390; it is emptied there (line 389) regardless of the outcome, so a hand-off can be spent on an attempt that is subsequently rejected and be gone at the bootstrap that actually succeeds. When it is used: for accepted tracks whose id is in `old_mp`, the ratio of the old point's depth to the new point's depth in the anchor camera gives a per-track scale (lines 377-382); the median over >= 10 such tracks, if within `0.2 < s < 5`, rescales the unit bootstrap translation and re-triangulates (lines 383-388). Keyed by track id because map-point ids are forgotten when `tracks.mp` is reset.

**Alternatives considered.** Decision 2.15 options: implicit via map (each segment at its own arbitrary scale); explicit hand-off across re-bootstraps. Decision 3.3 options: scale as given by the map; per-frame normalisation.

**Why this choice.** 2.15 (sweep 3, Slurm 45453807): the hand-off took mean ATE 116.8 -> 111.9 mm and median 117.4 -> 97.7 mm at 6.93 / 0.871 RPE, firing on 12 of 46 re-bootstraps; per-scene it helped most on short scenes with several reboots (124.6 -> 64.0, 88.9 -> 71.5). The motivation is the sweep-2 reading that ATE "is set by the 30-58 re-bootstraps per 12 scenes, each starting a new arbitrary-scale segment that Sim(3) cannot repair". 3.3: scale is left as the map gives it, since Sim(3) scoring absorbs one global scale and "scale DRIFT is not" free.

### Lines 300-301: `new_bootstrap` signature

```python
300:     def new_bootstrap(frame_idx):
301:         nonlocal booted, boot_anchor_T, pre_hist, anchor_frame, old_mp
```

**What it does.** Closure over `run_scene`'s state; it rebinds five outer variables, hence the `nonlocal` list (the other state it touches, `tracks`, `mp_xyz`, `last_T`, is mutated or only read, so no declaration is needed). Called at frame 0 (line 340), when the not-yet-booted tracker has fewer than `min_inliers` tracks or more than `MAX_TRACK_LEN_BEFORE_BOOT = 400` entries of `pre_hist` (lines 417-418), and on a map-dead / persistent-failure reboot (line 439).

**Alternatives considered.** A tracker class with methods; keeping the bootstrap logic inline in the loop. Design-wise, the alternative to re-bootstrapping at all is to wait for the map to starve (3.2b option "only at map starvation", `--reboot_after_fails 0`).

**Why this choice.** 3.2b: re-bootstrap after 3 consecutive PnP failures cut failed frames 18 -> 14% and RPE-r 1.11 -> 1.01 vs waiting for starvation; 10 was "no better" (sweep 2: rb3 119.9 / 7.99 / 1.009, rb10 122.6 / 7.92 / 1.020). The frozen command passes `--reboot_after_fails 3`.

### Lines 302-305: hand-off path, keep tracks alive and remember their points

```python
302:         if args.scale_handoff:
303:             old_mp = {int(i): mp_xyz[int(m)] for i, m in zip(tracks.ids, tracks.mp) if m >= 0 and int(m) in mp_xyz}
304:             tracks.mp[:] = -1                       # keep the tracks alive, forget their (old-scale) points
305:             tracks.kf_pos = tracks.pos.copy()
```

**What it does.** With `--scale_handoff`, every live track that currently references a map point contributes `track_id -> world xyz` to `old_mp` (the `int(m) in mp_xyz` test is a defensive guard; culling at lines 452-454 pops the point and drops its track together, so the test should never exclude anything). Then all map-point references are cleared **in place** (`tracks.mp[:] = -1`, so the arrays keep their identity and length), while the 2D positions and ids survive; `kf_pos` is reset to the current positions so the bootstrap parallax gate (median `|pos - kf_pos| >= 5 px`, line 360) and the bootstrap correspondences (`kf_pos` -> `pos`, line 361) are measured from this frame. A consequence of line 303: if `new_bootstrap` is called again before the new segment boots (the pre-boot re-anchor at lines 417-418), every `tracks.mp` is already `-1`, so `old_mp` is rebuilt as an empty dict and the pending hand-off is discarded.

**Alternatives considered.** Wiping the tracks (the `else` branch); matching scale through the map points themselves (impossible: no descriptors, so the only identity that persists across the reset is the track id); using the last PnP pose's translation magnitude (not run).

**Why this choice.** 2.15: "keep tracks alive, match new segment's scale to the old points' depth through >= 10 shared tracks, 0.2 < s < 5". Keeping the LK tracks running is what makes the hand-off possible at all, since the persistent-track frontend (1.6) has no re-association. Numbers: sweep 3 as above; the frozen configuration passes `--scale_handoff` (argparse default is off).

### Lines 306-308: no hand-off path, wipe the tracks

```python
306:         else:
307:             old_mp = {}
308:             tracks.keep(np.zeros(len(tracks), bool))
```

**What it does.** Without the hand-off, no old points are kept and `tracks.keep(all-False)` empties all four track arrays, so the new segment starts from a clean detection over the whole frame (the `detect` mask on line 309 then has no exclusion circles).

**Alternatives considered.** Keeping tracks alive without using them for scale (would only change which corners survive; not run separately).

**Why this choice.** This is the pre-2.15 behaviour, i.e. the "cull+lever+rb3+mi30" row of sweep 3 (116.8 / 7.07 / 0.861, 41 reboots, 0 hand-offs), retained as the switch-off state so the option can be scored head to head (module docstring: "every still-open decision is a CLI switch").

### Lines 309-313: detect corners, reset the anchor and the retro-fill history

```python
309:         tracks.add(detect(G[frame_idx], tracks, W, H))
310:         booted = False
311:         boot_anchor_T = last_T.copy()
312:         anchor_frame = frame_idx
313:         pre_hist = [(tracks.ids.copy(), tracks.pos.copy())]
```

**What it does.** `detect` (lines 145-150) builds a `uint8` mask that is 255 everywhere except filled circles of radius `MIN_DIST = 5` px around every existing track, then calls `cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=5, mask=mask)`; OpenCV returns `(N, 1, 2) float32` corners (integer pixel positions stored as float32; no `cornerSubPix` refinement is applied, so sub-pixel precision only enters later through LK) or `None` when nothing passes, which `detect` converts to `(N, 2)` (empty on `None`). `qualityLevel` is relative: corners with a Shi-Tomasi min-eigenvalue response below `qualityLevel = 0.01` times the strongest corner in the frame are discarded, and `minDistance` suppresses corners closer than 5 px to a stronger one. `tracks.add` appends them with `kf_pos = pos`, `mp = -1`, fresh ids. Then the segment is marked un-booted, anchored at the current held pose `last_T` (identity at frame 0) and frame index, and `pre_hist` is restarted with the anchor frame's own `(ids, pos)` as element 0; the retro-fill later iterates `pre_hist[1:-1]` (line 396), i.e. frames `anchor+1 .. b-1`.

**Alternatives considered.** 1.3 detector options: Shi-Tomasi; FAST; ORB; SIFT/AKAZE; fixed grid. 1.4 spread options: none; grid bucketing; top-up in empty regions. Anchoring a new segment at identity instead of `last_T` (would make each segment's trajectory start at the origin; Sim(3) scoring cannot repair that either).

**Why this choice.** 1.3: Shi-Tomasi `(1000, 0.01, 5)` are probe values ("cap never reached, ~140-300 corners/frame"; on the RAIL probe scene ~215 per frame, min 139), chosen because it is "designed for LK; relative quality threshold adapts across exposure; no wasted descriptor". 1.4: no bucketing, "minDistance=5 already spreads corners at this resolution (probe)", zero parameters. 1.5 rejected ORB/SIFT for correspondence (gap-4 rotErr 3.53 / 2.54 deg vs LK+FB 2.55, and only 57 / 54 correspondences vs 103), so descriptor detectors have no role. The anchor-at-`last_T` choice follows from "every frame must get a pose" and hold-last-pose (3.2). `pre_hist` exists for the retro-fill (2.9, audit item 1).

### Lines 314-317: `declare_keyframe` signature

```python
314: 
315:     def declare_keyframe(frame_idx, T_w2c):
316:         """Triangulate KF-to-KF, register observations, detect new corners, reset kf_pos."""
317:         nonlocal next_mp
```

**What it does.** (Line 314 is blank.) Closure called with the frame index and its accepted **w2c** pose. It rebinds only `next_mp`; `kf_T`, `kf_obs`, `kf_frames`, `mp_xyz`, `mp_outl`, `tracks`, `diag` are mutated in place. Two call sites: the bootstrap frame `b` (line 410, after the anchor has been pushed as the first keyframe on lines 408-409) and every frame whose median track displacement since the last keyframe reaches `PARALLAX_PX = 5` (lines 467-469).

**Alternatives considered.** Decision 2.12 options for when this is called: fixed stride 2 / 4 / 8; median parallax since last KF >= 5 / 10 / 20 px; tracked-inlier ratio < 0.9 / 0.8 / 0.7.

**Why this choice.** 2.12: parallax 5 px (keyframe-trigger sweep, Slurm 45443937): median KF gap 2 frames (p10-p90 1-8), PnP-at-+4 rotation 2.96 deg, "never fires on a paused camera"; stride 2 scored 2.62 deg but "25% of 2-frame windows have < 2 px motion, where a stride keyframe would triangulate at ~zero baseline and pass cheir+rep". The same 5 px threshold is shared with the bootstrap gate (2.9 revised from 10 to 5).

### Lines 318-322: previous keyframe and triangulation candidates

```python
318:         if kf_T:
319:             T_prev = kf_T[-1]
320:             cand = (tracks.mp < 0)
321:             if args.new_points == "kf":
322:                 pass
```

**What it does.** Triangulation needs a previous keyframe; `kf_T` is empty only if `declare_keyframe` were called before any bootstrap, which the loop never does (the anchor is pushed first). `T_prev` is the previous keyframe's w2c. Candidates are all live tracks without a map point: corners detected at the previous keyframe, tracks whose triangulation was rejected at the previous keyframe (they stay `-1` and are retried over the next keyframe gap only, since `kf_pos` is reset at every keyframe on line 336, not from their original detection frame), and, on the bootstrap call, tracks the essential-matrix RANSAC marked as outliers. Lines 321-322 are a placeholder: under `--new_points kf` (the default and the decided value) nothing further filters `cand`; the `every` option additionally triangulates between keyframes in the main loop (lines 459-464) with an extra per-track 5 px parallax requirement, while lines 323-327 still run at every keyframe in both modes.

**Alternatives considered.** Decision 2.13 options: triangulate KF-to-KF only; every frame. A per-track parallax requirement at keyframes (the `every` path has one; 2.11 rejected a 1 deg parallax filter).

**Why this choice.** 2.13: KF-to-KF only. Sweep 1: every-frame 126.2 / 10.40 / 1.402 vs KF-only 122.9 / 9.65 / 1.129 (ATE mm / RPE-t mm / RPE-r deg). The E-diagnostics explain why: chained LK tracks behave like ~3 px iid noise at gap 4 and get worse with chain length, so "keyframes must be close (gap 2-4), not far", and adding points at every frame between keyframes adds near-zero-baseline geometry.

### Lines 323-327: KF-to-KF triangulation and map registration

```python
323:             if cand.sum() >= 1:
324:                 X, acc = triangulate_pair(Ks[kf_frames[-1]], Ks[frame_idx], T_prev, T_w2c, tracks.kf_pos[cand].astype(np.float64), tracks.pos[cand].astype(np.float64))
325:                 ci = np.where(cand)[0][acc]
326:                 for row, Xi in zip(ci, X[acc]):
327:                     tracks.mp[row] = next_mp; mp_xyz[next_mp] = Xi; mp_outl[next_mp] = 0; next_mp += 1
```

**What it does.** `triangulate_pair` (lines 166-178) forms projection matrices `P0 = K_prev @ T_prev[:3]`, `P1 = K_now @ T_w2c[:3]` (each view with its own per-frame K; w2c is the convention `cv2.triangulatePoints` expects), calls `cv2.triangulatePoints(P0, P1, a.T, b.T)` with `(2, N) float64` pixel correspondences (`a` = positions at the previous keyframe, `b` = positions now), dehomogenises the `(4, N)` result into world `X (N, 3)` (guarding `|w| < 1e-12`), and returns an accept mask `acc` = depth `> 0` in **both** cameras (cheirality) and reprojection error `< TRI_REPROJ = 2` px in **both** views and finite. `ci` maps accepted candidate rows back to track rows; each accepted track gets a fresh map-point id, its world point is stored, and its outlier counter zeroed. Rejected candidates keep `mp = -1`. The bootstrap segment is at the arbitrary unit scale of `recoverPose` (or the hand-off-rescaled one), so `X` is in that segment's units.

**Alternatives considered.** Decision 2.11 options: none; cheirality only; + reprojection < 2 px; + parallax >= 1 deg; all three. Also single-K triangulation (previous per-scene-median design).

**Why this choice.** 2.11 (triangulation benchmark, Slurm 45443315): with the GT pose cheirality + reprojection is best on every metric (PnP rot 0.56 deg vs 0.63 cheirality-only vs 0.73 none; bad-depth 22% vs 32% vs 46%); the parallax filter is excluded because "at keyframe-scale baselines (~36 mm) a 1 deg floor discards half the points and raises PnP failures 2x" (E-pose fails 329 for cheirality + parallax vs 163 for cheirality + reprojection). The 2 px value is the PnP threshold (2.7), so "no new parameter". The doc's key finding is that with the E-matrix pose the map is ~53% bad regardless of filter, i.e. the two-view bootstrap pose, not the acceptance test, bounds map quality. Per-view K is the 0.1b revision.

### Lines 328-329: push the keyframe and its observations

```python
328:         kf_T.append(T_w2c.copy()); kf_frames.append(frame_idx)
329:         kf_obs.append({int(m): (float(u), float(v)) for m, (u, v) in zip(tracks.mp, tracks.pos) if m >= 0})
```

**What it does.** Appends a copy of the w2c pose (a defensive copy; nothing in the file mutates `kf_T` entries in place, `local_ba` rebinds them at line 247) and the frame index, then a dict of every currently mapped track's observation at this keyframe (pixel `(u, v)` as Python floats), including points just triangulated and older points still tracked. These observations are read only by `local_ba`.

**Alternatives considered.** Recording only the newly triangulated points; recording observations for every frame (a full BA would need it).

**Why this choice.** Keeps the BA option (3.4) functional at zero cost; since 3.4 was decided "none for v0", `kf_obs` has no effect on the reported rows.

### Lines 330-334: optional windowed bundle adjustment (rejected option 3.4)

```python
330:         if args.refine == "ba" and len(kf_T) >= 3:
331:             try:
332:                 local_ba(K, kf_T, kf_obs, mp_xyz, window=args.ba_window)
333:             except Exception as e:  # BA must never kill a scene
334:                 diag.append(dict(frame=frame_idx, event=f"ba_error:{type(e).__name__}"))
```

**What it does.** Under `--refine ba` and once three keyframes exist, `local_ba` (lines 203-249) runs a scipy `least_squares` (Huber loss, `f_scale = PNP_THRESH`, sparse Jacobian, `max_nfev=30`) over the last `--ba_window` keyframes (default 5; oldest fixed, 6-DoF rotation-vector + translation for the rest) and the map points they observe, minimising pixel reprojection residuals with the single matrix `K = Ks[0]`; it rebinds the refined `kf_T` entries (line 247) and `mp_xyz` values (line 249) inside the caller's list/dict. Any exception is swallowed and logged as a `ba_error:<ExceptionName>` event row in `diag` (a bare event dict, not a per-frame record, which the CSV writer tolerates via `extrasaction="ignore"` and `r.get`). Note BA edits `kf_T` but not `last_T`/`poses_c2w`, so already-written frame poses are not refined; only future PnP benefits through the refined map.

**Alternatives considered.** Decision 3.4 options: none; sliding-window BA (scipy); pose graph.

**Why this choice.** 3.4: none for v0. Sweep 1: local BA window 5 / 10 gave ATE 128.8 / 130.7 vs 122.9, RPE-r 1.068 / 1.175 vs 1.129, RPE-t 8.49 / 9.04 vs 9.65, at 104 / 189 s per scene vs 23 (5-9x runtime). The argparse default is `none`; with a per-frame focal the single-K BA would also be slightly mis-modelled, which is moot while it is off.

### Lines 335-337: corner top-up and keyframe reference reset

```python
335:         tracks.add(detect(G[frame_idx], tracks, W, H))
336:         tracks.kf_pos = tracks.pos.copy()
337: 
```

**What it does.** (Line 337 is blank.) New Shi-Tomasi corners are detected on the keyframe image with all existing track positions masked out by 5 px discs (see lines 309-313 for `detect` semantics) and appended as unmapped tracks; they will be triangulated at the next keyframe. Then `kf_pos` of every track (old and new) is set to its current position, which does two things: it re-arms the keyframe trigger (median `|pos - kf_pos|` restarts at 0 and must climb back to 5 px), and it fixes the "view a" of the next KF-to-KF triangulation to this keyframe, so the baseline used for triangulation is always exactly one keyframe gap (median 2 frames under 2.12), including for tracks whose triangulation was rejected at this keyframe.

**Alternatives considered.** Decision 1.6 options: re-detect every frame; persistent tracks + top-up on a count floor; persistent tracks + top-up only at keyframes. Triangulating rejected tracks from their original detection frame instead of the last keyframe (longer baseline, but longer LK chains; the E-diagnostics show chain drift dominates beyond gap 8).

**Why this choice.** 1.6: top-up only at keyframes, "No floor parameter: the keyframe trigger (2.12) governs refill." Excluding existing tracks in the mask is what makes the persistent-track set non-redundant without a spread parameter (1.4). Resetting `kf_pos` at every keyframe is the mechanism behind the 2.12 result that closer keyframes are monotonically better ("every family improves monotonically as keyframes get closer") while never firing on a paused camera.

### Known limitations / honesty notes for this block

- **Retro-fill is non-causal.** `pre_hist` exists so that frames between the anchor and the bootstrap frame are re-posed against a map built from later frames (honesty audit item 1, ACCEPTED 2026-09-07 with a table footnote). Ablation: +0.2 mm ATE, +0.2 mm RPE-t, +0.02 deg RPE-r; `--no_retro_fill` gives the strictly causal arm. The retro PnP accepts >= 4 inliers vs 30 live and un-flags those frames (audit items 2 and 12, pending).
- **Whole sequence in memory.** `G` holds every frame before the loop starts (audit item 3, pending); the per-frame focal loop likewise reads all `n` npz files up front, although the values used at frame t depend only on frames `<= t`.
- **Focal provenance.** `--focal gt` reads GT intrinsics (audit item 6); the "model-free" 203 px value came from GT calibration (item 4); `--focal model` rows are retroactive per-scene medians and are marked stale (item 13). Only `perframe:` / `causal:` are inference-time, and only `perframe:` is a reported configuration; the design doc records `causal:` numbers without a ruling. Rulings on 4, 6 and 13 are pending.
- **Only `fx` is used.** `K_of` forces `fy = fx` and pp = image centre regardless of what the model's `intrinsics` matrix contains (decision 0.1b); the pipeline is focal-tolerant (2x error ~10 mm ATE), so this is not what limits the arm.
- **Segments, not one trajectory.** Each `new_bootstrap` starts a new arbitrary-scale segment anchored at the held pose; the hand-off (2.15) fired on 12 of 46 reboots on the 12 smoke scenes and cannot bridge the rest. The design doc attributes the ~117-125 mm ATE plateau (parity with zero-shot CUT3R, 60% worse than finetuned) to these re-bootstraps.
- **Hand-off bookkeeping is fragile.** `old_mp` is emptied (line 389) by the first attempt that reaches the `>= 10` accepted-triangulation gate, even if that attempt then fails the 30-inlier floor on line 390, and a pre-boot re-anchor (lines 417-418) rebuilds it as an empty dict because all `tracks.mp` are already `-1`. Neither case was isolated as a variable; the 12-of-46 firing rate is the measured outcome of the code as written.
- **`mp_xyz` is never cleared** at a re-bootstrap; old-segment points persist in the dict (unreferenced by any track), so `map_points` in `summary.json` is a cumulative count. Conversely `keyframes` (`len(kf_frames)`) is cleared at every successful bootstrap (line 407), so it counts the final segment only, and `reboots` counts only the line-439 reboots, not pre-boot re-anchors.
- **Bootstrap-rejected tracks can enter the map.** On the bootstrap call of `declare_keyframe`, candidates include tracks that the essential-matrix RANSAC rejected; they are re-triangulated between anchor and bootstrap frame under cheirality + 2 px reprojection only. The triangulation benchmark shows the acceptance filter is not the bottleneck (map ~53% bad with any filter under the E pose), so this was never isolated as a variable.
- **BA path is single-K and does not touch written poses** (`K = Ks[0]`; `kf_T` only). It is off in v0 (3.4) and was measured worse (ATE 128.8 / 130.7 vs 122.9).
- **All thresholds here (5 px parallax, 2 px reprojection, 5 px `minDistance`, hand-off bounds 10 tracks / 0.2-5) were tuned on the 12 reported smoke scenes** (audit item 7), addressed by reporting all 4292 scenes with the configuration frozen.
