## `eval_pipeline/opencv_vo.py` lines 202-251 — `local_ba`: optional windowed bundle adjustment (decision 3.4, measured and rejected for v0)

This block is the only back-end refinement the classical arm has, and it is **switched off in the frozen v0 configuration**. Decision 3.4 of `OPENCV_VO_DESIGN.md` weighed three options for trajectory refinement — none, a sliding-window bundle adjustment in scipy, or a pose graph — and settled on **none for v0**, because the scipy BA was built, benchmarked in pipeline sweep 1 (2026-09-05, Slurm 45453490, 12 smoke scenes) and lost: ATE 128.8 mm (window 5) / 130.7 mm (window 10) against 122.9 mm without BA, at 5-9x the runtime. The code is kept as the `--refine ba` switch (`argparse` line 507, default `"none"`; window size `--ba_window`, default 5, line 508) so the measurement is reproducible. When enabled, the nested keyframe helper `declare_keyframe` (defined at line 315 inside `run_scene`, called from line 410 for the bootstrap keyframe and from line 469 for a parallax-triggered keyframe) calls `local_ba` at line 332, right after the new keyframe has been appended to `kf_T` / `kf_frames` / `kf_obs` (lines 328-329), guarded by `args.refine == "ba" and len(kf_T) >= 3` (line 330) and a blanket `try/except` so "BA must never kill a scene" (lines 331-334; a caught exception is logged as a `ba_error:<ExceptionName>` diag event, line 334). At the bootstrap call site the segment's keyframe list has just been reset to the anchor keyframe (lines 407-408), so after the append `len(kf_T) == 2` and the guard cannot pass; BA can only fire on the parallax keyframes of line 469. It sits therefore at the very end of the keyframe step: after PnP has localised the frame, after new map points have been triangulated keyframe-to-keyframe (decision 2.13, lines 318-327), and before the corner top-up (decision 1.6, line 335). It refines, in place, the world-to-camera poses of the last `window` keyframes (the oldest one held fixed) and the 3D map points those keyframes observe, by minimising Huber-robustified pixel reprojection error with `scipy.optimize.least_squares` and a hand-built Jacobian sparsity pattern.

### Lines 202-207: section banner, signature and docstring

```python
202: # ----------------------------------------------------------------------------- optional local BA (3.4)
203: def local_ba(K, kf_T_w2c, kf_obs, mp_xyz, window):
204:     """Windowed bundle adjustment over the last `window` keyframes (oldest in window fixed) and the map
205:     points they observe. Reprojection residuals, Huber loss, scipy least_squares with a sparse Jacobian.
206:     kf_T_w2c: list of 4x4 (modified in place for the window); kf_obs: list of dict mp_id -> (u,v);
207:     mp_xyz: dict mp_id -> np.array(3) (modified in place)."""
```

**What it does.** The banner comment ties the block to design decision 3.4. The function takes five arguments and returns nothing; its effect is entirely through in-place mutation of two of them.

- `K` — one 3x3 pinhole camera matrix in pixel units of the 320x192 eval-loader frames (`[[f,0,W/2],[0,f,H/2],[0,0,1]]`, principal point at the image centre, no distortion). Note what the caller passes (line 280): `K = Ks[0]`, i.e. the camera matrix of frame 0 only, with the comment "kept for the (unused in v0) local BA". Under the per-frame focal of the revised decision 0.1b, the rest of the pipeline uses `Ks[frame]` for E, triangulation (line 324) and PnP, but BA would project every keyframe with frame 0's K (see honesty notes).
- `kf_T_w2c` — Python list of 4x4 float64 **world-to-camera** rigid transforms, one per keyframe in the current scale segment (`kf_T` in `run_scene`), `X_cam = R X_world + t`. This is the convention `solvePnP` returns and that `rt_to_T` (lines 117-118) packs; it is the inverse of the c2w pose written to `camera/*.npz`.
- `kf_obs` — list parallel to `kf_T_w2c`; entry k is a dict `map_point_id -> (u, v)` of the pixel position (float, same 320x192 units as `K`) at which keyframe k observed that map point (built at line 329 from the live tracks, or at line 409 for the anchor keyframe of a new segment).
- `mp_xyz` — dict `map_point_id -> np.array(3)` of map-point positions in the world frame of the segment, in the arbitrary metric fixed by the bootstrap's unit translation (decision 2.10). The window's poses and these points are overwritten on exit.
- `window` — number of most-recent keyframes to optimise over (`args.ba_window`).

**Alternatives considered.** The design doc's decision 3.4 options are: no refinement; a sliding-window BA in scipy (this block); a pose graph. The standard literature alternatives for a keyframe-based monocular pipeline are ORB-SLAM-style local BA over the covisibility neighbourhood (with every out-of-window keyframe that sees a window point held fixed), a g2o/Ceres/GTSAM implementation with analytic Jacobians and Schur complement, or a motion-only BA (poses only, map frozen).

**Why this choice.** Decision 3.4: **none for v0**. Sweep 1 numbers (means over the 12 smoke scenes, ATE mm / RPE-trans mm / RPE-rot deg): `vo_v0` 122.9 / 9.65 / 1.129 at 23 s/scene; local BA window 5 → 128.8 / 8.49 / 1.068 at 104 s/scene; window 10 → 130.7 / 9.04 / 1.175 at 189 s/scene. BA lowered RPE-trans and (at window 5) RPE-rot slightly but raised ATE on both windows and multiplied runtime by 5-9x, so the switch defaults to `none`. The design doc records no explicit reason for implementing the BA option in scipy rather than g2o/Ceres; decision 3.4 only lists the option as "sliding-window BA (scipy)". The author's inference is that it is consistent with the doc's stated purpose — a bare-bones classical pose baseline with as few free parameters as possible, serving as a control next to the CUT3R arms — and with the file depending on nothing beyond numpy/OpenCV (plus scipy for this one switch).

### Lines 208-210: lazy scipy imports

```python
208:     from scipy.optimize import least_squares
209:     from scipy.sparse import lil_matrix
210:     from scipy.spatial.transform import Rotation as Rot
```

**What it does.** Imports the three scipy pieces inside the function body: the trust-region least-squares solver, a row-based sparse matrix class used to declare the Jacobian's non-zero pattern, and the `Rotation` class used to convert between 3x3 rotation matrices and rotation vectors (axis-angle, radians — the same parameterisation as OpenCV's `Rodrigues`/`rvec`).

**Alternatives considered.** A module-level import; OpenCV's `cv2.Rodrigues` for the rotation conversions (already used elsewhere in the file, e.g. line 198).

**Why this choice.** Local imports mean scipy is only imported when `--refine ba` is used: `scipy` is referenced nowhere else in `opencv_vo.py` (the only other occurrence of the word is the docstring at line 205), so the frozen v0 (`--refine none`, decision 3.4) never imports it. The doc gives no reason for the local import; it is a plumbing convenience, not a benchmarked decision.

### Lines 211-214: select the window; bail out if it is too short

```python
211:     kfs = list(range(max(0, len(kf_T_w2c) - window), len(kf_T_w2c)))
212:     if len(kfs) < 2:
213:         return
214:     fixed = kfs[0]; free = kfs[1:]
```

**What it does.** `kfs` is the index list of the last `window` keyframes (all of them if fewer exist), i.e. the window holds `min(len(kf_T), ba_window)` keyframes. With fewer than two keyframes there is nothing to adjust, so the function returns untouched. Otherwise the **oldest** keyframe in the window is declared `fixed` (its pose is a constant during the optimisation) and the remaining `free` keyframes carry 6 unknowns each. Fixing one camera removes the 6-DoF gauge freedom of a rigid-body ambiguity; it does **not** remove the 7th, scale, gauge direction of a monocular problem (scaling every free translation and every point about the fixed camera's centre leaves all residuals unchanged). The caller's own guard (`len(kf_T) >= 3`, line 330) means that for the default `--ba_window 5` the window always has at least 3 keyframes (and at most 5); with `--ba_window 2` it would have exactly 2 (one fixed, one free), and the `< 2` test at line 212 is then never the binding one.

**Alternatives considered.** Fix the two oldest keyframes (pins scale as well); ORB-SLAM's rule of fixing every keyframe outside the window that observes any window point; add a gauge prior instead of hard-fixing; optimise all keyframes (full BA).

**Why this choice.** Simplest gauge fix for a windowed problem; nothing in the design doc benchmarks the fixing rule separately — the whole 3.4 arm was rejected on the sweep-1 numbers above (ATE 128.8 / 130.7 vs 122.9), so the sub-choices inside it were never optimised. The open scale gauge is a known limitation (see honesty notes).

### Lines 215-219: collect the window's map points and observations

```python
215:     pts = sorted({m for k in kfs for m in kf_obs[k] if m in mp_xyz})
216:     if len(pts) < 10:
217:         return
218:     pidx = {m: i for i, m in enumerate(pts)}; kidx = {k: i for i, k in enumerate(free)}
219:     obs = [(k, m, kf_obs[k][m]) for k in kfs for m in kf_obs[k] if m in pidx]
```

**What it does.** `pts` is the sorted set of map-point ids observed by any keyframe in the window *and* still present in `mp_xyz` (points culled under decision 2.14, `mp_xyz.pop` at line 453, may still appear in old `kf_obs` dicts and are skipped here). Fewer than 10 points → return without changes. `pidx` maps a map-point id to its column block in the parameter vector; `kidx` maps a **free** keyframe index to its 6-parameter block (the fixed keyframe is absent from `kidx` on purpose). `obs` is the flat list of (keyframe index, map-point id, `(u, v)` pixel) triples over every keyframe in the window, *including the fixed one* — its observations still constrain the points. Each entry contributes a 2-vector residual.

**Alternatives considered.** Requiring a minimum number of observations per point (e.g. ≥ 2 keyframes, so each point is determined); a minimum per free keyframe; weighting by track age.

**Why this choice.** The `< 10` floor is a plain degeneracy guard, not a well-posedness check and not a tuned value: it bounds the number of *points*, whereas the residual count is `2*len(obs)` (line 232) and the unknown count is `6*n_free + 3*n_pts` (line 220-222), so 10 points by themselves say nothing about whether the system is over-determined. It appears nowhere in the design doc's tables. No per-point observation-count filter is applied, so points seen by only one window keyframe enter with 2 residuals against 3 unknowns and are locally underdetermined (see honesty notes). Again, none of this was tuned because decision 3.4 rejected the block on the sweep-1 outcome.

### Lines 220-223: initial parameter vector

```python
220:     x0 = np.concatenate([np.concatenate([Rot.from_matrix(kf_T_w2c[k][:3, :3]).as_rotvec(), kf_T_w2c[k][:3, 3]]) for k in free]
221:                         + [mp_xyz[m] for m in pts])
222:     n_cam = 6 * len(free)
223:     fixed_T = kf_T_w2c[fixed]
```

**What it does.** Builds the unknown vector `x0` of length `6*len(free) + 3*len(pts)`: first, for each free keyframe in order, its **absolute** world-to-camera rotation as a 3-vector rotation vector (`Rotation.from_matrix(...).as_rotvec()`, axis-angle in radians, matching OpenCV's `rvec` convention) followed by the 3-vector w2c translation `t` (map units); then, for each map point in `pts` order, its world XYZ. `n_cam` is the offset where the point block starts. `fixed_T` snapshots the fixed keyframe's 4x4 so it is not affected by the in-place write-back at line 247 (which only touches `free` anyway). The initial values are the pipeline's current PnP poses and triangulated points, so BA starts from a consistent, already-good estimate and is a local polish; the keyframes in a window are close together (median inter-keyframe gap 2 frames under decision 2.12; median per-step motion 9 mm / 1.3 deg per the dataset facts), so the initial guess is near the solution.

**Alternatives considered.** Quaternion or SO(3)-manifold parameterisation with local perturbations (increment relative to the current pose); inverse-depth points anchored to their first keyframe; parameterising c2w instead of w2c.

**Why this choice.** Rotation vectors are minimal (3 parameters, no unit constraint) and are what `solvePnP` already delivers. Because the parameter is the absolute w2c rotation of each keyframe (not the increment from the previous keyframe), the chart's singularity sits at a rotation angle of π *from the segment's world frame*; the world frame is fixed at the segment's bootstrap (decision 2.10), which makes a near-π absolute rotation unlikely within a window but not impossible over a long segment — the small inter-keyframe motion does not protect against it. Parameterising w2c means the residual (lines 234-237) needs no matrix inverse. Standard choice, no benchmark in the doc.

### Lines 224-228: unpack the camera block

```python
224:     def unpack(x):
225:         Ts = {fixed: fixed_T}
226:         for k in free:
227:             i = kidx[k]; rv = x[6 * i:6 * i + 3]; t = x[6 * i + 3:6 * i + 6]
228:             Ts[k] = rt_to_T(Rot.from_rotvec(rv).as_matrix(), t)
```

**What it does.** `unpack` is the inverse of the packing at lines 220-221. It returns a dict `Ts` from keyframe index to 4x4 w2c transform: the fixed keyframe's constant `fixed_T`, and for each free keyframe the 6-slice of `x` re-expanded via `Rotation.from_rotvec(...).as_matrix()` and `rt_to_T` (lines 117-118, which writes `R` and `t` into an identity 4x4). It is called on every residual evaluation, so it runs once per solver function evaluation plus once per finite-difference Jacobian column group.

**Alternatives considered.** Keeping poses as `(rvec, tvec)` pairs and using `cv2.Rodrigues` + `cv2.projectPoints` per keyframe (vectorised over points); a fully vectorised numpy unpack.

**Why this choice.** Readability over speed in a block that was expected to be, and was measured as, a single-variable experiment. The per-observation Python loops (here and in `resid`) and the finite-difference Jacobians are the obvious cost centres, but the design doc records only the outcome — runtime went from 23 s/scene to 104 (window 5) and 189 (window 10) s/scene — not a profile, so the split between loops, finite differencing and the LSMR solve is not known. That runtime is one of the two grounds on which decision 3.4 rejected the block.

### Lines 229-230: unpack the point block

```python
229:         P = x[n_cam:].reshape(-1, 3)
230:         return Ts, P
```

**What it does.** The tail of `x` from offset `n_cam` is viewed as an `(len(pts), 3)` array of world-frame map points, in `pts` order (row `pidx[m]` is point `m`). Returns both blocks.

**Alternatives considered / why.** Plain slicing; nothing to decide here. The `reshape` is a view, so no copy is made per evaluation.

### Lines 231-234: residual function, part 1 — transform each observed point into its camera

```python
231:     def resid(x):
232:         Ts, P = unpack(x); out = np.empty(2 * len(obs))
233:         for i, (k, m, uv) in enumerate(obs):
234:             Xc = Ts[k][:3, :3] @ P[pidx[m]] + Ts[k][:3, 3]
```

**What it does.** `resid` is the vector function `least_squares` minimises (½ Σ ρ(r_i²) with ρ the Huber loss chosen at line 244). It allocates `2*len(obs)` outputs — one `(du, dv)` pair per observation — and, for each observation, maps the point's current world position into keyframe `k`'s camera frame with the w2c transform: `X_cam = R X_world + t`. No inversion, because the parameters are w2c.

**Alternatives considered.** `cv2.projectPoints(P, rvec, tvec, K, None)` per keyframe, which also returns an analytic Jacobian with respect to `rvec`, `tvec`, focal, principal point and distortion — usable for the camera block of a supplied `jac=`, but `projectPoints` gives no derivative with respect to the 3D point, so the point block of the Jacobian (half the parameter vector here, line 221) would still have to be derived by hand (e.g. `d(u,v)/dX = J_proj · R`); a vectorised gather over all observations at once.

**Why this choice.** Direct and convention-transparent (w2c multiply, then pinhole projection); the `rt_to_T` / w2c convention is the same one that the plumbing check in the design doc requires ("solvePnP returns world-to-camera (rvec, tvec): invert before writing c2w"). Not benchmarked separately.

### Lines 235-238: residual function, part 2 — pinhole projection and pixel residual

```python
235:             z = Xc[2] if abs(Xc[2]) > 1e-9 else 1e-9
236:             out[2 * i] = K[0, 0] * Xc[0] / z + K[0, 2] - uv[0]
237:             out[2 * i + 1] = K[1, 1] * Xc[1] / z + K[1, 2] - uv[1]
238:         return out
```

**What it does.** Guards the depth `z` against exact zero (replaces |z| ≤ 1e-9 by +1e-9), then projects with the pinhole model `u = fx·X/z + cx`, `v = fy·Y/z + cy` using `K[0,0]`, `K[1,1]` (focal in px) and `K[0,2]`, `K[1,2]` (principal point = image centre), and subtracts the observed `(u, v)`. Residual units are therefore **pixels in the 320x192 frame** — the same unit as the PnP RANSAC threshold (decision 2.7) and the triangulation acceptance threshold (decision 2.11), which is what allows `f_scale=PNP_THRESH` on line 244 to be meaningful. Pitfalls: the guard only handles `z ≈ 0`; a point that moves *behind* a camera (`z < 0`) is not rejected — it projects to the mirrored pixel and keeps contributing a (Huber-capped) residual rather than being dropped, unlike the cheirality tests in the bootstrap (decision 2.9/2.10) and triangulation (decision 2.11).

**Alternatives considered.** Normalised-coordinate residuals (divide by `f`) with a threshold in radians; dropping observations with `z ≤ 0` inside the loop; angular (bearing) residuals, which are robust to the `z → 0` singularity.

**Why this choice.** Pixel residuals keep every threshold in the arm in one unit ("All pixel thresholds are in these units", decision 1.1). The `1e-9` clamp is a numerical guard, not a tuned value. The missing negative-depth handling is a limitation of the unused block (honesty notes).

### Lines 239-243: Jacobian sparsity pattern

```python
239:     S = lil_matrix((2 * len(obs), len(x0)), dtype=int)
240:     for i, (k, m, _) in enumerate(obs):
241:         if k in kidx:
242:             S[2 * i:2 * i + 2, 6 * kidx[k]:6 * kidx[k] + 6] = 1
243:         S[2 * i:2 * i + 2, n_cam + 3 * pidx[m]:n_cam + 3 * pidx[m] + 3] = 1
```

**What it does.** Declares which entries of the `(2·n_obs) x (6·n_free + 3·n_pts)` Jacobian can be non-zero: the two residual rows of observation `i` depend on the 6 parameters of keyframe `k` (only if `k` is free — `k in kidx` is exactly the "not the fixed keyframe" test) and on the 3 coordinates of point `m`. Everything else is structurally zero. `least_squares` uses this pattern to estimate the Jacobian by finite differences with column grouping (many mutually independent columns perturbed per evaluation), which turns an `O(n_params)` finite-difference cost into a handful of residual evaluations per Jacobian. `jac_sparsity` is rejected by `method='lm'` (`'trf'`, the default used here, and `'dogbox'` both accept it) and forces `tr_solver='lsmr'` (the `'exact'` dense solver is incompatible with it), so the sparse normal equations are solved iteratively with LSMR — verified against the installed scipy 1.13.1 in the `cuteanything` env.

**Alternatives considered.** Supplying an analytic `jac=` (camera block from `cv2.projectPoints`, point block by hand) — exact, and no finite differencing at all; a dense finite-difference Jacobian (no `jac_sparsity`), infeasible at hundreds of points; Schur-complement solvers in g2o/Ceres.

**Why this choice.** The sparsity pattern is the cheapest way to make scipy's generic solver usable on a BA-sized problem without writing derivatives; the docstring (line 205) names "sparse Jacobian" as a design feature. No benchmark of Jacobian strategies exists in the design doc — the block was rejected as a whole.

### Line 244: the solve

```python
244:     sol = least_squares(resid, x0, jac_sparsity=S, loss="huber", f_scale=PNP_THRESH, max_nfev=30, x_scale="jac")
```

**What it does.** Runs scipy's trust-region-reflective least squares from `x0`:

- `loss="huber"` — robust loss: quadratic for residuals with |r| ≤ `f_scale`, linear beyond, so a gross outlier (a wrong association or a gripper track surviving decision 1.2's lever) is bounded in influence instead of dragging the solution.
- `f_scale=PNP_THRESH` — the Huber transition point, in pixels. `PNP_THRESH` is the module global `2.0` (line 41; overridable by `--pnp_thresh`, lines 511/518-519), i.e. the PnP RANSAC reprojection threshold of decision 2.7. Reusing it means BA introduces no new pixel parameter: a residual is "inlier-like" exactly where PnP would have called it an inlier.
- `max_nfev=30` — hard cap on the number of residual evaluations. For `'trf'` scipy counts only the evaluations of trial steps toward this budget, not the finite-difference evaluations used to build each Jacobian (checked in scipy 1.13.1's `trf.py`), so it is effectively a cap of at most 30 trial steps. This bounds the runtime per keyframe rather than letting the solver converge to tolerance.
- `x_scale="jac"` — scales each variable by the inverse norm of its Jacobian column, so that rotation vectors (radians), translations and points (map units, arbitrary scale after the unit-translation bootstrap of decision 2.10) are conditioned comparably; this matters because the map's metric is arbitrary per segment.
- Default `method='trf'`, default `jac='2-point'` finite differences, scipy's default `ftol/xtol/gtol` tolerances.

The return value `sol` carries `sol.x`, `sol.cost`, `sol.success`, `sol.nfev`; **only `sol.x` is used** — no success or cost-decrease check gates the write-back below.

**Alternatives considered.** `loss="cauchy"` or `"soft_l1"`; a Cauchy/Tukey kernel, or ORB-SLAM's chi-square-based Huber threshold; a convergence-based stop instead of `max_nfev`; two-pass BA (Huber, then drop residuals beyond the threshold and re-solve without robust loss, as ORB-SLAM does); checking `sol.cost` against the initial cost before accepting.

**Why this choice.** Decision 2.7 sets 2 px as the arm's pixel scale on the evidence that "~65% of consecutive-frame tracks fall within 2 px of GT (corr benchmark)" and "1 px is at the LK noise floor", and decisions 2.11 and 3.4 deliberately reuse it rather than adding a parameter. Huber is the standard robust kernel for BA. `max_nfev=30` was a runtime guard; even so, sweep 1 measured 104 s/scene (window 5) and 189 s/scene (window 10) against 23 s without BA. With ATE going 122.9 → 128.8 / 130.7 mm, decision 3.4 rejected the block, so none of these knobs were tuned further.

### Lines 245-251: write the solution back in place

```python
245:     Ts, P = unpack(sol.x)
246:     for k in free:
247:         kf_T_w2c[k] = Ts[k]
248:     for m in pts:
249:         mp_xyz[m] = P[pidx[m]]
250: 
251: 
```

**What it does.** Unpacks the optimised vector and overwrites, in place, the w2c 4x4 of every **free** keyframe in the caller's `kf_T` list and the XYZ of every window map point in the caller's `mp_xyz` dict. The fixed keyframe is untouched. Nothing is returned. Lines 250-251 are the two blank lines that close the block before the `# --- the pipeline` banner (line 252). Two consequences of the write-back that a reader must know:

1. **The emitted trajectory is not updated.** `poses_c2w` is written at eight sites in `run_scene`: line 339 (frame 0 = identity), lines 402 and 406 (the retro-fill of pre-bootstrap frames against the bootstrap map — audit item 1, accepted with a footnote — which re-poses frames that were already passed), lines 415, 421, 434 and 455 (the frame being processed: pre-bootstrap hold, bootstrap frame, PnP-failure hold, PnP success), and line 477 (a save-time backfill of any still-`None` entry with the last written pose). Apart from the retro-fill and the backfill, each frame's pose is written once when it is processed; nothing reads `kf_T` back into `poses_c2w`, so BA never alters a written pose. BA's refined keyframe poses influence the future only — through the refined map points that subsequent PnP frames register to, and through `T_prev = kf_T[-1]` (line 319), the pose used to triangulate the next keyframe's points. From the output's point of view the block is therefore **causal** (no already-written pose changes), but the map and the written trajectory can disagree by the BA correction.
2. **Acceptance is unconditional.** If the solver returns an `x` with higher cost (e.g. hits `max_nfev` mid-step, or a degenerate point pulls the window), it is still written back; the only safety net is the caller's `except Exception` (lines 333-334), which catches crashes (and records them as a `ba_error:<ExceptionName>` diag event), not bad solutions.

**Alternatives considered.** Return the refined poses/points and let the caller decide; rewrite `poses_c2w` for keyframe frames (and re-propagate the non-keyframe frames between them) so the trajectory benefits directly; accept only if `sol.cost` decreased; motion-only BA of the *current* frame after every PnP so the written pose is the refined one.

**Why this choice.** In-place mutation matches the rest of `run_scene`'s state handling (`tracks`, `mp_xyz`, `kf_T` are all mutated in place). That the trajectory itself is never rewritten is consistent with the harness requirement that "every frame must get a pose" and with the causal reading of the arm — but it also means the sweep-1 ATE numbers (128.8 / 130.7 vs 122.9) measure BA acting only through the map, which is a limitation to state, not a tuned design.

### Known limitations / honesty notes

- **Rejected, not frozen.** Decision 3.4: refinement = none for v0. The only measurement is sweep 1 (2026-09-05, Slurm 45453490): window 5 → ATE 128.8 / RPE-t 8.49 / RPE-r 1.068, failed 32.5 %, 20 reboots, median inliers 78, 104 s/scene; window 10 → 130.7 / 9.04 / 1.175, 27.7 %, 20, 81, 189 s/scene; baseline `vo_v0` 122.9 / 9.65 / 1.129, 37.1 %, 23, 63, 23 s/scene. BA improved local metrics (RPE-t both windows, RPE-r at window 5, fewer failed frames, higher median inliers) but worsened ATE and cost 5-9x runtime. It was measured on the *pre-cull, pre-lever, pre-hand-off* v0 (sweep 1 tested each switch singly against defaults) and never re-run on the v0 FINAL configuration (decisions 2.14, 1.2 ON, 3.2b, 3.1 = 30, 2.15).
- **Single K.** The caller passes `Ks[0]` (line 280, "kept for the (unused in v0) local BA"). When BA was measured (2026-09-05) the focal was a per-scene constant, so this was exact; under the revised decision 0.1b (per-frame inference-time focal, 2026-09-06) re-enabling BA would project all window keyframes with frame 0's focal while PnP and triangulation use per-frame K. Would need to be fixed before any re-measurement.
- **Scale gauge is open.** Only one keyframe is fixed; a monocular window has a 7th gauge (scale) that nothing pins. `x_scale="jac"` and the `max_nfev=30` cap keep the solver from wandering along it, but it is not constrained by design.
- **Underdetermined points.** No minimum-observations filter: a point seen by a single window keyframe has 2 residuals for 3 unknowns.
- **No cheirality in the residual.** Only `|z| ≤ 1e-9` is guarded; points that go behind a camera are not dropped (contrast decisions 2.9/2.10/2.11, which all test cheirality).
- **Absolute-rotation chart.** The rotation parameter is the absolute w2c rotation vector; its singularity at angle π from the segment's world frame is not excluded by the small inter-keyframe motion.
- **Unconditional write-back.** `sol.success` / cost decrease are never checked.
- **Trajectory not refined.** Refined keyframe poses reach the map and the next triangulation only; `poses_c2w` is never rewritten, so the scored trajectory and the map can disagree. This keeps the block causal (relevant to the audit's causality items, e.g. item 1 on the retroactive pre-bootstrap fill, which this block does *not* touch) but also means the measurement above is of "BA through the map" only.
- **Runtime.** 104-189 s/scene vs 23 s, the second ground for rejection. The per-observation Python loops and finite-difference Jacobians are the obvious cost centres, but the doc records only the outcome, not a profile; an analytic Jacobian (camera block from `cv2.projectPoints`, point block derived by hand) is the untried remedy — never attempted because the ATE result did not justify it.
- **Tuning caveat (audit item 7).** Like every other threshold in the arm, the BA settings and the window sizes 5/10 were evaluated on the 12 reported smoke scenes; the audit's mitigation (report the frozen configuration on all 4292 scenes) does not apply here because the block is off in the frozen configuration.
