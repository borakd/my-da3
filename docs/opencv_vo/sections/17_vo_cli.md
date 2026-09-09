## `eval_pipeline/opencv_vo.py` lines 493-540: `main()` — CLI, shard loop, skip-if-done, depth symlink, error handling

This block is the entry point of the OpenCV monocular VO control arm. It declares one `argparse` switch per *open* design decision (every *decided* choice is hard-coded as a module constant above, per the file docstring: "Every decided choice is hard-coded; every still-open decision is a CLI switch so the options can be scored head to head"), overrides the one module constant that was left tunable (`PNP_THRESH`, decision 2.7), slices the scene list into shards for CPU-parallel Slurm runs, and drives `run_scene()` (lines 253-490, which tracks one scene end to end and writes `camera/%06d.npz` + `diag.csv` + `summary.json`) once per scene. It sits at the very top of the classical pipeline: upstream is the DROID scene store (`$SCENES_ROOT/<scene>/dense/rgb/*.png`) and a CUT3R preds tree that supplies the focal (decision 0.1b); downstream is the unchanged harness scorer (`eval_pipeline/opencv_vo_eval.py` → `eval_depth_poses.py`, or `pose_sim3_both.py`), which reads `<out_root>/<label>/preds/<scene>/{camera,depth}` exactly as it does for a model arm (decision 0.4, "Plumbing and day-one checks"). Nothing here touches geometry; every numerical claim below comes from the switch's consumer inside `run_scene()` and the benchmark tables of `OPENCV_VO_DESIGN.md`.

Note on defaults: the argparse defaults are the sweep-1 baseline `vo_v0` **plus the static-track lever**, which was flipped to default ON on 2026-09-05 (decision 1.2; line 509). The design doc has no row for a bare run of the current code; the nearest is sweep 1's "1.2 static-track lever ON" row (116.4 mm ATE / 7.93 mm RPE-t / 1.009 deg RPE-r on the 12 smoke scenes), and the lever-OFF `vo_v0` row is 122.9 / 9.65 / 1.129. Either way it is *not* the frozen v0 final. The frozen configuration (111.9 / 6.93 / 0.871 retroactive-focal; 110.4 / 7.53 / 0.858 with the inference-time per-frame focal) is the explicit command line in the design doc's "v0 FINAL configuration" section: `--cull outlier --reboot_after_fails 3 --min_inliers 30 --scale_handoff --focal perframe:$OUT/augfull_lr1e5/preds`. Running the script with no switches does not reproduce the reported rows.

### Lines 493-494: function header and parser

```
493: def main():
494:     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
```

**What it does.** Builds the parser with the module docstring (lines 2-21) as its description, so `python eval_pipeline/opencv_vo.py -h` prints the whole pipeline summary (frontend, bootstrap, tracking, keyframe, output conventions) verbatim; `RawDescriptionHelpFormatter` preserves the docstring's line breaks and indentation instead of re-flowing them.

**Alternatives considered.** A YAML/JSON config file (as the CUT3R training arms use), or environment variables (as `cg_launch_full.sh` uses for the gate arms). Neither is in the design doc.

**Why this choice.** Follows directly from the doc's frame for the script: constants for decided items, switches for open ones. Using the docstring as `-h` text keeps the one-line design record and the CLI in the same file, so a reader can see which decision each switch belongs to without opening the design doc.

### Lines 495-499: required paths and the output layout

```
495:     ap.add_argument("--scenes_root", required=True)
496:     ap.add_argument("--scene_list", required=True)
497:     ap.add_argument("--preds_root", required=True, help="model preds root providing the per-scene focal (e.g. .../augfull_lr1e5/preds)")
498:     ap.add_argument("--out_root", required=True, help="harness out root; writes <out_root>/<label>/preds/<scene>/camera/")
499:     ap.add_argument("--label", required=True)
```

**What it does.**
- `--scenes_root`: the DROID store root; `run_scene()` reads `<scenes_root>/<scene>/dense/rgb/*.png` (line 255) and, only in the diagnostic `--focal gt` mode, `dense/cam/000000.npz` (line 99). The design doc records the store layout as `$SCENES_ROOT/<scene>/dense/{rgb,cam,depth,outlier_mask,sky_mask}` with 320x180 PNGs; `run_scene` re-applies the eval loader's cover geometry (uniform 1.067x scale + centre crop to 320x192, decision 1.1) so pixel units match the model arm's.
- `--scene_list`: a text file, one scene name per line (blank lines ignored at line 521). The smoke list is `eval_pipeline/cg_smoke_scenes_12.txt` (decision 0.3); the full harness is 4292 scenes.
- `--preds_root`: a CUT3R preds tree (`<preds_root>/<scene>/camera/*.npz`). With the default `--focal model`, `model_K()` takes the per-scene **median** of `intrinsics[0,0]` over those npz files (line 110) — the decision-0.1b "model-estimated focal" (~211-217 px at 320x192 for the finetuned checkpoint, vs calibrated ~190 px at 320x180). The switch is `required=True` even when `--focal` is a fixed number, `gt`, `perframe:`/`causal:` (which carry their own root), or `selfcal`: in those modes the path is passed through to `run_scene` and only consulted if `selfcal` finds fewer than 5 usable pairs and falls back to `model_K(..., "model", ...)` (line 264).
- `--out_root` / `--label`: outputs land at `<out_root>/<label>/preds/<scene>/camera/%06d.npz` (assembled at lines 523-525). This is the harness convention: `opencv_vo_eval.py` looks for `<out_root>/<label>/preds/<scene>/{camera,depth}` and writes `<out_root>/<label>/eval/<scene>/eval_depth_pose_metrics.csv`; `build_opencv_vo_table.py` / `build_opencv_full_table.py` index rows by `<label>`.

**Alternatives considered.** Reading the focal from the GT `cam/*.npz` (calibrated K), a single nominal dataset K, or self-calibration — the four options of decision 0.1b. Writing to a fresh directory layout instead of the harness's `preds/` layout.

**Why this choice.** Decision 0.1b: the closed-loop rule says model output is not privileged, so the model's own focal is the default; keeping the layout identical to a model arm is what lets `opencv_vo_eval.py` score the classical row with "the SAME harness eval as every model arm" (decision 0.4). The doc records no rationale for `required=True` on `--preds_root`; in practice every reported row pairs the classical arm with a checkpoint's preds (the frozen command passes `--preds_root $OUT/augfull_lr1e5/preds`), and the focal source was shown to be immaterial in the full pipeline (finetuned-model focal 111.9/6.93/0.871, fixed nominal 203 px 109.1/7.04/0.869, calibrated 116.0/6.91/0.885; a 2x focal error costs ~10 mm ATE and ~0.1 deg RPE-r).

### Lines 500-501: inherited depth, and the "open decisions" marker

```
500:     ap.add_argument("--depth_link", default="", help="if set, symlink <preds_root-like>/<scene>/depth into each scene (inherited depth)")
501:     # open decisions
```

**What it does.** `--depth_link` takes a preds root (same shape as `--preds_root`, e.g. `$OUT/augfull_lr1e5/preds`). It is consumed at lines 530-533: after a scene succeeds, `<out_scene>/depth` is made a symlink to `<depth_link>/<scene>/depth`. Empty string (the default) disables it. The comment on line 501 separates the plumbing switches above from the per-decision switches below.

**Alternatives considered.** Decision 0.1 lists the depth-arm options — GT depth (oracle), model depth, triangulated monocular map — but those concern the *pose solver's* 3D source, not the reported depth columns. For the depth columns themselves the doc considered scoring ATE/RPE only vs. carrying AbsRel/d1 from a paired model (decision 0.4).

**Why this choice.** Decision 0.4: "harness scores ATE / RPE-trans / RPE-rot only (AbsRel, d1 are INHERITED from augfull_lr1e5 depth via symlink; mark as inherited in tables)". The VO arm produces no depth, and `opencv_vo_eval.py` refuses a scene without both `camera/` and `depth/` (its `missing` status), so the symlink is what makes the harness run at all. The full-4292 table confirms the intended effect: depth columns are "identical by construction (symlinked)" between each model row and its OpenCV row (AbsRel 0.479 / d1 0.558 for zero-shot pairs, 0.179 / 0.786 for finetuned pairs). Honesty-audit item 10 ("depth columns would be the paired model's") records this as a pending ruling; the frozen-config section says "pass `--depth_link $OUT/augfull_lr1e5/preds` for the full harness".

### Lines 502-504: map growth and culling (decisions 2.13, 2.14)

```
502:     ap.add_argument("--new_points", choices=["kf", "every"], default="kf", help="2.13")
503:     ap.add_argument("--cull", choices=["none", "outlier"], default="none", help="2.14")
504:     ap.add_argument("--cull_hits", type=int, default=3)
```

**What it does.**
- `--new_points kf|every`: `kf` triangulates new map points only when a keyframe is declared (KF-to-KF, line 321); `every` additionally runs `triangulate_pair()` between the last keyframe and the current frame on every frame whose PnP succeeded, for still-unmapped tracks whose displacement since the keyframe is at least `PARALLAX_PX` = 5 px (lines 459-464).
- `--cull none|outlier`: with `outlier`, after each successful PnP every map point observed in that frame either resets its counter (inlier) or increments it (RANSAC outlier); a point whose counter reaches `--cull_hits` is dropped from `mp_xyz` and its track is discarded (lines 443-454).
- `--cull_hits`: the integer hit count for that rule; default 3.

**Alternatives considered.** 2.13: KF-to-KF only vs. every frame. 2.14: none; drop points unseen for N keyframes; drop after k PnP-outlier hits.

**Why this choice.** Sweep 1 (decision 2.13): every-frame 126.2 / 10.40 / 1.402 vs KF-only 122.9 / 9.65 / 1.129 (ATE mm / RPE-t mm / RPE-r deg), so `kf` is the decided value and is the default. Sweep 1 (decision 2.14): culling after 3 hits cut PnP failures 26% → 5.6% of frames, RPE-t 9.65 → 8.18, and halved runtime (23 → 13 s/scene) with ATE unchanged (122.9 → 123.4); it is DECIDED and part of the frozen command, but the argparse default remains `none` (the sweep-1 baseline). The value 3 for `--cull_hits` was not swept in the design doc; it is the value the decision was benchmarked with. "Unseen for N keyframes" was rejected implicitly: dead tracks drop out on their own because there is no descriptor-based re-association (decision 2.14 note).

### Lines 505-508: failure floor, failure policy, refinement (decisions 3.1, 3.2, 3.4)

```
505:     ap.add_argument("--min_inliers", type=int, default=20, help="3.1")
506:     ap.add_argument("--fail_policy", choices=["hold", "cv"], default="hold", help="3.2")
507:     ap.add_argument("--refine", choices=["none", "ba"], default="none", help="3.4")
508:     ap.add_argument("--ba_window", type=int, default=5)
```

**What it does.**
- `--min_inliers`: one integer reused as a floor at six places in `run_scene`, each testing a different quantity:
  - line 360: the number of live tracks needed before a bootstrap E is even attempted (`len(tracks) >= args.min_inliers and np.median(disp) >= PARALLAX_PX`);
  - line 369: the `recoverPose` cheirality count a bootstrap pair must reach (`npass`);
  - line 390: the number of accepted triangulations a bootstrap map must reach (`acc.sum()`);
  - line 417: pre-bootstrap, when live tracks (`len(tracks)`) fall below the floor — or `pre_hist` exceeds `MAX_TRACK_LEN_BEFORE_BOOT` = 400 frames — `new_bootstrap()` re-detects corners;
  - line 427: the PnP RANSAC inlier count below which `pnp()` returns `None` (a "failed frame");
  - line 437: post-bootstrap, after a PnP failure, when the number of tracks that still carry a map point (`has.sum()`, with `has = tracks.mp >= 0` at line 425) is below the floor, the map is declared starved and a re-bootstrap starts.
  Note that the retro-fill PnP at line 400 uses a hard-coded floor of 4, not this switch.
- `--fail_policy hold|cv`: on a PnP failure, `hold` copies the last world-to-camera pose (`last_T`); `cv` extrapolates one step of constant velocity, `T_w2c = (last_T @ inv(prev_T)) @ last_T` (lines 430-433). Both are in w2c convention and inverted to c2w at line 434.
- `--refine none|ba`: `ba` calls `local_ba()` (scipy `least_squares`, Huber loss with `f_scale=PNP_THRESH`) over the last `--ba_window` keyframes once at least 3 keyframes exist (lines 330-332). Note `local_ba` is called with `K = Ks[0]` (line 280, commented "kept for the (unused in v0) local BA"); with a per-frame focal (`--focal perframe:`/`causal:`) it does not see the per-frame K that PnP and triangulation use, which is one more reason the BA path is ablation-only.
- `--ba_window`: keyframe count for that window; default 5.

**Alternatives considered.** 3.1: inlier count 10 / 20 / 30, or a ratio. 3.2: hold last pose; constant velocity; drop frame. 3.4: none; sliding-window BA (scipy); pose graph.

**Why this choice.** Decision 3.1 chose **30**: sweeps 1-2 show 10 → RPE-r 1.795 ("bad PnP accepted"); 30 vs 20 gives 0.861 vs 0.923 RPE-r, 7.07 vs 7.34 RPE-t, ATE 116.8 vs 120.0. Sweep 3 re-checked on the final config: `min_inliers 20` = 116.3 / 7.28 / 0.927 with 9.2% failed frames vs 111.9 / 6.93 / 0.871 with 21.5% at 30 — the higher floor trades more flagged frames for better pose. The default here is still 20 (sweep-1 baseline); the frozen command passes `--min_inliers 30`. Decision 3.2 chose **hold + flag**: constant velocity scored 121.9 / 8.47 / 1.375 vs hold 122.9 / 9.65 / 1.129 — mixed, rotation worse — so `hold` is default and decided; "drop frame" is impossible because `eval_depth_poses.py` numbers frames by position and every frame must get a pose (dataset facts). Decision 3.4 chose **none**: local BA window 5 / 10 gave ATE 128.8 / 130.7 vs 122.9 and RPE-r 1.068 / 1.175 vs 1.129 at 5-9x runtime (104 s / 189 s vs 23 s per scene); `--refine ba` and `--ba_window` remain only so the ablation is reproducible.

### Lines 509-511: the static-track lever and the PnP threshold (decisions 1.2, 2.7)

```
509:     ap.add_argument("--lever", dest="lever", action="store_true", default=True, help="1.2 reject_static_tracks lever (ON by default since 2026-09-05)")
510:     ap.add_argument("--no_lever", dest="lever", action="store_false", help="disable the static-track lever")
511:     ap.add_argument("--pnp_thresh", type=float, default=2.0, help="2.7 PnP RANSAC reprojection threshold (px)")
```

**What it does.**
- `--lever` / `--no_lever` share `dest="lever"`. Because `default=True`, passing `--lever` is a no-op; `--no_lever` is the only flag that changes anything. When `args.lever` is true, `run_scene` (lines 346-351) drops tracks whose frame-to-frame displacement is < 1 px on frames whose median displacement (over the tracks that survived LK) is ≥ 2 px, *before* any geometry (E, PnP, triangulation). It never fires on low-parallax frames, so a paused camera is unaffected.
- `--pnp_thresh`: float in pixels of the 320x192 cover frame; becomes `PNP_THRESH` at line 519 and is passed as `reprojectionError=` to `cv2.solvePnPRansac` (line 187) — the maximum pixel distance between a projected map point and its tracked position for the correspondence to count as an inlier. The same value is the Huber `f_scale` of the optional local BA (line 244).

**Alternatives considered.** 1.2: no mask; fixed dataset polygon; per-scene temporal-variance (Otsu) mask; flow-based rejection; GT outlier mask; and the displacement lever itself, ON or OFF. 2.7: OpenCV's default `reprojectionError` of 8 px, or 1-3 px.

**Why this choice.** Decision 1.2: the gripper is image-static and "votes for zero motion in PnP" (dataset facts); masking options all failed the probe (GT outlier_mask does not cover the gripper; Otsu over-masks 58%; a fixed polygon fails because finger geometry differs per lab). The lever was added DEFAULT OFF on 2026-09-04, then sweeps 1-3 showed ON better on every metric (v0 final 111.9 / 6.93 / 0.871 ON vs 115.6 / 8.40 / 0.971 OFF; correspondence benchmark: parallax-gated gap-4 E-rotation error 2.14 → 1.70 deg). **The default was flipped to ON on 2026-09-05 by user decision**, which is why the switch pair is shaped as it is. Decision 2.7 chose **2 px** over OpenCV's 8 px default: 8 px ≈ 2.2 deg at f ≈ 210 and admits gripper tracks; 1 px is at the LK noise floor; ~65% of consecutive-frame tracks fall within 2 px of GT. Sweep 2 confirmed it stays 2 px, against the same configuration at 2 px (`cull + lever + rb3`: 120.0 / 7.34 / 0.923, 42 reboots): PnP 3 px 124.6 / 8.07 / 1.084 (worse everything), PnP 1 px 120.4 / 7.71 / 0.866 with reboots 42 → 58. The `--pnp_thresh` switch exists so that sweep is reproducible; RANSAC confidence/iterations (0.999 / 1000, decision 2.8) are *not* exposed.

### Lines 512-513: focal source and strict causality (decision 0.1b, audit item 1)

```
512:     ap.add_argument("--focal", default="model", help="0.1b: 'model' (per-scene median, default), a fixed focal in px at 320x192 (strict zero-shot), 'selfcal', 'gt' (diagnostic), 'perframe:<preds_root>' (the model's focal for frame t; inference-time), or 'causal:<preds_root>' (running median of the model's focals for frames 0..t; inference-time)")
513:     ap.add_argument("--no_retro_fill", action="store_true", help="strictly causal: do NOT re-pose pre-bootstrap frames against the bootstrap map (they keep the held anchor pose and stay flagged failed)")
```

**What it does.** `--focal` is a free string dispatched in `run_scene` (lines 262-279) and `model_K()`; in every mode the principal point is the image centre and pixels are square (`K = [[f,0,W/2],[0,f,H/2],[0,0,1]]` in the 320x192 frame):
- `model` (default): per-scene median of `<preds_root>/<scene>/camera/*.npz` `intrinsics[0,0]`; one K for all frames. **Retroactive** (uses all frames' focals).
- a float, e.g. `203`: fixed focal; one K for all frames; no model input.
- `selfcal`: `selfcal_focal()` (lines 61-91) sweeps candidate focals 100-500 px (coarse step 10, then fine step 2 around the best) and keeps the one with the most `findEssentialMat` inliers, summed over up to 40 candidate frame pairs at gap 6; falls back to `model` if fewer than 5 pairs qualify (line 80, then line 264). Each pair must keep ≥ 30 tracks after the forward-backward check, the parallax gate (median displacement ≥ 5 px) and the static lever (lines 70, 74, 76-78); scoring uses `findEssentialMat` at RANSAC 1 px / 0.999 (line 85). The docstring's Sampson-error tie-break is not implemented. The design doc's focal-sensitivity table describes this run as a "sweep over 30 parallax-gated pairs"; the code samples up to 40 — the two are not reconciled. Retroactive.
- `gt`: reads the calibrated `intrinsic` from `dense/cam/000000.npz` and rescales by the cover factor. **Reads GT; diagnostic only.**
- `perframe:<root>`: frame t uses the focal the model produced for frame t (`<root>/<scene>/camera/%06d.npz`); one K per frame, threaded through bootstrap E, KF-to-KF triangulation and PnP (but not the optional BA, see above). Inference-time.
- `causal:<root>`: running median of those focals over frames 0..t. Inference-time.

`--no_retro_fill` disables the loop at lines 394-406 that, once a bootstrap succeeds at frame f, re-poses frames anchor+1..f-1 by PnP against the fresh map (needs ≥ 6 mapped tracks, floor 4 inliers) and marks them not-failed. With the flag, those frames keep the held anchor pose and stay flagged `failed`.

**Alternatives considered.** Decision 0.1b's four sources (calibrated K, nominal K, model focal, self-calibration), the per-scene-median vs per-frame vs running-median variants of the model focal, and the zero-shot checkpoint's focal. For retro-fill: keep (non-causal) vs strictly causal.

**Why this choice.** Decision 0.1b, as **REVISED 2026-09-06**: the user's constraint is that "OpenCV at frame t may only use what CUT3R has produced up to t; retroactive per-scene medians are not an inference-time method", so `perframe:` is the reported configuration and `model` is kept only as the historical default. The numbers: finetuned per-frame 110.4 / 7.53 / 0.858 ≈ retroactive median 111.9 / 6.93 / 0.871; running median 113.8 / 6.85 / 0.844; zero-shot per-frame (207-819 px) 124.8 / 10.33 / 1.059 vs zero-shot running median 114.7 / 7.92 / 0.891. `--focal 203` is the strictly model-free row (109.1 / 7.04 / 0.869) — but audit item 4 notes that 203 px "came from GT calibration". `selfcal` is "not recommended": 116.6 / 7.86 / 0.935, with per-scene focals scattering 132-510 px against a true ~203. `gt` is audit item 6 and the "GT intrinsics" row of the full table (ATE 0.103 vs 0.104 m: "worth ~2 mm ATE"). `--no_retro_fill` is the ablation for audit item 1 (**ACCEPTED 2026-09-07 with a footnote** in `build_opencv_full_table.py`): strictly causal 110.6 / 7.7 / 0.881 vs 110.4 / 7.5 / 0.858, i.e. +0.2 mm ATE, +0.2 mm RPE-t, +0.02 deg RPE-r, "does not change any ordering". The retained default is therefore **non-causal by design**.

### Lines 514-516: scale hand-off, reboot rule, sharding (decisions 2.15, 3.2b)

```
514:     ap.add_argument("--scale_handoff", action="store_true", help="2.15: at a re-bootstrap keep the old tracks and match the new segment's scale to the old one")
515:     ap.add_argument("--reboot_after_fails", type=int, default=0, help="3.2b recovery: re-bootstrap after K consecutive PnP failures (0 = only when the map starves)")
516:     ap.add_argument("--shard_id", type=int, default=0); ap.add_argument("--num_shards", type=int, default=1)
```

**What it does.**
- `--scale_handoff`: at a re-bootstrap, `new_bootstrap()` keeps the old tracks alive instead of discarding them, stashing each track's old-scale world point in `old_mp` keyed by track id and then clearing the map assignments (`tracks.mp[:] = -1`) and resetting `kf_pos` (lines 302-305; without the flag, lines 307-308 wipe the tracks). When the new two-view map is triangulated, the old points are looked up by track id (lines 378-382): if ≥ 10 shared tracks have positive old and new depth in the anchor camera, their median depth ratio `sc` is taken, and if 0.2 < sc < 5 the new unit-norm `recoverPose` translation is rescaled by `sc` and the pair re-triangulated, so the new segment inherits the old segment's scale (lines 383-388; logged as `handoff:<sc>:<n>` in `diag.csv`).
- `--reboot_after_fails K`: with K > 0, K consecutive PnP failures trigger a re-bootstrap (line 437); with 0, a re-bootstrap happens only when the number of tracks that still carry a map point (`has.sum()`, line 425) drops below `--min_inliers` ("map starves"). Unmapped live tracks do not count toward that test.
- `--shard_id` / `--num_shards`: two switches on one line (the file's compact style); consumed at line 522 as a stride slice of the scene list.

**Alternatives considered.** 2.15: implicit scale via the map only, vs explicit hand-off across re-bootstraps. 3.2b: wait for starvation; reboot after 3 fails; reboot after 10 fails. Sharding: a single process, or a multiprocessing pool inside the script (as `opencv_vo_eval.py` does with `--workers`).

**Why this choice.** Decision 2.15 (sweep 3): hand-off took mean ATE 116.8 → 111.9 and median 117 → 98 (117.4 → 97.7 in the table), firing on 12 of 46 reboots; short scenes with several reboots gain most (124.6 → 64.0, 88.9 → 71.5). Sweep 2's reading explains why it matters: ATE is "set by the 30-58 re-bootstraps per 12 scenes, each starting a new arbitrary-scale segment that Sim(3) cannot repair". Decision 3.2b (sweep 2): reboot-after-3 cut failed frames 18 → 14% and RPE-r 1.11 → 1.01 vs waiting for starvation; rb10 was no better (122.6 / 7.92 / 1.020 vs rb3 119.9 / 7.99 / 1.009). Both are decided and in the frozen command; both default OFF here (sweep-1 baseline). Sharding by process is the harness pattern: the sweep launchers `eval_pipeline/opencv_vo_probes/{sweep,pf}.sbatch` start 12 shards per label (`--num_shards 12`), and the full 4292-scene rows ran on 48 cores in ~16 min per row (design doc, "FULL HARNESS"). `cv2.setNumThreads(1)` at line 34 makes one process = one core, so shards, not threads, provide the parallelism. The doc's plumbing note: run as a CPU job, because the login node has a 300 s per-process CPU cap.

### Lines 517-520: parse, and the global threshold override

```
517:     args = ap.parse_args()
518:     global PNP_THRESH
519:     PNP_THRESH = args.pnp_thresh
520: 
```

**What it does.** Parses `sys.argv`, then rebinds the module-level constant `PNP_THRESH` (declared at line 41 as 2.0, with the comment "PNP_THRESH may be overridden by --pnp_thresh") to the CLI value. `pnp()` (line 187) and `local_ba()` (line 244) read the module global at call time, so the assignment takes effect for every scene in this process. `PNP_CONF` (0.999) and `PNP_ITERS` (1000) are not overridable. Line 520 is a blank separator.

**Alternatives considered.** Threading the threshold through `args` into `pnp()` like every other switch; keeping it a pure constant with no switch.

**Why this choice.** Decision 2.7 was DECIDED at 2 px on 2026-09-04. The doc records sweep 2's 1 px / 3 px check (2.7 "stays 2 px"); the switch is what makes that check reproducible. The doc gives no reason for implementing it as a module-global override rather than a `pnp()` argument. The module default equals the decided value, so a run without `--pnp_thresh` is at 2 px. `TRI_REPROJ` (line 42, decision 2.11) is deliberately *not* exposed: it was set equal to the PnP threshold ("No new parameter (2 px = PnP threshold)"), and changing `--pnp_thresh` does not move it.

### Lines 521-523: scene list, shard slice, output root

```
521:     scenes = [l.strip() for l in open(args.scene_list) if l.strip()]
522:     scenes = scenes[args.shard_id::args.num_shards]
523:     out_preds = os.path.join(args.out_root, args.label, "preds")
```

**What it does.** Reads the scene list, dropping whitespace and empty lines; keeps every `num_shards`-th entry starting at `shard_id` (a stride slice, so shard k of N gets scenes k, k+N, k+2N, ...). Then forms the preds root `<out_root>/<label>/preds`. Nothing is created yet; `run_scene` creates `<out_scene>/camera/` at line 473.

**Alternatives considered.** Contiguous block slicing (`scenes[k*len//N:(k+1)*len//N]`); a work queue. The design doc does not discuss sharding at all.

**Why this choice.** Stride slicing is the same rule as the GPU eval worker's default mode (`infer_and_eval_worker.py` lines 146-147, `i % num_shards == shard_id`; in that worker's `--claim_dir` mode, lines 138-145, `shard_id` only sets a starting offset into a cooperative queue, which this script does not have). Not a doc finding, but the author's reading: a stride interleaves scene lengths (57 to ~1700 frames, dataset facts) across shards, whereas contiguous blocks of a sorted list could load one shard with all the long scenes. The `<out_root>/<label>/preds` layout is the harness convention described under lines 495-499.

### Lines 524-527: per-scene loop and skip-if-done

```
524:     for s in scenes:
525:         out_scene = os.path.join(out_preds, s)
526:         if os.path.isfile(os.path.join(out_scene, "summary.json")):
527:             continue
```

**What it does.** Scenes are processed sequentially within a shard. A scene is skipped when `<out_scene>/summary.json` already exists. `summary.json` is the last file `run_scene` writes (line 489, after all `camera/*.npz` at line 479 and `diag.csv` at lines 480-484), so its presence means the scene completed; a scene interrupted mid-write (partial `camera/`) has no summary and is redone from scratch on the next launch.

**Alternatives considered.** A separate sentinel file (the GPU workers use a skip sentinel, cf. `--ignore_skip_sentinel` in `run_fix_best.sh`); checking the `camera/` frame count against the rgb count; no resume at all.

**Why this choice.** Re-launch safety under Slurm: a shard can be killed by a job time limit or a node failure, and re-submitting the same command then resumes where it stopped. Using the summary as the completion marker needs no extra file and matches `opencv_vo_eval.py`, which likewise "skips scenes already scored". The cost is that **the skip is keyed on `<label>` only, not on the switches**: re-running under an existing label with different flags silently keeps the old scenes. Audit item 13 records "stale per-scene-median rows on disk"; a label-keyed skip is one way such staleness arises, so every configuration must get its own `--label` or the label directory must be removed first.

### Lines 528-529: run the scene

```
528:         try:
529:             summ = run_scene(os.path.join(args.scenes_root, s), os.path.join(args.preds_root, s), out_scene, args)
```

**What it does.** Opens the per-scene guard and calls `run_scene(scene_dir, preds_scene_dir, out_scene_dir, args)` — the full tracker for one scene, which loads and grayscales every frame, resolves the per-frame K, bootstraps, tracks, writes `camera/%06d.npz` (`pose` = c2w 4x4 float32, `intrinsics` = that frame's K) for **every** frame, `diag.csv`, and `summary.json` — and returns the summary dict (`scene, frames, failed_frames, reboots, keyframes, median_inliers, map_points, seconds, focal, focal_min, focal_max, selfcal_pairs`, lines 486-488). `preds_scene_dir` is `<preds_root>/<scene>` and is only read by the `model` focal mode and the `selfcal` fallback.

**Alternatives considered.** None recorded; this is the natural call.

**Why this choice.** Every frame must receive a pose because "eval_depth_poses.py numbers frames by position" (dataset facts); the `run_scene` output block guarantees that (held-pose fill at lines 474-478). The summary is what decision 0.4 asked for as side diagnostics ("failed-frame count and median inliers", "median inliers moves before ATE does") and what the sweep tables report as `failed%`, `reboots`, `KF/100f`, `med inl`, `s/scene`.

### Lines 530-534: depth symlink and the stdout record

```
530:             if args.depth_link:
531:                 dst = os.path.join(out_scene, "depth"); src = os.path.join(args.depth_link, s, "depth")
532:                 if not os.path.exists(dst) and os.path.isdir(src):
533:                     os.symlink(src, dst)
534:             print(json.dumps(summ), flush=True)
```

**What it does.** After a successful `run_scene`, and only if `--depth_link` was given: create `<out_scene>/depth` as a symlink whose target is `<depth_link>/<scene>/depth` (an absolute path if `--depth_link` was absolute — the frozen command uses `$OUT/...`, so it is). The link is created only if `dst` does not already exist (any existing file/dir/link wins) *and* the source directory exists; a missing source is silently skipped and the scene will later be reported as `missing` ("no camera/ or depth/ under preds") by `opencv_vo_eval.py`. Finally the summary dict is printed as one JSON line to stdout with `flush=True`; the launchers redirect stdout to a file (`$OUT/logs/vo_sweep_<label>_<shard>.log`, `vo_pf_...`), where Python would otherwise block-buffer it, so the flush is what makes per-scene progress visible in real time.

**Alternatives considered.** Copying the paired model's depth files into each label; pointing the scorer at a second root; scoring pose only with a modified evaluator.

**Why this choice.** Decision 0.4 and the frozen-config note ("Depth columns (AbsRel, d1) are inherited from augfull_lr1e5 ... pass `--depth_link`"). Symlinking keeps the harness scorer untouched and makes the inheritance auditable (the per-scene check in the "metric floors" section found depth columns "identical by construction (symlinked)" while 0 of 4292 scenes had identical *pose* metrics between a model row and its OpenCV row). The symlink placement inside the `try:` after `run_scene` means a scene that raised gets neither poses nor depth, so it cannot be half-scored. Consequence for housekeeping: the OpenCV label's depth columns are live references into the paired model run's `preds/`; deleting that run's preds breaks every OpenCV row that inherited from it (the same class of dependency `CLAUDE.md` warns about for `augfull_cg_fuse_g7`'s depth symlinks).

### Lines 535-537: error handling

```
535:         except Exception as e:
536:             print(f"ERROR {s}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
537: 
```

**What it does.** Any exception from `run_scene` is caught, reported on stderr as `ERROR <scene>: <ExcType>: <msg>`, and the loop moves to the next scene. Typical sources: `FileNotFoundError` from `model_K` when the preds tree lacks `camera/*.npz` (line 109) or from `np.load` of a missing per-frame npz in `perframe:`/`causal:` mode (line 273); or a `cv2.error` from an unguarded OpenCV call — `findEssentialMat`/`recoverPose` at lines 365-368 and `triangulatePoints` at line 170 have no try/except; only `pnp()` swallows its own `cv2.error` and returns `None` (lines 185-190, 194-197), and `local_ba` errors are caught inside `declare_keyframe` (lines 331-334). No `summary.json` is written for a scene that raised, so a re-launch retries it. Line 537 is the blank line closing the function.

**Alternatives considered.** Letting the exception propagate (kills the whole shard and loses the remaining scenes); retrying with a fallback solver.

**Why this choice.** Decision 2.5 rules out silent fallbacks *inside* the solver ("a failed PnP is logged as a failed frame, not retried with SQPnP ... a guess or a fallback would hide instability the diagnostics must expose"); at the process level the analogous rule is to make the failure loud (stderr) and leave the scene absent rather than fabricate output. Decision 2.9 anticipates the one legitimate whole-scene failure — "Scene where no pair ever passes = total failure, flagged" — but note that case does *not* raise: `run_scene` still writes held poses for every frame with `failed=1` on every `diag.csv` row after frame 0, so it is visible only as `failed_frames == frames - 1` in `summary.json` (frame 0 is recorded with `failed=0` at line 341 and is never counted), not on stderr.

### Lines 538-540: script guard

```
538: 
539: if __name__ == "__main__":
540:     main()
```

**What it does.** Standard entry guard: `main()` runs only when the file is executed as a script, so the module could be imported without running the CLI. Nothing in the repo currently imports it; the probe scripts under `eval_pipeline/opencv_vo_probes/` (`corr_bench.py`, `e_diag.py`, `pnp_bench.py`, `tri_bench.py`) are self-contained and carry their own OpenCV code. Line 538 is the blank line before it.

**Alternatives considered.** None; idiomatic.

**Why this choice.** Idiomatic Python; nothing in the design doc bears on it.

### Known limitations / honesty notes for this block

- **Defaults ≠ frozen config.** Five decided values (`--cull outlier`, `--min_inliers 30`, `--reboot_after_fails 3`, `--scale_handoff`, `--focal perframe:<root>`) are OFF/old in argparse; the switches whose default *is* the decided value are `--new_points kf`, `--fail_policy hold`, `--refine none`, `--lever` (ON) and `--pnp_thresh 2.0`. The reported rows are the design doc's "v0 FINAL configuration" command line, not `python opencv_vo.py` bare.
- **Retro-fill is non-causal and ON by default** (audit items 1, 2, 12; item 1 accepted 2026-09-07 with a table footnote). Pre-bootstrap frames are posed against a map built later, at a 4-inlier floor instead of 30, and are counted as not-failed. `--no_retro_fill` is the strictly causal ablation: +0.2 mm ATE, +0.2 mm RPE-t, +0.02 deg RPE-r.
- **`--focal gt` reads GT intrinsics** (audit item 6); `--focal 203` is model-free at run time but the 203 px value came from GT calibration (audit item 4). Only `perframe:`/`causal:` are inference-time; `model` and `selfcal` are retroactive over the whole scene. `selfcal`'s pair count (40 in code, "30" in the design doc's table) is unreconciled, and its docstring promises a Sampson tie-break the code does not implement.
- **Optional BA uses frame 0's K** (`K = Ks[0]`, line 280) even under a per-frame focal; the BA path is ablation-only (decision 3.4) so this does not affect any reported row.
- **Whole sequence in memory** (audit item 3): `run_scene` loads every grayscale frame before tracking; `main()` does nothing to stream.
- **Skip-if-done is keyed on `<label>` only.** Re-running a label with changed switches keeps stale scenes (cf. audit item 13). Use a fresh `--label` per configuration.
- **Depth columns are inherited, not produced** (decision 0.4; audit item 10, pending). The symlinks are live references into the paired model's `preds/`; a missing source directory is skipped silently and surfaces only as `missing` in `opencv_vo_eval.py`.
- **All thresholds were tuned on the 12 reported smoke scenes** (audit item 7), addressed by reporting the frozen configuration on all 4292 scenes; the CLI exposes those thresholds precisely so that tuning is reproducible, not because they are meant to be re-tuned per run.
- **Whole-scene bootstrap failure is not an error.** A scene where no parallax-gated pair ever passes writes held poses for every frame and is flagged only through `summary.json` (`failed_frames == frames - 1`) / `diag.csv`, never on stderr.
- **`--lever` is a no-op flag** (already the default); only `--no_lever` changes behaviour. `--preds_root` is required even in modes that never read it.
