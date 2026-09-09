# OpenCV visual-odometry pose backbone for the DROID wrist harness

This document explains, line by line, everything that was added to this repository to build,
benchmark and evaluate a classical (OpenCV-only) camera-pose estimator as a control arm next to
CUT3R, and then analyses the evaluation results on the full 4292-scene DROID wrist test set.

It is written for a reader who knows structure-from-motion and OpenCV but has never seen this
codebase. Every design choice is traced to the numbered decision in `OPENCV_VO_DESIGN.md`
(repo root), which is the decision log: for each of the 22 decisions it records the options
that were considered, the benchmark that was run to settle it, the numbers, and the ruling.
No number in this README is asserted that does not appear in that log or in the code.

## 1. What the backbone is

The backbone takes the same 320x192 frames CUT3R sees, tracks Shi-Tomasi corners with pyramidal
Lucas-Kanade, bootstraps a 3D map from an essential matrix once the camera has moved 5 px of
parallax, and then registers every frame to the map by PnP. New map points are triangulated
between consecutive keyframes (declared every 5 px of parallax), outlier points are culled,
and if PnP fails three frames in a row the map is rebuilt with its scale handed over from the
old one. The only thing it takes from CUT3R at frame t is CUT3R's own predicted focal length
for frame t (an inference-time, closed-loop input). For the harness's depth columns the paired
CUT3R model's depth is used unchanged, because the backbone predicts poses only.

Runtime: 29.6 ms per frame on one CPU core (8.7 s per scene on average), against roughly 70 ms
per frame for CUT3R on one H100, so it adds no GPU cost and can run alongside inference.

## 2. Files added to the repository

| Path | Role |
|---|---|
| `eval_pipeline/opencv_vo.py` | The pipeline. Every decided choice is hard-coded as a constant; every choice that was measured as a switch is a CLI flag. |
| `eval_pipeline/opencv_vo_eval.py` | Scores the backbone's output with the *same* per-scene harness script (`eval_bundle/bin/eval_depth_poses.py`) used for every model arm, in parallel. |
| `eval_pipeline/build_opencv_vo_table.py` | LaTeX/PDF/PNG table for the 12-scene smoke comparison (pose-only scorer). |
| `eval_pipeline/build_opencv_full_table.py` | LaTeX/PDF/PNG table for the full 4292-scene harness comparison (all five metrics). |
| `eval_pipeline/opencv_vo_probes/corr_bench.py` | Benchmark that settled the correspondence method (decision 1.5). |
| `eval_pipeline/opencv_vo_probes/pnp_bench.py` | Benchmark that settled the PnP solver variant (decision 2.4). |
| `eval_pipeline/opencv_vo_probes/tri_bench.py` | Benchmark that settled triangulation acceptance, keyframe trigger, essential-matrix threshold and the static-track lever (decisions 2.11, 2.12, 2.10, 1.2). |
| `eval_pipeline/opencv_vo_probes/e_diag.py` | Diagnostic that explained why two-view translation direction is ~30 degrees off for the classical arm *and* for CUT3R. |
| `eval_pipeline/opencv_vo_probes/*.sbatch` | The Slurm launchers used for those benchmarks and for the pipeline sweeps. |
| `eval_pipeline/opencv_vo_probes/README.md` | Index of the four probes: which decision each settled, and how to re-run them from this checkout. |
| `OPENCV_VO_DESIGN.md` | The decision log (options, benchmarks, numbers, rulings, audit). |
| `docs/opencv_vo/opencv_backbone_full4292.{tex,png}` | The final full-test-set table. |
| `docs/opencv_vo/opencv_vo_smoke12.{tex,png}` | The 12-scene smoke table. |
| `docs/opencv_vo/README.md` | This document (generated). |
| `docs/opencv_vo/sections/*.md` | The 18 source sections this document is assembled from; edit these, not `README.md`. |
| `docs/opencv_vo/assemble_readme.py` | Concatenates the sections into `README.md` with a generated table of contents. |

Nothing in the existing evaluation code was modified; the backbone plugs into the harness by
writing the same `camera/%06d.npz` files a model run writes.

## 3. How to run

Frozen v0 configuration (the configuration reported in the tables), on any scene list:

```bash
source eval_pipeline/mn5_paths.sh        # sets SCENES_ROOT, OUT, ...
python eval_pipeline/opencv_vo.py \
    --scenes_root "$SCENES_ROOT" --scene_list "$OUT/scene_list.txt" \
    --preds_root  "$OUT/augfull_lr1e5/preds" \
    --out_root    "$OUT" --label vo_full_ft \
    --cull outlier --reboot_after_fails 3 --min_inliers 30 --scale_handoff \
    --focal perframe:"$OUT/augfull_lr1e5/preds" \
    --depth_link "$OUT/augfull_lr1e5/preds" \
    --shard_id 0 --num_shards 48          # run 48 of these in parallel on one node
```

The static-track lever is on by default (`--no_lever` disables it). `--focal perframe:<preds>`
gives frame t exactly the focal CUT3R produced for frame t; `--focal causal:<preds>` gives it
the running median of the model's focals up to t; both are inference-time. `--focal 203` is a
fixed model-free focal; `--focal gt` reads the calibrated intrinsics and is a diagnostic only.
`--no_retro_fill` makes the pipeline strictly causal (see section 5).

Scoring with the harness and building the table:

```bash
python eval_pipeline/opencv_vo_eval.py --out_root "$OUT" --label vo_full_ft \
    --scenes_root "$SCENES_ROOT" --scene_list "$OUT/scene_list.txt" --workers 48
python eval_pipeline/build_opencv_full_table.py      # -> $OUT/tables/opencv_backbone_full4292.{tex,pdf,png}
```

Per-scene diagnostics land next to the predictions: `diag.csv` (per frame: live tracks,
map points, PnP inliers, failed flag, keyframe flag, bootstrap state, event) and
`summary.json` (failed frames, re-bootstraps, keyframes, median inliers, focal range, seconds).

## 4. Decision index

Every decision below is documented with options, benchmark and ruling in `OPENCV_VO_DESIGN.md`;
the line-by-line sections that follow cite these numbers wherever a line of code implements one.

| # | Decision | Ruling |
|---|---|---|
| 0.1 | Depth source / arms | Pure monocular map (closed loop, images only); no model depth, no GT |
| 0.1b | Intrinsics source | CUT3R's predicted focal for frame t (inference-time); per-scene median superseded; 203 px fixed = model-free row; GT = diagnostic |
| 0.2 | Reference frame | Local map, forced by 0.1 |
| 0.3 | Eval scope | 12-scene smoke list (shared with existing smoke tests) then the full 4292 |
| 0.4 | Failure reporting | Harness scores ATE / RPE-trans / RPE-rot; AbsRel and delta inherited from the paired model; per-frame diagnostics in a side file |
| 1.1 | Preprocessing | Grayscale conversion only |
| 1.2 | Masking | No image mask; static-track lever (drop tracks < 1 px on frames whose median flow >= 2 px), ON by default since 2026-09-05 |
| 1.3 | Detector | Shi-Tomasi, 1000 / 0.01 / 5 px |
| 1.4 | Spatial spread | None |
| 1.5 | Correspondence | Pyramidal LK, OpenCV defaults, forward-backward check < 1 px (benchmarked against ORB, SIFT, DIS, Farneback) |
| 1.6 | Track lifetime | Persistent tracks, top-up only at keyframes |
| 2.1 | Geometry | 2D-3D PnP against the map; essential matrix only inside the bootstrap |
| 2.2 / 2.3 | Direction / depth read | Dropped (meaningless with a map) |
| 2.4 | PnP variant | solvePnPRansac ITERATIVE + solvePnPRefineLM (all variants measured within 0.05 deg) |
| 2.5 | Initial guess | None; no fallback solver |
| 2.6 | Robust estimator | Plain RANSAC |
| 2.7 | PnP RANSAC threshold | 2 px (OpenCV default 8 px is 2.2 deg at this focal) |
| 2.8 | Confidence / iterations | 0.999 / 1000 |
| 2.9 | Bootstrap pair | Frame 0 vs first frame with >= 5 px median track displacement; cheirality guard; pre-bootstrap frames retro-filled (see audit) |
| 2.10 | Bootstrap geometry | findEssentialMat RANSAC at 0.5 px + recoverPose |
| 2.11 | Triangulation acceptance | Cheirality in both cameras + reprojection < 2 px in both views |
| 2.12 | Keyframe trigger | Median track displacement since last keyframe >= 5 px |
| 2.13 | New map points | Keyframe-to-keyframe only |
| 2.14 | Map culling | Drop a map point after 3 consecutive PnP-outlier hits |
| 2.15 | Scale propagation | Implicit within a segment; scale hand-off across re-bootstraps |
| 3.1 | Failure floor | 30 PnP inliers |
| 3.2 | Failure policy | Hold last pose and flag; re-bootstrap after 3 consecutive failures |
| 3.3 | Scale | As given by the map (Sim(3) scoring absorbs one global scale) |
| 3.4 | Refinement | None (windowed BA measured worse and 5-9x slower) |

## 5. Honesty audit and rulings

The pipeline was audited for anything that could flatter it. The full list is in the design
log ("Honesty audit"); the items that matter for the reported tables are:

1. **Retroactive fill of pre-bootstrap frames (non-causal).** Frames before the first map
   exists are re-posed by PnP against the map built at the bootstrap frame, which is a future
   frame. Ruling: accepted, with a footnote in the table. Measured cost of the non-causal step
   on the 12 smoke scenes: 0.2 mm ATE, 0.2 mm RPE-trans, 0.02 deg RPE-rot (`--no_retro_fill`
   gives the strictly causal version).
2. **Retro-filled frames accept a 4-inlier PnP** (live frames require 30) and are not counted
   as failed in the diagnostics. Disclosed; part of item 1.
3. **Thresholds were selected on the 12 smoke scenes**, which are part of the test set. The
   reported full-harness numbers were produced with the configuration frozen and every one of
   the 4292 scenes scored; the 12 tuning scenes are 0.3 percent of that set.
4. **The "model-free 203 px focal" row of the smoke table** used a value read from the
   dataset's calibrated intrinsics; it is model-free but not GT-free. It is not in the
   full-harness table.
5. **The finetuned CUT3R row is a causal single pass; the champion fusion row (smoke table
   only) is forward-times-backward.** Only the finetuned row is a like-for-like causal comparison.
6. **Depth columns of every "+ OpenCV" row are the paired model's own depth** by construction.
7. **`--focal gt` exists** and is used only for the explicitly labelled diagnostic row.

Everything else in the loop (tracking, keyframing, triangulation, PnP, culling, recovery,
scale hand-off, the per-frame focal) uses only past frames and the images.
