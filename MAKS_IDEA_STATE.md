# maks_idea — state of the branch

Read this first. It is the entry point for picking the work up cold.
Last updated 2026-09-13.

The branch started as a pilot ("skip the frames that hurt the metrics") and turned into
two things: a **negative result that closed that idea**, and a **working causal correction**
that came out of understanding why it failed.

---

## 1. Where the work stands

| | |
|---|---|
| Headline result | **-9.0% ATE on all 4292 DROID wrist test scenes**, training-free, causal, one forward pass, 8.7 us/frame |
| Paper-level write-up | [`CAUSAL_POSE_RECALIBRATION.md`](CAUSAL_POSE_RECALIBRATION.md) — 9 sections, every number measured |
| Original pilot plan + verdict | [`HARMFUL_FRAME_SKIP_PLAN.md`](HARMFUL_FRAME_SKIP_PLAN.md) |
| Master table | `$OUT/master/master_4292.{png,pdf,tex,csv}` — 20 rows; `./rebuild.sh` to regenerate |
| Venue assessment | Not a CVPR submission as it stands; see §6 |

`$OUT` = `/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval` throughout.

### The numbers to quote

All 4292 scenes, official evaluator (`eval_bundle/bin/eval_depth_poses.py`, published defaults).

| Method | AbsRel | δ<1.25 | ATE | RPE trans | RPE rot |
|---|---|---|---|---|---|
| CUT3R finetuned (`augfull_lr1e5`) | 0.1794 | 0.7863 | 0.0759 | 0.00794 | 1.1010 |
| **`causal_posefix_tc`** | 0.1794 | 0.7863 | **0.0690** | **0.00713** | 1.1010 |
| **`causal_posefix_tc_ema`** | 0.1794 | 0.7863 | 0.0702 | **0.00510** | **0.7691** |

**Cite the `_tc` rows.** They calibrate on the train split. The older `causal_posefix` /
`causal_posefix_ema` rows calibrated on 100 *test* scenes and are retained only for the
comparison; `_tc` is statistically indistinguishable on ATE and `_tc_ema` is significantly
better. Depth is symlinked through, so AbsRel/δ<1.25 are bit-identical to the baseline by
construction — that identity is a built-in correctness check on every sweep.

`_tc_ema` is the **best non-oracle row in the master table on both RPE columns**, in one
causal pass. On ATE it is 5th (behind Scal3R 0.0490, RayMap3R 0.0600, TTT3R 0.0632, and our
own fwd/bwd fusion 0.0641 which needs two passes and future frames).

---

## 2. The method in six lines

Post-processes the pose stream CUT3R already emitted. Never touches the model or depth.

```
T_t  = P_{t-1}^{-1} P_t          # increment; its translation is in the BODY frame
tau  = (tau - b) * 1/(1 + t/200) # subtract bias, damp late steps
P'_t = P'_{t-1} @ [R_t | tau]    # re-integrate
```

**The bias is subtracted from the increment, never from the absolute pose.** Subtracting a
constant from absolute positions is an *exact Sim3 no-op* — the evaluator's alignment
absorbs it perfectly. In the body frame it accumulates as `sum_t R_{t-1} b`, which is
path-dependent because the camera rotates. That is the whole reason it does anything.

`b` is the only fitted quantity. Optional causal EMA on the increments (world-frame
translation a=0.25, rotation-vector a=0.40) trades ~2 points of ATE for a third off both RPEs.

Implementation: `eval_pipeline/maks_posefix_sweep.py`.

---

## 3. Why it works, and the three things that confirm it

The mechanism is **progressive scale inflation**: the model over-predicts its own step size
and the over-prediction grows through the sequence. A similarity alignment removes a
*constant* scale but not a *drifting* one, so this component survives evaluation as
trajectory-shape error, which is what ATE measures.

Three independent confirmations, each from a different direction:

1. **Generality.** Re-fitted on `cut3r_zeroshot` (never trained on DROID, 6x worse depth):
   **-9.0% ATE, identical to three significant figures**, with a bias of twice the norm and
   a different sign pattern. The relative gain is invariant; the fitted object is not.
2. **Anti-stacking.** On `augfull_cg_fuse_g7` it makes ATE **5.4% worse**. Averaging a
   forward and a backward pass already cancels monotone drift — the bias refits 32% smaller
   — so the shrinkage over-corrects. Failure in exactly the predicted place.
3. **Per-scene regressions.** It hurts on ~34% of scenes, and those are the scenes with
   already-tiny ATE where there was no drift to remove. Visible directly in the
   `regression_*` point-cloud video, where the corrected trajectory comes out too short.

### Why cutting RPE does not cut ATE (asked and measured)

ATE accumulates the **vector sum** of step errors, not their size. Measured over 200 scenes:

| | RMS step error | coherence C | ATE |
|---|---|---|---|
| baseline | 0.00778 | **1.25** | 0.0717 |
| + EMA | 0.00521 (-33%) | **1.75** | 0.0697 (-2.8%) |

where `C = ||sum s_t|| / sqrt(sum ||s_t||^2)`; 1.0 = independent errors that cancel,
sqrt(N)=16.4 = perfectly aligned. **C is 1.25 out of 16.4** — the per-step errors are nearly
incoherent and cancel in the integral, so deleting a third of them barely moves ATE. The EMA
additionally *raises* coherence to 1.75 because smoothing lag is a systematic accumulating
error. That is why the two operating points trade rather than one dominating.

---

## 4. What is dead. Do not re-open without new evidence.

- **Frame blocking / skipping, every variant.** Suppressing the state+memory write of
  "harmful" frames loses on all five metrics, whether keyed on GT error, model confidence,
  or window detection. Killed at 1, 100 and 430 scenes with paired bootstrap. Root cause:
  absolute-scale metrics make fast-motion frames look bad, but **per unit of motion the
  model is relatively best there** — the flagged frames are victims, not culprits.
- **Pre-window blocking.** Neutral overall on 100 scenes. On the single demo scene a
  *shorter* pre-block (10 frames) beats a length-matched one (12/20/20), consistent with
  long contiguous blocks compounding state staleness — but that is one scene, no statistics.
- **Image appearance as an explanation.** Sharpness, texture, entropy, brightness,
  saturation, exposure: all inert (|d| <= 0.24, most < 0.07). Blur does not explain declines.
- **Magnitude calibration beyond one constant.** Only 2.6-3.4% of the variance of the ideal
  per-frame log-factor is explainable by any function of predicted magnitude.
- **Degenerate GT channels.** `outlier_mask` is byte-identical to `depth<=0` on every frame
  checked; `sky_mask` is identically zero (indoor wrist camera).

---

## 5. The input-property study (second deliverable)

`eval_pipeline/maks_window_properties.py` + `_figure.py`, over 100 scenes / 367 decline
windows / 30,808 frames. 18 properties, all from GT poses, GT depth and pixels — **no model
output enters any feature**, so nothing is circular.

Outputs in `$OUT/maks_windows100/properties/`:

| file | what |
|---|---|
| `window_properties_effects.png` | which frames become decline windows (Cohen's d, 95% CI) |
| `window_properties_affects.png` | what each property affects (pose vs depth) |
| `window_properties.md` | exact values + the rho interpretation section |
| `window_properties.json`, `per_frame_properties.npz` | machine-readable |
| `window_properties.png` | **superseded** by the two split figures; kept only for reference |

Findings:

- **Pose error and depth error have disjoint causes.** Motion and overlap drive the RPE
  metrics only (+0.27..+0.55); scene geometry drives AbsRel/δ<1.25 only (+0.33..+0.43).
  Essentially no crossover.
- **Largest single effect: overlap with the previous frame, d = -0.72.** Bigger than raw
  speed. Speed matters because it destroys overlap, not on its own.
- Sideways motion is 30-40% more damaging than forward motion at matched magnitude
  (d +0.61 vs +0.44) — it sweeps content out of frame; forward motion merely rescales it.
- **Nothing per-frame predicts ATE** (largest |rho| 0.19). Confirms ATE is an accumulated
  quantity, from a completely independent measurement.
- Decline windows sit **late** in episodes (d = +0.43) — the drift signature again.

**Statistical convention, keep it:** bootstrap over **scenes**, never frames. Within-episode
lag-1 autocorrelation is 0.49-1.00, so a frame-level bootstrap invents precision. Pooled vs
within-scene rho was checked and agrees (gap <= 0.08), so pooled rho is not a Simpson's artifact.

---

## 6. Honest venue assessment

Not a CVPR submission as it stands. Blockers, in order:

1. **One dataset.** Everything is DROID wrist cameras — smooth robot-arm trajectories, one
   geometry. No other benchmark dataset exists on this cluster (checked). The EMA's gain is
   explained by the smoothness of that motion, which openly invites "does this hold on
   handheld video?". **TUM-RGBD and Sintel are public, small, and the cheapest way to find out.**
2. **One architecture.** Two CUT3R checkpoints is not model generality. Spann3R or a
   StreamVGGT-style model would separate "recurrent 3D models do this" from "CUT3R does this".
3. **5th on the headline metric.** The genuine wins are on both RPE columns.
4. **The functional form is provably uninformative** (§6.2 of the method doc): a random
   monotone-decreasing gain of matched RMS ties it exactly. The contribution is one scalar.
5. **TTT3R / RayMap3R numbers are imported**, not reproduced here (marked † in the table).

Better-fitting venues given the robot data: **RA-L / CoRL**, where wrist-camera specificity
and the causal/8.7us framing are features rather than limitations; or 3DV / WACV.

**Recommended next action: TUM-RGBD.** Cheap, and it is the experiment most likely to reveal
the work is narrower than it looks — better to learn that now than in a rebuttal.

---

## 7. Reproduction

```bash
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything

# the correction, train-split calibration, all 4292 scenes (2-stage: infer then sweep)
sbatch eval_pipeline/maks_posefix_traincalib.sbatch

# input-property analysis + both figures
sbatch eval_pipeline/maks_window_properties.sbatch
python eval_pipeline/maks_window_properties_figure.py \
  --json $OUT/maks_windows100/properties/window_properties.json \
  --npz  $OUT/maks_windows100/properties/per_frame_properties.npz \
  --manifest $OUT/maks_windows100/manifest.json \
  --out_dir $OUT/maks_windows100/properties

# 3D point-cloud stream videos (GT | CUT3R | corrected)
sbatch eval_pipeline/maks_pointcloud_stream.sbatch

# master table
cd $OUT/master && PATH=/apps/GPP/LATEX/20240430/bin/x86_64-linux:$PATH ./rebuild.sh
```

### Cluster gotchas that have bitten this work

- **Login nodes SIGKILL at 300s CPU.** Anything heavy (renders, flow, sweeps) must go
  through `sbatch`, or you get a truncated output file and a misleading exit 0 through a pipe.
- **pdflatex needs `PATH=/apps/GPP/LATEX/20240430/bin/x86_64-linux:$PATH`** — the system TeX
  lacks `standalone.cls` and has broken font maps.
- **`acc` partition requires >= 20 CPUs per GPU.** `--cpus-per-task=16 --gres=gpu:1` is rejected.
- **`acc_debug` allows ONE job per user**; use `acc_ehpc` for anything parallel.
- `eval_depth_poses.py` renumbers frames **positionally**, so subset scoring needs a kept-GT
  symlink tree (`build_kept_gt`). See the `eval-subset-positional-renumbering` memory note.
- No `node` on the cluster, so the dataviz palette validator cannot be executed here —
  the figures use documented pre-validated palette slots unchanged.

---

## 8. Key jobs (for provenance)

| job | what |
|---|---|
| 45671897 | `causal_posefix` / `_ema`, test-split calibration, 4292 scenes |
| 45679237 | generality (`zs_posefix`) + stacking (`fuse_posefix`, `_ema`), 4292 scenes |
| 45682586 | **`causal_posefix_tc` / `_tc_ema`, train-split calibration** — the canonical rows |
| 45703882 | point-cloud stream videos, 4 scenes |
| 45720678 | 10-frame pre-window blocking on the demo scene |
| 45720939 | 24-property input analysis over 100 scenes |

## 9. File map

```
CAUSAL_POSE_RECALIBRATION.md      the method paper. Start here for the science.
HARMFUL_FRAME_SKIP_PLAN.md        original pilot plan, metric conventions, §11 results
MAKS_IDEA_STATE.md                this file

eval_pipeline/
  maks_posefix_sweep.py           THE method. --calib_root/--calib_pred_base pick the split
  maks_posefix_traincalib.sbatch  2-stage train-split calibration driver
  maks_window_properties.py       18 input properties from GT + pixels
  maks_window_properties_figure.py  the two figures, scene-level bootstrap
  maks_render_pointcloud_stream.py  3D stream videos; --still for framing tuning
  find_decline_windows.py         multi-scale step detector over aggregate badness
  render_metrics_video.py         per-frame metrics overlay UI
  plot_metrics_over_time.py       five-panel metrics-over-frames plots
  maks_windows100_*.py            the 100-scene window pipeline
  maks_window_tables.py           LaTeX tables (3dp, top-3 colour or bold-best)
  maks_preblock10.sbatch          fixed-length pre-window blocking controls

modified in place:
  eval_pipeline/infer_and_eval_worker.py   --skip_frames_file/--control_json/--no_plots
  src/CUT3R/src/dust3r/model.py            update_alpha, token gate, confidence trigger
  src/CUT3R/demo.py                        --gt_root/--eval_script, evaluate_and_plot()
```
