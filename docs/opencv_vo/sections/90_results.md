# Evaluation results

All numbers below come from the full DROID wrist test set, **all 4292 scenes**, scored by the
harness's own per-scene script (`eval_bundle/bin/eval_depth_poses.py`, default arguments: depth
median-scale-aligned per frame, pose Sim(3)-aligned and RMSE-reduced per scene) and averaged over
scenes with `nanmean`, exactly as `eval_pipeline/aggregate_results.py` does for every model arm.
The OpenCV rows were scored by `eval_pipeline/opencv_vo_eval.py`, which runs the *same* script,
one subprocess per scene. No row uses a different scorer, a different scene list, or a different
reduction. The table lives at `docs/opencv_vo/opencv_backbone_full4292.{tex,png}` and is rebuilt by
`eval_pipeline/build_opencv_full_table.py`.

## 1. The five rows

| Method | AbsRel ↓ | δ<1.25 ↑ | ATE (m) ↓ | RPE-trans (m) ↓ | RPE-rot (°) ↓ |
|---|---|---|---|---|---|
| CUT3R zero-shot (pretrained ckpt) | 0.4786 | 0.5580 | 0.1197 | 0.0120 | 1.7132 |
| CUT3R finetuned (`augfull_lr1e5`) | **0.1794** | **0.7863** | **0.0759** | **0.0079** | **1.1010** |
| CUT3R zero-shot + OpenCV backbone | 0.4786 | 0.5580 | 0.1203 | 0.0124 | 1.3011 |
| CUT3R Finetuned + OpenCV backbone | **0.1794** | **0.7863** | 0.1043 | 0.0084 | 1.1141 |
| CUT3R Finetuned + OpenCV backbone, GT intrinsics | **0.1794** | **0.7863** | 0.1026 | **0.0080** | 1.1018 |

The backbone at frame *t* uses only the images and CUT3R's predicted focal for frame *t*
(`--focal perframe:<preds>`, decision 0.1b); the last row substitutes the calibrated intrinsics and
is a **privileged diagnostic**, not a deployable configuration.

## 2. Every metric, every pairing, with the exact difference

### 2.0 Verification of the table (re-run 2026-09-09)

Before interpreting anything, the table was re-derived from scratch:

1. **Independent aggregation.** Every one of the 4292 per-scene
   `eval_depth_pose_metrics.csv` files was re-read and re-averaged for all five rows, without
   going through the table builder. All five rows reproduce to four decimals, and all five have
   n = 4292 scenes with no NaN scenes dropped.
2. **Hand-recomputed ground truth.** For one scene the Sim(3) alignment and per-frame ATE were
   recomputed directly from the raw `camera/*.npz` pose files and compared with what the harness
   wrote. They agree to five decimals for all five rows, which also confirms the harness's
   `MEAN` row holds the **RMSE** over frames, not the mean:

   | label | hand-computed ATE mean | hand-computed ATE RMSE | harness `ate` |
   |---|---|---|---|
   | finetuned | 0.03936 | 0.04688 | 0.04688 |
   | ft + OpenCV | 0.06281 | 0.07122 | 0.07122 |
   | ft + OpenCV GT-K | 0.07931 | 0.08315 | 0.08315 |
   | zs + OpenCV | 0.06462 | 0.06892 | 0.06892 |
   | zero-shot | 0.10971 | 0.11425 | 0.11425 |

3. **The OpenCV rows really contain OpenCV poses.** The per-frame camera files differ from the
   paired model's (frame-10 translation differs by 0.01–1.44 m across sample scenes), the
   md5 of every row's concatenated pose bytes is distinct, and **0 of 4292 scenes** share a
   pose metric value with their base model row.
4. **The depth directory of each OpenCV row is a symlink** to the paired model's `depth/`, which
   is why §2.1 is exactly identical rather than approximately so.

### 2.1 Depth: AbsRel and δ<1.25 — neutral by construction

| Pairing | AbsRel | δ<1.25 |
|---|---|---|
| zero-shot → zero-shot + OpenCV | 0.4786 → 0.4786 (+0.0000, +0.0%) | 0.5580 → 0.5580 (+0.0000, +0.0%) |
| finetuned → finetuned + OpenCV | 0.1794 → 0.1794 (+0.0000, +0.0%) | 0.7863 → 0.7863 (+0.0000, +0.0%) |

**Classification: neutral, exactly and by construction.** The backbone predicts camera poses only.
Its output directory symlinks the paired CUT3R run's `depth/` (the `--depth_link` flag), so the
depth metrics of a "+ OpenCV" row are bit-identical to its base model row on all 4292 scenes.
These two columns carry no information about the backbone and must not be read as a result.

### 2.2 Zero-shot CUT3R vs zero-shot + OpenCV backbone

| Metric | Base | + OpenCV | Absolute | Relative | Paired *t* over 4292 scenes | Verdict |
|---|---|---|---|---|---|---|
| ATE | 0.1197 m | 0.1203 m | +0.0006 m (+0.6 mm) | +0.5% | 0.9 (**not significant**) | **statistical tie** |
| RPE-trans | 0.0120 m | 0.0124 m | +0.0004 m (+0.4 mm) | +3.3% | 4.8 (significant) | **significant but negligible** (0.4 mm is 5% of the 7.9 mm no-motion floor) |
| RPE-rot | 1.7132° | 1.3011° | **−0.4121°** | **−24.1%** | −28.4 (significant) | **improvement** |

This is the pairing where the backbone earns its place. Replacing the pretrained network's pose
head with classical geometry, while still feeding the classical arm that same network's (poor)
per-frame focal, removes a quarter of the relative-rotation error. Section 4 shows why this
matters: 1.7132° is *worse than predicting no rotation at all* (floor 1.2105°), whereas the
backbone's 1.3011° is close to that floor. ATE and RPE-trans move by less than a millimetre and
sit at the metric floors for both rows, so neither is evidence either way.

Per-scene evidence (all 4292 scenes): the backbone's ATE is better on **2013 scenes (46.9%)** and
worse on 2279; the per-scene ATE ratio has median **1.015** with p10 0.631 and p90 1.478. So the
two arms trade wins almost evenly and the aggregate tie is a real tie, not two identical
trajectories: per-scene correlation is only 0.677 (ATE), 0.525 (RPE-trans), 0.336 (RPE-rot), and
the median absolute per-scene ATE difference is 0.0204 m. **Zero of 4292 scenes** have identical
pose metrics between the two rows.

### 2.3 Finetuned CUT3R vs finetuned + OpenCV backbone

| Metric | Base | + OpenCV | Absolute | Relative | Paired *t* over 4292 scenes | Verdict |
|---|---|---|---|---|---|---|
| ATE | 0.0759 m | 0.1043 m | **+0.0284 m (+28.4 mm)** | **+37.4%** | 44.8 (significant) | **regression** |
| RPE-trans | 0.0079 m | 0.0084 m | +0.0005 m (+0.5 mm) | +6.3% | 7.5 (significant) | **significant but negligible** (both rows sit at the no-motion floor) |
| RPE-rot | 1.1010° | 1.1141° | +0.0131° | +1.2% | 1.5 (**not significant**) | **statistical tie** |

Against the finetuned network the backbone loses decisively on global trajectory accuracy and ties
on both local metrics. The ATE regression is the headline: 37.4% worse, 28.4 mm in absolute terms.
Per-scene, the backbone is better on **1030 scenes (24.0%)** and worse on 3262; the ATE ratio has
median **1.377** (p10 0.732, p90 2.736), i.e. a typical scene is ~38% worse but the spread is wide
and a quarter of scenes still favour the classical arm. Correlations are 0.618 / 0.444 / 0.461 and
the median absolute per-scene ATE difference is 0.0319 m.

The mechanism behind the ATE gap is documented in the design log and is *not* the solver: it is
loss of a single global scale. Each time PnP fails for three consecutive frames the map is
rebuilt, and a rebuilt map has its own arbitrary scale. The scale hand-off (decision 2.15) bridges
part of that, and measurably helped on the smoke set (116.8 → 111.9 mm mean ATE, 117.4 → 97.7 mm
median), but every un-bridged rebuild leaves a scale seam that Sim(3) alignment, which fits **one**
global scale per trajectory, cannot repair.

### 2.4 Predicted focal vs ground-truth intrinsics (privileged diagnostic)

| Metric | Predicted K | GT K | Absolute | Relative | Verdict |
|---|---|---|---|---|---|
| ATE | 0.1043 m | 0.1026 m | −0.0017 m (−1.7 mm) | −1.6% | **neutral** |
| RPE-trans | 0.0084 m | 0.0080 m | −0.0004 m (−0.4 mm) | −4.8% | **neutral** |
| RPE-rot | 1.1141° | 1.1018° | −0.0123° | −1.1% | **neutral** |

Perfect calibration is worth 1.7 mm of ATE and a hundredth of a degree. This is the strongest
evidence that the closed-loop constraint costs nothing here: the finetuned model's predicted focal
(205–221 px across the smoke scenes) is already good enough that replacing it with the true value
changes nothing material. The design log's focal-sensitivity sweep found the same at larger
perturbations: a 2× focal error costs about 10 mm of ATE, because the map and PnP share the same
(wrong) camera and Sim(3) absorbs the resulting scale error.

### 2.5 One cross-comparison worth stating

The classical backbone driven by the **finetuned** model's focal beats the **zero-shot network**
on every pose metric: ATE 0.1197 → 0.1043 (−12.9%), RPE-trans 0.0120 → 0.0084 (−30.0%), RPE-rot
1.7132 → 1.1141 (−35.0%). Since the backbone contributes no learning of its own, this says the
pretrained network's pose head is the weak component, not the pose problem itself.

## 3. Summary of what the backbone changes

| | Improvement | Regression | Neutral |
|---|---|---|---|
| On zero-shot CUT3R | RPE-rot −24.1% | — | ATE +0.5%, RPE-trans +3.3%, both depth columns |
| On finetuned CUT3R | — | ATE +37.4% | RPE-rot +1.2%, RPE-trans +6.3%, both depth columns |

One improvement, one regression, everything else neutral. The backbone is a *rotation* fix for a
weak pose head and a *translation-scale* liability against a strong one.

## 4. Why ATE and RPE-trans look alike everywhere: the metric floors

Reading the table without knowing the floors invites the wrong conclusion. Two trivial
trajectories were pushed through the identical Sim(3) scorer on a 144-scene sample:

| Trajectory | ATE (m) | RPE-trans (m) | RPE-rot (°) |
|---|---|---|---|
| Constant pose (camera never moves) | 0.1685 | 0.0079 | 1.2068 |
| Constant velocity from the first GT step | 0.1248 | 0.0083 | 1.2150 |

(Computed on the same 4292 scenes with the same scorer. An earlier 144-scene sample gave
0.1710 / 0.0079 / 1.2105 and 0.1252 / 0.0082 / 1.2294; the full-set values above supersede it.)

Expressed as headroom below the constant-velocity baseline, the table reads:

| Row | ATE | Better than trivial |
|---|---|---|
| CUT3R zero-shot | 0.1197 | 4.1% |
| zero-shot + OpenCV | 0.1203 | 3.6% |
| CUT3R finetuned | 0.0759 | **39.2%** |
| finetuned + OpenCV | 0.1043 | 16.4% |
| finetuned + OpenCV, GT K | 0.1026 | 17.8% |

This is the single most useful way to read the ATE column: **the entire zero-shot pairing lives in
a 4%-wide band above a trivial baseline**, so a tie there means "both are nearly uninformative
about the global trajectory", not "both are good". The finetuned model is the only arm with real
headroom, and the backbone gives back well over half of it.

Ground truth moves 3.7 mm per frame at the median, 7.9 mm RMS per scene, and rotates 0.56° per
frame at the median. Three consequences:

1. **RPE-trans is saturated.** Predicting no motion at all scores 0.0079 m — which equals the
   finetuned model's 0.0079 m to four decimals and is within 6% of the backbone's 0.0084 m. The
   per-frame translation is smaller than every method's per-frame error, so this column measures
   the step size, not the method. It cannot separate arms and its near-identical values across the
   table are expected, not suspicious.
2. **Zero-shot rotation is below trivial.** At 1.7132° the pretrained network is 0.50° *worse*
   than predicting no rotation. The backbone's 1.3011° is 0.09° above that floor and the finetuned
   model's 1.1010° is 0.11° below it. Only the finetuned arms extract real rotation information.
3. **ATE has a ceiling of usefulness too.** Constant velocity from a single GT step scores 0.1252 m.
   Zero-shot (0.1197) and both zero-shot/OpenCV rows (0.1203) sit essentially at that floor; only
   the finetuned model (0.0759) and the finetuned+OpenCV rows (0.1043, 0.1026) beat it clearly.

**Therefore: ATE and RPE-rot are the informative columns of this table.** RPE-trans should be read
as "no method resolves per-frame translation at 320×192 and this frame rate", and the depth columns
are inherited.

## 4b. Why so many cells look unchanged — three distinct mechanisms

Two of the five columns are *exactly* identical between a model row and its "+ OpenCV" row, and
three more differ by less than a percent. That is suspicious on its face, so each case has a
separate, verified explanation. None of them is shared data: **0 of 4292 scenes** share a pose
metric value between a base row and its OpenCV row.

**Mechanism 1 — architectural identity (AbsRel, δ<1.25).** These are identical to every decimal on
every scene because the OpenCV row's `depth/` directory *is* the model's, via a filesystem symlink
(`--depth_link`). The backbone emits poses only, so there is nothing else it could report. This is
the only genuinely identical case, and it is identity by construction, not agreement.

**Mechanism 2 — symmetric disagreement across scenes (ATE against zero-shot).** "Cancellation"
here does not mean errors cancel inside a scene; it means that across scenes the wins and losses
are the same size and equally common, so the *average* difference is near zero while the *typical*
difference is not. The per-scene ATE difference (OpenCV minus model, mm, n = 4292):

| pairing | mean | std | SE of mean | p25 / p50 / p75 | reading |
|---|---|---|---|---|---|
| zero-shot | +0.53 | 36.90 | 0.56 (0.9 SE from 0) | −19.0 / +1.8 / +21.0 | wide, near-symmetric, centred on zero |
| finetuned | +28.42 | 41.57 | 0.63 (44.8 SE from 0) | +1.1 / +26.6 / +54.1 | same spread, shifted right |

The spread is the same in both pairings (~37–42 mm at one sigma): the two arms always produce very
different trajectories. What differs is the *centre*. Against zero-shot the difference distribution
straddles zero (biggest wins −173 mm, biggest losses +161 mm), so nothing systematic survives
averaging. Against the finetuned model the whole distribution is shifted right and only 24% of
scenes fall below zero, so the regression is real.

Per-scene the two arms disagree violently; in the mean the zero-shot pairing ties:

| pairing / metric | OpenCV better | worse | mean difference | median per-scene abs. difference | ratio |
|---|---|---|---|---|---|
| zero-shot, ATE | 2013 (46.9%) | 2279 | +0.00053 m | 0.0204 m | **39×** |
| zero-shot, RPE-trans | 2186 (50.9%) | 2106 | +0.00040 m | 0.0024 m | 6× |
| finetuned, RPE-rot | 2183 (50.9%) | 2109 | +0.01314° | 0.2256° | **17×** |

The typical scene moves 39× further than the average of all scenes moves. Summing signed
differences for zero-shot ATE: wins total −57.49 m and losses total +59.75 m, so 2.26 m of net
difference survives out of 117 m of gross movement. A paired *t* test over the 4292 scenes puts
zero-shot ATE at *t* = 0.9 and finetuned RPE-rot at *t* = 1.5, i.e. **statistically
indistinguishable from no change**. These cells are not "the same number twice"; they are two
different methods that are, on average over this test set, equally good.

**Mechanism 3 — floor saturation (RPE-trans everywhere).** Ground truth moves 3.7 mm per frame at
the median and 7.9 mm RMS per scene. A trajectory that never moves at all scores 0.0079 m — the
finetuned model's exact score. Every arm in the table lands between 1.00× and 1.57× that floor,
because none of them resolves a 3.7 mm inter-frame translation from 320×192 images. The column has
almost no dynamic range left, so all five rows are compressed into it. The differences that *are*
statistically significant there (*t* = 4.8 and 7.5) amount to 0.4 and 0.5 mm, which is 5–6% of the
floor: real, and meaningless.

**What this leaves.** Only two cells in the table carry a real, large effect: the backbone's
−24.1% RPE-rot on zero-shot CUT3R (*t* = −28.4, better on 79.4% of scenes) and its +37.4% ATE on
the finetuned model (*t* = 44.8, worse on 76.0% of scenes). Everything else is identity by
construction, a statistical tie, or a difference below the metric's resolution.

**Sanity check that the backbone is not degenerating.** If the OpenCV rows were quietly collapsing
to "no motion" they would score the 0.1710 m constant-pose floor, not 0.1043 m. Over the full run
the backbone holds the previous pose on 21.4% of frames and re-bootstraps 15071 times across
4292 scenes (median 2 per scene; 669 scenes need none, 1035 need five or more), with a median of
85 PnP inliers on the frames it does solve.

## 5. How the reported configuration was reached (smoke-set history)

Every threshold was chosen by benchmark on the 12-scene smoke list before the full run. The path,
with the numbers from the design log (12-scene means, pose-only Sim(3) scorer, ATE mm / RPE-t mm /
RPE-rot °):

| Step | Configuration | ATE | RPE-t | RPE-rot |
|---|---|---|---|---|
| Sweep 1 | v0 defaults | 122.9 | 9.65 | 1.129 |
| Sweep 1 | + outlier culling (2.14) | 123.4 | 8.18 | 1.113 |
| Sweep 1 | + static-track lever (1.2) | 116.4 | 7.93 | 1.009 |
| Sweep 2 | + reboot after 3 failures, inlier floor 30 | 116.8 | 7.07 | 0.861 |
| Sweep 3 | + scale hand-off (2.15) — **v0 final** | 111.9 | 6.93 | 0.871 |
| — | CUT3R finetuned, same 12 scenes | 72.3 | 6.54 | 0.960 |

Measured but rejected along the way: every-frame triangulation (126.2 / 10.40 / 1.402, worse on all
three, decision 2.13); local bundle adjustment at windows 5 and 10 (128.8 and 130.7 ATE, 5–9×
slower, decision 3.4); inlier floor 10 (RPE-rot 1.795, bad poses accepted) and 20 (0.923 vs 0.861 at
30, decision 3.1); PnP threshold 1 px (more re-bootstraps) and 3 px (worse on all three, decision
2.7 confirmed at 2 px).

Two ablations that answer specific fairness questions:

- **Focal source** (full pipeline, 12 scenes): finetuned per-frame focal 110.4 / 7.53 / 0.858;
  running median to *t* 113.8 / 6.85 / 0.844; fixed 203 px 109.1 / 7.04 / 0.869; calibrated focal
  116.0 / 6.91 / 0.885; OpenCV self-calibration 116.6 / 7.86 / 0.935 (and its per-scene estimates
  scattered 132–510 px against a true ~203 px, so classical self-calibration is not viable on this
  footage). The focal source is immaterial within noise — which is why the full-harness GT-intrinsics
  row in §2.4 moves so little.
- **Retro-fill** (the one retained non-causal step, §6): with 110.4 / 7.53 / 0.858, strictly causal
  (`--no_retro_fill`) 110.6 / 7.71 / 0.881. The non-causal step is worth 0.2 mm ATE and 0.02° and
  changes no ordering; it touched 267 of 4243 frames (6.3%) on the smoke set.

## 6. Honest limitations

1. **Thresholds were tuned on 12 scenes that are part of the test set.** The mitigation is that the
   reported table is the full 4292 scenes with the configuration frozen, so the tuning scenes are
   0.3% of the reported set — but the tuning was not done on a held-out split, and a stricter
   protocol would have used one.
2. **One non-causal step is retained.** Frames before each (re-)bootstrap succeeds are posed
   retroactively by PnP against the map built at the bootstrap frame — typically 2–16 frames per
   scene plus a few after each rebuild. Accepted by the user with the cost measured (§5) and
   footnoted in the table. `--no_retro_fill` gives the strictly causal variant. Those retro-filled
   frames also accept a 4-inlier PnP where live frames require 30, and are not counted as failed
   in the per-scene diagnostics, so the reported failed-frame counts are optimistic by that amount.
3. **The GT-intrinsics row is privileged** and labelled as such; it is a sensitivity probe, not a
   competitive entry.
4. **The comparison is causal-vs-causal only for the finetuned row.** The champion fusion arm
   (`augfull_cg_fuse_g7`, in the smoke table only) is a forward×backward two-pass method and is not
   a like-for-like comparison for a single-pass causal backbone.
5. **The backbone contributes nothing to depth.** Any table row combining it with a model inherits
   that model's depth exactly.

## 7. What would be needed to close the ATE gap

The design log's diagnostics point at three specific causes, in order of measured size:

1. **Scale seams from re-bootstraps.** Each rebuild starts a new arbitrary-scale segment. The
   hand-off bridges some of them; a proper fix is a pose-graph or windowed BA that estimates a
   per-segment scale jointly, which is decision 3.4 territory and was rejected for v0 only in its
   naive windowed form.
2. **Two-view translation direction is ~30° off at these baselines** — and this is *not* specific
   to the classical arm: the diagnostic measured the same 30–40° error for CUT3R's own relative
   poses. Synthetic correspondences with 1 px noise reduce it to 4–14°, so the cause is
   correspondence noise at 320×192, not the estimator. Higher-resolution frames or a learned
   matcher would attack this directly.
3. **Long-scene drift.** The 400–700-frame scenes reach 130–195 mm ATE while short scenes sit near
   30–90 mm. Loop closure or any global optimisation would address this, and neither exists in v0.

None of these is a defect of the OpenCV implementation as such; all three are the expected
limitations of a single-pass, map-based monocular VO with no global optimisation, quantified here
against a learned baseline on the same 1.26 million frames.
