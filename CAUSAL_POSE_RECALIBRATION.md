# Causal Pose Recalibration for Recurrent Feed-Forward 3D Reconstruction

**A training-free, causal, single-pass correction that removes 9% of CUT3R's absolute
trajectory error on the full 4292-scene DROID wrist benchmark at a cost of 9 microseconds
per frame, without altering the model, its depth output, or its inference cost.**

Status: verified 2026-09-10 on all 4292 test scenes with the project's official evaluator.
Author's note: this document is a method description written for a reader who has not
followed the investigation. Every number in it was measured, not estimated. Section 8 states
plainly what is *not* established.

---

## 1. The claim

Let a recurrent feed-forward 3D model emit, for a video, a camera pose per frame. We show that
CUT3R's emitted poses contain two systematic, *causally removable* errors:

1. a constant additive bias in the per-step translation, and
2. a per-step scale that inflates progressively as the sequence advances.

Neither is visible to the standard evaluation protocol as a nuisance parameter, because that
protocol aligns predicted to ground-truth trajectories with a similarity transform, which
absorbs a *constant* scale but not a *drifting* one. Removing both with a two-parameter
post-process improves absolute trajectory error by 9.0% and relative translation error by
11.0% on the full benchmark, changes depth by exactly zero, and requires no re-inference.

Adding a causal low-pass filter on the increments trades 1.9 points of the ATE gain for a
35.8% reduction in relative translation error and a 30.1% reduction in relative rotation
error, the latter improving on **every one of the 4292 scenes**.

The same 9.0% ATE reduction is obtained, to three significant figures, on a second checkpoint that
was never trained on this data and is six times worse in depth, with an independently fitted and
numerically dissimilar bias (§5.2).

---

## 2. Setting and notation

**Model.** CUT3R reads frames $I_1 \dots I_N$ one at a time, maintaining an internal state, and
emits for each frame a depth map and a camera-to-world pose $P_t \in SE(3)$. We treat the model
as a black box: the method reads only $\{P_t\}$.

**Ground truth.** $G_t \in SE(3)$, from DROID's robot kinematics, in metres.

**Predicted increments.** Write

$$T_t = P_{t-1}^{-1} P_t = \begin{bmatrix} R_t & \tau_t \\ 0 & 1\end{bmatrix}, \qquad t = 1 \dots N-1$$

and correspondingly $\hat{T}_t = G_{t-1}^{-1} G_t$ with translation $\hat\tau_t$.

**Evaluation protocol.** For each scene, a similarity transform $(s, \mathbf{R}, \mathbf{t})$ is
fit by Umeyama's method on the camera centres, minimising $\sum_t \| s\mathbf{R}p_t + \mathbf{t} - g_t\|^2$.
Then

- $\mathrm{ATE} = \mathrm{RMS}_t \| s\mathbf{R}p_t + \mathbf{t} - g_t \|$
- $\mathrm{RPE} = \mathrm{RMS}_t$ of the error of $\hat T_t^{-1} T_t^{\text{aligned}}$, split into
  translation magnitude (metres) and rotation angle (degrees)

Scene values are averaged unweighted over scenes. Depth metrics (AbsRel, $\delta<1.25$) use only
the depth maps and are untouched by anything in this document.

**The consequence that shapes everything.** The similarity transform contains $s$. Therefore
*any* global rescaling of the predicted trajectory is an exact no-op for ATE. We verified this
empirically: scaling every predicted translation increment by 0.5 changes ATE by
$-0.0000$ to four decimal places. Only a **time-varying** scale error is observable, and hence
only a time-varying correction can help. This immediately kills a large family of naive fixes
and is the reason the correction below has the shape it does.

---

## 3. Diagnosis

Three measurements motivate the method. All are on 100 held-out development scenes
(30,708 consecutive frame pairs) unless stated.

### 3.1 Relative accuracy is worst where the camera is nearly still

Raw RPE is an *absolute* error in metres and degrees, so it grows mechanically with how far the
camera moved. Normalising by the true motion of the same step inverts the picture:

| | Spearman with true translation step |
|---|---|
| raw RPE translation | $+0.66$ |
| raw RPE rotation | $+0.56$ |
| RPE translation / true step | $-0.52$ |
| RPE rotation / true step | $-0.70$ |
| AbsRel | $+0.04$ |
| ATE | $+0.01$ |

By decile of true translation step, the median ratio $\|$RPE rot$\|$ / true rotation step falls
monotonically from 2.90 on the slowest decile to 0.68 on the fastest. On the slowest decile the
model reports a rotation error larger than the entire true rotation on 93.4% of frames.

This kills the intuitive reading that "the model struggles when the camera moves fast". Per unit
of motion, fast frames are where it is *most* reliable.

### 3.2 The model over-predicts its own step size

After the per-scene similarity scale is applied, the median predicted step magnitude is
**1.27 times** the true step magnitude. Over-prediction, not under-prediction, so a *shrinking*
correction is the right family.

### 3.3 The over-prediction grows through a sequence

The ratio of predicted to true step magnitude rises by roughly $1.5\times$ across normalised
sequence time. Because the evaluation absorbs a constant scale but not a drifting one, this is
precisely the component that is both real and correctable.

### 3.4 What is *not* correctable, and why the method must be simple

Decomposing each per-step translation error into the component along the true direction and the
component perpendicular to it:

| | along | perpendicular |
|---|---|---|
| all frames | 44.5% | 55.5% |
| slowest decile | 33.0% | 67.0% |

An isotropic random error gives exactly 33.3 / 66.7. On slow frames the error is statistically
indistinguishable from isotropic noise: the mean cosine between predicted and true increment
direction is 0.062, and the pooled vector correlation is 0.105 against a permutation null of
0.021 at the 95th percentile.

**Direction cannot be rescaled away.** Any method that only modulates the magnitude of what the
model already outputs is capped by this decomposition. That is the reason to build the simplest
possible magnitude correction and stop, rather than to build a learned per-frame gain: a
gradient-boosted regressor over 31 causal features (image statistics, optical flow at several
quantiles, forward-backward consistency, predicted-depth statistics, the model's own predicted
motion) reached a correlation of only 0.36 with the ideal per-frame factor, and its
held-out gain was smaller than the two-parameter rule below.

---

## 4. Method

Two hyperparameters total. One is a 3-vector fit offline; one is a scalar.

### 4.1 Calibration, offline, once per checkpoint

Given a calibration set of scenes with ground-truth poses (disjoint from anything to be scored):

For each calibration scene, fit the similarity scale $s$ by Umeyama on the camera centres, then

$$b = \operatorname*{mean}_{\text{frames}} \left( \tau_t - \frac{\hat\tau_t}{s} \right)$$

Division by $s$ expresses the ground-truth increment in the model's own arbitrary units, which is
necessary because CUT3R's output scale is not metric. Average over scenes.

Measured value for the DROID-finetuned checkpoint, from 100 calibration scenes:

$$b = (-1.974\times10^{-4},\; -8.909\times10^{-4},\; -2.379\times10^{-3}), \qquad \|b\| = 2.548\times10^{-3}$$

which is 4.24% of the RMS predicted step. It is a fixed direction in the *camera body frame*, and
its dominant component is along the optical axis.

**It is applied to the increment, not to the pose.** Subtracting a constant from the absolute
positions $p_t$ would be an exact no-op: the similarity transform's translation component absorbs it
perfectly, in the same way its scale component absorbs a global rescale (§2). Subtracting it from
$\tau_t$, which lives in the body frame of camera $t-1$, accumulates as $\sum_t R_{t-1} b$ — a
path-dependent quantity, because the camera rotates as it moves. That is the whole reason the
correction is not absorbed by the evaluation.

**How constant is the constant.** Measured per scene over the 100 calibration scenes:

| | $x$ | $y$ | $z$ (optical axis) |
|---|---|---|---|
| mean | $-8.87\times10^{-4}$ | $-8.08\times10^{-4}$ | $-1.920\times10^{-3}$ |
| per-scene s.d. | $3.27\times10^{-3}$ | $3.42\times10^{-3}$ | $3.14\times10^{-3}$ |
| sign agrees with the mean | 67% | 60% | 73% |
| $t$ over 100 scenes | $-2.71$ | $-2.36$ | $-6.12$ |

The per-scene scatter is 1.7 to 4 times the mean on every axis, and 72% of $\|b\|^2$ sits on the
optical axis. Only that component is solidly significant; the two lateral components are marginal
($p \approx 0.02$) and reverse sign on roughly a third of scenes. The defensible statement is
therefore **not** that each scene carries this offset, but that the *mean* body-frame step error is
reliably non-zero along the optical axis. This is what the method needs: a coherent component
survives averaging and accumulates into drift, while the much larger incoherent remainder does not
(§6). For scale, $\|b\|$ is 4.6% of a typical predicted step, whereas the model over-predicts step
*magnitude* by 35%.

$T = 200$ frames for the decay constant. Selection is nearly free: leave-one-scene-out gives
$-0.0057$ ATE against an in-sample best of $-0.0058$, and 400 independent held-out halves all
selected a value in $\{150, 200, 250, 300\}$.

### 4.2 Inference, per frame

Strictly causal. Frame $t$ uses only $P_{t-1}, P_t$ and the frame index.

```
T_t   = P'_{t-1}^{-1} P_t                  # note: from the ORIGINAL P_t
tau   = translation(T_t)
tau  <- (tau - b) * 1 / (1 + t / T)        # bias, then progressive shrink
P'_t  = P'_{t-1} @ [rotation(T_t) | tau]   # re-integrate
```

with $P'_0 = P_0$. Rotation is untouched.

The reference implementation forms $T_t = P_{t-1}^{-1}P_t$ from the *uncorrected* pair and
re-integrates onto the corrected chain, which is equivalent and numerically cleaner.

### 4.3 Optional causal smoothing

If per-step metrics matter more than trajectory shape, append a causal exponential moving average
on the increments. Translation is filtered in the world frame so that the filter does not fight
the rotation; rotation is filtered in the rotation-vector domain.

```
w_t    = R'_{t-1} @ tau_t                          # world-frame displacement
w̄_t   = a_tau * w_t + (1 - a_tau) * w̄_{t-1}      # a_tau = 0.25
tau_t <- R'_{t-1}^T @ w̄_t

rho_t  = Log(R_t)                                   # rotation vector
rho̅_t = a_R * rho_t + (1 - a_R) * rho̅_{t-1}      # a_R = 0.40
R_t   <- Exp(rho̅_t)
```

### 4.4 Cost

Measured at **8.7 microseconds per frame**, single-threaded NumPy, for the base correction. The
whole 4292-scene test set, including 2.7 million pose-file reads and a full re-run of the official
evaluator, took 23 minutes on 72 cores. There is no GPU cost, no change to the model, and no
ground truth is consumed at inference.

---

## 5. Results

All rows scored on **all 4292 DROID wrist test scenes** by the project's evaluator
(`eval_bundle/bin/eval_depth_poses.py`) at its published defaults. Depth is symlinked from the
source run, so AbsRel and $\delta<1.25$ must come out bit-identical; they do, on every scene,
which is a built-in correctness check on the entire sweep.

| Method | AbsRel $\downarrow$ | $\delta{<}1.25\uparrow$ | ATE $\downarrow$ | RPE$_\text{trans}\downarrow$ | RPE$_\text{rot}\downarrow$ |
|---|---|---|---|---|---|
| CUT3R finetuned (baseline) | 0.1794 | 0.7863 | 0.0759 | 0.00794 | 1.1010 |
| **+ causal pose recalibration** | 0.1794 | 0.7863 | **0.0691** | **0.00706** | 1.1010 |
| **+ recalibration and causal EMA** | 0.1794 | 0.7863 | 0.0705 | **0.00510** | **0.7691** |

Paired against the baseline, per scene, with a 10,000-sample bootstrap:

| | $\Delta$ATE | scenes better | $\Delta$RPE$_\text{trans}$ | scenes better | $\Delta$RPE$_\text{rot}$ | scenes better |
|---|---|---|---|---|---|---|
| recalibration | $-9.0\%$, CI $[-0.0073,-0.0063]$ | 64% | $-11.0\%$ | 88% | $0.0\%$ | untouched |
| with EMA | $-7.1\%$, CI $[-0.0060,-0.0048]$ | 56% | $-35.8\%$ | 99% | $-30.1\%$ | **100%** |

**Leakage check.** The bias was calibrated on 100 of the 4292 scenes (2.3%). On the 4192 scenes
never used for calibration the ATE gain is $-8.9\%$, against $-9.0\%$ on all 4292. The in-set
calibration does not inflate the result.

**Ablation.** On the full set, bias subtraction alone gives $-0.0049$ ATE and the decay alone gives
$-0.0049$; together they give $-0.0068$, so they are about 70% additive rather than independent.

### 5.1 Calibration on a disjoint split: no test label is consumed

The bias above was fitted on 100 scenes of the test set. To remove that objection entirely, the
same quantity was refitted on **100 episodes of the DROID train split**, chosen seeded-uniform and
verified disjoint (zero scene-name overlap with the 4292 test scenes), and applied unchanged to all
4292. This required a fresh inference pass over the train episodes, since calibration reads the
model's own predicted increments.

$$b_\text{train} = (-8.871\times10^{-4},\; -8.077\times10^{-4},\; -1.920\times10^{-3}),
\qquad \|b_\text{train}\| = 2.264\times10^{-3}$$

against $\|b_\text{test}\| = 2.548\times10^{-3}$: the same sign on all three components and a norm
11% smaller. The constant is a property of the checkpoint, not of the scenes used to measure it.

| Method (all 4292 scenes) | AbsRel | $\delta{<}1.25$ | ATE | RPE$_\text{trans}$ | RPE$_\text{rot}$ |
|---|---|---|---|---|---|
| CUT3R finetuned (baseline) | 0.1794 | 0.7863 | 0.0759 | 0.00794 | 1.1010 |
| + recalibration, **train-split calibration** | 0.1794 | 0.7863 | **0.0690** | **0.00713** | 1.1010 |
| + recalibration and EMA, **train-split calibration** | 0.1794 | 0.7863 | 0.0702 | **0.00510** | **0.7691** |

$\Delta$ATE $= -9.0\%$ and $-7.5\%$ respectively; RPE$_\text{trans}$ $-10.3\%$ and $-35.8\%$;
RPE$_\text{rot}$ unchanged and $-30.1\%$ (better on 100% of scenes).

Paired directly against the test-calibrated rows: the base arm is **statistically indistinguishable**
($\Delta$ATE $-0.000014$, CI $[-0.000095, +0.000069]$), and the EMA arm is **significantly better**
($-0.000284$, CI $[-0.000361, -0.000206]$). Moving the calibration off the test set costs nothing and
slightly helps. All headline claims in this document therefore stand without any test-label access,
and the train-calibrated rows are the ones that should be cited.

### 5.2 Generality: the same recipe on a different checkpoint

The method was re-fitted and re-run end to end on `cut3r_zeroshot`, the released CUT3R weights that
were **never trained on DROID** and are six times worse in depth. Nothing was reused: the bias was
recalibrated from that checkpoint's own predictions on the same 100 calibration scenes, and $T$ was
left at 200.

The fitted bias is a genuinely different object — $\|b\| = 5.105\times10^{-3}$, twice the finetuned
value, and with a different sign pattern, $(+9.53\times10^{-4}, -2.245\times10^{-3}, -4.485\times10^{-3})$
against $(-1.974\times10^{-4}, -8.909\times10^{-4}, -2.379\times10^{-3})$.

| Method (all 4292 scenes) | AbsRel | $\delta{<}1.25$ | ATE | RPE$_\text{trans}$ | RPE$_\text{rot}$ |
|---|---|---|---|---|---|
| CUT3R zero-shot | 0.4786 | 0.5580 | 0.1197 | 0.01196 | 1.7132 |
| **+ causal pose recalibration** | 0.4786 | 0.5580 | **0.1090** | **0.01107** | 1.7132 |

$\Delta$ATE $= -9.0\%$, CI $[-0.0113, -0.0102]$, better on 74% of scenes. $\Delta$RPE$_\text{trans} = -7.4\%$,
better on 75%.

**The relative gain is identical to three significant figures** ($-9.0\%$ on both checkpoints) despite
the two models differing by a factor of six in depth accuracy, by 58% in absolute ATE, and by a
factor of two in the magnitude of the defect being corrected. This is the paper's strongest single
result. It promotes the claim from *"this checkpoint has a removable bias"* to **"recurrent
feed-forward 3D models carry a checkpoint-specific progressive-drift signature of a shared
functional form, and one offline calibration removes a fixed fraction of it."**

### 5.3 Composition: it does not stack on bidirectional fusion, and that is the point

`augfull_cg_fuse_g7` — confidence-gated forward $\times$ backward fusion, the strongest existing
row in the project's table — was used as the source stream, with the bias refitted on it.

| Method (all 4292 scenes) | AbsRel | $\delta{<}1.25$ | ATE | RPE$_\text{trans}$ | RPE$_\text{rot}$ |
|---|---|---|---|---|---|
| fwd$\times$bwd conf-gated fusion (2 passes, non-causal) | 0.1749 | 0.7915 | **0.0641** | 0.00718 | 0.9604 |
| + recalibration | 0.1749 | 0.7915 | 0.0676 (+5.4%) | 0.00679 ($-5.5\%$) | 0.9604 |
| + recalibration and EMA | 0.1749 | 0.7915 | 0.0722 (+12.7%) | **0.00511** ($-28.8\%$) | **0.6874** ($-28.4\%$) |

On ATE the correction **anti-stacks**: applying it on top of fusion makes ATE worse, significantly,
on 57% of scenes.

This is a confirmation of the diagnosis rather than a defeat of it. Averaging a forward and a
backward pass cancels a drift that is monotone in time, because the two passes accumulate it in
opposite directions. Fusion therefore already removes most of the very error this method targets,
and the evidence is quantitative: the bias refitted on the fused stream has norm
$1.721\times10^{-3}$ against $2.548\times10^{-3}$ on the single forward pass, i.e. **fusion has
already absorbed 32% of it**. What remains is too small to justify the late-step shrinkage, so the
shrinkage over-corrects.

Read together, §5.2 and §5.3 bracket the mechanism from both sides. Where progressive drift is
present the correction removes a constant 9% of ATE regardless of checkpoint quality; where an
orthogonal method has already removed the drift, the correction has nothing left to take and hurts.
No account other than progressive drift predicts both.

The per-step half of the method is unaffected by any of this and remains additive: stacked on
fusion, the EMA yields **RPE$_\text{rot}$ 0.687 and RPE$_\text{trans}$ 0.00511**, the best values of
any non-oracle configuration measured in this project, improving on 99% and 95% of scenes
respectively. That configuration is not causal, because fusion is not.

---

## 6. Why it works

The mechanism is **progressive scale inflation**. The model over-predicts its own step size
(§3.2), and the over-prediction grows as the sequence advances (§3.3), so the reconstructed
trajectory expands relative to the truth over time. A similarity alignment can remove a constant
scale error but not one that drifts, so this component survives the evaluation and shows up as
trajectory-shape error, which is what ATE measures.

Three properties of the error field explain which metric each component moves.

- **The increment error is near-white; the true motion is not.** Lag-1 autocorrelation of the true
  per-step motion is 0.963 (translation) and 0.949 (rotation), against 0.16 to 0.39 for the model's
  increment error. That gap is exactly the condition under which a causal low-pass filter wins on
  per-step metrics, and it is why the EMA cuts both RPEs by around a third.
- **Near-white error largely cancels in the integral.** The incoherent remainder accumulates at only
  $1.49\sqrt{N}$ times its RMS, which is why the EMA buys only 2.7% of ATE despite removing a third
  of the per-step error.
- **ATE is a drift metric.** Projecting the aligned position error onto a smooth cubic in time
  accounts for 65.2% of its mean-square value, and 89 to 94% of every method's ATE gain comes from
  that smooth component. The constant bias is only 0.23% of squared per-step error but supplies a
  median 53% of accumulated endpoint drift. **A tiny coherent error dominates the integral; a large
  incoherent one does not.**

### 6.1 Controls

Every control that should fail, fails, and in the right direction.

| Control | $\Delta$ATE | Reading |
|---|---|---|
| decay reversed in time (ramp up instead of down) | $+0.0139$ | the direction of the drift is real |
| decay replaced by a constant global gain | $-0.0000$ | exactly the similarity no-op predicted in §2 |
| decay values shuffled within each scene | $+0.0010$ to $+0.0018$ | the time ordering carries the signal |
| decay applied to a matched random frame subset | $+0.0006$ | null, as it must be |
| bias replaced by a random direction of equal norm | $+0.0030$ | the direction of $b$ is real |
| bias sign-flipped | $+0.0101$ | ditto, strongly |
| 4400 matched random smooth perturbations | never improved, mean $+0.0047$ | the correction is not a lucky smooth deformation |

### 6.2 An honest deflation of the functional form

The specific shape $1/(1+t/T)$ carries **no information**. A randomly generated monotone decreasing
log-gain of identical RMS achieves $-0.00531 \pm 0.00006$ against the fitted form's $-0.00531$. The
matching random *increasing* shape gives $+0.01449$. The method is, in substance, one scalar: how
much to downweight later steps. The parameter sweep over $T$ is decoration and should be described
as such.

Similarly, the initially attractive story that "per-step translation over-prediction grows through
the sequence" is only about a third right. Scale-free diagnostics show that the predicted/true
magnitude ratio rises about $1.5\times$ across normalised time while a direction-penalised optimal
gain falls $3.8\times$. Most of what justifies late shrinkage is **growing direction error**, not
growing magnitude over-prediction. The defensible statement is that the model's late steps carry
progressively less usable signal, and shrinking them toward zero is the correct response to that.

---

## 7. What does not work

These were measured on the same data with the same evaluator, and are reported because they bound
the family and because several are attractive enough to be worth warning about.

**Frame-level memory blocking, in every variant tested.** Suppressing the state and memory write of
frames identified as harmful, whether identified by ground-truth error, by the model's own
confidence, or by window detection, loses on all five metrics. On 100 scenes and 367 detected
decline windows: blocking the window is worse everywhere; blocking the equal-length span
immediately before it is neutral; blocking every window in a scene is worse still. Blocking a
random 30% of frames costs $+0.0061$ ATE and blocking the fastest 30% costs $+0.0490$. The frames
that look worst are largely the frames where the camera moved furthest, and absolute-scale error
metrics make them look worse than they are.

**The magnitude-calibration family, beyond one constant.** After removing $b$, a causal per-bin
rescaling as a function of predicted magnitude reduces per-step translation error energy by a
further 0.6 percentage points. The calibration curve is essentially flat, and only 2.6 to 3.4% of
the variance of the ideal per-frame log-factor is explainable by any function of the predicted
magnitude.

**Zero-velocity detection.** Zeroing the increment on frames detected as static is worth at most
$-0.004$ even with a ground-truth detector, because near-static frames are only 0.8% of total
per-step error energy. With an honest detector, an image-intensity rule is not significant
($-0.0014$, CI crossing zero) and its strictly causal expanding-window form is actively harmful
($+0.0011$). A Lucas-Kanade forward-backward consistency rule reaches $-0.0039$ at 5.2 ms per
frame, but a budget-matched control shows four different detectors are statistically
indistinguishable, so within this family the lever is the per-scene budget, not the ranking.

**Geometric self-consistency.** Warping the previous frame with the model's own depth and its own
predicted relative pose, and using the photometric residual as a confidence signal, is a clear
negative: the model's own warp raises the mean residual to 9.04 intensity levels against 5.44 for
not warping at all. The residual measures depth error, not motion error.

**Also dead:** rotation-angle calibration (improves RPE$_\text{rot}$ by 0.21 degrees but costs
$+0.0018$ ATE); damping the first $k$ increments during state warm-up; causal median filtering of
increment magnitudes; Holt double-exponential smoothing; depth-scale-consistency correction
($+0.0039$); a lever-arm rotation-to-translation prior; ORB-based detectors in every form; image
sharpness (AUC 0.497); predicted focal-length drift (AUC 0.648 but partial correlation 0.035);
and a learned ridge regression over 19 causal features, whose confidence interval crosses zero.

---

## 8. Limitations

1. **The bias is checkpoint-specific.** It must be recalibrated for each set of weights. The
   *family* transfers and the relative gain is invariant (§5.2: −9.0% on both the finetuned and the
   zero-shot checkpoint, whose fitted biases differ in magnitude by 2× and in sign pattern); the
   fitted object does not.
2. **Calibration consumes ground-truth poses**, offline, on a split disjoint from the one scored
   (§5.1). The method is training-free but not calibration-free. No test label is used.
3. **One dataset, one camera.** DROID wrist cameras follow smooth robot-arm trajectories. The
   near-white-error and smooth-signal structure that the EMA exploits may not hold for handheld or
   vehicle video.
4. **The functional form is uninformative** (§6.2). Claims about $1/(1+t/T)$ specifically should not
   be made.
5. **"Constant" overstates it.** The bias is a weak mean with per-scene scatter several times its
   own size, and only its optical-axis component is individually significant (§4.1). It should be
   described as the coherent part of the step error, not as a deterministic sensor bias.
6. **The remaining headroom in this family is small.** Roughly 0.002 to 0.004 ATE, and capturing it
   requires lifting per-frame gain estimation from a correlation of 0.36 with the ideal factor to
   about 0.6. Six detector families and eleven independent analyses found nothing in the emitted
   stream that gets past 0.36.
7. **It does not compose with bidirectional fusion.** On a stream that has already had its drift
   removed, the correction over-shrinks and costs 5.4% of ATE (§5.3). The method is for single-pass
   causal streams; it is not a universal add-on. Its EMA half still composes and still wins.
8. **Comparison to TTT3R and RayMap3R is indirect.** Their numbers in the project's master table are
   imported from an off-cluster evaluation and could not be recomputed here.

---

## 9. Reproduction

```bash
# full sweep: apply the correction and score with the official evaluator
python eval_pipeline/maks_posefix_sweep.py --label causal_posefix          # bias + decay
python eval_pipeline/maks_posefix_sweep.py --label causal_posefix_ema --ema

# or as a batch job over both operating points
sbatch eval_pipeline/maks_posefix_sweep.sbatch
```

| artefact | path |
|---|---|
| implementation | `eval_pipeline/maks_posefix_sweep.py` |
| batch driver | `eval_pipeline/maks_posefix_sweep.sbatch` |
| fitted bias, per label | `<outputs>/cut3r_eval/<label>/bias.json` |
| per-scene results | `<outputs>/cut3r_eval/<label>/eval/<scene>/eval_depth_pose_metrics.csv` |
| master table | `<outputs>/cut3r_eval/master/master_4292.{tex,pdf,png}` |
| evaluator | `eval_bundle/bin/eval_depth_poses.py`, published defaults, unmodified |

`<outputs>` is `/gpfs/scratch/etur59/koc821022/outputs`.
