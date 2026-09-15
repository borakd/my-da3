# What Kendall & Gal (NeurIPS 2017) can and cannot contribute to a confidence-weighted memory-update module for CUT3R

Date: 2026-09-13. Worktree `maks_idea`. Written for a reader who knows the conf-gate campaign
(`my-da3/CONF_GATE_CAMPAIGN.md`) and the maks_idea state (`MAKS_IDEA_STATE.md`) but did not follow this
analysis. Every number below was measured in this session or is quoted from the record; nothing is estimated.

The paper: Kendall and Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?",
NIPS 2017. Two uncertainty types (aleatoric = input-dependent observation noise, learned by a heteroscedastic
head; epistemic = model/weight uncertainty, approximated by MC dropout), a combined predictive variance
(Eq. 9), the observation that the heteroscedastic loss is a learned loss attenuation (Sec. 3.2), a
classification variant (Sec. 3.3), and diagnostics (precision-recall by uncertainty percentile, calibration
plots, train-size and out-of-distribution ablations in Table 3).

---

## 1. The answer in one page

The paper's most direct contribution is not a new gate signal. It is a vocabulary and a set of tests that,
applied to the existing gate, change what we believe the gate is doing. Applying them this session produced
the following measured facts (details in Sec. 3):

1. **CUT3R's confidence head is exactly the paper's heteroscedastic aleatoric head**, in the Laplace form the
   paper itself uses in practice: `conf * ||r|| - 0.2 * log conf` with `conf = 1/sigma`, regulariser weight
   0.2 instead of the paper's 1, and a floor `conf = 1 + exp(x) >= 1` the paper does not have. Under that
   loss the per-pixel optimum is `conf* = max(1, 0.2/L)`, so every pixel whose residual exceeds 0.2
   normalised units sits at the floor where the loss no longer constrains `x`. On DROID that is the majority:
   68% of frames have a frame-mean `x < -5` for the cross-view head. **The gate ranks frames by an
   unconstrained pre-activation, not by a calibrated 1/sigma.** Raw confidence has no usable dynamic range as
   a weight (a Kalman-style weight `c_t/(c_t + c_{t-1})` spans 0.495 to 0.504 between the 5th and 95th
   percentiles), so the paper's Eq. 7-9 must never be applied to raw conf; only ranks and empirical monotone
   maps are legitimate.

2. **The two confidence heads split cleanly at the frame level**, exactly as the paper's "sigma is attached to
   its own output" implies: the cross-view (world-frame points) confidence tracks pose error (within-scene
   Spearman -0.28 vs rpe_rot) and is blind to depth error (-0.06 vs absrel); the self-view confidence tracks
   depth error (-0.48) and is blind to pose error (-0.04).

3. **Both heads are blind to the magnitude of camera motion** (Spearman of conf vs the GT translation step
   +0.005; rotation step -0.06), although the cross-view head does respond to covisibility on the 18 scenes
   where it could be checked (+0.22 to +0.24, tercile d = -0.74). The one signal that tracks motion, `dstate`
   (relative state rewrite, +0.64 vs GT step), is a motion-magnitude detector, and the record says fast frames
   are where the model is most reliable per unit motion. The paper's Sec. 5.2 ("aleatoric uncertainty does not
   increase for out-of-data examples") applies to distribution shift, so blindness to step size is a fact about
   this saturated, age-dominated head rather than a paper prediction.

4. **The cross-view confidence, the champion gate's signal, is dominated by processing history, not by image
   content.** Decoding the same frames forward and backward, its per-frame ranking on identical images agrees
   between orders at only +0.03 raw (+0.34 after removing each order's monotone age trend), against +0.66 for
   the self-view head and +0.58 for `dstate`; it declines with the number of frames processed in both
   directions (-0.57 forward, -0.40 backward). A bare clock ranks per-frame rotation error as well as it does
   (-0.29 vs -0.28). In the paper's own terms the gate reads a state-dependent quantity through a head that
   was trained as an input-dependent aleatoric sigma. This is why hand-set thresholds do not transfer across
   checkpoints and why the gate-trained model's confidence distribution shifted.

5. **The per-pixel sigma of the paper maps onto the per-STATE-TOKEN gate, and doing that mapping correctly
   is worth measurable pose accuracy.** The finetuned model has 768 state tokens on a 28-wide RoPE grid, of
   which exactly 240 overlap the 20x12 image patches 1:1; the worktree's legacy per-token gate resampled a
   28x28 pooled map onto 768 tokens and was progressively misaligned. A RoPE-aligned gate, confirmed on 430
   scenes, gives ATE -0.0053* (298/430 wins) with no significant rotation cost, where the misaligned gate
   with the same weights paid rpe_rot +0.03*; a shuffled-alignment control behaves like the misaligned gate.
   Attenuating the 528 off-image register tokens is a separate lever that buys the depth gain (absrel
   -0.0020*, a1 +0.0023*) and extra ATE (-0.0072*, the largest causal single-pass ATE gain in the record) at
   a rotation cost (+0.04*).

6. **The gate's gain is orthogonal to the causal recalibration and the two stack.** The paper's zero-mean
   likelihood (Eq. 5/7) says attenuation reweights and never de-biases; measured on 40 pilot scenes, gated poses
   keep the same step-ratio (1.168 vs 1.157), the same growth (slope +0.286 vs +0.287) and the same refitted
   bias norm, and gate + recalibration gives ATE -.00755* against a sum of singles of -.00833 (91% additive;
   gate + recalibration beats recalibration alone by -.00234*, 29/40). This is directly actionable on the
   single-pass causal system.

7. **What the paper cannot buy.** ATE is the vector sum of nearly incoherent step errors and no per-frame
   quantity predicts it; the gate's ATE gain is dose-linear (ATE ~ -0.021 x attenuation). Per-frame
   precision-recall logic (Fig. 2) applies to per-frame rotation error, where the honest headroom is the gap
   between the best signal (dstate retention@20 = 0.83, conf 0.90) and the realised-error oracle (0.62), and
   where an oracle-keyed hard gate loses on all five metrics. Four dose-matched control arms on 430 scenes
   (Sec. 5) settle it: a constant write weight with no confidence reproduces the champion's whole ATE gain
   (-.0030* vs -.0032*) with no rotation cost, a position clock does the same, and even soft ORACLES keyed on
   realised per-frame error do not beat the constant. **The frame gate's gain is dose, not selection**; the only
   ranking that beats dose is the spatial per-token one (-.0023* ATE vs the constant).

---

## 2. Primer: the paper's notation mapped onto CUT3R

| Paper | CUT3R |
|---|---|
| Heteroscedastic head `[y_hat, sigma^2] = f(x)` (Eq. 6) | The `conf` channel of each point head: `conf_self` (self-view points) and `conf` (world-frame points), per pixel |
| Laplace loss `|r|/sigma + log sigma` (Sec. 4) | `ConfLoss`: `conf * L21(r) - alpha * log conf`, `alpha = 0.2`, `conf = 1 + exp(x)` (`losses.py` ConfLoss, `postprocess.py` reg_dense_conf) |
| `s = log sigma^2` unbounded (Eq. 8) | `x` is unbounded but `conf` is floored at 1: `sigma <= 1` always, and `x` is unconstrained by the loss once `L >= alpha` |
| Learned loss attenuation (Sec. 3.2) | Already in effect for the point maps; the pose loss (`compute_pose_loss`, L2 on translation and quaternion) has no attenuation and no sigma |
| Epistemic via MC dropout (Sec. 2.1) | Not available: every dropout module is constructed with p = 0 and the model was never trained with dropout |
| Predictive variance `Var(y) = spread + mean sigma^2` (Eq. 9) | No estimator exists; nearest free analogs are `dstate` (belief change on one write) and the two-head disagreement (world points vs pose-composed self points), both untested as gates |
| PR by uncertainty percentile (Sec. 5.1, Fig. 2) | The retention curves in Sec. 3 below, per frame and per signal |
| Train-size / OOD ablation (Table 3) | Forward-vs-backward order test and the checkpoint sweep (zero-shot vs finetuned) |
| Memory write `state <- g*new + (1-g)*old` | Not in the paper. The paper supplies R (observation noise) at best; P (state uncertainty) and the gain are filtering constructs |

The gate itself (`STATE_GATE_*` in `my-da3/src/CUT3R/src/dust3r/model.py`): `g = max(gmin, sigmoid((s - tau)/temp))`
on an EMA of `s = mean log(conf - 1)`, tau at a pooled percentile; champion `augfull_cg_g7ema` (tau = -11.99,
temp 1.5, gmin .70, EMA .7): ATE -4.0%* on 4292, rpe_rot flat. Per-token variant in this worktree:
`token_gate_q/gmin` (+ new `token_gate_align`, `token_gate_offg`).

---

## 3. What was measured this session

All per-frame statistics: 430 scenes (`maks_sweep430/scene_list.txt`), 134,448 frames, clean model telemetry
(`diag_cg_log430`), within-scene Spearman, mean over scenes with a 95% bootstrap CI over scenes (never over
frames). Scripts and tables in the session scratchpad (`probe_perframe_calibration.py`,
`probe_signal_vs_motion.py`, `probe_position_confound.py`, plus the verifiers' `verify_*.py`).

### 3.1 Per-frame precision-recall of the existing signals (paper Fig. 2 analog)

| signal | rpe_rot_t | rpe_trans_t | absrel_t | rpe_rot_{t+1} | GT trans step | position |
|---|---|---|---|---|---|---|
| conf_mean (cross-view) | -0.28 | -0.24 | -0.06 | -0.27 | +0.005 | -0.57 |
| conf_p50 (cross-view) | -0.31 | -0.26 | +0.03 | -0.30 | -0.002 | -0.61 |
| conf_self_mean (self-view) | -0.04 | -0.12 | -0.48 | -0.03 | -0.04 | -0.17 |
| dstate | +0.46 | +0.55 | +0.16 | +0.39 | +0.64 | +0.04 |
| dmem | -0.29 | -0.27 | +0.12 | -0.29 | -0.03 | -0.999 |

Retention (drop the k% worst frames by within-scene percentile; RMS rpe_rot of retained / all; 1.00 = no
ranking power): at k = 20, conf_mean 0.896, conf_p50 0.890, dstate 0.825, conf_self 0.942, shuffled null
0.996, oracle by realised rpe_rot 0.616. For the NEXT frame's rpe_rot (what a write gate on frame t can
influence): conf_mean 0.906, dstate 0.876. No signal predicts the change of rpe_rot from t to t+1
(|rho| <= 0.06). A ridge regression over the seven telemetry scalars, fitted on 215 scenes and scored on the
other 215, reaches +0.61 vs rpe_rot_t (retention 0.78) and +0.49 next-frame: a combination beats every single
signal.

### 3.2 Position confound and the order test (paper Sec. 5.2 analog)

Rank-partial correlations: conf_mean vs rpe_rot -0.28 -> -0.11 given position -> -0.15 given position and GT
step. dstate +0.46 -> +0.22. conf_self ~ +0.03. Order test on identical frames decoded forward
(`diag_cg_log430`) and backward (`augfull_cg_g7ema_bwd`; its telemetry rows are in original scene order,
verified with the dmem clock, rho +0.999 vs row):

| signal | fwd vs frames processed | bwd vs frames processed | fwd-vs-bwd agreement, same frame (raw / detrended) |
|---|---|---|---|
| conf_mean | -0.57 | -0.40 | +0.03 / +0.34 |
| conf_p50 | -0.61 | -0.37 | +0.03 / n.a. |
| conf_self_mean | -0.17 | -0.09 | +0.66 / +0.71 |
| dstate | +0.04 | -0.18 | +0.58 |

Endogeneity (does attenuating writes lower the signal the gate reads?): real and positive-feedback for hard
write blocks (a blocked write lowers the next frame's cross conf; first-difference Spearman -0.28; the loop
inflates the skip budget of the absolute-threshold hard gate by 44%), immaterial for the soft champion
(median delta at the first attenuated frame -0.0008).

### 3.3 Saturation of the confidence head

Frame-mean `x = log(conf - 1)` pooled percentiles: p50 -8.79, p90 -1.21, p99 +0.34 (cross-view); the self-view
head is less saturated (p50 -2.28). Fraction of frames with frame-mean x < -5: 0.68 cross, 0.31 self.

### 3.4 The RoPE-aligned per-token gate (paper per-pixel sigma -> per-token weight), 430 scenes, paired vs augfull_lr1e5

| arm | absrel | a1 | ATE | rpe_trans | rpe_rot |
|---|---|---|---|---|---|
| legacy tok_all_q50_g50 (misaligned) | -.0016* | +.0030* | -.0048* (293/430) | +.0001 | +.03* |
| shuffled-alignment control | -.0011* | +.0024* | -.0041* (284/430) | -.0000 | +.01 |
| aligned, registers untouched (tok_al_q50_g50) | -.0008 | +.0011 | -.0053* (298/430) | +.0000 | +.01 |
| aligned + registers at .5 (off50) | -.0020* | +.0023* | -.0072* (322/430) | +.0002* | +.04* |
| aligned gmin .7 (tok_al_q50_g70) | -.0004 | +.0010 | -.0026* (260/430) | -.0001* | -.01 |
| aligned gmin .7 + registers .7 | -.0003 | +.0004 | -.0037* (277/430) | -.0000 | +.01 |
| reference frame gate augfull_cg_g7ema | -.0010* | +.0021* | -.0032* | +.0002* | +.02* |

Mechanics: `register_tokens` is (768, 1024); with `state_pe='2d'` token i sits at RoPE position (i//28,
i%28); eval images are 320x192 = 20x12 patches, so tokens with row < 12 and col < 20 overlap the image 1:1;
RoPE is applied to both queries and keys in the state-image cross attention. Implemented as view keys
`token_gate_align` (0 legacy, 1 aligned, 2 shuffled control) and `token_gate_offg` in
`src/CUT3R/src/dust3r/model.py`, parsed from `token_gate.align/offg` in `eval_pipeline/infer_and_eval_worker.py`;
arm files `eval_pipeline/maks_arm_tok_al_*.json`; results `$OUT/maks_align12/`, `$OUT/maks_align430/compare/`.
The legacy path is byte-identical when the keys are absent (re-run of the legacy arm matched the earlier
sweep on all 12 smoke scenes).

### 3.5 Dose-matched controls for the frame gate (paper Sec. 5.1 "strictly decreasing beyond a trivial ranking")

Four arms on the same 430 scenes, all built purely from per-frame `update_alpha` control files (the same commit
equation as the STATE_GATE soft mode, frame 0 always written), all at the champion's mean attenuation on this
subset (14.7%): UNIFORM (alpha = 0.853 on every frame, no confidence); CLOCK (alpha by normalised-position
decile, taken from the champion's own realised g, no confidence); ORACLE-SAME (g = .7 on the worst 49% of frames
by realised same-frame rpe_rot, EMA .7); ORACLE-NEXT (same, keyed on the NEXT frame's realised rpe_rot, i.e.
"does writing frame t hurt t+1"). Results: see the table in Sec. 5 (filled in when the jobs finished).

---

## 4. The contributions, ranked

Sixty-five candidate contributions were generated from seven reading lenses, merged to 42, and each was checked
against the paper text, the empirical record and the code; 29 checks ran as separate verifiers (several ran
their own CPU probes), the other 13 were checked inline against the same evidence. Below, each entry gives the
paper element, what it maps onto in CUT3R, what the evidence says, and the cheapest decisive test with its
decision rule. Metric deltas are paired vs `augfull_lr1e5`; * = 95% scene-bootstrap CI excludes zero.

### Tier 1: what the paper has already contributed (validated this session)

**R1. The identity: CUT3R's confidence IS the paper's heteroscedastic aleatoric head, and reading it that way
explains the gate.** (Sec. 2.2, 3.1, 3.2, Sec. 4 Laplace form.) Consequences that are now measured facts rather
than beliefs: the head is floored and saturated (Sec. 3.3), so only ranks and empirical monotone maps are
legitimate; the cross-view head's sigma includes registration to memory, which is why it tracks pose error while
the self-view head tracks depth (Sec. 3.1); and, per the paper's own Sec. 5.2, an aleatoric head cannot see the
low-overlap frames that drive pose error (conf vs GT step = +0.005). Nothing to test; this is the frame every
later entry uses.

**R2. The paper's diagnostics transplanted to a recurrent model expose that the champion gate reads a
state-history quantity and that its gain is a dose effect.** (Sec. 5.1 precision-recall; Sec. 5.2 / Table 3
in-vs-out-of-distribution logic, here realised as the forward-vs-backward order test.) Cross-view confidence:
fwd-vs-bwd agreement on identical frames +0.03 raw, +0.34 detrended; declines with frames processed in both
orders; a bare clock ranks per-frame rotation error as well as it does; more than half of its correlation with
error is position. The dose-matched controls (Sec. 3.5 / Sec. 5) then decide whether any frame-level ranking
matters at all. This is the single most consequential output of applying the paper: it moves the design
question from "which signal" to "how much, where, and how smoothly".

**R3. Per-pixel sigma maps onto per-STATE-TOKEN weights, and the alignment is worth measuring.** (Sec. 3.1
Eq. 6-7: sigma is attached to the same output; Sec. 4.2: aleatoric depth uncertainty is spatially structured.)
Measured (Sec. 3.4): aligned q50/g50 ATE -.0053* with no rotation cost; misaligned/shuffled weights pay +.03*;
attenuating the off-image registers adds the depth gain and more ATE (-.0072*) at a rotation cost (+.04*).
Honest attribution (verifier probe on the same 430 scenes): a confidence-free random half-mask already gives
-.0041*, so the confidence-attributable increment of alignment is ATE -.0013* (paired aligned-minus-shuffled,
230/430, CI [-.0024, -.0001]) with no rotation difference. Real, small, and the only place in the record where a
ranking (spatial, not temporal) moved the ATE-vs-rotation frontier. Next test (T4 below, R-C08): conf-free spatial
keys (predicted depth, fixed border ring) at the same budget, to learn whether the increment is confidence or
saliency.

**R4. Use confidence as a rank or through an empirical monotone map; never as a variance.** (Fig. 3 calibration
logic applied to a floored head.) Raw conf fed as a precision has no dynamic range (Kalman-style weight
c_t/(c_t + c_{t-1}) spans 0.495 to 0.504 between the 5th and 95th percentiles). An isotonic map
R(s) = E[rpe_rot_{t+1}^2 | s] fitted on half the scenes calibrates on the other half (decile mean relative error
11%, 7.8x dynamic range) but cannot change any ranking (held-out Spearman 0.417 vs 0.416): it is a calibration
artefact that turns tau into degrees-squared, useful for portability across checkpoints, not a gain.

**R4b. Stack the write gate with the causal pose recalibration.** (Eq. 5/7 zero-mean likelihood: the gate cannot
de-bias or de-inflate the emitted step, so its ATE gain lives elsewhere.) Measured on 40 scenes with the existing
gated poses: gate alone -.00312*, train-calibrated recalibration alone -.00521*, stacked -.00755* (30/40), i.e.
91% of the sum; the recalibration refits an unchanged bias on the gated stream (unlike fusion, where the bias
shrank 32% and the two anti-stacked). Cheapest test: apply `maks_posefix_sweep.py` with the train-split bias to
the champion's 4292 poses and score paired against the `_tc` row; expected order -10 to -12% ATE for one causal
pass.

### Tier 2: testable at inference, no training (surviving with corrections)

**R5. Two-route pose consistency (world-view points vs pose-composed self-view points).** (Spread term of
Eq. 4/9 as an analogy only; the paper's estimator needs dropout draws the model lacks.) Both heads and the pose
are supervised in the view-0 frame at one scale, so `P_cross = R P_self + t` holds at the optimum; the per-pixel
residual and a weighted Kabsch self-to-cross transform vs the pose head give a causal, single-pass, GT-free
pose-inconsistency signal in metric units that does not depend on the saturated pre-activation. Untested; zero
training; ~40 lines in the STATE_GATE sig block; one 12-scene then one 430-scene log run. Keep only if the
rank-partial vs next-frame rpe_rot given position and GT step exceeds 0.15, |rho| vs GT step < 0.3, and next-frame
retention@20 beats the ridge over existing signals (0.837); then a soft arm at matched dose against the CLOCK
control. Weights must be `conf - 1` (or 1/z^2), not conf.

**R6. Continuous, smoothed per-token weights on the aligned grid.** (Sec. 3.2 smooth exp(-s) rather than a hard
mask; not the paper's absolute thresholds, which need calibration the head lacks.) Replace the within-frame rank
step {1, gmin} by gmin + (1-gmin) sigmoid((x_i - tau_tok)/T_tok) with pooled constants, plus a per-token EMA (.7)
of the weight; registers as a separate constant. Bar: tok_al_q50_g50 (ATE -.0053*, rot n.s.) at matched
on-image mean weight 0.75; three 430 arms (continuous, continuous + EMA, rank + EMA as the smoothing control).

**R7. Cross-view-keyed token gate as an attribution experiment.** (Eq. 6/7: sigma of the world-frame output.)
Key the on-image tokens on the cross map or the per-patch gap instead of the self map. The paper's own model
predicts the within-frame ranking of the cross map equals the self map re-weighted by depth, so this is a
control on whether the on-image increment carries any pose information, not an expected improvement. Same job as
R-C08; decision rule: if rho(cross map, self map) > 0.9 within frames the arm is a relabelling and the question is
closed.

**R8. Recursive gain schedule (Kalman-form) for the frame commit.** (Only the additive P + R analogy of Eq. 4/9;
the recursion is filtering, not the paper.) g_t = P_t/(P_t + R_t) with P growing between writes and shrinking on
writes replaces gmin/EMA/max_skip by two scalars and re-opens writes smoothly after attenuated runs. At the
champion's budget the recursion is effectively memoryless (steady state in 0-1 steps), so it can only matter in
the deep-attenuation regime, which the record says is dead; R_t must be an isotonic map, never raw conf.
Cheapest test is CPU (open-loop schedules on the 12 smoke scenes injected through control files) against the
CLOCK control at the same budget. Low expected value.

**R9. Temporal reprojection residual (frame t-1's depth reprojected with the predicted relative pose vs frame t's
depth).** Measured offline on the 12 smoke scenes from saved depth and poses (4231 frames): Spearman +0.40 vs
rpe_rot_t, +0.50 vs rpe_trans_t, +0.06 vs absrel (pose-specific, depth-blind); on the same scenes it beats
conf_mean and dstate, is not a motion detector (rank-partial given GT step +0.35) and not a clock. Caveat from the
verifier: its pose-error component carries almost no ranking power (a GT-pose residual ranks equally), because the
parallax from the measured step over-prediction lies below the consecutive-depth-disagreement floor; it is a
registration-difficulty signal in the SfMLearner tradition, not an Eq. 4/9 quantity. Zero training, causal,
~50 lines; next: 430-scene log run and a soft arm at matched dose against the CLOCK control.

**R10. Output-side uncertainty-aware step shrink (refuted as a method, kept as a mechanism).** (Eq. 5 noise model pushed through the step norm: with
isotropic noise of variance 3 sigma^2 on the pose, E||v + e||^2 = ||v||^2 + 3 sigma^2, so the predicted/true
step ratio exceeds 1 most where the true step is small; one mechanism for the measured 1.27x over-prediction,
the isotropic slow-frame direction noise and the per-unit-motion facts.) This is the only entry that addresses
progressive scale inflation, and it lives in the recalibration, not the write gate. Two negative priors: a
31-feature per-frame gain reached only 0.36 correlation with the ideal factor and lost to the two-parameter rule.
Verified: the per-frame shrink k_t correlates with the ideal gain at only +0.24 within scene (predicted magnitude
alone +0.27), below the 0.36 a 31-feature regressor already failed with, and the "magnitude calibration beyond one
constant" family is recorded dead. What survives is the reading: after the per-scene Sim3 the 1.27x step
over-prediction is path-length inflation by per-step noise (excess ||v_hat||^2 - ||v||^2 rises with step
magnitude), which reframes "the model over-predicts its step" as a noise artefact and explains why a late shrink
keyed on falling signal-to-noise works. No new arm.

### Tier 3: requires training (all carry the co-adaptation caveat; none before the Tier 2 and Tier 4 items)

**R11. Unbounded confidence parametrisation (conf = exp(x), the paper's s = log sigma^2 without the floor).**
(Eq. 8; Sec. 3.2 regulariser.) With vmin = 1 every pixel with residual >= alpha sits at the floor with vanishing
gradient (68% of frames at frame-mean x < -5). Stage 1 duplicates the conf branch of each DPT head on the frozen
model (xyz and pose byte-identical, so no co-adaptation), retrains it with vmin = 0, logs raw x everywhere.
Decision rule: within-scene Spearman vs rpe_rot must improve beyond the scene CI AND the rank-partial given
position must improve from -0.11, otherwise the new head is a better clock, not a better gate. Because every gate
is a rank thresholder, the expected gain is calibration, not ranking.

**R12. Per-state-token log-variance readout on a frozen backbone.** (Eq. 7-8 with D = number of state tokens.)
769 parameters trained online during a forward-only rollout with the SAME-frame aligned self-view patch residual
as Laplace target (not the next frame's residual, which is the victim criterion). Kill gate first: per-token
retention on held-out train episodes must beat pooled conf_self at k = 20; then a 12-scene arm ranked by s_i at
the tok_al_q50_g50 budget with a shuffled-s control.

**R13. Learned scalar write gate trained through the recurrence: refuted as a frame-level module.** The
dose-matched controls show no frame-level headroom for a better ranker; the paper's log-sigma term does not even
point the right way in a write gate (s -> inf raises, not lowers, the next frames' losses). Salvageable only as
a per-token head with an explicit dose prior and a longer gradient horizon, and only if the final controls show
the champion beating both UNIFORM and CLOCK.

### Tier 4: diagnostics (done, or cheap and decisive)

| id | question | status |
|---|---|---|
| R-C09 | dose-matched confidence-free controls (uniform, clock) and soft oracles at the champion's budget | done: see Sec. 5 |
| R-C10 | order test: is the gate signal input- or history-driven | done: history-dominated (+0.03 raw, +0.34 detrended agreement) |
| R-C11 | cumulative endogeneity of attenuation on the signal | done: real for hard blocks (skip budget +44%), immaterial for the soft champion |
| R-C12 | is conf better calibrated early (small state error)? | done: no; first third of episodes is the WORST window (rho null), level not ranking tracks position |
| R-C08 | conf-free spatial controls for the aligned token gate (depth-keyed, border ring) | planned; decides whether the -.0013* increment is confidence or saliency |
| R-C31 | pixel-level PR and Laplace calibration of both conf heads (needs a conf-map dump, 12 scenes) | planned; fraction of pixels at the floor inside the gated band |
| R-C36 | does the gate's ATE gain stack with the causal recalibration; is it an implicit late-step shrink? | done (40-scene pilot): not a shrink (step ratio, growth and bias unchanged); stacks 91% additively; promoted to R4b |
| R-C35 | does any signal see the DRIFT (log pred/true step ratio) rather than noise? | 60-scene pilot: no; cross conf +0.02 given position and steps, dstate sign-unstable, conf_self +0.16 with the wrong sign. Drift is invisible to the uncertainty heads |
| R-C34 | cross-correlogram: is any signal frame-specific? | done, 430 scenes: conf_mean flat (lag-1 sharpness +0.01, 8% of the metric's own ceiling; two-thirds of its correlation is the linear clock; first-differenced ~0); dstate frame-specific at lag 0 only (differenced +0.18 at 0, 0.00 at +1). A per-frame gate on conf is a noisy schedule |
| R-C33 | within-motion-stratum ranking power of each head | pilot on 18 scenes: cross conf responds to covisibility (+0.22, tercile d -0.74) and to sideways motion at matched speed (d -0.50) but not to step magnitude; full 430 with GT depth pending |
| R-C32 | Table 3 analog across checkpoints | done from existing ckpt-10 telemetry: the conf LEVEL is checkpoint-dependent (p50 -16.2 at ckpt-10 vs -8.8 final, 2.1 within-scene SDs; frozen tau attenuates 26% vs 12%) while ranking power is stable (retention@20 0.885 vs 0.902). Gates must be rank-keyed |
| R-C23 | forward/backward disagreement as a per-frame error ranker (non-causal teacher) | CPU on existing cameras; must beat the ridge (0.778) to matter |
| R-C22 | memory-sensitivity variance (one extra batched decode over perturbed states) | one 21-scene job; kill if it is another motion detector |
| R-C21 | pose-head sensitivity (exact Jacobian of the pose MLP) | CPU from a pose-token dump; sensitivity, not epistemic |
| R-C27 | per-token state innovation tail as a diagnostic | telemetry extension; separate on-image and register groups |
| R-C42 | should the write weight rise or fall late in the episode? | the CLOCK control already answers the schedule question |

### Do not do (negative results the paper and the record jointly support)

- **Do not treat DUSt3R confidence as a calibrated variance** (inverse-variance, Kalman or Eq. 9 weights fed raw
  conf). No dynamic range; measured.
- **Do not retrain the finetune with an attenuated (Eq. 8) pose loss on absolute pose targets.** CPU emulation on
  1858 real 64-view windows: the training residual is a clock (Spearman +0.77 with position, +0.07 with motion,
  3.9x larger in the late half), so the Eq. 8 optimum halves the drift supervision (late-half gradient share
  0.50 -> 0.23). ~550 GPU-hours to learn "late = uncertain".
- **Do not build a frozen-backbone pose-sigma head as a gate signal.** Its ceiling is the soft oracle, and the
  soft oracles at matched dose do not beat uniform attenuation or the clock (Sec. 5).
- **Do not motion-normalise a learned sigma.** The ratio form over-normalises (dstate/step vs GT step -0.95),
  flips the sign of the error correlation and is worse than the shuffled null (retention 1.06).
- **Do not build MC dropout.** Every dropout module is p = 0 and the model was never trained with dropout; a
  test-time mask is a sensitivity analysis, not a posterior sample; the paper's own Table 3 says epistemic
  uncertainty is explained away in this big-data in-distribution regime; T passes violate the single-pass goal.
- **Do not compose "aleatoric + epistemic pose variance" (Eq. 9).** Neither term exists; the paper's own Table 2a
  shows the combination is not better; the combo-mode family is dominated by the plain soft gate.
- **Do not gate on step-residualised dstate or NIS.** Residualising on the predicted step leaves +0.15..0.21
  correlation but ranks worse than raw dstate, conf, or the predicted step itself (retention 0.92-0.95 vs 0.88).
- **Do not condition the gate on predicted overlap, and do not exempt fast frames ("slow-only").** The champion's
  attenuation is already motion-neutral (35% of it on the fastest tercile, rho(attenuation, step) = -0.015) and
  uniform attenuation has no rotation cost, so there is no motion-placed cost to remove. The gate's rotation cost
  is a fast-VICTIM effect: staleness accumulated during attenuated slow runs surfaces when the camera moves
  (+.038* at fast victims, 0 at slow); attenuating a fast frame itself is benign (cap_p90 shows no cost).
- **Do not add per-scene tau offsets** (measured: dominated by the global tau). Note the opposite result for a
  per-scene CONSTANT write level, which is the uniform control in Sec. 5.
- **Do not train an end-to-end gate with any CUT3R weight unfrozen, or with realised per-frame error as a label.**
  Co-adaptation measured three times; realised error marks victims (oracle hard block loses on all five).
- **Do not build novelty / confidence-drop triggers.** Measured dead (all n.s. at 40 scenes; hard block on trigger
  rpe_rot +.17*); the paper itself says aleatoric uncertainty does not spike out-of-data.
- **Do not put a sigma on the pose-retriever write.** The write is a single-key cross-attention broadcast
  identical across all 256 slots (verified on CPU with the repo's block); scope = mem was a wash.
- **Do not build a scene-level sigma.** Scene-conditional tau adds nothing, and the signal's within-scene decline
  cannot be expressed by one scalar.
- **Do not expect better per-frame uncertainty to buy ATE.** ATE is an integral of nearly incoherent step errors;
  no per-frame quantity predicts it; the frame gate's gain is dose-linear. The one exception measured is spatial
  (the aligned token gate), and even there three quarters of the gain is confidence-free.

---

## 5. Dose-matched control results (430 scenes)

All four control arms and the two references on the same 430 scenes, paired vs `augfull_lr1e5`
(`$OUT/maks_ctl430/compare/subset_compare.md`; jobs 45819758-61). Mean attenuation 14.7% for every arm (the
champion's own on this subset).

| arm (no model change, control files only) | absrel | a1 | ATE | rpe_trans | rpe_rot | sig wins / sig losses |
|---|---|---|---|---|---|---|
| UNIFORM alpha = .853 on every frame (no confidence) | -.0011* | +.0027* | -.0030* (277/430) | -.0001 | -.000 | 3 / 0 |
| CLOCK alpha by position decile (no confidence) | -.0005 | +.0012* | -.0033* (276/430) | +.0001 | +.006 | 2 / 0 |
| ORACLE same-frame realised rpe_rot, soft, EMA | -.0010* | +.0018* | -.0021* (255/430) | +.0000 | +.011* | 3 / 1 |
| ORACLE next-frame realised rpe_rot, soft, EMA | -.0007 | +.0017* | -.0028* (276/430) | +.0000 | +.020* | 2 / 1 |
| champion frame gate augfull_cg_g7ema (confidence) | -.0010* | +.0021* | -.0032* (261/430) | +.0002* | +.016* | 3 / 2 |
| aligned token gate tok_al_q50_g50 (spatial confidence) | -.0008 | +.0011 | -.0053* (298/430) | +.0000 | +.014 | 1 / 0 |

Paired arm-vs-arm (per scene, 95% bootstrap CI over scenes; negative ATE = first arm better):

| pair | ATE | rpe_trans | rpe_rot |
|---|---|---|---|
| champion minus UNIFORM | -.0001 [-.0012, +.0009] | +.0003* | +.016* [+.008, +.024] |
| champion minus CLOCK | +.0001 [-.0009, +.0011] | +.0002* | +.010* [+.002, +.018] |
| ORACLE-same minus UNIFORM | +.0010 [-.0000, +.0019] | +.0001* | +.011* |
| ORACLE-next minus UNIFORM | +.0002 [-.0008, +.0012] | +.0001* | +.020* |
| CLOCK minus UNIFORM | -.0002 [-.0012, +.0008] | +.0001* | +.006 |
| aligned token gate minus UNIFORM | **-.0023* [-.0035, -.0011]** (251/430) | +.0001 | +.014 [-.008, +.050] |
| aligned token gate minus CLOCK | **-.0021* [-.0032, -.0009]** (250/430) | -.0000 | +.009 |
| aligned token gate minus SHUFFLED alignment | -.0013* [-.0024, -.0002] (230/430) | +.0000 | +.001 |

**Reading.** At matched dose, a constant write weight with no confidence at all reproduces the entire ATE gain of
the confidence-keyed frame gate (-.0030* vs -.0032*, paired difference -.0001, CI straddles zero), with zero
rotation cost and the same depth wins; the confidence-keyed per-frame variation adds only a rotation and
translation penalty (+.016*, +.0003*). Even a perfect per-frame error signal, used with exactly the champion's
soft/floor/EMA recipe, does not beat the constant (paired ATE +.0010 and +.0002, rotation worse). The frame gate's
benefit is therefore a DOSE effect: how much of the state is preserved per step, not which frames. Time shape
(CLOCK) adds nothing over the constant either. The only ranking that beats the dose is SPATIAL: the aligned
per-token gate beats the constant by -.0023* ATE (251/430) and the shuffled-alignment control by -.0013*, at no
significant rotation cost. This also settles the training-side puzzle: a constant partial write is what the
gate-trained finetune co-adapted to, so its eval-time gate had nothing left to add.

---

## 6. Recommended order of work

1. **Act on the control table.** UNIFORM ties the champion on ATE with no rotation cost: retire tau/temp tuning
   and the search for frame-level signals. The module becomes "how much (dose), where (which state tokens), how
   smoothly (no per-frame jitter)"; the per-frame confidence is not load-bearing. Promote the constant write
   weight to the master table (one scalar, zero cost) and re-run the fwd x bwd fusion on top of it.
2. **CPU only, existing outputs:** stacking of the gate with the causal recalibration and the step-ratio
   decomposition of gated poses (R-C36); drift visibility of each signal (R-C35); 430-scene correlogram (R-C34).
3. **One 4-GPU job, 12 then 430 scenes:** conf-free spatial controls and the cross-keyed token gate (R-C08 / R7),
   to attribute the aligned gate's -.0013* increment.
4. **One log-mode telemetry job, 12 then 430:** two-route pose consistency (R5) and temporal reprojection (R9)
   keys, plus a conf-map dump on 12 scenes for the pixel-level calibration audit (R-C31).
5. **Only if step 3 shows the increment is confidence:** continuous smoothed per-token weights (R6).
6. **Training, only after 1-5 and only on a frozen backbone:** unbounded conf head (R11), per-token readout (R12).
7. **Independently of the gate:** the noise-norm shrink test for the recalibration (R10).

---

## 7. Reproduction

```bash
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh && conda activate cuteanything
export OMP_NUM_THREADS=1   # login node: 300 s CPU cap counts BLAS threads
# per-frame audits on the 430-scene telemetry (scratchpad copies; ~40-200 s each)
python probe_perframe_calibration.py; python probe_signal_vs_motion.py; python probe_position_confound.py
# aligned per-token gate, 12 smoke scenes (4 arms on one node)
WT=$PWD SCENE_LIST_OVERRIDE=$WT/../my-da3/eval_pipeline/cg_smoke_scenes_12.txt PILOT=maks_align12 \
  ARMS="tok_all_q50_g50 tok_al_q50_g50 tok_alshuf_q50_g50 tok_al_q50_g70" sbatch eval_pipeline/maks_subset.sbatch
# 430 scenes, one arm per job
for ARM in tok_al_q50_g50 tok_al_q50_g70 tok_alshuf_q50_g50 tok_al_q50_g50_off50 tok_al_q50_g70_off70; do
  WT=$PWD ARM=$ARM PILOT=maks_align430 sbatch eval_pipeline/maks_sweep430.sbatch; done
# dose-matched controls (control files eval_pipeline/maks_arm_ctl_*.json)
for ARM in ctl_uniform ctl_clock ctl_orsame ctl_ornext; do WT=$PWD ARM=$ARM PILOT=maks_ctl430 sbatch eval_pipeline/maks_sweep430.sbatch; done
# paired comparison
python eval_pipeline/maks_subset_compare.py --scene_list $OUT/maks_sweep430/scene_list.txt --baseline augfull_lr1e5=$OUT/augfull_lr1e5/eval --arm <name>=<eval dir> ... --out_dir <dir>
```

Jobs: 45815771 (12-scene aligned smoke), 45816460-64 (430 aligned arms), 45819758-61 (430 controls).
