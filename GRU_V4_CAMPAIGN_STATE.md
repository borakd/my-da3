# GRU v4 Autonomous Campaign — state file

**Mission (user directive, 2026-08-13):** Fully autonomous campaign to find a
GRU-based, strictly closed-loop design that beats the regular CUT3R finetune
baseline (arm A) significantly on as many of the 5 metrics as possible (ATE,
RPE-trans, RPE-rot, AbsRel, a1), ideally degrading none. Then launch a
full-dataset finetune, 32 GPUs, lr 1e-5 (the benchmark recipe all baselines
use). Hard evidence required that (a) constraints were obeyed and (b) metrics
are really better. Do not modify code outside branch `captain_gru_v3`.

## Hard constraints
1. GRU must be the corrector/integrator.
2. Closed loop: no GT poses or other privileged information at inference.
3. No retraining of baselines A/B/C.
4. All code changes on `captain_gru_v3` only.
5. Wins must clear the measured training-noise scatter (harness itself is
   byte-deterministic, 4292/4292); §0 acceptance gates of
   GRU_GAP_CLOSURE_DIRECTIVE.md are the pass rules.
6. Final run: full dataset, 32 GPUs, lr 1e-5.
7. GPU-hour awareness (user, 2026-08-13): don't burn hours unless necessary —
   aware, not restrictive. Budget at campaign start: etur59 on MN5 ACC has
   ~3150 khours remaining of 6700 (53% used), expires 2026-11-22. Reference
   costs: 8-node/32-GPU job ≈ 15.4 khours/day; a 5-day full finetune ≈ 77
   khours (~2.4% of remainder). Policy: cheapest-decisive-measurement first
   (CPU/1-GPU probes before short arms before full runs); no speculative
   full-scale launches; prefer scoring existing checkpoints over retraining;
   kill arms early when a pre-registered kill criterion fires.

## Standing evidence (do not re-derive)
- Drift unobservable from trajectory-only inputs: oracle recovers 83–93% of
  prevpred→GT gap, honest arms ~0%, velocity-extrapolation ceiling 20–25%.
- R lever = weighting, not iteration (RAFT-style iteration ruled out).
- Rung 0: raw GT at the GRU input makes things WORSE → scale normalization
  (P3.4) is a prerequisite.
- p43 v7 (untrained probe, OOD): incremental dR² of probe over velocity
  ~0.01–0.04 median per band, CIs graze 0 → untrained probe adds ≈nothing.
  Trained-probe results (p43_trained_e5/e10 vs placebo/shuffle) pending
  assessment (workflow ground:probe-evidence).

## Pre-registered STEP 1: noise-tolerance oracle (user directive, 2026-08-13)
No full finetune may launch before this curve exists. Eval-only on the
GT-conditioned checkpoint (B): inject noise into the conditioning pose at
magnitudes 0 → pose-head's own error level, trace metrics-vs-conditioning-error
for all 5 metrics. Run TWICE: white noise AND random-walk (temporally
correlated) noise of matched magnitude. Decides:
- Response-curve shape: monotone/shallow (partial GRU success = real metric
  gain) vs cliff (GRU must clear a bar) vs poisoned (B over-trusts
  conditioning; moderately-wrong conditioning WORSE than baseline A).
- White-vs-correlated gap = how much of the GT benefit is drift-anchoring
  (closed-loop GRU fundamentally struggles) vs local accuracy (realistic).
  Predicts which metrics move first — expect RPE before ATE.
- Trust calibration is trainable (escape hatch): training the integrated model
  on the GRU's actual imperfect outputs (not teacher-forced GT) converts
  cliff into graceful degradation — exposure-bias fix viewed from the other side.
Key advantage: harness is byte-deterministic and this is inference-only ⇒ the
curve has ZERO training scatter; every difference is real. Cost: a few eval
jobs, no training.

## Campaign log
- 2026-08-13: Campaign started. Preflight: branch=captain_gru_v3 confirmed;
  verify_plan_anchors.py rc=0, 22/43 anchors exact, 21 drifted (content
  authoritative, line numbers not). In-flight: jobs 44521973 (gru_f1r8np_ft)
  and 44521974 (gru_f1d8np_ft), 8 nodes each, ~16h elapsed.
- 2026-08-13: Design-lock workflow launched (run wf_9e736a0c-4cd):
  4 grounding readers → 3 designers → 3 adversarial judges → synthesizer.
  Output = ordered execution plan with pre-registered pass/kill criteria.

## Execution log 2026-08-13 (early morning)
Plan of record = design-lock workflow synthesis (wf_9e736a0c-4cd, full plan in
scratchpad tasks/wjnpaguic.output): Refine-50 backbone (r8 + refine_passes=2,
probe_every=1, single-variable vs scored gru_a4g3r8) behind the noise oracle.
Baseline table (4292 scenes): A ate .0759 rpet .00794 rper 1.101 absrel .1794
a1 .7863 | B ate .0134 rpet .00373 rper .337 absrel .1886 a1 .7866 | C ate
.0821 | gru_a4g3r8 ate .0799. Win rule: per-scene win% vs C on ATE AND
rpe_rot, >54.2% = progress, >63.0% = beats A; depth inside family band.
- Commits: 49f5819 (noise hook, unit-checked), d1b9828 (P4.3 mechanism +
  configs + instruments), 0cd7e83 (eval registration: refine levers
  whitelisted, unset-list extended, grad_norm logged, watcher .ignored),
  94ec8a5 (noise-oracle instruments + evidence).
- Verify battery: verify_gru_probe_pass + verify_gru_ray_gate exit 0
  (archived); preflight PASS on refine_short final; probe-block weight norm
  8.26 (alive).
- sigma_ref calibrated (zero GPU, from C's saved preds): t=0.0775 GT units,
  r=36.68 deg median per-view. Injection scaled /1.5382 so realized median
  matches grid.
- IN FLIGHT: 10 curve jobs 44547303-312 (diag_noise_*, 1 node each, subset
  430); smoke 44547317 (debug); p43 regen 44547322-326 (trained/placebo
  e5+e10, shuffle e10 on refine_short; acc_ehpc 1-GPU); f1r/f1d twins
  44521973/74 finish ~06:00, watcher auto-evals, then MANUAL:
  aggregate_and_merge.sh gru_a4g3f1rr8np gru_a4g3f1dr8np + build table.
- Analysis TODO when curve lands: B1-B4 readouts per plan Step 1 (shape,
  sigma_poison, walk/white D_ATE vs D_RPEt, exposure-bias note); sig0check
  must byte-match diag_noise_clean preds; then Step-6 GO/NO-GO.

## Gate results so far (2026-08-13 ~03:50)
- Step 5 smoke: ALL PASS (startup line refine_passes=2 probe_every=1 dim 21;
  probe block learned 0.0297; purity assert run exit 0 = throwaway probe holds
  in real train steps).
- Step 2c causal: PASS — probe shuffle degrades committed e_head_t
  +27/+49/+42% (early/mid/late), e_head_r +50/+41/+40%.
- Step 2d paired: PASS — trained-minus-placebo ddR2_t e5: early +0.112
  CI[+0.056,+0.172], mid +0.157 CI[+0.090,+0.223], late +0.137 CI[+0.009,
  +0.217]. Rotation ddR2 ~0 (known risk). At e10 ddR2_t collapses while
  shuffle-dependence + committed-error improvement persist = ABSORPTION
  (pre-registered Step-11 signature), not probe overfit.
- Step 1 curve: 10 diag jobs running since ~03:22 (44547303-312), ETA ~04:15.
  Analyzer + bytecheck ready (noise_oracle_analyze.py / _bytecheck.sh).
- REMAINING before Step-7 launch: byte-identity PASS, B1-B4 readouts,
  Step-6 GO/NO-GO record, wipe stale refine output dir, watcher registration.

## LAUNCH (2026-08-13 ~04:45)
- sigma=0 byte-identity: PASS (134,448 camera + 7,626 depth files, 0
  mismatches). Last Step-6 blocker cleared.
- STEP 7 LAUNCHED: job 44548347, 8 nodes/32 GPUs,
  captain_gru_v3_a4_g3_r8_refine_finetune (lr 1e-5, 50 ep, full set,
  single-variable vs gru_a4g3r8). ETA ~26h -> ~2026-08-14 ~07:00.
- Startup-line monitor armed (abort if "refine_passes=2 (probe_every=1)"
  absent). Step-8 kill gate due at checkpoint-10 (~5.5h in, ~10:15):
  (i) probe-block weight norm >0.01; (ii) paired clean-vs-shuffle e_head_t
  degradation >10% (100-scene probe set, 1 GPU); (iii) e_head trend at or
  below sigma_ref-equivalent. Fail => scancel 44548347.
- Watcher will auto-arm the eval (label gru_a4g3r8refine) on completion;
  then aggregate_and_merge.sh + build_gru_v3_table.sh + Gate 2 falsifiers
  + Gate 3 win% vs C (54.2%/63.0% tiers) + curve-consistency check.
- f1r/f1d twins finish ~07:00; their evals auto-submit; manual aggregate
  after (context rows, 0.00281 floor read).

## Next action on wake
Read the design-lock workflow result; execute its step list (implementation →
smoke → short arms → score vs gates → full 32-GPU launch). Keep this file
updated after every phase.

## f1r/f1d twins scored (2026-08-13 07:15, 4292/4292)
f1rr8np ate .0788 rot 1.1916 | f1dr8np ate .0814 rot 1.1991. Spread .00258
<= .00281 floor => pair ranks nothing; F-lever verdict unchanged. Win% vs C:
f1rr8np 55.6/54.4 (grazes 54.2 tier, within the +/-2-point replicate band,
indistinguishable from r8's 54.2/55.5) — recorded as floor-limited context,
NOT progress; refine line keeps priority. f1dr8np 49.4/53.8 fails.

## Checkpoint-10 kill gate (2026-08-13 09:45): RUN CONTINUES
(i) probe-block norm 9.50 PASS. (ii) shuffle degradation e_head_t +26.4%
mid / +21.7% late PASS. (iii) committed error 1.7-2.1x arm-A head — mid-
anneal vs annealed references, inconclusive by construction; kill branch
(B2 poison) never armed since no poison region exists. Risk carried: error
must anneal below sigma_ref-equivalent by e50; checkpoint-30 trend probe
scheduled (~19:30) to confirm the anneal slope before completion.

## Checkpoint-30 trend (2026-08-13 18:25): SLOPE CONFIRMS
e_head_t mid 0.592->0.383 (-35%), ratio vs arm A 1.90->1.23; all six bands
-27..-42%. 20 steepest-anneal epochs remain. Gate-(iii) risk retired.
Training ETA ~03:15 2026-08-14 (27 min/epoch); eval auto-arms on completion.

## DELIVERABLE SCORED (2026-08-14 04:00, 4292/4292) — PRE-REGISTERED NEGATIVE
gru_a4g3r8refine: ate .0903 rpet .00845 rper 1.3317 absrel .1981 a1 .7614.
Win% vs C: ATE 34.23, rot 26.82 — 20.0/27.4 points BELOW the 54.22 progress
tier, far outside the replicate bands (±4.0/±13.5) in the failing direction
=> clean single-run FAILURE, no replicate owed under PREREG_VERDICT_RULES.md.
Depth inside family band (pre-registered depth expectation held). Delta vs
gru_a4g3r8: +0.0104 ATE = 3.9x the corrected floor — a REAL regression, not
scatter. rpe_rot 1.332 = worst in family (open-risk #5, rotation
co-adaptation, likely fired). Ray-shuffle Gate-2 falsifier VOID at eval:
model.py:2028 requires batch>1; harness runs batch 1 (instrument no-op —
recorded as an instrument limitation, not arm evidence). Probe-shuffle
falsifier (batch-4 instrument) pending: p43_final_clean re-running after
transient OOM; p43_final_shuffle.json already on disk.

## B CLOSED-LOOP COLLAPSE (2026-08-14 08:10) — the curve's caveat BINDS
diag_gtrayB_prevpred430: ate .1307 rpet .01346 rper 3.291 absrel .2242 a1
.6727 — collapse below every baseline; 3x worse than the walk-1.0 curve
prediction (.0425). Converges with the concurrent-session finding
(gru_a4g3otr honest eval: ate .1192, rot 4.21). VERDICT TRIAD now complete:
(1) closed-loop-trained arms co-adapt conditioning to ~zero-or-negative net
value (refine: .0903); (2) accurate-conditioning-trained trunks collapse
when self-fed (B: .1307; oracle-trained: .1192); (3) exogenous-noise
tolerance does NOT price endogenous feedback error — the "no poison region"
was an artifact of exogenous injection. Frozen-B + GRU add-on is dead on
arrival. Surviving untested design: noise-conditioned TRAINING (train the
trunk on GT-conditioning corrupted with walk noise matched to deployment
error statistics — the user's trust-calibration escape hatch, applied at
trunk level). Cheap decisive rung: 10-epoch short arm (~5.5 khours), scored
closed-loop on the subset, BEFORE any 50-epoch bet.

## NGC-WalkDrift short rung LAUNCHED (2026-08-14 ~10:30, job 44583714)
All pre-launch falsifiers PASS (F1-F5 battery 20/20 byte-identity + parity,
archived; F6 drift fit 1.23x naive; F7 graft loads with refine sniff + 8
scenes scored; F8 config diff exact; F9 t_frac round-trip). Gates frozen in
NGC_WALKDRIFT_GATES.md BEFORE launch. 10 epochs, 8 nodes, trunk-pure,
label gru_ngcwalkdriftshort marked .ignored (rung, never a table arm).
Next: ckpt-5 early-out (closed-loop subset vs .1307), then G1/G2/G3/G4 on
ckpt-10, then S1-or-promotion per the frozen table.

## NGC ckpt-10 gate battery SUBMITTED (2026-08-14 13:05, jobs 44592185-88)
Storage note: checkpoints moved (new runs -> /gpfs/scratch/.../checkpoints,
old -> /gpfs/scratch/.../checkpoints_projects; mv from /gpfs/projects still
draining — ckpt-10 read explicitly from the projects-side source). Battery:
G1 diag_ngc10_clean (prev_gt), G2 diag_ngc10_drift10 (walk-1.0 + frozen
warmstart drift 0.0362/0.34), G3 DECISIVE diag_ngc10_closed (prev_pred),
shuffle spot check (LIMIT=5). R-denominator replays at 0.5/1.5x wait for the
checkpoint-specific drift fit from G3's own residuals. Claim message sent to
peer session (koc821022-6b) to avoid duplicate gate work; a4_g1 twins' evals
left to the peer unless they delegate.
