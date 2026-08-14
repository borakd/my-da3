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
Storage correction (peer session 13:15): the checkpoints_projects mv DIED
partway (RLIMIT_CPU on glogin1) — cut3r_multinode never copied to scratch;
projects side is source-of-truth for it, READ-ONLY (group over quota, 1.02TB
/1024GB, 6.9-day grace). a4_g1 twins are NEW runs at
/gpfs/scratch/.../checkpoints/captain_cut3r_finetune_aug_full/; their evals
resubmitted with CKPT_ROOT override: gru_a4g1f1np + gru_a4g1f1r8np, 8 shards
each (44592216+/44592234+). NGC battery unchanged (reads projects source).
Coordination (3rd session, known_good_fsrc_noproj, ~13:40): a4_g1 twin evals
run to completion under my labels but AGGREGATION IS THEIRS (single-pass
merge incl. eval_gru companions; my-da3's aggregate_results.py drops _gru
rows — do not merge from this worktree). They also score the two oracle
lr1e5 arms ORACLE=gt as user-directed table toplines (unmissable labels,
--no_rank). My remaining ownership: NGC ckpt-10 gate battery + readouts.

## NGC ckpt-10 GATE RESULTS (2026-08-14 14:10) — RUNG KILLED
Frozen bars from NGC_WALKDRIFT_GATES.md; subset refs A .0771 / C .0850 /
r8 .0816 / refine .0929 / B-closed .1307.
- G1 clean ceiling (prev_gt): ate .0663 — KILL (bar: PASS <=.045, KILL >.060).
  The noise-trained trunk retains little of the conditioning benefit even on
  CLEAN prev-GT rays.
- G2 drifted replay walk-1.0: delta_ate +.0533 vs clean — NO FLATTENING
  (bar <=.015). Trained on walk+drift noise, still detonates on walk+drift
  replay: the curriculum did not transfer even within its own noise family.
- G3 DECISIVE closed-loop (prev_pred): ate .1126 — KILL (bar >.0850).
  Between C-class failure and B-closed collapse.
- Shuffle spot check: VOID — paired delta exactly 0.00000 on 20/20 scenes =
  instrument no-op (known batch-1 limitation, model.py ray-shuffle requires
  batch>1). NOT evidence of a dead channel: the +.0533 replay delta proves
  the channel is alive.
- S1 stop rule does not technically fire (requires G2 flattening), but the
  measured conclusion is stronger: noise-conditioned training BOTH loses the
  clean ceiling AND buys no robustness. No retune exception applies (the
  CONDITIONAL band .0771-.0850 was not reached; no early rot divergence).
- G4/R-replays: moot per frozen ladder (P1 requires G3 PROMOTE); not run
  (GPU-awareness).
DESIGN MATRIX NOW MEASURED-DEAD: (1) A/G/F/R grid — function class lacks
target; (2) refine/P4.3 — real observation, co-adapts net-negative; (3)
teacher-forced trunks — collapse self-fed; (4) noise-bridged trunk — trades
ceiling away, gains nothing. Campaign is at its pre-registered negative
endgame; further GPU spend requires a user scope decision.
Verdict-doc dependency (14:30): hold the final dossier's quantitative section
for the FOUR-CORNER SQUARE — honest-trained/honest-eval (C .0821),
honest-trained/GT-fed (rung 0 / curve), oracle-trained/honest-eval
(gru_a4g3otr .1192), oracle-trained/GT-fed (gru_a4g{1,3}oraclegtfed, in
flight ~650/4292, peer my-da3-ca will send numbers). The matched corner
bounds every "if the corrector were accurate" counterfactual. User decision
pending: accept verdict / P4.2 last family / constraint relaxation. No new
GPU from this session until decided.

## Four-corner square COMPLETE (peer delivery, 2026-08-14 ~15:20; 4292/4292,
## banners verified, 16/16 jobs clean)
ATE, a4/G3 family: honest/honest .0837 | honest/GT-fed .1104 |
oracle/honest .1192 | oracle/GT-fed .0133 (= gtray topline .0134 exactly).
Both matched corners work; both mismatched corners fail SYMMETRICALLY —
the cleanest statement of the train/test conditioning-distribution principle
the campaign has. Cautions for the dossier (peer's, verified sound):
1. Oracle-fed _gru rotation rows (0.044/0.039 deg) are ECHO, not capability
   (residual mode, input IS GT, trained residual~0). Only honest _gru rows
   are interpretable.
2. Ceiling is exactly the GT baseline, no headroom beyond: oracle-fed head
   ate .0133 vs gtray .0134; rpe_rot WORSE through the GRU (0.446 vs 0.337).
3. NO DEPTH HEADROOM behind an accurate corrector: absrel .1876 with perfect
   pose vs .1794 unconditioned — the counterfactual is pose-only.
4. G1-vs-G3 null under oracle feeding (.0131 vs .0133) — e2e lever inert
   once input is correct.
Pending from peer: honest _gru columns for a4_g1 twins (~2146/4292).

## Honest _gru columns COMPLETE (peer delivery, 2026-08-14 ~16:10) — QUANT
## SECTION CLOSED
a4_g1 twins (4292/4292, banners verified, 0 oracle contamination):
f1np head ate .0828 / gru .0823, rot 1.263 -> 1.326; f1r8np head .0800 /
gru .0794, rot 1.215 -> 1.277. Across ALL SEVEN instrumented honest arms:
GRU pose is a WASH on ATE (deltas ±0.0006, sign-inconsistent) and WORSE on
rotation in every non-degenerate case (+0.055..+0.089 deg). The corrector
moves rotation the wrong way while not helping translation — at iters 1 and
8, with/without features, G1 and G3.
Complete campaign statement: ray channel worth 6.3x ATE when pose correct
(.0837->.0133); ceiling exactly gtray, no further; depth zero headroom;
GRU-as-built degrades rotation; and train/test conditioning distributions
must match or BOTH directions collapse symmetrically.
G1-vs-G3 matched pair (F1 noproj): e2e buys ~.002 ATE, costs ~.002 AbsRel —
below attribution floor, not a lever effect.
Table: summary/fresh_table_gru_pose.* (24 rows) with a real generator now at
eval_pipeline/build_gru_pose_table.{py,sh} (validated by regenerate+diff).

════════════════════════════════════════════════════════════════════════
## ORIENTATION FOR FUTURE SESSIONS (written 2026-08-15, campaign paused)
════════════════════════════════════════════════════════════════════════

READ THIS FILE TOP-TO-BOTTOM before touching anything GRU/conditioning-
related; it is the authoritative chronicle. Then read, in order:
1. eval_pipeline/evidence/STEP6_GO_RECORD.md — how the refine launch was
   gated (the method template every later rung followed).
2. eval_pipeline/evidence/NGC_WALKDRIFT_GATES.md — frozen-gates pattern +
   the S1 stop rule.
3. eval_pipeline/evidence/FINAL_DOSSIER.md — refine autopsy.
4. REPORT_2026-08-13.md — plain-language history of every lever (its
   original ending is superseded by the addendum at its foot).

### Where the campaign stands (as of 2026-08-15)
ALL FOUR design families are measured-dead (grid / probe-pass refine /
teacher-forced trunks / noise-bridged trunk). The complete empirical
statement is in the "Honest _gru columns COMPLETE" section above. GPU spend
is PAUSED awaiting the user's choice: (a) adversarial audit + final dossier,
(b) P4.2 cross-attention short rung (last untested observability route,
weighs against finding 3), (c) constraint relaxation (probe-informed gate is
the one untrained variant). DO NOT launch training arms before that choice.

### What this session (2026-08-13→15, "GRU v4 autonomous campaign") built
Commits 49f5819..7485e5d on captain_gru_v3, roughly in order:
- 49f5819 noise hook: GT_RAY_NOISE_* env-gated SE(3) perturbation in
  src/CUT3R/demo_ray.py (later extended by a peer with DRIFT_* terms).
  Unit-checked: sigma=0 byte-identity, white/walk seq-RMS matched.
- d1b9828 committed the P4.3 probe-pass mechanism (model.py,
  train_cut3r_baseline.py, refine configs, probe instrument, verifier).
- 0cd7e83 eval-path registration: refine levers whitelisted in
  preflight_ckpt.py, falsifier+noise envs added to arm_eval_for_run.sh
  unset list, grad_norm logging (inference.py returns max chunk norm),
  watcher .ignored state.
- 94ec8a5 noise-oracle instruments: noise_oracle_calibrate.py (sigma_ref
  from arm C's saved preds, zero GPU), noise_oracle_launch.sh (grid),
  430-scene subset list.
- fea153f noise curve + STEP6_GO_RECORD (B1-B4 readouts).
- ckpt10_kill_gate.sh, p43_regen_compare.py, noise_oracle_analyze.py,
  noise_oracle_bytecheck.sh — the gate/readout instruments.
- bd3d698 NGC ckpt-10 gate verdict (KILL); bcd46af path migration commit
  (peers' sed); 880d71a/7485e5d four-corner square + honest _gru closure.
(Table generator build_gru_pose_table.{py,sh} and eval_gru companion
instrument are PEER work from known_good_fsrc_noproj, not this session.)

### Traps future sessions must not rediscover (all verified here)
- Shuffle-type falsifiers are batch-gated: GT_RAY_MAP_SHUFFLE and the ray
  shuffle in model.py need batch>1; the 4292-harness runs batch 1 => a
  paired delta of exactly 0.00000 means INSTRUMENT VOID, not dead channel.
- Weight-norm check must precede any probe-shuffle reading (zero-init dead
  window).
- getattr-defaulted config keys: an uncommitted-lever arm launched from a
  committed-only tree silently trains the base arm under the new name —
  always verify the startup log line.
- diag_ labels never enter averages_table.csv; my-da3's aggregate_results
  drops _gru rows (merge from known_good only).
- Storage: new runs -> /gpfs/scratch/.../checkpoints; old ->
  /gpfs/scratch/.../checkpoints_projects; cut3r_multinode still ONLY on the
  /gpfs/projects source side (mv died; group over quota — READ-ONLY there).
- Exogenous noise tolerance does NOT price endogenous feedback error. Any
  future "condition on a predicted pose" idea must be scored closed-loop
  from day one; the noise curve alone WILL mislead.
