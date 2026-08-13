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

## Next action on wake
Read the design-lock workflow result; execute its step list (implementation →
smoke → short arms → score vs gates → full 32-GPU launch). Keep this file
updated after every phase.
