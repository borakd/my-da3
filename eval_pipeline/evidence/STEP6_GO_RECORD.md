# Step-6 GO/NO-GO record — 2026-08-13 ~04:20 CEST

**DECISION: GO** for the Step-7 deliverable launch
(`captain_gru_v3_a4_g3_r8_refine_finetune`, 8 nodes / 32 GPUs, lr 1e-5,
50 epochs, full dataset), with the Step-8 checkpoint-10 kill gate EXTENDED
(see below) beyond what the branch table strictly requires.

## Prerequisites (all PASS, artifacts in eval_pipeline/evidence/)
- Step 2c causal: probe shuffle degrades committed e_head_t +49.3% mid /
  +42.4% late (rot +40-50%) — channel load-bearing. (p43_regen_gates log)
- Step 2d paired: trained-minus-placebo ddR2_t e5 mid +0.157 CI95
  [+0.090, +0.223] excludes zero. e10 collapse + persistent shuffle
  dependence + improving committed error = ABSORPTION signature.
- Step 3: preflight PASS on refine checkpoint with levers restored;
  falsifier/noise envs added to the harness unset list.
- Step 5 smoke: startup line `refine_passes=2 (probe_every=1)` dim 21;
  probe columns learned (0.0297 after 2 ep); purity assert exit 0 in real
  train steps.
- sigma=0 byte-identity: 0 mismatches through the full camera sweep
  (final count in the bytecheck log; launch was gated on its completion).

## Noise-curve verdicts (noise_curve.json; 430-scene subset, paired, zero
training scatter by harness determinism; sigma_ref_t=0.0775, sigma_ref_r=36.7°)
- B1 SHAPE = INTERMEDIATE. Monotone, graceful; white-noise gap loss
  5.7/18.1/46.3/97.5% at 0.125/0.25/0.5/1.0 sigma_ref. No cliff: no hard
  accuracy bar below which conditioning is worthless.
- B2 POISON = NONE. No sigma up to 1.0 sigma_ref pushes noisy-B ATE above
  arm A (white 1.0: 0.0755 vs A 0.0771; walk 1.0: 0.0425). No
  "confidently misled" region above A ⇒ the CLIFF/poison protections do
  not bind.
- B3 DRIFT: D_ATE = 0.39, D_RPEt = 0.05 — correlated (walk) error is
  ~2.6x MORE tolerable on ATE than white, and barely moves RPE. The
  closed-loop GRU's errors are the correlated kind ⇒ operating point is
  the forgiving branch; ATE closure from partial accuracy is realistic.
  The drift-anchoring-dominant signature (D_ATE>=2) did NOT fire.
- B4: not CLIFF ⇒ no mandatory record; noted anyway that the refine arm
  trains on its own in-loop outputs (no GT teacher-forcing in any
  conditioning path) — the exposure-bias fix holds by construction.

## Branch-table reading
B1 fell between the two named branches (SHALLOW-GO / CLIFF-short-first).
The protections those branches encode are both satisfied: no A-crossing
(the CLIFF branch's fear) and proportional returns to accuracy (the
SHALLOW branch's promise). GO is therefore taken on the branch table's
intent, and the B2-poison branch's EXTRA safeguard is adopted although
its trigger did not fire:

## Step-8 kill gate (checkpoint-10, ~5.5h in) — extended
1. ||cell.weight_ih[:,14:21]|| > 0.01 (dead-window guard);
2. paired clean-vs-POSE_GRU_PROBE_SHUFFLE committed e_head_t degradation
   > 10% on the 100-scene probe set;
3. ADOPTED EXTRA: probe-measured committed e_head (normalized) trending at
   or below the honest-head reference (sigma_ref-equivalent), where the
   curve guarantees >=50% gap retention under correlated error.
Fail any ⇒ scancel (saves ~20h), record the negative, fall back to the
curve+probe endgame.

## Caveat carried forward (open risk #1 of the plan)
The curve measures B's tolerance to EXOGENOUS noise around GT. The refine
arm's trunk trains closed-loop on its own conditioning distribution; the
curve is a directional gate, expected conservative for an
exposure-bias-trained arm, not a point forecast.
