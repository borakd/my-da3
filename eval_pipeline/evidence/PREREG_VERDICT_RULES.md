# Pre-registered verdict rules for gru_a4g3r8refine — FROZEN BEFORE THE SCORE EXISTS
Written 2026-08-14 ~00:15 CEST. Job 44548347 is at ~epoch 44; no eval output
for this arm exists anywhere. These rules amend the campaign state per the
adversarial stats audit (wf_dfe5f36f-abf) and supersede the unpinned
"+/-2-win-point" band, which was never pre-registered and is hereby struck.

## 1. Headline verdict (unchanged, as registered in the directive)
Per-scene win% vs arm C (prevpred_lr1e5) on ATE and RPE-rot over the 4292
common scenes, strict '<', computed by the pinned instrument
eval_pipeline/win_rate.py (asserts n==4292 and zero NaNs, else fails loudly).
- > 54.22% on BOTH  => "progress over gru_a4g3r8"
- > 63.02% on BOTH  => "RESULT: beats arm A"
(Values re-derived from raw CSVs by the audit: A-vs-C ATE 63.02%,
gru_a4g3r8-vs-C 54.22% ATE / 55.45% rot; zero ties, zero NaNs.)
Depth read: AbsRel/a1 within the GRU-family band (0.194-0.201 / 0.757-0.773)
= "no depth harm" (pre-registered: depth is not expected to move).

## 2. Noise floor — CORRECTED
The directive's 0.00281 is the upper-median (15th of 28 order statistic).
The standard median of the 28 same-family arm-pair |mean paired ATE diffs| is
0.00264. OPERATIVE FLOOR: 0.00264 (report both). Audit confirmed no existing
decision flips in the window [0.00264, 0.00281] (twins spread 0.00258 still
inside; all "floor-limited" calls unchanged).

## 3. Replicate-trigger band — DERIVED, replacing the struck +/-2
Calibration: pairwise |win% difference| vs C across the 24 arm pairs whose
mean ATE difference is <= the corrected floor (training-noise-equivalent
pairs; full table in the audit log and reproducible from summary CSVs):
- ATE win%: median 2.90, p75 3.87, max 6.17
- rot win%: median 10.03, p75 13.21, max 24.51
BAND: +/-4.0 win-points (ATE), +/-13.5 win-points (rot) — the p75 values,
rounded conservatively outward. Caveat recorded: these pairs differ in
configuration, not only seed, so the band is an UPPER bound on pure seed
noise; using it makes replicate-triggering MORE likely, never less.
APPLICATION (symmetric, verdict-neutral): if the refine arm's win% on either
metric lands within its band of EITHER tier boundary (54.22 / 63.02), in the
passing OR failing direction, a seed replicate is REQUIRED before the tier is
claimed or denied. The band never flips a verdict by itself. Clearing a
boundary by more than the band on both metrics = tier claimable from the
single run (mean-delta floor check in §2 still applies to any mean-metric
claim).

## 4. Checkpoint-10 gate (ii) band reading — clarified post-hoc, flagged
The gate was registered without a view-band qualifier; the measured early
band was +7.9% (below 10%), mid +26.4% / late +21.7% (above). The PASS was
recorded on mid+late, consistent with the Step-2c causal-gate precedent
(mid+late), but this qualifier was not in the registration text. Recorded
here as a transparency item; the gate is not retroactively altered.

## 5. Instrument corrections applied with this commit
- p43_regen_compare.py: three-zone causal verdict (KILL <5% / INCONCLUSIVE
  5-10% / PASS >10%) now encoded; paired bootstrap asserts identical scene
  key sets across arms.
- noise_oracle_analyze.py: per-point statistics now computed strictly on the
  paired scene set; <100% coverage at any grid point is a hard failure
  rather than a silently biased readout.
- run_captain_ray_eval_node.sh: CONDITIONING is now required (no 'gt'
  default); the worker path already receives it from preflight derivation.
- demo_ray.py walk scaling: sqrt(2/(T+1)) makes walk sequence-RMS ~0.8%
  weaker than white at matched sigma (ratio sqrt(64/65)); documented, not
  changed — the existing curve's B3 margin (0.39 vs 2.0 threshold) dwarfs it
  and changing the generator would invalidate the scored curve points.
- ORACLE env: added to the eval unset list and pinned off in the export
  (closed before any deliverable eval fires; watcher env verified clean).

## 6. Curve-consistency check (unchanged)
Map the arm's measured conditioning error (probe e_head vs GT, converted via
the sigma_ref calibration) onto the noise curve to predict delta-ATE vs B;
report predicted vs realized. Agreement = mechanism evidence; disagreement
is recorded and investigated before any claim.
