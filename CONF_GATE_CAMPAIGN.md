# Confidence-gated memory writes — campaign chronicle (started 2026-08-17)

User directive: find out whether "frame is damaging to the memory" is
detectable via confidence, gate/drop such frames, and beat the regular CUT3R
finetune baseline (`augfull_lr1e5`) on a majority of {absrel, a1, ATE,
RPE-rot, RPE-trans} without noticeable regression on the rest. Inference-time
only; no training arms (respects the GRU-v4 pause; this is a sibling of the
recorded open option (c) "untrained gate").

## Target (full 4292 harness, from summary/averages_table.csv)

| arm | absrel | a1 | ate | rpe_trans | rpe_rot |
|---|---|---|---|---|---|
| augfull_lr1e5 (beat this) | .179384 | .786312 | .075869 | .007941 | 1.100981 |

Directions: a1 higher-better, all others lower-better. Depth has ~no headroom
from pose-side changes (gtray oracle absrel .1886 > A's .1794), so the
expected win route is the three pose metrics; skipall smoke shows memory also
feeds depth, so absrel/a1 must be watched, not assumed safe.

## Mechanism (commit 8bb550e)

`STATE_GATE_*` env hook at the state/mem commit in
`src/CUT3R/src/dust3r/model.py::_forward_decoder_group_step`. Frame i's own
predictions are already final when its write is gated — only what frame i+1
sees changes. Modes: log / thresh / soft / rel; signals in log(conf−1) space
(conf_mean, conf_p10, conf_p50, conf_self_mean, conf_self_p10, dstate, dmem);
sub-vars WARMUP (default 1), MAX_SKIP, SCOPE (both|mem|state), TAU, TEMP,
REL_ALPHA. Telemetry per frame → `preds/<scene>/state_gate.json`
(config-stamped). OFF ⇒ byte-identical (verified).

Instruments: `eval_pipeline/cg_analyze.py` (paired per-scene deltas vs
`per_scene_augfull_lr1e5.csv` — legitimate clean reference because the
harness is byte-deterministic; re-confirmed 12/12 exact metric ties),
`cg_calibrate.py` (signal distributions + midrank-Spearman vs clean per-scene
metrics), `cg_bytecheck.sh`.

## Verified-trap compliance

- Gate acts at batch 1 and raises loudly otherwise (no GT_RAY_MAP_SHUFFLE
  silent void). Proven by skipall smoke: all 5 metrics significantly worse.
- STATE_GATE_* added to arm_eval_for_run.sh unset list ⇒ full-eval gate arms
  must be launched with an explicit --export list (NOT via arm_eval_for_run.sh,
  which now strips them — by design, protecting honest arms).
- diag_ labels never touch averages_table.csv.
- Closed-loop from day one: every readout is the real harness loop.
- Alternate forward paths (_da3_forward_impl, inference_step,
  forward_recurrent) carry tripwires that raise if STATE_GATE_MODE is set.

## Ledger

- 2026-08-17 smoke (12 scenes, ckpt=augfull final, CONDITIONING=none):
  - diag_cgsmk_log: exact no-op, 12/12 ties all metrics. Byte-check vs clean:
    8486/8486 files identical. Clean vs skipall: 4504 diffs (only frame-0/
    warmup files match). Clean vs ORIGINAL augfull full-run preds: 4532/4532
    identical — cross-session determinism certified, so
    per_scene_augfull_lr1e5.csv is a valid paired clean reference forever.
  - diag_cgsmk_skipall (thresh tau=999, freeze mem after frame 0): absrel
    +.0297*, a1 −.0575*, ATE +.0611*, rpe_trans +.0297*, rpe_rot +3.65°* —
    memory is load-bearing for BOTH pose and depth on this image-only arm.
- 2026-08-17 diag_cg_log430 (job 44691188): 430/430 scenes, 42 min, 0 fail.
  Calibration (evidence/cg_calibration_430.json, 134,448 frames): the
  confidence family is ALIVE at scene level — midrank-Spearman of per-scene
  signal means vs clean per-scene metrics: conf_mean −.64 ATE / −.61 rpe_rot /
  −.58 absrel; conf_self_mean −.80 absrel / +.81 a1; dstate +.53 rpe_trans /
  +.46 rpe_rot. (Contrast: the trajectory-only harm classifier was AUC .50.)
  Caveat held open: per-scene correlation ≠ per-frame causal harm; the grid
  is the intervention test. Pooled conf_mean percentiles used as taus:
  p5 −15.75 / p10 −14.68 / p20 −13.26 / p30 −11.99; conf_self_mean p10 −10.10.
- 2026-08-17 round-1 grid launched (jobs 44692260-67), 430 scenes each,
  WARMUP=1 MAX_SKIP=8: th_p05/p10/p20/p30 (thresh conf_mean at pooled
  percentiles), rel_a2/rel_a4 (per-scene adaptive, alpha in log units),
  mem_p20 (scope=mem dissection), self_p10 (conf_self_mean signal).
- 2026-08-17 ROUND-1 RESULTS (evidence/cg_round1_430.json, all paired vs
  per_scene_augfull_lr1e5 on the 430):
  - ATE improves monotonically with skip rate; rel_a2 (54% skip) ATE
    −.00554 CI[−.0078,−.0033]* with 253w/16t/161l — hypothesis CONFIRMED
    for global trajectory: dropping low-conf frames' writes helps ATE.
  - RPE degrades ~linearly with skip fraction (rpe_rot +.11°…+.48°*, worst
    at rel_a2) — suspected mechanism: post-skip frames decode from stale
    state ⇒ local pose discontinuities. THE tension of the campaign.
  - scope=mem is a wash on everything ⇒ the whole effect (help and harm)
    flows through state_feat, not the pose-retriever mem.
  - Depth: neutral at low skip, mildly worse* at ≥20% skip.
- 2026-08-17 hook v2 (STATE_GATE_GMIN floor for soft, mode=cap trust-region
  g=min(1,tau/dstate), STATE_GATE_INVERT for high-signal gating); round-2
  arms launched (44694245-50): soft_g3, soft_g5, cap_p90, cap_p70,
  state_p20 (dissection), inv_dst (skip top-1% rewrites). Attribution
  analysis of per-frame RPE vs skip adjacency running concurrently.
- 2026-08-17 ATTRIBUTION (evidence/cg_frame_attrib.json, cg_frame_attrib.py):
  per-frame RPE deltas by class — accepted-after-accepted ≈ 0 (median exactly
  0); every stale-state frame pays (+0.36…+1.0° rot), worst at the FIRST
  frame of a skip run; forced writes +0.5…0.75°. Partial correlations: harm
  tracks n_skipped/run-length, NOT transition count (more transitions at
  fixed skips slightly protective). ATE gains live in low/mid-conf scenes
  (rel_a2 terciles −.0073/−.0100/+.0006). ⇒ floor-attenuation (soft+GMIN)
  is the mechanistically indicated fix; "clustered skips" contraindicated.
