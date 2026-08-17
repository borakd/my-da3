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
- 2026-08-17 ROUND-2 RESULTS (evidence/cg_round2_430.json):
  - cap_p90 (trust-region, only 1.9% mean atten): absrel −.00060*, a1
    +.00164*, ATE −.00028, rpe_trans −.000003, rpe_rot +.0185 n.s.
    ⇒ 4/5 better in the mean, both depth wins SIGNIFICANT, no significant
    loss. Capping the top-decile state rewrites is nearly a free win.
  - soft_g5 (gmin .5): ATE −.00419* + a1 +.00173* but rpe pair still
    significantly worse (+.053° rot) — partial staleness still costs,
    ~10× less than hard skip.
  - state-only scope ≈ both-scope (state_feat is the whole story);
    inv_dst (hard-skip top-1% rewrites) dominated by cap.
- 2026-08-17 round-3 launched (44696338-43): combo1 (soft g.5 τ p30 ×
  cap .1837), combo2 (soft g.7 × cap .21), soft_g7, cap_p95, cap_p98,
  soft_g5t3 (τ p20, temp 3 — smoother sigmoid).
- 2026-08-17 ROUND-3 RESULTS (evidence/cg_round3_430.json): the 3/5
  majority is REACHED with all three wins significant —
  - combo1: absrel −.00106* a1 +.00209* ATE −.00414* | rot +.0534* (+4.8%)
    trans +.00046* (+5.6%)
  - soft_g7: absrel −.00085* a1 +.00185* ATE −.00293* | rot +.0326* (+2.9%)
    trans +.000149* (+1.8%)
  - combo2: 3 sig wins | rot +.0281* (+2.5%) trans +.000125* (+1.5%)
  Frontier ≈ linear in mean attenuation (ATE ≈ −.021×atten, rot ≈
  +.26×atten); cap .1837 adds depth wins ~orthogonally (combo1 vs soft_g5:
  same pose deltas, absrel win +50%). cap_p95/p98 shrink to washes.
- 2026-08-17 round-4 launched (44697524-27): low-atten frontier c_g75/g8/g85
  (combo, gmin .75/.80/.85 × cap .1837) + g7ema (EMA .7 signal smoothing,
  jitter-mechanism test). STATE_GATE_EMA added to hook + unset list.
- 2026-08-17 ROUND-4 RESULTS (evidence/cg_round4_430.json): EMA WORKS —
  g7ema: absrel −.00101* a1 +.00213* ATE −.00316* | rot +.0155* (+1.4%)
  trans +.000205* (+2.5%). Halves the rot cost of soft_g7 while growing
  every win; dominates the whole combo-gmin family (c_g8: ATE −.0018,
  rot +.0249). Flicker-jitter mechanism confirmed.
- 2026-08-17 round-5 launched (44699254-57): g7emacap (EMA×cap), g7ema85
  (EMA .85), g5ema (aggressive+smooth), g6ema8cap (middle). Winner → full
  4292 promotion.
- 2026-08-17 ROUND-5 RESULTS (evidence/cg_round5_430.json): EMA×cap boosts
  depth but doubles rot cost; EMA .85 pose-safest but depth n.s.; g7ema
  stays champion. Losing diag arms' depth/camera preds purged (~700 GB);
  telemetry + eval CSVs kept.

## FINAL VERDICT (2026-08-17, evidence/cg_full4292_verdict.json)

Full 4292-scene harness, paired vs augfull_lr1e5, byte-deterministic:

**augfull_cg_g7ema** (STATE_GATE_MODE=soft, SIGNAL=conf_mean, TAU=−11.99,
TEMP=1.5, GMIN=0.70, EMA=0.7, WARMUP=1; mean attenuation 11.7%, zero hard
skips):
| metric | clean | gated | Δ | CI | w/l |
|---|---|---|---|---|---|
| absrel | .179384 | .178647 | −.000736* | [−.000999,−.000470] | 2275/2017 |
| a1 | .786312 | .787188 | +.000875* | [+.000491,+.001257] | 2198/2094 |
| ate | .075869 | .072802 | −.003067* | [−.003389,−.002740] | 2686/1606 |
| rpe_trans | .007941 | .008091 | +.000150* | [+.000125,+.000175] | 1871/2421 |
| rpe_rot | 1.100981 | 1.106619 | +.005638 n.s. | [−.000571,+.011915] | 2020/2272 |

WIN CONDITION MET: 3/5 metrics significantly better (absrel, a1, ATE −4.0%
with 62.6% per-scene win rate); rpe_rot statistically flat (+0.5%);
rpe_trans +1.9% (the one small cost). augfull_cg_g7ema85 (EMA .85) is the
confirming sibling: absrel −.000801*, a1 +.000913*, ATE −.002766*, rot
+.004942 n.s., trans +.000146* — same shape, so the operating point is not
a lottery ticket. Full-harness costs came in SMALLER than the 430 subset
predicted (rot +1.4%* → +0.5% n.s.).

Both rows merged into summary/averages_table.csv (honest inference-time
arms; no oracle information — the gate signal is the model's own conf).

### Why this worked where the GRU could not (one paragraph)
The GRU campaign died because drift is unobservable from trajectory-only
inputs. This gate never estimates drift: it only needs to detect frame
badness, and image-derived confidence carries that signal (Spearman −.6 to
−.8 vs per-scene error). Skipping writes trades staleness for wrongness —
and the prev_gt result (staleness nearly free, wrongness compounds) said
that trade is favorable. The two refinements that made it a clean win:
(1) floor attenuation (GMIN) so no frame's state goes fully stale (the
attribution showed stale frames themselves pay the RPE cost), and (2) EMA
on the signal so gate flicker cannot inject frame-to-frame jitter.
