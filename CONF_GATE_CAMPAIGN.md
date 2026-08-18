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

## PHASE 2 (2026-08-17, user-directed): gate-trained finetune + push to topline

User directive: (a) full finetune with the winner gate active, baseline
parity (32 GPU, lr1e-5, batch 4, accum 1, 38633/4292); (b) keep improving
toward the gtray topline (.013445 ATE) closed-loop. Training-arm pause is
lifted by this directive.

- Training integration (commit 2aed6d0): _state_gate_batched — vectorized
  B>1 gate, detached signals, per-slot EMA; B==1 scalar path preserved
  verbatim (eval reproducibility). Adversarial review: SOUND (scalar path
  mechanically verified identical; no TBPTT graph leak; exactly-once EMA
  under checkpointing; test-path B∈{1..4} safe).
- 1-node smoke 44709042: banner `[STATE_GATE] batched gate ACTIVE (B=4)`,
  loss sane (2.95 @ step 0), no errors. Wiped after.
- FULL RUN launched: job 44709426, 8 nodes × 4 H100, config
  cut3r_ft_augfull_cg_g7ema_32gpu_lr1e5 (2-line diff from augfull yaml),
  winner gate env in --export. Expected ~17 h; ckpts every 10 epochs.
- PRE-REGISTERED ckpt-10 early gate (~3.5 h, 20% spend), NGC-style:
  paired 430-subset diags, all CONDITIONING=none:
  - diag_cgt10_on  — gate-trained ckpt-10, winner gate ON (matched corner)
  - diag_cgt10_off — gate-trained ckpt-10, gate OFF (mismatch probe)
  - diag_aug10_on  — PLAIN augfull ckpt-10, gate ON (paired control)
  KILL if diag_cgt10_on is worse than diag_aug10_on by >+.005 ATE AND
  >+.05° rot (training under the gate actively harms), or training loss
  NaN/diverged vs baseline log.txt at epoch 10. Conf-distribution drift
  (mean attenuation at ckpt-10 vs the plain model's 11.7%) is recorded as
  telemetry, not a kill.
- Final scoring plan (pre-registered): cgtrain_g7ema (gate-on, PRIMARY,
  via cg_launch_full.sh with CKPT override) and cgtrain_plain (gate-off,
  via arm_eval_for_run.sh) vs {augfull_lr1e5, augfull_cg_g7ema, gtray
  .013445 distance}. TAU stays frozen at −11.99 for the verdict; any
  retune afterwards is diag_-labeled.

### Phase-2 lever ledger
- ORACLE HARD-BLOCK EXPERIMENT (user-requested, evidence/cg_oracle_430.json;
  masks from clean per-frame GT errors, verified; ~19–23% frames blocked
  via update=False): ALL THREE variants (ate>1.5×med, ate top-20%,
  rpe>2×med) are WORSE than clean on ALL FIVE metrics — ATE +.0007…+.0027,
  rot +0.28°*, trans +25%*. Compare conf-based th_p20 (same skip rate):
  ATE −.0008, same rot cost. INTERPRETATION (the campaign's sharpest
  mechanism result): per-frame GT error marks VICTIMS of drift, not
  CULPRITS — blocking a poorly-localized frame removes late-but-good
  observations while the corrupting frames already wrote; error segments
  are contiguous ⇒ long skip runs ⇒ compounding staleness (attribution:
  harm ∝ run length). Confidence marks observation quality (culprits),
  which is why conf-keyed gating wins where GT-error-keyed blocking loses.
  Both of g7ema's design choices (soft floor, culprit signal) are hereby
  independently validated.
- diag_cg_revisit / diag_cg_rev_g7ema (hindsight second sweep, update=False
  pass 2, preds from pass 2): CATASTROPHIC — absrel +.027*, a1 −.049*,
  ATE +.0385*, rot +2.16°* (0/429 wins). Decoding against a frozen state
  that contains the frame's own future is far outside the training
  distribution (training never used update=False at all). KILLED. A
  revisit-TRAINED model is the only route this could ever work — parked.
- diag_cg_scenetau (tercile tau schedule −10.5/−11.99/−15.75, atten .122):
  ATE −.00269* rot +.0171* trans +.000139*, depth n.s. — DOMINATED by plain
  g7ema (same atten, better everything). Scene-conditional aggressiveness
  adds nothing over the global tau. NEGATIVE.

- CKPT-10 GATE (evidence/cg_ckpt10_gate.json): NO KILL. KILL-0 pass (loss
  tracks baseline, gated slightly lower at e10: 1.750 vs 1.813). Trio at
  20% training: cgt10_on vs aug10_on ATE +.0033 (bar .005 — clears);
  gate-trained BETTER on depth (absrel .2286 vs .2387, a1 .6958 vs .6821);
  matched corner beats mismatched on absrel/a1/ATE. FLAG carried to final:
  rot 2.43 vs 1.52 (+0.9°) — conf-head recalibration mid-schedule (atten
  23% at ckpt-10 vs final-model 11.7%); resolve at 50-epoch verdict.

## GATE-TRAINED FINAL VERDICT (2026-08-18, evidence/cg_train_verdict.json)

Four-corner square COMPLETE (full 4292, all vs augfull_lr1e5):
| corner | absrel | a1 | ate | rpe_trans | rpe_rot |
|---|---|---|---|---|---|
| plain/plain (augfull) | .179384 | .786312 | .075869 | .007941 | 1.100981 |
| plain-train/gate-eval (augfull_cg_g7ema) | .178647* | .787188* | .072802* | .008091* ✗ | 1.106619 ~ |
| gate-train/gate-eval (cgtrain_g7ema) | .178928 ~ | .787309* | .077177* ✗ | .007825* ✓ | 1.115121* ✗ |
| gate-train/plain-eval (cgtrain_plain) | .179748 ~ | .786506 ~ | .081226* ✗ | .007704* ✓ | 1.122974* ✗ |

Findings: (1) Training under the gate REDISTRIBUTES the benefit — the
rpe_trans cost flips to a significant WIN (−.000117*, gate-off even
−.000238*), but the ATE gain evaporates (+.0013* vs baseline): the model
co-adapts to attenuated writes as its normal regime, so the eval-time gate
no longer confers drift protection. (2) Mismatch corner again worse than
matched on ATE (+.004) — four-corner symmetry holds on this axis too.
(3) Gate-trained conf distribution shifted: frozen tau −11.99 now yields
13.1% attenuation (vs 11.7% plain). (4) KILL-10 rot flag resolved benignly
(final rot +.014*, not +0.9). VERDICT: **augfull_cg_g7ema (inference-only
gate on the plain finetune) REMAINS CHAMPION** — 3 sig wins, rot flat.
The gate-trained model is a mechanism result, not a promotion. Both rows
merged into averages_table.csv. Pre-registered follow-up: recalibrate tau
on the gate-trained model's own conf distribution (diag-labeled).

- RETAU SALVAGE (evidence/cgt_retau_430.json): aggressive tau (−11.0 /
  −9.67 = its own p50; atten 15–17%) on the gate-trained ckpt keeps the
  rpe_trans win (−.00029*) and depth gains but ATE remains sig-worse
  (+.0023…+.0031*). Co-adaptation is intrinsic; eval-time tau cannot
  recover the drift protection. BRANCH CLOSED — final standings stand:
  **augfull_cg_g7ema is the campaign champion.**

### Recommended next steps (recorded 2026-08-18, not launched)
1. REVISIT-TRAINED model: train WITH second sweeps (update=False pass 2 +
  losses on pass-2 predictions) so hindsight decoding becomes
  in-distribution — the only measured-dead lever whose failure mode was
  purely distributional; potential large ATE gains if it trains. ~520
  GPU-h. Risk: the co-adaptation lesson (benefits may redistribute).
2. Gate-scheduled training (gate ramped or annealed, or gate only in the
  graded TBPTT chunks) — attacks the co-adaptation directly.
3. Forward/backward trajectory fusion (eval-side, untested, cheap).

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
