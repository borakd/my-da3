# GRU Gap-Closure Directive (captain_gru v4 program)

Executable decomposition of the final gap analysis
(`/gpfs/projects/etur59/koc821022/captain_gru_autoresearch/FINAL_ANALYSIS.md`,
artifact b060a83b) into implementation directions. **That autoresearch
directory is now ON HOLD** — all work described here lands on the
`captain_gru_v3` branch of this repo.

Every direction below is keyed to the analysis finding that motivates it, and
every proposal states the exact files, functions, config keys, and launch
commands. Proposals are numbered in the order they should be implemented; the
sequencing rationale is at the end (§6).

**GOVERNING CONSTRAINT (2026-08-11): no proposal here may require retraining
baselines A/B/C.** Concretely, nothing may change the trunk's forward math or
its optimization in a way that also applies to a non-GRU run. Every proposal
below is therefore scoped to the `pose_gru` module, its loss term, or its
optimizer/gradient handling, and is gated behind a default-off config key so
existing configs and checkpoints reproduce bit-for-bit. The one common-mode
item the analysis identified — full `ddp_grad_sync` for the trunk — is
**deferred**, kept behind its existing flag, and replaced by a GRU-only
variant (§1). Any future proposal that violates this constraint must be
listed as deferred, not silently folded into a run.

**GOVERNING CONSTRAINT 2 — BACKWARD COMPATIBILITY IS NON-NEGOTIABLE.**
Every change must remain *isolatable* and *negatable*: if a run goes wrong,
you must be able to identify which single change caused it and switch that one
off, without editing code and without losing anything else. Concretely:

1. **Config-revertable.** Every change rides its own default-off key. Setting
   that key absent (or `null` on the CLI) restores the exact previous
   behavior — not "approximately", byte-identical. No change may be reachable
   only by editing source.
2. **One key, one mechanism.** Never fold two mechanisms behind one key. If
   two changes ship together and the arm regresses, you must be able to
   bisect them by flipping keys, not by re-reading diffs.
3. **Old checkpoints load under new code**, strict — `All keys matched
   successfully`, never `strict=False` and never a silent drop. A new
   parameter or buffer must be ABSENT when its key is absent, not present-
   and-neutral: an all-ones buffer still adds a state-dict key that every
   earlier checkpoint lacks, and the loader hard-fails on that.
4. **New state must be recoverable from the checkpoint itself**, via
   `_sniff_pose_gru_config`, so an eval never silently scores a differently-
   configured module. Levers invisible in weight shapes (iteration counts,
   modes, gains) are the dangerous ones — they fail silently rather than
   loudly.
5. **Total kill-switch.** With every new key absent, the tree must reproduce
   the pre-program baseline exactly. That is the escape hatch: one revert
   path back to known-good, always available.
6. **Each proposal ships its own falsifier** proving (1) and (3) for itself —
   absent-key byte-identity, presence-changes-output, and checkpoint
   round-trip. `verify_gru_input_gain.py` is the reference implementation.

Enforcement: `python verify_backward_compat.py` loads a fixed set of
reference checkpoints spanning every era (no-GRU baseline, pre-program GRU
arms, post-change arms) and asserts strict key-match plus correct lever
recovery. **Run it before every commit.** Verified green 2026-08-12 on the
R1 arm, the R8 arm, and the no-GRU baseline.

---

## ⚠ NOISE-FLOOR WARNING — READ BEFORE INTERPRETING ANY ARM COMPARISON

**Measured 2026-08-12, verified twice independently. Run-to-run variance in
this pipeline has NEVER been measured — there is exactly one run per arm — and
the lever effects the whole program reasons about sit inside the arm-to-arm
scatter.**

Paired per-scene ATE over all 4292 scenes, all 28 pairs among the 8 GRU arms:

| quantity | value |
|---|---|
| median \|ATE diff\| between arms | **0.00281** |
| max | 0.00814 |
| paired standard error | 0.000346 |
| pairs "significant" at \|t\| > 3 | **25 / 28** |
| R8 − R1 (the result the R lever rests on) | 0.00378 — **rank 10/28, mid-pack** |

With 4292 paired scenes the SE is ~0.0003, so **any two runs differing by more
than ~0.001 come out "significant."** The paired t-test controls SCENE
sampling only; it says nothing about whether a rerun of the same config would
reproduce the number.

**The demonstration that it does not:** the F lever, measured two ways —

    gru_a4g3f1   − gru_a4g3     = **+0.00436**  (t = +13.5)
    gru_a4g3f1np − gru_a4g3f0np = **−0.00116**  (t = −3.4)

Same lever. Opposite signs. Both "highly significant." An effect of ~0.004 in
this pipeline is not distinguishable from configuration idiosyncrasy.

### R-LEVER VERDICT (2026-08-12): WEIGHTING, NOT ITERATION

The anytime sweep settled it. Trained r8 checkpoint, `POSE_GRU_FORCE_ITERS` =
1/2/4/8/16, 32 scenes, all paired:

| N | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| ATE | 0.078496 | 0.078881 | 0.082204 | 0.080104 | 0.080834 |

No trend, no monotonicity, largest \|t\| across 20 paired contrasts = 2.1 —
what 20 tests give by chance. **N=1 is not worse than N=8** (t = −0.9):
running the model at ONE iteration instead of the eight it trained with costs
nothing.

Override verified three ways, so a flat curve cannot be a no-op:
`GRUCell.forward` counts ratio exactly 8.0000; forced N=8 reproduces the
arm's native scoreboard run **bit-identically on 32/32 scenes**; forced N=1
differs on 32/32, per-scene ATE moving up to 0.0346.

Grounds, with iteration now inert on every observable axis:
- **Forward refinement, training distribution:** the γ-weighted mean
  intermediate iterate is within **0.3%** of the final one at convergence
  (rel = 0.9968) and was 0.915 early — iterating made the pose WORSE. Solving
  through the γ mass, even iterate 0 is ~0.95× the final's error.
- **Inference:** output independent of N over a 16× range.
- **γ is the ONLY channel** by which intermediate iterates reach the loss, and
  the final iterate's value is essentially independent of N — so R8−R1 IS the
  γ term, nearly by construction.
- **R8 is not a better pose module:** on the held-out 4-view eval (byte-
  identical criterion in both arms) it is worse on pose loss in 44/50 epochs,
  rotation in 48/50, and every headline metric.

NOT claimed: this rules out iteration-as-refinement, not a training-DYNAMICS
effect of iterating — which is inseparable from γ, since γ is how iterates
reach the loss. Closing that needs `pose_gru_iter_gamma=0.0` at N=8 (objective
bit-identical to R1, pure CLI override, 19.3h).

**A third mechanism, unconfirmed and worth more attention than either
hypothesis:** AdamW is invariant to a constant loss multiplier, so 4.16× cannot
act by "making the objective bigger". The GRU's 136k params sit in their own
group and the aux graph touches nothing else — so the weight can only act via
(a) the aux:main gradient ratio on those params, or (b) `clip_grad=1.0` applied
to ONE GLOBAL norm over `model.parameters()`, where an inflated aux term
tightens the clip and throttles the **trunk's** effective LR. R8's trunk did
move less and fit worse (train ConfLoss 71.21 vs R1's 66.04). Under (b), R8's
ATE "win" is a smaller-finetuning-damage artifact, not a pose win. Unconfirmable
as things stand: **the clip norm is returned by the scaler and discarded, never
logged.** Logging it is a one-line change and would settle a mechanism that
affects every arm.

**Within-run jitter, a lower bound on run variance:** checkpoint-to-checkpoint
ATE jitter over epochs 41–50 is already **35–64% of the entire R8−R1 arm gap**.

### REPO TRAP — duplicate module objects (verified independently)

`import dust3r.model` and `import src.dust3r.model` load the same file as
**two module objects with two `PoseGRU` classes** (`m1 is m2` → False;
`m1.PoseGRU is m2.PoseGRU` → False). `$CUT3R_DIR` is on `sys.path` and contains
a `src/` package, so both spellings resolve.
`infer_and_eval_worker_ray.py:151` imports `src.dust3r.model`; every repo
`probe_*`/`verify_*` imports `dust3r.*`.

Consequence: any in-process monkeypatch, `isinstance` check, or call counter
**silently reads zero** for code running under the other spelling — it looks
like "the mechanism never ran" rather than raising. A counter read 0 while the
GRU was making 14296 calls on the other copy in the same process. No existing
diagnostic is corrupted (the two never share an interpreter today), but it is
live for the next one. **Patch a `torch` symbol instead** — `torch` is one
module and `GRUCell` is instantiated exactly once — or run the probe in a
subprocess.

### What this invalidates, and what survives

**SUSPECT — do not treat as established:**
- "R8 is the best arm" / "iteration is the lever that worked." The gap is
  mid-pack in the scatter.
- Every individual lever attribution in the A/G/F/R grid at the ~0.004 scale.
- Any future single-arm result whose expected effect is ≲0.004 — **including
  the P4.3 refine arm, whose ΔR²-implied effect is ~0.001, roughly 3× BELOW
  the median scatter.** A working refine arm could not be distinguished from
  noise by a single run.

**SURVIVES — larger than the scatter, or consistent across many arms:**
- Arm B's superiority (0.0134 vs 0.0759+, 99.9% per-scene win rate).
- **All 8 GRU arms lose to arm A.** Eight-for-eight sign consistency is a far
  stronger claim than any single pairwise contrast.
- Prev-pred conditioning being net-negative vs A (63.0% win rate).
- Within-checkpoint measurements, which do not depend on run variance at all:
  the ΔR² partial-correlation result, the input-scale probe, the velocity
  collapse.

### The implied priority

**A SEED REPLICATE IS NOW THE HIGHEST-VALUE 19h ON THE BOARD** — rerun an
existing arm unchanged with a different seed and measure the run-to-run ATE
spread. That single number retro-calibrates every comparison ever made here
and decides whether any of the queued arms are worth running at all. Spending
19h on a lever control before knowing the noise floor risks measuring nothing.

#### PRE-REGISTERED: the harness-determinism re-score must run FIRST

**Authorized by the user 2026-08-12 16:xx; ~1h on 8 nodes; NO training slot.**
Re-score an already-scored checkpoint under a fresh label, changing nothing
else. This decomposes the 0.00281 arm-to-arm scatter into the two sources it
currently conflates, and it gates the 19h replicate: if the scoreboard is not
deterministic, a seed replicate measures training variance PLUS harness
variance and cannot isolate either.

**The reading is fixed here, before the number lands, so neither side can move
the goalposts afterwards:**

- **Per-scene ATE reproduces EXACTLY on 4292/4292 scenes** → harness/sharding/
  claim-order noise is identically zero, and 100% of the arm-to-arm scatter is
  attributable to training. The 0.00281 median becomes a clean statement about
  training variance, and the seed replicate is then the ONLY instrument that
  can calibrate it. Proceed to the replicate.
- **Per-scene ATE does NOT reproduce** → this is the more urgent finding and it
  outranks every queued arm. It would mean the scoreboard itself is
  nondeterministic, that all 12 rows in `averages_table.csv` carry an unmeasured
  harness term, and that every lever attribution ever made on this program —
  including the ones this document treats as settled — is confounded by it. In
  that case: STOP arming new evals, quantify the harness term first, and do not
  spend a 19h training slot until it is bounded.
- **Ambiguous middle** (reproduces on most scenes, differs on a few): report
  the per-scene diff distribution, NOT a mean. A handful of large per-scene
  disagreements and a uniform small jitter are different failures with
  different causes, and the mean hides both.

Do NOT read a reproducing re-score as evidence that any arm result is real. It
bounds harness noise only; the training-variance term stays unmeasured until
the replicate runs.

#### PARTIALLY ANSWERED ALREADY, FOR FREE (measured 2026-08-12, this window)

Before spending the node-hour, the question was checked against evidence
already on disk, and most of it was already answered.

`diag_r8_iters8` (2026-08-12 14:15, forced `POSE_GRU_FORCE_ITERS=8`) and the
native `gru_a4g3r8` scoreboard run (2026-08-10 20:05) score the SAME checkpoint
on 32 common scenes. Their per-scene `eval_depth_pose_metrics.csv` files are
**BYTE-IDENTICAL on 32/32 scenes** — verified by direct byte comparison in this
window, not relayed. Two separate invocations, two days apart, different job,
different label.

**What that establishes:** the forward pass, the data loading, and the metric
computation are deterministic. Whatever produces the 0.00281 arm-to-arm scatter
is NOT in any of them. It also independently reproduces the bit-identity claim
in 0634c97 rather than taking it on trust.

**What it does NOT establish, and this is the precise residual the full
re-score still buys:** the diag run has **no `claims/` directory** — it
bypassed the distributed claim queue entirely, while every scoreboard run
shards 4292 scenes across 32 GPUs through a 4292-entry claim queue. So
multi-node sharding and claim ORDER remain untested. A priori the risk is low,
because scenes are scored independently and written to their own directories,
so claim order should not be able to affect a per-scene number. But "should not
by construction" is exactly the class of reasoning that has failed five times
on this program (the oracle bound, 70x starvation, input scales,
expected-value bands, the 20-25% velocity ceiling), which is why it is worth an
hour rather than an assumption.

**Consequence for sequencing:** the re-score is now CONFIRMATORY on a narrow
residual rather than an open question. The dominant unmeasured term is
training variance, and only the seed replicate addresses it.

#### RESULT (2026-08-12 17:45): EXACT. THE HARNESS IS DETERMINISTIC.

The full re-score ran and the pre-registered **EXACT** branch fired.

`gru_a4g3r8_rescore` — same checkpoint, fresh label, 8 nodes × 4 H100, 8/8
jobs COMPLETED rc=0 in ~58 min each, 4292 claims / 4292 eval / 4292 preds, so
the 32-GPU distributed claim queue WAS exercised (this is the precise residual
the 32-scene diag check bypassed, since that run had no `claims/` directory).

    BYTE-IDENTICAL: 4292/4292    DIFFERING: 0    UNREADABLE: 0
    only-in-A: 0                 only-in-B: 0

Every per-scene `eval_depth_pose_metrics.csv` is byte-for-byte identical to
the original 2026-08-10 scoreboard run. Scored by
`compare_rescore_determinism.py`, which was written and committed while the
jobs were still PENDING, precisely so the reading could not be tuned to the
number.

**WHAT THIS SETTLES:**
- Harness, sharding, and claim-order noise are **identically zero**. Not
  "small" — zero, at byte granularity, across the full 4292-scene sharded path.
- Therefore **100% of the 0.00281 median arm-to-arm scatter is attributable to
  TRAINING**. The scoreboard is not adding variance; the runs are.
- Re-scoring can be removed from the list of things any disagreement might be
  blamed on. Two differing table rows differ because their TRAINING differed.

**WHAT IT DOES NOT SETTLE, AND THIS IS THE POINT:** it bounds harness noise
only. It is NOT evidence that any arm result is real, and it makes the
noise-floor warning at the top of this document STRONGER, not weaker — the
scatter is now confirmed to be genuine run-to-run training variance, which is
exactly the term that makes every single-run lever attribution on this program
unreliable. The seed replicate remains the only instrument that can calibrate
it, and it is now unambiguously interpretable: any spread it shows is pure
training variance with no harness component to subtract.

**Calibration note:** five inherited figures have failed on contact with
measurement (the oracle bound, 70× starvation, input scales, expected-value
bands, the ≤20-25% velocity ceiling). This is the first pre-registered
prediction on this program that held exactly as stated. Recorded so the
program's track record is not read as "measurement always overturns" — it
overturns *unverified secondary inferences*, and this was a direct measurement.

**Correction to an earlier framing in this document:** the 4.1611 aux-weight
factor was NOT "silent". The `captain_gru_v3_a4_g3_r8_finetune.yaml` header
already documents it and even names `pose_gru_loss_weight=0.2403` as the
matching control. It was known and not acted on — a different and more
uncomfortable failure than an unnoticed coupling.

---

### Reference ladder — FULL BENCHMARK (source of truth)

All 4292 DROID test scenes, `checkpoint-final.pth` (epoch 50), from
`$OUT/summary/averages_table.csv` (produced by `eval_pipeline/aggregate_results.py`;
`$OUT = /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval`). Win% columns are
per-scene wins against arm C, recomputed from `summary/per_scene_*.csv` (n=4292).

| arm | label in the table | ATE | rpe_rot | ATE win% vs C | rot win% vs C |
|---|---|---|---|---|---|
| A | `augfull_lr1e5` (plain finetune) | 0.075869 | 1.100981 | **63.0%** | — |
| B | `gtray_lr1e5` (GT ray-map, the bound) | **0.013445** | **0.337232** | **99.9%** | — |
| C | `prevpred_lr1e5` (prev-pred, no GRU) | 0.082068 | 1.197303 | — | — |
| D | `gru_a4g3` (A4×G3, R1, F0) | 0.083699 | 1.194713 | 46.2% | 54.1% |
| E | `gru_a4g3f1` (+F1 pooled) | 0.088054 | 1.306286 | 39.2% | 37.1% |
| — | `gru_a4g3r8` (**R8 — best GRU arm**) | 0.079918 | 1.192692 | **54.2%** | **55.5%** |
| — | `gru_a4g3f1r8` (F1×R8) | 0.085247 | 1.291252 | 43.2% | 29.6% |
| — | `gru_a4g3f0np` (F0 no-proj) | 0.081981 | 1.213474 | 50.0% | 42.3% |
| — | `gru_a4g3f1np` (F1 no-proj) | 0.080817 | 1.202899 | 52.3% | 47.0% |

Read this table before anything else:

- **The prize is enormous and unambiguous**: B beats C on 99.9% of scenes
  (6.1× ATE, 3.5× rotation). Nothing about the gap is marginal.
- **Prev-pred conditioning is net-negative**: A beats C on 63.0% of scenes.
- **Every GRU arm sits between C and A, none reaches A.** The best,
  `gru_a4g3r8`, closes just (0.082068−0.079918)/(0.082068−0.013445) = **3.1%**
  of the C→B ATE gap at a 54.2% win rate — a real but tiny majority.
- **R8 (iters=8) is the only lever that helped**; F1 pooled features hurt
  (39.2%), consistent with the analysis' shift-invariant-pooling finding.
- Arms `f1c` (corr), `f1d` (DINOv2) and `f1r` (resnet18) **have since finished
  and been scored at 4292 scenes** — see the experiment-state section for the
  rows and their two confounds. They land at ATE 0.0799–0.0822, a spread of
  0.0023 that is at or below the arm-to-arm noise floor, so they do not rank
  their feature sources. All three also inherited `img_feat_proj=True`.

The analysis' own 50-sequence subset ladder (A 0.001745/0.574°, B
0.000822/0.177°, C 0.001866/0.609°, D 0.001982/0.633°, E 0.002031/0.606°,
F oracle 0.000996/0.235°) is a DIFFERENT normalization and scene count — the
two are not comparable numerically, but every ordering above matches it.

**The upper bound is already in the table — it is arm B, not the oracle.**
An earlier draft of this directive called for scoring
`captain_gru_v3_a4_g3_oracle_finetune_32gpu` at 4292 scenes as "the cheapest
missing measurement". That was **refuted on 2026-08-11**, on three
independently verified grounds:

1. **It duplicates arm B.** Under `--oracle gt` the GRU input becomes the
   CURRENT view's GT pose (model.py:1668), which flows to the ray build — the
   same information `feed_gt_ray_map` already encodes. B is measured:
   ate 0.013445, rpe_rot 0.337232.
2. **It is not a perfect-pose ceiling anyway.** That run's own final-epoch
   `log.txt` reads `gru_in_trans_err_gtscale = 0.0` (injection is real and
   exact) but `gru_trans_err_gtscale = 3.028` with `gru_gain_trans = 0.332` —
   handed an exact pose, the GRU shrinks the translation ~3× and emits one
   that is 3 GT-scales wrong. Its rotation is near pass-through
   (`gru_quat_loss = 0.00017`). The run's own training log states outright
   that its "recon/pose metrics are not comparable to an honest arm."
3. **10× LR confound.** That run used `lr 1.0e-06 / min_lr 1.0e-07` against
   `1.0e-05 / 1.0e-06` for every arm in the table, at identical batch, epochs
   and steps.

A fourth fact worth carrying: **perfect poses do not lift depth.** B's absrel
(0.188603) is *worse* than A's (0.179384) — the conditioning gap is a pose
gap, so do not expect §4's observability work to move the depth columns.

Two corollaries for the oracle's 83–93% figure: it is a 50-scene number, and
point 2 means the oracle's benefit comes almost entirely from ROTATION (its
translation is badly corrupted and it still recovers most of the gap). That
strengthens P3.1's rotation re-weighting considerably.

### RUNG 0 RESULT (job 44514478, 2026-08-11, 40 scenes) — GT AT THE GRU INPUT MAKES THINGS WORSE

`--oracle gt` on the **honest** `captain_gru_v3_a4_g3_finetune` checkpoint
(lr 1e-5 — no LR confound, no circular training). All four shards logged the
GT-injection banner; 40 scenes, zero errors. Means over those same 40 scenes:

| arm | ATE | rpe_rot | rpe_trans | absrel |
|---|---|---|---|---|
| B truth-conditioned | **0.014706** | **0.4951** | **0.004188** | 0.2055 |
| A no conditioning | 0.077676 | 1.1956 | 0.008877 | **0.1810** |
| C prev-pred, no GRU | 0.080295 | 1.3105 | 0.009189 | 0.2005 |
| D GRU honest (same ckpt) | 0.088764 | 1.2999 | 0.009430 | 0.2101 |
| **PROBE: same ckpt, GT at GRU input** | **0.116545** | **1.7682** | 0.009034 | 0.2222 |

Per-scene ATE win rates for the probe: **0/40 vs B**, 4/40 vs A, 3/40 vs C,
**6/40 vs its own honest self (D)**. On the C→B gap it scores **−55.3%** ATE
and **−56.1%** rotation: handing the module the right answer moves it
backwards by more than half the gap width.

**Reading it honestly — this is NOT a clean "the GRU wrecks perfect
information" result.** The injected GT comes from `demo_ray` raw `cam["pose"]`
with no scene-scale normalization, while the GRU's residual anchor P(x−1)
lives in the model's own head scale (round-2 forward math: head scale ≈ 4× GT;
`verify_gru_oracle.py` carries `HEAD_SCALE = 0.37`; the oracle-TRAINED arm
learned `gru_gain_trans = 0.332` to compensate and still had
`gru_trans_err_gtscale = 3.028`). So the probe is a pure **out-of-distribution
scale shock**, and what it proves is:

> The GRU and the conditioning path have **no scale invariance whatsoever**.
> An input that is numerically correct but in the wrong scale is worse than a
> stale input in the right scale.

Three consequences, all of which change the plan:

1. **No GT-injection diagnostic is trustworthy until scale is handled.** That
   retroactively undermines the oracle arm's 83–93% figure from the analysis
   (already flagged for its 10× LR confound) — do not plan against it.
2. **P3.4 is promoted from a minor lever to a PREREQUISITE.** The round-2
   evidence already prescribed exactly the fix: a per-sequence running scale
   on `absT`/`delta_t` with the emitted translation correction multiplied back
   by the same scale, motivated by the 11.6× per-chunk factor spread. This
   probe is direct confirmation that the module cannot survive a scale it was
   not trained on.
3. **P2 (ray-level abstention) is validated as the milestone-1 route.** A
   module this brittle needs a way to be switched off; its floor must be A.

Cheapest follow-up that would separate scale from competence: rescale the
injected GT into head scale before the `torch.where` in the oracle block, and
re-run this exact probe. If the probe then lands near B, the GRU is fine and
scale is the whole story; if it stays bad, the emitter is genuinely broken.

---

## §0 — Standing acceptance gates (analysis: honest arms are coin flips; A beats C on 8/8)

Applied to EVERY new arm before it is declared progress. No code changes; these
harnesses exist.

1. **Full-benchmark eval (4292 scenes, 32 GPUs)** — the ONLY scoring path:

   ```
   bash eval_pipeline/arm_eval_for_run.sh <RUN_DIR> <LABEL> [--dry-run]
   ```

   `RUN_DIR` is the trainer `output_dir` (holds `.hydra/` and
   `checkpoint-final.pth`); `LABEL` keys the outputs and the table row. It
   gates on writer-settled, `preflight_ckpt.py` config/ckpt agreement, no
   prior output for the label (`FORCE=1` to override), and disk headroom, then
   submits 8 nodes × 4 GPUs. The `--conditioning` arm is DERIVED from
   `.hydra/config.yaml` — never pass it by hand, that is how an R8 GRU gets
   scored at R1. To fire automatically on job completion, add the run to
   `$OUT/armed/registry.txt` and run `eval_pipeline/armed_eval_watcher.sh`.
   Then `eval_pipeline/aggregate_results.py` refreshes
   `$OUT/summary/averages_table.csv` and `per_scene_<label>.csv`, and
   `eval_pipeline/build_lr1e5_table.py` renders the comparison table.

   NOT `run_gru_window64_eval.sbatch` — that is the 77-window sliding eval of
   the single OVERFIT episode, takes no arguments, and reads a different
   checkpoint root. It is a diagnostic, not a gate.

2. **Falsifier battery** (env vars, all already wired in
   `model.py` `_forward_decoder_group_step`, lines 1518–1618):
   `PREV_PRED_RAY_SHUFFLE=1`, `POSE_GRU_HIDDEN_SHUFFLE=1`,
   `POSE_GRU_HIDDEN_ZERO=1`, `POSE_GRU_IMG_FEAT_SHUFFLE=1`/`_ZERO=1` (F arms),
   `POSE_GRU_FORCE_ITERS=n` (R arms). A new arm whose metrics survive its
   relevant shuffle is not using that channel — treat as failure, not neutral.
   `submit_gru_v3_eval.sh` explicitly `unset`s all of them before scoring;
   any new eval path must do the same or it silently contaminates a run.

3. **Promotion criteria**, in per-scene win% against C over the 4292-scene
   `per_scene_*.csv` (this is measurable today for every arm — see the ladder):

   | tier | ATE win% vs C | meaning |
   |---|---|---|
   | noise | ≤50% | no better than the no-GRU closed loop |
   | current best | 54.2% | `gru_a4g3r8`, today's ceiling |
   | **promotion bar** | **>63.0%** | beats arm A, i.e. conditioning finally pays for itself |
   | bound | 99.9% | arm B |

   An arm is progress only if it clears 54.2% on BOTH ate and rpe_rot; it is a
   RESULT only if it clears 63.0%, because below that the honest answer is
   still "turn the conditioning off". Report the closure fraction
   (C−X)/(C−B) alongside; `gru_a4g3r8` sits at 3.1%.

   **Two numbers that are NOT on this axis — do not mix them in.** (i) The
   oracle's "83–93%" is a closure fraction on the 50-scene subset, and that
   run used base lr 1e-6, 10× lower than every other arm — it is a bound with
   a confound, which is why rung 0 below re-measures it. (ii) The
   "≤20–25% trajectory-only ceiling" is a fraction of the GRU's own
   NORMALIZED AUX TRANSLATION error at eval (a perfect velocity extrapolator
   takes 1.09 → 0.84), not a fraction of the ATE/rot metric gap. Nothing in
   the evidence converts one axis to the other, so never subtract them.

---

## §0b — P0: retarget the loss so the GRU is asked a LEARNABLE question ← DO THIS FIRST

The analysis names this the single highest-information experiment left
(~2 h of code, one run), and it was missing from the first draft of this
directive. It is entirely `PoseGRULoss`-local, so it is trivially compatible
with the no-baseline-retraining constraint.

**The problem it removes.** `PoseGRULoss` currently supervises the GRU's
ABSOLUTE pose against `camera_to_pose_encoding(inv(cam1) @ gt["camera_pose"])`
(losses.py:1138–1141). That target contains the accumulated head drift, which
is unobservable from the GRU's inputs and dominates the aux translation term
~13:1 over the one learnable component. So ~94% of the gradient pushes the
module toward something it structurally cannot predict, and the optimizer
spends a 136k-param recurrent module's capacity on noise. Re-weighting
(P3.1) and more graded chunks (P3.2) change the volume of that signal; only
retargeting changes its CONTENT.

**The change.** Supervise the MOTION instead of the pose. Both endpoints
carry the same drift, so differencing cancels it and prices the learnable
velocity skill at 100% instead of ~6%:

- `PoseGRULoss.__init__` (losses.py:1154): add `target="absolute"`,
  accepting `"relative"`.
- In `compute_loss`, under `target="relative"`, replace the per-view pair
  (`gru_pose[i]`, `gt_poses[i]`) with
  (`pose_delta_encoding(pr_poses[i-1], gru_pose[i])`,
  `pose_delta_encoding(gt_poses[i-1], gt_poses[i])`) — the GRU's motion
  relative to the head's own previous pose, against the GT motion. Skip the
  first graded view of each chunk (no `i-1` inside the window) rather than
  reaching across a TBPTT boundary.
- `pose_delta_encoding` already exists at **model.py:488** and is NOT
  currently imported by `losses.py` — add the import (losses.py already
  imports from `dust3r.utils.camera`, so follow that pattern and avoid a
  circular import by importing inside the method if needed).
- Translation normalization: keep the existing per-side factors
  (`factor_pr` / `factor_gt`, losses.py:1150–1163) — a relative translation
  lives in the same scene scale as an absolute one, so nothing else changes.
- Config: `pose_gru_loss_target: relative`, interpolated into BOTH criterion
  strings exactly like `pose_gru_iter_gamma` already is.

**Run it as a strict A/B** on the current best arm: two runs identical except
`pose_gru_loss_target`, based on `captain_gru_v3_a4_g3_r8_finetune.yaml`.

**Honest caveat to record.** With a relative target the absolute pose is no
longer penalized at all, so the pose the GRU hands the ray build may drift
freely. That may be fine (drift is unobservable anyway, and the conditioning
signal that matters is rotation — the oracle recovers 83–93% with an exact
rotation and a badly corrupted translation) or it may hurt. Measuring which
is exactly the point of the A/B, and it is why this runs BEFORE the levers
that assume the current target.

---

## §1 — Direction: GRU-only gradient sync (32× gradient variance on a from-scratch recurrent module; trunk untouched)

**CONSTRAINT (2026-08-11, user): no change may force baselines A/B/C to be
retrained.** Full `ddp_grad_sync` is common-mode — it changes the trunk's
optimization for every long-context run, so arms D/E would no longer be
comparable to C without re-running C. It is therefore **DEFERRED**, kept on
its existing flag and config, and replaced in the near-term ladder by the
GRU-only variant below.

Background on the defect (unchanged): the TBPTT path drives the rollout
through `accelerator.unwrap_model(model)`
(`src/CUT3R/src/dust3r/inference.py:118`), so DDP's reducer is never armed by
`prepare_for_backward()` and **no** gradient is synchronized — every rank
optimizes on its own local gradient at an effective batch of 4, not 128. The
first full-fix launch (job 44445467) additionally **deadlocked**: graded
rounds per batch = `min(num_chunks, 4)` with `num_chunks = ceil(views/4)` and
views drawn uniformly from [4, 64] *per rank*
(`batched_sampler.py:69–71`), so ranks issue 1–4 collectives per batch and
the sequences mismatch.

### Why syncing ONLY the GRU is both safe and the higher-value half

- **Baselines A/B/C have no `pose_gru` module at all**, so a sync gated on
  its parameters cannot touch them. Zero retraining.
- **The trunk stays byte-identical to today's regime** in GRU arms too: each
  rank still steps its trunk on its own local gradient, exactly as arm C and
  every existing checkpoint did. D-vs-C stays apples-to-apples on the
  common-mode substrate.
- **The GRU is where averaging buys the most.** The intended estimator
  averages 32 ranks × 4 sequences × 4 supervised views = 512 pose samples per
  step; with the all-reduce dead it averages 16 — **32× the gradient
  variance** (round1 dossier, AUDIT grad-routing). The fingerprint is on the
  RECURRENT weights: `cell.weight_hh`'s Adam consistency ratio
  mean(|m|/√v) = 0.097–0.108 sits **below the 0.22–0.33 noise floor** every
  other parameter including the backbone shows — i.e. pure noise. The
  pretrained trunk at lr 1e-5 tolerates the same bug (which is why A/B/C
  train fine); the damage is specific to a from-scratch recurrent module at
  lr 1e-3.

  Do NOT justify this with the 348× or 0.355× figures. 348× is
  `rms(h)` by algebra (grad_W = δ⊗h, grad_b = δ), a signal-scale fact
  invariant to how many ranks you average; and 0.355× is the score of a
  closed-form linear readout *relative to the lag baseline* (trained arms sit
  at 0.85×), not a fraction anything realizes. Neither is a noise measure.

- **Expected upside is bounded and must be stated as such.** The defect is
  common-mode across all six analysed runs, so it does not explain the
  *relative* GRU failure, and averaging shrinks the variance of an estimator
  whose expectation still prices the learnable skill at ~6%. Sync cannot beat
  the ~20–25% trajectory-only ceiling.
- **Cost is negligible**: the whole module is ~136k params (≈0.5 MB fp32),
  one bucket, microseconds per round.
- **The deadlock shrinks with the payload.** Padding is still required (the
  round count is still data-dependent), but a pad round is now one 0.5 MB
  all-reduce plus a scalar — **no `optimizer.step()`, no trunk contact** — so
  short-batch ranks' trunks are not perturbed by extra AdamW decay steps the
  way the full fix's `dummy_sync_step` requires.

### P1 — `pose_gru_grad_sync`: all-reduce the PoseGRU gradients only  ← FIRST

A NEW flag, orthogonal to and independently togglable from `ddp_grad_sync`.
Both default False; `ddp_grad_sync` is not removed and its config
(`captain_gru_v3_a4_g3_ddpsync_finetune.yaml`) stays on the shelf for
whenever baseline retraining is acceptable. Setting both at once must raise
(the GRU params would be averaged twice).

**(a) `src/CUT3R/src/croco/utils/misc.py` — `all_reduce_grads_` (line 320):**
add a `contributing: bool = True` parameter. Before the bucket walk, all
ranks join one extra scalar collective to count real contributors this round:

```python
flag = torch.tensor([1.0 if contributing else 0.0],
                    device=params[0].device, dtype=torch.float32)
dist.all_reduce(flag)
scale = 1.0 / max(float(flag.item()), 1.0)   # replaces 1.0 / world_size at line 374
```

Dividing by contributing ranks (not `world_size`) keeps the averaged
gradient's magnitude comparable to the un-synced baseline on rounds where
only some ranks had a real chunk — otherwise late rounds would be silently
scaled down by the fraction of long batches. The existing `1/world_size`
path never completed a training step (that launch hung), so no real
checkpoint's semantics change here.

Also relax the docstring's "the non-TBPTT path must not use this" note to
name the two callers explicitly (full sync vs GRU-only), and keep the
`p.grad is None → zero-fill` and fixed-registration-order bucketing intact —
both are what make the collective sequence rank-identical, and the GRU-only
list is walked the same way.

**(b) `misc.py` — `NativeScalerWithGradNormCount` (line 402):**

- `__init__` (line 405) gains `sync_params=None`. Store as
  `self.sync_params` (a list, or None). Keep `ddp_grad_sync` exactly as is.
- In `__call__` (line 413), inside `if update_grad:` and **before** the clip
  at line 431 (so the clip sees the synced norm — DDP's own ordering):

  ```python
  if self.sync_params:
      all_reduce_grads_(self.sync_params, contributing=True)
  ```

  placed next to, and mutually exclusive with, the existing
  `if self.ddp_grad_sync:` branch.
- New method for the pad rounds:

  ```python
  def pad_gru_sync(self):
      """Issue the collectives a short batch would otherwise skip.

      Ranks whose batch produced fewer than GRADED_CHUNKS graded rounds must
      still join every collective the long-batch ranks issue, or the job
      deadlocks. contributing=False keeps this rank OUT of the average; the
      reduced values are then discarded (grad=None) so nothing leaks into the
      next batch's backward. No clip, no optimizer.step() — the trunk is
      never touched on a pad round.
      """
      if not self.sync_params:
          return
      all_reduce_grads_(self.sync_params, contributing=False)
      for p in self.sync_params:
          p.grad = None
  ```

**(c) `src/CUT3R/src/dust3r/inference.py`:**

- Module top: `GRADED_CHUNKS = 4` and use it at line 162
  (`if chunk_id < num_chunks - GRADED_CHUNKS:`) so the window constant and
  the pad count cannot drift apart.
- After the chunk loop (after the line-233 `optimizer.zero_grad()` block,
  before `result = dict(...)` at line 235):

  ```python
  for _ in range(GRADED_CHUNKS - min(num_chunks, GRADED_CHUNKS)):
      loss_scaler.pad_gru_sync()
  ```

  `pad_gru_sync` self-noops when `sync_params` is empty/None, so the
  baseline path executes one integer comparison and nothing else.

**(d) `src/CUT3R/src/train_cut3r_baseline.py`:**

- Next to the `ddp_grad_sync` block (lines 356–362), read the new flag:

  ```python
  pose_gru_grad_sync = bool(getattr(args, "pose_gru_grad_sync", False))
  if pose_gru_grad_sync and not use_pose_gru:
      raise ValueError("pose_gru_grad_sync=True requires pose_gru=True")
  if pose_gru_grad_sync and not bool(getattr(args, "long_context", False)):
      raise ValueError("pose_gru_grad_sync=True requires long_context=True "
                       "(the non-TBPTT path already syncs via the DDP wrapper)")
  if pose_gru_grad_sync and ddp_grad_sync:
      raise ValueError("pose_gru_grad_sync and ddp_grad_sync are mutually "
                       "exclusive — the GRU params would be averaged twice")
  ```

- Construct the scaler as today (line 395), then attach the parameter list
  **after** `accelerator.prepare` — `base_model` is already unwrapped at
  line 404, and this avoids any doubt about whether `prepare` rebound the
  module:

  ```python
  if pose_gru_grad_sync:
      loss_scaler.sync_params = list(base_model.pose_gru.parameters())
  ```

  This is the same parameter SET that `split_pose_gru_param_groups`
  (line 110, applied at 344) already isolates for the 100× LR group, so the
  synced set and the LR-scaled set are identical by construction.
- Extend the existing log line (364–367) with
  `pose_gru_grad_sync=<bool> (<N> GRU tensors averaged across ranks | trunk
  UNCHANGED: local gradients, baseline-comparable)` — the "trunk unchanged"
  half is the claim reviewers of this run will need.

**(e) `verify_ddp_grad_sync.py` (repo root)** — add two checks beyond the
existing six:

- **Check 7 — GRU-only selectivity.** A 2-rank model with a "trunk" Linear
  and a "gru" Linear; sync only the gru params. Assert the gru gradient
  equals the cross-rank mean AND the trunk gradient is still each rank's
  purely local value (this is the *no-common-mode-change* guarantee, so it
  is the single most important assertion in the file).
- **Check 8 — divergent round counts.** The case the original six missed and
  the one that hung job 44445467: rank 0 issues 4 real rounds, rank 1 issues
  2 real + 2 `pad_gru_sync` rounds. Assert (i) no deadlock — wrap the
  `mp.spawn` join in a 120 s watchdog and fail loudly on timeout, (ii) rounds
  where only rank 0 contributed are divided by 1, not 2, (iii) after the pad
  rounds rank 1's `p.grad` is None, so the discarded values cannot
  contaminate the next step.

**(f) Config** — `src/CUT3R/config/captain_gru_v3_a4_g3_grusync_finetune.yaml`,
a 2-line semantic diff from `captain_gru_v3_a4_g3_finetune.yaml`
(`pose_gru_grad_sync: true`, new `exp_name`) plus a header comment recording
that the trunk is deliberately left un-synced so arm C remains the valid
control.

**(g) Smoke then launch.**

`smoke_train_gru_v3.sbatch` is a **1-GPU** script (`--gres=gpu:1`,
`accelerate launch --num_processes 1`), and `all_reduce_grads_` returns
immediately when `world_size < 2` — so the smoke as it stands **cannot
exercise a single collective and cannot catch a deadlock**. Two steps, in
order:

1. *Collective correctness, on a login node, no allocation*:
   `python verify_ddp_grad_sync.py` (2 gloo CPU ranks, checks 1–8 incl. the
   GRU-only selectivity and divergent-round-count cases added in (e)).
2. *Real-trainer liveness, ≥2 GPUs*: make the smoke's process count settable —
   change the launch line to
   `accelerate launch --multi_gpu --num_processes ${SMOKE_GPUS:-1}` — then
   `SMOKE_GPUS=2 sbatch --gres=gpu:2 --export=ALL,SMOKE_GPUS=2 smoke_train_gru_v3.sbatch <config>`.
   Confirm the step counter passes ~50 steps (the earlier deadlock froze at
   `Epoch: [0] [0/301]` and produced nothing for 14 minutes) and that the log
   reports `pose_gru_grad_sync=True`.

Then:

```
sbatch --nodes=8 --job-name=gru_a4g3_grusync \
       train_gru_v3_finetune_64node.sbatch captain_gru_v3_a4_g3_grusync_finetune
```

Fresh from-scratch start with a new wandb id; wipe any stale run dir first
(auto-resume + persisted `wandb_run_id.txt`). This run is the **isolated
measurement of GRU gradient noise** — 2-line config diff vs arm D. Do not
stack §2/§3 changes into it.

**Pair the sync with the LR arm — this is not optional.** The evidence's
only empirical corroboration of the noise story is an LR contrast, not a
batch contrast: the oracle run at 10× lower lr kept its gate biases at
std 0.16 instead of 0.55–1.08 and did not collapse as hard (rms(h) 0.041 vs
0.004–0.007). And the trained (not init) gate saturation is live and
unrefuted — reset-gate bias sum mean +5.04 → sigmoid 0.993 on **100% of 128
units** — with the dossier's prescribed fix being
`pose_gru_lr_scale 100 → 10`. A sync run at `lr_scale=100` is therefore NOT
the arm the cited mechanism points at. Launch **two** grusync arms
(`lr_scale=100` and `lr_scale=10`, one CLI override apart) or the result is
uninterpretable.

**Two honest caveats to record in the run's header comment:**

1. *GRU replicas drift slightly across ranks*, because pad rounds do not
   step: rank 1 takes fewer GRU optimizer steps per batch than rank 0. This
   is benign — only rank 0's checkpoint is ever saved, and every one of rank
   0's GRU steps uses the full contributing-rank average. It is mild
   staleness in other ranks' local copies, not corruption of the saved model.
   (If it ever needs eliminating, the fix is to make pad rounds step on the
   averaged gradient too — i.e. the deferred full-sync semantics for the GRU
   group only.)
2. *The trunk's effective-batch-4 defect remains, by design.* This defers the
   common-mode issue rather than fixing it. That is the point: a synced-GRU
   arm that beats C did so under the same trunk regime as C.

---

## §2 — Direction: guaranteed non-harm (analysis: GRU emits 1.41–1.50° rotation error vs a 0.43° total prize; D loses to C)

The GRU must be able to abstain per-step instead of injecting noise.

### P2 — abstention: DEMOTED TO AN INSTRUMENT, NOT A SHIPPED FEATURE

**STRATEGY DECISION 2026-08-12 (user).** The objective is to win ACTIVELY on
essentially every scene, using our own predictions. Abstention wins by
declining to play, so it is not that objective and must not stand in for it.
It is deferred until everything else has been tried. Four reasons, and the
first two are the load-bearing ones:

1. **It masks the defect.** A gate improves the headline number while the
   corrector stays exactly as broken, removing the pressure that produces the
   real fix and creating a result that needs caveating forever.
2. **Its value evaporates on success.** The 17.4% below exists BECAUSE the
   corrector is right on half the scenes and wrong on the other half. Fix the
   corrector and the gate's prize collapses toward zero. It is a measure of
   today's brokenness, not a durable asset.
3. **It forecloses nothing.** A bolt-on requiring no architectural commitment,
   attachable to any corrector at any time, dormant behind a default-off key
   under GOVERNING CONSTRAINT 2. Deferring it is free.
4. **It muddies attribution.** An always-on arm measures corrector quality; a
   gated arm's number conflates corrector quality with gate quality.

**But run it ONCE, as an instrument.** A learned gate is a free map of exactly
WHERE the corrector fails — information the always-on arm destroys by
averaging good and bad scenes into one number. Its open/close pattern is the
highest-value input to the §4 refinement design. Rules: never in the headline,
never claimed as a result, telemetry only.

The implementation and its pricing follow.

### P2 implementation — at the RAY level (a head-level gate cannot clear §0)

**PRICED 2026-08-12 — this is the highest-leverage proposal in the plan.**
An ORACLE gate (per scene, pick the better of {arm, A} using the realized
metric) computed over all 4292 scenes from `summary/per_scene_*.csv`:

| arm | always-on ATE | oracle gate vs A | C→B closure, gated | always-on |
|---|---|---|---|---|
| `gru_a4g3r8` | 0.079918 | **0.070125** | **17.4%** | 3.1% |
| `gru_a4g3` | 0.083699 | 0.071401 | 15.6% | −2.6% |
| `gru_a4g3f1np` | 0.080817 | 0.070577 | 16.8% | 1.8% |
| `gru_a4g3f1` (worst arm) | 0.088054 | 0.072200 | 14.4% | −8.7% |

(A = 0.075869, C = 0.082068, B = 0.013445. rpe_rot behaves the same:
r8 gated 14.5% vs 0.5% always-on.)

Three consequences:

1. **Every arm — including the worst — beats A under gating.** The GRU emits
   genuinely useful conditioning on a large subset of scenes and harmful
   conditioning elsewhere; always-on averages them to ~zero. The information
   is there; the failure is deployment, not capability.
2. **The gate dominates the lever grid.** All six arms land within ~0.002 of
   each other when gated. The entire A/G/F/R grid moved always-on ATE by 3.1%
   of the gap; gating unlocks 17.4% from checkpoints that already exist.
3. **P2 alone cannot approach B.** 17.4% is the perfect-gating ceiling — it
   clears milestone 1 (beat A) and stops. Milestone 2 needs P4.3.

**Caveat, state it in any writeup:** this is an oracle gate (it uses the
realized metric to choose) and it switches between two separately trained
models per scene, whereas P2 gates per-view inside one model. Per-view is
finer-grained — more headroom — but harder to learn. 0.070125 is a ceiling,
not a forecast; a learned gate capturing half the available margin still
beats A.


**Design correction.** Gating the GRU's *correction* makes its floor plain
`feed_prev_pred` — arm C — and C is itself net-negative (A beats C on 63.0%
of scenes). A correction gate therefore cannot satisfy this directive's own
promotion bar. The gate has to sit where it can switch the conditioning OFF
entirely, i.e. at the ray add (`model.py:2033`):

```python
feat_group = [feat_group[0] + a * ray_out[-1].to(...) + (1 - a) * self.masked_ray_map_token]
```

`a → 0` reproduces the pose-free construction **view 0** takes (the same
`masked_ray_map_token` added at model.py:1555).

**CORRECTION 2026-08-12 — that is NOT arm A's regime, and this section
previously claimed it was.** Verified in code: DL3DV sets `ray_mask=False`
for every view (dl3dv.py:339) and flips it True only under the GT-ray flags
(dl3dv.py:356); arm A's config sets no `feed_*` key at all, so
`selected_ray_maps.size(0) > 0` is False and `_encode_views` takes the else
branch, whose two adds are **both multiplied by 0.0** (model.py:1541-1542) —
that branch exists to keep the ray encoder in the graph, not to add a token.
The `+= masked_ray_map_token` at model.py:1555 sits inside
`if feed_prev_pred` and applies to `[:batch_size]`, i.e. **view 0 only**.
Three distinct mid-sequence constructions:

| arm | mid-sequence tokens |
|---|---|
| A (no conditioning) | `feat` |
| GRU arm, `a → 0` | `feat + masked_ray_map_token` |
| GRU arm, normal | `feat + ray_out` |

So "the floor is arm A by construction" is FALSE. `verify_gru_ray_gate.py`
check 4 is still correct — `a=0` really does equal `feat + masked_token`
bit-for-bit — it is the interpretation on top that was wrong.

Mitigating, and why the design is still sound: `feat + masked_ray_map_token`
is base CUT3R's NATIVE pose-free signal. `get_img_and_ray_masks`
(base_multiview_dataset.py:281) draws `raymap_mask` True with p ≈ 0.20 on
metric data, and any ray-less view in such a batch takes the masked-token
path — so the token is heavily trained in pretraining, and it is what view 0
of every GRU sequence sees. The floor is a well-trained regime; it is simply
not arm A's *finetuned* regime, which drifted toward "nothing added" over 50
epochs of never seeing the token. **Whether the two behave alike is
measurable and unmeasured.** Treat "stop losing to A" as an empirical hope
about an unmeasured regime, not a structural guarantee. Implement `a`
as a scalar per sample from a 1-unit head off the GRU hidden (or off the
pooled tokens when the GRU is off), sigmoid, bias-init +4 so it starts open
and init-equivalence holds to within 2%. The `pose_gru_e2e` tape already
reaches this point, so the main reconstruction loss trains `a` directly —
this is 3 lines plus the head, and it is the only proposal here that can
make the conditioning channel non-harmful by construction.

Config: `pose_gru_ray_gate: true`. Log the per-view mean of `a`; a trained
`a` collapsing toward 0 on mid-sequence views is itself the cleanest possible
confirmation of the misapplied-tail-correction finding in P3.2.

### P2b — (optional) learned gate on the correction

**`src/CUT3R/src/dust3r/model.py` — `PoseGRU`:**

- `__init__` (line 639): new kwarg `gate=False`. When true, after
  `self.head` (line 876): `self.gate = nn.Linear(hidden_dim, 2)` — one logit
  for the translation correction, one for rotation — with
  `nn.init.zeros_(self.gate.weight)` and `self.gate.bias.data.fill_(4.0)`
  (sigmoid ≈ 0.982: the gate starts open, preserving init-equivalence to the
  ungated module to within 2%, and learns to CLOSE where the correction hurts).
- `forward` (line 977): after `raw = self.head(hidden)`:

  ```python
  if self.gate is not None:
      g = torch.sigmoid(self.gate(hidden))          # (B, 2)
      raw = torch.cat([raw[:, :3] * g[:, :1], raw[:, 3:7] * g[:, 1:2]], dim=-1)
  ```

  This composes with all three modes (it scales the correction/regression
  term uniformly; in `residual` mode g→0 reproduces plain `feed_prev_pred`
  exactly — the abstention semantics we want).
- **Checkpoint sniffing** — `_sniff_pose_gru_config` (line 77): detect
  `gate` from key presence, mirroring the `img_norm` pattern at line 121:
  `gate = _get("gate.weight") is not None`, pass through to `enable_pose_gru`.
- **Config plumbing**: `pose_gru_gate: true` key, added to the
  `enable_pose_gru(...)` kwargs assembled at `train_cut3r_baseline.py:245–250`. Default absent→False so every existing
  config and checkpoint is untouched.
- **Diagnostics**: log `g` means into the loss details — in `PoseGRULoss`
  this is not visible, so instead have the call site stash
  `res_group[-1]["gru_gate"] = g.detach()` (model.py:2131, next to the
  `res_group[-1]["gru_pose"]` write) and
  add `gru_gate_t/gru_gate_q` means in `PoseGRULoss.compute_loss` details
  when the key exists. A trained gate sitting at ~0 for rotation is the
  analysis' prediction — that readout alone is informative.

---

## §3 — Direction: training-signal repair (analysis: learnable skill priced at ~6% of the objective; rotation ~11%; head grad-starved 348×; hidden rms 0.005)

Four independent, cheap changes. All ride the same future launch (§5, run
v4a); none changes any default.

### P3.1 — rotation re-weight (+ optional geodesic) in PoseGRULoss

**`src/CUT3R/src/dust3r/losses.py:1091–1217`:**

- `__init__` (1103): add `rot_weight=1.0, rot_geodesic=False`.
- Line 1164: with `rot_geodesic`, replace the chordal norm by the sign-safe
  angle: `L = min(‖q̂−q‖, ‖q̂+q‖)` then `q_err = 4·asin(clamp(L/2, max=1−1e−7))`
  — the actual rotation angle in radians (the analysis' θ = 4·asin(L/2)),
  removing the q/−q double-cover ambiguity the chordal form has.
- Line 1177: `loss = t_loss + self.rot_weight * q_loss`; apply the same
  weight inside the `iter_gamma` branch (line 1303).
- `get_name` (1126): include `rot_weight`/`rot_geodesic` when non-default so
  train/test criterion strings stay comparable in logs.
- **Config**: add `pose_gru_rot_weight: 1.0` and interpolate into BOTH
  criterion strings of the v4 config:
  `...PoseGRULoss(iter_gamma=${pose_gru_iter_gamma}, rot_weight=${pose_gru_rot_weight})`.
  Starting value for v4a: **12.0**, inside the dossier's geometric-impact
  range (w_q ≈ 10–20; one audit argues 50–100). Rotation is 11.6–11.8% of the
  train-side unweighted aux term for arms D/E — the PoseGRU docstring's
  "3–4%" is the same definition measured on the *overfit* grid, and the eval
  side reads 1.1%; the train figure is the one that sets the gradient.
  **Interaction warning**: `rot_geodesic` replaces the chord L by
  θ = 4·asin(L/2) ≈ 2L rad, so enabling both multiplies the rotation term by
  ~2·w_q. Set one lever at a time, or halve w_q when the geodesic form is on.
  Supporting evidence for weighting rotation heavily: the oracle recovers
  83–93% of the gap with an essentially exact rotation (0.020°) and a badly
  corrupted translation (3.03 GT-scale units) — rotation is the component of
  the conditioning ray map that produces the benefit.

### P3.2 — full-window GRU supervision (4× aux gradient budget)

Only the last 4 of up to 16 chunks are graded (`inference.py:162`); views
1–47 give the GRU zero gradient. The GRU path is 136k params on 7-d tensors —
grading it everywhere is nearly free.

**The real justification is not "4× budget", it is a MATERIAL measured
defect**: per-view conf_loss shows the GRU arm *degrading relative to plain
prev-pred* exactly in the ungraded chunks 8–11 (0.659/0.716/0.781/0.871 vs
prevpred 0.612/0.639/0.675/0.685) and improving sharply in the graded chunks
12–15. The GRU learns a **tail-regime correction and misapplies it to views
1..47**, actively making mid-sequence conditioning worse than no GRU at all.
That is the mechanism behind "D loses to C", so P3.2 is a candidate FIX for
the headline regression, not merely more gradient.

- **`model.py`**: wrap the PoseGRU island (from `pose_gru =
  getattr(self, "pose_gru", None)`, model.py:1527, through the stash writes
  and the e2e detach choice at 1706–1729) in `with torch.enable_grad():` **when**
  `getattr(self, "pose_gru_full_window", False)`. `enable_grad` re-enables
  the tape inside the outer `no_grad` chunk; inputs are detached stashes so
  only GRU parameters enter the graph. (`pose_gru_e2e` is naturally inert in
  the no-grad window — the ray build stays gradient-free — which is correct.)
- **`inference.py`** — early-chunk branch (lines 183–194): after the
  criterion call, when the flag is on and `loss.requires_grad` (the
  `gru_pose` terms now carry a graph; everything else in `loss` is detached):

  ```python
  loss_scaler(loss, optimizer, update_grad=False)
  ```

  before `del loss`. `update_grad=False` is backward-only
  (`misc.py:413–441`) — no collective, no step — so it composes with P1
  without perturbing the round count: ungraded chunks issue zero collectives
  no matter how many a rank has, and their accumulated GRU gradient simply
  rides into the first graded round's all-reduce and step. Only the GRU's own
  gradient accumulates here (every other term in `loss` is detached inside
  the no-grad window), so the trunk is untouched — this proposal is
  GRU-local too, and safe under the no-baseline-retraining constraint.
- **Boundary detach** at `inference.py:156–158` already truncates the hidden
  every chunk, so per-chunk backwards never cross freed graphs.
- **Config**: `pose_gru_full_window: true` in the v4 config; model attr set
  next to `pose_gru_bptt`/`pose_gru_e2e` wherever those are assigned in
  `train_cut3r_baseline.py`.

### P3.3 — separate gradient clip for the GRU group

`loss_scaler(..., clip_grad=1.0, parameters=model.parameters())`
(`inference.py:226–232`) computes ONE global norm; the trunk's norm dominates,
so trunk-driven clipping rescales the already 348×-starved GRU gradient.

- **`misc.py` `NativeScalerWithGradNormCount`**: add `gru_clip_grad=None` to
  `__init__` and reuse the `self.sync_params` list P1 already attaches (same
  parameter set — do not thread a second list through the call site). In
  `__call__`, AFTER the sync and around the existing clip at line 431: clip
  the GRU set by its own norm
  (`self.accelerator.clip_grad_norm_(self.sync_params, self.gru_clip_grad)`)
  and **exclude** those params from the global norm.
- **What the evidence actually supports** (do not overclaim a confound): the
  GRU is 136k of ~1B trainable parameters, so its contribution to the global
  norm is negligible and D's trunk clip is essentially C's already. The
  supported claim is one-directional — whenever the reconstruction loss
  produces a gradient norm ≫ 1, *every* GRU gradient is uniformly attenuated
  by 1/‖g_total‖ — and the dossier flags this as "not fatal on its own (lr is
  not the binding constraint)". Treat P3.3 as cheap hygiene, not a fix.
  Baselines are untouched either way (no `pose_gru` module → `sync_params`
  empty → both branches noop).
- **Config**: `pose_gru_clip: 5.0` for v4a (an order looser than the trunk's
  1.0; the GRU group already runs at 100× LR — `split_pose_gru_param_groups`,
  `train_cut3r_baseline.py:110`, applied at line 376).

### P3.4 — scale normalization ← PREREQUISITE (promoted by the rung-0 result)

**Status: this is no longer a tuning lever.** The rung-0 probe showed the
module is not scale-invariant at all (ATE 0.1165 with a numerically PERFECT
input, vs 0.0888 with its own stale one). Everything that feeds the GRU a
better pose — P0, corr, xattn, the probe pass — assumes the module can
consume a pose whose scale it was not trained on. It cannot. Do this before
trusting any of them.

**MEASURED 2026-08-12 by `probe_gru_input_stats.py`** (200 batches × 4 × 64
views = 50,400 rows, 800 sequences, real `a4_g3` checkpoint, run on BOTH
splits). These numbers SUPERSEDE the round-2 audit estimates this section
previously carried:

| quantity | train split | test split | old directive value |
|---|---|---|---|
| `absT` mean @ graded views (48–63) | **0.9947** | 0.9374 | 0.5–0.8 |
| `delta_t` p50 | **0.04125** | 0.03194 | 0.018 |
| `delta_t` p95 | **0.13686** | 0.11835 | 0.068 |
| `absT` / `delta_t` median ratio | **12.86** | 16.86 | ~70× |
| per-sequence scale spread p99/p5 | **17.24** | 12.74 | 11.6× |

Provenance, so this check never becomes circular: the old values were a
SECONDARY inference — round-2 audit GT statistics (`GT_DT_P50 = 0.0045`,
`GT_ABST_SUP_MEAN = 0.206`) multiplied by an assumed head scale of **4.0×**.
The new ones are a PRIMARY measurement. The head-scale-invariant p95/p50 SHAPE
check passes, so the distribution was predicted correctly and only the scale
factor was wrong.

**(Correction, 2026-08-12: an earlier revision of this section attributed the
assumption to `HEAD_SCALE = 0.37` in `verify_gru_oracle.py` and inferred a
"true head scale nearer 0.8". Both were wrong — that is a different quantity.
The probe's constant is `ASSUMED_HEAD_SCALE = 4.0`.)**

The measured factors are NOT a single constant, and the discrepancy is
informative:

| channel | GT value | measured | implied factor |
|---|---|---|---|
| `absT` @ views 48–63 | 0.206 | 0.99473 | **4.83×** |
| `delta_t` p50 | 0.0045 | 0.04125 | **9.17×** |

So the head's frame-to-frame motion is inflated **1.90× more** than its
absolute displacement, relative to GT. That is the signature of per-frame
prediction NOISE adding on top of true motion: the fed-back trajectory jitters.
Two consequences worth carrying:

- The velocity channel the A4 lever hands the GRU is proportionally noisier
  than the pose channel — it is not a clean motion signal.
- **This bears directly on P0.** A relative-motion target supervises exactly
  this jitter-inflated quantity, so P0's gain may be smaller than its
  drift-cancellation argument suggests. Log `delta_t` magnitude alongside the
  P0 A/B rather than assuming the target is clean.

Two corrections that change the prescription:

- **The velocity channel sits ~13× below the pose columns, not ~70×.** The
  "must grow 50–70×" figure was overstated ~5×; the measured-optimal delta
  gain is 20–30.
- **The per-sequence spread is WORSE than the dossier's 11.6×** (17.2× on
  train). That strengthens, not weakens, the case that a static buffer is
  only a partial fix.

Caveat to carry: the rung-0 "scale shock" reading leaned on `HEAD_SCALE=0.37`.
With the real head scale ~2× larger, the GT-vs-head mismatch is smaller than
stated there, so the "GRU is genuinely brittle" explanation gains relative
weight. The rescaled-GT re-probe settles it.

**A frozen per-dimension buffer is only a partial fix.** The per-chunk
supervision factor spans **11.6×** across batches (p5 0.0295 → p99 0.344), so
the module must be scale-equivariant; the dossier prescribes a **per-sequence
running scale** (divide `absT`/`delta_t` by e.g. the mean |t| of the fed-back
trajectory) with the **emitted translation correction multiplied back by the
same scale**. Implement the static buffer first (cheap, checkpoint-safe), but
log the per-sequence factor so the dynamic version can be priced.

- **Probe first**: new 30-line script `probe_gru_input_stats.py` (repo root,
  pattern-copy `probe_gru_hidden_falsifier.py`): load a D-arm checkpoint,
  run 200 batches, dump per-dimension std of `gru_in` (built at
  model.py:1923) to JSON.
- **`model.py` `PoseGRU.__init__`**: `self.register_buffer("input_gain",
  torch.ones(self.input_dim))`; in `forward` at model.py:952: `cell_in = gru_in *
  self.input_gain` (before the img_feat concat — image features are already
  LayerNormed). Buffer serializes with the checkpoint, so eval rebuilds
  bit-exact; all-ones default keeps every existing checkpoint loadable
  (`load_state_dict` picks the buffer up; for OLD checkpoints missing the
  key, load with `strict=False` handled by the existing hard-fail sniff —
  simplest: only build the buffer when a new `pose_gru_input_gain` config
  key is present, mirroring the gate's presence-gating).
- **Config**: use the TRAIN-split gains (gradients flow there), measured
  2026-08-12:

  ```yaml
  pose_gru_input_gain: [2, 2, 2,  1, 1, 1, 1,  30, 30, 20,  1, 1, 1, 1]
  #                    |absT(3)| |  quat(4) | | delta_t(3)| | delta_q(4) |
  ```

  Quaternion dims stay 1.0 (already unit-norm). The test split independently
  gives [2,2,2, 1,1,1,1, 30,30,30, 1,1,1,1] — agreement to one digit. **Length must
  be `self.input_dim`, which is NOT always 14**: A4×F0 is 14, but the F1 arms
  widen the cell input to 46 (pose 14 + appended 32). Gate the buffer on the
  pose block only, or build it at `self.input_dim` and pad with ones —
  a 14-long vector will not broadcast on any F arm.
- **Probe validation**: `all_checks_passed=False` on both splits is EXPECTED
  and already resolved — the three failing gates compared against the
  superseded estimates above, while the shape gate passed. Do NOT relax the
  bands to make the gate green; re-point them at the measured values and keep
  the old ones recorded as superseded, or the check becomes circular.

---

## §4 — Direction: observability (analysis: head drift dominates the aux translation loss ~13:1 over the one learnable component, at train)

Everything in §1–§3 optimizes within the ≤25% ceiling. Only giving the
corrector an *observation* of drift can approach the oracle. Three proposals,
strictly increasing cost.

### P4.1 — the corr arm is ALREADY IN FLIGHT (score it, do not relaunch)

`img_feat_src="corr"` (`dust3r/img_encoders.py:corr_motion_stats`, 54-d
correlation/flow statistics BETWEEN current and previous token sets) is the
only existing F source carrying explicit relative-motion evidence — exactly
what the analysis says pooled F1 destroys (shift-invariant mean over tokens,
model.py:1797).

**Status as of 2026-08-11 19:00: `captain_gru_v3_a4_g3_f1c_r8_finetune` is
TRAINING** — job 44504130, 8 nodes, started 14:49, epoch 10/50, run dir clean
(wiped before relaunch; the earlier attempt 44496324 was cancelled). Its two
source siblings are also running: `f1d` (DINOv2, job 44496323, epoch 21) and
`f1r` (resnet18, job 44496322, epoch 21). Nothing to launch. Actions:

1. **Do not resubmit and do not touch those run dirs.** ~2.5 days remain on
   each 3-day walltime; f1c needs ~40 more epochs.
2. **The evals are ALREADY ARMED — do not arm them again.** Verified
   2026-08-11 19:44: watcher PID 374785 is live on glogin1, and
   `$OUT/armed/registry.txt` already carries all three rows
   (`44504130|…f1c_r8_finetune|gru_a4g3f1cr8|…`, and the f1d/f1r twins),
   auto-discovered at 15:04. Starting a second watcher would give every arm
   two submitters that both pass the gates — 16 eval nodes per arm. Just
   check for the `.submitted` marker after each job lands; a `.failed`
   marker means re-gate, not re-train.
3. **Read them against the R8 column, not against D.** All three are R8 arms,
   so the correct control is `gru_a4g3r8` (ATE 0.079918, 54.2% win vs C) —
   comparing f1c to `gru_a4g3` would confound the F source with the R lever.
4. **Falsify before believing**: re-score the winner with
   `POSE_GRU_IMG_FEAT_SHUFFLE=1`. If metrics do not degrade, the cell is
   ignoring the corr evidence and P4.2 becomes the priority regardless of the
   headline number.
5. Only if corr clears the §0 promotion bar is an R1 twin
   (`pose_gru_iters: 1`) worth running to isolate the source from the R lever.

### P4.2 — cross-attention RelPose source (`img_feat_src="xattn"`)

A learned two-view relative-motion reader where corr's hand-crafted
statistics stop. Slots into the EXISTING source-lever machinery so all stash /
falsifier / sniff plumbing is inherited:

- **`dust3r/img_encoders.py`**: new `XAttnRelPose(nn.Module)` — inputs the
  current and previous views' pre-ray token sets (B, N, 1024) (the same tensors corr
  consumes — the full-token stash at model.py:1802 and the previous-view
  read at 1572); two blocks of
  cross-attention (dim 256, 4 heads, queries = current tokens projected,
  keys/values = previous) → mean-pool → MLP(256→256→64). Output: a 64-d
  motion embedding (not a hard 7-d pose — let the cell read it). Export
  `XATTN_FEAT_DIM = 64`.
- **`model.py` `PoseGRU`**: register the source alongside `"corr"` — it is
  pair-consuming (`img_feat_frames` must be 2, `img_feat_blocks = 1`,
  `img_feat_src_dim = XATTN_FEAT_DIM`), TRAINABLE (unlike the frozen
  encoders — exclude it from the frozen-source assumption at 599–606 and put
  its params in the pose_gru LR group, which happens automatically since it
  lives under `self.pose_gru`). Keep the zero-init `img_proj` gate
  (line 727) so init-equivalence holds.
- **Call site** (the F-source branch, model.py:1560–1585): `src == "xattn"`
  follows the corr branch shape — stash full token sets, call the module on (cur, prev).
  Under TBPTT the tokens are detached data; the xattn module trains through
  the aux loss via the cell (and e2e), same as everything else.
- **Sniff** (`_sniff_pose_gru_config:120–232`): detect via
  `pose_gru.img_encoder? no — via `img_norm` width 64 plus presence of
  `pose_gru.xattn.` keys; add the width to the identification table at 156–162.
- **Config**: `captain_gru_v3_a4_g3_f1x_finetune.yaml` — 2-line semantic diff
  from the f1c config (`pose_gru_img_feat_src: xattn`).
- Optional (only if the shuffle falsifier shows the cell ignores the
  embedding): direct supervision — stash `res["relpose"]` (a 7-d head off the
  embedding) and add a `RelPoseLoss` term patterned byte-for-byte on
  `PoseGRULoss` against `inv(P_gt(x−1))·P_gt(x)`.

### P4.3 — probe-pass drift observation (the only design that can reach F)

Give the GRU a measurement of the CURRENT view's pose (which contains the
drift) before the conditioning ray is built: run the decoder group step once
WITHOUT pose conditioning (view gets `masked_ray_map_token`, exactly the
view-0 path at model.py:1555) on a THROWAWAY copy of
`(state_feat, mem)`, read the probe's `res["camera_pose"]` — this is
P̂_probe(x), an observation the oracle proved sufficient — then run the real
pass with the ray built from the GRU's refinement of
`[P(x−1), Δ, P̂_probe(x)]`.

**DESIGN SETTLED 2026-08-12** by the implementing session, which read the
code rather than trusting this section. Five corrections to the prose above
and below — all verified here, all superseding the original prescription:

1. **No state clone.** `DecoderBlock.forward` is `x = x + f(x)` with no
   in-place op anywhere on the path; `LocalMemory.inquire` is read-only and
   `update_mem` returns a new tensor — and the probe never calls it at all
   (saving 3.4e10 FLOPs/view). A defensive clone would also HIDE an in-place
   write on `mem`, which at view 0 is a stride-0 `.expand()` of a trainable
   Parameter. Ship bitwise-unchanged ASSERTIONS instead of a clone. The probe
   must be a separate stateless helper that never re-enters
   `_forward_decoder_group_step` (which would rewrite the pose/hidden/img
   stashes and double-count the ray-gate telemetry).
2. **Cost is +24% per pass, not "~2× decoder FLOPs".** Measured with
   FlopCounterMode at real shapes against the 3.75 s/step in
   `logs/gru_v3_ft_44496322.out`: decoder 2.29e11, full DPT head 1.82e11,
   `pose_head` alone 4.8e6 (0.003% of the head). Reading
   `downstream_head.pose_head` directly costs +24%/pass; re-entering
   `_downstream_head` costs +43%/pass because the three DPT adapters
   (1.75e11, 96% of the head) run in FP32. N=2 → 4.65 s/step (26 h);
   N=4 → 6.4 s/step (36 h). Both fit the 3-day walltime; eval is +39% per
   extra pass over 4292 scenes. **Take the cheap path.**
3. **No new `input_mode` value.** Do NOT introduce `"pose_delta_probe"`.
   `input_mode` stays `"pose_delta"` and the width is derived from
   `pose_gru_refine_passes` alone — one key, one mechanism, per GOVERNING
   CONSTRAINT 2. Two parallel expressions of the same state can desync; one
   cannot. The sniff hard-fails both directions (width 21 with passes<2, or
   width 7/14 with passes≥2).
4. **The sniff has FOUR edit sites, not one**: the `base_width` assert, the
   no-projector candidate solver `for base in (7, 14)` above it,
   `PoseGRU.__init__`'s `assert input_mode in (...)`, and the `input_dim`
   line.
5. **`probe_every` is not a pure cost knob.** A held observation is an
   ABSOLUTE pose from j frames ago and the cell gets no staleness signal to
   distinguish fresh from stale. Default k=1; measure what staleness actually
   costs with a `POSE_GRU_PROBE_LAG` falsifier on a trained checkpoint rather
   than assuming drift is slow enough.

**Why this cannot repeat the rung-0 failure:** the observation comes out of
the same `pose_head`, the same postprocess, and the same view-0-relative
frame as the fed-back pose, so it is in head scale BY CONSTRUCTION. The
scale-shock failure mode is structurally unavailable here. That is the
strongest argument for P4.3 over anything GT-flavoured.

**Init-equivalence is stronger than the F lever's:** zero-init
`cell.weight_ih[:, 14:21]` across all three gates makes the 21-wide arm
reproduce the 14-wide arm bit-for-bit *including the hidden trajectory*.

**Lever placement:** `pose_gru_refine_passes` lives on the PoseGRU module
(like `iters`), never as a `base_model` attribute — no eval surface sets
model attrs, so a trainer-only attribute would be silently OFF at eval and a
trained 21-wide cell would meet a 14-wide input. Because N is invisible in
weight shapes (only "≥2" is), a `POSE_GRU_FORCE_REFINE` env override mirrors
`POSE_GRU_FORCE_ITERS` and yields the anytime-refinement curve from ONE
trained arm without retraining.

**Loss-scale caution:** iterates/view = (N−1)·R, so the γ-sum moves with N.
N=2 gives 8 iterates — identical to the scored `gru_a4g3r8` row, making N=2 a
strict single-variable A/B. N=3 → 4.8593, N=4 → 4.9764; renormalise with
`pose_gru_loss_weight` 0.8563 / 0.8362 if the N sweep should not also be a
loss-scale sweep.

**Not independent of P2.** The ray gate reads the POST-update hidden, which
under P4.3 has consumed the probe observation. **The first refine arm must
run gate-off**; a refine-vs-raygate A/B would confound the two.

**THE PREMISE IS TESTABLE BEFORE THE 8-NODE LAUNCH.** The entire bet is that
a pose-free readout of the CURRENT frame beats the stale fed-back pose. That
needs no trained refine arm: run the ordinary conditioned rollout on the
already-scored `gru_a4g3r8` checkpoint and additionally decode pose-free at
each view, logging the error and feeding it to nothing. One GPU, dozens of
scenes, minutes. Design rules, all learned the hard way:

- **One-sided.** The probe on r8 is OUT OF DISTRIBUTION — that model was
  trained with a ray token on every view x>0, and "rich state, no ray token"
  is a condition it never saw (view 0 has no accumulated state, so it is not
  the same case). So `e_probe < e_lag` ⇒ STRONG GO; `e_probe ≥ e_lag` ⇒
  SUGGESTIVE, NOT REFUTED — you measured that an untrained-for probe fails.
- **The arm-A bracket is DEGENERATE — my error, corrected 2026-08-12.** Arm A
  has no conditioning loop, so its "readout" and its "trajectory" are the SAME
  TENSOR: `e_probe_t` and `e_head_t` are bit-identical at every band in
  `p43_premise_v3.json` (and correctly differ for r8). Comparing them tests
  whether one estimator beats an extrapolation of its own two previous
  outputs — i.e. temporal smoothness, guaranteed for any state-recurrent
  model. **A model cannot observe its own error by asking itself.** So the
  bracket does NOT close the OOD escape hatch; only the r8 end is usable, and
  that end is pre-registered as SUGGESTIVE, NOT REFUTING.
  What arm A DOES establish, non-degenerately: by late views its pose readout
  is nearly predictable from its own two previous outputs (p/vel 0.985), i.e.
  the head has become a smooth extrapolator barely reading the current image.
  That is architectural and it is the strongest pessimistic evidence here.
- **Normalize within each model.** Arm A and r8 have different absolute head
  scales, so cross-model error comparison is a units comparison. Report the
  per-view RATIO `e_readout / e_lag` computed inside each model's own
  rollout — scale-free, and the convention the PoseGRU docstring already uses
  ("1.00 means the module learned nothing").
- **The real bar is velocity extrapolation, not lag.** `e_lag` contains the
  true inter-frame MOTION, so anything modelling motion beats it — including
  a pure trajectory extrapolator that never looks at an image (§0.3: a
  perfect one takes 1.09 → 0.84, ~23% below lag). The claim P4.3 rests on is
  that the probe observes DRIFT, which is precisely **`e_probe < e_vel`**
  where `e_vel = err(P̂(x−1) ∘ Δ, GT(x))`. Δ is already computed at the call
  site; composing is one matmul in `utils.camera`. Beating `e_lag` alone is
  consistent with the probe merely re-deriving motion the cell already has —
  i.e. the ≤20–25% trajectory-only ceiling restated.
- **The discriminator is the SLOPE over view index, not the mean.** The claim
  is that the probe sees ACCUMULATED drift, so `e_lag` should grow with x
  while `e_probe` grows more slowly. A sequence mean destroys exactly that.
  Log per view. (`e_lag`'s slope is also a measurement of head drift nobody
  has logged, worth having whether or not this arm runs.)
- Free assertion: at step 0 the zero-init residual head makes
  `P_committed = P̂(x−1)` exactly, so `e_gru` must EQUAL `e_lag` bit-for-bit.
- Caveat: these are RAW errors, not the per-batch-normalized quantities
  `PoseGRULoss` reports as `gru_trans_err_gtscale`. Never cross-read the two.

### PREMISE TEST RESULT (2026-08-12, 100 test scenes × 64 views, both arms)

Per-band ratios, normalized per side by each sequence's own `?avg_dis` factor
(v1 of the instrument differenced head-scale against GT-scale RAW and measured
the ~4.8× scale mismatch instead of pose error — the same class of failure as
rung 0, caught mid-flight):

| arm | band | e_probe/e_vel | e_probe/e_lag |
|---|---|---|---|
| r8 (OOD probe) | mid 17–47 | **1.066** | 1.077 |
| r8 (OOD probe) | late 48–63 | **1.051** | 1.068 |
| arm A (degenerate) | mid | 0.974 | 0.985 |
| arm A (degenerate) | late | 0.985 | 0.990 |

Not saturated: `e_probe/e_null` is 0.325–0.375 late, so every estimator is
~3× better than predicting the origin and the ratios have real dynamic range.
Common-mode share at late views: r8 **59%**, arm A 83%.

**Verdict: SUGGESTIVE NEGATIVE, NOT A REFUTATION.** Established: an
untrained-for probe inside a conditioned model is 5–7% worse than velocity
extrapolation mid/late. NOT established, and untestable without training it:
whether a TRAINED-FOR probe inside a CONDITIONED model behaves differently —
the one configuration P4.3 proposes and the one nothing here measures.

**Caveat on the slope argument.** "Advantage decays with view index" was
offered as corroboration, but arm A cannot corroborate (degenerate above), so
it rests on r8 alone — and OOD severity itself grows with x, since late views
carry more conditioning-dependent state the probe was never trained to
decode. Decaying advantage and growing OOD-ness predict the same curve; r8
cannot separate them.

**THE MAGNITUDE COMPARISON IS THE WRONG TEST.** `e_probe > e_vel` says the
probe is a worse POINT ESTIMATE. It does not say the probe carries no drift
information — a signal with larger error is still informative if its error is
DECORRELATED from the trajectory's, and r8's probe is only 59% common-mode,
i.e. ~41% independent content whose direction nothing has measured. The
decisive quantity is the cosine between the proposed and the true correction:

    u = O(x) − P̂(x−1)      w = GT(x) − P̂(x−1)      cos = <u,w>/(|u||w|)

with optimal gain `a* = <u,w>/<u,u>` leaving residual `|w|²(1 − cos²)`, so
**`cos²` is exactly the best fractional error reduction ANY corrector could
extract from this observation** — one scalar per band, directly comparable to
the ≤20–25% trajectory-only ceiling. `cos² ≈ 0` late ⇒ genuinely refuted, on
the right quantity. `cos²` materially > 0 while `e_probe > e_vel` ⇒ the probe
is informative and the premise test measured the wrong thing. Costs three
extra logged scalars per view on the same minutes-long rollout.

**THE DECIDING NUMBER IS PARTIAL, NOT MARGINAL.** `cos²_probe > cos²_vel` is
NOT sufficient: both are marginal fits and the two proposals overlap, since
the probe and velocity both partly encode inter-frame motion. A probe can
score a higher marginal cos² while adding nothing the cell cannot already get
from Δ — the trajectory-only ceiling restated in new units, which is the
confusion this program keeps re-hitting. The P4.3-specific claim is
INCREMENTAL:

    minimize over (a,b):  |w − a·u_vel − b·u_probe|²
    R²_joint = [S_vw S_pw] G⁻¹ [S_vw S_pw]ᵀ / S_ww ,  G = [[S_vv, S_vp],[S_vp, S_pp]]
    **ΔR² = R²_joint − cos²_vel**   ← P4.3 lives or dies here

Six pooled scalars per band, and the load-bearing one is the cross term
`S_vp = <u_vel, u_probe>` — without it the joint fit is not identifiable and
ΔR² cannot be recovered after the fact. Pre-registered reading:

- **ΔR² ≈ 0 mid/late** ⇒ the probe is REDUNDANT with velocity. P4.3 refuted on
  the right quantity and for the right reason — the observation carries
  nothing the trajectory lacks. A cleaner negative than the magnitude result.
- **ΔR² materially > 0** ⇒ drift-correlated signal no trajectory-only
  corrector can reach. ΔR² is then literally the headroom ABOVE the ≤20–25%
  ceiling, in the same units.
- **cos²_probe large with ΔR² ≈ 0 is the trap.** Always report both.

Guard: near-collinear `u_vel`/`u_probe` makes G singular — which is itself the
ΔR² ≈ 0 finding. Report `S_vp/√(S_vv·S_pp)` and refuse to invert past |corr|
> 0.99.

Pool per band (`a* = ΣS_uw/ΣS_uu`), never mean-of-per-sample-cos²: the latter
is a per-sample ORACLE gain, a looser bound than any deployable corrector.
Rotation via the log map with `w ≥ 0` quaternion standardization so the double
cover cannot flip the axis. Form `u` and `w` AFTER per-side scale
normalization — `u` is pred-side while `w` mixes pred- and GT-side, so raw
inner products put the ~4.8× head/GT mismatch straight into the numerator, and
a scale offset along the GT direction INFLATES `<u,w>` rather than cancelling.
That would manufacture a false positive, the worst available failure here.

**ΔR² RESULT (2026-08-12, r8, 100 scenes, scene-block bootstrap 1000 draws).**
The pre-registered negative did NOT occur. Every band's 95% CI excludes zero:

| band | ΔR² trans (95% CI) | ΔR² rot | corr(u_vel, u_probe) |
|---|---|---|---|
| early | +0.0120 [+0.0003, +0.0505] | +0.0498 [+0.0231, +0.0936] | +0.258 |
| mid | **+0.0400 [+0.0092, +0.0819]** | +0.0108 [+0.0000, +0.0812] | +0.105 |
| late | +0.0287 [+0.0003, +0.1177] | +0.0154 [+0.0001, +0.0892] | **−0.013** |

The redundancy trap did NOT fire: probe and velocity are near-orthogonal where
drift dominates, so the contribution is genuinely additive. Given the probe is
OOD, this is a LOWER bound.

**Read it with four honest caveats:**

1. **Magnitude is weakly determined.** ΔR² = 0.03 is a **1.5% best-case RMS
   reduction** on the correction vector, NOT 3% — cos² is a fraction of
   SQUARED error while the ≤20–25% ceiling is a fraction of error. Do not
   state ΔR² as headroom against that ceiling; that mixes axes exactly as the
   retired "60–70 points" figure did. CIs span a factor of ~4.
2. **The thesis's distinctive prediction is UNCONFIRMED.** Point estimates go
   0.0120 → 0.0400 → 0.0287: flat-to-declining where the thesis says the
   effect should GROW with accumulated drift. The OOD confound (late views are
   more OOD) makes declining ΔR² and rising OOD severity observationally
   identical here, so this does not contradict the thesis — but "strongest
   where drift dominates" is NOT established, and late is the weakest cell.
3. **Six bands from 100 scenes are not six independent confirmations.** The
   coherence argument is sign-consistency across two independent physical
   quantities and three regimes plus a predicting mechanism — not the count.
4. **Scale check.** The C→B ATE gap is 0.0686; a naive linear read of 1.5% is
   ~0.001, i.e. LESS than the 0.0021 the existing R8 arm already delivers. The
   pose→ATE map is strongly nonlinear (B has zero pose error and ~6× better
   ATE), so this is a sanity check, not a forecast.

**INDEPENDENT AND POSSIBLY BIGGER: cos²_vel COLLAPSES.** Translation
0.0207 early → 0.0001 mid → 0.0031 late. Constant-velocity extrapolation
explains essentially nothing of the needed correction where drift dominates.
MEASURED consequence: the A4 `pose_delta` lever hands the cell a velocity
block that is near-useless in the regime that matters, its usable content
concentrated early (and small even there). INFERRED, and stated no more
strongly: this CONTRADICTS the dossier's "a perfect velocity extrapolator
recovers ≤20–25%" carried in §0.3 — that figure cannot describe the late
regime. Record the contradiction and the measurement; **do not substitute a
replacement number**, since nobody has reproduced the original computation.
Fifth dossier figure to fail on contact with measurement.

Caveat riding with every number above: one checkpoint family, 100 scenes, one
dataset.

**THE SHORT-RUN RUNG (~3 h, not 26 h).** The premise test structurally cannot
answer whether a TRAINED-FOR probe beats an OOD one. A 5–10 epoch refine run
can, with three pre-registered readouts:
(1) does `||cell.weight_ih[:, 14:21]||` move off zero at all;
(2) does ΔR² on the partially-trained checkpoint EXCEED the 0.03 OOD lower
bound;
(3) **does ΔR² INCREASE with view index once trained** — the thesis's
signature, currently unconfirmed, and cleanly measurable only here because
training removes the OOD confound. A trained probe still peaking mid and
decaying late would mean it carries MOTION information (early/mid), not DRIFT
— the trajectory-only ceiling wearing a new hat.

**PROGRAM-LEVEL RULE — the dead-falsifier window.** EVERY zero-init channel
this program adds has a window in which its own falsifier reports "unused"
regardless of whether the mechanism works: a zero-weight channel is immune to
a shuffled input. `POSE_GRU_PROBE_SHUFFLE` is a no-op while
`cell.weight_ih[:, 14:21]` is zero, exactly as `POSE_GRU_FORCE_ITERS` is under
a locked zero-init head. Applies to `img_proj`, the residual head, the ray
gate's bias, and the probe columns. **The weight-norm check must precede the
falsifier**, always.

**Scoreability:** `pose_gru_refine_passes` / `pose_gru_probe_every` must be
registered in `preflight_ckpt.py`'s lever list — but only AFTER the load path
handles them. Registering a key whose loader does not exist is the exact
silent-wrong-row failure that list guards against.

---

## §5 — Launch ladder

**Base every new arm on A4×G3×R8, not R1.** `gru_a4g3r8` is the only lever
combination that beats C (54.2% ATE win, 0.079918); building on the R1 arm
would start 2.2% of the gap further back for no reason. Its config is
`captain_gru_v3_a4_g3_r8_finetune.yaml`.

| # | run | training cost | contents | measures | baselines retrained? |
|---|---|---|---|---|---|
| 0 | **gru-noise-floor probe** | 1 node, ~5 min | `--oracle gt` on the **honest** `captain_gru_v3_a4_g3_finetune` ckpt, `LIMIT=20`, `diag_` label, scratch OUT_ROOT | how much the GRU DEGRADES a perfect input — the noise floor P2's abstention gate has to beat. NOT a bound (arm B already is the bound) | no |
| 1 | f1c / f1d / f1r | **already running, already armed** | nothing to build, nothing to arm (P4.1) | whether any existing image source carries motion evidence | no |
| 2 | **v4-reltarget A/B** | 8 nodes × 2 | P0 (§0b): `pose_gru_loss_target` absolute vs relative, on R8 | whether the module can learn at all once the unlearnable drift term leaves the objective | no |
| 3 | v4-grusync ×2 | 8 nodes × 2 | §1, on top of the P0 winner, at `pose_gru_lr_scale` 100 AND 10 | GRU gradient noise; the lr arm is required because the evidence's gate-saturation fix is lr_scale 100→10 | no |
| 4 | **v4-refine** ← MAIN LINE | 8 nodes | **P4.3 iterative refinement** (§4): image-grounded pose observation, refinement count as a lever | the ONLY rung whose ceiling is the stated objective — winning actively, not abstaining | no (GRU-input change only; trunk weights and forward unchanged) |
| — | v4-raygate | 8 nodes | §2 gate, **as an instrument** | a map of WHERE the corrector fails; feeds the v4-refine design. Never a headline result | no |
| 5 | v4c | 8 nodes | v4-refine + `img_feat_src=xattn` (P4.2) | whether learned motion evidence adds anything on top of a real observation | no |
| — | *(deferred)* v5-ddp | 8 nodes ×4 | full `ddp_grad_sync` for the trunk | the common-mode effective-batch-4 defect | **YES — A/B/C all re-run. Do not launch under the current constraint.** |

Rung 0 replaces the refuted oracle-benchmark idea (see the ladder section).
It runs on the HONEST checkpoint — same LR family as every table arm, no
circularity — and answers a question nothing else answers: given a perfect
pose at its input, how much does this GRU still wreck the conditioning?
`gru_gain_trans = 0.332` from the oracle run says the answer is "a lot", and
20 scenes is enough to confirm it. Requires the `ORACLE` passthrough in
`eval_pipeline/run_captain_ray_eval_node.sh` (landed 2026-08-11, defaults to
`off` so every existing caller is byte-identical).

**Two hazards that apply to ANY hand-rolled eval submission:**

- `eval_pipeline/mn5_paths.sh:33` defaults `OUT_ROOT` to
  **`/gpfs/projects/...`**, which has only ~381 GB free against a ~332 GB
  prediction footprint — and that volume holds the live `output_dir` of the
  three running training jobs. `arm_eval_for_run.sh:28` forces scratch;
  a bare `sbatch run_captain_ray_eval_node.sh` does NOT. **Always export
  `OUT_ROOT=/gpfs/scratch/etur59/koc821022/outputs` explicitly**, and pass
  `--output/--error` so the 8 `#SBATCH` log lines do not land on projects
  either.
- Bypassing `arm_eval_for_run.sh` also drops its `unset` of the falsifier env
  vars (it submits with `--export=ALL`). `POSE_GRU_FORCE_ITERS` would
  override the R lever and `PREV_PRED_RAY_SHUFFLE` is re-applied *after* the
  oracle substitution — it would silently roll the injected GT. Run
  `env | grep -E 'PREV_PRED_RAY_SHUFFLE|GT_RAY_MAP_SHUFFLE|POSE_GRU_'`
  before submitting anything by hand.

Every rung leaves arm C a valid control, so the closure fraction (C−X)/(C−B)
stays meaningful throughout. Each run passes the §0 gates before promotion.
v4a's config (`captain_gru_v3_a4_g3_r8_v4a_finetune.yaml`) is a semantic diff
from the grusync config adding `pose_gru_gate`, `pose_gru_rot_weight: 8.0`,
`pose_gru_full_window`, `pose_gru_clip: 5.0`, `pose_gru_input_gain: [...]` —
every lever individually revertable from the CLI.

**Capacity note.** Three 8-node jobs are running (24 nodes, ~2.5 days left
each). `gpfs_projects` holds 643 GB of a 1024 GB group quota with each 50-epoch
run costing ~50 GB, so there is room for roughly seven more runs before the
quota bites — and a full disk is exactly what killed jobs 44422987/44422993 at
epoch 40 on 08-09. Check `bsc_quota` before each launch and prune superseded
run dirs.

## §6 — Sequencing rationale

0a. **P0 is the first code change** (§0b): retargeting `PoseGRULoss` to the
   relative motion is ~2 h, touches one file, and is the only proposal that
   changes WHAT the module is asked to learn rather than how hard it is
   pushed. Every other §3 lever is a multiplier on a signal that is 94%
   unlearnable until this lands, so doing it first changes how the rest
   should be tuned.
0b. **Rung 0 before any GPU-hours** — the GRU noise-floor probe (1 node,
   ~5 min). NOT the 8-node oracle benchmark, which was refuted (see the
   ladder section: it duplicates arm B, its GRU emits a translation 3
   GT-scales wrong, and it carries a 10× LR confound).

   ```bash
   env | grep -E 'PREV_PRED_RAY_SHUFFLE|GT_RAY_MAP_SHUFFLE|POSE_GRU_'   # must be empty
   CR=/gpfs/scratch/etur59/koc821022/checkpoints_projects/captain_cut3r_finetune_aug_full
   OUT_ROOT=/gpfs/scratch/etur59/koc821022/outputs \
   sbatch --nodes=1 --gres=gpu:4 --cpus-per-task=80 \
          --job-name=diag_oracle_honest \
          --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/diag_%j.out \
          --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/diag_%j.err \
          --export=ALL,OUT_ROOT=/gpfs/scratch/etur59/koc821022/outputs,\
LABEL=diag_a4g3_oraclegt,CONDITIONING=prev_pred_gru,ORACLE=gt,\
CKPT=$CR/captain_gru_v3_a4_g3_finetune/checkpoint-final.pth,\
BASE_SHARD=0,NUM_SHARDS=4,LIMIT=20 \
          eval_pipeline/run_captain_ray_eval_node.sh
   ```

   Uses the **honest** a4_g3 checkpoint (lr 1e-5, same family as every table
   arm — no circularity, no LR confound). Compare its 20 per-scene rows
   against the same scenes in `summary/per_scene_gtray_lr1e5.csv`. Two
   possible readings, both informative: if it lands near B, the GRU passes a
   perfect pose through and the whole problem is input quality; if it lands
   far from B (which `gru_gain_trans = 0.332` predicts), the GRU degrades even
   a perfect input and P2's abstention gate is the priority. The `diag_`
   prefix keeps it out of the arm namespace — **never aggregate it into
   `averages_table.csv`**.
1. **P1 first (code)** — it is the cheapest large change to the GRU's optimization
   (effective batch 4 → 128 for 136k params), it is confined to the module by
   construction, and everything downstream is evaluated under it. Most of the
   machinery already exists (`all_reduce_grads_`, the flag plumbing, the
   6-check falsifier); what is new is the `contributing` accounting, the
   `sync_params` list, `pad_gru_sync`, and falsifier checks 7–8. Blocking for
   all launches.
2. **P3.1, P3.3 next** — a few lines each in `losses.py` / `misc.py`, no
   probe dependency, and P3.3 reuses the `sync_params` list P1 introduces.
3. **P2 gate** — small module change plus a sniff entry; belongs in v4a so
   the non-harm guarantee is part of the first repaired run rather than
   retrofitted after another D-style regression.
4. **P3.2 full-window** — touches the TBPTT loop, so implement after P1
   (same file; its backward-only calls must be written against P1's
   round-count invariant so the pad arithmetic stays correct).
5. **P3.4 input gain** — needs the measurement from
   `probe_gru_input_stats.py`; run that probe against a D-arm snapshot while
   v4-grusync trains.
6. **Launch v4a**; only then **P4.1** (config-only) and, in parallel with its
   training, implement **P4.2**. **P4.3 last**, and only if v4c misses the
   promotion bar — it is the most invasive and its necessity depends on
   v4b/v4c results.

Rationale in one line: repair the GRU's own training substrate first — it is
where the analysis found the waste and it is reachable without touching the
trunk — then buy observability in order of implementation cost, because
observability, not optimization, is what separates a motion-term corrector
from the oracle. (Do not quote a "60–70 points" figure — that subtracted two
quantities measured on different axes; see §0.3.) The trunk's own defect is real but common-mode, so it waits
until retraining the baselines is acceptable.

---

## §7 — Preflight: run this before touching any code

Line numbers in this document drift. `model.py` shifted +9 lines in the
call-site region on 2026-08-11 *while this directive was being written*, so
treat every `file:line` here as a hint and the SYMBOL as the truth.

```
cd /gpfs/home/koc/koc821022/my-da3
python verify_plan_anchors.py          # 0 = all anchors exact; 1 = drift/unresolved
python verify_plan_anchors.py --fix-doc   # rewrite stale numbers into this file
```

`verify_plan_anchors.py` resolves all 39 edit sites cited below by regex and
reports where each lives now, plus the presence of every config, launcher and
harness the plan invokes. Verified 39/39 exact on 2026-08-11. An `UNRESOLVED`
line means the code around that anchor genuinely changed — re-read that
region before editing, never edit blind.

### Experiment state as of 2026-08-12 16:30 (verify before acting)

- **Running**: two 8-node finetunes — `f1r_r8_noproj` (job 44521973) and
  `f1d_r8_noproj` (44521974), launched ~10:00 2026-08-12 at the user's direct
  request, ~2d18h remaining. Each is an exact re-run of its projected twin
  with ONE key added (`pose_gru_img_feat_proj: False`); verified mechanically
  as 73→74 yaml keys, zero removals, `exp_name` the only other change. Runtime
  wiring confirmed from the logs: f1r cell input 1038 (14 + 2×512, resnet18),
  f1d 782 (14 + 2×384, dinov2_vits14), both `proj=False`, R8 preserved
  (iters=8, γ=0.8), lr 1e-5, 32 GPUs, no new-code paths active. Both are armed
  and will score automatically. Their configs are still UNTRACKED:
  `src/CUT3R/config/captain_gru_v3_a4_g3_f1{r,d}_r8_noproj_finetune.yaml`.
  Also running: `p43_premise` (44536126), a P4.3 diagnostic, not an arm.
- **`f1c` / `f1d` / `f1r` are FINISHED — this file said "training now" until
  2026-08-12 and was wrong.** All three completed 50 epochs in a SINGLE
  contiguous segment (verified: one `Start training` line each, 51 log rows,
  a `Training time` line each): f1r 19:46:11 and f1d 19:45:55, both ending
  06:23 2026-08-12; f1c 19:32:45, ending 10:23 2026-08-12. All three are now
  scored at 4292/4292 scenes.
- **Finished and scored** (rows in `averages_table.csv`, now 12): `gru_a4g3`,
  `gru_a4g3f1`, `gru_a4g3r8`, `gru_a4g3f1r8`, `gru_a4g3f0np`, `gru_a4g3f1np`,
  `gru_a4g3f1rr8`, `gru_a4g3f1dr8`, `gru_a4g3f1cr8`, plus baselines
  `augfull_lr1e5` / `gtray_lr1e5` / `prevpred_lr1e5`.

  | arm | absrel | a1 | ATE | rpe_t | rpe_rot | feature source |
  |---|---|---|---|---|---|---|
  | `gru_a4g3f1rr8` | 0.1979 | 0.7609 | 0.0822 | 0.0089 | 1.2566 | ResNet-18 |
  | `gru_a4g3f1dr8` | 0.2000 | 0.7577 | 0.0812 | 0.0087 | 1.2334 | DINOv2-S |
  | `gru_a4g3f1cr8` | 0.1989 | 0.7579 | 0.0799 | 0.0090 | 1.2016 | corr stats |

  All three: 4292/4292 scenes, 0 orphan claims, all workers rc=0,
  `conditioning=prev_pred_gru`, no falsifier env leakage.

  **READ BOTH CONFOUNDS BEFORE USING THIS TABLE.** (1) The total ATE spread
  across all three feature sources is ≤0.0023 — at or below the median
  arm-to-arm scatter of 0.00281, so per the noise-floor warning these three
  rows do NOT rank their feature sources. (2) None of the three configs sets
  `pose_gru_img_feat_proj`, so all inherited the default `True` and every
  encoder was squeezed through the zero-init 32-D projector (DINOv2
  2×384→32 is a 24:1 bottleneck). "DINOv2 is no better than pooled" therefore
  does NOT separate feature quality from the bottleneck — that is what the two
  running noproj arms exist to test.

- **Trainer-side test loss, same three arms** (`loss_avg`, best epoch and
  epochs 41–50 mean). This is the trainer's 4-view test metric and is a
  DIFFERENT quantity from the 4292-scene harness ATE above — do not
  cross-read the two:

  | arm | best `loss_avg` | @epoch | tail-10 mean |
  |---|---|---|---|
  | `f1r` (resnet18) | 2.6009 | 47 | 3.065 |
  | `f1d` (dinov2) | 2.6673 | 47 | 3.116 |
  | `f1_r8` (pooled, proj) | 2.8281 | 47 | 3.140 |
  | **`r8` (F-OFF control)** | **2.5725** | 27 | **2.794** |

  Within R8 — identical iters, γ, LR, schedule, criterion strings and test set
  — **the F-off control beats all three projected image-feature arms** on both
  best-epoch and tail-10 mean. Adding a projected image feature at R8 cost
  ~0.27–0.35 test loss. resnet18 vs dinov2 are indistinguishable: 7/6
  metric-by-metric split, and the e47 gap of 0.066 sits well inside either
  run's tail-10 stdev (±0.261, ±0.296).

- **THREE LOG-READING HAZARDS, verified, that corrupt any naive arm
  comparison** (found 2026-08-12):
  1. `f0_noproj` (FOUR `Start training` lines, i.e. 3 restarts; duplicate
     epoch rows at 2 and 50), `f1_finetune` and base `_finetune` (2 segments
     each, duplicate at 40) have RESTARTS in `log.txt`. Their `Training time`
     line covers only the FINAL segment. Resolve duplicates by taking the LAST
     row per epoch, or you double-count and mix segments.
  2. **`best loss_avg` is not a like-for-like statistic across arms.** `base`
     (2.121), `f0_noproj` (2.293) and `oracle32` (1.825) all have their best at
     **epoch 1** — an early-training value, not a converged one. Any arm-to-arm
     gap computed off best-epoch inherits this, independent of seed variance.
  3. **`orig/loss_avg` does not discriminate and must not be used to score the
     F lever.** It moves the WRONG WAY in every arm — roughly doubles over 50
     epochs, minimum always at epoch 1–5 — including both F-off controls and
     the oracle. At e50 all nine arms land in 18.3–20.5, a spread comparable to
     one arm's own tail-10 stdev.
- **Within-run epoch scatter, as a LOWER bound on run-to-run variance**:
  tail-10 stdev of `loss_avg` is ±0.261 (f1r) and ±0.296 (f1d). A single-epoch
  arm-vs-arm comparison already carries ~±0.28 from temporal scatter alone,
  before any seed effect. This is a floor, NOT the run-to-run number the
  program still lacks.
- **Finished but NEVER scored at 4292 scenes**:
  `captain_gru_v3_a4_g3_oracle_finetune_32gpu` (idle since 08-07). Rung 0.
- **Baselines A/B/C live in
  `/gpfs/scratch/etur59/koc821022/checkpoints_projects/cut3r_finetune_baselines/`**
  as `cut3r_finetune_aug_full{,_gtray,_prevpred}_32gpu_lr1e5`. NOTE:
  `checkpoints/cut3r_multinode/` — the 64-GPU b2_lr1e5 family — **has been
  deleted**, though 15 configs under `src/CUT3R/config/` still reference that
  path. Do not point anything at it.
- **Dead run to wipe before any grusync/ddpsync relaunch**:
  `captain_cut3r_finetune_aug_full/captain_gru_v3_a4_g3_ddpsync_finetune` —
  no checkpoints, but it holds `wandb_run_id.txt`
  (`fta4g3ddpsync20260809173946`) and one offline wandb dir, which auto-resume
  would pick up.
- **Data**: both DROID split roots are intact — 38633 train / 4292 test
  symlinks, zero dangling.
- **Disk**: gpfs_projects 643 GB / 1024 GB group quota (406 GB of it is
  `captain_cut3r_finetune_aug_full`); gpfs_scratch 18.6 TB / 29.3 TB. Eval
  writes ~332 GB of predictions per arm to gpfs_scratch.

### Repo state this plan assumes

- Branch `captain_gru_v3`; HEAD `aab1778`. **The working tree carries large
  uncommitted changes** — `model.py` (+669 lines), `train_cut3r_baseline.py`,
  `croco/utils/misc.py`, `losses.py` — which include the entire A5/F/R lever
  grid and the existing `ddp_grad_sync` scaffolding. Do NOT `git checkout` or
  `stash` these; the plan builds on them.
- Untracked and load-bearing: `GRU_GAP_CLOSURE_DIRECTIVE.md`,
  `verify_plan_anchors.py`, `verify_ddp_grad_sync.py`,
  `src/CUT3R/config/captain_gru_v3_a4_g3_ddpsync_finetune.yaml`.
- Commit before starting a proposal, so each proposal is one reviewable diff.

### Conventions every proposal follows

1. Default-off config key; absent key → exact previous behavior. See
   GOVERNING CONSTRAINT 2 at the top — a hard invariant, not a style
   preference, enforced by `verify_backward_compat.py`.
2. New `pose_gru` sub-modules must be recoverable by `_sniff_pose_gru_config`
   (model.py:77) from weight-key presence, because the loader hard-fails on
   shape mismatch and mode is otherwise invisible in weight shapes.
3. New buffers/params must serialize into the checkpoint so offline eval
   nodes rebuild bit-exact without pretrained files.
4. Anything claiming to use a channel must have a falsifier that kills it.

---

## §8 — How to execute this from a fresh session

One proposal per session, one commit per session. Open every session with:

```
Read GRU_GAP_CLOSURE_DIRECTIVE.md, then run `python verify_plan_anchors.py`
and stop if it exits non-zero. Implement <PROPOSAL> exactly as specified,
default-off, and do not touch the trunk or the three running jobs.
```

### Session order

| # | session | GPU? | duration | produces |
|---|---|---|---|---|
| S1 | Rung 0 + arm the watcher | eval only | ~30 min | oracle scored at 4292; f1c/f1d/f1r armed |
| S2 | **P0** — `pose_gru_loss_target: relative` (§0b) | no | ~2 h | one loss file + config key |
| S3 | **P1** — `pose_gru_grad_sync` (§1) | login-node falsifier | ~3 h | misc.py + inference.py + trainer + checks 7–8 |
| S4 | **P2** — ray-level abstention gate (§2) | no | ~2 h | model.py blend + sniff + telemetry |
| S5 | **P3.1 + P3.3 + P3.4 probe** (§3) | 1 GPU for the probe | ~2 h | loss weights, GRU clip, input-stat JSON |
| S6 | **P3.2** — full-window GRU supervision (§3) | no | ~2 h | enable_grad island + backward-only calls |
| S7 | configs + launches | 8 nodes/arm | ~1 h + queue | v4-reltarget A/B, then grusync ×2 |
| S8 | **P4.2** xattn source (§4) | no | ~4 h | img_encoders.py + source lever |
| S9 | **P4.3** probe pass (§4) | no | ~1 day | only if v4c misses the bar |

S2–S6 need no allocation and can run in any order among themselves; they are
listed in the order that makes each later one easier to tune.

### Scheduling against the cluster

Three 8-node jobs (`f1c`, `f1d`, `f1r`) occupy 24 nodes until roughly
**2026-08-14**. Do S1 immediately (the eval queue is separate from those
allocations), then spend the waiting time on S2–S6, so that the moment those
jobs land there is a fully-built, falsified code base ready to launch. Do not
queue new 8-node trainings before then — they would contend with the arms
whose results decide what v4a should even contain.

### Hard guardrails for every session

1. `python verify_plan_anchors.py` first; non-zero exit means re-read, not edit.
2. Never `git checkout`/`stash` the working tree — it carries the uncommitted
   lever grid **that the three running jobs are executing**.
3. Never touch `captain_gru_v3_a4_g3_f1{c,d,r}_r8_finetune` run dirs.
4. Every new key defaults off; absent key ⇒ byte-identical previous behavior.
5. No trunk changes. If a proposal seems to need one, it is deferred, not
   folded in (see the governing constraint at the top).
6. Wipe `captain_gru_v3_a4_g3_ddpsync_finetune/` before any sync relaunch —
   it has no checkpoints but does have `wandb_run_id.txt` and an offline run.
