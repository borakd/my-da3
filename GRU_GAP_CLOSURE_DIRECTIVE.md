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
- Arms `f1c` (corr), `f1d` (DINOv2) and `f1r` (resnet18) are **training now**
  and are not in this table yet.

The analysis' own 50-sequence subset ladder (A 0.001745/0.574°, B
0.000822/0.177°, C 0.001866/0.609°, D 0.001982/0.633°, E 0.002031/0.606°,
F oracle 0.000996/0.235°) is a DIFFERENT normalization and scene count — the
two are not comparable numerically, but every ordering above matches it. The
oracle arm F has no row in the full-benchmark table: it was never evaluated at
4292 scenes. **Evaluating the oracle checkpoint on the full benchmark is the
cheapest missing measurement in this whole program** — it is the only number
that sets the real upper bound for the fix ladder, its checkpoint is finished
and idle (`captain_gru_v3_a4_g3_oracle_finetune_32gpu`, untouched since
08-07), and it is one `arm_eval_for_run.sh` call.

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

- `PoseGRULoss.__init__` (losses.py:1103): add `target="absolute"`,
  accepting `"relative"`.
- In `compute_loss`, under `target="relative"`, replace the per-view pair
  (`gru_pose[i]`, `gt_poses[i]`) with
  (`pose_delta_encoding(pr_poses[i-1], gru_pose[i])`,
  `pose_delta_encoding(gt_poses[i-1], gt_poses[i])`) — the GRU's motion
  relative to the head's own previous pose, against the GT motion. Skip the
  first graded view of each chunk (no `i-1` inside the window) rather than
  reaching across a TBPTT boundary.
- `pose_delta_encoding` already exists at **model.py:459** and is NOT
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

- Construct the scaler as today (line 363), then attach the parameter list
  **after** `accelerator.prepare` — `base_model` is already unwrapped at
  line 372, and this avoids any doubt about whether `prepare` rebound the
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

### P2 — abstention, at the RAY level (the head-level gate cannot clear §0)

**Design correction.** Gating the GRU's *correction* makes its floor plain
`feed_prev_pred` — arm C — and C is itself net-negative (A beats C on 63.0%
of scenes). A correction gate therefore cannot satisfy this directive's own
promotion bar. The gate has to sit where it can switch the conditioning OFF
entirely, i.e. at the ray add (`model.py:1758`):

```python
feat_group = [feat_group[0] + a * ray_out[-1].to(...) + (1 - a) * self.masked_ray_map_token]
```

`a → 0` reproduces the pose-free path view 0 already takes (the same
`masked_ray_map_token` added at model.py:1318), which is arm A's regime — so
the floor becomes A, and "stop losing to A" becomes reachable. Implement `a`
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
  `self.head` (line 751): `self.gate = nn.Linear(hidden_dim, 2)` — one logit
  for the translation correction, one for rotation — with
  `nn.init.zeros_(self.gate.weight)` and `self.gate.bias.data.fill_(4.0)`
  (sigmoid ≈ 0.982: the gate starts open, preserving init-equivalence to the
  ungated module to within 2%, and learns to CLOSE where the correction hurts).
- `forward` (line 796): after `raw = self.head(hidden)`:

  ```python
  if self.gate is not None:
      g = torch.sigmoid(self.gate(hidden))          # (B, 2)
      raw = torch.cat([raw[:, :3] * g[:, :1], raw[:, 3:7] * g[:, 1:2]], dim=-1)
  ```

  This composes with all three modes (it scales the correction/regression
  term uniformly; in `residual` mode g→0 reproduces plain `feed_prev_pred`
  exactly — the abstention semantics we want).
- **Checkpoint sniffing** — `_sniff_pose_gru_config` (line 77): detect
  `gate` from key presence, mirroring the `img_norm` pattern at line 120:
  `gate = _get("gate.weight") is not None`, pass through to `enable_pose_gru`.
- **Config plumbing**: `pose_gru_gate: true` key, added to the
  `enable_pose_gru(...)` kwargs assembled at `train_cut3r_baseline.py:245–250`. Default absent→False so every existing
  config and checkpoint is untouched.
- **Diagnostics**: log `g` means into the loss details — in `PoseGRULoss`
  this is not visible, so instead have the call site stash
  `res_group[-1]["gru_gate"] = g.detach()` (model.py:1856, next to the
  `res_group[-1]["gru_pose"]` write) and
  add `gru_gate_t/gru_gate_q` means in `PoseGRULoss.compute_loss` details
  when the key exists. A trained gate sitting at ~0 for rotation is the
  analysis' prediction — that readout alone is informative.

---

## §3 — Direction: training-signal repair (analysis: learnable skill priced at ~6% of the objective; rotation ~11%; head grad-starved 348×; hidden rms 0.005)

Four independent, cheap changes. All ride the same future launch (§5, run
v4a); none changes any default.

### P3.1 — rotation re-weight (+ optional geodesic) in PoseGRULoss

**`src/CUT3R/src/dust3r/losses.py:1084–1217`:**

- `__init__` (1103): add `rot_weight=1.0, rot_geodesic=False`.
- Line 1164: with `rot_geodesic`, replace the chordal norm by the sign-safe
  angle: `L = min(‖q̂−q‖, ‖q̂+q‖)` then `q_err = 4·asin(clamp(L/2, max=1−1e−7))`
  — the actual rotation angle in radians (the analysis' θ = 4·asin(L/2)),
  removing the q/−q double-cover ambiguity the chordal form has.
- Line 1177: `loss = t_loss + self.rot_weight * q_loss`; apply the same
  weight inside the `iter_gamma` branch (line 1211).
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
  `train_cut3r_baseline.py:110`, applied at line 344).

### P3.4 — input gain normalization (hidden throttled: rms 0.005, z≈0.89)

Measured, not assumed (round-2 forward-math audit): fed-back `absT` runs
**0.5–0.8 head-units** at supervised views (full range 0.01–3.1), while
`delta_t` is **0.018–0.068** — the informative velocity channel sits ~**70×
below** the nuisance pose columns, and its gate pre-activation std is 0.0007
vs ~0.08 for the pose block. The velocity columns must grow ~50–70× before
the hidden state is velocity-dominated. (An earlier draft said "~1e-2/1e-3
scale" — that was the pre-activation std, not the input scale. A `1/std`
probe still lands in the right place.)

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
  model.py:1685) to JSON.
- **`model.py` `PoseGRU.__init__`**: `self.register_buffer("input_gain",
  torch.ones(self.input_dim))`; in `forward` at model.py:783: `cell_in = gru_in *
  self.input_gain` (before the img_feat concat — image features are already
  LayerNormed). Buffer serializes with the checkpoint, so eval rebuilds
  bit-exact; all-ones default keeps every existing checkpoint loadable
  (`load_state_dict` picks the buffer up; for OLD checkpoints missing the
  key, load with `strict=False` handled by the existing hard-fail sniff —
  simplest: only build the buffer when a new `pose_gru_input_gain` config
  key is present, mirroring the gate's presence-gating).
- **Config**: `pose_gru_input_gain` filled from the probe's `1/std` values,
  rounded to one significant digit; quaternion dims stay 1.0. **Length must
  be `self.input_dim`, which is NOT always 14**: A4×F0 is 14, but the F1 arms
  widen the cell input to 46 (pose 14 + appended 32). Gate the buffer on the
  pose block only, or build it at `self.input_dim` and pad with ones —
  a 14-long vector will not broadcast on any F arm.
- **Probe validation**: the probe must reproduce the known numbers before its
  output is trusted — `absT` ≈ 0.5–0.8 at views 48–63, `delta_t` p50 ≈ 0.018 /
  p95 ≈ 0.068. If it does not, the probe is wrong, not the model.

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
model.py:1559).

**Status as of 2026-08-11 19:00: `captain_gru_v3_a4_g3_f1c_r8_finetune` is
TRAINING** — job 44504130, 8 nodes, started 14:49, epoch 10/50, run dir clean
(wiped before relaunch; the earlier attempt 44496324 was cancelled). Its two
source siblings are also running: `f1d` (DINOv2, job 44496323, epoch 21) and
`f1r` (resnet18, job 44496322, epoch 21). Nothing to launch. Actions:

1. **Do not resubmit and do not touch those run dirs.** ~2.5 days remain on
   each 3-day walltime; f1c needs ~40 more epochs.
2. **Arm the evals** so they score the moment they land: add the three labels
   to `$OUT/armed/registry.txt` and keep `armed_eval_watcher.sh` running
   (a standing watcher log already exists from 15:04 today). Verify each
   arm's `.submitted` marker appears; a `FAILED` marker means re-gate, not
   re-train.
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
  consumes — the full-token stash at model.py:1564 and the previous-view
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
view-0 path at model.py:1318) on a THROWAWAY copy of
`(state_feat, mem)`, read the probe's `res["camera_pose"]` — this is
P̂_probe(x), an observation the oracle proved sufficient — then run the real
pass with the ray built from the GRU's refinement of
`[P(x−1), Δ, P̂_probe(x)]`.

- **`model.py` `_forward_decoder_group_step`** (def at line 1443): under flag
  `pose_gru_probe_pass`, before the GRU island: clone `state_feat`/`mem`
  (NOT `.detach().clone()` on the graph path — probe runs under
  `torch.no_grad()` entirely), call the inner decode once with the masked
  token, extract the pose, discard the cloned state. Feed the GRU a widened
  input: `gru_in = cat([est, delta_static, probe_pose])` — input width 21,
  which the sniff already generalizes over via `base_width`
  (line 232–238: extend the accepted set {7, 14} with 21 → mode
  `"pose_delta_probe"`).
- Cost: ~2× decoder FLOPs per view. Mitigation if needed: probe every k-th
  view and hold the last observation (the drift is slow — that's why it's
  drift).
- This is a bigger change than everything above combined; build it ONLY
  after P4.1/P4.2 results are in — if xattn already clears the §0 promotion
  bar, the probe pass may be unnecessary.

---

## §5 — Launch ladder

**Base every new arm on A4×G3×R8, not R1.** `gru_a4g3r8` is the only lever
combination that beats C (54.2% ATE win, 0.079918); building on the R1 arm
would start 2.2% of the gap further back for no reason. Its config is
`captain_gru_v3_a4_g3_r8_finetune.yaml`.

| # | run | training cost | contents | measures | baselines retrained? |
|---|---|---|---|---|---|
| 0 | **oracle-eval** | **none** | score the finished, idle `captain_gru_v3_a4_g3_oracle_finetune_32gpu` on the 4292-scene benchmark | the REAL upper bound for every rung below; currently unmeasured at full scale | no |
| 1 | f1c / f1d / f1r | **already running** | nothing to build (P4.1) | whether any existing image source carries drift evidence | no |
| 2 | **v4-reltarget A/B** | 8 nodes × 2 | P0 (§0b): `pose_gru_loss_target` absolute vs relative, on R8 | whether the module can learn at all once the unlearnable drift term leaves the objective | no |
| 3 | v4-grusync ×2 | 8 nodes × 2 | §1, on top of the P0 winner, at `pose_gru_lr_scale` 100 AND 10 | GRU gradient noise; the lr arm is required because the evidence's gate-saturation fix is lr_scale 100→10 | no |
| 4 | v4a | 8 nodes | §2 (ray gate) + §3 | non-harm floor at A, plus repaired training signal; gate telemetry | no |
| 5 | v4c | 8 nodes | v4a + `img_feat_src=xattn` (P4.2) | **motion-term** observability (still under the trajectory-only ceiling) | no |
| 6 | v4d | 8 nodes | v4c + probe pass (P4.3) | the only rung that changes the ceiling — a real drift OBSERVATION | no (GRU-input change only; trunk weights and forward unchanged) |
| — | *(deferred)* v5-ddp | 8 nodes ×4 | full `ddp_grad_sync` for the trunk | the common-mode effective-batch-4 defect | **YES — A/B/C all re-run. Do not launch under the current constraint.** |

Rung 0 costs one command and no GPU-hours of training, and it is the number
every later decision is measured against — do it first, regardless of where
the code work stands.

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
0b. **Rung 0 before any GPU-hours**: `bash eval_pipeline/arm_eval_for_run.sh
   /gpfs/scratch/etur59/koc821022/checkpoints_projects/captain_cut3r_finetune_aug_full/captain_gru_v3_a4_g3_oracle_finetune_32gpu
   gru_a4g3oracle`. Zero training, and it converts "the oracle recovers
   83–93%" from a 50-scene claim into a full-benchmark bound. If the oracle
   turns out NOT to dominate at 4292 scenes, the §4 observability program
   loses its justification and the whole plan should be re-scoped — so this
   is a genuine go/no-go, not a formality.
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
cd /gpfs/home/koc/koc821022/vggt_features
python verify_plan_anchors.py          # 0 = all anchors exact; 1 = drift/unresolved
python verify_plan_anchors.py --fix-doc   # rewrite stale numbers into this file
```

`verify_plan_anchors.py` resolves all 39 edit sites cited below by regex and
reports where each lives now, plus the presence of every config, launcher and
harness the plan invokes. Verified 39/39 exact on 2026-08-11. An `UNRESOLVED`
line means the code around that anchor genuinely changed — re-read that
region before editing, never edit blind.

### Experiment state as of 2026-08-11 19:00 (verify before acting)

- **Running**: three 8-node finetunes — `f1c` (job 44504130, epoch 10/50),
  `f1d` (44496323, epoch 21/50), `f1r` (44496322, epoch 21/50). Do not
  resubmit or touch their run dirs.
- **Finished and scored** (rows in `averages_table.csv`): `gru_a4g3`,
  `gru_a4g3f1`, `gru_a4g3r8`, `gru_a4g3f1r8`, `gru_a4g3f0np`, `gru_a4g3f1np`,
  plus baselines `augfull_lr1e5` / `gtray_lr1e5` / `prevpred_lr1e5`.
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

1. Default-off config key; absent key → exact previous behavior.
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
