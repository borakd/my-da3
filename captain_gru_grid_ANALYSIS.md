# PoseGRU lever grid — what actually learns something (2026-07-30)

16 completed overfit runs: `captain_gru_v2_a{1..4}_g{0,1,2}` (12) + `captain_gru_v3_a4_{g1,g2}{,_f1}_r8` (4).
All identical recipes — one DROID wrist episode (`RAIL+eh61f232+…/13062452+wrist`), 64-view train
windows, 4-view test windows, 50 epochs, lr 1e-6, `pose_gru_lr_scale` 100, hidden 128 — differing
only in the lever keys. Verified by diffing the `.hydra/config.yaml` of every arm against the
controls: the only deltas are the lever keys and the `PoseGRULoss` term.

## Evidence base

Three regimes, which **disagree**, so all three are used:

1. **In-training test** (4-view windows, `log.txt`) — short-baseline, tiny motion.
2. **Uniform epoch-50 falsifier probe** (32-view windows × 12 windows) — *run for this analysis*,
   `probe_gru_grid_all.py` → `logs/gru_grid_probe/*.json`. Replaces the earlier one-off probes,
   which were at mixed epochs (20/30/40) and gave a materially different A-lever ranking.
3. **672-frame closed-loop rollout** on the episode (`eval_gru_grid_overfit.sbatch`) — *v3 rows
   run for this analysis*; v2 rows already existed.

Controls (same recipe, only conditioning flags differ), from `captain_cut3r_sim3rmse`:
`cut3r_overfit` (no conditioning), `captain_ray_prev_pred` (prev-pred rays, no GRU),
`captain_ray_prev_gt` / `captain_ray_current_gt` (GT-ray oracles).

---

## 1. A lever — the only lever that changes whether the GRU learns

Epoch-50 probe, 32-view windows. "gain" = how much the GRU beats the **lag baseline** (reuse
P(x−1) unchanged), the threshold for "learned anything at all".

| arm | input → head | gain over lag, G0 / G1 / G2 | verdict |
|-----|--------------|---------------------------|---------|
| A1 | pose → direct      | −10.2% / −8.1% / −12.5% | **never learns** — worse than copying the previous pose |
| A2 | pose → residual    | +7.4% / +11.8% / +4.8%  | marginal, and unstable |
| A3 | pose_delta → direct| **+24.2% / +21.5% / +22.1%** | best on the GRU's own metric |
| A4 | pose_delta → residual| +16.2% / +17.9% / +17.2% | second, but by far the most consistent |

**The `pose_delta` input is the information.** A1→A3 is +32pp, A2→A4 is +7pp. Residual-vs-direct is
not an information channel — it is a conditioning/optimisation choice (residual starts at the lag
baseline by construction; direct must learn identity from scratch).

**This ranking changed with training length.** At epoch 30 the earlier probes had A3 at −4.8% /
−10.4% ("not learned"); by epoch 50 it leads. `test_gru` minima land at epoch 46–49 for nearly
every arm → **50 epochs is not converged**, and the direct-mode arms are the ones still moving.

**Apparent instability on the 672-frame rollout** — ATE across an arm's own three G variants:

- A3: 0.0486 / 0.0710 / 0.0738 — spread 0.025
- A4: 0.0670 / 0.0642 / 0.0634 — spread 0.004
- A2: 0.0718 / **0.1020** / 0.0735 — G1 diverged (rpe_rot 3.91°)
- A1: 0.0704 / 0.0694 / 0.0691 — tight but the GRU is inert

### 1a. CORRECTION — the A3 "instability" does not reproduce (77-window test)

The spread above is **n=1** (one 672-frame rollout per arm). Re-run properly over **77 sliding
64-frame windows** (stride 8, `run_gru_window64_eval.sbatch`, job 1421570):

| arm | ATE mean | sd | p90 | max/median | rpe_rot mean | absrel mean |
|-----|---------|-----|-----|-----------|--------------|-------------|
| a3_g0 / g1 / g2 | 0.0023 / 0.0023 / 0.0024 | 0.0010–0.0011 | 0.0033–0.0038 | 1.73–2.38 | 0.1990 / 0.1997 / 0.1943 | 0.1115–0.1121 |
| a4_g0 / g1 / g2 | 0.0024 / 0.0024 / 0.0023 | 0.0011 | 0.0038–0.0039 | 2.07–2.30 | **0.1784 / 0.1784 / 0.1783** | 0.1116–0.1119 |
| a4_g1_r8 | 0.0024 | 0.0011 | 0.0037 | 2.14 | 0.2065 | 0.1130 |

**ATE and absrel are a statistical tie**, and A3 shows **no fat tail** (max/median 1.7–2.4, same as
A4). Paired per-window: A3 wins 42–53% of windows, mean Δ ≈ ±0.00005 ≈ 2% of the sd.

**The real, reproducible A4 advantage is rotation**: rpe_rot 0.178 vs 0.194–0.200, ≈11%, 3-vs-3
with no overlap.

### 1b. Why — the direct head fails specifically on rotation

Probe error split by component, each as a multiple of that arm's own lag baseline component:

| arm | head | translation | rotation |
|-----|------|-------------|----------|
| A3 g0/g1/g2 | direct | **0.80 / 0.81 / 0.81** | 1.03 / 1.08 / 0.99 |
| A4 g0/g1/g2 | residual | 0.86 / 0.85 / 0.85 | **0.90 / 0.89 / 0.89** |
| A1 g1 | direct (pose-only) | 1.08 | 1.40 |
| A2 g1 | residual (pose-only) | 0.89 | 0.93 |

The direct head **beats the anchor on translation** and is **worthless on rotation** (≈1.0× lag =
learned nothing). Frame-to-frame rotation is near-identity, so q(x−1) is an excellent prior that a
zero-init correction exploits and a from-scratch regression cannot match. This motivates the **A5**
arm (below).

---

## 2. G lever — noise, with exactly one real (non-accuracy) effect

Every aggregate difference between G0/G1/G2 at fixed A sits inside the epoch-to-epoch sd of the
last 10 epochs (e.g. A4 `test_gru` 0.7780 / 0.7744 / 0.7853, sd ≈ 0.010–0.014).

The one thing G1 measurably does — exactly what it was designed to do — is make the GRU **use its
memory more**. `POSE_GRU_HIDDEN_ZERO` degradation for A4: **13.5% (G0) → 25.9% (G1) → 11.9% (G2)**.
It buys reliance on the recurrent state, not accuracy.

G2 (main-loss gradient into the GRU) never helps on any metric and owns the worst v3 rollout
(`a4_g2_r8`: absrel 0.1357, rpe_rot 0.532°).

**Verdict: keep G1 (free), drop G2.** G is not where the information is.

---

## 3. R lever (8 iterations) — real but shallow refinement, mispriced at N=8

**Anytime-N probe** (`POSE_GRU_FORCE_ITERS`, the designed go/no-go), gru pose error on a trained
R8 checkpoint:

| eval N | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| a4_g1_r8    | 0.15644 | 0.15180 | **0.14567** | 0.15316 |
| a4_g1_f1_r8 | 0.16301 | 0.15573 | **0.14526** | 0.15008 |
| a4_g2_r8    | 0.15836 | 0.15231 | **0.14564** | 0.15156 |
| a4_g2_f1_r8 | 0.16209 | 0.15698 | **0.15043** | 0.16200 |

Refinement is **real** (−7% from N=1 to N=4, consistently across all four arms) but **non-monotone**
— it over-corrects past 4. `iters=8` is the wrong count; **4** is the sweet spot. (Forcing N>1 on
the R1 v2 checkpoints blows up as expected, which validates the probe.)

**R8 destroyed the recurrence.** `HIDDEN_ZERO` degradation collapses from 25.9% (v2 a4_g1) to
**−0.4% / +2.8%** for `a4_g1_r8` / `a4_g2_r8` — the within-view iteration replaced the cross-view
memory and the GRU went effectively stateless.

**Net accuracy vs the R1 controls:** worse on the 4-view test (`test_gru` 0.934 vs 0.774), worse
depth (absrel 0.0502 vs 0.0422, ≈27× the 0.0003 sd), worse ate/rpe. *Better* 672-frame rollout ATE
(0.0547–0.0575 vs 0.0634–0.0670) but worse rpe_rot.

**Known confound:** `PoseGRULoss(iter_gamma=0.8)` at N=8 adds `1 + Σ_{k=1..7} 0.8^k` = **4.16×** the
aux gradient mass of an R1 run. Part of the R8-vs-R1 delta is a 4× aux-loss reweighting, not the
iteration. Renormalise the γ sum (or set `pose_gru_loss_weight≈0.25`) before drawing a final verdict.

---

## 4. F lever (2-frame image features) — the strongest *new* information channel

**The falsifiers are unambiguous.** Degradation of gru pose error when the feature channel is
corrupted:

| arm | `IMG_FEAT_ZERO` | `IMG_FEAT_SHUFFLE` |
|-----|-----------------|--------------------|
| a4_g1_f1_r8 | **+93.0%** | +59.0% |
| a4_g2_f1_r8 | **+64.9%** | +53.4% |

The GRU genuinely reads the images — this is not a dead zero-init input.

- Under G1 it **recovers most of R8's damage**: gain over lag 10.5% → **16.7%** (back to A4/R1 level).
- It is the **only lever with a nonzero downstream effect**: stripping the GRU costs head-pose
  **+2.0%** (g1_f1) / +1.5% (g2_f1), versus ≈0.0–0.6% for every v2 arm.
- It **stabilises**: it rescued the G2 blowup (absrel 0.1357 → 0.1187, rpe_rot 0.532° → 0.265°).

**Caveat:** there is no trained F1×R1 arm, so F is only measured on top of R8. Its isolated effect
is inferred from the falsifiers plus the N=1 column above (F1 is *worse* than F0 at forced N=1:
0.16301 vs 0.15644 — F needs the iterations to pay off).

---

## 5. The uncomfortable result: almost none of this reaches the reconstruction

**Probe, downstream columns.** Stripping the GRU entirely at eval changes head-pose by ≤3% and
scale-aligned pts3d by **≤0.1%** for every v2 arm. The GRU improves *its own* pose estimate and
almost nothing else.

**Against the true controls** (4-view in-training test, last-10-epoch mean):

| run | absrel | ate | rpe_rot | pose_loss |
|-----|--------|-----|---------|-----------|
| `cut3r_overfit` (no conditioning) | **0.0376** | **0.00054** | **0.198** | **0.00457** |
| `captain_ray_prev_pred` (rays, no GRU) | 0.0395 | 0.00058 | 0.202 | 0.00502 |
| best GRU arm (`a4_g1`) | 0.0422 | 0.00061 | 0.226 | 0.00535 |
| `a4_g1_r8` | 0.0502 | 0.00082 | 0.277 | 0.00649 |

On short windows the whole prev-pred + GRU line is a **net negative**.

**On the 672-frame rollout the ATE sign flips** — which is the regime the filter exists for:

| method | AbsRel | ATE | RPE_rot |
|--------|--------|-----|---------|
| Regular CUT3R (overfit, no conditioning) | **0.0981** | 0.0780 | 0.398 |
| Prev-frame **GT** rays (oracle) | 0.1314 | **0.0183** | **0.195** |
| Prev-frame predicted rays, no GRU | 0.1244 | 0.0765 | 0.212 |
| best GRU (a3_g0) | 0.1213 | **0.0486** | 0.282 |
| a4_g1 / a4_g2 | 0.1215 / 0.1218 | 0.0642 / 0.0634 | 0.290 / 0.285 |
| a4_g1_r8 / a4_g1_f1_r8 | 0.1233 / 0.1241 | 0.0575 / 0.0574 | 0.311 / 0.274 |

The GRU closes up to **~48%** of the predicted-ray → GT-ray ATE gap. Depth never recovers to the
no-conditioning baseline, and rpe_rot is worse than the no-GRU control for every arm.

**One thing that does move consistently: metric scale.** `orig/Regr3DPose_pts3d/4` = 1.06–1.10
(direct arms) vs 1.18–1.21 (residual arms) vs 0.85 (GT-ray oracles) vs 1.00 (no conditioning),
while the *scale-invariant* version is flat at 0.0222–0.0239 and absrel is flat at 0.042. So the A
mode shifts **global scale calibration**, not shape — do not read the metric-scale pts3d gap as a
reconstruction-quality gap.

---

## Ranking — what contributes meaningful information

1. **`pose_delta` input (A3/A4)** — the single biggest contributor; +32pp / +7pp over the
   pose-only input. Everything else is built on it.
2. **F, 2-frame image features** — strongest *new* channel; falsifier-verified (+93% when zeroed);
   the only lever with a measurable downstream effect.
3. **R, iterative refinement** — real but shallow (−7% at N=4); currently mispriced at N=8 and
   confounded by the 4.16× γ reweighting; it also kills the recurrence.
4. **residual vs direct (A4 vs A3)** — a stability/convergence knob, not information. A3 wins the
   metric at ep50 and is still improving; A4 is reproducible across G.
5. **G** — noise. G1 doubles memory reliance for free; G2 adds risk with no gain.
6. **A1 (pose → direct)** — inert. Never beats the lag baseline. Drop.

## Recommended next arm

`A4` (and an `A3` twin) **+ F1 + R4**, γ-sum renormalised, **G1**, trained well past 50 epochs.
Plus the one missing control: **F1 × R1**, to isolate F from R.

## Artifacts produced by this analysis

- `probe_gru_grid_all.py` / `.sbatch` — uniform epoch-50 falsifier sweep, 4-way GPU shard
  (job 1421200). Results: `logs/gru_grid_probe/<label>.json`.
- 672-frame rollout eval for the four v3 arms (job 1421204) →
  `outputs/cut3r_eval/overfit_test_scene/gru_grid/gru_v3_*`.

---

## 6. A5 — "anchored A3" (`pose_gru_mode: split_anchor`), launched 2026-07-30

Section 1b showed the A3/A4 difference decomposes cleanly by component. A5 splices the two
winning halves, in one shared head, with **zero new parameters and zero new config keys**:

```python
raw = self.head(hidden)
t = raw[:, :3]                                          # direct   — A3's 0.80x-lag translation
q = normalize(gru_in[:, 3:7] + raw[:, 3:7])             # anchored — A4's 0.89x-lag rotation
```

Only the rotation rows `head.weight[3:7]` / `bias[3:7]` are zero-init; the translation rows keep
the default init, bitwise identical to A3's.

**Why the anchor must be structural rather than learned.** Rotation is only **3–4%** of the
unweighted `PoseGRULoss` (`t_loss + q_loss`: trans 0.141 vs quat 0.0047), so the aux gradient
cannot teach rotation regardless of architecture. A zero-init correction on top of q(x−1) gets it
for free; a from-scratch regression never does.

**Two designs rejected.** (a) A learned convex gate `lerp(anchor, direct, sigmoid(gate(hidden)))`:
the gate reads the same corrupted hidden the head reads, so its "fallback" is illusory; and with
~1800 optimizer steps at lr 1e-6 × lr_scale 100, AdamW per-element travel is ~0.10 total, so a gate
bias (or an `anchor_gain_t` initialised at 1.0) cannot leave its init and the arm collapses to A4.
(b) Separate `head_t`/`head_q` tensors: renaming pose_gru tensors trips the silent-drop hazard —
every loader uses `strict=False` and the existing pose_gru hard-fail only catches *size* mismatches,
so a new or renamed tensor is silently dropped or silently fresh-init.

**HONEST LIMIT.** A5 anchors rotation only. Translation remains a free regression off the hidden,
and **98.3% of A3's hidden-zero degradation is translation** — so A5 buys essentially **no** tail
protection. It targets the measured rotation deficit, not the fallback property.

| arm | head | trans (× lag) | rot (× lag) | hidden-zero trans | hidden-zero rot |
|-----|------|---------------|-------------|-------------------|-----------------|
| A3 | direct | **0.80–0.81** | 0.99–1.08 | 3.43–3.58 | 2.01–2.45 |
| A4 | residual | 0.85–0.86 | **0.89–0.90** | 0.96–1.07 | 0.90–0.94 |
| A5 | split_anchor | *target ≤0.81* | *target ≤0.90* | *expected ≈ A3* | *expected ≈ A4* |

**Success criterion:** translation ≤0.81× **and** rotation ≤0.90× of lag simultaneously, plus
rpe_rot matching A4's 0.178 on the 77-window eval. Failure to hit both means the two halves
interact through the shared cell/hidden rather than being independent.

**Hardenings landed alongside** (both are pre-existing hazards, not A5-specific): `forward`'s
catch-all `else` — which silently executed the DIRECT algebra for any unrecognised mode string —
is now explicit `elif` + terminal `raise`; and `load_model` hard-fails when a checkpoint carries
pose_gru weights but its args lack `pose_gru_mode` (the mode is unrecoverable from weight shapes,
since all three modes share one head shape, and the old code silently defaulted to `residual`).

**Verification:** `verify_gru_a5.py` / `.sbatch` — **31/31 PASS** (job 1421826), including
cross-worktree BITWISE parity of `residual` and `direct` against the v2 worktree, per-half bitwise
equality with A3/A4 under identical weights, and the checkpoint round-trip + args hard-fail.

**Runs:** `captain_gru_v3_a5_g1` (job 1422852) and `_a5_g2` (job 1422853), 2× L40S each,
R1/F0/50 epochs, configs differing from `captain_gru_v2_a4_g{1,2}.yaml` in only `pose_gru_mode`,
`exp_name` and `save_dir`.
