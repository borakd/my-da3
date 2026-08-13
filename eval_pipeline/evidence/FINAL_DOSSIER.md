# FINAL EVIDENCE DOSSIER — GRU v4 campaign, deliverable arm `captain_gru_v3_a4_g3_r8_refine_finetune`

Audit date: 2026-08-13 (T-6h before job 44548347 completes). Branch `captain_gru_v3`, HEAD `fcc988e`.
Adversarial pre-score audit run across three lenses (legality, compliance, statistics). Status of every
claim below is one of: **VERIFIED** (independently reproduced from files/commands), **DEFECT** (substantiated,
see register, section 6), or **TODO** (arrives with the score).

Rule of this document: nothing in a TODO slot may be filled by the same person/session that fixes a DEFECT
touching that slot without noting the fix's commit hash next to the value.

---

## 1. Constraint proofs (what the deliverable is allowed to be) — all VERIFIED

- **Run config legality** — VERIFIED. `.hydra/config.yaml` of the deliverable run
  (`/gpfs/projects/etur59/koc821022/checkpoints/captain_cut3r_finetune_aug_full/captain_gru_v3_a4_g3_r8_refine_finetune/`)
  has `feed_gt_ray_map=false`, `feed_prev_gt_ray_map=false`, `feed_prev_pred=true`, `pose_gru=true`, no
  `pose_gru_oracle` key (trainer default `off`, asserted `off|gt` at `train_cut3r_baseline.py:469-471`).
  `resume: null`, zero 'Resume checkpoint' lines in the train log — not a resume rewrite. Train-log banner
  (iters=8, refine_passes=2, GRU input dim 21, hidden 128) matches the config exactly.
- **Single-variable A/B vs scored r8 arm** — VERIFIED. Config diff (comments stripped) is exactly
  `pose_gru_refine_passes: 2`, `pose_gru_probe_every: 1`, `exp_name`. Nothing else.
- **Recipe match vs baselines** — VERIFIED. lr 1.0e-05, min_lr 1.0e-06, epochs 50, batch 4, num_views 64,
  warmup 0.5, seed 0, same DL3DV_Multi roots (38633 train / 4292 test), same pretrained
  `cut3r_512_dpt_4_64.pth` — identical to `cut3r_pointworld_droid_aug_mn5_prevpred_32gpu_lr1e5.yaml`.
  Omitted dataset kwargs default to the baseline's explicit values (`dl3dv.py`).
- **Runtime truth** — VERIFIED. `sacct -j 44548347 -X`: 8 nodes x 4 GPU = 32; job stdout confirms global
  batch 128; `.hydra/overrides.yaml` is `[]` (no CLI overrides).
- **Baselines untouched** — VERIFIED. All three dirs under `cut3r_finetune_baselines/` mtime 2026-08-07;
  `find -newermt 2026-08-12` returns zero files in each.
- **Branch purity (commits)** — VERIFIED for committed history (full campaign chain on `captain_gru_v3`:
  6dbfa12 launch → 0cd7e83 eval-path registration → d1b9828 P4.3 mechanism → 191013b / fcc988e gates).
  Working tree is DIRTY with score-path scripts — **DEFECT MAJOR-4**, see section 6.
- **Preflight lever whitelist** — VERIFIED. `preflight_ckpt.py` LEVERS_RESTORED includes
  `pose_gru_refine_passes` + `pose_gru_probe_every` (committed 0cd7e83, file clean); the config header's
  'will fail preflight until this edit lands' warning is discharged.

## 2. Instrument audit (whether the eval can lie) 

- **Probe purity** — VERIFIED. `verify_gru_probe_pass.py` re-run fresh: exit 0, all 10 sections pass.
  Pass-1 probe is bit-for-bit the pose-free `masked_ray_map_token` construction; probe never calls
  `update_mem`, leaves `state_feat`/mem bit-identical; zero-init probe columns reproduce the 14-wide arm's
  hidden trajectory bit-identically; sniff round-trips and mis-sniffs hard-fail both directions.
- **Model honest path cannot see current-view GT** — VERIFIED. `gt_pose_encoding` read only inside the
  `oracle=='gt'` gate (`model.py:2171`); dataset-side GT ray feeding impossible under `feed_prev_pred`
  (hard assert `model.py:1690`; `ray_mask=False` on every view for prev_pred arms); pass>0 re-probe rays
  built from the RUNNING PREDICTED estimate + prev-view GT intrinsics only (`model.py:1823-1843`), the
  convention every legal arm uses.
- **GRU centrality** — VERIFIED, code + empirical. Probe observation consumed at exactly one point
  (cat into cell input cols 14:21, under `no_grad`); committed ray map comes from GRU output; no bypass.
  Empirically: `weight_ih[:,14:21]` norm 9.50 at ck10; PROBE_SHUFFLE degrades committed e_head_t
  +7.9/+26.4/+21.7% by band; regen causal gate PASS; paired trained-minus-placebo ddR2 e5 mid_t
  +0.157 CI [+0.090, +0.223] excluding zero.
- **Arm derivation** — VERIFIED. `arm_eval_for_run.sh` takes no conditioning argument; arm comes only from
  `preflight_ckpt.py`'s `PREFLIGHT_OK ARM=` token; preflight cross-checks ckpt args vs weights vs config,
  gates `pose_gru_oracle==off`, requires final epoch, refuses un-whitelisted levers; `--prevalidate` emits a
  token the arm script's grep cannot match. Watcher path inherits all of it (`armed_eval_watcher.sh:256`).
- **Env-hook coverage** — DEFECT. 12 of 13 falsifier env hooks on the eval model/data path are in the
  unset list; **ORACLE is not, and is not pinned in `--export`** → silent GT-injection contamination path
  for exactly the deliverable's conditioning mode. **MAJOR-1**, must be fixed before the watcher arms.
  Secondary: node script defaults `CONDITIONING=gt` on hand submission (**MINOR-2**).
- **GT_RAY_NOISE hook containment** — VERIFIED. Commit 49f5819 touches only `demo_ray.py`; all four env
  vars in the unset list; absent env ⇒ byte-identical loader; data-side ray maps inert under prev_pred_gru
  anyway.
- **Scoring isolation** — VERIFIED. Metrics computed by a separate subprocess comparing saved predictions
  against GT read from disk; out of reach of conditioning-side perturbation.
- **Win% instrument** — DEFECT (**MINOR-1**): no committed script; historical convention re-derived as
  strict `<` over the full 4292-scene intersection (reproduces every recorded tier exactly). Must be pinned
  in git before the deliverable CSV exists.

## 3. Gate results to date

| Gate | Registered rule | Result | Status |
|---|---|---|---|
| ck10 (i) weight norm | probe-column channel not dead | 9.50 | PASS — VERIFIED |
| ck10 (ii) shuffle degradation | e_head_t >10% (no band registered) | early +7.9%, mid +26.4%, late +21.7% | PASS as recorded, but band chosen at read time — **MINOR-3**; raw JSONs committed, numbers reproduce |
| ck10 (iii) arm-A head ratio | inconclusive-by-construction (mid-anneal) | recorded as such | VERIFIED as registered |
| ck30 trend | e_head declining vs ck10 | -27..-42%, 1.23-1.35x arm A | VERIFIED (fcc988e); reduction ad-hoc — **MINOR-4** |
| P4.3 causal (step-2c) | PASS >10% / KILL <5% | +49.3% / +42.4% mid/late | PASS — VERIFIED; code collapses 3-zone to 2-zone (**MINOR-7**), no effect at this margin |
| P4.3 paired ddR2 (step-2d) | CI excludes zero | e5 mid_t +0.1568, CI [0.0898, 0.2229] | PASS — independently reproduced (point exact, CI to bootstrap noise) |
| Noise oracle B1 | SHALLOW/CLIFF/INTERMEDIATE | INTERMEDIATE (0.181 / 0.463) | VERIFIED, recomputed |
| Noise oracle B2 poison | noisy-B ≥ arm-A subset ATE | none (white 1.0: 0.07554 < 0.07714) | VERIFIED, incl. exact arm-A subset re-derivation; partial-coverage latent bias (**MINOR-5**) confirmed not fired |
| Noise oracle B3 | D_ATE≥2 and D_RPEt<1.5 ⇒ drift-dominant | D_ATE 0.391, D_RPEt 0.049 | VERIFIED; walk ~0.8% under-scaled vs white (**MINOR-8**), margin-irrelevant |
| sigma=0 byte-identity | byte-identical outputs | 134,448 + 7,626 files, 0 mismatches | VERIFIED (evidence/bytecheck_sigma0_20260813_0443.log) |

Supporting: 430-scene subset is deterministically `full[::10]` (no selection freedom); noise injection is a
pure crc32-keyed function of (scene, view, seed) — shard/order independent; sigma calibration (Umeyama sim3,
chi-3 median 1.5382) verified correct and consistently applied.

## 4. Statistical foundations for the verdict

- **Noise floor** — DEFECT (**MAJOR-3**). Directive's 0.00281 is `median_high` of the 28 pairwise
  |ATE diffs|; standard median is **0.00264** (same data confirmed: max 0.00814, R8−R1 0.00378 rank 10/28,
  25/28 |t|>3 all reproduce; quoted SE 0.000346 does not). Decision-flip window [0.00264, 0.00281].
  Existing calls survive under either convention (closest margin: 0.00006). Convention must be pinned
  before the refine-vs-r8 delta is read.
- **Registered promotion tiers** — VERIFIED verbatim between directive and campaign state:
  >54.2% progress / >63.0% beats-A, on ATE AND rpe_rot, vs baseline C. Re-derived from raw CSVs:
  A 63.02% ATE; r8 54.22% ATE / 55.45% rot; f1rr8np 55.57% / 54.38%; f1dr8np 49.39% / 53.82%.
  n=4292 common scenes, zero ties, zero NaNs.
- **Replicate band** — DEFECT (**MAJOR-2**). The +/-2-win-point band is unregistered, uncalibrated (no seed
  replicate ever run), and its one historical application overrode the registered rule (f1rr8np cleared
  54.2 on both metrics, recorded NOT progress). Must be registered-or-struck before the score exists.

## 5. Pre-registration addendum required BEFORE score time (fixes MAJOR-2/-3, MINOR-1/-3/-7)

To be committed within the 6h window, then referenced by hash in section 7:

1. ORACLE added to unset list / pinned `ORACLE=off` in `arm_eval_for_run.sh` (MAJOR-1). Commit: **TODO**
2. Replicate-band rule: registered with derivation + symmetric application, or struck (MAJOR-2). Commit: **TODO**
3. Floor convention pinned: 0.00264 standard median (or explicit upper-median 0.00281) (MAJOR-3). Commit: **TODO**
4. Score-path scripts committed: `build_lr1e5_table.py` edit, `aggregate_and_merge.sh` (after dry-run of the
   averages_table.csv merge against a copy), `build_gru_v3_table.sh`; p43_*.json moved under
   `eval_pipeline/evidence/` (MAJOR-4). Commit: **TODO**
5. Win% script committed: strict `<`, assert n==4292, zero NaNs, fail loudly (MINOR-1). Commit: **TODO**
6. Falsifier-battery band convention for checkpoint-final: mid AND late must clear >10%, early
   informational (MINOR-3); three-zone causal verdict encoded (MINOR-7). Commit: **TODO**

## 6. Defect register (full)

| ID | Sev | One-line | Fix-by |
|---|---|---|---|
| MAJOR-1 | major | ORACLE env not unset/pinned in arm_eval_for_run.sh — silent GT-injection path for the deliverable arm | before watcher arms eval |
| MAJOR-2 | major | +/-2-win-point band unregistered, uncalibrated, historically overrode the registered rule | before score exists |
| MAJOR-3 | major | Floor 0.00281 is median_high; standard median 0.00264; flip window [0.00264, 0.00281] | before delta read |
| MAJOR-4 | major | Score-path scripts uncommitted at T-6h, incl. shared-CSV-rewriting merge heredoc | before eval completes |
| MINOR-1 | minor | No committed win% instrument (convention: strict `<`, full 4292 intersection) | before deliverable CSV |
| MINOR-2 | minor | Node script defaults CONDITIONING=gt; mismatch is a warning, not a refusal | opportunistic |
| MINOR-3 | minor | Gate (ii) band qualifier chosen at read time (early +7.9% < 10%) | pre-register for final battery |
| MINOR-4 | minor | Gate-log reductions ad-hoc / unscripted (raw JSONs committed, numbers reproduce) | fold into ckpt10_kill_gate.sh |
| MINOR-5 | minor | noise_oracle_analyze partial-coverage bias (latent; verified not fired, 430/430 everywhere) | before any noise rerun |
| MINOR-6 | minor | p43_regen_compare positional pairing, no key-identity assert (latent; artifacts verified aligned) | one-line assert |
| MINOR-7 | minor | Causal gate 3-zone rule collapsed to 2 zones in code (no effect at +49/+42%) | with item 6 of §5 |
| MINOR-8 | minor | Walk noise ~0.8% weaker than white at same multiplier (margin-irrelevant at D_ATE 0.39 vs 2.0) | document only |

## 7. Score-time slots — ALL TODO (job 44548347, ~6h)

- [ ] **Final table row** for `captain_gru_v3_a4_g3_r8_refine_finetune` (via committed
      `aggregate_and_merge.sh` + `build_gru_v3_table.sh` — MAJOR-4 fix hash: ____): ATE ____, RPE-t ____,
      RPE-rot ____, n=4292 asserted.
- [ ] **Win% vs C** (committed instrument, MINOR-1 fix hash ____): ATE ____%, rpe_rot ____%.
      Verdict against REGISTERED tiers (>54.2 both = progress; >63.0 both = beats A): ____.
      Band rule applied per §5.2 registration (hash ____): ____.
- [ ] **Refine-vs-r8 delta read**: ATE delta ____ vs pinned floor convention (§5.3, hash ____).
      If |delta| lands in [0.00264, 0.00281]: verdict is governed by the pinned convention, recorded here
      with the pin's commit hash, not chosen at read time.
- [ ] **Falsifier battery on checkpoint-final** (band convention per §5.6, hash ____):
      (i) probe-column weight norm ____ ; (ii) PROBE_SHUFFLE degradation early ____ / mid ____ / late ____
      (mid AND late must clear >10%); (iii) arm-A head ratio ____ (now conclusive, post-anneal);
      PROBE_LAG ____ ; POSE_GRU_HIDDEN_ZERO ____ .
- [ ] **Curve-consistency delta**: ck10 → ck30 → final e_head trend point ____ vs recorded
      -27..-42% trajectory; consistent decline YES/NO ____.
- [ ] **Contamination check on the scoring run itself**: grep all worker logs under the deliverable label
      for 'pose_gru_oracle' / 'DIAGNOSTIC RUN' banners → must be ZERO hits ____ (MAJOR-1 fix hash ____);
      confirm eval-side conditioning printed as `prev_pred_gru` in every shard log ____.
- [ ] **Replicate decision**: seed replicate launched YES/NO ____; if the win% lands within the (now
      registered-or-struck) band of a tier boundary, the replicate is the pre-registered tiebreaker.
- [ ] **averages_table.csv integrity**: post-merge diff vs backup shows baseline A/B/C rows byte-identical ____.

*Verdict may not be declared until every checkbox above is filled and MAJOR-1..4 fix hashes are recorded.*
