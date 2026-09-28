# Write-gate variants: confidence-weighted memory updates (design + implementation contract)

Date opened: 2026-09-21. Branch `maks_idea`. Status: IMPLEMENTING.

Two proof-of-concept variants of one idea: CUT3R commits each frame into memory in proportion to a
Kendall-Gal confidence about the thing being written. Frozen backbone (`augfull_lr1e5` finetuned checkpoint,
`$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth`), fully causal,
no oracle information at inference. Constants and oracles appear only as control rows.

The concept is the deliverable, not the number. Everything that departs from the paper or from the DUSt3R
design is listed under "Departures" and must stay listed.

## Background facts (from the code, verified 2026-09-21)

* Two memories. (1) The state: 768 tokens x 1024 (`register_tokens`, `state_size=768` in the 512 config), embedded to
  the 768-d decoder width by `decoder_embed_state`; committed per frame in `_forward_decoder_group_step` as
  `state_feat = new_state_feat * state_mask + state_feat * (1 - state_mask)` (model.py ~L2066), where `state_mask`
  is `update_mask` (1 for image frames with `update=True`) times an optional per-view scalar `update_alpha`
  (~L1941, applied to ALL tokens) times an optional per-token vector (the token gate). (2) The pose retriever
  memory: `LocalMemory`, 256 slots; written once per frame at ~L1809,
  `new_mem = self.pose_retriever.update_mem(mem, global_img_feat_group, pooled_pose_feat)`, read at ~L1781 to seed
  the pose token on frames > 0 (`pose_seed = self.pose_retriever.inquire(global_img_feat_group, mem)`).
* `global_img_feat_group` = mean over tokens of the ENCODER features of the frame, (B,1,1024).
* `pooled_pose_feat` = the pose token's output after the last decoder block, (B,1,768) = `dec[-1][:, :1]` at
  group size 1. The pose head (`PoseDecoder`, dust3r/utils/camera.py) is `Mlp(768 -> 3072 -> 7)` = fc1, GELU, fc2;
  it reads exactly this feature (dpt_head.py ~L296-336, `pose = self.pose_head(pose_token)`, then
  `postprocess_pose`). Output `camera_pose` is a 7-d absT_quaR encoding relative to frame 0.
* Pose residual as the finetune loss defines it (losses.py `compute_pose_loss`, `get_all_pts3d` ~L494-560):
  per frame `||t_pred - t_gt|| + ||q_pred - q_gt||` on absT_quaR encodings, GT expressed in camera 1
  (`camera_to_pose_encoding(in_camera1 @ gt["camera_pose"])`), translations divided by the per-sequence point
  norm factor (`norm_mode='?avg_dis'`). `criterion.get_all_pts3d(views, preds)` returns
  `(gt_pts_self, gt_pts_cross, pr_pts_self, pr_pts_cross, gt_poses, pr_poses, valids, skys, pose_masks, {})`
  with `gt_poses`/`pr_poses` lists of (B,7) tensors already normalised.
* Frame 0: `init_state_feat` is the learned prior; the first write turns it into scene memory. Both variants
  write frame 0 in full.
* Training data: `train_unc_gate.build_loader(TRAIN_ROOT, num_views=64, batch_size, num_workers, accelerator)`,
  consecutive-frame episodes, `set_epoch`, `to_device`; rollout pattern in `train_conf_gate.run_batch`
  (`_forward_encoder` once, then `_forward_decoder_group_step` per frame with `view_indices=[t]`).

## Variant 1: confidence-weighted state update (frame gate)

* Write: per frame t>0 one scalar `a_t in [a_min, 1]`; `state = a_t * proposal + (1 - a_t) * old` for all 768
  tokens. Implemented by multiplying `update_mask` (the existing `update_alpha` path) so the token gate, resets
  and `update=False` all compose unchanged. Frame 0: `a_0 = 1`.
* Confidence: `FrameConfHead`, a two-layer MLP (in -> 256 -> GELU -> 1) outputting `s = log sigma^2` for the frame.
  Input vector (all detached, float32), computed inside `_forward_decoder_group_step` right after
  `new_state_feat` is available and BEFORE the commit:
  `x = concat( mean_tokens(delta) [768], ||delta||_F / sqrt(768*768) [1], mean_tokens(|delta|) over channels... )`
  — precisely: `delta = new_state_feat - state_feat` (B,768,768);
  `f_delta_mean = delta.mean(1)` (768); `f_delta_norm = delta.norm(dim=-1).mean(1, keepdim=True)` (1);
  `f_delta_tok = delta.norm(dim=-1)` summarised by its mean, std, max over tokens (3);
  `f_img = global_img_feat_group[:, 0]` (1024). Total in = 768 + 1 + 3 + 1024 = 1796. Layer-norm the input
  inside the head (LayerNorm(1796)).
* Target: the frame's own error, `r_t = ||t_pred - t_gt|| + ||q_pred - q_gt|| + mean self-view point L21`
  (normalised as the finetune loss does; the point term via `get_all_pts3d` outputs, masked by valids).
  Loss: Gaussian `exp(-s) * r^2 + s` (paper eq. 8 form, on the scalar residual).
* sigma -> weight: `a_t = clip(sigma_ref / sigma_t, a_min, 1)`, `sigma_ref` = median sigma over the training
  table (stored in the head checkpoint), `a_min = 0.5` fixed.
* Controls: (i) dose-matched constant `alpha_all = mean a_t over the 430 scenes` (read from the trace dump);
  (ii) oracle: `a_t` from the same map with sigma replaced by the GT residual (normalised by its median), via the
  existing per-frame `alpha` control JSON; (iii) frame-0-gated control.

## Variant 2: confidence-weighted pose-memory update (mem gate)

* Write: per frame t>0 one scalar `b_t in [b_min, 1]`;
  `mem = b_t * update_mem(mem, key, value) + (1 - b_t) * mem`. Frame 0: `b_0 = 1`.
* Confidence: `PoseSigmaHead` = the pose head's own penultimate activation `h = GELU(fc1(pose_feat))` (3072-d,
  frozen, shared with the pose decoder) followed by a NEW linear layer `3072 -> 2` giving `(s_t, s_R)` =
  log sigma^2 for translation and rotation. This is "extra output rows on the same head" (paper) / "same head,
  extra channel" (DUSt3R). Only the new linear layer trains.
* Target: `r_t = ||t_pred - t_gt||`, `r_R = ||q_pred - q_gt||`, same normalisation. Loss: sum of the two
  Gaussian NLLs.
* sigma -> weight: `sigma_comb = sqrt( (sigma_t / ref_t) * (sigma_R / ref_R) )`, `b_t = clip(1 / sigma_comb,
  b_min, 1)`, `b_min = 0.5` fixed. Refs = training medians, stored in the checkpoint.
* Leverage probe (before training): control keys `mem_gate.const` (b constant for all t>0; 0 = memory frozen at
  its initial state) and `mem_gate.freeze_after=k` (b=0 for t>k). Arms on 430: const 0, freeze_after 8, const .5.
* Controls: dose-matched constant `b`; oracle `b_t` from GT pose residual.

## Implementation contract (names are fixed; all three implementers code against these)

Model (`src/CUT3R/src/dust3r/model.py`) + new package `src/CUT3R/src/dust3r/wgate/`:
* `dust3r/wgate/heads.py`: `FrameConfHead(in_dim=1796, hidden=256)`, `PoseSigmaHead(pose_decoder)` (holds a
  reference to the frozen `PoseDecoder`'s `mlp.fc1`/`mlp.act`, owns `self.out = nn.Linear(3072, 2)`),
  `frame_features(new_state_feat, state_feat, global_img_feat) -> (B,1796) float32 detached`,
  `sigma_to_weight(log_var, log_ref, wmin) -> (B,)`, `combine_pose_sigma(log_var2, log_ref2) -> (B,)`.
* Model view keys (per-view tensors, shape (1,), like the existing keys): `frame_gate_on` (1.0 = apply the
  attached FrameConfHead), `frame_gate_wmin`; `mem_gate_on`, `mem_gate_wmin`, `mem_gate_const` (if present:
  b_t = const for t>0, overrides the head), `mem_gate_freeze_after` (b_t = 0 for t > k).
* Model methods: `attach_frame_gate(head_or_state_dict, log_ref)`, `attach_mem_gate(head_or_state_dict,
  log_ref_t, log_ref_R)`; both set `.eval()`, no grad. Recording hook: `model.wgate_record = []` (a list) ->
  when it is a list, every decoder step appends a dict `{t, frame_feat (1796,), pose_feat (768,), pose_h (3072,)
  is OPTIONAL and NOT stored (recompute from pose_feat via the frozen fc1), camera_pose (7,)}` as CPU float16/32.
* Trace: when a gate is on, the model appends `(t, weight)` to `model._wgate_trace` (reset at t==0); the worker
  dumps it to `<eval_dir>/<scene>/wgate_trace.json`.
* Parity: with no keys set, outputs must be bit-identical to before (verify on CPU with a fixed seed: run the
  decoder step on random tensors twice, once with the patch's paths disabled by absence of keys).

Recording + training (`src/CUT3R/src/record_wgate_features.py`, `src/CUT3R/src/train_wgate_heads.py`,
sbatch files in `eval_pipeline/`):
* Recorder: frozen model, no gates, bf16 autocast, `--episodes N` (default 4000) from the train split with a
  fixed seed, sharded over GPUs via accelerate (or `--shard i/n`), writes one `.npz` per episode under
  `/gpfs/scratch/etur59/koc821022/checkpoints/wgate/features/<split>/<episode_idx>.npz` with arrays
  `frame_feat (T,1796) f16`, `pose_feat (T,768) f16`, `camera_pose (T,7) f32`, `res_pose_t (T,)`, `res_pose_R (T,)`,
  `res_pts_self (T,)` f32, computed with `criterion.get_all_pts3d` on the full 64-frame sequence (criterion =
  `Regr3DPoseBatchList(L21, norm_mode='?avg_dis')`, `camera1=batch[0]["camera_pose"]`). Also records 64 held-out
  episodes from the test-split loader (`TEST_ROOT`) under `<split>=heldout` for calibration reporting.
* Trainer: loads all npz, trains `FrameConfHead` (AdamW 1e-3, 20 epochs, batch 4096 frames, frames t>0 only) and
  `PoseSigmaHead` (same; needs the frozen `fc1` from the checkpoint's pose decoder); reports held-out NLL and
  Spearman(sigma, residual) per head; writes `checkpoints/wgate/frame_conf_head.pth` = {"state_dict", "log_ref",
  "wmin", "args", "metrics"} and `checkpoints/wgate/pose_sigma_head.pth` = {"state_dict" (out layer only),
  "log_ref_t", "log_ref_R", "wmin", "args", "metrics"}.

Eval plumbing (`eval_pipeline/infer_and_eval_worker.py`, arm JSONs, `eval_pipeline/wgate_table.py`):
* Control JSON keys under `"*"`: `"frame_gate": {"head": path, "wmin": .5}`;
  `"mem_gate": {"head": path, "wmin": .5}` or `{"const": 0.0}` or `{"freeze_after": 8}`. Existing `alpha_all` /
  `alpha` keys remain the constant / oracle controls for variant 1.
* Arm JSONs: `maks_arm_fg_head.json`, `maks_arm_mg_head.json`, `maks_arm_mg_const0.json`,
  `maks_arm_mg_const50.json`, `maks_arm_mg_freeze8.json`; constant/oracle arms are generated after the head arms
  run (they need the trace / GT residuals): `eval_pipeline/wgate_make_controls.py`.
* Table: `eval_pipeline/wgate_table.py` -> `$OUT/wgate/table/wgate_<subset>.{md,tex,png}`, paired vs
  `augfull_lr1e5`, rows: plain; V1 head, V1 dose-matched constant, V1 oracle; V2 leverage probes, V2 head,
  V2 constant, V2 oracle.

## Departures from the paper / DUSt3R (keep this list honest)

1. Backbone frozen; the paper trains the sigma jointly with the predictor.
2. Variant 1's head has no counterpart in either source: a frame-level confidence read from the proposed state
   update (state delta summary + global image feature). Motivated by our finding that pose error is motion-driven
   and the state delta tracks motion.
3. sigma -> weight map `clip(sigma_ref / sigma, wmin, 1)` is ours; the paper only uses sigma inside the loss.
4. Gaussian NLL on a scalar residual (paper eq. 8); DUSt3R's ConfLoss is the Laplace-like `conf*L - a*log conf`.

## Run log

(appended as things run)

### 2026-09-21 Component C (eval plumbing, arm JSONs, control generator, table) -- implemented, CPU-tested

Files: `eval_pipeline/infer_and_eval_worker.py` (edit), `eval_pipeline/maks_arm_{fg_head,mg_head,mg_const0,mg_const50,mg_freeze8}.json`,
`eval_pipeline/wgate_make_controls.py`, `eval_pipeline/wgate_table.py`, `eval_pipeline/wgate_probe430.sh`. Nothing submitted.

* Worker: control keys exactly as the contract (`frame_gate {head, wmin}`, `mem_gate {head, wmin} | {const} | {freeze_after}`),
  parsed by a module-level `apply_wgate_controls(views, ctl, attach_fg, attach_mg)` that sets the per-view (1,) tensors
  `frame_gate_on/frame_gate_wmin/mem_gate_on/mem_gate_wmin/mem_gate_const/mem_gate_freeze_after` on ALL views (the model
  owns a_0 = b_0 = 1). Heads attach once per path (`attach_fg` -> `model.attach_frame_gate(ck["state_dict"], ck["log_ref"])`,
  `attach_mg` -> `model.attach_mem_gate(ck["state_dict"], ck["log_ref_t"], ck["log_ref_R"])`); a `wmin` missing from the json
  falls back to the checkpoint's `wmin`, then 0.5. After inference `dump_wgate_trace` writes
  `<eval_dir>/<scene>/wgate_trace.json = {"scene", "gates", "trace": [[t, weight], ...]}` from `model._wgate_trace`
  (tensors/numpy converted; extra tuple fields kept; a warning is printed if the list is empty while a gate is active).
* ADDED key (per contract instruction): `mem_gate.alpha {frame: b}` -> per-view `mem_gate_const` (the V2 oracle's per-scene
  schedule). The const/freeze_after controls set ONLY `mem_gate_const` / `mem_gate_freeze_after`, no `mem_gate_on`
  (the contract defines `mem_gate_on` as "apply the attached head"); Component A: const/freeze must not be gated behind
  `mem_gate_on`.
* `wgate_make_controls.py`: constants = mean weight POOLED over all frames t>0 of all scenes with a trace (mean of scene
  means also reported in `<pilot_dir>/wgate_controls_summary.json`); refuses to write a constant when < 95% of the
  scenes have a trace. Oracles: V1 `a_t = clip(median(r)/r_t, .5, 1)` with `r_t = e_trans + e_rot`; V2
  `b_t = clip(1/sqrt((e_trans/med_t)(e_rot/med_R)), .5, 1)` (the `combine_pose_sigma` map with ref = subset median);
  medians pooled over all frames t>0 of the subset; frame 0 and NaN/zero residuals -> 1.0; weights rounded to 4 dp.
* DEPARTURE (GT residual definition): `e_trans` = evaluator-style Sim(3) umeyama over the whole scene on frame-0-relative
  poses -- verified equal to the evaluator's per-frame ATE to 3e-9. `e_rot` is the geodesic angle of `R_gt^T R_pred` on the
  frame-0-relative rotations WITHOUT the umeyama rotation: the centre-only fit is undetermined about the axis of a near-linear
  wrist path and applying it added a constant 15-33 deg offset to every frame of the three scenes checked
  (e_rot(t=1) 0.58 rad vs 0.003 rad without it). Consecutive-frame rotation from the same poses matches the evaluator's
  rpe_rot to 1e-3 deg, so the pose loading/convention is right.
* 430-scene oracle run (login node, scratchpad only, not written to eval_pipeline/): 430/430 scenes, 134018 frames t>0,
  median e_trans .0681 m, median e_rot .3008 rad; mean oracle a = .819 (19% at the floor, 50% at 1), mean oracle b = .848
  (13% floor, 54% at 1); 42 s CPU. The 4292 list needs ~7 min CPU -> run it under Slurm, not on a login node.
* `wgate_table.py`: rows in the contract order, `<pilot_dir>/<arm>/eval` (default `$OUT/wgate430`), paired vs
  `augfull_lr1e5` on `--scene_list`, constants read from the arm's `control.json` into the label, `--ref NAME=DIR=LABEL`
  extra rows, missing arms skipped, incomplete arms skipped unless `--allow_partial` (dagger). Verified on a fake pilot of
  symlinked existing 430 trees: reproduces ctl_uniform ATE -.0030*, tok_al_q50_g50 -.0053*, g7ema -.0032*;
  tex -> pdf -> png OK (`/apps/GPP/LATEX/...` prepended to PATH when present). Output `$OUT/wgate/table/wgate_<subset>.*`.
* `wgate_probe430.sh`: one `maks_sweep430.sbatch` per arm, `PILOT=wgate430`, job names `wg430_<arm>`; skips a head arm whose
  `.pth` is missing (FORCE=1 overrides) and an arm already fully scored; job ids appended to `$OUT/wgate430.jids`.
* Tests run: worker parsing on fake views for all five JSONs + wmin fallback + per-scene mem_gate.alpha + error paths;
  trace dump with tensor/numpy/list entries; JSONs validated with json.tool; generator on a fabricated 6-scene pilot.
  Not tested: an actual gated inference (needs Component A's model patch + a GPU job).

### 2026-09-21 — Component A (model hooks + heads) implemented and verified on CPU

Files: `src/CUT3R/src/dust3r/wgate/{__init__.py,heads.py,tests/test_wgate.py,tests/parity_cpu.py}` (new),
`src/CUT3R/src/dust3r/model.py` (hooks in `_forward_decoder_group_step`, methods after `_unc_forward`).

Contract names are all in place: `FrameConfHead(in_dim=1796, hidden=256)`, `PoseSigmaHead(pose_decoder)` (owns
`self.out = Linear(3072, 2)`), `frame_features`, `sigma_to_weight`, `combine_pose_sigma`; view keys `frame_gate_on`,
`frame_gate_wmin`, `mem_gate_on`, `mem_gate_wmin`, `mem_gate_const`, `mem_gate_freeze_after`; `attach_frame_gate(...)`,
`attach_mem_gate(...)`; `model.wgate_record` (list) and `model._wgate_trace`.

Hook points (verified by `test_model_hooks_wrapped_once`): features are computed right after `_recurrent_rollout`
returns (`_wgate_frame_pre`, detached fp32, autocast off); the mem gate is the line directly after
`new_mem = self.pose_retriever.update_mem(...)`; the frame gate multiplies `update_mask` right after the
`update_alpha` block and before `state_mask = update_mask` / the token gate; the commit line is untouched.

Results:
* `test_wgate.py`: 8/8 pass (pure functions, heads with fixed seeds, source grep, hook methods on a stub).
* `parity_cpu.py` on the real `augfull_lr1e5` checkpoint, fp32, 3 random 320x192 frames, 103 s CPU single-thread:
  keys absent vs (i) heads attached but no keys, (ii) `frame_gate_on`+`mem_gate_on` with heads forced to weight 1,
  (iii) `mem_gate_const=1`+`freeze_after=10` are ALL `torch.equal` on camera_pose, pts3d_in_self_view, state_feat, mem
  at t=0,1,2; recording (`wgate_record=[]`) is inert; negative controls behave as derived (both gates at .5:
  state_1 = .5 new + .5 old, `mem_gate_const=0` freezes mem bit-for-bit, t=2 pose changes).

Departures / additions vs the contract text (small, all additive):
1. `wgate_record` entries are CPU **float32** (contract said "float16/32"); the recorder casts to f16 for the npz.
   `frame_feat`/`pose_feat`/`camera_pose` have the batch dim squeezed at B=1, kept as (B, ...) otherwise.
2. Trace: `_wgate_trace` holds `(t, w)`; when BOTH gates are on in the same step the entry is `(t, a_t, b_t)`.
   Per-gate lists `_wgate_trace_frame` / `_wgate_trace_mem` and raw log-variances `_wgate_trace_logvar`
   `[(t, "frame"|"mem", [s...])]` are kept alongside (reset together at t == 0 whenever any wgate key is present).
3. `attach_frame_gate(head_or_sd, log_ref=None, wmin=0.5)` / `attach_mem_gate(head_or_sd, log_ref_t=None,
   log_ref_R=None, wmin=0.5)` also accept the whole `.pth` dict (`{"state_dict", "log_ref*", "wmin", ...}`) and then
   read the refs / wmin from it; `attach_mem_gate` accepts `{"weight","bias"}` or `{"out.weight","out.bias"}`.
   The view keys `*_wmin` override the attached wmin. Pass `None` to detach.
4. `PoseSigmaHead` borrows `fc1`/`act` WITHOUT registering them: `state_dict()` = out layer only (as the contract's
   checkpoint format wants), `parameters()` = out layer only, and `.to()/.float()` never touch the model. The trainer
   must place the pose decoder on its device itself; `penultimate(pose_feat)` recomputes h (fp32, no grad).
5. `combine_pose_sigma` returns `sigma_comb` (the contract's formula); the weight is `pose_sigma_to_weight(log_var2,
   log_ref2, wmin) = clip(1/sigma_comb, wmin, 1)`. `FrameConfHead.forward` returns `(B,)`. Helper `gaussian_nll(s, r)`
   = `exp(-s) r^2 + s` added for the trainer.
6. t == 0: the heads are not run at all (`a_0 = b_0 = 1` hard-coded, trace records 1.0). Any step whose weight is
   exactly 1 for the whole batch skips the blend arithmetic, which is what makes parity bit-exact rather than close.
   When `b != 1` the mem blend is computed in fp32 (matches the existing promotion at the `update_mask` line).
7. Gates and recording assert `views_per_step == 1` and a pose-head model (they need `global_img_feat_group`).

Observation for the group (not a departure — it is what the contract's "existing update_alpha path" does): the frame
gate multiplies `update_mask`, and `update_mask` is ALSO the commit mask of the retriever memory
(`mem = new_mem * update_mask + mem * (1 - update_mask)`), so a_t scales both writes; with both gates on the mem write
is a_t * b_t (checked numerically in parity arm D). The V1 dose-matched constant (`alpha_all`) and oracle (`alpha`)
controls go through the same path, so V1 comparisons stay like-for-like. Restricting V1 to the state only would be a
one-line change (multiply `state_mask` instead of `update_mask`).

CPU-only note: the compiled curope kernel's CPU path rejects the float16 q/k that croco's `Attention` feeds RoPE, so
`parity_cpu.py` swaps every `cuRoPE2D` for the pure-PyTorch `RoPE2D` (croco's ImportError fallback) in ALL arms; GPU
numerics are untouched. GPU/bf16 rollouts were not run here (login node); the heads upcast and disable autocast, so
they see the same fp32 inputs the pose head reads.

### 2026-09-21 Component B (recording pass + head trainer) implemented

Files: `src/CUT3R/src/record_wgate_features.py`, `src/CUT3R/src/train_wgate_heads.py`,
`eval_pipeline/record_wgate.sbatch`, `eval_pipeline/train_wgate.sbatch`. Both scripts have `--smoke` (2 episodes /
1 epoch) and `--selftest` (CPU, synthetic tensors / synthetic npz table); both selftests pass on a login node
(recorder ~4 s, trainer ~16 s, the latter against Component A's real `dust3r.wgate.heads`).

Recorder, as contracted: frozen model, `.eval()`, no gates, bf16 autocast, `build_loader(TRAIN_ROOT, 64, 1, ...)`,
`set_epoch(loader, seed)` (seed 0), `--episodes 4000` = first N of that order, accelerate-sharded (rank r's i-th
batch is global episode `i*n_proc + r`; verified against accelerate 1.10.1 `BatchSamplerShard`), one npz per
episode `<root>/features/<split>/<idx:06d>.npz`, 64 heldout episodes from `TEST_ROOT` under `heldout`. Residuals
from `Regr3DPoseBatchList(L21, norm_mode='?avg_dis').get_all_pts3d(views, preds, camera1=batch[0]["camera_pose"])`
with autocast off (the criterion's kwarg is `camera1`, same as `train_conf_gate` passes it through `ConfLoss`).
Selftest facts: exact predictions give residual 0 to 1e-4; a 0.05 translation offset gives the analytic
`0.05*sqrt(3)/norm_factor` on every frame with rotation residual 0.
Additions beyond the contract (all extra, nothing renamed):
* extra npz arrays `t`, `pts_valid_frac` (res_pts_self is 0 where no pixel is valid), `pose_mask` (criterion's
  per-sequence validity), `scene`, `first_frame`, `last_frame`, `dataset_idx`, `split`;
* `<root>/pose_decoder.pth` (the checkpoint's `downstream_head.pose_head` state dict, written by rank 0) so the
  trainer needs neither the 3 GB checkpoint nor a GPU; the trainer falls back to an mmap read of the checkpoint
  (`module.`-prefixed keys handled);
* existing npz are skipped on re-run (`--overwrite` to redo); `camera_pose` in the npz is the head output the
  criterion saw; a mismatch with the hook's copy is logged once, not fatal.

Trainer, as contracted: frames t>0 only; FrameConfHead on r = res_pose_t + res_pose_R + res_pts_self, PoseSigmaHead
on (r_t, r_R) with the frozen fc1/GELU recomputed on the fly (only `out.*` trains; the contract head's
`state_dict()` is already out-only); Gaussian NLL `exp(-s) r^2 + s` (s clamped to +-10 inside exp only); AdamW 1e-3
(wd 1e-4), 20 epochs, batch 4096, per-step cosine to 1% of lr; residuals divided by their TRAIN median (stored as
`res_scale` / `res_scale_t` / `res_scale_R`); `log_ref*` = median s over the TRAIN table; `wmin` 0.5. Reports per
head: heldout NLL vs the constant-s baseline (`s = log mean r^2` on train), Spearman(sigma, r) (+ `spearman_comb` =
Spearman(s_t + s_R, r_t + r_R) for the pose head, monotone in the contract's `combine_pose_sigma`), a 5-bin
sigma-quantile calibration table (`r_rms / sigma_rms` ~ 1 when calibrated), and the mean gate weight the
`sigma_to_weight` map would give (dose preview). Frames with non-finite residuals and episodes with `pose_mask`
False are dropped and counted. Checkpoints carry exactly the contract keys plus the divisors, `head_cfg`, and a
`standin` flag (False when trained with the real heads).
Options beyond the contract, all OFF by default: `--r_clip K` (cap the standardised residual at K medians),
`--min_valid_frac`, `--val_frac F --early_stop` (holds back F of the TRAIN episodes and restores the best-val
epoch; the heldout split never selects). Motivation: on the synthetic selftest table the 1796->256->1
FrameConfHead (460k params) overfits small tables (train NLL 1.1 vs heldout 7.6 on 2k iid-feature frames; fine on
low-rank features at 4k frames). The real table is ~252k frames; the per-epoch train/heldout NLL curve is in
`train_wgate_metrics.json` -- if heldout NLL rises while train NLL falls, rerun with `EXTRA="--val_frac 0.1
--early_stop"` (or fewer epochs) and say so here.
Launch: `sbatch eval_pipeline/record_wgate.sbatch` (4 GPUs, acc_ehpc, 4 h; env EPISODES/HELDOUT/EXTRA/SMOKE/WGATE_ROOT)
then `sbatch eval_pipeline/train_wgate.sbatch` (1 GPU; env HEADS/EXTRA/SMOKE/WGATE_ROOT).
* Addendum (same day, after Component A's model patch landed): the worker passes the trainer's `wmin` through to
  `attach_frame_gate/attach_mem_gate(..., wmin=)` (the per-view `*_wmin` key still overrides it), and `wgate_trace.json`
  also carries the model's per-gate lists when present: `trace_frame` [[t, a_t]], `trace_mem` [[t, b_t]],
  `trace_logvar` [[t, "frame"|"mem", [log sigma^2 ...]]] (raw head outputs; weights can be recomputed for another wmin
  without rerunning). `wgate_make_controls.py` prefers `trace_frame` / `trace_mem` and falls back to `trace`. Verified
  against model.py: trace lists reset at t == 0 whenever any wgate key is present; the mem-gate path fires on any of
  `mem_gate_on / mem_gate_const / mem_gate_freeze_after`, so the const / freeze arms are correct without `mem_gate_on`.
Smoke (Slurm 46267143, acc_debug, 1 GPU, `WGATE_ROOT=.../checkpoints/wgate/smoke`, SMOKE=1): recorder end-to-end
with Component A's hook, 2 train + 2 heldout episodes, ~8 s/episode incl. loading after warm-up (=> 4000 episodes on
4 GPUs ~ 2-2.5 h; raise `--time` or split EPISODES if the queue is slow). npz arrays exactly as contracted
(frame_feat (64,1796) f16, pose_feat (64,768) f16, camera_pose (64,7) f32, residuals (64,) f32); frame-0 pose
residuals 0.001-0.003 (identity reference), t>0 per-episode means r_t 0.07-0.30, r_R 0.02-0.12, r_pts 0.08-0.35,
all finite, pose_mask True; the recorded `delta_norm` feature (index 768) falls from ~24.5 at t=0 (first write over
the learned prior) to ~1.4-3 by t=63; hook camera_pose == head output (no mismatch warning). Note the train loader
exposes 8.79M start indices (every frame of every train scene can start a 64-frame window), so "4000 episodes" are
4000 random windows of the seeded permutation, not 4000 distinct scenes; heldout windows come from TEST_ROOT.
Trainer smoke (Slurm 46267834, acc_debug, 1 GPU, 22 s, SMOKE=1 on the same smoke root): both checkpoints written with
the contract keys (`pose_sigma_head.pth` state_dict = {out.weight (2,3072), out.bias}); train medians on the
2-episode table r_t 0.097, r_R 0.030, r_pts+r_t+r_R 0.191. Metrics at 2 episodes / 1 step are noise (heldout NLL >
constant baseline, as expected with 126 frames) -- not a result. The same table trained 5 epochs on a login-node CPU
in 8 s (the trainer only needs a GPU for speed on the full table). Train Spearman 0.81 vs heldout 0.16 for the
frame head on 126 frames is the overfitting signature to watch for on the full run (see options above).
Full recording NOT yet launched (needs 4 GPUs on acc_ehpc): `sbatch eval_pipeline/record_wgate.sbatch`, then
`sbatch eval_pipeline/train_wgate.sbatch`.

### 2026-09-21 Integration smoke test (CPU tests + two acc_debug jobs) -- PASSED

Scope: Components A+B+C wired together on the real checkpoint. Nothing committed. New files:
`eval_pipeline/wgate_smoke_record_train.sbatch` (job 1), `eval_pipeline/wgate_smoke_worker.sbatch` (job 2),
`eval_pipeline/maks_arm_wgsmoke_{plain,mg_const1,fg_wmin1,fg_head,mg_head,mg_const0,mg_freeze8}.json` (smoke arms;
the head arms point at `checkpoints/wgate/smoke/{frame_conf_head,pose_sigma_head}.pth`).

1. CPU (login node, `OMP_NUM_THREADS=1 timeout 280`, env cuteanything): `python -m pytest -q dust3r/wgate/tests/test_wgate.py`
   -> 8 passed in 4.2 s; `python dust3r/wgate/tests/parity_cpu.py` -> `PARITY OK`, rc 0, 107 s (real augfull_lr1e5 ckpt,
   fp32, 3 frames: arms A/A2/A3/B/C bit-identical on camera_pose, pts3d_in_self_view, state_feat, mem; D/E negative
   controls as derived).
2. Job 1 = Slurm 46269786 (acc_debug, 1 GPU, as01r5b11, 1:05 wall, exit 0):
   `sbatch -A etur59 -p acc -q acc_debug --gres=gpu:1 -c 20 --time=00:30:00 eval_pipeline/wgate_smoke_record_train.sbatch`.
   `record_wgate_features.py --smoke --overwrite --root checkpoints/wgate/features_smoke` (52 s; npz at
   `features_smoke/features/{train,heldout}/00000{0,1}.npz` -- the recorder's fixed `<root>/features/<split>/` layout, so
   the task's "into features_smoke/" is one level deeper than literal) then `train_wgate_heads.py --smoke --features
   features_smoke/features --out checkpoints/wgate/smoke --pose_decoder features_smoke/pose_decoder.pth` (5 s) in the same
   job. In-job verify: every npz has exactly `frame_feat (64,1796) f16, pose_feat (64,768) f16, camera_pose (64,7) f32,
   res_pose_t/res_pose_R/res_pts_self (64,) f32`, all finite (+ Component B's extra arrays); `frame_conf_head.pth` keys
   {state_dict (norm/fc1/out), log_ref .1017, wmin .5, args, metrics, +res_scale, head_cfg, residual, standin=False};
   `pose_sigma_head.pth` keys {state_dict = out.weight (2,3072)/out.bias, log_ref_t .0921, log_ref_R .0774, wmin .5, args,
   metrics, +res_scale_t/R, pose_decoder_src, residual, standin=False}; `FrameConfHead.from_state_dict` loads it. The
   seeded order reproduced the implementer's smoke (same 4 episodes: WEIRD+f1c42455, TRI+52ca9b6a+2023-11-01 / CLVR+13759f6e,
   TRI+52ca9b6a+2024-02-07). The implementer's earlier smoke checkpoints were moved to `checkpoints/wgate/smoke/prev_46267834/`
   before being overwritten. NOTE: these smoke heads are one gradient step on 126 frames, i.e. effectively untrained
   (weights within 0.5% of 1); they exercise the plumbing, not the idea.
3. Job 2 = Slurm 46269951 (acc_debug, 1 GPU, as01r1b09, 2:13 wall, exit 0):
   `sbatch -A etur59 -p acc -q acc_debug --gres=gpu:1 -c 20 --time=00:40:00 eval_pipeline/wgate_smoke_worker.sbatch`
   = `infer_and_eval_worker.py` with the maks_sweep430 flags (`--size 320 --skip_mode block --no_plots
   --ignore_skip_sentinel --shard_id 0 --num_shards 1`, control.json copied per arm) on ONE scene
   `RAIL+80edfcb1+2023-07-14-14h-28m-45s` (128 frames), PILOT=wgate_smoke -> `$OUT/wgate_smoke/<arm>/{eval,preds,control.json,
   worker_g0.log}`, eight arms serially, 14-23 s each. Departure: predictions were KEPT (KEEP_PREDS=1, 258 MB total) so the
   camera npz could be diffed against `$OUT/augfull_lr1e5/preds`. Comparison script output in
   `$OUT/wgate_smoke/smoke_compare.json`:
   * plain, plain2 (same-job determinism control), `mg_gate.const=1.0`, `frame_gate` smoke head with wmin 1.0: eval CSV
     byte-identical to `$OUT/augfull_lr1e5/eval/<scene>/eval_depth_pose_metrics.csv` (ALL row absrel .128941 a1 .906821
     ATE .025208 rpe_trans .005691 rpe_rot .871241; max |diff| 0.0 on every column) AND all 128 camera npz (pose 4x4,
     intrinsics) bit-identical to the reference preds. So the gated code path at weight 1 is exact on GPU, not just on CPU,
     and this GPU rollout is deterministic across processes and against the historical run.
   * fg_head (wmin .5): trace 128 entries t=0..127 consecutive (a_0 = 1.0 hard-coded, 127 entries t>0), a_t in
     [.9987, 1] (11 frames < 1), each a_t == clip(exp(.5(log_ref - s_t)), .5, 1) recomputed from `trace_logvar` and the
     checkpoint's log_ref; outputs differ from plain (ATE .025702, rpe_rot .916833).
   * mg_head (wmin .5): 127 entries t>0, b_t in [.9957, 1] (23 < 1), b_t == clip(1/sigma_comb, .5, 1) from the two
     log-variances and the checkpoint refs; ATE .025306, rpe_rot .930633.
   * mg_const0: 127 entries t>0 all exactly 0.0; differs from plain (ATE .054888 vs .025208, max per-frame camera pose
     diff .70). mg_freeze8: b_t = 1.0 for t=1..8 and exactly 0.0 for t=9..127 (8 + 119 entries); ATE .074801.
   * The worker's `wgate_trace.json` carries `trace` + the per-gate `trace_frame`/`trace_mem` and `trace_logvar` lists,
     `gates` echoes the control spec (incl. the resolved wmin); no WARNING lines, no tracebacks in any worker log.
   Observation for the group: a gate perturbation of <= 0.13% on 11 frames (fg_head) moves single-scene ATE by 2% and
   rpe_rot by 5% -- the recurrent rollout is very sensitive to tiny write-scale changes, so per-scene deltas from a
   near-identity gate are not signal; only the paired 430/4292 tables are.
Known reviewer items were not blockers here: the a_t-also-scales-mem-commit question (Component A) is unchanged; the
trace-reset-only-when-gated case does not arise (every smoke arm keys frame 0); `--seed` is only exercised at 0.
Minor logging finding (Component C, not a behaviour defect): the "frame gate head attached from ... (log_ref=..., wmin_train=...)"
/ "mem gate head attached ..." lines never reach the worker log because `apply_wgate_controls` (and hence `attach_fg` /
`attach_mg`) runs inside the `contextlib.redirect_stdout(os.devnull)` block that silences `demo.prepare_input` (worker
L544-604; the pre-existing `attach_unc` / `attach_cb` calls in the same block are swallowed the same way). The head path
and resolved wmin are still recorded in `wgate_trace.json["gates"]`, and the attachment was verified here from the traced
log-variances, so nothing is lost; moving the wgate call (or the print) outside the redirect would restore the log line.

### 2026-09-21 Component C fix pass (review items on the eval component) -- CPU-tested, nothing submitted

Files edited: `eval_pipeline/wgate_make_controls.py`, `eval_pipeline/wgate_table.py`, `eval_pipeline/wgate_probe430.sh`,
`eval_pipeline/infer_and_eval_worker.py` (prints only). New: `eval_pipeline/wgate_selftest.py` (CPU self-test, 3 s).
No model / recorder / trainer file touched; nothing committed.

1. MAJOR fixed -- the V1 dose-matched constant no longer scales frame 0. `wgate_make_controls.py` emits
   `{"*": {"alpha_all": c, "alpha": {"0": 1.0}}}` (`v1_const_spec`): the worker sets `update_alpha` from `alpha_all` on
   EVERY view incl. view 0 and then applies the per-frame `alpha` entries on top (alpha_all block precedes the alpha
   block; the self-test asserts that order in the worker source), and model.py's update_alpha path has no t == 0
   exemption, whereas the frame-gate head hard-codes a_0 = 1 (`_wgate_frame_pre`). Frame 0 is the dominant write
   (delta_norm ~24.5 at t=0 vs 1.4-3 later, Component A's log), so the old form was not dose-matched where it mattered
   most. `frame_weight(spec, t)` mirrors the worker's precedence and every emitted constant is asserted to give frame 0
   -> 1.0 and frame 1 -> c; `wgate_table.const_of` still reads `alpha_all`. The V2 constant is unaffected (the model
   writes b_0 = 1 for every mem_gate key). HISTORICAL NOTE for the group: the conf-gate campaign's `ctl_uniform` arm
   (`maks_arm_ctl_uniform.json` = `{"*": {"alpha_all": 0.8527}}`, the ATE -.0030* row) DID write frame 0 at alpha
   .8527 too -- it is a "constant on all frames incl. 0" control, not a t>0-only dose match, so it is not directly
   comparable to the new fg_const row.
2. V1 oracle residual made unit-free (closes the implementer's open question): `r_t = e_trans/median(e_trans) +
   e_rot/median(e_rot)`, `a_t = clip(median(r)/r_t, wmin, 1)`, medians pooled over all frames t>0 of the subset -- the
   term-wise reading of "normalised by its median". Before, `r = e_trans[m] + e_rot[rad]`, and at the subset medians
   (.068 m vs .30 rad) the oracle ranked frames almost entirely by rotation. Summary key renamed `median_r_norm`
   (+ a `v1_residual` string); the V2 oracle was already term-wise and is unchanged. The self-test checks both oracles
   are invariant to m -> mm and rad -> deg rescaling. The 430-scene oracle numbers quoted in the Component C entry
   (mean oracle a = .819, 19% at floor / 50% at 1) were computed with the OLD residual; regenerate after the head arms run.
3. Oracle scenes with no control entry are no longer silent. Generator: `maks_arm_{fg,mg}_oracle.skipped.txt` sidecars
   next to the jsons (+ a WARNING line). Worker: `WARNING: <scene> has no control entry in <json> -> plain` whenever a
   per-scene control json (no "*") lacks the scene, and a one-time `WARNING: --control_json with --skip_mode drop: ...
   controls are IGNORED` at start-up (the whole control block is gated on `--skip_mode block`). Table: a scene absent
   from the arm's copied `control.json` (per-scene keys, no "*") or listed in `eval_pipeline/maks_arm_<arm>.skipped.txt`
   counts as UNSCORED for that arm (it ran plain), so it leaves the common set instead of being scored as an oracle row.
4. `wgate_table.py`: ONE common scene set. Pass 1 loads and admits every arm (`--min_frac` default .999 as before;
   incomplete arms skipped unless `--allow_partial`); pass 2 computes every row INCLUDING the baseline means on
   `common = listed ∩ baseline ∩ (every complete arm)`. `--allow_partial` rows (dagger) are computed on their scored part of
   the common set and show their own n. The json now carries `n_listed`, `common_n`, `dropped_scenes`, `skipped:
   [{arm, reason}]`; the md/tex end with "Common scene set k/N ..." and "Rows not shown: fg_const (missing); mg_const0
   (1/20 scored)". Verified on a fabricated 20-scene pilot: mg_head 19/20, oracle lacking 2 (+1 sidecar) -> every shown
   row n = 16 (baseline included), skipped = mg_const0 + the five absent contract arms; the footnoted tex builds to png.
5. `wgate_probe430.sh`: before submitting, `squeue -h -u $USER -n <jobname> -o %i`; a queued/running job of the same arm
   prints `QUEUED/RUNNING <arm> (job <id>) -- not resubmitting`. Job name is now `wg${PILOT#wgate}_<arm>` (wgate430 ->
   `wg430_<arm>`, unchanged; wgate4292 -> `wg4292_<arm>`) so the check is per pilot. Dry run with stub sbatch/squeue on
   PATH and OUT redirected to the scratchpad: run 1 submits, run 2 (stub squeue lists 4242) skips, run 3 with
   PILOT=wgate4292 submits as `wg4292_mg_const0` -- no duplicate job on the same eval tree.
6. Smoke-test logging finding closed: `attach_fg` / `attach_mg` print via `sys.__stdout__`, bypassing the
   `redirect_stdout(os.devnull)` block they run inside, so the "head attached (log_ref, wmin_train)" line reaches
   `worker_g*.log`. (`attach_unc` / `attach_cb` are pre-existing and left as they were.)
Tests: `OMP_NUM_THREADS=1 timeout 280 python eval_pipeline/wgate_selftest.py` -> `wgate_selftest: ALL OK`, 3.0 s real
(generator on a 6-scene fabricated pilot with GT/pred cameras; `apply_wgate_controls` on the emitted mem-gate jsons; table
on a 20-scene fabricated pilot with and without `--allow_partial`, incl. pdflatex + gs). The GPU smoke was not re-run (no
Slurm in this pass); none of these edits touch the model or the inference path -- the worker changes are prints only --
so the smoke-test parity evidence above stands.

### 2026-09-21 Component A FIX PASS (review items on model.py) -- applied, CPU-tested, no Slurm

Files: `src/CUT3R/src/dust3r/model.py`, `src/CUT3R/src/dust3r/wgate/tests/{test_wgate.py,parity_cpu.py}`. Nothing committed,
nothing submitted. Other components' files untouched (the one needed change there is more than a key rename, see 1c).

1. MAJOR -- V1 frame gate is now STATE-ONLY, as the contract's write formula says (`state = a*proposal + (1-a)*old`;
   the variant is named "confidence-weighted state update"; V2 owns the retriever memory). Decision taken here so the two
   variants are orthogonal; the joint reading is kept reachable, not deleted:
   a. Code: `a_t` multiplies `state_mask` on the statement immediately before `state_feat = new_state_feat * state_mask + ...`
      (after the token gate, so tok / `update=False` / reset compose as before); the mem commit keeps `update_mask`.
      The multiply at the `update_alpha` site remains only for the JOINT mode: new per-view key `frame_gate_joint` (> 0 =>
      `a_t` scales `update_mask`, i.e. state AND retriever-mem commit, exactly the pre-fix behaviour and what
      `update_alpha` does). Absent key = state-only.
   b. New per-view key `frame_gate_const` (a_t = const for t > 0, overrides the head, per-view values give a schedule,
      a_0 = 1, traced in `_wgate_trace(_frame)`; the head's log-variance is still traced when `frame_gate_on` is also set;
      no head / no features needed) -- the mirror of `mem_gate_const`, so the V1 dose-matched constant and oracle rows run
      through the SAME state-only branch as the head arm.
   c. CONSEQUENCE FOR COMPONENT C (not edited here): the current V1 controls `alpha_all` / `alpha` -> `update_alpha` are
      JOINT (state + mem) and therefore no longer like-for-like with the state-only head arm. Needed: worker
      `apply_wgate_controls` accepts `"frame_gate": {"const": a}` and `"frame_gate": {"alpha": {t: a}}` -> per-view
      `frame_gate_const` (and drops the "frame_gate needs a 'head' path" ValueError; optionally `"joint": 1` -> per-view
      `frame_gate_joint`); `wgate_make_controls.py` emits `{"*": {"frame_gate": {"const": c}}}` and
      `{scene: {"frame_gate": {"alpha": w}}}` instead of `alpha_all` / `alpha`; `wgate_table.py` reads the V1 constant from
      `control.json["*"]["frame_gate"]["const"]`; worker header comment (L141-143) is stale. `update_alpha` itself is
      untouched (earlier campaigns' `ctl_uniform` etc. keep their meaning). Until C lands, `alpha_all` / `alpha` are valid
      controls only for a `frame_gate_joint = 1` head arm.
   d. The integration smoke's `fg_head` row (ATE .025702 / rpe_rot .916833 on the one scene) was measured with the JOINT
      gate and must be re-run; `fg_wmin1` (weight 1, bit-identical to plain) is unaffected by construction.
2. MINOR -- trace lists (`_wgate_trace`, `_wgate_trace_frame`, `_wgate_trace_mem`, `_wgate_trace_logvar`) are now reset
   UNCONDITIONALLY at every view index 0 (keys or not; the `mem_keys` lookups in the pre-hook are gone), so a scene keyed
   `frame_gate_on = 0` or not keyed at all never inherits the previous scene's trace. A plain scene therefore leaves the
   four attributes as empty lists (the parity check "no trace written without keys" accepts `[]`).
3. MINOR -- `attach_frame_gate(head_or_sd, log_ref=None, wmin=None)` / `attach_mem_gate(head_or_sd, log_ref_t=None,
   log_ref_R=None, wmin=None)`: an explicit argument always wins, a `None` is read from the whole-dict checkpoint, `wmin`
   finally defaults to 0.5 (same precedence as `log_ref*`). The worker passes `wmin=` explicitly -> unaffected. Also the
   stored refs are exact Python floats now (`_wgate_scalar`; previously routed through a float32 tensor, e.g. 0.1 ->
   0.10000000149; downstream fp32 math identical).
4. MINOR -- heads stay REGISTERED submodules (`self.frame_gate` / `self.mem_gate`, like `unc_gate` / `conf_branch`), now
   documented in the wgate comment block and both docstrings: `model.to()` carries them; `model.state_dict()` gains
   `frame_gate.*` / `mem_gate.*` keys while attached (detach with `None` before saving a backbone checkpoint);
   `model.train()` is harmless (LayerNorm / Linear / GELU only, no dropout / BN); frozen, so an optimiser built from
   parameters requiring grad never sees them.

Tests (login node, env cuteanything, `OMP_NUM_THREADS=1 timeout 280`):
* `python -m pytest -q dust3r/wgate/tests/test_wgate.py` -> 8 passed (4.2 s). Source grep now pins the state-only
  multiply as the statement right before the commit and the joint multiply between `update_alpha` and the token gate,
  both guarded on `wg_frame_w is not None` and mutually exclusive on `wg_frame_joint`; stub tests add
  `frame_gate_const` (fires without `frame_gate_on`, overrides the head, per-view schedule, not clipped by wmin), the
  unconditional reset (gated scene -> keyed-off scene / unkeyed scene at t = 0 -> `[]`), and the attach precedence
  (explicit > checkpoint > 0.5; both mem state-dict formats).
* `python dust3r/wgate/tests/parity_cpu.py` -> `PARITY OK`, rc 0, 132.6 s CPU (real augfull_lr1e5 ckpt, fp32, T = 3):
  A/A2/A3/B/C bit-identical as before; D (both heads at .5) now gives state .5/.5 and mem == .5*update + .5*old (b only;
  was .25/.75); NEW F (`frame_gate_const = .5`, no head keys): state .5/.5, mem bit-identical to A at t = 1, t = 1 pose
  untouched, t = 2 pose differs, trace `[(0,1),(1,.5),(2,.5)]`, no logvar; NEW G (D's keys + `frame_gate_joint = 1`):
  state == D, mem == .25*update + .75*old (the pre-fix numbers reproduced through the joint key); E unchanged; one plain
  t = 0 step after the gated arms leaves all four trace lists empty. Logs: scratchpad `pytest_wgate_fix.log`,
  `parity_cpu_fix.log`.
Not run: any GPU job (per the fix-pass instructions). The state-only head arm and the `frame_gate_const` path have been
verified on CPU only; the acc_debug smoke (`eval_pipeline/wgate_smoke_worker.sbatch`) should be re-run for `fg_head` once
Component C's control keys land.

### 2026-09-21 Component B fix pass (recorder / trainer / sbatch) -- review items addressed, CPU-tested, nothing submitted

Files: `src/CUT3R/src/record_wgate_features.py` (rewritten sampling path), `src/CUT3R/src/train_wgate_heads.py`
(`gaussian_nll`), `eval_pipeline/record_wgate.sbatch`, `eval_pipeline/train_wgate.sbatch` (headers), NEW
`eval_pipeline/record_wgate_precision_check.sbatch`. The production `checkpoints/wgate/features/` root was still
empty, so no recorded table is invalidated. `features_smoke/` (bf16, plain order) is unchanged and still loads.

1. MAJOR `--seed` ignored (accelerate `DataLoaderShard.__iter__` re-applies `set_epoch(self.iteration)` = 0 over
   whatever `train_unc_gate.set_epoch` set): FIXED by removing `accelerator.prepare(loader)` from the recorder
   altogether. The recorder now takes the dataset and its `CustomRandomSampler` from `build_loader` (never iterated),
   calls `sampler.set_epoch(seed)` itself and walks the permutation in `select_episodes` (identical on every rank), then
   reads its share through a plain `torch.utils.data.DataLoader(batch_sampler=FixedBatchSampler([...]))` of explicit
   `(dataset_idx, feat_idx, nview)` tuples. Rank layout unchanged (episode g -> rank g % n_proc, `--shard i/n` for
   non-accelerate launches). `manifest.json` carries `seed` and `selection.sampler_epoch` (the epoch the permutation
   was drawn at), `episodes.json` the full list; the loop asserts scene + first frame of every batch against the list.
   Selftest step 6 checks that seed 0 and seed 5 give different orders and that the plain order equals the sampler's.
   NOT changed: `train_unc_gate.set_epoch` (not this component's file, not a key-name mismatch); the same latent
   override exists in `train_unc_gate` / `train_conf_gate` but only matters there when resuming at a non-zero epoch,
   since `DataLoaderShard.iteration` increments per full pass anyway. Suggested one-liner for its owner: add `loader`
   itself to the candidate list (`DataLoaderShard.set_epoch` sets `.iteration`).
2. MINOR overlapping windows: FIXED (DEPARTURE from the literal "first N of the seeded order"; the contract's
   "fixed seed" and "sharded via accelerate (or --shard i/n)" are kept). `select_episodes` keeps the first N windows
   whose [start, start+63] does not intersect an already kept window of the same scene; `--allow_overlap` restores
   the plain order. Measured on the REAL indices on a login node (`scratchpad/select_real.py`, 149 s total of which
   130 s is the cold DL3DV index cache; the selection itself is 0.3 s on 8,786,039 starts / 38,442 scenes): train
   4000 kept from 4103 candidates, 103 overlapping windows skipped (the plain order had 102 overlapping pairs, i.e.
   the reviewer's ~100 estimate), 3715 distinct scenes vs 3630 for the plain order, first 291 episodes identical to
   the old numbering (episodes 0-2 = the smoke's 8783096 / 1202982 / 5180043), 1000 episodes per rank at n_proc 4;
   heldout: 64 kept from 64 candidates, 0 overlaps, 64 scenes, list identical to the smoke's. Re-runs no longer load
   already recorded episodes (the skip happens before the loader is built).
3. MINOR bf16 recording vs fp32 eval rollout: DEPARTURE -- the recorder now defaults to fp32 (autocast OFF), exactly
   the worker's `inference()` path (`dust3r/inference.py` L279/L307 `autocast(enabled=False)`), so the heads see the
   feature distribution they are served; `--bf16` (sbatch `PRECISION=bf16`) restores the contract's autocast, `--fp32`
   is kept as a no-op alias, the npz gains a `precision` string and the manifest records it. Expected cost ~1.5-2x on
   the rollout only (loading dominates the 8 s/episode measured in bf16); `record_wgate.sbatch` wall raised 4 h -> 6 h.
   The 2-episode bf16-vs-fp32 comparison the reviewer asked for is packaged as
   `eval_pipeline/record_wgate_precision_check.sbatch` (records the 2+2 smoke episodes both ways into
   `checkpoints/wgate/features_prec/{fp32,bf16}` and prints per-episode feature / log-variance / a_t / b_t gaps
   relative to their across-frame spread, `precision_compare.json`); NOT run here (no Slurm submissions in this pass).
4. MINOR `gaussian_nll` clamp trap: FIXED. The trainer now uses `dust3r.wgate.heads.gaussian_nll` (Component A,
   |s| clamped at 30 inside exp only) when importable; the local fallback (only for `--selftest` without Component A)
   clamps the overflow side only, `exp(-max(s, -30))`, and agrees with Component A to 0.0 on [-35, 35]. Numerical
   check: at s = -12, r = 0.005 medians the old +-10 clamp gave dL/ds = +1 (runaway downwards); now -3.07 (pulls s back
   towards s* = log r^2 = -10.6). Docstring explains why.
5. MINOR sbatch header: FIXED. `record_wgate.sbatch` smoke line is now `SMOKE=1 sbatch -q acc_debug --gres=gpu:1 -c 20
   --time=00:30:00 ...` (note that the file's 4-GPU / 80-CPU defaults are for acc_ehpc); same `--gres=gpu:1 -c 20`
   added to `train_wgate.sbatch`'s smoke line for uniformity; `PRECISION` and the new `EXTRA` flags documented.

Tests (login node, `OMP_NUM_THREADS=1 timeout 280`, env cuteanything): `record_wgate_features.py --selftest` OK in
4 s (residual derivation, npz round trip incl. `precision`, selection on a 3-scene synthetic index: 4 distinct of 10
requested from 175 starts, prefix-consistency with the plain order, seed sensitivity, rank sharding, fixed loader
returns the listed windows in order, `--shard` parsing); `train_wgate_heads.py --selftest` OK in 13 s (frame head
heldout NLL 1.93 vs const 2.14, Spearman .50; pose head 4.65 vs 6.73, Spearman t .58 / R .54) now through Component
A's `gaussian_nll`. Not re-run: GPU smoke (would be `SMOKE=1 sbatch -q acc_debug --gres=gpu:1 -c 20 --time=00:30:00
eval_pipeline/record_wgate.sbatch`, then the integration job `wgate_smoke_record_train.sbatch`, which needs no change:
same flags, now fp32, npz extras gain `precision`). Launch order for the real table is unchanged:
`sbatch eval_pipeline/record_wgate.sbatch` then `sbatch eval_pipeline/train_wgate.sbatch`.

### 2026-09-21 17:20 launches (orchestrator)
* Recording pass: `EPISODES=4000 HELDOUT=64 sbatch eval_pipeline/record_wgate.sbatch` -> job 46272493 (acc_ehpc, 4 GPU,
  6 h limit), features under `/gpfs/scratch/etur59/koc821022/checkpoints/wgate/features/{train,heldout}/`.
* V2 leverage probes on the 430 subset (`wgate_probe430.sh`, PILOT=wgate430): mg_const0 job 46272495, mg_freeze8 job
  46272496, mg_const50 job 46272497.
* Pre-launch checks after the fix pass: `test_wgate.py` 8/8, `parity_cpu.py` PARITY OK (132 s CPU), frame gate state-only.

### 2026-09-21 17:40 orchestrator fix: V1 controls moved to the STATE-ONLY path
The fix pass made the frame-gate head state-only (`state_mask *= a_t`, retriever-mem commit untouched) but left the
V1 dose-matched constant and oracle on `alpha_all` / `alpha`, which multiply `update_mask` and therefore ALSO scale the
retriever-mem commit (the conf-gate campaign's joint frame gate). That would have compared a state-only head against
joint controls. Now: worker accepts `frame_gate: {"const": c}` and `frame_gate: {"alpha": {t: a_t}}` (no head needed)
-> per-view `frame_gate_const` (model: a_0 = 1, state only); `wgate_make_controls.py` emits
`{"*": {"frame_gate": {"const": c}}}` and `{scene: {"frame_gate": {"alpha": ...}}}`; `wgate_table.const_of` reads
`frame_gate.const`. `alpha_all` / `alpha` are unchanged and remain the JOINT gate (historical rows). Self-test updated
and passing; worker parse checked on both forms.
* 17:25 queued behind the recording pass: head training job 46272620 (`train_wgate.sbatch`, EXTRA="--val_frac 0.1
  --early_stop"), then the 430 head arms fg_head job 46272621 and mg_head job 46272622 (dependency afterok).
* 17:30 detached supervisor `eval_pipeline/wgate_pipeline.sh` started on glogin (state `$OUT/wgate_pipeline/`): waits
  for the head arms, generates the dose-matched constants + oracles, runs the four control arms and the 430 table,
  then 8-shard 4292 for fg_head/mg_head/fg_const/mg_const/mg_const0 and the 4292 table.

### 2026-09-21 21:00 first results (430 subset; `$OUT/wgate/table/wgate_430pre.md`, paired vs augfull_lr1e5)
Recording: 4002 train + 66 heldout episodes in 35 min (job 46272493). Training (job 46272620, 92 s): FrameConfHead
heldout NLL 2.17 vs 3.91 for a constant-sigma baseline; log_ref 1.17; PoseSigmaHead log_ref_t 1.05 / log_ref_R 0.83.

| arm | AbsRel | d<1.25 | ATE | RPE trans | RPE rot |
|---|---|---|---|---|---|
| V1 frame gate, FrameConfHead (mean a = .818) | -.0015* | +.0026* | -.0038* | +.0002* | +.029* |
| V2 probe: pose memory frozen (b = 0) | +.0058* | -.0095* | +.0268* | -.0000 | +.090* |
| V2 probe: frozen after frame 8 | +.0066* | -.0111* | +.0306* | -.0001 | +.077* |
| V2 probe: constant b = .5 | +.0003 | -.0006 | +.0005 | +.0000 | +.019* |
| V2 mem gate, PoseSigmaHead (mean b = .602) | +.0006* | -.0002 | +.0018* | +.0000 | +.001 |

Reading so far: (1) the pose memory has LARGE leverage: freezing it costs +35% ATE and +8% rotation, so the earlier
"wash" hypothesis (seed dominated by the decoder) is refuted; the memory is where a good part of pose lives.
(2) Halving its dose (b = .5) costs rotation only; the head-gated version at a similar mean dose (.60) has no rotation
cost but a small ATE cost -- the dose-matched constant (b = .6022) decides whether the head selects. (3) The V1 head
gives ATE -.0038* with a depth gain but a rotation cost of +2.6%; its dose-matched constant (a = .8176, state-only)
and the oracle decide whether that is selection or dose. Head weights sit low on test scenes (mean b .60 vs the
map's design point of 1 for a median frame): sigma on the 430 test scenes is systematically above the TRAIN median,
i.e. a train/test shift in the sigma scale -- noted as a calibration issue for the write-up.
Supervisor: S2 failed on `conda activate` under `set -u` (fixed: env python called directly); controls generated by
hand at 21:00 (V1 const .8176, V2 const .6022, oracles mean a .836 / b .848); supervisor relaunched at stage S3.

### 2026-09-21 21:15 finding: the literal residual is a clock (orchestrator analysis on the 430 traces)
* Recording ran in fp32 (the fix pass made fp32 the recorder default to match the worker), so the bf16/fp32 shift is
  not the cause of the low weights (a precision check with the trained heads is running anyway, job 46283563).
* Heads are well ranked on held-out windows: FrameConfHead heldout Spearman(sigma, r) .71 (NLL 2.23 vs 3.91 const);
  PoseSigmaHead .83 (t) / .76 (R), combined .85. Mean weights on heldout: .81 / .81.
* On the 430 eval scenes the mean weight is .836 (V1) / .636 (V2) and it FALLS with frame index:
  V1 [1,16) .93 -> [256,end) .78; V2 [1,16) 1.00, [32,64) .73, [64,96) .60, [128+) .53 (floor). Eval scenes are
  median 226 frames (p90 610) vs 64-frame training windows.
* Cause: the contract's target is the finetune loss's pose residual = ABSOLUTE pose error relative to frame 0, which
  accumulates: on the heldout table median res_t .039 at t<16 -> .171 at t in [48,64), Spearman(t, res_t) .52,
  (t, res_R) .60. A Kendall-Gal sigma trained on it learns elapsed drift, and the gate becomes "full writes early,
  half writes late". Scene-mean weight still tracks difficulty (Spearman with plain scene ATE: V1 -.42, V2 -.63),
  so the sigma is informative, but the dominant factor is time.
* Consequence for the concept: "confidence of THIS frame's estimate" wants a residual that does not accumulate.
  DEPARTURE 5 (added): a second pair of heads trained on the consecutive-frame RELATIVE pose residual (the RPE-like
  quantity, same normalisation, computed from the same normalised encodings the loss uses) -- arms fg_head_rel /
  mg_head_rel. The literal (absolute) heads stay in the table as the paper-faithful rows.

### 2026-09-21 RELATIVE (consecutive-frame) pose residual as an alternative training target -- implemented end to end, CPU-tested

Motivation: the entry above ("the literal residual is a clock"). Files: `src/CUT3R/src/record_wgate_features.py`,
`src/CUT3R/src/train_wgate_heads.py`, `eval_pipeline/wgate_make_controls.py`, `eval_pipeline/wgate_table.py`,
`eval_pipeline/wgate_selftest.py`, `eval_pipeline/train_wgate.sbatch`, `eval_pipeline/record_wgate.sbatch`,
`eval_pipeline/wgate_probe430.sh` (edits); NEW `eval_pipeline/maks_arm_fg_head_rel.json`, `eval_pipeline/maks_arm_mg_head_rel.json`.
Nothing committed, nothing submitted, no file under `checkpoints/wgate/` touched. The absolute target's behaviour, file names
and formats are unchanged (the trainer selftest reproduces the pre-change abs numbers to the digit: frame heldout NLL 1.933 vs
const 2.136 / Spearman .495, pose 4.652 vs 6.729).

1. Recorder: two new npz arrays `res_rel_t`, `res_rel_R` (float32, (T,), 0 at t = 0). For t >= 1 they are the norms of the
   differences between the GT and predicted CONSECUTIVE-frame relative encodings (t-1 -> t), from
   `dust3r.losses.relative_pose_absT_quatR` (the function `compute_relative_pose_loss` uses) applied to the SAME normalised
   `gt_poses` / `pr_poses` lists of `Regr3DPoseBatchList(L21,'?avg_dis').get_all_pts3d` that the absolute residual reads,
   so the translation scale is the loss's. Written for every episode; a re-recorded npz is rewritten whole, so existing
   arrays are overwritten. NEW `--fill_rel` (sbatch `EXTRA="--fill_rel"`): re-record ONLY the episodes whose npz lacks the
   rel arrays (the current 4002+66 table was recorded before they existed) -- default skip-existing behaviour unchanged.
   The rel arrays cannot be back-filled without the rollout (the normalised encodings are not stored), so the full table
   needs one more 4-GPU recording pass (~35 min last time) before the rel heads can be trained.
   Selftest additions: exact predictions -> rel residual 0; a constant absolute translation offset on ALL frames -> abs
   residual > 0 on every frame (0.0482) while the rel translation residual is 2e-8; an offset on frame 2 only -> abs nonzero
   on [2], rel nonzero on [2, 3] (both hops touching the frame) with equal norm; npz round trip incl. dtype/shape, and the
   `--fill_rel` guard (`npz_lacks`) flags an npz whose rel arrays were dropped. 8.6 s.
2. Trainer: `--target {abs,rel}` (default abs = current behaviour). rel: FrameConfHead r = res_rel_t + res_rel_R + res_pts_self,
   PoseSigmaHead (r_t, r_R) = (res_rel_t, res_rel_R); median standardisation, log_ref = train median of the predicted
   log-variance, wmin .5, NLL / Spearman / calibration all identical. Outputs get the suffix: `frame_conf_head_rel.pth`,
   `pose_sigma_head_rel.pth`, `train_wgate_metrics_rel.json`, `train_wgate_log_rel.txt`; every checkpoint dict and the
   metrics json carry `"target"` (abs files too) plus `residual_keys`. `load_split` now loads every residual array it finds and
   NaN-fills an absent one that is not part of the target (a pre-rel npz under `--target abs` loads as before, with a note),
   but raises `KeyError ... re-record with --fill_rel` when the target needs it. Clock diagnostic recorded for both targets:
   `metrics["clock"]["spearman_t_<array>"]` = Spearman(t, r) on the train table for every residual array, and per head
   `train/heldout["spearman_t_r"]` (frame) / `["spearman_t_r_t", "spearman_t_r_R"]` (pose), also printed. Selftest: the
   synthetic generator emits the rel arrays (own RNG stream, so the abs arrays are bit-identical to before); full run for
   BOTH targets (rel: frame 1.860 vs 2.141 / .505; pose 4.274 vs 6.452 / t .499 R .564), suffixed files present, "target"
   in both checkpoints, abs files untouched by the rel run, res_scale = train median of the rel sum; `clock_diagnostic` exact
   on a handmade table (accumulating -> 1.0, shuffled -> ~0, NaNs ignored) and ~0 on the iid synthetic table; pre-rel npz:
   abs loads, rel raises with the hint. 27 s.
   Real-table check (login node, 3 episodes, pose head, CPU, output to the scratchpad): abs target loads the current
   `checkpoints/wgate/features` with "res_rel_t x3 / res_rel_R x3 NaN-filled" and reports the clock Spearman(t, r_t) .458,
   (t, r_R) .560 -- the finding, reproduced by the new diagnostic; `--target rel` on the same table fails immediately with the
   `--fill_rel` message.
3. Eval plumbing: `maks_arm_fg_head_rel.json` / `maks_arm_mg_head_rel.json` point at the `_rel.pth` heads, wmin .5 (the worker
   attaches them through the same `frame_gate.head` / `mem_gate.head` keys; checkpoint keys identical). `wgate_table.py`:
   six new rows in `ROWS` (+ `NAMES` list) -- `fg_head_rel`, `fg_const_rel`, `fg_oracle_rel` (group "V1 rel", after the V1
   triple) and `mg_head_rel`, `mg_const_rel`, `mg_oracle_rel` (group "V2 rel", after the V2 triple); missing arms are skipped
   as before. `wgate_make_controls.py --suffix S` (default ""): outputs `maks_arm_{fg,mg}_const<S>.json`,
   `maks_arm_{fg,mg}_oracle<S>.json` (+ `.skipped.txt`), `<pilot_dir>/wgate_controls_summary<S>.json`; the head arms default
   to `fg_head<S>` / `mg_head<S>`; `--oracle_residual {auto,abs,rel}` (auto = rel iff S ends in `_rel`). The rel oracle uses
   the same sigma -> weight maps with the GT residual = CONSECUTIVE-frame error from the same GT/pred cameras
   (`gt_pose_errors_rel`): e_trans_rel[t] = ||(C[t]-C[t-1]) - (G[t]-G[t-1])|| on the Sim(3)-aligned predicted centres
   (same umeyama fit as the abs oracle; a wrong global scale or offset cancels), e_rot_rel[t] = geodesic angle between the
   GT and predicted frame-to-frame rotations (= the evaluator's per-pair rpe_rot); term-wise median normalisation as
   before, frame 0 -> 1.0. Default invocation (no suffix) is byte-identical in behaviour (`wgate_pipeline.sh` S2 unaffected).
   `wgate_probe430.sh` is generic (any `maks_arm_<ARM>.json`; the head path is read from the json) -- verified the lookup on
   both rel jsons; usage line `ARMS="fg_head_rel mg_head_rel"` added to its header. `train_wgate.sbatch`: `TARGET=rel`
   passes `--target rel` (validated abs|rel), `EXTRA` still passes through; `record_wgate.sbatch` header documents
   `EXTRA="--fill_rel"` (recording needs no TARGET: both residual pairs are always stored).
4. Tests (login node, env cuteanything, `OMP_NUM_THREADS=1 timeout 280`): `record_wgate_features.py --selftest` OK 8.6 s;
   `train_wgate_heads.py --selftest` OK 27 s (covers both targets); `eval_pipeline/wgate_selftest.py` ALL OK 5 s (generator
   with and without `--suffix _rel` on the fabricated 6-scene pilot incl. fg_head_rel / mg_head_rel traces -> constants .75 /
   .85, rel oracles differ from the abs ones and stay in [.5, 1] with weight 1 at frame 0, sidecars, `_rel` summary with
   `residual: rel`, abs control files unchanged by the rel run; `gt_pose_errors_rel` on a globally 2x-scaled prediction ->
   0, on a single perturbed frame k -> top-2 at {k, k+1} with rotation error 0; worker parse; table with 12 contract rows,
   the six `_rel` rows reported as missing, tex -> pdf -> png).
Not run: any GPU job. Next steps to get the rel rows: `EXTRA="--fill_rel" sbatch eval_pipeline/record_wgate.sbatch` (4 GPU,
re-records all 4068 episodes), `TARGET=rel sbatch eval_pipeline/train_wgate.sbatch`, `ARMS="fg_head_rel mg_head_rel" bash
eval_pipeline/wgate_probe430.sh`, `python eval_pipeline/wgate_make_controls.py --suffix _rel`, then the four `_rel` control
arms and `wgate_table.py`.

### 2026-09-21 21:31 rel-residual chain launched (orchestrator)
Re-record into a SEPARATE root so the abs table stays pristine: WGATE_ROOT=/gpfs/scratch/etur59/koc821022/checkpoints/wgate_rel, job 46284922 (same seed -> same episodes);
train_rel job 46284924 (TARGET=rel, --features /gpfs/scratch/etur59/koc821022/checkpoints/wgate_rel/features, heads written next to the abs ones as *_rel.pth);
430 arms fg_head_rel / mg_head_rel queued behind it (see /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/wgate430.jids). Review of the rel change: ok, minor issues only.
* Precision check (job 46283563; compare step re-run on the login node after exporting CUT3R_DIR, sbatch fixed): bf16 vs fp32
  recordings of the same 4 episodes differ by a median 8-14% of the frame-to-frame spread in frame_feat and 16-43% in
  pose_feat -- large enough that recording in fp32 (what the eval worker runs) was the right choice; `features_prec/precision_compare.json`.

### 2026-09-21 23:30 rel heads trained (job 46284924 on the wgate_rel table, 4000+64 episodes, fp32)
Clock diagnostic on the training table, Spearman(t, residual): abs pose .49 (t) / .57 (R) vs REL .14 / .18 -- the relative
residual is close to time-free, as intended (points .-02). Heldout calibration: FrameConfHead_rel Spearman .77 (NLL 2.56 vs
4.83 const), PoseSigmaHead_rel .66 (t) / .57 (R), combined .67; heldout mean weights .80 / .85; log_ref .913 / .913 / .784.
Arms fg_head_rel / mg_head_rel released on 430 (jobs 46284925 / 46284926).

### 2026-09-21 23:45 controls in (430; `$OUT/wgate/table/wgate_430pre2.md`)
| arm | AbsRel | d<1.25 | ATE | RPE trans | RPE rot |
|---|---|---|---|---|---|
| V1 head (mean a .818) | -.0015* | +.0026* | -.0038* | +.0002* | +.029* |
| V1 dose-matched constant a = .818 (state-only) | -.0012* | +.0021* | -.0039* (287/430) | -.0001 | +.005 |
| V1 ORACLE (GT residual -> same map) | -.0004 | +.0010* | -.0016* | +.0001* | +.044* |
| V2 head (mean b .602) | +.0006* | -.0002 | +.0018* | +.0000 | +.001 |
| V2 dose-matched constant b = .602 | -.0001 | +.0002 | +.0004 | +.0000 | +.005 |
| V2 ORACLE (GT pose residual -> same map) | +.0002 | -.0001 | +.0004 | -.0000 | -.002 |

Reading: BOTH variants, as Kendall-Gal-confidence-keyed dose gates, are closed by their oracles.
* V1: the learned head ties its dose-matched constant on ATE (-.0038 vs -.0039) and adds a rotation cost the constant
  does not have; the ORACLE keyed on the true per-frame residual is WORSE than the constant (ATE -.0016, rot +.044*).
  Keying the state dose to per-frame error, even perfectly, hurts: the "dose, not selection" law holds for the
  state-only gate too, and error-keyed selection damages rotation. The state-only constant .818 is itself a useful
  number: ATE -.0039* with no rotation cost on 430 (the joint alpha_all .853 row gave -.0030*).
* V2: constant, oracle and head are all within noise of plain on every metric except the head's small ATE cost
  (its clock behaviour). The oracle says the ceiling for a per-frame-error-keyed pose-memory dose is ~0 -- the memory
  has leverage (frozen = +35% ATE) but modulating its write by the frame's pose error does not use it.
* What is NOT bounded by these oracles: a gate trained on the CONSEQUENCE of the write (gradient through the memory
  from a causal ATE/RPE loss), since the oracles key on per-frame error, not on write value.

### 2026-09-22 00:35 rel-target arms on 430 (`$OUT/wgate/table/wgate_430rel.md`)
| arm | AbsRel | d<1.25 | ATE | RPE trans | RPE rot |
|---|---|---|---|---|---|
| V1 head, ABS target (mean a .818) | -.0015* | +.0026* | -.0038* | +.0002* | +.029* |
| V1 head, REL target (mean a .810) | -.0016* | +.0027* | -.0042* | +.0002* | +.035* |
| V2 head, ABS target (mean b .602) | +.0006* | -.0002 | +.0018* | +.0000 | +.001 |
| V2 head, REL target (mean b .699) | +.0000 | +.0003 | +.0015* | +.0000 | +.012 |

The clock fix works on the TARGET (Spearman(t, r) .49/.57 -> .14/.18) but only partly on the DEPLOYED weights: V1 rel is
flat in t (.84 -> .78 across the scene, vs .93 -> .78 for abs), V2 rel still decays (.98 -> .60, vs 1.00 -> .53). Residual
cause: even the per-step error grows with elapsed drift on these scenes, and 64-frame training windows never show the
model at t > 64. Outcome unchanged: V1 rel matches the V1 dose-matched constant (-.0042 vs -.0039 ATE) and still pays
rotation (+.035*) that the constant does not (+.005); V2 rel is a small ATE cost like V2 abs. Both variants therefore
land in the same place as their constants, and the V1 ORACLE remains the binding evidence: keying the state dose to
per-frame error is the wrong signal, not just a hard-to-learn one.

## Variant C: the CONSEQUENCE-trained gate (opened 2026-09-22)

The two variants above learn a Kendall-Gal sigma that PREDICTS a frame's own error and use it as a proxy for the value
of that frame's write. Their oracles close that family (V1 oracle is worse than the V1 constant; V2 oracle is flat).
Variant C drops the proxy: the gate weight is trained by its CONSEQUENCE, i.e. the gradient of a causal trajectory
error with respect to the weight, flowing back through the memory and the frozen decoder. Same two write sites:

* **C1 (state write)**: per-frame scalar a_t on the 768-token state commit (state-only, as V1).
* **C2 (pose-memory write)**: per-frame scalar b_t on the retriever-memory commit (as V2).

Head: `GateWeightHead` in `dust3r/wgate/heads.py`, SAME inputs as the corresponding sigma head, but the output IS the
weight, not a log-variance:
* C1: `frame_features` (1796) -> LayerNorm -> Linear(256) -> GELU -> Linear(1) -> u; `a_t = wmin + (1-wmin)*sigmoid(u)`.
* C2: `pose_feat` (768) -> the FROZEN pose-decoder `mlp.fc1` + `act` (3072) -> Linear(1) -> u; same map.
* Init: last layer weight zeroed, bias `--init_logit` (default 2.0) so every frame starts at a_t ~ .94 (near-plain CUT3R)
  and training can only move it. `wmin` default .5 (same floor as every other arm). Frame 0 always writes in full.

Loss (`dust3r/wgate/traj_loss.py`): causal, differentiable, the quantity the benchmark scores.
* `causal_ate(pred_centres, gt_centres, t)`: Sim(3) Umeyama WITH scale on the camera centres 0..t (pre-chunk centres
  enter detached, so the alignment sees the real drift context), then RMSE. torch SVD, fp32, autocast off.
* Chunk loss = mean over t in the chunk with t >= `--ate_min_t` (default 4; below that Sim(3) is degenerate) of ATE_t,
  plus with `--loss ate_rpe` a weighted consecutive-frame translation + rotation term (`--rpe_w`, default 0.1).
* Predicted centres = `pred["camera_pose"][:, :3]` (absT of c2w = camera centre in frame-0 coords); GT centres =
  `camera_to_pose_encoding(inv(views[0]["camera_pose"]) @ views[t]["camera_pose"])[:, :3]`. ATE with Sim(3) is
  scale-invariant, so no norm factor is needed. NO point loss and NO Kendall-Gal term: depth is a measured outcome here.

Training (`src/CUT3R/src/train_wgate_consequence.py`): frozen CUT3R, only the gate head trains; TBPTT over 16-frame
chunks with state AND mem detached at the boundary; optimiser step per chunk; AdamW, `--lr` (1e-4 and 1e-5 arms),
clip 1.0, 4 GPUs x batch 1 via accelerate, NCCL timeout 3 h; held-out eval every `--eval_every` episodes on 16 test
episodes reporting the gate-on causal ATE and the mean weight; checkpoints every `--save_every`.
Model support needed: `attach_frame_gate(..., train=True)` / `attach_mem_gate(..., train=True)` keep `requires_grad`
and leave the head in train mode, and the hooks must NOT detach the weight on that path (features stay detached).
Checkpoint format: `{"state_dict", "mode": "weight", "wmin", "init_logit", "args", "metrics"}` -> the worker's
`attach_*` dispatches on `mode` ("sigma" = the existing log_ref map, "weight" = use the head output directly).

Arms: `cg1_head`, `cg2_head` (+ `_lr5` variants), dose-matched constants from their traces via
`wgate_make_controls.py --suffix`. Rows added to `wgate_table.py`.

DEPARTURE 6: the paper's sigma is gone in variant C; only the gating mechanism and the frozen backbone are shared with
Kendall & Gal. This is the user's design ("we updated this state and got this ATE, so update in a way that minimises
ATE"), kept as a separate labelled family, not as a Kendall-Gal row.

### 2026-09-22 Variant C, Component D (eval plumbing + campaign scripts) -- implemented, CPU-tested, nothing submitted

Files: `eval_pipeline/infer_and_eval_worker.py`, `eval_pipeline/wgate_table.py`, `eval_pipeline/wgate_make_controls.py`,
`eval_pipeline/wgate_selftest.py` (edits); NEW `eval_pipeline/maks_arm_{cg1_head,cg2_head,cg1_head_lr5,cg2_head_lr5}.json`,
NEW `eval_pipeline/wgate_cons_pipeline.sh`. No model / heads / traj_loss / trainer file touched, nothing committed,
no Slurm job submitted.

1. Worker: the existing `frame_gate {head, wmin}` / `mem_gate {head, wmin}` control keys now take a weight-mode
   checkpoint transparently. New module-level `attach_wgate_head(model, kind, ck, wmin=None) -> (head, mode)` and
   `wgate_mode(ck)` ("sigma" when the checkpoint has no "mode"); the `attach_fg` / `attach_mg` closures call it, so the
   per-view keys, the trace dump, the per-scene controls and the oracle schedules are untouched. Dispatch: sigma mode =
   the historical `attach_*(ck["state_dict"], log_ref...)` call; weight mode = the WHOLE dict is passed through
   (`attach_frame_gate(ck, wmin=..., train=False)`), NO log_ref, which is what model.py's weight branch reads
   (`mode` / `init_logit` / `wmin`; the trainer's extra `kind` / `site` keys are ignored there). `train=False` is passed
   only when `inspect.signature(attach_*)` has the parameter (so the worker also runs against a pre-variant-C model.py),
   and the returned head is forced to `.eval()` + `requires_grad_(False)` as a belt-and-braces inference check. A
   weight-mode checkpoint against a model.py without the dispatch raises a RuntimeError naming the fix instead of the
   bare `assert log_ref is not None`. The attach log line now prints `mode=` and, in weight mode, `init_logit=` instead
   of `log_ref=`. Verified against the landed signatures: `attach_frame_gate(self, head_or_state_dict, log_ref=None,
   wmin=None, train=False)`, `attach_mem_gate(..., log_ref_t=None, log_ref_R=None, wmin=None, train=False)`.
2. Arms: `maks_arm_cg1_head.json` / `maks_arm_cg1_head_lr5.json` = `{"*": {"frame_gate": {"head":
   "$CKPT_ROOT/wgate/cg1[_lr5]/gate_head_best.pth", "wmin": 0.5}}}`, `maks_arm_cg2_head[_lr5].json` the same with
   `mem_gate` and `cg2[_lr5]`. NOTE for whoever launches the training: `train_wgate_consequence.py` writes
   `--out/<name>` with `--out` defaulting to `.../checkpoints/wgate/consequence`, i.e. its natural output is
   `.../wgate/consequence/cg1_head/gate_head_best.pth`, NOT the contract path above. Either launch with
   `--out /gpfs/scratch/etur59/koc821022/checkpoints/wgate --name cg1|cg2|cg1_lr5|cg2_lr5`, or leave the defaults:
   the supervisor's `resolve_ckpt` symlinks `consequence/<arm>|<name>/gate_head_best.pth` into the path the arm json
   names (logged) rather than editing the shared arm jsons.
3. `wgate_table.py`: six rows in `ROWS`, group "C", in the order `cg1_head, cg2_head, cg1_head_lr5, cg2_head_lr5,
   cg1_const, cg2_const` (the order the component task lists them in; the constants' labels take their value from the
   arm's `control.json` through the existing `const_of`, which already reads `frame_gate.const` / `mem_gate.const`).
   New `GROUP_HEADINGS = {"C": "Variant C (consequence-trained)"}`: a group in that dict gets a heading row above its
   first row (`| **Variant C (consequence-trained)** | ... |` in the md, `\multicolumn{7}{l}{\textit{...}}` in the tex);
   every other group renders exactly as before (blank row / `\midrule` only). Verified the tex with the heading still
   builds to pdf + png.
4. `wgate_make_controls.py`: `--suffix` already covered the cg arms for the TRACE side (the cg heads are ordinary
   `frame_gate` / `mem_gate` arms, `--fg_arm cg1_head --mg_arm cg2_head --suffix _cg` reads their traces and would write
   `maks_arm_fg_const_cg.json`), but those names do not match the table rows, so DECISION: explicit output names were
   added instead of renaming the table rows -- `--fg_const_name / --mg_const_name / --fg_oracle_name / --mg_oracle_name`
   (default = the old `<kind><suffix>` scheme; the explicit name wins over `--suffix`, which still names the summary and
   the default head arms). The campaign command (what the supervisor runs) is
   `wgate_make_controls.py --only const --fg_arm cg1_head --mg_arm cg2_head --fg_const_name cg1_const
   --mg_const_name cg2_const --suffix _cg` -> `maks_arm_cg1_const.json` (`{"*": {"frame_gate": {"const": c}}}`, the
   state-only path, frame 0 at 1), `maks_arm_cg2_const.json` (`{"*": {"mem_gate": {"const": c}}}`) and
   `<pilot>/wgate_controls_summary_cg.json` (which now also records `out_names`). The V1/V2/_rel behaviour and file names
   are unchanged (asserted in the self-test). The two constants are dose-matched to the lr 1e-4 head arms; the `_lr5`
   arms have no separate constant.
5. `wgate_cons_pipeline.sh`: detached supervisor with the shape of `wgate_pipeline.sh` -- state dir
   `$OUT/wgate_cons_pipeline` (`stage`, `pipeline.log`, `heartbeat`, `<arm>.430.jid`, `<arm>.4292.<shard>.jid`,
   `make_controls.out`), safe re-run, one resubmit per failed eval job. Stages: **W** waits for the training jobs listed
   in `$OUT/wgate_cons.jids`, one per line `NAME JOBID [ARM]` (`#` comments ignored; NAME = cg1 | cg2 | cg1_lr5 |
   cg2_lr5, arm defaults to NAME with `_head` spliced in, an explicit third field wins; SEVERAL lines with the same NAME
   are the resubmits, waited in order, first COMPLETED wins) and a training job that fails is LOGGED AND SKIPPED, not
   fatal; **E430** one `maks_sweep430.sbatch` per arm whose checkpoint resolves (PILOT=wgate430), in parallel;
   **C430** the make_controls command of (4) then the two constant arms; **T430** the table job
   (`--subset $SUBSET430`, default `430cons`, pilot `$OUT/wgate430` -- a fresh name so the earlier `wgate_430*.md`
   tables are not overwritten); **E4292** 8 shards per arm that is scored on 430 (PILOT=wgate4292, shard lists of
   `unc_full4292`), one resubmit per failed shard; **T4292** the 4292 table (`4292cons`). No `conda activate` anywhere
   in the script (the env python is called directly as `$PY`; the table job's `--wrap` still activates, as in
   `wgate_pipeline.sh`, because that runs in a fresh job shell without `set -u`), and no `local a=$1 b="$a"` -- every
   `local` is declared on its own line.
   Launch: `setsid nohup bash eval_pipeline/wgate_cons_pipeline.sh >> $OUT/wgate_cons_pipeline/supervisor.out 2>&1 &`
   after writing `$OUT/wgate_cons.jids`. `WGATE_CONS_LIB=1 source`ing the file defines the functions without running any
   stage (the dry-run hook).
6. Tests (login node, env cuteanything, `OMP_NUM_THREADS=1 timeout 280`, no Slurm):
   * `eval_pipeline/wgate_selftest.py` -> `ALL OK` in 3.3 s, extended with a variant-C section: the four arm jsons
     (gate key, wmin, checkpoint path, presence in `wgate_table.NAMES`, row order + group heading); the C430
     make_controls command on the fabricated pilot (cg1_head trace .65 -> `{"frame_gate": {"const": 0.65}}`, cg2_head
     .88 -> mem_gate, `--only const` writes no oracle, the V1/V2/_rel files are byte-unchanged, `maks_arm_fg_const_cg.json`
     is NOT written); a table run on a fabricated 6-scene C pilot (heading row in md and tex, `C1 dose-matched constant
     a = 0.720` label from `control.json`, rows in group C); and the weight-mode control path on FABRICATED checkpoints
     (`torch.save({"mode": "weight", "state_dict", "wmin", "init_logit"})`) against stub models: whole dict + no log_ref
     + `train=False` in weight mode, no `train` kwarg for a model.py without it, the sigma-mode call unchanged
     (state_dict + log_ref / log_ref_t + log_ref_R), the head left in eval with grads off, the actionable RuntimeError
     on a sigma-only model.py and on a model without `attach_mem_gate`, and `apply_wgate_controls` resolving `wmin` from
     the weight-mode checkpoint when the json omits it. The 20-scene table test's skipped-arm assertion is now derived
     from `wt.NAMES` instead of a hard-coded count.
   * `python -m json.tool` on all four new arm jsons: OK.
   * `bash -n eval_pipeline/wgate_cons_pipeline.sh` OK, plus two dry runs with stub `sbatch` / `squeue` / `sacct` on
     PATH and everything pointed at the scratchpad: (a) helper-level -- `name_to_arm` / `arm_to_name` round trip,
     `arm_head_path`, `live_arms`, `wait_job` (COMPLETED / FAILED / an id neither sacct nor squeue knows -> gives up),
     `run_arm430` (submit + wait, already-scored short circuit, missing json, failure -> exactly two attempts),
     `table_job` (propagates the job's failure, wrap contains the table command), `resolve_ckpt` (both fallback layouts
     symlinked, missing everywhere -> 1); (b) whole stage machine end to end on a fabricated 5-scene pilot with the REAL
     `wgate_make_controls.py`: W (a failed then a COMPLETED job for the same NAME; an all-failed NAME logged and its arm
     skipped) -> E430 -> C430 (wrote `maks_arm_cg1_const.json` .7 / `maks_arm_cg2_const.json` .9 into the sandbox) ->
     T430 -> E4292 (32 shard submissions for 4 arms) -> T4292 -> DONE, and a second pass where every eval job failed:
     two attempts each, logged, non-fatal, still DONE.
   * Bug found and fixed in the dry run: `resolve_ckpt`'s log line went to STDOUT and was captured by
     `arms=$(live_arms)` as a bogus arm name; there is now a `log2` (log file + stderr) for functions whose stdout is
     read back.
   Not tested here: an actual gated inference with a real variant-C checkpoint (needs the trained heads + a GPU job) and
   the 4 GPU eval arms themselves. The V1 / V2 / _rel plumbing is unchanged and its evidence above still stands.
   * Addendum (checked after `eval_pipeline/train_wgate_consequence.sbatch` landed): its defaults are
     `WGATE_OUT=.../checkpoints/wgate/consequence` and `NAME=cg1_head | cg2_head | cg1_head_lr5 | cg2_head_lr5`, i.e.
     `.../wgate/consequence/<arm>/gate_head_best.pth` -- exactly `resolve_ckpt`'s FIRST candidate, so the training jobs
     can be launched with the documented commands unchanged and the supervisor links the checkpoints into the paths the
     arm jsons name (nothing to edit, `$OUT/wgate_cons.jids` is the only file the launcher must write).

### 2026-09-22 01:19 FULL 4292 RESULT (`$OUT/wgate/table/wgate_4292.png`), paired vs augfull_lr1e5
| arm | AbsRel | d<1.25 | ATE | RPE trans | RPE rot |
|---|---|---|---|---|---|
| V1 head (sigma-keyed state dose, mean a .818) | -.0017* | +.0023* | -.0039* | +.0002* | +.017* |
| V1 dose-matched constant a = .818 (state-only) | -.0015* | +.0022* | -.0036* | -.0000* | -.004 |
| V2 head (pose-memory dose, mean b .602) | +.0004* | -.0004* | +.0015* | +.0001* | +.003 |
| V2 dose-matched constant b = .602 | +.0001 | -.0001 | +.0002 | +.0000* | -.003 |
| V2 probe: pose memory frozen | +.0056* | -.0097* | +.0249* | +.0000 | +.076* |
| (ref) frame gate g7ema, joint state+mem | -.0007* | +.0009* | -.0031* | +.0001* | +.006 |
| (ref) aligned token gate, self-view conf (SPATIAL) | -.0007* | +.0011* | -.0053* | +.0001* | -.004 |

VERDICT for the Kendall-Gal write gates at full scale: V1's learned sigma buys ATE -.0039* but pays rotation +.017*,
while its dose-matched constant gets -.0036* with NO rotation cost -- the learned key is not worth its cost, exactly as
the 430 oracle predicted. V2 is a small loss at any dose. The spatial (per-token) gate remains the only member of this
family that beats dose: -.0053* ATE with rotation -.004. Variant C (consequence-trained) is the open question.

### 2026-09-22 Variant C, Component B (differentiable causal trajectory loss) -- implemented, CPU-verified

Files (owned by this component, nothing else touched, nothing committed, no Slurm job submitted):
`src/CUT3R/src/dust3r/wgate/traj_loss.py` (new), `src/CUT3R/src/dust3r/wgate/tests/test_traj_loss.py` (new).
`model.py`, `heads.py`, `train_wgate_consequence.py` and `eval_pipeline/` were NOT edited.

Contract names in place: `umeyama_sim3(src, dst)` -> `(s, R, t)` (torch port of
`eval_bundle/bin/eval_depth_poses.py::_estimate_umeyama`, scale on the camera centres, det-corrected
reflection branch, differentiable through `torch.linalg.svd`); `causal_ate(pred_centres, gt_centres)`;
`chunk_traj_loss(pred_centres_chunk, gt_centres_chunk, prev_pred_centres, prev_gt_centres, ate_min_t=4,
rpe_w=0.0)` -> `(loss, stats)`; `pred_centres(preds)`; `gt_centres(views, ref_idx=0)`. Everything runs in
float32 or better with autocast disabled (`torch.autocast(..., enabled=False)` on the input's device);
float64 inputs are preserved so the tests gradcheck in double. The trainer's shim calls
(`traj.pred_centres(preds)`, `traj.gt_centres(views)`, `traj.causal_ate`, `traj.pred_pose_encodings`,
`traj.gt_pose_encodings`) all resolve against these signatures -- no fallback path in
`train_wgate_consequence.py` is taken.

DEPARTURE 7 (the one that matters) -- `causal_ate` does NOT differentiate through the Umeyama SVD by
default. `ATE(x)^2 = (1/N) min_T sum_i ||T(x_i) - y_i||^2`, i.e. the alignment is the argmin of the very
quantity being differentiated, so by Danskin's envelope theorem the `dT*/dx` term vanishes and fitting
the Sim(3) under `no_grad` on detached inputs gives the SAME gradient. That matters because the SVD
backward carries `1/(sigma_i^2 - sigma_j^2)` terms that blow up exactly in the degenerate case the
contract asks us to survive (2-3 collinear centres = a wrist that has not turned yet, which is the FIRST
thing a causal loss sees at small t). `align_grad="danskin"` is the default; `align_grad="svd"` forces the
literal path; the tests gradcheck BOTH and check they agree (~1e-12) on well-conditioned input, and that
only the default survives collinear centres. `umeyama_sim3` itself is differentiable as contracted.

Other departures / additions, all additive:
1. The optional RPE term is NOT covered by the envelope argument (the Sim(3) minimises the absolute error,
   not the consecutive-frame one), so it differentiates THROUGH the SVD (`rpe_align_grad="svd"`, ~1% of the
   gradient, caught by the central-difference test) with an automatic per-chunk fallback to "danskin" when
   the singular values are degenerate (relative gap < `RPE_DEGEN_TOL=1e-6`); the fallback is reported in
   `stats["rpe_fallback"]`, never silent. RPE translation = `||s R (C_t - C_{t-1}) - (G_t - G_{t-1})||`;
   RPE rotation = geodesic angle of `R_{t-1}^T R_t` from the quaternion part of the 7-d `absT_quaR`
   encodings (alignment-invariant, so no encoding is transformed) -- same definition as
   `wgate_make_controls.py::gt_pose_errors_rel`.
2. CHOICE made where the contract left it open: with `rpe_w > 0` the 7-d encodings are REQUIRED. Pass them
   either inside the centre arguments (last dim 7 -> centres and encodings both taken from them) or as the
   optional `pred_enc_chunk / gt_enc_chunk / prev_pred_enc / prev_gt_enc` arguments; a missing encoding
   raises ValueError rather than silently dropping the rotation term.
3. Extra keyword arguments (all defaulted to the contract behaviour): `causal_ate(..., t=None,
   align_grad, reduce="mean"|"none", eps, ridge)`; `chunk_traj_loss(..., align_grad, rpe_align_grad, eps,
   ridge)`. `chunk_traj_loss` skips frames by GLOBAL index `len(prev) + i < ate_min_t` and returns
   `0 * sum` when no frame qualifies, so `backward()` is always safe. `stats` = plain floats
   `ate, ate_last, n_ate, rpe_trans, rpe_rot, rpe_rot_deg, n_rpe, rpe_fallback, loss`.
4. `gt_pose_encodings / gt_centres` take `ref_pose=` as well as `ref_idx=`: in a chunked rollout pass
   `ref_pose=batch[0]["camera_pose"]` (as the criterion is called with `camera1=...`), otherwise each chunk
   would take its own first frame as reference. ATE is invariant to that choice, the RPE rotation term and
   cross-chunk comparisons are not. GT is detached inside. No `?avg_dis` norm factor is applied anywhere:
   Sim(3) ATE is scale-invariant (contract).
5. Numerical guards, all named constants at the top of the module: `RIDGE=1e-12 * max|cov|` on the
   covariance diagonal (a FORWARD guard only -- it shifts every singular value equally and so does NOT
   separate degenerate ones, which is why departure 7 and not the ridge is what makes the backward finite);
   `VAR_EPS=1e-15` floor on the Umeyama source variance (scale set to 1 below it, as the evaluator does);
   `ATE_EPS=1e-16` clamp on the mean squared error before `sqrt` (an exactly aligned trajectory is a real
   input -- test (a) -- and `d sqrt/dx` is infinite at 0; `clamp_min` passes zero gradient there, the correct
   limit) -> ATE floor 1e-8; `NORM_EPS=1e-20` inside every vector norm.
6. Also exported (used by the tests and available to the trainer): `sim3_align`, `apply_sim3`,
   `quat_geodesic`, `pred_pose_encodings`, `gt_pose_encodings`.

Tests -- `OMP_NUM_THREADS=1 timeout 280 python -m pytest -q dust3r/wgate/tests/test_traj_loss.py` from
`src/CUT3R/src` (env cuteanything, login node): **25 passed in 2.5 s**, no GPU, no Slurm. The contract's
five:
* (a) exactness: for `pred = s*R*gt + t` with 4 random `(s, R, t)`, `causal_ate` and the full `ate + rpe`
  loss are both < 1e-5 (they sit at the 1e-8 ATE floor).
* (b) scale invariance: ate 0.034248754063 for the raw prediction, for `7 * pred`, and for a
  rotated+shifted one -- identical to all printed digits.
* (c) gradcheck: `torch.autograd.gradcheck` on `causal_ate` (8 frames, float64) PASSES for BOTH
  `align_grad="danskin"` and `"svd"`, with an explicit `t` and with a batch dim; plus a central-difference
  check of `d(chunk_traj_loss)/d(one frame's centre)` (ate and ate+rpe), a danskin-vs-svd gradient
  agreement check, and a "backward at exact alignment is finite" check.
* (d) AGREEMENT WITH THE EVALUATOR on `RAIL+80edfcb1+2023-07-14-14h-28m-45s` (128 frames; preds
  `$OUT/cut3r_eval/augfull_lr1e5/preds/<scene>/camera/*.npz` key "pose", GT
  `.../pointworld_droid_splits/test/dl3dv_multi/wrist/<scene>/dense/cam/*.npz` key "pose"):
  CSV `sqrt(mean(ate^2))` = **0.025207676419**, `causal_ate` float64 = **0.025207676419**
  (abs diff **3.47e-18**, per-frame max 2.07e-13), float32 = 0.025207675993 (diff **4.25e-10**). Both are
  far inside the requested ~1e-6. A second test checks the causal prefix is monotone in information.
* (e) degeneracy: 6 cases (2 collinear, 3 collinear exact, 3 collinear with ate > 0, 3 near-collinear, 3
  identical, 1 point) -- forward and backward finite in all of them, directly and through
  `chunk_traj_loss`; a separate test shows the RPE term falls back (fallback=1, loss 0.3034, finite grad) on
  a collinear chunk while a healthy chunk does not (fallback=0), and another asserts the `"svd"` path is
  genuinely the unsafe one (so the test is not vacuous).
Plus: `umeyama_sim3` vs a numpy reimplementation of `_estimate_umeyama` (incl. the reflection branch and a
batched call), the detachment of the pre-chunk history and the `ate_min_t` skip, RPE zero on a perfect
trajectory / weighting / alignment-invariance of the rotation term, `quat_geodesic` vs dust3r's own
quaternion helpers, the rollout helpers against `camera_to_pose_encoding(inv(ref) @ pose)`, dtype/autocast
behaviour, and the error paths.
NOT tested here: a real gated GPU rollout (that is the trainer's smoke, Component A/D territory) and any
4-GPU training. `dust3r.utils.camera` is imported lazily inside `gt_pose_encodings` (circular import with
`dust3r.heads`), so the pure-math part of the module imports with nothing but torch.

### 2026-09-22 Variant C, Component C (the consequence TRAINER) -- implemented, self-test green, nothing submitted

Files owned and touched: `src/CUT3R/src/train_wgate_consequence.py`, `eval_pipeline/train_wgate_consequence.sbatch`
(both NEW, untracked, not committed). `model.py`, `heads.py`, `traj_loss.py` and the eval plumbing were NOT touched --
verified with `git status --porcelain` (only the two files above are mine; the pre-existing ` M model.py` / ` M
infer_and_eval_worker.py` are Components A and D). No Slurm job submitted.

Shape, as the contract specifies it and mirroring `train_conf_gate.py`: frozen
`ARCroco3DStereo.from_pretrained(CKPT_DEFAULT)` with every backbone parameter `requires_grad_(False)`,
`gradient_checkpointing_enable()` + `model.train()` (checkpointing needs train mode; CUT3R has no dropout/BN),
`build_loader(TRAIN_ROOT, 64, 1, workers, accelerator)` with `set_epoch` per pass; per episode `_forward_encoder`
once under `no_grad`, then TBPTT over `--chunk` (16) frames stepping `_forward_decoder_group_step` one frame at a
time with `state_feat` AND `mem` detached at each boundary, grad enabled inside the chunk, `chunk_traj_loss` over
that chunk with the pre-chunk centres passed DETACHED as `prev_pred` / `prev_gt`, `accelerator.backward`,
`clip_grad_norm_(params, 1.0)`, one AdamW step per chunk. Only `camera_pose` is retained from each step's
prediction, so the pts3d/conf subgraph is freed per frame. Accelerate DDP, 4 GPUs x batch 1,
`InitProcessGroupKwargs(timeout=3 h)`. All contract args are present (`--site {state,mem} --wmin .5 --init_logit 2.0
--lr 1e-4 --chunk 16 --episodes 38633 --loss {ate,ate_rpe} --rpe_w .1 --ate_min_t 4 --eval_every 2000
--eval_episodes 16 --save_every 1000 --resume --name --out`), plus the additive ones listed under Departures.
Components A and B are imported by name (`GateWeightHead`; `chunk_traj_loss` / `pred_centres` / `gt_centres`) through
`load_components()`, which fails with a message naming the missing name AND its owning component instead of an
AttributeError hours into a 4-GPU job.

Logging / checkpoints, as contracted: every `--log_every` episodes a line with episode index, running mean chunk
loss, running mean gate weight and elapsed minutes; at `--eval_every` the held-out episodes run with the gate on and
NO gradient and print `[wgcons] EVAL {"tag": "ep<N>", "ate": ..., "loss": ..., "w_mean": ...}` -- same shape as
`train_conf_gate.py`'s `[conf_gate] EVAL {...}`, so `grep 'EVAL {'` works for the supervisor (the record also
carries `n_ep`, `n_chunks`, `seen`, `step`, and is appended to `<out>/<name>/eval.json`). Atomic writes of
`<out>/<name>/gate_head.pth` = exactly `{"state_dict", "mode": "weight", "kind", "site", "wmin", "init_logit",
"args", "episode", "metrics"}` (asserted key-for-key in the self-test), `gate_head_best.pth` at the lowest held-out
ATE, and `gate_head_opt.pth` (optimiser + episode + step + best_ate) for `--resume`. Defaults put a run at
`.../checkpoints/wgate/consequence/<name>/`, i.e. Component D's `resolve_ckpt` FIRST candidate -- nothing to edit.

sbatch: `acc_ehpc`, `--gres=gpu:4`, `-c 80`, `--time=2-00:00:00`, logs to `$OUT/logs/slurm_wgcons_%j.{out,err}`,
env `NAME SITE LR EPISODES CHUNK WMIN INIT_LOGIT LOSS RPE_W WGATE_OUT PRECISION WORKERS EXTRA RESUME SMOKE CKPT`,
`NUMEXPR_MAX_THREADS=64`, `accelerate launch --multi_gpu` when `nvidia-smi -L` shows > 1 GPU else plain python,
every failure path `exit`s non-zero, and it refuses a `/gpfs/projects` output root (CLAUDE.md). It runs the CPU
self-test BEFORE launching, so a missing Component A/B name costs seconds, not a 4-GPU allocation.

**BUG FOUND AND FIXED in this pass (the self-test was RED as delivered; the sbatch would therefore have refused to
launch every training job).** `--selftest` asserted `g > 0` on the gate head's gradient at EVERY one of its 80 toy
optimisation steps. The toy problem is `pred_c = gt_c + (1 - a) * err`, whose optimum is exactly `a == 1`: there
`pred == gt`, the causal ATE is exactly 0.0 and so is its gradient. The loop converged at step 7 (loss .012417 ->
0.0, mean w .9404 -> .9993) and step 8 tripped the assert -- a successful optimisation reported as
`AssertionError: zero gradient reached the gate head through chunk_traj_loss`. Diagnosed, not guessed: with `a` as a
leaf tensor `chunk_traj_loss` gives `grad a = [0, -.0144, -.0602, ...]` (non-zero), `GateWeightHead` backprops
(`out.weight` grad 13.3 on a toy square loss), and a single wired-up step gives head grads `out.weight` .3773 /
`out.bias` .0109 -- so every link of the chain was fine and only the assertion's placement was wrong. FIX (in my
file only): `step()` now returns the gradient magnitude, the `g > 0` assertion fires on the FIRST optimisation step
only -- which is the real "the consequence gradient reaches the head" check, and it is now printed -- the loop
breaks at convergence (`g == 0` or loss <= 1e-8), and a `math.isfinite(g)` assertion is kept on every step so a
NaN gradient is still caught. The final `l1 < .9 * l0` and `w1 > w0` assertions are unchanged.

Tests run (login node, env cuteanything, `OMP_NUM_THREADS=1 timeout 280`, no Slurm):
* `python src/CUT3R/src/train_wgate_consequence.py --selftest` -> `ALL OK`, rc 0, 4.4 s. It checks: Component A/B
  names import; `attach_frame_gate` / `attach_mem_gate` both accept `train=`; `pred_centres` / `gt_centres` shapes
  and the frame-0 convention (`gt_centres == gt_c - gt_c[:, :1]`); init mean weight .940398 ==
  `wmin + (1-wmin)*sigmoid(2.0)` to 1e-5; `--loss ate_rpe` (rpe_w .1) strictly exceeds the pure ATE loss
  (.014409 > .012417); a non-zero head gradient on the first backward (.388214); the toy 1-chunk optimisation falling
  .012417 -> 0.0 in 8 steps with the mean weight rising .9404 -> .9993; and the checkpoint key set + reload
  (`load_state_dict` round trip reproduces the head output to 1e-6).
* `bash -n eval_pipeline/train_wgate_consequence.sbatch` OK; `python -m py_compile` on the trainer OK.
NOT run here: `--smoke` and any real rollout (both need a GPU, and this pass was told not to submit Slurm), so the
gated TBPTT rollout through the real `_forward_decoder_group_step` is still unexercised end to end. Before the 4-GPU
launches, run `SMOKE=1 NAME=cgsmoke SITE=state sbatch -q acc_debug --gres=gpu:1 -c 20 --time=00:40:00
eval_pipeline/train_wgate_consequence.sbatch` (and once with `SITE=mem`) -- the trainer's own `init-weight check`
line on the first episode is the canary that model.py is routing the weight-mode head through the weight branch and
not the sigma map.

Departures from the contract text (all additive, nothing renamed):
1. Extra args, all defaulted to the contract's behaviour: `--ckpt`, `--wd 1e-4`, `--log_every 10`, `--batch_size 1`,
   `--num_views 64`, `--num_workers 8`, `--seed 0`, `--precision {fp32,bf16}` (fp32 default = the eval worker's
   `inference()` path; bf16 autocast is only faster), `--no_ddp_sync` (the codebase's historical no-sync TBPTT
   behaviour, OFF by default).
2. The gate head is deliberately NOT `accelerator.prepare()`d: the model calls the raw submodule, so a DDP wrapper's
   forward would never run and its reducer would never fire (the known no-sync trap in this repo's TBPTT trainers).
   Gradients are averaged explicitly in `sync_grads` via `accelerator.reduce(..., "mean")`; episodes are
   fixed-length so every rank runs the same number of chunks and the all-reduce cannot deadlock (the failure mode
   recorded in the DDP/TBPTT memory note). Only the loaders are prepared.
3. `_flex_call` binds Components A and B by parameter NAME through an alias table and passes keywords, so this file
   is insensitive to their argument order; an unresolvable REQUIRED parameter is a hard `SystemExit` naming it
   (never a silent default), and the resolved binding is printed once into the log. Against the landed signatures it
   binds `chunk_traj_loss(pred_centres_chunk<-pred_c, gt_centres_chunk<-gt_c, ate_min_t, rpe_w, pred_enc_chunk,
   gt_enc_chunk)` and `GateWeightHead(kind, hidden, wmin, init_logit)`.
4. `--loss ate` is implemented as `rpe_w = 0.0` (Component B's `chunk_traj_loss` has no loss-mode argument; 0
   disables the term, which is the same thing).
5. `_set_epoch` calls both `train_unc_gate.set_epoch(loader, e)` and `loader.set_epoch(e)` when present -- the
   accelerate `DataLoaderShard.__iter__` override that Component B's fix pass documented for the recorder.
6. A chunk with no frame at or above `--ate_min_t` (only reachable for chunk 0 when `--chunk <= --ate_min_t`)
   carries no gradient; it is logged and skipped, not an error.

### 2026-09-22 Variant C, Component A (GateWeightHead + trainable gate path in the model) -- implemented, CPU-verified, nothing submitted

Files: `src/CUT3R/src/dust3r/wgate/heads.py` (edit), `src/CUT3R/src/dust3r/model.py` (edit), NEW
`src/CUT3R/src/dust3r/wgate/tests/test_wgate_consequence.py`, NEW `src/CUT3R/src/dust3r/wgate/tests/parity_cpu_consequence.py`.
No loss module, no trainer, no `eval_pipeline/` file touched; nothing committed; no Slurm job submitted.

1. `GateWeightHead(kind, in_dim=None, hidden=256, wmin=0.5, init_logit=2.0, pose_decoder=None)` in `heads.py`, exactly the
   contract's signature. `kind="frame"`: LayerNorm(1796) -> Linear(1796, 256) -> GELU -> Linear(256, 1). `kind="pose"`: the
   model's frozen `PoseDecoder.mlp.fc1` + `act` borrowed in a TUPLE (`self._borrowed`, so `nn.Module` never registers them,
   exactly as `PoseSigmaHead`) -> new `Linear(3072, 1)`; the pose input and the borrowed fc1 weights are detached, so nothing
   flows into the backbone. `forward(x, wmin=None) -> (B,) w = wmin + (1 - wmin) * sigmoid(u)`, split into `logit()` /
   `weight_from_logit()` so the model hook can trace the raw `u`. The last `Linear` is initialised `weight = 0`,
   `bias = init_logit`, so `weight_at_init = wmin + (1 - wmin) * sigmoid(init_logit)` holds EXACTLY for every input (tested to
   0 ulp on random inputs, not to a tolerance). `state_dict()` holds only the trainable layers (frame: norm/fc1/out; pose:
   `out.*` only -- no `fc1.*`). fp32 with autocast disabled throughout, like the two sigma heads. Helpers: `head_cfg()`,
   `checkpoint(args, metrics)` (writes the contract dict `{"state_dict", "mode": "weight", "wmin", "init_logit", "head_cfg"}`),
   `from_state_dict()` (infers the kind from `"norm.weight"`), `load_head_state_dict()` (pose out layer).
   The `forward`/`weight_from_logit` `wmin` override is a DEPARTURE-sized addition the contract did not ask for: the hook passes
   the per-view `frame_gate_wmin` / `mem_gate_wmin` key through, so one trained head can be evaluated at another floor and
   `wmin=1` reproduces plain CUT3R bit-exactly (used as the parity arm below). The head's own `wmin` is the default.
2. `attach_frame_gate(head_or_state_dict, log_ref=None, wmin=None, train=False)` and
   `attach_mem_gate(head_or_state_dict, log_ref_t=None, log_ref_R=None, wmin=None, train=False)` gained `train` and the
   weight-mode dispatch. A `GateWeightHead` instance, or a dict with `"mode": "weight"`, selects weight mode; anything else
   (a bare state_dict, a `FrameConfHead` / `PoseSigmaHead`, a checkpoint with no `"mode"` or `"mode": "sigma"`) takes the
   historical sigma path unchanged, including the `assert log_ref is not None`. In weight mode `log_ref` is forced to None and
   never consulted; `wmin` falls back to the checkpoint's, then the head's, then 0.5. The kind is cross-checked
   (`attach_frame_gate` refuses a pose-kind checkpoint and vice versa, naming the other function). `train=False` keeps today's
   behaviour (`head.eval()`, `requires_grad_(False)`); `train=True` leaves the head in `train()` mode with `requires_grad=True`.
   New state on the model: `frame_gate_mode` / `mem_gate_mode` ("sigma" | "weight") and `frame_gate_train` / `mem_gate_train`.
3. Hooks. `_wgate_frame_pre` / `_wgate_mem_gate` run the head under module-level `_wgate_grad_ctx(train)` =
   `torch.set_grad_enabled(train and torch.is_grad_enabled())` -- so the weight carries gradient when and only when the head was
   attached with `train=True`, and a trainable head left attached under an outer `torch.no_grad()` can never re-enable grad on
   the eval path. The head INPUT stays detached in both modes (`frame_features` is `no_grad`; `pose_feat.detach().float()`).
   `a_0 = b_0 = 1` (frame 0 written in full, no multiplication at all), the `*_const` override and `mem_gate_freeze_after` are
   unchanged and still win over the head, and the trace still records everything -- detached, with the weight-mode tags
   `frame_u` / `mem_u` carrying the raw logit instead of `log sigma^2`. The mem gate's bit-exact fast path
   (`b == 1` for the whole batch -> return `new_mem` untouched) now also requires `not b.requires_grad`, because returning
   `new_mem` there would silently cut the trainer's only path to the head; the state gate has no such fast path (the weight is
   always multiplied into `state_mask` when it is not None), so nothing to guard there.
4. Tests, all on a login node in `cuteanything` with `OMP_NUM_THREADS=1` and `timeout 280`, no Slurm, no GPU:
   * `dust3r/wgate/tests/test_wgate_consequence.py` -- 7/7 pass in ~7 s, fixed seeds, no checkpoint needed: exact init weight
     for random inputs and for several `(wmin, init_logit)`; `state_dict` keys for both kinds (and that the pose kind saves no
     borrowed `fc1`); the weight map, its `wmin` override and the fp32/autocast-off behaviour; the `checkpoint()` dict; a
     synthetic end-to-end gradient test with a differentiable stand-in (`w = head(x); y = w*a + (1-w)*b; loss = y.sum()`) that
     asserts finite non-zero grads on every head parameter; and attach-mode tests on a stub model covering sigma vs weight
     checkpoints, `train=True/False`, the kind cross-check, and the hooks in weight mode (a_0 = 1, const override, freeze_after,
     the grad-carrying blend vs the bit-exact fast path).
   * `dust3r/wgate/tests/test_wgate.py` -- 8/8 still green (the variant-1/2 machinery is untouched).
   * `dust3r/wgate/tests/test_traj_loss.py` -- 25/25 green (Component B's file, run only to confirm heads.py did not break it).
   * NEW `dust3r/wgate/tests/parity_cpu_consequence.py` on the REAL `augfull_lr1e5` checkpoint, 68 s wall: arm H = weight-mode
     heads on BOTH gates at `wmin = 1`, `train=False` -> bit-identical to plain on `camera_pose`, `pts3d_in_self_view`,
     `state_feat` and `mem` for t = 0..2, with the `frame_u` / `mem_u` tags and every traced weight exactly 1.0; arm I =
     both heads `train=True`, wmin .5, loss `||camera_pose||^2` of the LAST frame only -> `a_0 = b_0 = 1`, gated weights
     strictly inside (wmin, 1) at .9439/.9445 (= `weight_at_init`), the last frame's pose carries a graph, and the gradient of
     that later frame reaches EVERY parameter of both heads through the earlier frames' writes (frame head max|g| 1.8e-05 ..
     1.1e-03, mem head 2.2e-06 .. 3.0e-06), while the frozen backbone accumulates none; detaching the heads returns to the
     bit-identical plain path. This is the one claim no synthetic test can make and the whole of Component B rests on it.
   * `dust3r/wgate/tests/parity_cpu.py` (the variant-1/2 parity on the real checkpoint) re-run after the edits: 28/28 `[ok]`,
     133 s -- the sigma arms, the joint-mode arm and the negative controls are unchanged.
   Not tested here: any GPU run, the trainer, and an actual variant-C checkpoint trained by consequence (Component B).

### 2026-09-22 Variant C, Component D -- ADVERSARIAL REVIEW (read-only; no file in the repo was edited)

Reviewer agent. Nothing committed, no Slurm job submitted, no source file changed. Tests run on a login node
(env `cuteanything`, `OMP_NUM_THREADS=1 timeout 280`).

CONFIRMED (tried to refute, failed to):
* Weight mode is really the head output, and nothing else. With a REAL `GateWeightHead` (both kinds, randomised
  away from the zero-init so the map is observable), saved by the REAL `train_wgate_consequence.save_ckpt`, loaded
  by `load_wgate_head` and attached by `attach_wgate_head` onto a stub that binds the REAL `ARCroco3DStereo`
  methods: `_wgate_frame_pre` returns exactly `wmin + (1-wmin)*sigmoid(head.logit(frame_features(...)))`
  (0.5196414 vs 0.5196414), which is NOT what the sigma map would give on the same scalar (1.0), and the per-view
  `frame_gate_wmin` key overrides the floor (0.8 -> 0.8 + 0.2*sigmoid(u)). `_wgate_mem_gate` returns
  `b*new_mem + (1-b)*mem` with the same map on `head.logit(pose_feat)`. Both heads come back with
  `training is False` and every parameter `requires_grad False`, `frame_gate_mode == "weight"`,
  `frame_gate_log_ref is None`. The borrowed pose-decoder `fc1` is NOT frozen or re-cast by the attach
  (checked after attaching the mem head: `pose_head` parameters still `requires_grad`, still fp32) -- the
  tuple-not-registered trick in `GateWeightHead` holds through the worker's `requires_grad_(False)` loop.
  The consequence trainer does not `accelerator.prepare` the head, so the saved `state_dict` has no `module.`
  prefix and `from_state_dict` round-trips (verified through the real `save_ckpt` for both kinds).
* Sigma mode untouched: `eval_pipeline/wgate_selftest.py` -> `ALL OK` (3.4 s), and
  `src/CUT3R/src/dust3r/wgate/tests/test_wgate.py` -> 8/8 `[ok]`. The 20-scene table test's skipped-arm
  expectation is derived from `wgate_table.NAMES` but still pins the cg rows explicitly, so it is not vacuous.
* `python -m json.tool` on the four new arm jsons: OK. `bash -n eval_pipeline/wgate_cons_pipeline.sh`: OK. No
  `conda activate` outside the sbatch `--wrap`, no `local a=$1 b=$a` (every `local` on its own line). The state dir
  `$OUT/wgate_cons_pipeline` does not exist yet and `$OUT/wgate_pipeline/stage` is `DONE`, so nothing is clobbered;
  table subsets `430cons` / `4292cons` do not collide with the existing `wgate_430*` / `wgate_4292*` files.
  `ARMS` matches the four arm jsons that exist; `maks_sweep430.sbatch`'s `CKPT` default is the augfull backbone the
  heads are trained against. Table rows/labels resolve and a missing arm is skipped loudly (stdout + `skipped` in
  the json + a "Rows not shown" footnote in md and tex); a group whose rows are all missing prints no heading.

DEFECTS FOUND (not fixed here -- Component D's agent owns these files):
1. MAJOR -- the four arm jsons name `$CKROOT/{cg1,cg2,cg1_lr5,cg2_lr5}/gate_head_best.pth`, but
   `train_wgate_consequence.py` + `.sbatch` with their documented commands write
   `$CKROOT/consequence/{cg1_head,cg2_head,cg1_head_lr5,cg2_head_lr5}/gate_head_best.pth` (the FILENAME is right,
   the directory is not). So `ARM=cg1_head sbatch eval_pipeline/maks_sweep430.sbatch` -- the normal way to score an
   arm, and literally what the supervisor submits -- fails unless `resolve_ckpt` has run first. Fix: repoint the
   four jsons at the trainer's real output path; `resolve_ckpt` then becomes a harmless fallback.
2. MAJOR -- `resolve_ckpt` symlinks into the absolute path it reads out of the arm json, which no env var
   (`CKROOT` included) can redirect, and it accepts whatever already sits there without checking it
   (`[ -f "$want" ] && return 0`). Two consequences: (a) the documented dry-run hook is NOT sandboxable -- sourcing
   the script with `CKROOT` in the scratchpad and calling `live_arms` created four real dirs with dangling links
   under `/gpfs/scratch/etur59/koc821022/checkpoints/wgate/` (cg1, cg2, cg1_lr5, cg2_lr5; this reviewer removed
   them); (b) a stale link or file left at that path from an earlier run makes the arm score an OLD checkpoint
   silently. Fix 1 removes the need to link; if the link stays, refuse targets outside `$CKROOT` and log
   target + mtime.
3. MINOR -- stage W checks `$CKROOT/$(arm_to_name $arm)/gate_head_best.pth` instead of calling `resolve_ckpt`, so
   with the documented training launch it logs "NO checkpoint at ..." for every arm even when training succeeded.
4. MINOR -- `table_job` drops the `--ref g7ema=... --ref tok_al=...` arguments `wgate_pipeline.sh` passes, so the
   `430cons` / `4292cons` tables lose the two reference rows every earlier wgate table carries.
5. MINOR -- the worker's weight-mode wrapper turns any `AssertionError` from `attach_*` into "does not accept a
   weight-mode checkpoint ... Update model.py". The model's kind guard ("attach_frame_gate got a 'pose'-kind wgate
   checkpoint") is exactly the error a mis-resolved symlink (2) raises, and it is reported as a missing model patch.

### 2026-09-22 Variant C, ADVERSARIAL REVIEW of Components A (GateWeightHead + trainable gate path) and B (traj_loss)

Read-only review; no source file changed, nothing committed, no Slurm job. Every test the two components ship was
re-run on a login node (`OMP_NUM_THREADS=1`, `timeout 280`), plus three independent probe scripts written for this
review (kept in the reviewer's scratchpad, not in the repo).

Re-ran, all green: `test_wgate_consequence.py` 7/7; `test_wgate.py` 8/8 (V1/V2 unregressed); `test_traj_loss.py`
25 passed in 3.0 s; `parity_cpu_consequence.py` 67.5 s -> "VARIANT-C PARITY OK" (wmin=1 weight mode bit-identical
to plain on camera_pose/pts3d_in_self_view/state_feat/mem for t=0..2, and the consequence gradient reaches both
heads through the frozen rollout while the backbone accumulates none); `parity_cpu.py` 131.8 s, 28 [ok] ->
"PARITY OK".

Independently CONFIRMED (probes, not the components' own tests):
* No keys -> unchanged forward (parity_cpu, real augfull_lr1e5 checkpoint).
* Mode isolation both ways: with `sigma_to_weight` / `pose_sigma_to_weight` monkey-patched to raise, a weight-mode
  rollout completes (a_1 = .8216, b_1 = .8422, `*_log_ref is None`, trace tags `frame_u`/`mem_u`); with
  `GateWeightHead.weight_from_logit` patched to raise, a sigma-mode rollout completes and reproduces the legacy
  closed forms (b = .7260532 vs .7260534 recomputed by hand), trace tags `frame`/`mem`.
* Cross-gate checkpoint guards fire in both directions.
* a_0 = b_0 = 1 (frame hook returns None, mem hook returns the identical object); `frame_gate_const` .25 /
  `mem_gate_const` .75 / `mem_gate_freeze_after` 2 all override the weight-mode head exactly.
* train=True inside an outer `no_grad` builds no graph; outside it both weights carry grad and the mem gate's
  `b == 1` fast path is correctly bypassed.
* `causal_ate` is causal: ATE_6 is bit-identical after moving every frame > 6 by +100, and d ATE_6 / d p[7:] == 0.
* Scale invariance to 1.7e-17 (float64); `umeyama_sim3(ridge=0)` matches `eval_depth_poses._estimate_umeyama` to
  <= 1e-15 on both a random and a mirrored (det < 0) configuration.
* Degenerate backward (1 / 2 / 3-collinear / 5-identical points): finite with the default `align_grad="danskin"`,
  NaN with the forced `"svd"` path -- i.e. DEPARTURE 7 is load-bearing, not cosmetic. `chunk_traj_loss` with
  `rpe_w=0.1` on a perfectly collinear chunk is finite and reports `rpe_align_fallback = 1`.

DEFECT FOUND (one, latent): `_rpe_terms` and `_svd_well_conditioned` in `traj_loss.py` run their matmuls OUTSIDE
any `_no_autocast(...)` guard -- `causal_ate` and `umeyama_sim3` each open one, `chunk_traj_loss` does not. Under
`torch.autocast(bf16)`, `dp @ R.transpose(-1, -2)` (traj_loss.py:370) executes in bf16 and `stats["rpe_trans"]`
went from 1.106e-06 to 5.663e-03 on a 12-frame probe (the ATE term and `rpe_rot` are unaffected). This contradicts
the module docstring's "Everything runs in float32 or better with autocast DISABLED". It is currently MASKED:
`train_wgate_consequence.py:393` wraps the whole loss call in `torch.autocast(dev_type, enabled=False)`, and the
eval worker runs fp32, so no launched run is wrong today. Fix: wrap the body of `_rpe_terms` (and of
`_svd_well_conditioned`) in `with _no_autocast(full_p):`, as `causal_ate` already does.

Noted, not defects: (a) `GateWeightHead` with `--init_logit >= ~17` saturates float32 sigmoid to exactly 1.0 and
the gradient is then identically zero (measured: 2.0 -> max|g| 2.1e-01, 12.0 -> 1.2e-05, 17.0 -> 0.0), so such a
run would train nothing -- the trainer's selftest assertion `g > 0` catches it, and the default 2.0 is safe;
(b) `stats` carries `rpe_align_fallback`, not the `rpe_fallback` name the Component B report quotes, and the
trainer only prints chunk stats for the FIRST episode, so the fallback RATE is not actually observable over a
run; (c) the model hook calls `head.logit(...)` on the raw submodule, so a DDP wrapper around the head would
never fire -- Component C already compensates with its explicit `sync_grads` all-reduce.

### 2026-09-22 Variant C, Component C -- ADVERSARIAL REVIEW of the consequence trainer (no code changed)

Reviewer pass over `src/CUT3R/src/train_wgate_consequence.py` + `eval_pipeline/train_wgate_consequence.sbatch`,
cross-checked against `dust3r/wgate/heads.py`, `dust3r/model.py` and `dust3r/wgate/traj_loss.py` as they stand on
disk. NOTHING was edited (no source file, no sbatch); no Slurm job submitted; nothing committed. Only this Run log
entry was appended.

Tests actually run (login node, env `cuteanything`, `OMP_NUM_THREADS=1 timeout 280`, no GPU, no checkpoint):
* `python src/CUT3R/src/train_wgate_consequence.py --selftest` -> rc 0, `ALL OK` (init mean weight 0.940398,
  first backward |grad| 0.388214, toy loss .012417 -> 0.0 in 8 steps, checkpoint keys + reload OK). The
  implementer's assertion-placement fix is confirmed green.
* NEW reviewer probe (scratchpad only, not added to the repo): `run_episode` driven against a STUB model that
  reproduces the real contract exactly where it matters -- `_forward_encoder` / `_forward_decoder_group_step`
  signatures, the pose read BEFORE the commit (so `w_t` can only affect frames > t), `state = a*new + (1-a)*old`,
  `mem = b*new + (1-b)*mem`, `a_0 = b_0 = 1`, detached head input, detached trace lists. Nothing of the trainer is
  stubbed. Results, 24 frames / chunk 8:
  - `--site state`: 3 chunks, 3 optimiser steps, `max|grad|` 9.27e-04 on the head, 0/6 parameters with a zero
    gradient, parameters moved 3.0e-02. `--site mem`: `max|grad|` 3.44e-04, 0/2 zero. `--loss ate_rpe`:
    `max|grad|` 7.4e-03. **Claim (1) holds: the chunk loss really does reach the head through the memory.**
  - three consecutive episodes: live `torch.Tensor` count constant at 81, Python heap flat -> **claim (2) holds**;
    no "backward through a freed graph" in any configuration. Every cross-chunk carrier is detached
    (`state_feat`, `mem`, `all_pred_c`, `all_pred_pose`) and `model._wgate_trace*` stores Python floats.
  - eval path (`train=False`): 0 optimiser steps, every head parameter still has `p.grad is None`, 23 gate
    weights traced, full-episode ATE returned -> **claim (5) holds** (gate ON, no gradient, `final_ate` =
    `causal_ate(..., t=T-1)` = the benchmark quantity).
* NEW reviewer probe of the CHECKPOINT HANDOFF: `save_ckpt` -> `torch.load` -> `ARCroco3DStereo.attach_frame_gate`
  / `attach_mem_gate` (bound onto a minimal carrier, as Component A's own tests do) for BOTH kinds. Both land in
  `mode="weight"` with the right `kind`, `wmin` 0.5 and bit-identical head output (atol 1e-6); the kind
  cross-check refuses the wrong gate in both directions; `train=True` gives `requires_grad=True` and
  `training=True`. `save_ckpt` writes no `head_cfg`, which is fine because `GateWeightHead.from_state_dict`
  reads the sizes off the state dict. **Claim (4) holds** (`--site mem` -> `kind="pose"` built from
  `model.downstream_head.pose_head`, attached to the MEM gate).

Claims verified by reading, with the code cited:
* (3) frozen backbone: the `requires_grad_(False)` sweep runs BEFORE `build_head`/`attach_head`, so the head
  (registered as `model.frame_gate` / `model.mem_gate`) is untouched by it; `opt` is built from
  `head.parameters()` only; the pose kind's `fc1`/`act` live in a tuple and its weights are `.detach()`ed inside
  `logit()`. Gradient checkpointing cannot cut the head's gradient: `ARCroco3DStereo.__init__` sets
  `fixed_input_length = True`, so `_decoder` passes `use_reentrant=not self.fixed_input_length` = **False**
  (non-reentrant checkpointing propagates grad correctly even when the checkpointed inputs do not require it),
  and the gate is applied OUTSIDE the checkpointed blocks anyway.
* (2) also: episodes are fixed length -- `build_loader` passes `num_views=64` (an int) with
  `allow_repeat=False` and `fixed_length=True`, and `EasyDataset.make_sampler` then sets
  `min_view_size = max_view_size = num_views`. Every rank therefore runs exactly `ceil(64/chunk)` chunks, so the
  explicit `sync_grads` all-reduce cannot deadlock (the "random view counts" trap of the DDP/TBPTT memory note
  does not apply here). `accelerate.Accelerator.reduce` is an ALL-reduce ("All processes get the reduced value"),
  so the ranks stay in step.
* (6) resume restores head state dict, optimiser state, `episode`, `step` and `best_ate`; the in-place
  `load_state_dict` keeps the optimiser's parameter references valid.
* (7) sbatch: `--qos=acc_ehpc`, `--gres=gpu:4`, `--time=2-00:00:00`, `-c 80`, logs under
  `$OUT/logs/slurm_wgcons_%j.{out,err}` (the directory exists); `NAME`/`SITE`/`LR` all reach the ARGS array;
  every failure path exits non-zero (conda, mn5_paths, SITE, NAME, `/gpfs/projects`, LOSS, PRECISION,
  `mn5_require`, mkdir, no GPU, `--selftest`, train); the default `CKPT`
  (`$CKPT_ROOT` = `/gpfs/scratch/etur59/koc821022/checkpoints_projects`) resolves to an existing 3.1 GB file.

DEFECTS FOUND (all minor; none blocks the launches, nothing was changed -- Component C's owner should decide):
1. A chunk with no frame at or above `--ate_min_t` is NOT skipped, contrary to the implementer's departure 6.
   `chunk_traj_loss` returns `p_chunk.sum() * 0.0`, so `loss.requires_grad` is True and the trainer runs
   `zero_grad / backward / clip / opt.step()` with an all-zero gradient -- and AdamW's decoupled weight decay
   still shrinks the head (`p *= 1 - lr*wd`). Measured: `--chunk 4 --ate_min_t 4` gives `opt_steps=6` for 6
   chunks with chunk 0's loss `-0.00000`. Unreachable at the defaults (chunk 16 > ate_min_t 4). Fix: gate the
   step on `st.get("n_ate", 1) > 0`.
2. `run_w` / `run_l` grow without bound (63 gate weights per episode x 38633 episodes ~ 2.4M floats,
   ~100-150 MB of host RAM) although only `run_w[-256:]` / `run_l[-40:]` are ever read. Fix: truncate after
   each extend.
3. The save / eval cadence is an exact modulo on a counter that advances by `batch_size * n_proc`. 1000 % 4 == 0
   and 2000 % 4 == 0, so the documented 4-GPU launches are fine, but a 3-GPU relaunch (or `--batch_size 3`)
   would silently never checkpoint or evaluate for the whole 2-day job. Fix: compare against a `last_saved`
   counter (`seen - last >= save_every`).
4. `run_eval` averages per-rank MEANS (`g[:, 2].mean()`) and a rank with no eval episode contributes a literal
   0.0 (`sum(ates)/max(len(ates),1)`), which would both print a wrong held-out ATE and write
   `gate_head_best.pth` from it. Needs a test split with >= 16 episodes to be safe. Fix: gather sums and counts
   and divide after the gather.
5. `--resume` restores the episode INDEX but not the data position: `epoch = args.seed + (1 if seen > 0 else 0)`
   restarts epoch 1 from its beginning, so a job resumed after the 2-day wall re-trains on episodes it has
   already seen and never reaches the rest of the first pass. Fix: derive the epoch from
   `seen // len(loader)` and skip `seen % len(loader)` batches.
6. Nothing asserts that the head is identical on all ranks. It is today (same seed, same deterministic
   construction order), but `sync_grads` only keeps ranks together if they START together. Fix: broadcast
   `head.state_dict()` after `build_head` and after `--resume`.
7. `_wrap_traj` falls back SILENTLY: `final_ate` swallows any exception from `causal_ate` and uses the local
   `_sim3_rmse`, and `pred_centres_of` / `gt_centres_of` fall back to local definitions. Against the landed
   Component B every one of these paths is dead (verified: `pred_centres`, `gt_centres`, `causal_ate`,
   `pred_pose_encodings`, `gt_pose_encodings` all bind), but a future signature change would quietly alter the
   reported EVAL metric instead of failing the way `load_components` does.

NOT covered here (unchanged from the implementer's report): the gated TBPTT rollout through the REAL
`_forward_decoder_group_step` on a GPU. The stub probe above and Component A's `parity_cpu_consequence.py`
(real checkpoint, gradient from a later frame reaching every head parameter through earlier writes) together
cover the mechanism, but the GPU smoke (`SMOKE=1 NAME=cgsmoke SITE=state` and once with `SITE=mem`) is still
the thing that must run before the 4-GPU launches, and the `init-weight check` line remains the canary.

### 2026-09-22 Variant C -- INTEGRATION SMOKE TEST (CPU suites + 3 acc_debug GPU jobs)

Integration agent. Nothing committed. No source file of Components A/B/C/D was edited. The only repo files this
pass created are four new smoke arm jsons (`eval_pipeline/maks_arm_cgsmoke_{ident1,ident2,cg1,cg2}.json`); every
other artefact lives under `/gpfs/scratch/etur59/koc821022/checkpoints/wgate/`.

**VERDICT: variant C is wired end to end. The consequence gradient is real on a GPU with the real decoder, the
trainer produces `mode="weight"` checkpoints, and the eval worker loads them and reproduces plain CUT3R
BIT-EXACTLY when the head's weight is 1.**

**(1) CPU suites, login node, `OMP_NUM_THREADS=1 timeout 280`, all green.**
* `python -m pytest -q dust3r/wgate/tests/` (from `src/CUT3R/src`) -> **40 passed in 4.01 s** (25 traj_loss,
  8 test_wgate, 7 test_wgate_consequence; the two `parity_cpu*.py` scripts are not `test_*` so pytest does not
  collect them -- run separately below).
* `python eval_pipeline/wgate_selftest.py` -> `wgate_selftest: ALL OK` (3.4 s), including the line
  `variant C OK (4 arm jsons, table rows + heading, weight/sigma attach dispatch, eval mode, control path)`.
* `python src/CUT3R/src/train_wgate_consequence.py --selftest` -> rc 0, `ALL OK`; init mean weight 0.940398,
  first backward `|grad| 0.388214`, toy loss 0.012417 -> 0.000000 in 8 steps, checkpoint keys + reload OK.
* `dust3r/wgate/tests/parity_cpu.py` -> 28 `[ok]`, 134.4 s, `PARITY OK: keys absent / present-with-weight-1 are
  bit-identical; negative controls behave as derived`.
* `dust3r/wgate/tests/parity_cpu_consequence.py` -> 21 `[ok]`, 66.6 s, `VARIANT-C PARITY OK: wmin=1 weight-mode
  is bit-identical to plain; the consequence gradient reaches both heads through the frozen rollout`.

**(2) Evaluator agreement, the actual number.** `pytest -q -s -rA dust3r/wgate/tests/test_traj_loss.py -k eval`:
on `RAIL+80edfcb1+2023-07-14-14h-28m-45s`, 128 frames --
`CSV RMSE(ate) 0.025207676419 | causal_ate f64 0.025207676419 (diff 3.47e-18, per-frame max 2.07e-13) |
f32 0.025207675993 (diff 4.25e-10)`. Both far inside the ~1e-6 the task asked for. The companion test also
reports `umeyama_sim3(ridge=0) == the evaluator's _estimate_umeyama (sim3 + se3) to < 1e-14`.

**(3) GPU job A -- the trainer smoke, job `46322219` (acc_debug, 1 GPU, 20 cpus, node as01r2b14, 00:01:20).**
`--smoke --site state --out .../checkpoints/wgate --name smoke_cg1` then the same with `--site mem --name
smoke_cg2`. Both rc 0. Canary lines, both sites identical:
`init-weight check: 15 gate calls, mean w 0.9404 vs wmin+(1-wmin)*sigmoid(2.0) = 0.9404 -> OK`, i.e. model.py
routes the weight-mode head through the WEIGHT branch and not the sigma map on a real GPU rollout.
state: `first episode: 1 chunks, losses ['0.00470'], ate 0.00516, opt steps 1`, chunk stats
`n_ate 12, t_first 0, t_last 15`; EVAL init ate 0.00530987 -> final 0.00531153, w_mean 0.94040 -> 0.93998.
mem: `losses ['0.00482'], ate 0.00537, opt steps 1`; EVAL init 0.00538060 -> final 0.00538103,
w_mean 0.94040 -> 0.94105. (Two episodes cannot move the held-out metric; the point is that it runs.)
Checkpoint inspection, all four files: `mode='weight'` as contracted, plus
`smoke_cg1/gate_head.pth kind='frame' site='state' wmin=0.5 init_logit=2.0 episode=2
sd_keys=['fc1.bias','fc1.weight','norm.bias','norm.weight','out.bias','out.weight']` and
`smoke_cg2/gate_head.pth kind='pose' site='mem' ... sd_keys=['out.bias','out.weight']` (the pose kind stores
only its new out layer, exactly as designed). NOTE: in a 2-episode smoke `gate_head_best.pth` is the INIT head
(`episode=0`, `|out.weight| = 0`, so w is the constant 0.9404) because the best held-out ATE was the init eval.
`gate_head.pth` has moved off the zero-init: `|out.weight|` 2.70e-03 (cg1) / 5.34e-03 (cg2).

**(4) GPU job A2 -- THE CRUX, job `46323416` (acc_debug, 1 GPU, node as01r2b14, 00:01:02).** Driver
`.../wgate/cons_smoke/crux_gpu.py`: builds the model / head / attach / traj shim through
`train_wgate_consequence`'s own functions, takes the FIRST episode of the train loader, and runs SIX optimiser
steps on the SAME first chunk (frames 0..15), restoring the post-encoder state each time. `--lr 1e-3`
(the arms use 1e-4; six steps at 1e-4 move nothing visible). Both sites rc 0:

* `VERDICT site=state: grad_norm_first_chunk 1.243632e-05 (NON-ZERO OK); loss 0.00122004 -> 0.00121973
  (DECREASED OK); mean gate weight 0.940399 -> 0.938464 (delta -0.001935)`
* `VERDICT site=mem: grad_norm_first_chunk 1.261137e-06 (NON-ZERO OK); loss 0.00122280 -> 0.00122238
  (DECREASED OK); mean gate weight 0.940399 -> 0.911933 (delta -0.028465)`

`backbone_params_with_grad 0` at every step of both arms -- the frozen backbone accumulates nothing. Per-step
weights spread away from the constant (state step 5 `w[min,max] [0.937463,0.942184]`, mem step 5
`[0.911398,0.912908]`), so the head is genuinely reading its input and not just moving its bias.
HONEST READING of the loss column: the decrease is real but TINY and NOT monotone (state
0.00122004, 0.00122028, 0.00121989, 0.00121994, 0.00121978, 0.00121973 -- steps 1 and 3 go up). On this chunk the
causal ATE is almost flat in the gate weight at this dose; the load-bearing claim of this step is the non-zero
gradient and the moving weights, not the size of the drop.
One sub-result worth recording: at step 0 the FRAME head's `norm.*` / `fc1.*` gradients are EXACTLY 0
(`per-param max|g|: {'norm.weight': 0.0, ..., 'out.weight': 3.14e-06, 'out.bias': 1.72e-06}`, `nonzero_grad_params
2/6`), and from step 1 all six are non-zero (`{'norm.weight': 1.53e-08, ..., 'out.weight': 2.64e-06}`). That is
the contract's zero-init (`du/dh = out.weight = 0` at init), not a cut path -- but it means the MLP body of the
frame head only starts learning after the out layer has moved, which is worth knowing for the lr-1e-5 arm.

**(5) GPU job B -- the eval worker on one scene, job `46324664` (acc_debug, 1 GPU, node as01r2b14, 00:01:14).**
`PILOT=wgate_cons_smoke`, scene `RAIL+80edfcb1+2023-07-14-14h-28m-45s` (128 frames), backbone = the augfull
`cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth`, four arms, all rc 0, all scored 1/1.
Identity heads (written for this test, `init_logit = 40` so `sigmoid(u) == 1.0` in fp32 and
`w = 0.5 + 0.5*1.0 == 1.0` EXACTLY for every input, last Linear zeroed) live in
`.../wgate/smoke_identity/identity_{frame,pose}.pth`. Against
`$OUT/augfull_lr1e5/eval/<scene>/eval_depth_pose_metrics.csv` (130 rows), max |delta| per column:

| arm | gate | absrel | a1 | ate | rpe_trans | rpe_rot | trace w[min,max] | mean w | t=0 |
|---|---|---|---|---|---|---|---|---|---|
| `cgsmoke_ident1` | frame, w==1 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | [1.000000, 1.000000] | 1.000000 | `[0, 1.0]` |
| `cgsmoke_ident2` | mem, w==1 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | [1.000000, 1.000000] | 1.000000 | `[0, 1.0]` |
| `cgsmoke_cg1` | frame, smoke head | 1.693e-02 | 1.769e-02 | 1.685e-02 | 3.343e-03 | 6.267e-01 | [0.939914, 1.000000] | 0.940434 | `[0, 1.0]` |
| `cgsmoke_cg2` | mem, smoke head | 7.322e-03 | 1.317e-02 | 6.655e-03 | 3.615e-03 | 7.331e-01 | [0.940405, 1.000000] | 0.941093 | `[0, 1.0]` |

The two identity rows are BIT-IDENTICAL (0.000e+00, not merely ~1e-6) on all five metrics and both index
columns: the whole weight-mode inference path -- worker dispatch, `attach_wgate_head`, model.py's weight branch,
both hooks -- is a provable no-op at w = 1. All four arms wrote `wgate_trace.json` with 128 rows, every weight
inside `[wmin, 1] = [0.5, 1]`, and `t = 0` at exactly 1.0 (a_0 = b_0 = 1 confirmed on the real eval path). The
worker's attach log prints the new fields: `frame gate head attached from ... (mode=weight, init_logit=40.0,
wmin_train=0.5)`. The two smoke-head rows differ from the baseline as they must (they gate at ~0.94).

**Independent confirmation of the reviewer's Component D issue 1** (not fixed here; Component D owns the files):
all four production arm jsons name `.../wgate/{cg1,cg2,cg1_lr5,cg2_lr5}/gate_head_best.pth` and all four paths
are MISSING on disk, while `train_wgate_consequence.py`'s `OUT_ROOT` is `.../wgate/consequence`. The smoke arms
this pass added sidestep it by naming the real files directly.

DEPARTURES from the task text, all deliberate:
1. **`/scratch/tmp` (the agent scratchpad) is NODE-LOCAL NVMe** (`/dev/nvme0n1p2 on /scratch type ext4`), so a
   compute node cannot see it. The first submission (`46321750`) had its sbatch `--output` and its driver script
   there; it was cancelled after 29 s and everything was re-staged under
   `/gpfs/scratch/etur59/koc821022/checkpoints/wgate/cons_smoke/` (scripts + `logs/`). Anyone writing an sbatch
   for this project must keep every path the job touches on GPFS.
2. GPU job A was SPLIT into two acc_debug jobs. `46322219` ran the smokes fine but both crux steps exited rc 2:
   MY driver cleared `sys.argv` before importing the trainer module and never restored it, so its own
   `parse_args` saw no `--site`. Fixed in `crux_gpu.py` and re-run as `46323416`; the smoke half of `46322219`
   is unaffected and is the evidence quoted in (3).
3. The crux is a separate driver, not `--smoke`. The trainer takes exactly one optimiser step per chunk and
   never revisits a chunk, so "loss before and after a few steps on the SAME chunk" cannot come out of it.
   The driver calls the trainer's own `build_head` / `attach_head` / `_wrap_traj` / `_flex_call` /
   `gate_weights`, so it tests the shipped code, not a re-implementation.
4. Job A asked for `--time=01:20:00` (acc_debug allows 2 h) rather than the 40 min in the task, to leave room for
   four model loads; A2 and B used 40 min. Jobs were run strictly one at a time (acc_debug allows one per user);
   queue waits were 7-14 min each.
5. `maks_arm_cgsmoke_cg1.json` / `..._cg2.json` point at `gate_head.pth`, NOT `gate_head_best.pth`: in a
   2-episode smoke `best` is the init head with a zero out layer, whose weight is the constant 0.9404 for every
   frame, which would not exercise a per-frame-varying head.
6. New repo files: the four `eval_pipeline/maks_arm_cgsmoke_*.json`. No existing arm json, table, pipeline,
   trainer, model, heads or traj_loss file was modified.

NOT covered by this pass: any multi-GPU run, any trained-to-convergence variant-C head, the 430/4292 arms, and
the reviewers' open defects (traj_loss's RPE autocast guard, the trainer's save/eval cadence and per-rank eval
average, Component D's arm-json checkpoint paths and `resolve_ckpt` symlink behaviour) -- none of which blocked
any step above, and none of which this pass touched.

### 2026-09-22 Variant C, Component D -- FIX PASS (arm checkpoint paths + resolve_ckpt fencing)

Scope: the two `major` findings against the eval group (arm jsons + `wgate_cons_pipeline.sh`). Files touched:
`eval_pipeline/maks_arm_{cg1_head,cg2_head,cg1_head_lr5,cg2_head_lr5}.json`, `eval_pipeline/wgate_cons_pipeline.sh`,
`eval_pipeline/wgate_selftest.py`. No model / heads / traj_loss / trainer file touched, no Slurm job submitted,
nothing committed.

1. **Arm jsons repointed at the trainer's real output (blocker for standalone scoring).** All four now name
   `/gpfs/scratch/etur59/koc821022/checkpoints/wgate/consequence/<ARM>/gate_head_best.pth` with
   ARM = `cg1_head, cg2_head, cg1_head_lr5, cg2_head_lr5` -- exactly `train_wgate_consequence.py`'s
   `OUT_ROOT` (`--out` default) joined with the `--name` the sbatch header documents. The old
   `.../wgate/{cg1,cg2,cg1_lr5,cg2_lr5}/gate_head_best.pth` was a directory nothing writes, so
   `ARM=cg1_head sbatch eval_pipeline/maks_sweep430.sbatch` could only work after the supervisor had
   symlinked. It now works standalone and `resolve_ckpt` is a pure verify+log step in the normal case.
   DECISION recorded: the previous entry's "do not edit the shared arm jsons, symlink instead" is REVERSED --
   editing four json literals is cheaper and safer than a symlink into the production checkpoint tree, and the
   launcher no longer has a choice of `--out`/`--name` to get wrong (leave both at their defaults).
2. **`resolve_ckpt` fenced and made loud.** New `ck_desc()` (size + mtime, and the link target when the path is a
   symlink) and `under_ckroot()`. Behaviour now: (a) if the json's head path is NOT under `$CKROOT` the function
   never creates anything there -- it accepts an existing file with a loud `OUTSIDE CKROOT` line, otherwise
   refuses; this is what stops a dry run with `CKROOT=<tmp>` from planting directories + dangling links in
   `/gpfs/scratch/.../checkpoints/wgate/` (the exact damage the review reproduced); (b) an existing path is
   logged with size/mtime EVERY time, short-circuit included, so the log always records which file an arm scored;
   (c) a symlink whose target is outside `$CKROOT`, or that dangles, is REMOVED and re-resolved instead of being
   accepted -- a stale link from an earlier/smoke run can no longer silently feed the wrong checkpoint to an arm;
   (d) a candidate equal to `$want` is skipped (no self-link), a fourth fallback layout `$CKROOT/<name>/` was
   added, and a failed resolve now logs why. Stage W's status line now reports the arm json's own head path
   (`arm_head_path`) instead of the old `$CKROOT/<name>/gate_head_best.pth` guess, and the header comment
   documents the real layout and the `NAME`/arm identity.
3. **Selftest guard against the regression.** `test_variant_c` expects the new `consequence/<arm>` paths and, new,
   cross-checks them against the sources of truth: it regexes `OUT_ROOT` out of `train_wgate_consequence.py` and
   the `WGATE_OUT` default out of `train_wgate_consequence.sbatch`, asserts the two agree, and asserts each arm
   json's head is `<OUT_ROOT>/<arm>/gate_head_best.pth`. If another agent moves the trainer's output root the
   selftest now fails instead of the 430 sweep.

Tests re-run on a login node (`OMP_NUM_THREADS=1`, `timeout 280`), nothing submitted:
* `python eval_pipeline/wgate_selftest.py` -> rc 0, `wgate_selftest: ALL OK`, including the new line
  `variant-C arm heads agree with the trainer output root /gpfs/scratch/etur59/koc821022/checkpoints/wgate/consequence/<NAME>`.
* `python -m pytest -q dust3r/wgate/tests/` -> `41 passed in 4.07s`. HONEST NOTE: an earlier run of the same
  command at 09:28 reported `1 failed, 40 passed` on Component B's brand-new
  `test_autocast_is_inert_on_the_whole_chunk_loss` (gradient `max|dg| 1.4e-4` between the autocast and non-autocast
  runs; loss and stats already equal) while `traj_loss.py`/`test_traj_loss.py` were being written by that group
  seconds earlier. It has passed 6/6 since (5 isolated `-k autocast` runs + the full suite). Not my file, not
  touched, flagged here for Component B rather than silently dropped.
* `resolve_ckpt` dry run in a sandbox (`WGATE_CONS_LIB=1`, `WT`/`OUT`/`CKROOT` all under the scratchpad), 5 cases:
  (A) a production arm json whose head is outside the sandbox `$CKROOT` -> `live_arms` empty, rc 1, loud refusal,
  and `md5` of `ls -a /gpfs/scratch/.../checkpoints/wgate` UNCHANGED, no `cg1/cg2/cg1_lr5/cg2_lr5` directories
  created (the review's failure, now impossible); (B) head present -> rc 0 with `size=3 mtime=...`;
  (C) head missing, `$CKROOT/<arm>/` fallback present -> link created and logged with its target + size/mtime;
  (D) stale symlink to a file outside `$CKROOT` -> `replacing symlink ...` then relinked to the in-root fallback
  (content `fallback`, not `stale`); (E) dangling symlink, no fallback -> link removed, rc 1, `no checkpoint at ...`.
  Production tree md5 unchanged at the end.

Not fixed here (other groups own the files): Component B's RPE autocast guard (now apparently landed), the
trainer's save/eval cadence, per-rank eval average, resume position and rank broadcast.

### 2026-09-22 Variant C, model+loss FIX PASS (traj_loss autocast guard) -- CPU-tested, nothing submitted

Scope: the one blocker/major raised against the **model+loss** group (`dust3r/wgate/heads.py`,
`dust3r/model.py`, `dust3r/wgate/traj_loss.py`, `dust3r/wgate/tests/`).  No Slurm job was submitted and
nothing was committed.

**The defect (reproduced first, then fixed).** `traj_loss.py`'s module docstring advertises "Everything
runs in float32 or better with autocast DISABLED", but only `umeyama_sim3` and `causal_ate` opened
`_no_autocast`.  `_rpe_terms` and `_svd_well_conditioned` did not, so under `torch.autocast(bfloat16)`
the aligned consecutive-frame translation `dp @ R^T` ran in bf16.  Reproduced on a 12-frame chunk
(login node, cuteanything, `OMP_NUM_THREADS=1`), `rpe_w=0.1`:

```
             plain                   autocast(bf16)          same
ate          0.026397915557026863    0.026397915557026863    True
rpe_trans    0.03750526160001755     0.03749231994152069     False
rpe_rot      2.219606637954712       2.219606637954712       True
loss         0.2521091103553772      0.2521078288555145      False
apply_sim3   float32                 float32                 maxdiff 3.24e-03
```

The error is isolated to the aligned-translation matmul, exactly as the review said.  It was MASKED in
every launchable path (`train_wgate_consequence.py:393` wraps the loss call in
`torch.autocast(dev_type, enabled=False)`; the eval worker uses no autocast), so no run ever produced a
wrong number -- the defect was that the module advertised a guard that lived in another component.

**The fix** (`dust3r/wgate/traj_loss.py`, three edits): the guard is now the module's own, at every
autocast-eligible op rather than at one outer wrap, so any future entry point is covered too.
  * `_rpe_terms`: whole body under `with _no_autocast(full_p):` (the `sim3_align` call, the dp/dg
    differences, `dp_al`, `_safe_norm` and the quaternion terms).  The early `lo` return stays outside;
    it contains no matmul.
  * `_svd_well_conditioned`: `with torch.no_grad(), _no_autocast(src):`, i.e. alongside the existing
    `no_grad`, covering its `dc^T sc` covariance and `svdvals`.
  * `apply_sim3`: guarded too -- it is exported in `__all__` and its `x @ R^T` moved an aligned centre
    by 3.2e-3 under autocast when called directly (its `causal_ate` caller was already guarded).
  * the "Precision" section of the module docstring now names the four matmul sites, states the
    bit-identity contract for `causal_ate` AND `chunk_traj_loss`, and records the old behaviour.

After the fix the reviewer's own probe gives `same=True` on all four numbers and `apply_sim3` maxdiff
`0.0`.

**New test** `test_autocast_is_inert_on_the_whole_chunk_loss` (`tests/test_traj_loss.py`, added to the
script runner's `order`; the file's docstring coverage list was updated).  It runs `chunk_traj_loss`
with a 4-frame detached history and `rpe_w=0.1` inside and outside `torch.autocast('cpu', bfloat16)`
and asserts all **11 stats + the loss + the gradient** are bit-identical, that `rpe_trans > 0` and
`n_rpe > 0` (so the term is genuinely exercised), and that `umeyama_sim3` / `apply_sim3` are
bit-identical on their own.

DEPARTURE / honest caveat worth recording: the assertion is on the FORWARD under autocast with
`backward()` called OUTSIDE the region.  Calling `backward()` *inside* a live bf16 region still drifts
(measured 1.43e-04 max abs on this input) because the autograd engine inherits the ambient autocast TLS
and re-autocasts the backward matmuls -- that is torch's documented caller-side semantics, not
something this module can guard.  The test asserts that drift is non-zero as a tripwire, and
`train_wgate_consequence.py` already backwards outside every autocast block
(`accelerator.backward(loss)` at :415 is dedented out of both `with` statements), so the shipped
trainer is correct as written.

**Re-run of the CPU suites (all from `src/CUT3R/src`, cuteanything, PYTHONPATH set,
`OMP_NUM_THREADS=1`, `timeout 280`):**
  * `python -m pytest -q dust3r/wgate/tests/` -> **41 passed** in 4.09s (was 40; +1 is the new test).
  * `python dust3r/wgate/tests/test_traj_loss.py` (script runner) -> `test_traj_loss: 26/26 PASS`.
  * `python dust3r/wgate/tests/parity_cpu_consequence.py` -> 68.6s, `VARIANT-C PARITY OK`.
  * `python dust3r/wgate/tests/parity_cpu.py` -> exit 0, `PARITY OK: keys absent /
    present-with-weight-1 are bit-identical; negative controls behave as derived` (variant-1/2
    regression guard: this fix touches nothing they use).
  * `python eval_pipeline/wgate_selftest.py` (repo root) -> `wgate_selftest: ALL OK`.
  * `python src/CUT3R/src/train_wgate_consequence.py --selftest` -> `ALL OK`, init mean weight
    0.940398, first backward `|grad| 0.388214`, toy loss 0.012417 -> 0.000000.
  Loss values are unchanged by the fix everywhere the code actually runs (no autocast there), so none
  of the above numbers moved.

**Not fixed here, other groups own the files** (reported for the record, no edit made):
  * The smoke report's item 3 (production arm jsons naming `wgate/{cg1,cg2,...}` while the trainer's
    OUT_ROOT is `wgate/consequence`) is **already resolved by Component D**: all four jsons now read
    `.../checkpoints/wgate/consequence/<NAME>/gate_head_best.pth` and `wgate_selftest.py` prints
    `variant-C arm heads agree with the trainer output root`.  Verified, not edited.
  * The smoke report's items 2 and 4 are operational notes (node-local `/scratch/tmp` invisible to
    compute nodes; the crux driver's `sys.argv`), not defects in any file of this group.
  * The trainer's save/eval cadence, per-rank eval average, resume position and rank-broadcast items
    belong to the trainer group.
No key-name mismatch in another group's file needed fixing.

### 2026-09-22 Variant C, Component C (trainer) -- FIX PASS on the adversarial review's defects 1-7

Trainer-group fix pass. Files touched: `src/CUT3R/src/train_wgate_consequence.py` and
`eval_pipeline/train_wgate_consequence.sbatch` (the two files this group owns). NOTHING else was edited -- no
model/loss file, no eval file, no arm json, no test in `dust3r/wgate/tests/` (owned by the other groups).
No Slurm job submitted, nothing committed.

All seven defects of the 2026-09-22 "Component C -- ADVERSARIAL REVIEW" entry are now fixed:

1. **No optimiser step on a chunk with no trajectory signal.** `run_episode` reads `n_ate` out of
   `chunk_traj_loss`'s stats dict; when it is exactly 0 the chunk is logged and skipped instead of stepping.
   This matters because `chunk_traj_loss` returns `p_chunk.sum() * 0.0`, whose `requires_grad` is True, so the
   old code ran `opt.step()` with an all-zero gradient and AdamW's DECOUPLED weight decay still shrank the head
   (`p *= 1 - lr*wd`). `n_ate` absent (-1) keeps the old behaviour, so a loss that reports no count still steps.
2. **Bounded running buffers.** `run_l` / `run_w` are truncated to their last 400 / 2048 entries (only the last
   40 / 256 are ever read), instead of growing ~63 floats per episode x 38633 episodes of host RAM.
3. **Cadence by counter, not modulo.** save / eval / log now fire on `seen - last_* >= every`, with `last_*`
   carried in `state` and reset on `--resume`. The old exact modulo of a counter advancing by
   `batch_size * n_proc` worked only because the shipped launches are 4 GPUs x batch 1; a 3-GPU relaunch would
   have silently never checkpointed or evaluated for the whole 2-day job.
4. **Held-out ATE from gathered SUMS AND COUNTS.** `run_eval` gathers `[loss_sum, n_chunks, w_sum, n_w,
   ate_sum, n_ep]` (fp64), sums over ranks and divides afterwards, in the new module-level `eval_record()`.
   The old mean-of-per-rank-means let a rank with no eval episode contribute a literal 0.0 ATE -- which would
   have been written to `gate_head_best.pth` as a record. With no episode at all the record is now `nan`
   (and `rec["ate"] == rec["ate"]` already refuses to save on nan).
5. **`--resume` restores the DATA position.** New module-level `resume_position(seen, steps_per_epoch,
   per_step, seed) -> (epoch, skip, ep_episodes)`; the loop re-enters that epoch and skips `skip` batches of it.
   The old `epoch = seed + (seen > 0)` restarted epoch 1 from its beginning, so a job resumed after the 2-day
   wall re-trained seen episodes for ever and never finished the first pass.
6. **Rank broadcast.** New `broadcast_head(tag)` (`torch.distributed.broadcast` from rank 0 over the head's
   state dict, then a barrier) is called after `attach_head` and again after `--resume`. `sync_grads` only keeps
   the ranks together if they START together; that was true by construction but nothing enforced it. No-op at
   `n_proc == 1` or without an initialised process group.
7. **No silent fallback in the traj_loss binding.** `_wrap_traj`'s `pred_centres_of` / `gt_centres_of` /
   `final_ate` / `pred_pose_of` / `gt_pose_of` now raise `SystemExit` naming the function and the owner hint
   (like `load_components` does) instead of quietly substituting a local definition; signature PROBING
   (`TypeError` -> try the next spelling) is kept, and a name the module simply does not export still falls back
   to the contract definition with a logged note. The now-unreachable `_sim3_rmse` fallback was deleted.
   Rationale: the training path already had no fallback (an exception inside `chunk_traj_loss` was always
   fatal), so making the EVAL path fatal adds no new fragility -- it only stops a future signature change from
   silently changing the reported held-out metric.

Also (from the integration agent's process notes): the sbatch now refuses `WGATE_OUT` under `/scratch/*` or
`/tmp/*` next to its existing `/gpfs/projects` refusal -- on MN5 those are NODE-LOCAL NVMe and invisible to any
other node, which is what cost job 46321750. `/gpfs/scratch/...` is unaffected (the pattern is anchored).

NOT fixed here, and why: the integration agent's Component D finding (the four production arm jsons naming
checkpoints that do not exist) is **already closed by Component D** -- the jsons now read
`.../checkpoints/wgate/consequence/<ARM>/gate_head_best.pth`, which is exactly the trainer's
`OUT_ROOT/<NAME>`, and `eval_pipeline/wgate_selftest.py` asserts that agreement (it prints
`variant-C arm heads agree with the trainer output root .../consequence/<NAME>`). Nothing in another group's
file was edited by this pass. Component B's RPE autocast guard is likewise the other group's, and remains
masked on the trainer's path (`autocast(enabled=False)` around the loss call).

CPU tests re-run (login node, env `cuteanything`, `OMP_NUM_THREADS=1 timeout 280`):
* `python src/CUT3R/src/train_wgate_consequence.py --selftest` -> rc 0, `ALL OK`, with the pre-existing numbers
  UNCHANGED (init mean weight 0.940398, first backward |grad| 0.388214, toy loss 0.012417 -> 0.000000 in 8
  steps, checkpoint keys + reload OK) plus a new line covering the two extracted helpers:
  `eval_record (sums/counts, empty rank -> nan) + resume_position OK`
  (`resume_position(400,100,4)==(1,0,400)`, `(404,100,4)==(1,1,400)`, `(1204,100,4,seed=7)==(10,1,400)`,
  no-length loader -> the old guess; `eval_record` with 2 of 4 ranks empty still gives ate 0.04, all-empty nan).
* `python -m pytest -q dust3r/wgate/tests/` -> **41 passed** in 5.1 s (40 before this pass; the extra test is
  another group's, landed in parallel).
* `python eval_pipeline/wgate_selftest.py` -> rc 0, `wgate_selftest: ALL OK`, including the variant-C block and
  the arm-head/OUT_ROOT agreement line above.
* `bash -n eval_pipeline/train_wgate_consequence.sbatch` -> OK.
* NEW scratchpad probe (not added to the repo, tests dir is another group's): `run_episode` against a stub model
  that keeps the real contract (pose read BEFORE the commit, `state = a*new + (1-a)*old`, `a_0 = 1`, detached
  head input, `_wgate_trace_*`), 24 frames, AdamW `wd=0.5` and the head moved off its zero init so weight decay
  would be visible:
  - `--chunk 4 --ate_min_t 4`: 6 chunks, `n_ate = [0,4,4,4,4,4]`, **5 optimiser steps** (was 6).
  - the same run stopped after chunk 0 (`max_chunks=1`): **0 steps and the head bit-identical** (max |delta| ==
    0.0). Counterfactual measured separately: an `opt.step()` on an all-zero gradient with AdamW(lr 1e-3,
    wd 0.5) moves a parameter of 0.05 by 2.5e-05 = `lr*wd*p` -- that is exactly what defect 1 was doing.
  - `--chunk 8 --ate_min_t 4`, sites `state` and `mem`: 3 chunks, 3 steps, all 6 head parameters with a
    gradient, head moved 4.6e-03 -> the normal path is unchanged.
  - `train=False`: 0 steps, no `p.grad`, 23 traced weights -> eval mode unchanged.
* NEW scratchpad probe of defect 7: with `causal_ate` / `pred_centres` / `gt_centres` replaced by a raising stub
  (on a COPY of the module namespace, the real module untouched) each path now exits with
  `[wgcons] FATAL: traj_loss.<name> ... did not behave as the contract states`, and a missing `causal_ate` with
  `[wgcons] FATAL: traj_loss exports no 'causal_ate', so the held-out metric cannot be the benchmark quantity`.

Still not covered by anything in this pass: a multi-rank run. `broadcast_head` and the gathered `run_eval` are
no-ops / identities at `n_proc == 1`, so the GPU smoke evidence above (jobs 46322219 / 46323416 / 46324664)
still stands, but the 4-GPU behaviour of defects 3/4/6 is first exercised by the production launch itself.

### 2026-09-22 08:1x Variant C launched (orchestrator)
Smoke PASSED, incl. the crux on GPU: gate-head grad norm non-zero on the first chunk (state 1.24e-05, mem 1.26e-06),
loss decreases over repeated steps on the SAME chunk, mean weight moves off its init (.9404 -> .9385), zero backbone
params with grad; identity weight-mode heads (init_logit 40 -> w == 1) reproduce augfull_lr1e5 BIT-IDENTICALLY on a
real scene; causal_ate matches the evaluator's RMSE-reduced ATE to 3e-18 (float64) on RAIL+80edfcb1.
Full-data trainings (38,633 episodes, 4 GPUs each, acc_ehpc, 2-day limit, loss=ate, wmin .5, init_logit 2.0, chunk 16):
cg1_head (state, lr 1e-4) 46325772 | cg1_head_lr5 (state, 1e-5) 46325773 | cg2_head (mem, 1e-4) 46325774 |
cg2_head_lr5 (mem, 1e-5) 46325775. Supervisor eval_pipeline/wgate_cons_pipeline.sh started detached
($OUT/wgate_cons_pipeline, jids in $OUT/wgate_cons.jids): waits for the trainings, then 430 arms, cg constants,
430 table, 4292 shards, 4292 table.

### 2026-09-22 ~11:00 Variant C first held-out checkpoints (16 test episodes, gate on, no gradient)
| run | init ate | ep2000 ate | init w | ep2000 w |
|---|---|---|---|---|
| cg1_head (state, lr 1e-4) | .01181 | .01166 | .940 | .514 |
| cg1_head_lr5 (state, lr 1e-5) | .01181 | .01175 | .940 | .912 |
| cg2_head (mem, lr 1e-4) | .01149 | .01212 | .940 | .977 |
| cg2_head_lr5 (mem, lr 1e-5) | .01149 | .01190 | .940 | .949 |
Early reading (2000/38633 episodes, do not quote): the STATE gate moves in the expected direction and lr 1e-4 has
already driven the mean weight from .94 to .514, i.e. essentially to the wmin floor -- the optimiser's first move is to
lower the DOSE, exactly the law the variant-1 controls established, now rediscovered by gradient descent from ATE alone.
The MEM gate moves the other way (w up to .977) and its held-out ATE gets worse, consistent with variant 2's finding that
attenuating the pose-memory write does not help. Watch whether cg1 develops any per-frame SPREAD (the thing a constant
cannot do) rather than just sitting on the floor: the trace dumps at eval time will show it.
NOTE for the monitor: the trainer prints `[wgcons] EVAL {...}`, not `EVAL {...}` at line start.

### 2026-09-22 11:40 Variant C RELAUNCHED at batch 8 (the batch-1 launch was a mistake)
The first launch used the trainer's default `--batch_size 1`, giving 4 episodes in flight on 4 GPUs: 29.6 ep/min,
i.e. ~29 h for the single 38,633-episode epoch. Reference point that makes that indefensible: the FULL finetune
(`cut3r_finetune_aug_full_32gpu_lr1e5`, every weight trainable) does an epoch in 19.8 min (checkpoint-10..50 mtimes,
4 x 19.8 min apart) on 32 GPUs at batch 8 = 256 episodes in flight, 150 optimiser steps/epoch. We had 64x less
parallelism for a 0.44M-parameter head.
No code change was needed: `--batch_size` already existed and `traj_loss` was already written for (B,T,3). Verified
before relaunching: batched `causal_ate` is BIT-IDENTICAL to looping over episodes (max|d| = 0.0 on a 5x20 float64
case) and `chunk_traj_loss` gives every batch element non-zero gradient. GPU benchmark (job 46332736, 1 GPU,
acc_debug): batch 8 = 320 episodes in 8.5 min = 37.6 ep/min/GPU vs 7.4 ep/min/GPU at batch 1 -> 5.1x per GPU, no OOM.
Old jobs 46325772-75 cancelled at ~ep 3200. New: cg1_head 46334305 | cg1_head_lr5 46334306 | cg2_head 46334307 |
cg2_head_lr5 46334308, all `EXTRA="--batch_size 8"`, expected ~4.3 h/epoch on 4 GPUs. Supervisor restarted on the new ids.
NOTE: batch 8 means 8x fewer optimiser steps per epoch (4832 vs 38632); lr 1e-4 and 1e-5 are both in flight as before.

### 2026-09-22 NEW ARM `mg_conf`: the pose-memory write keyed by the model's OWN self-view confidence -- implemented, CPU-verified, 430 submitted

The missing experiment: every mem_gate arm so far is a constant, a leverage probe, a trained sigma head or a
consequence head -- the retriever memory has NEVER been keyed with the same existing, untrained self-view
confidence that gives our best state-side arm (`tok_al_q50_g50`, ATE -.0053* on 4292). This arm does exactly that.

Map (no training, causal, no GT): with `c_t` = mean over pixels of `log(conf_self.clamp(min=1))` for THIS frame
(the identical expression the conf trigger computes at model.py ~L1967) and a causal history of `c` over frames
`0..t-1`,
    `b_t = wmin + (1 - wmin) * sigmoid((c_t - median(c_0..c_{t-1})) / soft)`,  `b_0 = 1` exactly,
and `b_t` multiplies the MEM COMMIT MASK only: `mem = new_mem * (update_mask * b) + mem * (1 - update_mask * b)`.
Defaults `soft = 0.5`, `wmin = 0.5`. Frame at the running median -> `b = .75`; more confident -> 1, less -> .5.

Code sites (all strictly additive; four trainings were running and import model.py):
* `dust3r/wgate/heads.py`: new pure function `conf_mem_weight(c_t, hist, soft=.5, wmin=.5)` (+ `__all__`); returns
  exactly 1.0 on an empty history, i.e. the `b_0 = 1` rule lives in the helper and is unit-testable.
* `dust3r/model.py`: new method `_wgate_mem_conf(views, view_indices, res_group)` (right before
  `_wgate_record_step`) and, at the commit site in `_forward_decoder_group_step`, `mem_mask = update_mask` ->
  `mem = new_mem * mem_mask + mem * (1 - mem_mask)` with `mem_mask` multiplied by `b` only when the hook returns
  one. New per-view (1,) keys `mem_gate_src` (1.0 = "conf"), `mem_gate_conf_soft`, `mem_gate_conf_wmin`; causal
  history `self._memconf_hist`, reset at view index 0, `c_t` appended AFTER use; fp32, autocast off,
  `torch.no_grad()`. Without `mem_gate_src` the hook returns None on its first line, `mem_mask` IS `update_mask`
  and the commit is the same expression as before -- and at `b == 1` (incl. t = 0) it also returns None, so the
  blend arithmetic is skipped and the write stays bit-exact. The state commit and the token gate are untouched.
  `b_t` goes into `_wgate_trace` / `_wgate_trace_mem` via the existing `_wgate_trace_append`, so the worker's trace
  dump and `wgate_make_controls.py` work unchanged. Orthogonal to `_wgate_mem_gate` (which rescales `new_mem` at
  the `update_mem` site): both on => the write is `b_head * b_conf`.
* `eval_pipeline/infer_and_eval_worker.py`: `apply_wgate_controls` mem_gate block extended with
  `{"src": "conf", "soft": .5, "wmin": .5}` -> the three per-view keys on EVERY view (the model owns `b_0 = 1`);
  head / const / freeze_after / alpha paths untouched, an unknown `src` raises. Spec recorded as
  `{"src","soft","conf_wmin"}` in `wgate_trace.json["gates"]`.
* NEW `eval_pipeline/maks_arm_mg_conf.json` = `{"*": {"mem_gate": {"src": "conf", "soft": 0.5, "wmin": 0.5}}}`.
* `eval_pipeline/wgate_table.py`: row `mg_conf` = "V2 mem gate keyed by the model's OWN self-view confidence
  (untrained)", in group V2 after `mg_oracle`.
* NEW `dust3r/wgate/tests/test_mem_conf_gate.py` (10 tests, CPU, ~1 s): the `b_t` formula against an independently
  written reference (incl. the lower-median convention, the temperature, the `[wmin, 1]` range, monotonicity,
  saturation, bad-arg rejection), the `b_0 == 1` rule for every (soft, wmin) and for `hist` None/[]/(), then the
  hook on a stub -- inert without the key (no state, no trace), `b_0 = 1` returning None, the CAUSAL median
  (never sees the current frame), the trace entries, soft/wmin defaults and per-view overrides, history reset at
  view index 0, batched shape -- plus a source grep pinning the wrapped commit.
* `dust3r/wgate/tests/test_wgate.py`: `test_model_hooks_wrapped_once` updated for the renamed commit line
  (`mem_mask`), with `mem_mask = update_mask` added to the ordering assertion. Only change to an existing test.

Tests (login node, env cuteanything, `OMP_NUM_THREADS=1 timeout 280`):
* `python dust3r/wgate/tests/parity_cpu.py` -> `PARITY OK: keys absent / present-with-weight-1 are bit-identical;
  negative controls behave as derived`, rc 0, total wall 133.7 s / cpu 132.6 s (real `augfull_lr1e5`).
* `python dust3r/wgate/tests/parity_cpu_consequence.py` -> `VARIANT-C PARITY OK: wmin=1 weight-mode is
  bit-identical to plain; the consequence gradient reaches both heads through the frozen rollout`, 67.8 s.
* `python -m pytest -q dust3r/wgate/tests/` -> `51 passed in 8.42 s` (was 50 + the 10 new, with the one grep test
  updated as above; the intermediate `1 failed, 50 passed` was exactly that stale line, fixed, not suppressed).
* `python dust3r/wgate/tests/test_mem_conf_gate.py` -> `all 10 tests passed`.
* Worker parse check on the new json (keys + values on all views) and on `mg_const0/mg_freeze8/mg_const50`
  (unchanged: only `mem_gate_const` / `mem_gate_freeze_after` set); `eval_pipeline/wgate_selftest.py` ->
  `wgate_selftest: ALL OK`.

Submitted (the only job): `WT=$PWD ARM=mg_conf PILOT=wgate430 sbatch --job-name=wg430_mg_conf
eval_pipeline/maks_sweep430.sbatch` -> Slurm **46348457**. No 4292 run, no constants; the dose-matched constant
for this arm needs the trace (`wgate_make_controls.py`) and is a follow-up. NOT verified on GPU yet: this arm has
only been run on CPU stubs -- the first real rollout is job 46348457.
