# Harmful-frame skip pilot: assessment and execution plan

Date: 2026-09-02. Worktree: `maks_idea` (branch `maks_idea`, base = known_good_fsrc_noproj shelf).

## 1. The idea, restated

1. Pick one random scene from the 4292-scene DROID wrist test split.
2. Run pretrained CUT3R (`cut3r_512_dpt_4_64.pth`, the "Regular CUT3R" row) on it, score with the
   table-standard eval.
3. From the per-frame rows, flag frames that hurt the scene metrics with a simple threshold.
4. Re-run inference with those frames "ignored", re-score, compare.

## 2. Verdict

**Feasible, cheap, and ~40 lines of code. But the write-block form of this experiment has already
been run at scale and lost on all five metrics.** Run the pilot anyway as a mechanism demo on the
pretrained checkpoint, but design it so the result is interpretable (three variants + a control,
see §5), and do not treat one scene as evidence.

Prior art (main checkout `my-da3`, `CONF_GATE_CAMPAIGN.md` §"Phase-2 lever ledger", evidence in
`eval_pipeline/evidence/cg_oracle_430.json`):

- "ORACLE HARD-BLOCK": masks built from the clean run's per-frame GT errors, three thresholds
  (`ate > 1.5×median`, top-20% ATE, `rpe > 2×median`), ~19–23% of frames blocked via
  `views[i]["update"]=False`, 430-scene paired subset, finetuned augfull checkpoint.
- Result: all three variants worse than clean on all 5 metrics. ATE +0.0007…+0.0027,
  rpe_rot +0.28° (significant), rpe_trans +25% (significant).
- Interpretation recorded there: per-frame GT error marks **victims** of drift, not **culprits**.
  The corrupting frames already wrote to memory; the poorly-localised frames that follow are often
  late-but-good observations, and blocking them in contiguous runs compounds staleness.
  Confidence-keyed gating (the model's own signal) wins where GT-error-keyed blocking loses.

What is genuinely new in this proposal:

- Pretrained checkpoint instead of the finetuned one (the finding may or may not transfer).
- The option of **hard-dropping** frames from the input rather than blocking their memory write.
  Nobody has measured that, and it changes the RPE step structure (see §3).

## 3. Metric conventions to keep (the "latest tables")

Every table row (`summary/summary_table_final.tex`, `averages_table.*`) is produced by
`eval_bundle/bin/eval_depth_poses.py` with **default args**, invoked by the worker as
`--pred_root <preds/scene> --gt_root <scene>/dense --output_csv <eval/scene>/eval_depth_pose_metrics.csv`.
Do not pass `--eval_like_cut3r`; it is a different code path.

| metric | per-frame value | per-scene (MEAN row) | across scenes |
|---|---|---|---|
| absrel, a1 | per-frame median-scale alignment, mask `pred>eps & gt>eps` | nanmean over frames | nanmean |
| ate | Sim3 Umeyama fit on ALL frame centres, then per-frame ‖p−g‖ | nan-RMSE over frames | nanmean |
| rpe_trans, rpe_rot | consecutive-pair relative-pose error after the same Sim3, attributed to the later frame; frame 0 is NaN | nan-RMSE over frames | nanmean |

Regular CUT3R reference row (full 4292): absrel 0.1794, a1 0.7863, ate 0.0759, rpe_trans 0.0079,
rpe_rot 1.1010. **Correction (2026-09-03):** that row is `augfull_lr1e5`, i.e. the finetuned
`cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth` (50 epochs, lr 1e-5), NOT the
pretrained zero-shot `cut3r_512_dpt_4_64.pth` that the `regular` label denotes in the older
`run_regular_parallel.sh`. The §11 pilot ran the pretrained checkpoint; its per-scene trees under
`augfull_lr1e5/{preds,eval}` are complete on scratch if the finetuned one is wanted.

Consequences that shape the experiment:

1. Per-frame ATE depends on a global Sim3 fit over all frames. Removing frames from the scored set
   changes the alignment for the frames that remain. This is a scoring effect, not a model effect.
2. RPE is pairwise over whatever frames are scored. Hard-dropping frame t turns (t−1,t),(t,t+1) into
   one pair (t−1,t+1) with roughly double the motion. Both arms must be scored on the same frame
   set or the comparison is meaningless.
3. RMSE over frames is dominated by exactly the outlier frames you are about to flag. Rescoring
   the baseline on the kept set will already improve it a lot. **That is the control (V0).**

## 4. Mechanics already in place (no model surgery needed)

- Inference path for the image-only worker is `inference()` → `model(views)` → `_forward_impl`
  → `_forward_decoder_group_step` with `views_per_step=1`. That step reads
  `views[i].get("update")` (`src/CUT3R/src/dust3r/model.py:1867-1885`) and gates the state/memory
  commit. So setting `views[i]["update"] = torch.tensor(False).unsqueeze(0)` after
  `demo.prepare_input` is a live switch here. The forced `update = torch.ones(...)` at
  model.py:2020/2048 is in the unused `_da3_forward_impl` path; ignore it.
- `my-da3/eval_pipeline/infer_and_eval_worker_ray.py:366-383` has the `SKIP_FRAMES_FILE` hook
  (json `{scene: [frame idx]}`) to port. `my-da3/eval_pipeline/cg_build_oracle_masks.py` has the
  threshold logic to port (never flag frame 0; cap 40% per scene).
- `eval_pipeline/infer_and_eval_worker.py` (this worktree) is the worker that produced the
  Regular row (`run_regular_parallel.sh`). Use it, not the ray worker, to stay on the identical path.

## 5. Experimental design: three variants and one control

All on the same scene, same checkpoint, same eval defaults. F = all frames, K = kept frames
(F minus flagged), B = flagged frames.

| arm | inference | scored on | answers |
|---|---|---|---|
| P1  | plain pass | F | baseline; source of the per-frame rows used to flag |
| V0  | none (reuse P1 preds) | K | how much of any "gain" is pure scoring (dropping outliers from the RMSE and re-fitting Sim3) |
| V1  | B get a forward pass but `update=False` | F, and K | does keeping bad frames out of memory help the frames that follow? (the prior experiment; expected to lose) |
| V2  | B removed from the input sequence entirely | K | does the model do better when it never sees them? Compare **V2 vs V0**, never V2 vs P1 |

Optional V3 (only if V1 or V2 shows anything): flag again from V1's own per-frame rows and iterate
once. Skip in the pilot.

Flagging rules (simple per-scene-median thresholds, one json per rule; the first two are the
prior experiment's rules so the pilot is directly comparable):

- `ate15`: `ate_t > 1.5·median(ate)`.
- `rpe2x`: `rpe_trans_t > 2·median` OR `rpe_rot_t > 2·median`.
- `absrel15`: `absrel_t > 1.5·median(absrel)` (depth-only, new).
- `any`: union of the three. On the pilot scene's finetuned CSV the union hits the cap
  (ate15 21%, rpe2x ~30%, absrel15 9%, union 40% capped), so read it as "aggressive".
- Always: never flag frame 0, cap flagged fraction at 40% (keep the worst by normalised score),
  NaN never flags. Record the flagged fraction and the run-length histogram; contiguous runs were
  the mechanism in the prior loss.

Scoring a subset, caveat found in the dry run: `eval_depth_poses.py` renumbers files
**positionally** per camera stream (`_split_stream_by_camera`), so a subset of predictions scored
against the full GT is misaligned from the first gap on. Every K-scoring therefore builds a kept-GT
symlink tree with the same frame numbers (`build_kept_gt` in the worker and in
`maks_rescore_subset.py`), and the resulting CSV's `local_timestep` is the position within K
(`kept_frames.json` maps it back).

## 6. Scene

Seeded uniform pick over `outputs/cut3r_eval/scene_list.txt` with `random.Random(0).choice`:

    TRI+52ca9b6a+2023-10-26-15h-31m-47s   (268 frames, list index 3155)

Write it to `eval_pipeline/maks_scene_1.txt`. `augfull_lr1e5/eval/<scene>/` already holds a
per-frame CSV for this scene from the finetuned model; useful as a plausibility check on P1, not
as a baseline.

## 7. Code changes (this worktree only)

1. `eval_pipeline/infer_and_eval_worker.py`
   - `--skip_frames_file PATH` (json `{scene: [idx]}`) and `--skip_mode {block,drop}`.
   - `block`: after `prepare_input`, set `views[i]["update"]` to False for listed i.
   - `drop`: filter `img_paths` before `prepare_input`; pass the surviving original indices to
     `save_depth_camera` so depth/camera files keep their **original** frame numbers (the eval
     joins pred and GT on file index).
2. `save_depth_camera(..., frame_ids=None)`: optional list of output indices (default `range(B)`).
3. `eval_pipeline/maks_flag_harmful_frames.py`: reads P1's CSV (camera 0, numeric
   `local_timestep`), applies the rules in §5, writes `<label>/skip_<rule>.json` plus a one-line
   summary (n_flagged, fraction, longest run).
4. `eval_pipeline/maks_rescore_subset.py`: builds a temp pred dir of symlinks for the kept indices
   and runs `eval_depth_poses.py` with default args into a separate eval dir. Used for V0 and for
   the "V1 scored on K" column. The eval script itself is not modified, so the numbers stay
   comparable with every existing table.
5. `eval_pipeline/maks_harmful_1scene.sbatch`: `--partition acc --qos acc_debug --gres gpu:1
   --cpus-per-task 20 --time 00:45:00`, self-locating `WT` like the other launchers, sources
   `mn5_paths.sh`, then runs in order: P1 → flag → V0 → V1 (block) → V2 (drop) → a small
   aggregation that prints the §5 table for both rules. Outputs under
   `$OUT/maks_harmful_1scene/{p1,v1_<rule>,v2_<rule>}/{preds,eval,eval_kept}`.
   Never run this on a login node (`mn5_require_slurm`).

Expected cost: a few minutes of one GPU; well under a gigabyte of predictions.

## 8. Sanity gates before reading the numbers

- P1 per-frame CSV has 268 numeric rows plus MEAN rows, no NaN in ate for t ≥ 0.
- V1 parity check that the switch actually fired: predictions for every frame **before** the first
  flagged frame must be byte-identical to P1; the first frame **after** it must differ. If
  everything is identical, the update flag did not reach the model.
- V2 file-index check: `preds/<scene>/depth/` has exactly |K| files, named with the original
  indices, and the eval CSV's numeric rows are exactly K.
- Flagged fraction between 5% and 40%; if the rule flags nothing or everything, the scene is a
  bad pilot, pick the next seed.

## 9. Reporting

One table, five metrics, columns P1(F), V0(K), V1(F), V1(K), V2(K), per rule, with n_flagged and
longest flagged run. Plus per-frame ATE curves for P1 vs V1 with flagged frames shaded (the prior
campaign found the interesting signal in where the errors sit, not in the means).

## 10. What would make this worth scaling

n = 1 cannot show a win. If V1(F) beats P1(F) on 3 of 5, or V2(K) beats V0(K) on 3 of 5, repeat on
`my-da3/eval_pipeline/cg_smoke_scenes_12.txt`, then on the 430-scene subset with the paired
bootstrap in `my-da3/eval_pipeline/cg_analyze.py`. Prior evidence says V1 loses; V2 vs V0 is the
open question. If even that is flat, the honest conclusion is the one already recorded: GT error
identifies victims, and any frame selection has to be keyed on a signal the model has at
inference time (confidence), which is the conf-gate line of work.

## 11. Pilot result (2026-09-02, job 45337367, 3.5 min on one H100)

Outputs: `/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/maks_harmful_1scene/` (report_<scene>.md,
ate_per_frame_<rule>.png, masks/, p1/, v1_<rule>/, v2_<rule>/). Both §8 gates passed for all four
rules (update switch verified by byte-identical preds before the first flagged frame and changed
preds after it; drop-mode files = |K| with original indices).

Pretrained CUT3R on this scene is far worse than the 4292-mean (P1: absrel .430, a1 .608,
ate .189, rpe_trans .021, rpe_rot 2.75°); it is a hard, drifting scene.

| rule (flagged) | comparison | absrel | a1 | ate | rpe_trans | rpe_rot |
|---|---|---|---|---|---|---|
| ate15 (14%)   | V1(F) vs P1(F) | −.022 | +.024 | +.018 | +.0006 | +2.55° |
| ate15         | V1(K) vs V0(K) | −.009 | +.014 | +.032 | −.0035 | +0.65° |
| rpe2x (35%)   | V1(F) vs P1(F) | −.047 | +.031 | −.009 | +.026  | +5.47° |
| rpe2x         | V1(K) vs V0(K) | −.038 | +.030 | −.027 | −.021  | +0.97° |
| absrel15 (4%) | V1(F) vs P1(F) | −.016 | +.010 | +.015 | −.0001 | +1.43° |
| absrel15      | V1(K) vs V0(K) | −.011 | +.010 | +.019 | −.0067 | −0.10° |
| any (40%)     | V1(F) vs P1(F) | −.046 | +.031 | −.013 | +.029  | +5.50° |
| any           | V1(K) vs V0(K) | −.029 | +.023 | −.013 | −.026  | +0.92° |

Findings (n = 1 scene, so mechanism observations, not evidence):

1. **Block ≡ drop.** V1 and V2 predictions are bit-identical on every kept frame for every rule
   (231/231, 173/173, 256/256, 161/161). The model carries nothing across frames except
   `state_feat` and `mem`, so suppressing the write is the same as never showing the frame. V2 is
   therefore redundant with "V1 scored on K"; the only real choice is whether flagged frames are
   scored.
2. **Scored on all frames, blocking destroys rotation RPE** (+1.4° to +5.5°): the flagged frames
   are decoded against an increasingly stale state inside each contiguous run and their poses
   saw-tooth (visible in ate_per_frame_rpe2x.png, frames 100–150). This is the prior campaign's
   "stale frames themselves pay RPE" law, reproduced on the pretrained checkpoint.
3. **Scored on kept frames only, the picture is mixed and rule-dependent:** depth improves for
   every rule (absrel −.01 to −.04, a1 +.01 to +.03); rpe_trans improves for every rule; ATE
   improves only for the aggressive rules (rpe2x, any) and worsens for ate15/absrel15; rpe_rot
   worsens for 3 of 4 rules. No rule wins on all five.
4. The V0 control matters: rescoring P1 on K alone moves ATE by −.03 to −.05 and rpe by +50% to
   +200% (gaps), so any "V2 vs P1" comparison would have been dominated by scoring artefacts.

Next step if pursued: run `maks_harmful_1scene.sbatch` per scene over
`my-da3/eval_pipeline/cg_smoke_scenes_12.txt` (SCENE_LIST_OVERRIDE, PILOT=<label>), then the 430
subset with paired bootstrap, before drawing any conclusion. Drop V2 from that sweep (identical to
V1 on K). If the depth gain survives at scale it is the one signal not present in the prior
finetuned-checkpoint experiment.
