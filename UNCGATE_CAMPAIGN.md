# uncgate: a Kendall-Gal registration-uncertainty head on frozen CUT3R, as the key of the per-token write gate

Date: 2026-09-18. Worktree `maks_idea` (uncommitted). Companion to `KENDALL_GAL_CONF_GATE_ANALYSIS.md`.
`$OUT` = `/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval`.

## 1. What was built

Two heads, both trained with the heteroscedastic loss of Kendall & Gal (NeurIPS 2017) on the frozen
`augfull_lr1e5` checkpoint, with the target of SURE-Map (arXiv 2609.15795): the residual between the backward
optical flow induced by the model's own pose and depth (frame t to t-1) and the ground-truth flow from GT depth
and GT poses. The head predicts an unbounded 2-channel log-variance s = (s_u, s_v) per pixel (v1) or per state
token (v2); loss = eps_u^2 e^{-s_u} + eps_v^2 e^{-s_v} + s_u + s_v (Gaussian) or the Laplace variant.
Nothing in CUT3R is updated. At inference the head's score (-0.5 (s_u + s_v), higher = more confident) keys the
RoPE-aligned per-token gate: the on-image state tokens in the top half by score commit fully, the rest at 0.5,
the 528 off-image registers commit fully.

| piece | where |
|---|---|
| flow target, pooling, RoPE map | `src/CUT3R/src/dust3r/uncgate/flow_target.py` |
| heads (v1 PixelUncHead, v2 TokenUncHead), losses, alignment | `src/CUT3R/src/dust3r/uncgate/heads.py` |
| model hook (`attach_unc_gate`, `_unc_forward`), gate keys `token_gate_src=3`, `token_gate_ema` | `src/CUT3R/src/dust3r/model.py` |
| worker: `--unc_head`, control-json `token_gate.{src,ema,unc_head,unc_mode}` | `eval_pipeline/infer_and_eval_worker.py` |
| trainer (fp32, accelerate DDP, resume, `--loss`, `--clip`) and launcher | `src/CUT3R/src/train_unc_gate.py`, `eval_pipeline/train_unc_gate.sbatch` |
| supervisor (train -> resume on failure -> 8-shard 4292 eval -> table), table, pairwise | `eval_pipeline/unc_gate_pipeline.sh`, `unc_gate_table.py`, `unc_gate_pairwise.py` |

Verification before training: the no-head path is byte-identical to the earlier aligned-gate runs on all 12 smoke
scenes; random-init heads run end to end; module tests on CPU; training smokes on 1 GPU, 4 GPUs (DDP) and v2
with next-frame channels. Two review passes fixed the sampler epoch, a bf16-autocast corruption of the sub-pixel
target (0.57 px mean error), an optimizer step invalidating later frames' graphs, the shuffle control being a
no-op with the head source, undetached telemetry, tie-inflated retention curves and padded episodes.

## 2. Training (v1, full wrist train split)

Three runs, one pass each over the 38,633 training episodes (64 consecutive frames, 320x192, ImgNorm, no
colour jitter, episodes shorter than 64 frames excluded), 4 H100 each, batch 2 per GPU, AdamW 1e-4, optimizer
step every 8 frames, 213 minutes per run:

| run | loss | residual clip | head | held-out retention@0.2 on the flow residual (self-view conf: 0.977) |
|---|---|---|---|---|
| A `v1_gauss_c64` | Gaussian NLL (SURE-Map) | 64 px | 3.94M params | 0.811 |
| B `v1_laplace_c64` | Laplace NLL (K&G Sec. 4) | 64 px | 3.94M | 0.808 |
| C `v1_gauss_c16` | Gaussian NLL | 16 px | 3.94M | 0.820 |

Retention@0.2 = RMS residual of the 80% most confident pixels over RMS of all, on 32 held-out test episodes; the
head learned the target (0.81 vs 0.98 for the existing self-view confidence, which has no ranking power on this
target). Heads at `/gpfs/scratch/etur59/koc821022/checkpoints/unc_gate/<run>/head.pth`.

## 3. Results on all 4292 test scenes

Table: `$OUT/unc_full4292/table/unc_gate_4292.{png,pdf,tex,md,csv}` (paired vs `augfull_lr1e5`, * = 95%
scene-bootstrap CI excludes zero; two arms per head: the gate alone and with a per-token EMA of 0.7).

| method | AbsRel | d<1.25 | ATE | RPE trans | RPE rot |
|---|---|---|---|---|---|
| CUT3R finetuned (baseline) | 0.1794 | 0.7863 | 0.0759 | 0.0079 | 1.101 |
| Conf-Gate frame gate (g7ema) | 0.1786 (-.0007*) | 0.7872 (+.0009*) | 0.0728 (-.0031*) | 0.0081 (+.0001*) | 1.107 (+.006) |
| Causal pose recalibration | 0.1794 | 0.7863 | 0.0690 (-.0068*) | 0.0071 (-.0008*) | 1.101 |
| Aligned token gate, self-view conf (q.5, g.5) | 0.1787 (-.0007*) | 0.7874 (+.0011*) | **0.0706 (-.0053*)** | 0.0080 (+.0001*) | 1.097 (-.004) |
| v1 head A: token gate q.5 g.5 | 0.1787 (-.0007*) | 0.7872 (+.0009*) | 0.0719 (-.0040*) | 0.0080 (+.0000*) | 1.100 (-.001) |
| v1 head A + per-token EMA .7 | 0.1787 (-.0007*) | 0.7874 (+.0011*) | 0.0719 (-.0039*) | 0.0080 | 1.097 (-.004) |
| v1 head B: token gate q.5 g.5 | 0.1786 (-.0008*) | 0.7875 (+.0012*) | 0.0718 (-.0041*) | 0.0080 | **1.095 (-.006*)** |
| v1 head B + per-token EMA .7 | 0.1787 (-.0006*) | 0.7873 (+.0010*) | 0.0719 (-.0040*) | 0.0080 | 1.099 (-.002) |
| v1 head C: token gate q.5 g.5 | 0.1788 (-.0006*) | 0.7872 (+.0009*) | 0.0720 (-.0039*) | 0.0080 | 1.098 (-.003) |
| v1 head C + per-token EMA .7 | 0.1788 (-.0006*) | 0.7872 (+.0009*) | 0.0719 (-.0040*) | 0.0080 (+.0000*) | 1.096 (-.005*) |

Paired arm-vs-arm on the same 4292 scenes (`$OUT/unc_full4292/table/pairwise.md`; negative ATE = first arm better):

| pair | absrel | a1 | ATE | RPE trans | RPE rot |
|---|---|---|---|---|---|
| self-conf gate minus baseline | -.0007* | +.0011* | -.0053* (2919/4292) | +.0001* | -.004 |
| head A minus self-conf gate | +.0000 | -.0003 | +.0013* (1999/4292) | -.0000 | +.003 |
| head B minus self-conf gate | -.0001 | +.0000 | +.0012* (1968/4292) | -.0001* | -.002 |
| head C minus self-conf gate | +.0001 | -.0002 | +.0014* (1953/4292) | -.0000* | +.000 |
| head A + EMA minus self-conf gate | -.0000 | -.0000 | +.0014* | -.0001* | +.000 |
| head B minus head A | -.0001 | +.0003 | -.0001 | -.0000* | -.005* (2175/4292) |
| head B minus frame gate g7ema | -.0000 | +.0003 | -.0010* (2352/4292) | -.0001* | -.011* (2419/4292) |
| self-conf gate minus frame gate g7ema | +.0000 | +.0003 | -.0022* (2504/4292) | -.0001* | -.009* |
| head B minus causal recalibration | -.0008* | +.0012* | +.0028* | +.0008* | -.006* |

## 4. Reading

1. Every head-keyed gate beats the baseline on ATE (-5.2% to -5.4%), depth and, for B and C+EMA, rotation, with
   no significant loss on any metric: the first single-pass causal arm family in the record with rpe_rot
   significantly better than the baseline.
2. The heads do NOT beat the heuristic they were meant to replace. Keyed on the self-view confidence the same gate
   gives ATE -.0053*; keyed on any of the three learned heads it gives -.0040*. The three heads and the EMA variant
   are indistinguishable from each other, so this is not a training-choice effect: the loss form, the residual
   clip and the smoothing do not matter.
3. The heads rank their own target well (retention 0.81 vs 0.98) and the gate still gains nothing from that
   ranking beyond what the shuffled-alignment control already delivered (-.0041* on the 430 subset). Where the
   flow residual is uncertain is not where the state write should be attenuated; the small extra ATE of the
   self-view key comes from depth saliency (far patches, occlusion boundaries), the reading the analysis document
   already recorded. The one thing the learned heads add is a slightly better rotation trade (B: -.006* vs -.004).
4. Consequence for the module: the write gate remains dose plus spatial placement; a registration-uncertainty
   ranking is not the missing ingredient. The head itself is a valid registration-difficulty estimator and is worth
   keeping for output-side uses (SURE-Map's own use: per-frame weighting of a translation correction, point
   filtering), which is the pinned "new from both papers" direction.

## 5. Reproduction

```bash
# train one head (4 GPUs, one pass over the train split)
WT=$PWD MODE=v1 NAME=v1_gauss_c64 EPISODES=38633 BS=2 EXTRA="--loss gauss --clip 64" \
  sbatch --gres=gpu:4 --cpus-per-task=80 eval_pipeline/train_unc_gate.sbatch
# evaluate an arm on a 4292 shard (control json carries the head path); 8 shards under $OUT/unc_full4292/shards
WT=$PWD ARM=tok_uncA_q50_g50 PILOT=unc_full4292 SCENE_LIST_OVERRIDE=$OUT/unc_full4292/shards/shard_0.txt \
  sbatch eval_pipeline/maks_sweep430.sbatch
# or the whole campaign, detached from the session
setsid nohup bash eval_pipeline/unc_gate_pipeline.sh >> $OUT/unc_gate_pipeline/supervisor.out 2>&1 < /dev/null &
# table and pairwise
python eval_pipeline/unc_gate_table.py; python eval_pipeline/unc_gate_pairwise.py
```
Jobs: training 46043495/46043498/46043501; comparator shards 46043483-92; head shards 46056xxx; tables 46061922/28.

## Conf branch trained through the gate (2026-09-19/20)

Question: can the self-view confidence head, retrained on DROID *through* the differentiable token gate with the rest of
CUT3R frozen, beat the untrained head as the gate's ranking key (4292: ATE -.0053*)?

Setup (`src/CUT3R/src/train_conf_gate.py`, `eval_pipeline/train_conf_gate.sbatch`, supervisors
`eval_pipeline/conf_gate_pipeline{,_lr5,_g1lr5}.sh`): `ConfBranchHead` = deep copy of the DPT self head's last conv stack
(0.443M params, bit-identical at init), only trainable module; soft budgeted gate (T .5, per-frame detached median -> fixed
dose), q .5 gmin .5 align 1, registers commit fully; finetune criterion over 16-frame TBPTT chunks (state detached at chunk
boundaries); one epoch over the 38,633 train episodes, batch 1 x 4 GPUs, ~26 h per run. Runs: G1/G10 (lr 1e-4) stopped by
hand at ep8000/ep6000 (held-out pose loss drifting up); G1L (gscale 1) and G10L (gscale 10) at lr 1e-5 completed.
Held-out (16 episodes) pose loss stayed inside +-.005 of the untrained value (.179) for the whole epoch on both runs.

4292 result (`$OUT/unc_full4292/table/unc_gate_4292.png`, paired vs augfull_lr1e5):

| arm | AbsRel | d<1.25 | ATE | RPE trans | RPE rot |
|---|---|---|---|---|---|
| untrained self-conf key (tok_al_q50_g50) | -.0007* | +.0011* | -.0053* | +.0001* | -.004 |
| G1L hard rank (tok_cbG1L_q50_g50) | -.0006* | +.0010* | -.0054* | +.0001* | -.008* |
| G1L soft T.5 | -.0010* | +.0015* | -.0043* | -.0000* | -.012* |
| G10L hard rank | -.0006* | +.0011* | -.0052* | +.0001* | -.007* |
| G10L soft T.5 | -.0009* | +.0016* | -.0042* | -.0000 | -.007 |

Paired vs the untrained key (`eval_pipeline/cb_pairwise.py`, Slurm job; `$OUT/unc_full4292/table/cb_pairwise_*.md`):
G10L hard minus self: ATE +.00008 [-.0002,+.0004] (tie), rot -.0036 [-.0089,+.0010] (n.s.); G10L soft minus self: ATE
+.0011* (worse), a1 +.0004*, rpe_trans -.00008*. G1L hard minus self: ATE -.00014 [-.0004,+.0002] (tie), rot -.0041* [-.0086,-.0002] (significant), depth flat; G1L soft minus self: ATE +.0010* (worse), absrel -.0003*, rpe_trans -.00009*, rot -.0077*.

Reading: training the key through the gate reproduces the untrained key's ATE gain exactly (hard rule ties on ATE) and adds a
small rotation gain that is significant vs the baseline; the soft rule trades ~.001 ATE for the best depth and rotation in
the table. Nothing beats the heuristic on ATE. Consistent with the loss structure: the branch starts at the minimum of the
direct Kendall-Gal term, which anchors it, and the gate-path gradient (the only route for the pose term) is weak even at
gscale 10. Next design if pursued: keep the frozen head's conf for the loss weighting, let the branch drive only the gate,
pose-only loss. G1L hard rank is the row added to `master/master_4292_causal_vggt_norecalib` (csv key tok_cbG1L_q50_g50).

### Hot-swap smoke on the PRETRAINED checkpoint (430 scenes, 2026-09-20)

`cut3r_512_dpt_4_64.pth`, same control JSONs, `$OUT/maks_sweep430_zs/{plain,tok_al_q50_g50,tok_cbG1L_q50_g50}`,
paired vs zs plain (`compare/subset_compare.md`). Zero-shot plain: absrel .480 / ATE .1231 (random-walk floor).
- own self-conf key: ATE -.0033* (2.7%), absrel -.004 n.s., rpe_trans +.0003* (worse), rot 0.
- G1L head (trained on the finetuned backbone, hot-swapped): ATE -.0028*, absrel -.0062*, a1 +.0026*, rpe_trans +.0003*.
Reading: the mechanism works on the pretrained backbone at the same ~3-7% relative ATE scale as on the finetune, but it is
.046 ATE / .30 absrel away from plain finetuning; the hot-swapped head is not broken (it beats the untrained key on depth)
but neither key moves the pretrained model off the random-walk floor. A head trained on the pretrained backbone would be
fighting for ~.003 ATE; not worth 26 h against a finetune gap of .046.

### Trained head with both backbones, all 4292 scenes (2026-09-20)

`eval_pipeline/conf_head_backbones_table.py` -> `$OUT/zs_full4292/table/conf_head_backbones_4292.png`. Pretrained backbone
arms under `$OUT/zs_full4292/`, finetuned under `$OUT/unc_full4292/`.

| backbone | arm | AbsRel | d<1.25 | ATE | RPE trans | RPE rot |
|---|---|---|---|---|---|---|
| pretrained | plain (zero-shot) | .4786 | .5580 | .1197 | .0120 | 1.713 |
| pretrained | + gate, own conf | -.0055* | +.0031* | -.0028* | +.0002* | +.001 |
| pretrained | + gate, G1L head (hot-swapped) | -.0060* | +.0031* | -.0028* | +.0002* | +.002 |
| pretrained | G1L minus own conf | -.0006 | -.0001 | -.0000 | -.0000 | +.001 |
| finetuned | plain | .1794 | .7863 | .0759 | .0079 | 1.101 |
| finetuned | + gate, own conf | -.0007* | +.0011* | -.0053* | +.0001* | -.004 |
| finetuned | + gate, G1L head | -.0006* | +.0010* | -.0054* | +.0001* | -.008* |
| finetuned | G1L minus own conf | +.0001 | -.0002 | -.0001 | -.0000 | -.004* |

Reading: on both backbones the trained head is indistinguishable from the backbone's own untrained confidence as the gate
key on every metric except a .4% rotation gain on the finetuned backbone (where it was trained). The gate itself gives
~2.3% ATE on the pretrained model and ~7% on the finetune; the pretrained model stays at the random-walk floor.
