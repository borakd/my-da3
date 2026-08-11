# captain_gru_v3 lever grid — F (image features) x R (iterative GRU)

Four overfit configs, every one the exact `captain_gru_v2_a4_g1` / `_a4_g3` recipe
(single wrist_test episode set, 64 views, batch 4, 50 epochs, TBPTT chunk 4,
`pose_gru_lr_scale` 100, hidden 128, save_dir `captain_cut3r_sim3rmse`) differing
ONLY in the new lever keys + `exp_name`. **A4 is fixed** (pose_delta input,
residual head); **G0 and A1-A3 are retired**. The trained v2 `a4_g1` / `a4_g2`
runs are the R1/F0 controls — do not retrain them.

**G arm change:** the `*_g2` configs were converted in place to **G3 =
`bptt=True, e2e=True`**, the true superset of G1. As shipped, G2 was
`bptt=False` — G0 + e2e, *not* a superset of G1 — so a g1-vs-g2 comparison
traded multi-step credit away instead of adding the main-loss path on top of
it. See the G-lever table in `captain_gru_v2_grid_README.md`. The already-trained
`*_g2` checkpoints stay on disk under their old names as valid G0+e2e runs.

Launch one arm: `sbatch train_captain_gru_overfit.sbatch captain_gru_v3_a4_<arm>`
(2x L40S, accelerate num_processes=2 — identical to how the v2 grid ran).

| config                        | G  | F (img feats) | R | γ (train) |
|-------------------------------|----|---------------|---|-----------|
| `captain_gru_v2_a4_g1` (ctrl) | G1 | off           | 1 | —         |
| `captain_gru_v2_a4_g3` (ctrl) | G3 | off           | 1 | —         |
| `captain_gru_v3_a4_g1_r8`     | G1 | off           | 8 | 0.8       |
| `captain_gru_v3_a4_g3_r8`     | G3 | off           | 8 | 0.8       |
| `captain_gru_v3_a4_g1_f1_r8`  | G1 | 2-frame, D=32 | 8 | 0.8       |
| `captain_gru_v3_a4_g3_f1_r8`  | G3 | 2-frame, D=32 | 8 | 0.8       |

## F lever — image features into the GRU (`pose_gru_img_feat`)

`input` feeds the GRUCell the pooled **PRE-ray** image-encoder features of the
current view t and (frames=2) the previous view t-1: mean over the 240 tokens
(the `_get_img_level_feat` statistic), per-frame LayerNorm (shared affine),
then a **ZERO-INIT** `Linear(1024*frames -> D)` appended AFTER the pose/delta
block. gru_in keeps its pose-first 14-d layout; the residual anchor is
untouched. View 0's stash subtracts the constant `masked_ray_map_token` so all
frames carry the same pure-image statistic. The stash (`_prev_img_feat`) is
always-detached data — no TBPTT boundary handling. The encoder is frozen AND
TBPTT-detached: features can never backprop into it. Zero-init img_proj ⇒ an
untrained F1 model is output-identical to F0 (the cell's default-init feature
columns keep img_proj's gradient alive — do NOT zero them, e.g. in surgery).

### F SOURCE sub-lever (`pose_gru_img_feat_src`, suffix on the F arm name)

Selects WHAT produces the per-step feature; the frames/proj machinery above is
shared by every source, only the block width changes:

| suffix | src               | feature (one block)                             | width |
|--------|-------------------|-------------------------------------------------|-------|
| (none) | `pooled` (default)| mean over the CUT3R pre-ray tokens (original)   | 1024  |
| `r`    | `resnet18`        | frozen ImageNet resnet18 avgpool of the RAW img | 512   |
| `d`    | `dinov2_vits14`   | frozen DINOv2 ViT-S/14 CLS of the RAW img       | 384   |
| `c`    | `corr`            | cur<->prev token correlation/flow statistic     | 54    |

Arms: `f0r/f1r`, `f0d/f1d` (per-frame sources, F0/F1 as before) and `f1c`
(corr REQUIRES frames=2 but appends ONE 54-wide block — the previous view's
FULL token set is stashed instead of its mean). `corr` is the only source
carrying explicit relative-motion evidence (`corr_motion_stats` in
`dust3r/img_encoders.py`); the per-frame sources are global summaries the cell
must learn to compare. The frozen encoders live at `pose_gru.img_encoder.*`
(eval-locked, requires_grad=False, excluded from optimizer/DDP) and their
weights serialize into every ckpt, so offline eval needs no pretrained file —
`pose_gru_img_encoder_weights` (default `src/CUT3R/src/pretrained_encoders/`)
matters at TRAIN init only. The load_model sniff identifies the source from
key fingerprints (`img_encoder.net.conv1.weight` / `img_encoder.net.cls_token`)
or the `img_norm` width (54=corr, enc width=pooled), cross-checked against
`ckpt['args'].pose_gru_img_feat_src` with a hard fail on disagreement.
Zero-init img_proj ⇒ init-equivalence holds for every source; the
`POSE_GRU_IMG_FEAT_SHUFFLE/ZERO` falsifiers act downstream of all of them.
CPU test suite: `verify_gru_encoder_levers.py` (repo root).
`pose_gru_expand_ckpt.py` refuses source-lever checkpoints.

## R lever — iterative GRU (`pose_gru_iters`)

N back-to-back cell iterations per view, shared weights, zero new parameters
(RAFT-style). Fixed conventions (not flags):
- the pose slice re-fed each iterate is the RUNNING estimate; iterate 0 starts
  from the detached stash P(x-1);
- the delta `inv(P(x-2))·P(x-1)` and the image features are STATIC per-view
  context, re-injected unchanged at every iterate (RAFT's `inp` stream);
- the estimate is detached between iterates (`pose_gru_iter_detach: True`,
  RAFT's `coords1.detach()`); the hidden is NEVER detached inside a view —
  G1/G2 apply at the view boundary on the LAST iterate's hidden (under G2,
  which has bptt=False, a within-view hidden tape therefore still exists);
- the ray map is built ONCE, from the final iterate — G2 semantics unchanged;
- the fed-back stash stays the raw HEAD pose, detached (same as v2).
Compute cost is ~1e-5 of the per-view ray-encoder pass — R is free.

`pose_gru_iters` is INVISIBLE in weight shapes: it persists via ckpt args and
is restored by `load_model` (mis-restoring N is silent — the round-trip check
in `verify_gru_v3_levers.py` guards it). Eval-time override:
`POSE_GRU_FORCE_ITERS=<n>` — the **anytime-N probe**: run a trained R8 ckpt at
n ∈ {1,2,4,8}; monotone improvement with n ⇒ refinement learned, flat ⇒
1-step fixed point.

## Supervision (`pose_gru_iter_gamma`)

TRAIN criterion only: `PoseGRULoss(iter_gamma=0.8)` adds the RAFT sequence
loss `w_k = γ^(N-1-k)` over intermediate iterates (final iterate keeps its
weight-1.0 term — gradient scale matches R1). The TEST criterion deliberately
stays `PoseGRULoss()` (final-iterate), so best-ckpt selection and loss curves
are comparable across R arms. This breaks the v2 train==test string symmetry
BY DESIGN. `gru_trans_loss`/`gru_quat_loss` stay final-iterate-only;
`gru_pose_loss_iters` reports the weighted intermediate mass.

## Protocol

After each run, falsify against a keep_freq snapshot (never
`checkpoint-last/best.pth` while the job lives):
- `PREV_PRED_RAY_SHUFFLE=1` — pose-feedback channel
- `POSE_GRU_IMG_FEAT_SHUFFLE=1` / `POSE_GRU_IMG_FEAT_ZERO=1` — feature channel
- `POSE_GRU_HIDDEN_SHUFFLE=1` / `POSE_GRU_HIDDEN_ZERO=1` — recurrence
- `POSE_GRU_FORCE_ITERS={1,2,4,8}` — the R go/no-go curve
Code-level checks: `sbatch verify_gru_v3_levers.sbatch` (flag-off byte-identity
vs the v2 worktree, init-equivalence, iterate mechanics, feature wiring, loss
gating, ckpt round-trip + hard-fail).

## Warm starts (not used by the grid, available)

The 4 arms train from `cut3r_512_dpt_4_64.pth` exactly like v2 (comparability).
To warm-start an F arm from a trained v2 GRU instead:
`python pose_gru_expand_ckpt.py --src <v2 ckpt> --dst <out> [--iters 8]`
(copies the 14 base weight_ih columns, fresh-inits the D new ones, zero
img_proj ⇒ function-preserving at load). The loader HARD-FAILS on any
pose_gru shape mismatch — silent random-GRU eval is impossible by design.
