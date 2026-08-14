# The F lever — image features into the PoseGRU

*Scope: the `known_good_fsrc_noproj` branch. Throughout this document the A and
G levers are held fixed at **A4** and **G3** (defined below); the F lever and
all of its sub-levers (F0 vs F1, proj vs noproj, and the pooled / resnet18 /
DINOv2 / corr sources) are what varies. The companion document
`R_LEVER_EXPLAINED.md` covers the R lever.*

---

## 1. Background: the system the lever lives in

You need four pieces of context to understand anything below. None of them
require reading the code.

### 1.1 CUT3R and the closed pose loop

The base model (CUT3R, class `ARCroco3DStereo` in `src/CUT3R/src/dust3r/model.py`)
is a sequential 3D reconstruction network. It consumes a video one frame
("view") at a time, and for each view predicts, among other things, a **camera
pose**: where the camera was when that frame was taken. A pose is encoded as a
7-number vector called **absT_quaR**: 3 numbers of translation (position) plus
a 4-number quaternion (orientation). All poses are expressed **relative to
view 0** — the first frame's camera defines the coordinate system.

The training arm this project studies is the **closed loop**
(`feed_prev_pred: True`): when the model processes view *x*, it is shown the
pose *it itself predicted* at view *x−1*. That pose is not fed in as raw
numbers — it is rendered into a **ray map** (a per-pixel image of camera ray
directions, built using view *x−1*'s camera intrinsics), passed through a
frozen ray-map encoder, and *added* to view *x*'s image tokens. So the model
literally sees its own previous answer painted into the current frame's
features. View 0, which has no previous prediction, instead gets a learned
constant placeholder token (`masked_ray_map_token`).

The danger of a closed loop is drift: an error at view *x−1* contaminates the
conditioning of view *x*, and so on.

### 1.2 The PoseGRU

The **PoseGRU** (class `PoseGRU`, same file) is a small recurrent filter
inserted *between* the fed-back pose and the ray-map build. Instead of
rendering the raw fed-back pose directly, the loop first passes it through the
GRU, which may correct it, and renders the **GRU's output** instead.

Concretely it is a single `nn.GRUCell` (hidden size 128) plus a linear "head"
that maps the hidden state to a 7-d pose correction. A GRU cell is a standard
recurrent unit: at each step it takes an input vector and its own previous
**hidden state** (a 128-number memory vector), and produces a new hidden
state. The hidden state is reset to zeros at view 0 of every sequence, so
memory never leaks across videos.

The GRU is trained by an **auxiliary loss** (`PoseGRULoss` in
`dust3r/losses.py`): its output is compared against the current view's
ground-truth pose. This loss trains the GRU and *only* the GRU (everything
else it reads is detached), and — except through the G lever's e2e path
below — nothing else trains the GRU.

### 1.3 A4 (fixed): what the GRU consumes and emits

The A lever fixes the GRU's input/output contract. **A4** means:

- **Input = `pose_delta` (14-d).** The cell input's pose block is the fed-back
  pose P(x−1) *concatenated with* the relative rigid transform
  inv(P(x−2))·P(x−1) — i.e. the motion between the last two fed-back poses,
  the loop's "velocity", handed to the module explicitly. This delta is a
  proper relative transform, not a componentwise subtraction (subtracting
  quaternions is meaningless because q and −q encode the same rotation). At
  view 1, where no P(x−2) exists yet, the delta is the identity motion.
- **Output = `residual`.** The head is **initialized to all zeros** and its
  output is *added* to the input pose (then the quaternion is re-normalized).
  Consequence: an untrained PoseGRU reproduces plain `feed_prev_pred`
  bit-for-bit — the module starts as an exact no-op and must *learn* to
  deviate. This "zero-init head" fact matters repeatedly below.

### 1.4 G3 (fixed): where gradients flow

The G lever routes gradients without changing any forward computation. **G3**
means both of its switches are on:

- `pose_gru_bptt=True`: the hidden state's gradient tape is kept across the
  views of one TBPTT chunk (training processes 64 views in chunks of 4;
  "TBPTT" = truncated backpropagation through time), so a loss at a later view
  can train the GRU calls of earlier views in the same chunk.
- `pose_gru_e2e=True`: the GRU output entering the ray-map build is left
  attached, so the main reconstruction loss *also* reaches the GRU through
  the pose → ray map → frozen ray encoder chain.

Since G changes gradient routing only, a G3 checkpoint is indistinguishable
from a G0 one at eval time. Nothing in the F lever depends on G, but the
detachment story in §8 references it.

---

## 2. What the F lever is

**F asks: should the PoseGRU be allowed to *look at the images*, and if so,
through what summary?**

With F off, the GRU sees only pose numbers — the 14-d pose+delta vector. It
can smooth or extrapolate the trajectory, but it has no visual evidence about
what actually moved between frames. The F lever appends an **image feature
vector** to the cell input at every step, so the input becomes:

```
cell input = [ pose (7) | delta (7) | image features (width varies) ]
```

The pose block always comes **first** and is untouched — this matters because
the residual head anchors its correction on the first 7 dimensions of the
input, and that anchor must remain the fed-back pose.

The lever is controlled by config keys (all `pose_gru_*`):

| key | values | meaning |
|---|---|---|
| `pose_gru_img_feat` | `none` / `input` | master switch: F off / F on |
| `pose_gru_img_feat_frames` | `1` / `2` | **F0 vs F1**: current view only, or current + previous view (§4) |
| `pose_gru_img_feat_proj` | `True` / `False` | **proj vs noproj**: compress through a 32-d bottleneck, or feed raw (§5) |
| `pose_gru_img_feat_dim` | int (32) | bottleneck width when proj=True; **ignored** when proj=False |
| `pose_gru_img_feat_src` | `pooled` / `resnet18` / `dinov2_vits14` / `corr` | **source sub-lever**: what produces the feature (§6) |
| `pose_gru_img_encoder_weights` | path | pretrained-weights dir for the resnet/DINOv2 sources — matters at *training init only* (§10) |

When F is off, the feature machinery (`img_norm`, `img_proj`, `img_encoder`)
is **not built at all** — an F-off module is genuinely smaller, not a zeroed
one. (A config that sets a non-default source while F is off is refused at
construction: it would train F-off while recording a source arm in the
checkpoint args, giving one run three possible names.)

---

## 3. The feature pipeline, step by step

Everything here happens once per view, at the GRU call site inside the
decode loop (`_forward_decoder_group_step` in `model.py`).

1. **Extract** the current view's feature, per the source (§6). For the
   default `pooled` source this is: take the view's image tokens as they come
   out of CUT3R's image encoder — at the training resolution of 320×192 with
   16×16-pixel patches that is 20×12 = **240 tokens of 1024 numbers each** —
   and average them into a single 1024-d vector (the model's own
   `_get_img_level_feat` statistic). Crucially these are the **PRE-ray**
   tokens: captured *before* the previous-pose ray map is added, so the
   feature is pure image appearance with no pose information smuggled in.

2. **Pair with the previous view** (only if frames=2): the previous view's
   feature was stashed at the previous step (`self._prev_img_feat`), and the
   two are concatenated **current first, then previous**. The stash is plain
   detached data — it crosses TBPTT chunk boundaries with no special
   handling, unlike the hidden state.

3. **Normalize**: each frame's block is passed through a `LayerNorm` of the
   source width. It is a *single* LayerNorm module whose learned scale/shift
   are **shared across both frames** (the tensor is reshaped to
   `(batch, blocks, width)`, normalized, and flattened back). LayerNorm is
   applied in *both* projector modes — it is feature conditioning, not part
   of the projection. Raw encoder activations are not unit-scaled, and
   feeding 1024–2048 unnormalized columns into a GRU cell whose input weights
   are initialized for O(1) inputs would swamp the 14 pose columns that carry
   the residual anchor.

4. **Project — or don't** (§5): with proj=True the normalized features go
   through a zero-initialized `Linear(width × blocks → 32)`; with proj=False
   they enter the cell raw.

5. **Concatenate** after the pose block and run the cell.

6. **Stash** the current view's (honest, never-falsified) feature as
   `_prev_img_feat` for the next step.

Two structural properties hold for every source:

- **The features are data, never a gradient path.** The producing network is
  frozen, and the feature tensors are explicitly `.detach()`-ed. No loss can
  ever backpropagate into the image encoder through the F lever.
- **The GRU block runs in float32** with mixed-precision autocast disabled —
  a pose correction is a handful of precision-sensitive scalars.

---

## 4. The frames sub-lever: F0 vs F1 — and a naming trap

`pose_gru_img_feat_frames` selects *which* frames contribute:

| arm | frames | what the cell sees | raw width (pooled source) |
|---|---|---|---|
| **F0** | 1 | current view *t* only | 1024 |
| **F1** | 2 | current view *t* + previous view *t−1* | 2048 |

The motivation for F1: a *single* frame's global appearance says little about
camera motion, but a *pair* lets the cell compare two appearances and infer
that something moved. (Whether a GRU can actually do that from two pooled
global vectors is exactly what the source sub-lever's `corr` arm probes —
see §6.4.)

Note that under F0, the previous-view stash is still *written* every step; it
is simply never consumed. This keeps the code path uniform.

**⚠ The naming trap.** "F0" has meant two different things over the project's
history:

- In the **older v3 grid README**, "F0" meant *F entirely off* (the
  feature-less control arms, e.g. `captain_gru_v3_a4_g3_finetune`).
- In the **PoseGRU docstring convention used by every newer config** —
  and in this document — "F0" means *F ON with frames=1*.

So `captain_gru_v3_a4_g3_f0_noproj_finetune` is **not** a control; it feeds
the current view's 1024-d feature into the cell. The true F-off control is
always `captain_gru_v3_a4_g3_finetune` (or the v2 `a4_g3` arm at overfit
scale). Config headers on this branch call this out explicitly ("NAMING —
READ FIRST").

---

## 5. The projection sub-lever: proj vs noproj

`pose_gru_img_feat_proj` decides whether the normalized features are squeezed
through a bottleneck before entering the cell.

### 5.1 proj=True (the original arms)

A `Linear(source_width × blocks → 32)` whose **weight and bias are both
initialized to exactly zero** maps the features to 32 columns. Cell input
width: 14 + 32 = **46**. For the pooled F1 arm this adds `img_proj`
(2048×32 + 32 = 65,568 params) and `img_norm` (2×1024 = 2,048 params) and
widens the cell's input weight matrix `weight_ih` to 3·128×46 = 17,664
params; the whole PoseGRU is 136,103 params vs the F-off module's 56,199.

The zero-init projector has a precise purpose: **at initialization the
appended columns are exactly zero**, so an untrained F1 model computes the
*identical hidden-state trajectory* to an F-off model — the F lever
provably contributes nothing until training moves the projector. F on/off is
therefore a clean single-variable comparison even at step 0.

Why doesn't a zero projector get stuck (zero output ⇒ zero gradient)? Because
the *cell's* input columns for the feature block keep their default random
initialization, and the gradient to `img_proj.weight` flows through those
columns. Corollary, stated in the docstring as a warning: **never zero the
cell's feature columns** (e.g. during checkpoint surgery) — that would kill
`img_proj`'s gradient permanently.

The cost of the bottleneck: the module is *forced* to summarize appearance
into 32 numbers. If pose-relevant visual evidence doesn't survive that
compression, F reads as useless when the real culprit is the bottleneck.

### 5.2 proj=False (the "noproj" arms — this branch's namesake)

There is **no Linear at all** (`img_proj` is never constructed). The
LayerNormed features enter the cell raw:

| arm | cell input width | `weight_ih` params |
|---|---|---|
| F-off | 14 | 5,376 |
| F0/F1 proj (any per-frame source) | 46 | 17,664 |
| **F0 noproj** (pooled) | 14 + 1024 = **1038** | **398,592** |
| **F1 noproj** (pooled) | 14 + 2048 = **2062** | **791,808** |

`pose_gru_img_feat_dim` is **ignored** in this mode — the appended width is
dictated by source width × blocks. The key is kept in the configs anyway so
flipping proj back on is a one-word change. (The module attribute
`img_feat_dim` always reports the width the cell *actually* receives.)

The point of noproj: remove the bottleneck as a suspect. The cell's own input
matrix gets full-rank access to the appearance vector; nothing forces a
32-d summary. The price is ~45×/22× more input-weight parameters and a
weaker init story:

**Init-equivalence is weaker under noproj, and for a different reason.**
There is no zero gate anywhere on the feature path, so image content reaches
the hidden state **from step one** through the cell's default-init input
columns — the hidden trajectory differs from F-off immediately. What still
holds is *output* equivalence: because A4's residual head is zero-init, the
module's output is the unchanged fed-back pose regardless of what the hidden
is doing. (Under `mode="direct"` there would be no init-equivalence at all in
this mode; with A4 fixed this doesn't arise.)

**The bootstrap gate (bites silently, not a defect).** In both proj modes,
the zero-init head is the gradient gate: while `head.weight` is zero, the
gradient reaching the cell (and `img_proj`/`img_norm`) is *exactly* zero —
only the head's own weight and bias move at first. Until the head "unlocks"
(becomes nonzero), the image features are provably inert. Implications:

- An early "F does nothing" reading is an **artifact**, not a result.
- Do **not** run the `POSE_GRU_IMG_FEAT_*` falsifiers (§9) on a checkpoint
  from before the head unlocked — they will trivially show no effect.
- The finetune configs' 10× learning rate (peak 1e-5, with the GRU group
  scaled a further 100× to 1e-3 — see §8) shortens this window considerably.

### 5.3 What the noproj arms price (the comparison design)

Each config is a single-variable change against a named twin:

| comparison | isolates |
|---|---|
| `f1_noproj_finetune` vs `f1_finetune` (frames=2, proj flips) | **the 32-d bottleneck itself** |
| `f1_noproj_finetune` vs `f0_noproj_finetune` (proj=False, frames flips) | **the previous-frame contribution** at raw access |
| either vs `captain_gru_v3_a4_g3_finetune` (F off) | the F lever as a whole |

---

## 6. The source sub-lever: what produces the feature

`pose_gru_img_feat_src` selects *what network computes* the per-step feature.
The frames/LayerNorm/proj machinery of §3–5 is shared by every source; only
the block width changes. Sources are suffixed onto the arm name:

| suffix | source | feature (one block) | width | learned params? |
|---|---|---|---|---|
| *(none)* | `pooled` (default) | mean over CUT3R's 240 pre-ray tokens | 1024 | no (reuses the backbone) |
| `r` | `resnet18` | frozen ImageNet ResNet-18, avgpool output, on the **raw image** | 512 | no (frozen, 11.7M params) |
| `d` | `dinov2_vits14` | frozen DINOv2 ViT-S/14, final-norm CLS token, on the **raw image** | 384 | no (frozen) |
| `c` | `corr` | hand-crafted correlation/flow statistic **between** the two views' token sets | 54 | no (pure math) |

So the arm names compose as: `f1` = pooled/frames2, `f1r` = resnet/frames2,
`f0d` = DINOv2/frames1, `f1c` = corr, plus `_noproj` for proj=False. The
implemented code lives in `src/CUT3R/src/dust3r/img_encoders.py`.

### 6.1 `pooled` — the model's own tokens

Described in §3. Its virtue is being free (the tokens already exist); its
weakness is that CUT3R's encoder was never trained to make its *global mean*
informative about camera motion, and averaging 240 tokens destroys all
spatial structure.

### 6.2 `resnet18` — an independent CNN's global appearance

A torchvision ResNet-18 with full ImageNet-pretrained weights, classifier
head replaced by identity so the forward returns the 512-d global-average-pool
vector. Run on the **raw image** (not CUT3R tokens): the loader's [−1, 1]
normalization is mapped back to [0, 1] and then ImageNet-normalized. The
point of this arm: test whether the *pooled-CUT3R* feature is the problem —
maybe a representation trained for recognition carries different (better or
worse) evidence.

### 6.3 `dinov2_vits14` — a self-supervised ViT's global token

The official DINOv2 ViT-S/14 (architecture vendored under
`dust3r/vendor/dinov2` so no internet or hub access is needed), returning the
384-d CLS token after the final norm. Because its patch size is 14, the
320×192 training image is bilinearly resized (with antialiasing) up to the
nearest multiple of 14 — 322×196 — before encoding. Same rationale as
resnet18, with a self-supervised representation instead of a supervised one.

Both encoder sources share hard freezing guarantees:

- Every parameter is `requires_grad=False`, and the module is excluded from
  the optimizer and from DDP gradient machinery.
- The module is **eval-locked**: its `train()` method is overridden to a
  no-op that keeps it in eval mode, because a parent `.train()` call would
  otherwise flip ResNet's BatchNorm into batch-statistics mode and silently
  update its running estimates.
- The forward runs inside a `no_grad` + autocast-disabled fp32 island and
  returns a detached fp32 vector.
- The **weights serialize into every checkpoint** (`pose_gru.img_encoder.*`),
  so offline evaluation reconstructs bit-exactly from `ckpt["model"]` alone.
  The pretrained file (`src/CUT3R/src/pretrained_encoders/`, overridable via
  `pose_gru_img_encoder_weights`) is needed at **training init only**; on the
  cluster it must be fetched once from a login node (compute nodes have no
  internet), and the code raises a clear error with the exact `curl` command
  if it is missing.
- The ImageNet mean/std tensors are registered as *non-persistent* buffers —
  constants, deliberately kept out of the checkpoint.

### 6.4 `corr` — the only source with explicit motion evidence

All three sources above are **per-frame global summaries**: even under F1 the
cell receives two independent descriptions and must *learn to compare* them.
`corr` inverts that: it computes, in pure tensor math with **zero learned
parameters**, a 54-d correlation/flow statistic *between* the current and
previous views' CUT3R pre-ray token sets (`corr_motion_stats`):

- For each of the 240 current tokens, find its best-matching previous token
  by cosine similarity (argmax over all 240 previous tokens). That match
  defines a displacement (dx, dy) on the 20×12 token grid — normalized by
  grid size, in the current→previous direction (approximately *negative*
  optical flow) — and a match confidence m (the max cosine similarity).
- Aggregate: the grid is partitioned into 4×4 cells and each cell reports
  its mean (dx, dy, m) → 48 numbers; then 6 global numbers (mean and std of
  dx, dy, m) are appended. 48 + 6 = **54**.

Special-cases that follow from being a *pair* statistic:

- `corr` **requires frames=2** (asserted at construction — the arm is always
  `f1c`, an `f0c` cannot exist), but it appends **one** 54-wide block, not
  two. The code distinguishes `img_feat_frames` (semantic: is the previous
  view consumed? = 2) from `img_feat_blocks` (how many width-blocks the
  LayerNorm reshape and projector width follow; = 1 for corr, = frames
  everywhere else).
- The stash must keep the previous view's **full 240×1024 token set**, not
  its mean (~1 MB per sample of detached fp32 data), because the statistic
  needs per-token correspondence.
- The argmax is non-differentiable — harmless, since the inputs are detached
  data by construction.
- The statistic is computed under autocast-disabled fp32. This is not
  pedantry: matmul is autocast-listed, so without the guard the feature would
  compute in fp16 during training but fp32 at eval — a silent train/eval
  feature mismatch.

**Known limitation, documented in the code:** m is a *max* cosine similarity,
not a two-peak match confidence. A textureless region matches many previous
tokens at m ≈ 1, so its argmax displacement is arbitrary while m still reads
"confident". The GRU has to learn that per-cell (dx, dy) is only meaningful
where tokens are distinctive. A peak-ratio confidence would fix this at the
statistic level, but it changes the feature width — that would be a new arm,
not a patch to this one.

---

## 7. View 0 and the masked-ray-token correction

There is a subtle contamination hazard at sequence start. View 0's tokens, as
they reach the GRU call site, already carry the constant
`masked_ray_map_token` added by the encoder stage (the "no previous pose yet"
placeholder). If view 0's feature were stashed as-is, then at view 1 the
"previous frame" feature would differ from every later step's not because the
image differs but because of a constant additive offset — a systematic
first-step artifact.

So, for the **token-based sources** (`pooled` and `corr`), view 0's stash
subtracts the `masked_ray_map_token` before pooling/stashing, making the
stash the same pure-image statistic every other view provides. The **encoder
sources** (resnet18, DINOv2) read the raw image, which never carries the ray
token — nothing to correct.

---

## 8. The gradient story, in one place

- The image features are **structurally detached** on every path: frozen
  producers, explicit `.detach()`, stash is data. The F lever can never train
  the image encoder — F changes what the GRU *sees*, never what the backbone
  *learns*.
- The **zero-init residual head** is the single gradient gate for the whole
  GRU (§5.2). All F machinery is inert until it unlocks.
- The GRU's trainable parameters — including `img_norm` and `img_proj` —
  live in a dedicated optimizer group with learning rate multiplied by
  `pose_gru_lr_scale` (default 100). This is applied *on top of* the config's
  base lr (`split_pose_gru_param_groups` in `train_cut3r_baseline.py`):
  rationale, a from-scratch module at a finetune's tiny lr would stay
  effectively dead. In the lr=1e-5 finetune family the GRU group therefore
  peaks at **1e-3**. The frozen `img_encoder` parameters are excluded from
  the optimizer entirely. When comparing arms, keep `pose_gru_lr_scale`
  identical across them or the comparison stops being single-variable.

---

## 9. Falsifiers — how "F is used" is actually tested

The evaluation protocol never trusts a metric improvement alone; each channel
into the GRU has an environment-variable "falsifier" that corrupts *only that
channel* at eval time, on a trained checkpoint. If corrupting a channel does
not degrade metrics, the model provably ignores it.

| env var | corrupts | how |
|---|---|---|
| `POSE_GRU_IMG_FEAT_SHUFFLE=1` | **the F lever** | batch-roll: every sample sees another sample's image pair (rolled coherently; pose feedback and supervision stay honest). Needs batch > 1. |
| `POSE_GRU_IMG_FEAT_ZERO=1` | **the F lever** | features replaced by zeros — the featureless control; works at batch 1. |
| `PREV_PRED_RAY_SHUFFLE=1` | pose feedback | rolls *both* pose stashes coherently, so the (pose, delta) pair stays consistent per (wrong) source sample. |
| `POSE_GRU_HIDDEN_SHUFFLE=1` / `POSE_GRU_HIDDEN_ZERO=1` | recurrence | roll / zero the hidden state. |

Protocol rules: run falsifiers only against a `keep_freq` snapshot (never
`checkpoint-last/best.pth` while the job is alive — the file may be
mid-write), and never before the zero-init head has unlocked (§5.2). The
shuffle falsifiers stash-roll only the *consumed* copy; the stash that is
written for the next step stays honest.

---

## 10. Checkpoints: serialization, sniffing, hard-fail, warm starts

**Everything needed to rebuild the module is in the checkpoint.** The F
configuration is recoverable from the weights themselves, and `load_model`
(the eval entry point) *sniffs* it (`_sniff_pose_gru_config` in `model.py`)
rather than trusting external configuration:

- `img_norm.weight` exists **iff** F is on.
- `img_proj.weight` exists **iff** proj=True (its output width gives the
  bottleneck dim, its input width the source width × blocks).
- The source is fingerprinted by key names (`img_encoder.net.conv1.weight` ⇒
  resnet18; `img_encoder.net.cls_token` ⇒ DINOv2) or, failing those, by the
  `img_norm` width (54 ⇒ corr; encoder width ⇒ pooled).
- frames is recovered uniquely from the cell input width
  (14 + blocks × width).
- The sniffed source is **cross-checked** against the checkpoint's stored
  training args, with a **hard fail on disagreement**.

Because each F arm has different weight shapes, checkpoints are mutually
shape-incompatible across arms — and the loader **hard-fails on any
`pose_gru` shape or key mismatch** instead of skipping tensors. This is a
deliberate design goal stated in the code: *silently evaluating a random GRU
is impossible.* (For non-GRU weights the loader's legacy fallback does skip
mismatches with a log line; the GRU is exempted from that leniency.)

At load time the sniff passes `img_encoder_pretrained=False`: the frozen
encoder is built with random init and then overwritten by the checkpoint's
own serialized weights — so eval nodes never need the pretrained files.

**Warm starts.** `pose_gru_expand_ckpt.py` can graft a trained v2 (F-off)
GRU into a projected-F shape: it copies the 14 base input columns, fresh-inits
the 32 new ones, and zeroes `img_proj`, which is function-preserving at load.
It **refuses** source-lever checkpoints and **refuses noproj destinations**
(it will not splice a projector onto — or invent 1024 raw columns for — a
proj=False checkpoint). Consequently *no warm start exists for the noproj
arms*; they train from base CUT3R like every other v3 grid arm, which is
precisely what keeps the arm comparison clean. None of the grid arms actually
use warm starts, for the same comparability reason.

**Verification.** `verify_gru_encoder_levers.py` (CPU test suite for the
source lever) and `verify_gru_v3_levers.py` / `.sbatch` (flag-off
byte-identity vs the v2 code, init-equivalence, feature wiring, checkpoint
round-trip including the hard-fail) cover this machinery;
`verify_gru_f0_proj.py` covers the noproj round-trip.

---

## 11. The config inventory on this branch (A4 × G3 slice)

All full-dataset finetunes: 32 GPUs, DROID splits (38,633 train / 4,292 test
scenes), 64 views/sequence, batch 4/GPU, lr 1e-5, `pose_gru_loss_weight` 1.0,
`ddp_grad_sync: false` (kept explicitly for comparability with the family —
see the memory note on the TBPTT/DDP no-sync behavior).

| config (`captain_gru_v3_a4_g3_…`) | frames | proj | source | R | cell input |
|---|---|---|---|---|---|
| `…finetune` (F **off** — the control) | — | — | — | 1 | 14 |
| `…f1_finetune` | 2 | ✓ | pooled | 1 | 46 |
| `…f0_noproj_finetune` | 1 | ✗ | pooled | 1 | 1038 |
| `…f1_noproj_finetune` | 2 | ✗ | pooled | 1 | 2062 |
| `…f1_r8_finetune` | 2 | ✓ | pooled | 8 | 46 |
| `…f1r_r8_finetune` | 2 | ✓ | resnet18 | 8 | 46 |
| `…f1d_r8_finetune` | 2 | ✓ | dinov2 | 8 | 46 |
| `…f1c_r8_finetune` | 2 | ✓ | corr | 8 | 46 |
| `…r8_finetune` (F off, R8) | — | — | — | 8 | 14 |

(Overfit-scale twins of most arms exist under `a4_g1`/`a4_g3` without the
`_finetune` suffix, including the frames=1 source arms `f0r_r8`/`f0d_r8`;
`_smoke` configs are short shakedown runs.)

One cross-lever caveat when reading F comparisons: the R8 arms multiply the
auxiliary-loss gradient mass by ×4.1611 relative to R1 (see
`R_LEVER_EXPLAINED.md` §6), so an F comparison is only clean *within* a fixed
R (and fixed `pose_gru_loss_weight`), and the F1×R1 arm exists precisely to
price F alone with both confounds removed.
