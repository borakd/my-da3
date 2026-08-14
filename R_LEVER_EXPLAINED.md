# The R lever — iterative refinement inside the PoseGRU

*Scope: the `known_good_fsrc_noproj` branch. Throughout this document the A
and G levers are held fixed at **A4** and **G3**; the R lever
(`pose_gru_iters`) is what varies. The companion document
`F_LEVER_EXPLAINED.md` covers the F lever and contains the fuller system
background (§1 there); the short version needed here is repeated below so
this file stands alone.*

---

## 1. Background in brief

The base model (CUT3R) reconstructs a video one frame ("view") at a time and
predicts a **camera pose** per view — a 7-number encoding: 3 of translation
plus a 4-number quaternion of rotation, all relative to view 0's camera. In
the **closed-loop** training arm (`feed_prev_pred`), view *x* is conditioned
on the pose the model itself predicted at view *x−1*, rendered into a
per-pixel **ray map** and added to view *x*'s image tokens.

The **PoseGRU** sits between that fed-back pose and the ray-map build: a
single 128-unit `nn.GRUCell` plus a linear head, which may correct the
fed-back pose before it is rendered. It is trained by an auxiliary loss
(`PoseGRULoss`) against the current view's ground-truth pose.

The fixed levers:

- **A4** — the cell input is 14-d: the fed-back pose P(x−1) concatenated
  with the relative transform inv(P(x−2))·P(x−1) (the loop's "velocity");
  the head is **zero-initialized** and outputs a *residual* added to the
  input pose, so an untrained module is an exact no-op.
- **G3** — gradient routing only (forward math identical to every other G
  arm): the hidden state's tape spans a 4-view TBPTT chunk, *and* the GRU
  output entering the ray build stays attached so the main reconstruction
  loss also trains the GRU.
- The F lever may be on or off orthogonally; where it interacts with R it is
  called out below.

---

## 2. What the R lever is

**R asks: does the GRU get one shot at correcting the pose, or several?**

`pose_gru_iters = N` runs the *same* GRU cell **N times back-to-back within a
single view**, each iteration re-feeding the previous iteration's pose
estimate as input. R1 (one iteration) is the classic setup; R8 gives the
module eight refinement steps per view.

**Zero new parameters.** The same cell and head are re-entered N times, so an
R8 checkpoint has *exactly* the same parameter count and weight shapes as its
R1 twin (56,199 params for the F-off module). This has a consequence that
runs through everything below: **R is invisible in the weights** (§7).

The design is a deliberate transplant of **RAFT's iterative update
operator** — the repo's own comments cite RAFT by name at the call site
(`model.py`), in the loss (`losses.py`), and in the grid README. So before
the PoseGRU loop itself, here is the RAFT code it copies from, and then the
loop with the parallels drawn line by line.

---

## 3. The RAFT blueprint (the code this lever transplants)

RAFT (Teed & Deng, ECCV 2020, "Recurrent All-Pairs Field Transforms for
Optical Flow", `princeton-vl/RAFT`) estimates **optical flow** — a per-pixel
displacement field between two images. Its central idea: instead of
predicting flow in one shot, maintain a *running estimate* and apply a small
recurrent **update operator** N times with **shared weights**, each pass
emitting a correction. Every quoted block below is verbatim from the RAFT
repository.

### 3.1 The iterative loop (`core/raft.py`, `RAFT.forward`)

Setup — a context encoder splits its output into the GRU's initial hidden
state `net` and a static context stream `inp`, and the flow estimate starts
at zero (`coords0` is the identity pixel grid; `coords1` is the running
"where did each pixel go" grid):

```python
cnet = self.cnet(image1)
net, inp = torch.split(cnet, [hdim, cdim], dim=1)
net = torch.tanh(net)
inp = torch.relu(inp)

coords0, coords1 = self.initialize_flow(image1)
```

The loop:

```python
flow_predictions = []
for itr in range(iters):
    coords1 = coords1.detach()                    # (a) cut the tape on the estimate
    corr = corr_fn(coords1)                       # (b) re-index evidence at the estimate
    flow = coords1 - coords0                      # (c) current flow estimate as input
    net, up_mask, delta_flow = self.update_block(
        net, inp, corr, flow)                     # (d) shared-weight recurrent update
    coords1 = coords1 + delta_flow                # (e) residual: estimate += correction
    ...
    flow_predictions.append(flow_up)              # (f) keep EVERY iterate for the loss
```

The pieces to hold on to:

- **(a)** the running estimate is detached at the top of every iteration —
  gradients never flow through the chain of estimate feedback, only through
  the hidden state `net`. This is the famous `coords1.detach()`.
- **inp** is computed once and re-fed unchanged every iteration — a static
  per-pair context stream.
- **(b)** the correlation features are **re-looked-up every iteration** at
  the current estimate (`corr_fn(coords1)` indexes a precomputed all-pairs
  correlation volume at the currently-estimated correspondences) — the
  evidence itself is re-read as the estimate moves.
- **(d)/(e)** one shared update block, residual output.
- **(f)** all intermediate predictions are kept, because the loss supervises
  every one of them (§3.3).

### 3.2 The update block (`core/update.py`)

The update block wraps a **convolutional GRU** — a standard GRU cell whose
matrix multiplies are replaced by convolutions so it runs on a 2-D feature
map (RAFT's separable variant does a 1×5 pass then a 5×1 pass):

```python
class SepConvGRU(nn.Module):
    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        # ... (same again with 5x1 convolutions) ...
        return h

class BasicUpdateBlock(nn.Module):
    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)          # hidden state threaded through iterations
        delta_flow = self.flow_head(net)  # linear-ish head reads the correction off h
        ...
        return net, mask, delta_flow
```

Note the shape: *hidden in, hidden out, correction read off the hidden by a
small head*. `nn.GRUCell(input, hidden)` computes exactly the `z`/`r`/`q`
algebra above with plain matrix multiplies — the PoseGRU's cell **is** this
update block with convolutions degenerated to vectors, and `head` playing
`flow_head`.

### 3.3 The sequence loss (`train.py`)

RAFT supervises **every** iterate, with exponentially decaying weights so
early (rough) iterates count less and the final iterate counts fully:

```python
def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    ...
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()
```

`gamma**(n_predictions - i - 1)` is the exact formula the R lever reuses
(§8): the final iterate gets γ⁰ = 1.0, the one before γ¹, and so on. Note —
this will matter — RAFT's default γ is 0.8, the same value every R arm on
this branch uses.

### 3.4 SEA-RAFT (the modern successor, for contrast)

SEA-RAFT (Wang, Lipson & Deng, ECCV 2024, `princeton-vl/SEA-RAFT`) keeps the
same skeleton but changes the trimmings. Its loop:

```python
# BEFORE the loop: regress an actual initial flow from context (not zero)
flow_update = self.flow_head(net)
flow_8x = flow_update[:, :2]

for itr in range(iters):
    flow_8x = flow_8x.detach()                        # same detach convention
    coords2 = (coords_grid(N, H, W, ...) + flow_8x).detach()
    corr = corr_fn(coords2, dilation=dilation)        # same per-iterate re-lookup
    net = self.update_block(net, context, corr, flow_8x)
    flow_update = self.flow_head(net)
    flow_8x = flow_8x + flow_update[:, :2]            # same residual accumulation
```

Differences from classic RAFT: it **regresses a non-zero initial estimate**
before iterating (classic RAFT starts from zero flow); its head also emits a
mixture-of-Laplace uncertainty (`info_8x`) used in the loss; and it is built
to need far fewer iterations. Where the R lever sides with SEA-RAFT rather
than classic RAFT: **the PoseGRU also starts from a meaningful initial
estimate, not zero** — iterate 0 consumes the fed-back pose P(x−1), which in
a closed loop is already a strong prior (frame-to-frame motion is small).
The uncertainty head has no counterpart in the R lever.

---

## 4. The loop, exactly — with the parallels drawn

Per view, at the call site in `_forward_decoder_group_step` (`model.py`),
after the F-lever features and the A4 delta have been computed once:

```python
est = P(x-1)                                  # detached fed-back pose stash
for k in range(N):
    gru_in = concat([est, delta_static])      # A4: pose 7-d + delta 7-d
    out_k, hidden = cell(gru_in, hidden, img_feat=img_feat_vec)
    gru_iterates.append(out_k)
    if k+1 < N:
        est = out_k.detach()                  # pose_gru_iter_detach
ray map is built from out_{N-1} ONLY          # the final iterate
```

Line-by-line against §3.1:

| RAFT (`core/raft.py`) | PoseGRU R lever | correspondence |
|---|---|---|
| `coords1` (running flow estimate) | `est` (running pose estimate) | the state being refined |
| `coords0, coords1 = initialize_flow(...)` — zero init | `est = P(x-1)` — the fed-back pose | **SEA-RAFT-style non-zero init** (§3.4), not classic RAFT's zero start |
| `coords1 = coords1.detach()` | `est = out_k.detach()` (`pose_gru_iter_detach=True`) | identical convention: gradient reaches earlier iterates through the hidden only |
| `inp` (static context stream, computed once) | `delta_static` **and** `img_feat_vec` (computed once per view) | identical: per-target context re-injected unchanged every iteration |
| `corr = corr_fn(coords1)` — **re-indexed at the current estimate every iteration** | **no counterpart** — nothing is re-read as the estimate moves | **deliberate divergence, see below** |
| `net` (hidden, threaded through iterations) | `hidden` (threaded through iterations, never detached inside a view) | identical inside a view; extra temporal role across views (below) |
| `SepConvGRU` (conv GRU on a 2-D map) + `flow_head` | `nn.GRUCell` (vector GRU) + `head` | same algebra; convolutions degenerate to matrix multiplies because a pose is 7 numbers, not a field |
| `coords1 = coords1 + delta_flow` | residual head: `pose = gru_in[:, :7] + head(hidden)` (then quat renormalize) | identical residual discipline (this is the A-lever's `residual` mode — under A4 it holds at every iterate) |
| `flow_predictions.append(flow_up)` | `gru_iterates.append(out_k)` → `gru_pose_iters` (N, batch, 7) | identical: every iterate kept for the sequence loss |
| final prediction = last iterate | ray map built from `out_{N-1}` only | identical: intermediates never touch the output/conditioning |

Now the conventions themselves, each with its rationale (these are *fixed*,
not flags, except where a key is named):

1. **The pose slice is the running estimate.** Iterate 0 starts from the
   detached stash P(x−1); each later iterate consumes the previous iterate's
   output. Only this slice of the input evolves across iterations — RAFT's
   `coords1`.

2. **The delta and the image features are static per-view context** —
   RAFT's `inp` stream. Both are computed **once** per view and re-injected
   *unchanged* at every iterate. Two implications: the delta is always the
   velocity between the last two *raw fed-back* poses (never recomputed from
   intermediate estimates), and under F the appearance vector is identical
   across all N iterates, so the iterations refine the pose against a fixed
   visual context rather than re-reading the image.

3. **The estimate is detached between iterates** (`pose_gru_iter_detach:
   True`; the key exists but every config keeps it True — at R1 it is inert
   since there is nothing between iterates to detach). Gradient therefore
   reaches earlier iterates **through the hidden state only**, never through
   an N-deep chain of pose feedback. This is `coords1.detach()` verbatim,
   and it is what keeps an 8-deep unroll numerically stable — in both
   systems the recurrent state is the gradient highway and the estimate is
   data.

4. **The hidden is never detached inside a view.** It threads through all N
   iterates as one recurrent chain — the N calls are *not* independent.
   Detaching happens only at the **view boundary**, governed by the G lever
   (under G3's `bptt=True` the tape also survives view boundaries within a
   TBPTT chunk). Side effect noted in the README: even under a `bptt=False`
   arm, a *within-view* hidden tape still exists when N > 1.

5. **The ray map is built once, from the final iterate.** Intermediate
   estimates never touch the conditioning; G-lever ("e2e") semantics are
   unchanged. Under G3 the final iterate enters the ray build attached, so
   the main reconstruction loss reaches the GRU through it. (RAFT likewise
   reports its last iterate as *the* flow.)

6. **The fed-back stash stays the raw head pose.** What is stashed for the
   *next* view's input is the model's own pose-head prediction (detached),
   not the GRU's refined pose — identical to the v2 (R1) behavior. The GRU's
   influence on the next step travels only through the ray map it
   conditioned and through its own hidden state.

7. **The whole loop runs in float32** with autocast disabled, regardless of
   the ambient mixed-precision setting — pose corrections are a handful of
   precision-sensitive scalars. (RAFT instead runs its update block *under*
   autocast — at flow's pixel scale that is safe; at quaternion scale it is
   not.)

### 4.1 Where the transplant genuinely differs from RAFT

Three structural differences, all worth keeping in mind when reasoning about
what R can and cannot learn here:

- **No per-iterate evidence lookup.** RAFT's iterations get *fresh
  information* each pass: `corr_fn(coords1)` re-indexes the correlation
  volume at the current estimate, so a better estimate literally retrieves
  better-matched evidence, which is what makes its refinement loop so
  effective. The PoseGRU's iterations receive the **same** input evidence
  every pass (static delta + static image features); only the pose slice
  changes. There is no mechanism by which a better intermediate pose fetches
  new measurements. The closest thing in the project is the F lever's `corr`
  *source* — a correlation/flow statistic between the two views' token sets
  (see `F_LEVER_EXPLAINED.md` §6.4) — but it too is computed **once per
  view** from the images, not re-indexed at the running pose. So the R lever
  tests a strictly weaker hypothesis than RAFT's: *can more shared-weight
  compute against fixed evidence refine the pose*, not *can re-grounded
  evidence lookups refine it*.

- **The hidden state has a second, temporal job.** RAFT's `net` is
  re-initialized from the context encoder for every image pair; it lives
  only inside one refinement loop. The PoseGRU's hidden is **also the
  cross-view memory** of the closed loop — it persists from view to view
  (reset only at view 0) and carries whatever trajectory history the module
  has accumulated. The R loop threads that same vector. So iterations here
  both refine the current pose *and* rewrite the temporal memory N times per
  view.

- **Scale of the update operator.** RAFT refines a dense H×W×2 field with a
  convolutional GRU over feature maps; the PoseGRU refines 7 numbers with a
  128-unit vector GRU. That is why R's compute cost is negligible (§6) —
  and why the residual head, not capacity, is the binding constraint early
  in training (§9).

---

## 5. R changes forward math (unlike G)

Unlike the G lever — which is pure gradient routing and invisible at eval —
**R changes the forward computation**: an R8 model computes a genuinely
different pose than the same weights run at R1. That is exactly what the
anytime-N probe (§7) exploits, and it mirrors RAFT's own headline property:
RAFT can be run at any iteration count at test time ("anytime inference"),
and more iterations monotonically improve a healthy model.

---

## 6. Cost

Effectively zero. The loop re-runs a 128-unit GRU cell on a ≤2062-wide input
N−1 extra times; the config headers put the whole thing at roughly **1e-5 of
one view's ray-encoder pass**. R is a free lever at train and eval time —
its price is paid elsewhere: the supervision-scale confound (§8) and the
bootstrap subtleties (§9). (Contrast RAFT, where iterations dominate runtime
— its N is a real speed/accuracy dial; SEA-RAFT's redesign was largely about
needing fewer of them.)

---

## 7. R is invisible in the weights — persistence and the anytime-N probe

Because iterating adds no parameters, nothing in a checkpoint's weight shapes
reveals N. Instead:

- `iters` is stored on the module and **persisted via the checkpoint's
  training args**; `load_model` restores it when rebuilding the module (the
  sniff, `_sniff_pose_gru_config`, reads `pose_gru_iters` from
  `ckpt["args"]`).
- Mis-restoring N would be **silent** — the weights load fine either way and
  the model just computes something else. The checkpoint round-trip check in
  `verify_gru_v3_levers.py` guards this.

**`POSE_GRU_FORCE_ITERS=<n>`** overrides the count at eval time. This is the
**anytime-N probe**, the R lever's go/no-go instrument — the direct analogue
of evaluating RAFT at different `iters`: take one trained R8 checkpoint and
evaluate it at n ∈ {1, 2, 4, 8}.

- **Monotone improvement with n** ⇒ the module genuinely learned iterative
  refinement (each pass adds value, RAFT-style).
- **Flat curve** ⇒ the module collapsed to a one-step fixed point; the
  iterations are decoration.

The probe is *legal* on an R1 checkpoint too (N isn't in the weights), but
there it is an out-of-distribution stress test — the model never trained an
iterate re-feed — not a verdict on R.

---

## 8. Supervision: the γ-weighted sequence loss — and the confound it creates

### 8.1 Mechanics

`PoseGRULoss` (in `dust3r/losses.py`) compares a predicted pose against the
GT pose in view-0-relative coordinates, as translation error + quaternion
error, with translations normalized per side exactly like the main pose
loss: GT by the GT scene scale, the GRU output by the scale of the *head's
own predicted* poses — so the GRU is taught the true pose expressed in the
model's current scene scale, which is the scale the ray-map build consumes.

The R-lever extension is the constructor's `iter_gamma` (config key
`pose_gru_iter_gamma`, train criterion only), and it is RAFT's
`sequence_loss` from §3.3 re-expressed:

| RAFT `sequence_loss` | `PoseGRULoss(iter_gamma=γ)` |
|---|---|
| `flow_preds` (list, one per iterate) | `gru_pose_iters` ((N, batch, 7), final slice == `gru_pose`) |
| `i_weight = gamma**(n_predictions - i - 1)` | intermediate iterate k gets weight **γ^(N−1−k)** |
| final prediction inside the same sum at weight 1.0 | final iterate as a **separate** weight-1.0 term (the value the R1 arms already compute) |
| `(flow_preds[i] - flow_gt).abs()` per pixel | translation + quaternion error per view, same normalization/masks for every iterate |
| default `gamma=0.8` | every R arm on this branch uses γ = 0.8 |

The split bookkeeping (final term separate, intermediates added on top) is
mathematically identical to RAFT's single sum; it exists so that the final
iterate's gradient scale exactly matches an R1 run, and so γ = 0 degrades to
*exactly* the R1 code path:

- **γ = 0.0 (R1 arms):** the final-only path. The gate is on the *value*,
  not on key presence — even if `gru_pose_iters` were attached, γ=0 ignores
  it. The loss also *reports its name* as `PoseGRULoss()` in this case,
  keeping the train and test criterion strings byte-identical (the "v2
  train==test symmetry").
- **γ > 0 (R>1 arms):** RAFT weighting over the intermediates, same
  normalization factors and validity masks as the final term.

Logged keys: `gru_trans_loss` / `gru_quat_loss` stay **final-iterate-only**
(comparable across R arms); `gru_pose_loss_iters` reports the weighted
intermediate mass; `gru_pose_loss` is the total.

### 8.2 ⚠ The ×4.1611 confound — R8 is not a single-variable change

The weights sum:

| N | weights (γ = 0.8) | total gradient mass |
|---|---|---|
| 1 | [1.0] | **1.0000** |
| 4 | [0.512, 0.64, 0.8, 1.0] | **2.9520** |
| 8 | [0.2097, …, 0.64, 0.8, 1.0] | **4.1611** |

With `pose_gru_loss_weight: 1.0` (what every R arm on disk uses), an R8 arm's
auxiliary term carries **4.1611× the gradient magnitude** of its R1 twin. An
R8-vs-R1 comparison therefore changes *two* things at once: the iterate count
**and** the aux-loss scale.

This confound is **inherited from RAFT verbatim** — `sequence_loss` at more
iterations also sums to a larger total — but RAFT never *compared* iteration
counts as an experimental lever, so it never mattered there; N was fixed
(12) and the loss scale rode along. Here N *is* the lever, so the inherited
scale change becomes a second variable. The config headers flag it loudly
and give the correction: to isolate iteration alone, renormalize on the
command line with `pose_gru_loss_weight=0.2403` (= 1/4.1611) — and whatever
value is chosen must be identical across the F0/F1 R8 arms, or the *F*
comparison loses single-variable status too. (This is the "R lever verdict:
WEIGHTING, not iteration" thread in the project's recent history.)

### 8.3 Train ≠ test criterion, by design

The **test** criterion deliberately stays `PoseGRULoss()` — final-iterate
only — even for R8 arms. (RAFT does the analogous thing: its evaluation
metric, endpoint error, is computed on `flow_preds[-1]` alone — see the
`epe` lines in §3.3 — while only training uses the full sequence.)
Rationale: best-checkpoint selection and test-loss curves remain comparable
across all R arms — everyone is judged on the final answer. Consequences:

- In γ>0 arms the train and test criterion *strings* differ, so a train-vs-
  test curve within such a run is not an apples-to-apples overfitting watch;
  compare each side against the same side of another run.
- The R1 arms keep the key at `pose_gru_iter_gamma: 0.0` rather than
  deleting it, precisely so their train and test strings are identical.
- `best_ckpt_agg` selects on the test loss *including* the GRU term — so
  compare arms at fixed epochs, or subtract `gru_pose_loss_avg` from
  headline losses.

---

## 9. Interactions to keep in mind

**Zero-init head bootstrap (shared with every arm, but R makes it showier).**
Under A4 the head starts at zero, so on the first backward the gradient
reaching the cell is exactly zero — only the head's own weight/bias move
until it "unlocks". While the head is zero, **all N iterates emit the same
value** (each pass adds a zero correction), the iterate sequence is
degenerate, and the γ-weighted loss is just 4.1611× the final-only loss.
(This failure mode does not exist in RAFT: its `flow_head` is default-init
and emits nonzero corrections from step one; the zero-init head is the
A-lever's init-equivalence guarantee, and this bootstrap window is its
price.) Two rules follow: do **not** read "the iterates do nothing" from
early training, and do **not** run `POSE_GRU_FORCE_ITERS` probes in that
window. In residual mode, any number of *untrained* iterations is an
identity chain — bit-exact at N=1, up to repeated-quaternion-normalization
float noise at N>1.

**With G.** The within-view hidden chain belongs to R; the view-boundary and
chunk-boundary handling belongs to G. Under G3 the aux loss at a later view
reaches earlier views' cell calls through the hidden tape, and the main loss
reaches the GRU through the final iterate's ray build. Forward math is
G-independent, so the anytime-N probe reads identically on any G arm.

**With F.** Image features are static per-view context (§4.2): R gives the
module more compute against the *same* evidence, F gives it more evidence
per unit compute. RAFT needs both halves at once — its refinement works
*because* each iteration re-grounds in the correlation volume (§4.1) — which
is exactly why the grid prices them separately: a 2×2 over
{F off, F1} × {R1, R8} plus the F-source arms at fixed R8.

**TBPTT gradient window (affects every arm equally).** Training runs all but
the last 4 chunks of the 64-view sequence under `no_grad`: only views 48–63
generate gradient at all; views 1–47 are pure forward. The effective aux
gradient budget is a quarter of what "64 views" suggests. This does not bias
R8-vs-R1 comparisons, but it is part of why learning is slow in absolute
terms.

---

## 10. Verification and falsifier protocol

After each run, against a `keep_freq` snapshot only (never
`checkpoint-last/best.pth` while the job lives), and never before the head
has unlocked (§9):

| probe | channel | reading |
|---|---|---|
| `POSE_GRU_FORCE_ITERS={1,2,4,8}` | **the R lever** | monotone-in-n improvement ⇒ refinement learned; flat ⇒ 1-step fixed point (§7) |
| `PREV_PRED_RAY_SHUFFLE=1` | pose feedback | rolls both pose stashes coherently |
| `POSE_GRU_HIDDEN_SHUFFLE=1` / `POSE_GRU_HIDDEN_ZERO=1` | recurrence | roll / zero the hidden |
| `POSE_GRU_IMG_FEAT_SHUFFLE=1` / `POSE_GRU_IMG_FEAT_ZERO=1` | F lever (if on) | see `F_LEVER_EXPLAINED.md` §9 |

Code-level checks live in `verify_gru_v3_levers.py` / `.sbatch`: flag-off
byte-identity against the v2 code, residual init-equivalence, the iterate
mechanics (static context, detach placement, hidden threading), loss gating
(γ=0 ⇒ exact final-only path), and the checkpoint round-trip including the
silent-N guard.

---

## 11. The config inventory on this branch (A4 × G3 slice)

| config | F | R | γ (train) | notes |
|---|---|---|---|---|
| `captain_gru_v3_a4_g3_finetune` | off | 1 | 0.0 | the R control at full scale |
| `captain_gru_v3_a4_g3_r8_finetune` | off | 8 | 0.8 | prices R at full scale vs the control |
| `captain_gru_v3_a4_g3_f1_finetune` | F1 proj | 1 | 0.0 | the F1×R1 cell (both confounds removed) |
| `captain_gru_v3_a4_g3_f1_r8_finetune` | F1 proj | 8 | 0.8 | completes the 2×2 |
| `captain_gru_v3_a4_g3_f1{r,d,c}_r8_finetune` | source arms | 8 | 0.8 | F-source sweep at fixed R8 |
| `captain_gru_v3_a4_g3_r8` (+ `a4_g1` twin) | off | 8 | 0.8 | overfit-scale arms |
| `captain_gru_v3_a4_g3_f1_r4{,_finetune}`, `a4_g0_f1_r{4,8}_finetune` | F1 proj | 4/8 | 0.8 | R4 arms (γ mass 2.952×) and G-ablation twins |

All R>1 arms use `pose_gru_iters: 8` (or 4), `pose_gru_iter_detach: True`,
`pose_gru_iter_gamma: 0.8` (RAFT's default), `pose_gru_loss_weight: 1.0` —
i.e. as shipped they carry the §8.2 confound; renormalized comparisons
require the CLI override. The trained v2 `a4_g1` / `a4_g3` runs serve as the
R1/F-off controls at overfit scale and are deliberately not retrained.
