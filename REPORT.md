# REPORT — captain_gru_v2 → captain_gru_v3: F/R lever implementation

Every code change between the `captain_gru_v2` branch tip (`251606e`, the state this
worktree was created from) and the current `captain_gru_v3` working tree. Generated
from `git diff HEAD` on the tracked files plus the new (untracked) files.

**What was built:**
- **F lever** (`pose_gru_img_feat`): pooled **pre-ray** image-encoder features of
  views *t* and *t−1* enter the PoseGRU cell input through a per-frame LayerNorm +
  **zero-init** `Linear(1024·frames → D)`.
- **R lever** (`pose_gru_iters`): N back-to-back GRUCell iterations per view,
  RAFT-style (running estimate re-fed, static context re-injected, estimate detached
  between iterates, hidden never detached inside a view), with an optional
  γ-weighted sequence loss over the iterates (train criterion only).

**Verification:** `verify_gru_v3_levers.py` — 28/28 PASS (SLURM job 1417506, 1× L40S),
including cross-worktree **flag-off byte-identity** against the unmodified v2 worktree
code and the loader **hard-fail** on mismatched `pose_gru.*` weights.

Diffstat (tracked files):

```
 src/CUT3R/src/dust3r/inference.py     |   5 +
 src/CUT3R/src/dust3r/losses.py        |  47 +++++-
 src/CUT3R/src/dust3r/model.py         | 262 +++++++++++++++++++++++++++++++---
 src/CUT3R/src/train_cut3r_baseline.py |  27 +++-
 verify_gru_grid.py                    |  16 ++-
 5 files changed, 332 insertions(+), 25 deletions(-)
```

New files: `verify_gru_v3_levers.py`, `verify_gru_v3_levers.sbatch`,
`pose_gru_expand_ckpt.py`, `src/CUT3R/config/captain_gru_v3_a4_{g1,g2}{,_f1}_r8.yaml`
(4 configs), `src/CUT3R/config/captain_gru_v3_grid_README.md`, this `REPORT.md`.

---

## 1. `src/CUT3R/src/dust3r/model.py` (+262/−16, 8 hunks)

### 1.1 `load_model` checkpoint sniff — args+weights authoritative (now lines ~104–164)

`pose_gru_iters` is invisible in weight shapes, so it is read from `ckpt["args"]`
(the same pattern as `pose_gru_mode`); the F-lever config is recovered purely from
the `img_proj.weight` shape (rows = D, cols/enc_width = frame count); the cell input
width must then decompose as `base(7|14) + D` or the load **asserts**. Legacy v2
checkpoints have no `img_proj` key → D=0 → the sniff reduces bit-for-bit to the old
`==14` rule.

Added inside the `if any(k.startswith(("pose_gru.", ...)))` block:

```python
        # R lever: the iteration count is INVISIBLE in weight shapes (shared-
        # weight iteration) — ckpt args are the only source, like mode.
        iters = int(getattr(train_args, "pose_gru_iters", 1))
```

```python
        # F lever: projector presence + shapes are authoritative — D from
        # img_proj.weight rows, frame count from its columns / enc width.
        w_proj = ckpt["model"].get(
            "pose_gru.img_proj.weight", ckpt["model"].get("module.pose_gru.img_proj.weight")
        )
        img_feat = "none" if w_proj is None else "input"
        img_feat_dim = 32
        img_feat_frames = 2
        if w_proj is not None:
            img_feat_dim = w_proj.shape[0]
            src_dim = net.enc_embed_dim
            assert w_proj.shape[1] % src_dim == 0, (
                f"pose_gru.img_proj input width {w_proj.shape[1]} is not a "
                f"multiple of the encoder width {src_dim}"
            )
            img_feat_frames = w_proj.shape[1] // src_dim
```

The input-width rule (replacing `input_mode = "pose_delta" if w_ih.shape[1] == 14 else "pose"`):

```python
        if w_ih is not None:
            base_width = w_ih.shape[1] - (img_feat_dim if img_feat == "input" else 0)
            assert base_width in (7, 14), (
                f"pose_gru cell input width {w_ih.shape[1]} minus projector dim "
                f"{img_feat_dim if img_feat == 'input' else 0} = {base_width}; "
                f"expected 7 (pose) or 14 (pose_delta)"
            )
            input_mode = "pose_delta" if base_width == 14 else "pose"
```

And the enable call + verbose print gained the four new kwargs / fields:

```python
        net.enable_pose_gru(
            hidden_dim=hidden_dim,
            mode=mode,
            input_mode=input_mode,
            img_feat=img_feat,
            img_feat_dim=img_feat_dim,
            img_feat_frames=img_feat_frames,
            iters=iters,
        )
        if verbose:
            print(
                f"... pose_gru enabled from ckpt: mode={mode}, "
                f"hidden_dim={hidden_dim}, input={input_mode}, "
                f"img_feat={img_feat} (dim={img_feat_dim}, frames={img_feat_frames}), "
                f"iters={iters}"
            )
```

### 1.2 `PoseGRU.__init__` — F/R ctor (now lines ~337–424)

Docstring additions (F-lever and R-lever contracts):

```python
    img_feat="input" (F lever): the cell input is additionally conditioned on
    pooled PRE-ray image-encoder features of the current view (and, with
    img_feat_frames=2, the previous view): per-frame LayerNorm (shared
    affine) then a ZERO-INIT Linear(src_dim*frames -> img_feat_dim) appended
    AFTER the pose block — so gru_in keeps its pose-first 7/14-d layout, the
    residual anchor is untouched, and an untrained module is output-identical
    to img_feat="none" (the feature term is exactly zero at init, while the
    cell's default-init input columns keep the gradient to img_proj alive).

    iters (R lever): number of back-to-back cell iterations per view. The
    CALL SITE owns the loop (this module stays a pure single-step cell — the
    probe/spy surface); the count lives here so it rides enable_pose_gru and
    the checkpoint sniff (it is invisible in weight shapes). In residual mode
    the zero-init head makes any number of untrained iterations an identity
    chain (up to repeated-quat-normalize fp noise; bit-exact at iters=1).
```

New signature (all new kwargs defaulted so flag-off constructs the exact v2 module —
identical state_dict keys/shapes, identical RNG consumption):

```python
    def __init__(
        self,
        pose_dim=7,
        hidden_dim=128,
        mode="residual",
        input_mode="pose",
        img_feat="none",
        img_feat_dim=32,
        img_feat_frames=2,
        img_feat_src_dim=1024,
        iters=1,
    ):
```

New body lines (asserts, attribute stores, and the F-branch module construction —
`self.cell` widens by D only when the F lever is on):

```python
        assert img_feat in ("none", "input"), f"unknown pose_gru img_feat {img_feat!r}"
        assert int(iters) >= 1, f"pose_gru iters must be >= 1, got {iters!r}"
        ...
        self.img_feat = img_feat
        self.img_feat_src_dim = int(img_feat_src_dim)
        self.iters = int(iters)
        ...
        if img_feat == "input":
            assert int(img_feat_frames) in (1, 2), (
                f"img_feat_frames must be 1 (current view) or 2 (current + previous), "
                f"got {img_feat_frames!r}"
            )
            assert int(img_feat_dim) > 0, "img_feat_dim must be positive when img_feat='input'"
            self.img_feat_dim = int(img_feat_dim)
            self.img_feat_frames = int(img_feat_frames)
            self.img_norm = nn.LayerNorm(self.img_feat_src_dim)
            self.img_proj = nn.Linear(
                self.img_feat_src_dim * self.img_feat_frames, self.img_feat_dim
            )
            nn.init.zeros_(self.img_proj.weight)
            nn.init.zeros_(self.img_proj.bias)
            self.input_dim += self.img_feat_dim
        else:
            self.img_feat_dim = 0
            self.img_feat_frames = 0
```

Design invariants encoded here: `img_proj` is **zero-init** (feature contribution is
exactly 0 at init ⇒ init-equivalence with F0), while the GRUCell's feature input
columns keep their default init (⇒ the gradient path to `img_proj` stays alive —
zeroing both sides would deadlock the channel permanently). All new parameters live
**inside** `PoseGRU`, so `split_pose_gru_param_groups`' id-membership filter
auto-includes them in the ×100-lr group and the ckpt sniff scope stays correct.

### 1.3 `PoseGRU.forward` — 3-arg signature with internal concat (now lines ~426–450)

```python
    def forward(self, gru_in, hidden, img_feat=None):
        """gru_in: (B, 7) or (B, 14) detached prev-step pose (+delta); hidden:
        (B, H) or None at sequence start; img_feat: (B, frames*src_dim) pooled
        pre-ray image features (current view first, then previous) when
        img_feat="input", else None. Returns (refined_pose_enc, new_hidden)."""
        if hidden is None:
            hidden = gru_in.new_zeros(gru_in.shape[0], self.hidden_dim)
        cell_in = gru_in
        if self.img_feat == "input":
            assert img_feat is not None, "img_feat='input' needs the pooled image features"
            f = img_feat.reshape(
                img_feat.shape[0], self.img_feat_frames, self.img_feat_src_dim
            )
            f = self.img_norm(f).reshape(img_feat.shape[0], -1)
            cell_in = torch.cat([gru_in, self.img_proj(f)], dim=-1)
        hidden = self.cell(cell_in, hidden)
```

The concat happens **inside** forward, so `gru_in` stays 7/14-wide at the module
boundary: the residual anchor `gru_in[:, :3]` / `gru_in[:, 3:7]` (unchanged lines
just below) and every harness that reconstructs `gru_in` remain valid. The
per-frame LayerNorm shares one affine across both frames (reshape → LN → flatten).

### 1.4 `load_state_dict` last-resort fallback — hard-fail on `pose_gru.*` (now lines ~619–687)

The old fallback silently skipped size-mismatched keys with a `printer.info`, which
would let a trained GRU evaluate as random init with no error. Two raises added
(non-`pose_gru` keys keep the lenient behavior):

```python
                        elif key.startswith("pose_gru."):
                            # NEVER silently drop a trained GRU: a skipped
                            # pose_gru tensor means the module was rebuilt
                            # with the wrong hyperparameters and would
                            # evaluate as random init with no error.
                            raise RuntimeError(
                                f"pose_gru weight '{key}' size mismatch "
                                f"(ckpt: {new_ckpt[key].size()}, model: "
                                f"{self.state_dict()[key].size()}) — rebuild the "
                                f"module with the checkpoint's hyperparameters "
                                f"(enable_pose_gru / load_model sniff) or use the "
                                f"pose_gru_expand_ckpt.py surgery script"
                            )
```

```python
                    elif key.startswith("pose_gru."):
                        raise RuntimeError(
                            f"pose_gru weight '{key}' not present in the model — "
                            f"call enable_pose_gru with the checkpoint's "
                            f"configuration (img_feat/iters included) before loading"
                        )
```

### 1.5 `enable_pose_gru` — new kwargs, direct-mode warning, feature-stash reset (now lines ~689–735)

```python
    def enable_pose_gru(
        self,
        hidden_dim=128,
        mode="residual",
        input_mode="pose",
        img_feat="none",
        img_feat_dim=32,
        img_feat_frames=2,
        iters=1,
    ):
        ...
        if mode == "direct" and int(iters) > 1:
            printer.info(
                "WARNING: pose_gru mode='direct' with iters>1 — nothing anchors "
                "iterate k on iterate k-1, so the loop can collapse to a 1-step "
                "fixed point. The R lever is designed for residual mode."
            )
        self.pose_gru = PoseGRU(
            hidden_dim=hidden_dim,
            mode=mode,
            input_mode=input_mode,
            img_feat=img_feat,
            img_feat_dim=img_feat_dim,
            img_feat_frames=img_feat_frames,
            img_feat_src_dim=self.enc_embed_dim,
            iters=iters,
        )
        self._pose_gru_hidden = None
        self._prev_img_feat = None
        return self.pose_gru
```

`img_feat_src_dim` is bound to the model's own `enc_embed_dim` (1024). Every legacy
caller (`trainer`, `verify_pose_gru_e2e.py`, `verify_gru_eval_and_direct.py`, the
eval worker's identity-init fallback) passes no new kwargs and keeps constructing
the exact v2 module.

### 1.6 `_forward_decoder_group_step` — iterate collector init (now line ~1062)

```python
        gru_pose_pred = None
        gru_iterates = []  # all R-lever iterates of this view (last == gru_pose_pred)
        pose_gru_e2e = False  # set True inside the GRU block when lever G2 is live
```

### 1.7 `_forward_decoder_group_step` — view-0 branch: feature-stash reset + view-0 stash (now lines ~1078–1098)

Added to the `if view_indices[0] == 0:` sequence-start reset (alongside the pose
stashes and hidden):

```python
                self._prev_img_feat = None
                pose_gru = getattr(self, "pose_gru", None)
                if pose_gru is not None and pose_gru.img_feat == "input":
                    # Stash view 0's pooled PRE-ray appearance for the next
                    # step's (current, previous) feature pair. View 0's tokens
                    # carry the constant, pose-free masked_ray_map_token added
                    # in _encode_views — subtract it so the stash is the same
                    # pure-image statistic every other view provides.
                    self._prev_img_feat = (
                        (feat_group[0] - self.masked_ray_map_token.to(feat_group[0].dtype))
                        .detach()
                        .mean(dim=1)
                        .float()
                    )
```

View 0 never runs the GRU, but its feature must exist as "previous frame" at view 1.
`_encode_views` adds the constant `masked_ray_map_token` to view 0's tokens under
feed_prev_pred; subtracting it makes the stash the same pure-image mean statistic
that every view ≥ 1 provides.

### 1.8 `_forward_decoder_group_step` — the core: feature build + falsifiers + iteration loop (now lines ~1130–1211)

Replaces the old 11-line single-call block (`gru_in = prev_pose_enc; ... gru_pose_pred,
new_hidden = pose_gru(gru_in.float(), hidden)`). Added, in order:

**(a) F-lever feature pair + falsifiers** (runs before the ray add at the
`feat_group[0] + ray_out[-1]` line, so the features are pre-ray by construction):

```python
                    img_feat_vec = None
                    cur_img_feat = None
                    if pose_gru.img_feat == "input":
                        # F lever: pooled PRE-ray appearance of the current
                        # view (the ray add below happens later), plus the
                        # previous view's stash for the two-frame pair. The
                        # detach is structural: the encoder is frozen and
                        # (under TBPTT) already detached — the feature is
                        # data, never a gradient path.
                        cur_img_feat = feat_group[0].detach().mean(dim=1).float()
                        if pose_gru.img_feat_frames == 2:
                            prev_img_feat = getattr(self, "_prev_img_feat", None)
                            assert prev_img_feat is not None, (
                                "pose_gru img_feat: no stashed view x-1 feature — "
                                "views must be processed sequentially from view 0"
                            )
                            img_feat_vec = torch.cat([cur_img_feat, prev_img_feat], dim=-1)
                        else:
                            img_feat_vec = cur_img_feat
                        if (
                            os.environ.get("POSE_GRU_IMG_FEAT_SHUFFLE") == "1"
                            and img_feat_vec.shape[0] > 1
                        ):
                            # Falsifier: every sample sees another sample's
                            # image pair (rolled coherently, mirroring
                            # PREV_PRED_RAY_SHUFFLE); pose feedback and
                            # supervision stay honest. If metrics don't
                            # degrade, the GRU ignores image evidence.
                            img_feat_vec = torch.roll(img_feat_vec, shifts=1, dims=0)
                        if os.environ.get("POSE_GRU_IMG_FEAT_ZERO") == "1":
                            # Falsifier: featureless control (works at B=1).
                            img_feat_vec = torch.zeros_like(img_feat_vec)
```

**(b) R-lever setup** (count, eval override, detach discipline, static delta hoist):

```python
                    # R lever: n_iters back-to-back cell iterations, RAFT
                    # style. The pose slice of the input is the RUNNING
                    # estimate (detached between iterations per
                    # pose_gru_iter_detach — RAFT's coords1.detach()); the
                    # delta and image features are STATIC per-view context
                    # re-injected every iteration; the hidden is NEVER
                    # detached inside a view (only at the view boundary, per
                    # the G lever below). POSE_GRU_FORCE_ITERS=<n> overrides
                    # the count at eval — the anytime-inference probe.
                    n_iters = int(getattr(pose_gru, "iters", 1))
                    if os.environ.get("POSE_GRU_FORCE_ITERS"):
                        n_iters = max(1, int(os.environ["POSE_GRU_FORCE_ITERS"]))
                    iter_detach = bool(getattr(self, "pose_gru_iter_detach", True))
                    delta_static = None
                    if pose_gru.input_mode == "pose_delta":
                        # Hand the loop's velocity to the GRU explicitly: the
                        # relative motion between the last two RAW head poses
                        # (identity at view 1, where no P(x-2) exists yet).
                        delta = pose_delta_encoding(prev_prev_pose_enc, prev_pose_enc)
                        delta_static = delta.to(prev_pose_enc.dtype)
```

**(c) The iteration loop** (fp32 outside autocast, exactly like the old single call;
`n_iters=1` unrolls to the byte-identical v2 op sequence — verified bitwise):

```python
                    est = prev_pose_enc
                    with torch.autocast(device_type=feat_group[0].device.type, enabled=False):
                        for _it in range(n_iters):
                            gru_in = (
                                est
                                if delta_static is None
                                else torch.cat([est, delta_static], dim=-1)
                            )
                            gru_pose_pred, hidden = pose_gru(
                                gru_in.float(), hidden, img_feat=img_feat_vec
                            )
                            gru_iterates.append(gru_pose_pred)
                            if _it + 1 < n_iters:
                                est = (
                                    gru_pose_pred.detach() if iter_detach else gru_pose_pred
                                )
                    new_hidden = hidden
```

**(d) Honest stash rotation** (the falsified `img_feat_vec` never enters the stash):

```python
                    if pose_gru.img_feat == "input":
                        # Stash the current view's (honest, unfalsified)
                        # pooled pre-ray appearance for the next step's pair.
                        # Pure detached data — crosses TBPTT chunks like the
                        # pose stashes, no boundary handling needed.
                        self._prev_img_feat = cur_img_feat
```

Everything after this point is **unchanged v2 code** operating on the final iterate:
the G1 hidden keep/detach, the G2 e2e gate, the ray build
(`pose_encoding_to_camera → get_ray_map_torch → _encode_ray_map → token add`),
and the raw-head-pose feedback stash.

### 1.9 res surfacing — stacked iterates (now lines ~1354–1360)

```python
                res_group[-1]["gru_pose"] = gru_pose_pred
                if len(gru_iterates) > 1:
                    # R lever: all iterates (final == gru_pose) for the
                    # gamma-weighted sequence loss. ONE stacked (N, B, 7)
                    # tensor, not a list — the TBPTT all_preds collection
                    # blanket-detaches tensor values and passes it through.
                    res_group[-1]["gru_pose_iters"] = torch.stack(gru_iterates, dim=0)
```

Key absent at N=1 ⇒ flag-off res dicts are unchanged.

---

## 2. `src/CUT3R/src/dust3r/losses.py` — `PoseGRULoss` (+47/−2, 3 hunks)

### 2.1 Ctor: `iter_gamma`, value-gated

```python
    def __init__(self, norm_mode="?avg_dis", iter_gamma=0.0):
        ...
        # R lever: gamma-weighted RAFT sequence loss over pred["gru_pose_iters"]
        # ((N, B, 7), final iterate == gru_pose). Gated on the VALUE, not key
        # presence: iter_gamma=0.0 runs the exact final-only code path even
        # when the iterates tensor is attached. Intermediate iterates get
        # weight gamma**(N-1-k); the final iterate keeps its existing
        # weight-1.0 term, so its gradient scale matches an iters=1 run.
        self.iter_gamma = float(iter_gamma)
```

### 2.2 `get_name`

```python
    def get_name(self):
        if self.iter_gamma > 0.0:
            return f"PoseGRULoss(iter_gamma={self.iter_gamma:g})"
        return "PoseGRULoss()"
```

### 2.3 `compute_loss`: sequence-loss block

`gru_pose_loss` detail moved after the total (so it reports final+iterates);
`gru_trans_loss`/`gru_quat_loss` stay **final-iterate-only** for cross-arm curve
comparability. Added block:

```python
        if self.iter_gamma > 0.0:
            # Intermediate iterates 0..N-2 only (the final iterate IS
            # gru_pose, already counted above at weight 1.0). Same norm
            # factors and validity masks as the final term — every iterate
            # lives in the head's scene scale.
            iter_t_terms, iter_q_terms = {}, {}
            for i in gru_idx:
                seq = preds[i].get("gru_pose_iters")
                if seq is None:
                    continue
                seq = seq.float()
                n_it = seq.shape[0]
                valid = base_mask & torch.isfinite(gt_poses[i]).all(dim=-1)
                for k in range(n_it - 1):
                    t_err_k = torch.norm(
                        seq[k][:, :3] / factor_pr.clip(eps)
                        - gt_poses[i][:, :3] / factor_gt.clip(eps),
                        dim=-1,
                    )
                    q_err_k = torch.norm(seq[k][:, 3:] - gt_poses[i][:, 3:], dim=-1)
                    iter_t_terms.setdefault((n_it, k), []).append(t_err_k[valid])
                    iter_q_terms.setdefault((n_it, k), []).append(q_err_k[valid])
            iters_loss = None
            for (n_it, k), terms in iter_t_terms.items():
                t_k = torch.cat(terms)
                q_k = torch.cat(iter_q_terms[(n_it, k)])
                if t_k.numel() == 0:
                    continue
                w_k = self.iter_gamma ** (n_it - 1 - k)
                term = w_k * (t_k.mean() + q_k.mean())
                iters_loss = term if iters_loss is None else iters_loss + term
            if iters_loss is not None:
                loss = loss + iters_loss
                details["gru_pose_loss_iters"] = float(iters_loss.detach())
        details["gru_pose_loss"] = float(loss.detach())
```

Semantics: RAFT weighting `w_k = γ^(N−1−k)` over intermediates only — the final
iterate is already the existing weight-1.0 `gru_pose` term, so an R8 run's final
gradient scale matches an R1 run exactly. Norm factors and validity masks are the
same tensors the final term uses (all iterates live in the head's scene scale).

---

## 3. `src/CUT3R/src/train_cut3r_baseline.py` (+27/−2, 3 hunks)

### 3.1 `enable_pose_gru` call — new key plumbing (also persists them into `ckpt["args"]`)

```python
            # F lever: pooled pre-ray image features (current [+ previous]
            # view) into the cell input via a zero-init projector.
            img_feat=str(getattr(args, "pose_gru_img_feat", "none")),
            img_feat_dim=int(getattr(args, "pose_gru_img_feat_dim", 32)),
            img_feat_frames=int(getattr(args, "pose_gru_img_feat_frames", 2)),
            # R lever: back-to-back cell iterations per view. Persisted in
            # ckpt args (invisible in weight shapes — load_model reads it
            # back exactly like pose_gru_mode).
            iters=int(getattr(args, "pose_gru_iters", 1)),
```

### 3.2 Extended startup log line

```python
        printer.info(
            "pose_gru enabled: mode=%s, input=%s (dim %d), hidden_dim=%d, "
            "img_feat=%s (dim %d, frames %d), iters=%d, params=%d"
            % (
                ...
                model.pose_gru.img_feat,
                model.pose_gru.img_feat_dim,
                model.pose_gru.img_feat_frames,
                model.pose_gru.iters,
                ...
            )
        )
```

### 3.3 `pose_gru_iter_detach` stamp + R-lever log (after `accelerator.prepare`)

```python
    # R lever inner-iteration discipline (RAFT convention, default True):
    # detach the running pose estimate between iterations; the hidden is
    # never detached inside a view. Training-graph-only — inert at eval
    # (no_grad) and at iters=1.
    base_model.pose_gru_iter_detach = bool(getattr(args, "pose_gru_iter_detach", True))
    ...
    if use_pose_gru and base_model.pose_gru.iters > 1:
        printer.info(
            f"pose_gru R lever: iters={base_model.pose_gru.iters}, "
            f"iter_detach={base_model.pose_gru_iter_detach}, "
            f"iter_gamma(train)={float(getattr(args, 'pose_gru_iter_gamma', 0.0)):g}"
        )
```

No changes to `split_pose_gru_param_groups` (id-membership auto-includes
`img_norm`/`img_proj` at ×100 lr) or to the optimizer/DDP/TBPTT plumbing.

---

## 4. `src/CUT3R/src/dust3r/inference.py` (+5, comment only)

Zero functional change — verified that the existing chunk-boundary detach and the
`all_preds` blanket-detach handle both levers; documented at the boundary:

```python
            # v3 levers add NO new boundary state: with iters>1 (R lever) the
            # stashed hidden is the view's LAST iterate — this detach is
            # N-agnostic; the _prev_img_feat stash (F lever) is always
            # detached data; gru_pose_iters is a plain res tensor that the
            # all_preds collection below blanket-detaches like everything else.
```

---

## 5. `verify_gru_grid.py` (+16/−5) — v2-grid harness kept runnable

Two kinds of change:

**(a) Worktree path retarget** (from the worktree-setup step, not the lever work):
`WORKTREE = ".../captain_gru_v2"` → `".../captain_gru_v3"`, and the same in the
`SCRATCH_DIR` fallback for the mini-ckpt path.

**(b) `GRUSpy` 3-arg signature** — mandatory: the call site now always passes
`img_feat=` (None on the F0 arms this file exercises), so the old 2-arg spy would
`TypeError` on every run:

```python
        def spy(gru_in, hidden, img_feat=None):
            self.calls.append(
                dict(
                    gru_in=gru_in.detach().clone(),
                    img_feat=(None if img_feat is None else img_feat.detach().clone()),
                    ...
                )
            )
            return self.orig(gru_in, hidden, img_feat=img_feat)
```

plus a docstring note that all its checks assume the default N=1 (true for every v2
grid arm) and that the v3 levers have their own verifier. All other harnesses
(`verify_pose_gru_e2e.py`, `probe_gru_hidden_falsifier.py`,
`verify_gru_eval_and_direct.py`, `eval_pipeline/infer_and_eval_worker_ray.py`)
required **zero** changes — they call `enable_pose_gru` with defaulted kwargs and
never wrap `PoseGRU.forward`.

---

## 6. New file: `verify_gru_v3_levers.py` (+ `verify_gru_v3_levers.sbatch`)

Standalone GPU verifier for both levers (whole file is new; 28 checks, all PASS on
job 1417506). Stage map with the load-bearing code:

| Stage | Checks | What it proves |
|---|---|---|
| PAR | P1–P2 | **Cross-worktree flag-off byte-identity**: subprocesses re-run the same rollout under the v2 worktree's code and this worktree's code (`--dump-rollout` mode, `PYTHONPATH`-isolated, seeded GRU init) — plain prev_pred AND A4-GRU arms bitwise equal |
| UNIT | U1–U4 | ctor widths (14+32=46, `img_proj (32,2048)` zero), asserts on bad values, zero-init residual identity with features attached |
| INIT | I0–I2 | F1×R8 at init == plain prev_pred on real data/ckpt; iterate translation chain bit-stable through 8 zero-head iterates |
| ITER | T1–T8 | 56 calls = 7 views × 8; iterate k's pose slice == iterate k−1's output; delta and features bit-constant per view; `gru_pose_iters` (8,B,7) with final row == `gru_pose`; **`FORCE_ITERS=1` bitwise == an iters=1 module** (anytime-N is exact); key absent at N=1 |
| FEAT | F1–F3 | recorded features bit-equal independently recomputed pre-ray means (current + previous halves, view-0 masked-token subtraction); `IMG_FEAT_ZERO` perturbs views ≥ 1 only; `IMG_FEAT_SHUFFLE` == coherent batch-roll of the honest pair |
| LOSS | S1–S4 | `iter_gamma=0` bit-identical with/without the iterates key; γ-weighted total == hand-computed per-iterate sum; detail keys; gradients reach every iterate |
| LOAD | L1–L4 | round-trip restores img_feat/frames/D/iters; legacy v2 ckpt sniffs exactly as before; **loading an F1 GRU into a 14-wide module HARD-FAILS** |

The PAR-stage subprocess core:

```python
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            r = subprocess.run(
                [sys.executable, os.path.abspath(__file__),
                 "--dump-rollout", out, "--worktree", wt, "--arm", arm],
                env=env, capture_output=True, text=True,
            )
```

with `dump_rollout()` seeding `torch.manual_seed(GRU_SEED)` immediately before
`enable_pose_gru` so the freshly-initialized GRU weights are identical across the
two worktrees' processes.

`verify_gru_v3_levers.sbatch`: 1× `lovelace_l40s`, 8 CPUs, 60G, 1h — same pattern as
`verify_gru_grid.sbatch`.

---

## 7. New file: `pose_gru_expand_ckpt.py` (warm-start surgery)

Widens a trained v2 GRU checkpoint into the F-lever layout, function-preserving at
load. The core transformation:

```python
    torch.manual_seed(args.seed)
    # Fresh default GRUCell init for the new feature columns (see module doc).
    bound = 1.0 / math.sqrt(hidden)
    new_cols = torch.empty(hidden3, D, dtype=w_ih.dtype).uniform_(-bound, bound)
    sd[prefix + "cell.weight_ih"] = torch.cat([w_ih, new_cols], dim=1)

    src_total = args.enc_dim * args.frames
    sd[prefix + "img_proj.weight"] = torch.zeros(D, src_total, dtype=w_ih.dtype)
    sd[prefix + "img_proj.bias"] = torch.zeros(D, dtype=w_ih.dtype)
    sd[prefix + "img_norm.weight"] = torch.ones(args.enc_dim, dtype=w_ih.dtype)
    sd[prefix + "img_norm.bias"] = torch.zeros(args.enc_dim, dtype=w_ih.dtype)
```

The new `weight_ih` columns are deliberately **fresh-init, not zero** (zero on both
sides would permanently deadlock the feature channel's gradients — zero `img_proj`
already guarantees function preservation on its own). It also stamps
`pose_gru_img_feat/_dim/_frames` (and optionally `pose_gru_iters`) into the ckpt
args (OmegaConf `open_dict` or plain namespace), refuses to re-expand an
already-expanded ckpt, and asserts the base width ∈ {7, 14}. Not used by the
launched grid (all 4 arms train from `cut3r_512_dpt_4_64.pth` for v2 comparability).

---

## 8. New configs: `src/CUT3R/config/captain_gru_v3_*.yaml` (4 files) + grid README

Each is the **byte-identical** v2 `captain_gru_v2_a4_g1.yaml` / `_a4_g2.yaml` overfit
recipe (same wrist_test data ROOT, `num_views` 64, batch 4, 50 epochs, TBPTT via
`long_context`, lr 1e-6, `save_dir captain_cut3r_sim3rmse`, same `extra_test_criteria`)
except for the header comments, `exp_name`, and these functional lines:

**All four arms** (vs the v2 twin):

```yaml
pose_gru_iter_gamma: 0.8

train_criterion: ... + ${pose_gru_loss_weight}*PoseGRULoss(iter_gamma=${pose_gru_iter_gamma})
# Final-iterate-only on the test side (see pose_gru_iter_gamma note above).
test_criterion:  ... + ${pose_gru_loss_weight}*PoseGRULoss()

pose_gru_iters: 8
pose_gru_iter_detach: True
```

(The train≠test criterion asymmetry is deliberate: the γ-summed sequence loss grows
with N, and the test criterion feeds best-ckpt selection — keeping test final-only
keeps `loss_avg` comparable across R arms and against the v2 controls.)

**F0 arms** (`_r8`) additionally:

```yaml
pose_gru_img_feat: none
```

**F1 arms** (`_f1_r8`) additionally:

```yaml
pose_gru_img_feat: input
pose_gru_img_feat_dim: 32
pose_gru_img_feat_frames: 2
```

**Per-G pair** — exactly the v2 G split: `pose_gru_bptt: True / pose_gru_e2e: False`
(g1 arms) vs `pose_gru_bptt: False / pose_gru_e2e: True` (g2 arms). **Superseded —
see §8b: the g2 arms were G0+e2e, not a superset of G1, and were converted to G3.**

| Config | exp_name | Trained as |
|---|---|---|
| `captain_gru_v3_a4_g1_r8.yaml` | `captain_gru_v3_a4_g1_r8` | job 1417515 (2× L40S, COMPLETED) |
| `captain_gru_v3_a4_g2_r8.yaml` | `captain_gru_v3_a4_g2_r8` | job 1417516 (2× L40S, COMPLETED) |
| `captain_gru_v3_a4_g1_f1_r8.yaml` | `captain_gru_v3_a4_g1_f1_r8` | job 1417517 (2× L40S) |
| `captain_gru_v3_a4_g2_f1_r8.yaml` | `captain_gru_v3_a4_g2_f1_r8` | job 1417518 (2× A6000) |

`captain_gru_v3_grid_README.md` documents the lever table, the fixed R conventions,
the train-only-γ decision, the falsifier suite, the `POSE_GRU_FORCE_ITERS` anytime-N
protocol, and the warm-start surgery path.

### 8b. G3 — the superset arm (2026-07-31)

`pose_gru_bptt` and `pose_gru_e2e` are independent booleans, and every `*_g2`
config set `bptt: False`. **G2 was therefore G0 + e2e, not a superset of G1.**
With `pose_gru_iters: 8` and TBPTT chunk 4 the practical difference is the depth
of the tape credit reaches: 8 cell calls under G0/G2 (one view's inner
iterations) versus up to 32 under G1 (4 views × 8 iterations, one chunk). Every
g1-vs-g2 comparison in §8 and in `captain_gru_grid_ANALYSIS.md` therefore traded
multi-step credit *away* to buy the main-loss path, rather than adding one on
top of the other — the two arms are not nested, so neither dominates by
construction.

**G3 = `bptt: True, e2e: True`** is the intended superset. No code change was
needed; `_forward_decoder_group_step` already reads the two flags independently
and the chunk-boundary detach in `inference.py:156` is unconditional, so each
chunk's backward stays inside its own graph exactly as it does under G1. The
fed-back stash stays detached (`model.py:1406`), so even under G3 the main loss
never reaches an earlier view's decoder or heads — the added credit is confined
to the GRU's own recurrence, which costs essentially nothing (a 128-d GRUCell
at batch 4).

The A3/A4/A5 `*_g2` configs were **converted in place and renamed to `_g3`**;
A1/A2 keep their original G2 configs, and all previously trained `*_g2`
checkpoints stay on disk under their old names as valid G0+e2e runs.
`verify_gru_grid.py` gained checks T10–T14 for the combination, which had never
been exercised before. **34/34 on job 1428236:**

- **T10** no freed-graph crash when the hidden tape and the ray-build tape
  coexist in one chunk backward;
- **T11** G3 reproduces G1's tape topology exactly — tape at views 2,3,5,6,7 and
  cut at the view-4 chunk boundary;
- **T12** G3 keeps G2's main-loss path (main-only criterion still grads head and
  cell);
- **T13** G3 owns G2's mechanism *exactly*: under main-only, G1 receives
  identically zero GRU gradient while G3 receives 2.47;
- **T14** G3 owns G1's mechanism *exactly*: under main-only, a gradient
  physically arrives at the incoming hidden of every in-chunk call (views
  2,3,5,6,7) and at none under G2.

**Methodology note — do not re-litigate.** T13 and T14 were first written as
differences between gradient tensors, gated at `10 × floor` where `floor` is the
G0/full-criterion nondeterminism (2.9e-07). That gate is invalid for any e2e
arm: G0's GRU gradient is almost entirely the aux `PoseGRULoss`, a tiny nearly
deterministic path, whereas every e2e arm routes gradient through the ray build,
ray encoder, decoder and DPT heads — a backward ~70× noisier. Measured paired
floors: a G2-vs-G2 repeat differs by **2.11e-05**, *larger* than the 1.92e-05
G3-vs-G2 difference the original T13 called proof of a strict superset. Two
repair attempts failed and are recorded in the file so they are not retried:

- run 1428195 amplified the head ×100 to lift the signal — signal +48×, paired
  noise +300×, SNR 6× → 3.8×. A ×100 head emits wild poses, making ray maps and
  hence the backward's reduction order far more variable.
- run 1428210 measured at the normal head against a paired floor — 7.28e-06 vs
  2.11e-05, correctly reported as inconclusive.

The term is genuinely smaller than the noise of the path it travels, so no
threshold on that comparison can be both honest and decisive. Both checks were
therefore rebuilt as **exact** tests — an all-or-nothing zero-gradient
orthogonality check (T13) and a `register_hook` on the hidden *argument* of each
GRU call (T14), the sole route by which credit leaves a view for the previous
one. Note the *returned* hidden is unusable for this: it is also consumed by its
own view's head, so a hook there cannot separate the two gradient sources.
Separately, `torch.use_deterministic_algorithms(True, warn_only=True)` plus a
max-over-3-repeats floor fixed the pre-existing flaky **T3**, whose 1e-6
threshold sat exactly at the noise level (it drew 5.89e-07 on job 1409920 and
1.25e-06 on 1428021 while the signal it gates was bit-stable at 1.51e-05); the
floor is now 2.9e-07.

| Config | exp_name | Trained as |
|---|---|---|
| `captain_gru_v2_a3_g3.yaml` | `captain_gru_v2_a3_g3` | job 1428022 (2× L40S) |
| `captain_gru_v2_a4_g3.yaml` | `captain_gru_v2_a4_g3` | job 1428023 (2× L40S) |
| `captain_gru_v3_a4_g3_r8.yaml` | `captain_gru_v3_a4_g3_r8` | job 1428024 (2× L40S) |
| `captain_gru_v3_a4_g3_f1_r8.yaml` | `captain_gru_v3_a4_g3_f1_r8` | job 1428025 (2× L40S) |
| `captain_gru_v3_a5_g3.yaml` | `captain_gru_v3_a5_g3` | job 1428026 (2× L40S) |

Verifier: job 1428021 (`verify_gru_grid.sbatch`, 1× L40S).

---

## 9. Explicitly NOT changed (the invariants the levers preserve)

- The **fed-back stash** stays the raw head pose, always detached
  (`res_group[-1]["camera_pose"].detach()`) — the refiner never contaminates its own
  future inputs.
- The **ray build** runs once per view, from the final iterate, under the unchanged
  `set_grad_enabled(pose_gru_e2e)` block — G2 semantics identical to v2.
- The **G1/G2 hidden keep/detach** lines and the TBPTT chunk-boundary detaches in
  `inference.py` — untouched (operate on the last iterate's hidden, N-agnostic).
- `PoseGRU`'s residual math, quaternion renormalization, `pose_delta_encoding`, the
  `_encode_views` / dataset code, `split_pose_gru_param_groups`, and all v2 configs.
- Flag-off behavior end to end: verified **byte-identical** to the pre-change v2
  worktree code (verifier stages P1/P2).
