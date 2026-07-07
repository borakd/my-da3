# Captain Ray — Ground-Truth-Camera Conditioning of CUT3R via its Pretrained Ray-Map Encoder

**Branch:** `captain_ray` (git worktree `/scratch/bdursun25/cuteanything/captain_ray`, rooted at
`9ec1c39`, i.e. before any of the abandoned pose-adapter code).
**Core commits:** `d6090d7` (dataset flag + falsifier + config), `25240b3` (carried-over pointworld
configs + eval pipeline), `410b36f` (`demo_ray.py`).
**All paths below are relative to this directory (`src/CUT3R/` inside the `my-da3` repo).**

---

## TL;DR

Captain Ray is an **oracle experiment**: at every timestep of CUT3R's recurrent rollout, the model
is additionally given the **ground-truth camera (extrinsics + intrinsics) of that same timestep**,
and we measure how much this improves online 3D reconstruction (depth / pointmap quality) on DROID
wrist-camera sequences. GT cameras are not available at deployment, so the result is an **upper
bound**: it tells us how much headroom exists for any future method that estimates or memorizes
camera geometry better (e.g. a pose-aware memory).

The conditioning channel is **not** a new module. CUT3R's public checkpoint
(`cut3r_512_dpt_4_64.pth`) already contains a fully pretrained **ray-map encoder branch** — 29
tensors (`patch_embed_ray_map.*`, `enc_blocks_ray_map.{0,1}.*`, `enc_norm_ray_map.*`,
`masked_ray_map_token`) — whose entire job is to inject a *known* camera into a view's encoder
tokens. Upstream training simply never activates it for non-metric datasets. Captain Ray flips it
on **deterministically for every non-reference view**, with **zero new parameters**, a strict
(`<All keys matched successfully>`) checkpoint load, and a **built-in falsifier**
(`GT_RAY_MAP_SHUFFLE=1`) that feeds *wrong* cameras with identical masks to prove the model is
using ray *content*, not just the mask pattern.

Everything is behind one dataset flag, `feed_gt_ray_map` (default `False` → byte-identical
behavior to upstream), wired into one config, `config/captain_ray.yaml`, trained with the stock
`train_cut3r_baseline.py` entrypoint, and served at inference by `demo_ray.py` (a `demo.py` clone
that additionally streams the per-frame pose `.npz` files and reproduces the *training* image/ray
preprocessing bit-exactly).

---

## Table of contents

1. [Problem statement and motivation](#1-problem-statement-and-motivation)
2. [Background: the CUT3R facts the design hinges on](#2-background-the-cut3r-facts-the-design-hinges-on)
3. [The design space: every route we considered](#3-the-design-space-every-route-we-considered)
4. [What exactly was implemented (with code citations)](#4-what-exactly-was-implemented)
5. [Mechanism walk-through and diagrams](#5-mechanism-walk-through)
6. [Verification and experimental evidence](#6-verification-and-experimental-evidence)
7. [Caveats and threats to validity](#7-caveats-and-threats-to-validity)
8. [Evaluation protocol](#8-evaluation-protocol)
9. [Reproduction commands](#9-reproduction-commands)

---

## 1. Problem statement and motivation

CUT3R (`ARCroco3DStereo`, `src/dust3r/model.py`) performs *online* metric-ambiguous 3D
reconstruction: views arrive one at a time, each view's encoder tokens are integrated into a
recurrent state (768-d state tokens + a `LocalMemory` pose memory, `model.py:145` /
`model.py:266`), and per-view heads decode pointmaps (`pts3d_in_self_view`,
`pts3d_in_other_view`), depth, confidence, and camera pose. Every prediction after view 0 depends
on how well the recurrent state has *implicitly* localized the current camera relative to the
past.

The research question: **how much of CUT3R's reconstruction error is attributable to camera
localization error, as opposed to per-view geometry (depth) error?** If we hand the model the
exact camera of every frame — an oracle it can never have at test time in the real world — and
depth/pointmap metrics improve substantially, then camera-side improvements (better pose
estimation, pose-aware memory, external odometry) are worth pursuing. If they don't improve, the
bottleneck is elsewhere and pose-side engineering is wasted effort. This kind of oracle
upper-bound probe is only meaningful if the conditioning pathway itself is sound, which drives
every design decision below.

Constraints we imposed on ourselves:

- **C1 — No new trainable parameters.** If we bolt on a fresh adapter, an improvement could come
  from added capacity rather than from the GT information, and a *failure* could come from the
  adapter being undertrained rather than the information being useless. The oracle must be
  confounder-free.
- **C2 — Deterministic conditioning at every timestep.** The GT camera of frame *t* must condition
  frame *t*'s own prediction, for all *t* (the reference frame *t*=0 excepted, see §5.3), during
  both finetuning and inference. No stochastic masking as in upstream pretraining.
- **C3 — No change to the optimization semantics.** Loss definition, normalization, sampling — all
  unchanged, so the only difference between the oracle run and its baseline is the conditioning
  bit.
- **C4 — Falsifiable.** We need a control that keeps the computational graph identical (same
  masks, same token additions) but destroys the *information*, to distinguish "the model exploits
  GT geometry" from "the model reacts to a distribution shift in its inputs".

## 2. Background: the CUT3R facts the design hinges on

### 2.1 Two encoder modalities, fused additively

`_encode_views` (`model.py:576`) accepts, for every view, an image and a ray map, plus two
independent booleans `img_mask` and `ray_mask`. The fusion is a per-token **sum**
(`model.py:632–648`):

```
tokens(view) =  ⎧ img_tokens(view)         if img_mask   ⎫     ⎧ ray_tokens(view)        if ray_mask ⎫
                ⎨                          +              ⎬  +  ⎨                                     ⎬
                ⎩ masked_img_token         otherwise      ⎭     ⎩ masked_ray_map_token    otherwise*  ⎭
```

- Image branch: `_encode_image` — the full 24-block ViT-Large encoder.
- Ray branch: `_encode_ray_map` (`model.py:533–541`) — `patch_embed_ray_map` (a Conv2d
  `1024×6×16×16` patchifier over the 6-channel ray map, `model.py:330`) → **two** transformer
  blocks (`enc_blocks_ray_map`, `model.py:246`) → `enc_norm_ray_map` (`model.py:259`).
- `*` the masked-ray token is only added when at least one view in the batch has rays — the quirk
  dissected in §5.4.

Because `img_mask` and `ray_mask` are independent, a view can be fed **image AND rays together**:
appearance for depth, rays for camera geometry. That is exactly the captain-ray configuration.

**All 29 parameters of the ray branch ship in the public checkpoint** (verified by loading
`cut3r_512_dpt_4_64.pth` and enumerating keys matching `ray_map`):

```
patch_embed_ray_map.proj.{weight,bias}          (1024,6,16,16), (1024,)
enc_blocks_ray_map.{0,1}.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,mlp.fc2}.{weight,bias}   ×24
enc_norm_ray_map.{weight,bias}                  (1024,), (1024,)
masked_ray_map_token                            (1,1024)
```

So the checkpoint loads **strictly** with the flag on — there is no `strict=False`, no randomly
initialized module anywhere. This satisfies **C1** exactly.

### 2.2 What a ray map is (the pose encoding)

`get_ray_map(c2w1, c2w2, intrinsics, h, w)`
(`src/dust3r/datasets/base/base_multiview_dataset.py:14`) computes the camera of view *i*
**relative to view 0**:

```
c2w  = inv(c2w1) @ c2w2                    # view-i camera expressed in view-0's frame
o    = c2w[:3, 3]                          # ray origin  (broadcast to all pixels)   → 3 channels
d(u,v) = normalize( R · K⁻¹ · [u, v, 1]ᵀ ) # unit ray direction per pixel            → 3 channels
ray_map ∈ R^{H×W×6} = concat(o, d)
```

This is an (origin, direction) per-pixel camera parameterization — informationally it is a
**lossless encoding of the relative pose *and* the intrinsics** (up to the pixel grid's angular
resolution): the origin channel carries translation (in the metric units of the source poses), and
the direction field carries rotation and the full calibration (focal lengths, principal point).
It is the same family of conditioning used by raymap-conditioned view-synthesis and pose-free
reconstruction models; crucially for us, **CUT3R's decoder was pretrained to consume it**.

### 2.3 The frame convention matches the loss — no mismatch to fix

The dataset builds every window's ray maps against the window's **first view**:
`first_view_camera_pose = views[0]["camera_pose"]` (`base_multiview_dataset.py:376`) and
`ray_map = get_ray_map(first_view_camera_pose, view_pose, K, h, w)`
(`base_multiview_dataset.py:400–401`). The regression loss relativizes ground truth in the exact
same way: `in_camera1 = inv(gts[0]["camera_pose"])` (`src/dust3r/losses.py:397`). Conditioning
input and supervision target therefore live in the **same coordinate frame by construction** —
there is nothing to align, and this held *before* our change (the ray map was being computed for
every sample and then discarded; see §4.1).

### 2.4 How upstream pretraining used (and didn't use) rays

`get_img_and_ray_masks` (`base_multiview_dataset.py:307–323`) is upstream's mask sampler. Two
facts matter:

1. It is only ever invoked by the *metric* datasets (ARKitScenes, ScanNet, …) — the DL3DV-format
   loader hardcodes `img_mask=True, ray_mask=False` for every view
   (`src/dust3r/datasets/dl3dv.py:345–346`), so on this data family the ray branch was **never**
   exercised in pretraining.
2. Even on metric data the distribution was: view 0 **always** image-only
   (`base_multiview_dataset.py:309–311`), and views *v*>0 sampled `p = [0.80, 0.15, 0.05]` for
   (image-only, ray-only, image+ray). So "image AND rays together" — our regime — was seen only
   ~5% of the time.

Consequences: (a) the pretrained model demonstrably *can* read rays (§6.2), (b) it may
**under-use** them when the image is present (hence the finetune), and (c) view 0 with rays is
strictly out-of-distribution, which independently justifies keeping the reference view ray-free.

### 2.5 Why "encode a pose into the memory" is not a pretrained operation

The recurrent pose memory (`pose_retriever`, a `LocalMemory`, `model.py:145,266`) stores and
retrieves opaque **768-d decoder features**. The only pretrained bridge between that feature space
and actual SE(3) poses is `pose_head`, a `PoseDecoder`
(`src/dust3r/heads/dpt_head.py:296`, class at `src/dust3r/utils/camera.py:13`) — a **one-way**
map, features → 7-d (quaternion + translation) encoding. Its inverse does not exist in the
checkpoint: `PoseEncoder` (`src/dust3r/utils/camera.py:46`) is defined and even imported by the
model (`model.py:27`) but **never instantiated by any pretrained configuration and has no weights
in `cut3r_512_dpt_4_64.pth`**. This asymmetry is what kills the most obvious alternative route
(§3.1).

## 3. The design space: every route we considered

### 3.1 Route A — pose→memory adapter ("captain adapter") — **rejected, but preserved**

*Idea:* encode the GT pose with `PoseEncoder` (or a new MLP) into a 768-d feature and fuse it into
`LocalMemory` at each step, so the memory "knows" the current camera.

*Why it fails the constraints:*
- **Violates C1 fatally.** There is no pretrained pose→feature encoder (§2.5), so the adapter must
  be trained **from scratch** to approximate the inverse of `PoseDecoder` — a 768→7 map that is
  massively non-injective; its "inverse" is a 761-dimensional coset per pose, and nothing
  identifies *which* feature in that coset the memory dynamics expect. The adapter has to discover
  this by gradient descent through the recurrent rollout.
- **Empirically weak.** This route was actually built earlier (it lives on `temp_clean_main`,
  commit `9045208`, "mem_same_step fusion": `gt_pose_input` / `gt_pose_encoder` /
  `gt_pose_fusion` + `captain_cut3r_pointworld_droid.yaml`). With the sensible safety choices
  (zero-initialized output projection so step 0 is a no-op, sharing the finetune LR of 1e-6,
  truncated BPTT), the adapter barely moves: zero-init × tiny LR × truncated credit assignment
  through a recurrence is a recipe for a near-dead pathway. It also initially had an off-by-one
  (pose fused into the memory used at the *next* step, not the current one) — fixed by
  `mem_same_step`, but symptomatic of how invasive the route is.
- **Uninterpretable as an oracle.** If it helps → maybe capacity. If it doesn't → maybe
  undertrained. Either way the upper-bound question stays unanswered.

The code is deliberately **not** on this branch (`captain_ray` roots at `9ec1c39`, the commit
before any adapter code; `git grep gt_pose` returns nothing here).

### 3.2 Route B — any new conditioning module (FiLM, cross-attention adapter, extra pose tokens)

Same C1 violation as Route A in generalized form: every variant introduces from-scratch parameters
between the GT signal and a pretrained representation, so oracle gains/losses are confounded by
adapter capacity/training. Also strictly more machinery than Route E for the same information
content (a ray map already encodes pose+intrinsics losslessly, §2.2).

### 3.3 Route C — flip `is_metric=True` to activate the existing stochastic ray path

*Idea:* the upstream ray machinery already turns on for metric datasets; declare DROID metric.

*Rejected because it violates C3 twice:* (a) `is_metric` also flips the **loss** to metric-scale
supervision via `not_metric_mask` (`losses.py:446–448`) — the comparison against the baseline
would then confound conditioning with a supervision change; (b) the masks it produces are
**stochastic** (80/15/5, §2.4), violating C2 — we want the oracle *always on*. The correct move is
an **independent flag** that only touches `ray_mask`, which is what `feed_gt_ray_map` is. (The
loader also exposes `is_metric` as a constructor arg defaulting to the historical `False`,
`dl3dv.py:22–23`; captain_ray leaves it `False`.)

### 3.4 Route D — ray-only feeding (`img_mask=False` for conditioned views)

In-distribution (15% of upstream's metric-data samples) and maximally forces the model to rely on
geometry — but it deletes the appearance signal that depth prediction needs, so it answers a
different question ("can the model do geometry from cameras alone") rather than ours ("does known
camera geometry *add* to what images provide"). Kept in the back pocket as a fallback if the
image+ray regime under-uses rays even after finetuning (§7), not used as the primary design.

### 3.5 Route E — deterministic GT ray maps through the pretrained branch — **chosen**

Satisfies every constraint simultaneously:

- **C1**: zero new parameters; strict checkpoint load (§2.1, verified §6.2).
- **C2**: `feed_gt_ray_map=True` sets `ray_mask=True` for *every* view *v*≥1, every sample, train
  and test and inference (`demo_ray.py`), no randomness.
- **C3**: the flag touches only the two mask booleans; loss, normalization, sampling, and the
  ray-map computation itself (which was already running, §4.1) are untouched. Baseline = same
  config with `feed_gt_ray_map=False`, which is **byte-identical** to upstream behavior (verified,
  §6.1).
- **C4**: the `GT_RAY_MAP_SHUFFLE=1` falsifier feeds wrong-but-plausible cameras under identical
  masks (§4.2).
- And the conditioning provably reaches the predictions **zero-shot** — before any finetuning, the
  pretrained model's outputs for a view move by up to ~0.94 (pose encoding) / ~0.6 (pointmap
  max-abs) when that view's ray content changes (§6.2). The pathway is alive; finetuning only has
  to strengthen it, not create it.

Note on "conditioning the memory": rays enter at the **encoder**, not the memory — but the
recurrent decoder alternates attention between view tokens and state/memory at every block, so
ray-conditioned tokens influence (i) the current view's own heads *in the same step* (satisfying
the "decoded for that same timestep" requirement) and (ii) everything the state/memory carries
forward. Encoder-side injection is thus a *superset* of memory-side injection, using only
pretrained plumbing.

## 4. What exactly was implemented

Three code changes (commit `d6090d7`), one config, one inference script (commit `410b36f`). Diff
surface is intentionally minimal.

### 4.1 `src/dust3r/datasets/dl3dv.py` — the `feed_gt_ray_map` flag

- Constructor (`dl3dv.py:22–30`): new keyword `feed_gt_ray_map=False`, stored on the instance.
  Default `False` reproduces upstream **byte-for-byte** (verified, §6.1).
- `_get_views` (`dl3dv.py:277`) still builds every view dict with the upstream defaults
  `img_mask=True, ray_mask=False` (`dl3dv.py:345–346`). After the per-view loop
  (`dl3dv.py:353–360`):

```python
if self.feed_gt_ray_map:
    # GT-pose oracle: expose each view's ground-truth camera to the model
    # through the pretrained ray-map encoder branch (img_mask stays True,
    # so the image is fed alongside the rays). View 0 keeps ray_mask=False:
    # it defines the reference frame, so its relative pose is identity and
    # its ray map carries no pose information.
    for v in range(1, len(views)):
        views[v]["ray_mask"] = True
```

The key observation that makes this a ~8-line change: **the base dataset already computes the GT
ray map for every view of every sample** (`base_multiview_dataset.py:400–401`, from the cam
`.npz`'s `pose` and `intrinsic` entries, relativized per §2.3) — upstream just threw it away for
this data family by leaving `ray_mask=False`. We only stop throwing it away.

### 4.2 `src/dust3r/datasets/base/base_multiview_dataset.py` — the falsifier

After the per-view assembly in `__getitem__` (`base_multiview_dataset.py:425–433`):

```python
if os.environ.get("GT_RAY_MAP_SHUFFLE") == "1" and len(views) > 1:
    # Falsifier for GT-ray-map conditioning: cyclically shift the ray maps
    # so every ray-fed view receives another view's (wrong) camera, while
    # supervision (camera_pose/pts3d) stays correct. If metrics do not
    # degrade vs. correct ray maps, the model is ignoring the rays.
    shifted = [views[-1]["ray_map"]] + [w["ray_map"] for w in views[:-1]]
    for view, rmap in zip(views, shifted):
        view["ray_map"] = rmap
```

Design points: it is an **environment variable**, not a config field, so it can never be silently
left on in a training config; it permutes only `ray_map` (masks, images, `camera_pose`, `pts3d`
supervision untouched), so oracle-vs-falsifier is a *pure ray-content* contrast with an identical
computational graph — the clean control demanded by C4 and by the masked-token quirk (§5.4). The
cyclic shift (rather than random noise) keeps the wrong rays *marginally plausible* — real
cameras from the same window — so the model cannot detect and discount them as off-manifold. This
is **not part of the main experiment**; it exists solely as the validation control.

### 4.3 `config/captain_ray.yaml` — the experiment config

A copy of the overfit config `cut3r_pointworld_droid.yaml` with exactly two semantic differences:

1. Top-level `feed_gt_ray_map: True` (`captain_ray.yaml:38`, with an explanatory comment block),
   interpolated into **both** the train and test `DL3DV_Multi(...)` dataset strings as
   `feed_gt_ray_map=${feed_gt_ray_map}` (`captain_ray.yaml:40–41`) — so evaluation is conditioned
   the same way as training, and the **unconditioned baseline is a one-token CLI override**:
   `feed_gt_ray_map=False`.
2. `exp_name: 'captain_ray'` (`captain_ray.yaml:79`), routing checkpoints/logs to
   `/scratch/bdursun25/cuteanything/checkpoints/cut3r_overfit/captain_ray/`.

Everything else is inherited from the overfit setup: ROOT
`/frozen/avg/bora_data/droid_datasets/training_data/pointworld_droid_wrist_test/dl3dv_multi`,
resolution `[[320,192]]`, `num_views: 64` (train) / 4 (test), 672 samples per epoch per split,
`force_consecutive_frame_sampling: True` (strictly consecutive interval-1 windows),
`pretrained: cut3r_512_dpt_4_64.pth`, entrypoint `train_cut3r_baseline.py` (**no** DA3 bridge, no
monkeypatch — this line of work is orthogonal to the DA3→CUT3R bridge).

### 4.4 `demo_ray.py` — oracle-conditioned inference (functionally `demo.py` + a pose stream)

`demo.py` with one interface change and one fidelity upgrade:

- **Interface** (`demo_ray.py:45–101, 336–356`): new `--pose_path` pointing at the `dense/cam`
  directory of per-frame `.npz` files (keys `pose` = absolute c2w, `intrinsic` = K). Pairing is by
  basename (`000123.png` ↔ `000123.npz`; missing npz = hard error). If `--seq_path` ends in
  `/rgb`, `--pose_path` defaults to the sibling `/cam` (`default_pose_path`, `demo_ray.py:350`).
  Video input is rejected (no pose stream exists for a video file).
- **Training-parity preprocessing** (`crop_resize_training_style`, `demo_ray.py:103`;
  `load_frames_training_style`, `demo_ray.py:136`): frames and intrinsics go through a verbatim
  replica of the training loader's `_crop_resize_if_necessary`
  (`base_multiview_dataset.py:468`) built from the same `src.dust3r.datasets.utils.cropping`
  primitives — **including the principal-point-centered pre-crop** that `demo.py`'s
  `load_images_cover`-style loading does *not* do. This matters: the naive
  cover-resize+center-crop version produced ray directions off by ~0.9° from training; the final
  version is **bit-exact** (max |Δ| = 0.0 on both pixels and ray maps) against the actual training
  loader on real frames (§6.4).
- **Conditioning** (`prepare_input`, `demo_ray.py:193`): each view dict carries the image, its GT
  ray map (built by the same `get_ray_map` against frame 0's pose), `img_mask=True` for all, and
  `ray_mask = (i > 0)` — the exact captain-ray training pattern. Downstream (`inference()` →
  `loss_of_one_batch(..., inference=True)` → the same `_forward_impl`/`_encode_views` as
  training), output saving (`prepare_output`, `demo_ray.py:233`) and the viewer are identical to
  `demo.py`, so `check_depths.py` and existing tooling consume the outputs unchanged.

**`demo.py` and `eval_pipeline/` were deliberately not modified** — they remain ray-free, so the
same checkpoint can be evaluated conditioned (demo_ray) and unconditioned (demo) without code
switches.

## 5. Mechanism walk-through

### 5.1 End-to-end data flow

```
 dense/cam/000123.npz                dense/rgb/000123.png
 ┌───────────────────┐               ┌──────────────────┐
 │ pose  (abs. c2w)  │               │  RGB frame       │
 │ intrinsic (K)     │               └────────┬─────────┘
 └─────────┬─────────┘                        │  _crop_resize_if_necessary (base:468)
           │ relativize to window's view 0    │  (pp-centered crop → Lanczos → pp-centered
           │ c2w = inv(P₀) · Pᵢ  (base:376)   │   final crop; K adjusted alongside)
           ▼                                  ▼
   get_ray_map(P₀,Pᵢ,K,H,W)  (base:14)   img ∈ R^{3×192×320}  (ImgNorm → [-1,1])
   ray_map ∈ R^{192×320×6}                    │
           │                                  │
           │ ray_mask=True (v≥1)              │ img_mask=True (all v)
           │ [dl3dv.py:353-360]               │
           ▼                                  ▼
   _encode_ray_map (model:533)         _encode_image
   patch_embed 6ch/16px (model:330)    (24-block ViT-L)
   → 2 tx blocks (model:246)                  │
   → LayerNorm (model:259)                    │
           │       ┌──────── + ───────────────┘
           └──────►│  token-wise SUM  (model:645: full_out[ray_mask] += ray_out)
                   ▼
        fused view tokens (view v now "knows" its GT camera)
                   ▼
   recurrent decoder: view tokens ⇄ state tokens ⇄ LocalMemory (pose_retriever)
   — conditioning reaches BOTH the current step's heads AND the carried-forward state —
                   ▼
   DPT head → pts3d/depth/conf        pose_head (PoseDecoder, dpt_head:296) → camera
```

### 5.2 Per-view mask pattern (a window of N views)

```
view:        0 (reference)   1        2        ...      N-1
img_mask:    True            True     True     ...      True
ray_mask:    False           True     True     ...      True
ray_map:     identity rays   GT rel.  GT rel.  ...      GT rel.   (computed for all,
             (uninformative) pose 1   pose 2            pose N-1   encoded only where masked True)
```

View 0 stays ray-free for two independent reasons: (i) its relative pose is the identity by
construction, so its ray map carries zero pose information (only K, which the image already
implies); (ii) upstream pretraining *never* fed view-0 rays (`base_multiview_dataset.py:309–311`),
so doing so would be gratuitously out-of-distribution.

### 5.3 Windows, reference frames, and memory initialization

With `force_consecutive_frame_sampling=True`, a training sample is a strictly consecutive window
of `num_views` frames starting anywhere in an episode. Everything is **re-anchored per window at
load time**: the npz poses are absolute, but the model only ever consumes the *relative*
transforms `inv(P_start) · P_i` (rays, §2.3) and is only ever supervised on the same relative
quantities (loss, `losses.py:397`). The recurrent state and `LocalMemory` are initialized fresh
for every window from learned initial tokens — there is no cross-window carry-over — so a window
starting mid-episode is self-consistent: its view 0 *is* the origin of that window's world frame.
The same semantics hold at `demo_ray.py` inference: frame 0 of whatever directory you point at
becomes the reference, and all npz poses are relativized against it.

### 5.4 The masked-token quirk (upstream, `model.py:639–659`) — why the falsifier is the control

`_encode_views` has two branches:

- **Some view in the batch has rays** (`model.py:639–648`): ray tokens are added where
  `ray_mask=True`, and the learned `masked_ray_map_token` is added to every view where
  `ray_mask=False` (`model.py:646`) — including our view 0.
- **No view anywhere has rays** (`model.py:650–659`): a dummy ray encoding is computed but
  multiplied by `0.0` — *nothing* is added, not even the masked token.

Consequence: the oracle run and the no-ray baseline differ at view 0 by a **constant, pose-free
token offset** (empirically ~4e-3 max-abs in pts3d). It cannot leak pose (the token is a single
learned constant), but it means "oracle vs baseline" is not a *perfectly* pure contrast at the
reference view. The **shuffle falsifier is immune to this**: it keeps the exact mask pattern (and
therefore the exact token additions) and changes only ray *content*, making it the clean control
for "is the model using the GT geometry". This is also why the model-level causality test (§6.2)
compares correct-rays vs wrong-rays (bit-identical at view 0) rather than rays vs no-rays.

## 6. Verification and experimental evidence

### 6.1 Loader level

On a real DROID wrist episode (`RAIL+eh61f232+…/13062452+wrist`, 672 frames):

- `feed_gt_ray_map=False` produces views **byte-identical** to the pre-change code (masks, images,
  ray maps, every dict field) — the flag is a true no-op when off (C3).
- Flag on: `ray_mask == [False, True, …, True]`, `img_mask` all `True`, ray maps exactly
  `get_ray_map(P₀, Pᵢ, K, h, w)` of the window's GT poses.
- `GT_RAY_MAP_SHUFFLE=1` changes ray maps only — supervision tensors verified untouched.
- Expected physical sanity: for consecutive windows at an episode start, inter-view ray deltas are
  tiny (~1e-5 — the arm barely moves), growing to O(1e-1…1) deeper into the episode. This is a
  property of the data (near-static early windows), not a bug; it is why `num_views: 64` (real
  motion within a window) matters for the oracle to have signal.

### 6.2 Model level, zero-shot (pretrained checkpoint, before any finetuning)

CPU forward of the real `cut3r_512_dpt_4_64.pth` on a real 3-view sample, three conditions —
**A**: GT rays (`ray_mask=[F,T,T]`); **B**: grossly wrong rays, same masks; **C**: no rays:

- Checkpoint loads **strictly** (`<All keys matched successfully>`) with all 29 ray tensors — zero
  new parameters (C1).
- Mixed img+ray masks forward cleanly.
- **Causality:** view 0's `pts3d_in_self_view` and `camera_pose` are **bit-identical** A vs B
  (|Δ| = 0.0): other views' ray content cannot reach the reference view (the rollout is causal and
  view 0's rays are never encoded). A vs C at view 0 shows only the expected constant masked-token
  offset (§5.4).
- **Sensitivity:** for ray-fed views, wrong-vs-GT rays move `camera_pose` by up to **~0.94**
  (max-abs on the 7-d encoding) and pointmaps by **~0.6** — with the image present. The pretrained
  model does **not** ignore ray content even zero-shot; Route E's pathway exists before we train a
  single step.

### 6.3 The first oracle finetune (2026-07-06)

`accelerate launch --multi_gpu --num_processes 2 train_cut3r_baseline.py --config-name
captain_ray` (2× L40S, `NCCL_P2P_DISABLE=1`), 50 epochs over the 672-scene overfit split, 6h22m
wall-clock. Checkpoints at `/scratch/bdursun25/cuteanything/checkpoints/cut3r_overfit/captain_ray/`
(`checkpoint-{10,20,30,40,50,best,last,final}.pth`).

- Test loss: 0.715 → ~0.006–0.007 plateau; test `pose_loss` ≈ 0.0054 (expected to be near-trivial:
  under conditioning, pose prediction is largely reading back the input — which is why **depth /
  pointmap metrics, not pose metrics, are the oracle's judgment criteria**).
- Per-view test `self_pts3d` at convergence (4 test views): **view 1 (reference, no rays) 0.158
  vs. ray-conditioned views 2–4 at 0.087 / 0.079 / 0.075** — a stable ~2× gap that emerged early
  and persisted. Consistent with (though not yet proof of — see §8) the rays doing real work: the
  one view without GT-camera conditioning is reconstructed markedly worse than its neighbors from
  the same scenes.

### 6.4 `demo_ray.py` verification

- **Preprocessing parity:** on real frames, images and ray maps produced by
  `load_frames_training_style` match the training loader **bit-exactly** (max |Δ| = 0.00e+00 for
  both), after fixing the initial version's ~0.9° ray mismatch caused by omitting the
  principal-point-centered pre-crop. Inference is therefore a true replica of the training-time
  forward, not an approximation.
- **End-to-end:** 6-frame sequence, `checkpoint-final.pth`, CPU: runs clean; pose read-back
  against GT relative poses — rotation errors 0.44° / 0.19° / 0.11° at frames 1/3/5 (GT rotations
  0.24° / 0.77° / 1.24°), translation *direction* cosines +0.967 / +0.997 / +0.998 (direction is
  the meaningful quantity at millimeter-scale baselines under scale-normalized supervision). The
  conditioned model reproduces the injected cameras almost exactly, confirming the full
  npz → ray → encoder → decoder → pose-head loop at inference.

## 7. Caveats and threats to validity

1. **Distribution mismatch, image+ray:** upstream saw this combination ~5% of the time and never
   on DL3DV-format data (§2.4); zero-shot the model may under-weight rays when images are present.
   The finetune is the remedy; if post-finetune gains are weak *but the falsifier still shows
   sensitivity*, escalate to (a) ray-only feeding for some views (Route D) or (b) longer/joint
   finetuning — the information is reaching the network, it just isn't being exploited.
2. **Near-degenerate windows:** consecutive windows at episode starts have ~identity relative
   poses; their rays are nearly uninformative (§6.1). With `num_views=64` most windows span real
   motion, but per-window conditioning strength is heterogeneous.
3. **Scale semantics:** ray origins carry the npz poses' metric translation, while supervision is
   scale-normalized (`is_metric=False` → `not_metric_mask` all-true, `losses.py:446–448`). The
   network must internally reconcile a metric conditioning signal with normalized targets; for an
   overfit oracle this is learnable, but it is a genuine representational wrinkle to keep in mind
   when reading absolute numbers.
4. **Pose metrics are contaminated by construction:** a conditioned model that merely echoes its
   input aces pose evaluation. All oracle conclusions must rest on depth / pointmap metrics
   (abs_rel, δ<1.25, `self_pts3d`).
5. **View-0 masked-token offset:** oracle-vs-baseline is impure at the reference view by a
   constant token (§5.4); use the falsifier for the clean contrast, and exclude view 0 (or report
   it separately) in per-view comparisons.
6. **Oracle ≠ method:** GT cameras do not exist at deployment. A positive result licenses
   *research directions* (pose-aware memory, external odometry injection), not a system.

## 8. Evaluation protocol

The go/no-go rests on a three-arm comparison, identical in everything but the conditioning signal:

| Arm | Command delta | Question it answers |
|---|---|---|
| **Oracle** | `--config-name captain_ray` (done, §6.3) | ceiling with GT cameras |
| **Baseline** | `--config-name captain_ray feed_gt_ray_map=False exp_name=captain_ray_baseline` | same recipe, no conditioning |
| **Falsifier** | `GT_RAY_MAP_SHUFFLE=1 … --config-name captain_ray exp_name=captain_ray_shuffled` | same graph, wrong cameras |

Decision logic on depth/pointmap metrics (per-view, view 0 excluded per §7.5):

- Oracle ≫ Baseline **and** Oracle ≫ Falsifier → GT camera geometry genuinely improves
  reconstruction; pose-side research has headroom. (The falsifier arm is expected to be *worse*
  than baseline — actively wrong geometry should hurt.)
- Oracle ≈ Falsifier → the model ignores ray content; any Oracle-vs-Baseline gap is the mask/token
  distribution shift, not information. Escalate per §7.1.
- Oracle ≈ Baseline with Falsifier ≪ both → rays are read but redundant given images on this data;
  the bottleneck is not camera localization.

Inference-side evaluation uses `demo_ray.py` (conditioned) vs `demo.py` (unconditioned) on held-out
sequences, with `check_depths.py` / `eval_pipeline/` downstream of both (outputs are
format-identical).

## 9. Reproduction commands

All from the worktree, env `cuteanything`. Never on the login node — GPU work goes through SLURM.

```bash
cd /scratch/bdursun25/cuteanything/captain_ray/src/CUT3R/src
export PYTHONPATH=/scratch/bdursun25/cuteanything/captain_ray:$PYTHONPATH
export DL3DV_CACHE_DIR=/scratch/bdursun25/cuteanything/.dl3dv_cache
export NCCL_P2P_DISABLE=1     # broken PCIe P2P on this cluster — required for multi-GPU

# Oracle (GT rays at every timestep)
HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu --num_processes 2 \
    train_cut3r_baseline.py --config-name captain_ray

# Baseline (identical, conditioning off)
HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu --num_processes 2 \
    train_cut3r_baseline.py --config-name captain_ray \
    feed_gt_ray_map=False exp_name=captain_ray_baseline

# Falsifier (identical graph, cyclically wrong cameras)
GT_RAY_MAP_SHUFFLE=1 HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu --num_processes 2 \
    train_cut3r_baseline.py --config-name captain_ray exp_name=captain_ray_shuffled
```

```bash
# Oracle-conditioned inference (pose stream auto-derived from .../rgb → .../cam)
cd /scratch/bdursun25/cuteanything/captain_ray/src/CUT3R
python demo_ray.py \
    --model_path /scratch/bdursun25/cuteanything/checkpoints/cut3r_overfit/captain_ray/checkpoint-final.pth \
    --seq_path  <episode>/dense/rgb \
    --output_dir <out> --size 320 [--device cuda] [--disable_viewer]
```

Outputs are standard `demo.py` format (`depth/`, `conf/`, `color/`, `camera/` per frame).
