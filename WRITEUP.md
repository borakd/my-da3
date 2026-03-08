# DA3-to-CUT3R Memory Bridge: Implementation Analysis

This document explains the implementation in `with_cut3r.py` as a research-style, line-by-line design rationale.

The goal of this script is not to train a new model. The goal is to run a controlled integration experiment:

1. keep DA3 preprocessing and transformer extraction faithful to DA3 internals,
2. map DA3 tokens into CUT3R encoder token space,
3. write these mapped tokens into CUT3R memory with CUT3R-native recurrent logic,
4. decode outputs with CUT3R's own downstream head and persist all artifacts for audit.

## 1. Why this design is correct for this codebase

The bridge is anchored to existing "native" entry points in each subsystem:

- DA3 side uses:
  - `DepthAnything3._preprocess_inputs` (`src/depth_anything_3/api.py:279`)
  - `DepthAnything3._prepare_model_inputs` (`src/depth_anything_3/api.py:305`)
  - `DepthAnything3._normalize_extrinsics` (`src/depth_anything_3/api.py:331`)
  - `DepthAnything3Net.backbone(...)` via `DepthAnything3Net.forward` semantics (`src/depth_anything_3/model/da3.py:125`, `:132`)
- CUT3R side uses:
  - `dust3r.utils.image.load_images` for image preprocessing (`src/CUT3R/src/dust3r/utils/image.py:75`)
  - `ARCroco3DStereo._forward_encoder` for state initialization (`src/CUT3R/src/dust3r/model.py:735`)
  - `ARCroco3DStereo._recurrent_rollout` for recurrent state transition (`src/CUT3R/src/dust3r/model.py:713`)
  - `LocalMemory.update_mem` for memory writes (`src/CUT3R/src/dust3r/model.py:204`)
  - `ARCroco3DStereo._downstream_head` for decoding (`src/CUT3R/src/dust3r/model.py:700`)

Using these native paths avoids speculative re-implementations and keeps behavior comparable to each model's intended runtime.

## 2. High-level pipeline in `with_cut3r.py`

`with_cut3r.py` is split into four explicit stages in `main()`:

1. `extract_da3_tokens(...)`:
   - generate DA3 transformer tokens using DA3-native preprocessing and backbone execution.
2. `initialize_cut3r_runtime(...)`:
   - build CUT3R views with CUT3R-native preprocessing and initialize recurrent state/memory.
3. `bridge_tokens_and_decode(...)`:
   - adapt DA3 token shape -> CUT3R token shape, run recurrent rollout, update memory, decode.
4. persistence:
   - save decoded outputs, memory-norm trajectory, and metadata.

This stage split is deliberate because each stage corresponds to one conceptual claim you can discuss in research notes:

- Stage 1 claim: "input/token semantics are DA3-faithful"
- Stage 2 claim: "state semantics are CUT3R-faithful"
- Stage 3 claim: "bridge only changes token source, not recurrent/memory mechanics"
- Stage 4 claim: "outputs are reproducible and inspectable"

## 3. Function-by-function analysis

This section explains every function in `with_cut3r.py`: what it calls, what files it interacts with, and why its logic is correct.

### 3.1 Data containers

#### `DA3TokenBundle`
- Purpose:
  - carries DA3 output tokens and DA3 patch-grid metadata.
- Why needed:
  - token adaptation requires both tensor data and source patch geometry (`N = H*W`).

#### `CUT3RRuntime`
- Purpose:
  - stores CUT3R state tensors returned by `_forward_encoder` plus view metadata.
- Why needed:
  - recurrent bridging needs the exact tensors CUT3R expects across steps.

### 3.2 Utility / setup functions

#### `parse_args()`
- Calls:
  - `argparse.ArgumentParser(...)`.
- Interacts with:
  - none (standard library only).
- Correctness rationale:
  - exposes all bridge-critical knobs:
    - DA3 model and processing policy,
    - CUT3R model and preprocessing size,
    - DA3 reference-view strategy (wired to DA3 backbone).

#### `add_repo_python_paths(repo_root)`
- Calls:
  - `sys.path.insert(...)`.
- Interacts with:
  - `src/depth_anything_3` import path resolution,
  - `src/CUT3R/src` (`dust3r` package path).
- Correctness rationale:
  - this repo uses source-layout imports; adding these paths makes script execution independent of installation mode.

#### `resolve_device(device_arg)`
- Calls:
  - `torch.device(...)`, `torch.cuda.is_available()`.
- Interacts with:
  - torch runtime only.
- Correctness rationale:
  - explicit CPU fallback keeps script deterministic on non-CUDA environments.

#### `collect_input_images(input_path)`
- Calls:
  - `glob.glob(...)`.
- Interacts with:
  - filesystem input directory.
- Correctness rationale:
  - extension set matches DA3/CUT3R demos and produces deterministic sorted order.

#### `move_view_to_device(view, device)`
- Calls:
  - `.to(device, non_blocking=True)` on tensor fields.
- Interacts with:
  - CUT3R view dictionaries.
- Correctness rationale:
  - mirrors tensor-transfer style in `src/CUT3R/src/dust3r/inference.py`; leaves metadata keys unchanged.

#### `get_da3_backbone_owner(da3_model_obj)`
- Calls:
  - `hasattr(...)`.
- Interacts with:
  - DA3 nested vs non-nested model structures (`src/depth_anything_3/model/da3.py`).
- Correctness rationale:
  - nested models expose any-view branch at `.da3`; non-nested models expose backbone directly.

#### `as_patch_hw(patch_size)`
- Calls:
  - type checks / casts.
- Interacts with:
  - DA3/CUT3R patch-size attributes that may be scalar or tuple.
- Correctness rationale:
  - removes shape-interpretation ambiguity in grid computations.

#### `get_da3_patch_size(da3_owner)`
- Calls:
  - attribute inspection + `as_patch_hw(...)`.
- Interacts with:
  - DINOv2 `pretrained.patch_size` from `src/depth_anything_3/model/dinov2/vision_transformer.py`.
- Correctness rationale:
  - avoids hard-coded 14 where possible, while preserving fallback to 14 if field is missing.

### 3.3 Token-space adaptation

#### `resize_and_match_tokens(da3_frame_tokens, src_grid_hw, tgt_grid_hw, tgt_dim)`
- Calls:
  - `reshape`, `permute`, `F.interpolate`, `F.pad`.
- Interacts with:
  - tensor geometry only.
- Correctness rationale:
  - CUT3R recurrent decoder expects token tensors with CUT3R encoder token count and embedding width.
  - This function performs minimal deterministic mapping:
    - spatial interpolation for token count mismatch,
    - truncate/pad for channel-width mismatch.
  - This isolates the bridge assumption into one explicit operator.

### 3.4 CUT3R view and state initialization

#### `build_cut3r_views(image_paths, cut3r_size)`
- Calls:
  - `dust3r.utils.image.load_images(...)`.
- Interacts with:
  - `src/CUT3R/src/dust3r/utils/image.py:75`.
- Correctness rationale:
  - by using CUT3R preprocessing directly, image normalization/cropping semantics remain CUT3R-native.
  - uses image-only mode (`ray_mask=False`) exactly like CUT3R demo-style setup.

#### `initialize_cut3r_runtime(args, device, image_paths)`
- Calls:
  - `ARCroco3DStereo.from_pretrained(...)`
  - `build_cut3r_views(...)`
  - `move_view_to_device(...)`
  - `model._forward_encoder(...)`
  - `as_patch_hw(...)`.
- Interacts with:
  - `src/CUT3R/src/dust3r/model.py`:
    - `_forward_encoder` (`:735`)
    - pose-head configuration (`pose_head_flag`, `pose_retriever` setup around `:256`).
- Correctness rationale:
  - `_forward_encoder` is the same state split CUT3R uses in its own recurrent/TBPTT workflows.
  - explicit guard on `pose_head_flag` prevents invalid memory-bridge execution with incompatible checkpoints.

### 3.5 DA3 token extraction

#### `extract_da3_tokens(args, device, image_paths)`
- Calls:
  - `DepthAnything3.from_pretrained(...)`
  - `da3._preprocess_inputs(...)`
  - `da3._prepare_model_inputs(...)`
  - `da3._normalize_extrinsics(...)`
  - `get_da3_backbone_owner(...)`
  - optional `da3_owner.cam_enc(...)`
  - `da3_owner.backbone(...)`
  - `get_da3_patch_size(...)`.
- Interacts with:
  - `src/depth_anything_3/api.py` (`_preprocess_inputs`, `_prepare_model_inputs`, `_normalize_extrinsics`)
  - `src/depth_anything_3/model/da3.py` (backbone and camera-token flow)
  - `src/depth_anything_3/model/dinov2/vision_transformer.py` (intermediate token format).
- Correctness rationale:
  - reproduces DA3's own input and camera-conditioning flow before the head.
  - extracts final-layer patch tokens (`da3_feats[-1][0]`) that DA3 heads consume.
  - validates token count against patch-grid dimensions.

### 3.6 Compatibility and state update policy

#### `validate_bridge_compatibility(da3_tokens, cut3r)`
- Calls:
  - shape comparisons only.
- Interacts with:
  - DA3 token batch/view shapes and CUT3R initialized view/state shapes.
- Correctness rationale:
  - early failure for structural mismatches avoids silent incorrect bridging.

#### `apply_update_and_reset_masks(...)`
- Calls:
  - tensor mask arithmetic.
- Interacts with:
  - CUT3R per-view control fields (`img_mask`, `update`, `reset`).
- Correctness rationale:
  - matches CUT3R policy in `_forward_decoder_step` (`src/CUT3R/src/dust3r/model.py:798-814`).

### 3.7 Core bridge and decode

#### `bridge_tokens_and_decode(da3_tokens, cut3r)`
- Calls:
  - `resize_and_match_tokens(...)`
  - `model.pose_retriever.inquire(...)`
  - `model._recurrent_rollout(...)`
  - `model.pose_retriever.update_mem(...)`
  - `model._downstream_head(...)`
  - `apply_update_and_reset_masks(...)`.
- Interacts with:
  - `src/CUT3R/src/dust3r/model.py`:
    - `_recurrent_rollout` (`:713`)
    - `LocalMemory.update_mem` (`:204`)
    - `_downstream_head` (`:700`)
    - `_forward_decoder_step` logic it mirrors (`:763-814`).
- Correctness rationale:
  - follows CUT3R's recurrent decoding recipe but replaces CUT3R encoder tokens with mapped DA3 tokens.
  - preserves memory read/write and downstream decoding operators, isolating the experimental variable to token source.

### 3.8 Persistence

#### `save_decoded_outputs(decoded_outputs, output_path)`
- Calls:
  - `np.save(...)`, directory creation.
- Interacts with:
  - filesystem outputs under `decoded/<key>/<frame>.npy`.
- Correctness rationale:
  - per-key, per-frame storage enables inspection without hidden post-processing.

#### `save_run_metadata(args, da3_tokens, cut3r, memory_norms, output_path)`
- Calls:
  - `json.dump(...)`.
- Interacts with:
  - `bridge_metadata.json`.
- Correctness rationale:
  - captures bridge-relevant dimensions/configuration for reproducibility and paper notes.

#### `main()`
- Calls:
  - all orchestrating stage functions in sequence.
- Interacts with:
  - both model stacks and output filesystem.
- Correctness rationale:
  - four-stage structure makes the experimental logic explicit and auditable.

## 4. Code changes made beyond comments

The implementation was updated beyond documentation to improve correctness and analysis quality:

1. Introduced explicit stage containers (`DA3TokenBundle`, `CUT3RRuntime`) to avoid implicit tensor coupling.
2. Refactored monolithic flow into stage functions with clear contracts.
3. Added `--da3_ref_view_strategy` to control DA3 backbone routing behavior.
4. Replaced DA3 patch-size hard-code with introspection (`get_da3_patch_size`).
5. Added explicit CUT3R compatibility guard for pose-memory checkpoints.
6. Added metadata artifact (`bridge_metadata.json`) for reproducibility.
7. Added path bootstrapping for source-layout execution in this repo.

## 5. Known bridge assumptions (important for research discussion)

1. Spatial adaptation uses bilinear interpolation over token grids.
2. Channel adaptation is truncate-or-zero-pad (no learned projection).
3. Memory key uses mean pooled DA3 tokens to mirror CUT3R `_get_img_level_feat` behavior.
4. Decoder head is run with CUT3R-native 4-scale token taps from recurrent decoder outputs.

These assumptions are intentionally simple to keep attribution clear: if output behavior changes, the main intervention is token replacement, not an extra learned adaptor.

## 6. Suggested ablations to deepen analysis

1. Replace truncate/pad with a learned linear projection and compare memory norms/output confidence.
2. Compare pooled-key strategies: mean pool vs learned attention pool vs cls-like token synthesis.
3. Evaluate per-layer DA3 tokens (not only final layer) as memory write keys.
4. Sweep DA3 reference-view strategy and CUT3R input size to isolate geometric sensitivity.

