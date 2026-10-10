# Shelf: the single-episode success (RAIL+80edfcb1+2023-07-14-14h-28m-45s), 2026-09-29

The reference point every later candidate is compared against. Everything on this shelf is training-free, causal
except where noted, and reproducible from the files listed here. Nothing below depends on the seeding-study rounds.

## The masks (human-prompted SAM3, the one thing that is not automatic)

`ext_cams/sam3_gripper_boxclick_3cams.py` (+ `.sbatch`): SAM3 video tracker, box + 3 positive clicks per camera on a
high-contrast frame, propagated backward then forward (non-causal, fine for a label). Prompts are hand-picked for this
episode. Products:

```
$OUT/ext_cams/gripper_sam3/RAIL+80edfcb1+2023-07-14-14h-28m-45s/
    ext1_20521388__gripper_boxclick_masks.npz      gripper visible from frame 27
    ext2_24259877__gripper_boxclick_masks.npz      gripper visible from frame 37
    wrist_13062452__gripper_boxclick_*             wrist mask (not used by the rig)
    all_cams_gripper_boxclick_side_by_side.mp4
```

`$OUT = /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval`. Mask npz: `union` uint8 (T, 720, 160) bit-packed along
width, `shape` [T, 720, 1280], `frames_present` (T,).

## The rig pipeline on those masks

`ext_cams/rig_kabsch_probe.py --res 1280`: Shi-Tomasi seeds in the eroded mask, pyramidal LK forward with a 1.5 px
forward-backward check, re-seeding, cross-view pairing by accumulated epipolar residual (NOTE: over each pair's whole
lifetime, so not strictly causal; fixed in `rig_track2.py`), triangulation with PointWorld extrinsics + factory
intrinsics, growing body model, RANSAC + weighted Kabsch with 5 mm inliers, transfer to the lens with the GT wrist pose
at the reference frame (hand-eye stand-in, the only GT in the method path).

Frames 37..127 (gripper in both exterior views), 84 solved, reference frame 45:

| | ATE | RPE trans | RPE rot | Sim3 scale |
|---|---|---|---|---|
| rig alone, 84 frames | 0.0032 | 0.0015 | 0.799 | 0.991 |
| CUT3R finetuned, same frames | 0.0290 | 0.0056 | 0.870 | 0.213 |

Lens error vs kinematic GT: 0.59 cm median / 0.84 p90 / 0.99 max; rotation 4.3 deg median (half of it a constant 3.4 deg
frame offset). At 320x180 the same pipeline fails (2.1 cm, 17 deg): exterior geometry runs at native resolution.

Products: `$OUT/ext_cams/rig_kabsch/RAIL+.../{summary_res1280.json, per_frame_res1280.csv, tracks_res1280.npz,
all_cams_rig_kabsch_side_by_side_res1280.mp4}`.

## The fusion on those anchors

`ext_cams/rig_fuse_probe.py --backbone finetuned|zeroshot`: CUT3R's wrist-only increments always; on anchored frames the
metric anchor is mapped into CUT3R's frame by a Sim3 estimated once from the first anchored frames and frozen, then a
scalar Kalman pull whose variances are measured online. No constant tuned on GT.

| row | frames | finetuned ATE / RPE_t / RPE_rot | zero-shot ATE / RPE_t / RPE_rot |
|---|---|---|---|
| CUT3R alone | 128 | 0.0252 / 0.0047 / 0.694 | 0.0690 / 0.0097 / 1.220 |
| rig alone | 84 | 0.0032 / 0.0015 / 0.799 | same |
| fused | 128 | 0.0073 / 0.0027 / 0.557 | 0.0798 / 0.0066 / 0.785 |
| CUT3R, anchored frames | 84 | 0.0290 / 0.0056 / 0.870 | 0.0303 / 0.0071 / 1.443 |
| fused, anchored frames | 84 | 0.0038 / 0.0022 / 0.661 | 0.0050 / 0.0024 / 0.777 |

Zero-shot loses on whole-trajectory ATE because its internal scale halves between the unanchored start and the anchored
part (2.3 vs 4.8 units per metre); the first 48 frames cannot be revised causally.

Products: `$OUT/ext_cams/rig_kabsch/RAIL+.../{fuse_summary.json, fused_poses.npz, all_cams_cut3r_rig_fused_side_by_side.mp4}`
and the `_zeroshot` variants.

## What is NOT on this shelf

Automatic seeding (`seed_*.py`), the per-episode trackers (`rig_track.py`, `rig_track2.py`) and the seeding-study
reports under `$OUT/ext_cams/seedstudy*/` are candidates under evaluation, not the reference.

## Later result, for comparison with this shelf (2026-09-30)

`ext_cams/rig_fuse2.py`: the honest fusion (no GT anywhere in the method path; tracker v2 anchors; segment-aware; CUT3R's own
rotation). RAIL, shelf masks, finetuned: fused all frames 0.0169 / 0.0038 / 0.694 against 0.0073 / 0.0027 / 0.557 for the
shelf fusion that used the GT lens pose at the reference frame. Full tables, both backbones, plus the smoke-13 means with the
dino_exemplar and verified_motion_v2 anchors: `$OUT/ext_cams/fuse2/fuse2_results.md`.
