# v2 (2026-10-07): birth frames, masks and tracks with the PointWorld exterior extrinsics

v1 (`README.md`) projected the gripper with the raw DROID metadata `ext{1,2}_cam_extrinsics`. MN5 found that
the PointWorld `optimized_extrinsics` (`<ep>_cameras.json`, keyed by camera serial, 4x4 base->camera, used
as-is with the ZED factory intrinsics; on Leonardo in `$CINECA_SCRATCH/ext_build/meta/cameras/`) remove most
of the "22.7 % wrong extrinsics" problem. v2 redoes the pipeline with them. **Nothing in v1 was modified or
deleted**; everything new is in `recheck_pointworld/`, `birth_frames_v2/`, `masks_run_v2/`, `trackon_run_v2/`
and the `*_v2.*` files next to the v1 ones.

## 1. Decisive check: the kinematic flag recomputed with PointWorld extrinsics (`recheck_pointworld/`)

`recheck_projection_pointworld.py` re-evaluates every v1 RobotSeg mask (13 430 of 13 432; two cameras have
no PointWorld entry) with the SAME rule (> 20 % of the automatic mask outside the projected 54-point box +
60 px) under both calibrations. The v1 flag is reproduced exactly with the raw extrinsics (13 430 / 13 430).

| set | masks | disagree raw | disagree PointWorld | raw only | PointWorld only | both |
|---|---|---|---|---|---|---|
| b050 | 8520 | 1967 (23.1 %) | 392 (4.6 %) | 1640 | 65 | 327 |
| b100 | 4910 | 1074 (21.9 %) | 115 (2.3 %) | 988 | 29 | 86 |
| all | 13430 | 3041 (22.6 %) | 507 (3.8 %) | 2628 | 94 | 413 |

Per lab (b050): TRI 25.8 -> 3.2 %, AUTOLab 15.9 -> 2.4 %, RAIL 27.5 -> 17.4 %, IPRL 11.1 -> 1.3 %, REAL 33.4 -> 2.1 %,
IRIS 42.2 -> 3.0 %, CLVR 33.7 -> 11.9 %, ILIAD 19.0 -> 4.6 %, RPL 9.8 -> 0.8 %, PennPAL 16.1 -> 2.1 %,
WEIRD 17.6 -> 1.1 %, GuptaLab 5.1 -> 1.4 %, RAD 8.8 -> 1.0 %. Scenes with >= 1 flagged camera: 1547 -> 415.

Projected gripper-centroid shift raw -> PointWorld at the b050 frame: flagged cameras median 194 px
(p10 66, p90 351; 92 % > 30 px), unflagged median 11 px — the same picture MN5 got from the box centroid
(220 / 12.5 px). Distance mask centroid -> projected centroid: raw median 45 px, p90 253 px, > 100 px in 28 %;
PointWorld median 30 px, p90 69 px, > 100 px in 5.8 %.

Contact sheets (red box = raw, green = PointWorld, blue tint = automatic mask): `sheet_fixed_by_pw.jpg` — in all
40 sampled cameras the green box sits on the mask and the red one elsewhere. `sheet_still_disagree_pw.jpg` —
the residual 4.6 % is a mix of (a) the gripper cut by the top edge of the frame at birth, with the mask
extending along the arm beyond the clipped box, (b) RobotSeg false positives on background (dark scenes,
black gripper), (c) genuinely remaining offsets, mostly RAIL and CLVR (the two labs still > 10 %).
Conclusion: PointWorld confirmed; the PnP + RANSAC re-estimation of v1's "suggested next step" is unnecessary.

## 2. Birth frames v2 (`birth_frames_v2/`)

`birth_frames.py --extrinsics pointworld` (new default; `raw` reproduces v1). `compare_v1_v2.md/.csv`:

- b050: same 2960, moved 1301, recovered 29, lost 2 (of 4292). Moved: earlier in 634 / later in 667; |delta|
  median 4 frames, p90 33, max 321 (<= 2 frames: 555, 3-10: 283, 11-30: 315, > 30: 148); 177 moved scenes
  now have the gripper visible from frame 0.
- b100: same 2247, moved 1939, recovered 66, lost 40.
- Changes concentrate where v1 flagged: of the 463 scenes moved by > 10 frames, 383 were in
  `scenes_extrinsics_suspect.txt` (266 in the severe list). Per lab the b050 change rate is RAIL 58 %, REAL 65 %,
  CLVR 64 %, ILIAD 47 %, IRIS 44 %, AUTOLab 24 %, TRI 18 %, IPRL 13 %, RPL 12 %, WEIRD 5 %, RAD 2 %.
- The 30 v1 "never visible in both" scenes: 29 get a b050 birth frame (all 21 judged "bad extrinsics" plus
  the borderline and "genuine" ones, several at frame 0); `TRI+52ca9b6a+2024-01-25-13h-35m-10s` stays at
  ext2 max 0.44. The 2 lost scenes (`IRIS+226b9f35+2023-07-11-11h-46m-06s`, `IRIS+f94b8622+2023-12-12-19h-39m-25s`,
  ext2 serial 29838012) have ext2 max fraction 0.17 / 0.0 under PointWorld; their v1 ext2 masks were already
  flagged (spill 1.0 / 0.98 vs the raw projection), so neither calibration agrees with the mask there.
- One scene (`TRI+52ca9b6a+2023-11-07-14h-29m-08s`) has no PointWorld entry for either serial and falls back
  to the raw extrinsics (`ext{1,2}_extrinsics = raw_fallback` in `birth_frames.csv`).

## 3. Masks v2 (`masks_run_v2/`, `birth_masks_v2.csv`, `birth_masks_summary_v2.md`)

RobotSeg re-run (same model, same single-frame automatic prompt, same box fallback and refine policy) on the
2235 scenes whose b050 or b100 frame changed: 8381 masks (b050 4465, b100 3916), box used only for 11
(automatic mask empty). The kinematic flag under PointWorld is 5.2 % at b050 (RAIL 16 %, CLVR 7.6 %, others
<= 5.9 %) — consistent with step 1. Scenes whose frames did not change keep their v1 masks.

## 4. Tracks v2 (`trackon_run_v2/`)

Track-On-R re-run with IDENTICAL settings to v1 (`track_on_r.pt`, 20x20 support grid, visibility threshold
0.8, same grid sampling inside the eroded b050 mask, target 64 / max 100 points; verified in the npz `meta`) for
the 2659 (scene, camera) items whose b050 frame moved or was recovered (1330 scenes), 16 A100 shards, ~35 min
each, 0 failures. `trackon_run_v2/tracks/` is a COMPLETE set of 8576 items (4289 scenes): 5917 unchanged items
are symlinks to the v1 npz files, 2659 are new. v1 had 8522 items / 4262 scenes: +58 items from the 29 recovered
scenes, -4 from the 2 lost ones. `tracks_index.csv` and `summary.md` regenerated over the full set (v1 log
records of the reused items are in `logs/track_shard_v1_reused.jsonl`, `reused_from_v1 = true`).

Over the whole set the visibility statistics are unchanged (median visible fraction 0.74, 16.5 % of items
end with < 25 % visible). For the 2600 re-tracked items that also existed in v1: birth frame earlier in 1267,
later in 1333 (median |delta| 4 frames); median visible fraction 0.77 -> 0.76, last-frame visibility
0.79 -> 0.77, i.e. the tracker behaves the same, only the start frame moved. The new birth frame is the
kinematically correct one; nothing else about the tracks changed.

Shipping: `ship/trackon_run_v2.tar` (+ `.sha256` manifests, symlinks dereferenced), same layout as v1.

## Code (branch `vggt_features`)

`vggt_probe/birth_frames.py` (`--extrinsics`, `load_T_bc`), `recheck_projection_pointworld.py`,
`compare_birth_frames_v1_v2.py`, `summarize_birth_masks.py --suffix _v2`,
`robotseg_stage/{segment_birth_frames.py --birth_json/--scenes/--out/--extrinsics, refine_birth_masks.py --run,
run_segment_birth_frames_v2.sbatch, run_refine_v2.sbatch}`,
`trackon_stage/{build_worklist.py, build_trackon_v2.py --phase worklist|assemble, track_gripper_birth_v2.sbatch,
summarize_gripper_tracks.py --run}`.

## Exclusions (after the MN5 cross-check, 2026-10-07)

`trackon_run_v2/scenes_to_exclude.tsv`: `TRI+52ca9b6a+2023-11-07-14h-29m-08s` has tracks in v1 and v2 but its MP4s
are the success recording while the store wrist poses are the failure recording with the same id (video and poses
from different episodes) — exclude it. The two IRIS scenes without a v2 birth frame are listed for completeness;
MN5's depth-based extrinsics check fails on their ext2 camera as well.
