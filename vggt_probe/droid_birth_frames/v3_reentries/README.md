# v3 (2026-10-08): gripper RE-ENTRIES — a RobotSeg mask and a Track-On-R run for every entry event

v2 covers the FIRST entry of the gripper into both exterior views per scene. v3 adds every later re-entry
(the gripper drops below 50 % visibility in either camera and comes back). Only the re-entries were run; the
first entries are the v2 files. v1 and v2 are untouched.

## Events (`events_v3/`)

`gripper_entry_events.py`, definition agreed with MN5: `both[t] = min(frac_ext1, frac_ext2)` from
`birth_frames.analyze_episode` with the v2 settings (PointWorld extrinsics, raw fallback), `vis = both >= 0.5`,
runs of `vis`, consecutive runs merged when `next_start - prev_end < 3`, one event per merged run with
`entry_frame` = run start and `exit_frame` = run end (exclusive, `n_frames` if it lasts to the end). The 3
scenes in `trackon_run_v2/scenes_to_exclude.tsv` are dropped.

Counts, identical to MN5's: event 0 == v2 `birth_f050` in 4292/4292 scenes; after exclusions 5420 events in
4288 scenes, 1132 re-entries in 682 scenes, max 16 events in one scene (1173 re-entries without the
debounce). Re-entry run length median 58 frames (p10 10, p90 264, min 1); 516 re-entries last to the end of
the episode; frames from re-entry to episode end median 141, 252 k in total. Re-entries per lab: TRI 339,
AUTOLab 281, IPRL 116, ILIAD 81, IRIS 59, REAL 58, WEIRD 50, PennPAL 46, RAIL 44, RPL 29, CLVR 27, GuptaLab 1, RAD 1.
Files: `events.csv` (all 5420 events), `reentries.csv` (the 1132), `curves.jsonl` (per-scene `both[t]`), `summary.md`.

## Masks (`masks_run_v3_reentries/`)

`segment_birth_frames.py --events events_v3/events.csv` (new option; `which = e01, e02, ...`), same settings as
the v2 b050 masks: automatic RobotSeg prompt on the entry frame, kinematic box fallback when the automatic mask
is empty, spill flag against the PointWorld projection, then `refine_birth_masks.py`. 2264 masks (1132 events x 2
cameras), all decoded. Final mask automatic for 2248, box for 16 (automatic empty); 20 automatic masks empty.
`projection_disagree` 205 / 2264 = 9.1 % (v2 first entries: 5.2 %) — re-entry frames have the gripper at the
image border more often, so the clipped 54-point box misses part of the mask. `qa_random_reentry_masks.jpg`:
40 random re-entry masks, all on the gripper.

## Tracks (`trackon_run_v3_reentries/`)

Track-On-R with the settings of v1/v2 (`track_on_r.pt`, 20x20 support grid, visibility threshold 0.8, same grid
sampling inside the eroded mask; verified in the npz `meta`) from every re-entry frame to the END of the episode:
2264 items, 8 A100 shards, 50-63 min each (the decoder walks the frames before the entry), 0 failures,
504 k frames tracked. Files `tracks/<ep>/<cam>_f<entry>.npz`, same format as v1/v2, named by entry frame so they
sit next to the v2 file of the same scene without clashing. Points per item median 66 (5 items < 10 points);
visible fraction median 0.84 over the tracked span, 0.85 at the last frame; 466 items (20.6 %) end with < 25 %
of points visible — a re-entry is by definition a gripper that moves in and out of view, and 616 of the 1132
re-entries end before the episode does. `tracks_index.csv`, `summary.md` cover the re-entries only; use
`events_index.csv` for the full picture.

## Index over ALL events (`trackon_run_v3_reentries/events_index.csv`)

One row per (event, camera): `episode, cam, event, entry_frame, exit_frame, n_frames, run, npz, status, n_points,
tracked_frames, vis_frac_mean, vis_frac_last, mask_prompt, projection_disagree, auto_empty, mask_area`.
Event 0 rows point into `trackon_run_v2/tracks/...` (with the v2 mask flags), events >= 1 into
`trackon_run_v3_reentries/tracks/...`. `exit_frame` lets downstream code cut a track where the gripper leaves
the view. `npz` is relative to this outputs root. `events_index_summary.md` has the status counts: 8574 first-entry rows ok, 2 `no_v2_item` (RAIL+d027f2ae+2023-06-08-11h-59m-05s ext1, REAL+4f8ca688+2023-07-11-12h-42m-31s ext2: their MP4 could not be decoded at the birth frame in v1/v2, no track exists), 2264 re-entry rows ok.

Shipped as `ship/trackon_run_v3_reentries.tar` (tracks, worklist, events_index.csv, events_index_summary.md,
tracks_index.csv, summary.md, events.csv, and the per-file manifest `sha256_manifest.txt` INSIDE the tar);
`ship/trackon_run_v3_reentries.tar.sha256` is the tar checksum.

## Code (branch `vggt_features`)

`vggt_probe/gripper_entry_events.py`, `segment_birth_frames.py --events`, `summarize_gripper_tracks.py`
(records keyed by (scene, camera, entry frame)), `trackon_stage/build_events_run.py --phase worklist|index`,
launchers `robotseg_stage/run_segment_reentries_v3.sbatch`, `run_refine_reentries_v3.sbatch`,
`trackon_stage/track_gripper_reentries_v3.sbatch`.
