> Snapshot of `$WORK/bora/outputs/droid_birth_frames` on Leonardo (2026-10-06), small files only. The per-scene masks (`masks_run/`, 3.6 GB), tracks (`trackon_run/tracks/`, 1.1 GB), QA videos, `examples/*.mp4`, `failed_scenes_video_*.mp4`, raw `metadata/` and `logs/` stay on `$WORK`. Stage code: `../robotseg_stage/` (RobotSeg masks) and `../trackon_stage/` (Track-On-R tracks, checkpoint comparison, videos), folded from `~/RobotSeg/test` and `~/track_on/leonardo` on 2026-10-06.

# DROID gripper birth frames + RobotSeg birth-frame masks (test split, 4292 scenes)

Produced 2026-10-05/06 on Leonardo. Code: `~/vggt_features/vggt_probe/{birth_frames.py, fetch_droid_metadata.py,
fetch_droid_mp4s_all.py, make_failed_scenes_videos.py, summarize_*.py}` and `~/RobotSeg/test/{segment_birth_frames.py,
refine_birth_masks.py}`. Raw exterior MP4s (64 GB): `$CINECA_SCRATCH/robotseg_demo/raw/<ep>/recordings/MP4/<serial>.mp4`.

## Deliverables

| file | content |
|---|---|
| `birth_frames.csv` | per scene: birth frame at 25/50/75/100 % gripper visibility in BOTH exterior cameras, per-camera first-visible frames, max visibility (-1 = never). `summary.md` + `birth_frames_hist.png` summarise it. |
| `birth_masks.csv` | per (scene, camera, b050/b100): the frame used, RobotSeg prompt, mask path, kinematic-consistency scores. `birth_masks_summary.md` summarises it. |
| `masks_run/masks/<ep>/<cam>_f<t>.png` | binary gripper mask (255 = gripper) on the birth frame; `_auto.png` / `_box.png` kept where both exist |
| `masks_run/frames/`, `masks_run/overlays/` | the decoded birth frame and the mask overlay |
| `failed_scenes_never_visible_both.{md,csv}`, `failed_scenes.txt`, `failed_scenes_video_{1,2,3}.mp4` | the 30 scenes with no birth frame: per-scene verdict from the review videos (21 bad extrinsics, 7 genuine, 2 borderline) |
| `scenes_extrinsics_suspect.txt` (1548), `scenes_extrinsics_suspect_severe.txt` (853) | scenes whose RobotSeg mask disagrees with the kinematic projection in >= 1 camera (severe: mask entirely outside the projected box at the 50 % birth frame) |
| `validation/`, `validation2/`, `validation3_at_birth/` | kinematic-vs-RobotSeg entry-frame checks (14 scenes), single-frame mask test at the birth frame |
| `qa_random_consistent_b050.jpg`, `qa_random_projection_disagree.jpg`, `qa_disagree_with_projection.jpg`, `qa_auto_empty_box_used.jpg` | QA contact sheets |

## What holds and what does not

- The kinematic chain (store wrist pose = robot FK, Euler 'xyz', ZED factory intrinsics, 54-point gripper box) is exact where
  the DROID exterior extrinsics are right: projected points land on the gripper, entry frames match RobotSeg within 4 frames.
- RobotSeg's automatic single-frame prompt on the birth frame segments the gripper cleanly whenever the gripper is in view
  (random QA sample: all 40 on the gripper; failures seen only for a black gripper on a black curtain). It is kept as the
  final mask for 13403 of 13432 masks; the kinematic box prompt is used only where the automatic mask is empty (29).
- **The exterior extrinsics in the DROID metadata are wrong for a large minority of scenes.** The RobotSeg mask sits where
  the gripper is while the projection sits elsewhere in 22.7 % of camera-birth-frame pairs; 1548 / 4262 scenes have at
  least one such camera, 853 have a severe case at the 50 % birth frame. Rates by lab: IRIS 42 %, CLVR 34 %, REAL 33 %,
  RAIL 27 %, TRI 26 %, ILIAD 19 %, WEIRD 18 %, AUTOLab 16 %, PennPAL 16 %, IPRL 11 %, RPL 10 %, RAD 9 %, GuptaLab 5 %.
  For those scenes the birth frame itself is computed from the wrong projection and is suspect; the mask on that frame is
  still usually a correct gripper mask, just not necessarily at the true birth frame. 21 of the 30 'never visible' scenes
  are this failure (the review videos show the gripper plainly in both cameras).
- Other limitations: occlusion is not modelled (a curtain hides the robot in two PennPAL scenes); 21 scenes have an MP4
  frame count different from the store (frame index taken as the store index); 2 REAL scenes have a 0-frame MP4.


## Stage 3 (added 2026-10-06): Track-On-R gripper tracks from the birth frame

`trackon_run/` — for every (scene, exterior camera) with a birth frame (8522 items, 4262 scenes): query points sampled on a grid
inside the eroded birth-frame RobotSeg mask (median 65 per item), tracked causally with **Track-On-R** (`track_on_r.pt`;
20x20 hidden support grid, visibility threshold 0.8) from the birth frame to the end of the episode. RobotSeg ran on the birth
frame only; nothing is re-segmented. 2.40 M frames, 36 GPU-hours on 32 A100s (1 h 20 min wall), 0 failures.

| file | content |
|---|---|
| `trackon_run/tracks/<ep>/<cam>_f<birth>.npz` | `tracks` (T,N,2) float32 px in the 1280x720 frame, NaN before the birth frame; `visibility` (T,N) bool; `queries` (N,3) [t,x,y]; `meta` json (birth_frame, n_frames_store/mp4/tracked, mp4_shorter_than_store, store_scale 0.25 for the 320x180 `pointworld_droid_ext` frames, store_fps 15, support_grid, delta_v, ckpt). Row i == MP4 frame i == store frame i. |
| `trackon_run/tracks_index.csv`, `trackon_run/summary.md` | per-item index (points, frames, visibility stats, fps) and the run summary |
| `trackon_run/qa_videos/` (96), `trackon_run/qa_tracks_sheet.jpg` | rendered tracks for every 100th item |
| `examples/` | full-pipeline showcase videos (birth frame -> mask -> tracks) and Track-On-R vs Track-On2 side-by-sides |
| `comparisons/` | Track-On-R vs Track-On2 on two scenes: identical queries, RobotSeg-mask and kinematic proxies, `metrics.md` |

Visibility (tracker's own flag): median 0.74 of points visible over the tracked span; 16 % of items end with < 25 % of points
visible, which includes the genuine cases where the gripper leaves the view or is occluded (scene-2 ext1 in `examples/` is one).
Remember the extrinsics caveat above: in the ~1500 `scenes_extrinsics_suspect*` scenes the birth frame itself may be early or
late, so the tracks there start from a frame that may not be the true birth frame (the mask on it is still usually correct).

## Suggested next step

Re-estimate each exterior camera's pose per scene from the data itself: RobotSeg gripper-mask centroids over ~16 frames
(2D) against the FK gripper centre (3D), PnP + RANSAC with the ZED intrinsics; accept when the reprojection error is small,
then recompute the birth frames with the corrected extrinsics. The 1548 suspect scenes are the obvious target and the
consistent scenes give a built-in check.
