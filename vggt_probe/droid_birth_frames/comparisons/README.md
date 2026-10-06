# Track-On-R vs Track-On2 on two example scenes

Which checkpoint is which (same architecture, 509 tensors each):

| file | model | training |
|---|---|---|
| `$WORK/bora/checkpoints/track_on/track_on_r.pt` | **Track-On-R** (CVPR 26) | Kubric + real-world verifier-guided pseudo-label fine-tuning |
| `$WORK/bora/checkpoints/track_on/trackon2_dinov3.pt` | **Track-On2** (TPAMI 26) | Kubric only |

The production gripper-track run (`../trackon_run/`) used `track_on_r.pt`, i.e. Track-On-R.

Scenes: `AUTOLab+5d05c5aa+2023-08-12-20h-31m-55s` (birth frame 21, 164 frames) and
`AUTOLab+5d05c5aa+2023-09-02-09h-55m-32s` (birth frame 34, 236 frames; the gripper leaves ext1's view near the end;
ext2's extrinsics failed the kinematic consistency check, so kinematic metrics are not used there).

## Videos (`../examples/`)
- `<scene>__pipeline_track_on_r.mp4`, `<scene>__pipeline_trackon2.mp4` — full pipeline per checkpoint (birth frame, RobotSeg mask
  on that frame only, tracks to the end), both exterior cameras side by side.
- `<scene>__compare_R_vs_2.mp4` — rows = checkpoints, columns = cameras, same query points, synchronous playback.

## Quantitative comparison
`metrics.md` (table), `metrics.json`, `in_mask_over_time.png`. Tracks from both checkpoints on identical query points are in
`track_on_r/tracks/` and `trackon2/tracks/`; RobotSeg evaluation masks (every 3rd frame, evaluation only) in `eval_masks/`.

Summary (means over the 4 camera-videos, gated to frames where the gripper is in view and to query points on the gripper body):

| | Track-On-R | Track-On2 |
|---|---|---|
| visible points inside the RobotSeg mask (+5 px) | 0.982 | 0.980 |
| all points inside the mask | 0.971 | 0.965 |
| points lost (> 60 px from the mask) | 0.7 % | 0.8 % |
| points still on the gripper at the last valid frame | 0.954 | 0.912 |
| visibility recall on in-mask points | 0.872 | 0.848 |
| points flagged visible while the gripper is out of view | 0.047 | 0.091 |
| visibility flicker (toggles / point / 100 frames) | 2.4 | 3.6 |
| kinematic drift of the point cloud centroid (px) | 11.1 | 12.2 |
| jitter (px / frame^2) | 1.84 | 2.17 |
| speed (median fps, 720p, 20x20 support grid, A100) | 19.6 | 19.6 |

Where both trackers flag a point visible they agree to within 1.5-2 px (median). The differences are in what happens around
occlusion and exits: Track-On-R keeps a few percent more points on the gripper to the end, flags fewer false "visible" points
when the gripper leaves the view, flickers less and is slightly smoother. Track-On2 is marginally better on raw in-mask
precision without dilation (0.943 vs 0.940) and on survival in scene 2 ext1. All mask-metric gaps are at or below the
pseudo-GT noise level (bounded by the 0 px vs 5 px dilation columns); the visibility-behaviour gaps are the consistent ones.
Two episodes from one rig cannot rank the checkpoints in general.

Scripts: `~/track_on/leonardo/compare_checkpoints_track.py` (re-track with several checkpoints on identical queries),
`~/RobotSeg/test/eval_masks_for_tracks.py` (evaluation masks), `~/track_on/leonardo/compare_checkpoints_metrics.py`,
`make_pipeline_video.py`, `make_comparison_video.py`.
