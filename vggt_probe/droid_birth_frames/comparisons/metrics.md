# Track-On checkpoint comparison on the two example scenes

Checkpoints: track_on_r, trackon2 (track_on_r = Track-On-R, trackon2 = Track-On2). Identical query points from the birth-frame RobotSeg mask; causal tracking from the birth frame to the end. No ground truth exists: the proxies below use RobotSeg masks on every 3rd frame (evaluation only) and, where the extrinsics are trustworthy, the kinematic gripper projection. Metric definitions and gating rules are in the script docstring (`~/track_on/leonardo/compare_checkpoints_metrics.py`).

## Frame and point gating

| scene | cam | eval frames | valid | gripper-out frames | kinematics usable | strict-kin frames |
|---|---|---|---|---|---|---|
| 2023-08-12-20h-3 | ext1 | 48 | 48 | 0 | True | 46 |
| 2023-08-12-20h-3 | ext2 | 48 | 48 | 0 | True | 45 |
| 2023-09-02-09h-5 | ext1 | 68 | 53 | 7 | True | 21 |
| 2023-09-02-09h-5 | ext2 | 68 | 68 | 0 | False | 0 |

## Per camera-video

| scene | cam | ckpt | n_points_eval | in_mask_vis_d0 | in_mask_vis_d5 | in_mask_all_d5 | lost_frac | boundary_frac | survive_end | vis_recall | false_vis_out | flicker | drift_px | step_err_px | jitter_px | jump_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023-08-12-20h-3 | ext1 | track_on_r | 52.0 | 0.961 | 0.993 | 0.989 | 0.002 | 0.004 | 0.981 | 0.877 | - | 1.049 | 5.484 | 1.009 | 1.372 | 0.000 |
| 2023-08-12-20h-3 | ext1 | trackon2 | 52.0 | 0.965 | 0.987 | 0.982 | 0.004 | 0.007 | 0.942 | 0.837 | - | 1.587 | 4.517 | 0.909 | 1.614 | 0.001 |
| 2023-08-12-20h-3 | ext2 | track_on_r | 61.0 | 0.893 | 0.968 | 0.959 | 0.008 | 0.019 | 0.984 | 0.966 | - | 0.952 | 5.349 | 1.701 | 1.295 | 0.001 |
| 2023-08-12-20h-3 | ext2 | trackon2 | 61.0 | 0.902 | 0.964 | 0.946 | 0.020 | 0.012 | 0.836 | 0.906 | - | 2.178 | 7.287 | 1.474 | 1.582 | 0.001 |
| 2023-09-02-09h-5 | ext1 | track_on_r | 52.0 | 0.944 | 0.981 | 0.975 | 0.008 | 0.007 | 0.904 | 0.777 | 0.047 | 5.084 | 22.4 | 3.867 | 2.569 | 0.007 |
| 2023-09-02-09h-5 | ext1 | trackon2 | 52.0 | 0.948 | 0.986 | 0.979 | 0.000 | 0.008 | 0.962 | 0.766 | 0.091 | 6.502 | 24.7 | 3.863 | 3.030 | 0.013 |
| 2023-09-02-09h-5 | ext2 | track_on_r | 64.0 | 0.964 | 0.985 | 0.961 | 0.009 | 0.004 | 0.948 | 0.869 | - | 2.614 | - | - | 2.117 | 0.002 |
| 2023-09-02-09h-5 | ext2 | trackon2 | 64.0 | 0.956 | 0.982 | 0.952 | 0.008 | 0.006 | 0.907 | 0.885 | - | 4.007 | - | - | 2.453 | 0.008 |

## Mean over the 4 camera-videos (kinematic columns over the 3 with usable extrinsics)

| ckpt | in_mask_vis_d0 | in_mask_vis_d5 | in_mask_all_d5 | lost_frac | boundary_frac | survive_end | vis_recall | false_vis_out | flicker | drift_px | step_err_px | jitter_px | jump_frac | fps (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| track_on_r | 0.940 | 0.982 | 0.971 | 0.007 | 0.008 | 0.954 | 0.872 | 0.047 | 2.425 | 11.1 | 2.192 | 1.838 | 0.003 | 19.6 |
| trackon2 | 0.943 | 0.980 | 0.965 | 0.008 | 0.008 | 0.912 | 0.848 | 0.091 | 3.569 | 12.2 | 2.082 | 2.170 | 0.006 | 19.6 |

Direction: higher is better for in_mask_*, survive_end, vis_recall; lower is better for lost_frac, boundary_frac, false_vis_out, flicker, drift_px, step_err_px, jitter_px, jump_frac. vis_mean/vis_last are descriptive only (the gripper leaves the view at the end of scene 2 ext1, so low visibility there is correct). Speed is identical by construction (same architecture); the median fps removes the first-item CUDA warm-up.

## Cross-checkpoint agreement

- 2023-08-12-20h-3 ext1: median distance 1.5 px where both visible, 96% within 8 px, both visible on 82% of point-frames
- 2023-08-12-20h-3 ext2: median distance 1.5 px where both visible, 95% within 8 px, both visible on 88% of point-frames
- 2023-09-02-09h-5 ext1: median distance 2.0 px where both visible, 87% within 8 px, both visible on 61% of point-frames
- 2023-09-02-09h-5 ext2: median distance 2.0 px where both visible, 91% within 8 px, both visible on 80% of point-frames

Caveat: two episodes from one lab rig (four camera views) are enough to see whether the checkpoints behave differently on this data, not to rank them in general; differences below ~0.02 in the mask metrics are within the pseudo-GT noise (the 0 px vs 5 px dilation columns bound it).
