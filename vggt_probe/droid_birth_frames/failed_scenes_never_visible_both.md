# Side note: the 30 scenes where the gripper is never visible to both exterior cameras

Criterion: no frame where >= 50 % of the 54 gripper-model points project inside BOTH exterior images
(`birth_f050 = -1` in `birth_frames.csv`). Per-camera maxima show which camera is the blocker;
`birth@25%` is the birth frame at the looser 25 % threshold (-1 = none there either).
Review videos: `failed_scenes_video_{1,2,3}.mp4` (10 scenes each; rows = scenes, columns = wrist | ext1 | ext2,
all playing simultaneously, shorter episodes hold their last frame). Scene lists: `failed_scenes.txt`, this csv.

| # | episode | frames | ext1 max | ext2 max | ext1 first@50% | ext2 first@50% | birth@25% | video |
|---|---|---|---|---|---|---|---|---|
| 1 | CLVR+13759f6e+2023-06-03-00h-37m-51s | 197 | 0.44 | 1.00 | -1 | 0 | 19 | 1 / row 1 |
| 2 | CLVR+13759f6e+2023-06-03-00h-51m-46s | 290 | 0.39 | 1.00 | -1 | 0 | 92 | 1 / row 2 |
| 3 | CLVR+13759f6e+2023-06-03-17h-19m-53s | 300 | 0.19 | 1.00 | -1 | 21 | -1 | 1 / row 3 |
| 4 | CLVR+13759f6e+2023-06-03-17h-24m-42s | 193 | 0.30 | 1.00 | -1 | 0 | 133 | 1 / row 4 |
| 5 | CLVR+236539bc+2023-08-06-19h-44m-31s | 208 | 0.09 | 1.00 | -1 | 0 | -1 | 1 / row 5 |
| 6 | ILIAD+7ae1bcff+2023-05-24-18h-47m-07s | 348 | 1.00 | 0.20 | 0 | -1 | -1 | 1 / row 6 |
| 7 | IPRL+7790ec0a+2023-06-30-17h-15m-06s | 259 | 0.67 | 0.31 | 0 | -1 | -1 | 1 / row 7 |
| 8 | IPRL+7790ec0a+2023-06-30-17h-28m-48s | 243 | 0.85 | 0.31 | 50 | -1 | 50 | 1 / row 8 |
| 9 | IPRL+edf28ef3+2024-01-02-20h-28m-23s | 295 | 0.00 | 1.00 | -1 | 0 | -1 | 1 / row 9 |
| 10 | IPRL+edf28ef3+2024-01-02-20h-59m-43s | 776 | 0.00 | 1.00 | -1 | 0 | -1 | 1 / row 10 |
| 11 | IPRL+edf28ef3+2024-01-02-21h-03m-31s | 404 | 0.00 | 1.00 | -1 | 0 | -1 | 2 / row 1 |
| 12 | IPRL+edf28ef3+2024-01-02-21h-13m-34s | 499 | 0.26 | 1.00 | -1 | 0 | 0 | 2 / row 2 |
| 13 | IRIS+226b9f35+2023-07-11-13h-15m-33s | 367 | 1.00 | 0.22 | 0 | -1 | -1 | 2 / row 3 |
| 14 | IRIS+7dfa2da3+2023-05-25-14h-00m-19s | 231 | 1.00 | 0.00 | 0 | -1 | -1 | 2 / row 4 |
| 15 | PennPAL+acda9df3+2023-06-18-20h-09m-33s | 287 | 0.44 | 1.00 | -1 | 0 | 225 | 2 / row 5 |
| 16 | PennPAL+acda9df3+2023-06-18-20h-21m-34s | 260 | 0.37 | 1.00 | -1 | 0 | 210 | 2 / row 6 |
| 17 | RAIL+c3d50939+2023-10-20-17h-36m-30s | 297 | 0.48 | 0.85 | -1 | 167 | 176 | 2 / row 7 |
| 18 | TRI+52ca9b6a+2023-10-17-13h-51m-19s | 564 | 1.00 | 0.28 | 0 | -1 | 0 | 2 / row 8 |
| 19 | TRI+52ca9b6a+2023-10-17-13h-59m-42s | 466 | 1.00 | 0.17 | 0 | -1 | -1 | 2 / row 9 |
| 20 | TRI+52ca9b6a+2023-10-19-10h-00m-12s | 196 | 0.02 | 1.00 | -1 | 0 | -1 | 2 / row 10 |
| 21 | TRI+52ca9b6a+2023-11-01-11h-15m-22s | 123 | 0.67 | 0.31 | 103 | -1 | 105 | 3 / row 1 |
| 22 | TRI+52ca9b6a+2023-11-28-17h-48m-01s | 168 | 0.39 | 1.00 | -1 | 0 | 0 | 3 / row 2 |
| 23 | TRI+52ca9b6a+2023-11-28-17h-51m-50s | 239 | 0.22 | 1.00 | -1 | 0 | -1 | 3 / row 3 |
| 24 | TRI+52ca9b6a+2023-11-28-18h-03m-23s | 216 | 0.07 | 1.00 | -1 | 0 | -1 | 3 / row 4 |
| 25 | TRI+52ca9b6a+2024-01-02-13h-34m-47s | 176 | 0.04 | 1.00 | -1 | 0 | -1 | 3 / row 5 |
| 26 | TRI+52ca9b6a+2024-01-25-13h-35m-10s | 166 | 1.00 | 0.17 | 0 | -1 | -1 | 3 / row 6 |
| 27 | TRI+52ca9b6a+2024-02-05-10h-45m-15s | 186 | 1.00 | 0.28 | 0 | -1 | 0 | 3 / row 7 |
| 28 | TRI+52ca9b6a+2024-02-06-10h-45m-47s | 355 | 1.00 | 0.33 | 0 | -1 | 0 | 3 / row 8 |
| 29 | TRI+52ca9b6a+2024-02-06-10h-49m-08s | 307 | 1.00 | 0.28 | 0 | -1 | 220 | 3 / row 9 |
| 30 | WEIRD+d5edb57c+2023-12-01-16h-02m-20s | 185 | 0.93 | 0.37 | 0 | -1 | 145 | 3 / row 10 |

Blocker summary: ext1 never >=50%: 17, ext2 never >=50%: 13

By lab: TRI: 12, IPRL: 6, CLVR: 5, IRIS: 2, PennPAL: 2, ILIAD: 1, RAIL: 1, WEIRD: 1

Inspected by eye (validation/never_*.jpg): TRI+52ca9b6a+2023-11-01 (both cameras aimed at a door, robot at the edge of view) and
TRI+52ca9b6a+2024-01-25 (ext2 looks at the couch below the arm's working height; ext1 tracks the gripper fine).

## Visual verdict from the review videos (frames at 20/50/80 % of each video)

Counts: {'borderline': 2, 'extrinsics': 21, 'genuine': 7}

`extrinsics` = the gripper IS visible in the camera the detector calls blind, so that camera's `ext*_cam_extrinsics` in the
DROID metadata does not match where the camera actually was (stale or wrong calibration). `genuine` = the camera really
never shows the gripper (aimed elsewhere, or occluded by a curtain). `borderline` = gripper hugs the image edge.

| # | episode | verdict | note |
|---|---|---|---|
| 1 | CLVR+13759f6e+2023-06-03-00h-37m-51s | borderline | gripper at the top edge of ext1 for much of the episode; projection says max 44 % |
| 2 | CLVR+13759f6e+2023-06-03-00h-51m-46s | extrinsics | gripper visible top-centre of ext1; projection 0.22 |
| 3 | CLVR+13759f6e+2023-06-03-17h-19m-53s | extrinsics | gripper fully visible in ext1 (power strip); projection 0.00 |
| 4 | CLVR+13759f6e+2023-06-03-17h-24m-42s | extrinsics | gripper fully visible in ext1 (kettle); projection 0.15 |
| 5 | CLVR+236539bc+2023-08-06-19h-44m-31s | genuine | ext1 looks at a kitchen counter, robot never in view |
| 6 | ILIAD+7ae1bcff+2023-05-24-18h-47m-07s | extrinsics | gripper visible top-right of ext2 above the bin; projection 0.00 |
| 7 | IPRL+7790ec0a+2023-06-30-17h-15m-06s | extrinsics | gripper visible in BOTH exterior views; projections 0.15 / 0.28 |
| 8 | IPRL+7790ec0a+2023-06-30-17h-28m-48s | extrinsics | gripper visible in both; projections 0.48 / 0.09 |
| 9 | IPRL+edf28ef3+2024-01-02-20h-28m-23s | extrinsics | gripper visible top-centre of ext1 (bag); projection 0.00 |
| 10 | IPRL+edf28ef3+2024-01-02-20h-59m-43s | extrinsics | gripper visible in ext1; projection 0.00 |
| 11 | IPRL+edf28ef3+2024-01-02-21h-03m-31s | extrinsics | gripper visible in ext1; projection 0.00 |
| 12 | IPRL+edf28ef3+2024-01-02-21h-13m-34s | extrinsics | gripper visible in ext1; projection 0.00 |
| 13 | IRIS+226b9f35+2023-07-11-13h-15m-33s | extrinsics | gripper visible top-right of ext2 by the microwave; projection 0.00 |
| 14 | IRIS+7dfa2da3+2023-05-25-14h-00m-19s | extrinsics | gripper visible top-right of ext2 at the drawer; projection 0.00 |
| 15 | PennPAL+acda9df3+2023-06-18-20h-09m-33s | genuine | a curtain hides the robot from ext1 (occlusion, which the model does not represent) |
| 16 | PennPAL+acda9df3+2023-06-18-20h-21m-34s | genuine | curtain hides the robot from ext1 |
| 17 | RAIL+c3d50939+2023-10-20-17h-36m-30s | genuine | neither exterior camera shows the robot at the sampled frames |
| 18 | TRI+52ca9b6a+2023-10-17-13h-51m-19s | genuine | ext2 aimed at a shelf with cables in front; gripper not in view |
| 19 | TRI+52ca9b6a+2023-10-17-13h-59m-42s | genuine | same rig as above |
| 20 | TRI+52ca9b6a+2023-10-19-10h-00m-12s | extrinsics | arm + gripper centred in ext1; projection 0.00 |
| 21 | TRI+52ca9b6a+2023-11-01-11h-15m-22s | genuine | both cameras face a door, robot at the edge of view |
| 22 | TRI+52ca9b6a+2023-11-28-17h-48m-01s | extrinsics | gripper visible top-centre of ext1; projection 0.22 |
| 23 | TRI+52ca9b6a+2023-11-28-17h-51m-50s | extrinsics | gripper visible right of centre in ext1; projection 0.00 |
| 24 | TRI+52ca9b6a+2023-11-28-18h-03m-23s | extrinsics | gripper visible in both views; projections 0.00 |
| 25 | TRI+52ca9b6a+2024-01-02-13h-34m-47s | extrinsics | gripper with TV remote centred in ext1; projection 0.00 |
| 26 | TRI+52ca9b6a+2024-01-25-13h-35m-10s | borderline | gripper appears at the top-left corner of ext2 at times; projection max 0.17 |
| 27 | TRI+52ca9b6a+2024-02-05-10h-45m-15s | extrinsics | gripper visible top-right of ext2; projection 0.00 |
| 28 | TRI+52ca9b6a+2024-02-06-10h-45m-47s | extrinsics | gripper visible in both views; projections 0.00 |
| 29 | TRI+52ca9b6a+2024-02-06-10h-49m-08s | extrinsics | gripper visible in both views; projections 0.00 |
| 30 | WEIRD+d5edb57c+2023-12-01-16h-02m-20s | extrinsics | gripper visible top-right of ext2; projection 0.00 |

Implication: the 'never visible' set is dominated by bad exterior extrinsics, not by camera placement. The same error
source can shift birth frames in scenes that did get one; the full-split RobotSeg run records a mask-vs-projection
consistency flag per mask (`flagged_final`) which is the per-scene probe for this.
