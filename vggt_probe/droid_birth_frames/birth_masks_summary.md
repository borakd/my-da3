# RobotSeg gripper masks on the birth frames, DROID wrist test split

rows: 13494  status: {'ok': 13432, 'no_birth_frame': 60, 'decode_failed': 2}
scenes with masks: 4262; masks: 13432 (b050: 8522, b100: 4910)
MP4 frame count != store frame count: 27 masks in 21 scenes (frame index still taken as the store index)

## Mask policy and the kinematic consistency flag

Final mask = RobotSeg automatic prompt on the birth frame; the kinematic box prompt is used only when the
automatic mask is empty (< 0.05 % of the image). `projection_disagree` = the automatic mask does not sit
where the kinematic projection puts the gripper (> 20 % of mask pixels outside the projected box + 60 px).
QA showed that in these cases the MASK is usually right and the scene's exterior extrinsics are off, so the
flag is best read as 'extrinsics (and therefore birth frame) suspect for this camera'.

| set | masks | automatic mask kept | box used (auto empty) | projection_disagree |
|---|---|---|---|---|
| b050 | 8522 | 8500 | 22 | 1969 (23.1%) |
| b100 | 4910 | 4903 | 7 | 1074 (21.9%) |
| all | 13432 | 13403 | 29 | 3043 (22.7%) |

scenes with projection_disagree in at least one camera: 1548 of 4262 (both cameras at b050: 468); listed in `scenes_extrinsics_suspect.txt`

## Per lab (b050, projection_disagree rate = extrinsics-suspect rate)

| lab | masks | disagree | rate |
|---|---|---|---|
| TRI | 2436 | 630 | 25.9% |
| AUTOLab | 1226 | 195 | 15.9% |
| RAIL | 850 | 234 | 27.5% |
| IPRL | 702 | 78 | 11.1% |
| REAL | 566 | 189 | 33.4% |
| IRIS | 562 | 237 | 42.2% |
| CLVR | 540 | 182 | 33.7% |
| ILIAD | 394 | 75 | 19.0% |
| RPL | 388 | 38 | 9.8% |
| PennPAL | 280 | 45 | 16.1% |
| WEIRD | 262 | 46 | 17.6% |
| GuptaLab | 214 | 11 | 5.1% |
| RAD | 102 | 9 | 8.8% |


QA sheets: `qa_random_consistent_b050.jpg` (40 random masks consistent with the projection), `qa_random_projection_disagree.jpg` (40 random masks that disagree with it), `qa_auto_empty_box_used.jpg` (40: automatic mask empty, box mask used).

Files: `masks_run/masks/<ep>/<cam>_f<t>.png` (binary mask), `masks_run/frames/...jpg` (the decoded birth frame),
`masks_run/overlays/...jpg`; `birth_masks.csv` has one row per mask with the frame index, prompt used and scores.
