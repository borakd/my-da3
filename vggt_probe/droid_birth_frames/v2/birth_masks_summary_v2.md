# RobotSeg gripper masks on the birth frames, DROID wrist test split

rows: 8386  status: {'ok': 8381, 'no_birth_frame': 4, 'decode_failed': 1}
scenes with masks: 2233; masks: 8381 (b050: 4465, b100: 3916)
MP4 frame count != store frame count: 12 masks in 7 scenes (frame index still taken as the store index)

## Mask policy and the kinematic consistency flag

Final mask = RobotSeg automatic prompt on the birth frame; the kinematic box prompt is used only when the
automatic mask is empty (< 0.05 % of the image). `projection_disagree` = the automatic mask does not sit
where the kinematic projection puts the gripper (> 20 % of mask pixels outside the projected box + 60 px).
QA showed that in these cases the MASK is usually right and the scene's exterior extrinsics are off, so the
flag is best read as 'extrinsics (and therefore birth frame) suspect for this camera'.

| set | masks | automatic mask kept | box used (auto empty) | projection_disagree |
|---|---|---|---|---|
| b050 | 4465 | 4455 | 10 | 231 (5.2%) |
| b100 | 3916 | 3915 | 1 | 73 (1.9%) |
| all | 8381 | 8370 | 11 | 304 (3.6%) |

scenes with projection_disagree in at least one camera: 242 of 2233 (both cameras at b050: 14); listed in `scenes_extrinsics_suspect.txt`

## Per lab (b050, projection_disagree rate = extrinsics-suspect rate)

| lab | masks | disagree | rate |
|---|---|---|---|
| TRI | 1072 | 40 | 3.7% |
| AUTOLab | 624 | 14 | 2.2% |
| RAIL | 613 | 98 | 16.0% |
| REAL | 474 | 5 | 1.1% |
| CLVR | 446 | 34 | 7.6% |
| IRIS | 344 | 19 | 5.5% |
| ILIAD | 270 | 8 | 3.0% |
| IPRL | 224 | 4 | 1.8% |
| PennPAL | 162 | 7 | 4.3% |
| RPL | 122 | 0 | 0.0% |
| GuptaLab | 70 | 0 | 0.0% |
| WEIRD | 34 | 2 | 5.9% |
| RAD | 10 | 0 | 0.0% |


QA sheets: `qa_random_consistent_b050.jpg` (40 random masks consistent with the projection), `qa_random_projection_disagree.jpg` (40 random masks that disagree with it), `qa_auto_empty_box_used.jpg` (16: automatic mask empty, box mask used).

Files: `masks_run/masks/<ep>/<cam>_f<t>.png` (binary mask), `masks_run/frames/...jpg` (the decoded birth frame),
`masks_run/overlays/...jpg`; `birth_masks.csv` has one row per mask with the frame index, prompt used and scores.
