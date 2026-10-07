# Gripper birth frames, DROID wrist test split (4292 scenes)

analysed: 4292   errors: 0  
frames per scene: median 220, min 32, max 2437

## Birth frame by visibility threshold (fraction of the 54 gripper points inside both images)

| threshold | never (both) | birth = 0 | median birth | p90 | median birth / n_frames |
|---|---|---|---|---|---|
| 25% | 2 (0.0%) | 3027 (70.5%) | 0 | 30 | 0.00 |
| 50% | 3 (0.1%) | 2724 (63.5%) | 0 | 38 | 0.00 |
| 75% | 25 (0.6%) | 2254 (52.5%) | 0 | 48 | 0.00 |
| 100% | 93 (2.2%) | 1741 (40.6%) | 17 | 61 | 0.06 |

## Which camera is the bottleneck (50% threshold)

- ext1 sees the gripper first in 709 scenes, ext2 first in 800, same frame in 2780
- never seen by ext1 at 50%: 0; never by ext2: 3; never by both: 0
- |ext1 - ext2| entry gap: median 0 frames, p90 26

## Per lab (50% threshold)

| lab | scenes | never | birth = 0 | median birth | p90 |
|---|---|---|---|---|---|
| TRI | 1230 | 1 | 1008 | 0 | 21 |
| AUTOLab | 613 | 0 | 433 | 0 | 25 |
| RAIL | 426 | 0 | 14 | 29 | 58 |
| IPRL | 357 | 0 | 305 | 0 | 20 |
| REAL | 284 | 0 | 106 | 15 | 35 |
| IRIS | 283 | 2 | 179 | 0 | 38 |
| CLVR | 275 | 0 | 66 | 25 | 51 |
| ILIAD | 198 | 0 | 86 | 16 | 74 |
| RPL | 194 | 0 | 167 | 0 | 19 |
| PennPAL | 142 | 0 | 87 | 0 | 57 |
| WEIRD | 132 | 0 | 127 | 0 | 0 |
| GuptaLab | 107 | 0 | 97 | 0 | 0 |
| RAD | 51 | 0 | 49 | 0 | 0 |

## Stability after birth (50%)

- fraction of post-birth frames where both cameras still see >= 50% of the gripper: median 1.00, 10th percentile 0.85; scenes below 0.5: 144

## Scenes never visible to both cameras at 50% (3)

- IRIS+226b9f35+2023-07-11-11h-46m-06s  ext1 max 0.83  ext2 max 0.17
- IRIS+f94b8622+2023-12-12-19h-39m-25s  ext1 max 1.00  ext2 max 0.00
- TRI+52ca9b6a+2024-01-25-13h-35m-10s  ext1 max 1.00  ext2 max 0.44
