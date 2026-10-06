# Gripper birth frames, DROID wrist test split (4292 scenes)

analysed: 4292   errors: 0  
frames per scene: median 220, min 32, max 2437

## Birth frame by visibility threshold (fraction of the 54 gripper points inside both images)

| threshold | never (both) | birth = 0 | median birth | p90 | median birth / n_frames |
|---|---|---|---|---|---|
| 25% | 15 (0.3%) | 3041 (70.9%) | 0 | 30 | 0.00 |
| 50% | 30 (0.7%) | 2704 (63.0%) | 0 | 38 | 0.00 |
| 75% | 57 (1.3%) | 2255 (52.5%) | 0 | 49 | 0.00 |
| 100% | 119 (2.8%) | 1718 (40.0%) | 17 | 61 | 0.06 |

## Which camera is the bottleneck (50% threshold)

- ext1 sees the gripper first in 745 scenes, ext2 first in 770, same frame in 2747
- never seen by ext1 at 50%: 17; never by ext2: 13; never by both: 0
- |ext1 - ext2| entry gap: median 0 frames, p90 26

## Per lab (50% threshold)

| lab | scenes | never | birth = 0 | median birth | p90 |
|---|---|---|---|---|---|
| TRI | 1230 | 12 | 1015 | 0 | 21 |
| AUTOLab | 613 | 0 | 429 | 0 | 25 |
| RAIL | 426 | 1 | 16 | 29 | 57 |
| IPRL | 357 | 6 | 302 | 0 | 20 |
| REAL | 284 | 0 | 90 | 19 | 42 |
| IRIS | 283 | 2 | 168 | 0 | 42 |
| CLVR | 275 | 5 | 64 | 23 | 45 |
| ILIAD | 198 | 1 | 89 | 13 | 73 |
| RPL | 194 | 0 | 167 | 0 | 19 |
| PennPAL | 142 | 2 | 83 | 0 | 52 |
| WEIRD | 132 | 1 | 130 | 0 | 0 |
| GuptaLab | 107 | 0 | 101 | 0 | 0 |
| RAD | 51 | 0 | 50 | 0 | 0 |

## Stability after birth (50%)

- fraction of post-birth frames where both cameras still see >= 50% of the gripper: median 1.00, 10th percentile 0.80; scenes below 0.5: 162

## Scenes never visible to both cameras at 50% (30)

- CLVR+13759f6e+2023-06-03-00h-37m-51s  ext1 max 0.44  ext2 max 1.00
- CLVR+13759f6e+2023-06-03-00h-51m-46s  ext1 max 0.39  ext2 max 1.00
- CLVR+13759f6e+2023-06-03-17h-19m-53s  ext1 max 0.19  ext2 max 1.00
- CLVR+13759f6e+2023-06-03-17h-24m-42s  ext1 max 0.30  ext2 max 1.00
- CLVR+236539bc+2023-08-06-19h-44m-31s  ext1 max 0.09  ext2 max 1.00
- ILIAD+7ae1bcff+2023-05-24-18h-47m-07s  ext1 max 1.00  ext2 max 0.20
- IPRL+7790ec0a+2023-06-30-17h-15m-06s  ext1 max 0.67  ext2 max 0.31
- IPRL+7790ec0a+2023-06-30-17h-28m-48s  ext1 max 0.85  ext2 max 0.31
- IPRL+edf28ef3+2024-01-02-20h-28m-23s  ext1 max 0.00  ext2 max 1.00
- IPRL+edf28ef3+2024-01-02-20h-59m-43s  ext1 max 0.00  ext2 max 1.00
- IPRL+edf28ef3+2024-01-02-21h-03m-31s  ext1 max 0.00  ext2 max 1.00
- IPRL+edf28ef3+2024-01-02-21h-13m-34s  ext1 max 0.26  ext2 max 1.00
- IRIS+226b9f35+2023-07-11-13h-15m-33s  ext1 max 1.00  ext2 max 0.22
- IRIS+7dfa2da3+2023-05-25-14h-00m-19s  ext1 max 1.00  ext2 max 0.00
- PennPAL+acda9df3+2023-06-18-20h-09m-33s  ext1 max 0.44  ext2 max 1.00
- PennPAL+acda9df3+2023-06-18-20h-21m-34s  ext1 max 0.37  ext2 max 1.00
- RAIL+c3d50939+2023-10-20-17h-36m-30s  ext1 max 0.48  ext2 max 0.85
- TRI+52ca9b6a+2023-10-17-13h-51m-19s  ext1 max 1.00  ext2 max 0.28
- TRI+52ca9b6a+2023-10-17-13h-59m-42s  ext1 max 1.00  ext2 max 0.17
- TRI+52ca9b6a+2023-10-19-10h-00m-12s  ext1 max 0.02  ext2 max 1.00
- TRI+52ca9b6a+2023-11-01-11h-15m-22s  ext1 max 0.67  ext2 max 0.31
- TRI+52ca9b6a+2023-11-28-17h-48m-01s  ext1 max 0.39  ext2 max 1.00
- TRI+52ca9b6a+2023-11-28-17h-51m-50s  ext1 max 0.22  ext2 max 1.00
- TRI+52ca9b6a+2023-11-28-18h-03m-23s  ext1 max 0.07  ext2 max 1.00
- TRI+52ca9b6a+2024-01-02-13h-34m-47s  ext1 max 0.04  ext2 max 1.00
- TRI+52ca9b6a+2024-01-25-13h-35m-10s  ext1 max 1.00  ext2 max 0.17
- TRI+52ca9b6a+2024-02-05-10h-45m-15s  ext1 max 1.00  ext2 max 0.28
- TRI+52ca9b6a+2024-02-06-10h-45m-47s  ext1 max 1.00  ext2 max 0.33
- TRI+52ca9b6a+2024-02-06-10h-49m-08s  ext1 max 1.00  ext2 max 0.28
- WEIRD+d5edb57c+2023-12-01-16h-02m-20s  ext1 max 0.93  ext2 max 0.37
