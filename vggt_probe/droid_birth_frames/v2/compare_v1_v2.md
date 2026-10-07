# Birth frames: v1 (raw DROID extrinsics) vs v2 (PointWorld optimized_extrinsics)

scenes compared: 4292 (v1 4292, v2 4292)
v2 extrinsics used per camera: {'pointworld': 8582, 'raw_fallback': 2}

## b050

- same 2960, moved 1301, recovered (v1 -1 -> v2 frame) 29, lost (v1 frame -> v2 -1) 2
- moved: v2 earlier in 634, later in 667; |delta| frames median 4, p90 33, max 321; |delta| <= 2: 555, 3-10: 283, 11-30: 315, > 30: 148
- moved, with v2 frame 0 (gripper visible from the start under PointWorld): 177

## b100

- same 2247, moved 1939, recovered (v1 -1 -> v2 frame) 66, lost (v1 frame -> v2 -1) 40
- moved: v2 earlier in 945, later in 994; |delta| frames median 4, p90 38, max 466; |delta| <= 2: 849, 3-10: 439, 11-30: 389, > 30: 262
- moved, with v2 frame 0 (gripper visible from the start under PointWorld): 173

## Per lab (b050 changed = moved or recovered or lost)

| lab | scenes | changed | moved | recovered | lost | median \|delta\| (moved) |
|---|---|---|---|---|---|---|
| TRI | 1230 | 222 | 211 | 11 | 0 | 13.0 |
| AUTOLab | 613 | 149 | 149 | 0 | 0 | 2.0 |
| RAIL | 426 | 246 | 245 | 1 | 0 | 2.0 |
| IPRL | 357 | 48 | 42 | 6 | 0 | 2.0 |
| REAL | 284 | 184 | 184 | 0 | 0 | 5.0 |
| IRIS | 283 | 125 | 121 | 2 | 2 | 14.0 |
| CLVR | 275 | 175 | 170 | 5 | 0 | 4.0 |
| ILIAD | 198 | 94 | 93 | 1 | 0 | 3.0 |
| RPL | 194 | 23 | 23 | 0 | 0 | 15.0 |
| PennPAL | 142 | 49 | 47 | 2 | 0 | 2.0 |
| WEIRD | 132 | 7 | 6 | 1 | 0 | 51.0 |
| GuptaLab | 107 | 9 | 9 | 0 | 0 | 7.0 |
| RAD | 51 | 1 | 1 | 0 | 0 | 15.0 |

## The 30 v1 scenes with no birth frame

| scene | v1 verdict | v2 b050 | v2 b100 | ext1 max frac v2 | ext2 max frac v2 |
|---|---|---|---|---|---|
| CLVR+13759f6e+2023-06-03-00h-37m-51s | borderline | 0 | 0 | 1.00 | 1.00 |
| CLVR+13759f6e+2023-06-03-00h-51m-46s | extrinsics | 35 | 66 | 1.00 | 1.00 |
| CLVR+13759f6e+2023-06-03-17h-19m-53s | extrinsics | 66 | 126 | 1.00 | 1.00 |
| CLVR+13759f6e+2023-06-03-17h-24m-42s | extrinsics | 60 | -1 | 0.94 | 1.00 |
| CLVR+236539bc+2023-08-06-19h-44m-31s | genuine | 0 | 68 | 1.00 | 1.00 |
| ILIAD+7ae1bcff+2023-05-24-18h-47m-07s | extrinsics | 106 | 182 | 1.00 | 1.00 |
| IPRL+7790ec0a+2023-06-30-17h-15m-06s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| IPRL+7790ec0a+2023-06-30-17h-28m-48s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| IPRL+edf28ef3+2024-01-02-20h-28m-23s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| IPRL+edf28ef3+2024-01-02-20h-59m-43s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| IPRL+edf28ef3+2024-01-02-21h-03m-31s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| IPRL+edf28ef3+2024-01-02-21h-13m-34s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| IRIS+226b9f35+2023-07-11-13h-15m-33s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| IRIS+7dfa2da3+2023-05-25-14h-00m-19s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| PennPAL+acda9df3+2023-06-18-20h-09m-33s | genuine | 224 | -1 | 0.80 | 1.00 |
| PennPAL+acda9df3+2023-06-18-20h-21m-34s | genuine | 234 | -1 | 0.50 | 1.00 |
| RAIL+c3d50939+2023-10-20-17h-36m-30s | genuine | 179 | -1 | 0.65 | 1.00 |
| TRI+52ca9b6a+2023-10-17-13h-51m-19s | genuine | 0 | 0 | 1.00 | 1.00 |
| TRI+52ca9b6a+2023-10-17-13h-59m-42s | genuine | 0 | 378 | 1.00 | 1.00 |
| TRI+52ca9b6a+2023-10-19-10h-00m-12s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| TRI+52ca9b6a+2023-11-01-11h-15m-22s | genuine | 103 | -1 | 0.67 | 1.00 |
| TRI+52ca9b6a+2023-11-28-17h-48m-01s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| TRI+52ca9b6a+2023-11-28-17h-51m-50s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| TRI+52ca9b6a+2023-11-28-18h-03m-23s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| TRI+52ca9b6a+2024-01-02-13h-34m-47s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| TRI+52ca9b6a+2024-01-25-13h-35m-10s | borderline | -1 | -1 | 1.00 | 0.44 |
| TRI+52ca9b6a+2024-02-05-10h-45m-15s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| TRI+52ca9b6a+2024-02-06-10h-45m-47s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| TRI+52ca9b6a+2024-02-06-10h-49m-08s | extrinsics | 0 | 0 | 1.00 | 1.00 |
| WEIRD+d5edb57c+2023-12-01-16h-02m-20s | extrinsics | 0 | 0 | 1.00 | 1.00 |

scenes to re-segment (b050 or b100 changed): 2235 -> `scenes_to_resegment.txt`; (scene, camera) items to re-track (b050 moved or recovered): 2660 -> `items_to_retrack.txt`; scenes with a v1 birth frame but none in v2: 2 -> `scenes_lost_in_v2.txt`
