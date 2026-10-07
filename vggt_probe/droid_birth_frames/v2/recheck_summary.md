# Kinematic projection vs RobotSeg mask: raw DROID extrinsics vs PointWorld optimized_extrinsics

masks: 13432; evaluated with both calibrations: 13430; errors: 2 ({"'28451778'": 1, "'28813166'": 1})
v1 flag reproduced with the raw extrinsics: 13430/13430

rule: projection_disagree = > 20 % of the automatic mask's pixels outside the projected 54-point box padded by 60 px

| set | masks | disagree raw | disagree pointworld | raw only | pointworld only | both |
|---|---|---|---|---|---|---|
| b050 | 8520 | 1967 (23.1%) | 392 (4.6%) | 1640 | 65 | 327 |
| b100 | 4910 | 1074 (21.9%) | 115 (2.3%) | 988 | 29 | 86 |
| all | 13430 | 3041 (22.6%) | 507 (3.8%) | 2628 | 94 | 413 |

## Per lab (b050)

| lab | masks | disagree raw | disagree pointworld |
|---|---|---|---|
| TRI | 2434 | 628 (25.8%) | 79 (3.2%) |
| AUTOLab | 1226 | 195 (15.9%) | 29 (2.4%) |
| RAIL | 850 | 234 (27.5%) | 148 (17.4%) |
| IPRL | 702 | 78 (11.1%) | 9 (1.3%) |
| REAL | 566 | 189 (33.4%) | 12 (2.1%) |
| IRIS | 562 | 237 (42.2%) | 17 (3.0%) |
| CLVR | 540 | 182 (33.7%) | 64 (11.9%) |
| ILIAD | 394 | 75 (19.0%) | 18 (4.6%) |
| RPL | 388 | 38 (9.8%) | 3 (0.8%) |
| PennPAL | 280 | 45 (16.1%) | 6 (2.1%) |
| WEIRD | 262 | 46 (17.6%) | 3 (1.1%) |
| GuptaLab | 214 | 11 (5.1%) | 3 (1.4%) |
| RAD | 102 | 9 (8.8%) | 1 (1.0%) |

## Projected gripper-centroid shift raw -> pointworld at the b050 frame (px)

- all: n=8408 median 15.0, p10 4.2, p90 214.2, > 30 px: 37.0%
- flagged raw: n=1883 median 194.3, p10 65.5, p90 351.0, > 30 px: 91.7%
- unflagged raw: n=6525 median 11.1, p10 3.8, p90 72.9, > 30 px: 21.3%

- mask centroid to projected centroid, raw: median 45.4 px, p90 253.1 px, > 100 px: 28.1%
- mask centroid to projected centroid, pointworld: median 29.6 px, p90 69.3 px, > 100 px: 5.8%

scenes with >= 1 flagged camera: raw 1547, pointworld 415 (of 4261); pointworld list in `scenes_still_disagree_pointworld.txt`

contact sheets: `sheet_still_disagree_pw.jpg` (40 random b050 cameras still flagged under pointworld; red box = raw, green = pointworld, orange = automatic mask), `sheet_fixed_by_pw.jpg` (40 flagged raw, clean under pointworld)
