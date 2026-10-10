# Erroneous / anomalous scenes identified in this session (exterior-camera work), 2026-10-02

Compiled before any action is taken; NO data file has been modified. Sources: a full scan of all 42,925 episodes (38,633 train, 4,292 test) for exterior MP4s, metadata, factory intrinsics, PointWorld calibration sidecars and store entries; the 40-scene calibration audit of 2026-09-08 (`$OUT/vggt_probe/gt_audit_2026-09-08`, $OUT = /gpfs/scratch/etur59/koc821022/outputs/cut3r_eval); and the per-scene records of the seeding studies on the 13 test smoke scenes. Full enumeration with every reason per episode: `ERRONEOUS_SCENES.csv` (same directory).

Checks that came back clean on every one of the 42,925 episodes: metadata JSON present with all three serials; both exterior MP4 files present; PointWorld camera sidecar present with `optimization_success` true and two exterior extrinsics; store `dense/cam` present. Not checked: MP4 frame count against the store for every episode (only the four size outliers were probed).

## TEST split: 59 episodes listed

### Data-level (corpus scan)

Reason counts: {'calibration': 12, 'intrinsics': 35, 'exterior MP4 for serial 20103212 is small (189 KB) but complete': 1}

Episodes absent from the DROID authors' intrinsics.json, grouped by rig session (lab+user id); every one is recoverable because both exterior serials appear under other episodes:

- IPRL+edf28ef3: 6 episodes: IPRL+edf28ef3+2024-01-02-10h-22m-08s, IPRL+edf28ef3+2024-01-02-10h-39m-13s, IPRL+edf28ef3+2024-01-02-10h-50m-10s, IPRL+edf28ef3+2024-01-02-13h-02m-06s, IPRL+edf28ef3+2024-01-02-15h-21m-01s, IPRL+edf28ef3+2024-01-02-22h-27m-04s
- TRI+52ca9b6a: 5 episodes: TRI+52ca9b6a+2023-11-27-14h-19m-34s, TRI+52ca9b6a+2024-01-29-14h-57m-13s, TRI+52ca9b6a+2024-01-31-16h-34m-18s, TRI+52ca9b6a+2024-02-06-10h-35m-40s, TRI+52ca9b6a+2024-02-06-10h-46m-18s
- AUTOLab+44bb9c36: 4 episodes: AUTOLab+44bb9c36+2023-11-24-18h-22m-36s, AUTOLab+44bb9c36+2023-11-25-11h-39m-41s, AUTOLab+44bb9c36+2023-11-25-12h-14m-46s, AUTOLab+44bb9c36+2023-11-30-11h-25m-20s
- RAIL+t3d58310: 3 episodes: RAIL+t3d58310+2023-08-12-15h-02m-31s, RAIL+t3d58310+2023-08-12-16h-51m-15s, RAIL+t3d58310+2023-08-12-17h-11m-35s
- AUTOLab+5d05c5aa: 2 episodes: AUTOLab+5d05c5aa+2023-07-26-16h-57m-14s, AUTOLab+5d05c5aa+2023-11-12-16h-54m-03s
- AUTOLab+t3d58310: 2 episodes: AUTOLab+t3d58310+2023-08-12-13h-03m-32s, AUTOLab+t3d58310+2023-08-19-17h-41m-40s
- TRI+7dfa2da3: 2 episodes: TRI+7dfa2da3+2023-10-12-13h-20m-22s, TRI+7dfa2da3+2023-10-12-13h-22m-11s
- WEIRD+bcd0b679: 2 episodes: WEIRD+bcd0b679+2023-11-21-21h-50m-54s, WEIRD+bcd0b679+2023-11-29-15h-14m-00s
- AUTOLab+84bd5053: 1 episodes: AUTOLab+84bd5053+2023-11-11-08h-52m-38s
- AUTOLab+9aed6d7a: 1 episodes: AUTOLab+9aed6d7a+2023-11-30-14h-48m-00s
- ILIAD+7ae1bcff: 1 episodes: ILIAD+7ae1bcff+2023-05-12-16h-44m-32s
- IPRL+5085c3ce: 1 episodes: IPRL+5085c3ce+2023-10-14-22h-05m-10s
- IRIS+89b42cd2: 1 episodes: IRIS+89b42cd2+2023-12-02-14h-24m-28s
- RAD+62828209: 1 episodes: RAD+62828209+2023-11-26-13h-45m-02s
- TRI+4b1a56cc: 1 episodes: TRI+4b1a56cc+2023-09-05-16h-47m-08s
- TRI+a8754edb: 1 episodes: TRI+a8754edb+2023-10-12-18h-13m-33s
- WEIRD+d5edb57c: 1 episodes: WEIRD+d5edb57c+2023-11-22-22h-21m-10s

Other data-level entries:

- AUTOLab+0d4edc83+2023-10-27-19h-55m-09s: calibration: raw DROID metadata exterior extrinsics wrong (relative rotation 14.7 deg off VGGT-Omega; PointWorld optimized extrinsics agree with VGGT to 2.7 deg, so PointWorld-based pipelines are unaffected)
- CLVR+13759f6e+2023-06-13-01h-43m-49s: exterior MP4 for serial 20103212 is small (189 KB) but complete: 32 frames, store has 32 frames; NOT a fault, listed for the record
- GuptaLab+553d1bd5+2023-06-18-21h-07m-50s: calibration: raw DROID metadata exterior extrinsics wrong (relative rotation 11.9 deg off VGGT-Omega; PointWorld optimized extrinsics agree with VGGT to 3.1 deg, so PointWorld-based pipelines are unaffected)
- ILIAD+50aee79f+2023-08-19-11h-23m-10s: calibration: raw DROID metadata exterior extrinsics wrong (relative rotation 14.8 deg off VGGT-Omega; PointWorld optimized extrinsics agree with VGGT to 1.8 deg, so PointWorld-based pipelines are unaffected)
- ILIAD+5e938e3b+2023-07-17-11h-31m-36s: calibration: raw DROID metadata exterior extrinsics wrong (relative rotation 10.5 deg off VGGT-Omega; PointWorld optimized extrinsics agree with VGGT to 4.8 deg, so PointWorld-based pipelines are unaffected)
- ILIAD+5e938e3b+2023-07-21-22h-16m-04s: calibration: raw DROID metadata exterior extrinsics wrong (relative rotation 17.2 deg off VGGT-Omega; PointWorld optimized extrinsics agree with VGGT to 4.5 deg, so PointWorld-based pipelines are unaffected)
- TRI+52ca9b6a+2023-10-24-17h-40m-31s: calibration: raw DROID metadata exterior extrinsics wrong (relative rotation 67.1 deg off VGGT-Omega; PointWorld optimized extrinsics agree with VGGT to 3.6 deg, so PointWorld-based pipelines are unaffected)
- TRI+52ca9b6a+2023-11-07-13h-44m-24s: calibration: raw DROID metadata exterior extrinsics wrong (relative rotation 23.7 deg off VGGT-Omega; PointWorld optimized extrinsics agree with VGGT to 2.6 deg, so PointWorld-based pipelines are unaffected)
- TRI+52ca9b6a+2023-12-11-11h-52m-46s: calibration: raw DROID metadata exterior extrinsics wrong (relative rotation 12.1 deg off VGGT-Omega; PointWorld optimized extrinsics agree with VGGT to 1.9 deg, so PointWorld-based pipelines are unaffected)
- TRI+52ca9b6a+2024-01-25-16h-02m-47s: calibration: raw DROID metadata exterior extrinsics wrong (relative rotation 32.0 deg off VGGT-Omega; PointWorld optimized extrinsics agree with VGGT to 3.7 deg, so PointWorld-based pipelines are unaffected)
- TRI+52ca9b6a+2024-02-06-12h-16m-44s: calibration: raw DROID metadata exterior extrinsics wrong (relative rotation 34.3 deg off VGGT-Omega; PointWorld optimized extrinsics agree with VGGT to 4.2 deg, so PointWorld-based pipelines are unaffected)
- TRI+52ca9b6a+2024-02-07-15h-36m-24s: calibration: raw DROID metadata exterior extrinsics wrong (relative rotation 23.9 deg off VGGT-Omega; PointWorld optimized extrinsics agree with VGGT to 1.3 deg, so PointWorld-based pipelines are unaffected)
- WEIRD+f61abce9+2023-12-09-17h-45m-41s: calibration: raw DROID metadata exterior extrinsics wrong (relative rotation 39.8 deg off VGGT-Omega; PointWorld optimized extrinsics agree with VGGT to 1.3 deg, so PointWorld-based pipelines are unaffected)

### Scene-level observations from this session (13 smoke scenes; method-level unless marked calibration)

These are NOT data faults: they describe where the automatic gripper identification or the tracker failed on a scene, and are listed because a pattern across them is informative.

- AUTOLab+0d4edc83+2023-10-21-19h-48m-55s:
    - identification (method-level): motion_sam3 masks 81 px from the lens
- AUTOLab+0d4edc83+2023-10-27-19h-55m-09s:
    - coverage (method-level): silhouette model never initialised; motion_sam3 masks 93 px from the lens
- AUTOLab+0d4edc83+2023-10-27-20h-28m-46s:
    - coverage (method-level): 71-frame episode; three seeders produced zero rig anchors (no segment with >= 4 causally confirmed cross-view pairs), silhouette model never initialised
- AUTOLab+0d4edc83+2023-12-02-14h-17m-47s:
    - coverage (method-level): silhouette model never initialised
- AUTOLab+0d4edc83+2023-12-02-15h-41m-32s:
    - coverage (method-level): motion_sam3 zero anchors (2 cross-view pairs); silhouette model never initialised
- AUTOLab+44bb9c36+2023-11-23-10h-45m-46s:
    - identification (method-level): text_arm_sam3 masks 137 px from the lens (wrong object); egomotion_verify max jump 10.07 cm
- AUTOLab+44bb9c36+2023-11-23-18h-32m-23s:
    - identification (method-level): motion_only 159 px and text_arm_sam3 152 px from the lens (manipulated cloth / wrong object); rounds 2-3 wrong-body segments on a chair and hanging cloth (47% of that scene's anchors in round 2)
- AUTOLab+44bb9c36+2023-11-25-11h-05m-25s:
    - identification (method-level): motion_only 141 px and text_arm_sam3 188 px from the lens
- AUTOLab+44bb9c36+2023-11-25-13h-39m-05s:
    - identification (method-level): motion_only 89 px and text_arm_sam3 154 px from the lens; round-2 segments on the elbow
- AUTOLab+44bb9c36+2023-11-28-09h-03m-58s:
    - identification (method-level): motion_only 2 anchors at 129 px, motion_sam3 220 px from the lens; round-2/3 wrong-body segments on the elbow and shoulder; the seeder's own verified point on the elbow at 0.58 m
    - tracker prepass warning: mask-centroid triangulation behind a camera (two mask sources); possible calibration inconsistency on this rig, where one exterior camera sits against the robot base
- AUTOLab+44bb9c36+2023-11-29-23h-38m-45s:
    - identification (method-level): text_arm_sam3 zero cross-view pairs, mask 309 px from the lens; motion_sam3 1 anchor
    - tracker prepass warning: mask-centroid triangulation behind a camera (one mask source)
- RAIL+80edfcb1+2023-07-14-14h-28m-45s:
    - calibration: constant 3.4 deg rotation offset between the rig's body rotation and the GT wrist orientation on the 84-frame window (PointWorld world frame vs store base frame, or residual exterior calibration error); invisible to ATE/RPE

## TRAIN split: 292 episodes listed

### Data-level (corpus scan)

Reason counts: {'intrinsics': 289, 'exterior MP4 for serial 24013089 is truncated': 3}

Episodes absent from the DROID authors' intrinsics.json, grouped by rig session (lab+user id); every one is recoverable because both exterior serials appear under other episodes:

- TRI+52ca9b6a: 32 episodes
- IPRL+edf28ef3: 24 episodes
- WEIRD+e604eeed: 24 episodes
- TRI+7dfa2da3: 21 episodes
- AUTOLab+5d05c5aa: 20 episodes
- WEIRD+5a211037: 20 episodes
- AUTOLab+9aed6d7a: 18 episodes
- AUTOLab+44bb9c36: 14 episodes
- AUTOLab+84bd5053: 14 episodes
- TRI+4b1a56cc: 13 episodes
- TRI+a8754edb: 13 episodes
- RAIL+t3d58310: 11 episodes
- IPRL+5085c3ce: 10 episodes
- IRIS+89b42cd2: 9 episodes
- WEIRD+f61abce9: 8 episodes
- AUTOLab+t3d58310: 7 episodes
- WEIRD+9338a0f6: 6 episodes
- IPRL+89b42cd2: 4 episodes
- AUTOLab+cf2a60c6: 3 episodes
- RAD+62828209: 3 episodes
- WEIRD+417ecf07: 3 episodes
- REAL+abf65a9e: 2 episodes
- WEIRD+bcd0b679: 2 episodes
- ILIAD+49fe161f: 1 episodes
- ILIAD+7ae1bcff: 1 episodes
- IPRL+c850f181: 1 episodes
- IPRL+w026bb9b: 1 episodes
- RAD+c6cf6b42: 1 episodes
- REAL+4dbb5646: 1 episodes
- WEIRD+5047dd9a: 1 episodes
- WEIRD+bccfde6e: 1 episodes

Other data-level entries:

- RPL+32cba90c+2023-12-05-15h-19m-29s: exterior MP4 for serial 24013089 is truncated: 1057 bytes, 1 frame, store has {'RPL+32cba90c+2023-12-05-15h-19m-29s':886,'RPL+cb4f6842+2023-06-05-18h-55m-39s':197,'RPL+cb4f6842+2023-11-30-20h-42m-38s':243}[e] frames; one exterior stream unusable, rig impossible
- RPL+cb4f6842+2023-06-05-18h-55m-39s: exterior MP4 for serial 24013089 is truncated: 1057 bytes, 1 frame, store has {'RPL+32cba90c+2023-12-05-15h-19m-29s':886,'RPL+cb4f6842+2023-06-05-18h-55m-39s':197,'RPL+cb4f6842+2023-11-30-20h-42m-38s':243}[e] frames; one exterior stream unusable, rig impossible
- RPL+cb4f6842+2023-11-30-20h-42m-38s: exterior MP4 for serial 24013089 is truncated: 1057 bytes, 1 frame, store has {'RPL+32cba90c+2023-12-05-15h-19m-29s':886,'RPL+cb4f6842+2023-06-05-18h-55m-39s':197,'RPL+cb4f6842+2023-11-30-20h-42m-38s':243}[e] frames; one exterior stream unusable, rig impossible

## Pattern analysis

1. **Missing intrinsics cluster by rig session, not by episode.** 324 episodes (35 test, 289 train) are absent from the DROID authors' intrinsics file, and they come from 32 rig sessions; four sessions are missing almost entirely (TRI+4b1a56cc 14/14, TRI+7dfa2da3 23/28, TRI+a8754edb 14/59, IRIS+89b42cd2 10/71). This looks like whole recording sessions the post-hoc calibration release skipped. Impact on the rig pipeline: none once intrinsics are looked up by camera serial, because factory intrinsics are a property of the camera and every affected serial is present elsewhere. No file needs to change; the loader does.
2. **Truncated exterior videos are a single-camera, single-lab issue.** Three RPL train episodes carry a 1057-byte, one-frame MP4 for serial 24013089 while their other exterior camera and the store are complete. The rig cannot run there (one view). The CLVR test entry is a false alarm: a genuinely 32-frame episode. Only 3 of 42,925 episodes, 0.007%.
3. **Raw-metadata extrinsics are wrong on about 30% of episodes, lab-dependent.** 12 of the 40 audited test scenes, concentrated on TRI+52ca9b6a (6), ILIAD (3), plus GuptaLab, WEIRD and one AUTOLab+0d4edc83 scene; errors of 10 to 67 degrees in the relative exterior rotation. PointWorld's optimized extrinsics agree with an independent registration to within 5 degrees on all 12, so every pipeline on this branch, which reads PointWorld, is unaffected. Anything that reads `metadata_*.json` extrinsics is not. Extrapolated to the corpus this is roughly 1,300 test and 11,600 train episodes, but only the 40 audited ones are known individually.
4. **The method-level failures concentrate on one rig.** Every misplacement above 100 px and every wrong-body segment is on AUTOLab+44bb9c36, where one exterior camera sits against the robot base so the arm fills the view and manipulated cloth moves with the gripper; both tracker 'behind a camera' warnings are on that rig too, which is worth one calibration check there. The 0d4edc83 rig fails by coverage instead: short episodes where the gripper is in both views too briefly for pairs to be confirmed.
5. **The RAIL 3.4-degree constant rotation offset** is the only calibration-level observation on the shelf scene; it does not affect the benchmark metrics and has not been traced to the PointWorld world frame versus the store base frame.

## Recommended order if action is taken later (none taken now)

- Intrinsics: switch the loader to per-serial lookup with per-episode as the first choice; no data edit.
- RPL truncated MP4s: re-download the three files from the public bucket (the bucket copies are 3-8 MB) or exclude the three episodes from rig runs; verify by frame count against the store.
- Calibration audit: extend the 40-scene cross-check to the smoke scenes' rigs, AUTOLab+44bb9c36 first, before trusting PointWorld there blindly.
- Everything in the scene-level section is the seeding work's problem, not the data's.
