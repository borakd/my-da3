# Validation of the kinematic birth-frame detector against RobotSeg

Detector: `vggt_features/vggt_probe/birth_frames.py` (54-point box model of the Robotiq 2F-85 in
the wrist-camera frame, projected into ext1/ext2 with DROID metadata extrinsics + ZED factory
intrinsics; "visible" = fraction of points inside the image with positive depth).

Reference: RobotSeg (showlab) gripper masks, prompted with the kinematic box on a frame where the
gripper is fully in both views, propagated forward and backward. RobotSeg entry frame = first frame
with mask area >= 0.2 % of the image. Compared against the kinematic 25 % threshold (closest in
meaning to a 0.2 % mask).

| set | scenes | camera-scene pairs | agree (|diff| <= 5 frames, or both 0) | disagree |
|---|---|---|---|---|
| 1: RAIL smoke episode + 5 AUTOLab | 6 | 12 | 12 | 0 |
| 2: late-birth scenes, one per lab (TRI, REAL, CLVR, ILIAD, IRIS, PennPAL, IPRL, RPL) | 8 | 16 | 12 | 4 |

Pairs where both methods see a real entry (> 0): RAIL ext1 34 vs 32, RAIL ext2 38 vs 38,
REAL ext1 16 vs 19, CLVR ext2 32 vs 36, IRIS ext1 15 vs 16, IRIS ext2 20 vs 22, PennPAL ext1 40 vs 36
(RobotSeg vs kinematic-25 %): all within 4 frames.

The 4 disagreements (ILIAD ext1, IPRL ext1, RPL ext1, RPL ext2) were inspected frame by frame
(`validation2/mismatch_inspection.jpg`): in every case the kinematic model says the gripper is
above the top image border, and the RobotSeg mask that exists in those frames sits on the robot
BASE / mount or is a residual blob left by backward propagation. The RobotSeg mask collapses to
~0 % right before the kinematic entry and re-grows together with it. The kinematic frame is the
correct one in all four.

Two of the 30 scenes flagged "never visible to both cameras at 50 %" were inspected
(`validation/never_*.jpg`): both are genuine camera placements (one scene has both cameras aimed
at a door with the robot at the edge of view; in the other, ext2 looks at the couch below the
arm's working height while ext1 tracks the gripper perfectly).

Known limitations: occlusion by the arm, objects or people is not modelled (only the image
border is); the gripper geometry is a coarse box; the exterior extrinsics are DROID's own
per-scene calibration and inherit its errors. Frame indices are 0-based store/MP4 frame indices
(the store drops the last row of trajectory.h5).
