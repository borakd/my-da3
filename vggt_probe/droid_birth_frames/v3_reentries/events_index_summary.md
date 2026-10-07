# Index over all gripper entry events (v2 first entries + v3 re-entries)

events: 5420 in 4288 scenes; rows (event x camera): 10840
first entries (event 0, trackon_run_v2): {'ok': 8574, 'no_v2_item': 2}
re-entries (event >= 1, trackon_run_v3_reentries): {'ok': 2264}
re-entry masks: 2264; projection_disagree 205 (9.1 %); box used (auto empty) 20
re-entry tracks ok: 2264; points median 66; visible fraction median 0.84; tracked frames total 503964

columns: entry_frame = first frame of the event (track start), exit_frame = first frame after the gripper drops below 50 % in either camera (n_frames if it stays), npz relative to the outputs root, mask flags from the RobotSeg stage (projection_disagree = mask inconsistent with the PointWorld projection).
