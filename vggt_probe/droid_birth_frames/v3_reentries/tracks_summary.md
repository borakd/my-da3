# Track-On gripper tracks from the birth frame (DROID test split)

work items: 2264  logged: 2264  status: {'ok': 2264}
scenes with >= 1 track file: 682; items with both cameras tracked: 443
npz files on disk: 2264
frames tracked: 503964  GPU time: 7.4 h  mean throughput: 18.9 frames/s per GPU
GPU peak memory (max over items): 0.7 GB

## Query points
points per item: median 66  p10 58  p90 74  min 2  (items with < 10 points: 5)

## Visibility (Track-On's own flag, threshold 0.8)
mean visible fraction over the tracked span: median 0.84  p10 0.41  p90 1.00
visible fraction at the LAST frame: median 0.85  p10 0.01;  items with < 25 % of points visible at the end: 466 (20.6 %)

## Per lab

| lab | items | median vis (mean) | median vis (last) | items with < 25 % visible at end |
|---|---|---|---|---|
| TRI | 678 | 0.84 | 0.84 | 149 |
| AUTOLab | 562 | 0.76 | 0.78 | 143 |
| IPRL | 232 | 0.89 | 0.92 | 25 |
| ILIAD | 162 | 0.87 | 0.86 | 35 |
| IRIS | 118 | 0.90 | 0.88 | 20 |
| REAL | 116 | 0.87 | 0.87 | 17 |
| WEIRD | 100 | 0.75 | 0.72 | 32 |
| PennPAL | 92 | 0.93 | 0.93 | 14 |
| RAIL | 88 | 0.83 | 0.78 | 20 |
| RPL | 58 | 0.90 | 0.97 | 2 |
| CLVR | 54 | 0.93 | 0.94 | 9 |
| GuptaLab | 2 | 0.65 | 0.59 | 0 |
| RAD | 2 | 0.42 | 0.35 | 0 |

## Failures (0)


QA: `qa_videos/` (24 rendered items), `qa_tracks_sheet.jpg` (mid and last frame of 24 of them).

Files: `tracks/<ep>/<cam>_f<birth>.npz` with tracks (T,N,2) float32 in 1280x720 px (NaN before birth), visibility (T,N) bool,
queries (N,3) [t,x,y], meta (json: birth_frame, n_frames_store, n_frames_mp4, fps, store_scale=0.25 for the 320x180 ext store, support_grid, delta_v, ckpt).
`tracks_index.csv` is the per-item index.
