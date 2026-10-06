# Track-On gripper tracks from the birth frame (DROID test split)

work items: 8522  logged: 8522  status: {'ok': 8522}
scenes with >= 1 track file: 4262; items with both cameras tracked: 4260
npz files on disk: 8522
frames tracked: 2396150  GPU time: 36.1 h  mean throughput: 18.4 frames/s per GPU
GPU peak memory (max over items): 0.7 GB

## Query points
points per item: median 65  p10 59  p90 71  min 1  (items with < 10 points: 5)

## Visibility (Track-On's own flag, threshold 0.8)
mean visible fraction over the tracked span: median 0.74  p10 0.35  p90 0.96
visible fraction at the LAST frame: median 0.77  p10 0.08;  items with < 25 % of points visible at the end: 1390 (16.3 %)

## Per lab

| lab | items | median vis (mean) | median vis (last) | items with < 25 % visible at end |
|---|---|---|---|---|
| TRI | 2436 | 0.70 | 0.73 | 493 |
| AUTOLab | 1226 | 0.72 | 0.78 | 204 |
| RAIL | 850 | 0.76 | 0.79 | 96 |
| IPRL | 702 | 0.78 | 0.81 | 102 |
| REAL | 566 | 0.80 | 0.82 | 72 |
| IRIS | 562 | 0.66 | 0.67 | 109 |
| CLVR | 540 | 0.83 | 0.87 | 36 |
| ILIAD | 394 | 0.68 | 0.61 | 108 |
| RPL | 388 | 0.80 | 0.91 | 29 |
| PennPAL | 280 | 0.75 | 0.76 | 45 |
| WEIRD | 262 | 0.72 | 0.76 | 46 |
| GuptaLab | 214 | 0.80 | 0.72 | 41 |
| RAD | 102 | 0.81 | 0.89 | 9 |

## Failures (0)


QA: `qa_videos/` (96 rendered items), `qa_tracks_sheet.jpg` (mid and last frame of 24 of them).

Files: `tracks/<ep>/<cam>_f<birth>.npz` with tracks (T,N,2) float32 in 1280x720 px (NaN before birth), visibility (T,N) bool,
queries (N,3) [t,x,y], meta (json: birth_frame, n_frames_store, n_frames_mp4, fps, store_scale=0.25 for the 320x180 ext store, support_grid, delta_v, ckpt).
`tracks_index.csv` is the per-item index.
