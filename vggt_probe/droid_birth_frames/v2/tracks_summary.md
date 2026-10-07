# Track-On gripper tracks from the birth frame (DROID test split)

work items: 8576  logged: 8576  status: {'ok': 8576}
scenes with >= 1 track file: 4289; items with both cameras tracked: 4287
npz files on disk: 8576
frames tracked: 2409534  GPU time: 35.7 h  mean throughput: 18.8 frames/s per GPU
GPU peak memory (max over items): 0.7 GB

## Query points
points per item: median 65  p10 59  p90 71  min 6  (items with < 10 points: 1)

## Visibility (Track-On's own flag, threshold 0.8)
mean visible fraction over the tracked span: median 0.74  p10 0.35  p90 0.95
visible fraction at the LAST frame: median 0.77  p10 0.08;  items with < 25 % of points visible at the end: 1414 (16.5 %)

## Per lab

| lab | items | median vis (mean) | median vis (last) | items with < 25 % visible at end |
|---|---|---|---|---|
| TRI | 2458 | 0.70 | 0.73 | 502 |
| AUTOLab | 1226 | 0.72 | 0.77 | 205 |
| RAIL | 851 | 0.75 | 0.77 | 99 |
| IPRL | 714 | 0.78 | 0.82 | 103 |
| REAL | 567 | 0.78 | 0.82 | 75 |
| IRIS | 562 | 0.66 | 0.66 | 116 |
| CLVR | 550 | 0.83 | 0.84 | 40 |
| ILIAD | 396 | 0.66 | 0.59 | 111 |
| RPL | 388 | 0.80 | 0.91 | 25 |
| PennPAL | 284 | 0.76 | 0.74 | 43 |
| WEIRD | 264 | 0.72 | 0.74 | 46 |
| GuptaLab | 214 | 0.80 | 0.72 | 40 |
| RAD | 102 | 0.81 | 0.89 | 9 |

## Failures (0)


QA: `qa_videos/` (32 rendered items), `qa_tracks_sheet.jpg` (mid and last frame of 24 of them).

Files: `tracks/<ep>/<cam>_f<birth>.npz` with tracks (T,N,2) float32 in 1280x720 px (NaN before birth), visibility (T,N) bool,
queries (N,3) [t,x,y], meta (json: birth_frame, n_frames_store, n_frames_mp4, fps, store_scale=0.25 for the 320x180 ext store, support_grid, delta_v, ckpt).
`tracks_index.csv` is the per-item index.
