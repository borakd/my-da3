
## Verification log 2026-09-07 (13 smoke episodes)

Supersedes the "Status: PROPOSAL, nothing implemented" line above for Stage 0,
Stage 1 and Stage 3 steps 1-3. Everything below was produced on 2026-09-07
between 16:47 and 17:52 CEST by five build/run agents, three score agents and
three adversarial verifiers. Code: `vggt_probe/` (uncommitted) plus an
uncommitted edit to `eval_pipeline/infer_and_eval_worker_ray.py`
(`--gt_scenes_root`, `--context_dir`). Results root:
`/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/` (tables in
`tables/`, per-run logs in `logs/`). Cache root:
`/gpfs/scratch/etur59/koc821022/vggt_cache/` (raw MP4s + t0 PNGs, manifest,
t0 token caches, pseudo_scenes trees). Slurm jobs: 45534245 (RAIL test),
45534376/45534377 (VGGT sweep), 45534440 (controls/baselines),
45535156/45535157 (F feeds; finished after the score reports were forced,
tables regenerated 17:52 with `vggt_probe/aggregate_feed_tables.py`).

Scenes: RAIL+80edfcb1+2023-07-14-14h-28m-45s plus `eval_pipeline/cg_smoke_scenes_12.txt`
(= `vggt_probe/smoke13_scenes.txt`, verified identical). Splits: short7 =
T<=200 (128,131,158,67,71,117,178 = RAIL + AUTOLab 0d4edc83), long6 = T>200
(705,453,588,742,693,340 = AUTOLab 44bb9c36), off8 = short7 + T=340.
Evaluator: `eval_bundle/bin/eval_depth_poses.py` defaults (per-frame median
depth scale, Sim(3) Umeyama on positions, RMSE pose reduce) for every row.
Lower is better except a1. All-13 reference rows come from the cached
`summary/per_scene_<label>.csv`; fresh re-runs reproduce them exactly (see D9).

Every number below was re-derived from the raw eval CSVs / meta JSONs by at
least one verifier unless tagged UNVERIFIED (taken from the producing agent's
report, not independently reproduced).

### Table A: VGGT-Omega pose sources scored directly (no CUT3R)

Variants (all wrist_0-first unless noted, VGGT-Omega-1B-512, 384x688):
anchored_t0 = [w0,e1,e2,w_t] per t (S=4); anchored_sliding = [w0,e1,e2,w_{t-1},w_t]
(S=5); wrist_pair = [w0,w_t] (S=2, no exterior anchors); ext_first =
[e1,e2,w0,w_t] re-expressed in w0; offline_all = one non-causal forward over
all wrist frames (T>400 skipped); offline_all_ext = same + e1,e2;
anchored_t0_rawwrist = anchored_t0 with 1280x720 raw wrist frames (RAIL only).

| variant | subset | n | absrel | a1 | ATE | rpe_t | rpe_rot | med ATE | med rpe_rot |
|---|---|---|---|---|---|---|---|---|---|
| anchored_t0 | all13 | 13 | 0.1583 | 0.8126 | 0.0904 | 0.0429 | 15.23 | 0.0883 | 6.43 |
| anchored_sliding | all13 | 13 | 0.1566 | 0.8178 | 0.0838 | 0.0336 | 11.32 | 0.0788 | 4.57 |
| wrist_pair | all13 | 13 | 0.1732 | 0.7734 | 0.1012 | 0.0384 | 10.22 | 0.1058 | 4.99 |
| ext_first | all13 | 13 | 0.1655 | 0.8365 | 0.0815 | 0.0509 | 13.94 | 0.0768 | 6.25 |
| augfull_lr1e5 (Regular CUT3R, no rays) | all13 | 13 | 0.1582 | 0.8031 | 0.0759 | 0.0076 | 1.12 | 0.0709 | 1.06 |
| augfull_cg_fuse_g7 (CUT3R champion) | all13 | 13 | 0.1662 | 0.7865 | 0.0642 | 0.0070 | 0.98 | 0.0475 | 1.03 |
| prevgt_lr1e5 (CUT3R, prev-GT rays) | all13 | 13 | 0.1702 | 0.7971 | 0.0139 | 0.0044 | 0.38 | 0.0131 | 0.36 |
| gtray_lr1e5 (CUT3R, GT rays) | all13 | 13 | 0.1779 | 0.7895 | 0.0129 | 0.0038 | 0.24 | 0.0116 | 0.23 |
| anchored_t0 | short7 | 7 | 0.1541 | 0.8240 | 0.0865 | 0.0423 | 20.22 | 0.0920 | 17.33 |
| anchored_sliding | short7 | 7 | 0.1534 | 0.8282 | 0.0793 | 0.0348 | 14.59 | 0.0807 | 6.88 |
| wrist_pair | short7 | 7 | 0.1660 | 0.7953 | 0.0742 | 0.0298 | 6.50 | 0.0787 | 3.39 |
| ext_first | short7 | 7 | 0.1611 | 0.8516 | 0.0731 | 0.0507 | 15.29 | 0.0898 | 10.72 |
| offline_all | short7 | 7 | 0.1288 | 0.8711 | 0.0117 | 0.0054 | 0.54 | 0.0117 | 0.45 |
| offline_all_ext | short7 | 7 | 0.1349 | 0.8703 | 0.0116 | 0.0055 | 0.57 | 0.0131 | 0.46 |
| augfull_lr1e5 | short7 | 7 | 0.1368 | 0.8442 | 0.0431 | 0.0074 | 1.03 | 0.0364 | 0.98 |
| augfull_cg_fuse_g7 | short7 | 7 | 0.1430 | 0.8236 | 0.0314 | 0.0060 | 0.81 | 0.0324 | 0.77 |
| prevgt_lr1e5 | short7 | 7 | 0.1425 | 0.8476 | 0.0099 | 0.0041 | 0.38 | 0.0099 | 0.36 |
| gtray_lr1e5 | short7 | 7 | 0.1445 | 0.8448 | 0.0097 | 0.0035 | 0.20 | 0.0106 | 0.20 |
| anchored_t0 | long6 | 6 | 0.1632 | 0.7993 | 0.0949 | 0.0436 | 9.42 | 0.0824 | 4.25 |
| anchored_sliding | long6 | 6 | 0.1602 | 0.8057 | 0.0890 | 0.0321 | 7.50 | 0.0731 | 3.66 |
| wrist_pair | long6 | 6 | 0.1815 | 0.7477 | 0.1327 | 0.0484 | 14.55 | 0.1121 | 14.97 |
| ext_first | long6 | 6 | 0.1705 | 0.8188 | 0.0913 | 0.0512 | 12.37 | 0.0712 | 5.44 |
| augfull_lr1e5 | long6 | 6 | 0.1832 | 0.7552 | 0.1141 | 0.0079 | 1.22 | 0.1135 | 1.26 |
| augfull_cg_fuse_g7 | long6 | 6 | 0.1932 | 0.7432 | 0.1025 | 0.0081 | 1.18 | 0.1052 | 1.10 |
| prevgt_lr1e5 | long6 | 6 | 0.2025 | 0.7381 | 0.0185 | 0.0047 | 0.38 | 0.0153 | 0.37 |
| gtray_lr1e5 | long6 | 6 | 0.2168 | 0.7251 | 0.0167 | 0.0041 | 0.29 | 0.0141 | 0.31 |
| offline_all | off8 | 8 | 0.1260 | 0.8691 | 0.0118 | 0.0054 | 0.56 | 0.0120 | 0.54 |
| offline_all_ext | off8 | 8 | 0.1311 | 0.8682 | 0.0117 | 0.0055 | 0.59 | 0.0128 | 0.53 |
| anchored_t0 | off8 | 8 | 0.1474 | 0.8283 | 0.0824 | 0.0420 | 18.11 | 0.0868 | 11.88 |
| anchored_sliding | off8 | 8 | 0.1465 | 0.8333 | 0.0752 | 0.0331 | 13.03 | 0.0795 | 5.72 |
| wrist_pair | off8 | 8 | 0.1603 | 0.8042 | 0.0739 | 0.0310 | 6.13 | 0.0752 | 3.45 |
| ext_first | off8 | 8 | 0.1550 | 0.8512 | 0.0722 | 0.0501 | 13.91 | 0.0777 | 7.46 |
| RAIL only: anchored_t0 | T=128 | 1 | 0.2583 | 0.8634 | 0.0567 | 0.0307 | 6.15 | | |
| RAIL only: anchored_t0_rawwrist | T=128 | 1 | 0.4128 | 0.8609 | 0.0332 | 0.0393 | 5.38 | | |

Source: `tables/vggt_variants_per_scene.csv`, `tables/vggt_variants_means.csv`
(all 345 VGGT cells and 260 CUT3R cells re-derived at maxdiff 0 by verifier 3).

### Table B: CUT3R fed with VGGT poses (F rows), exterior-context controls (C1), fresh baselines (B0)

F rows: pseudo-scene trees `vggt_cache/pseudo_scenes/<variant>_<scale>` carry
VGGT c2w poses (translation x s) + store K in `cam/*.npz`, store rgb symlinked;
the worker reads frames + conditioning from the pseudo tree and evaluates
against the REAL store GT (`--gt_scenes_root`, echoed `gt=` in both slurm
logs and confirmed on the live worker command lines by verifier 2).
`prevgt_prevgt` = prevgt_lr1e5 ckpt with `--conditioning prev_gt` (view t
carries view t-1's VGGT ray map); `gtray_gt` = gtray_lr1e5 ckpt with
`--conditioning gt` (view t carries its own VGGT ray map). raw = s=1; calib =
s_calib from DROID metadata extrinsics (see D4); oracle = Umeyama vs GT
(GT-USING DIAGNOSTIC, never an arm). slide = anchored_sliding poses instead of
anchored_t0. C1 = augfull ckpt, no rays, two 320x180 exterior t0 frames
prepended as unscored context views; C1b = same with the last-frame exterior
views. B0 = fresh re-runs of the cached baselines through the same worker.

| label | subset | n | absrel | a1 | ATE | rpe_t | rpe_rot |
|---|---|---|---|---|---|---|---|
| augfull_lr1e5 (cached) | all13 | 13 | 0.1582 | 0.8031 | 0.0759 | 0.00763 | 1.120 |
| gtray_lr1e5 (cached) | all13 | 13 | 0.1779 | 0.7895 | 0.0129 | 0.00382 | 0.239 |
| prevgt_lr1e5 (cached) | all13 | 13 | 0.1702 | 0.7971 | 0.0139 | 0.00437 | 0.381 |
| augfull_cg_fuse_g7 (cached) | all13 | 13 | 0.1662 | 0.7865 | 0.0642 | 0.00698 | 0.984 |
| B0_augfull_none (fresh) | all13 | 13 | 0.1582 | 0.8031 | 0.0759 | 0.00763 | 1.120 |
| B0_gtray_gt (fresh) | all13 | 13 | 0.1779 | 0.7895 | 0.0129 | 0.00382 | 0.239 |
| C1_ext_context | all13 | 13 | 0.1774 | 0.7829 | 0.1062 | 0.00832 | 1.657 |
| C1b_ext_context_last | all13 | 13 | 0.1760 | 0.7842 | 0.1007 | 0.00818 | 1.522 |
| F1_prevgt_prevgt_raw | all13 | 13 | 0.1802 | 0.7777 | 0.0927 | 0.04813 | 15.06 |
| F2_prevgt_prevgt_calib | all13 | 13 | 0.1783 | 0.7793 | 0.0927 | 0.04778 | 15.52 |
| F3_prevgt_prevgt_oracle (GT-using) | all13 | 13 | 0.1804 | 0.7761 | 0.0931 | 0.04805 | 15.82 |
| F4_gtray_gt_raw | all13 | 13 | 0.1856 | 0.7742 | 0.0893 | 0.04336 | 14.59 |
| F5_gtray_gt_calib | all13 | 13 | 0.1840 | 0.7755 | 0.0891 | 0.04285 | 14.91 |
| F6_gtray_gt_oracle (GT-using) | all13 | 13 | 0.1868 | 0.7696 | 0.0891 | 0.04278 | 15.08 |
| F7_prevgt_prevgt_slide_calib | all13 | 13 | 0.1790 | 0.7789 | 0.0859 | 0.03721 | 12.05 |
| F8_gtray_gt_slide_calib | all13 | 13 | 0.1821 | 0.7775 | 0.0826 | 0.03398 | 11.38 |
| augfull_lr1e5 | short7 | 7 | 0.1368 | 0.8442 | 0.0431 | 0.00739 | 1.031 |
| gtray_lr1e5 | short7 | 7 | 0.1445 | 0.8448 | 0.0097 | 0.00354 | 0.197 |
| C1_ext_context | short7 | 7 | 0.1529 | 0.8229 | 0.0698 | 0.00898 | 1.677 |
| C1b_ext_context_last | short7 | 7 | 0.1536 | 0.8182 | 0.0639 | 0.00883 | 1.656 |
| F2_prevgt_prevgt_calib | short7 | 7 | 0.1539 | 0.8232 | 0.0878 | 0.04669 | 20.36 |
| F5_gtray_gt_calib | short7 | 7 | 0.1575 | 0.8214 | 0.0841 | 0.04286 | 19.59 |
| F7_prevgt_prevgt_slide_calib | short7 | 7 | 0.1531 | 0.8238 | 0.0810 | 0.03823 | 15.63 |
| F8_gtray_gt_slide_calib | short7 | 7 | 0.1552 | 0.8238 | 0.0775 | 0.03571 | 14.64 |
| augfull_lr1e5 | long6 | 6 | 0.1832 | 0.7552 | 0.1141 | 0.00792 | 1.224 |
| gtray_lr1e5 | long6 | 6 | 0.2168 | 0.7251 | 0.0167 | 0.00414 | 0.289 |
| C1_ext_context | long6 | 6 | 0.2059 | 0.7361 | 0.1487 | 0.00755 | 1.633 |
| C1b_ext_context_last | long6 | 6 | 0.2021 | 0.7446 | 0.1436 | 0.00741 | 1.365 |
| F2_prevgt_prevgt_calib | long6 | 6 | 0.2067 | 0.7281 | 0.0985 | 0.04905 | 9.87 |
| F5_gtray_gt_calib | long6 | 6 | 0.2150 | 0.7220 | 0.0948 | 0.04283 | 9.44 |
| F7_prevgt_prevgt_slide_calib | long6 | 6 | 0.2091 | 0.7264 | 0.0915 | 0.03601 | 7.88 |
| F8_gtray_gt_slide_calib | long6 | 6 | 0.2136 | 0.7236 | 0.0885 | 0.03196 | 7.58 |

Per-scene wins vs augfull_lr1e5 (ATE / rpe_t / rpe_rot): every F row 5/13
(F8: 6/13) / 0/13 / 0/13; vs augfull_cg_fuse_g7 3/13 / 0 / 0; vs prevgt or
gtray 0/13 on all three pose metrics. C1 1/13 / 6/13 / 1/13; C1b 4 / 8 / 2.
gtray vs augfull 13/13 on all three; prevgt vs gtray ATE 4/13, rpe 0/13.
Source: `tables/cut3r_feeds_{per_scene,mean,wins,parity}.csv`,
`tables/cut3r_feeds_tables.md` (regenerated 17:52 with all 16 labels x 13
scenes; no `_failures_*` files under any F label).

### Table C: per-scene ATE / rpe_rot (scene = date suffix; T in parentheses)

| scene | T | anch_t0 | anch_slid | wrist_pair | ext_first | offline_all | augfull | prevgt | gtray |
|---|---|---|---|---|---|---|---|---|---|
| RAIL 07-14 | 128 | .0567/6.15 | .0334/4.57 | .0206/6.31 | .0115/1.27 | .0061/0.35 | .0252/0.87 | .0108/0.26 | .0070/0.20 |
| 10-21-19h11 | 131 | .1015/21.0 | .0977/20.5 | .1120/3.39 | .0411/10.7 | .0099/0.43 | .0469/1.05 | .0088/0.44 | .0090/0.23 |
| 10-21-19h48 | 158 | .0920/2.90 | .0807/2.32 | .0851/2.33 | .0900/2.19 | .0117/0.46 | .0820/1.36 | .0110/0.60 | .0106/0.28 |
| 10-27-19h55 | 67 | .1202/43.3 | .1294/44.0 | .1382/23.6 | .1200/43.4 | .0123/0.78 | .0326/1.28 | .0131/0.40 | .0108/0.22 |
| 10-27-20h28 | 71 | .0816/6.43 | .0880/6.88 | .0399/2.48 | .0987/14.6 | .0097/0.75 | .0160/0.98 | .0062/0.30 | .0037/0.15 |
| 12-02-14h17 | 117 | .0982/44.4 | .0782/21.6 | .0787/4.99 | .0898/30.9 | .0174/0.62 | .0629/0.85 | .0094/0.36 | .0138/0.16 |
| 12-02-15h41 | 178 | .0551/17.3 | .0475/2.36 | .0450/2.38 | .0605/3.89 | .0147/0.42 | .0364/0.84 | .0099/0.29 | .0129/0.14 |
| 11-23-10h45 | 705 | .0411/1.97 | .0419/1.42 | .1106/4.42 | .0554/4.63 | skip | .1040/1.04 | .0131/0.41 | .0092/0.32 |
| 11-23-18h32 | 453 | .0883/3.93 | .0788/3.45 | .1058/19.1 | .0512/3.73 | skip | .0923/1.22 | .0138/0.46 | .0137/0.33 |
| 11-25-11h05 | 588 | .0765/4.56 | .0675/3.87 | .1136/10.8 | .0768/6.25 | skip | .1295/1.31 | .0233/0.39 | .0174/0.29 |
| 11-25-13h39 | 742 | .0924/7.17 | .0832/5.44 | .1719/21.5 | .0830/6.69 | skip | .1229/1.33 | .0169/0.36 | .0145/0.24 |
| 11-28-09h03 | 693 | .2171/35.5 | .2160/28.7 | .2224/28.0 | .2155/48.7 | skip | .1650/1.39 | .0311/0.34 | .0335/0.23 |
| 11-29-23h38 | 340 | .0540/3.38 | .0467/2.14 | .0717/3.52 | .0656/4.20 | .0129/0.68 | .0709/1.06 | .0131/0.35 | .0116/0.32 |

F rows per scene (ATE / rpe_rot):

| scene | T | F1 raw | F2 calib | F3 oracle | F4 raw | F5 calib | F6 oracle | F7 slide | F8 slide |
|---|---|---|---|---|---|---|---|---|---|
| RAIL 07-14 | 128 | .0540/6.70 | .0536/6.69 | .0545/6.71 | .0512/6.24 | .0507/6.24 | .0515/6.24 | .0311/4.85 | .0286/4.62 |
| 10-21-19h11 | 131 | .1039/18.4 | .1041/18.8 | .1045/19.4 | .1005/17.8 | .1005/17.9 | .1008/18.6 | .1003/17.9 | .0971/17.6 |
| 10-21-19h48 | 158 | .0946/2.99 | .0950/3.00 | .0958/3.01 | .0912/2.93 | .0910/2.92 | .0910/2.91 | .0834/2.44 | .0793/2.33 |
| 10-27-19h55 | 67 | .1224/42.2 | .1227/43.4 | .1225/43.4 | .1177/41.6 | .1152/43.3 | .1153/43.3 | .1328/44.2 | .1270/43.9 |
| 10-27-20h28 | 71 | .0814/6.58 | .0828/6.73 | .0845/6.84 | .0776/6.34 | .0780/6.38 | .0796/6.37 | .0882/7.08 | .0853/6.71 |
| 12-02-14h17 | 117 | .1011/42.6 | .1020/46.2 | .1026/48.6 | .0986/41.7 | .0985/43.4 | .0993/45.1 | .0816/30.5 | .0772/25.0 |
| 12-02-15h41 | 178 | .0559/17.2 | .0542/17.7 | .0551/18.0 | .0544/16.5 | .0549/17.0 | .0523/17.2 | .0497/2.40 | .0479/2.35 |
| 11-23-10h45 | 705 | .0432/2.09 | .0427/2.10 | .0423/2.09 | .0401/2.00 | .0396/2.00 | .0398/2.00 | .0431/1.50 | .0409/1.45 |
| 11-23-18h32 | 453 | .0887/4.16 | .0884/4.15 | .0886/4.16 | .0878/3.94 | .0878/3.93 | .0875/3.93 | .0792/3.66 | .0783/3.44 |
| 11-25-11h05 | 588 | .0828/4.79 | .0825/4.81 | .0826/4.85 | .0760/4.59 | .0760/4.59 | .0762/4.61 | .0726/4.08 | .0665/3.91 |
| 11-25-13h39 | 742 | .1000/7.56 | .0999/7.60 | .1000/7.64 | .0946/7.15 | .0943/7.16 | .0938/7.17 | .0898/5.83 | .0853/5.48 |
| 11-28-09h03 | 693 | .2187/37.0 | .2188/36.9 | .2183/37.2 | .2184/35.6 | .2185/35.6 | .2178/35.2 | .2182/29.9 | .2177/29.0 |
| 11-29-23h38 | 340 | .0581/3.55 | .0589/3.64 | .0592/3.64 | .0528/3.37 | .0527/3.39 | .0529/3.41 | .0462/2.31 | .0423/2.16 |

### Decisions resolved

D1. Wrist resolution (store 320x180 vs raw 1280x720). NOT ACTIONABLE, n=1.
RAIL only: anchored_t0_rawwrist ATE 0.0332 vs 0.0567 (-41%), rpe_rot 5.38 vs
6.15, but rpe_trans 0.0393 vs 0.0307 (+28%) and absrel 0.413 vs 0.258 (+60%).
Keep the store 320x180 wrist frames (no download/decoding cost, same evaluator
input as CUT3R); revisit only if a per-t VGGT source is ever carried forward.
Evidence: Table A RAIL rows; `anchored_t0_rawwrist/eval/RAIL.../`.

D2. Frame order (wrist_0 first vs exterior first). KEEP WRIST_0 FIRST.
ext_first vs anchored_t0 on 13: ATE 0.0815 vs 0.0904 (wins 8/13), a1 0.837 vs
0.813, but rpe_trans 0.0509 vs 0.0429 (+19%, wins 4/13), absrel 0.166 vs 0.158
(wins 5/13), most pose jumps of any variant (200 vs 181). The RAIL result
(0.0115 vs 0.0567) does not generalise: without RAIL the 12-scene ATE is 0.0873
vs 0.0932; ext_first is worse than anchored_t0 on 2 of the 7 short episodes
(T=71, T=178; the score report's "4 of 7" was wrong). Order matters but not in
a consistent direction; wrist_0-first keeps VGGT world == CUT3R world.
Evidence: Table A, Table C, `tables/vggt_jump_analysis.csv`.

D3. Anchored VGGT pose quality vs the CUT3R arms. NO PER-T VGGT VARIANT IS A
USABLE CAUSAL POSE SOURCE. Best per-t variant anchored_sliding: ATE 0.0838 /
rpe_trans 0.0336 / rpe_rot 11.3 deg vs augfull 0.0759 / 0.0076 / 1.12 and
gtray 0.0129 / 0.0038 / 0.24. All four per-t variants lose rpe_trans and
rpe_rot on 13/13 scenes against all four CUT3R rows (per-scene multipliers
vs augfull: rpe_trans 1.9-7.5x, rpe_rot 1.4-34x; the 13-mean is 4-7x / 10x).
Exterior anchors do bound long-horizon drift: on long6 anchored_t0/sliding ATE
0.095/0.089 vs augfull 0.114 and wrist_pair 0.133; anchored_t0 beats augfull
on ATE 5/13, exactly the five 44bb9c36 episodes (T=705,453,588,742,340), but
loses rpe_rot 0/13 and never beats prevgt/gtray on any pose metric. Only the
non-causal offline_all is competitive: off8 ATE 0.0118 / rpe_rot 0.56 vs gtray
0.0099 / 0.21, i.e. 3.7x better than augfull on ATE but 2.8x worse than gtray on
rpe_rot; offline_all_ext is identical (0.0117 / 0.59): exterior frames add
nothing to the joint forward. Causal-vs-offline gap on short7: 7.4x ATE, 7.8x
rpe_trans, 37x rpe_rot. The prize identified in the proposal (GT rays ATE
0.0129 vs 0.0759) is real, but per-t VGGT does not deliver it; the failure is
per-frame orientation/translation jitter between independent forwards, not
global scale (see D4, D7). Evidence: Tables A/C, per-scene win counts
recomputed by verifiers 1 and 3.

D4. Scale handling (raw / calib / oracle). IRRELEVANT UNDER THIS EVALUATOR;
calib is the honest default if a scale is ever needed. F1/F2/F3 ATE 0.0927 /
0.0927 / 0.0931, rpe_rot 15.06 / 15.52 / 15.82; F4/F5/F6 0.0893 / 0.0891 /
0.0891, rpe_rot 14.59 / 14.91 / 15.08. Max per-scene |calib-raw| ATE 0.0016
(prevgt) / 0.0024 (gtray), |oracle-calib| 0.0017 / 0.0026, including the
11-28 outlier where s_calib/s_oracle = 5.64 (ATE 0.2187 / 0.2188 / 0.2183).
The evaluator's Sim(3) absorbs a global scale and CUT3R's ray channel is
scale-tolerant to this degree, so the sweep cannot separate the options;
s_calib/s_oracle is 0.92-1.58 on 12/13 episodes (median ~1.2; four within 15%:
10-27-19h55 0.92, 11-23-10h45 1.07, 11-29 1.13, RAIL 1.145) and 5.64 on 11-28.
Note (verifier 2): s_calib is the median of three pair ratios from the DROID
metadata, two of which use the t=0 kinematic wrist position (numerically the
store GT frame-0 translation); it coincides with the pure ext1-ext2 ratio on
11/13 episodes (RAIL 1.0814 vs 1.1022, 10-21-19h48 0.8836 vs 0.8855 differ).
Declare calib rows as "calibration incl. t=0 kinematic wrist position". Answers
proposal open decision 1: no metric rescale is needed for the ray channel;
if scale matters downstream (metric depth, Arm L), use s_calib.
Evidence: Table B, `vggt_cache/pseudo_scenes/anchored_t0_calib/scale_report.csv`.

D5. Sliding vs t0 anchors. SLIDING WINS CONSISTENTLY BUT NOT ENOUGH.
anchored_sliding vs anchored_t0: ATE 0.0904 -> 0.0838 (-7%, wins 10/13),
rpe_trans 0.0429 -> 0.0336 (-22%, 12/13), rpe_rot 15.2 -> 11.3 (-26%, 11/13;
median 6.4 -> 4.6), absrel 9/13, jumps 181 -> 145; the only improvement that
is consistent across scenes. Through CUT3R the same holds: F7 vs F2 ATE 0.0859
vs 0.0927, rpe_rot 12.05 vs 15.52; F8 vs F5 0.0826 vs 0.0891, 11.38 vs 14.91.
Cost 1.25x (0.1033 vs 0.0825 s/frame end-to-end, 16.1 vs 13.7 GiB at B=16).
Still 10x worse than augfull on rpe_rot. Evidence: Tables A/B/C.

D6. Exterior views natively through CUT3R (unscored context, zero training).
NEGATIVE CONTROL, IT HURTS. C1: ATE 0.1062 (+40% vs augfull 0.0759), rpe_rot
1.657 (+48%), rpe_trans +9%, absrel +12%, a1 -2.5%; ATE better than augfull on
1/13 (11-25-13h39, 0.1201 vs 0.1229), worst on RAIL (0.136 vs 0.025). C1b
(last-frame exterior views) 0.1007 / 1.522, same pattern (4/13 ATE wins); which
exterior frame does not matter. Penalty is larger on short7 (ATE +62%, rpe_rot
+63%) than long6 (+30% / +33%); C1b long6 rpe_trans 0.00741 is 6% better than
augfull, so it is rotation and accumulated drift that degrade, not step
translation. Verifier 1 refinement: CUT3R's registration of wrist frame 0
against ext1 is 60-150 deg off the GT relative rotation on all 13 episodes, so
the seeded state is geometrically wrong (Sim(3) absorbs the global offset; the
comparison stays fair). Consequence: CUT3R's state does not absorb static
exterior views without training; the proposal's "revisit gain" argument does
not transfer. Evidence: Table B, `summary_control_rows.csv`.

D7. Failure handling for Arm P (wrist close-ups that do not register).
depth_conf IS A USABLE GATE; hold-last-good is not yet tested. Jump =
translation step > 5x episode-median step (scale-free, no alignment):
anchored_t0 181/4358 steps (4.2%); 100/181 (55%) adjacent to a bottom-decile
depth_conf frame vs 24.5 (14%) expected under independence (4x enrichment);
sliding 79/145, ext_first 104/200, wrist_pair 84/180. Zero of these jumps
coincide with a GT jump (GT has >5x-median steps only in 12-02-15h41, 13 of
them; 0/181 overlap for anchored_t0, 1/200 ext_first): they are VGGT failures,
not motion. Mean step on low-conf frames is 2-4x the mean elsewhere (RAIL 0.060
vs 0.014). Anchored variants do not fail more on long episodes (anchored_t0
pooled jump rate short7 5.0% [42/843], long6 4.0% [139/3515]; the "5.4%" in the
score report was a mean of per-scene percentages) but wrist_pair does (1.9% vs
5.3%). Scale drift across t is a global per-forward scale, not anchor noise:
median CV(|e1-e2|) 0.061 (0.021-0.139), corr(|e1-e2|,|w0-e1|) median 0.991
(min 0.918), ratio CV 0.8%; peak-to-peak spread 8-46% per episode (the "15-40%"
in the score report was too narrow). The 11-28-09h03 episode (T=693) has
406/693 frames with conf<1.5 (median 1.32), ATE 0.216-0.222 in every per-t
variant and every F row, and is the worst scene for CUT3R too (augfull 0.165,
gtray 0.0335); there the relative-decile enrichment vanishes (2 vs 3.9
expected). Answers proposal open decision 3 partially: gating on bottom-decile
conf would remove about half the jumps; a per-t rescale by
d_e1e2(0)/d_e1e2(t) is well-posed and free (data in meta). Neither was applied;
both should be expected to move ATE, not rpe_rot (sliding has higher scale CV
0.087 yet better poses), so they cannot rescue the per-t feed on their own.
Evidence: `tables/vggt_jump_analysis.csv`, `tables/vggt_scale_cv.csv`,
`<variant>/meta/<EP>.json`.

D8. Cost per episode. MEASURED PER FRAME; THE SCORE REPORT'S EXTRAPOLATION
IS RETRACTED. anchored_t0 S=4 B=16 on H100: 0.0825 s/frame end-to-end (360.8 s
/ 4371 frames incl. t=0 forward and batching; median pure forward 0.0738
s/frame, the report's 0.0745 is UNVERIFIED), peak 13.74 GiB; sliding 0.1033
s/frame (1.25x), 16.12 GiB (UNVERIFIED peak); wrist_pair 0.59x, 9.0 GiB
(UNVERIFIED); offline S=340 33 s, 30.2 GiB (UNVERIFIED, ~88 MB/frame). Per
process ~8.6 s model load and ~5.9 s/episode CPU preprocess (both UNVERIFIED).
The report assumed 113 frames/episode; verifier 3 counted the real test store:
4292 scenes = 1,260,994 frames (mean 293.8, median 220, max 2437) -> 28.9 GPU-h
and ~1.26 TB of fp32 384x688 depth for anchored_t0 (not 11.1 GPU-h / 485 GB);
train 38,633 episodes, 300-episode sample mean 291.8 frames -> ~11.3M frames,
~258 GPU-h, ~11 TB (not 100 GPU-h / 4.4 TB); sliding ~36 / ~325 GPU-h. The
proposal's budget table (25 / 215 GPU-h) is therefore roughly right for
anchored_t0 and low for sliding. Any train-scale extraction must store poses
+ fp16 or no depth. The data-acquisition side (2 exterior MP4s per episode,
3-16 MB each, login-node 300 s CPU cap; ~8 MB x 2 x 38,633 = ~600 GB for train)
is unbudgeted and likely the real bottleneck. Moot unless an arm survives D3.
Evidence: `anchored_t0/meta/*.json`, verifier 3 re-derivation.

D9. Are the cached 4292-scene baselines protocol-identical? YES, EXACTLY.
B0_augfull_none vs cached per_scene_augfull_lr1e5 and B0_gtray_gt vs cached
per_scene_gtray_lr1e5: max abs diff 0.000e+00 on all five metrics over 13/13
scenes, and camera/intrinsics/depth preds bit-identical to
`outputs/cut3r_eval/{augfull_lr1e5,gtray_lr1e5}/preds`. The cached
per_scene CSVs are valid reference rows for every table here.
Evidence: `tables/cut3r_feeds_parity.csv`, `vggt_probe/parity_*.csv`.

D10 (added; the actual Arm P test). Arm P at zero training is DEAD. Every F
row re-scores its VGGT input trajectory: per-scene corr(F ATE, input variant
ATE) 0.998-0.999, median ratio 0.986-1.029; rpe_rot corr 0.984-0.999. With
gt/prev_gt conditioning the trained ray channel is a near pass-through, so
"F beats augfull" reduces to "anchored_t0 beats augfull" (5/13 ATE, 0/13 rpe).
Best row F8 (gtray ckpt, gt-style, sliding, calib): ATE 0.0826 / rpe_trans
0.0340 / rpe_rot 11.38 vs augfull 0.0759 / 0.0076 / 1.12, cg_fuse_g7 0.0642 /
0.0070 / 0.98, gtray 0.0129 / 0.0038 / 0.24. Distance to topline: F8/gtray =
6.4x ATE, 8.9x rpe_trans, 48x rpe_rot. The proposal's noise-oracle premise
("exogenous error up to head magnitude is tolerated") is falsified for this
error magnitude: the feed error is ~10x the head's own. Depth: F rows absrel
0.178-0.187 (the expected ray-ckpt regression, not a VGGT artefact).
The win rule of Stage 3 step 6 (per-scene win% vs augfull on ATE AND rpe_rot)
is failed by every F row at 0/13 on rpe_rot.

Proposal open decision 2 (benchmark definition / rig-benchmark labelling):
NEEDS USER; moot until an arm passes the win rule. All rows above that use
exterior frames or metadata are labelled as such.

### Path to proceed

Nothing from Arm P goes to the 430 subset. Every zero-training feed of VGGT
per-t poses through the ray channel (F1-F8) loses rpe_rot and rpe_trans on
13/13 scenes to plain CUT3R, and the native exterior-context control (C1)
loses on all five metrics; the win rule cannot be met by any settings change
that this sweep exposes (order, resolution, scale, sliding). The two cheap
post-hoc probes still open (per-t rescale by d_e1e2(0)/d_e1e2(t); bottom-decile
conf gate with hold-last-good) can be run on the 13 in minutes from the
existing meta, but both target ATE while the blocking metric is rpe_rot, so
they are diagnostics, not a path. What remains of the proposal is Arm L (t=0
camera+register tokens as trained decoder context; caches for all 13 exist at
`vggt_cache/t0/<EP>.npz`), which is untested and is the next step in the
scoring order (Stage 3 step 4: bit-identical plumbing check at zero-init, then
single-episode overfit liveness) before any 430-subset run; note that D6 is a
warning for it (CUT3R at zero training mis-registers exterior views by 60-150
deg, so the adapter must learn something CUT3R cannot do natively). The only
VGGT pose source that is competitive is the non-causal offline forward
(off8 ATE 0.0118 / rpe_rot 0.56, within 1.2x of gtray on ATE), which is
incompatible with the online setting and skips T>400; whether a windowed /
latency-K offline variant is admissible is a NEEDS-USER decision that would
redefine the benchmark. If the user nonetheless wants one VGGT feed carried to
the 430 subset for the record, the settings are: anchored_sliding, wrist_0
first, store 320x180 wrist frames, s_calib scale, gtray_lr1e5 ckpt with
`--conditioning gt` (F8), B=16, S=5, fp16 depth or no depth, expected ~36
GPU-h for the full test split and 0/13 on rpe_rot at this scale.

### NUANCES TRACKER

Status legend: resolved = answered by evidence in this log; open = affects a
conclusion or a next step and is not yet settled; needs-user = policy choice.

Data / exterior frames
- N1. Two of the 13 are failure-status DROID episodes (11-23-10h45 T=705,
  11-29-23h38 T=340). Consequence: task semantics differ, data/calibration
  fine; they are among the scenes where exterior context hurts most. resolved.
- N2. Store pose world == robot base frame (metadata extrinsics reproduce
  store c2w to 3e-8 m / 0.012 deg with scipy Euler 'xyz' extrinsic).
  Consequence: ext c2w usable in the store world without alignment. resolved.
- N3. Exterior cameras are assumed static (one calibration per episode, no
  per-frame ext poses). Consequence: a camera bump would be invisible; last
  -frame PNGs allow a visual check. open (low).
- N4. The Franka arm enters the exterior views in essentially every episode
  (last-vs-first mean gray diff 6-35/255) and is partly visible at t0 in the
  44bb9c36 and 12-02 episodes. Consequence: ext-vs-wrist multi-view geometry
  sees a non-static object; t0 frames are cleanest but not arm-free. open.
- N5. In AUTOLab 0d4edc83 10-27/12-02 episodes ext1 is ~40% occluded by a
  black panel and ext2 looks past the table edge. Consequence: ext/wrist
  overlap smaller than calib distances suggest; these are the 43-46 deg
  rpe_rot scenes (10-27-19h55, 12-02-14h17). open (explains, not fixed).
- N6. Same camera serials across the two AUTOLab rigs but per-episode
  extrinsics differ. Consequence: never reuse one episode's extrinsics.
  resolved.
- N7. Exterior intrinsics are not in the metadata (only store wrist K0).
  Consequence: any stage that needs ext K (ray maps for ext views, Arm L
  geometry) must get it from ZED calibration by serial or VGGT FoV. open.
- N8. MP4 fps tag 60 is nominal (store keeps every MP4 frame). Consequence:
  do not use it for timing. resolved.
- N9. MP4 frame-count check is container-level; store==MP4 pixel equality
  rests on the earlier shared verification. resolved.
- N10. raw/ dir pre-existed from an earlier attempt; the downloader script was
  only exercised on its skip-existing branch. Consequence: none for
  correctness; untested over the network. open (low).

VGGT extraction / sweep
- N11. Per-t forwards have independent scale (|e1-e2| CV 0.06-0.09, peak-to-
  peak 8-46%); a Sim(3) cannot absorb it; per-t rescale by
  d_e1e2(0)/d_e1e2(t) is well-posed and unapplied. Consequence: could recover
  ATE, not rpe_rot. open (cheap probe).
- N12. VGGT scale is forward-composition dependent (4-frame vs 130-frame
  forwards differ ~2x). Consequence: never compare t_norm / distances across
  variants without normalising. resolved.
- N13. rpe_rot of per-t variants is orientation jitter between independent
  forwards, not drift; sliding reduces it 26% but leaves it 10x augfull.
  resolved (D5).
- N14. Five episodes (705,453,588,742,693) exceed the offline cap of 400, so
  offline_* rows exist for 8/13 and every offline comparison is short7/off8;
  the skipped set includes the hardest scene. Consequence: offline_all's 0.012
  ATE is unproven on long low-conf episodes; T=453 would fit (~40 GiB,
  UNVERIFIED), T=742 would not. open.
- N15. 11-28-09h03 (T=693) is an outlier in every per-t variant and F row
  (~400 low-conf frames, ATE ~0.22, rpe_rot 28-49) and the worst CUT3R scene
  (augfull 0.165). Cause not diagnosed. Consequence: dominates 13-means;
  medians and win counts reported alongside. open.
- N16. Means are not robust (rpe_rot dominated by 10-27-19h55 and
  12-02-14h17; ATE by 11-28). Consequence: rank conclusions use per-scene wins
  and medians, both reported. resolved.
- N17. RAIL rows for anchored_t0 / wrist_pair / ext_first / offline_* /
  rawwrist come from the B=8 test job 45534245, the other 12 scenes from B=16
  jobs; parity 1e-4 pose / 2e-3 depth. Consequence: metrics comparable;
  RAIL timing/memory rows are not. Their per-run logs were overwritten by the
  sweep's "skipping" lines; values remain traceable to the test job stdout.
  resolved.
- N18. offline_all's RAIL 128-frame forward took 95 s vs 6 s for the 130-frame
  offline_all_ext forward; not root-caused. Consequence: offline per-frame
  timing is unreliable; excluded from cost figures. open (timing only).
- N19. "Low confidence" threshold conf_mean<1.5 is ad hoc; offline forwards
  run systematically higher conf (~4-6) than per-t (~1.5-4). Consequence:
  treat lowconf counts as relative; the jump analysis uses per-episode
  bottom-decile instead. resolved.
- N20. The bottom-decile gate is per-episode relative, so ~10% of frames
  qualify everywhere; for 11-28 it is meaningless. Consequence: an absolute
  threshold would gate that whole episode, a relative one would not; the 4x
  enrichment holds for the 12 normal episodes. open (gate design).
- N21. Jump definition and GT-jump check use unaligned c2w in each frame's
  own scale; scale-free by construction, but 6-14% per-t scale drift can
  itself produce apparent steps (unlikely at 5x median). resolved.
- N22. Jump-rate split figures in the score report mixed pooled and
  mean-of-percentages definitions (short7 5.4% vs pooled 5.0%); the
  "5-27x" step ratio holds for anchored_t0 only (3.6-73.7 across variants).
  Consequence: direction of conclusions unchanged; pooled numbers used in D7.
  resolved.
- N23. Sliding at t=1 duplicates wrist_0 as wrist_{t-1} (S=5 constant);
  parity 2e-8 vs 1e-4 for S=4 not investigated. Numerically irrelevant.
  resolved.
- N24. t=0 pose written as exact identity while VGGT's own E_0 deviates by
  up to 2.3e-3; t=0 depth comes from the S=3 ext-only forward. resolved.
- N25. Saved `intrinsics` = store K (320x180), VGGT's under `intrinsics_vggt`;
  evaluator ignores intrinsics. Consequence: unprojecting the 384x688 depth
  needs `intrinsics_vggt`. resolved.
- N26. Depth stored fp32 at 384x688 (1 MB/frame): sweep added ~20 GB, test
  extraction would be ~1.26 TB, train ~11 TB. Consequence: never store fp32
  depth at scale. open (policy for any future extraction).
- N27. anchored_t0_rawwrist is single-episode evidence (D1). resolved.
- N28. Both sweep jobs shared node as02r3b18; per-forward timings match the
  single-job test within 5%. resolved.
- N29. offline_all_ext == offline_all on all 8 episodes. Consequence:
  exterior frames add nothing non-causally; any anchoring value must be
  causal. resolved.
- N30. The first RAIL test job crashed on a batching index bug, fixed and
  rerun; failed-run logs were overwritten. resolved.
- N31. Depth resolution/resampling (384x688 vs 192x320, evaluator bilinear
  point sampling) does not bias absrel measurably (verifier 1: <0.001 on two
  scenes); VGGT preprocessing does not crop 16:9. resolved.

Pseudo-scenes / feeds
- N32. s_calib vs s_oracle ratio 0.92-1.58 on 12/13 (median ~1.2), 5.64 on
  11-28 (s_oracle biased by ~400 mis-registered frames); the score report's
  "1.07-1.58, only two within 15%" was wrong (four within 15%). Consequence:
  none for the F rows (D4). resolved.
- N33. s_calib includes two wrist-involving pair ratios using the t=0
  kinematic wrist position (= store GT frame-0 translation); inert on 11/13.
  Consequence: label calib rows accordingly (D4); legitimately available at
  deployment. resolved.
- N34. s_calib identical between anchored_t0 and anchored_sliding trees (shared
  t=0 forward); only s_oracle differs. Consequence: F7/F8 vs F2/F5 isolate
  trajectory quality at equal scale. resolved.
- N35. F3/F6 are GT-using diagnostics; raw/calib trees carry an s_oracle column
  but apply s=1 / s_calib (verified 13/13). resolved.
- N36. Pseudo trees symlink the store GT depth/sky/outlier masks; nothing
  reads them today (worker reads rgb+cam only), but a future worker change
  could silently become an oracle. Consequence: drop the depth symlink from
  `build_pseudo_scenes.py`. open.
- N37. Ray-conditioned worker runs store the cam pose in views[i]['camera_pose'];
  the only consumer is the `--oracle gt` + prev_pred_gru branch, never enabled
  here (live command lines checked). Consequence: inert; never add `--oracle`
  to a pseudo-tree run. resolved.
- N38. gt_scenes_root check: worker never prints its evaluator command; both
  slurm headers echo scenes=13 and gt=<real store> for all 8 labels, and live
  `srun --overlap ps` showed `--scenes_root` = pseudo tree with no `--oracle`
  / `--context_dir`. resolved.
- N39. Pseudo-scene sanity: 4371 cam npz per tree, no non-finite values, det R
  within 1e-3, frame counts match the store, Sim(3)-ATE of the raw pseudo
  trajectory reproduces the anchored_t0 evaluator ATE per scene (GT-vs-GT
  control ~1e-16). resolved.
- N40. Store path resolves through a symlink to
  `pointworld_droid_wrist_VALAR/<EP>`; pseudo rgb links point at the resolved
  path. Consequence: both roots must stay in place. resolved.
- N41. Worker resume: scenes with an existing eval CSV are skipped; rebuilt
  pseudo trees or regenerated preds require deleting `<label>/eval` (and
  preds) or a new label. resolved (documented).
- N42. F label names are the long form (F1_prevgt_prevgt_raw ...), not the
  example config's short names. resolved.
- N43. Feed jobs were pending when the score reports were forced; they ran
  17:43-17:5x, 13/13 per label, no failures; tables regenerated 17:52.
  resolved.
- N44. wrist_pair has no anchors so s_calib is NaN there (calib fails loudly).
  resolved.

Controls / baselines / evaluator
- N45. `run_cut3r_feed.sbatch` originally defaulted SCENE_LIST after sourcing
  mn5_paths.sh (4292 scenes); first control submission 45534412 cancelled at
  56 s, fixed, resubmitted. Consequence: check `scenes=13` in every feed log
  (done for 45534440, 45535156, 45535157). resolved.
- N46. Exact protocol parity (D9). resolved.
- N47. Under context, CUT3R's world is ext1 and wrist frame 0 is 60-150 deg
  mis-registered vs GT on all 13; Sim(3) absorbs the rigid part, RPE metrics
  degrade too, so the C1 penalty is real. resolved.
- N48. Context PNGs (1280x720 -> 320x180 INTER_AREA, then 320x192 in the
  worker) vs store wrist frames from another downscale pipeline: mild domain
  gap, not separable from the penalty. open (low).
- N49. Worker prints only 3 scene lines + DONE; completeness verified via file
  mtimes/counts, not logs. resolved.
- N50. Evaluator defaults: per-frame median depth scaling then mean over frames
  (not CUT3R-paper pooled absrel), Sim(3) on all frames, RMSE reduce. Identical
  for every row, but no row measures metric scale and absrel is not
  paper-comparable. resolved (documented).
- N51. Short/long split (T=200) coincides exactly with rig/site identity
  (RAIL+0d4edc83 vs 44bb9c36); all 5 anchored_t0 ATE wins over augfull are
  44bb9c36. Consequence: "anchors help long episodes" is confounded with
  "anchors help that rig". open (design caveat for any 430 run).
- N52. gtray topline worsens depth (absrel 0.158 -> 0.178); F rows regress
  depth likewise. Consequence: judge feeds on pose metrics. resolved.
- N53. augfull_cg_fuse_g7 wins ATE only 9/13 here and gains little on long6;
  the practical non-GT bar on this subset is 0.0642 / 0.984. resolved.
- N54. C1b long6 rpe_trans 0.00741 < augfull 0.00792 while ATE is +26%:
  rotation/accumulated error degrade, not step translation. resolved.
- N55. CUT3R reference rows taken from the cached CSVs; evaluator settings
  assumed identical -> confirmed by exact parity. resolved.

Cost / bookkeeping
- N56. Cost extrapolation used 113 frames/episode (unsupported); real means
  are 294 (test) / ~292 (train sample). Consequence: GPU-h and disk figures
  x2.6 (D8). resolved (corrected).
- N57. Exterior MP4 acquisition for 4292/38,633 episodes (login-node 300 s
  CPU cap, ~70 GB / ~600 GB) is unbudgeted. open (moot unless an arm survives).
- N58. Monitors on the feed jobs had 1 h caps; jobs finished within them.
  resolved.
- N59. Results root also holds other agents' B0_*/C1_* dirs (excluded from
  Table A via --variants) and `cut3r_feeds_*` tables (now regenerated).
  resolved.
- N60. Git state: `M eval_pipeline/infer_and_eval_worker_ray.py`, `??
  vggt_probe/`, `?? VGGT_FEATURES_*.md`, `?? eval_pipeline/cg_smoke_scenes_12.txt`.
  Nothing committed. open (commit the shelf).
- N61. No writes under /gpfs/projects; scal3r_sweep job 45532813 untouched.
  resolved.

### Verifier issues

Major (verbatim):

V1 (verifier 1, major): "feedScore report is now stale: all eight F labels
have 13/13 eval CSVs (jobs 45535156/45535157 started 17:43:56, both finished;
no F*/eval/_failures_* files). The blank F rows and the 'PARTIAL' status must
be regenerated with vggt_probe/aggregate_feed_tables.py before anything is
concluded from that table." -> ADDRESSED: regenerated at 17:52; Table B and
D4/D5/D10 use the regenerated rows, which match the verifier's own
re-derivation (F1 0.0927/15.06 ... F8 0.0826/11.38) exactly.

V2 (verifier 1, major): "The F rows are not an apples-to-apples test of
'CUT3R + VGGT info' vs augfull: with gt/prev_gt conditioning the scored CUT3R
output pose is essentially a pass-through of the injected ray pose, so the F
rows re-score the VGGT trajectories, not CUT3R's drift recovery. Consequently
the raw/calib/oracle scale sweep (F1/F2/F3, F4/F5/F6) is uninformative by
construction under the evaluator's global Sim3 (which absorbs any global
scale), and 'F beats augfull' reduces to 'anchored_t0 beats augfull' (5/13 ATE
wins, 0/13 rpe wins)." -> ADDRESSED as a finding, not a fix: reproduced here
(corr 0.998-0.999, median ratio 0.986-1.029, calib-vs-raw max per-scene ATE
delta 0.0016-0.0024) and written into D4 and D10; it is the reason Arm P is
declared dead rather than "needs a better scale". Not addressable by re-running:
it is a property of the trained ray channel.

V3 (verifier 3, major): "vggtScore finding 5 (cost extrapolation) uses an
unsupported '113 frames per episode' assumption; the real test store is 2.6x
larger, so every GPU-hour and disk figure for test and train is understated by
~2.6x. Test: 4292 scenes = 1,260,994 frames (mean 293.8, median 220, min 32,
max 2437), not 485k -> at the measured 0.0825 s/frame that is 104,000 s = 28.9
GPU-h (report: 11.1) and ~1.26 TB of fp32 384x688 depth (report: 485 GB).
Train: 38,633 episodes confirmed, but a 300-episode random sample gives mean
291.8 frames (sem 13), i.e. ~11.3M frames -> ~930,000 s = ~258 GPU-h (report:
100) and ~11 TB disk (report: 4.4 TB); anchored_sliding scales to ~36 / ~325
GPU-h. The per-episode figures (model load, preprocess) are unaffected. No file
on disk contains a 113-frame mean; the only '113' in the design doc is an
unrelated CUT3R paper metric." -> ADDRESSED: original figures retracted, D8
carries the verifier's counts (train figure is sample-based, sem 13 frames).

Minor issues and how they were handled: verifier 1: RAIL VGGT CSV provenance
(logs overwritten; traceable via test-job stdout, accepted, N17); C1 registration
60-150 deg off (folded into D6/N47); evaluator is not the CUT3R-paper path
(N50); depth resolution does not bias absrel (N31); short/long confounded with
rig (N51). Verifier 2: s_calib includes t=0 kinematic wrist position (D4/N33);
camera_pose stored but inert (N37); GT depth symlinks in pseudo trees (N36,
open); F3/F6 GT-using and labelled (N35). Verifier 3: per-scene multipliers
overstated (fixed in D3); jump-rate definitions mixed (fixed in D7/N22); "5-27x"
anchored_t0 only (N22); peak-to-peak 8-46% not 15-40% (fixed in D7); 0.0745 vs
0.0738 s/frame (D8, UNVERIFIED tag); ext_first worse on 2/7 not 4/7 short
episodes (fixed in D2); feedScore staleness (V1); s_calib/s_oracle range
0.92-1.58 (fixed in D4/N32). All three verifier verdicts: sound / sound /
flawed (the "flawed" verdict is V3, now corrected). Every table cell above was
re-derived from the raw eval CSVs at maxdiff 0 by verifier 3 (VGGT and cached
CUT3R rows) or by verifier 1 (F rows, after regeneration).
