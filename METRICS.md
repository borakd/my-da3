# Metrics In `train_mine.py`

This document lists the scalar metrics currently computed and logged by `src/CUT3R/src/train_mine.py` with the active criteria setup used in `src/CUT3R/config/debug.yaml`.

## 1) Core Loop Metrics

These are always computed in the training loop and tracked by `MetricLogger`:

- `loss`: total training loss returned by the training criterion for the batch.
- `lr`: optimizer learning rate.
- `step`: global training step index used for scalar/image logging.
- `epoch`: fractional epoch progress (`epoch + iter/len(loader)`).

TensorBoard train-only convenience tags:

- `train_loss`: reduced training loss written at logging intervals.
- `train_lr`: learning rate written at logging intervals.
- `train_iter`: `int(epoch_f * 1000)` convenience x-axis scalar.

## 2) Criterion-Derived Training Metrics

Training criterion (from `debug.yaml`):

- `ConfLoss(Regr3DPoseBatchList(L21Loss())) + RGBLoss(MSE)`

Scalar metric families produced by this criterion:

- `Regr3DPoseBatchList_self_pts3d/<i>`: self-view 3D point regression loss for view `i`.
- `Regr3DPoseBatchList_pts3d/<i>`: cross-view 3D point regression loss for view `i`.
- `ConfLoss(Regr3DPoseBatchList(L21Loss()))_conf_loss_self/<i>`: confidence-weighted self-view loss term for view `i`.
- `ConfLoss(Regr3DPoseBatchList(L21Loss()))_conf_loss/<i>`: confidence-weighted cross-view loss term for view `i`.
- `RGBLoss_rgb/<i>`: RGB reconstruction loss for view `i`.
- `pose_loss`: camera pose regression loss (translation + quaternion error).

Notes:

- `i` is a view index (1-based). With your current config (`num_views: 24`), these appear for `i = 1..24`.
- Non-scalar tensors such as `gt_img*`, `pred_rgb_*`, `conf_*`, masks, descriptors are produced for visualization but are not scalar metrics.

## 3) Criterion-Derived Testing Metrics

Testing criterion (from `debug.yaml`):

- `Regr3DPose(L21, ...) + Regr3DPose_ScaleInv(L21, ...) + RGBLoss(L21)`

Per-batch scalar metric keys tracked during test:

- `loss`: total test loss from the summed test criterion.
- `Regr3DPose_self_pts3d/<i>`
- `Regr3DPose_pts3d/<i>`
- `Regr3DPose_ScaleInv_self_pts3d/<i>`
- `Regr3DPose_ScaleInv_pts3d/<i>`
- `RGBLoss_rgb/<i>`
- `pose_loss`

Notes:

- `pose_loss` is a shared key across summed pose losses; last contributing criterion with that key wins when details are merged.
- With your current config, `i` appears as `1..24`.

## 4) Test Aggregation Metrics

At the end of each test epoch, each tracked test key `k` is aggregated into:

- `k_avg`: global average over the test epoch.
- `k_med`: median over the test epoch.

So for example:

- `loss_avg`, `loss_med`
- `pose_loss_avg`, `pose_loss_med`
- `RGBLoss_rgb/1_avg`, `RGBLoss_rgb/1_med`
- `Regr3DPose_pts3d/7_avg`, `Regr3DPose_pts3d/7_med`
- `Regr3DPose_ScaleInv_self_pts3d/24_avg`, `Regr3DPose_ScaleInv_self_pts3d/24_med`

## 5) Where Names Appear

- TensorBoard train scalars: prefixed as `train_<metric_key>` for criterion-derived training keys, plus `train_loss`, `train_lr`, `train_iter`.
- TensorBoard test scalars: `<test_prefix>_<aggregated_metric_name>` (for example `32 @ DL3DV_Multi_loss_avg`).
- `log.txt`: training stats are logged as `train_<k>`, test stats as `<test_prefix>_<k>`.

