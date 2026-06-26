# CUT3R Loss Definitions

This note documents the loss definitions used by the training scripts:

- `train.py`
- `train_mine.py`
- `train_cut3r_baseline.py`

For the relevant configs, the criterion definitions are:

```text
train_criterion:
ConfLoss(Regr3DPoseBatchList(L21, norm_mode='?avg_dis'), alpha=0.2) + RGBLoss(MSE)

test_criterion:
Regr3DPose(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)
+ Regr3DPose_ScaleInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)
+ RGBLoss(L21)
```

Notation:

- `V`: number of views.
- `b`: batch item index.
- `v`: view index.
- `p`: valid pixel or point index.
- `M`: pose mask.
- `M_k`: valid point mask for a self-view or cross-view point-loss term.
- `t`: 3D translation part of the pose encoding.
- `q`: quaternion rotation part of the pose encoding.
- `X`: ground-truth 3D point.
- `\hat{X}`: predicted 3D point.
- `I`: ground-truth RGB image.
- `\hat{I}`: predicted RGB image.

## `train_pose_loss`

`train_pose_loss` is the scalar detail named `pose_loss` emitted by
`Regr3DPoseBatchList`.

Without a pose mask:

```math
\text{train\_pose\_loss}
=
\operatorname{mean}_{b,v}
\left[
\left\|\hat{t}_{b,v} - t_{b,v}\right\|_2
\right]
+
\operatorname{mean}_{b,v}
\left[
\left\|\hat{q}_{b,v} - q_{b,v}\right\|_2
\right]
```

With a pose mask:

```math
\text{train\_pose\_loss}
=
\operatorname{mean}_{b \in M, v}
\left\|\hat{t}_{b,v} - t_{b,v}\right\|_2
+
\operatorname{mean}_{b \in M, v}
\left\|\hat{q}_{b,v} - q_{b,v}\right\|_2
```

In the training implementation, `Regr3DPoseBatchList` applies an additional
image-mask factor to the pose mask before computing this value.

## `train_loss`

The training criterion is:

```text
ConfLoss(Regr3DPoseBatchList(L21, norm_mode='?avg_dis'), alpha=0.2) + RGBLoss(MSE)
```

Therefore:

```math
\text{train\_loss}
=
\text{Conf3DTrainLoss}
+
\text{train\_pose\_loss}
+
\text{RGBMSELoss}
```

The confidence-weighted 3D term is:

```math
\text{Conf3DTrainLoss}
=
2 \cdot
\operatorname{mean}_{k}
\left[
\operatorname{mean}_{p \in M_k}
\left(
e_{k,p} \cdot c_{k,p}
-
0.2 \log c_{k,p}
\right)
\right]
```

where `k` ranges over all self-view and cross-view point-loss terms, and:

```math
e_{k,p}
=
\left\|
\hat{X}_{k,p} - X_{k,p}
\right\|_2
```

For self-view terms, `c = conf_self`. For cross-view terms, `c = conf`.

The RGB term is:

```math
\text{RGBMSELoss}
=
\frac{1}{V}
\sum_{v=1}^{V}
\operatorname{mean}_{b,p,c}
\left[
\left(
\hat{I}_{b,v,p,c} - I_{b,v,p,c}
\right)^2
\right]
```

## `test_set_size @ DL3DV_multi_pose_loss_avg`

This metric is the test-set global average of the batch-level `pose_loss`
detail. It is not added into the test scalar loss.

```math
\text{DL3DV\_multi\_pose\_loss\_avg}
=
\frac{1}{N_\text{test batches}}
\sum_i
\text{pose\_loss}^{(i)}
```

For each test batch:

```math
\text{pose\_loss}^{(i)}
=
\operatorname{mean}_{b,v}
\left\|\hat{t}_{b,v} - t_{b,v}\right\|_2
+
\operatorname{mean}_{b,v}
\left\|\hat{q}_{b,v} - q_{b,v}\right\|_2
```

The test criterion contains both `Regr3DPose` and `Regr3DPose_ScaleInv`.
Both emit a detail named `pose_loss`; because the loss details are merged with
Python dict union, the later `Regr3DPose_ScaleInv` value overwrites the earlier
one. `Regr3DPose_ScaleInv` changes the point-cloud scale handling, but it does
not change the pose-loss formula above.

## `test_set_size @ DL3DV_multi_loss_avg`

The test criterion is:

```text
Regr3DPose(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)
+ Regr3DPose_ScaleInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)
+ RGBLoss(L21)
```

Therefore, for each test batch:

```math
\text{test\_loss}
=
\text{Regr3DPoseL21}
+
\text{Regr3DPoseScaleInvL21}
+
\text{RGBL21Loss}
```

The normal 3D point term is:

```math
\text{Regr3DPoseL21}
=
\sum_{v=1}^{V}
\operatorname{mean}_{p \in M_v}
\left\|
\hat{X}^{self}_{v,p} - X^{self}_{v,p}
\right\|_2
+
\sum_{v=1}^{V}
\operatorname{mean}_{p \in M_v}
\left\|
\hat{X}^{cross}_{v,p} - X^{cross}_{v,p}
\right\|_2
```

The scale-invariant 3D point term has the same form, but with the predicted
point-cloud scale aligned to the ground-truth point-cloud scale:

```math
\text{Regr3DPoseScaleInvL21}
=
\sum_{v=1}^{V}
\operatorname{mean}_{p \in M_v}
\left\|
\tilde{X}^{self}_{v,p} - X^{self}_{v,p}
\right\|_2
+
\sum_{v=1}^{V}
\operatorname{mean}_{p \in M_v}
\left\|
\tilde{X}^{cross}_{v,p} - X^{cross}_{v,p}
\right\|_2
```

The test RGB term uses `L21`, not MSE:

```math
\text{RGBL21Loss}
=
\frac{1}{V}
\sum_{v=1}^{V}
\operatorname{mean}_{b,p}
\left\|
\hat{I}_{b,v,p} - I_{b,v,p}
\right\|_2
```

The logged average is:

```math
\text{DL3DV\_multi\_loss\_avg}
=
\frac{1}{N_\text{test batches}}
\sum_i
\text{test\_loss}^{(i)}
```

## Key Distinction

`train_loss` includes `train_pose_loss`, because `ConfLoss` explicitly adds
`details["pose_loss"]` to its final scalar.

`test_set_size @ DL3DV_multi_loss_avg` does not include
`test_set_size @ DL3DV_multi_pose_loss_avg`, because the test criterion is not
wrapped in `ConfLoss`. The test pose loss is logged as an auxiliary metric.
