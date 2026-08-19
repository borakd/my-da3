import tqdm
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf
from dust3r.model import ARCroco3DStereo
from accelerate import Accelerator
import os
import re

# REVISIT-TRAIN sampler RNG: dedicated generator so the global torch stream
# stays byte-identical to the ungated recipe (data aug / sampler order).
_REVISIT_TRAIN_GEN = None


def custom_sort_key(key):
    text = key.split("/")
    if len(text) > 1:
        text, num = text[0], text[-1]
        return (text, int(num))
    else:
        return (key, -1)


def merge_chunk_dict(old_dict, curr_dict, add_number):
    new_dict = {}
    for key, value in curr_dict.items():

        match = re.search(r"(\d+)$", key)
        if match:

            num_part = int(match.group()) + add_number

            new_key = re.sub(r"(\d+)$", str(num_part), key, 1)
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    new_dict = old_dict | new_dict
    return {k: new_dict[k] for k in sorted(new_dict.keys(), key=custom_sort_key)}


def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2


def loss_of_one_batch(
    batch,
    model,
    criterion,
    accelerator: Accelerator,
    symmetrize_batch=False,
    use_amp=False,
    ret=None,
    img_mask=None,
    inference=False,
):
    if len(batch) > 2:
        assert (
            symmetrize_batch is False
        ), "cannot symmetrize batch with more than 2 views"
    if symmetrize_batch:
        batch = make_batch_symmetric(batch)

    # print("********** DEBUG PRINT 5 **********")
    # print(f"Size of batch (should be >1 for multiview): {len(batch)}")

    with torch.cuda.amp.autocast(enabled=not inference):
        if inference:
            output, state_args = model(batch, ret_state=True)
            preds, batch = output.ress, output.views
            result = dict(views=batch, pred=preds)
            print(f"DEBUG got outputs from loss_of_one_batch")
            return result[ret] if ret else result, state_args
        else:
            output = model(batch)
            preds, batch = output.ress, output.views

        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(batch, preds) if criterion is not None else None

    result = dict(views=batch, pred=preds, loss=loss)
    return result[ret] if ret else result


def loss_of_one_batch_tbptt(
    batch,
    model,
    criterion,
    chunk_size,
    loss_scaler,
    optimizer,
    accelerator: Accelerator,
    log_writer=None,
    symmetrize_batch=False,
    use_amp=False,
    ret=None,
    img_mask=None,
    inference=False,
):
    if len(batch) > 2:
        assert (
            symmetrize_batch is False
        ), "cannot symmetrize batch with more than 2 views"
    if symmetrize_batch:
        batch = make_batch_symmetric(batch)
    all_preds = []
    all_loss = 0.0
    all_grad_norms = []
    all_loss_details = {}
    base_model = accelerator.unwrap_model(model)
    views_per_step = max(1, int(getattr(base_model, "views_per_step", 1)))
    chunk_groups = max(1, (chunk_size + views_per_step - 1) // views_per_step)
    with torch.cuda.amp.autocast(enabled=not inference):
        with torch.no_grad():
            (feat, pos, shape), (
                init_state_feat,
                init_mem,
                state_feat,
                state_pos,
                mem,
            ) = base_model._forward_encoder(batch)
        feat = [f.detach() for f in feat]
        pos = [p.detach() for p in pos]
        shape = [s.detach() for s in shape]
        init_state_feat = init_state_feat.detach()
        init_mem = init_mem.detach()
        # REVISIT-TRAIN: append K update=False copies of uniformly sampled
        # earlier frames so hindsight decoding against the frozen whole-scene
        # state (eval REVISIT=1) is supervised in-distribution. Appended at
        # the END so the existing last-4-chunks grad rule lands 2 forward +
        # 2 revisit grad chunks at K=8, keeping exactly 4 optimizer steps.
        # Encoder feats are duplicated BY INDEX (free, already detached);
        # copies share GT tensor storage (read-only on this path).
        revisit_k = int(os.environ.get("REVISIT_TRAIN_K", "0") or 0)
        if revisit_k > 0:
            global _REVISIT_TRAIN_GEN
            if _REVISIT_TRAIN_GEN is None:
                _REVISIT_TRAIN_GEN = torch.Generator()
                _REVISIT_TRAIN_GEN.manual_seed(torch.initial_seed() + 0x5EED)
                print(f"[REVISIT_TRAIN] ACTIVE K={revisit_k} "
                      f"(seed base {torch.initial_seed()})", flush=True)
            assert views_per_step == 1, \
                "revisit-train designed for views_per_step=1"
            # The loader emits VARIABLE view counts (known TBPTT quirk), so no
            # divisibility is assumed: a chunk straddling the forward/revisit
            # boundary is fine (views are processed singly; forward views
            # write state, copies never do). randperm[:K] self-caps K on
            # short batches. K<=3*chunk_size keeps >=1 forward grad chunk on
            # full-length (64-view) batches.
            assert revisit_k <= 3 * chunk_size, (
                f"REVISIT_TRAIN_K={revisit_k} > {3*chunk_size}: would leave "
                f"no forward grad chunk on full-length batches")
            n_fwd = len(batch)
            src_idxs = sorted(torch.randperm(
                n_fwd, generator=_REVISIT_TRAIN_GEN)[:revisit_k].tolist())
            _B = batch[0]["img_mask"].shape[0]
            _dev = batch[0]["img_mask"].device
            batch = list(batch)  # rebind, never mutate the caller's list
            _upd_false = torch.zeros(_B, dtype=torch.bool, device=_dev)
            _rst_false = torch.zeros(_B, dtype=torch.bool, device=_dev)
            for _src in src_idxs:
                _nv = dict(batch[_src])  # shallow: GT tensors shared
                _nv["update"] = _upd_false
                _nv["reset"] = _rst_false
                batch.append(_nv)
                feat.append(feat[_src])
                pos.append(pos[_src])
                shape.append(shape[_src])
        group_ranges = base_model._group_view_ranges(len(batch))
        num_chunks = (len(group_ranges) - 1) // chunk_groups + 1
        seen_views = 0

        for chunk_id in range(num_chunks):
            preds = []
            chunk = []
            state_feat = state_feat.detach()
            state_pos = state_pos.detach()
            mem = mem.detach()
            # PoseGRU within-chunk BPTT (pose_gru_bptt): the hidden's tape may
            # span the steps of ONE chunk only — truncate at the boundary,
            # exactly like state/mem above. Each of the last chunks runs its
            # own backward and frees its graph, so an undetached hidden
            # crossing here would crash the next chunk's backward. No-op under
            # the per-step detach (grad_fn already None) or without a GRU.
            # v3 levers add NO new boundary state: with iters>1 (R lever) the
            # stashed hidden is the view's LAST iterate — this detach is
            # N-agnostic; the _prev_img_feat stash (F lever) is always
            # detached data; gru_pose_iters is a plain res tensor that the
            # all_preds collection below blanket-detaches like everything else.
            _gru_hidden = getattr(base_model, "_pose_gru_hidden", None)
            if _gru_hidden is not None:
                base_model._pose_gru_hidden = _gru_hidden.detach()
            start_group = chunk_id * chunk_groups
            end_group = min(start_group + chunk_groups, len(group_ranges))
            active_groups = group_ranges[start_group:end_group]
            if chunk_id < num_chunks - 4:
                with torch.no_grad():
                    for group_start, group_end in active_groups:
                        view_indices = list(range(group_start, group_end))
                        res_group, (state_feat, mem) = base_model._forward_decoder_group_step(
                            views=batch,
                            view_indices=view_indices,
                            feat_group=[feat[i] for i in view_indices],
                            pos_group=[pos[i] for i in view_indices],
                            shape_group=[shape[i] for i in view_indices],
                            init_state_feat=init_state_feat,
                            init_mem=init_mem,
                            state_feat=state_feat,
                            state_pos=state_pos,
                            mem=mem,
                        )
                        for local_idx, view_idx in enumerate(view_indices):
                            res = res_group[local_idx]
                            preds.append(res)
                            all_preds.append({k: v.detach() for k, v in res.items()})
                            chunk.append(batch[view_idx])
                with torch.cuda.amp.autocast(enabled=False):
                    loss, loss_details = (
                        criterion(chunk, preds, camera1=batch[0]["camera_pose"])
                        if criterion is not None
                        else None
                    )
                    all_loss += float(loss)
                    all_loss_details = merge_chunk_dict(
                        all_loss_details, loss_details, seen_views
                    )
                    seen_views += len(chunk)
                    del loss
            else:
                for group_start, group_end in active_groups:
                    view_indices = list(range(group_start, group_end))
                    res_group, (state_feat, mem) = base_model._forward_decoder_group_step(
                        views=batch,
                        view_indices=view_indices,
                        feat_group=[feat[i] for i in view_indices],
                        pos_group=[pos[i] for i in view_indices],
                        shape_group=[shape[i] for i in view_indices],
                        init_state_feat=init_state_feat,
                        init_mem=init_mem,
                        state_feat=state_feat,
                        state_pos=state_pos,
                        mem=mem,
                    )
                    for local_idx, view_idx in enumerate(view_indices):
                        res = res_group[local_idx]
                        preds.append(res)
                        all_preds.append({k: v.detach() for k, v in res.items()})
                        chunk.append(batch[view_idx])
                with torch.cuda.amp.autocast(enabled=False):
                    loss, loss_details = (
                        criterion(chunk, preds, camera1=batch[0]["camera_pose"])
                        if criterion is not None
                        else None
                    )
                    all_loss += float(loss)
                    all_loss_details = merge_chunk_dict(
                        all_loss_details, loss_details, seen_views
                    )
                    seen_views += len(chunk)
                    norm = loss_scaler(
                        loss,
                        optimizer,
                        parameters=model.parameters(),
                        update_grad=True,
                        clip_grad=1.0,
                    )
                    if norm is not None:
                        all_grad_norms.append(float(norm))
                    optimizer.zero_grad()
                    del loss
    result = dict(
        views=batch,
        pred=all_preds,
        loss=(all_loss / num_chunks, all_loss_details),
        # pre-clip gradient norm, max over this batch's TBPTT chunks
        grad_norm=(max(all_grad_norms) if all_grad_norms else None),
        already_backprop=True,
    )
    return result[ret] if ret else result


@torch.no_grad()
def inference(groups, model, device, verbose=True):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for view in groups:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            if isinstance(view[name], tuple) or isinstance(view[name], list):
                view[name] = [x.to(device, non_blocking=True) for x in view[name]]
            else:
                view[name] = view[name].to(device, non_blocking=True)

    if verbose:
        print(f">> Inference with model on {len(groups)} image/raymaps")

    res, state_args = loss_of_one_batch(groups, model, None, None, inference=True)
    result = to_cpu(res)
    return result, state_args


@torch.no_grad()
def inference_step(view, state_args, model, device, verbose=True):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for name in view.keys():  # pseudo_focal
        if name in ignore_keys:
            continue
        if isinstance(view[name], tuple) or isinstance(view[name], list):
            view[name] = [x.to(device, non_blocking=True) for x in view[name]]
        else:
            view[name] = view[name].to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=False):
        state_feat, state_pos, init_state_feat, mem, init_mem = state_args
        pred, _ = model.inference_step(
            view, state_feat, state_pos, init_state_feat, mem, init_mem
        )

    res = dict(pred=pred)
    result = to_cpu(res)
    return result


@torch.no_grad()
def inference_recurrent(groups, model, device, verbose=True):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for view in groups:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            if isinstance(view[name], tuple) or isinstance(view[name], list):
                view[name] = [x.to(device, non_blocking=True) for x in view[name]]
            else:
                view[name] = view[name].to(device, non_blocking=True)

    if verbose:
        print(f">> Inference with model on {len(groups)} image/raymaps")

    with torch.cuda.amp.autocast(enabled=False):
        preds, batch, state_args = model.forward_recurrent(
            groups, device, ret_state=True
        )
        res = dict(views=batch, pred=preds)
    result = to_cpu(res)
    return result, state_args


def check_if_same_size(pairs):
    shapes1 = [img1["img"].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2["img"].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(
        shapes2[0] == s for s in shapes2
    )


def get_pred_pts3d(gt, pred, use_pose=False, inplace=False):
    if "depth" in pred and "pseudo_focal" in pred:
        try:
            pp = gt["camera_intrinsics"][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif "pts3d" in pred:

        pts3d = pred["pts3d"]

    elif "pts3d_in_other_view" in pred:

        assert use_pose is True
        return (
            pred["pts3d_in_other_view"]
            if inplace
            else pred["pts3d_in_other_view"].clone()
        )

    if use_pose:
        camera_pose = pred.get("camera_pose")
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(
    gt_pts1,
    gt_pts2,
    pr_pts1,
    pr_pts2=None,
    fit_mode="weiszfeld_stop_grad",
    valid1=None,
    valid2=None,
):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = (
        invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None
    )

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = (
        invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None
    )

    all_gt = (
        torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1)
        if gt_pts2 is not None
        else nan_gt_pts1
    )
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith("avg"):

        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith("median"):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith("weiszfeld"):

        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)

        for iter in range(10):

            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)

            w = dis.clip_(min=1e-8).reciprocal()

            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f"bad {fit_mode=}")

    if fit_mode.endswith("stop_grad"):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)

    return scaling
