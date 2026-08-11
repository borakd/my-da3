# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions for CroCo
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
import math
import json
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

import torch
import torch.distributed as dist
from torch import inf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import gather_object

printer = get_logger(__name__, log_level="DEBUG")


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values."""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, accelerator: Accelerator):
        """Synchronize the count and total across all processes."""
        if accelerator.num_processes == 1:
            return
        t = torch.tensor(
            [self.count, self.total], dtype=torch.float64, device=accelerator.device
        )
        accelerator.wait_for_everyone()
        accelerator.reduce(t, reduction="sum")
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                if v.ndim > 0:
                    continue
                v = v.item()
            if isinstance(v, list):
                continue
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self, accelerator):
        """Collective-safe sync of all meters across processes.

        The previous implementation issued one all-reduce per meter while
        iterating each rank's *own* ``self.meters`` dict. When ranks hold
        different key sets -- e.g. variable ``num_views`` produces a different
        number of per-view loss keys (``conf_loss/i``, ``pts3d/i``, ...) on each
        rank -- the number of collectives differs across ranks and NCCL
        dead-locks (watchdog timeout on the ``Numel=2`` all-reduce).

        Instead we agree on the rank-union of meter keys, pack the
        ``(count, total)`` of every key into a single vector (zero-filled for
        keys a rank is missing), and run ONE all-reduce. The summed
        ``count``/``total`` -> ``global_avg`` is exactly what a non-divergent run
        would log; logging-only, no effect on training. Single-process is a
        no-op, matching the original.
        """
        if accelerator.num_processes == 1:
            return
        # 1. Deterministic, rank-identical key ordering (union across all ranks).
        gathered_keys = gather_object(list(self.meters.keys()))
        union = sorted({k for rank_keys in gathered_keys for k in rank_keys})
        if not union:
            return
        # 2. Pack [count_k, total_k, ...] over the union; 0 where this rank lacks k.
        buf = torch.zeros(
            2 * len(union), dtype=torch.float64, device=accelerator.device
        )
        for i, k in enumerate(union):
            meter = self.meters.get(k)
            if meter is not None:
                buf[2 * i] = meter.count
                buf[2 * i + 1] = meter.total
        # 3. ONE all-reduce for every meter at once (identical shape on all ranks).
        accelerator.wait_for_everyone()
        accelerator.reduce(buf, reduction="sum")
        # 4. Write the global totals back into the meters this rank actually has
        #    (don't materialise empty meters -> keeps median/window stats valid).
        vals = buf.tolist()
        for i, k in enumerate(union):
            if k in self.meters:
                self.meters[k].count = int(vals[2 * i])
                self.meters[k].total = vals[2 * i + 1]

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(
        self, iterable, print_freq, accelerator: Accelerator, header=None, max_iter=None
    ):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        len_iterable = min(len(iterable), max_iter) if max_iter else len(iterable)
        space_fmt = ":" + str(len(str(len_iterable))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for it, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len_iterable - 1:
                eta_seconds = iter_time.global_avg * (len_iterable - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    if accelerator.is_main_process:
                        printer.info(
                            log_msg.format(
                                i,
                                len_iterable,
                                eta=eta_string,
                                meters=str(self),
                                time=str(iter_time),
                                data=str(data_time),
                                memory=torch.cuda.max_memory_allocated() / MB,
                            )
                        )
                else:
                    if accelerator.is_main_process:
                        printer.info(
                            log_msg.format(
                                i,
                                len_iterable,
                                eta=eta_string,
                                meters=str(self),
                                time=str(iter_time),
                                data=str(data_time),
                            )
                        )
            i += 1
            end = time.time()
            if max_iter and it >= max_iter:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if accelerator.is_main_process:
            printer.info(
                "{} Total time: {} ({:.4f} s / it)".format(
                    header, total_time_str, total_time / len_iterable
                )
            )


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process(accelerator: Accelerator):
    return accelerator.is_main_process


def save_on_master(accelerator: Accelerator, *args, **kwargs):
    if is_main_process(accelerator):
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    nodist = args.nodist if hasattr(args, "nodist") else False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not nodist:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def all_reduce_grads_(parameters, bucket_bytes=128 * 1024 * 1024):
    """Average gradients across ranks, in place — the all-reduce DDP would have done.

    Needed only on the TBPTT path (dust3r/inference.py:loss_of_one_batch_tbptt),
    which drives the rollout through accelerator.unwrap_model(model) because the
    stepwise API (_forward_encoder / _forward_decoder_group_step) consists of
    custom methods and DistributedDataParallel only intercepts forward(). With
    DDP.forward() never called, its Reducer is never armed by
    prepare_for_backward(), the per-parameter autograd hooks return early, and
    NO gradient synchronization happens: every rank steps on its own local
    gradient and the replicas diverge from optimizer step 1 onward. Only rank
    0's weights are ever saved, so a 32-GPU run silently optimizes at an
    effective batch of one rank's worth of sequences.

    The non-TBPTT path (loss_of_one_batch) calls model(batch) on the WRAPPED
    module and syncs correctly — it must not use this function, or gradients
    would be averaged twice.

    Call AFTER backward and BEFORE grad clipping, so the clip sees the true
    global norm (DDP's own ordering). Gradients are unscaled fp32 here
    (mixed_precision="bf16" => accelerator.scaler is None), so no unscale dance
    is required.

    Bucketed to keep the flatten buffer bounded: a flat copy of every gradient
    in this model would be several GB.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    world_size = dist.get_world_size()
    if world_size < 2:
        return 0

    # EVERY rank must issue the SAME sequence of collectives or the job
    # deadlocks. Two things guarantee that here, and both are load-bearing:
    #
    #  1. Materialize zeros for parameters that got no gradient on THIS rank.
    #     The graded chunks are data-dependent (that is why the Accelerator is
    #     built with find_unused_parameters=True, train_cut3r_baseline.py:139),
    #     so rank A can have a gradient where rank B has None. Filtering on
    #     `p.grad is not None` would then bucket differently per rank and
    #     mismatch the all_reduce. Zero-filling also matches DDP's own
    #     find_unused_parameters semantics: unused params get a zero gradient
    #     and are still touched by weight decay.
    #  2. Walk parameters in their fixed registration order and start a new
    #     bucket whenever dtype/device changes. Grouping via a set of
    #     (dtype, device) keys would iterate in hash order, which is not
    #     guaranteed identical across processes.
    params = [p for p in parameters if p.requires_grad]
    if not params:
        return 0
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    scale = 1.0 / world_size
    n_synced = 0

    def _flush(bucket):
        if not bucket:
            return
        flat = torch._utils._flatten_dense_tensors(bucket)
        dist.all_reduce(flat)
        flat.mul_(scale)
        for g, synced in zip(bucket, torch._utils._unflatten_dense_tensors(flat, bucket)):
            g.copy_(synced)

    bucket, bucket_numel, bucket_key = [], 0, None
    for p in params:
        g = p.grad
        key = (g.dtype, g.device)
        max_numel = max(1, bucket_bytes // g.element_size())
        if bucket and (key != bucket_key or bucket_numel + g.numel() > max_numel):
            _flush(bucket)
            bucket, bucket_numel = [], 0
        bucket.append(g)
        bucket_key = key
        bucket_numel += g.numel()
        n_synced += 1
    _flush(bucket)
    return n_synced


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True, accelerator: Accelerator = None, ddp_grad_sync=False):
        self.accelerator = accelerator
        # ddp_grad_sync (config key of the same name): restore the cross-rank
        # gradient averaging that the unwrapped TBPTT forward bypasses. OFF by
        # default so every existing config reproduces the previous baseline
        # byte-for-byte; the new arm turns it on. See all_reduce_grads_.
        self.ddp_grad_sync = bool(ddp_grad_sync)

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self.accelerator.backward(
            loss, create_graph=create_graph
        )  # .backward(create_graph=create_graph)
        if update_grad:
            if self.ddp_grad_sync:
                # Before the clip, so clip_grad_norm_ sees the synced norm.
                assert parameters is not None, "ddp_grad_sync needs the parameter list"
                parameters = list(parameters)
                all_reduce_grads_(parameters)
            if clip_grad is not None:
                assert parameters is not None
                # self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = self.accelerator.clip_grad_norm_(parameters, clip_grad)
            else:
                if self.accelerator.scaler is not None:
                    self.accelerator.unscale_gradients()
                norm = get_grad_norm_(parameters)
            optimizer.step()
        else:
            norm = None
        return norm

    def state_dict(self):
        if self.accelerator.scaler is not None:
            return self.accelerator.scaler.state_dict()
        else:
            return {}

    def load_state_dict(self, state_dict):
        if self.accelerator.scaler is not None:
            self.accelerator.scaler.load_state_dict(state_dict)


# class NativeScalerWithGradNormCount:
#     state_dict_key = "amp_scaler"

#     def __init__(self, enabled=True, accelerator:Accelerator=None):
#         self._scaler = torch.cuda.amp.GradScaler(enabled=enabled)
#         self.accelerator = accelerator

#     def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
#         # self.accelerator.backward(loss, create_graph=create_graph) #.backward(create_graph=create_graph)
#         self._scaler.scale(loss).backward(create_graph=create_graph)
#         if update_grad:
#             if clip_grad is not None:
#                 assert parameters is not None
#                 # #self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
#                 # norm = self.accelerator.clip_grad_norm_(parameters, clip_grad)
#                 self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
#                 norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
#             else:
#                 # if self.accelerator.scaler is not None:
#                 #     self.accelerator.unscale_gradients()
#                 # norm = get_grad_norm_(parameters)
#                 self._scaler.unscale_(optimizer)
#                 norm = get_grad_norm_(parameters)
#             # optimizer.step()
#             self._scaler.step(optimizer)
#             self._scaler.update()
#         else:
#             norm = None
#         return norm

#     # def state_dict(self):
#     #     if self.accelerator.scaler is not None:
#     #         return self.accelerator.scaler.state_dict()
#     #     else:
#     #         return {}

#     # def load_state_dict(self, state_dict):
#     #     if self.accelerator.scaler is not None:
#     #         self.accelerator.scaler.load_state_dict(state_dict)

#     def state_dict(self):
#         return self._scaler.state_dict()

#     def load_state_dict(self, state_dict):
#         self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def save_model(
    accelerator,
    args,
    epoch,
    model_without_ddp,
    optimizer,
    loss_scaler,
    fname=None,
    best_so_far=None,
):
    if accelerator.is_main_process:
        output_dir = Path(args.output_dir)
        if fname is None:
            fname = str(epoch)
        checkpoint_path = output_dir / ("checkpoint-%s.pth" % fname)
        to_save = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": loss_scaler.state_dict(),
            "args": args,
            "epoch": epoch,
        }
        if best_so_far is not None:
            to_save["best_so_far"] = best_so_far
        print(f">> Saving model to {checkpoint_path} ...")
        save_on_master(accelerator, to_save, checkpoint_path)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    args.start_epoch = 0
    best_so_far = None
    if args.resume is not None:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        printer.info("Resume checkpoint %s" % args.resume)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        args.start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        if "best_so_far" in checkpoint:
            best_so_far = checkpoint["best_so_far"]
            printer.info(" & best_so_far={:g}".format(best_so_far))
        else:
            printer.info("")
        printer.info("With optim & sched! start_epoch={:d}".format(args.start_epoch))
    return best_so_far


def all_reduce_mean(x, accelerator):
    """Use accelerator to all-reduce and compute mean."""
    if accelerator.state.num_processes > 1:
        x_reduce = torch.tensor(x).cuda()
        accelerator.reduce(x_reduce, reduce_op="SUM")
        x_reduce /= accelerator.state.num_processes
        return x_reduce.item()
    else:
        return x


def _replace(text, src, tgt, rm=""):
    """Advanced string replacement.
    Given a text:
    - replace all elements in src by the corresponding element in tgt
    - remove all elements in rm
    """
    if len(tgt) == 1:
        tgt = tgt * len(src)
    assert len(src) == len(tgt), f"'{src}' and '{tgt}' should have the same len"
    for s, t in zip(src, tgt):
        text = text.replace(s, t)
    for c in rm:
        text = text.replace(c, "")
    return text


def filename(obj):
    """transform a python obj or cmd into a proper filename.
    - \1 gets replaced by slash '/'
    - \2 gets replaced by comma ','
    """
    if not isinstance(obj, str):
        obj = repr(obj)
    obj = str(obj).replace("()", "")
    obj = _replace(obj, "_,(*/\1\2", "-__x%/,", rm=" )'\"")
    assert all(len(s) < 256 for s in obj.split(os.sep)), (
        "filename too long (>256 characters):\n" + obj
    )
    return obj


def _get_num_layer_for_vit(var_name, enc_depth, dec_depth):
    if var_name in ("cls_token", "mask_token", "pos_embed", "global_tokens"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("enc_blocks"):
        layer_id = int(var_name.split(".")[1])
        return layer_id + 1
    elif var_name.startswith("decoder_embed") or var_name.startswith(
        "enc_norm"
    ):  # part of the last black
        return enc_depth
    elif var_name.startswith("dec_blocks"):
        layer_id = int(var_name.split(".")[1])
        return enc_depth + layer_id + 1
    elif var_name.startswith("dec_norm"):  # part of the last block
        return enc_depth + dec_depth
    elif any(var_name.startswith(k) for k in ["head", "prediction_head"]):
        return enc_depth + dec_depth + 1
    else:
        raise NotImplementedError(var_name)


def get_parameter_groups(
    model, weight_decay, layer_decay=1.0, skip_list=(), no_lr_scale_list=[]
):
    parameter_group_names = {}
    parameter_group_vars = {}
    enc_depth, dec_depth = None, None
    # prepare layer decay values
    assert layer_decay == 1.0 or 0.0 < layer_decay < 1.0
    if layer_decay < 1.0:
        enc_depth = model.enc_depth
        dec_depth = model.dec_depth if hasattr(model, "dec_blocks") else 0
        num_layers = enc_depth + dec_depth
        layer_decay_values = list(
            layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
        )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        # Assign weight decay values
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            if "enc_blocks" in name:
                group_name = "no_decay_enc_blocks"
            else:
                group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            if "enc_blocks" in name:
                group_name = "decay_enc_blocks"
            else:
                group_name = "decay"
            this_weight_decay = weight_decay

        # Assign layer ID for LR scaling
        if layer_decay < 1.0:
            skip_scale = False
            layer_id = _get_num_layer_for_vit(name, enc_depth, dec_depth)
            group_name = "layer_%d_%s" % (layer_id, group_name)
            if name in no_lr_scale_list:
                skip_scale = True
                group_name = f"{group_name}_no_lr_scale"
        else:
            layer_id = 0
            skip_scale = True

        if group_name not in parameter_group_names:
            if not skip_scale:
                scale = layer_decay_values[layer_id]
            else:
                scale = 1.0

            if "enc_blocks" in group_name:
                scale *= 1.0
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    printer.info("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""

    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        # lr = args.lr
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args.warmup_epochs)
                / (args.epochs - args.warmup_epochs)
            )
        )

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr
