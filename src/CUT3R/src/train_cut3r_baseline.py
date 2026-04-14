# --------------------------------------------------------
# training code for CUT3R
# --------------------------------------------------------
# References:
# DUSt3R: https://github.com/naver/dust3r
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
import cv2
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import (
    PreTrainedModel,
    ARCroco3DStereo,
    ARCroco3DStereoConfig,
    inf,
    strip_module,
)  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch, loss_of_one_batch_tbptt  # noqa
from dust3r.viz import colorize
from dust3r.utils.geometry import inv, geotrf
from dust3r.utils.camera import pose_encoding_to_camera
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

import hydra
from omegaconf import OmegaConf
import logging
import pathlib
from tqdm import tqdm
import random
import builtins
import shutil

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import send_to_device
from datetime import timedelta
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")

try:
    import wandb
except Exception:
    print("Wandb not found")
    wandb = None


def setup_for_distributed(accelerator: Accelerator):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (accelerator.num_processes > 8)
        if accelerator.is_main_process or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def save_current_code(outdir):
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H-%M-%S")
    src_dir = "."
    dst_dir = os.path.join(outdir, "code", "{}".format(date_time))
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns(
            ".vscode*",
            "assets*",
            "example*",
            "checkpoints*",
            "OLD*",
            "logs*",
            "out*",
            "runs*",
            "*.png",
            "*.mp4",
            "*__pycache__*",
            "*.git*",
            "*.idea*",
            "*.zip",
            "*.jpg",
        ),
        dirs_exist_ok=True,
    )
    return dst_dir


def train(args):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_iter,
        mixed_precision="bf16",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(seconds=6000)),
        ],
    )
    device = accelerator.device

    setup_for_distributed(accelerator)

    printer.info("output_dir: " + args.output_dir)
    if args.output_dir:
        printer.info(f"Creating output_dir (mkdir -p): {args.output_dir}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        printer.info(f"output_dir ready: {args.output_dir}")

    save_code_snapshot = bool(getattr(args, "save_code_snapshot", False))
    printer.info(f"Code snapshot block reached; save_code_snapshot={save_code_snapshot}")
    if accelerator.is_main_process and save_code_snapshot:
        printer.info("Saving current code snapshot...")
        snapshot_start = time.time()
        dst_dir = save_current_code(outdir=args.output_dir)
        snapshot_sec = time.time() - snapshot_start
        printer.info(
            f"Saved current code to {dst_dir} in {snapshot_sec:.1f}s"
        )
    elif accelerator.is_main_process:
        printer.info(
            "Skipping code snapshot (set save_code_snapshot=true to enable)"
        )

    use_wandb = bool(getattr(args, "use_wandb", True))
    if accelerator.is_main_process and wandb is not None and wandb.run is None and use_wandb:
        # Added a default project name so it doesn't silently fail to log
        wandb_project = os.environ.get("WANDB_PROJECT", "da3-with-cut3r-training-dist")
        wandb_id_path = os.path.join(args.output_dir, "wandb_run_id.txt")
        tb_dir = os.path.join(args.output_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        if os.path.isfile(wandb_id_path):
            with open(wandb_id_path, "r", encoding="utf-8") as f:
                wandb_run_id = f.read().strip()
        else:
            wandb_run_id = wandb.util.generate_id()
            with open(wandb_id_path, "w", encoding="utf-8") as f:
                f.write(wandb_run_id)
        wandb.tensorboard.patch(root_logdir=tb_dir)
        printer.info("Initializing Weights & Biases...")
        wandb_start = time.time()
        wandb.init(
            project=wandb_project,
            sync_tensorboard=True,
            name=os.environ.get(
                "WANDB_NAME", os.path.basename(args.output_dir.rstrip("/"))
            ),
            dir=args.output_dir,
            id=wandb_run_id,
            resume=os.environ.get("WANDB_RESUME", "allow"),
            # Converted OmegaConf DictConfig to a standard Python dictionary
            config=OmegaConf.to_container(args, resolve=True),
        )
        printer.info(f"W&B initialized in {time.time() - wandb_start:.1f}s")

    # auto resume
    if not args.resume:
        last_ckpt_fname = os.path.join(args.output_dir, f"checkpoint-last.pth")
        args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    printer.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

    # fix the seed
    seed = args.seed + accelerator.state.process_index
    printer.info(
        f"Setting seed to {seed} for process {accelerator.state.process_index}"
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = args.benchmark

    # training dataset and loader
    printer.info("Building train dataset %s", args.train_dataset)
    #  dataset and loader
    data_loader_train = build_dataset(
        args.train_dataset,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        test=False,
        fixed_length=args.fixed_length
    )
    printer.info("Building test dataset %s", args.test_dataset)
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset,
            args.batch_size,
            args.num_workers,
            accelerator=accelerator,
            test=True,
            fixed_length=True
        )
        for dataset in args.test_dataset.split("+")
    }

    # model
    printer.info("Loading model: %s", args.model)
    model: PreTrainedModel = eval(args.model)
    printer.info(f"All model parameters: {sum(p.numel() for p in model.parameters())}")
    printer.info(
        f"Encoder parameters: {sum(p.numel() for p in model.enc_blocks.parameters())}"
    )
    printer.info(
        f"Decoder parameters: {sum(p.numel() for p in model.dec_blocks.parameters())}"
    )

    printer.info(f">> Creating train criterion = {args.train_criterion}")
    train_criterion = eval(args.train_criterion).to(device)
    printer.info(
        f">> Creating test criterion = {args.test_criterion or args.train_criterion}"
    )
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.long_context:
        model.fixed_input_length = False

    if args.pretrained and not args.resume:
        printer.info(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        load_only_encoder = getattr(args, "load_only_encoder", False)
        if load_only_encoder:
            filtered_state_dict = {
                k: v
                for k, v in ckpt["model"].items()
                if "enc_blocks" in k or "patch_embed" in k
            }
            printer.info(
                model.load_state_dict(strip_module(filtered_state_dict), strict=False)
            )
        else:
            printer.info(
                model.load_state_dict(strip_module(ckpt["model"]), strict=False)
            )
        del ckpt  # in case it occupies memory

    # # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler(accelerator=accelerator)

    accelerator.even_batches = False
    optimizer, model, data_loader_train = accelerator.prepare(
        optimizer, model, data_loader_train
    )
    # Add MVU to CUT3R
    base_model = accelerator.unwrap_model(model)
    base_model.views_per_step = int(getattr(args, "views_per_step", 1))
    printer.info(
        f"Configured grouped recurrence with views_per_step={base_model.views_per_step}"
    )

    def write_log_stats(epoch, train_stats, test_stats):
        if accelerator.is_main_process:
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    best_so_far = misc.load_model(
        args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler
    )
    if best_so_far is None:
        best_so_far = float("inf")
    tb_dir = os.path.join(args.output_dir, "tb")
    if accelerator.is_main_process:
        os.makedirs(tb_dir, exist_ok=True)
    log_writer = (
        SummaryWriter(log_dir=tb_dir, flush_secs=10, max_queue=10)
        if accelerator.is_main_process
        else None
    )

    printer.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}

    for epoch in range(args.start_epoch, args.epochs + 1):

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if (
                args.save_freq
                and np.allclose(epoch / args.save_freq, int(epoch / args.save_freq))
                or epoch == args.epochs
            ):
                save_model(epoch - 1, "last", best_so_far)

        # Test on multiple datasets
        new_best = False
        if epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(
                    model,
                    test_criterion,
                    testset,
                    accelerator,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                    prefix=test_name,
                )
                test_stats[test_name] = stats

                # Save best of all
                if stats["loss_med"] < best_so_far:
                    best_so_far = stats["loss_med"]
                    new_best = True
        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, "best", best_so_far)
        if epoch >= args.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            accelerator,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    printer.info("Training time {}".format(total_time_str))

    save_final_model(accelerator, args, args.epochs, model, best_so_far=best_so_far)

    if log_writer is not None:
        log_writer.flush()
        log_writer.close()

    if accelerator.is_main_process and wandb is not None and wandb.run is not None:
        wandb.finish()


def save_final_model(accelerator, args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": (
            model_without_ddp
            if isinstance(model_without_ddp, dict)
            else model_without_ddp.cpu().state_dict()
        ),
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    printer.info(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(accelerator, to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, accelerator, test=False, fixed_length=False):
    split = ["Train", "Test"][test]
    printer.info(f"Building {split} Data loader for dataset: {dataset}")
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not (test),
        drop_last=not (test),
        accelerator=accelerator,
        fixed_length=fixed_length
    )
    return loader


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    def save_model(epoch, fname, best_so_far):
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        with accelerator.accumulate(model):
            epoch_f = epoch + data_iter_step / len(data_loader)
            step = int(epoch_f * len(data_loader))
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                misc.adjust_learning_rate(optimizer, epoch_f, args)
            if not args.long_context:
                result = loss_of_one_batch(
                    batch,
                    model,
                    criterion,
                    accelerator,
                    symmetrize_batch=False,
                    use_amp=bool(args.amp),
                )
            else:
                result = loss_of_one_batch_tbptt(
                    batch,
                    model,
                    criterion,
                    chunk_size=4,
                    loss_scaler=loss_scaler,
                    optimizer=optimizer,
                    accelerator=accelerator,
                    symmetrize_batch=False,
                    use_amp=bool(args.amp),
                )
            loss, loss_details = result["loss"]  # criterion returns two values

            loss_value = float(loss)

            if not math.isfinite(loss_value):
                print(
                    f"Loss is {loss_value}, stopping training, loss details: {loss_details}"
                )
                sys.exit(1)
            if not result.get("already_backprop", False):
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=True,
                    clip_grad=1.0,
                )
                optimizer.zero_grad()

            is_metric = batch[0]["is_metric"]
            curr_num_view = len(batch)

            del loss
            tb_vis_img = (data_iter_step + 1) % accum_iter == 0 and (
                (step + 1) % (args.print_img_freq)
            ) == 0
            if not tb_vis_img:
                del batch
            else:
                torch.cuda.empty_cache()

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(epoch=epoch_f)
            metric_logger.update(lr=lr)
            metric_logger.update(step=step)

            metric_logger.update(loss=loss_value, **loss_details)

            if (data_iter_step + 1) % accum_iter == 0 and (
                (data_iter_step + 1) % (accum_iter * args.print_freq)
            ) == 0:
                loss_value_reduce = accelerator.gather(
                    torch.tensor(loss_value).to(accelerator.device)
                ).mean()  # MUST BE EXECUTED BY ALL NODES

                if log_writer is None:
                    continue
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int(epoch_f * 1000)
                log_writer.add_scalar("train_loss", loss_value_reduce, step)
                log_writer.add_scalar("train_lr", lr, step)
                log_writer.add_scalar("train_iter", epoch_1000x, step)
                for name, val in loss_details.items():
                    if isinstance(val, torch.Tensor):
                        if val.ndim > 0:
                            continue
                    if isinstance(val, dict):
                        continue
                    log_writer.add_scalar("train_" + name, val, step)

            if tb_vis_img:
                if log_writer is None:
                    continue
                with torch.no_grad():
                    depths_self, gt_depths_self = get_depth_results_direct(
                        batch, result["pred"], self_view=True
                    )
                    depths_cross, gt_depths_cross = get_depth_results_direct(
                        batch, result["pred"], self_view=False
                    )
                    for k in range(len(batch)):
                        loss_details[f"self_pred_depth_{k+1}"] = (
                            depths_self[k].detach().cpu()
                        )
                        loss_details[f"self_gt_depth_{k+1}"] = (
                            gt_depths_self[k].detach().cpu()
                        )
                        loss_details[f"pred_depth_{k+1}"] = (
                            depths_cross[k].detach().cpu()
                        )
                        loss_details[f"gt_depth_{k+1}"] = (
                            gt_depths_cross[k].detach().cpu()
                        )

                imgs_stacked_dict = get_vis_imgs_new(
                    loss_details, args.num_imgs_vis, curr_num_view, is_metric=is_metric
                )
                for name, imgs_stacked in imgs_stacked_dict.items():
                    log_writer.add_images(
                        "train" + "/" + name, imgs_stacked, step, dataformats="HWC"
                    )
                del batch

        if (
            data_iter_step % int(args.save_freq * len(data_loader)) == 0
            and data_iter_step != 0
            and data_iter_step != len(data_loader) - 1
        ):
            print("saving at step", data_iter_step)
            save_model(epoch - 1, "last", float("inf"))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    accelerator: Accelerator,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = "Test Epoch: [{}]".format(epoch)

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(0)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(0)

    def _compute_depth_absrel_a1(gts, preds, eps=1e-8):
        absrels = []
        a1s = []
        for gt, pred in zip(gts, preds):
            gt_depth = geotrf(inv(gt["camera_pose"]), gt["pts3d"])[..., -1]
            pr_depth = pred["pts3d_in_self_view"][..., -1]
            valid = gt.get(
                "valid_mask", torch.ones_like(gt_depth, dtype=torch.bool)
            ).bool()
            valid = valid & torch.isfinite(gt_depth) & torch.isfinite(pr_depth)
            valid = valid & (gt_depth > 0)

            for b in range(gt_depth.shape[0]):
                mask = valid[b]
                if mask.sum() == 0:
                    continue
                g = gt_depth[b][mask].double()
                p = pr_depth[b][mask].double()
                scale = torch.median(g) / (torch.median(p) + eps)
                p_aligned = p * scale
                abs_rel = torch.mean(torch.abs(g - p_aligned) / (g + eps))
                ratio = torch.maximum(
                    g / (p_aligned + eps), p_aligned / (g + eps)
                )
                a1 = torch.mean((ratio < 1.25).double())
                absrels.append(float(abs_rel))
                a1s.append(float(a1))

        if len(absrels) == 0:
            return {}
        return {"absrel": float(np.mean(absrels)), "a1": float(np.mean(a1s))}

    def _align_points_similarity(pr_pts, gt_pts, eps=1e-8):
        pr = pr_pts.double()
        gt = gt_pts.double()
        n = pr.shape[0]
        if n < 2:
            return None
        pr_mean = pr.mean(dim=0)
        gt_mean = gt.mean(dim=0)
        pr_c = pr - pr_mean
        gt_c = gt - gt_mean
        cov = (gt_c.T @ pr_c) / n
        U, S, Vh = torch.linalg.svd(cov)
        V = Vh.T
        d = torch.ones(3, dtype=pr.dtype, device=pr.device)
        if torch.det(U @ V.T) < 0:
            d[-1] = -1.0
        R = U @ torch.diag(d) @ V.T
        var_pr = (pr_c.square().sum()) / n
        if float(var_pr) < eps:
            return None
        scale = (S * d).sum() / var_pr
        t = gt_mean - scale * (R @ pr_mean)
        return scale, R, t

    def _pose_rot_err_deg(R_err):
        cos_theta = ((torch.trace(R_err) - 1.0) / 2.0).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.acos(cos_theta))

    def _compute_pose_metrics(gts, preds):
        pr_cams = [pose_encoding_to_camera(pred["camera_pose"]) for pred in preds]
        num_views = len(gts)
        batch_size = gts[0]["camera_pose"].shape[0]
        ate_list = []
        rpe_rot_list = []
        rpe_trans_list = []

        for b in range(batch_size):
            view_mask = torch.ones(num_views, dtype=torch.bool, device=device)
            for i in range(num_views):
                if "img_mask" in gts[i]:
                    view_mask[i] = bool(gts[i]["img_mask"][b].item())
            if int(view_mask.sum()) < 2:
                continue
            valid_ids = torch.where(view_mask)[0].tolist()

            gt_seq = torch.stack(
                [gts[i]["camera_pose"][b] for i in valid_ids], dim=0
            ).double()
            pr_seq = torch.stack([pr_cams[i][b] for i in valid_ids], dim=0).double()

            gt_seq = torch.linalg.inv(gt_seq[:1]) @ gt_seq
            pr_seq = torch.linalg.inv(pr_seq[:1]) @ pr_seq

            aligned = _align_points_similarity(pr_seq[:, :3, 3], gt_seq[:, :3, 3])
            if aligned is None:
                continue
            scale, R_align, t_align = aligned

            pr_aligned = pr_seq.clone()
            pr_aligned[:, :3, :3] = R_align @ pr_seq[:, :3, :3]
            pr_aligned[:, :3, 3] = (
                scale * (R_align @ pr_seq[:, :3, 3].T)
            ).T + t_align

            ate = torch.sqrt(
                torch.mean(torch.sum((pr_aligned[:, :3, 3] - gt_seq[:, :3, 3]) ** 2, dim=-1))
            )
            ate_list.append(float(ate))

            for i in range(pr_aligned.shape[0] - 1):
                gt_rel = torch.linalg.inv(gt_seq[i]) @ gt_seq[i + 1]
                pr_rel = torch.linalg.inv(pr_aligned[i]) @ pr_aligned[i + 1]
                err = torch.linalg.inv(gt_rel) @ pr_rel
                rpe_trans_list.append(float(torch.linalg.norm(err[:3, 3])))
                rpe_rot_list.append(float(_pose_rot_err_deg(err[:3, :3])))

        if len(ate_list) == 0:
            return {}

        out = {"ate": float(np.mean(ate_list))}
        out["rpe_rot"] = float(np.mean(rpe_rot_list)) if rpe_rot_list else 0.0
        out["rpe_trans"] = float(np.mean(rpe_trans_list)) if rpe_trans_list else 0.0
        return out

    for _, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        batch = send_to_device(batch, device)
        result = loss_of_one_batch(
            batch,
            model,
            criterion,
            accelerator,
            symmetrize_batch=False,
            use_amp=bool(args.amp),
        )

        loss_value, loss_details = result["loss"]  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)
        depth_metrics = _compute_depth_absrel_a1(batch, result["pred"])
        if depth_metrics:
            metric_logger.update(**depth_metrics)
        pose_metrics = _compute_pose_metrics(batch, result["pred"])
        if pose_metrics:
            metric_logger.update(**pose_metrics)

    printer.info("Averaged stats: %s", metric_logger)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    if log_writer is not None:
        for name, val in results.items():
            if isinstance(val, torch.Tensor):
                if val.ndim > 0:
                    continue
            if isinstance(val, dict):
                continue
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)

        depths_self, gt_depths_self = get_depth_results_direct(
            batch, result["pred"], self_view=True
        )
        depths_cross, gt_depths_cross = get_depth_results_direct(
            batch, result["pred"], self_view=False
        )
        for k in range(len(batch)):
            loss_details[f"self_pred_depth_{k+1}"] = depths_self[k].detach().cpu()
            loss_details[f"self_gt_depth_{k+1}"] = gt_depths_self[k].detach().cpu()
            loss_details[f"pred_depth_{k+1}"] = depths_cross[k].detach().cpu()
            loss_details[f"gt_depth_{k+1}"] = gt_depths_cross[k].detach().cpu()

        imgs_stacked_dict = get_vis_imgs_new(
            loss_details,
            args.num_imgs_vis,
            args.num_test_views,
            is_metric=batch[0]["is_metric"],
        )
        for name, imgs_stacked in imgs_stacked_dict.items():
            log_writer.add_images(
                prefix + "/" + name, imgs_stacked, 1000 * epoch, dataformats="HWC"
            )

    del loss_details, loss_value, batch
    torch.cuda.empty_cache()

    return results


def batch_append(original_list, new_list):
    for sublist, new_item in zip(original_list, new_list):
        sublist.append(new_item)
    return original_list


@torch.no_grad()
def get_depth_results_direct(gts, preds, self_view=False):
    """Return per-pixel depth (z) directly from point maps, without gsplat rendering."""
    depths = []
    gt_depths = []
    for i, (gt, pred) in enumerate(zip(gts, preds)):
        if self_view:
            gt_depth = geotrf(inv(gt["camera_pose"]), gt["pts3d"])[..., -1]
            pred_depth = pred["pts3d_in_self_view"][..., -1]
        else:
            gt_depth = geotrf(inv(gts[0]["camera_pose"]), gt["pts3d"])[..., -1]
            pred_depth = pred["pts3d_in_other_view"][..., -1]

        valid = gt.get("valid_mask", torch.ones_like(gt_depth, dtype=torch.bool)).bool()
        valid = valid & torch.isfinite(gt_depth) & torch.isfinite(pred_depth) & (gt_depth > 0)

        gt_depth = torch.where(valid, gt_depth, torch.zeros_like(gt_depth))
        pred_depth = torch.where(valid, pred_depth, torch.zeros_like(pred_depth))
        gt_depths.append(gt_depth)
        depths.append(pred_depth)

    return depths, gt_depths


def gen_mask_indicator(img_mask_list, ray_mask_list, num_views, h, w):
    output = []
    for img_mask, ray_mask in zip(img_mask_list, ray_mask_list):
        out = torch.zeros((h, w * num_views, 3))
        for i in range(num_views):
            if img_mask[i] and not ray_mask[i]:
                offset = 0
            elif not img_mask[i] and ray_mask[i]:
                offset = 1
            else:
                offset = 0.5
            out[:, i * w : (i + 1) * w] += offset
        output.append(out)
    return output


def _caption_row(img: torch.Tensor, caption: str) -> torch.Tensor:
    """Overlay a minimal caption in the top-left corner of a visualization row."""
    if img.ndim != 3 or img.shape[-1] != 3:
        return img

    device = img.device
    img_np = (img.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    img_np = np.ascontiguousarray(img_np)

    box_h = min(26, img_np.shape[0])
    box_w = min(max(140, 8 * len(caption) + 20), img_np.shape[1])
    cv2.rectangle(img_np, (0, 0), (box_w, box_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        img_np,
        caption,
        (6, min(18, box_h - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return torch.from_numpy(img_np).to(device=device).float() / 255.0


def vis_and_cat(
    gt_imgs,
    pred_imgs,
    cross_gt_depths,
    cross_pred_depths,
    self_gt_depths,
    self_pred_depths,
    cross_conf,
    self_conf,
    ray_indicator,
    is_metric,
):
    cross_depth_gt_min = torch.quantile(cross_gt_depths, 0.01).item()
    cross_depth_gt_max = torch.quantile(cross_gt_depths, 0.99).item()
    cross_depth_pred_min = torch.quantile(cross_pred_depths, 0.01).item()
    cross_depth_pred_max = torch.quantile(cross_pred_depths, 0.99).item()
    cross_depth_min = min(cross_depth_gt_min, cross_depth_pred_min)
    cross_depth_max = max(cross_depth_gt_max, cross_depth_pred_max)

    cross_gt_depths_vis = colorize(
        cross_gt_depths,
        cmap_name="viridis",
        range=(
            (cross_depth_min, cross_depth_max)
            if is_metric
            else (cross_depth_gt_min, cross_depth_gt_max)
        ),
        append_cbar=True,
    )
    cross_pred_depths_vis = colorize(
        cross_pred_depths,
        cmap_name="viridis",
        range=(
            (cross_depth_min, cross_depth_max)
            if is_metric
            else (cross_depth_pred_min, cross_depth_pred_max)
        ),
        append_cbar=True,
    )

    self_depth_gt_min = torch.quantile(self_gt_depths, 0.01).item()
    self_depth_gt_max = torch.quantile(self_gt_depths, 0.99).item()
    self_depth_pred_min = torch.quantile(self_pred_depths, 0.01).item()
    self_depth_pred_max = torch.quantile(self_pred_depths, 0.99).item()
    self_depth_min = min(self_depth_gt_min, self_depth_pred_min)
    self_depth_max = max(self_depth_gt_max, self_depth_pred_max)

    self_gt_depths_vis = colorize(
        self_gt_depths,
        cmap_name="viridis",
        range=(
            (self_depth_min, self_depth_max)
            if is_metric
            else (self_depth_gt_min, self_depth_gt_max)
        ),
        append_cbar=True,
    )
    self_pred_depths_vis = colorize(
        self_pred_depths,
        cmap_name="viridis",
        range=(
            (self_depth_min, self_depth_max)
            if is_metric
            else (self_depth_pred_min, self_depth_pred_max)
        ),
        append_cbar=True,
    )
    if len(cross_conf) > 0:
        cross_conf_vis = colorize(cross_conf, cmap_name="viridis", append_cbar=True)
    if len(self_conf) > 0:
        self_conf_vis = colorize(self_conf, cmap_name="viridis", append_cbar=True)
    gt_imgs_vis = torch.zeros_like(cross_gt_depths_vis)
    gt_imgs_vis[: gt_imgs.shape[0], : gt_imgs.shape[1]] = gt_imgs
    pred_imgs_vis = torch.zeros_like(cross_gt_depths_vis)
    pred_imgs_vis[: pred_imgs.shape[0], : pred_imgs.shape[1]] = pred_imgs
    ray_indicator_vis = torch.cat(
        [
            ray_indicator,
            torch.zeros(
                ray_indicator.shape[0],
                cross_pred_depths_vis.shape[1] - ray_indicator.shape[1],
                3,
            ),
        ],
        dim=1,
    )
    rows = [
        _caption_row(ray_indicator_vis, "mask indicator"),
        _caption_row(gt_imgs_vis, "gt rgb"),
        _caption_row(pred_imgs_vis, "pred rgb"),
        _caption_row(self_gt_depths_vis, "self-view gt depth"),
        _caption_row(self_pred_depths_vis, "self-view pred depth"),
        _caption_row(self_conf_vis, "self-view confidence"),
        _caption_row(cross_gt_depths_vis, "cross-view gt depth"),
        _caption_row(cross_pred_depths_vis, "cross-view pred depth"),
        _caption_row(cross_conf_vis, "cross-view confidence"),
    ]
    out = torch.cat(
        rows,
        dim=0,
    )
    return out


def get_vis_imgs_new(loss_details, num_imgs_vis, num_views, is_metric):
    ret_dict = {}
    num_imgs_vis = min(num_imgs_vis, loss_details["gt_img1"].shape[0])
    gt_img_list = [[] for _ in range(num_imgs_vis)]
    pred_img_list = [[] for _ in range(num_imgs_vis)]

    cross_gt_depth_list = [[] for _ in range(num_imgs_vis)]
    cross_pred_depth_list = [[] for _ in range(num_imgs_vis)]

    self_gt_depth_list = [[] for _ in range(num_imgs_vis)]
    self_pred_depth_list = [[] for _ in range(num_imgs_vis)]

    cross_view_conf_list = [[] for _ in range(num_imgs_vis)]
    self_view_conf_list = [[] for _ in range(num_imgs_vis)]
    cross_view_conf_exits = False
    self_view_conf_exits = False

    img_mask_list = [[] for _ in range(num_imgs_vis)]
    ray_mask_list = [[] for _ in range(num_imgs_vis)]

    if num_views > 30:
        stride = 5
    elif num_views > 20:
        stride = 3
    elif num_views > 10:
        stride = 2
    else:
        stride = 1
    for i in range(0, num_views, stride):
        gt_imgs = 0.5 * (loss_details[f"gt_img{i+1}"] + 1)[:num_imgs_vis].detach().cpu()
        width = gt_imgs.shape[2]
        pred_imgs = (
            0.5 * (loss_details[f"pred_rgb_{i+1}"] + 1)[:num_imgs_vis].detach().cpu()
        )
        gt_img_list = batch_append(gt_img_list, gt_imgs.unbind(dim=0))
        pred_img_list = batch_append(pred_img_list, pred_imgs.unbind(dim=0))

        cross_pred_depths = (
            loss_details[f"pred_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        cross_gt_depths = (
            loss_details[f"gt_depth_{i+1}"]
            .to(gt_imgs.device)[:num_imgs_vis]
            .detach()
            .cpu()
        )
        cross_pred_depth_list = batch_append(
            cross_pred_depth_list, cross_pred_depths.unbind(dim=0)
        )
        cross_gt_depth_list = batch_append(
            cross_gt_depth_list, cross_gt_depths.unbind(dim=0)
        )

        self_gt_depths = (
            loss_details[f"self_gt_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        self_pred_depths = (
            loss_details[f"self_pred_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        self_gt_depth_list = batch_append(
            self_gt_depth_list, self_gt_depths.unbind(dim=0)
        )
        self_pred_depth_list = batch_append(
            self_pred_depth_list, self_pred_depths.unbind(dim=0)
        )

        if f"conf_{i+1}" in loss_details:
            cross_view_conf = loss_details[f"conf_{i+1}"][:num_imgs_vis].detach().cpu()
            cross_view_conf_list = batch_append(
                cross_view_conf_list, cross_view_conf.unbind(dim=0)
            )
            cross_view_conf_exits = True

        if f"self_conf_{i+1}" in loss_details:
            self_view_conf = (
                loss_details[f"self_conf_{i+1}"][:num_imgs_vis].detach().cpu()
            )
            self_view_conf_list = batch_append(
                self_view_conf_list, self_view_conf.unbind(dim=0)
            )
            self_view_conf_exits = True

        img_mask_list = batch_append(
            img_mask_list,
            loss_details[f"img_mask_{i+1}"][:num_imgs_vis].detach().cpu().unbind(dim=0),
        )
        ray_mask_list = batch_append(
            ray_mask_list,
            loss_details[f"ray_mask_{i+1}"][:num_imgs_vis].detach().cpu().unbind(dim=0),
        )

    # each element in the list is [H, num_views * W, (3)], the size of the list is num_imgs_vis
    gt_img_list = [torch.cat(sublist, dim=1) for sublist in gt_img_list]
    pred_img_list = [torch.cat(sublist, dim=1) for sublist in pred_img_list]
    cross_pred_depth_list = [
        torch.cat(sublist, dim=1) for sublist in cross_pred_depth_list
    ]
    cross_gt_depth_list = [torch.cat(sublist, dim=1) for sublist in cross_gt_depth_list]
    self_gt_depth_list = [torch.cat(sublist, dim=1) for sublist in self_gt_depth_list]
    self_pred_depth_list = [
        torch.cat(sublist, dim=1) for sublist in self_pred_depth_list
    ]
    cross_view_conf_list = (
        [torch.cat(sublist, dim=1) for sublist in cross_view_conf_list]
        if cross_view_conf_exits
        else []
    )
    self_view_conf_list = (
        [torch.cat(sublist, dim=1) for sublist in self_view_conf_list]
        if self_view_conf_exits
        else []
    )
    # each elment in the list is [num_views,], the size of the list is num_imgs_vis
    img_mask_list = [torch.stack(sublist, dim=0) for sublist in img_mask_list]
    ray_mask_list = [torch.stack(sublist, dim=0) for sublist in ray_mask_list]

    ray_indicator = gen_mask_indicator(
        img_mask_list, ray_mask_list, len(img_mask_list[0]), 30, width
    )

    for i in range(num_imgs_vis):
        out = vis_and_cat(
            gt_img_list[i],
            pred_img_list[i],
            cross_gt_depth_list[i],
            cross_pred_depth_list[i],
            self_gt_depth_list[i],
            self_pred_depth_list[i],
            cross_view_conf_list[i],
            self_view_conf_list[i],
            ray_indicator[i],
            is_metric[i],
        )
        ret_dict[f"imgs_{i}"] = out
    return ret_dict


@hydra.main(
    version_base=None,
    config_path=str(os.path.dirname(os.path.abspath(__file__))) + "/../config",
    config_name="train.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    run()
