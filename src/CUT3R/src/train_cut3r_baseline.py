# --------------------------------------------------------
# training code for CUT3R
# --------------------------------------------------------
# References:
# DUSt3R: https://github.com/naver/dust3r
# --------------------------------------------------------
import datetime
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sized
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

import builtins
import pathlib
import random
import shutil
from datetime import timedelta
import croco.utils.misc as misc  # noqa
import dust3r.utils.path_to_croco  # noqa: F401
import hydra
import torch.multiprocessing
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import send_to_device
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa
from dust3r.datasets import get_data_loader
from dust3r.inference import loss_of_one_batch, loss_of_one_batch_tbptt  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.model import (  # noqa: F401, needed when loading the model
    ARCroco3DStereo,
    ARCroco3DStereoConfig,
    PreTrainedModel,
    inf,
    strip_module,
)
from dust3r.utils.camera import pose_encoding_to_camera
from dust3r.utils.geometry import geotrf, inv
from dust3r.viz import colorize
from omegaconf import OmegaConf

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
            builtin_print(f"[{now}] ", end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def save_current_code(outdir):
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H-%M-%S")
    src_dir = "."
    dst_dir = os.path.join(outdir, "code", f"{date_time}")
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


def split_pose_gru_param_groups(param_groups, pose_gru, lr_scale):
    """Give the from-scratch PoseGRU its own optimizer param groups with an
    lr multiplier (adjust_learning_rate multiplies each group's lr by its
    lr_scale) — at the finetune's tiny base lr a from-scratch module would
    stay effectively dead. Weight-decay assignment is inherited from the
    group each parameter came from (biases keep wd=0)."""
    gru_param_ids = {id(p) for p in pose_gru.parameters()}
    split = []
    for group in param_groups:
        gru_params = [p for p in group["params"] if id(p) in gru_param_ids]
        rest = [p for p in group["params"] if id(p) not in gru_param_ids]
        if rest:
            rest_group = dict(group)
            rest_group["params"] = rest
            split.append(rest_group)
        if gru_params:
            gru_group = {k: v for k, v in group.items() if k != "params"}
            gru_group["params"] = gru_params
            gru_group["lr_scale"] = float(group.get("lr_scale", 1.0)) * float(lr_scale)
            split.append(gru_group)
    return split


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
        printer.info(f"Saved current code to {dst_dir} in {snapshot_sec:.1f}s")
    elif accelerator.is_main_process:
        printer.info("Skipping code snapshot (set save_code_snapshot=true to enable)")

    use_wandb = bool(getattr(args, "use_wandb", True))
    if accelerator.is_main_process and wandb is not None and wandb.run is None and use_wandb:
        # Added a default project name so it doesn't silently fail to log
        wandb_project = os.environ.get("WANDB_PROJECT", "pointworld-droid-v2")
        wandb_id_path = os.path.join(args.output_dir, "wandb_run_id.txt")
        tb_dir = os.path.join(args.output_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        # Fresh run id per launch. Reusing a persisted id with resume="allow"
        # makes wandb re-attach to the old run; with sync_tensorboard the
        # restarted (non-monotonic) tensorboard steps then get dropped, so the
        # run looks "active" but records nothing. Generate a new id every launch
        # (export WANDB_RUN_ID to intentionally resume a specific run).
        wandb_run_id = os.environ.get("WANDB_RUN_ID") or wandb.util.generate_id()
        with open(wandb_id_path, "w", encoding="utf-8") as f:
            f.write(wandb_run_id)
        wandb.tensorboard.patch(root_logdir=tb_dir)
        printer.info("Initializing Weights & Biases...")
        wandb_start = time.time()
        wandb.init(
            project=wandb_project,
            sync_tensorboard=True,
            name=os.environ.get("WANDB_NAME", os.path.basename(args.output_dir.rstrip("/"))),
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

    printer.info(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")

    # fix the seed
    seed = args.seed + accelerator.state.process_index
    printer.info(f"Setting seed to {seed} for process {accelerator.state.process_index}")
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
        fixed_length=args.fixed_length,
    )
    printer.info("Building test dataset %s", args.test_dataset)
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset,
            args.batch_size,
            args.num_workers,
            accelerator=accelerator,
            test=True,
            fixed_length=True,
        )
        for dataset in args.test_dataset.split("+")
    }

    # model
    printer.info("Loading model: %s", args.model)
    model: PreTrainedModel = eval(args.model)
    printer.info(f"All model parameters: {sum(p.numel() for p in model.parameters())}")
    printer.info(f"Encoder parameters: {sum(p.numel() for p in model.enc_blocks.parameters())}")
    printer.info(f"Decoder parameters: {sum(p.numel() for p in model.dec_blocks.parameters())}")

    # PoseGRU refiner for the feed_prev_pred loop. Enabled BEFORE the
    # pretrained/resume loads (the load_state_dict guard requires the module
    # to exist when a checkpoint carries pose_gru weights), BEFORE .to(device)
    # (moves with the rest of the model), and BEFORE the optimizer is built
    # (its params need their own group — see split_pose_gru_param_groups).
    use_pose_gru = bool(getattr(args, "pose_gru", False))
    if use_pose_gru:
        assert bool(
            getattr(args, "feed_prev_pred", False)
        ), "pose_gru refines the fed-back pose — it requires feed_prev_pred=True"
        model.enable_pose_gru(
            hidden_dim=int(getattr(args, "pose_gru_hidden_dim", 128)),
            mode=str(getattr(args, "pose_gru_mode", "residual")),
            input_mode=str(getattr(args, "pose_gru_input", "pose")),
            # F lever: pooled pre-ray image features into the cell input.
            # frames=1 ("F0") is the current view only; frames=2 ("F1") adds
            # the previous view. img_feat_proj=False drops the zero-init
            # projector and hands the cell the full 1024/2048-d features.
            img_feat=str(getattr(args, "pose_gru_img_feat", "none")),
            img_feat_dim=int(getattr(args, "pose_gru_img_feat_dim", 32)),
            img_feat_frames=int(getattr(args, "pose_gru_img_feat_frames", 2)),
            img_feat_proj=bool(getattr(args, "pose_gru_img_feat_proj", True)),
            # F source: where the per-step feature comes from — "pooled"
            # (mean pre-ray tokens, the path above), a frozen resnet18 /
            # dinov2_vits14 encoder on the raw image, or "corr" motion stats.
            # The frozen encoder initializes from a local pretrained file
            # (img_encoder_weights, None => default under pretrained_encoders/)
            # and is serialized into every ckpt, so eval nodes never need it.
            img_feat_src=str(getattr(args, "pose_gru_img_feat_src", "pooled")),
            img_encoder_weights=getattr(args, "pose_gru_img_encoder_weights", None),
            # R lever: back-to-back cell iterations per view. Persisted in
            # ckpt args (invisible in weight shapes — load_model reads it
            # back exactly like pose_gru_mode).
            iters=int(getattr(args, "pose_gru_iters", 1)),
        )
        # Split the param count: the encoder sources hang a frozen network
        # under pose_gru.img_encoder (11.2M resnet18 / 21M dinov2), and a
        # single total would misread as the GRU having grown 100x. Only the
        # trainable count reaches the optimizer.
        gru_trainable = sum(
            p.numel() for p in model.pose_gru.parameters() if p.requires_grad
        )
        gru_frozen = sum(
            p.numel() for p in model.pose_gru.parameters() if not p.requires_grad
        )
        printer.info(
            "pose_gru enabled: mode=%s, input=%s (dim %d), hidden_dim=%d, "
            "img_feat=%s (src %s, appended %d, frames %d, proj=%s), iters=%d, "
            "trainable params=%d%s"
            % (
                model.pose_gru.mode,
                model.pose_gru.input_mode,
                model.pose_gru.input_dim,
                model.pose_gru.hidden_dim,
                model.pose_gru.img_feat,
                model.pose_gru.img_feat_src,
                model.pose_gru.img_feat_dim,
                model.pose_gru.img_feat_frames,
                model.pose_gru.img_feat_proj,
                model.pose_gru.iters,
                gru_trainable,
                f" (+{gru_frozen} frozen img_encoder)" if gru_frozen else "",
            )
        )

    printer.info(f">> Creating train criterion = {args.train_criterion}")
    train_criterion = eval(args.train_criterion).to(device)
    printer.info(f">> Creating test criterion = {args.test_criterion or args.train_criterion}")
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    # Optional extra diagnostic test criteria. These are evaluated on the SAME forward
    # pass as test_criterion (no extra model forward), so you can keep test_criterion
    # equal to train_criterion to watch overfitting while ALSO tracking the original
    # test loss. Each is logged under its own name, so every term gets its own plot:
    #   <test_prefix>_<name>/loss_{avg,med} and <test_prefix>_<name>/<term>_{avg,med}.
    extra_test_criteria = {}
    for name, expr in dict(getattr(args, "extra_test_criteria", {}) or {}).items():
        printer.info(f">> Creating extra test criterion '{name}' = {expr}")
        extra_test_criteria[name] = eval(expr).to(device)

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
                k: v for k, v in ckpt["model"].items() if "enc_blocks" in k or "patch_embed" in k
            }
            printer.info(model.load_state_dict(strip_module(filtered_state_dict), strict=False))
        else:
            printer.info(model.load_state_dict(strip_module(ckpt["model"]), strict=False))
        del ckpt  # in case it occupies memory

    # # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model, args.weight_decay)
    if use_pose_gru:
        pose_gru_lr_scale = float(getattr(args, "pose_gru_lr_scale", 100.0))
        param_groups = split_pose_gru_param_groups(
            param_groups, model.pose_gru, pose_gru_lr_scale
        )
        printer.info(f"pose_gru param groups split out with lr_scale x{pose_gru_lr_scale:g}")
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    # ddp_grad_sync: the TBPTT path forwards through the UNWRAPPED module, so
    # DDP's reducer is never armed and no cross-rank gradient averaging happens
    # (see croco/utils/misc.py:all_reduce_grads_). Default False reproduces the
    # historical baseline exactly; set true in the config to restore the sync.
    # Only meaningful with long_context=True — the non-TBPTT path already syncs
    # via the DDP wrapper, and enabling this there would double-average.
    ddp_grad_sync = bool(getattr(args, "ddp_grad_sync", False))
    if ddp_grad_sync and not bool(getattr(args, "long_context", False)):
        raise ValueError(
            "ddp_grad_sync=True requires long_context=True: the non-TBPTT path "
            "(loss_of_one_batch) forwards through the DDP wrapper and already "
            "all-reduces, so syncing again would average gradients twice."
        )
    loss_scaler = NativeScaler(accelerator=accelerator, ddp_grad_sync=ddp_grad_sync)
    printer.info(
        f"ddp_grad_sync={ddp_grad_sync} "
        f"({'cross-rank gradient averaging RESTORED' if ddp_grad_sync else 'baseline: NO cross-rank averaging on the TBPTT path'})"
    )

    accelerator.even_batches = False
    optimizer, model, data_loader_train = accelerator.prepare(optimizer, model, data_loader_train)
    # Add MVU to CUT3R
    base_model = accelerator.unwrap_model(model)
    base_model.views_per_step = int(getattr(args, "views_per_step", 1))
    # Closed-loop ray conditioning: at step x, build a ray map from the pose the
    # model predicted at step x-1 and add it to view x's encoder tokens (see
    # _forward_decoder_group_step). Dataset-side ray flags must be off.
    base_model.feed_prev_pred = bool(getattr(args, "feed_prev_pred", False))
    if base_model.feed_prev_pred:
        assert base_model.views_per_step == 1, "feed_prev_pred requires views_per_step=1"
        printer.info("feed_prev_pred=True: conditioning on previous predicted pose")
    # PoseGRU gradient-flow levers (read inside _forward_decoder_group_step /
    # loss_of_one_batch_tbptt; both change ONLY where gradients flow, never the
    # forward math — eval/inference is identical across settings):
    #   pose_gru_bptt (G1): keep the hidden's tape across the steps of one TBPTT
    #     chunk (detached at chunk boundaries) instead of every step.
    #   pose_gru_e2e (G2): don't detach the GRU output entering the ray build,
    #     so the main reconstruction loss also reaches the GRU.
    base_model.pose_gru_bptt = bool(getattr(args, "pose_gru_bptt", False))
    base_model.pose_gru_e2e = bool(getattr(args, "pose_gru_e2e", False))
    # R lever inner-iteration discipline (RAFT convention, default True):
    # detach the running pose estimate between iterations; the hidden is
    # never detached inside a view. Training-graph-only — inert at eval
    # (no_grad) and at iters=1.
    base_model.pose_gru_iter_detach = bool(getattr(args, "pose_gru_iter_detach", True))
    if base_model.pose_gru_bptt or base_model.pose_gru_e2e:
        assert use_pose_gru, "pose_gru_bptt / pose_gru_e2e require pose_gru=True"
        printer.info(
            f"pose_gru gradient levers: bptt={base_model.pose_gru_bptt} "
            f"(within-chunk BPTT), e2e={base_model.pose_gru_e2e} (main-loss grad)"
        )
    # ORACLE DIAGNOSTIC — NOT a lever, and unlike pose_gru_bptt/pose_gru_e2e it
    # DOES change the forward math. Feeds the GRU the CURRENT view's GT pose in
    # the aux loss's own frame and encoding, so a correctly wired zero-init
    # residual GRU is a pass-through and scores identically zero. Use it to
    # answer "is the GRU wired to the thing it is scored on"; it cannot answer
    # "is the GRU learning" (the correct answer is zero before any gradient
    # step). See gt_pose_encoding / the pose_gru_oracle block in model.py.
    #   'gt' — the GRU's POSE input becomes the current view's GT pose instead
    #          of the previous step's predicted pose P(x-1). That is the ONLY
    #          change: the delta half of the A4 input, the image features, the
    #          hidden, the pose stashes, the ray build (still fed the GRU's own
    #          output), the e2e/bptt tapes and res["gru_pose"] are all untouched.
    base_model.pose_gru_oracle = str(getattr(args, "pose_gru_oracle", "off") or "off")
    assert base_model.pose_gru_oracle in ("off", "gt"), (
        f"pose_gru_oracle must be off|gt, got {base_model.pose_gru_oracle!r}"
    )
    if base_model.pose_gru_oracle != "off":
        assert use_pose_gru, "pose_gru_oracle requires pose_gru=True"
        assert base_model.feed_prev_pred, "pose_gru_oracle requires feed_prev_pred=True"
        assert base_model.pose_gru.mode == "residual", (
            "pose_gru_oracle's zero-loss expectation needs pose_gru_mode='residual' "
            "(zero-init head => exact pass-through); mode="
            f"{base_model.pose_gru.mode!r} regresses the pose freely and has no "
            "zero-loss expectation (direct: t and q; split_anchor: t)"
        )
        printer.warning(
            f"*** pose_gru_oracle={base_model.pose_gru_oracle} — DIAGNOSTIC RUN, "
            "NOT a training arm *** GT pose is injected at the GRU input. "
            "Expect gru_quat_loss == 0 at step 0 (a zero-init residual head "
            "returns its input unchanged). gru_trans_loss/gru_pose_loss stay "
            "NONZERO by design: the criterion divides the GRU output by the "
            "HEAD's scene scale while the injected GT is in GT scale. The "
            "refined pose drives the conditioning ray build as always, so "
            "reconstruction runs on a GT-derived pose and recon/pose metrics "
            "are not comparable to an honest arm."
        )
    if use_pose_gru and base_model.pose_gru.iters > 1:
        printer.info(
            f"pose_gru R lever: iters={base_model.pose_gru.iters}, "
            f"iter_detach={base_model.pose_gru_iter_detach}, "
            f"iter_gamma(train)={float(getattr(args, 'pose_gru_iter_gamma', 0.0)):g}"
        )
    base_model.debug_grouped_updates = (
        bool(getattr(args, "debug_grouped_updates", False)) and accelerator.is_main_process
    )
    base_model.debug_grouped_updates_once = bool(getattr(args, "debug_grouped_updates_once", True))
    base_model._debug_grouped_updates_emitted = 0
    printer.info(f"Configured grouped recurrence with views_per_step={base_model.views_per_step}")
    if base_model.debug_grouped_updates:
        printer.info(
            "Grouped-update debug prints enabled"
            f" (once={base_model.debug_grouped_updates_once})"
        )

    def write_log_stats(epoch, train_stats, test_stats):
        if accelerator.is_main_process:
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()})
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
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

        # Keep all ranks in lockstep at the epoch boundary. The save_model() calls below
        # run on the main process only; without a barrier the other ranks race ahead into
        # the next collective (eval forward / next-epoch backward) while rank 0 is still in
        # torch.save(), which desyncs NCCL and hangs until the watchdog timeout.
        accelerator.wait_for_everyone()

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
                    extra_criteria=extra_test_criteria,
                )
                test_stats[test_name] = stats

                # Save best of all. Aggregation is configurable via best_ckpt_agg
                # ('med' or 'avg', default 'med'); falls back to avg if the chosen
                # key is unavailable (e.g. median logging disabled).
                best_key = f"loss_{getattr(args, 'best_ckpt_agg', 'med')}"
                if best_key not in stats:
                    best_key = "loss_avg"
                if stats[best_key] < best_so_far:
                    best_so_far = stats[best_key]
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

        # Barrier so the main-only keep/best saves above finish before any rank enters
        # the next training epoch's collectives.
        accelerator.wait_for_everyone()

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
    printer.info(f"Training time {total_time_str}")

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
        fixed_length=fixed_length,
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
    header = f"Epoch: [{epoch}]"
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
        printer.info(f"log_dir: {log_writer.log_dir}")

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
                print(f"Loss is {loss_value}, stopping training, loss details: {loss_details}")
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
                        loss_details[f"self_pred_depth_{k+1}"] = depths_self[k].detach().cpu()
                        loss_details[f"self_gt_depth_{k+1}"] = gt_depths_self[k].detach().cpu()
                        loss_details[f"pred_depth_{k+1}"] = depths_cross[k].detach().cpu()
                        loss_details[f"gt_depth_{k+1}"] = gt_depths_cross[k].detach().cpu()

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
    extra_criteria=None,
):

    model.eval()
    # Eval on the UNWRAPPED module. The test loaders are not sharded across ranks
    # (only data_loader_train is passed to accelerator.prepare), so every rank already
    # computes identical metrics. Keeping the DDP wrapper here makes each forward fire a
    # buffer-broadcast collective (broadcast_buffers=True, the model has RoPE/pos-embed
    # buffers) that deadlocks against rank 0 while it is busy in the main-only
    # save_model() torch.save(). Unwrapping removes all eval-time collectives.
    # Eval/logging only -> no effect on training optimization.
    model = accelerator.unwrap_model(model)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = f"Test Epoch: [{epoch}]"

    if log_writer is not None:
        printer.info(f"log_dir: {log_writer.log_dir}")

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(0)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(0)

    # CUT3R's OWN depth_evaluation, imported so the mono/video math is byte-identical
    # to eval/monodepth/tools.py and eval/video_depth/tools.py (the two functions are
    # identical). Lazy import: a missing optional dep must never abort a test epoch.
    try:
        from eval.monodepth.tools import depth_evaluation as _mono_depth_eval
        from eval.video_depth.tools import depth_evaluation as _video_depth_eval
    except Exception as _imp_err:  # pragma: no cover - optional-dep guard
        _mono_depth_eval = _video_depth_eval = None
        printer.info(
            f"[test] CUT3R depth_evaluation import failed ({_imp_err}); "
            "depth metrics disabled this run"
        )

    def _masked_gt_pred_depths(gts, preds):
        """Per view: (gt_depth, pred_depth) as (B,H,W) with invalid GT pixels zeroed.

        Invalid = ~valid_mask | non-finite | gt<=0. Zeroing GT there makes CUT3R's
        internal (gt>0) mask select exactly the valid pixels for BOTH the alignment
        and the metric. Detached: eval/logging only, no graph.
        """
        out = []
        for gt, pred in zip(gts, preds):
            gt_depth = geotrf(inv(gt["camera_pose"]), gt["pts3d"])[..., -1]
            pr_depth = pred["pts3d_in_self_view"][..., -1]
            valid = gt.get("valid_mask", torch.ones_like(gt_depth, dtype=torch.bool)).bool()
            valid = valid & torch.isfinite(gt_depth) & torch.isfinite(pr_depth) & (gt_depth > 0)
            gt_masked = torch.where(valid, gt_depth, torch.zeros_like(gt_depth))
            out.append((gt_masked.detach(), pr_depth.detach()))
        return out

    def _pixelweighted_avg(vals, weights):
        vals = np.asarray(vals, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        keep = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
        if not keep.any():
            return None
        return float(np.average(vals[keep], weights=weights[keep]))

    def _compute_depth_metrics(gts, preds):
        """MONO and VIDEO depth metrics, each identical to CUT3R's own protocol.

        MONO  (eval/monodepth/eval_metrics.py): each frame aligned INDEPENDENTLY by
               median(gt)/median(pred), abs_rel / (delta<1.25) per frame, then
               valid-pixel-weighted mean over frames.
        VIDEO (eval/video_depth/eval_depth.py --align scale): all num_views frames of
               one clip (= one batch element) aligned by ONE Weiszfeld scale
               (align_with_scale) over pooled pixels, metrics pooled, then
               valid-pixel-weighted mean over clips.
        Both call CUT3R's depth_evaluation verbatim, so the per-frame / per-clip math
        is identical. Logged as absrel/a1 (mono) and video_absrel/video_a1 (video).
        Per-batch values are reduced across batches by the MetricLogger the same
        (mean-of-batch-means) way as loss/pose. Eval/logging only.
        """
        if _mono_depth_eval is None:
            return {}
        views = _masked_gt_pred_depths(gts, preds)
        if not views:
            return {}
        batch_size = views[0][0].shape[0]

        # MONO: per-frame median alignment (default branch of depth_evaluation).
        m_absrel, m_a1, m_w = [], [], []
        for gt_v, pr_v in views:
            for b in range(gt_v.shape[0]):
                res, _, _, _ = _mono_depth_eval(pr_v[b], gt_v[b], max_depth=None)
                m_absrel.append(res["Abs Rel"])
                m_a1.append(res["δ < 1.25"])
                m_w.append(res["valid_pixels"])

        # VIDEO: one Weiszfeld scale per clip over its stacked (T,H,W) frames.
        v_absrel, v_a1, v_w = [], [], []
        for b in range(batch_size):
            gt_stack = torch.stack([gt_v[b] for gt_v, _ in views], dim=0)
            pr_stack = torch.stack([pr_v[b] for _, pr_v in views], dim=0)
            res, _, _, _ = _video_depth_eval(
                pr_stack, gt_stack, max_depth=None, align_with_scale=True
            )
            v_absrel.append(res["Abs Rel"])
            v_a1.append(res["δ < 1.25"])
            v_w.append(res["valid_pixels"])

        out = {}
        mono_absrel = _pixelweighted_avg(m_absrel, m_w)
        if mono_absrel is not None:
            out["absrel"] = mono_absrel
            out["a1"] = _pixelweighted_avg(m_a1, m_w)
        video_absrel = _pixelweighted_avg(v_absrel, v_w)
        if video_absrel is not None:
            out["video_absrel"] = video_absrel
            out["video_a1"] = _pixelweighted_avg(v_a1, v_w)
        return out

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

            gt_seq = torch.stack([gts[i]["camera_pose"][b] for i in valid_ids], dim=0).double()
            pr_seq = torch.stack([pr_cams[i][b] for i in valid_ids], dim=0).double()

            gt_seq = torch.linalg.inv(gt_seq[:1]) @ gt_seq
            pr_seq = torch.linalg.inv(pr_seq[:1]) @ pr_seq

            aligned = _align_points_similarity(pr_seq[:, :3, 3], gt_seq[:, :3, 3])
            if aligned is None:
                continue
            scale, R_align, t_align = aligned

            pr_aligned = pr_seq.clone()
            pr_aligned[:, :3, :3] = R_align @ pr_seq[:, :3, :3]
            pr_aligned[:, :3, 3] = (scale * (R_align @ pr_seq[:, :3, 3].T)).T + t_align

            ate = torch.sqrt(
                torch.mean(torch.sum((pr_aligned[:, :3, 3] - gt_seq[:, :3, 3]) ** 2, dim=-1))
            )
            ate_list.append(float(ate))

            # Per-sequence RPE aggregated as RMSE over this sequence's consecutive
            # pairs, then averaged across sequences below -- matching CUT3R's
            # eval/relpose/evo_utils.eval_metrics (evo rpe .stats["rmse"]) and
            # eval_depth_poses.py's --pose_reduce rmse. Previously all pairs from
            # every sequence were pooled into one arithmetic mean.
            seq_rpe_trans = []
            seq_rpe_rot = []
            for i in range(pr_aligned.shape[0] - 1):
                gt_rel = torch.linalg.inv(gt_seq[i]) @ gt_seq[i + 1]
                pr_rel = torch.linalg.inv(pr_aligned[i]) @ pr_aligned[i + 1]
                err = torch.linalg.inv(gt_rel) @ pr_rel
                seq_rpe_trans.append(float(torch.linalg.norm(err[:3, 3])))
                seq_rpe_rot.append(float(_pose_rot_err_deg(err[:3, :3])))
            if seq_rpe_trans:
                rpe_trans_list.append(float(np.sqrt(np.mean(np.square(seq_rpe_trans)))))
                rpe_rot_list.append(float(np.sqrt(np.mean(np.square(seq_rpe_rot)))))

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

        # Extra diagnostic criteria on the SAME preds/views (no extra forward).
        # Keys are namespaced by criterion so they get their own plots and never
        # collide with the primary criterion's terms (e.g. RGBLoss_rgb/i, pose_loss).
        if extra_criteria:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                for cname, crit in extra_criteria.items():
                    extra_loss, extra_details = crit(result["views"], result["pred"])
                    extra_kwargs = {f"{cname}/loss": float(extra_loss)}
                    extra_kwargs.update({f"{cname}/{k}": v for k, v in extra_details.items()})
                    metric_logger.update(**extra_kwargs)

        depth_metrics = _compute_depth_metrics(batch, result["pred"])
        if depth_metrics:
            metric_logger.update(**depth_metrics)
        pose_metrics = _compute_pose_metrics(batch, result["pred"])
        if pose_metrics:
            metric_logger.update(**pose_metrics)

    printer.info("Averaged stats: %s", metric_logger)

    # Always log avg; log median only when enabled (config: log_median, default True).
    aggs = [("avg", "global_avg")]
    if getattr(args, "log_median", True):
        aggs.append(("med", "median"))
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

    # Always visualize every view (stride=1) regardless of sequence length.
    stride = 1
    for i in range(0, num_views, stride):
        gt_imgs = 0.5 * (loss_details[f"gt_img{i+1}"] + 1)[:num_imgs_vis].detach().cpu()
        width = gt_imgs.shape[2]
        pred_imgs = 0.5 * (loss_details[f"pred_rgb_{i+1}"] + 1)[:num_imgs_vis].detach().cpu()
        gt_img_list = batch_append(gt_img_list, gt_imgs.unbind(dim=0))
        pred_img_list = batch_append(pred_img_list, pred_imgs.unbind(dim=0))

        cross_pred_depths = loss_details[f"pred_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        cross_gt_depths = (
            loss_details[f"gt_depth_{i+1}"].to(gt_imgs.device)[:num_imgs_vis].detach().cpu()
        )
        cross_pred_depth_list = batch_append(
            cross_pred_depth_list, cross_pred_depths.unbind(dim=0)
        )
        cross_gt_depth_list = batch_append(cross_gt_depth_list, cross_gt_depths.unbind(dim=0))

        self_gt_depths = loss_details[f"self_gt_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        self_pred_depths = loss_details[f"self_pred_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        self_gt_depth_list = batch_append(self_gt_depth_list, self_gt_depths.unbind(dim=0))
        self_pred_depth_list = batch_append(self_pred_depth_list, self_pred_depths.unbind(dim=0))

        if f"conf_{i+1}" in loss_details:
            cross_view_conf = loss_details[f"conf_{i+1}"][:num_imgs_vis].detach().cpu()
            cross_view_conf_list = batch_append(
                cross_view_conf_list, cross_view_conf.unbind(dim=0)
            )
            cross_view_conf_exits = True

        if f"self_conf_{i+1}" in loss_details:
            self_view_conf = loss_details[f"self_conf_{i+1}"][:num_imgs_vis].detach().cpu()
            self_view_conf_list = batch_append(self_view_conf_list, self_view_conf.unbind(dim=0))
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
    cross_pred_depth_list = [torch.cat(sublist, dim=1) for sublist in cross_pred_depth_list]
    cross_gt_depth_list = [torch.cat(sublist, dim=1) for sublist in cross_gt_depth_list]
    self_gt_depth_list = [torch.cat(sublist, dim=1) for sublist in self_gt_depth_list]
    self_pred_depth_list = [torch.cat(sublist, dim=1) for sublist in self_pred_depth_list]
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
