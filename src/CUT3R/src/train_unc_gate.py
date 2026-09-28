#!/usr/bin/env python
"""Train a registration-uncertainty head (uncgate v1 / v2) on a FROZEN CUT3R.

Kendall & Gal heteroscedastic head, SURE-Map target: per frame the frozen model's own pose+depth induce a
backward optical flow to the previous frame; the head predicts a 2-channel log-variance s=(s_u,s_v) of that
flow and is trained with the Gaussian NLL against the residual vs the GT flow (GT depth + GT poses + GT K):
    L = mean_valid[ eps_u^2 e^{-s_u} + eps_v^2 e^{-s_v} + s_u + s_v ]
Nothing in CUT3R is updated; the head consumes features the frozen forward already produces.

  v1: PixelUncHead on the final decoder image tokens -> per-pixel s (B,2,H,W); loss per pixel.
  v2: TokenUncHead on [new_state_feat_i ; RoPE-aligned image token_i ; on_flag] -> per-token s; loss per
      on-image token against the patch-mean squared residual; optional extra channel pair trained on the
      NEXT frame's residual at the same cell (--next_frame_channels, ablation).

Usage (Slurm, one GPU):
  python train_unc_gate.py --mode v2 --name v2_flow --episodes 2000 --batch_size 2
Outputs: <out_root>/<name>/head.pth (+ log.txt, eval.json). Load at eval with
  infer_and_eval_worker.py --unc_head <head.pth> and control_json token_gate.src="unc".
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import dust3r.utils.path_to_croco  # noqa
from dust3r.model import ARCroco3DStereo
from dust3r.datasets import get_data_loader, DL3DV_Multi  # noqa: F401 (eval() namespace)
from dust3r.uncgate.heads import PixelUncHead, TokenUncHead, gaussian_nll_2d, laplace_nll_2d, scalar_score
from dust3r.uncgate.flow_target import (gt_backward_flow, pred_backward_flow, flow_residual,
                                        pool_to_patches_masked, rope_token_map)
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

CKPT_DEFAULT = ("/gpfs/scratch/etur59/koc821022/checkpoints_projects/cut3r_finetune_baselines/"
                "cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth")
TRAIN_ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/train/dl3dv_multi"
TEST_ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi"
OUT_ROOT = "/gpfs/scratch/etur59/koc821022/checkpoints/unc_gate"
IGNORE = {"dataset", "label", "instance", "idx", "rng"}


def to_device(batch, device):
    for view in batch:
        for k, v in view.items():
            if k in IGNORE:
                continue
            if isinstance(v, torch.Tensor):
                view[k] = v.to(device, non_blocking=True)
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
                view[k] = [x.to(device, non_blocking=True) for x in v]
    return batch


def set_epoch(loader, epoch):
    """CustomRandomSampler needs set_epoch before iteration; the accelerate-prepared loader nests it one level deeper."""
    for obj in (getattr(loader, "dataset", None), getattr(loader, "batch_sampler", None),
                getattr(getattr(loader, "batch_sampler", None), "batch_sampler", None), getattr(loader, "sampler", None)):
        if obj is not None and hasattr(obj, "set_epoch"):
            obj.set_epoch(epoch)


NLL = {"gauss": gaussian_nll_2d, "laplace": laplace_nll_2d}
CFG = {"loss": "gauss", "clip": 64.0}


def build_loader(root, num_views, batch_size, num_workers, accelerator, seed_offset=0):
    # allow_repeat=False: episodes shorter than num_views are excluded instead of padded with a repeated
    # last frame (repeated frames have zero GT flow and would teach 'static = confident').
    ds = (f'DL3DV_Multi(allow_repeat=False, split="train", ROOT="{root}", aug_crop=0, resolution=[[320, 192]], '
          f'transform=ImgNorm, num_views={num_views}, n_corres=0, force_consecutive_frame_sampling=True)')
    return get_data_loader(ds, batch_size=batch_size, num_workers=num_workers, pin_mem=True, shuffle=True,
                           drop_last=True, accelerator=accelerator, fixed_length=True)


def frame_targets(batch, res_list, t, ph, pw, n_state):
    """Flow residual of frame t vs t-1 (pixel eps (B,H,W,2), valid (B,H,W)) and its per-token squared
    version (eps2_tok (B,n_state,2), valid_tok (B,n_state))."""
    cur, prev = batch[t], batch[t - 1]
    # autocast OFF: the geometry (einsum/bmm) would otherwise run in bf16 and the sub-pixel residual
    # would be dominated by rounding noise (measured 0.57 px mean error under bf16).
    dev_type = "cuda" if cur["depthmap"].is_cuda else "cpu"
    with torch.autocast(device_type=dev_type, enabled=False):
        return _frame_targets_fp32(cur, prev, res_list, t, ph, pw, n_state)


def _frame_targets_fp32(cur, prev, res_list, t, ph, pw, n_state):
    flow_gt, valid_gt = gt_backward_flow(cur["depthmap"].float(), cur["camera_intrinsics"].float(),
                                         prev["camera_intrinsics"].float(), cur["camera_pose"].float(),
                                         prev["camera_pose"].float(), cur["valid_mask"].bool())
    flow_pr, valid_pr = pred_backward_flow(res_list[t]["pts3d_in_self_view"].float(),
                                           res_list[t]["camera_pose"].float(), res_list[t - 1]["camera_pose"].float(),
                                           prev["camera_intrinsics"].float())
    eps, valid = flow_residual(flow_pr, flow_gt, valid_pr, valid_gt)
    eps = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
    # clip absurd residuals (gross failures dominate the NLL otherwise); 64 px ~ a fifth of the width
    eps = eps.clamp(-CFG["clip"], CFG["clip"])
    eps2 = (eps ** 2).permute(0, 3, 1, 2)  # (B,2,H,W)
    pooled, frac = pool_to_patches_masked(eps2, valid, ph, pw)  # (B,2,ph,pw), (B,1,ph,pw)
    on, r_idx, c_idx = rope_token_map(n_state, ph, pw)
    on, r_idx, c_idx = on.to(eps.device), r_idx.to(eps.device), c_idx.to(eps.device)
    B = eps.shape[0]
    eps2_tok = torch.zeros(B, n_state, 2, device=eps.device)
    valid_tok = torch.zeros(B, n_state, dtype=torch.bool, device=eps.device)
    eps2_tok[:, on] = pooled[:, :, r_idx[on], c_idx[on]].permute(0, 2, 1)
    valid_tok[:, on] = frac[:, 0, r_idx[on], c_idx[on]] > 0.2
    return eps, valid, eps2_tok, valid_tok


@torch.no_grad()
def retention_curve(scores, err, valid, ks=(0.1, 0.2, 0.3)):
    """Drop the worst-k fraction (lowest score, RANK-based so ties cannot inflate the kept set) of valid
    entries; RMS(err retained)/RMS(err all) (1.0 = no ranking power)."""
    s = scores[valid].float(); e = err[valid].float()
    n = s.numel()
    if n < 100:
        return {str(k): float("nan") for k in ks}
    base = torch.sqrt(e.mean())
    order = torch.argsort(s, descending=True)  # most confident first
    out = {}
    for k in ks:
        keep = order[: max(1, int(round(n * (1 - k))))]
        out[str(k)] = float(torch.sqrt(e[keep].mean()) / base)
    return out


def run_episode(model, batch, mode, ph, pw, n_state, next_ch, step_fn=None, accum=8):
    """Roll the frozen model through one batch of episodes. The head runs inside the model's group step
    (attach_unc_gate). When step_fn is given, every `accum` frames the chunk of per-frame losses is handed to it
    (backward + optimizer step) INSIDE the rollout, so no head graph ever straddles a weight update (an in-place
    step would invalidate the saved activations of later frames). Returns the per-frame loss values (floats) and
    monitoring stats."""
    with torch.no_grad():
        (feat, pos, shape), (init_state_feat, init_mem, state_feat, state_pos, mem) = model._forward_encoder(batch)
    res_list, losses, stats = [], [], {"n": 0, "ret_head": [], "ret_conf": []}
    chunk = []
    prev_inputs, prev_valid = None, None  # cached head inputs of frame t-1 for the next-frame channels (v2 ablation)
    for t in range(len(batch)):
        with torch.no_grad():
            res_group, (state_feat, mem) = model._forward_decoder_group_step(
                views=batch, view_indices=[t], feat_group=[feat[t]], pos_group=[pos[t]], shape_group=[shape[t]],
                init_state_feat=init_state_feat, init_mem=init_mem, state_feat=state_feat, state_pos=state_pos, mem=mem)
        res = res_group[0]
        res_list.append({k: (v.detach() if (isinstance(v, torch.Tensor) and not k.startswith("unc_")) else v)
                         for k, v in res.items()})
        if t == 0:
            continue
        with torch.no_grad():
            eps, valid, eps2_tok, valid_tok = frame_targets(batch, res_list, t, ph, pw, n_state)
        if mode == "v1":
            s = res["unc_s"].permute(0, 2, 3, 1)  # (B,H,W,2)
            loss = NLL[CFG["loss"]](s, eps, valid)
            score = scalar_score(s.detach())
            err = (eps ** 2).sum(-1)
            stats["ret_head"].append(retention_curve(score, err, valid))
            conf = torch.log(res["conf_self"].float().clamp(min=1.0))
            stats["ret_conf"].append(retention_curve(conf, err, valid))
        else:
            s_tok = res["unc_s_tok"]  # (B,n_state,2 or 4)
            loss = NLL[CFG["loss"]](s_tok[..., :2], torch.sqrt(eps2_tok), valid_tok)
            if next_ch and prev_inputs is not None:
                # fresh forward on the cached inputs of frame t-1 (current weights): its channels 2:4 predict
                # the residual of THIS frame at the same cell
                s_prev = model.unc_gate(*prev_inputs)
                loss = loss + NLL[CFG["loss"]](s_prev[..., 2:4], torch.sqrt(eps2_tok), valid_tok & prev_valid)
            if next_ch:
                prev_inputs, prev_valid = res.get("unc_inputs"), valid_tok
            score = scalar_score(s_tok[..., :2].detach())
            err = eps2_tok.sum(-1)
            stats["ret_head"].append(retention_curve(score, err, valid_tok))
            # reference ranking: pooled log conf_self on the same tokens
            logc = torch.log(res["conf_self"].float().clamp(min=1.0))[:, None]
            pooled_c = torch.nn.functional.adaptive_avg_pool2d(logc, (ph, pw))[:, 0]
            on, r_idx, c_idx = rope_token_map(n_state, ph, pw)
            on, r_idx, c_idx = on.to(logc.device), r_idx.to(logc.device), c_idx.to(logc.device)
            conf_tok = torch.full_like(score, float("nan")); conf_tok[:, on] = pooled_c[:, r_idx[on], c_idx[on]]
            stats["ret_conf"].append(retention_curve(conf_tok, err, valid_tok))
        losses.append(float(loss.detach()))
        stats["n"] += 1
        for k in ("unc_s", "unc_s_tok", "unc_inputs"):
            res_list[-1].pop(k, None)
        if step_fn is not None:
            chunk.append(loss)
            if len(chunk) >= accum:
                step_fn(chunk); chunk = []
        else:
            del loss
    if step_fn is not None and chunk:
        step_fn(chunk)
    return losses, stats


def mean_ret(lst):
    if not lst:
        return {}
    keys = lst[0].keys()
    return {k: float(torch.tensor([d[k] for d in lst if d[k] == d[k]]).mean()) if any(d[k] == d[k] for d in lst) else float("nan") for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["v1", "v2"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--episodes", type=int, default=2000, help="training episodes (batches x batch_size)")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_views", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--accum_frames", type=int, default=8, help="frames per optimizer step")
    ap.add_argument("--next_frame_channels", action="store_true", help="v2: extra (s_u,s_v) pair trained on frame t+1's residual")
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--eval_every", type=int, default=200, help="episodes")
    ap.add_argument("--eval_episodes", type=int, default=16)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for the frozen model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/head.pth (weights, optimizer, episode count)")
    ap.add_argument("--loss", choices=["gauss", "laplace"], default="gauss", help="2D Gaussian NLL (SURE-Map) or Laplace (K&G Sec.4)")
    ap.add_argument("--clip", type=float, default=64.0, help="residual clip in px before the NLL")
    args = ap.parse_args()
    CFG["loss"] = args.loss; CFG["clip"] = args.clip

    torch.manual_seed(args.seed)
    out_dir = os.path.join(args.out_root, args.name); os.makedirs(out_dir, exist_ok=True)
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=3))])  # GPFS stalls during eval loading exceeded the 10-min NCCL default
    device = accelerator.device
    is_main = accelerator.is_main_process
    log = open(os.path.join(out_dir, "log.txt"), "a") if is_main else None
    def P(*a):
        if not is_main:
            return
        msg = " ".join(str(x) for x in a); print(msg, flush=True); log.write(msg + "\n"); log.flush()
    P(f"[unc_gate] {time.strftime('%F %T')} procs={accelerator.num_processes} args={vars(args)}")
    model = ARCroco3DStereo.from_pretrained(args.ckpt).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if getattr(model, "pose_gru", None) is not None:
        model.pose_gru = None
    n_state = int(model.state_size); psz = int(model.patch_embed.patch_size[0]); ph, pw = 192 // psz, 320 // psz
    on, _, _ = rope_token_map(n_state, ph, pw)
    P(f"[unc_gate] frozen model loaded; n_state={n_state} patch={ph}x{pw} on_image={int(on.sum())}")
    out_ch = 4 if (args.mode == "v2" and args.next_frame_channels) else 2
    head_cfg = dict(dec_dim=model.dec_embed_dim, patch_size=psz, out_ch=2) if args.mode == "v1" else \
        dict(state_dim=model.dec_embed_dim, img_dim=model.dec_embed_dim, hidden=args.hidden, out_ch=out_ch)
    head = (PixelUncHead(**head_cfg) if args.mode == "v1" else TokenUncHead(**head_cfg)).to(device)
    model.attach_unc_gate(head, args.mode, train=True)
    P(f"[unc_gate] head {args.mode}: {sum(p.numel() for p in head.parameters())/1e6:.3f}M params cfg={head_cfg}")
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    train_loader = build_loader(TRAIN_ROOT, args.num_views, args.batch_size, args.num_workers, accelerator)
    eval_loader = build_loader(TEST_ROOT, args.num_views, 1, max(2, args.num_workers // 2), accelerator)
    head, opt, train_loader, eval_loader = accelerator.prepare(head, opt, train_loader, eval_loader)
    model.attach_unc_gate(head, args.mode, train=True)  # re-attach the (possibly DDP-wrapped) head
    set_epoch(train_loader, args.seed); set_epoch(eval_loader, 0)
    n_proc = accelerator.num_processes

    def evaluate(tag):
        # every rank scores its own shard of the eval loader; stats are gathered to rank 0
        head.eval(); rh, rc, ls = [], [], []
        model.unc_gate_train = False  # no head graphs during evaluation
        set_epoch(eval_loader, 0)
        with torch.no_grad():
            for i, b in enumerate(eval_loader):
                if i >= max(1, args.eval_episodes // n_proc):
                    break
                b = to_device(b, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                    losses, st = run_episode(model, b, args.mode, ph, pw, n_state, args.next_frame_channels)
                ls.append(float(sum(losses) / max(len(losses), 1))); rh += st["ret_head"]; rc += st["ret_conf"]
        head.train(); model.unc_gate_train = True
        nll_t = torch.tensor([sum(ls) / max(len(ls), 1), float(len(ls))], device=device)
        rh_t = torch.tensor([[d[k] for k in ("0.1", "0.2", "0.3")] for d in rh] or [[float("nan")] * 3], device=device)
        rc_t = torch.tensor([[d[k] for k in ("0.1", "0.2", "0.3")] for d in rc] or [[float("nan")] * 3], device=device)
        nll_all = accelerator.gather(nll_t.reshape(1, 2)); rh_all = accelerator.gather(rh_t); rc_all = accelerator.gather(rc_t)
        nll = float((nll_all[:, 0] * nll_all[:, 1]).sum() / nll_all[:, 1].sum().clamp(min=1))
        def _m(t):
            return {k: float(torch.nanmean(t[:, i])) for i, k in enumerate(("0.1", "0.2", "0.3"))}
        rec = {"tag": tag, "nll": nll, "retention_head": _m(rh_all), "retention_conf_self": _m(rc_all), "n_frames": int(nll_all[:, 1].sum())}
        P(f"[unc_gate] EVAL {json.dumps(rec)}")
        if is_main:
            with open(os.path.join(out_dir, "eval.json"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    def save(step, seen=0):
        if not is_main:
            return
        torch.save({"head": accelerator.unwrap_model(head).state_dict(), "mode": args.mode, "head_cfg": head_cfg,
                    "args": vars(args), "step": step, "seen": seen, "opt": opt.state_dict()}, os.path.join(out_dir, "head.pth"))

    head.train(); seen = 0; step = 0; t0 = time.time(); run_loss = []
    ck_path = os.path.join(out_dir, "head.pth")
    if args.resume and os.path.isfile(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(head).load_state_dict(ck["head"])
        try:
            opt.load_state_dict(ck["opt"])
        except Exception as e:  # optimizer state is optional
            P(f"[unc_gate] resume: optimizer state not restored ({e})")
        seen, step = int(ck.get("seen", 0)), int(ck.get("step", 0))
        P(f"[unc_gate] RESUMED from {ck_path}: seen={seen} step={step}")
        if seen >= args.episodes:
            P("[unc_gate] nothing left to do"); evaluate("final"); save(step, seen); return
    else:
        evaluate("init")
    done = False
    epoch = args.seed + (seen // max(args.episodes, 1)) + (1 if seen > 0 else 0)  # resumed runs draw a fresh order
    while not done:
        set_epoch(train_loader, epoch); epoch += 1
        for b in train_loader:
            b = to_device(b, device)
            def step_fn(chunk_losses):
                nonlocal step
                chunk = torch.stack(chunk_losses).mean()
                opt.zero_grad(set_to_none=True); accelerator.backward(chunk)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0); opt.step(); step += 1
                run_loss.append(float(chunk))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                losses, st = run_episode(model, b, args.mode, ph, pw, n_state, args.next_frame_channels,
                                         step_fn=step_fn, accum=args.accum_frames)
            seen += args.batch_size * n_proc
            if seen % (10 * args.batch_size * n_proc) == 0:
                P(f"[unc_gate] ep {seen}/{args.episodes} step {step} nll {sum(run_loss[-50:])/max(len(run_loss[-50:]),1):.4f} "
                  f"ret_head@0.2 {mean_ret(st['ret_head']).get('0.2', float('nan')):.3f} ret_conf@0.2 {mean_ret(st['ret_conf']).get('0.2', float('nan')):.3f} "
                  f"{(time.time()-t0)/60:.1f} min")
            if seen % args.eval_every == 0:
                evaluate(f"ep{seen}"); save(step, seen)
            if seen >= args.episodes:
                done = True; break
    evaluate("final"); save(step, seen)
    P(f"[unc_gate] done: {seen} episodes, {step} steps, {(time.time()-t0)/60:.1f} min -> {out_dir}/head.pth")


if __name__ == "__main__":
    main()
