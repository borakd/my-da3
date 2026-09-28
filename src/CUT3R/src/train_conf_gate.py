#!/usr/bin/env python
"""Train ONLY the self-view confidence branch of a frozen CUT3R so that the per-token state-write gate it keys
improves the following frames' pose and depth.

Mechanism (unchanged from the evaluated arm): per frame, the self-view confidence map is pooled to the
RoPE-aligned state grid; the most confident half of the 240 on-image state tokens commit fully, the rest at
gmin; registers always commit. For training the rank rule is relaxed to a budget-preserving soft rule
(sigmoid around the per-frame median, token_gate_soft=T) so gradient reaches the confidence.

Objective: the finetune's own criterion (ConfLoss(Regr3DPoseBatchList(L21)) + RGBLoss) over chunks of H frames,
backpropagated through the FROZEN decoder/heads into the confidence branch of earlier frames (the gate path),
plus the criterion's own Kendall-Gal term on each frame (the branch weights its frame's point loss). Everything
else in CUT3R is frozen; xyz/pose outputs change only through what the gate lets into the state.

  python train_conf_gate.py --name cb_g1 --episodes 38633 --batch_size 1 --chunk 16 --gscale 1
Outputs <out_root>/<name>/conf_branch.pth  (load with infer_and_eval_worker.py --conf_branch / token_gate.conf_branch)
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import dust3r.utils.path_to_croco  # noqa
from dust3r.model import ARCroco3DStereo
from dust3r.losses import *  # noqa: F401,F403 (criterion strings are eval'd in this namespace)
from dust3r.datasets import get_data_loader, DL3DV_Multi  # noqa: F401
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from train_unc_gate import build_loader, set_epoch, to_device, CKPT_DEFAULT, TRAIN_ROOT, TEST_ROOT

OUT_ROOT = "/gpfs/scratch/etur59/koc821022/checkpoints/unc_gate"
CRITERION = "ConfLoss(Regr3DPoseBatchList(L21, norm_mode='?avg_dis'), alpha=0.2) + RGBLoss(MSE)"


def set_gate_keys(batch, q, gmin, soft, gscale, offg=1.0):
    for v in batch:
        B = v["img"].shape[0]; dev = v["img"].device
        v["token_gate_q"] = torch.full((B,), float(q), device=dev)
        v["token_gate_gmin"] = torch.full((B,), float(gmin), device=dev)
        v["token_gate_align"] = torch.full((B,), 1.0, device=dev)
        v["token_gate_offg"] = torch.full((B,), float(offg), device=dev)
        v["token_gate_soft"] = torch.full((B,), float(soft), device=dev)
        v["token_gate_gscale"] = torch.full((B,), float(gscale), device=dev)


def run_batch(model, batch, criterion, chunk, accelerator, opt, params, train=True, stats=None):
    """TBPTT over one batch of episodes: state detached at chunk boundaries, gradient inside a chunk flows
    from later frames' losses through the frozen decoder into the confidence branch of earlier frames."""
    with torch.no_grad():
        (feat, pos, shape), (init_state_feat, init_mem, state_feat, state_pos, mem) = model._forward_encoder(batch)
    T = len(batch); losses = []
    for c0 in range(0, T, chunk):
        state_feat = state_feat.detach(); mem = mem.detach()
        preds, views = [], []
        with torch.set_grad_enabled(train):
            for t in range(c0, min(c0 + chunk, T)):
                res_group, (state_feat, mem) = model._forward_decoder_group_step(
                    views=batch, view_indices=[t], feat_group=[feat[t]], pos_group=[pos[t]], shape_group=[shape[t]],
                    init_state_feat=init_state_feat, init_mem=init_mem, state_feat=state_feat, state_pos=state_pos, mem=mem)
                preds.append(res_group[0]); views.append(batch[t])
            with torch.autocast("cuda", enabled=False):
                loss, details = criterion(views, preds, camera1=batch[0]["camera_pose"])
        if train:
            opt.zero_grad(set_to_none=True); accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        losses.append(float(loss.detach()))
        if stats is not None:
            with torch.no_grad():
                cs = torch.stack([p["conf_self"].float().mean() for p in preds]).mean()
                stats.setdefault("conf_mean", []).append(float(cs)); stats.setdefault("pose_loss", []).append(float(details.get("pose_loss", float("nan"))))
        del preds, views, loss
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--episodes", type=int, default=38633)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_views", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=16, help="TBPTT chunk = gradient horizon in frames")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--q", type=float, default=0.5); ap.add_argument("--gmin", type=float, default=0.5)
    ap.add_argument("--soft", type=float, default=0.5, help="temperature T of the soft rule, in log(conf) units")
    ap.add_argument("--gscale", type=float, default=1.0, help="gradient scale of the gate path vs the own-frame K&G term")
    ap.add_argument("--eval_every", type=int, default=2000); ap.add_argument("--eval_episodes", type=int, default=16)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = os.path.join(args.out_root, args.name); os.makedirs(out_dir, exist_ok=True)
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=3))])  # GPFS stalls during eval loading exceeded the 10-min NCCL default; device = accelerator.device; is_main = accelerator.is_main_process
    log = open(os.path.join(out_dir, "log.txt"), "a") if is_main else None
    def P(*a):
        if is_main:
            msg = " ".join(str(x) for x in a); print(msg, flush=True); log.write(msg + "\n"); log.flush()
    P(f"[conf_gate] {time.strftime('%F %T')} procs={accelerator.num_processes} args={vars(args)}")
    model = ARCroco3DStereo.from_pretrained(args.ckpt).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    if getattr(model, "pose_gru", None) is not None:
        model.pose_gru = None
    branch = model.attach_conf_branch(None, train=True)
    model.gradient_checkpointing_enable(); model.train()  # checkpointing needs train mode; no dropout/BN in CUT3R
    params = [p for p in branch.parameters() if p.requires_grad]
    P(f"[conf_gate] frozen model; trainable conf branch {sum(p.numel() for p in params)/1e6:.3f}M params; criterion {CRITERION}")
    criterion = eval(CRITERION).to(device)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    train_loader = build_loader(TRAIN_ROOT, args.num_views, args.batch_size, args.num_workers, accelerator)
    eval_loader = build_loader(TEST_ROOT, args.num_views, 1, max(2, args.num_workers // 2), accelerator)
    branch, opt, train_loader, eval_loader = accelerator.prepare(branch, opt, train_loader, eval_loader)
    n_proc = accelerator.num_processes
    ck_path = os.path.join(out_dir, "conf_branch.pth")

    def save(step, seen):
        if is_main:
            torch.save({"conf": accelerator.unwrap_model(branch).state_dict(), "args": vars(args), "step": step, "seen": seen,
                        "opt": opt.state_dict()}, ck_path)

    def evaluate(tag):
        set_epoch(eval_loader, 0); ls = []; st = {}
        for i, b in enumerate(eval_loader):
            if i >= max(1, args.eval_episodes // n_proc):
                break
            b = to_device(b, device); set_gate_keys(b, args.q, args.gmin, args.soft, 1.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls += run_batch(model, b, criterion, args.chunk, accelerator, opt, params, train=False, stats=st)
        t = torch.tensor([sum(ls) / max(len(ls), 1), float(len(ls)), sum(st.get("conf_mean", [0])) / max(len(st.get("conf_mean", [1])), 1),
                          sum(st.get("pose_loss", [0])) / max(len(st.get("pose_loss", [1])), 1)], device=device)
        g = accelerator.gather(t.reshape(1, 4)); rec = {"tag": tag, "loss": float(g[:, 0].mean()), "pose_loss": float(g[:, 3].mean()),
                                                         "conf_self_mean": float(g[:, 2].mean()), "n_chunks": int(g[:, 1].sum())}
        P(f"[conf_gate] EVAL {json.dumps(rec)}")
        if is_main:
            open(os.path.join(out_dir, "eval.json"), "a").write(json.dumps(rec) + "\n")

    seen = step = 0; t0 = time.time(); run = []
    if args.resume and os.path.isfile(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(branch).load_state_dict(ck["conf"])
        try: opt.load_state_dict(ck["opt"])
        except Exception as e: P(f"[conf_gate] resume: opt state not restored ({e})")
        seen, step = int(ck.get("seen", 0)), int(ck.get("step", 0)); P(f"[conf_gate] RESUMED seen={seen} step={step}")
        if seen >= args.episodes:
            evaluate("final"); save(step, seen); return
    else:
        evaluate("init")
    epoch = args.seed + (1 if seen > 0 else 0); done = False
    while not done:
        set_epoch(train_loader, epoch); epoch += 1
        for b in train_loader:
            b = to_device(b, device); set_gate_keys(b, args.q, args.gmin, args.soft, args.gscale)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls = run_batch(model, b, criterion, args.chunk, accelerator, opt, params, train=True)
            step += len(ls); run += ls; seen += args.batch_size * n_proc
            if seen % (10 * args.batch_size * n_proc) == 0:
                P(f"[conf_gate] ep {seen}/{args.episodes} step {step} loss {sum(run[-40:])/max(len(run[-40:]),1):.4f} {(time.time()-t0)/60:.1f} min")
            if seen % args.save_every == 0:
                save(step, seen)
            if seen % args.eval_every == 0:
                evaluate(f"ep{seen}")
            if seen >= args.episodes:
                done = True; break
    evaluate("final"); save(step, seen)
    P(f"[conf_gate] done: {seen} episodes, {step} steps, {(time.time()-t0)/60:.1f} min -> {ck_path}")


if __name__ == "__main__":
    main()
