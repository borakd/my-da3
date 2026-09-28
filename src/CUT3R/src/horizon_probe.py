#!/usr/bin/env python3
"""Horizon probe for the consequence-trained state gate (why zero-shot training ran to a = 1).

Hypothesis: the consequence trainer cannot see the long-horizon benefit of a low write dose, because
(i) its episodes are 64 frames long while test scenes run 100-600 frames, and (ii) its gradient is
truncated to 16-frame chunks.  Two measurements, constant state dose a (no learned variation):

  V  value.  Roll out a constant dose, no grad, and measure the error the trainer and the evaluator see:
       train split, 64-frame episodes : J64 = mean_{tau=4..63} causal_ATE(0..tau)  (the training objective)
       test split, --test_len episodes: ATE(0..H-1) and J_H = mean_{tau=4..H-1} causal_ATE(0..tau)
                                        for every horizon H in --horizons
     -> the dose that minimises each quantity.  If (i) is right, the zero-shot optimum is near 1 at
        H = 64 and falls with H.
  G  gradient.  On the same 64-frame train episodes, the gate is a GateWeightHead with wmin = 0 whose
     last layer has zero weight and bias logit(a), i.e. exactly the constant a on every frame but
     differentiable.  dJ64/da is computed twice:
       chunk 16 : the trainer's TBPTT (state/mem detached every 16 frames), chunk losses weighted by
                  their share of the 60 scored frames so the sum is J64
       chunk 64 : one chunk, the full gradient of J64
     -> if (ii) is right, the truncated gradient points towards larger a where the full one does not.
     The finite-difference slope of V's J64 curve is the ground truth the full gradient should match.

Episodes are loaded once and every dose runs on the same frames.  Output: --out JSON + a printed summary.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_train", type=int, default=32)
ap.add_argument("--n_test", type=int, default=24)
ap.add_argument("--test_len", type=int, default=256)
ap.add_argument("--horizons", default="16,32,64,128,256")
ap.add_argument("--doses", default="1.0,0.94,0.8,0.65,0.5,0.4,0.3,0.2,0.1")
ap.add_argument("--grad_doses", default="0.94,0.8,0.5,0.3,0.1")
ap.add_argument("--grad_chunks", default="8,16,32,64")
ap.add_argument("--min_t", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dust3r.utils.path_to_croco  # noqa: F401
from dust3r.model import ARCroco3DStereo
from dust3r.datasets import get_data_loader, DL3DV_Multi  # noqa: F401
from dust3r.wgate.heads import GateWeightHead
from dust3r.wgate import traj_loss as TL
from train_unc_gate import build_loader, set_epoch, to_device, TRAIN_ROOT, TEST_ROOT
from accelerate import Accelerator

torch.manual_seed(args.seed)
acc = Accelerator()
dev = acc.device
DOSES = [float(x) for x in args.doses.split(",")]
GDOSES = [float(x) for x in args.grad_doses.split(",")]
GCH = [int(x) for x in args.grad_chunks.split(",")]
HOR = [int(x) for x in args.horizons.split(",")]
t0 = time.time()
def P(*a): print(f"[hprobe {args.tag} {time.time()-t0:6.0f}s]", *a, flush=True)

model = ARCroco3DStereo.from_pretrained(args.ckpt).to(dev)
for p in model.parameters():
    p.requires_grad_(False)
if getattr(model, "pose_gru", None) is not None:
    model.pose_gru = None
head = GateWeightHead("frame", wmin=0.0, init_logit=0.0).to(dev).float()
model.attach_frame_gate(head, wmin=0.0, train=True)
head = model.frame_gate
model.gradient_checkpointing_enable()


def rollout(batch, chunk=None, grad=False):
    """Causal rollout.  Returns centres (B,T,3) [with grad if grad]; with chunk, detaches every chunk."""
    with torch.no_grad():
        (feat, pos, shape), (init_s, init_m, s, spos, m) = model._forward_encoder(batch)
    T, out = len(batch), []
    for t in range(T):
        if chunk and t % chunk == 0:
            s, m = s.detach(), m.detach()
        with torch.set_grad_enabled(grad), torch.autocast("cuda", enabled=False):
            res, (s, m) = model._forward_decoder_group_step(
                views=batch, view_indices=[t], feat_group=[feat[t]], pos_group=[pos[t]],
                shape_group=[shape[t]], init_state_feat=init_s, init_mem=init_m,
                state_feat=s, state_pos=spos, mem=m)
        out.append({"camera_pose": res[0]["camera_pose"]})
        del res
    return out


def centres(preds):
    return TL.pred_centres(preds).float()


def causal_curve(pc, gc):
    """causal ATE(0..tau) for tau = 0..T-1, float64 on CPU, per batch element -> (B,T)."""
    pc, gc = pc.detach().double().cpu(), gc.detach().double().cpu()
    T = pc.shape[1]
    cur = torch.zeros(pc.shape[0], T, dtype=torch.float64)
    for tau in range(args.min_t, T):
        cur[:, tau] = TL.causal_ate(pc[:, :tau + 1], gc[:, :tau + 1], reduce="none")
    return cur


def set_const(batch, a):
    for v in batch:
        B = v["img"].shape[0]
        for k in ("frame_gate_on", "frame_gate_wmin"):
            v.pop(k, None)
        v["frame_gate_const"] = torch.full((B,), float(a), device=dev)


def set_head(batch, a):
    with torch.no_grad():
        head.out.weight.zero_()
        head.out.bias.fill_(math.log(a / (1 - a)))
    for v in batch:
        B = v["img"].shape[0]
        v.pop("frame_gate_const", None)
        v["frame_gate_on"] = torch.ones(B, device=dev)
        v["frame_gate_wmin"] = torch.zeros(B, device=dev)


def J_of(pc_or_curve, H):
    c = pc_or_curve
    return c[:, args.min_t:H].mean(1)


def grad_dJda(batch, a, chunk):
    """dJ64/da via the bias of the head's last layer (every frame's a_t = sigmoid(bias))."""
    set_head(batch, a)
    head.zero_grad(set_to_none=True)
    model.train()
    gc = TL.gt_centres(batch).float().to(dev)
    T = len(batch)
    preds_all, n_scored = [], T - args.min_t
    with torch.no_grad():
        (feat, pos, shape), (init_s, init_m, s, spos, m) = model._forward_encoder(batch)
    for c0 in range(0, T, chunk):
        s, m = s.detach(), m.detach()
        c1 = min(c0 + chunk, T)
        preds = []
        for t in range(c0, c1):
            res, (s, m) = model._forward_decoder_group_step(
                views=batch, view_indices=[t], feat_group=[feat[t]], pos_group=[pos[t]],
                shape_group=[shape[t]], init_state_feat=init_s, init_mem=init_m,
                state_feat=s, state_pos=spos, mem=m)
            preds.append({"camera_pose": res[0]["camera_pose"]})
            del res
        pc = centres(preds)
        prev = torch.cat(preds_all, 1).detach() if preds_all else None
        loss = 0.0
        for tau in range(max(c0, args.min_t), c1):
            full = torch.cat([prev, pc[:, :tau - c0 + 1]], 1) if prev is not None else pc[:, :tau + 1]
            loss = loss + TL.causal_ate(full, gc[:, :tau + 1])
        if torch.is_tensor(loss):
            (loss / n_scored).backward()
        preds_all.append(pc.detach())
        del preds, pc
    g_bias = float(head.out.bias.grad)
    model.eval()
    return g_bias / (a * (1 - a))  # dJ/du * du/da, u = logit(a)


def load(root, n_views, n):
    ld = acc.prepare(build_loader(root, n_views, 1, 4, acc))
    set_epoch(ld, 0)
    if hasattr(ld, "set_epoch"):
        ld.set_epoch(0)
    out = []
    for i, b in enumerate(ld):
        if i >= n:
            break
        out.append(b)
    return out


R = {"tag": args.tag, "ckpt": args.ckpt, "doses": DOSES, "grad_doses": GDOSES, "horizons": HOR,
     "train": [], "test": []}

# ---- train split, 64-frame episodes: value J64 per dose + gradient at chunk 16 / 64
train_eps = load(TRAIN_ROOT, 64, args.n_train)
P(f"loaded {len(train_eps)} train episodes of 64 frames")
model.eval()
for i, b in enumerate(train_eps):
    b = to_device(b, dev)
    gc = TL.gt_centres(b).float().to(dev)
    rec = {"J64": {}, **{f"g{c}": {} for c in GCH}}
    for a in DOSES:
        set_const(b, a)
        with torch.no_grad():
            cur = causal_curve(centres(rollout(b)), gc)
        rec["J64"][a] = float(J_of(cur, 64)[0])
    for a in GDOSES:
        for c in GCH:
            rec[f"g{c}"][a] = grad_dJda(b, a, c)
    R["train"].append(rec)
    P(f"train ep {i}: J64 " + " ".join(f"{a}:{rec['J64'][a]:.4f}" for a in DOSES)
      + " | dJ/da " + " ".join(f"{a}:" + "|".join(f"{rec[f'g{c}'][a]:+.3g}" for c in GCH) for a in GDOSES))
    json.dump(R, open(args.out, "w"))
    del b
    torch.cuda.empty_cache()

# ---- test split, long episodes: ATE(0..H-1) and J_H per dose
test_eps = load(TEST_ROOT, args.test_len, args.n_test)
P(f"loaded {len(test_eps)} test episodes of {args.test_len} frames")
for i, b in enumerate(test_eps):
    b = to_device(b, dev)
    gc = TL.gt_centres(b).float().to(dev)
    rec = {"ATE": {}, "J": {}}
    for a in DOSES:
        set_const(b, a)
        with torch.no_grad():
            cur = causal_curve(centres(rollout(b)), gc)
        rec["ATE"][a] = {H: float(cur[0, H - 1]) for H in HOR}
        rec["J"][a] = {H: float(J_of(cur, H)[0]) for H in HOR}
    R["test"].append(rec)
    P(f"test ep {i}: ATE@H " + " ".join(f"{a}:" + "/".join(f"{rec['ATE'][a][H]:.3f}" for H in HOR) for a in DOSES))
    json.dump(R, open(args.out, "w"))
    del b
    torch.cuda.empty_cache()

# ---- summary
def best(vals):
    return DOSES[int(np.argmin(vals))]

tr = R["train"]
J64 = np.array([[r["J64"][a] for a in DOSES] for r in tr])
P("SUMMARY train J64 mean per dose: " + " ".join(f"{a}:{v:.5f}" for a, v in zip(DOSES, J64.mean(0)))
  + f"  -> argmin {best(J64.mean(0))}")
fd = {}  # finite-difference slope of mean J64 between neighbouring doses, at their midpoint
Jm = J64.mean(0)
for k in range(len(DOSES) - 1):
    fd[(DOSES[k] + DOSES[k + 1]) / 2] = (Jm[k] - Jm[k + 1]) / (DOSES[k] - DOSES[k + 1])
P("SUMMARY finite-difference dJ64/da (value curve): " + " ".join(f"@{m:.3f}:{v:+.5f}" for m, v in fd.items()))
for a in GDOSES:
    parts = []
    for c in GCH:
        g = np.array([r[f"g{c}"][a] for r in tr])
        parts.append(f"chunk{c}: median {np.median(g):+.4g} mean {g.mean():+.4g} median|g| {np.median(np.abs(g)):.3g} "
                     f"pos {(g > 0).mean():.2f}")
    P(f"SUMMARY dJ64/da at a={a}: " + " ; ".join(parts))
te = R["test"]
for H in HOR:
    A = np.array([[r["ATE"][a][H] for a in DOSES] for r in te]); Jh = np.array([[r["J"][a][H] for a in DOSES] for r in te])
    P(f"SUMMARY test H={H}: ATE(0..H-1) " + " ".join(f"{a}:{v:.4f}" for a, v in zip(DOSES, A.mean(0)))
      + f" -> argmin {best(A.mean(0))};  J_H argmin {best(Jh.mean(0))}")
json.dump(R, open(args.out, "w"))
P("done")
