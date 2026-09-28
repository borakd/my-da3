#!/usr/bin/env python3
"""Diagnose the per-frame search harness mismatch: determinism, batch cross-talk, amplification.

A  worker path (inference(views) -> model.forward) twice, vs the stored plain preds
B  step loop, batch 1, twice
C  step loop, batch 2 with identical rows
D  step loop, batch 2, row 1 perturbed (state write at 0.1 from frame 1): does row 0 still equal B?
E  step loop, batch 1, state perturbed by 1e-6 relative noise at frame 1: growth of the divergence
Prints per-frame max |centre diff| at frames 1, 2, 4, 8, ..., 127 and the scene ATE of each run.
"""
import glob, os, sys, time
import numpy as np
import torch

S_ = "RAIL+80edfcb1+2023-07-14-14h-28m-45s"
CK = "/gpfs/home/koc/koc821022/using_ext_cams/src/CUT3R/src/cut3r_512_dpt_4_64.pth"
ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
STORED = f"/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/cut3r_zeroshot/preds/{S_}/camera"
sys.path.insert(0, "/gpfs/home/koc/koc821022/using_ext_cams/src/CUT3R")
from add_ckpt_path import add_path_to_dust3r
add_path_to_dust3r(CK)
import demo
from src.dust3r.inference import inference
from src.dust3r.model import ARCroco3DStereo
from src.dust3r.wgate.traj_loss import causal_ate

dev = "cuda"
imgs = sorted(glob.glob(f"{ROOT}/{S_}/dense/rgb/*.png") + glob.glob(f"{ROOT}/{S_}/dense/rgb/*.jpg"))
T = len(imgs)
gt = torch.tensor(np.stack([np.load(c)["pose"][:3, 3] for c in sorted(glob.glob(f"{ROOT}/{S_}/dense/cam/*.npz"))]), dtype=torch.float64)
stored = torch.tensor(np.stack([np.load(p)["pose"][:3, 3] for p in sorted(glob.glob(f"{STORED}/*.npz"))]), dtype=torch.float64)
model = ARCroco3DStereo.from_pretrained(CK).to(dev).eval()
print("views_per_step", getattr(model, "views_per_step", None), "pose_head_flag", model.pose_head_flag, flush=True)
print("cudnn.deterministic", torch.backends.cudnn.deterministic, "benchmark", torch.backends.cudnn.benchmark,
      "sdpa flash", torch.backends.cuda.flash_sdp_enabled(), "mem_eff", torch.backends.cuda.mem_efficient_sdp_enabled(), flush=True)
ate = lambda c: float(causal_ate(c[None].double().cpu(), gt[None], reduce="none")[0])
FR = [1, 2, 4, 8, 16, 32, 64, 127]
def cmp(name, a, b):
    d = (a.double().cpu() - b.double().cpu()).abs().amax(-1)
    print(f"{name:34s} " + " ".join(f"{d[f]:.1e}" for f in FR) + f" | ATE {ate(a):.5f} vs {ate(b):.5f}", flush=True)

def worker_run():
    with open(os.devnull, "w") as dn:
        import contextlib
        with contextlib.redirect_stdout(dn):
            v = demo.prepare_input(img_paths=imgs, img_mask=[True] * T, size=320, revisit=1, update=True)
            with torch.no_grad():
                out, _ = inference(v, model, dev)
    return torch.cat([p["camera_pose"][:, :3] for p in out["pred"]]).double()

A1 = worker_run(); A2 = worker_run()
print("frames:", FR, flush=True)
cmp("A1 worker vs stored", A1, stored)
cmp("A2 worker vs A1 worker", A2, A1)

views = demo.prepare_input(img_paths=imgs, img_mask=[True] * T, size=320, revisit=1, update=True)
for v in views:
    for k, x in v.items():
        if k in ("depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"):
            continue
        if torch.is_tensor(x):
            v[k] = x.to(dev)
with torch.no_grad():
    shape, feat_ls, pos = model._encode_views(views)
    feat = feat_ls[-1]
    S0, SP = model._init_state(feat[0], pos[0])
    M0 = model.pose_retriever.mem.expand(feat[0].shape[0], -1, -1).clone()
rep = lambda x, n: x if x.shape[0] == n else x.repeat(n, *([1] * (x.dim() - 1)))

@torch.no_grad()
def loop(n, pert=None, noise=0.0):
    S, M = rep(S0, n), rep(M0, n)
    Si, Mi = S.clone(), M.clone()
    out = []
    for t in range(T):
        rg, (Sn, Mn) = model._forward_decoder_group_step(views=views, view_indices=[t], feat_group=[rep(feat[t], n)],
            pos_group=[rep(pos[t], n)], shape_group=[rep(shape[t], n)], init_state_feat=Si, init_mem=Mi,
            state_feat=S, state_pos=rep(SP, n), mem=M)
        out.append(rg[0]["camera_pose"][:, :3].double())
        if pert is not None and t >= 1:
            w = torch.tensor(pert, device=dev, dtype=Sn.dtype)[:, None, None]
            Sn = w * Sn + (1 - w) * S
        if noise and t == 1:
            Sn = Sn * (1 + noise * torch.randn_like(Sn))
        S, M = Sn, Mn
    return torch.stack(out, 1)  # (n,T,3)

B1 = loop(1)[0]; B2 = loop(1)[0]
cmp("B1 loop-b1 vs A1 worker", B1, A1)
cmp("B2 loop-b1 vs B1", B2, B1)
C = loop(2)
cmp("C row0 (b2 identical) vs B1", C[0], B1)
cmp("C row1 vs C row0", C[1], C[0])
D = loop(2, pert=[1.0, 0.1])
cmp("D row0 (b2, row1 perturbed) vs B1", D[0], B1)
cmp("D row0 vs C row0", D[0], C[0])
E11 = loop(11)
cmp("E b11 identical row0 vs B1", E11[0], B1)
cmp("E b11 row5 vs row0", E11[5], E11[0])
for nz in (1e-7, 1e-5, 1e-3):
    torch.manual_seed(0)
    N = loop(1, noise=nz)[0]
    cmp(f"noise {nz:g} at t=1 vs B1", N, B1)
