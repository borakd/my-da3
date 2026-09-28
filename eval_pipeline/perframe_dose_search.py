#!/usr/bin/env python3
"""Per-frame write-dose oracle for ONE scene (GT-driven headroom measurement, not a method).

For every frame t of the scene, find the write weight that minimises the scene's ATE, for one of the two
write sites of WRITE_DOSE_MECHANISM.md:

    site=state : S_t = a_t * S~_t + (1 - a_t) * S_{t-1}          (a_0 = 1)
    site=mem   : M_t = b_t * Write(M_{t-1}, k_t, v_t) + (1 - b_t) * M_{t-1}   (b_0 = 1)

The weight of frame t only changes frames t+1..T-1 (frame t's own depth and pose are read out before its
write), so "the weight that minimises error at frame t" is measured on the frames it influences. The search
is one or more forward passes of coordinate descent on the full-scene ATE (the number the evaluator
reports: Sim(3)-Umeyama on camera centres, RMSE):

    pass p, frame t = 1..T-2:
        frames < t keep the weights already chosen in this pass,
        frames > t keep the incumbent schedule (pass 1: the init; pass p > 1: pass p-1's result),
        each grid weight is tried for frame t and the WHOLE remainder of the scene is rolled out,
        and the weight with the lowest full-scene ATE is kept.

The incumbent weight is always in the grid, so the scene ATE is non-increasing along a pass. Frame T-1's
weight changes nothing and keeps its incumbent value; frame 0 is always written in full.

Batch 1 throughout (candidates run one after another), NOT batched: perframe_dose_diag.py showed that
zero-shot CUT3R is chaotic on this scene -- running the same plain rollout at batch 2 instead of 1 moves
frame-1 centres by 3e-5 and the scene ATE from .0690 to .0752, and a 1e-7 relative state perturbation at
frame 1 moves it to .0608. At batch 1 the harness equals the worker bit for bit, so every schedule found
here reproduces exactly when the worker scores it.

--placebo replaces the grid by 1 - k*1e-6 (k = 0..10): the same search, the same budget, weights that do
nothing systematic. Its ATE gain is what the search gets from steering the chaos alone.

Also run first: the per-scene CONSTANT sweep over the same grid (every frame t >= 1 at the same weight),
and a batch-1 plain rollout checked against the stored plain predictions of the worker (harness parity).

Outputs (in --out_dir): search_<site>_<tag>.json (grid, constant sweep, per pass: schedule, the full
ATE landscape [t][k], best ATE after each t) and maks_arm_pf_<site>_<tag>_p<p>.json control files for the
standard worker (per-scene frame_gate/mem_gate "alpha" schedules).
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--site", choices=["state", "mem"], required=True)
ap.add_argument("--init", default="1.0", help="incumbent schedule of pass 1: a float, or 'best' = the scene's best constant")
ap.add_argument("--tag", required=True)
ap.add_argument("--passes", type=int, default=2)
ap.add_argument("--grid", default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.65,0.8,1.0")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--scenes_root", default="/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist")
ap.add_argument("--plain_preds", default="", help="worker preds dir of the plain run of this backbone (parity check)")
ap.add_argument("--cut3r_dir", default="/gpfs/home/koc/koc821022/maks_idea/src/CUT3R")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--size", type=int, default=320)
ap.add_argument("--max_t", type=int, default=0, help="smoke: stop pass 1 after this many frames")
ap.add_argument("--placebo", action="store_true", help="grid = 1 - k*1e-6, k=0..10 (chaos control)")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

sys.path.insert(0, args.cut3r_dir)
from add_ckpt_path import add_path_to_dust3r  # noqa: E402
add_path_to_dust3r(args.ckpt)
import demo  # noqa: E402
from src.dust3r.model import ARCroco3DStereo  # noqa: E402  (same module path as the worker: duplicate-module trap)
from src.dust3r.wgate.traj_loss import causal_ate  # noqa: E402

dev = "cuda"
# NOTE: do not touch TF32 -- croco.py enables it at import and the worker runs with it on; overriding it here
# broke bit-parity with the worker (plain ATE .0730 vs .0690) on this chaotic scene.
grid = [1.0 - k * 1e-6 for k in range(11)] if args.placebo else [float(x) for x in args.grid.split(",")]
K = len(grid)
log = lambda *a: print(f"[pfdose {args.site}/{args.tag}]", *a, flush=True)

# ---------------------------------------------------------------- data (exactly as the worker builds it)
rgb_dir = os.path.join(args.scenes_root, args.scene, "dense", "rgb")
img_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")) + glob.glob(os.path.join(rgb_dir, "*.jpg")))
T = len(img_paths)
cams = sorted(glob.glob(os.path.join(args.scenes_root, args.scene, "dense", "cam", "*.npz")))
assert len(cams) == T, (len(cams), T)
gt = torch.tensor(np.stack([np.load(c)["pose"][:3, 3] for c in cams]), dtype=torch.float64)  # (T,3) c2w centres

model = ARCroco3DStereo.from_pretrained(args.ckpt).to(dev)
if getattr(model, "pose_gru", None) is not None:
    model.pose_gru = None
model.eval()
views = demo.prepare_input(img_paths=img_paths, img_mask=[True] * T, size=args.size, revisit=1, update=True)
for v in views:
    for k_, x in v.items():
        if k_ in ("depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"):
            continue
        if isinstance(x, (list, tuple)):
            v[k_] = [y.to(dev) for y in x]
        elif torch.is_tensor(x):
            v[k_] = x.to(dev)

with torch.no_grad():
    shape, feat_ls, pos = model._encode_views(views)
    feat = feat_ls[-1]
    S0, SP = model._init_state(feat[0], pos[0])
    M0 = model.pose_retriever.mem.expand(feat[0].shape[0], -1, -1).clone()
    S_init, M_init = S0.clone(), M0.clone()


def rep(x, n):
    return x if x.shape[0] == n else x.repeat(n, *([1] * (x.dim() - 1)))


@torch.no_grad()
def step(t, S, M):
    """One plain decoder step of frame t for a batch of states; returns (centres (n,3), proposal S~, updated M)."""
    n = S.shape[0]
    res_group, (S_new, M_new) = model._forward_decoder_group_step(
        views=views, view_indices=[t], feat_group=[rep(feat[t], n)], pos_group=[rep(pos[t], n)],
        shape_group=[rep(shape[t], n)], init_state_feat=rep(S_init, n), init_mem=rep(M_init, n),
        state_feat=S, state_pos=rep(SP, n), mem=M)
    return res_group[0]["camera_pose"][:, :3].double(), S_new, M_new


def commit(S_prev, M_prev, S_new, M_new, w):
    """Blend the frame's write at weight w ((n,) tensor) at the searched site; the other site writes in full."""
    if args.site == "state":
        wv = w.to(S_new.dtype)[:, None, None]
        return wv * S_new + (1 - wv) * S_prev, M_new
    wv = w.to(M_new.dtype)[:, None, None]
    return S_new, wv * M_new + (1 - wv) * M_prev


@torch.no_grad()
def rollout(t0, S, M, sched):
    """Roll frames t0..T-1 from the (n-batched) states entering t0; sched: (n, T) weights. Returns (n, T-t0, 3)."""
    out = []
    for u in range(t0, T):
        c, S_new, M_new = step(u, S, M)
        out.append(c)
        if u < T - 1:
            S, M = commit(S, M, S_new, M_new, sched[:, u])
    return torch.stack(out, 1)


def ate_of(cent):
    return causal_ate(cent, rep(gt[None].to(cent.device), cent.shape[0]), reduce="none").cpu().numpy()


result = {"scene": args.scene, "site": args.site, "tag": args.tag, "grid": grid, "T": T, "ckpt": args.ckpt}

# ---------------------------------------------------------------- frame 0 (always written in full) + parity
t_s = time.time()
c0, S1, M1 = step(0, S0, M0)  # state/mem after frame 0's full write
torch.cuda.synchronize()
plain1 = torch.cat([c0[:, None], rollout(1, S1, M1, torch.ones(1, T, device=dev))], 1)  # batch-1 plain
t_plain = time.time() - t_s
result["plain_ate_harness"] = float(ate_of(plain1)[0])
if args.plain_preds and os.path.isdir(args.plain_preds):
    pc = sorted(glob.glob(os.path.join(args.plain_preds, "camera", "*.npz")))
    if len(pc) == T:
        stored = torch.tensor(np.stack([np.load(p)["pose"][:3, 3] for p in pc]), dtype=torch.float64)
        dmax = float((stored - plain1[0].cpu()).abs().max())
        result["parity_plain_maxabs_centre"] = dmax
        result["plain_ate_stored"] = float(ate_of(stored[None])[0])
        log(f"parity vs stored plain preds: max |centre diff| {dmax:.3e}; ATE harness {result['plain_ate_harness']:.6f}"
            f" stored {result['plain_ate_stored']:.6f}")
log(f"plain batch-1 rollout {t_plain:.1f}s  ATE {result['plain_ate_harness']:.6f}")

# ---------------------------------------------------------------- per-scene constant sweep
t_s = time.time()
wgrid = torch.tensor(grid, dtype=torch.float64, device=dev)
sweep = np.array([ate_of(torch.cat([c0[:, None], rollout(1, S1, M1, torch.full((1, T), w, dtype=torch.float64, device=dev))], 1))[0]
                  for w in grid])
t_batch_step = (time.time() - t_s) / (T - 1) / K
result["const_sweep"] = {f"{w:g}": float(a) for w, a in zip(grid, sweep)}
kbest = int(np.argmin(sweep))
result["const_best"] = {"w": grid[kbest], "ate": float(sweep[kbest])}
log("constant sweep:", " ".join(f"{w:.7g}:{a:.5f}" for w, a in zip(grid, sweep)), f"| best {grid[kbest]:.7g} {sweep[kbest]:.5f}")
log(f"batch-1 step {t_batch_step*1000:.0f} ms; one pass ~{K*t_batch_step*(T*T/2)/60:.1f} min")

# ---------------------------------------------------------------- coordinate descent
init = grid[kbest] if args.init == "best" else float(args.init)
assert any(abs(init - g) < 1e-12 for g in grid), f"init {init} must be on the grid"
result["init"] = init
incumbent = np.full(T, init)
incumbent[0] = 1.0
passes = []
for p in range(1, args.passes + 1):
    t_s = time.time()
    sched = incumbent.copy()
    S, M = S1, M1  # entering frame 1
    prefix = [c0]
    land, best_after, drift = [], [], []
    prev_best = None
    last = T - 2 if not args.max_t else min(T - 2, args.max_t)
    for t in range(1, last + 1):
        c_t, S_new, M_new = step(t, S, M)
        sch = torch.tensor(sched, dtype=torch.float64, device=dev)[None]
        pre = torch.stack(prefix + [c_t], 1)
        ates, cands = [], []
        for kk in range(K):
            Sk, Mk = commit(S, M, S_new, M_new, wgrid[kk:kk + 1])
            ates.append(ate_of(torch.cat([pre, rollout(t + 1, Sk, Mk, sch)], 1))[0])
            cands.append((Sk, Mk))
        ates = np.array(ates)
        k = int(np.argmin(ates))
        kin = int(np.argmin([abs(g - sched[t]) for g in grid]))
        if prev_best is not None:
            drift.append(float(ates[kin] - prev_best))  # same trajectory as last iteration's pick: batch-numerics drift
        sched[t] = grid[k]
        prev_best = float(ates[k])
        land.append([float(a) for a in ates])
        best_after.append(prev_best)
        S, M = cands[k]
        del cands
        prefix.append(c_t)
        if t % 16 == 0 or t == last:
            log(f"pass {p} t={t:3d} w={grid[k]:.7g} ATE {prev_best:.5f}  ({time.time()-t_s:.0f}s)")
    # batch-1 check of the final schedule
    fin = torch.cat([c0[:, None], rollout(1, S1, M1, torch.tensor(sched, dtype=torch.float64, device=dev)[None])], 1)
    fin_ate = float(ate_of(fin)[0])
    passes.append({"pass": p, "schedule": [float(x) for x in sched], "landscape": land, "best_after_t": best_after,
                   "final_ate_batch1": fin_ate, "drift_maxabs": float(np.max(np.abs(drift))) if drift else 0.0,
                   "seconds": time.time() - t_s})
    log(f"pass {p} done: ATE {best_after[-1]:.5f} (full re-roll {fin_ate:.5f}); mean w {np.mean(sched[1:T-1]):.7g}; "
        f"max incumbent drift {passes[-1]['drift_maxabs']:.2e} (0 expected at batch 1); {time.time()-t_s:.0f}s")
    # worker control file: per-scene per-frame schedule at this site (frame 0 is written in full by the model)
    key = "frame_gate" if args.site == "state" else "mem_gate"
    arm = f"pf_{args.site}_{args.tag}_p{p}"
    json.dump({args.scene: {key: {"alpha": {str(t): float(sched[t]) for t in range(1, T)}}}},
              open(os.path.join(args.out_dir, f"maks_arm_{arm}.json"), "w"))
    incumbent = sched
    result["passes"] = passes
    json.dump(result, open(os.path.join(args.out_dir, f"search_{args.site}_{args.tag}.json"), "w"))
    if args.max_t:
        break
# constant-best control file for the worker (per-scene constant at this site)
key = "frame_gate" if args.site == "state" else "mem_gate"
if not args.placebo:
    json.dump({args.scene: {key: {"const": float(result["const_best"]["w"])}}},
              open(os.path.join(args.out_dir, f"maks_arm_pf_{args.site}_constbest.json"), "w"))
log("done")
