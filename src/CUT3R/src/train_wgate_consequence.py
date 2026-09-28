#!/usr/bin/env python
"""Variant C of WRITE_GATE_VARIANTS.md: train a write gate by its CONSEQUENCE.

Variants 1 and 2 learn a Kendall & Gal sigma that PREDICTS a frame's own error and use it as a proxy
for the value of that frame's write; their oracles closed that family.  Variant C drops the proxy: the
per-frame gate weight is trained by the gradient of a CAUSAL trajectory error with respect to the
weight, flowing back through the memory and the FROZEN decoder.

    C1  --site state : per-frame scalar a_t on the 768-token state commit   (state only, as V1)
    C2  --site mem   : per-frame scalar b_t on the retriever-memory commit  (as V2)

Everything except the gate head is frozen.  The head is `dust3r.wgate.heads.GateWeightHead`
(Component A): same inputs as the corresponding sigma head, but its output IS the weight
(`w = wmin + (1 - wmin) * sigmoid(u)`), with the last layer zeroed and its bias at `--init_logit`
so the run starts at w ~ .94, i.e. essentially plain CUT3R, and training can only move it.
The loss is `dust3r.wgate.traj_loss.chunk_traj_loss` (Component B): causal Sim(3) ATE on the camera
centres 0..t (pre-chunk centres enter detached), optionally plus a consecutive-frame RPE term.

Optimisation is TBPTT, exactly the shape of `train_conf_gate.py`: the encoder runs once per episode
under `no_grad`, then the decoder is stepped frame by frame; at every `--chunk` boundary BOTH the
state and the retriever memory are detached, the chunk's loss is backpropagated through the frozen
decoder into the gate head of the earlier frames of that chunk, and one AdamW step is taken.

    # self-test (CPU, no checkpoint, no data)
    python train_wgate_consequence.py --selftest
    # smoke (1 GPU): 2 episodes, 1 chunk each, 1 held-out episode
    python train_wgate_consequence.py --name cgsmoke --site state --smoke
    # real runs (4 GPUs via accelerate; see eval_pipeline/train_wgate_consequence.sbatch)
    NAME=cg1_head SITE=state             sbatch eval_pipeline/train_wgate_consequence.sbatch
    NAME=cg2_head SITE=mem               sbatch eval_pipeline/train_wgate_consequence.sbatch
    NAME=cg1_head_lr5 SITE=state LR=1e-5 sbatch eval_pipeline/train_wgate_consequence.sbatch

Outputs (atomic writes) under <out>/<name>/:
    gate_head.pth       {"state_dict", "mode": "weight", "kind", "site", "wmin", "init_logit",
                         "args", "episode", "metrics"}      <- what the eval worker attaches
    gate_head_best.pth  same, at the lowest held-out ATE
    gate_head_opt.pth   {"opt", "episode", "step", "best_ate"} -- optimiser state for --resume only
    log.txt, eval.json
"""
import argparse
import inspect
import json
import math
import os
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

OUT_ROOT = "/gpfs/scratch/etur59/koc821022/checkpoints/wgate/consequence"


# --------------------------------------------------------------------------------------- imports
# Components A (GateWeightHead) and B (traj_loss) are implemented by other agents against the names
# fixed in WRITE_GATE_VARIANTS.md.  Import them explicitly and fail with a message that says which
# name is missing and who owns it, instead of an AttributeError three hours into a 4-GPU job.
_A_HINT = ("Component A owes `GateWeightHead` in src/CUT3R/src/dust3r/wgate/heads.py "
           "(WRITE_GATE_VARIANTS.md, 'Variant C ... Head').")
_B_HINT = ("Component B owes `chunk_traj_loss`, `pred_centres`, `gt_centres` in "
           "src/CUT3R/src/dust3r/wgate/traj_loss.py (WRITE_GATE_VARIANTS.md, 'Variant C ... Loss').")


def _require(modname, names, hint):
    try:
        mod = __import__(modname, fromlist=["_"])
    except ImportError as e:
        raise SystemExit(f"[wgcons] FATAL: cannot import {modname} ({e}).\n         {hint}")
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        raise SystemExit(f"[wgcons] FATAL: {modname} lacks {missing}.\n         {hint}")
    return mod


def load_components():
    """Return (GateWeightHead, traj_loss module).  Raises SystemExit with a clear message."""
    heads = _require("dust3r.wgate.heads", ["GateWeightHead"], _A_HINT)
    traj = _require("dust3r.wgate.traj_loss", ["chunk_traj_loss", "pred_centres", "gt_centres"], _B_HINT)
    return heads.GateWeightHead, traj


# ------------------------------------------------------------------------------ flexible binding
# Components A and B are written in parallel with this file.  The NAMES are fixed by the contract;
# the exact parameter names / order of `GateWeightHead.__init__` and `chunk_traj_loss` are not.
# `_flex_call` binds every parameter by NAME through an alias table and passes it as a keyword, so
# the call is insensitive to argument order.  A required parameter that cannot be resolved is a hard
# error naming it (never a silent default), and the resolved binding is printed once into the log.
_MISSING = object()

_ALIASES = {
    # predicted / GT camera centres of the current chunk, (B, n, 3).  The first two names are the
    # ones traj_loss.chunk_traj_loss actually uses; the rest are tolerated spellings.
    "pred_centres_chunk": "pred_c", "gt_centres_chunk": "gt_c",
    "pred_c": "pred_c", "pred_centres": "pred_c", "pred_centers": "pred_c", "pc": "pred_c",
    "p_c": "pred_c", "centres_pred": "pred_c", "pred_xyz": "pred_c", "pred_t": "pred_c",
    "gt_c": "gt_c", "gt_centres": "gt_c", "gt_centers": "gt_c", "gc": "gt_c",
    "g_c": "gt_c", "centres_gt": "gt_c", "gt_xyz": "gt_c", "gt_t": "gt_c",
    # detached centres of the frames BEFORE this chunk, (B, c0, 3)
    "prev_pred_centres": "prev_pred", "prev_gt_centres": "prev_gt",
    "prev_pred": "prev_pred", "prev_pred_c": "prev_pred",
    "prev_p": "prev_pred", "pred_prev": "prev_pred", "prev": "prev_pred", "prev_c": "prev_pred",
    "prev_gt": "prev_gt", "prev_gt_c": "prev_gt", "gt_prev": "prev_gt",
    # full absT_quaR encodings, (B, n, 7) -- needed by the RPE term
    "pred_enc_chunk": "pred_pose", "gt_enc_chunk": "gt_pose",
    "prev_pred_enc": "prev_pred_pose", "prev_gt_enc": "prev_gt_pose",
    "pred_pose": "pred_pose", "pred_poses": "pred_pose", "pr_poses": "pred_pose", "pred_enc": "pred_pose",
    "gt_pose": "gt_pose", "gt_poses": "gt_pose", "gt_enc": "gt_pose",
    "prev_pred_pose": "prev_pred_pose", "prev_gt_pose": "prev_gt_pose",
    # rotations (quaternions) alone, if a component asks for them separately
    "pred_q": "pred_q", "pred_quat": "pred_q", "pred_quats": "pred_q", "pred_rot": "pred_q",
    "pred_R": "pred_q", "pr_quats": "pred_q",
    "gt_q": "gt_q", "gt_quat": "gt_q", "gt_quats": "gt_q", "gt_rot": "gt_q", "gt_R": "gt_q",
    # raw objects
    "preds": "preds", "pred_list": "preds", "res": "preds", "results": "preds", "pred": "preds",
    "views": "views", "batch": "views", "gt_views": "views", "gts": "views",
    # scalars
    "t0": "t0", "c0": "t0", "chunk_start": "t0", "offset": "t0", "start": "t0", "t_start": "t0",
    "ate_min_t": "min_t", "min_t": "min_t", "t_min": "min_t", "min_frames": "min_t",
    "rpe_w": "rpe_w", "w_rpe": "rpe_w", "rpe_weight": "rpe_w", "rpe": "rpe_w",
    "return_details": "_true", "return_parts": "_true", "with_details": "_true", "details": "_true",
    # GateWeightHead
    "wmin": "wmin", "w_min": "wmin", "init_logit": "init_logit", "bias_init": "init_logit",
    "init_bias": "init_logit", "site": "site", "in_dim": "in_dim", "hidden": "hidden",
    "kind": "kind",
    "pose_decoder": "pose_decoder", "pose_head": "pose_decoder", "decoder": "pose_decoder",
    "pose_dec": "pose_decoder",
}


def _flex_call(fn, pool, what, verbose=None):
    """Call `fn`, binding each of its parameters from `pool` through `_ALIASES` (by keyword)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # a builtin / C function: fall back to the contract's order
        return fn(pool["pred_c"], pool["gt_c"])
    kwargs, used, unresolved = {}, [], []
    for p in sig.parameters.values():
        if p.name in ("self", "cls") or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        key = _ALIASES.get(p.name, _MISSING)
        if key == "_true":
            kwargs[p.name] = True
            used.append(f"{p.name}=True")
            continue
        val = _MISSING if key is _MISSING else pool.get(key, _MISSING)
        if val is _MISSING:
            if p.default is p.empty:
                unresolved.append(p.name)
            continue
        if val is None and p.default is not p.empty:
            continue  # nothing to give: let the component's own default stand
        kwargs[p.name] = val
        used.append(f"{p.name}<-{key}" + ("(None)" if val is None else ""))
    if unresolved:
        raise SystemExit(
            f"[wgcons] FATAL: cannot bind {what}{sig}: no value for required parameter(s) "
            f"{unresolved}. Known aliases: {sorted(set(_ALIASES))}. Either use a contract name for "
            f"the parameter or add the alias to _ALIASES in train_wgate_consequence.py.")
    if verbose is not None:
        verbose(f"[wgcons] bound {what}{sig} as ({', '.join(used)})")
    return fn(**kwargs)


# ------------------------------------------------------------------------------------- GT poses
def gt_pose_encodings(views):
    """(B, T, 7) absT_quaR of every view expressed in view 0's frame -- the contract's GT.

    `camera_to_pose_encoding(inv(views[0]["camera_pose"]) @ views[t]["camera_pose"])`, fp32, no grad,
    autocast off.  ATE with Sim(3) is scale invariant, so no norm factor is applied (unlike the
    finetune loss, which divides the translations by the per-sequence point norm).
    """
    from dust3r.utils.camera import camera_to_pose_encoding
    from dust3r.utils.geometry import inv
    dev_type = "cuda" if views[0]["camera_pose"].is_cuda else "cpu"
    with torch.no_grad(), torch.autocast(device_type=dev_type, enabled=False):
        in0 = inv(views[0]["camera_pose"].float())
        return torch.stack([camera_to_pose_encoding(in0 @ v["camera_pose"].float()) for v in views], dim=1)


def pred_pose_encodings(preds):
    """(B, n, 7) predicted absT_quaR (the pose head's own output, already relative to frame 0)."""
    return torch.stack([p["camera_pose"].float() for p in preds], dim=1)


def _as_bt3(x, n=None):
    """Normalise a centres return value to (B, T, 3): a list of (B,3) is stacked on dim 1."""
    if isinstance(x, (list, tuple)):
        x = torch.stack([torch.as_tensor(e) for e in x], dim=1)
    x = torch.as_tensor(x)
    if x.dim() == 2 and (n is None or x.shape[0] == n):
        x = x[None]  # (T,3) -> (1,T,3)
    assert x.dim() == 3 and x.shape[-1] == 3, f"centres have shape {tuple(x.shape)}, expected (B,T,3)"
    return x


# ------------------------------------------------------------------------------------- the head
def build_head(GateWeightHead, kind, args, pose_decoder=None, P=print):
    pool = {"kind": kind, "site": "mem" if kind == "pose" else "state",
            "wmin": float(args.wmin), "init_logit": float(args.init_logit), "hidden": 256}
    if kind == "pose":
        if pose_decoder is None:
            raise SystemExit("[wgcons] FATAL: --site mem needs the frozen pose decoder "
                             "(model.downstream_head.pose_head); none was available.")
        pool["pose_decoder"] = pose_decoder
    head = _flex_call(GateWeightHead, pool, "GateWeightHead", verbose=P)
    if sum(p.numel() for p in head.parameters()) == 0:
        raise SystemExit("[wgcons] FATAL: GateWeightHead has no trainable parameters.")
    return head


def attach_head(model, site, head, P=print):
    """attach_frame_gate / attach_mem_gate with train=True (Component A: keeps requires_grad, leaves
    the head in train mode, and does NOT detach the weight on that path).  Returns the module the
    model will actually call."""
    fn = model.attach_frame_gate if site == "state" else model.attach_mem_gate
    if "train" not in inspect.signature(fn).parameters:
        raise SystemExit(
            f"[wgcons] FATAL: {fn.__name__} has no `train` parameter. Variant C needs "
            f"`{fn.__name__}(head, train=True)` to keep the gate weight in the graph "
            f"(WRITE_GATE_VARIANTS.md, 'Model support needed'). {_A_HINT}")
    fn(head, train=True)
    attached = getattr(model, "frame_gate" if site == "state" else "mem_gate", None)
    if attached is None:
        raise SystemExit(f"[wgcons] FATAL: {fn.__name__}(head, train=True) did not attach the head.")
    trainable = [p for p in attached.parameters() if p.requires_grad]
    if not trainable:
        raise SystemExit(f"[wgcons] FATAL: {fn.__name__}(..., train=True) left the head frozen "
                         f"(no parameter has requires_grad) -- the gradient can never reach it.")
    P(f"[wgcons] attached {site} gate ({type(attached).__name__}): "
      f"{sum(p.numel() for p in trainable)} trainable params")
    return attached


def set_gate_keys(batch, site, wmin):
    """Per-view (1,) tensors that switch the gate on, exactly like the eval worker's control JSON."""
    on_key = "frame_gate_on" if site == "state" else "mem_gate_on"
    wmin_key = "frame_gate_wmin" if site == "state" else "mem_gate_wmin"
    for v in batch:
        B, dev = v["img"].shape[0], v["img"].device
        v[on_key] = torch.ones(B, device=dev)
        v[wmin_key] = torch.full((B,), float(wmin), device=dev)


# ------------------------------------------------------------------------------- traj_loss shim
def _wrap_traj(traj, say):
    """Attach adapters so the rollout never guesses a call shape twice.

    `pred_centres` / `gt_centres` are contract names but their signatures are Component B's: call them
    with the shapes the contract states, normalise the return to (B,T,3), and fall back to the
    contract's own definitions (with a loud warning) if the call shape does not fit.  `final_ate` uses
    `causal_ate` when Component B exports it and otherwise a local Sim(3) RMSE used only for logging.
    """
    said = set()

    def say_once(msg):
        if say is not None and msg not in said:
            said.add(msg)
            say(msg)
    traj.say = say_once

    def _fatal(what, e=None):
        raise SystemExit(
            f"[wgcons] FATAL: traj_loss.{what} did not behave as the contract states"
            + (f" ({type(e).__name__}: {e})" if e is not None else "")
            + f".\n         {_B_HINT}")

    def pred_centres_of(preds):
        try:
            return _as_bt3(traj.pred_centres(preds), n=len(preds))
        except Exception as e:
            _fatal("pred_centres(preds)", e)

    def gt_centres_of(views):
        last = None
        for call in (lambda: traj.gt_centres(views),
                     lambda: traj.gt_centres(views, views[0]),
                     lambda: traj.gt_centres(views, ref=views[0])):
            try:
                return _as_bt3(call(), n=len(views))
            except TypeError as e:      # a different signature: try the next spelling
                last = e
                continue
            except Exception as e:
                _fatal("gt_centres(views)", e)
        _fatal("gt_centres(views[, ref]) -- no accepted call shape", last)

    def final_ate(pred_c, gt_c):
        fn = getattr(traj, "causal_ate", None)
        if fn is None:
            raise SystemExit(f"[wgcons] FATAL: traj_loss exports no `causal_ate`, so the held-out "
                             f"metric cannot be the benchmark quantity.\n         {_B_HINT}")
        last = None
        for call in (lambda: fn(pred_c, gt_c, pred_c.shape[1] - 1), lambda: fn(pred_c, gt_c)):
            try:
                return float(torch.as_tensor(call()).reshape(-1).float().mean())
            except TypeError as e:      # a different signature: try the next spelling
                last = e
            except Exception as e:
                _fatal("causal_ate(pred_c, gt_c, t)", e)
        _fatal("causal_ate(pred_c, gt_c[, t]) -- no accepted call shape", last)

    def pred_pose_of(preds):
        fn = getattr(traj, "pred_pose_encodings", None)
        if fn is None:
            say_once("[wgcons] note: traj_loss exports no pred_pose_encodings; using the contract "
                     "definition stack(pred['camera_pose'])")
            return pred_pose_encodings(preds)
        try:
            return fn(preds)
        except Exception as e:
            _fatal("pred_pose_encodings(preds)", e)

    def gt_pose_of(views):
        fn = getattr(traj, "gt_pose_encodings", None)
        if fn is None:
            say_once("[wgcons] note: traj_loss exports no gt_pose_encodings; using the contract "
                     "definition camera_to_pose_encoding(inv(v0) @ vt)")
            return gt_pose_encodings(views)
        try:
            return fn(views)
        except Exception as e:
            _fatal("gt_pose_encodings(views)", e)

    traj.pred_centres_of = pred_centres_of
    traj.gt_centres_of = gt_centres_of
    traj.pred_pose_of = pred_pose_of
    traj.gt_pose_of = gt_pose_of
    traj.final_ate = final_ate
    return traj


# ------------------------------------------------------------------------------------- rollout
def gate_weights(model, site):
    """The weights the model actually applied this episode (t > 0; a_0 = b_0 = 1 is hard-coded).

    Read from the model's own trace rather than a forward hook on the head: in weight mode the hook
    calls `head.logit()` / `head.weight_from_logit()` directly, so `head.forward` never runs.  The
    four trace lists are reset by `_wgate_frame_pre` at every view index 0, i.e. once per episode.
    """
    tr = getattr(model, "_wgate_trace_frame" if site == "state" else "_wgate_trace_mem", None) or []
    out = []
    for e in tr:
        t, w = e[0], e[1]
        if t <= 0:
            continue
        out.extend([float(x) for x in w] if isinstance(w, (list, tuple)) else [float(w)])
    return out


def run_episode(model, batch, traj, args, accelerator, opt, params, train, site, n_proc=1,
                max_chunks=None):
    """TBPTT over one episode.  Returns (chunk_losses, gate_weights, ate, n_opt_steps).

    The encoder runs once under no_grad; the decoder is stepped one frame at a time.  At every chunk
    boundary state AND mem are detached, so the gradient horizon is exactly `--chunk` frames; inside a
    chunk it flows from the later frames' trajectory error, through the frozen decoder, into the gate
    weight of the earlier frames.  Only `camera_pose` is kept from each step's prediction: nothing
    downstream reads pts3d/conf, so dropping those references frees that whole subgraph at once.
    """
    with torch.no_grad():
        (feat, pos, shape), (init_state_feat, init_mem, state_feat, state_pos, mem) = \
            model._forward_encoder(batch)
    T = len(batch)
    gt_pose = traj.gt_pose_of(batch)                        # (B,T,7) fp32, no grad
    gt_c_all = traj.gt_centres_of(batch).to(gt_pose.dtype)  # (B,T,3) via Component B
    # chunk_traj_loss has no loss-mode argument: `--loss ate` IS rpe_w = 0 (it disables the term).
    rpe_w = float(args.rpe_w) if args.loss == "ate_rpe" else 0.0
    losses, stats_all, all_pred_c, all_pred_pose, n_steps = [], [], [], [], 0
    dev_type = "cuda" if batch[0]["img"].is_cuda else "cpu"
    for ci, c0 in enumerate(range(0, T, args.chunk)):
        if max_chunks is not None and ci >= max_chunks:
            break
        state_feat = state_feat.detach()
        mem = mem.detach()
        c1 = min(c0 + args.chunk, T)
        preds = []
        amp = (torch.autocast(dev_type, dtype=torch.bfloat16) if args.precision == "bf16"
               else torch.autocast(dev_type, enabled=False))
        with torch.set_grad_enabled(train):
            with amp:
                for t in range(c0, c1):
                    res_group, (state_feat, mem) = model._forward_decoder_group_step(
                        views=batch, view_indices=[t], feat_group=[feat[t]], pos_group=[pos[t]],
                        shape_group=[shape[t]], init_state_feat=init_state_feat, init_mem=init_mem,
                        state_feat=state_feat, state_pos=state_pos, mem=mem)
                    preds.append({"camera_pose": res_group[0]["camera_pose"]})
                    del res_group
            with torch.autocast(dev_type, enabled=False):
                pred_pose = traj.pred_pose_of(preds)         # (B,n,7), WITH grad
                pred_c = traj.pred_centres_of(preds).to(pred_pose.dtype)  # (B,n,3)
                prev_pred = torch.cat(all_pred_c, dim=1).detach() if all_pred_c else None
                prev_pred_pose = torch.cat(all_pred_pose, dim=1).detach() if all_pred_pose else None
                pool = {
                    "pred_c": pred_c, "gt_c": gt_c_all[:, c0:c1],
                    "prev_pred": prev_pred, "prev_gt": gt_c_all[:, :c0] if c0 else None,
                    "pred_q": pred_pose[..., 3:], "gt_q": gt_pose[:, c0:c1, 3:],
                    "pred_pose": pred_pose, "gt_pose": gt_pose[:, c0:c1],
                    "prev_pred_pose": prev_pred_pose,
                    "prev_gt_pose": gt_pose[:, :c0] if c0 else None,
                    "preds": preds, "views": batch, "t0": c0,
                    "min_t": int(args.ate_min_t), "rpe_w": rpe_w,
                }
                out = _flex_call(traj.chunk_traj_loss, pool, "chunk_traj_loss", verbose=traj.say)
                loss, st = (out[0], out[1]) if isinstance(out, (tuple, list)) else (out, {})
        if not torch.is_tensor(loss) or loss.dim() != 0:
            raise SystemExit(f"[wgcons] FATAL: chunk_traj_loss returned {type(loss)} "
                             f"{tuple(getattr(loss, 'shape', ()))}; a 0-d tensor was expected. {_B_HINT}")
        # A chunk with no frame at or above --ate_min_t (only reachable when chunk <= ate_min_t)
        # carries no trajectory signal at all: chunk_traj_loss returns `p_chunk.sum() * 0.0`, which
        # still has requires_grad=True, so an unguarded step would let AdamW's DECOUPLED weight decay
        # (p *= 1 - lr*wd) shrink the head on the strength of an all-zero gradient.  n_ate == -1 means
        # the loss reported no count; then behave as before and step.
        n_ate = int(st.get("n_ate", -1)) if isinstance(st, dict) else -1
        if train and loss.requires_grad and n_ate != 0:
            opt.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            if n_proc > 1:
                sync_grads(params, accelerator)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            n_steps += 1
        elif train and n_ate == 0:
            traj.say(f"[wgcons] chunk at t0={c0} has no frame >= --ate_min_t {args.ate_min_t}: "
                     f"no gradient, no optimiser step (logged only)")
        losses.append(float(loss.detach()))
        if isinstance(st, dict):
            stats_all.append(st)
        all_pred_c.append(pred_c.detach())
        all_pred_pose.append(pred_pose.detach())
        del preds, loss, pred_c, pred_pose
    pc = torch.cat(all_pred_c, dim=1)
    with torch.no_grad(), torch.autocast(dev_type, enabled=False):
        ate = traj.final_ate(pc, gt_c_all[:, :pc.shape[1]])
    return losses, gate_weights(model, site), ate, n_steps, stats_all


def sync_grads(params, accelerator):
    """Average the gate head's gradients across ranks.

    The head is deliberately NOT wrapped in DDP: the model calls the raw submodule, so a DDP wrapper's
    forward would never run and its reducer would never fire (the long-standing no-sync on this
    codebase's TBPTT trainers).  Episodes are fixed-length (num_views, fixed_length=True), so every
    rank performs exactly the same number of chunks and this all-reduce cannot deadlock.
    """
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.data = accelerator.reduce(p.grad.data, reduction="mean")


# ----------------------------------------------------------------------- cadence / resume helpers
def eval_record(tag, totals, state):
    """Held-out record from GATHERED SUMS AND COUNTS (never a mean of per-rank means).

    `totals` = [loss_sum, n_chunks, w_sum, n_w, ate_sum, n_ep] summed over the ranks.  A rank that
    evaluated no episode contributes (0, 0) and therefore nothing, instead of a literal 0.0 ATE that
    would be written to gate_head_best.pth as a record.
    """
    f = [float(x) for x in totals]

    def d(num, den):
        return f[num] / f[den] if f[den] > 0 else float("nan")
    return {"tag": tag, "ate": d(4, 5), "loss": d(0, 1), "w_mean": d(2, 3),
            "n_ep": int(f[5]), "n_chunks": int(f[1]),
            "seen": state["seen"], "step": state["step"]}


def resume_position(seen, steps_per_epoch, per_step, seed=0):
    """(epoch, batches to skip in it) for `seen` episodes already trained.

    One pass over the loader is `steps_per_epoch` rank-steps = that many * `per_step` episodes.
    Restoring only the counter (the old `seed + (seen > 0)`) restarted epoch 1 from its beginning, so
    a job resumed after the 2-day wall re-trained episodes it had already seen and never finished the
    first pass.  `steps_per_epoch <= 0` (a loader with no length) falls back to that old guess.
    """
    per_step = max(1, int(per_step))
    ep_episodes = max(0, int(steps_per_epoch)) * per_step
    if ep_episodes <= 0:
        return seed + (1 if seen > 0 else 0), 0, 0
    return seed + seen // ep_episodes, (seen % ep_episodes) // per_step, ep_episodes


# ------------------------------------------------------------------------------------ checkpoint
def save_ckpt(path, head, kind, site, args, argdict=None, episode=0, metrics=None):
    """Atomic write of the contract dict (tmp file in the same directory + os.replace)."""
    sd = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    ck = {"state_dict": sd, "mode": "weight", "kind": kind, "site": site,
          "wmin": float(args.wmin), "init_logit": float(args.init_logit),
          "args": dict(argdict) if argdict is not None else {},
          "episode": int(episode), "metrics": metrics or {}}
    tmp = path + f".tmp{os.getpid()}"
    torch.save(ck, tmp)
    os.replace(tmp, path)
    return ck


# ------------------------------------------------------------------------------------- self-test
def selftest():
    """CPU, no checkpoint, no data: the contract names exist and a 1-chunk optimisation works."""
    print("[wgcons] selftest: importing the Component A / B names ...")
    GateWeightHead, traj = load_components()
    _wrap_traj(traj, say=print)
    print("[wgcons] selftest: dust3r.wgate.heads.GateWeightHead and dust3r.wgate.traj_loss OK")

    from dust3r.model import ARCroco3DStereo
    for nm in ("attach_frame_gate", "attach_mem_gate"):
        sig = inspect.signature(getattr(ARCroco3DStereo, nm))
        assert "train" in sig.parameters, f"{nm}{sig} has no `train` parameter -- {_A_HINT}"
    print("[wgcons] selftest: model.attach_frame_gate / attach_mem_gate accept train= OK")

    torch.manual_seed(0)
    B, T = 1, 12
    ang = torch.linspace(0, 1.2, T)
    gt_c = torch.stack([torch.stack([ang.sin(), ang.cos() - 1.0, 0.3 * ang], -1)], 0)   # (1,T,3)
    err = 0.15 * torch.randn(B, T, 3)
    err[:, 0] = 0.0                                    # frame 0 always writes in full
    gtq = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(B, T, 4).contiguous()

    views = []
    for t in range(T):
        c = torch.eye(4)[None].repeat(B, 1, 1)
        c[:, :3, 3] = gt_c[:, t]
        views.append({"camera_pose": c})
    gc = traj.gt_centres_of(views)
    assert gc.shape == (B, T, 3), gc.shape
    assert torch.allclose(gc, gt_c - gt_c[:, :1], atol=1e-4), (gc[0, :3], (gt_c - gt_c[:, :1])[0, :3])
    enc = gt_pose_encodings(views)
    assert enc.shape == (B, T, 7) and torch.allclose(enc[..., :3], gc, atol=1e-5)
    preds0 = [{"camera_pose": torch.cat([gt_c[:, t], gtq[:, t]], -1)} for t in range(T)]
    pc = traj.pred_centres_of(preds0)
    assert pc.shape == (B, T, 3) and torch.allclose(pc, gt_c, atol=1e-5), pc.shape
    print("[wgcons] selftest: pred_centres / gt_centres shapes + frame-0 convention OK")

    class _A:
        wmin, init_logit, ate_min_t, rpe_w, loss = 0.5, 2.0, 4, 0.1, "ate"
    head = build_head(GateWeightHead, "frame", _A, P=lambda *a: None)
    in_dim = int(getattr(head, "in_dim", 1796))
    x = torch.randn(T, in_dim)
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=5e-3)

    def step(do, rpe_w=0.0):
        a = torch.cat([head(x[t:t + 1]).reshape(1) for t in range(T)]).reshape(B, T)  # in [wmin, 1]
        pred_c = gt_c + (1.0 - a)[..., None] * err          # a higher weight -> less drift
        pred_pose = torch.cat([pred_c, gtq], -1)
        preds = [{"camera_pose": pred_pose[:, t]} for t in range(T)]
        pool = {"pred_c": pred_c, "gt_c": gt_c, "prev_pred": None, "prev_gt": None,
                "pred_q": gtq, "gt_q": gtq, "pred_pose": pred_pose,
                "gt_pose": torch.cat([gt_c, gtq], -1),
                "prev_pred_pose": None, "prev_gt_pose": None,
                "preds": preds, "views": views, "t0": 0,
                "min_t": _A.ate_min_t, "rpe_w": rpe_w}
        out = _flex_call(traj.chunk_traj_loss, pool, "chunk_traj_loss")
        loss = out[0] if isinstance(out, (tuple, list)) else out
        assert torch.is_tensor(loss) and loss.dim() == 0, f"chunk_traj_loss -> {loss}"
        assert torch.isfinite(loss), f"chunk_traj_loss -> {float(loss)}"
        assert loss.requires_grad, "chunk_traj_loss detached the gradient to the gate weight"
        g = None
        if do:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            g = sum(float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None)
            assert math.isfinite(g), f"non-finite gradient in the gate head ({g})"
            torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.0)
            opt.step()
        return float(loss.detach()), float(a.detach().mean()), g

    l0, w0, _ = step(False)
    exp = _A.wmin + (1 - _A.wmin) / (1 + math.exp(-_A.init_logit))
    assert abs(w0 - exp) < 1e-5, f"init weight {w0:.6f} != wmin+(1-wmin)*sigmoid(init_logit)={exp:.6f}"
    if hasattr(head, "weight_at_init"):
        assert abs(float(head.weight_at_init) - exp) < 1e-9, head.weight_at_init
    print(f"[wgcons] selftest: init mean weight {w0:.6f} == wmin+(1-wmin)*sigmoid({_A.init_logit}) OK")
    l_rpe, _, _ = step(False, rpe_w=_A.rpe_w)
    assert l_rpe > l0, f"--loss ate_rpe ({l_rpe:.6f}) should add to the pure ATE loss ({l0:.6f})"
    print(f"[wgcons] selftest: --loss ate rpe_w=0 -> {l0:.6f}; ate_rpe rpe_w={_A.rpe_w} -> {l_rpe:.6f} OK")
    # The toy problem's optimum is EXACTLY a == 1 (then pred_c == gt_c and the ATE is 0), where the
    # loss and hence the gradient are legitimately zero.  So a non-zero gradient is asserted on the
    # FIRST optimisation step -- that is the "the consequence gradient reaches the head" check -- and
    # the loop then stops at convergence instead of demanding grad > 0 from an already-solved problem.
    l1, w1, steps = l0, w0, 0
    for i in range(80):
        l1, w1, g = step(True)
        steps = i + 1
        if i == 0:
            assert g > 0, "zero gradient reached the gate head through chunk_traj_loss"
            print(f"[wgcons] selftest: first backward -> |grad| {g:.6g} in the gate head OK")
        if g == 0.0 or l1 <= 1e-8:
            break                       # converged: pred == gt, nothing left to differentiate
    print(f"[wgcons] selftest: toy 1-chunk optimisation ({steps} steps) loss {l0:.6f} -> {l1:.6f}, "
          f"mean weight {w0:.4f} -> {w1:.4f}")
    assert l1 < 0.9 * l0, f"toy loss did not fall ({l0:.6f} -> {l1:.6f})"
    assert w1 > w0, f"mean gate weight did not rise towards the optimum ({w0:.4f} -> {w1:.4f})"

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "gate_head.pth")
        save_ckpt(p, head, "frame", "state", _A, {"name": "selftest"}, 7, {"ate": 0.1})
        ck = torch.load(p, map_location="cpu", weights_only=False)
        assert set(ck) == {"state_dict", "mode", "kind", "site", "wmin", "init_logit", "args",
                           "episode", "metrics"}, sorted(ck)
        assert ck["mode"] == "weight" and ck["kind"] == "frame" and ck["site"] == "state"
        head2 = build_head(GateWeightHead, "frame", _A, P=lambda *a: None)
        head2.load_state_dict(ck["state_dict"])
        assert torch.allclose(head2(x[:1]), head(x[:1]), atol=1e-6)
    print("[wgcons] selftest: checkpoint keys + reload OK")

    st = {"seen": 4000, "step": 7}
    r = eval_record("t", [2.0, 4.0, 3.76, 4.0, 0.08, 2.0], st)
    assert abs(r["ate"] - 0.04) < 1e-12 and abs(r["loss"] - 0.5) < 1e-12 and r["n_ep"] == 2, r
    # 4 ranks, only 2 of which had an episode: the empty ranks must not drag the mean towards 0
    assert abs(eval_record("t", [0, 0, 0, 0, 0.08, 2.0], st)["ate"] - 0.04) < 1e-12
    assert math.isnan(eval_record("t", [0] * 6, st)["ate"]), "no episode -> nan, never 0.0"
    assert resume_position(0, 100, 4) == (0, 0, 400)
    assert resume_position(400, 100, 4) == (1, 0, 400)           # exactly one pass done
    assert resume_position(404, 100, 4) == (1, 1, 400)           # one batch into the second pass
    assert resume_position(1204, 100, 4, seed=7) == (10, 1, 400)
    assert resume_position(1204, 0, 4)[:2] == (1, 0)             # loader with no length: old guess
    assert resume_position(0, 0, 4)[:2] == (0, 0)
    print("[wgcons] selftest: eval_record (sums/counts, empty rank -> nan) + resume_position OK")
    print("[wgcons] selftest: ALL OK")


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default=None, help="run name; outputs go to <out>/<name>/")
    ap.add_argument("--site", choices=["state", "mem"], default="state",
                    help="state = C1 (768-token state commit, head kind 'frame'); "
                         "mem = C2 (pose-retriever memory commit, head kind 'pose')")
    ap.add_argument("--token", action="store_true",
                    help="site state only: PER-TOKEN weights a_{t,i} (head kind 'token', one weight per state "
                         "token from the frame descriptor + the token's own proposed change) instead of one "
                         "weight per frame; everything else (loss, TBPTT, init, floor) is unchanged")
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--ckpt", default=None, help="frozen backbone (default train_unc_gate.CKPT_DEFAULT)")
    ap.add_argument("--wmin", type=float, default=0.5)
    ap.add_argument("--init_logit", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--chunk", type=int, default=16, help="TBPTT chunk = gradient horizon in frames")
    ap.add_argument("--episodes", type=int, default=38633)
    ap.add_argument("--loss", choices=["ate", "ate_rpe"], default="ate")
    ap.add_argument("--rpe_w", type=float, default=0.1)
    ap.add_argument("--ate_min_t", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--eval_episodes", type=int, default=16)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_views", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                    help="fp32 (default) is the eval worker's inference() path; bf16 autocast is "
                         "faster but shifts the head's input features (recorder precision check)")
    ap.add_argument("--no_ddp_sync", action="store_true",
                    help="do NOT average the head's gradients across ranks (the historical no-sync "
                         "behaviour of this codebase's TBPTT trainers)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="2 episodes, 1 chunk each, 1 held-out episode")
    ap.add_argument("--selftest", action="store_true", help="CPU, no checkpoint, no data")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.smoke:
        args.episodes = min(args.episodes, 2)
        args.eval_episodes = 1
        args.eval_every = 0          # init + final evals only
        args.save_every = 1
        args.log_every = 1
        args.name = args.name or "cgsmoke"
    if not args.name:
        raise SystemExit("[wgcons] --name is required (outputs go to <out>/<name>/)")
    max_chunks = 1 if args.smoke else None

    GateWeightHead, traj = load_components()
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs
    import dust3r.utils.path_to_croco  # noqa: F401
    from dust3r.model import ARCroco3DStereo
    from dust3r.datasets import get_data_loader, DL3DV_Multi  # noqa: F401 (eval() namespace)
    from train_unc_gate import build_loader, set_epoch, to_device, CKPT_DEFAULT, TRAIN_ROOT, TEST_ROOT

    torch.manual_seed(args.seed)
    ckpt = args.ckpt or CKPT_DEFAULT
    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)
    # 3 h NCCL timeout: GPFS stalls while the eval loader cold-starts have exceeded the 10-min default.
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=3))])
    device, is_main = accelerator.device, accelerator.is_main_process
    n_proc = accelerator.num_processes
    log = open(os.path.join(out_dir, "log.txt"), "a") if is_main else None

    def P(*a):
        if is_main:
            msg = " ".join(str(x) for x in a)
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

    _wrap_traj(traj, say=P)
    if args.token and args.site != "state":
        raise SystemExit("[wgcons] --token needs --site state (per-token weights exist for the state commit only)")
    kind = ("token" if args.token else "frame") if args.site == "state" else "pose"
    P(f"[wgcons] {time.strftime('%F %T')} procs={n_proc} site={args.site} kind={kind} args={vars(args)}")

    model = ARCroco3DStereo.from_pretrained(ckpt).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    if getattr(model, "pose_gru", None) is not None:
        model.pose_gru = None
    pose_dec = getattr(getattr(model, "downstream_head", None), "pose_head", None)
    head = build_head(GateWeightHead, kind, args, pose_decoder=pose_dec, P=P).to(device).float()
    head = attach_head(model, args.site, head, P=P)
    model.gradient_checkpointing_enable()
    model.train()  # checkpointing needs train mode; CUT3R has no dropout / BN
    params = [p for p in head.parameters() if p.requires_grad]

    def broadcast_head(tag):
        """Make every rank's gate head bit-identical to rank 0's.

        `sync_grads` averages the gradients, which only keeps the ranks together if they START
        together.  They do today (same seed, same deterministic construction order, and --resume
        loads the same file on every rank), but nothing enforced it -- this does.
        """
        import torch.distributed as dist
        if n_proc <= 1 or not (dist.is_available() and dist.is_initialized()):
            return
        with torch.no_grad():
            sd = head.state_dict()
            for k in sorted(sd):
                if torch.is_tensor(sd[k]):
                    dist.broadcast(sd[k].data, src=0)
        dist.barrier()
        P(f"[wgcons] gate head broadcast from rank 0 ({tag})")

    broadcast_head("init")
    P(f"[wgcons] frozen backbone {ckpt}; trainable gate head "
      f"{sum(p.numel() for p in params)/1e3:.1f}k params; precision {args.precision}; "
      f"loss {args.loss} (rpe_w {args.rpe_w}, ate_min_t {args.ate_min_t}); "
      f"ddp_sync {'off' if args.no_ddp_sync else 'on'}")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    train_loader = build_loader(TRAIN_ROOT, args.num_views, args.batch_size, args.num_workers, accelerator)
    eval_loader = build_loader(TEST_ROOT, args.num_views, 1, max(2, args.num_workers // 2), accelerator)
    # The head is NOT accelerator.prepare()d: the model calls the raw submodule, so a DDP wrapper's
    # forward would never run (see sync_grads).  Only the loaders need accelerate -- that is what
    # shards the episodes across ranks.
    train_loader, eval_loader = accelerator.prepare(train_loader, eval_loader)
    ck_path = os.path.join(out_dir, "gate_head.pth")
    best_path = os.path.join(out_dir, "gate_head_best.pth")
    opt_path = os.path.join(out_dir, "gate_head_opt.pth")
    argdict = dict(vars(args))
    argdict["ckpt"] = ckpt
    state = {"seen": 0, "step": 0, "best_ate": float("inf"),
             "last_save": 0, "last_eval": 0, "last_log": 0}
    metrics = {"evals": []}

    def _set_epoch(loader, epoch):
        set_epoch(loader, epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)  # DataLoaderShard otherwise re-applies set_epoch(self.iteration)

    def save(tag="last"):
        if not is_main:
            return
        m = dict(metrics)
        m.update(seen=state["seen"], step=state["step"], best_ate=state["best_ate"])
        save_ckpt(ck_path if tag == "last" else best_path, head, kind, args.site, args,
                  argdict, state["seen"], m)
        if tag == "last":
            tmp = opt_path + f".tmp{os.getpid()}"
            torch.save({"opt": opt.state_dict(), "episode": state["seen"], "step": state["step"],
                        "best_ate": state["best_ate"]}, tmp)
            os.replace(tmp, opt_path)

    def run_eval(tag):
        _set_epoch(eval_loader, 0)
        model.eval()
        ls, ws, ates = [], [], []
        per_rank = max(1, args.eval_episodes // n_proc)
        for i, b in enumerate(eval_loader):
            if i >= per_rank:
                break
            b = to_device(b, device)
            set_gate_keys(b, args.site, args.wmin)
            l, w, ate, _, _ = run_episode(model, b, traj, args, accelerator, opt, params,
                                          train=False, site=args.site, n_proc=n_proc,
                                          max_chunks=max_chunks)
            ls += l
            ws += w
            if ate == ate:
                ates.append(ate)
        model.train()
        # SUMS and COUNTS are gathered and divided afterwards, never per-rank means: with
        # eval_episodes < n_proc (or a short test split) a rank can evaluate zero episodes, and a
        # mean of per-rank means would then either weight the ranks wrongly or fold in a literal 0.0
        # -- which would be written to gate_head_best.pth as a record-breaking held-out ATE.
        t = torch.tensor([sum(ls), float(len(ls)), sum(ws), float(len(ws)),
                          sum(ates), float(len(ates))], device=device, dtype=torch.float64)
        g = accelerator.gather(t.reshape(1, 6)).sum(0)
        rec = eval_record(tag, g, state)
        P(f"[wgcons] EVAL {json.dumps(rec)}")
        if is_main:
            metrics["evals"].append(rec)
            open(os.path.join(out_dir, "eval.json"), "a").write(json.dumps(rec) + "\n")
            if rec["ate"] == rec["ate"] and rec["ate"] < state["best_ate"]:
                state["best_ate"] = rec["ate"]
                save("best")
                P(f"[wgcons] new best held-out ATE {rec['ate']:.5f} -> {best_path}")
        return rec

    if args.resume and os.path.isfile(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        head.load_state_dict(ck["state_dict"])
        head.to(device).float()
        state["seen"] = int(ck.get("episode", 0))
        metrics = ck.get("metrics") or {}
        metrics.setdefault("evals", [])
        state["step"] = int(metrics.get("step", 0))
        state["best_ate"] = float(metrics.get("best_ate", float("inf")))
        if os.path.isfile(opt_path):
            try:
                o = torch.load(opt_path, map_location="cpu", weights_only=False)
                opt.load_state_dict(o["opt"])
                state["step"] = int(o.get("step", state["step"]))
                state["best_ate"] = float(o.get("best_ate", state["best_ate"]))
            except Exception as e:
                P(f"[wgcons] resume: optimiser state not restored ({e})")
        state["last_save"] = state["last_eval"] = state["last_log"] = state["seen"]
        broadcast_head("resume")
        P(f"[wgcons] RESUMED episode={state['seen']} step={state['step']} best_ate={state['best_ate']}")
        if state["seen"] >= args.episodes:
            run_eval("final")
            save()
            return
    else:
        run_eval("init")

    t0 = time.time()
    run_l, run_w = [], []
    # --resume restores the DATA position, not only the episode counter.  One pass over the loader is
    # len(train_loader) rank-steps, i.e. len * batch_size * n_proc episodes, so both the epoch to
    # re-enter and the number of batches to skip inside it follow from `seen`.  Without this, a job
    # resumed after the 2-day wall restarted epoch 1 from its beginning and would re-train the same
    # episodes for ever instead of finishing the first pass.
    try:
        steps_per_epoch = int(len(train_loader))
    except TypeError:
        steps_per_epoch = 0
    epoch, skip, ep_episodes = resume_position(state["seen"], steps_per_epoch,
                                               args.batch_size * n_proc, args.seed)
    if state["seen"] > 0:
        P(f"[wgcons] data position: {ep_episodes} episodes per pass -> epoch {epoch}, "
          f"skipping {skip}/{steps_per_epoch} batches of it")
    done = False
    while not done:
        _set_epoch(train_loader, epoch)
        epoch += 1
        for bi, b in enumerate(train_loader):
            if bi < skip:                    # already trained on in the interrupted run
                continue
            b = to_device(b, device)
            set_gate_keys(b, args.site, args.wmin)
            first = state["seen"] == 0
            ls, ws, ate, nst, st_all = run_episode(
                model, b, traj, args, accelerator, opt, params, train=True, site=args.site,
                n_proc=(1 if args.no_ddp_sync else n_proc), max_chunks=max_chunks)
            state["step"] += nst
            state["seen"] += args.batch_size * n_proc
            run_l += ls
            run_w += ws
            # only the last 40 losses / 256 weights are ever read: keep the buffers bounded instead
            # of growing 63 floats per episode x 38633 episodes (~100-150 MB of host RAM).
            if len(run_l) > 400:
                del run_l[:-400]
            if len(run_w) > 2048:
                del run_w[:-2048]
            if first:
                exp = args.wmin + (1 - args.wmin) / (1 + math.exp(-args.init_logit))
                mw = sum(ws) / max(len(ws), 1)
                ok = abs(mw - exp) < 0.05
                P(f"[wgcons] init-weight check: {len(ws)} gate calls, mean w {mw:.4f} vs "
                  f"wmin+(1-wmin)*sigmoid({args.init_logit}) = {exp:.4f} -> {'OK' if ok else 'MISMATCH'}"
                  + ("" if ok else "  (the model may be routing this head through the sigma map;"
                                   " attach_* must dispatch on mode='weight')"))
                P(f"[wgcons] first episode: {len(ls)} chunks, losses "
                  f"{['%.5f' % x for x in ls]}, ate {ate:.5f}, opt steps {nst}")
                P(f"[wgcons] first episode chunk stats: {json.dumps(st_all)}")
            # Cadences compare against a "last done" counter instead of taking an exact modulo of
            # `seen`, which advances by batch_size * n_proc: 1000 % 4 == 0 only because the shipped
            # launches use 4 GPUs x batch 1, and a 3-GPU relaunch would otherwise silently never
            # checkpoint or evaluate for the whole 2-day job.
            if args.log_every and state["seen"] - state["last_log"] >= args.log_every * args.batch_size * n_proc:
                state["last_log"] = state["seen"]
                P(f"[wgcons] ep {state['seen']}/{args.episodes} step {state['step']} "
                  f"loss {sum(run_l[-40:])/max(len(run_l[-40:]),1):.5f} "
                  f"w {sum(run_w[-256:])/max(len(run_w[-256:]),1):.4f} "
                  f"ate {ate:.5f} {(time.time()-t0)/60:.1f} min")
            if args.save_every and state["seen"] - state["last_save"] >= args.save_every:
                state["last_save"] = state["seen"]
                save()
            if args.eval_every and state["seen"] - state["last_eval"] >= args.eval_every:
                state["last_eval"] = state["seen"]
                run_eval(f"ep{state['seen']}")
            if state["seen"] >= args.episodes:
                done = True
                break
        skip = 0                             # only the first (resumed) pass skips anything
    run_eval("final")
    save()
    P(f"[wgcons] done: {state['seen']} episodes, {state['step']} steps, "
      f"{(time.time()-t0)/60:.1f} min -> {ck_path}")


if __name__ == "__main__":
    main()
