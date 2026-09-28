#!/usr/bin/env python3
"""Batched CUT3R inference + depth/pose evaluation over a shard of test scenes.

Loads ONE checkpoint once and processes a (round-robin) shard of scenes:
  1. CUT3R inference on all frames of <scene>/dense/rgb  (reuses demo.py's
     prepare_input + the model's inference(), so outputs match demo.py).
  2. Save per-frame depth (.npy) + camera (.npz: pose=c2w, intrinsics) -- the exact
     same depth/camera demo.py's prepare_output writes (conf/color are skipped).
  3. Run eval_depth_poses.py (all default args) comparing pred vs <scene>/dense GT.

Resumable: a scene whose eval CSV already exists (non-empty) is skipped.
"""
import os
# Reduce CUDA fragmentation OOMs on very long sequences (read before torch init).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import contextlib
import inspect
import gc
import glob
import subprocess
import sys
import time
import traceback

import numpy as np
import torch


def save_depth_camera(outputs, outdir, pose_encoding_to_camera, estimate_focal_knowing_depth,
                      frame_ids=None):
    """Replicates the depth + camera saving of demo.py:prepare_output (revisit=1),
    writing ONLY depth/ and camera/ (skips conf/ and color/).

    frame_ids: optional list of ORIGINAL frame indices, one per emitted view, used
    as the file numbers. Needed by --skip_mode drop, where the input sequence is a
    subset of the scene: the evaluator joins pred and GT on file index, so kept
    frames must keep their original numbers. Default: 0..B-1."""
    preds = outputs["pred"]

    pts3ds_self = torch.cat([p["pts3d_in_self_view"].cpu() for p in preds], 0)  # B,H,W,3

    pr_poses = [pose_encoding_to_camera(p["camera_pose"].clone()).cpu() for p in preds]
    cam2world = torch.cat(pr_poses)  # B,4,4

    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    depths = pts3ds_self[..., 2]  # B,H,W
    intrinsics = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    intrinsics[:, 0, 0] = focal.detach().cpu()
    intrinsics[:, 1, 1] = focal.detach().cpu()
    intrinsics[:, 0, 2] = pp[:, 0]
    intrinsics[:, 1, 2] = pp[:, 1]

    depth_dir = os.path.join(outdir, "depth")
    cam_dir = os.path.join(outdir, "camera")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(cam_dir, exist_ok=True)
    if frame_ids is None:
        frame_ids = list(range(B))
    assert len(frame_ids) == B, f"frame_ids has {len(frame_ids)} entries for {B} views"
    for i in range(B):
        fid = int(frame_ids[i])
        np.save(os.path.join(depth_dir, f"{fid:06d}.npy"), depths[i].cpu().numpy())
        np.savez(
            os.path.join(cam_dir, f"{fid:06d}.npz"),
            pose=cam2world[i].cpu().numpy(),
            intrinsics=intrinsics[i].cpu().numpy(),
        )
        if os.environ.get("DUMP_GATE") == "1":
            # per-frame self-view confidence and per-token gate weights (visualisation only)
            gdir = os.path.join(outdir, "gate"); os.makedirs(gdir, exist_ok=True)
            ex = {}
            for k in ("conf_self", "tok_gate_w", "tok_gate_key"):
                if k in preds[i]:
                    ex[k] = preds[i][k][0].float().cpu().numpy() if preds[i][k].dim() > 1 and preds[i][k].shape[0] == 1 else preds[i][k].float().cpu().numpy()
            np.savez(os.path.join(gdir, f"{fid:06d}.npz"), **ex)
    return B


def build_kept_gt(gt_dense, kept_ids, dst):
    """Symlink the GT depth/cam files of kept_ids into dst/{depth,cam}.

    eval_depth_poses.py renumbers files POSITIONALLY per camera stream
    (_split_stream_by_camera), so a subset of predictions scored against the
    full GT is misaligned from the first gap on. Scoring a subset therefore
    needs a GT tree holding exactly the same frame numbers."""
    keep = set(int(i) for i in kept_ids)
    for sub, suf in (("depth", ".npy"), ("cam", ".npz")):
        src_dir = os.path.join(gt_dense, sub)
        if sub == "cam" and not os.path.isdir(src_dir):
            src_dir = os.path.join(gt_dense, "camera")
        dst_dir = os.path.join(dst, sub)
        os.makedirs(dst_dir, exist_ok=True)
        for old in os.listdir(dst_dir):
            os.remove(os.path.join(dst_dir, old))
        for fn in sorted(os.listdir(src_dir)):
            stem, ext = os.path.splitext(fn)
            if ext == suf and stem.isdigit() and int(stem) in keep:
                os.symlink(os.path.abspath(os.path.join(src_dir, fn)),
                           os.path.join(dst_dir, fn))
    return dst


def _plot_metrics(eval_csv, out_dir, scene, tag):
    """Write metrics_over_time*.png next to the eval CSV. Never fatal: a plot
    failure must not fail a scene that has already been scored."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from plot_metrics_over_time import plot_metrics_over_time
        plot_metrics_over_time(eval_csv, out_dir, title=scene)
    except Exception as e:  # matplotlib missing, malformed CSV, ...
        print(f"{tag} WARNING: metrics plot failed for {scene}: {e!r}", flush=True)


def _release_claim(claim_path, eval_csv):
    """Remove a cooperative claim so the scene can be retried by another worker,
    unless it actually completed (a non-empty eval CSV exists)."""
    if not claim_path:
        return
    if os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0:
        return
    try:
        os.rmdir(claim_path)
    except OSError:
        pass


# ----------------------------------------------------------------------------- wgate
# Write-gate variants (WRITE_GATE_VARIANTS.md): confidence-weighted memory updates on
# the frozen model. Control-json keys (under "*" or per scene):
#   "frame_gate": {"head": <frame_conf_head.pth>, "wmin": .5}   variant 1 (state write
#                 weight a_t from the attached FrameConfHead)
#   "mem_gate":   {"head": <pose_sigma_head.pth>, "wmin": .5}   variant 2 (pose-memory
#                 write weight b_t from the attached PoseSigmaHead)
#             | {"const": b}           b_t = b for every t>0 (0 = pose memory frozen at
#                                      its initial state) -- leverage probe / dose control
#             | {"freeze_after": k}    b_t = 0 for t > k -- leverage probe
#             | {"alpha": {frame: b}}  per-frame b_t (the V2 oracle; per-scene json)
#             | {"src": "conf", "soft": .5, "wmin": .5}   b_t keyed by the model's OWN
#                                      (untrained) self-view confidence: b_t = wmin +
#                                      (1-wmin)*sigmoid((c_t - causal median of c)/soft),
#                                      c = mean log conf_self of the frame, b_0 = 1 (arm mg_conf)
# The existing "alpha_all" / "alpha" keys remain the constant / oracle controls of
# variant 1 (they drive update_alpha, the same multiplier the frame gate uses).
#   "frame_gate": {"const": a} | {"alpha": {frame: a}}   state-only constant / per-frame
#                 schedule (V1 dose-matched constant and oracle; the model keeps a_0 = 1)
# The SAME two "head" keys also take a variant-C consequence-trained checkpoint
# ({"mode": "weight", "state_dict", "wmin", "init_logit"}, train_wgate_consequence.py):
# attach_wgate_head dispatches on the checkpoint's "mode" ("sigma" = the Kendall-Gal
# log_ref map of variants 1/2, the default when the key is absent; "weight" = the head
# output IS the write weight, GateWeightHead, no log_ref). Nothing else changes: the
# same per-view keys, the same trace dump, the same controls.
# Per-view keys set here, (1,) float tensors like the token_gate_* keys, read by
# _forward_decoder_group_step: frame_gate_on, frame_gate_wmin, frame_gate_const,
# mem_gate_on, mem_gate_wmin, mem_gate_const, mem_gate_freeze_after, mem_gate_src,
# mem_gate_conf_soft, mem_gate_conf_wmin. Absent keys = plain model.
WGATE_HEAD_ROOT = "/gpfs/scratch/etur59/koc821022/checkpoints/wgate"


def _v1(x):
    """(1,) float tensor -- the per-view scalar-key convention of the eval controls."""
    return torch.tensor(float(x)).unsqueeze(0)


def load_wgate_head(path):
    """Load a wgate head checkpoint and return the raw dict (sigma mode or weight mode).

    sigma mode (train_wgate_heads.py, variants 1 / 2; "mode" absent or "sigma"):
      frame_conf_head.pth = {"state_dict", "log_ref", "wmin", "args", "metrics"}
      pose_sigma_head.pth = {"state_dict" (out layer only), "log_ref_t", "log_ref_R", "wmin",
                             "args", "metrics"}
    weight mode (train_wgate_consequence.py, variant C):
      gate_head_best.pth  = {"state_dict", "mode": "weight", "wmin", "init_logit", "args", "metrics"}
      -- the head output IS the write weight, so there is no log_ref / sigma map."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert "state_dict" in ck, f"{path}: not a wgate head checkpoint (no 'state_dict')"
    return ck


def wgate_mode(ck):
    """"sigma" (variants 1/2, the default when the checkpoint has no "mode") or "weight" (variant C)."""
    return str(ck.get("mode") or "sigma").lower()


def attach_wgate_head(model, kind, ck, wmin=None):
    """Attach a wgate head to `model` in EVAL mode, dispatching on the checkpoint's "mode".

    kind: "frame" (model.attach_frame_gate) or "mem" (model.attach_mem_gate).
    sigma mode -> the historical call (state_dict + the trainer's log_ref / log_ref_t, log_ref_R).
    weight mode -> the WHOLE checkpoint dict is passed through and the model dispatches on
    ck["mode"]; no log_ref is passed (there is none). `train=False` is passed only when the
    model's attach signature has it (variant C's trainer needs train=True; inference never does),
    and the returned head is forced to eval / requires_grad False here as a belt-and-braces check.
    Returns (head, mode)."""
    fn = getattr(model, "attach_frame_gate" if kind == "frame" else "attach_mem_gate", None)
    if fn is None:
        raise RuntimeError(f"model.py has no attach_{kind}_gate (wgate model patch missing)")
    mode = wgate_mode(ck)
    wmin = float(ck.get("wmin", 0.5)) if wmin is None else float(wmin)
    kw = {"wmin": wmin}
    try:
        if "train" in inspect.signature(fn).parameters:
            kw["train"] = False
    except (TypeError, ValueError):
        pass
    if mode == "weight":
        try:
            head = fn(ck, **kw)          # whole dict: the model reads "mode" / "init_logit" / "wmin"
        except (AssertionError, TypeError, KeyError) as e:
            raise RuntimeError(
                f"model.attach_{kind}_gate does not accept a weight-mode checkpoint "
                f"({{'mode': 'weight', ...}}): {e}. Update model.py (variant C, WRITE_GATE_VARIANTS.md)."
            ) from e
    elif kind == "frame":
        head = fn(ck["state_dict"], ck["log_ref"], **kw)
    else:
        head = fn(ck["state_dict"], ck["log_ref_t"], ck["log_ref_R"], **kw)
    if head is None:                      # some attach implementations return None
        head = getattr(model, "frame_gate" if kind == "frame" else "mem_gate", None)
    if hasattr(head, "eval"):
        head.eval()
        for p_ in getattr(head, "parameters", list)():
            p_.requires_grad_(False)
    return head, mode


def apply_wgate_controls(views, ctl, attach_fg=None, attach_mg=None):
    """Set the wgate per-view keys on `views` from one scene's control spec `ctl`.

    attach_fg(path) / attach_mg(path): callbacks for the head arms (the worker's are
    idempotent by path and return the loaded checkpoint dict, so a "wmin" missing from
    the json falls back to the value the trainer stored; default 0.5).
    Returns None when `ctl` carries no wgate key, else a json-able dict of the active
    gates that the caller stores next to the trace dump. Frame 0 gets the same keys as
    every other view: the model itself writes frame 0 in full (a_0 = b_0 = 1)."""
    fg = ctl.get("frame_gate")
    mg = ctl.get("mem_gate")
    if not fg and not mg:
        return None
    active = {}
    n = len(views)
    if fg:
        spec = {}
        if "head" in fg:
            ck = attach_fg(fg["head"]) if attach_fg else None
            wmin = float(fg.get("wmin", (ck or {}).get("wmin", 0.5)))
            for v_ in views:
                v_["frame_gate_on"] = _v1(1.0)
                v_["frame_gate_wmin"] = _v1(wmin)
            spec["head"] = fg["head"]
            spec["wmin"] = wmin
        # STATE-ONLY constant / per-frame schedule (model key frame_gate_const; the model keeps a_0 = 1 and
        # leaves the retriever-mem commit untouched) -- the like-for-like controls for the frame-gate head arm.
        # alpha_all / alpha remain the JOINT (state + retriever-mem) frame gate of the conf-gate campaign.
        if fg.get("const") is not None:
            for v_ in views:
                v_["frame_gate_const"] = _v1(fg["const"])
            spec["const"] = float(fg["const"])
        if fg.get("alpha"):
            nset = 0
            for fr_, a_ in fg["alpha"].items():
                fr_ = int(fr_)
                if 0 <= fr_ < n:
                    views[fr_]["frame_gate_const"] = _v1(a_)
                    nset += 1
            spec["alpha_frames"] = nset
        # STATE-ONLY write weight keyed by frame t's OWN (untrained) self-view confidence (arm fg_conf):
        # a_t = wmin + (1-wmin)*sigmoid((c_t - causal median of c)/soft), or with soft = 0 the hard rule
        # a_t = 1 if c_t >= median else wmin; c = mean log conf_self, a_0 = 1 (model: _wgate_frame_conf).
        if fg.get("src") is not None:
            if str(fg["src"]).lower() != "conf":
                raise ValueError(f"frame_gate.src must be 'conf', got {fg['src']!r}")
            if spec:
                raise ValueError(f"frame_gate.src cannot be combined with head / const / alpha: {fg}")
            soft, cwmin = float(fg.get("soft", 0.5)), float(fg.get("wmin", 0.5))
            for v_ in views:
                v_["frame_gate_src"] = _v1(1.0)
                v_["frame_gate_conf_soft"] = _v1(soft)
                v_["frame_gate_conf_wmin"] = _v1(cwmin)
            spec.update(src="conf", soft=soft, conf_wmin=cwmin)
        if not spec:
            raise ValueError("frame_gate needs 'head', 'const', 'alpha' or 'src'")
        active["frame_gate"] = spec
    if mg:
        spec = {}
        if "head" in mg:
            ck = attach_mg(mg["head"]) if attach_mg else None
            wmin = float(mg.get("wmin", (ck or {}).get("wmin", 0.5)))
            for v_ in views:
                v_["mem_gate_on"] = _v1(1.0)
                v_["mem_gate_wmin"] = _v1(wmin)
            spec["head"] = mg["head"]
            spec["wmin"] = wmin
        if mg.get("const") is not None:
            for v_ in views:
                v_["mem_gate_const"] = _v1(mg["const"])
            spec["const"] = float(mg["const"])
        if mg.get("freeze_after") is not None:
            for v_ in views:
                v_["mem_gate_freeze_after"] = _v1(mg["freeze_after"])
            spec["freeze_after"] = int(mg["freeze_after"])
        if mg.get("alpha"):
            nset = 0
            for fr_, b_ in mg["alpha"].items():
                fr_ = int(fr_)
                if 0 <= fr_ < n:
                    views[fr_]["mem_gate_const"] = _v1(b_)
                    nset += 1
            spec["alpha_frames"] = nset
        # Pose-memory write keyed by the model's OWN (untrained) self-view confidence (arm mg_conf):
        # model keys mem_gate_src / mem_gate_conf_soft / mem_gate_conf_wmin. Independent of the
        # head / const / freeze_after / alpha paths above (they may be combined).
        if mg.get("src") is not None:
            src = str(mg["src"]).lower()
            if src != "conf":
                raise ValueError(f"mem_gate.src must be 'conf', got {mg['src']!r}")
            soft = float(mg.get("soft", 0.5))
            cwmin = float(mg.get("wmin", 0.5))
            for v_ in views:
                v_["mem_gate_src"] = _v1(1.0)
                v_["mem_gate_conf_soft"] = _v1(soft)
                v_["mem_gate_conf_wmin"] = _v1(cwmin)
            spec["src"] = src
            spec["soft"] = soft
            spec["conf_wmin"] = cwmin
        if not spec:
            raise ValueError(f"mem_gate has none of head / const / freeze_after / alpha / src: {mg}")
        active["mem_gate"] = spec
    return active


def _trace_jsonable(trace):
    """model._wgate_trace entries (t, weight[, ...]) -> lists of python scalars
    (0-d / 1-element tensors and numpy scalars become floats; longer tensors become lists)."""
    out = []
    for e in trace:
        row = []
        for x in (list(e) if isinstance(e, (tuple, list)) else [e]):
            if hasattr(x, "detach"):
                x = x.detach().float().cpu().reshape(-1)
                x = float(x[0]) if x.numel() == 1 else x.tolist()
            elif isinstance(x, np.ndarray):
                x = float(x.reshape(-1)[0]) if x.size == 1 else x.reshape(-1).tolist()
            elif isinstance(x, (np.floating, np.integer)):
                x = float(x)
            row.append(x)
        out.append(row)
    return out


def dump_wgate_trace(model, eval_dir, scene, active):
    """Write <eval_dir>/wgate_trace.json = {"scene", "gates", "trace": [[t, weight], ...], ...}
    from model._wgate_trace (the model appends (t, weight) at every gated decoder step, (t, a_t, b_t)
    when both gates are on, and resets the lists at t == 0, so after inference they hold exactly this
    scene). The model's per-gate lists, when present, are dumped alongside: "trace_frame" [[t, a_t]],
    "trace_mem" [[t, b_t]], "trace_logvar" [[t, "frame"|"mem", [log sigma^2 ...]]] (the raw head
    outputs, so weights can be recomputed for another wmin without rerunning).
    Returns the number of "trace" rows written (0 when the model recorded nothing)."""
    import json
    trace = getattr(model, "_wgate_trace", None) or []
    os.makedirs(eval_dir, exist_ok=True)
    payload = {"scene": scene, "gates": active, "trace": _trace_jsonable(trace)}
    for key in ("frame", "mem", "logvar", "token"):  # token: per-token gate [mean on-image, mean register, sd, min, max]
        extra = getattr(model, f"_wgate_trace_{key}", None)
        if extra:
            payload[f"trace_{key}"] = _trace_jsonable(extra)
    with open(os.path.join(eval_dir, "wgate_trace.json"), "w") as f:
        json.dump(payload, f)
    return len(payload["trace"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--scenes_root", required=True, help=".../test/dl3dv_multi/wrist")
    ap.add_argument("--scene_list", required=True, help="txt of scene names (one per line)")
    ap.add_argument("--pred_base", required=True)
    ap.add_argument("--eval_base", required=True)
    ap.add_argument("--eval_script", required=True)
    ap.add_argument("--cut3r_dir", required=True)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="process at most N scenes (smoke test)")
    ap.add_argument(
        "--claim_dir",
        default="",
        help="If set, enable cooperative-queue mode: every worker walks ALL "
        "scenes and atomically claims each via mkdir under this dir, so any "
        "number of workers (across nodes / GPU types) cooperatively drain one "
        "shared queue with no double-processing and no stall if some workers "
        "never start. shard_id/num_shards then only set a starting offset.",
    )
    ap.add_argument(
        "--skip_frames_file",
        default="",
        help="json {scene: [frame indices]} of frames to IGNORE (harmful-frame "
        "pilot, see HARMFUL_FRAME_SKIP_PLAN.md). Scenes absent from the json run "
        "normally. Frame 0 is never skipped.",
    )
    ap.add_argument(
        "--control_json",
        default="",
        help="Richer eval-time write control, json {scene: spec}. spec keys (all "
        "optional): block [frames] (update=False); alpha {frame: a} (soft commit "
        "state<-a*new+(1-a)*old); reset [frames] (state/mem reset to init AFTER "
        "that frame's commit); token_gate {frames: [..] | 'all', q, gmin} "
        "(per-state-token commit keyed on the frame's own confidence); "
        "frame_gate {head, wmin} / mem_gate {head, wmin} | {const} | "
        "{freeze_after} | {alpha: {frame: b}} (write-gate variants, see "
        "WRITE_GATE_VARIANTS.md and apply_wgate_controls). Frame 0 "
        "is never blocked. Combines with --skip_frames_file.",
    )
    ap.add_argument("--unc_head", default="", help="uncgate head checkpoint (head.pth from train_unc_gate.py); "
                    "attaches the head to the frozen model so control_json token_gate.src='unc' can key the token gate on it")
    ap.add_argument("--unc_mode", default="", choices=["", "v1", "v2"], help="uncgate head type; read from the checkpoint if empty")
    ap.add_argument("--conf_branch", default="", help="trained self-view confidence branch (conf_branch.pth from train_conf_gate.py); "
                    "replaces the conf channel of the self-view head (xyz/pose untouched)")
    ap.add_argument(
        "--skip_mode",
        choices=["block", "drop"],
        default="block",
        help="block: listed frames still get a forward pass and a prediction but "
        "their state/memory write is suppressed (views[i]['update']=False). "
        "drop: listed frames are removed from the input sequence entirely; the "
        "kept frames' predictions are saved under their ORIGINAL indices.",
    )
    ap.add_argument(
        "--no_keep_preds",
        action="store_true",
        help="Delete the scene's predictions (depth/ + camera/) once its eval CSV "
        "is written, keeping only the eval outputs. For large sweeps.",
    )
    ap.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip the per-scene metrics-over-frames plots (metrics_over_time*.png "
        "next to the eval CSV, see eval_pipeline/plot_metrics_over_time.py).",
    )
    ap.add_argument(
        "--ignore_skip_sentinel",
        action="store_true",
        help="Run even if <OUT>/logs/SKIP_<label> exists. Default: respect the "
        "sentinel and exit immediately (used to disable a job's pass for a label "
        "that another job is handling).",
    )
    args = ap.parse_args()

    # Skip sentinel: lets us disable a specific (job, label) pass without editing
    # a running job's launch loop. OUT = parent-of-parent of eval_base.
    out_root = os.path.dirname(os.path.dirname(os.path.abspath(args.eval_base)))
    sentinel = os.path.join(out_root, "logs", f"SKIP_{args.label}")
    if (not args.ignore_skip_sentinel) and os.path.exists(sentinel):
        print(f"[{args.label} shard {args.shard_id}] SKIP sentinel present "
              f"({sentinel}); exiting without work.", flush=True)
        return

    sys.path.insert(0, args.cut3r_dir)
    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(args.ckpt)

    import demo  # provides prepare_input (no GPU/viser side effects on import)
    from src.dust3r.inference import inference
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    with open(args.scene_list) as f:
        all_scenes = [ln.strip() for ln in f if ln.strip()]
    if args.claim_dir:
        # Cooperative-queue mode: every worker considers ALL scenes and claims
        # each atomically (mkdir) below. shard_id/num_shards only pick a distinct
        # starting offset so workers begin in different parts of the list and
        # don't all contend for the same first scene.
        os.makedirs(args.claim_dir, exist_ok=True)
        n = len(all_scenes)
        off = (args.shard_id * (n // max(args.num_shards, 1))) % n if n else 0
        my_scenes = all_scenes[off:] + all_scenes[:off]
    else:
        my_scenes = [s for i, s in enumerate(all_scenes)
                     if i % args.num_shards == args.shard_id]
    if args.limit > 0:
        my_scenes = my_scenes[: args.limit]

    tag = f"[{args.label} shard {args.shard_id}/{args.num_shards}]"
    print(f"{tag} {len(my_scenes)} scenes (of {len(all_scenes)} total). device={device}",
          flush=True)

    print(f"{tag} loading model from {args.ckpt} ...", flush=True)
    t0 = time.time()
    model = ARCroco3DStereo.from_pretrained(args.ckpt).to(device)
    # This worker is IMAGE-ONLY (demo.py-style views, no ray conditioning, no
    # feed_prev_pred). A checkpoint trained with conditioning still loads —
    # but evaluating it here is an out-of-distribution open-loop probe, so say
    # so loudly and strip any PoseGRU (it would be inert anyway: the GRU only
    # runs inside the feed_prev_pred loop). The conditioned arms belong to
    # infer_and_eval_worker_ray.py.
    trained_cond = getattr(model, "trained_conditioning", "none")
    if trained_cond != "none":
        print(f"{tag} WARNING: ckpt was trained with conditioning "
              f"'{trained_cond}' but this worker runs image-only open loop — "
              "an out-of-distribution probe, NOT the ckpt's honest arm. Use "
              "eval_pipeline/infer_and_eval_worker_ray.py --conditioning "
              f"{trained_cond} for the honest evaluation.", flush=True)
    if getattr(model, "pose_gru", None) is not None:
        print(f"{tag} ckpt carries pose_gru — disabling it for this image-only "
              "worker (it only ever runs inside the feed_prev_pred loop).",
              flush=True)
        model.pose_gru = None
    model.eval()
    unc_state = {"path": None}

    def attach_unc(path, mode=""):
        # uncgate: attach a trained registration-uncertainty head (frozen model untouched).
        # Called from --unc_head and from control_json token_gate.unc_head (per-arm heads in sweeps).
        if path == unc_state["path"]:
            return
        from dust3r.uncgate.heads import PixelUncHead, TokenUncHead
        uck = torch.load(path, map_location="cpu", weights_only=False)
        umode = mode or uck.get("mode", "")
        assert umode in ("v1", "v2"), f"unc_mode unknown ({umode!r}); pass --unc_mode"
        hcfg = uck.get("head_cfg", {})
        head = PixelUncHead(**hcfg) if umode == "v1" else TokenUncHead(**hcfg)
        head.load_state_dict(uck["head"])
        head = head.to(device).eval()
        model.attach_unc_gate(head, umode, train=False)
        unc_state["path"] = path
        print(f"{tag} uncgate head attached: mode={umode} from {path} "
              f"({sum(p.numel() for p in head.parameters())/1e6:.2f}M params)", flush=True)

    cb_state = {"path": None}

    def attach_cb(path):
        if path == cb_state["path"]:
            return
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.attach_conf_branch(ck["conf"], train=False)
        model.downstream_head.dpt_self.head.to(device).eval()
        cb_state["path"] = path
        print(f"{tag} conf branch attached from {path} (step {ck.get('step')})", flush=True)

    # wgate (WRITE_GATE_VARIANTS.md): the two write-gate heads, attached once per path from
    # control_json frame_gate.head / mem_gate.head; the loaded dict is cached so the
    # per-scene control parsing can read the trainer's "wmin" without reloading.
    fg_state = {"path": None, "ck": None}
    mg_state = {"path": None, "ck": None}

    def attach_fg(path):
        if path != fg_state["path"]:
            ck = load_wgate_head(path)
            attach_wgate_head(model, "frame", ck)      # dispatches on ck["mode"] (sigma | weight)
            fg_state["path"], fg_state["ck"] = path, ck
            # sys.__stdout__: apply_wgate_controls runs inside the redirect_stdout(os.devnull)
            # block that silences prepare_input, which would swallow this line.
            print(f"{tag} frame gate head attached from {path} (mode={wgate_mode(ck)}, "
                  + (f"log_ref={float(ck['log_ref']):.4f}, " if wgate_mode(ck) != "weight" else
                     f"init_logit={ck.get('init_logit')}, ")
                  + f"wmin_train={ck.get('wmin')})", file=sys.__stdout__, flush=True)
        return fg_state["ck"]

    def attach_mg(path):
        if path != mg_state["path"]:
            ck = load_wgate_head(path)
            attach_wgate_head(model, "mem", ck)        # dispatches on ck["mode"] (sigma | weight)
            mg_state["path"], mg_state["ck"] = path, ck
            print(f"{tag} mem gate head attached from {path} (mode={wgate_mode(ck)}, "
                  + (f"log_ref_t={float(ck['log_ref_t']):.4f}, log_ref_R={float(ck['log_ref_R']):.4f}, "
                     if wgate_mode(ck) != "weight" else f"init_logit={ck.get('init_logit')}, ")
                  + f"wmin_train={ck.get('wmin')})", file=sys.__stdout__, flush=True)
        return mg_state["ck"]

    if args.conf_branch:
        attach_cb(args.conf_branch)
    if args.unc_head:
        attach_unc(args.unc_head, args.unc_mode)
    print(f"{tag} model loaded in {time.time()-t0:.1f}s", flush=True)

    skip_frames = {}
    if args.skip_frames_file:
        import json
        with open(args.skip_frames_file) as f:
            skip_frames = {k: sorted(set(int(i) for i in v) - {0})
                           for k, v in json.load(f).items()}
        print(f"{tag} skip-frames file: {args.skip_frames_file} "
              f"({len(skip_frames)} scenes, mode={args.skip_mode})", flush=True)

    controls = {}
    if args.control_json:
        import json
        with open(args.control_json) as f:
            controls = json.load(f)
        print(f"{tag} control json: {args.control_json} ({len(controls)} scenes)", flush=True)
        if args.skip_mode != "block":
            print(f"{tag} WARNING: --control_json with --skip_mode {args.skip_mode}: only the 'block' lists "
                  "are honoured (as drops); alpha/alpha_all/reset/trigger/token_gate/frame_gate/mem_gate "
                  "controls are IGNORED in this mode -- every scene runs plain apart from the drops", flush=True)

    fail_log = os.path.join(args.eval_base, f"_failures_shard{args.shard_id}.txt")
    os.makedirs(args.eval_base, exist_ok=True)

    done = 0
    skipped = 0
    failed = 0
    t_start = time.time()
    for idx, scene in enumerate(my_scenes):
        claim_path = None
        eval_dir = os.path.join(args.eval_base, scene)
        eval_csv = os.path.join(eval_dir, "eval_depth_pose_metrics.csv")
        if os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0:
            skipped += 1
            continue

        # Cooperative claim: atomically reserve this scene so concurrent workers
        # don't process it twice. mkdir is atomic on POSIX; FileExistsError means
        # another worker already owns it (in-progress or done).
        if args.claim_dir:
            claim_path = os.path.join(args.claim_dir, scene)
            try:
                os.mkdir(claim_path)
            except FileExistsError:
                skipped += 1
                continue

        rgb_dir = os.path.join(args.scenes_root, scene, "dense", "rgb")
        gt_dense = os.path.join(args.scenes_root, scene, "dense")
        pred_dir = os.path.join(args.pred_base, scene)
        img_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")) +
                           glob.glob(os.path.join(rgb_dir, "*.jpg")))
        if len(img_paths) < 2:
            skipped += 1
            _release_claim(claim_path, eval_csv)
            continue

        try:
            ts = time.time()
            # Run inference with one OOM retry (after a cache clear) for very long
            # sequences; silence load_images/inference per-frame prints.
            nfr = None
            last_exc = None
            frame_ids = None
            for attempt in range(2):
                outputs = state_args = views = None
                wg_active = None
                try:
                    ctl = controls.get(scene) or controls.get("*", {})
                    if controls and "*" not in controls and scene not in controls and attempt == 0:
                        # per-scene control json (an oracle arm) without this scene: it runs PLAIN and
                        # is scored under this arm's label; wgate_table.py drops it via the arm's
                        # control.json / .skipped.txt sidecar (wgate_make_controls.py)
                        print(f"{tag} WARNING: {scene} has no control entry in {args.control_json} -> plain",
                              flush=True)
                    bad = set(skip_frames.get(scene, [])) | set(int(b) for b in ctl.get("block", []))
                    bad = {b for b in bad if 0 < b < len(img_paths)}
                    frame_ids = None
                    run_paths = img_paths
                    if bad and args.skip_mode == "drop":
                        frame_ids = [i for i in range(len(img_paths)) if i not in bad]
                        run_paths = [img_paths[i] for i in frame_ids]
                    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
                        views = demo.prepare_input(
                            img_paths=run_paths,
                            img_mask=[True] * len(run_paths),
                            size=args.size,
                            revisit=1,
                            update=True,
                        )
                        if bad and args.skip_mode == "block":
                            # Native memory-write switch, honored by
                            # _forward_decoder_group_step (model.py ~1867):
                            # the frame is still decoded and predicted, but
                            # state_feat/mem keep their pre-frame values.
                            for vi_, v_ in enumerate(views):
                                if vi_ in bad:
                                    v_["update"] = torch.tensor(False).unsqueeze(0)
                        if ctl and args.skip_mode == "block":
                            if ctl.get("alpha_all") is not None:
                                for v_ in views:
                                    v_["update_alpha"] = torch.tensor(float(ctl["alpha_all"])).unsqueeze(0)
                            for fr_, a_ in (ctl.get("alpha") or {}).items():
                                fr_ = int(fr_)
                                if 0 <= fr_ < len(views):
                                    views[fr_]["update_alpha"] = torch.tensor(float(a_)).unsqueeze(0)
                            for fr_ in ctl.get("reset") or []:
                                fr_ = int(fr_)
                                if 0 <= fr_ < len(views):
                                    views[fr_]["reset"] = torch.tensor(True).unsqueeze(0)
                            trig = ctl.get("trigger")  # {"z": 1.0, "warmup": 8}: causal conf trigger
                            if trig:
                                for v_ in views:
                                    v_["conf_trigger_z"] = torch.tensor(float(trig.get("z", 1.0))).unsqueeze(0)
                                    v_["conf_trigger_warmup"] = torch.tensor(float(trig.get("warmup", 8))).unsqueeze(0)
                                    v_["conf_trigger_window"] = torch.tensor(float(trig.get("window", 0))).unsqueeze(0)
                            tg = ctl.get("token_gate")
                            if tg and tg.get("unc_head"):
                                attach_unc(tg["unc_head"], tg.get("unc_mode", ""))
                            if tg and tg.get("conf_branch"):
                                attach_cb(tg["conf_branch"])
                            if tg:
                                frs = range(len(views)) if tg.get("frames") == "all" else [int(x) for x in tg.get("frames", [])]
                                for fr_ in frs:
                                    if 0 <= fr_ < len(views):
                                        views[fr_]["token_gate_q"] = torch.tensor(float(tg["q"])).unsqueeze(0)
                                        views[fr_]["token_gate_gmin"] = torch.tensor(float(tg.get("gmin", 0.0))).unsqueeze(0)
                                        # align: none (legacy, default) | patch | patch_shuffle -> model.py token gate
                                        al_ = {"none": 0, "patch": 1, "patch_shuffle": 2}[tg.get("align", "none")]
                                        if al_:
                                            views[fr_]["token_gate_align"] = torch.tensor(float(al_)).unsqueeze(0)
                                            if tg.get("offg") is not None:
                                                views[fr_]["token_gate_offg"] = torch.tensor(float(tg["offg"])).unsqueeze(0)
                                        # src: conf_self (default) | unc (attached uncgate head score); ema: per-token EMA
                                        src_ = {"conf_self": 0, "unc": 3}[tg.get("src", "conf_self")]
                                        if src_:
                                            views[fr_]["token_gate_src"] = torch.tensor(float(src_)).unsqueeze(0)
                                        if tg.get("ema") is not None:
                                            views[fr_]["token_gate_ema"] = torch.tensor(float(tg["ema"])).unsqueeze(0)
                                        if tg.get("soft") is not None:
                                            views[fr_]["token_gate_soft"] = torch.tensor(float(tg["soft"])).unsqueeze(0)
                            # write-gate variants: frame_gate / mem_gate keys (see apply_wgate_controls)
                            wg_active = apply_wgate_controls(views, ctl, attach_fg, attach_mg)
                        with torch.no_grad():
                            outputs, state_args = inference(views, model, device)
                        nfr = save_depth_camera(
                            outputs, pred_dir, pose_encoding_to_camera,
                            estimate_focal_knowing_depth, frame_ids=frame_ids,
                        )
                        if ctl and ctl.get("trigger"):
                            import json as _json
                            trig_frames = sorted(int(t) for t in getattr(model, "_maks_triggered", []))
                            os.makedirs(eval_dir, exist_ok=True)
                            with open(os.path.join(eval_dir, "triggered_frames.json"), "w") as _f:
                                _json.dump({scene: trig_frames}, _f)
                            with open(os.path.join(eval_dir, "conf_trace.json"), "w") as _f:
                                _json.dump({"mean_log_conf_self": list(getattr(model, "_maks_conf_hist", [])),
                                            "z": list(getattr(model, "_maks_z_hist", []))}, _f)
                            model._maks_conf_hist = []
                            model._maks_triggered = []
                        if wg_active:
                            # (t, weight) per gated decoder step -> <eval_dir>/wgate_trace.json;
                            # wgate_make_controls.py reads these for the dose-matched constants.
                            ntr = dump_wgate_trace(model, eval_dir, scene, wg_active)
                            if ntr == 0:
                                print(f"{tag} WARNING: {scene}: wgate controls active "
                                      f"({sorted(wg_active)}) but model._wgate_trace is empty", flush=True)
                    if bad or ctl:
                        print(f"{tag} {scene}: {args.skip_mode} {len(bad)}/{len(img_paths)} frames"
                              + (f"; controls={sorted(ctl.keys())}" if ctl else ""), flush=True)
                    break
                except torch.cuda.OutOfMemoryError as oom:
                    last_exc = oom
                    print(f"{tag} OOM on {scene} (frames={len(img_paths)}) "
                          f"attempt {attempt+1}/2", flush=True)
                finally:
                    del outputs, state_args, views
                    gc.collect()
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
            if nfr is None:
                raise last_exc if last_exc is not None else RuntimeError("no output")

            os.makedirs(eval_dir, exist_ok=True)
            gt_for_eval = gt_dense
            if frame_ids is not None:
                # drop mode: score the kept frames against a GT tree with the
                # same frame numbers (positional join, see build_kept_gt).
                gt_for_eval = build_kept_gt(
                    gt_dense, frame_ids, os.path.join(pred_dir, "_gt_kept"))
            cmd = [
                sys.executable, args.eval_script,
                "--pred_root", pred_dir,
                "--gt_root", gt_for_eval,
                "--output_csv", eval_csv,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not (os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0):
                raise RuntimeError(f"eval failed rc={r.returncode}: {r.stderr[-500:]}")
            if not args.no_plots:
                _plot_metrics(eval_csv, eval_dir, scene, tag)
            if args.no_keep_preds:
                import shutil
                shutil.rmtree(pred_dir, ignore_errors=True)

            done += 1
            dt = time.time() - ts
            if done <= 3 or done % 25 == 0:
                rate = (time.time() - t_start) / max(done, 1)
                remaining = (len(my_scenes) - skipped - done - failed) * rate
                print(f"{tag} [{idx+1}/{len(my_scenes)}] {scene} frames={nfr} "
                      f"{dt:.1f}s | done={done} skip={skipped} fail={failed} "
                      f"ETA~{remaining/3600:.1f}h", flush=True)
        except Exception as e:
            failed += 1
            with open(fail_log, "a") as fh:
                fh.write(f"{scene}\t{repr(e)}\n")
            print(f"{tag} FAIL {scene}: {repr(e)[:200]}", flush=True)
            traceback.print_exc()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            # Release the claim so another worker (or a later sweep) can retry.
            _release_claim(claim_path, eval_csv)

    print(f"{tag} DONE done={done} skipped={skipped} failed={failed} "
          f"elapsed={(time.time()-t_start)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
