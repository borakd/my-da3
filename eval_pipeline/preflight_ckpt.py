#!/usr/bin/env python3
"""Soundness gate for a finished training run, BEFORE 32 GPUs are committed to it.

Given a run directory (the trainer's output_dir) this:

  1. reads ``<run>/.hydra/config.yaml`` -- the config the run ACTUALLY resolved,
     not the repo YAML (which may have been edited since the job was submitted)
  2. derives the eval ``--conditioning`` arm from the conditioning flags, using
     the same decision rule as infer_and_eval_worker_ray.py:172-195
  3. opens checkpoint-final.pth and confirms it agrees with that config
  4. rebuilds the model through the REAL eval load path
     (``ARCroco3DStereo.from_pretrained`` -> ``dust3r/model.py:load_model``),
     which is where a mis-sniffed PoseGRU geometry would actually blow up

On success it prints one machine-readable line for the caller::

    PREFLIGHT_OK ARM=<arm> CKPT=<path> EPOCH=<n>

and exits 0. Any failed check prints ``FAIL`` and exits 1, so an armed watcher
can refuse to submit rather than burning 8 nodes on a bad checkpoint.

Why each check is here -- all of these are live footguns in this tree
(MN5_EVAL_README.md section 6):

  * ``pose_gru_mode`` is INVISIBLE in the weights (residual / direct /
    split_anchor share one head shape). load_model hard-fails without it in
    ``ckpt["args"]`` -- better to catch that on a login node.
  * ``pose_gru_iters`` is invisible too and silently defaults to 1. An R8 run
    scored at R1 is a different model, and nothing raises.
  * ``img_feat`` / ``img_feat_frames`` / ``img_feat_proj`` are sniffed from
    weight shapes. The ``img_feat_proj: False`` arms (raw 1024/2048-D encoder
    features, no ``img_proj`` tensor at all) take a different branch in
    load_model (model.py:157-178, frames solved from the cell input width) --
    assert that sniff round-trips to what the config asked for.
  * ``pose_gru_oracle != off`` is a GT-injection diagnostic. Its numbers are
    not comparable to anything and it must never reach the table.
  * a checkpoint that carries no ``pose_gru.*`` tensors but is evaluated with
    ``--conditioning prev_pred_gru`` silently fabricates an identity-init GRU,
    i.e. it quietly becomes the plain prev_pred arm.

Usage:
    python eval_pipeline/preflight_ckpt.py --run_dir <output_dir> [--ckpt name]
"""
import argparse
import os
import re
import sys

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUT3R = os.path.join(WT, "src", "CUT3R")
sys.path.insert(0, CUT3R)

import torch  # noqa: E402  (must precede curope's transitive import)
import yaml  # noqa: E402

fails = []


def ck(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def cfg_get(c, k, default=None):
    """Read a key from either a plain dict or an omegaconf DictConfig."""
    try:
        if hasattr(c, "get"):
            v = c.get(k, default)
        else:
            v = getattr(c, k, default)
    except Exception:
        v = default
    return default if v is None and default is not None else v


# Every conditioning / PoseGRU config key this eval pipeline has been shown to
# handle, split by HOW it is handled. A key in neither set is a lever nobody has
# checked, and the only safe answer is to refuse: the eval would run happily and
# silently score a model that differs from the trained one.
#
# RESTORED -- the eval load path reads it (from ckpt["args"]) or recovers it
# from the weights, so the evaluated model matches the trained one.
LEVERS_RESTORED = {
    # arm selection (infer_and_eval_worker_ray.py:172-195)
    "feed_gt_ray_map", "feed_prev_gt_ray_map", "feed_prev_pred", "pose_gru",
    # PoseGRU geometry: mode/iters come from ckpt["args"] (invisible in the
    # weights), the rest is sniffed in _sniff_pose_gru_config (model.py:94-248)
    "pose_gru_mode", "pose_gru_input", "pose_gru_hidden_dim", "pose_gru_iters",
    "pose_gru_img_feat", "pose_gru_img_feat_dim", "pose_gru_img_feat_frames",
    "pose_gru_img_feat_proj",
    # source lever: frozen encoders are fingerprinted by their serialized keys,
    # corr/pooled by the img_norm width, then cross-checked against ckpt["args"]
    "pose_gru_img_feat_src",
    # inert BY CONSTRUCTION at eval: load_model passes img_encoder_pretrained=
    # False (model.py:283-286) and the frozen encoder's weights come out of the
    # checkpoint, so this never reads a weight file on an offline compute node
    "pose_gru_img_encoder_weights",
    # P3.4 (5036262) and P2 (885763f). Both are PRESENCE-GATED state that the
    # sniff recovers from weight-key presence, not from ckpt["args"]:
    # pose_gru.input_gain is a buffer, pose_gru.ray_gate.weight a Linear, and
    # each is absent entirely when its config key is absent. Verified restored:
    # load_model's banner prints "input_gain=..., ray_gate=..." for every
    # checkpoint (demo_ray.py / infer_and_eval_worker_ray.py print the same),
    # and verify_backward_compat.py asserts the recovery on 7 checkpoints.
    # Registered only AFTER confirming the load path handles them -- whitelisting
    # a key whose loader does not exist yet is precisely the silent-wrong-row
    # failure this list guards against.
    "pose_gru_input_gain", "pose_gru_ray_gate",
    # P4.3 (landed d1b9828): refine_passes is recovered from cell.weight_ih
    # width (probe block presence) and CROSS-CHECKED against ckpt["args"] with
    # a hard failure on mismatch in both directions (model.py sniff); the
    # round-trip is machine-checked by verify_gru_probe_pass.py (exit-0 log in
    # eval_pipeline/evidence/). probe_every rides in ckpt["args"] beside it.
    "pose_gru_refine_passes", "pose_gru_probe_every",
    # not restored -- gated to "off" below; a GT-injection arm is unscoreable
    "pose_gru_oracle",
}
# TRAINER_ONLY -- shapes the optimisation, not the forward pass. Provably inert
# under torch.no_grad() at eval: none of these appears in PoseGRU.forward.
LEVERS_TRAINER_ONLY = {
    "pose_gru_bptt", "pose_gru_e2e", "pose_gru_lr_scale",
    "pose_gru_loss_weight", "pose_gru_iter_gamma", "pose_gru_iter_detach",
    # P0: chooses WHAT PoseGRULoss supervises (absolute pose vs frame-to-frame
    # motion). Lives entirely inside the loss -- it builds no parameters, no
    # buffers and no state_dict keys, and PoseGRU.forward never reads it -- so
    # the eval rebuilds the identical module either way. Whitelisted here only
    # so check_lever_coverage stops refusing to score a reltarget arm.
    "pose_gru_loss_target",
}


def check_lever_coverage(cfg):
    """Refuse any conditioning/PoseGRU lever this pipeline has not been vetted for.

    New levers arrive by config key long before anyone remembers to teach the
    eval about them (pose_gru_img_feat_src landed exactly this way). Silence is
    the failure mode: an unknown key changes the trained model, the eval rebuilds
    the OLD geometry, load_state_dict is satisfied, and the row is wrong with no
    error anywhere. So enumerate and whitelist rather than trust.
    """
    keys = sorted(k for k in (cfg or {})
                  if k.startswith("pose_gru") or k.startswith("feed_"))
    unknown = [k for k in keys
               if k not in LEVERS_RESTORED and k not in LEVERS_TRAINER_ONLY]
    inert = [k for k in keys if k in LEVERS_TRAINER_ONLY]
    print(f"  levers seen: {len(keys)} "
          f"({len(keys) - len(inert) - len(unknown)} restored, "
          f"{len(inert)} trainer-only/inert, {len(unknown)} UNKNOWN)")
    if inert:
        print(f"    inert at eval: {', '.join(inert)}")
    ck(not unknown,
       "every pose_gru*/feed_* config key is a lever the eval pipeline handles"
       + (f" -- UNKNOWN: {', '.join(unknown)}; teach preflight_ckpt.py + the "
          f"load path about it before scoring" if unknown else ""))


def arm_from_flags(gt, prev_gt, prev_pred, pose_gru):
    """The eval decision rule (infer_and_eval_worker_ray.py:172-195)."""
    if pose_gru:
        return "prev_pred_gru"
    if gt:
        return "gt"
    if prev_gt:
        return "prev_gt"
    if prev_pred:
        return "prev_pred"
    return "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="trainer output_dir")
    ap.add_argument("--ckpt", default="checkpoint-final.pth")
    ap.add_argument("--min_gb", type=float, default=2.0,
                    help="reject a suspiciously small checkpoint")
    ap.add_argument("--prevalidate", action="store_true",
                    help="mid-training rehearsal: downgrade the 'ran to the last "
                         "epoch' gate to a warning so the PoseGRU sniff / load "
                         "path can be exercised on a checkpoint-last.pth long "
                         "before 32 GPUs are committed. NEVER use to score.")
    args = ap.parse_args()

    rd = args.run_dir.rstrip("/")
    ckpt_path = os.path.join(rd, args.ckpt)
    print(f"=== preflight {os.path.basename(rd)}")
    print(f"    {ckpt_path}")

    # ---- 1. the run's own resolved config ---------------------------------
    hydra_cfg = os.path.join(rd, ".hydra", "config.yaml")
    ck(os.path.isfile(hydra_cfg), f"{hydra_cfg} exists")
    if fails:
        return finish()
    cfg = yaml.safe_load(open(hydra_cfg))

    ck(os.path.isfile(ckpt_path), f"{args.ckpt} exists")
    if fails:
        return finish()
    gb = os.path.getsize(ckpt_path) / 1e9
    ck(gb >= args.min_gb, f"checkpoint size {gb:.2f} GB >= {args.min_gb} GB")

    # ---- 2. the arm, from the config only ---------------------------------
    c_gt = bool(cfg_get(cfg, "feed_gt_ray_map", False))
    c_pgt = bool(cfg_get(cfg, "feed_prev_gt_ray_map", False))
    c_pp = bool(cfg_get(cfg, "feed_prev_pred", False))
    c_gru = bool(cfg_get(cfg, "pose_gru", False))
    arm = arm_from_flags(c_gt, c_pgt, c_pp, c_gru)
    print(f"  config: gt={c_gt} prev_gt={c_pgt} prev_pred={c_pp} pose_gru={c_gru} "
          f"lr={cfg_get(cfg, 'lr')} epochs={cfg_get(cfg, 'epochs')} -> --conditioning {arm}")
    if c_gru:
        print(f"  levers: mode={cfg_get(cfg, 'pose_gru_mode')} "
              f"input={cfg_get(cfg, 'pose_gru_input')} "
              f"hidden={cfg_get(cfg, 'pose_gru_hidden_dim')} "
              f"iters={cfg_get(cfg, 'pose_gru_iters')} "
              f"img_feat={cfg_get(cfg, 'pose_gru_img_feat')} "
              f"frames={cfg_get(cfg, 'pose_gru_img_feat_frames')} "
              f"proj={cfg_get(cfg, 'pose_gru_img_feat_proj', True)} "
              f"src={cfg_get(cfg, 'pose_gru_img_feat_src', 'pooled')} "
              f"bptt={cfg_get(cfg, 'pose_gru_bptt')} e2e={cfg_get(cfg, 'pose_gru_e2e')}")
    check_lever_coverage(cfg)

    oracle = cfg_get(cfg, "pose_gru_oracle", "off") or "off"
    ck(str(oracle) == "off",
       f"pose_gru_oracle == off (got {oracle!r}; a GT-injection arm is unscoreable)")
    ck(not (c_gru and not c_pp),
       "pose_gru=True implies feed_prev_pred=True (the GRU refines the fed-back pose)")

    # A resume rewrites .hydra/ but NOT necessarily with the same flags, and it
    # is the one thing the config cannot tell you: args.resume is set by
    # train_cut3r_baseline.py:194-196 AFTER Hydra dumps the config, so
    # `resume: null` proves nothing. The log line is the real evidence.
    tl = os.path.join(rd, "train_cut3r_baseline.log")
    if os.path.isfile(tl):
        txt = open(tl, errors="replace").read()
        n_resume = txt.count("Resume checkpoint")
        print(f"  train log: 'Resume checkpoint' x{n_resume} "
              f"({'single launch' if n_resume == 0 else 'RESUMED -- .hydra may be a rewrite'})")
        m = re.search(r"pose_gru enabled: ([^\n]+)", txt)
        if m:
            print(f"  train log: pose_gru enabled: {m.group(1)}")

    # ---- 3. the checkpoint's own record -----------------------------------
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    a, sd = d["args"], d["model"]
    gru_keys = [k for k in sd if "pose_gru" in k]
    print(f"  ckpt: epoch={d.get('epoch')} best_so_far={d.get('best_so_far')} "
          f"n_tensors={len(sd)} pose_gru_tensors={len(gru_keys)}")

    if args.prevalidate:
        print(f"  WARN  --prevalidate: epoch {d.get('epoch')} vs configured "
              f"{cfg_get(cfg, 'epochs')} NOT enforced (rehearsal only, not scoreable)")
    else:
        ck(d.get("epoch") == cfg_get(cfg, "epochs"),
           f"ckpt epoch {d.get('epoch')} == configured epochs {cfg_get(cfg, 'epochs')} "
           f"(a short checkpoint means the run did not finish)")

    a_arm = arm_from_flags(bool(cfg_get(a, "feed_gt_ray_map", False)),
                           bool(cfg_get(a, "feed_prev_gt_ray_map", False)),
                           bool(cfg_get(a, "feed_prev_pred", False)),
                           bool(cfg_get(a, "pose_gru", False)))
    ck(a_arm == arm, f"ckpt['args'] arm {a_arm!r} == .hydra arm {arm!r}")
    ck(bool(gru_keys) == (arm == "prev_pred_gru"),
       f"pose_gru tensors present ({len(gru_keys)}) is consistent with arm {arm!r} "
       f"(an empty set + prev_pred_gru would fabricate an identity GRU)")
    if arm == "prev_pred_gru":
        ck(cfg_get(a, "pose_gru_mode", None) is not None,
           "ckpt['args'].pose_gru_mode present (load_model hard-fails without it)")
        ck(cfg_get(a, "pose_gru_iters", None) == cfg_get(cfg, "pose_gru_iters"),
           f"ckpt iters {cfg_get(a, 'pose_gru_iters', None)} == config "
           f"{cfg_get(cfg, 'pose_gru_iters')} (invisible in weights; defaults to 1 if lost)")
    del d, sd

    if fails:
        return finish()

    # ---- 4. the real eval load path ---------------------------------------
    from add_ckpt_path import add_path_to_dust3r  # noqa: E402

    add_path_to_dust3r(ckpt_path)
    from src.dust3r.model import ARCroco3DStereo  # noqa: E402

    model = ARCroco3DStereo.from_pretrained(ckpt_path)
    # Mirror the worker's runtime mutation (infer_and_eval_worker_ray.py:172-195)
    # so the arm we are about to submit is the arm we just validated.
    if arm in ("prev_pred", "prev_pred_gru"):
        model.feed_prev_pred = True
    elif arm == "gt":
        model.feed_gt_ray_map = True
    elif arm == "prev_gt":
        model.feed_prev_gt_ray_map = True

    trained = getattr(model, "trained_conditioning", None)
    print(f"  load_model: trained_conditioning={trained!r} "
          f"trained_pose_gru_oracle={getattr(model, 'trained_pose_gru_oracle', None)!r}")
    ck(trained in (None, arm),
       f"load_model's own trained_conditioning {trained!r} agrees with {arm!r}")

    if arm == "prev_pred_gru":
        g = getattr(model, "pose_gru", None)
        ck(g is not None, "pose_gru module restored (NOT identity-init)")
        if g is not None:
            enc = getattr(g, "img_encoder", None)
            got = dict(mode=g.mode, input_mode=g.input_mode, hidden=g.hidden_dim,
                       iters=getattr(model, "pose_gru_iters", g.iters),
                       img_feat=g.img_feat, frames=g.img_feat_frames,
                       proj=g.img_feat_proj, appended=g.img_feat_dim,
                       # getattr: checkpoints predating the source lever have no
                       # img_feat_src attribute at all, and those ARE "pooled".
                       src=getattr(g, "img_feat_src", "pooled"),
                       enc_params=(sum(p.numel() for p in enc.parameters())
                                   if enc is not None else 0),
                       n_params=sum(p.numel() for p in g.parameters()))
            print(f"  restored: {got}")
            ck(got["mode"] == cfg_get(cfg, "pose_gru_mode"),
               f"restored mode {got['mode']!r} == config {cfg_get(cfg, 'pose_gru_mode')!r}")
            ck(got["input_mode"] == cfg_get(cfg, "pose_gru_input"),
               f"restored input {got['input_mode']!r} == config "
               f"{cfg_get(cfg, 'pose_gru_input')!r}")
            ck(got["hidden"] == cfg_get(cfg, "pose_gru_hidden_dim"),
               f"restored hidden_dim {got['hidden']} == config")
            ck(int(got["iters"]) == int(cfg_get(cfg, "pose_gru_iters", 1)),
               f"restored iters {got['iters']} == config {cfg_get(cfg, 'pose_gru_iters')} "
               f"(R lever)")
            ck(str(got["img_feat"]) == str(cfg_get(cfg, "pose_gru_img_feat")),
               f"restored img_feat {got['img_feat']!r} == config "
               f"{cfg_get(cfg, 'pose_gru_img_feat')!r} (F lever)")
            if str(got["img_feat"]) == "input":
                want_proj = bool(cfg_get(cfg, "pose_gru_img_feat_proj", True))
                want_frames = int(cfg_get(cfg, "pose_gru_img_feat_frames", 2))
                want_src = str(cfg_get(cfg, "pose_gru_img_feat_src", "pooled"))
                ck(str(got["src"]) == want_src,
                   f"restored img_feat_src {got['src']!r} == config {want_src!r} "
                   f"(fingerprinted from the weights, model.py:132-176)")
                # The frozen encoder must come out of the CHECKPOINT. If its
                # weights were not serialized, load_model builds it with
                # pretrained=False -- i.e. random -- and every compute node
                # would score a different, untrained feature extractor with no
                # error raised (and no internet to fetch the real ones).
                if want_src in ("resnet18", "dinov2_vits14"):
                    ck(got["enc_params"] > 0,
                       f"frozen {want_src} encoder restored from the checkpoint "
                       f"({got['enc_params']} params; 0 would mean random weights)")
                if want_src == "corr":
                    # corr consumes the previous view to form pair statistics,
                    # so its semantic frame count is pinned at 2 regardless of
                    # the block count -- do not compare it to the config below.
                    print(f"  note: img_feat_src=corr -- frames pinned to 2 "
                          f"(config says {want_frames}), one appended block")
                    want_frames = int(got["frames"])
                ck(bool(got["proj"]) == want_proj,
                   f"restored img_feat_proj {got['proj']} == config {want_proj} "
                   f"(proj=False takes the raw-feature branch, model.py:157-178)")
                ck(int(got["frames"]) == want_frames,
                   f"restored img_feat_frames {got['frames']} == config {want_frames} "
                   f"(solved from the cell input width when proj=False)")
                if want_proj:
                    # With proj=True the appended width is the config's img_feat_dim.
                    # With proj=False it is frames*enc_width and the config key is
                    # documented as ignored -- comparing it there would be wrong.
                    ck(int(got["appended"]) == int(cfg_get(cfg, "pose_gru_img_feat_dim", 32)),
                       f"restored appended width {got['appended']} == config "
                       f"pose_gru_img_feat_dim")
    model.eval()
    del model

    return finish(arm=arm, ckpt=ckpt_path, prevalidate=args.prevalidate)


def finish(arm=None, ckpt=None, prevalidate=False):
    print("-" * 78)
    if fails:
        print(f"PREFLIGHT_FAIL ({len(fails)} failures)")
        for f in fails:
            print("  - " + f)
        return 1
    # A rehearsal must never emit the token arm_eval_for_run.sh greps for --
    # that is the whole safety interlock between "the load path works" and
    # "this checkpoint is scoreable".
    print(f"{'PREVALIDATE_OK' if prevalidate else 'PREFLIGHT_OK'} ARM={arm} CKPT={ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
