#!/usr/bin/env python3
"""Verify that demo_ray.py can run EVERY checkpoint under checkpoints/captain_gru_overfit
with the single argument ``--conditioning prev_pred_gru`` — i.e. that the run-name levers
are all recoverable from the checkpoint itself and are actually restored.

For each run directory it:
  1. resolves the LATEST checkpoint (checkpoint-final.pth if the run finished, else the
     highest numbered keep_freq snapshot — never checkpoint-last/best, which are rewritten
     mid-epoch),
  2. derives the expected PoseGRU levers from the DIRECTORY NAME alone (the naming
     convention the eval harness relies on),
  3. independently derives them from the run's own .hydra/config.yaml,
  4. loads the checkpoint through the exact code path demo_ray.py uses
     (``ARCroco3DStereo.from_pretrained``) and reads back the live module's levers,
  5. asserts all three agree, that load_state_dict reported no missing/unexpected keys,
     and that every ``pose_gru.*`` tensor in the file is bit-identical to the live module,
  6. (unless --no-falsifier) runs a short real-data closed-loop inference twice — once with
     the restored GRU, once with it stripped — and asserts the predicted poses differ, so a
     silently random/inert GRU cannot pass as a trained arm.

Exit code 0 iff every run passes. Usage:
    python verify_gru_ckpt_matrix.py [--ckpt_root DIR] [--frames 8] [--no-falsifier]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

ROOT = "/scratch/bdursun25/cuteanything"
CUT3R_DIR = os.path.join(ROOT, "captain_gru_v3", "src", "CUT3R")
SCENE_DENSE = (
    "/frozen/avg/bora_data/droid_datasets/training_data/pointworld_droid_wrist_test/"
    "dl3dv_multi/RAIL+eh61f232+2023-10-26-17h-33m-59s/13062452+wrist/dense"
)

# Naming convention -> (mode, input_mode). The A index fixes the head algebra and the
# GRU input; G only routes training-time gradients and is inert under no_grad.
A_LEVERS = {
    "a1": ("direct", "pose"),
    "a2": ("residual", "pose"),
    "a3": ("direct", "pose_delta"),
    "a4": ("residual", "pose_delta"),
    "a5": ("split_anchor", "pose_delta"),
}


def resolve_latest_ckpt(run_dir):
    """checkpoint-final.pth (run completed) else the highest checkpoint-<N>.pth."""
    final = os.path.join(run_dir, "checkpoint-final.pth")
    if os.path.isfile(final):
        return final, "final(50)"
    snaps = []
    for p in glob.glob(os.path.join(run_dir, "checkpoint-*.pth")):
        m = re.fullmatch(r"checkpoint-(\d+)\.pth", os.path.basename(p))
        if m:
            snaps.append((int(m.group(1)), p))
    if not snaps:
        return None, None
    n, p = max(snaps)
    return p, f"epoch {n}"


def levers_from_name(name):
    """captain_gru_v{2,3}_a<N>_g<M>[_f1][_r<K>] -> expected lever dict."""
    m = re.search(r"_(a[1-5])_g([0-2])", name)
    if not m:
        raise ValueError(f"cannot parse run name {name!r}")
    mode, input_mode = A_LEVERS[m.group(1)]
    r = re.search(r"_r(\d+)(?:_|$)", name)
    return dict(
        mode=mode,
        input_mode=input_mode,
        hidden_dim=128,
        img_feat="input" if "_f1" in name else "none",
        img_feat_dim=32 if "_f1" in name else 0,
        img_feat_frames=2 if "_f1" in name else 0,
        iters=int(r.group(1)) if r else 1,
        g=int(m.group(2)),
    )


def levers_from_config(run_dir):
    """Read the run's own hydra config. Missing keys fall back to the trainer defaults
    that were in force when that config was written (pose input, R1, F0)."""
    cfg = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.isfile(cfg):
        return None
    t = open(cfg).read()

    def get(key, default=None):
        m = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", t, re.M)
        return m.group(1) if m else default

    img_feat = get("pose_gru_img_feat", "none")
    return dict(
        mode=get("pose_gru_mode"),
        input_mode=get("pose_gru_input", "pose"),
        hidden_dim=int(get("pose_gru_hidden_dim", "128")),
        img_feat=img_feat,
        img_feat_dim=int(get("pose_gru_img_feat_dim", "32")) if img_feat == "input" else 0,
        img_feat_frames=int(get("pose_gru_img_feat_frames", "2")) if img_feat == "input" else 0,
        iters=int(get("pose_gru_iters", "1")),
        feed_prev_pred=get("feed_prev_pred", "false") == "true",
        feed_gt_ray_map=get("feed_gt_ray_map", "false") == "true",
        feed_prev_gt_ray_map=get("feed_prev_gt_ray_map", "false") == "true",
    )


def live_levers(gru):
    return dict(
        mode=gru.mode,
        input_mode=gru.input_mode,
        hidden_dim=int(gru.hidden_dim),
        img_feat=gru.img_feat,
        img_feat_dim=int(gru.img_feat_dim),
        img_feat_frames=int(gru.img_feat_frames),
        iters=int(gru.iters),
    )


LEVER_KEYS = ("mode", "input_mode", "hidden_dim", "img_feat", "img_feat_dim",
              "img_feat_frames", "iters")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default=os.path.join(ROOT, "checkpoints", "captain_gru_overfit"))
    ap.add_argument("--frames", type=int, default=8, help="frames used by the liveness falsifier")
    ap.add_argument("--no-falsifier", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, CUT3R_DIR)
    from add_ckpt_path import add_path_to_dust3r

    run_dirs = sorted(
        d for d in glob.glob(os.path.join(args.ckpt_root, "*"))
        if os.path.isdir(d)
    )
    device = args.device if torch.cuda.is_available() else "cpu"

    img_paths = sorted(glob.glob(os.path.join(SCENE_DENSE, "rgb", "*.png")))[: args.frames]
    cam_dir = os.path.join(SCENE_DENSE, "cam")

    n_pass = n_fail = n_skip = 0
    imported = False
    for run_dir in run_dirs:
        name = os.path.basename(run_dir)
        ckpt, which = resolve_latest_ckpt(run_dir)
        if ckpt is None:
            print(f"[SKIP] {name}: no checkpoint yet (run still training)", flush=True)
            n_skip += 1
            continue

        print(f"\n=== {name}  ckpt={os.path.basename(ckpt)} [{which}]", flush=True)
        errs = []
        try:
            exp_name = levers_from_name(name)
            exp_cfg = levers_from_config(run_dir)

            if not imported:
                add_path_to_dust3r(ckpt)
                imported = True
            import demo_ray  # noqa: F401  (same module demo_ray.py/the worker import)
            from src.dust3r.inference import inference
            from src.dust3r.model import ARCroco3DStereo

            # --- the exact load demo_ray.run_inference performs -------------------
            model = ARCroco3DStereo.from_pretrained(ckpt).to(device)
            model.feed_prev_pred = True  # demo_ray sets this for prev_pred_gru
            model.eval()

            if getattr(model, "pose_gru", None) is None:
                errs.append("model.pose_gru is None after load (would silently become an "
                            "identity-init GRU == plain prev_pred)")
            else:
                got = live_levers(model.pose_gru)
                for k in LEVER_KEYS:
                    if got[k] != exp_name[k]:
                        errs.append(f"name-derived {k}={exp_name[k]!r} but live {k}={got[k]!r}")
                if exp_cfg is not None:
                    for k in LEVER_KEYS:
                        if exp_cfg.get(k) != got[k]:
                            errs.append(f"config {k}={exp_cfg.get(k)!r} but live {k}={got[k]!r}")
                    if not exp_cfg["feed_prev_pred"]:
                        errs.append("config feed_prev_pred is False — prev_pred_gru is the "
                                    "wrong conditioning for this run")
                    if exp_cfg["feed_gt_ray_map"] or exp_cfg["feed_prev_gt_ray_map"]:
                        errs.append("config feeds data-side GT rays — prev_pred_gru is wrong")
                print(f"    live levers: {got}", flush=True)

            # --- every pose_gru tensor must be bit-identical to the file ----------
            raw = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
            gru_keys = [k for k in raw if k.startswith(("pose_gru.", "module.pose_gru."))]
            if not gru_keys:
                errs.append("checkpoint carries NO pose_gru weights")
            live = dict(model.state_dict())
            n_checked = 0
            for k in gru_keys:
                lk = k.replace("module.", "", 1)
                if lk not in live:
                    errs.append(f"ckpt tensor {k} has no live counterpart")
                    continue
                a = raw[k].to(torch.float32)
                b = live[lk].detach().cpu().to(torch.float32)
                if a.shape != b.shape:
                    errs.append(f"{lk}: shape {tuple(a.shape)} vs live {tuple(b.shape)}")
                elif not torch.equal(a, b):
                    errs.append(f"{lk}: NOT bit-identical (max|d|={(a - b).abs().max():.3e})")
                else:
                    n_checked += 1
            print(f"    pose_gru tensors bit-identical: {n_checked}/{len(gru_keys)}", flush=True)

            # --- non-pose_gru keys: nothing missing / unexpected -------------------
            sd_keys = set(live)
            ckpt_keys = {k.replace("module.", "", 1) for k in raw}
            missing = sorted(k for k in sd_keys - ckpt_keys)
            unexpected = sorted(k for k in ckpt_keys - sd_keys)
            if missing:
                errs.append(f"{len(missing)} model keys absent from ckpt, e.g. {missing[:3]}")
            if unexpected:
                errs.append(f"{len(unexpected)} ckpt keys unused by model, e.g. {unexpected[:3]}")

            # --- GRU liveness falsifier ------------------------------------------
            if not args.no_falsifier and len(img_paths) >= 2:
                images, ray_maps, intr = demo_ray.load_frames_training_style(
                    img_paths, cam_dir, 320)
                views = demo_ray.prepare_input(images, ray_maps, intr,
                                               conditioning="prev_pred")
                with torch.no_grad():
                    out_g, _ = inference(views, model, device)
                pose_g = torch.cat([p["camera_pose"].cpu() for p in out_g["pred"]])
                model.pose_gru = None  # the GRU-less ablation
                views = demo_ray.prepare_input(images, ray_maps, intr,
                                               conditioning="prev_pred")
                with torch.no_grad():
                    out_n, _ = inference(views, model, device)
                pose_n = torch.cat([p["camera_pose"].cpu() for p in out_n["pred"]])
                d = (pose_g - pose_n).abs().max().item()
                print(f"    GRU-liveness falsifier: max|pose(GRU) - pose(no GRU)| = {d:.6f}",
                      flush=True)
                if not np.isfinite(d) or d <= 0:
                    errs.append("GRU is INERT (stripping it changes nothing) — not a trained arm")
                del out_g, out_n
            del model, raw, live
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            errs.append(f"EXCEPTION {e!r}")

        if errs:
            n_fail += 1
            print(f"    [FAIL] {name}", flush=True)
            for e in errs:
                print(f"        - {e}", flush=True)
        else:
            n_pass += 1
            print(f"    [PASS] {name}", flush=True)

    print(f"\n===== verify_gru_ckpt_matrix: {n_pass} pass, {n_fail} fail, "
          f"{n_skip} skipped (no ckpt) =====", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
