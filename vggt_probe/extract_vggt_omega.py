#!/usr/bin/env python3
"""VGGT-Omega probe: exterior-camera-anchored wrist pose/depth for ONE DROID
episode and ONE variant, written in the CUT3R evaluator layout, then scored.

Outputs (out_root = --out_root, name = --out_name or --variant):
  <out_root>/<name>/preds/<EP>/camera/%06d.npz   pose = c2w 4x4 float32, world = wrist_0
                                                 camera frame (CUT3R convention),
                                                 VGGT scale; intrinsics = STORE wrist
                                                 K (320x180) from dense/cam/000000.npz;
                                                 intrinsics_vggt = VGGT's own K at the
                                                 network input resolution (384x688)
  <out_root>/<name>/preds/<EP>/depth/%06d.npy    (H,W) float32 at network res (384x688,
                                                 up-to-scale); the evaluator resizes it
  <out_root>/<name>/meta/<EP>.json               per-frame stats + run stats
  <out_root>/<name>/eval/<EP>/eval_depth_pose_metrics.csv   evaluator output vs the
                                                 REAL store dense/ (skipped if present)
  (anchored_t0 only) <cache_root>/t0/<EP>.npz    t=0 token cache

Variants (--variant):
  anchored_t0      per t: [wrist_0, ext1_0, ext2_0, wrist_t]           save idx 3
  anchored_sliding per t: [wrist_0, ext1_0, ext2_0, wrist_{t-1}, wrist_t] save idx 4
                   (t=1 uses wrist_0 as wrist_{t-1}, i.e. a duplicated frame, so S=5
                   is constant and batching works)
  wrist_pair       per t: [wrist_0, wrist_t]                             save idx 1
  ext_first        per t: [ext1_0, ext2_0, wrist_0, wrist_t]            save idx 3,
                   re-expressed relative to wrist_0 (index 2)
  offline_all      ONE forward over [wrist_0 .. wrist_{T-1}] (non-causal reference);
                   skipped with a note when T > --max_offline_frames (default 400)
  offline_all_ext  ONE forward over [wrist_0, ext1_0, ext2_0, wrist_1 .. wrist_{T-1}]
                   (same T cap)
  For every per-t variant, t=0 is a separate forward without wrist_t (identity pose,
  depth from that forward).  All poses are E_ref @ inv(E_t) with ref = wrist_0 in that
  same forward, so wrist_0 is exactly identity in every variant's world.

Per-frame forwards are batched: images [B, S, 3, H, W]; B halves on CUDA OOM.

Env: conda 'vggt-omega' (VGGT-Omega installed editable), PYTHONNOUSERSITE=1.
Must run under Slurm (login nodes have a 300 s CPU cap); ALLOW_NO_SLURM=1 overrides.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import glob
import json
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image

DEFAULT_CKPT = "/gpfs/home/koc/koc821022/vggt-omega/checkpoints/VGGT-Omega-1B-512/model.pt"
DEFAULT_SCENES_ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
DEFAULT_OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe"
DEFAULT_CACHE_ROOT = "/gpfs/scratch/etur59/koc821022/vggt_cache"
_WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EVAL_SCRIPT = os.path.join(_WT, "eval_bundle", "bin", "eval_depth_poses.py")

PER_T_VARIANTS = ("anchored_t0", "anchored_sliding", "wrist_pair", "ext_first")
OFFLINE_VARIANTS = ("offline_all", "offline_all_ext")
ALL_VARIANTS = PER_T_VARIANTS + OFFLINE_VARIANTS
NEEDS_EXT = ("anchored_t0", "anchored_sliding", "ext_first", "offline_all_ext")


# --------------------------------------------------------------------------- inputs
def preprocess_pil(img, image_resolution=512, patch_size=16):
    """Exact mirror of vggt_omega.utils.load_fn.load_and_preprocess_images for one
    in-memory PIL image (mode='balanced'): aspect crop -> balanced target -> bicubic
    resize -> ToTensor in [0,1].  Reuses the library's private helpers so any drift
    in the library shows up here too."""
    from vggt_omega.utils.load_fn import _balanced_target_shape, _crop_to_supported_aspect_ratio
    from torchvision import transforms as TF
    img = _crop_to_supported_aspect_ratio(img.convert("RGB"))
    w, h = img.size
    th, tw = _balanced_target_shape(h / max(w, 1), image_resolution, patch_size)
    img = img.resize((tw, th), Image.Resampling.BICUBIC)
    return TF.ToTensor()(img)


def load_paths(paths, image_resolution):
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    return load_and_preprocess_images(list(paths), image_resolution=image_resolution)


def load_mp4_frames(mp4_path, n_frames, image_resolution):
    """Decode the first n_frames of an MP4 (store frame i == MP4 frame i) and
    preprocess them like PNGs.  Returns [n,3,H,W]."""
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in range(n_frames):
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"{mp4_path}: decode failed at frame {i} (reported {total} frames, need {n_frames})")
        out.append(preprocess_pil(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), image_resolution))
    cap.release()
    return torch.stack(out), total


def load_source(spec, image_resolution):
    """One frame from 'path.png' or 'raw:<mp4>[:frame_idx]' -> [3,H,W]."""
    if spec.startswith("raw:"):
        rest = spec[4:]
        mp4, idx = rest, 0
        if ":" in rest and rest.rsplit(":", 1)[1].isdigit():
            mp4, idx = rest.rsplit(":", 1)
            idx = int(idx)
        frames, _ = load_mp4_frames(mp4, idx + 1, image_resolution)
        return frames[idx]
    return load_paths([spec], image_resolution)[0]


# --------------------------------------------------------------------------- geometry
def to44(e34):
    out = np.eye(4, dtype=np.float64)
    out[:3, :4] = e34
    return out


def rel_c2w(extr, ref_idx, idx):
    """c2w of frame idx expressed in the camera frame of frame ref_idx.
    extr: [S,3,4] camera-from-world (OpenCV).  c2w_rel = E_ref @ inv(E_idx)."""
    return to44(extr[ref_idx]) @ np.linalg.inv(to44(extr[idx]))


def cam_center(extr, idx):
    return np.linalg.inv(to44(extr[idx]))[:3, 3]


# --------------------------------------------------------------------------- model
class Runner:
    def __init__(self, ckpt, device):
        from vggt_omega.models import VGGTOmega
        t0 = time.time()
        self.model = VGGTOmega().to(device).eval()
        sd = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(sd)
        del sd
        self.device = device
        self.load_s = time.time() - t0
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def forward(self, imgs):
        """imgs [B,S,3,H,W] (cpu or gpu) -> dict of numpy: pose_enc [B,S,9],
        extr [B,S,3,4], intr [B,S,3,3], depth [B,S,H,W], conf [B,S,H,W]; plus wall s."""
        from vggt_omega.utils.pose_enc import encoding_to_camera
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            pred = self.model(imgs.to(self.device, non_blocking=True))
            extr, intr = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
            out = {
                "pose_enc": pred["pose_enc"].float().cpu().numpy(),
                "extr": extr.float().cpu().numpy(),
                "intr": intr.float().cpu().numpy(),
                "depth": pred["depth"][..., 0].float().cpu().numpy(),
                "conf": pred["depth_conf"].float().cpu().numpy(),
            }
        torch.cuda.synchronize()
        out["wall_s"] = time.time() - t0
        del pred, extr, intr
        return out

    def aggregator_tokens(self, imgs, layers):
        """imgs [S,3,H,W] -> ({layer: [S,P,2048] fp16 numpy}, patch_token_start)."""
        agg = self.model.aggregator
        agg.cached_layer_indices = set(layers) | {4, 11, 17, 23}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self.amp_dtype):
            toks, ps = agg(imgs[None].to(self.device))
        out = {l: toks[l][0].to(torch.float16).cpu().numpy() for l in layers}
        del toks
        return out, int(ps)


def forward_batched(runner, samples, bank, batch, log):
    """samples: list of index lists (all same length S).  Yields (sample_slice, out)
    per executed batch; halves batch on OOM."""
    i = 0
    while i < len(samples):
        b = min(batch, len(samples) - i)
        idx = samples[i:i + b]
        imgs = torch.stack([torch.stack([bank[i] for i in smp]) for smp in idx])  # [b,S,3,H,W]
        try:
            out = runner.forward(imgs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            log(f"OOM at B={b}; retrying with B={batch}")
            continue
        yield i, idx, out, batch
        i += b


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep", required=True, help="episode id, e.g. RAIL+80edfcb1+2023-07-14-14h-28m-45s")
    ap.add_argument("--variant", required=True, choices=ALL_VARIANTS)
    ap.add_argument("--out_name", default=None, help="output subdir name (default = --variant)")
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--scenes_root", default=DEFAULT_SCENES_ROOT)
    ap.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--ext1", default=None, help="ext1 t=0 frame: PNG path or raw:<mp4>[:idx]")
    ap.add_argument("--ext2", default=None, help="ext2 t=0 frame: PNG path or raw:<mp4>[:idx]")
    ap.add_argument("--wrist_src", default="store", help="'store' (dense/rgb 320x180 PNGs) or raw:<wrist.mp4>")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--eval_script", default=DEFAULT_EVAL_SCRIPT)
    ap.add_argument("--batch", type=int, default=8, help="forwards per call for per-t variants")
    ap.add_argument("--max_offline_frames", type=int, default=400)
    ap.add_argument("--image_resolution", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo inference even if meta says complete")
    ap.add_argument("--no_parity_check", action="store_true", help="skip the B=1 vs batched check on t=1")
    args = ap.parse_args()

    if not os.environ.get("SLURM_JOB_ID") and os.environ.get("ALLOW_NO_SLURM") != "1":
        sys.exit("ERROR: refuse to run inference outside Slurm (login nodes: 300 s CPU cap). Set ALLOW_NO_SLURM=1 to override.")

    name = args.out_name or args.variant
    tag = f"[{args.variant}->{name} {args.ep}]"

    def log(msg):
        print(f"{tag} {msg}", flush=True)

    scene = os.path.join(args.scenes_root, args.ep)
    gt_dense = os.path.join(scene, "dense")
    rgb_paths = sorted(glob.glob(os.path.join(gt_dense, "rgb", "*.png")))
    T = len(rgb_paths)
    if T < 2:
        sys.exit(f"{tag} ERROR: {T} store frames in {gt_dense}/rgb")
    store_K = np.load(os.path.join(gt_dense, "cam", "000000.npz"))["intrinsic"].astype(np.float32)

    pred_dir = os.path.join(args.out_root, name, "preds", args.ep)
    meta_dir = os.path.join(args.out_root, name, "meta")
    eval_dir = os.path.join(args.out_root, name, "eval", args.ep)
    meta_path = os.path.join(meta_dir, f"{args.ep}.json")
    eval_csv = os.path.join(eval_dir, "eval_depth_pose_metrics.csv")
    for d in (os.path.join(pred_dir, "camera"), os.path.join(pred_dir, "depth"), meta_dir, eval_dir):
        os.makedirs(d, exist_ok=True)

    if args.variant in NEEDS_EXT and not (args.ext1 and args.ext2):
        sys.exit(f"{tag} ERROR: --ext1/--ext2 required for {args.variant}")

    meta = {
        "ep": args.ep, "variant": args.variant, "out_name": name, "T": T,
        "wrist_src": args.wrist_src, "ext1": args.ext1, "ext2": args.ext2,
        "ckpt": args.ckpt, "image_resolution": args.image_resolution,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "complete": False, "skipped": None,
        "frames": [],
    }

    # ---------- resume check
    prev = None
    if os.path.isfile(meta_path) and not args.force:
        try:
            with open(meta_path) as f:
                prev = json.load(f)
        except Exception:
            prev = None
    n_cam = len(glob.glob(os.path.join(pred_dir, "camera", "*.npz")))
    n_dep = len(glob.glob(os.path.join(pred_dir, "depth", "*.npy")))
    inference_done = bool(prev and prev.get("complete") and n_cam == T and n_dep == T)
    if prev and prev.get("skipped"):
        log(f"previously skipped: {prev['skipped']}; nothing to do")
        return
    if inference_done:
        log(f"inference already complete ({T} frames); skipping to eval")
        meta = prev
    else:
        # ---------- offline cap
        if args.variant in OFFLINE_VARIANTS and T > args.max_offline_frames:
            note = f"T={T} > max_offline_frames={args.max_offline_frames}; offline variant skipped (memory ~74 MB/frame)"
            meta["skipped"] = note
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=1)
            log(f"SKIP: {note}")
            return

        # ---------- inputs
        t0 = time.time()
        if args.wrist_src == "store":
            wrist = load_paths(rgb_paths, args.image_resolution)
        elif args.wrist_src.startswith("raw:"):
            wrist, mp4_total = load_mp4_frames(args.wrist_src[4:], T, args.image_resolution)
            meta["wrist_mp4_frame_count"] = mp4_total
            if mp4_total != T:
                log(f"WARNING: MP4 reports {mp4_total} frames, store has {T} (using the first {T})")
        else:
            sys.exit(f"{tag} ERROR: bad --wrist_src {args.wrist_src}")
        bank = [wrist[i] for i in range(T)]           # 0..T-1 = wrist_t
        W0, E1, E2 = 0, T, T + 1
        if args.ext1 and args.ext2:
            bank.append(load_source(args.ext1, args.image_resolution))
            bank.append(load_source(args.ext2, args.image_resolution))
        shapes = {tuple(x.shape) for x in bank}
        if len(shapes) != 1:
            sys.exit(f"{tag} ERROR: mixed preprocessed shapes {shapes} (would need padding)")
        H, W = bank[0].shape[-2:]
        meta["input_hw"] = [int(H), int(W)]
        meta["preprocess_s"] = round(time.time() - t0, 2)
        log(f"T={T} frames at {H}x{W}, preprocess {meta['preprocess_s']}s, wrist_src={args.wrist_src}")

        # ---------- model
        runner = Runner(args.ckpt, args.device)
        meta["model_load_s"] = round(runner.load_s, 2)
        torch.cuda.reset_peak_memory_stats()
        log(f"model loaded in {runner.load_s:.1f}s")

        def anchor_stats(extr, amap):
            """distances between the anchor cameras in this forward (VGGT scale)."""
            c = {k: cam_center(extr, i) for k, i in amap.items()}
            st = {}
            if "wrist0" in c and "ext1" in c:
                st["d_w0_e1"] = float(np.linalg.norm(c["wrist0"] - c["ext1"]))
            if "wrist0" in c and "ext2" in c:
                st["d_w0_e2"] = float(np.linalg.norm(c["wrist0"] - c["ext2"]))
            if "ext1" in c and "ext2" in c:
                st["d_e1_e2"] = float(np.linalg.norm(c["ext1"] - c["ext2"]))
            return st

        def save_frame(t, pose, depth, K_vggt):
            np.save(os.path.join(pred_dir, "depth", f"{t:06d}.npy"), depth.astype(np.float32))
            np.savez(os.path.join(pred_dir, "camera", f"{t:06d}.npz"),
                     pose=pose.astype(np.float32), intrinsics=store_K,
                     intrinsics_vggt=K_vggt.astype(np.float32))

        def frame_meta(t, out_b, save_idx, ref_idx, amap, wall_s, S, extra=None):
            pe = out_b["pose_enc"][save_idx]
            pose = rel_c2w(out_b["extr"], ref_idx, save_idx)
            fm = {"t": t, "conf_mean": float(out_b["conf"][save_idx].mean()),
                  "conf_median": float(np.median(out_b["conf"][save_idx])),
                  "fov_h": float(pe[7]), "fov_w": float(pe[8]),
                  "t_norm": float(np.linalg.norm(pose[:3, 3])),
                  "fwd_wall_s": round(float(wall_s), 4), "S": int(S)}
            fm.update(anchor_stats(out_b["extr"], amap))
            if extra:
                fm.update(extra)
            return fm, pose

        t_inf = time.time()
        if args.variant in OFFLINE_VARIANTS:
            if args.variant == "offline_all":
                idx = list(range(T)); wpos = {t: t for t in range(T)}; amap = {"wrist0": 0}
            else:
                idx = [W0, E1, E2] + list(range(1, T)); wpos = {0: 0}; wpos.update({t: t + 2 for t in range(1, T)})
                amap = {"wrist0": 0, "ext1": 1, "ext2": 2}
            log(f"offline forward over S={len(idx)} frames")
            out = runner.forward(torch.stack([bank[i] for i in idx])[None])
            ob = {k: v[0] for k, v in out.items() if k != "wall_s"}
            for t in range(T):
                fm, pose = frame_meta(t, ob, wpos[t], 0, amap, out["wall_s"] / T, len(idx))
                save_frame(t, pose, ob["depth"][wpos[t]], ob["intr"][wpos[t]])
                meta["frames"].append(fm)
            meta["offline_forward_wall_s"] = round(out["wall_s"], 3)
            meta["batch_used"] = 1
        else:
            # ---- t = 0 forward (no wrist_t)
            if args.variant == "wrist_pair":
                idx0, ref0, amap0 = [W0], 0, {"wrist0": 0}
            elif args.variant == "ext_first":
                idx0, ref0, amap0 = [E1, E2, W0], 2, {"ext1": 0, "ext2": 1, "wrist0": 2}
            else:
                idx0, ref0, amap0 = [W0, E1, E2], 0, {"wrist0": 0, "ext1": 1, "ext2": 2}
            out0 = runner.forward(torch.stack([bank[i] for i in idx0])[None])
            ob0 = {k: v[0] for k, v in out0.items() if k != "wall_s"}
            pred0 = rel_c2w(ob0["extr"], ref0, ref0)  # identity by construction
            dev0 = to44(ob0["extr"][ref0])
            fm0, _ = frame_meta(0, ob0, ref0, ref0, amap0, out0["wall_s"], len(idx0),
                                {"ref_extr_dev_from_identity": float(np.abs(dev0 - np.eye(4)).max())})
            save_frame(0, np.eye(4), ob0["depth"][ref0], ob0["intr"][ref0])
            meta["frames"].append(fm0)
            meta["t0_pose_enc"] = ob0["pose_enc"].tolist()
            meta["t0_order"] = [{0: "wrist0", T: "ext1", T + 1: "ext2"}[i] for i in idx0]

            # ---- t=0 token cache (anchored_t0 only)
            if args.variant == "anchored_t0":
                cache_path = os.path.join(args.cache_root, "t0", f"{args.ep}.npz")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                toks, ps = runner.aggregator_tokens(torch.stack([bank[i] for i in idx0]), (11, 17, 23))
                tmp = cache_path + ".tmp.npz"
                np.savez(tmp,
                         camera_and_register_tokens=toks[23][:, :ps],
                         tokens_l11=toks[11], tokens_l17=toks[17], tokens_l23=toks[23],
                         pose_enc=ob0["pose_enc"].astype(np.float32),
                         extr=ob0["extr"].astype(np.float32), intr=ob0["intr"].astype(np.float32),
                         order=np.array(meta["t0_order"]), hw=np.array([H, W]),
                         patch_token_start=np.array(ps), patch_grid=np.array([H // 16, W // 16]),
                         sources=np.array([args.wrist_src, args.ext1, args.ext2]))
                os.replace(tmp, cache_path)
                meta["t0_token_cache"] = cache_path
                meta["t0_token_P"] = int(toks[23].shape[1])
                del toks
                log(f"t=0 token cache -> {cache_path} (P={meta['t0_token_P']}, ps={ps})")

            # ---- t >= 1 forwards, batched
            def sample(t):
                if args.variant == "anchored_t0":
                    return [W0, E1, E2, t], 3, 0, {"wrist0": 0, "ext1": 1, "ext2": 2}
                if args.variant == "anchored_sliding":
                    return [W0, E1, E2, t - 1, t], 4, 0, {"wrist0": 0, "ext1": 1, "ext2": 2}
                if args.variant == "wrist_pair":
                    return [W0, t], 1, 0, {"wrist0": 0}
                if args.variant == "ext_first":
                    return [E1, E2, W0, t], 3, 2, {"ext1": 0, "ext2": 1, "wrist0": 2}
                raise ValueError(args.variant)

            ts = list(range(1, T))
            samples = [sample(t)[0] for t in ts]
            _, save_idx, ref_idx, amap = sample(1)
            S = len(samples[0])
            n_batches = 0
            parity = None
            for i0, idx, out, b_used in forward_batched(runner, samples, bank, args.batch, log):
                b = len(idx)
                n_batches += 1
                for j in range(b):
                    t = ts[i0 + j]
                    ob = {k: v[j] for k, v in out.items() if k != "wall_s"}
                    fm, pose = frame_meta(t, ob, save_idx, ref_idx, amap, out["wall_s"] / b, S,
                                          {"batch": b})
                    save_frame(t, pose, ob["depth"][save_idx], ob["intr"][save_idx])
                    meta["frames"].append(fm)
                if n_batches == 1 and b > 1 and not args.no_parity_check:
                    # B=1 forward of t=1 vs its batched result (numerics-only check)
                    o1 = runner.forward(torch.stack([bank[i] for i in idx[0]])[None])
                    p_b = rel_c2w(out["extr"][0], ref_idx, save_idx)
                    p_1 = rel_c2w(o1["extr"][0], ref_idx, save_idx)
                    d_b = out["depth"][0, save_idx]; d_1 = o1["depth"][0, save_idx]
                    parity = {"t": ts[0], "B": b, "pose_max_abs_diff": float(np.abs(p_b - p_1).max()),
                              "depth_mean_abs_rel_diff": float((np.abs(d_b - d_1) / np.maximum(np.abs(d_1), 1e-6)).mean()),
                              "single_fwd_wall_s": round(o1["wall_s"], 4), "batched_fwd_wall_s": round(out["wall_s"], 4)}
                    log(f"batch parity B={b} vs 1 at t={ts[0]}: {parity}")
                if n_batches == 1 or n_batches % 20 == 0:
                    log(f"batch {n_batches} B={b} S={S} {out['wall_s']:.2f}s ({out['wall_s']/b:.3f}s/frame) "
                        f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
                meta["batch_used"] = b_used
            meta["batch_requested"] = args.batch
            meta["n_batches"] = n_batches
            meta["batch_parity"] = parity

        meta["frames"].sort(key=lambda r: r["t"])
        meta["inference_wall_s"] = round(time.time() - t_inf, 2)
        meta["peak_mem_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        meta["peak_mem_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        fw = [r["fwd_wall_s"] for r in meta["frames"][1:]] or [meta["frames"][0]["fwd_wall_s"]]
        meta["fwd_wall_s_per_frame_mean"] = round(float(np.mean(fw)), 4)
        meta["complete"] = len(meta["frames"]) == T
        meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(tmp, meta_path)
        log(f"inference done: {len(meta['frames'])}/{T} frames in {meta['inference_wall_s']}s, "
            f"B={meta.get('batch_used')}, peak alloc {meta['peak_mem_alloc_gib']} GiB, "
            f"{meta['fwd_wall_s_per_frame_mean']}s/frame")
        del runner
        torch.cuda.empty_cache()

    # ---------- evaluator
    if args.no_eval:
        return
    if os.path.isfile(eval_csv) and os.path.getsize(eval_csv) > 0:
        log(f"eval CSV exists; skipping evaluator")
    else:
        tmp_csv = os.path.join(eval_dir, "eval_depth_pose_metrics.csv.tmp")
        cmd = [sys.executable, args.eval_script, "--pred_root", pred_dir, "--gt_root", gt_dense,
               "--output_csv", tmp_csv]
        log("running evaluator: " + " ".join(cmd))
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not (os.path.isfile(tmp_csv) and os.path.getsize(tmp_csv) > 0):
            sys.exit(f"{tag} ERROR: evaluator rc={r.returncode}\n{r.stderr[-2000:]}")
        os.replace(tmp_csv, eval_csv)
        log(f"evaluator done in {time.time()-t0:.1f}s -> {eval_csv}")
    with open(eval_csv) as f:
        for line in f:
            if ",MEAN," in line:
                log("MEAN " + line.strip())


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
