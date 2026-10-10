#!/usr/bin/env python
"""RobotSeg (showlab/RobotSeg, CVPR 2026) out of the box on the two DROID exterior cameras of one episode.

Model: ~/RobotSeg @ dafb8c0, checkpoint checkpoints/robotseg.pt (OneDrive release, 165,436,221 B, sha256 1384c38c...),
config robotseg-infer, bf16 autocast + inference_mode exactly as test/demo.py. No finetuning, no hyper-parameter changes.
Env: /gpfs/scratch/etur59/koc821022/conda_envs/robotseg (venv over the recal3r env: py3.11, torch 2.5.1+cu121 = the
README pins; + iopath, kornia, timm, opencv-contrib for the guided-filter post-processing). Run with PYTHONNOUSERSITE=1.

Variants (all re-prompt the learned class token on every frame during propagation, as RobotSeg does):
  auto_f0_<cat>     AUTOMATIC, the demo.py protocol: add_new_robot(frame 0, robot=<cat>), propagate forward over all frames.
                    cat in {gripper, arm, robot}. This is the "out of the box" row (no prompt, no GT).
  auto_entry_gripper  AUTOMATIC from the first frame where the REFERENCE gripper mask is non-empty, forward only (the VRS
                    benchmark's "AU" protocol: start_idx = first GT-non-empty frame; PRIVILEGED start frame). Frames
                    before entry are left empty.
  image_gripper     per-frame IMAGE mode: reset, add_new_robot(t) on every frame independently (no memory).
  bbox_entry_gripper  VRS "BB" protocol: box = bounding rect of the reference mask on the entry frame, forward only
                    (PRIVILEGED prompt from the reference).
  auto_vis_gripper / bbox_vis_gripper  NON-VRS sensitivity rows: same as auto_entry / bbox_entry but started at the first
                    frame where the reference area is >= 10% of its episode maximum (the entry frame shows only a sliver
                    of the gripper at the top edge: ext1 f27 = 129 px). PRIVILEGED start frame / box.
Seeds: the robot prompt generator draws torch.randint on every tracked frame (unseeded upstream). --seed k seeds torch
before EACH variant so every cell is reproducible and order-independent; the first run (no --seed) was unseeded.
Reference = SAM3 box+click masks (human-prompted, approved 2026-09-28; NOT human-annotated GT), see SHELF_single_episode.md.

Resolutions (say which!): --res 1280x720 (native MP4, gripper ~1% of the image) and --res 320x180 (PointWorld store /
scoring resolution, standing rule; cv2 INTER_AREA downscale of the same decoded frames). RobotSeg resizes every input to
1024x1024 internally; masks come back at the input resolution.

Outputs, per res under OUT_ROOT/<res>/<EP>/:
  frames/<cam>/%05d.jpg                                 the JPEG folder RobotSeg reads (cv2 q=95)
  <cam>_<serial>__robotseg_<variant>_masks.npz          MASK FILE CONTRACT (seedstudy): 'union' packbits (T,720,1280) of the
                                                        guided-refined masks (320x180 runs are upsampled x4 nearest), 'shape',
                                                        'frames_present'
  <cam>_<serial>__robotseg_<variant>_raw.npz            at the processed res: 'raw' / 'refined' packbits (T,h,w),
                                                        'logits' float16 (T,180,320) video-res mask logits (area-resized),
                                                        'obj_score' float32 (T,) object-score logit (NaN = frame not run),
                                                        'ran' bool (T,)
  run_info.json                                         timings, prompt frames, boxes, versions

    sbatch robotseg_gripper_ext.sbatch [EP]          # GPU
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam3_gripper_masks import RAW_ROOT, read_frames  # noqa: E402  (stdlib + cv2 only at import time)

ROBOTSEG = "/home/koc/koc821022/RobotSeg"
CKPT = os.path.join(ROBOTSEG, "checkpoints", "robotseg.pt")
CFG = "../robotseg/configs/robotseg-infer"  # same string as test/demo.py (resolved inside the robotseg hydra module)
OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/robotseg"
REF_DIR = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/gripper_sam3"
REF_SUFFIX = "__gripper_boxclick_masks.npz"
VARIANTS = ["auto_f0_gripper", "auto_f0_arm", "auto_f0_robot", "auto_entry_gripper", "image_gripper", "bbox_entry_gripper",
            "auto_vis_gripper", "bbox_vis_gripper"]
VIS_FRAC = 0.10


def load_test_utils():
    spec = importlib.util.spec_from_file_location("robotseg_test_utils", os.path.join(ROBOTSEG, "test", "utils.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_contract(path):
    z = np.load(path)
    T, H, W = [int(v) for v in z["shape"]]
    return np.unpackbits(z["union"], axis=2)[:, :, :W].astype(bool)


def save_contract(path, masks_720):
    T, H, W = masks_720.shape
    np.savez_compressed(path, union=np.packbits(masks_720.astype(np.uint8), axis=2), shape=np.array([T, H, W]),
                        frames_present=masks_720.reshape(T, -1).any(1))


def write_jpegs(frames, d):
    os.makedirs(d, exist_ok=True)
    for i, f in enumerate(frames):
        p = os.path.join(d, f"{i:05d}.jpg")
        if not os.path.exists(p):
            cv2.imwrite(p, f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return sorted(os.path.join(d, p) for p in os.listdir(d) if p.endswith(".jpg"))


def obj_score_of(state, t):
    for store in ("output_dict_per_obj", "temp_output_dict_per_obj"):
        od = state[store].get(0)
        if od is None:
            continue
        for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            out = od[key].get(t)
            if out is not None and out.get("object_score_logits") is not None:
                return float(out["object_score_logits"].float().flatten()[0])
    return float("nan")


def run_variant(predictor, state, variant, T, entry, entry_box, vis, vis_box, seed=None):
    """Return logits (T,h,w) float32 numpy (-inf where the frame was not run), obj_score (T,), ran (T,), seconds."""
    import torch
    h, w = state["video_height"], state["video_width"]
    logits = np.full((T, h, w), -1e4, np.float32)
    score = np.full(T, np.nan, np.float32)
    ran = np.zeros(T, bool)
    cat = variant.split("_")[-1]
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize()
    t0 = time.time()
    if variant.startswith("image_"):
        for t in range(T):
            predictor.reset_state(state)
            _, _, m = predictor.add_new_robot(inference_state=state, frame_idx=t, obj_id=0, robot=cat)
            logits[t] = m[0, 0].float().cpu().numpy()
            score[t] = obj_score_of(state, t)
            ran[t] = True
    else:
        predictor.reset_state(state)
        if variant.startswith("auto_f0_"):
            start = 0
            predictor.add_new_robot(inference_state=state, frame_idx=0, obj_id=0, robot=cat)
        elif variant == "auto_entry_gripper":
            start = entry
            predictor.add_new_robot(inference_state=state, frame_idx=entry, obj_id=0, robot=cat)
        elif variant == "bbox_entry_gripper":
            start = entry
            predictor.add_new_points_or_box(inference_state=state, frame_idx=entry, obj_id=0, box=np.asarray(entry_box, np.float32), robots=cat)
        elif variant == "auto_vis_gripper":
            start = vis
            predictor.add_new_robot(inference_state=state, frame_idx=vis, obj_id=0, robot=cat)
        elif variant == "bbox_vis_gripper":
            start = vis
            predictor.add_new_points_or_box(inference_state=state, frame_idx=vis, obj_id=0, box=np.asarray(vis_box, np.float32), robots=cat)
        else:
            raise ValueError(variant)
        for t, _, m in predictor.propagate_in_video(inference_state=state, robot=cat):
            logits[t] = m[0, 0].float().cpu().numpy()
            ran[t] = True
        for t in range(T):
            if ran[t]:
                score[t] = obj_score_of(state, t)
        assert ran[start:].all() and not ran[:start].any(), (variant, start, np.nonzero(ran)[0][:5])
    torch.cuda.synchronize()
    return logits, score, ran, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="RAIL+80edfcb1+2023-07-14-14h-28m-45s")
    ap.add_argument("--res", nargs="+", default=["1280x720", "320x180"])
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--no_guided", action="store_true", help="skip the guided-filter refinement (raw masks only)")
    ap.add_argument("--seed", type=int, default=None, help="seed torch before every variant (default: unseeded, as upstream)")
    ap.add_argument("--out_root", default=OUT_ROOT)
    args = ap.parse_args()
    ep = args.episode
    meta = json.load(open(os.path.join(RAW_ROOT, ep, f"metadata_{ep}.json")))
    cams = {c: meta[f"{c}_cam_serial"] for c in ("ext1", "ext2")}

    import torch
    sys.path.insert(0, ROBOTSEG)
    from robotseg.build_robotseg import build_robotseg_video_predictor
    tu = load_test_utils()
    torch.cuda.set_device(0)
    predictor = build_robotseg_video_predictor(CFG, CKPT)
    print(f"RobotSeg built: {sum(p.numel() for p in predictor.parameters()) / 1e6:.1f} M params on {torch.cuda.get_device_name(0)}", flush=True)

    frames_native = {c: read_frames(os.path.join(RAW_ROOT, ep, "recordings", "MP4", f"{s}.mp4")) for c, s in cams.items()}
    refs = {c: load_contract(os.path.join(REF_DIR, ep, f"{c}_{s}{REF_SUFFIX}")) for c, s in cams.items()}

    for res in args.res:
        W, H = [int(v) for v in res.split("x")]
        out_dir = os.path.join(args.out_root, res, ep)
        os.makedirs(out_dir, exist_ok=True)
        info = dict(episode=ep, res=res, seed=args.seed, robotseg_commit="dafb8c0", ckpt=CKPT, cfg=CFG, guided_filter=not args.no_guided,
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(0), cams={})
        for cam, ser in cams.items():
            fr = frames_native[cam]
            if (W, H) != (fr[0].shape[1], fr[0].shape[0]):
                fr = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in fr]
            jpgs = write_jpegs(fr, os.path.join(out_dir, "frames", cam))
            T = len(jpgs)
            imgs = [cv2.imread(p) for p in jpgs]   # what RobotSeg sees (post-JPEG), also the guided-filter guide
            ref = refs[cam]
            assert len(ref) == T, (len(ref), T)
            present = ref.reshape(T, -1).any(1)
            entry = int(np.argmax(present))
            sx, sy = W / ref.shape[2], H / ref.shape[1]
            area = ref.reshape(T, -1).sum(1)
            vis = int(np.argmax(area >= VIS_FRAC * area.max()))

            def box_at(t):
                ys, xs = np.nonzero(ref[t])
                return [xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy]
            entry_box, vis_box = box_at(entry), box_at(vis)
            cinfo = dict(serial=ser, T=T, ref_entry=entry, entry_box_px=[round(v, 1) for v in entry_box],
                         ref_vis=vis, vis_box_px=[round(v, 1) for v in vis_box], variants={})
            t0 = time.time()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(video_path=os.path.join(out_dir, "frames", cam), async_loading_frames=False,
                                             offload_video_to_cpu=False, offload_state_to_cpu=False)
                print(f"[{res} {cam}] init_state {time.time() - t0:.1f}s, T={T}, video {state['video_width']}x{state['video_height']}, "
                      f"ref entry f{entry}, entry box {cinfo['entry_box_px']}, vis f{vis} box {cinfo['vis_box_px']}", flush=True)
                for v in args.variants:
                    logits, score, ran, sec = run_variant(predictor, state, v, T, entry, entry_box, vis, vis_box, args.seed)
                    raw = logits > 0.0
                    if args.no_guided:
                        refined = raw.copy()
                    else:   # exactly what demo.py / the VRS inference script save: guided_refine_mask(mask*255, BGR image)
                        refined = np.stack([tu.guided_refine_mask((raw[t].astype(np.uint8) * 255), imgs[t]) > 0 for t in range(T)])
                    small = np.stack([cv2.resize(np.maximum(l, -50.0), (320, 180), interpolation=cv2.INTER_AREA) for l in logits]).astype(np.float16)
                    np.savez_compressed(os.path.join(out_dir, f"{cam}_{ser}__robotseg_{v}_raw.npz"),
                                        raw=np.packbits(raw, axis=2), refined=np.packbits(refined, axis=2), shape=np.array([T, H, W]),
                                        logits=small, obj_score=score, ran=ran)
                    up = refined if (W, H) == (1280, 720) else np.stack([cv2.resize(m.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST) > 0 for m in refined])
                    save_contract(os.path.join(out_dir, f"{cam}_{ser}__robotseg_{v}_masks.npz"), up)
                    n_run = int(ran.sum())
                    vi = dict(seconds=round(sec, 2), ms_per_frame=round(1000 * sec / max(n_run, 1), 1), frames_run=n_run,
                              frames_nonempty=int(refined.reshape(T, -1).any(1).sum()),
                              mean_area_frac_nonempty=float(refined.reshape(T, -1).mean(1)[refined.reshape(T, -1).any(1)].mean()) if refined.any() else 0.0,
                              obj_score_pos_frames=int(np.nansum(score > 0)))
                    cinfo["variants"][v] = vi
                    print(f"[{res} {cam}] {v:20s} {vi}", flush=True)
            info["cams"][cam] = cinfo
            del state
            torch.cuda.empty_cache()
        json.dump(info, open(os.path.join(out_dir, "run_info.json"), "w"), indent=1)
        print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
