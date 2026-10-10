#!/usr/bin/env python
"""Depth-gated RobotSeg on the DROID exterior cameras: the gripper mask must go to the CLOSEST robot (user idea, 2026-10-02).

Out of the box RobotSeg (robotseg_gripper_ext.py, RESULTS.md) segments "a robot gripper" well but locks onto a background
robot arm or people when our gripper is absent or small, and never reports absence (object score > 0 on every frame).
Depth = PointWorld-DROID FoundationStereo depth, converted to the wrist-store format by pwdepth_extract_convert.py:
pointworld_droid_ext_all/dl3dv_multi/<ext1|ext2>/<EP>/dense/depth/NNNNNN.npy, float32 (180, 320) metres, 0 = invalid.
Causal and GT-free: frame t uses only the depth of frame t and global constants.

TAU = 1.5 m: the kinematic wrist-camera centre is within 1.46 m of the exterior camera in 100% of 15,821 in-view samples
(250 TRAIN episodes, every 8th frame; median 0.59, p99 1.04). DELTA = 0.3 m.

RobotSeg runs (category "gripper", robotseg-infer, bf16, torch seed 0 before every run; raw masks, no guided filter):
  plain_*   the frames as they are                          (baselines = robotseg_gripper_ext.py rows)
  blankF_*  pixels with depth > TAU OR invalid -> mid-gray  (INPUT gating; invalid counted as far)
  blankK_*  pixels with depth > TAU -> mid-gray, invalid kept
  each as *_auto (add_new_robot on frame 0 + propagate_in_video, the demo protocol) and *_image (per-frame, no memory).
Output selection "dselF"/"dselK" (post-hoc, per frame): connected components (8-conn) of the mask, each scored by the
median of its valid depth (needs >= 20% valid pixels, else depth unknown); keep components with depth <= TAU and, of
those, the ones within DELTA of the nearest. dselF drops unknown-depth components; dselK keeps them when a near
component exists. An empty near set gives an empty mask (the absence decision RobotSeg lacks).
Variants written (contract npz, (T,720,1280) union): plain_auto, plain_image, plain_auto_dselF, plain_auto_dselK,
plain_image_dselF, plain_image_dselK, blankF_auto, blankF_image, blankF_auto_dselF, blankF_image_dselF,
blankK_auto, blankK_image, blankK_auto_dselK, blankK_image_dselK.

Outputs: OUT_ROOT/<res>/<EP>/<cam>_<serial>__rsdg_<variant>_masks.npz (+ <cam>_<serial>__rsdg_runs.npz with the six raw
runs: packbits masks, logits (T,180,320) float16, obj scores), frames_gated_<cam>_<F|K>.jpg previews, run_info.json.

    sbatch robotseg_depthgate.sbatch --episodes EP [EP ...] [--res 320x180 1280x720]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robotseg_gripper_ext import CFG, CKPT, ROBOTSEG, obj_score_of, save_contract  # noqa: E402
from sam3_gripper_masks import RAW_ROOT, read_frames  # noqa: E402

EXT_DEPTH = "/gpfs/scratch/etur59/koc821022/pointworld_droid_ext_all/dl3dv_multi"
OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/robotseg_depthgate"
TAU, DELTA, MIN_VALID = 1.5, 0.3, 0.2
GRAY = 128
RUNS = ["plain_auto", "plain_image", "blankF_auto", "blankF_image", "blankK_auto", "blankK_image"]


def load_depth(cam, ep):
    d = os.path.join(EXT_DEPTH, cam, ep, "dense", "depth")
    fs = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    return np.stack([np.load(os.path.join(d, f)) for f in fs])


def gate_frames(frames, depth, invalid_far):
    out = []
    for f, d in zip(frames, depth):
        far = d > TAU
        if invalid_far:
            far |= d == 0
        g = f.copy()
        g[far] = GRAY
        out.append(g)
    return out


def select_near(mask, depth, keep_unknown):
    if not mask.any():
        return mask
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    near, unknown = [], []
    for k in range(1, n):
        px = lab == k
        d = depth[px]
        v = d[d > 0]
        if len(v) < MIN_VALID * len(d):
            unknown.append(k)
            continue
        z = float(np.median(v))
        if z <= TAU:
            near.append((z, k))
    if not near:
        return np.zeros_like(mask)
    zmin = min(z for z, _ in near)
    keep = [k for z, k in near if z <= zmin + DELTA] + (unknown if keep_unknown else [])
    return np.isin(lab, keep)


def write_jpegs(frames, d):
    os.makedirs(d, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(d, f"{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])


def run_robotseg(predictor, jpg_dir, mode, T):
    import torch
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=jpg_dir, async_loading_frames=False, offload_video_to_cpu=False, offload_state_to_cpu=False)
        h, w = state["video_height"], state["video_width"]
        logits = np.full((T, h, w), -50.0, np.float32)
        score = np.full(T, np.nan, np.float32)
        t0 = time.time()
        if mode == "image":
            for t in range(T):
                predictor.reset_state(state)
                _, _, m = predictor.add_new_robot(inference_state=state, frame_idx=t, obj_id=0, robot="gripper")
                logits[t] = m[0, 0].float().cpu().numpy()
                score[t] = obj_score_of(state, t)
        else:
            predictor.add_new_robot(inference_state=state, frame_idx=0, obj_id=0, robot="gripper")
            for t, _, m in predictor.propagate_in_video(inference_state=state, robot="gripper"):
                logits[t] = m[0, 0].float().cpu().numpy()
            for t in range(T):
                score[t] = obj_score_of(state, t)
        torch.cuda.synchronize()
        sec = time.time() - t0
        del state
    torch.cuda.empty_cache()
    return logits, score, sec


def to720(masks):
    if masks.shape[1:] == (720, 1280):
        return masks
    return np.stack([cv2.resize(m.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST) > 0 for m in masks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--res", nargs="+", default=["320x180", "1280x720"])
    args = ap.parse_args()
    import torch
    sys.path.insert(0, ROBOTSEG)
    from robotseg.build_robotseg import build_robotseg_video_predictor
    torch.cuda.set_device(0)
    predictor = build_robotseg_video_predictor(CFG, CKPT)
    print(f"RobotSeg on {torch.cuda.get_device_name(0)}; TAU={TAU} DELTA={DELTA}", flush=True)
    for ep in args.episodes:
        if not all(os.path.isdir(os.path.join(EXT_DEPTH, c, ep, "dense", "depth")) for c in ("ext1", "ext2")):
            print(f"SKIP {ep}: exterior depth not converted yet", flush=True)
            continue
        meta = json.load(open(os.path.join(RAW_ROOT, ep, f"metadata_{ep}.json")))
        cams = {c: str(meta[f"{c}_cam_serial"]) for c in ("ext1", "ext2")}
        st = json.load(open(os.path.join(EXT_DEPTH, "_status", ep + ".json")))   # MP4 frame offset when a stream has extra frames
        native = {c: read_frames(os.path.join(RAW_ROOT, ep, "recordings", "MP4", f"{s}.mp4"))[st.get(f"{c}_offset", 0):][:st[f"F_{c}"]]
                  for c, s in cams.items()}
        depth180 = {c: load_depth(c, ep) for c in cams}
        for res in args.res:
            W, H = [int(v) for v in res.split("x")]
            out = os.path.join(OUT_ROOT, res, ep)
            os.makedirs(out, exist_ok=True)
            info = dict(episode=ep, res=res, tau=TAU, delta=DELTA, min_valid=MIN_VALID, cams={})
            for cam, ser in cams.items():
                fr = native[cam] if (W, H) == (1280, 720) else [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in native[cam]]
                dep = depth180[cam] if (W, H) == (320, 180) else np.stack([cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST) for d in depth180[cam]])
                T = min(len(fr), len(dep))
                assert len(fr) == len(dep), (ep, cam, len(fr), len(dep))
                inputs = {"plain": fr, "blankF": gate_frames(fr, dep, True), "blankK": gate_frames(fr, dep, False)}
                for k in ("blankF", "blankK"):
                    t = T // 2
                    cv2.imwrite(os.path.join(out, f"frames_gated_{cam}_{k[-1]}_f{t}.jpg"), np.concatenate([fr[t], inputs[k][t]], 1))
                runs, ci = {}, dict(serial=ser, T=T, runs={})
                tmp_root = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"rsdg_{os.getpid()}")
                for name in RUNS:
                    src, mode = name.split("_")
                    jd = os.path.join(tmp_root, f"{res}_{cam}_{src}")
                    if not os.path.isdir(jd):
                        write_jpegs(inputs[src], jd)
                    logits, score, sec = run_robotseg(predictor, jd, mode, T)
                    runs[name] = (logits > 0, logits, score)
                    ci["runs"][name] = dict(seconds=round(sec, 1), frames_nonempty=int((logits > 0).reshape(T, -1).any(1).sum()))
                    print(f"[{ep[:40]} {res} {cam}] {name:13s} {ci['runs'][name]}", flush=True)
                os.system(f"rm -rf {tmp_root}")
                np.savez_compressed(os.path.join(out, f"{cam}_{ser}__rsdg_runs.npz"),
                                    **{f"{n}_mask": np.packbits(runs[n][0], axis=2) for n in RUNS},
                                    **{f"{n}_logits": np.stack([cv2.resize(np.maximum(l, -50), (320, 180), interpolation=cv2.INTER_AREA) for l in runs[n][1]]).astype(np.float16) for n in RUNS},
                                    **{f"{n}_score": runs[n][2] for n in RUNS}, shape=np.array([T, H, W]))
                variants = {n: runs[n][0] for n in RUNS}
                for n in ("plain_auto", "plain_image"):
                    variants[n + "_dselF"] = np.stack([select_near(m, d, False) for m, d in zip(runs[n][0], dep)])
                    variants[n + "_dselK"] = np.stack([select_near(m, d, True) for m, d in zip(runs[n][0], dep)])
                for n in ("blankF_auto", "blankF_image"):
                    variants[n + "_dselF"] = np.stack([select_near(m, d, False) for m, d in zip(runs[n][0], dep)])
                for n in ("blankK_auto", "blankK_image"):
                    variants[n + "_dselK"] = np.stack([select_near(m, d, True) for m, d in zip(runs[n][0], dep)])
                for v, m in variants.items():
                    save_contract(os.path.join(out, f"{cam}_{ser}__rsdg_{v}_masks.npz"), to720(m))
                ci["variants"] = {v: int(m.reshape(T, -1).any(1).sum()) for v, m in variants.items()}
                info["cams"][cam] = ci
            json.dump(info, open(os.path.join(out, "run_info.json"), "w"), indent=1)
            print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
