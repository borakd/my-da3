"""Single-frame RobotSeg gripper masks on the birth frame of every DROID test scene, both
exterior cameras, with the kinematic consistency check and box-prompt fallback.

Per scene and camera, two target frames: the 50 % birth frame (`b050`, the user's definition:
first frame with >= 50 % of the gripper model inside both views) and the 100 % birth frame
(`b100`, whole gripper model inside both views) when it exists and differs.

Steps per (scene, cam, frame):
  1. decode frame t from the exterior MP4 (frame index == store index; count mismatches logged)
  2. RobotSeg automatic prompt (RPG), single frame, no propagation
  3. kinematic check: project the gripper model at t; spill = fraction of mask pixels outside the
     projected bbox padded by 60 px; flagged if spill > 0.2 or mask area < 0.05 % of the image
  4. if flagged: re-run with the kinematic box prompt, keep whichever mask has lower spill
Outputs under $WORK/bora/outputs/droid_birth_frames/masks_run/:
  frames/<ep>/<cam>_f<t>.jpg   masks/<ep>/<cam>_f<t>.png   overlays/<ep>/<cam>_f<t>.jpg
  results_shard<k>.jsonl  (one row per mask, resumable)
Usage: segment_birth_frames.py --shard K --nshards N
"""
import argparse
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # vggt_probe: birth_frames.py
import birth_frames as bf  # noqa: E402
from robotseg.build_robotseg import build_robotseg_video_predictor  # noqa: E402
import robotseg  # noqa: E402

# RobotSeg clone (editable install). The config name below is resolved by Hydra relative to the
# robotseg package, not the cwd; only the checkpoint is a filesystem path.
ROBOTSEG_ROOT = os.environ.get("ROBOTSEG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(robotseg.__file__)))

RAW_ROOT = "/leonardo_scratch/large/userexternal/bdursun0/robotseg_demo/raw"
OUT = f"{bf.OUT_ROOT}/masks_run"
PAD = 60
SPILL_MAX = 0.2
AREA_MIN = 0.0005


def load_pose(ep, t):
    with np.load(f"{bf.STORE}/{ep}/dense/cam/{t:06d}.npz") as z:
        return z["pose"]


def decode_frame(mp4, t):
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(t, max(n - 1, 0)))
    ok, img = cap.read()
    cap.release()
    return (img if ok else None), n


def spill_area(mask, uv, vis):
    m = mask.astype(bool)
    area = float(m.mean())
    if vis.sum() and m.sum():
        x0, y0 = np.maximum(uv[vis].min(0) - PAD, 0).astype(int)
        x1, y1 = np.minimum(uv[vis].max(0) + PAD, [m.shape[1] - 1, m.shape[0] - 1]).astype(int)
        spill = float(1 - m[y0:y1 + 1, x0:x1 + 1].sum() / m.sum())
    else:
        spill = float("nan")
    return spill, area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    for sub in ("frames", "masks", "overlays", "tmp"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    tmpdir = f"{OUT}/tmp/shard{args.shard}"
    rows = json.load(open(f"{bf.OUT_ROOT}/birth_frames.json"))
    rows = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    if args.limit:
        rows = rows[: args.limit]
    res_path = f"{OUT}/results_shard{args.shard}.jsonl"
    done = set()
    if os.path.isfile(res_path):
        for line in open(res_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["episode"], r["cam"], r["which"]))
    print(f"shard {args.shard}/{args.nshards}: {len(rows)} scenes, {len(done)} masks already done", flush=True)

    torch.cuda.set_device(0)
    predictor = build_robotseg_video_predictor("../robotseg/configs/robotseg-infer", f"{ROBOTSEG_ROOT}/checkpoints/robotseg.pt")
    P_cam = bf.gripper_points_cam()
    fh = open(res_path, "a")
    t_start = time.time()
    n_done = 0
    for si, r in enumerate(rows):
        ep = r["episode"]
        if r["birth_f050"] < 0:
            for cam in ("ext1", "ext2"):
                if (ep, cam, "b050") not in done:
                    fh.write(json.dumps(dict(episode=ep, cam=cam, which="b050", status="no_birth_frame")) + "\n")
            fh.flush()
            continue
        targets = {"b050": r["birth_f050"]}
        if r["birth_f100"] >= 0 and r["birth_f100"] != r["birth_f050"]:
            targets["b100"] = r["birth_f100"]
        meta = json.load(open(f"{bf.META_DIR}/{ep}.json"))
        for cam in ("ext1", "ext2"):
            sn = meta[f"{cam}_cam_serial"]
            mp4 = f"{RAW_ROOT}/{ep}/recordings/MP4/{sn}.mp4"
            K = bf.load_zed_K(sn)
            T_bc = np.linalg.inv(bf.pose6_to_T(np.array(meta[f"{cam}_cam_extrinsics"])))
            for which, t in targets.items():
                if (ep, cam, which) in done:
                    continue
                row = dict(episode=ep, cam=cam, which=which, frame=int(t), n_frames=r["n_frames"], serial=sn)
                if not os.path.isfile(mp4):
                    row["status"] = "mp4_missing"
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    continue
                img, n_mp4 = decode_frame(mp4, t)
                row["mp4_frames"] = n_mp4
                if img is None:
                    row["status"] = "decode_failed"
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    continue
                if img.shape[:2] != (720, 1280):
                    img = cv2.resize(img, (1280, 720))
                    row["resized_from"] = list(img.shape[:2])
                pose = load_pose(ep, t)
                Pb = (pose[:3, :3] @ P_cam.T).T + pose[:3, 3]
                frac, uv, vis = bf.visible_fraction(T_bc, K, Pb)
                row["kin_frac"] = float(frac)
                # single-frame "video"
                shutil.rmtree(tmpdir, ignore_errors=True)
                os.makedirs(tmpdir)
                cv2.imwrite(f"{tmpdir}/00000.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    state = predictor.init_state(video_path=tmpdir, async_loading_frames=False,
                                                 offload_video_to_cpu=False, offload_state_to_cpu=False)
                    _, _, logits = predictor.add_new_robot(inference_state=state, frame_idx=0, obj_id=0, robot="gripper")
                    mask = (logits[0, 0] > 0).cpu().numpy()
                    sp, ar = spill_area(mask, uv, vis)
                    row.update(spill_auto=sp, area_auto=ar, prompt_used="auto")
                    flagged = (not np.isnan(sp) and sp > SPILL_MAX) or ar < AREA_MIN
                    row["flagged_auto"] = bool(flagged)
                    if flagged and vis.sum() > 0:
                        predictor.reset_state(state)
                        x0, y0 = np.maximum(uv[vis].min(0) - 25, 0)
                        x1, y1 = np.minimum(uv[vis].max(0) + 25, [1279, 719])
                        _, _, logits2 = predictor.add_new_points_or_box(
                            inference_state=state, frame_idx=0, obj_id=0, robots="gripper",
                            box=torch.tensor([x0, y0, x1, y1], dtype=torch.float32))
                        mask2 = (logits2[0, 0] > 0).cpu().numpy()
                        sp2, ar2 = spill_area(mask2, uv, vis)
                        row.update(spill_box=sp2, area_box=ar2, box=[float(x0), float(y0), float(x1), float(y1)])
                        if (np.isnan(sp) or (not np.isnan(sp2) and sp2 < sp)) and ar2 >= AREA_MIN:
                            mask, sp, ar = mask2, sp2, ar2
                            row["prompt_used"] = "box"
                    predictor.reset_state(state)
                del state
                row.update(spill_final=sp, area_final=ar,
                           flagged_final=bool((not np.isnan(sp) and sp > SPILL_MAX) or ar < AREA_MIN))
                os.makedirs(f"{OUT}/frames/{ep}", exist_ok=True)
                os.makedirs(f"{OUT}/masks/{ep}", exist_ok=True)
                os.makedirs(f"{OUT}/overlays/{ep}", exist_ok=True)
                stem = f"{cam}_f{t:05d}"
                cv2.imwrite(f"{OUT}/frames/{ep}/{stem}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(f"{OUT}/masks/{ep}/{stem}.png", (mask.astype(np.uint8) * 255))
                ov = img.copy()
                ov[mask] = (0.45 * ov[mask] + 0.55 * np.array([255, 80, 0])).astype(np.uint8)
                cv2.putText(ov, f"{ep} {cam} {which} f{t} {row['prompt_used']} spill {sp:.2f} area {100*ar:.2f}%",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(f"{OUT}/overlays/{ep}/{stem}.jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
                row.update(status="ok", mask=f"masks/{ep}/{stem}.png")
                fh.write(json.dumps(row) + "\n"); fh.flush()
                n_done += 1
        if (si + 1) % 50 == 0:
            el = time.time() - t_start
            print(f"  {si+1}/{len(rows)} scenes, {n_done} masks, {el/60:.1f} min, {el/max(n_done,1):.2f} s/mask", flush=True)
    fh.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"shard {args.shard} done: {n_done} new masks in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
