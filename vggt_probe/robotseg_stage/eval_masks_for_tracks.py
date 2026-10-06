"""EVALUATION ONLY (not part of the pipeline): RobotSeg automatic gripper masks on a subset of frames
after the birth frame, used as pseudo ground truth to score point tracks. Nothing here feeds back into
the tracking outputs.

Usage: eval_masks_for_tracks.py --items EP:cam ... [--every 3]
Output: $O/comparisons/eval_masks/<ep>/<cam>_f<t>.png (255 = gripper), index.jsonl
"""
import argparse
import json
import os
import shutil
import sys

import av
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

O = bf.OUT_ROOT
RAW = "/leonardo_scratch/large/userexternal/bdursun0/robotseg_demo/raw"
OUT = f"{O}/comparisons/eval_masks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--every", type=int, default=3)
    args = ap.parse_args()
    births = {r["episode"]: r for r in json.load(open(f"{O}/birth_frames.json"))}
    torch.cuda.set_device(0)
    predictor = build_robotseg_video_predictor("../robotseg/configs/robotseg-infer", f"{ROBOTSEG_ROOT}/checkpoints/robotseg.pt")
    tmp = f"{OUT}/_tmp"
    os.makedirs(OUT, exist_ok=True)
    idx = open(f"{OUT}/index.jsonl", "a")
    for item in args.items:
        ep, cam = item.split(":")
        B = births[ep]["birth_f050"]
        meta = json.load(open(f"{bf.META_DIR}/{ep}.json"))
        sn = meta[f"{cam}_cam_serial"]
        os.makedirs(f"{OUT}/{ep}", exist_ok=True)
        with av.open(f"{RAW}/{ep}/recordings/MP4/{sn}.mp4") as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for t, fr in enumerate(c.decode(s)):
                if t < B or (t - B) % args.every:
                    continue
                dst = f"{OUT}/{ep}/{cam}_f{t:05d}.png"
                if os.path.isfile(dst):
                    continue
                img = cv2.cvtColor(fr.to_ndarray(format="rgb24"), cv2.COLOR_RGB2BGR)
                shutil.rmtree(tmp, ignore_errors=True)
                os.makedirs(tmp)
                cv2.imwrite(f"{tmp}/00000.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    state = predictor.init_state(video_path=tmp, async_loading_frames=False,
                                                 offload_video_to_cpu=False, offload_state_to_cpu=False)
                    _, _, logits = predictor.add_new_robot(inference_state=state, frame_idx=0, obj_id=0, robot="gripper")
                    mask = (logits[0, 0] > 0).cpu().numpy()
                    predictor.reset_state(state)
                del state
                cv2.imwrite(dst, mask.astype(np.uint8) * 255)
                idx.write(json.dumps(dict(episode=ep, cam=cam, frame=t, area=float(mask.mean()))) + "\n")
        idx.flush()
        print(item, "done", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
