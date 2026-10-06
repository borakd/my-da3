"""Re-track selected (scene, camera) items with several Track-On checkpoints using IDENTICAL query
points (taken from the production run's npz), so checkpoints can be compared directly.

Output: <out_root>/<ckpt_name>/tracks/<ep>/<cam>_f<t>.npz  (same layout as the production run)
        <out_root>/<ckpt_name>/timing.jsonl
Usage: compare_checkpoints_track.py --items EP:cam ... --ckpts name=path ...
"""
import argparse
import json
import os
import sys
import time

import av
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # causal_track.py sits alongside
from causal_track import CausalTracker  # noqa: E402

O = "/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames"
PROD = f"{O}/trackon_run"
RAW = "/leonardo_scratch/large/userexternal/bdursun0/robotseg_demo/raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path")
    ap.add_argument("--out_root", default=f"{O}/comparisons")
    ap.add_argument("--support_grid", type=int, default=20)
    args = ap.parse_args()
    births = {r["episode"]: r for r in json.load(open(f"{O}/birth_frames.json"))}
    metas = {}
    for name_path in args.ckpts:
        name, path = name_path.split("=", 1)
        tracker = CausalTracker(ckpt=path, support_grid_size=args.support_grid)
        os.makedirs(f"{args.out_root}/{name}/tracks", exist_ok=True)
        tlog = open(f"{args.out_root}/{name}/timing.jsonl", "a")
        for item in args.items:
            ep, cam = item.split(":")
            B = births[ep]["birth_f050"]
            stem = f"{cam}_f{B:05d}"
            prod = np.load(f"{PROD}/tracks/{ep}/{stem}.npz")
            queries = prod["queries"]  # (N,3) [t,x,y] in frame pixels
            pmeta = json.loads(str(prod["meta"]))
            if ep not in metas:
                metas[ep] = json.load(open(f"{O}/metadata/{ep}.json"))
            sn = metas[ep][f"{cam}_cam_serial"]
            tracker.reset()
            tracks, viss = [], []
            t0 = time.time()
            n = 0
            with av.open(f"{RAW}/{ep}/recordings/MP4/{sn}.mp4") as c:
                s = c.streams.video[0]
                s.thread_type = "AUTO"
                for t, fr in enumerate(c.decode(s)):
                    n = t + 1
                    if t < B:
                        continue
                    rgb = fr.to_ndarray(format="rgb24")
                    if t == B:
                        p, v = tracker.step(rgb, new_points=queries[:, 1:].tolist())
                    else:
                        p, v = tracker.step(rgb)
                    tracks.append(p.numpy())
                    viss.append(v.numpy())
            dt = time.time() - t0
            T, N = n, len(queries)
            tr = np.full((T, N, 2), np.nan, np.float32)
            vi = np.zeros((T, N), bool)
            tr[B:B + len(tracks)] = np.stack(tracks)
            vi[B:B + len(viss)] = np.stack(viss)
            meta = dict(pmeta)
            meta.update(ckpt=path, ckpt_name=name, n_frames_mp4=T, n_frames_tracked=len(tracks),
                        queries_from=f"{PROD}/tracks/{ep}/{stem}.npz")
            os.makedirs(f"{args.out_root}/{name}/tracks/{ep}", exist_ok=True)
            np.savez_compressed(f"{args.out_root}/{name}/tracks/{ep}/{stem}.npz", tracks=tr, visibility=vi,
                                queries=queries, meta=json.dumps(meta))
            rec = dict(ckpt=name, episode=ep, cam=cam, birth_frame=B, n_points=N, tracked_frames=len(tracks),
                       seconds=round(dt, 2), fps=round(len(tracks) / dt, 1),
                       gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       vis_mean=float(vi[B:].mean()), vis_last=float(vi[B + len(viss) - 1].mean()))
            tlog.write(json.dumps(rec) + "\n")
            tlog.flush()
            print(rec, flush=True)
        tlog.close()
        del tracker
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
