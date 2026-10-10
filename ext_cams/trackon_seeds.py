#!/usr/bin/env python
"""trackon_seeds: Track-On-R (causal, frame by frame) on the guided cross-view seeds (idea 2, 2026-10-09).

Seeds = the epipolar-guided DISK matches at the FK birth frame b (rig_match_track.py, matches.npz x1/x2: the SAME physical
points in ext1 and ext2). Each camera is tracked independently with the birth-frame pipeline's CausalTracker
(vggt_features/vggt_probe/trackon_stage/causal_track.py, Track-On-R checkpoint, support grid 20 as on Leonardo), frames
b .. t_end fed one at a time (t_end = min(FK exit, store frames, MP4 frames) - 1). Nothing reads a pose.

Output DIR/<ep>/{ext1,ext2}_f<b:05d>.npz: tracks (T,N,2) float32 (NaN outside b..t_end), visibility (T,N) bool,
queries (N,3) [b,x,y], meta json. Run in the track_on_r env:
    python trackon_seeds.py --episodes LIST --seeds DIR --out DIR [--shard i --nshards n]
"""
import argparse, csv, json, os, sys, time
import numpy as np, torch, av

sys.path.insert(0, os.path.expanduser("~/vggt_features/vggt_probe/trackon_stage"))
from causal_track import CausalTracker  # noqa: E402

S = "/gpfs/scratch/etur59/koc821022"
RAW = f"{S}/vggt_cache/raw"; STORE = f"{S}/pointworld_droid_wrist_all/dl3dv_multi/wrist"
EVENTS = f"{S}/outputs/droid_birth_frames/entries/entry_events_gap3.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True); ap.add_argument("--seeds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--support_grid", type=int, default=20)
    a = ap.parse_args()
    eps = [l.strip() for l in open(a.episodes) if l.strip()]
    eps = [e for i, e in enumerate(eps) if i % a.nshards == a.shard]
    ev = {r["episode"]: (int(r["entry_frame"]), int(r["exit_frame"])) for r in csv.DictReader(open(EVENTS)) if r["event"] == "0"}
    tracker = CausalTracker(support_grid_size=a.support_grid)
    t0 = time.time(); nfr = 0
    for k, ep in enumerate(eps):
        sf = f"{a.seeds}/{ep}/matches.npz"
        if ep not in ev or not os.path.isfile(sf): continue
        b, e = ev[ep]
        meta = json.load(open(f"{RAW}/{ep}/metadata_{ep}.json"))
        n_store = len([f for f in os.listdir(f"{STORE}/{ep}/dense/cam") if f.endswith(".npz")])
        z = np.load(sf); seeds = {"ext1": z["x1"].astype(np.float32), "ext2": z["x2"].astype(np.float32)}
        N = len(seeds["ext1"])
        outs = {c: f"{a.out}/{ep}/{c}_f{b:05d}.npz" for c in seeds}
        if all(os.path.isfile(o) for o in outs.values()): continue
        os.makedirs(f"{a.out}/{ep}", exist_ok=True)
        for c in ("ext1", "ext2"):
            mp4 = f"{RAW}/{ep}/recordings/MP4/{meta[f'{c}_cam_serial']}.mp4"
            t_lim = min(e, n_store)                                  # exclusive
            tr, vi = [], []
            tracker.reset()
            with av.open(mp4) as cont:
                st = cont.streams.video[0]; st.thread_type = "AUTO"
                for t, fr in enumerate(cont.decode(st)):
                    if t >= t_lim: break
                    if t < b: continue
                    img = fr.to_ndarray(format="rgb24")
                    assert img.shape[:2] == (720, 1280), img.shape
                    p, v = tracker.step(img, new_points=seeds[c] if t == b else None)
                    tr.append(p.numpy().astype(np.float32)); vi.append(v.numpy().astype(bool))
            T = b + len(tr)
            TR = np.full((T, N, 2), np.nan, np.float32); VI = np.zeros((T, N), bool)
            if tr: TR[b:] = np.stack(tr); VI[b:] = np.stack(vi)
            np.savez_compressed(outs[c] + ".tmp.npz", tracks=TR, visibility=VI,
                                queries=np.c_[np.full(N, b, np.float32), seeds[c]].astype(np.float32),
                                meta=json.dumps(dict(episode=ep, cam=c, birth=b, exit=e, n_store=n_store, frames_tracked=len(tr),
                                                     support_grid=a.support_grid, source="trackon_seeds (guided DISK seeds)")))
            os.replace(outs[c] + ".tmp.npz", outs[c]); nfr += len(tr)
        if (k + 1) % 10 == 0:
            el = time.time() - t0; print(f"{k+1}/{len(eps)} eps, {nfr} frames, {nfr/el:.1f} fr/s", flush=True)
    print(f"done shard {a.shard}: {nfr} frames in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
