"""Window-definition rig dirs for the CUT3R controls: per scene record.json (t_ref = birth b, window = [b, n_store-1]) and
anchors.npz holding only the birth pose (GT(b), the hand-eye stand-in). Usage: make_window_dirs.py EPLIST OUTDIR"""
import sys, os, csv, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt, rig_anchor as ra
eps = [l.strip() for l in open(sys.argv[1]) if l.strip()]; out = sys.argv[2]
ev = {r["episode"]: int(r["entry_frame"]) for r in csv.DictReader(open(ra.EVENTS)) if r["event"] == "0"}
n = 0; lens = []
for ep in eps:
    if ep not in ev: continue
    b = ev[ep]; store = f"{rt.STORE_ROOT}/{ep}/dense/cam"; T = rt.count_files(store, ".npz")
    if T - b < 2: continue
    gtb = np.load(f"{store}/{b:06d}.npz")["pose"].astype(np.float64)
    os.makedirs(f"{out}/{ep}", exist_ok=True)
    json.dump(dict(episode=ep, tag="window_def", windows=[dict(episode=ep, t_ref=b, window=[b, T - 1], window_len=T - b, n_anchored=1)]), open(f"{out}/{ep}/record.json", "w"))
    np.savez(f"{out}/{ep}/anchors.npz", frames=np.array([b]), poses=np.array([gtb])); n += 1; lens.append(T - b)
lens = np.array(lens); print(f"{n} scenes; window length median {np.median(lens):.0f}, p10 {np.percentile(lens,10):.0f}, p90 {np.percentile(lens,90):.0f}, max {lens.max()}, total frames {lens.sum()}")
