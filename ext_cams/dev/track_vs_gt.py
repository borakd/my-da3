#!/usr/bin/env python
"""DEV DIAGNOSTIC (GT): distance of tracked seed points to the GT projection of their triangulated birth point, by time
since birth. Usage: track_vs_gt.py TRACKS_DIR EPLIST [label]"""
import sys, os, csv, glob
import numpy as np, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra
tdir, eps = sys.argv[1], [l.strip() for l in open(sys.argv[2]) if l.strip()]
ev = {r["episode"]: (int(r["entry_frame"]), int(r["exit_frame"])) for r in csv.DictReader(open(ra.EVENTS)) if r["event"] == "0"}
rows = []
for ep in eps:
    b, e = ev.get(ep, (None, None))
    if b is None: continue
    fs = {c: f"{tdir}/{ep}/{c}_f{b:05d}.npz" for c in ("ext1", "ext2")}
    if not all(os.path.isfile(f) for f in fs.values()): continue
    cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS)
    zt = {c: np.load(fs[c]) for c in fs}; x = {c: zt[c]["queries"][:, 1:].astype(np.float64) for c in fs}
    Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], x["ext1"].T, x["ext2"].T); X = (Xh[:3] / Xh[3]).T
    T = zt["ext1"]["tracks"].shape[0]; gt = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", T, "gt")
    an = ra.Anchor(type("A", (), dict(tau_in=4, kabsch_m=0.01, huber_px=2))(), K, E, X)
    for t in range(b, T):
        if t not in gt: break
        M = gt[t] @ np.linalg.inv(gt[b])
        for ci, c in enumerate(("ext1", "ext2")):
            uv, _ = an.project(ci, M[:3, :3], M[:3, 3]); tr = zt[c]["tracks"][t]; vi = zt[c]["visibility"][t]
            d = np.linalg.norm(tr - uv, axis=1)
            inimg = (uv[:, 0] >= 0) & (uv[:, 0] < rt.W) & (uv[:, 1] >= 0) & (uv[:, 1] < rt.H)
            for k in range(len(X)):
                rows.append((ep, t - b, ci, k, bool(vi[k]), float(d[k]), bool(inimg[k])))
d = pd.DataFrame(rows, columns=["ep", "dt", "cam", "k", "vis", "dist", "inimg"])
d["dtb"] = pd.cut(d.dt, [-1, 0, 5, 10, 20, 40, 80, 160, 1000])
g = d[d.inimg].groupby("dtb", observed=True).apply(lambda s: pd.Series(dict(n=len(s), vis=s.vis.mean(), dist_med_vis=s[s.vis].dist.median(),
    within3_vis=(s[s.vis].dist < 3).mean(), within6_vis=(s[s.vis].dist < 6).mean(), within6_all=(s.dist < 6).mean())))
print(sys.argv[3] if len(sys.argv) > 3 else tdir, "scenes", d.ep.nunique()); print(g.round(3).to_string())
