#!/usr/bin/env python
"""DEV DIAGNOSTIC (GT/FK labels): assign every birth seed to its nearest FK label link (positions_v1, at b), then measure
tracked-point distance to (a) the wrist/base-motion projection and (b) the seed's OWN link motion projection (Kabsch of
that link's label points b -> t), split by base vs finger. Usage: track_vs_gt_link.py TRACKS_DIR EPLIST label"""
import sys, os, csv
import numpy as np, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra
tdir, eps = sys.argv[1], [l.strip() for l in open(sys.argv[2]) if l.strip()]
ev = {r["episode"]: (int(r["entry_frame"]), int(r["exit_frame"])) for r in csv.DictReader(open(ra.EVENTS)) if r["event"] == "0"}

def label_world(z, s, t):
    uv = z[f"{s}_uv"][t].astype(np.float64); d = z[f"{s}_z"][t].astype(np.float64); K = z[f"{s}_K"]; W2C = z[f"{s}_W2C"]
    Xc = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0] * d, (uv[:, 1] - K[1, 2]) / K[1, 1] * d, d]
    return (Xc - W2C[:3, 3]) @ W2C[:3, :3]          # inv(W2C) applied: R^T (Xc - t)

rows = []; seeds_tab = []
for ep in eps:
    b, e = ev.get(ep, (None, None)); lf = f"{rte.POSITIONS}/{ep}_gripper.npz"
    if b is None or not os.path.isfile(lf): continue
    fs = {c: f"{tdir}/{ep}/{c}_f{b:05d}.npz" for c in ("ext1", "ext2")}
    if not all(os.path.isfile(f) for f in fs.values()): continue
    cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS); z = np.load(lf); s0 = str(z["serials"][0])
    zt = {c: np.load(fs[c]) for c in fs}; x = {c: zt[c]["queries"][:, 1:].astype(np.float64) for c in fs}
    Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], x["ext1"].T, x["ext2"].T); X = (Xh[:3] / Xh[3]).T
    T = min(zt["ext1"]["tracks"].shape[0], int(z["T"])); gt = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", T, "gt")
    Lb = label_world(z, s0, b); link = z["link"]
    dd = np.linalg.norm(X[:, None] - Lb[None], axis=2); nn = dd.argmin(1); nd = dd.min(1)
    cls = np.where(nd > 0.02, "far", np.where(link[nn] == "robotiq_85_base_link", "base", "finger"))
    for k in range(len(X)): seeds_tab.append(dict(ep=ep, k=k, cls=cls[k], nn_mm=nd[k] * 1000, link=link[nn[k]]))
    an = ra.Anchor(type("A", (), dict(tau_in=4, kabsch_m=0.01, huber_px=2))(), K, E, X)
    for t in range(b, T):
        if t not in gt: break
        M = gt[t] @ np.linalg.inv(gt[b]); Lt = label_world(z, s0, t)
        Xown = X.copy()
        for k in range(len(X)):
            ln = link[nn[k]]; m = link == ln
            R_, t_ = rt.kabsch(Lb[m], Lt[m]); Xown[k] = R_ @ X[k] + t_
        an2 = ra.Anchor(an.a, K, E, Xown)
        for ci, c in enumerate(("ext1", "ext2")):
            uvw, _ = an.project(ci, M[:3, :3], M[:3, 3]); uvo, _ = an2.project(ci, np.eye(3), np.zeros(3))
            tr = zt[c]["tracks"][t]; vi = zt[c]["visibility"][t]
            inimg = (uvo[:, 0] >= 0) & (uvo[:, 0] < rt.W) & (uvo[:, 1] >= 0) & (uvo[:, 1] < rt.H)
            for k in np.where(inimg)[0]:
                rows.append((ep, t - b, ci, k, cls[k], bool(vi[k]), float(np.linalg.norm(tr[k] - uvw[k])), float(np.linalg.norm(tr[k] - uvo[k])), float(np.linalg.norm(uvw[k] - uvo[k]))))
st = pd.DataFrame(seeds_tab); print(sys.argv[3], "scenes", st.ep.nunique(), "seed classes:", st.cls.value_counts().to_dict(), "per-scene base fraction median", st.groupby("ep").cls.apply(lambda s: (s == "base").mean()).median())
d = pd.DataFrame(rows, columns=["ep", "dt", "cam", "k", "cls", "vis", "d_wrist", "d_own", "wrist_vs_own"])
d["dtb"] = pd.cut(d.dt, [-1, 0, 5, 10, 20, 40, 80, 160, 1000])
g = d[d.vis].groupby(["cls", "dtb"], observed=True).agg(n=("k", "size"), d_own_med=("d_own", "median"), own_within3=("d_own", lambda v: (v < 3).mean()),
    own_within6=("d_own", lambda v: (v < 6).mean()), d_wrist_med=("d_wrist", "median"), wrist_vs_own_med=("wrist_vs_own", "median"))
print(g.round(3).to_string())
