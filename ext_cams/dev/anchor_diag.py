#!/usr/bin/env python
"""DEV DIAGNOSTIC (uses GT; not part of any method): for the birth model X_B of one scene, project it with the GT motion
gt[t] @ inv(gt[b]) and measure, per frame and view, the DISK dense similarity to the birth descriptor AT the true pixel,
the best similarity in a radius, and the distance of that best peak from the true pixel."""
import sys, os, json, csv
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra

ep, out = sys.argv[1], sys.argv[2]; radius = float(sys.argv[3]) if len(sys.argv) > 3 else 30
ev = [r for r in csv.DictReader(open(ra.EVENTS)) if r["episode"] == ep and r["event"] == "0"][0]; b, e = int(ev["entry_frame"]), int(ev["exit_frame"])
cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS)
store = f"{rt.STORE_ROOT}/{ep}/dense/cam"; T = min(e, rt.count_files(store, ".npz"))
zs = np.load(f"{ra.SEEDS}/{ep}/matches.npz"); x = {"ext1": zs["x1"], "ext2": zs["x2"]}
gray = {c: rt.read_frames(f"{rt.RAW_ROOT}/{ep}/recordings/MP4/{s}.mp4", T) for c, s in cams}; T = min(T, min(len(g) for g in gray.values()))
gt = rt.load_poses(store, T, "gt")
Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], x["ext1"].T, x["ext2"].T); X = (Xh[:3] / Xh[3]).T
an = ra.Anchor(type("A", (), dict(tau_in=4, kabsch_m=0.01, huber_px=2))(), K, E, X)
qz = {c: np.load(f"{ra.V2}/{ep}/{c}_f{b:05d}.npz") for c in x}; dn = ra.Dense(); refs = []; s_b = {}; zb = {}
import torch
for ci, c in enumerate(("ext1", "ext2")):
    m = ra.region_mask(qz[c]["queries"][:, 1:], 8); ys, xs = np.nonzero(m)
    box = (max(xs.min() - 20, 0), max(ys.min() - 20, 0), min(xs.max() + 21, rt.W), min(ys.max() + 21, rt.H))
    s_b[c] = 640 / max(box[2] - box[0], box[3] - box[1]); D, geo = dn.map(gray[c][b], box, s_b[c]); refs.append(dn.sample(D, geo, x[c]))
    zb[c] = float(an.project(ci, np.eye(3), np.zeros(3))[1].mean())
refs_both = [torch.stack([refs[0][k], refs[1][k]]) for k in range(len(X))]
rows = []
for t in range(b, T, 2):
    M = gt[t] @ np.linalg.inv(gt[b]); R, tt = M[:3, :3], M[:3, 3]
    for ci, c in enumerate(("ext1", "ext2")):
        uv, z = an.project(ci, R, tt); ok = (uv[:, 0] >= 0) & (uv[:, 0] < rt.W) & (uv[:, 1] >= 0) & (uv[:, 1] < rt.H)
        idx = np.where(ok)[0]
        if not len(idx): continue
        box = ra_box = (max(int(uv[idx, 0].min() - radius - 8), 0), max(int(uv[idx, 1].min() - radius - 8), 0), min(int(uv[idx, 0].max() + radius + 9), rt.W), min(int(uv[idx, 1].max() + radius + 9), rt.H))
        scale = float(np.clip(s_b[c] * np.median(z[idx]) / zb[c], 0.5 * s_b[c], 2 * s_b[c]))
        D, geo = dn.map(gray[c][t], box, scale)
        at = dn.sample(D, geo, uv[idx])
        for mode, rf in (("own", [refs[ci][k][None] for k in idx]), ("both", [refs_both[k] for k in idx])):
            pos, pk, sec = dn.search(D, geo, rf, uv[idx], radius, 4.0)
            sim_at = np.array([float((rf[j] @ at[j]).max()) for j in range(len(idx))])
            for j, k in enumerate(idx):
                rows.append(dict(t=t, dt=t - b, cam=c, k=k, mode=mode, sim_true=sim_at[j], peak=pk[j], second=sec[j], dist_px=float(np.linalg.norm(pos[j] - uv[k])), z=z[k], scale=scale))
import pandas as pd
d = pd.DataFrame(rows); os.makedirs(out, exist_ok=True); d.to_csv(f"{out}/{ep}_diag.csv", index=False)
d["dtb"] = pd.cut(d.dt, [-1, 0, 5, 10, 20, 40, 80, 160, 1000])
g = d.groupby(["mode", "dtb"], observed=True).agg(n=("k", "size"), sim_true=("sim_true", "median"), peak=("peak", "median"), sec=("second", "median"),
    dist_med=("dist_px", "median"), frac_within3=("dist_px", lambda v: (v < 3).mean()), frac_within6=("dist_px", lambda v: (v < 6).mean()))
print(ep); print(g.round(3).to_string())
