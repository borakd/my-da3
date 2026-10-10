#!/usr/bin/env python
"""DEV DIAGNOSTIC (FK labels + PointWorld exterior depth; never a method input): per frame and exterior view, the
fraction of the FK gripper label points (positions_v1) that are in the image, in front of the camera and NOT occluded
(|depth map - label depth| < OCC_M, depth map at the label pixel; 320x180). Writes OUT/<ep>.npz: vis (T, 2) float,
ninimg (T, 2). Usage: gripper_visibility.py EPLIST OUTDIR"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_trackon_eval as rte
DEP = "/gpfs/scratch/etur59/koc821022/pointworld_droid_ext_all/dl3dv_multi"; OCC_M = 0.03
eps = [l.strip() for l in open(sys.argv[1]) if l.strip()]; out = sys.argv[2]; os.makedirs(out, exist_ok=True)
for ep in eps:
    of = f"{out}/{ep}.npz"
    if os.path.isfile(of): continue
    lf = f"{rte.POSITIONS}/{ep}_gripper.npz"; sf = f"{DEP}/_status/{ep}.json"
    if not os.path.isfile(lf) or not os.path.isfile(sf): continue
    z = np.load(lf); st = json.load(open(sf)); meta = json.load(open(f"/gpfs/scratch/etur59/koc821022/vggt_cache/raw/{ep}/metadata_{ep}.json"))
    T = int(z["T"]); vis = np.full((T, 2), np.nan); nin = np.zeros((T, 2), int)
    for ci, c in enumerate(("ext1", "ext2")):
        s = str(meta[f"{c}_cam_serial"])
        if f"{s}_uv" not in z.files: continue
        off = int(st.get(f"{c}_offset", 0) or 0)
        uv = z[f"{s}_uv"]; dz = z[f"{s}_z"].astype(np.float32); ok = z[f"{s}_in_frame"] & z[f"{s}_front"]
        for t in range(T):
            f = f"{DEP}/{c}/{ep}/dense/depth/{t:06d}.npy"
            if not os.path.isfile(f): break
            D = np.load(f); m = ok[t]; nin[t, ci] = int(m.sum())
            if not m.any(): vis[t, ci] = 0.0; continue
            u = np.clip(np.round(uv[t, m, 0]).astype(int), 0, 319); v = np.clip(np.round(uv[t, m, 1]).astype(int), 0, 179)
            d = D[v, u]; good = (d > 0) & (np.abs(d - dz[t, m]) < OCC_M)
            vis[t, ci] = good.sum() / len(X) if (X := z["link"]) is not None else 0
    np.savez(of, vis=vis, ninimg=nin)
