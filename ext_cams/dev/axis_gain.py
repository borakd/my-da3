"""DEV (GT): rotation gain of a rig arm per axis class: increment rotation vectors (world frame) of rig vs GT, projected on
each camera's viewing direction (in-plane rotation for that camera) and on the direction perpendicular to both view axes.
Usage: axis_gain.py EPLIST name=DIR ..."""
import sys, os, json, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt, rig_trackon_eval as rte
eps = [l.strip() for l in open(sys.argv[1])]; rows = []
for spec in sys.argv[2:]:
    name, d = spec.split("=", 1)
    for ep in eps:
        f = f"{d}/{ep}/anchors.npz"
        if not os.path.isfile(f): continue
        z = np.load(f); fr = z["frames"]; M = z["motions"]; b = int(fr[0]); Mi = {int(a): m for a, m in zip(fr, M)}
        gt = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", int(fr.max()) + 1, "gt"); cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS)
        d1 = E["ext1"][:3, :3].T @ np.array([0, 0, 1.0]); d2 = E["ext2"][:3, :3].T @ np.array([0, 0, 1.0]); dp = np.cross(d1, d2); dp /= np.linalg.norm(dp)
        for t in fr[1:]:
            t = int(t)
            if t - 1 not in Mi or t not in gt or t - 1 not in gt: continue
            A = gt[t - 1] @ np.linalg.inv(gt[b]); B = gt[t] @ np.linalg.inv(gt[b])
            wg = Rot.from_matrix(B[:3, :3] @ A[:3, :3].T).as_rotvec(); we = Rot.from_matrix(Mi[t][:3, :3] @ Mi[t - 1][:3, :3].T).as_rotvec()
            for ax, v in (("view1", d1), ("view2", d2), ("perp_both", dp)):
                rows.append(dict(arm=name, ep=ep, ax=ax, g=float(we @ v), e=float(wg @ v)))
d = pd.DataFrame(rows)
for (arm, ax), g in d.groupby(["arm", "ax"]):
    big = np.abs(g.e) > np.radians(0.2)
    slope = float((g.g[big] * g.e[big]).sum() / (g.e[big] ** 2).sum()); corr = float(np.corrcoef(g.g[big], g.e[big])[0, 1])
    print(f"{arm:22s} {ax:10s} gain {slope:.2f} corr {corr:.2f} n {big.sum()}")
