import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt
from scipy.spatial.transform import Rotation as Rot
d, eps = sys.argv[1], [l.strip() for l in open(sys.argv[2])]
res = {-2: [], -1: [], 0: [], 1: [], 2: []}; mags = []
for ep in eps:
    f = f"{d}/{ep}/anchors.npz"
    if not os.path.isfile(f): continue
    z = np.load(f); fr = z["frames"]; M = z["motions"]; gt = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", int(fr.max()) + 3, "gt")
    Mi = {int(a): m for a, m in zip(fr, M)}
    b = int(fr[0])
    def ginc(t):   # GT world-frame rotation increment of the gripper motion between t-1 and t (rotvec, deg)
        if t - 1 not in gt or t not in gt: return None
        A = gt[t - 1] @ np.linalg.inv(gt[b]); B = gt[t] @ np.linalg.inv(gt[b]); return np.degrees(Rot.from_matrix(B[:3, :3] @ A[:3, :3].T).as_rotvec())
    for t in fr[1:]:
        t = int(t)
        if t - 1 not in Mi or t - b < 6: continue
        e = np.degrees(Rot.from_matrix(Mi[t][:3, :3] @ Mi[t - 1][:3, :3].T).as_rotvec())
        for L in res:
            g = ginc(t + L)
            if g is not None: res[L].append(np.r_[e, g])
for L, v in res.items():
    v = np.array(v); e, g = v[:, :3], v[:, 3:]
    err = np.linalg.norm(e - g, axis=1); corr = (e * g).sum() / np.sqrt((e * e).sum() * (g * g).sum())
    print(f"lag {L:+d}: median |est - gt| {np.median(err):.3f} deg, vector correlation {corr:.3f}, |est| {np.median(np.linalg.norm(e,axis=1)):.3f}, |gt| {np.median(np.linalg.norm(g,axis=1)):.3f}")
