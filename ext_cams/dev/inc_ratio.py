import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt
d, eps = sys.argv[1], [l.strip() for l in open(sys.argv[2])]
rows = []
for ep in eps:
    f = f"{d}/{ep}/anchors.npz"
    if not os.path.isfile(f): continue
    z = np.load(f); fr, Pp = z["frames"], z["poses"]; gt = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", int(fr.max()) + 1, "gt")
    for i in range(1, len(fr)):
        if fr[i] != fr[i - 1] + 1: continue
        dp = np.linalg.inv(Pp[i - 1]) @ Pp[i]; dg = np.linalg.inv(gt[fr[i - 1]]) @ gt[fr[i]]
        rows.append(dict(ep=ep, dt=fr[i] - fr[0], est=rt.rot_angle_deg(dp[:3, :3]), gt=rt.rot_angle_deg(dg[:3, :3]),
                         est_t=np.linalg.norm(dp[:3, 3]) * 100, gt_t=np.linalg.norm(dg[:3, 3]) * 100))
r = pd.DataFrame(rows); r["b"] = pd.cut(r.dt, [0, 5, 20, 50, 100, 1e4])
print(r.groupby("b", observed=True).agg(n=("est", "size"), est_rot=("est", "median"), gt_rot=("gt", "median"), est_cm=("est_t", "median"), gt_cm=("gt_t", "median"),
      sum_est=("est", "sum"), sum_gt=("gt", "sum")).round(3).to_string())
