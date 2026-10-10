"""DEV (GT): error of the rig's birth-model CENTROID position (translation part) vs the GT motion of the same centroid,
pooled by frames since the reference. Usage: centroid_err.py EPLIST name=DIR ..."""
import sys, os, json, numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra
eps = [l.strip() for l in open(sys.argv[1])]; rows = []
for spec in sys.argv[2:]:
    name, d = spec.split("=", 1)
    for ep in eps:
        f = f"{d}/{ep}/anchors.npz"
        if not os.path.isfile(f) or not os.path.isfile(f"{ra.SEEDS}/{ep}/matches.npz"): continue
        z = np.load(f); w = json.load(open(f"{d}/{ep}/record.json"))["windows"][0]; b = int(w["t_ref"])
        if "model_centroid" in z.files: cb = z["model_centroid"]
        else:
            cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS); zs = np.load(f"{ra.SEEDS}/{ep}/matches.npz")
            Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], zs["x1"].T, zs["x2"].T); cb = (Xh[:3] / Xh[3]).T.mean(0)
        fr = z["frames"]; gt = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", int(fr.max()) + 1, "gt")
        for t, M in zip(fr, z["motions"]):
            if t == b or t not in gt: continue
            Mg = gt[t] @ np.linalg.inv(gt[b]); e = np.linalg.norm((M[:3, :3] @ cb + M[:3, 3]) - (Mg[:3, :3] @ cb + Mg[:3, 3])) * 100
            rows.append((name, ep, t - b, e))
d = pd.DataFrame(rows, columns=["arm", "ep", "dt", "cm"]); d["b"] = pd.cut(d.dt, [0, 5, 20, 50, 100, 1e4], labels=["1-5", "6-20", "21-50", "51-100", ">100"])
print("model-centroid position error (cm), pooled median [p90] by frames since reference")
print(d.groupby(["arm", "b"], observed=False).cm.agg(lambda s: f"{s.median():.2f} [{s.quantile(.9):.1f}]").unstack("b").to_string())
print("\nmedian of scene medians:", d.groupby(["arm", "ep"]).cm.median().groupby("arm").median().round(2).to_dict())
