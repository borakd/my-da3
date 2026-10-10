"""DEV: per-frame rotation error of rig arms split by GT gripper visibility (dev/vis) and frames since reference."""
import sys, os, json, numpy as np, pandas as pd
V = sys.argv[1]; eps = [l.strip() for l in open(sys.argv[2])]; arms = [a.split("=", 1) for a in sys.argv[3:]]
VB = [-0.01, 0.1, 0.3, 0.5, 1.01]; VL = ["<0.1", "0.1-0.3", "0.3-0.5", ">=0.5"]
# window visibility profile (all window frames, independent of any arm)
prof = []
for ep in eps:
    vf = f"{V}/{ep}.npz"; rf = f"{arms[0][1]}/{ep}/record.json"
    if not os.path.isfile(vf) or not os.path.isfile(rf): continue
    w = json.load(open(rf))["windows"][0]
    if "window" not in w: continue
    v = np.load(vf)["vis"]; b, e = w["window"]
    for t in range(b, min(e + 1, len(v))): prof.append(dict(dt=t - b, vmin=np.nanmin(v[t]), vmax=np.nanmax(v[t])))
p = pd.DataFrame(prof); p["b"] = pd.cut(p.dt, [-1, 5, 20, 50, 100, 1e4], labels=["0-5", "6-20", "21-50", "51-100", ">100"])
print("window frames: fraction with min-view visibility >= 0.3 / >= 0.5, and max-view >= 0.3, by frames since birth")
print(p.groupby("b", observed=True).agg(n=("dt", "size"), min_ge03=("vmin", lambda s: (s >= .3).mean()), min_ge05=("vmin", lambda s: (s >= .5).mean()), max_ge03=("vmax", lambda s: (s >= .3).mean())).round(2).to_string())
print("overall:", round((p.vmin >= .3).mean(), 3), "of window frames have >= 30% of the gripper visible in BOTH views")
rows = []
for name, d in arms:
    for ep in eps:
        vf = f"{V}/{ep}.npz"; pf = f"{d}/{ep}/per_frame.csv"; rf = f"{d}/{ep}/record.json"
        if not (os.path.isfile(vf) and os.path.isfile(pf)): continue
        w = json.load(open(rf))["windows"][0]; tr = w.get("t_ref")
        if tr is None: continue
        q = pd.read_csv(pf); q = q[q.rot_err_deg.notna()]
        if "solved" in q: q = q[q.solved.astype(bool)]
        v = np.load(vf)["vis"]
        for f, r in zip(q.frame, q.rot_err_deg):
            if f < len(v): rows.append((name, ep, f - tr, np.nanmin(v[f]), r))
d = pd.DataFrame(rows, columns=["arm", "ep", "dt", "vmin", "rot"]); d["vb"] = pd.cut(d.vmin, VB, labels=VL)
d["b"] = pd.cut(d.dt, [-1, 20, 50, 1e4], labels=["<=20", "21-50", ">50"])
print("\nper-frame rot err median [solved frames] by min-view visibility x frames since reference")
t = d.groupby(["arm", "b", "vb"], observed=False).rot.agg(["median", "size"])
t["s"] = t["median"].round(1).astype(str) + " [" + t["size"].astype(str) + "]"
print(t["s"].unstack("vb").to_string())
