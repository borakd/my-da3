"""DEV (GT): one-step LK accuracy of every birth seed, started at its GT projection at t-1, vs its GT projection at t;
split by inside/outside the RobotSeg birth mask (both views). Usage: lk_step_accuracy.py EPLIST"""
import sys, os, csv, numpy as np, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra
eps = [l.strip() for l in open(sys.argv[1]) if l.strip()]; it = pd.read_csv(f"{ra.MASKS}/items_v2.csv"); rows = []
ev = {r["episode"]: (int(r["entry_frame"]), int(r["exit_frame"])) for r in csv.DictReader(open(ra.EVENTS)) if r["event"] == "0"}
for ep in eps:
    sf = f"{ra.SEEDS}/{ep}/matches.npz"
    if ep not in ev or not os.path.isfile(sf): continue
    b, e = ev[ep]; cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS); zs = np.load(sf); x = {"ext1": zs["x1"], "ext2": zs["x2"]}
    Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], x["ext1"].T, x["ext2"].T); X = (Xh[:3] / Xh[3]).T
    T = min(e, rt.count_files(f"{rt.STORE_ROOT}/{ep}/dense/cam", ".npz")); gt = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", T, "gt")
    gray = {c: rt.read_frames(f"{rt.RAW_ROOT}/{ep}/recordings/MP4/{s}.mp4", T) for c, s in cams}; T = min(T, min(len(g) for g in gray.values()))
    an = ra.Anchor(type("A", (), dict(tau_in=4, kabsch_m=0.01, huber_px=2))(), K, E, X)
    inside = np.ones(len(X), bool)
    for c in x:
        r_ = it[(it.episode == ep) & (it.frame == b) & (it.cam == c)]; m = cv2.imread(f"{ra.MASKS}/{r_.mask_relpath.iloc[0]}", 0)
        m = cv2.resize(m, (rt.W, rt.H), interpolation=cv2.INTER_NEAREST) > 0; xi = np.clip(np.round(x[c]).astype(int), [0, 0], [rt.W - 1, rt.H - 1]); inside &= m[xi[:, 1], xi[:, 0]]
    for ci, c in enumerate(("ext1", "ext2")):
        for t in range(b + 1, T):
            if t not in gt or t - 1 not in gt: break
            M0 = gt[t - 1] @ np.linalg.inv(gt[b]); M1 = gt[t] @ np.linalg.inv(gt[b])
            u0, _ = an.project(ci, M0[:3, :3], M0[:3, 3]); u1, _ = an.project(ci, M1[:3, :3], M1[:3, 3])
            ok = (u0[:, 0] > 20) & (u0[:, 0] < rt.W - 20) & (u0[:, 1] > 20) & (u0[:, 1] < rt.H - 20)
            if not ok.any(): continue
            p, st, _ = cv2.calcOpticalFlowPyrLK(gray[c][t - 1], gray[c][t], u0[ok].astype(np.float32).reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3)
            p = p.reshape(-1, 2); err = np.linalg.norm(p - u1[ok], axis=1); mot = np.linalg.norm(u1[ok] - u0[ok], axis=1)
            for j, k in enumerate(np.where(ok)[0]): rows.append((ep, c, k, t - b, bool(inside[k]), float(err[j]), float(mot[j]), int(st[j])))
d = pd.DataFrame(rows, columns=["ep", "cam", "k", "dt", "inside", "err", "motion", "st"])
d["mb"] = pd.cut(d.motion, [-0.01, 0.5, 2, 5, 15, 1000])
print("seeds inside both masks: %.2f" % d.drop_duplicates(["ep", "k"]).inside.mean())
print(d.groupby(["inside", "mb"], observed=True).err.describe(percentiles=[.25, .5, .75]).round(2).to_string())
