"""DEV (GT check): fresh epipolar-guided DISK matches at frame t (true 3D points at t from the calibration alone) mapped
into the gripper frame with the GT pose at t; nearest-neighbour distance to the birth seeds mapped with the GT pose at b.
If GT (FK + extrinsics + timing) is consistent with the images, re-detected physical points coincide within a few mm at
any t. Usage: gt_consistency.py EPLIST OUTCSV"""
import sys, os, csv, numpy as np, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra, rig_match_track as rmt
eps = [l.strip() for l in open(sys.argv[1]) if l.strip()]; rows = []
ev = {r["episode"]: (int(r["entry_frame"]), int(r["exit_frame"])) for r in csv.DictReader(open(ra.EVENTS)) if r["event"] == "0"}
for ep in eps:
    sf = f"{ra.SEEDS}/{ep}/matches.npz"
    if ep not in ev or not os.path.isfile(sf): continue
    b, e = ev[ep]; cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS); zs = np.load(sf)
    Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], zs["x1"].T, zs["x2"].T); XB = (Xh[:3] / Xh[3]).T
    T = min(e, rt.count_files(f"{rt.STORE_ROOT}/{ep}/dense/cam", ".npz")); gt = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", T, "gt")
    gray = {c: rt.read_frames(f"{rt.RAW_ROOT}/{ep}/recordings/MP4/{s}.mp4", T) for c, s in cams}; T = min(T, min(len(g) for g in gray.values()))
    R1, t1, R2, t2 = E["ext1"][:3, :3], E["ext1"][:3, 3], E["ext2"][:3, :3], E["ext2"][:3, 3]; Rr = R2 @ R1.T; trel = t2 - Rr @ t1
    F = np.linalg.inv(K["ext2"]).T @ rt.skew(trel) @ Rr @ np.linalg.inv(K["ext1"])
    qz = {c: np.load(f"{ra.V2}/{ep}/{c}_f{b:05d}.npz") for c in ("ext1", "ext2")}
    YB = (XB - gt[b][:3, 3]) @ gt[b][:3, :3]                                         # inv(gt[b]) X
    for dt in (0, 10, 20, 40, 80, 120):
        t = b + dt
        if t >= T or t not in gt: break
        m = {}
        for c in ("ext1", "ext2"):
            p = qz[c]["tracks"][t] if t < len(qz[c]["tracks"]) else None
            ok = qz[c]["visibility"][t] & ~np.isnan(p[:, 0]) if p is not None else None
            if p is None or ok.sum() < 3: m = None; break
            m[c] = rmt.region_mask(p[ok], 8)
        if m is None: continue
        x1, x2, conf = rmt.match_disk_epi(gray["ext1"][t], gray["ext2"][t], m["ext1"], m["ext2"], F, 640, 2.5)
        sel, res = rmt.gate(x1, x2, conf, m["ext1"], m["ext2"], F, P, E, 2.5, 0.3)
        if len(sel) < 3: continue
        Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], x1[sel].T, x2[sel].T); Xt = (Xh[:3] / Xh[3]).T
        Yt = (Xt - gt[t][:3, 3]) @ gt[t][:3, :3]
        dd = np.linalg.norm(Yt[:, None] - YB[None], axis=2).min(1) * 1000
        rot = rt.rot_angle_deg(gt[t][:3, :3] @ gt[b][:3, :3].T)
        for v in dd: rows.append((ep, dt, rot, v))
d = pd.DataFrame(rows, columns=["ep", "dt", "rotb", "nn_mm"]); d.to_csv(sys.argv[2], index=False)
d["rb"] = pd.cut(d.rotb, [-1, 1, 5, 10, 20, 40, 180])
print(d.groupby("rb", observed=True).nn_mm.describe(percentiles=[.1, .25, .5]).round(2).to_string())
