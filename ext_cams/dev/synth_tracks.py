"""DEV (GT): synthetic observation tracks = GT projection of the triangulated birth seeds + Gaussian pixel noise, in the
Track-On layout (first N columns = seeds). Usage: synth_tracks.py EPLIST OUTDIR SIGMA_PX"""
import sys, os, csv, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra
eps = [l.strip() for l in open(sys.argv[1]) if l.strip()]; out = sys.argv[2]; sig = float(sys.argv[3]); rng = np.random.default_rng(0)
ev = {r["episode"]: (int(r["entry_frame"]), int(r["exit_frame"])) for r in csv.DictReader(open(ra.EVENTS)) if r["event"] == "0"}
for ep in eps:
    sf = f"{ra.SEEDS}/{ep}/matches.npz"
    if ep not in ev or not os.path.isfile(sf): continue
    b, e = ev[ep]; cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS); zs = np.load(sf); x = {"ext1": zs["x1"], "ext2": zs["x2"]}
    Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], x["ext1"].T, x["ext2"].T); X = (Xh[:3] / Xh[3]).T
    T = min(e, rt.count_files(f"{rt.STORE_ROOT}/{ep}/dense/cam", ".npz")); gt = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", T, "gt")
    an = ra.Anchor(type("A", (), dict(tau_in=4, kabsch_m=0.01, huber_px=2))(), K, E, X); os.makedirs(f"{out}/{ep}", exist_ok=True)
    for ci, c in enumerate(("ext1", "ext2")):
        tr = np.full((T, len(X), 2), np.nan, np.float32); vi = np.zeros((T, len(X)), bool)
        for t in range(b, T):
            if t not in gt: break
            M = gt[t] @ np.linalg.inv(gt[b]); uv, z = an.project(ci, M[:3, :3], M[:3, 3])
            uv = uv + rng.normal(0, sig, uv.shape) * (t > b); tr[t] = uv; vi[t] = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < rt.W) & (uv[:, 1] >= 0) & (uv[:, 1] < rt.H)
        tr[b] = x[c]
        np.savez_compressed(f"{out}/{ep}/{c}_f{b:05d}.npz", tracks=tr, visibility=vi, queries=np.c_[np.full(len(X), b), x[c]].astype(np.float32))
