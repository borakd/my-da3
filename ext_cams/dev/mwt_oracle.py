"""DEV (GT): masked model-warped birth templates with the GT motion as the prediction. Per seed/view/frame: accepted?,
NCC, error of the match vs the GT projection; binned by rotation from birth. Usage: mwt_oracle.py EPLIST OUTCSV [ncc] [r]"""
import sys, os, csv, numpy as np, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra
eps = [l.strip() for l in open(sys.argv[1]) if l.strip()]; outf = sys.argv[2]
NCC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7; RR = int(sys.argv[4]) if len(sys.argv) > 4 else 12
ev = {r["episode"]: (int(r["entry_frame"]), int(r["exit_frame"])) for r in csv.DictReader(open(ra.EVENTS)) if r["event"] == "0"}
it = pd.read_csv(f"{ra.MASKS}/items_v2.csv"); rows = []
for ep in eps:
    sf = f"{ra.SEEDS}/{ep}/matches.npz"
    if ep not in ev or not os.path.isfile(sf): continue
    b, e = ev[ep]; cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS); zs = np.load(sf); x = {"ext1": zs["x1"], "ext2": zs["x2"]}
    Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], x["ext1"].T, x["ext2"].T); X = (Xh[:3] / Xh[3]).T
    T = min(e, rt.count_files(f"{rt.STORE_ROOT}/{ep}/dense/cam", ".npz")); gt = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", T, "gt")
    gray = {c: rt.read_frames(f"{rt.RAW_ROOT}/{ep}/recordings/MP4/{s}.mp4", T) for c, s in cams}; T = min(T, min(len(g) for g in gray.values()))
    an = ra.Anchor(type("A", (), dict(tau_in=4, kabsch_m=0.01, huber_px=2))(), K, E, X)
    Xc_ = X.mean(0)
    for ci, c in enumerate(("ext1", "ext2")):
        r_ = it[(it.episode == ep) & (it.frame == b) & (it.cam == c)]
        if not len(r_): continue
        Mb = cv2.imread(f"{ra.MASKS}/{r_.mask_relpath.iloc[0]}", 0)
        if Mb is None: continue
        Mb = (cv2.resize(Mb, (rt.W, rt.H), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32); Gb = gray[c][b].astype(np.float32)
        dep = float((E[c][:3, :3] @ Xc_ + E[c][:3, 3])[2]); sz = min(1.0, K[c][0, 0] / (max(dep, 0.05) * 100) / rt.PPCM_RAIL); h_ = max(5, (int(max(7, 31 * sz)) | 1) // 2)
        Kc_ = an.Kc[ci]; Ki_ = np.linalg.inv(Kc_)
        for t in range(b + 1, T, 3):
            M = gt[t] @ np.linalg.inv(gt[b]); R, tt = M[:3, :3], M[:3, 3]; rotb = rt.rot_angle_deg(R)
            A_ = an.Rc[ci] @ R @ an.Rc[ci].T; a_ = an.Rc[ci] @ tt + an.tc[ci] - A_ @ an.tc[ci]
            ugt, _ = an.project(ci, R, tt)
            for k in range(len(X)):
                Xc = an.Rc[ci] @ X[k] + an.tc[ci]; d_ = np.linalg.norm(Xc); n_ = Xc / d_
                Hk = Kc_ @ (A_ + np.outer(a_, n_) / d_) @ Ki_; v0 = Hk @ np.r_[x[c][k], 1.0]; v0 = v0[:2] / v0[2]
                vx, vy = int(round(v0[0])), int(round(v0[1]))
                if vx - h_ - RR < 0 or vy - h_ - RR < 0 or vx + h_ + RR >= rt.W or vy + h_ + RR >= rt.H: continue
                gy_, gx_ = np.mgrid[vy - h_:vy + h_ + 1, vx - h_:vx + h_ + 1].astype(np.float64)
                q_ = np.linalg.inv(Hk) @ np.stack([gx_.ravel(), gy_.ravel(), np.ones(gx_.size)])
                mx_ = (q_[0] / q_[2]).reshape(gx_.shape).astype(np.float32); my_ = (q_[1] / q_[2]).reshape(gx_.shape).astype(np.float32)
                Tm = cv2.remap(Gb, mx_, my_, cv2.INTER_LINEAR); Mm = (cv2.remap(Mb, mx_, my_, cv2.INTER_LINEAR) > 0.5).astype(np.float32)
                if Mm.mean() < 0.25 or Tm[Mm > 0].std() < 4: rows.append((ep, c, k, t - b, rotb, np.nan, np.nan, 0)); continue
                reg = gray[c][t][vy - h_ - RR:vy + h_ + RR + 1, vx - h_ - RR:vx + h_ + RR + 1].astype(np.float32)
                ncc = np.nan_to_num(cv2.matchTemplate(reg, Tm, cv2.TM_CCOEFF_NORMED, mask=Mm), nan=-1, posinf=-1, neginf=-1)
                py, px = np.unravel_index(int(ncc.argmax()), ncc.shape); pk = float(ncc[py, px])
                u = np.array([vx + px - RR, vy + py - RR]) + (v0 - [vx, vy])
                rows.append((ep, c, k, t - b, rotb, pk, float(np.linalg.norm(u - ugt[k])), 1))
d = pd.DataFrame(rows, columns=["ep", "cam", "k", "dt", "rotb", "ncc", "err", "tmpl"]); d.to_csv(outf, index=False)
d["rb"] = pd.cut(d.rotb, [0, 5, 10, 20, 30, 45, 60, 180])
g = d.groupby("rb", observed=True).apply(lambda s: pd.Series(dict(n=len(s), tmpl_ok=s.tmpl.mean(), acc=(s.ncc >= NCC).mean(),
     err_acc_med=s[s.ncc >= NCC].err.median(), acc_within2=(s[s.ncc >= NCC].err < 2).mean(), err_all_med=s.err.median())))
print(g.round(3).to_string())
