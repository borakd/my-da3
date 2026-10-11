#!/usr/bin/env python
"""DEV DIAGNOSTIC (PointWorld exterior depth = GT, diagnostic only): label every stereo_clouds v2 match as TRUE (triangulated
depth within TOL of the depth map at its pixel in BOTH views, min over a 3x3 neighbourhood at 320x180) or PHANTOM, then report
the fraction of true matches per source and the ROC of GT-FREE filter variables: s*m3d (MASt3R two-view consistency scaled
by the per-frame scale), |z1 - s*zm1| / z1, pointmap conf, desc conf, similarity.
Usage: phantom_oracle.py CLOUDDIR EPLIST [--n 30] [--tol 0.01] [--dt 0,10,30,60] [--out CSV]"""
import argparse, csv, json, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra
DEP = "/gpfs/scratch/etur59/koc821022/pointworld_droid_ext_all/dl3dv_multi"
ap = argparse.ArgumentParser(); ap.add_argument("clouds"); ap.add_argument("eps"); ap.add_argument("--n", type=int, default=30); ap.add_argument("--tol", type=float, default=0.01)
ap.add_argument("--dt", default="0,5,10,20,40,80"); ap.add_argument("--out", default=None); ap.add_argument("--all_frames", action="store_true")
a = ap.parse_args(); eps = [l.strip() for l in open(a.eps) if l.strip()][:a.n]; dts = [int(v) for v in a.dt.split(",")]
ev = {r["episode"]: int(r["entry_frame"]) for r in csv.DictReader(open(ra.EVENTS)) if r["event"] == "0"}
rows = []
for ep in eps:
    f = f"{a.clouds}/{ep}.npz"; stf = f"{DEP}/_status/{ep}.json"
    if not os.path.isfile(f) or not os.path.isfile(stf) or ep not in ev: continue
    z = np.load(f); meta = json.loads(str(z["meta"])); b = meta["birth"]; st = json.load(open(stf)); cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS)
    scale = {p["t"]: p.get("scale", np.nan) for p in meta["per_frame"]}
    fr = z["frame"]; frames = sorted(set(fr.tolist())) if a.all_frames else [b + d for d in dts]
    for t in frames:
        sel = np.where(fr == t)[0]
        if len(sel) < 3: continue
        X = z["X"][sel].astype(np.float64); x = {"ext1": z["x1"][sel], "ext2": z["x2"][sel]}; dd = {}; okd = True
        for c in ("ext1", "ext2"):
            off = int(st.get(f"{c}_offset", 0) or 0); df = f"{DEP}/{c}/{ep}/dense/depth/{t + off:06d}.npy"
            if not os.path.isfile(df): okd = False; break
            D = np.load(df); zc = (X @ E[c][:3, :3].T + E[c][:3, 3])[:, 2]
            u = np.clip(np.round(x[c][:, 0] / 4).astype(int), 0, 319); v = np.clip(np.round(x[c][:, 1] / 4).astype(int), 0, 179)
            best = np.full(len(X), np.inf)
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    d_ = D[np.clip(v + dv, 0, 179), np.clip(u + du, 0, 319)]; e_ = np.where(d_ > 0, np.abs(d_ - zc), np.inf); best = np.minimum(best, e_)
            dd[c] = best
        if not okd: continue
        s = scale.get(t, np.nan); dmax = np.maximum(dd["ext1"], dd["ext2"])
        for i, k in enumerate(sel):
            rows.append(dict(ep=ep, dt=t - b, src=int(z["src"][k]), true=bool(dmax[i] < a.tol), dmax_mm=float(dmax[i] * 1000) if np.isfinite(dmax[i]) else np.nan,
                             m3d_s=float(z["m3d"][k] * s) if s == s else np.nan, zrel=float(abs(z["z1"][k] - s * z["zm1"][k]) / z["z1"][k]) if s == s else np.nan,
                             pconf=float(min(z["pconf"][k])), conf=float(min(z["conf"][k])), sim=float(z["sim"][k]), epi=float(z["epi"][k])))
d = pd.DataFrame(rows)
if a.out: d.to_csv(a.out, index=False)
pd.set_option("display.width", 220)
print(f"{d.ep.nunique()} scenes, {len(d)} matches; TRUE = depth-consistent within {a.tol*1000:.0f} mm in both views")
print("\nfraction true by source (1 recip, 2 guided, 3 both) and frames since birth:")
print(d.groupby(["src", pd.cut(d.dt, [-1, 0, 5, 10, 20, 40, 80, 10000])], observed=True).agg(n=("true", "size"), frac_true=("true", "mean"), n_true_per_frame=("true", lambda s: s.sum() / max(d.loc[s.index].groupby(["ep", "dt"]).ngroups, 1))).round(2).to_string())
print("\nGT-free filters (all sources pooled): keep rate of TRUE and of PHANTOM matches, per threshold")
for var, ths, lower in (("m3d_s", [0.005, 0.01, 0.02, 0.03, 0.05], True), ("zrel", [0.02, 0.05, 0.1, 0.2], True), ("pconf", [1.5, 2, 3, 5, 10], False), ("conf", [1.0, 1.2, 1.5, 2.0], False), ("sim", [0.5, 0.6, 0.7, 0.8], False)):
    v = d[var].values; tr = d.true.values; line = []
    for th in ths:
        k = (v <= th) if lower else (v >= th); k &= np.isfinite(v)
        line.append(f"{'<=' if lower else '>='}{th}: true {k[tr].mean():.2f} phantom {k[~tr].mean():.2f} (precision {tr[k].mean():.2f})")
    print(f"  {var}: " + " | ".join(line))
print("\nCOMBINED: src in (1,3) & m3d_s <= 0.01 -> ", end="")
k = d.src.isin([1, 3]) & (d.m3d_s <= 0.01); print(f"keeps {k[d.true].mean():.2f} of true, {k[~d.true].mean():.2f} of phantoms, precision {d.true[k].mean():.2f}, per frame {k.sum()/max(d.groupby(['ep','dt']).ngroups,1):.1f}")
k = d.src.isin([1, 3]); print("src in (1,3) only -> ", f"precision {d.true[k].mean():.2f}, per frame {k.sum()/max(d.groupby(['ep','dt']).ngroups,1):.1f}")
k = (d.m3d_s <= 0.01) & (d.pconf >= 2); print("m3d_s<=0.01 & pconf>=2 (any src) -> ", f"precision {d.true[k].mean():.2f}, per frame {k.sum()/max(d.groupby(['ep','dt']).ngroups,1):.1f}")
