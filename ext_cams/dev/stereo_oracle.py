#!/usr/bin/env python
"""DEV DIAGNOSTIC (GT): information content of the per-frame stereo clouds (stereo_clouds.py).
Per scene: (1) validity = fraction of cloud points within 15 mm of the FK label cloud at t (positions_v1), by frames since
birth; (2) ICP-only pose: register S_t (point-to-plane) against the BIRTH cloud S_b (model), started at the GT motion
perturbed by +3 deg about a random axis and +1 cm; report the converged rotation/position error vs GT by rotation-from-birth
bin. Pass criterion: converged error < 1 deg / 5 mm on most frames with >= 50 associated points.
Usage: stereo_oracle.py CLOUDDIR EPLIST [--n 30] [--sim 0.5] [--every 3]"""
import argparse, csv, os, sys, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra

ap = argparse.ArgumentParser(); ap.add_argument("clouds"); ap.add_argument("eps"); ap.add_argument("--n", type=int, default=30); ap.add_argument("--sim", type=float, default=0.5)
ap.add_argument("--every", type=int, default=3); ap.add_argument("--assoc", type=float, default=12.0); ap.add_argument("--out", default=None); ap.add_argument("--no_normals", action="store_true"); ap.add_argument("--f1", type=float, default=0.0, help="apply the GT-free F1 filter: s*m3d <= this and pconf >= 2"); ap.add_argument("--src", default="all"); ap.add_argument("--model_vox", type=float, default=0.0)
a = ap.parse_args(); eps = [l.strip() for l in open(a.eps) if l.strip()][:a.n]
ev = {r["episode"]: int(r["entry_frame"]) for r in csv.DictReader(open(ra.EVENTS)) if r["event"] == "0"}
rng = np.random.default_rng(0); rows = []; vrows = []

def label_world(z, s, t):
    uv = z[f"{s}_uv"][t].astype(np.float64); d = z[f"{s}_z"][t].astype(np.float64); K = z[f"{s}_K"]; W2C = z[f"{s}_W2C"]; ok = z[f"{s}_front"][t]
    Xc = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0] * d, (uv[:, 1] - K[1, 2]) / K[1, 1] * d, d]
    return ((Xc - W2C[:3, 3]) @ W2C[:3, :3])[ok]

for ep in eps:
    f = f"{a.clouds}/{ep}.npz"
    if not os.path.isfile(f) or ep not in ev: continue
    z = np.load(f); ok = z["sim"] >= a.sim; b = ev[ep]
    if a.f1 > 0 and "m3d" in z.files:
        import json as _j; mt = _j.loads(str(z["meta"])); sc = {p_["t"]: p_.get("scale", np.nan) for p_ in mt["per_frame"]}
        s_pt = np.array([sc.get(int(t_), np.nan) for t_ in z["frame"]]); ok &= (z["m3d"] * s_pt <= a.f1) & (z["pconf"].min(1) >= 2)
    if a.src == "recip" and "src" in z.files: ok &= np.isin(z["src"], [1, 3])
    fr = z["frame"][ok]; X = z["X"][ok].astype(np.float64)
    if not (fr == b).any(): continue
    T = int(fr.max()) + 1; gt = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", T, "gt")
    lf = f"{rte.POSITIONS}/{ep}_gripper.npz"; zl = np.load(lf) if os.path.isfile(lf) else None; s0 = str(zl["serials"][0]) if zl is not None else None
    Sb = X[fr == b]; cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS)
    if a.model_vox > 0: Sb = Sb[ra.voxel_keep(Sb, a.model_vox * 1e-3, 100000)]
    an = ra.Anchor(type("A", (), dict(tau_in=4, kabsch_m=0.01, huber_px=2))(), K, E, Sb)
    if a.no_normals: an.normals = None
    for t in range(b, T, a.every):
        S = X[fr == t]
        if len(S) < 6 or t not in gt: continue
        Mg = gt[t] @ np.linalg.inv(gt[b]); Rg, tg = Mg[:3, :3], Mg[:3, 3]; rotb = rt.rot_angle_deg(Rg)
        if zl is not None:
            lab = label_world(zl, s0, t); nl = np.linalg.norm(S[:, None] - lab[None], axis=2).min(1) * 1000
            vrows.append(dict(ep=ep, dt=t - b, rotb=rotb, n=len(S), frac15=float((nl < 15).mean()), frac30=float((nl < 30).mean()), med_mm=float(np.median(nl))))
        if t == b: continue
        # perturbed start: +3 deg random axis, +1 cm random direction
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax); dR = Rot.from_rotvec(np.radians(3.0) * ax).as_matrix()
        R0 = ra.ortho(dR @ Rg); cen = Sb.mean(0); t0 = (Rg @ cen + tg + rng.normal(size=3) / np.linalg.norm(rng.normal(size=3)) * 0.01) - R0 @ cen
        n0 = an.stereo_stats(R0, t0, S, a.assoc * 1e-3)[0]
        empty = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2)))
        R1, t1 = an.refine(R0, t0, empty, None, (S, 1.0, a.assoc * 1e-3, 4))
        n1, rms1, med1 = an.stereo_stats(R1, t1, S, a.assoc * 1e-3)
        rows.append(dict(ep=ep, dt=t - b, rotb=rotb, n=len(S), n_assoc0=n0, n_assoc=n1, rms_mm=rms1,
                         rot_err0=rt.rot_angle_deg(R0.T @ Rg), rot_err=rt.rot_angle_deg(R1.T @ Rg),
                         pos_err_cm=np.linalg.norm((R1 @ cen + t1) - (Rg @ cen + tg)) * 100))
d = pd.DataFrame(rows); v = pd.DataFrame(vrows)
if a.out: d.to_csv(a.out, index=False)
if len(v):
    v["rb"] = pd.cut(v.rotb, [-1, 1, 5, 10, 20, 45, 180]); print("cloud validity vs FK labels (fraction of points within 15 / 30 mm of a label point), by rotation from birth")
    print(v.groupby("rb", observed=True).agg(n_frames=("n", "size"), pts=("n", "median"), frac15=("frac15", "median"), frac30=("frac30", "median"), med_mm=("med_mm", "median")).round(2).to_string())
if len(d):
    d["rb"] = pd.cut(d.rotb, [-1, 1, 5, 10, 20, 45, 180]); print("\nICP-only registration of S_t to the birth cloud from GT+3deg/+1cm, by rotation from birth")
    print(d.groupby("rb", observed=True).agg(n_frames=("n", "size"), pts=("n", "median"), assoc=("n_assoc", "median"), rms_mm=("rms_mm", "median"), rot_err_med=("rot_err", "median"),
          rot_err_p75=("rot_err", lambda s: s.quantile(.75)), pos_err_cm=("pos_err_cm", "median"), conv_lt1deg=("rot_err", lambda s: (s < 1).mean())).round(2).to_string())
