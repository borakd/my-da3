#!/usr/bin/env python
"""rig_hybrid: causal hybrid pose source = exterior-rig TRANSLATION + CUT3R-finetune ROTATION (2026-10-09).

The rig (rig_anchor.py) recovers the gripper's translation well (model-centroid motion) but its per-frame rotation sits at
the measurement floor of this small, specular, black gripper. CUT3R finetuned (augfull_lr1e5, online, frames <= t) predicts a
smoother rotation. Hybrid lens pose at t:
    R(t)   = R_GT(b) R_ft(b)^T R_ft(t)                 (finetune rotation relative to the birth frame, in the base frame)
    motion = (R(t) R_GT(b)^T,  c_rig(t) - R(t) R_GT(b)^T c_b)    (moves the birth model centroid c_b to the rig's centroid)
    pose(t) = motion @ GT(b)
GT is used only at the birth frame (as in every rig arm). Frames the rig did not solve stay unsolved.
    python rig_hybrid.py --rig DIR --out DIR --episodes LIST [--rot_source ft]
Writes the rig_anchor layout (record.json via rig_anchor.evaluate, per_frame.csv, anchors.npz)."""
import argparse, csv, json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra

ap = argparse.ArgumentParser(); ap.add_argument("--rig", required=True); ap.add_argument("--out", required=True); ap.add_argument("--episodes", required=True)
ap.add_argument("--rot_blend", type=float, default=1.0, help="1 = finetune rotation; 0 = rig rotation; in between = geodesic blend")
a = ap.parse_args()
for ep in [l.strip() for l in open(a.episodes) if l.strip()]:
    rf, af = f"{a.rig}/{ep}/record.json", f"{a.rig}/{ep}/anchors.npz"
    if not (os.path.isfile(rf) and os.path.isfile(af)): continue
    rec = json.load(open(rf)); w = rec["windows"][0]
    if w.get("skipped") or "t_ref" not in w: continue
    b, t_end = int(w["t_ref"]), int(w["window"][1])
    z = np.load(af); fr = [int(f) for f in z["frames"]]; Mo = {f: m for f, m in zip(fr, z["motions"])}
    if "model_centroid" in z.files: cb = z["model_centroid"]
    else:
        cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS); zs = np.load(f"{rec['args']['seeds']}/{ep}/matches.npz")
        Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], zs["x1"].T, zs["x2"].T); cb = (Xh[:3] / Xh[3]).T.mean(0)
    store = f"{rt.STORE_ROOT}/{ep}/dense/cam"
    try: ft = rt.load_poses(f"{rt.CUT3R_ROOT}/{ep}/camera", t_end + 1, "CUT3R")
    except rt.Failure: continue
    gtb = rt.load_poses(store, b + 1, "gt")[b]
    from scipy.spatial.transform import Rotation as Rot
    R_s, t_s, solved = {}, {}, {}
    for f in fr:
        if f not in ft: continue
        Rl = gtb[:3, :3] @ ft[b][:3, :3].T @ ft[f][:3, :3]          # finetune lens rotation in the base frame
        Rm_ft = Rl @ gtb[:3, :3].T                                    # world-frame motion rotation
        Rm_rig = Mo[f][:3, :3]
        if a.rot_blend < 1.0:
            dw = Rot.from_matrix(Rm_ft @ Rm_rig.T).as_rotvec(); Rm = Rot.from_rotvec(a.rot_blend * dw).as_matrix() @ Rm_rig
        else:
            Rm = Rm_ft
        c_rig = Mo[f][:3, :3] @ cb + Mo[f][:3, 3]
        R_s[f], t_s[f], solved[f] = Rm, c_rig - Rm @ cb, True
    rows = []
    if os.path.isfile(f"{a.rig}/{ep}/per_frame.csv"):
        rows = [dict(frame=int(r["frame"]), n_inl=int(float(r.get("n_inl", 0) or 0)), solved=r.get("solved")) for r in csv.DictReader(open(f"{a.rig}/{ep}/per_frame.csv"))]
    rows = [r for r in rows if r["frame"] in R_s or r["frame"] > b]
    for r in rows: r["solved"] = r["frame"] in solved
    new = dict(episode=ep, tag=f"hybrid:{os.path.basename(a.rig.rstrip('/'))}", n_seeds=rec.get("n_seeds"), args=vars(a))
    pred = ra.evaluate(new, rows, (R_s, t_s, solved), b, t_end, store)
    os.makedirs(f"{a.out}/{ep}", exist_ok=True)
    fr2 = sorted(pred); np.savez(f"{a.out}/{ep}/anchors.npz", frames=np.array(fr2), poses=np.array([pred[t] for t in fr2]),
                                 motions=np.array([ra.se3(R_s[t], t_s[t]) for t in fr2]), model_centroid=cb)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if rows and k in rows[0] else 99)
    with open(f"{a.out}/{ep}/per_frame.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, keys); wr.writeheader(); wr.writerows(rows)
    json.dump(new, open(f"{a.out}/{ep}/record.json", "w"), indent=1, default=float)
print("done")
