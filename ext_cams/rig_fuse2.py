#!/usr/bin/env python
"""Honest causal fusion of CUT3R (wrist stream, every frame) with the GT-free exterior-rig anchors of rig_track2.py.

No ground truth anywhere in the method path. Per frame t:
  predict   P_f(t) = P_f(t-1) * (P_c(t-1)^-1 P_c(t))           CUT3R's own relative motion, always (gaps = dead reckoning)
  anchor    when the rig solved frame t (segment k, body-fixed point c_t = R_t c0 + t_t in the metric base frame):
    Sim3    rig -> CUT3R frame (scale s, rotation Q, offset o): Umeyama on (centroid_t, CUT3R camera centre) pairs over the
            first >= 8 anchored frames with >= 5 cm of travel, then FROZEN. (Positions only; the lens-centroid offset is
            absorbed by o up to the body extent.)
    (i)  relative anchor, precise, within the segment:  p_A = p_f(t_ref,k) + s Q (c_t - c_ref,k)
         variance (s sigma_rig)^2 + (s extent theta_t)^2   [theta_t = body rotation since the segment reference: the
         unknown lens-to-centroid lever arm turns rotation into position uncertainty]
    (ii) absolute anchor, coarse, across segments:      p_A = s Q c_t + o
         variance (s extent)^2 + (s sigma_rig)^2           [which body point the segment tracks is unknown within the extent]
    each applied as a scalar Kalman update; the process variance is measured from the running innovations of (i).
    rotation: R_A = Q R_t Q^T R_f(t_ref,k), blended with the relative gain (slerp).
  segment   a new segment's reference pose is the fused PREDICTION at its first anchored frame (no GT); a linked segment
            keeps its inherited reference (the tracker expresses R,t relative to it).
sigma_rig = rigid-fit residual x (extent / extent) ~ the residual itself scaled by 1.25 (lever/extent stand-in without GT).

    python rig_fuse2.py --episode EP --anchors ANCHORS.npz --backbone finetuned|zeroshot --out DIR
    python rig_fuse2.py --batch ROOT --rig_subdir rig2 --list SCENES.txt --backbone finetuned --out DIR   (one row per scene + mean)
"""
import argparse, csv, json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig_kabsch_probe import umeyama, rot_angle_deg, traj_metrics  # noqa: E402

OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
STORE = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
BACKBONES = {"finetuned": "augfull_lr1e5", "zeroshot": "cut3r_zeroshot"}


def rotvec(R): return cv2.Rodrigues(R)[0].ravel()
def rotmat(v): return cv2.Rodrigues(np.asarray(v, float).reshape(3, 1))[0]


def fuse(Pc, A, init_frames=6, init_travel=0.03, extent_floor=0.04, rot_gain='none'):
    """Pc: dict t -> CUT3R c2w. A: anchors npz. Returns fused poses dict, log."""
    T = max(Pc) + 1
    fr = A["frames"].astype(int); seg = A["segment_id"].astype(int); R = A["R"]; C = A["centroid"]
    resid = A["rigid_resid_mm"] * 1e-3; extent = np.maximum(A["model_extent"], extent_floor)
    idx = {int(t): i for i, t in enumerate(fr)}
    Pf = {0: Pc[0].copy()}
    S = None; buf = []
    Pvar, q, innov = 0.0, None, []
    Rvar, qr, innov_r = 0.0, None, []      # rotation filter (rad^2)
    segref = {}     # seg -> dict(pos, R_f, c_ref)
    log = {}
    for t in range(1, T):
        D = np.linalg.inv(Pc[t - 1]) @ Pc[t]
        Pm = Pf[t - 1] @ D
        if q is not None:
            Pvar += q
        if qr is not None:
            Rvar += qr
        e_ = dict(anchored=False, K_rel=0.0, K_abs=0.0, seg=-1)
        if t in idx:
            i = idx[t]; k = int(seg[i])
            if k not in segref:
                segref[k] = dict(pos=Pm[:3, 3].copy(), R_f=Pm[:3, :3].copy(), c_ref=C[i].copy(), t_ref=t)
            if S is None:
                buf.append((C[i].copy(), Pc[t][:3, 3].copy()))
                cen = np.array([b[0] for b in buf])
                travel = np.linalg.norm(cen - cen[0], axis=1).max() if len(buf) > 1 else 0.0
                spread = np.linalg.svd(cen - cen.mean(0), compute_uv=False) if len(buf) >= 3 else np.zeros(3)
                if travel >= init_travel and ((len(buf) >= init_frames and spread[1] > 0.2 * spread[0]) or len(buf) >= 15):
                    s, Q, o = umeyama(cen, np.array([b[1] for b in buf]))
                    S = (s, Q, o); Pvar = 0.0; q = 0.0; innov = []; Rvar = 0.0; qr = 0.0; innov_r = []
                    e_["sim3_init"] = dict(scale=float(s), n=len(buf), travel_m=float(travel))
            if S is not None:
                s, Q, o = S
                sr = segref[k]
                sig = 1.25 * max(resid[i], 5e-4)
                theta = np.linalg.norm(rotvec(R[i]))
                # (i) relative anchor within the segment
                pA = sr["pos"] + s * (Q @ (C[i] - sr["c_ref"]))
                e = pA - Pm[:3, 3]
                Rm = (s * sig) ** 2 + (s * extent[i] * theta) ** 2
                innov.append(float(e @ e / 3))
                q = max(np.mean(innov[-10:]) - (s * sig) ** 2, 1e-12)
                Pvar = max(Pvar, q)
                K = Pvar / (Pvar + Rm)
                pos = Pm[:3, 3] + K * e
                Pvar = (1 - K) * Pvar
                # (ii) absolute anchor across segments (coarse)
                pA2 = s * (Q @ C[i]) + o
                e2 = pA2 - pos
                Rm2 = (s * extent[i]) ** 2 + (s * sig) ** 2
                K2 = Pvar / (Pvar + Rm2)
                pos = pos + K2 * e2
                Pvar = (1 - K2) * Pvar
                # rotation: own scalar Kalman gain; rig rotation noise ~ residual / extent (rad), CUT3R noise from innovations
                RA = Q @ R[i] @ Q.T @ sr["R_f"]
                dv = rotvec(Pm[:3, :3].T @ RA)
                Rm_r = (sig / extent[i]) ** 2
                innov_r.append(float(dv @ dv / 3))
                qr = max(np.mean(innov_r[-10:]) - Rm_r, 1e-10)
                Rvar = max(Rvar, qr)
                Kr = Rvar / (Rvar + Rm_r)
                Rvar = (1 - Kr) * Rvar
                if rot_gain == 'none':
                    Kr = 0.0
                elif rot_gain == 'shared':
                    Kr = K
                Rn = Pm[:3, :3] @ rotmat(Kr * dv)
                Pn = np.eye(4); Pn[:3, :3] = Rn; Pn[:3, 3] = pos
                Pm = Pn
                e_.update(anchored=True, K_rel=float(K), K_abs=float(K2), K_rot=float(Kr), seg=k, innov_rel_cm=float(np.linalg.norm(e) / s * 100),
                          innov_abs_cm=float(np.linalg.norm(e2) / s * 100))
        Pf[t] = Pm
        log[t] = e_
    return Pf, log, S


def run_episode(ep, anchors_path, backbone, rot_gain="none"):
    T = len([f for f in os.listdir(f"{STORE}/{ep}/dense/cam") if f.endswith(".npz")])
    gt = {t: np.load(f"{STORE}/{ep}/dense/cam/{t:06d}.npz")["pose"].astype(np.float64) for t in range(T)}
    Pc = {t: np.load(f"{OUT}/{BACKBONES[backbone]}/preds/{ep}/camera/{t:06d}.npz")["pose"].astype(np.float64) for t in range(T)}
    A = np.load(anchors_path, allow_pickle=True)
    Pf, log, S = fuse(Pc, A, rot_gain=rot_gain)
    allf = list(range(T)); anc = [int(t) for t in A["frames"] if int(t) < T]
    rows = {"cut3r_all_frames": traj_metrics([Pc[t] for t in allf], [gt[t] for t in allf]),
            "fused_all_frames": traj_metrics([Pf[t] for t in allf], [gt[t] for t in allf])}
    if len(anc) >= 3:
        cen = {int(t): c for t, c in zip(A["frames"], A["centroid"])}
        rigP = []
        for t in anc:
            P = np.eye(4); P[:3, 3] = cen[t]; rigP.append(P)
        rows["rig_anchored_frames_only(centroid)"] = traj_metrics(rigP, [gt[t] for t in anc])
        rows["cut3r_anchored_frames_only"] = traj_metrics([Pc[t] for t in anc], [gt[t] for t in anc])
        rows["fused_anchored_frames_only"] = traj_metrics([Pf[t] for t in anc], [gt[t] for t in anc])
    n_upd = sum(1 for t in log if log[t]["anchored"])
    info = dict(episode=ep, backbone=backbone, n_frames=T, n_anchors=len(anc), n_segments=int(len(np.unique(A["segment_id"]))),
                anchored_updates=n_upd, sim3=(dict(scale=float(S[0])) if S else None),
                first_update=(min(t for t in log if log[t]["anchored"]) if n_upd else None), rows=rows)
    return info, Pf, log


def fmt(v): return f"{v['ate']:.4f} / {v['rpe_trans']:.4f} / {v['rpe_rot']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode"); ap.add_argument("--anchors"); ap.add_argument("--batch"); ap.add_argument("--rig_subdir", default="rig2")
    ap.add_argument("--list"); ap.add_argument("--rot_gain", choices=["none","shared","own"], default="none"); ap.add_argument("--backbone", choices=list(BACKBONES), default="finetuned"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.batch:
        eps = [l.strip() for l in open(a.list) if l.strip()]
        infos = []
        for ep in eps:
            p = f"{a.batch}/{ep}/{a.rig_subdir}/anchors.npz"
            if not os.path.exists(p):
                print("no anchors:", ep); continue
            try:
                info, Pf, log = run_episode(ep, p, a.backbone, a.rot_gain)
            except Exception as ex:
                print("FAILED", ep, ex); continue
            infos.append(info)
            print(f"{ep[:38]:38s} T={info['n_frames']:4d} anch={info['n_anchors']:4d} seg={info['n_segments']:2d} | cut3r {fmt(info['rows']['cut3r_all_frames'])} | fused {fmt(info['rows']['fused_all_frames'])}")
        keys = ["cut3r_all_frames", "fused_all_frames", "rig_anchored_frames_only(centroid)", "cut3r_anchored_frames_only", "fused_anchored_frames_only"]
        mean = {}
        for k in keys:
            vals = [i["rows"][k] for i in infos if k in i["rows"]]
            if vals:
                mean[k] = {m: float(np.mean([v[m] for v in vals])) for m in ("ate", "rpe_trans", "rpe_rot")}; mean[k]["n_scenes"] = len(vals)
        json.dump(dict(backbone=a.backbone, scenes=infos, mean=mean), open(f"{a.out}/fuse2_{a.backbone}.json", "w"), indent=1)
        print("\nMEAN over scenes:")
        for k in keys:
            if k in mean: print(f"  {k:38s} n={mean[k]['n_scenes']:2d}  {fmt(mean[k])}")
    else:
        info, Pf, log = run_episode(a.episode, a.anchors, a.backbone, a.rot_gain)
        json.dump(dict(info=info, per_frame=[dict(frame=t, **log[t]) for t in log]), open(f"{a.out}/fuse2_{a.backbone}.json", "w"), indent=1)
        np.savez(f"{a.out}/fused2_{a.backbone}.npz", fused=np.array([Pf[t] for t in range(info["n_frames"])]))
        print(json.dumps({k: v for k, v in info.items() if k != "rows"}))
        for k, v in info["rows"].items(): print(f"  {k:38s} n={v['n']:4d}  ATE {v['ate']:.4f}  RPE_t {v['rpe_trans']:.4f}  RPE_rot {v['rpe_rot']:.3f}  scale {v['sim3_scale']:.3f}")


if __name__ == "__main__":
    main()
