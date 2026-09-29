#!/usr/bin/env python
"""Fuse regular CUT3R (wrist stream only, existing augfull_lr1e5 predictions) with the exterior-camera rig anchors from
rig_kabsch_probe.py, causally, on one episode. Three rows are scored against the kinematic ground truth:

    cut3r      regular CUT3R, all frames
    rig        the rig pipeline alone, only the frames where the gripper was solved from both exterior cameras
    fused      this method, all frames

The fusion runs in CUT3R's own (non-metric) frame so that no past output is ever revised:
  * predict:  P_f(t) = P_f(t-1) * (P_c(t-1)^-1 P_c(t))          CUT3R's own relative motion, always
  * anchor:   when the rig solved frame t, its metric base-frame pose is mapped into CUT3R's frame by a Sim3 (scale, rotation,
              offset) estimated ONCE from the first anchored frames (rotation from the paired orientations, scale from the ratio
              of travelled distance, offset from the positions), then frozen; the predicted pose is pulled toward the mapped
              anchor with a scalar Kalman gain whose two variances are measured online: the rig's from its rigid-fit residual
              scaled by the lens lever arm, CUT3R's from the running innovation. No constant is tuned on ground truth.
  * no anchor: the prediction stands (CUT3R dead reckoning from the last anchored pose).
Ground truth enters only through the rig's reference pose (already inside the anchors, once per episode) and the scoring.

    OMP_NUM_THREADS=1 python rig_fuse_probe.py
"""
import json, os, sys
import numpy as np, cv2

EP = "RAIL+80edfcb1+2023-07-14-14h-28m-45s"
OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
RIG = f"{OUT_ROOT}/ext_cams/rig_kabsch/{EP}"
MASKS = f"{OUT_ROOT}/ext_cams/gripper_sam3/{EP}"
RAW = f"/gpfs/scratch/etur59/koc821022/vggt_cache/raw/{EP}/recordings/MP4"
STORE = f"/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist/{EP}/dense"
BACKBONES = {"finetuned": ("augfull_lr1e5", "CUT3R finetuned"), "zeroshot": ("cut3r_zeroshot", "CUT3R zero-shot")}
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--backbone", choices=list(BACKBONES), default="finetuned")
ARGS = _ap.parse_args(); LABEL, NAME = BACKBONES[ARGS.backbone]
CUT3R = f"{OUT_ROOT}/{LABEL}/preds/{EP}/camera"
TAG = "" if ARGS.backbone == "finetuned" else "_zeroshot"
CAMS = [("ext1", "20521388"), ("ext2", "24259877")]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam3_gripper_masks import FfmpegWriter  # noqa: E402
from rig_kabsch_probe import umeyama, rot_angle_deg, traj_metrics, load_masks  # noqa: E402


def rotvec(R):
    return cv2.Rodrigues(R)[0].ravel()


def rotmat(v):
    return cv2.Rodrigues(v.reshape(3, 1))[0]


def mean_rotation(Rs):
    U, _, Vt = np.linalg.svd(sum(Rs))
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt


def fuse(Pc, anchors, sigma_rig_m, n_init_travel_m=0.05):
    """Pc: dict t -> CUT3R c2w (its frame). anchors: dict t -> rig c2w (metric base frame). sigma_rig_m: dict t -> anchor
    position noise (m) from the rig's own residuals. Returns fused poses (CUT3R frame) for every t, plus a log."""
    T = max(Pc) + 1
    Pf = {0: Pc[0].copy()}
    Tinit = None; buf = []
    Pvar = 0.0; q = None; innov = []   # scalar position variance (CUT3R units^2), process noise, innovation history
    log = {}
    for t in range(1, T):
        D = np.linalg.inv(Pc[t - 1]) @ Pc[t]
        Pm = Pf[t - 1] @ D
        if q is not None:
            Pvar += q
        entry = dict(anchored=False, K=0.0)
        if t in anchors:
            if Tinit is None:
                buf.append((t, anchors[t], Pc[t]))
                travel = sum(np.linalg.norm(buf[i][1][:3, 3] - buf[i - 1][1][:3, 3]) for i in range(1, len(buf)))
                if len(buf) >= 5 and travel >= n_init_travel_m:
                    Q = mean_rotation([pc[:3, :3] @ pr[:3, :3].T for _, pr, pc in buf])
                    dc = sum(np.linalg.norm(buf[i][2][:3, 3] - buf[i - 1][2][:3, 3]) for i in range(1, len(buf)))
                    s = dc / travel
                    o = np.mean([pc[:3, 3] - s * Q @ pr[:3, 3] for _, pr, pc in buf], 0)
                    Tinit = (s, Q, o)
                    # initial uncertainty of the fused position: CUT3R has run unanchored for t frames
                    Pvar = 0.0; q = 0.0; innov = []
                    entry["sim3_init"] = dict(scale=float(s), rot_deg=float(rot_angle_deg(Q)), n=len(buf))
            if Tinit is not None:
                s, Q, o = Tinit
                pA = s * Q @ anchors[t][:3, 3] + o
                RA = Q @ anchors[t][:3, :3]
                e = pA - Pm[:3, 3]
                Rm = (s * sigma_rig_m[t]) ** 2
                innov.append(float(e @ e / 3))
                # process noise from the innovation sequence (what the prediction misses per step beyond the anchor noise)
                q = max(np.mean(innov[-10:]) - Rm, 1e-12)
                Pvar = max(Pvar, q)
                K = Pvar / (Pvar + Rm)
                Pn = np.eye(4)
                Pn[:3, 3] = Pm[:3, 3] + K * e
                Pn[:3, :3] = Pm[:3, :3] @ rotmat(K * rotvec(Pm[:3, :3].T @ RA))
                Pvar = (1 - K) * Pvar
                Pm = Pn
                entry.update(anchored=True, K=float(K), innovation=float(np.linalg.norm(e)))
        Pf[t] = Pm
        log[t] = entry
    return Pf, log, Tinit


def main():
    d = np.load(f"{RIG}/tracks_res1280.npz")
    T = 128
    t0, t1 = int(d["t0"]), int(d["t1"])
    gt = {t: np.load(f"{STORE}/cam/{t:06d}.npz")["pose"].astype(np.float64) for t in range(T)}
    Pc = {t: np.load(f"{CUT3R}/{t:06d}.npz")["pose"].astype(np.float64) for t in range(T)}
    anchors = {int(f): p for f, p in zip(d["pred_frames"], d["pred"])}
    lever = float(np.linalg.norm(d["lens_ref"] - d["model_centroid"])); extent = float(d["model_extent"])
    resid = {t0 + i: r for i, r in enumerate(d["rigid_resid_mm"])}
    sigma = {t: max(resid[t] * 1e-3, 1e-4) * lever / extent for t in anchors}  # point residual x lever arm -> lens noise
    Pf, log, Tinit = fuse(Pc, anchors, sigma)
    n_anch = sum(1 for t in log if log[t]["anchored"])
    t_first = min(t for t in log if log[t]["anchored"])
    Ks = [log[t]["K"] for t in log if log[t]["anchored"]]
    print(f"anchors available {len(anchors)} frames ({t0}..{t1} window); Sim3 rig->CUT3R frozen at frame {t_first}: "
          f"scale {Tinit[0]:.3f}, rotation {rot_angle_deg(Tinit[1]):.1f} deg; anchored updates {n_anch}, gain median {np.median(Ks):.2f} min {np.min(Ks):.2f}")

    allf = list(range(T)); anc = sorted(anchors)
    rows = {
        "cut3r_all_frames": traj_metrics([Pc[t] for t in allf], [gt[t] for t in allf]),
        "fused_all_frames": traj_metrics([Pf[t] for t in allf], [gt[t] for t in allf]),
        "rig_anchored_frames_only": traj_metrics([anchors[t] for t in anc], [gt[t] for t in anc]),
        "cut3r_anchored_frames_only": traj_metrics([Pc[t] for t in anc], [gt[t] for t in anc]),
        "fused_anchored_frames_only": traj_metrics([Pf[t] for t in anc], [gt[t] for t in anc]),
    }
    print(f"{'row':30s} {'n':>4s} {'ATE':>8s} {'RPE_t':>8s} {'RPE_rot':>8s} {'scale':>7s}")
    for k, v in rows.items():
        print(f"{k:30s} {v['n']:4d} {v['ate']:8.4f} {v['rpe_trans']:8.4f} {v['rpe_rot']:8.3f} {v['sim3_scale']:7.3f}")
    out = dict(episode=EP, window=[t0, t1], n_anchors=len(anchors), first_anchored_update=t_first,
               sim3_rig_to_cut3r=dict(scale=float(Tinit[0]), rot_deg=float(rot_angle_deg(Tinit[1]))),
               gain=dict(median=float(np.median(Ks)), min=float(np.min(Ks))), rows=rows,
               per_frame=[dict(frame=t, **log[t]) for t in log])
    json.dump(out, open(f"{RIG}/fuse_summary{TAG}.json", "w"), indent=1)
    np.savez(f"{RIG}/fused_poses{TAG}.npz", fused=np.array([Pf[t] for t in allf]), cut3r=np.array([Pc[t] for t in allf]),
             gt=np.array([gt[t] for t in allf]), anchor_frames=np.array(anc), anchors=np.array([anchors[t] for t in anc]))

    # ------------------------------------------------------------- video: wrist + inset | ext1 | ext2
    def align_all(P):
        s, R, tt = umeyama(np.array([P[t][:3, 3] for t in allf]), np.array([gt[t][:3, 3] for t in allf]))
        return {t: s * R @ P[t][:3, 3] + tt for t in allf}
    cut_c, fus_c = align_all(Pc), align_all(Pf)      # display only: Sim3-aligned over ALL frames
    m = rows
    M = {cam: load_masks(cam, s_, 1280, 720) for cam, s_ in CAMS}
    caps = {cam: cv2.VideoCapture(f"{RAW}/{s_}.mp4") for cam, s_ in CAMS}
    Pm = {"ext1": d["P_ext1"], "ext2": d["P_ext2"]}
    tr = {"ext1": d["tr1"], "ext2": d["tr2"]}
    pair1 = {int(i) for i, _ in d["pairs"]}; pair2 = {int(j) for _, j in d["pairs"]}
    PW, PH = 640, 360
    G = np.array([gt[t][:3, 3] for t in allf]); lo, hi = G[:, :2].min(0) - 0.05, G[:, :2].max(0) + 0.05
    span = (hi - lo).max(); cen = (lo + hi) / 2; lo = cen - span / 2
    IW, IH, IX, IY = 230, 230, PW - 240, PH - 240
    def to_inset(xy):
        return int(IX + (xy[0] - lo[0]) / span * (IW - 1)), int(IY + (IH - 1) - (xy[1] - lo[1]) / span * (IH - 1))
    writer = FfmpegWriter(f"{RIG}/all_cams_cut3r_rig_fused_side_by_side{TAG}.mp4", 3 * PW, PH, 15)
    GREEN, BLUE, RED, MAG = (0, 220, 0), (255, 120, 0), (0, 0, 255), (255, 0, 255)
    for t in allf:
        anchored = log.get(t, {}).get("anchored", False)
        wimg = cv2.resize(cv2.imread(f"{STORE}/rgb/{t:06d}.png"), (PW, PH), interpolation=cv2.INTER_CUBIC)
        ov = wimg.copy(); cv2.rectangle(ov, (IX - 6, IY - 6), (IX + IW + 6, IY + IH + 6), (30, 30, 30), -1)
        wimg = cv2.addWeighted(ov, 0.75, wimg, 0.25, 0)
        cv2.putText(wimg, "top view (base x,y)", (IX + 4, IY + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        for tt in range(1, t + 1):
            cv2.line(wimg, to_inset(G[tt - 1]), to_inset(G[tt]), GREEN, 2)
            cv2.line(wimg, to_inset(cut_c[tt - 1]), to_inset(cut_c[tt]), BLUE, 1)
            cv2.line(wimg, to_inset(fus_c[tt - 1]), to_inset(fus_c[tt]), MAG, 2)
        for tt in anc:
            if tt <= t:
                cv2.circle(wimg, to_inset(anchors[tt][:3, 3]), 2, RED, -1)
        for c, col in ((G[t], GREEN), (cut_c[t], BLUE), (fus_c[t], MAG)):
            cv2.circle(wimg, to_inset(c), 4, col, -1)
        lines = [(f"wrist f{t}   {NAME} (wrist only) + exterior rig anchors, causal fusion", (255, 255, 255)),
                 (f"green GT  blue {NAME}  magenta fused  red dots rig anchors", (255, 255, 255)),
                 (f"all 128 f:  ATE  {NAME.split()[-1]} {m['cut3r_all_frames']['ate']:.4f}  fused {m['fused_all_frames']['ate']:.4f}", (255, 255, 255)),
                 (f"            RPE_t {m['cut3r_all_frames']['rpe_trans']:.4f} / {m['fused_all_frames']['rpe_trans']:.4f}   RPE_rot {m['cut3r_all_frames']['rpe_rot']:.2f} / {m['fused_all_frames']['rpe_rot']:.2f}", (255, 255, 255)),
                 (f"rig alone, {len(anc)} anchored f: ATE {m['rig_anchored_frames_only']['ate']:.4f}  RPE_t {m['rig_anchored_frames_only']['rpe_trans']:.4f}  RPE_rot {m['rig_anchored_frames_only']['rpe_rot']:.2f}", (255, 255, 255)),
                 (f"this frame err vs GT: CUT3R {np.linalg.norm(cut_c[t] - G[t]) * 100:.1f} cm   fused {np.linalg.norm(fus_c[t] - G[t]) * 100:.1f} cm" +
                  (f"   gain {log[t]['K']:.2f}" if anchored else ""), (0, 255, 255))]
        for i, (txt, col) in enumerate(lines):
            cv2.putText(wimg, txt, (8, 22 + 19 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1)
        panels = [wimg]
        for cam, _ in CAMS:
            ok, img = caps[cam].read()
            cnts, _ = cv2.findContours(M[cam][t], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 200, 0), 2)
            pset = pair1 if cam == "ext1" else pair2
            for n in range(len(tr[cam])):
                if not np.isnan(tr[cam][n, t, 0]):
                    x, y = tr[cam][n, t]
                    cv2.circle(img, (int(x), int(y)), 5, (255, 200, 0) if n in pset else (140, 140, 140), -1)
            def proj(pt):
                q = Pm[cam] @ np.r_[pt, 1]; return int(q[0] / q[2]), int(q[1] / q[2])
            cv2.drawMarker(img, proj(G[t]), GREEN, cv2.MARKER_CROSS, 30, 3)
            c = np.array(proj(cut_c[t])); cv2.rectangle(img, tuple(c - 9), tuple(c + 9), BLUE, 3)
            cv2.drawMarker(img, proj(fus_c[t]), MAG, cv2.MARKER_DIAMOND, 26, 3)
            if t in anchors:
                cv2.circle(img, proj(anchors[t][:3, 3]), 12, RED, 3)
            img = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_AREA)
            cv2.putText(img, f"{cam} f{t}   + GT   [] CUT3R   <> fused   o rig   (CUT3R/fused aligned for display)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            panels.append(img)
        if not anchored:
            txt = "NO RIG ANCHOR: CUT3R DEAD RECKONING" if Tinit is not None and t >= t_first else "NO RIG ANCHOR YET: CUT3R ONLY"
            for pnl in panels:
                cv2.rectangle(pnl, (0, PH // 2 - 20), (PW, PH // 2 + 20), (0, 140, 180), -1)
                cv2.putText(pnl, txt, (PW // 2 - 7 * len(txt), PH // 2 + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        writer.write(np.concatenate(panels, axis=1))
    writer.close()
    print("wrote", f"{RIG}/all_cams_cut3r_rig_fused_side_by_side{TAG}.mp4")


if __name__ == "__main__":
    main()
