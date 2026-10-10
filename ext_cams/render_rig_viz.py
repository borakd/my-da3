#!/usr/bin/env python
"""Video of the exterior-camera rig evaluation on one scene: both exterior cameras side by side with the labelled gripper
points and the wrist-camera position of every method projected into them, plus top-down and side trajectory plots.

Methods (wrist-camera c2w, robot base frame): GT (store), exterior rig on the labelled points (rig_points_eval.py anchors),
CUT3R finetuned (augfull_lr1e5), CUT3R zero-shot (cut3r_zeroshot). Every method is Sim3-aligned to GT over the frames shown,
as in the evaluation (rig_track.traj_metrics). The video covers the rig's windows: it starts at the first frame where the
gripper is in view in both exterior cameras.

    python render_rig_viz.py --episode EP --positions DIR --rig DIR_WITH_anchors.npz --out OUT.mp4
"""
import argparse, os, subprocess, sys
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_track as rt                     # noqa: E402
from rig_points_eval import calib          # noqa: E402

FF = os.environ.get("FFMPEG", "/gpfs/scratch/etur59/koc821022/conda_envs/recal3r/bin/ffmpeg")
PW, PH = 640, 360
COL = {"GT": (255, 255, 255), "rig": (60, 60, 255), "cut3r_ft": (255, 160, 40), "cut3r_zs": (0, 165, 255)}   # BGR
NAME = {"GT": "GT wrist camera", "rig": "exterior rig (label points)", "cut3r_ft": "CUT3R finetuned", "cut3r_zs": "CUT3R zero-shot"}


def sim3_apply(P, s, R, t):
    Q = P.copy(); Q[:, :3, 3] = (s * (R @ P[:, :3, 3].T)).T + t; Q[:, :3, :3] = R @ P[:, :3, :3]; return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True); ap.add_argument("--positions", required=True); ap.add_argument("--rig", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--trail", type=int, default=20)
    a = ap.parse_args(); ep = a.episode
    z = np.load(f"{a.positions}/{ep}_gripper.npz"); cams, K, E, P, _ = calib(ep, z)
    an = np.load(f"{a.rig}/anchors.npz"); fr = an["frames"].astype(int)
    T = int(fr.max()) + 1
    gt = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", T, "GT")
    ft = rt.load_poses(f"{rt.CUT3R_ROOT}/{ep}/camera", T, "CUT3R ft")
    zs = rt.load_poses(f"{rt.OUT_ROOT}/cut3r_zeroshot/preds/{ep}/camera", T, "CUT3R zs")
    G = np.array([gt[t] for t in fr])
    traj = {"GT": G, "rig": an["poses"]}
    for k, d in (("cut3r_ft", ft), ("cut3r_zs", zs)):
        Pm = np.array([d[t] for t in fr]); s, R, t = rt.umeyama(Pm[:, :3, 3], G[:, :3, 3]); traj[k] = sim3_apply(Pm, s, R, t)
    s, R, t = rt.umeyama(traj["rig"][:, :3, 3], G[:, :3, 3]); traj["rig"] = sim3_apply(traj["rig"], s, R, t)
    metr = {k: rt.traj_metrics(list(traj[k]), list(G)) for k in ("rig", "cut3r_ft", "cut3r_zs")}
    err = {k: np.linalg.norm(traj[k][:, :3, 3] - G[:, :3, 3], axis=1) * 100 for k in metr}
    C = {k: v[:, :3, 3] for k, v in traj.items()}
    # plot frames: top-down (x, y) and side (x, z), common scale with a margin
    allc = np.concatenate(list(C.values())); lo, hi = allc.min(0), allc.max(0); mid = (lo + hi) / 2
    spans = {ax: max((hi[ax[0]] - lo[ax[0]]) / (PW - 40), (hi[ax[1]] - lo[ax[1]]) / (PH - 60)) * 1.08 + 1e-6 for ax in ((0, 1), (0, 2))}   # m per pixel
    def to_px(c, ax):
        m = spans[ax]
        return np.stack([PW / 2 + (c[..., ax[0]] - mid[ax[0]]) / m, PH / 2 + 10 - (c[..., ax[1]] - mid[ax[1]]) / m], -1).astype(int)
    sc = np.array([PW / rt.W, PH / rt.H])
    caps = {c: cv2.VideoCapture(f"{rt.RAW_ROOT}/{ep}/recordings/MP4/{s_}.mp4") for c, s_ in cams}
    for c in caps: caps[c].set(cv2.CAP_PROP_POS_FRAMES, int(fr[0]))
    tmp = a.out + ".tmp.mp4"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 15, (2 * PW, 30 + 2 * PH + 26))
    idx = {int(f): i for i, f in enumerate(fr)}
    for t in range(int(fr[0]), T):
        i = idx.get(t)
        panels = []
        for c, s_ in cams:
            ok, img = caps[c].read(); img = cv2.resize(img, (PW, PH)) if ok else np.zeros((PH, PW, 3), np.uint8)
            uv = z[f"{s_}_uv"][t] * 2; inf = z[f"{s_}_in_frame"][t]; front = z[f"{s_}_front"][t]
            for (x, y), a_, f_ in zip(uv, inf, front):
                if a_: cv2.circle(img, (int(x), int(y)), 2, (0, 200, 0) if f_ else (0, 0, 140), -1, cv2.LINE_AA)
            if i is not None:
                for k in ("cut3r_zs", "cut3r_ft", "rig", "GT"):
                    lo_ = max(0, i - a.trail); pts = C[k][lo_:i + 1]
                    pr = (P[c] @ np.c_[pts, np.ones(len(pts))].T); ok_ = pr[2] > 0; pr = (pr[:2] / pr[2]).T * sc
                    pr = pr[ok_].astype(int)
                    if len(pr) > 1: cv2.polylines(img, [pr.reshape(-1, 1, 2)], False, COL[k], 1, cv2.LINE_AA)
                    if len(pr):
                        if k == "GT": cv2.circle(img, tuple(pr[-1]), 9, COL[k], 2, cv2.LINE_AA)
                        else: cv2.circle(img, tuple(pr[-1]), 5, COL[k], -1, cv2.LINE_AA)
            cv2.putText(img, f"{c} {s_}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(img)
        plots = []
        for ax, title in (((0, 1), "top-down (x right, y up), base frame"), ((0, 2), "side (x right, z up), base frame")):
            pl = np.full((PH, PW, 3), 25, np.uint8)
            cv2.putText(pl, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            bar = 0.05 / spans[ax]; cv2.line(pl, (PW - 20 - int(bar), PH - 12), (PW - 20, PH - 12), (200, 200, 200), 2)
            cv2.putText(pl, "5 cm", (PW - 20 - int(bar), PH - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            for k in ("cut3r_zs", "cut3r_ft", "rig", "GT"):
                px = to_px(C[k], ax)
                cv2.polylines(pl, [px.reshape(-1, 1, 2)], False, tuple(int(v * 0.35) for v in COL[k]), 1, cv2.LINE_AA)
                if i is not None:
                    if i > 0: cv2.polylines(pl, [px[:i + 1].reshape(-1, 1, 2)], False, COL[k], 2 if k == "GT" else 1, cv2.LINE_AA)
                    cv2.circle(pl, tuple(px[i]), 6 if k == "GT" else 4, COL[k], 2 if k == "GT" else -1, cv2.LINE_AA)
            plots.append(pl)
        leg = np.zeros((26, 2 * PW, 3), np.uint8); x0 = 8
        for k in ("GT", "rig", "cut3r_ft", "cut3r_zs"):
            txt = NAME[k] if k == "GT" else f"{NAME[k]}: {err[k][i]:.1f} cm now, ATE {metr[k]['ate'] * 100:.2f} cm" if i is not None else NAME[k]
            cv2.circle(leg, (x0 + 6, 13), 5, COL[k], -1); cv2.putText(leg, txt, (x0 + 16, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL[k], 1, cv2.LINE_AA)
            x0 += 16 + cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] + 22
        hdr = np.zeros((30, 2 * PW, 3), np.uint8)
        cv2.putText(hdr, f"{ep}  frame {t}  {'(gripper in both views)' if i is not None else '(not evaluated)'}   points: green = labelled, facing camera; dark red = facing away   all methods Sim3-aligned to GT",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(np.vstack([hdr, np.hstack(panels), np.hstack(plots), leg]))
    vw.release()
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], check=True); os.remove(tmp)
    print("wrote", a.out, {k: round(v["ate"], 5) for k, v in metr.items()})


if __name__ == "__main__":
    main()
