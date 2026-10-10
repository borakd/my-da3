#!/usr/bin/env python
"""Aggregate rig_points_eval.py runs (one dir per episode with record.json + per_frame.csv) into windows.csv + SUMMARY.md.

    python rig_points_summary.py RUN_DIR [EPISODE_LIST]
"""
import glob, json, os, sys
import numpy as np, pandas as pd

run = sys.argv[1]
eps = open(sys.argv[2]).read().split() if len(sys.argv) > 2 else sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{run}/*/record.json"))
W, F, missing = [], [], []
for ep in eps:
    p = f"{run}/{ep}/record.json"
    if not os.path.isfile(p): missing.append(ep); continue
    r = json.load(open(p))
    for w in r["windows"]: W.append(dict(w, episode=ep, lab=ep.split("+")[0], T=r["T"], frames_min_pts_both=r["frames_with_min_pts_both"]))
    pf = f"{run}/{ep}/per_frame.csv"
    if os.path.isfile(pf):
        d = pd.read_csv(pf); d["episode"] = ep; d["lab"] = ep.split("+")[0]; F.append(d)
W = pd.DataFrame(W); F = pd.concat(F) if F else pd.DataFrame()
W["window"] = W["window"].astype(str); W.to_csv(f"{run}/windows.csv", index=False)
ok = F[F.n_inl >= 3] if len(F) else F
ev = W[W.n_anchored.notna() & (W.n_anchored >= 2)] if "n_anchored" in W else W.iloc[0:0]
nwin = W.groupby("episode").size()
q = lambda s, p: float(np.nanpercentile(s, p))
L = ["# Exterior-camera rig on labelled gripper points, test split\n",
     "Method: rig_track.py (RANSAC + weighted Kabsch, 5 mm inliers, growing body model, store GT wrist pose at each window's reference "
     "frame as the hand-eye stand-in), points = kinematic gripper labels (positions_v1). Window = maximal run of frames with >= 3 labelled "
     "points in view (inside the image, facing the camera) in both exterior cameras; each window is a separate evaluation, reference = its "
     "first frame with >= 4 such points. Scored against the store GT wrist poses; CUT3R finetuned (augfull_lr1e5) on the same frames.\n",
     "## Coverage\n",
     f"- scenes evaluated: {W.episode.nunique()} (record missing: {len(missing)})",
     f"- scenes with no window: {int((W.groupby('episode').n_anchored.apply(lambda s: s.notna().sum()) == 0).sum()) if 'n_anchored' in W else 'n/a'}",
     f"- windows: {len(W)}; scenes with more than one window: {int((nwin > 1).sum())} (max {int(nwin.max())} windows in one scene); "
     f"windows skipped (no frame with >= 4 points): {int(W.get('skipped', pd.Series(dtype=object)).notna().sum())}",
     f"- frames with >= 3 points in both cameras: {int(W.drop_duplicates('episode').frames_min_pts_both.sum())}; anchored (solved) frames: {len(ok)}\n",
     "## Per-frame wrist-pose error (all solved frames pooled)\n",
     "| | median | p90 | p99 | max | frames > 1 cm / 5° | frames > 5 cm / 20° |", "|---|---|---|---|---|---|---|",
     f"| position (cm) | {ok.pos_err_cm.median():.3f} | {q(ok.pos_err_cm, 90):.3f} | {q(ok.pos_err_cm, 99):.2f} | {ok.pos_err_cm.max():.1f} | "
     f"{(ok.pos_err_cm > 1).mean():.1%} | {(ok.pos_err_cm > 5).mean():.1%} |",
     f"| rotation (deg) | {ok.rot_err_deg.median():.3f} | {q(ok.rot_err_deg, 90):.2f} | {q(ok.rot_err_deg, 99):.1f} | {ok.rot_err_deg.max():.1f} | "
     f"{(ok.rot_err_deg > 5).mean():.1%} | {(ok.rot_err_deg > 20).mean():.1%} |",
     f"| centroid-only position (cm) | {ok.centroid_pos_err_cm.median():.2f} | {q(ok.centroid_pos_err_cm, 90):.2f} | {q(ok.centroid_pos_err_cm, 99):.1f} | "
     f"{ok.centroid_pos_err_cm.max():.1f} | {(ok.centroid_pos_err_cm > 1).mean():.1%} | {(ok.centroid_pos_err_cm > 5).mean():.1%} |\n",
     "## Per-window trajectory metrics (windows with >= 2 solved frames; Sim3-aligned, as in rig_track)\n",
     f"- windows: {len(ev)}",
     f"- rig ATE (m): median {ev.rig_ate.median():.5f}, mean {ev.rig_ate.mean():.5f}, p90 {q(ev.rig_ate, 90):.4f}",
     f"- rig RPE_trans (m) median {ev.rig_rpe_t.median():.5f}; RPE_rot (deg) median {ev.rig_rpe_rot.median():.3f}",
     f"- CUT3R finetuned, same frames: ATE median {ev.cut3r_ate_same_frames.median():.4f}, mean {ev.cut3r_ate_same_frames.mean():.4f}",
     f"- rig ATE lower than CUT3R's in {(ev.rig_ate < ev.cut3r_ate_same_frames).mean():.1%} of windows",
     f"- windows with a propagated failure (max per-frame position error > 5 cm): {int((ev.pos_err_max_cm > 5).sum())} "
     f"({(ev.pos_err_max_cm > 5).mean():.1%}); > 1 cm: {int((ev.pos_err_max_cm > 1).sum())}\n",
     "## Per lab\n", "| lab | scenes | windows | solved frames | pos median (cm) | pos p90 | rot median (deg) | rot p90 | windows max err > 5 cm |",
     "|---|---|---|---|---|---|---|---|---|"]
for lab, g in ok.groupby("lab"):
    wl = ev[ev.lab == lab]
    L.append(f"| {lab} | {g.episode.nunique()} | {len(wl)} | {len(g)} | {g.pos_err_cm.median():.3f} | {q(g.pos_err_cm, 90):.3f} | "
             f"{g.rot_err_deg.median():.3f} | {q(g.rot_err_deg, 90):.2f} | {int((wl.pos_err_max_cm > 5).sum())} |")
open(f"{run}/SUMMARY.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
