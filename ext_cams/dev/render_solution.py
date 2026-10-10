"""DEV: overlay model projection under the SOLVED motion (green) and under GT motion (red) for frames of one scene."""
import sys, os, csv, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra
d, ep, out = sys.argv[1], sys.argv[2], sys.argv[3]; frames = [int(v) for v in sys.argv[4].split(",")]
ev = [r for r in csv.DictReader(open(ra.EVENTS)) if r["episode"] == ep and r["event"] == "0"][0]; b = int(ev["entry_frame"])
cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS); zs = np.load(f"{ra.SEEDS}/{ep}/matches.npz")
Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], zs["x1"].T, zs["x2"].T); X = (Xh[:3] / Xh[3]).T
an = ra.Anchor(type("A", (), dict(tau_in=4, kabsch_m=0.01, huber_px=2))(), K, E, X)
z = np.load(f"{d}/{ep}/anchors.npz"); M = {int(f): m for f, m in zip(z["frames"], z["motions"])}
T = max(frames) + 1; gt = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", T, "gt")
col = {c: rt.read_frames(f"{rt.RAW_ROOT}/{ep}/recordings/MP4/{s}.mp4", T, gray=False) for c, s in cams}
rows = []
for ci, c in enumerate(("ext1", "ext2")):
    tiles = []
    for t in frames:
        Mg = gt[t] @ np.linalg.inv(gt[b]); ug, _ = an.project(ci, Mg[:3, :3], Mg[:3, 3]); cx, cy = ug.mean(0)
        x0, y0 = int(np.clip(cx - 120, 0, rt.W - 240)), int(np.clip(cy - 120, 0, rt.H - 240)); img = col[c][t][y0:y0 + 240, x0:x0 + 240].copy()
        img = cv2.resize(img, (360, 360)); s = 1.5
        for u, v in ug: cv2.circle(img, (int((u - x0) * s), int((v - y0) * s)), 4, (0, 0, 255), 1)
        if t in M:
            us, _ = an.project(ci, M[t][:3, :3], M[t][:3, 3])
            for u, v in us: cv2.circle(img, (int((u - x0) * s), int((v - y0) * s)), 3, (0, 255, 0), -1)
        cv2.putText(img, f"{c} t{t}{'' if t in M else ' unsolved'}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2); tiles.append(img)
    rows.append(np.concatenate(tiles, 1))
cv2.imwrite(out, np.concatenate(rows, 0))
