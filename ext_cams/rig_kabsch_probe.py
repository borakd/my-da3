#!/usr/bin/env python
"""Rig probe: recover the wrist camera's 6-DoF trajectory from the two STATIC exterior cameras alone, by tracking
surface points on the gripper in both views, triangulating them, and rigidly aligning each frame to a reference frame
(RANSAC + weighted Kabsch). Compare against the kinematic ground truth on the frames where the gripper is visible in
every camera. One episode, no training, no ground truth at any step except (a) the wrist pose at the REFERENCE frame,
used only to transfer the body motion to the lens (stands in for the hand-eye offset), and (b) the scoring.

Inputs on disk: exterior MP4s (vggt_cache/raw), SAM gripper masks from ext_cams/sam3_gripper_boxclick_3cams.py
(bit-packed union masks; used ONLY to seed points and to define the visibility window), PointWorld optimized extrinsics
(world-to-camera, world = robot base), factory intrinsics (hf_intrinsics.json), store GT wrist poses (c2w, base frame),
CUT3R finetuned predictions (augfull_lr1e5/preds) for the same-window comparison.

Cross-view correspondence is NOT given: points are seeded independently in each view (Shi-Tomasi inside the eroded mask),
tracked forward only with pyramidal LK + forward-backward check, and paired across views by the epipolar constraint
accumulated over time (a wrong pair violates it once the gripper moves). Everything is forward/causal.

    OMP_NUM_THREADS=1 python rig_kabsch_probe.py --res 1280  # native
    OMP_NUM_THREADS=1 python rig_kabsch_probe.py --res 320   # store resolution
"""
import argparse, csv, json, os, sys, time
import numpy as np, cv2

EP = "RAIL+80edfcb1+2023-07-14-14h-28m-45s"
OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
AUDIT = f"{OUT_ROOT}/vggt_probe/gt_audit_2026-09-08"
MASKS = f"{OUT_ROOT}/ext_cams/gripper_sam3/{EP}"
RAW = f"/gpfs/scratch/etur59/koc821022/vggt_cache/raw/{EP}/recordings/MP4"
STORE = f"/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist/{EP}/dense/cam"
CUT3R = f"{OUT_ROOT}/augfull_lr1e5/preds/{EP}/camera"
CAMS = [("ext1", "20521388"), ("ext2", "24259877")]


# ----------------------------------------------------------------------------- geometry helpers
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def kabsch(A, B, w=None):
    """R, t minimising sum w |R a + t - b|^2 (A, B: N x 3)."""
    w = np.ones(len(A)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    ca, cb = (w[:, None] * A).sum(0), (w[:, None] * B).sum(0)
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def ransac_kabsch(A, B, w, thr, iters=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(A)
    if n < 3:
        return None, None, np.zeros(n, bool)
    best, best_inl = None, np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        if np.linalg.matrix_rank(A[idx] - A[idx].mean(0)) < 2:
            continue
        R, t = kabsch(A[idx], B[idx])
        inl = np.linalg.norm((A @ R.T + t) - B, axis=1) < thr
        if inl.sum() > best_inl.sum():
            best_inl = inl
    if best_inl.sum() < 3:
        return None, None, best_inl
    R, t = kabsch(A[best_inl], B[best_inl], w[best_inl])
    return R, t, best_inl


def umeyama(src, dst):
    """Sim3 (s, R, t) with dst ~ s R src + t; src, dst: N x 3."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    H = S.T @ D / len(src)
    U, sig, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1, 1, d])
    R = Vt.T @ Dm @ U.T
    s = np.trace(np.diag(sig) @ Dm) / (S ** 2).sum() * len(src)
    return s, R, mu_d - s * R @ mu_s


def traj_metrics(pred, gt):
    """pred, gt: lists of 4x4 c2w. ATE = RMSE of camera centres after Sim3 alignment; RPE over consecutive frames on the
    aligned trajectory (translation in metres, rotation in degrees), mean over steps."""
    P, G = np.array(pred), np.array(gt)
    s, R, t = umeyama(P[:, :3, 3], G[:, :3, 3])
    Pa = P.copy()
    Pa[:, :3, 3] = (s * (R @ P[:, :3, 3].T)).T + t
    Pa[:, :3, :3] = R @ P[:, :3, :3]
    ate = np.sqrt(np.mean(np.sum((Pa[:, :3, 3] - G[:, :3, 3]) ** 2, 1)))
    rt, rr = [], []
    for i in range(len(P) - 1):
        dg = np.linalg.inv(G[i]) @ G[i + 1]
        dp = np.linalg.inv(Pa[i]) @ Pa[i + 1]
        e = np.linalg.inv(dg) @ dp
        rt.append(np.linalg.norm(e[:3, 3])); rr.append(rot_angle_deg(e[:3, :3]))
    return dict(ate=float(ate), rpe_trans=float(np.mean(rt)), rpe_rot=float(np.mean(rr)), sim3_scale=float(s), n=len(P))


# ----------------------------------------------------------------------------- inputs
def load_masks(cam, serial, W, H):
    m = np.load(f"{MASKS}/{cam}_{serial}__gripper_boxclick_masks.npz")
    T, h, w = m["shape"]
    M = np.unpackbits(m["union"], axis=2)[:, :, :w].astype(np.uint8)
    if (h, w) != (H, W):
        M = np.array([cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST) for x in M])
    return M


def read_frames(serial, t_end, W, H):
    cap = cv2.VideoCapture(f"{RAW}/{serial}.mp4")
    fr = []
    for t in range(t_end + 1):
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[1] != W:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        fr.append(img)
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=1280, help="processing width: 1280 (native) or 320 (store resolution)")
    ap.add_argument("--max_pts", type=int, default=120)
    ap.add_argument("--out", default=f"{OUT_ROOT}/ext_cams/rig_kabsch/{EP}")
    a = ap.parse_args()
    W = a.res; H = W * 9 // 16; sc = W / 1280.0
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()

    intr = json.load(open(f"{AUDIT}/docs/hf_intrinsics.json"))[EP]
    cams = json.load(open(f"{AUDIT}/pointworld/droid/cameras/{EP}_cameras.json"))
    K, E, P = {}, {}, {}
    for cam, s in CAMS:
        fx, cx, fy, cy = intr[s]["cameraMatrix"]
        K[cam] = np.array([[fx * sc, 0, cx * sc], [0, fy * sc, cy * sc], [0, 0, 1]])
        E[cam] = np.array(cams[s]["optimized_extrinsics"])  # world(base) -> camera
        P[cam] = K[cam] @ E[cam][:3]
    # fundamental matrix ext1 -> ext2 from the calibration
    R1, t1, R2, t2 = E["ext1"][:3, :3], E["ext1"][:3, 3], E["ext2"][:3, :3], E["ext2"][:3, 3]
    Rr = R2 @ R1.T; tr = t2 - Rr @ t1
    F = np.linalg.inv(K["ext2"]).T @ skew(tr) @ Rr @ np.linalg.inv(K["ext1"])

    M = {cam: load_masks(cam, s, W, H) for cam, s in CAMS}
    T = min(len(M[c]) for c in M)
    vis = np.array([all(M[c][t].any() for c in M) for t in range(T)])
    t0 = int(np.argmax(vis))
    t1 = t0
    while t1 + 1 < T and vis[t1 + 1]:
        t1 += 1
    print(f"visibility window (gripper in both exterior views, wrist always): frames {t0}..{t1} ({t1 - t0 + 1} frames)")

    frames = {cam: read_frames(s, T - 1, W, H) for cam, s in CAMS}
    gray = {cam: [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[cam]] for cam in frames}

    # ------------------------------------------------------------- seed + track (forward only, per view, re-seeding)
    area = {cam: np.array([M[cam][t].sum() for t in range(T)], float) for cam in M}
    med_area = {cam: np.median(area[cam][t0:t1 + 1]) for cam in M}
    t_seed = t0
    while t_seed < t1 and any(area[c][t_seed] < 0.3 * med_area[c] for c in M):
        t_seed += 1
    print(f"first frame where both masks reach 30% of their median area (reference frame): {t_seed}")
    er = np.ones((max(3, int(11 * sc)) | 1,) * 2, np.uint8)
    dil = np.ones((max(5, int(31 * sc)) | 1,) * 2, np.uint8)
    lk = dict(winSize=(max(7, int(21 * sc)) | 1,) * 2, maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    tracks = {}  # cam -> array (n, T, 2) with nan when lost; tracks are appended by re-seeding
    for cam in gray:
        tr_ = np.full((0, T, 2), np.nan, np.float32)
        alive = np.zeros(0, bool)
        n_seed_events = 0
        for t in range(t_seed, t1 + 1):
            if t > t_seed and alive.any():
                idx = np.where(alive)[0]
                p_prev = tr_[idx, t - 1].reshape(-1, 1, 2)
                p_next, st, _ = cv2.calcOpticalFlowPyrLK(gray[cam][t - 1], gray[cam][t], p_prev, None, **lk)
                p_back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[cam][t], gray[cam][t - 1], p_next, None, **lk)
                fb = np.linalg.norm(p_back - p_prev, axis=2).ravel()
                ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.5 * max(sc, 0.5))
                pn = p_next.reshape(-1, 2)
                dm = cv2.dilate(M[cam][t], dil)
                inm = np.array([0 <= x < W and 0 <= y < H and dm[int(y), int(x)] > 0 for x, y in pn])
                ok &= inm
                tr_[idx[ok], t] = pn[ok]
                alive[idx[~ok]] = False
            # re-seed whenever the live set is thin
            if alive.sum() < a.max_pts // 2 and area[cam][t] >= 0.3 * med_area[cam]:
                seedmask = cv2.erode(M[cam][t], er)
                if alive.any():
                    for x, y in tr_[alive, t]:
                        cv2.circle(seedmask, (int(x), int(y)), max(3, int(8 * sc)), 0, -1)
                pts = cv2.goodFeaturesToTrack(gray[cam][t], a.max_pts - int(alive.sum()), 0.005, max(3, int(6 * sc)), mask=seedmask)
                if pts is not None:
                    pts = pts.reshape(-1, 2).astype(np.float32)
                    new = np.full((len(pts), T, 2), np.nan, np.float32); new[:, t] = pts
                    tr_ = np.concatenate([tr_, new]); alive = np.concatenate([alive, np.ones(len(pts), bool)])
                    n_seed_events += 1
        tracks[cam] = tr_
        lens = (~np.isnan(tr_[:, :, 0])).sum(1)
        print(f"{cam}: {len(tr_)} tracks over {n_seed_events} seeding events; track length median {np.median(lens):.0f} frames, "
              f"max {lens.max()}; live at {t1}: {int(alive.sum())}")

    # ------------------------------------------------------------- cross-view pairing by accumulated epipolar residual
    A1, A2 = tracks["ext1"], tracks["ext2"]
    ok1, ok2 = ~np.isnan(A1[:, :, 0]), ~np.isnan(A2[:, :, 0])
    n1, n2 = len(A1), len(A2)
    res = np.full((n1, n2), np.inf)
    min_common = 10
    for i in range(n1):
        for j in range(n2):
            com = ok1[i] & ok2[j]
            if com.sum() < min_common:
                continue
            x1 = np.c_[A1[i, com], np.ones(com.sum())]; x2 = np.c_[A2[j, com], np.ones(com.sum())]
            l2 = x1 @ F.T; l1 = x2 @ F          # epipolar lines in view 2 and view 1
            d2 = np.abs((x2 * l2).sum(1)) / np.hypot(l2[:, 0], l2[:, 1])
            d1 = np.abs((x1 * l1).sum(1)) / np.hypot(l1[:, 0], l1[:, 1])
            res[i, j] = np.median(0.5 * (d1 + d2))
    thr_ep = 2.5 * max(sc, 0.4)
    pairs = []
    used2 = set()
    for i in np.argsort(res.min(1)):
        j = int(np.argmin(res[i]))
        if res[i, j] < thr_ep and j not in used2 and np.argmin(res[:, j]) == i:
            pairs.append((i, j)); used2.add(j)
    print(f"cross-view pairs by epipolar consistency (thr {thr_ep:.2f} px, >= {min_common} common frames): {len(pairs)} "
          f"of {n1} x {n2}; median residual of accepted pairs {np.median([res[i, j] for i, j in pairs]) if pairs else float('nan'):.2f} px")
    if len(pairs) < 3:
        print("FAIL: fewer than 3 cross-view pairs"); sys.exit(1)

    # ------------------------------------------------------------- triangulate per frame
    X = np.full((len(pairs), T, 3), np.nan)
    reproj = []
    for k, (i, j) in enumerate(pairs):
        com = np.where(ok1[i] & ok2[j])[0]
        x1 = A1[i, com].T.astype(np.float64); x2 = A2[j, com].T.astype(np.float64)
        Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], x1, x2)
        Xw = (Xh[:3] / Xh[3]).T
        X[k, com] = Xw
        for cam, xx in (("ext1", x1), ("ext2", x2)):
            pr = (P[cam] @ np.c_[Xw, np.ones(len(Xw))].T); pr = pr[:2] / pr[2]
            reproj.append(np.linalg.norm(pr - xx, axis=0))
    reproj = np.concatenate(reproj)
    print(f"triangulation reprojection error: median {np.median(reproj):.2f} px, p90 {np.percentile(reproj, 90):.2f} px")

    # ------------------------------------------------------------- GT + CUT3R
    gt = {t: np.load(f"{STORE}/{t:06d}.npz")["pose"].astype(np.float64) for t in range(t0, t1 + 1)}
    cut = {t: np.load(f"{CUT3R}/{t:06d}.npz")["pose"].astype(np.float64) for t in range(t0, t1 + 1)}

    # ------------------------------------------------------------- rigid alignment to the reference frame (growing body model)
    X0 = np.full((len(pairs), 3), np.nan)          # each paired point's coordinates in the REFERENCE frame
    n_tri = (~np.isnan(X[:, :, 0])).sum(0)                       # triangulated pairs per frame
    cand = np.where(n_tri >= 4)[0]
    if len(cand) == 0:
        print("FAIL: no frame with >= 4 triangulated pairs"); sys.exit(1)
    t_ref = int(cand[0])
    seed0 = ~np.isnan(X[:, t_ref, 0])
    print(f"triangulated pairs per frame over the window: median {np.median(n_tri[t0:t1 + 1]):.0f}, min {n_tri[t0:t1 + 1].min()}, "
          f"frames with >= 3: {(n_tri[t0:t1 + 1] >= 3).sum()} of {t1 - t0 + 1}")
    X0[seed0] = X[seed0, t_ref]
    spread = np.linalg.norm(X0[seed0] - X0[seed0].mean(0), axis=1)
    print(f"reference frame {t_ref}: {seed0.sum()} points, extent (max radius) {spread.max() * 100:.1f} cm, "
          f"depth from ext1 {np.linalg.norm(E['ext1'][:3, :3] @ X0[seed0].T + E['ext1'][:3, 3:], axis=0).mean():.2f} m")
    P0 = gt[t_ref]  # wrist pose at the reference frame: the only GT the rig uses (hand-eye stand-in)
    thr_in = 0.005
    rows, pred, centroid_pred, gtl, cutl = [], [], [], [], []
    c0 = P0[:3, 3]; cen0 = X0[seed0].mean(0)
    n_inl = []
    for t in range(t0, t1 + 1):
        have = ~np.isnan(X0[:, 0]) & ~np.isnan(X[:, t, 0])
        R, tt, inl = ransac_kabsch(X0[have], X[have, t], np.ones(have.sum()), thr_in) if have.sum() >= 3 else (None, None, None)
        if R is None:
            rows.append(dict(frame=t, n_pts=int(have.sum()), n_inl=0, rigid_resid_mm=np.nan, pos_err_cm=np.nan, rot_err_deg=np.nan,
                             centroid_pos_err_cm=np.nan)); continue
        Mt = np.eye(4); Mt[:3, :3] = R; Mt[:3, 3] = tt
        # grow the body model: points first triangulated at this frame are registered into the reference frame
        newp = np.isnan(X0[:, 0]) & ~np.isnan(X[:, t, 0])
        if newp.any():
            X0[newp] = (X[newp, t] - tt) @ R      # R^T (x - t)
        Pt = Mt @ P0
        e_pos = np.linalg.norm(Pt[:3, 3] - gt[t][:3, 3]) * 100
        e_rot = rot_angle_deg(Pt[:3, :3].T @ gt[t][:3, :3])
        cen_t = (X0[have][inl] @ R.T + tt).mean(0); cen_ref = X0[have][inl].mean(0)
        c_cen = c0 + (cen_t - cen_ref)          # translation-only transfer through the inlier centroid (ignores the lever arm)
        e_cen = np.linalg.norm(c_cen - gt[t][:3, 3]) * 100
        resid = np.linalg.norm((X0[have][inl] @ R.T + tt) - X[have, t][inl], axis=1)
        rows.append(dict(frame=t, n_pts=int(have.sum()), n_inl=int(inl.sum()), rigid_resid_mm=float(np.median(resid) * 1000),
                         pos_err_cm=float(e_pos), rot_err_deg=float(e_rot), centroid_pos_err_cm=float(e_cen)))
        pred.append(Pt); gtl.append(gt[t]); cutl.append(cut[t]); n_inl.append(int(inl.sum()))
        Pc = np.eye(4); Pc[:3, :3] = Pt[:3, :3]; Pc[:3, 3] = c_cen; centroid_pred.append(Pc)
    have0 = seed0
    # ------------------------------------------------------------- refined body model (generalized Procrustes over all frames)
    # Each point's reference coordinates are re-estimated as the robust average of M_t^{-1} X[k,t] over the frames where it
    # was an inlier, then every frame is re-aligned; 3 rounds. Uses all frames of the window (batch), so this variant reports
    # what the causal model would reach with a well-estimated body model rather than a causal row by itself.
    Xr = X0.copy(); Ms = {}
    for it in range(3):
        Ms = {}; acc = {k: [] for k in range(len(pairs))}
        for t in range(t0, t1 + 1):
            have = ~np.isnan(Xr[:, 0]) & ~np.isnan(X[:, t, 0])
            if have.sum() < 3:
                continue
            R, tt, inl = ransac_kabsch(Xr[have], X[have, t], np.ones(have.sum()), thr_in, seed=it)
            if R is None:
                continue
            Ms[t] = (R, tt)
            for k, ok_ in zip(np.where(have)[0], inl):
                if ok_:
                    acc[k].append((X[k, t] - tt) @ R)
        for k in acc:
            if len(acc[k]) >= 3:
                Xr[k] = np.median(np.array(acc[k]), 0)
    if t_ref in Ms:
        Rr_, tr_ = Ms[t_ref]; Mref = np.eye(4); Mref[:3, :3] = Rr_; Mref[:3, 3] = tr_
    else:
        Mref = np.eye(4)
    pred_r, gt_r, cut_r, pe_r, re_r = [], [], [], [], []
    for t in range(t0, t1 + 1):
        if t not in Ms:
            continue
        R, tt = Ms[t]; Mt = np.eye(4); Mt[:3, :3] = R; Mt[:3, 3] = tt
        Pt = Mt @ np.linalg.inv(Mref) @ P0
        pred_r.append(Pt); gt_r.append(gt[t]); cut_r.append(cut[t])
        pe_r.append(np.linalg.norm(Pt[:3, 3] - gt[t][:3, 3]) * 100); re_r.append(rot_angle_deg(Pt[:3, :3].T @ gt[t][:3, :3]))
    m_ref = traj_metrics(pred_r, gt_r); m_cut_r = traj_metrics(cut_r, gt_r)
    refined = dict(frames_solved=len(pred_r), pos_cm=dict(median=float(np.median(pe_r)), p90=float(np.percentile(pe_r, 90)), max=float(np.max(pe_r))),
                   rot_deg=dict(median=float(np.median(re_r)), p90=float(np.percentile(re_r, 90)), max=float(np.max(re_r))),
                   window_metrics=dict(rig_refined=m_ref, cut3r_finetuned_same_frames=m_cut_r))
    print("refined body model:", json.dumps(refined))
    np.savez(f"{a.out}/tracks_res{W}.npz", X=X, X0=X0, pairs=np.array(pairs), t_ref=t_ref, t0=t0, t1=t1,
             gt=np.array([gt[t] for t in range(t0, t1 + 1)]), cut=np.array([cut[t] for t in range(t0, t1 + 1)]),
             pred=np.array(pred), pred_frames=np.array([r["frame"] for r in rows if r["n_inl"] >= 3]), pred_refined=np.array(pred_r),
             tr1=tracks["ext1"], tr2=tracks["ext2"], P_ext1=P["ext1"], P_ext2=P["ext2"], E_ext1=E["ext1"], E_ext2=E["ext2"],
             n_inl=np.array([r["n_inl"] for r in rows]), rigid_resid_mm=np.array([r["rigid_resid_mm"] if r["n_inl"] >= 3 else np.nan for r in rows]),
             lens_ref=P0[:3, 3], model_centroid=np.nanmean(X0, 0), model_extent=float(np.nanmax(np.linalg.norm(X0 - np.nanmean(X0, 0), axis=1))))
    with open(f"{a.out}/per_frame_res{W}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[-1].keys())); wr.writeheader(); wr.writerows(rows)
    pe = np.array([r["pos_err_cm"] for r in rows if r["n_inl"] >= 3]); re_ = np.array([r["rot_err_deg"] for r in rows if r["n_inl"] >= 3])
    ce = np.array([r["centroid_pos_err_cm"] for r in rows if r["n_inl"] >= 3])
    m_rig = traj_metrics(pred, gtl); m_cut = traj_metrics(cutl, gtl); m_cen = traj_metrics(centroid_pred, gtl)
    summary = dict(episode=EP, res=[W, H], window=[t0, t1], t_seed=t_seed, t_ref=t_ref, n_frames=t1 - t0 + 1, frames_solved=len(pred),
                   seeds={c: int(len(tracks[c])) for c in tracks}, pairs=len(pairs),
                   reproj_px=dict(median=float(np.median(reproj)), p90=float(np.percentile(reproj, 90))),
                   ref_cloud=dict(n=int(have0.sum()), extent_cm=float(spread.max() * 100)),
                   inliers=dict(median=float(np.median(n_inl)), min=int(np.min(n_inl))),
                   per_frame_vs_gt=dict(pos_cm=dict(median=float(np.median(pe)), p90=float(np.percentile(pe, 90)), max=float(pe.max())),
                                        rot_deg=dict(median=float(np.median(re_)), p90=float(np.percentile(re_, 90)), max=float(re_.max())),
                                        centroid_only_pos_cm=dict(median=float(np.median(ce)), p90=float(np.percentile(ce, 90)))),
                   window_metrics=dict(rig_kabsch=m_rig, rig_centroid_only=m_cen, cut3r_finetuned=m_cut), refined_body_model=refined,
                   elapsed_s=time.time() - t_start)
    json.dump(summary, open(f"{a.out}/summary_res{W}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))

    # ------------------------------------------------------------- overlay video: tracked points + predicted vs GT lens
    vw = cv2.VideoWriter(f"{a.out}/overlay_res{W}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (2 * 1280, 720))
    pair1 = {i for i, _ in pairs}; pair2 = {j for _, j in pairs}
    pi = 0
    for t in range(t0, t1 + 1):
        panels = []
        solved = rows[t - t0]["n_inl"] >= 3
        for cam, tr_, pset in (("ext1", A1, pair1), ("ext2", A2, pair2)):
            img = frames[cam][t].copy()
            s_ = 1280 / W
            cnts, _ = cv2.findContours(M[cam][t], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if s_ != 1:
                img = cv2.resize(img, (1280, 720)); cnts = [(c * s_).astype(np.int32) for c in cnts]
            cv2.drawContours(img, cnts, -1, (0, 200, 0), 1)
            for n in range(len(tr_)):
                if np.isnan(tr_[n, t, 0]):
                    continue
                x, y = tr_[n, t] * s_
                cv2.circle(img, (int(x), int(y)), 4, (255, 200, 0) if n in pset else (128, 128, 128), -1)
            g = gt[t][:3, 3]; pg = P[cam] @ np.r_[g, 1] / s_; pg = pg[:2] / pg[2] * s_
            cv2.drawMarker(img, (int(pg[0]), int(pg[1])), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
            if solved:
                p_ = pred[pi][:3, 3]; pp = P[cam] @ np.r_[p_, 1] / s_; pp = pp[:2] / pp[2] * s_
                cv2.circle(img, (int(pp[0]), int(pp[1])), 7, (0, 0, 255), 2)
            cv2.putText(img, f"{cam} f{t}  cyan=paired tracks grey=unpaired  green+=GT lens  red o=rig lens", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            panels.append(img)
        if solved:
            r = rows[t - t0]
            cv2.putText(panels[0], f"pos err {r['pos_err_cm']:.1f} cm  rot err {r['rot_err_deg']:.1f} deg  inliers {r['n_inl']}/{r['n_pts']}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            pi += 1
        vw.write(np.hstack(panels))
    vw.release()
    print("wrote", a.out)
    gt_all = {t: np.load(f"{STORE}/{t:06d}.npz")["pose"].astype(np.float64) for t in range(T)}
    cut_all = {t: np.load(f"{CUT3R}/{t:06d}.npz")["pose"].astype(np.float64) for t in range(T)}
    make_side_by_side(a, W, T, t0, t1, frames, M, tracks, pairs, rows, pred, [r["frame"] for r in rows if r["n_inl"] >= 3], P, gt_all, cut_all, summary)


# ----------------------------------------------------------------------------- 3-panel video (same format as the SAM3 side-by-side)
def make_side_by_side(a, W, T, t0, t1, frames, M, tracks, pairs, rows, pred, pred_frames, P, gt_all, cut_all, summary):
    from sam3_gripper_masks import FfmpegWriter
    PW, PH = 640, 360
    s_ = 1280 / W
    pair1 = {i for i, _ in pairs}; pair2 = {j for _, j in pairs}
    pred_by_t = {int(f): p for f, p in zip(pred_frames, pred)}
    # CUT3R poses Sim3-aligned to GT on the solved frames (evaluation-style), for display only
    sf = [int(f) for f in pred_frames]
    sc_, Rc, tc = umeyama(np.array([cut_all[t][:3, 3] for t in sf]), np.array([gt_all[t][:3, 3] for t in sf]))
    cut_c = {t: sc_ * Rc @ cut_all[t][:3, 3] + tc for t in range(T)}
    rig_m, cut_m = summary["window_metrics"]["rig_kabsch"], summary["window_metrics"]["cut3r_finetuned"]
    # top-down plot geometry (base-frame x,y over the window)
    G = np.array([gt_all[t][:3, 3] for t in range(t0, t1 + 1)])
    lo, hi = G[:, :2].min(0) - 0.05, G[:, :2].max(0) + 0.05
    span = (hi - lo).max(); cen = (lo + hi) / 2; lo, hi = cen - span / 2, cen + span / 2
    IW, IH, IX, IY = 230, 230, PW - 240, PH - 240
    def to_inset(xy):
        u = IX + (xy[0] - lo[0]) / span * (IW - 1); v = IY + (IH - 1) - (xy[1] - lo[1]) / span * (IH - 1)
        return int(u), int(v)
    writer = FfmpegWriter(f"{a.out}/all_cams_rig_kabsch_side_by_side_res{W}.mp4", 3 * PW, PH, 15)
    def banner(img, text, color):
        cv2.rectangle(img, (0, PH // 2 - 22), (PW, PH // 2 + 22), color, -1)
        cv2.putText(img, text, (PW // 2 - 7 * len(text), PH // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for t in range(T):
        in_win = t0 <= t <= t1
        r = rows[t - t0] if in_win else None
        solved = in_win and r["n_inl"] >= 3
        # --- wrist panel: store frame + trajectory inset
        wimg = cv2.imread(f"{STORE.replace('/cam', '/rgb')}/{t:06d}.png")
        wimg = cv2.resize(wimg, (PW, PH), interpolation=cv2.INTER_CUBIC)
        cv2.putText(wimg, f"wrist f{t}  rig = 2 static cams, tracked gripper points, Kabsch", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ov = wimg.copy(); cv2.rectangle(ov, (IX - 6, IY - 6), (IX + IW + 6, IY + IH + 6), (30, 30, 30), -1)
        wimg = cv2.addWeighted(ov, 0.75, wimg, 0.25, 0)
        cv2.putText(wimg, "top view (base x,y)", (IX + 4, IY + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        if in_win:
            for tt in range(t0, min(t, t1) + 1):
                if tt > t0:
                    cv2.line(wimg, to_inset(gt_all[tt - 1][:3, 3]), to_inset(gt_all[tt][:3, 3]), (0, 220, 0), 2)
                    cv2.line(wimg, to_inset(cut_c[tt - 1]), to_inset(cut_c[tt]), (255, 120, 0), 1)
                    if tt in pred_by_t and (tt - 1) in pred_by_t:
                        cv2.line(wimg, to_inset(pred_by_t[tt - 1][:3, 3]), to_inset(pred_by_t[tt][:3, 3]), (0, 0, 255), 2)
            cv2.circle(wimg, to_inset(gt_all[t][:3, 3]), 4, (0, 220, 0), -1)
            cv2.circle(wimg, to_inset(cut_c[t]), 4, (255, 120, 0), -1)
            if t in pred_by_t:
                cv2.circle(wimg, to_inset(pred_by_t[t][:3, 3]), 4, (0, 0, 255), -1)
        y = 44
        for text, col in ((f"green GT   red rig   blue CUT3R (Sim3-aligned)", (255, 255, 255)),
                          (f"window f{t0}-{t1}: ATE rig {rig_m['ate']:.4f} vs CUT3R {cut_m['ate']:.4f}", (255, 255, 255)),
                          (f"RPE_t rig {rig_m['rpe_trans']:.4f} vs {cut_m['rpe_trans']:.4f}   RPE_rot {rig_m['rpe_rot']:.2f} vs {cut_m['rpe_rot']:.2f}", (255, 255, 255))):
            cv2.putText(wimg, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1); y += 20
        if solved:
            cv2.putText(wimg, f"this frame: lens pos err {r['pos_err_cm']:.1f} cm  rot err {r['rot_err_deg']:.1f} deg  inliers {r['n_inl']}/{r['n_pts']}",
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        panels = [wimg]
        # --- exterior panels
        for cam, tr_, pset in (("ext1", tracks["ext1"], pair1), ("ext2", tracks["ext2"], pair2)):
            img = frames[cam][t].copy()
            if img.shape[1] != 1280:
                img = cv2.resize(img, (1280, 720))
            cnts, _ = cv2.findContours(M[cam][t], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [(c * s_).astype(np.int32) for c in cnts]
            cv2.drawContours(img, cnts, -1, (0, 200, 0), 2)
            n_tr = 0
            for n in range(len(tr_)):
                if np.isnan(tr_[n, t, 0]):
                    continue
                n_tr += 1
                x, y_ = tr_[n, t] * s_
                cv2.circle(img, (int(x), int(y_)), 5, (255, 200, 0) if n in pset else (140, 140, 140), -1)
            def proj(pt):
                q = P[cam] @ np.r_[pt, 1] / s_; return int(q[0] / q[2] * s_), int(q[1] / q[2] * s_)
            if in_win:
                cv2.drawMarker(img, proj(gt_all[t][:3, 3]), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
                cv2.rectangle(img, tuple(np.array(proj(cut_c[t])) - 9), tuple(np.array(proj(cut_c[t])) + 9), (255, 120, 0), 3)
                if solved:
                    cv2.circle(img, proj(pred_by_t[t][:3, 3]), 12, (0, 0, 255), 3)
            img = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_AREA)
            cv2.putText(img, f"{cam} f{t}  tracks {n_tr}  cyan=cross-view paired  + GT lens  o rig lens  [] CUT3R", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            panels.append(img)
        if not in_win:
            for pnl in panels:
                banner(pnl, "GRIPPER NOT IN ALL CAMERAS", (0, 0, 160))
        elif not solved:
            for pnl in panels:
                banner(pnl, "RIG NOT SOLVED (<3 tracked points)", (0, 140, 180))
        writer.write(np.concatenate(panels, axis=1))
    writer.close()
    print("wrote", f"{a.out}/all_cams_rig_kabsch_side_by_side_res{W}.mp4")


if __name__ == "__main__":
    main()
