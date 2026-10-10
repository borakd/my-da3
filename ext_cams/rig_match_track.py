#!/usr/bin/env python
"""rig_match_track: GT-free cross-view correspondences for the exterior rig (fix 1).

At the birth frame b (entry_events.csv, event 0), the two exterior images are MATCHED inside the gripper region, and
every matched point is then TRACKED forward in its own view. The two cameras therefore carry the same physical points
with identical indexing: pairs are known by construction and the rig's epipolar pairing stage is not needed.

Nothing here reads a wrist pose. Inputs: the raw exterior MP4s, the calibration (factory intrinsics + PointWorld
extrinsics, as in the rig), the birth/exit frames (FK, the pipeline's existing assumption) and the gripper REGION per
camera = dilated convex hull of Track-On's query points (which were sampled inside the RobotSeg mask at b; an
image-based proxy for the mask, which is not on MN5).

matchers
  sift   OpenCV SIFT on the full 1280x720 grey frame inside the region, ratio 0.8, mutual nearest neighbour
  loftr  kornia LoFTR (outdoor weights) on the two region crops, upscaled so the crop's long side is --loftr_side px
gates (all GT-free): symmetric epipolar distance under the calibration's F < --epi_px, both points inside the regions,
triangulated depth in both cameras within [0.15, 2.5] m, one-to-one, >= 3 px apart in each view.
tracking: pyramidal LK forward with the forward-backward check, rig_track's validated 720p parameters scaled by the
gripper's apparent size (same estimator as the rig: pixels per cm from the triangulated centroid of the matches);
a track dies when the check fails or it leaves the image. --reseed re-matches at frame t whenever fewer than half of
the live pairs remain (region = dilated hull of the live tracks), appending new pairs seeded at t.

Output DIR/<ep>/{ext1,ext2}_f<b:05d>.npz in the Track-On layout: tracks (T,N,2) float32 NaN before the seed frame
and after loss, visibility (T,N) bool, queries (N,3) [t_seed,x,y], meta json; plus DIR/<ep>/matches.npz (birth-frame
matches and residuals) and, with --viz, DIR/<ep>/matches_f<b>.jpg.
    python rig_match_track.py --episode EP --tracks_v2 DIR --events CSV --out DIR --matcher sift|loftr [--reseed] [--viz]
"""
import argparse, csv, json, os, sys, time
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_track as rt            # noqa: E402
import rig_trackon_eval as rte    # noqa: E402

DEPTH_RANGE = (0.15, 2.5)


def region_mask(pts, dil):
    m = np.zeros((rt.H, rt.W), np.uint8)
    if len(pts) >= 3:
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts).astype(np.int32)), 255)
    elif len(pts):
        for x, y in pts:
            cv2.circle(m, (int(x), int(y)), dil, 255, -1)
    return cv2.dilate(m, np.ones((2 * dil + 1,) * 2, np.uint8))


def epi_dist(F, x1, x2):
    """symmetric epipolar distance (px) for N correspondences, x1 in view 1, x2 in view 2."""
    h1 = np.c_[x1, np.ones(len(x1))]; h2 = np.c_[x2, np.ones(len(x2))]
    l2 = h1 @ F.T; l1 = h2 @ F
    d2 = np.abs((h2 * l2).sum(1)) / np.hypot(l2[:, 0], l2[:, 1]); d1 = np.abs((h1 * l1).sum(1)) / np.hypot(l1[:, 0], l1[:, 1])
    return 0.5 * (d1 + d2)


def match_sift(g1, g2, m1, m2):
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    k1, d1 = sift.detectAndCompute(g1, m1); k2, d2 = sift.detectAndCompute(g2, m2)
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    fwd = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(d1, d2, k=2) if len(m) == 2 and m[0].distance < 0.8 * m[1].distance}
    bwd = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(d2, d1, k=2) if len(m) == 2 and m[0].distance < 0.8 * m[1].distance}
    pairs = [(i, j) for i, j in fwd.items() if bwd.get(j) == i]
    x1 = np.array([k1[i].pt for i, _ in pairs], np.float64).reshape(-1, 2); x2 = np.array([k2[j].pt for _, j in pairs], np.float64).reshape(-1, 2)
    return x1, x2, np.ones(len(pairs))


_LOFTR = None


def match_loftr(g1, g2, m1, m2, side):
    global _LOFTR
    import torch, kornia
    if _LOFTR is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _LOFTR = (kornia.feature.LoFTR(pretrained="outdoor").eval().to(dev), dev)
    model, dev = _LOFTR
    crops, boxes, scales = [], [], []
    for g, m in ((g1, m1), (g2, m2)):
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
        x0, x1_, y0, y1_ = max(xs.min() - 20, 0), min(xs.max() + 21, rt.W), max(ys.min() - 20, 0), min(ys.max() + 21, rt.H)
        crop = g[y0:y1_, x0:x1_]; s = side / max(crop.shape)
        cw, ch = max(8, int(round(crop.shape[1] * s / 8)) * 8), max(8, int(round(crop.shape[0] * s / 8)) * 8)
        crops.append(cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_CUBIC)); boxes.append((x0, y0)); scales.append((cw / crop.shape[1], ch / crop.shape[0]))
    with torch.no_grad():
        inp = {f"image{i}": torch.from_numpy(c).float().div(255)[None, None].to(dev) for i, c in enumerate(crops)}
        out = model(inp)
    k0, k1, conf = out["keypoints0"].cpu().numpy(), out["keypoints1"].cpu().numpy(), out["confidence"].cpu().numpy()
    x1 = k0 / np.array(scales[0]) + np.array(boxes[0]); x2 = k1 / np.array(scales[1]) + np.array(boxes[1])
    return x1.astype(np.float64), x2.astype(np.float64), conf


_DISK = None


def _disk_features(g, m, side, n=2048):
    """DISK keypoints + descriptors on the region crop upscaled to `side` px (long side); keypoints mapped back to full-res."""
    global _DISK
    import torch, kornia
    if _DISK is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _DISK = (kornia.feature.DISK.from_pretrained("depth").eval().to(dev), dev)
    model, dev = _DISK
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return np.zeros((0, 2)), np.zeros((0, 128), np.float32)
    x0, x1_, y0, y1_ = max(xs.min() - 20, 0), min(xs.max() + 21, rt.W), max(ys.min() - 20, 0), min(ys.max() + 21, rt.H)
    crop = g[y0:y1_, x0:x1_]; s = side / max(crop.shape)
    cw, ch = max(16, int(round(crop.shape[1] * s / 16)) * 16), max(16, int(round(crop.shape[0] * s / 16)) * 16)
    up = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_CUBIC)
    with torch.no_grad():
        f = model(torch.from_numpy(up).float().div(255)[None, None].repeat(1, 3, 1, 1).to(dev), n=n, window_size=5, score_threshold=0.0, pad_if_not_divisible=True)[0]
    kp = f.keypoints.cpu().numpy() / np.array([cw / crop.shape[1], ch / crop.shape[0]]) + np.array([x0, y0])
    return kp.astype(np.float64), f.descriptors.cpu().numpy().astype(np.float32)


def match_disk_epi(g1, g2, m1, m2, F, side, epi_px, ratio=0.9):
    """Epipolar-GUIDED matching: for every DISK keypoint in view 1 the candidates are the view-2 keypoints within epi_px of
    its epipolar line (and vice versa); best by descriptor distance with a ratio test against the second candidate ON THE
    LINE; mutual. The constraint does the heavy lifting on the textureless gripper, where unconstrained matching fails."""
    k1, d1 = _disk_features(g1, m1, side); k2, d2 = _disk_features(g2, m2, side)
    if len(k1) == 0 or len(k2) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    h1 = np.c_[k1, np.ones(len(k1))]; h2 = np.c_[k2, np.ones(len(k2))]
    l2 = h1 @ F.T; l1 = h2 @ F                                              # lines in view 2 of view-1 points, and vice versa
    dep = 0.5 * (np.abs(h2 @ l2.T).T / np.hypot(l2[:, 0], l2[:, 1])[:, None] + np.abs(h1 @ l1.T) / np.hypot(l1[:, 0], l1[:, 1])[None])
    d1n = d1 / np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-9); d2n = d2 / np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-9)
    D = 1.0 - d1n @ d2n.T; D[dep >= epi_px] = np.inf
    best2 = np.argmin(D, 1); fwd = {}
    for i in range(len(k1)):
        j = best2[i]
        if not np.isfinite(D[i, j]): continue
        row = np.sort(D[i][np.isfinite(D[i])])
        if len(row) == 1 or row[0] < ratio * row[1]: fwd[i] = j
    best1 = np.argmin(D, 0); pairs = [(i, j) for i, j in fwd.items() if best1[j] == i]
    x1 = np.array([k1[i] for i, _ in pairs]).reshape(-1, 2); x2 = np.array([k2[j] for _, j in pairs]).reshape(-1, 2)
    return x1, x2, np.array([1.0 - D[i, j] for i, j in pairs])


_LG = None


def match_lightglue(g1, g2, m1, m2, side):
    """DISK features + LightGlue (learned matcher), unconstrained; the epipolar gate is applied afterwards like LoFTR/SIFT."""
    global _LG
    import torch, kornia
    k1, d1 = _disk_features(g1, m1, side); k2, d2 = _disk_features(g2, m2, side)
    if len(k1) < 2 or len(k2) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    dev = _DISK[1]
    if _LG is None:
        _LG = kornia.feature.LightGlueMatcher("disk").eval().to(dev)
    laf = lambda k: kornia.feature.laf_from_center_scale_ori(torch.from_numpy(k).float()[None].to(dev))
    with torch.no_grad():
        dists, idx = _LG(torch.from_numpy(d1).to(dev), torch.from_numpy(d2).to(dev), laf(k1), laf(k2))
    idx = idx.cpu().numpy()
    return k1[idx[:, 0]], k2[idx[:, 1]], np.ones(len(idx))


def gate(x1, x2, conf, m1, m2, F, P, E, epi_px, conf_min):
    """GT-free gates; returns the kept indices and their epipolar residuals."""
    if len(x1) == 0:
        return np.zeros(0, int), np.zeros(0)
    res = epi_dist(F, x1, x2)
    keep = (conf >= conf_min) & (res < epi_px)
    ii = np.round(x1).astype(int); jj = np.round(x2).astype(int)
    inside = lambda p: (p[:, 0] >= 0) & (p[:, 0] < rt.W) & (p[:, 1] >= 0) & (p[:, 1] < rt.H)
    keep &= inside(ii) & inside(jj)
    keep[keep] &= (m1[ii[keep, 1], ii[keep, 0]] > 0) & (m2[jj[keep, 1], jj[keep, 0]] > 0)
    if keep.any():
        Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], x1[keep].T, x2[keep].T); Xw = (Xh[:3] / Xh[3]).T
        ok = np.ones(keep.sum(), bool)
        for c in ("ext1", "ext2"):
            z = (E[c][:3, :3] @ Xw.T + E[c][:3, 3:]).T[:, 2]; ok &= (z > DEPTH_RANGE[0]) & (z < DEPTH_RANGE[1])
        keep[np.where(keep)[0][~ok]] = False
    idx = np.where(keep)[0][np.argsort(res[keep])]
    sel = []
    for i in idx:   # one-to-one and >= 3 px apart in each view (near-duplicates carry no extra constraint)
        if all(np.hypot(*(x1[i] - x1[j])) >= 3 and np.hypot(*(x2[i] - x2[j])) >= 3 for j in sel):
            sel.append(i)
    sel = np.array(sel, int)
    return sel, res[sel]


ARGS = [
    ("--episode", dict(required=True)),
    ("--tracks_v2", dict(required=True, help="Track-On v2 tracks dir (query points / per-frame region)")),
    ("--events", dict(required=True)),
    ("--out", dict(required=True)),
    ("--matcher", dict(default="disk_epi", choices=["sift", "loftr", "disk_epi", "lightglue"])),
    ("--loftr_side", dict(type=int, default=640, help="long side (px) of the upscaled region crop fed to LoFTR / DISK")),
    ("--epi_px", dict(type=float, default=2.5)),
    ("--conf_min", dict(type=float, default=0.3)),
    ("--region_dilate", dict(type=int, default=8, help="px; keep tight: static background inside the region passes the epipolar gate trivially")),
    ("--reseed", dict(action="store_true", help="LK tracker only: re-match when fewer than max(4, half) of the pairs remain")),
    ("--reseed_every", dict(type=int, default=0, help="LK + --reseed: also re-match every K frames (0 = only on attrition)")),
    ("--tracker", dict(default="cotracker", choices=["lk", "cotracker", "external"], help="cotracker = CoTracker3 online (GPU); external = --ext_tracks")),
    ("--ext_tracks", dict(default=None, help="--tracker external: dir with <ep>/{ext1,ext2}_f<b>.npz tracks seeded on these matches")),
    ("--seeds_from", dict(default=None, help="load the birth-frame matches from DIR/<ep>/matches.npz instead of matching")),
    ("--epi_track_px", dict(type=float, default=3.0, help="per-frame epipolar gate on tracked pairs (0 = off)")),
    ("--lk_relax", dict(action="store_true", help="LK: forward-backward threshold x2, window x1.5 (fast-moving gripper)")),
    ("--max_pairs", dict(type=int, default=150)),
    ("--viz", dict(action="store_true")),
]


def run(a):
    ep = a.episode; t_start = time.time()
    ev = [r for r in csv.DictReader(open(a.events)) if r["episode"] == ep and r["event"] == "0"]
    if not ev:
        raise rt.Failure("no event 0 (no birth frame / excluded)")
    b, e = int(ev[0]["entry_frame"]), int(ev[0]["exit_frame"])
    cams, K, E, P, ksrc = rte.calib(ep, rte.POSITIONS)
    R1, t1_, R2, t2_ = E["ext1"][:3, :3], E["ext1"][:3, 3], E["ext2"][:3, :3], E["ext2"][:3, 3]
    Rr = R2 @ R1.T; trel = t2_ - Rr @ t1_
    F = np.linalg.inv(K["ext2"]).T @ rt.skew(trel) @ Rr @ np.linalg.inv(K["ext1"])
    meta = json.load(open(f"{rt.RAW_ROOT}/{ep}/metadata_{ep}.json"))
    store = f"{rt.STORE_ROOT}/{ep}/dense/cam"; n_store = rt.count_files(store, ".npz")
    T = min(e, n_store)
    qz = {c: np.load(f"{a.tracks_v2}/{ep}/{c}_f{b:05d}.npz") for c, _ in cams}
    q = {c: qz[c]["queries"][:, 1:] for c in qz}
    qtr = {c: qz[c]["tracks"] for c in qz}; qvis = {c: qz[c]["visibility"] for c in qz}   # (T2,N,2), (T2,N): region at any frame t

    def region_at(c, t):
        """dilated hull of the Track-On points visible & in-image at frame t (None when fewer than 3)."""
        if t >= len(qtr[c]): return None
        p = qtr[c][t]; ok = qvis[c][t] & ~np.isnan(p[:, 0]) & (p[:, 0] >= 0) & (p[:, 0] < rt.W) & (p[:, 1] >= 0) & (p[:, 1] < rt.H)
        return region_mask(p[ok], a.region_dilate) if ok.sum() >= 3 else None
    gray = {c: rt.read_frames(f"{rt.RAW_ROOT}/{ep}/recordings/MP4/{s}.mp4", T) for c, s in cams}
    T = min(T, min(len(gray[c]) for c in gray))
    if T - b < 2:
        raise rt.Failure(f"window too short (b={b}, T={T})")
    reg = {c: region_mask(q[c], a.region_dilate) for c in q}
    matcher = {"sift": lambda g1, g2, m1, m2: match_sift(g1, g2, m1, m2),
               "loftr": lambda g1, g2, m1, m2: match_loftr(g1, g2, m1, m2, a.loftr_side),
               "disk_epi": lambda g1, g2, m1, m2: match_disk_epi(g1, g2, m1, m2, F, a.loftr_side, a.epi_px),
               "lightglue": lambda g1, g2, m1, m2: match_lightglue(g1, g2, m1, m2, a.loftr_side)}[a.matcher]

    def do_match(t, m1, m2):
        x1, x2, conf = matcher(gray["ext1"][t], gray["ext2"][t], m1, m2)
        sel, res = gate(x1, x2, conf, m1, m2, F, P, E, a.epi_px, a.conf_min)
        return x1[sel][: a.max_pairs], x2[sel][: a.max_pairs], res[: a.max_pairs], len(x1)

    if a.seeds_from:   # reuse the birth-frame matches of an earlier run (identical seeds across tracker arms)
        sf = f"{a.seeds_from}/{ep}/matches.npz"
        if not os.path.isfile(sf): raise rt.Failure(f"no seeds file {sf}")
        zs = np.load(sf); x1, x2 = zs["x1"].astype(np.float64), zs["x2"].astype(np.float64); res = epi_dist(F, x1, x2); n_raw = len(x1)
    else:
        x1, x2, res, n_raw = do_match(b, reg["ext1"], reg["ext2"])
    rec = dict(episode=ep, matcher=a.matcher, birth=b, exit=e, T=T, n_raw_matches=int(n_raw), n_pairs_birth=int(len(x1)),
               epi_res_median_px=float(np.median(res)) if len(res) else None, region_px={c: int((reg[c] > 0).sum()) for c in reg}, K_src=ksrc)
    if len(x1) < 3:
        rec["failure_reason"] = f"fewer than 3 gated matches at the birth frame ({len(x1)} of {n_raw} raw)"
        os.makedirs(f"{a.out}/{ep}", exist_ok=True); json.dump(rec, open(f"{a.out}/{ep}/match_record.json", "w"), indent=1)
        print(json.dumps({k: v for k, v in rec.items() if k != "K_src"})); return
    # apparent size -> LK parameters (rig_track's estimator on the matched centroid)
    Xc = rt.triangulate_point(P["ext1"], P["ext2"], x1.mean(0), x2.mean(0))
    depth = {c: float((E[c][:3, :3] @ Xc + E[c][:3, 3])[2]) for c in E}
    sz = {c: min(1.0, K[c][0, 0] / (max(depth[c], 0.05) * 100) / rt.PPCM_RAIL) for c in E}
    prm = {c: rt.scaled_params(sz[c]) for c in E}
    if a.lk_relax:
        for c in E:
            prm[c]["fb"] *= 2; prm[c]["lk_win"] = int(prm[c]["lk_win"] * 1.5) | 1
    rec.update(depth_birth_m=depth, size_scale=sz, lk_relax=bool(a.lk_relax), epi_track_px=a.epi_track_px)
    # tracks (N, T, 2) per camera, identical indexing
    tr = {c: np.full((len(x1), T, 2), np.nan, np.float32) for c in E}
    tr["ext1"][:, b] = x1; tr["ext2"][:, b] = x2
    seed_t = np.full(len(x1), b, np.int32); alive = np.ones(len(x1), bool); n_reseed = 0; n0 = len(x1); last_reseed = b
    lk = {c: dict(winSize=(prm[c]["lk_win"],) * 2, maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)) for c in E}
    if a.tracker == "cotracker":
        # CoTracker3 online (causal, sliding window of 2*step frames) per view, seeded once at b; a pair is observed on a
        # frame when CoTracker flags both views visible. Stand-in for Track-On-R (not available on MN5); same protocol.
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(dev).eval()
        vis_ct = {}
        for c, s in cams:
            col = rt.read_frames(f"{rt.RAW_ROOT}/{ep}/recordings/MP4/{s}.mp4", T, gray=False)[b:T]
            video = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in col]).transpose(0, 3, 1, 2)   # (Tw,3,H,W) uint8
            qry = torch.from_numpy(np.c_[np.zeros(len(x1)), tr[c][:, b]]).float()[None].to(dev)
            step = model.step; Tw = len(video); trk = visb = None
            with torch.no_grad():
                model(video_chunk=torch.from_numpy(video[:1]).float()[None].to(dev), is_first_step=True, queries=qry)
                for i0 in range(0, Tw - step, step):
                    chunk = torch.from_numpy(video[i0:i0 + 2 * step]).float()[None].to(dev)
                    trk, visb = model(video_chunk=chunk)
                if trk is None or trk.shape[1] < Tw:   # short window: one padded call covers it
                    pad = np.concatenate([video, np.repeat(video[-1:], max(0, 2 * step - Tw), 0)])
                    model(video_chunk=torch.from_numpy(pad[:1]).float()[None].to(dev), is_first_step=True, queries=qry)
                    trk, visb = model(video_chunk=torch.from_numpy(pad[:2 * step]).float()[None].to(dev))
            trk = trk[0, :Tw].cpu().numpy(); visb = visb[0, :Tw].cpu().numpy().astype(bool)   # (Tw,N,2), (Tw,N)
            tr[c][:, b:T] = np.transpose(trk, (1, 0, 2)); vis_ct[c] = visb.T
        inimg = {c: (tr[c][..., 0] >= 0) & (tr[c][..., 0] < rt.W) & (tr[c][..., 1] >= 0) & (tr[c][..., 1] < rt.H) for c in E}
        both = np.ones((len(x1), T), bool); both[:, b:T] = vis_ct["ext1"] & vis_ct["ext2"]
        both &= inimg["ext1"] & inimg["ext2"]; both[:, :b] = False
        for c in E:
            tr[c][~both] = np.nan
        rec["tracker"] = "cotracker3_online"
    if a.tracker == "external":
        # tracks computed elsewhere (e.g. trackon_seeds.py: Track-On-R on these seeds), same N and order as x1/x2
        vis_ex = {}
        for c in E:
            f = f"{a.ext_tracks}/{ep}/{c}_f{b:05d}.npz"
            if not os.path.isfile(f): raise rt.Failure(f"external track file missing: {os.path.basename(f)}")
            zt = np.load(f); trk = zt["tracks"]; vb = zt["visibility"]
            if trk.shape[1] != len(x1) or np.abs(zt["queries"][:, 1:] - (x1 if c == "ext1" else x2)).max() > 0.01:
                raise rt.Failure("external tracks do not match the seeds")
            n = min(T, trk.shape[0]); tr[c][:, b:n] = np.transpose(trk[b:n], (1, 0, 2)); vis_ex[c] = np.zeros((len(x1), T), bool); vis_ex[c][:, b:n] = vb[b:n].T
        inimg = {c: (tr[c][..., 0] >= 0) & (tr[c][..., 0] < rt.W) & (tr[c][..., 1] >= 0) & (tr[c][..., 1] < rt.H) for c in E}
        both = vis_ex["ext1"] & vis_ex["ext2"] & inimg["ext1"] & inimg["ext2"]; both[:, :b] = False; both[:, b] = True
        for c in E:
            tr[c][~both] = np.nan
        rec["tracker"] = f"external:{a.ext_tracks}"
    for t in (range(b + 1, T) if a.tracker == "lk" else []):
        idx = np.where(alive)[0]
        if len(idx):
            for c in E:
                p_prev = tr[c][idx, t - 1].reshape(-1, 1, 2)
                p_next, st, _ = cv2.calcOpticalFlowPyrLK(gray[c][t - 1], gray[c][t], p_prev, None, **lk[c])
                p_back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[c][t], gray[c][t - 1], p_next, None, **lk[c])
                fb = np.linalg.norm(p_back - p_prev, axis=2).ravel(); pn = p_next.reshape(-1, 2)
                ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < prm[c]["fb"]) & (pn[:, 0] >= 0) & (pn[:, 0] < rt.W) & (pn[:, 1] >= 0) & (pn[:, 1] < rt.H)
                tr[c][idx[ok], t] = pn[ok]; alive[idx[~ok]] = False   # a pair dies when either view loses it
            for c in E:   # keep the two views consistent: a pair that died in one view is NaN in both from t on
                tr[c][idx[~alive[idx]], t] = np.nan
        if a.reseed and (alive.sum() < max(rt.RAIL["min_ref_pairs"], n0 // 2) or (a.reseed_every and t - last_reseed >= a.reseed_every)) \
                and t - last_reseed >= 3:
            last_reseed = t
            live = {c: tr[c][alive, t] for c in E}
            # region at t: Track-On's visible points at t (image-based); fall back to the live tracks' hull; never the stale birth region
            m = {c: region_at(c, t) for c in E}
            for c in E:
                if m[c] is None and alive.any(): m[c] = region_mask(live[c], a.region_dilate)
            if any(m[c] is None for c in E):
                continue
            for c in E:
                for x, y in live[c]:
                    cv2.circle(m[c], (int(x), int(y)), prm[c]["excl_r"], 0, -1)
            nx1, nx2, nres, _ = do_match(t, m["ext1"], m["ext2"])
            if len(nx1):
                for c, nx in (("ext1", nx1), ("ext2", nx2)):
                    new = np.full((len(nx), T, 2), np.nan, np.float32); new[:, t] = nx; tr[c] = np.concatenate([tr[c], new])
                alive = np.concatenate([alive, np.ones(len(nx1), bool)]); seed_t = np.concatenate([seed_t, np.full(len(nx1), t, np.int32)])
                n0 = max(n0, int(alive.sum())); n_reseed += 1
    # motion gate (GT-free): a static-background pair satisfies the epipolar geometry exactly, so it survives the match gate;
    # it is recognised by NOT moving while the body does. A pair observed >= 10 frames whose maximum displacement from its
    # seed is < 3 px in BOTH views, while the median pair moved > 10 px, is dropped.
    obs = ~np.isnan(tr["ext1"][:, :, 0]) & ~np.isnan(tr["ext2"][:, :, 0])
    disp = {c: np.nanmax(np.linalg.norm(tr[c] - tr[c][np.arange(len(seed_t)), seed_t][:, None], axis=2), axis=1) for c in E}
    disp = np.nan_to_num(np.minimum(disp["ext1"], disp["ext2"]))
    n_static = 0
    if len(disp) and np.median(disp[obs.sum(1) >= 10]) > 10 if (obs.sum(1) >= 10).any() else False:
        static = (obs.sum(1) >= 10) & (disp < 3)
        n_static = int(static.sum())
        for c in E:
            tr[c][static] = np.nan
        obs = ~np.isnan(tr["ext1"][:, :, 0]) & ~np.isnan(tr["ext2"][:, :, 0])
    rec["n_static_dropped"] = n_static
    # per-frame epipolar gate (GT-free): a pair that is still the same physical point satisfies the calibration's epipolar
    # geometry on every frame; a track that slid onto a neighbouring point in one view does not. Frames where the pair's
    # symmetric epipolar residual exceeds --epi_track_px are marked unobserved (NaN in both views).
    n_epi_dropped = 0
    if a.epi_track_px > 0 and obs.any():
        kk, tt = np.nonzero(obs)
        h1 = np.c_[tr["ext1"][kk, tt].astype(np.float64), np.ones(len(kk))]; h2 = np.c_[tr["ext2"][kk, tt].astype(np.float64), np.ones(len(kk))]
        l2 = h1 @ F.T; l1 = h2 @ F
        r = 0.5 * (np.abs((h2 * l2).sum(1)) / np.hypot(l2[:, 0], l2[:, 1]) + np.abs((h1 * l1).sum(1)) / np.hypot(l1[:, 0], l1[:, 1]))
        bad = r > a.epi_track_px; n_epi_dropped = int(bad.sum())
        for c in E:
            tr[c][kk[bad], tt[bad]] = np.nan
        obs = ~np.isnan(tr["ext1"][:, :, 0]) & ~np.isnan(tr["ext2"][:, :, 0])
        rec["epi_track_res_median_px"] = float(np.median(r))
    rec["n_epi_dropped_frames"] = n_epi_dropped
    lens = obs.sum(1)
    rec.update(n_pairs_total=int(len(alive)), reseed_events=n_reseed, pair_len_median=float(np.median(lens)),
               frames_ge3_pairs=int((obs.sum(0)[b:] >= 3).sum()), frames_ge4_pairs=int((obs.sum(0)[b:] >= 4).sum()), window_len=int(T - b),
               elapsed_s=time.time() - t_start)
    os.makedirs(f"{a.out}/{ep}", exist_ok=True)
    for c in E:
        tk = np.transpose(tr[c], (1, 0, 2)).astype(np.float32)   # (T, N, 2)
        np.savez_compressed(f"{a.out}/{ep}/{c}_f{b:05d}.npz", tracks=tk, visibility=~np.isnan(tk[..., 0]),
                            queries=np.c_[seed_t, np.array([tr[c][k, seed_t[k]] for k in range(len(seed_t))])].astype(np.float32),
                            meta=json.dumps(dict(rec, cam=c, serial=str(meta[f"{c}_cam_serial"]), source="rig_match_track")))
    np.savez(f"{a.out}/{ep}/matches.npz", x1=x1, x2=x2, epi_res=res, F=F)
    json.dump(rec, open(f"{a.out}/{ep}/match_record.json", "w"), indent=1)
    if a.viz:
        col = {c: cv2.cvtColor(gray[c][b], cv2.COLOR_GRAY2BGR) for c in E}
        for c in E:
            cnt, _ = cv2.findContours(reg[c], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(col[c], cnt, -1, (255, 200, 0), 2)
        side = np.concatenate([col["ext1"], col["ext2"]], 1)
        for k in range(len(x1)):
            colr = tuple(int(v) for v in np.random.default_rng(k).integers(60, 255, 3))
            p1, p2 = (int(x1[k, 0]), int(x1[k, 1])), (int(x2[k, 0]) + rt.W, int(x2[k, 1]))
            cv2.circle(side, p1, 4, colr, -1); cv2.circle(side, p2, 4, colr, -1); cv2.line(side, p1, p2, colr, 1)
        cv2.putText(side, f"{ep} f{b} {a.matcher}: {len(x1)} pairs (raw {n_raw})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imwrite(f"{a.out}/{ep}/matches_f{b:05d}.jpg", side, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(json.dumps({k: v for k, v in rec.items() if k != "K_src"}))


def main():
    ap = argparse.ArgumentParser()
    for name, kw in ARGS: ap.add_argument(name, **kw)
    a = ap.parse_args()
    try:
        run(a)
    except (rt.Failure, FileNotFoundError, OSError) as ex:
        os.makedirs(f"{a.out}/{a.episode}", exist_ok=True)
        rec = dict(episode=a.episode, matcher=a.matcher, failure_reason=str(ex))
        json.dump(rec, open(f"{a.out}/{a.episode}/match_record.json", "w"), indent=1); print(json.dumps(rec))


if __name__ == "__main__":
    main()
