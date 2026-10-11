#!/usr/bin/env python
"""DEV probe: MASt3R dense descriptors for cross-view matching of the gripper at the birth frame (and at later frames), compared
with epipolar-guided DISK. Runs in the mast3r env from ~/mast3r. For every scene and frame t in FRAMES (b + offsets):
  crops = RobotSeg birth-mask bbox (t = b) or the Track-On v2 visible-point hull bbox (t > b), + margin, each resized so the
  long side is SIDE px (MASt3R works at 512); MASt3R on the crop pair -> desc (H,W,24) per view;
  (i) fast_reciprocal_NNs (official) on the crops, mapped back to full-res, gated by the calibration's epipolar residual;
  (ii) epipolar-GUIDED NN on the dense descriptors (for each view-1 pixel inside the mask/hull the best view-2 pixel within
       EPI px of its epipolar line; mutual), gated the same way.
GT CHECK (diagnostic only): triangulated points vs the FK label cloud at t (positions_v1): nearest-label distance (mm);
a gripper-surface point is within ~15 mm of a label point; background points are far.
Usage: mast3r_probe.py EPLIST OUTCSV [--side 512] [--epi 2.5] [--offsets 0,10,30,60]"""
import argparse, csv, json, os, sys, time
import numpy as np, cv2, torch, pandas as pd
sys.path.insert(0, os.path.expanduser("~/mast3r")); sys.path.insert(0, os.path.expanduser("~/using_ext_cams/ext_cams"))
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.inference import inference
from dust3r.utils.image import ImgNorm
import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra, rig_match_track as rmt

ap = argparse.ArgumentParser(); ap.add_argument("eps"); ap.add_argument("out"); ap.add_argument("--side", type=int, default=512)
ap.add_argument("--epi", type=float, default=2.5); ap.add_argument("--offsets", default="0,10,30,60"); ap.add_argument("--n", type=int, default=0)
a = ap.parse_args(); offs = [int(v) for v in a.offsets.split(",")]
eps = [l.strip() for l in open(a.eps) if l.strip()]; eps = eps[:a.n] if a.n else eps
dev = "cuda"; model = AsymmetricMASt3R.from_pretrained(os.path.expanduser("~/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")).to(dev).eval()
ev = {r["episode"]: (int(r["entry_frame"]), int(r["exit_frame"])) for r in csv.DictReader(open(ra.EVENTS)) if r["event"] == "0"}
it = pd.read_csv(f"{ra.MASKS}/items_v2.csv")

def label_world(z, s, t):
    uv = z[f"{s}_uv"][t].astype(np.float64); d = z[f"{s}_z"][t].astype(np.float64); K = z[f"{s}_K"]; W2C = z[f"{s}_W2C"]
    ok = z[f"{s}_front"][t]
    Xc = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0] * d, (uv[:, 1] - K[1, 2]) / K[1, 1] * d, d]
    return ((Xc - W2C[:3, 3]) @ W2C[:3, :3])[ok]

def crop_resize(g, box, side):
    x0, y0, x1, y1 = box; crop = g[y0:y1, x0:x1]; s = side / max(crop.shape)
    cw, ch = max(16, int(round(crop.shape[1] * s / 16)) * 16), max(16, int(round(crop.shape[0] * s / 16)) * 16)
    img = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_CUBIC)
    return img, (x0, y0, cw / crop.shape[1], ch / crop.shape[0])

def view(img_rgb, idx):
    t_ = ImgNorm(img_rgb)[None]
    return dict(img=t_, true_shape=np.int32([img_rgb.shape[:2]]), idx=idx, instance=str(idx))

def epi_lines(F, pts):
    h = np.c_[pts, np.ones(len(pts))]; return h @ F.T

rows = []; t0 = time.time()
for ei, ep in enumerate(eps):
    if ep not in ev: continue
    b, e = ev[ep]; cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS)
    R1, t1, R2, t2 = E["ext1"][:3, :3], E["ext1"][:3, 3], E["ext2"][:3, :3], E["ext2"][:3, 3]; Rr = R2 @ R1.T; trel = t2 - Rr @ t1
    F = np.linalg.inv(K["ext2"]).T @ rt.skew(trel) @ Rr @ np.linalg.inv(K["ext1"])
    T = min(e, rt.count_files(f"{rt.STORE_ROOT}/{ep}/dense/cam", ".npz"))
    col = {c: rt.read_frames(f"{rt.RAW_ROOT}/{ep}/recordings/MP4/{s}.mp4", T, gray=False) for c, s in cams}; T = min(T, min(len(v) for v in col.values()))
    lf = f"{rte.POSITIONS}/{ep}_gripper.npz"; z = np.load(lf) if os.path.isfile(lf) else None; s0 = str(z["serials"][0]) if z is not None else None
    qz = {c: np.load(f"{ra.V2}/{ep}/{c}_f{b:05d}.npz") for c in ("ext1", "ext2")}
    masks = {}
    for c in ("ext1", "ext2"):
        r_ = it[(it.episode == ep) & (it.frame == b) & (it.cam == c)]
        m = cv2.imread(f"{ra.MASKS}/{r_.mask_relpath.iloc[0]}", 0) if len(r_) else None
        masks[c] = (cv2.resize(m, (rt.W, rt.H), interpolation=cv2.INTER_NEAREST) > 0) if m is not None else None
    for off in offs:
        t = b + off
        if t >= T or any(masks[c] is None for c in masks): continue
        reg = {}
        for c in ("ext1", "ext2"):
            if off == 0: reg[c] = masks[c].astype(np.uint8) * 255
            else:
                p = qz[c]["tracks"][t] if t < len(qz[c]["tracks"]) else None
                ok = (qz[c]["visibility"][t] & ~np.isnan(p[:, 0]) & (p[:, 0] >= 0) & (p[:, 0] < rt.W) & (p[:, 1] >= 0) & (p[:, 1] < rt.H)) if p is not None else None
                reg[c] = rmt.region_mask(p[ok], 8) if (ok is not None and ok.sum() >= 3) else None
        if any(reg[c] is None for c in reg): continue
        boxes, imgs, geos = {}, {}, {}
        for c in ("ext1", "ext2"):
            ys, xs = np.nonzero(reg[c]); box = (max(xs.min() - 24, 0), max(ys.min() - 24, 0), min(xs.max() + 25, rt.W), min(ys.max() + 25, rt.H))
            img, geo = crop_resize(cv2.cvtColor(col[c][t], cv2.COLOR_BGR2RGB), box, a.side); imgs[c], geos[c], boxes[c] = img, geo, box
        with torch.no_grad():
            out = inference([(view(imgs["ext1"], 0), view(imgs["ext2"], 1))], model, dev, batch_size=1, verbose=False)
        d1 = out["pred1"]["desc"].squeeze(0).detach(); d2 = out["pred2"]["desc"].squeeze(0).detach()
        c1 = out["pred1"]["desc_conf"].squeeze(0).detach().cpu().numpy(); c2 = out["pred2"]["desc_conf"].squeeze(0).detach().cpu().numpy()
        h1, w1 = d1.shape[:2]; h2, w2 = d2.shape[:2]
        tofull = lambda m, c: m / np.array([geos[c][2], geos[c][3]]) + np.array([geos[c][0], geos[c][1]])
        inreg = lambda pts, c: reg[c][np.clip(np.round(pts[:, 1]).astype(int), 0, rt.H - 1), np.clip(np.round(pts[:, 0]).astype(int), 0, rt.W - 1)] > 0
        lab = label_world(z, s0, t) if z is not None else None
        def record(name, x1, x2):
            if len(x1) == 0: rows.append(dict(ep=ep, off=off, method=name, n_raw=0, n_gated=0)); return
            res = rmt.epi_dist(F, x1, x2); keep = (res < a.epi) & inreg(x1, "ext1") & inreg(x2, "ext2")
            n_raw = len(x1); x1, x2, res = x1[keep], x2[keep], res[keep]
            if len(x1) == 0: rows.append(dict(ep=ep, off=off, method=name, n_raw=n_raw, n_gated=0)); return
            Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], x1.T, x2.T); X = (Xh[:3] / Xh[3]).T
            zc = [(E[c][:3, :3] @ X.T + E[c][:3, 3:]).T[:, 2] for c in ("ext1", "ext2")]; okz = (zc[0] > 0.15) & (zc[1] > 0.15) & (zc[0] < 2.5) & (zc[1] < 2.5)
            X = X[okz]; nl = np.linalg.norm(X[:, None] - lab[None], axis=2).min(1) * 1000 if (lab is not None and len(X)) else np.full(len(X), np.nan)
            rows.append(dict(ep=ep, off=off, method=name, n_raw=n_raw, n_gated=int(okz.sum()), epi_med=float(np.median(res)), lab_mm_med=float(np.nanmedian(nl)) if len(nl) else np.nan,
                             frac_lab15=float(np.mean(nl < 15)) if len(nl) else np.nan, frac_lab30=float(np.mean(nl < 30)) if len(nl) else np.nan))
        # (i) official reciprocal NN on the crops
        m0, m1 = fast_reciprocal_NNs(d1, d2, subsample_or_initxy1=2, device=dev, dist="dot", block_size=2 ** 13)
        record("mast3r_recipNN", tofull(m0.astype(np.float64), "ext1"), tofull(m1.astype(np.float64), "ext2"))
        # (ii) epipolar-guided NN on dense descriptors: view-1 pixels inside the region (stride 2) vs view-2 pixels within EPI px of the line
        D1 = torch.nn.functional.normalize(d1, dim=-1); D2 = torch.nn.functional.normalize(d2, dim=-1)
        yy, xx = np.mgrid[0:h1:2, 0:w1:2]; p1c = np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float64); p1f = tofull(p1c, "ext1"); sel = inreg(p1f, "ext1"); p1c, p1f = p1c[sel], p1f[sel]
        yy2, xx2 = np.mgrid[0:h2, 0:w2]; p2c = np.stack([xx2.ravel(), yy2.ravel()], 1).astype(np.float64); p2f = tofull(p2c, "ext2"); sel2 = inreg(p2f, "ext2"); p2c, p2f = p2c[sel2], p2f[sel2]
        if len(p1c) and len(p2c):
            l2 = epi_lines(F, p1f); h2f = np.c_[p2f, np.ones(len(p2f))]
            dist = np.abs(l2 @ h2f.T) / np.hypot(l2[:, 0], l2[:, 1])[:, None]            # (n1, n2) full-res px
            f1 = D1[p1c[:, 1].astype(int), p1c[:, 0].astype(int)]; f2 = D2[p2c[:, 1].astype(int), p2c[:, 0].astype(int)]
            S = (f1 @ f2.T).cpu().numpy(); S[dist > a.epi] = -np.inf
            j = S.argmax(1); ok = np.isfinite(S[np.arange(len(j)), j])
            # mutual: for the chosen view-2 pixel, is this view-1 pixel its best among the gated view-1 pixels?
            Sc = S.copy(); i_best = np.where(np.isfinite(Sc).any(0), Sc.argmax(0), -1)
            mut = ok & (i_best[j] == np.arange(len(j)))
            x1g, x2g = p1f[mut], p2f[j[mut]]
            # one-to-one and >= 3 px apart (rmt.gate's rule) is enforced by mutuality + stride; keep sim >= 0.5
            sim = S[np.arange(len(j)), j][mut]; keep = sim >= 0.5
            record("mast3r_guided", x1g[keep], x2g[keep])
        # DISK guided (existing)
        g1, g2 = cv2.cvtColor(col["ext1"][t], cv2.COLOR_BGR2GRAY), cv2.cvtColor(col["ext2"][t], cv2.COLOR_BGR2GRAY)
        x1, x2, conf = rmt.match_disk_epi(g1, g2, reg["ext1"], reg["ext2"], F, 640, a.epi)
        selg, _ = rmt.gate(x1, x2, conf, reg["ext1"], reg["ext2"], F, P, E, a.epi, 0.3)
        record("disk_guided", x1[selg], x2[selg])
    if (ei + 1) % 5 == 0: print(f"{ei+1}/{len(eps)} {time.time()-t0:.0f}s", flush=True)
d = pd.DataFrame(rows); d.to_csv(a.out, index=False)
print(d.groupby(["off", "method"]).agg(scenes=("ep", "size"), n_gated_med=("n_gated", "median"), n_gated_p25=("n_gated", lambda s: s.quantile(.25)), epi=("epi_med", "median"),
                                     lab_mm=("lab_mm_med", "median"), frac15=("frac_lab15", "median"), frac30=("frac_lab30", "median")).round(2).to_string())
