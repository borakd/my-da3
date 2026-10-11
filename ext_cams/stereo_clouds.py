#!/usr/bin/env python
"""stereo_clouds v2: per-frame cross-view 3D points of the gripper from the two static exterior cameras, with GT-free
phantom evidence (GPU pass, 2026-10-10).

Why v2: epipolar-GUIDED matching (DISK or MASt3R) on this black, textureless gripper picks the best descriptor ON the line,
which on texture-less stretches is a lottery: ~60% of the triangulated points are "phantoms" (two different surface points on
one epipolar plane; off the surface by > 1 cm) and phantoms track with a rotation gain near zero. v2 therefore
  (1) matches with MASt3R's OFFICIAL reciprocal nearest neighbours (mast3r.fast_nn.fast_reciprocal_NNs, one-cycle
      reciprocity over the whole crop, started from the region pixels) and, optionally, the guided search (--matcher), and
  (2) records MASt3R's own two-view 3D consistency per match: pts3d of the view-1 pixel and pts3d_in_other_view of the view-2
      pixel are two independent predictions of the SAME point (both in view-1's frame, MASt3R metric units); their distance
      m3d is small for a true match and ~ the surface separation for a phantom. Also stored: MASt3R depth at both pixels,
      the calibrated triangulated depth z1 (view 1), the pointmap confidences, so a downstream GT-free filter can gate on
      s*m3d and |z1 - s*zm1| with the per-frame scale s = median(z1 / zm1).
Region at t: t = b -> RobotSeg birth mask; t > b -> dilated hull of the Track-On v2 points visible at t (image-based, causal).
Crops (region bbox + margin) are resized so the long side is --side px. GT is never read.

Output DIR/<ep>.npz: frame (N,), x1, x2 (N,2) full-res px, X (N,3) world metres, sim (N,), epi (N,) px, conf (N,2) desc_conf,
src (N,) 1 = reciprocal, 2 = guided, 3 = both, m3d (N,), zm1, zm2 (N,) MASt3R units, z1 (N,) m, pconf (N,2), plus meta json
(birth, T, per-frame region source, counts, scale s, timing). Run in the mast3r env from ~/mast3r:
    python stereo_clouds.py --episodes LIST --out DIR [--shard i --nshards n] [--matcher both|recip|guided]"""
import argparse, csv, json, os, sys, time
import numpy as np, cv2, torch, pandas as pd
sys.path.insert(0, os.path.expanduser("~/mast3r")); sys.path.insert(0, os.path.expanduser("~/using_ext_cams/ext_cams"))
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.inference import inference
from dust3r.utils.image import ImgNorm
import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra, rig_match_track as rmt

DEPTH = (0.15, 2.5)
CKPT = os.path.expanduser("~/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")


def crop_resize(img, box, side):
    x0, y0, x1, y1 = box; crop = img[y0:y1, x0:x1]; s = side / max(crop.shape[:2])
    cw, ch = max(16, int(round(crop.shape[1] * s / 16)) * 16), max(16, int(round(crop.shape[0] * s / 16)) * 16)
    return cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA), (x0, y0, cw / crop.shape[1], ch / crop.shape[0])


def view(img_rgb, idx):
    return dict(img=ImgNorm(img_rgb)[None], true_shape=np.int32([img_rgb.shape[:2]]), idx=idx, instance=str(idx))


def sample_map(M, p):
    """nearest-pixel values of a (H,W,C) numpy map at crop coords p (n,2) float."""
    H, W = M.shape[:2]; xi = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1); yi = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
    return M[yi, xi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--matcher", default="both", choices=["both", "recip", "guided"])
    ap.add_argument("--side", type=int, default=512); ap.add_argument("--epi", type=float, default=2.5); ap.add_argument("--sim", type=float, default=0.5)
    ap.add_argument("--recip_sub", type=int, default=2, help="stride of the reciprocal-NN start pixels inside the view-1 region")
    ap.add_argument("--recip_tol", type=float, default=2.0, help="pixel tolerance of the reciprocity cycle (crop px)")
    ap.add_argument("--stride", type=int, default=2, help="guided search: stride of the view-1 region pixels")
    ap.add_argument("--margin", type=int, default=24); ap.add_argument("--hull_dilate", type=int, default=12)
    ap.add_argument("--frame_step", type=int, default=1, help="process every k-th frame after the birth frame (birth always)")
    ap.add_argument("--max_frames", type=int, default=0, help="debug: stop after this many frames per scene")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    eps = [l.strip() for l in open(a.episodes) if l.strip()]; eps = [e for i, e in enumerate(eps) if i % a.nshards == a.shard]
    dev = "cuda"; model = AsymmetricMASt3R.from_pretrained(CKPT).to(dev).eval()
    ev = {r["episode"]: (int(r["entry_frame"]), int(r["exit_frame"])) for r in csv.DictReader(open(ra.EVENTS)) if r["event"] == "0"}
    items = pd.read_csv(f"{ra.MASKS}/items_v2.csv")
    t_start = time.time(); n_done = 0; n_frames = 0; tim = dict(mast3r=0.0, recip=0.0, guided=0.0, io=0.0)
    for ep in eps:
        of = f"{a.out}/{ep}.npz"
        if os.path.isfile(of) or ep not in ev: continue
        b, _ = ev[ep]
        try:
            cams, K, E, P, ksrc = rte.calib(ep, rte.POSITIONS)
        except Exception as ex:  # noqa: BLE001
            json.dump(dict(episode=ep, failure=f"calib: {ex}"), open(f"{a.out}/{ep}.fail.json", "w")); continue
        R1, t1, R2, t2 = E["ext1"][:3, :3], E["ext1"][:3, 3], E["ext2"][:3, :3], E["ext2"][:3, 3]; Rr = R2 @ R1.T; trel = t2 - Rr @ t1
        F = np.linalg.inv(K["ext2"]).T @ rt.skew(trel) @ Rr @ np.linalg.inv(K["ext1"])
        n_store = rt.count_files(f"{rt.STORE_ROOT}/{ep}/dense/cam", ".npz")
        masks = {}
        for c in ("ext1", "ext2"):
            r_ = items[(items.episode == ep) & (items.frame == b) & (items.cam == c)]
            m = cv2.imread(f"{ra.MASKS}/{r_.mask_relpath.iloc[0]}", 0) if len(r_) else None
            masks[c] = ((cv2.resize(m, (rt.W, rt.H), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) * 255) if m is not None else None
        if any(masks[c] is None for c in masks):
            json.dump(dict(episode=ep, failure="no birth mask"), open(f"{a.out}/{ep}.fail.json", "w")); continue
        try:
            qz = {c: np.load(f"{ra.V2}/{ep}/{c}_f{b:05d}.npz") for c in ("ext1", "ext2")}
        except Exception as ex:  # noqa: BLE001
            json.dump(dict(episode=ep, failure=f"v2 tracks: {ex}"), open(f"{a.out}/{ep}.fail.json", "w")); continue
        caps = {c: cv2.VideoCapture(f"{rt.RAW_ROOT}/{ep}/recordings/MP4/{s}.mp4") for c, s in cams}
        keys = ("frame", "x1", "x2", "X", "sim", "epi", "conf", "src", "m3d", "zm1", "zm2", "z1", "pconf")
        out = {k: [] for k in keys}; per_frame = []
        t = 0; T_eff = n_store
        while t < n_store:
            t0 = time.time(); fr = {}
            for c in caps:
                ok, img = caps[c].read()
                if not ok: fr = None; break
                if img.shape[1] != rt.W or img.shape[0] != rt.H: img = cv2.resize(img, (rt.W, rt.H), interpolation=cv2.INTER_AREA)
                fr[c] = img
            tim["io"] += time.time() - t0
            if fr is None: T_eff = t; break
            if t < b or (t > b and (t - b) % a.frame_step): t += 1; continue
            if a.max_frames and t - b >= a.max_frames: break
            reg = {}
            for c in ("ext1", "ext2"):
                if t == b: reg[c] = masks[c]; continue
                tr = qz[c]["tracks"]; p = tr[t] if t < len(tr) else None
                if p is None: reg[c] = None; continue
                ok = qz[c]["visibility"][t] & ~np.isnan(p[:, 0]) & (p[:, 0] >= 0) & (p[:, 0] < rt.W) & (p[:, 1] >= 0) & (p[:, 1] < rt.H)
                reg[c] = rmt.region_mask(p[ok], a.hull_dilate) if ok.sum() >= 3 else None
            if any(reg[c] is None for c in reg):
                per_frame.append(dict(t=t, region="none", n=0)); t += 1; continue
            boxes, imgs, geos = {}, {}, {}
            for c in ("ext1", "ext2"):
                ys, xs = np.nonzero(reg[c]); box = (max(xs.min() - a.margin, 0), max(ys.min() - a.margin, 0), min(xs.max() + a.margin + 1, rt.W), min(ys.max() + a.margin + 1, rt.H))
                if box[2] - box[0] < 16 or box[3] - box[1] < 16: boxes = None; break
                imgs[c], geos[c] = crop_resize(cv2.cvtColor(fr[c], cv2.COLOR_BGR2RGB), box, a.side); boxes[c] = box
            if boxes is None:
                per_frame.append(dict(t=t, region="tiny", n=0)); t += 1; continue
            t0 = time.time()
            with torch.no_grad():
                o = inference([(view(imgs["ext1"], 0), view(imgs["ext2"], 1))], model, dev, batch_size=1, verbose=False)
            d1 = torch.nn.functional.normalize(o["pred1"]["desc"].squeeze(0).to(dev), dim=-1); d2 = torch.nn.functional.normalize(o["pred2"]["desc"].squeeze(0).to(dev), dim=-1)
            c1 = o["pred1"]["desc_conf"].squeeze(0).cpu().numpy(); c2 = o["pred2"]["desc_conf"].squeeze(0).cpu().numpy()
            pm1 = o["pred1"]["pts3d"].squeeze(0).cpu().numpy(); pm2 = o["pred2"]["pts3d_in_other_view"].squeeze(0).cpu().numpy()
            pc1 = o["pred1"]["conf"].squeeze(0).cpu().numpy(); pc2 = o["pred2"]["conf"].squeeze(0).cpu().numpy()
            tim["mast3r"] += time.time() - t0
            h1, w1 = d1.shape[:2]; h2, w2 = d2.shape[:2]
            tofull = lambda m, c: m / np.array([geos[c][2], geos[c][3]]) + np.array([geos[c][0], geos[c][1]])
            inreg = lambda pts, c: reg[c][np.clip(np.round(pts[:, 1]).astype(int), 0, rt.H - 1), np.clip(np.round(pts[:, 0]).astype(int), 0, rt.W - 1)] > 0
            yy, xx = np.mgrid[0:h1, 0:w1]; pall = np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float64); inr1 = inreg(tofull(pall, "ext1"), "ext1").reshape(h1, w1)
            cand = []    # list of (p1c (n,2), p2c (n,2), sim (n,), src)
            if a.matcher in ("both", "recip"):
                t0 = time.time()
                ys_, xs_ = np.nonzero(inr1[::a.recip_sub, ::a.recip_sub]); xs_, ys_ = xs_ * a.recip_sub, ys_ * a.recip_sub
                if len(xs_):
                    m0, m1 = fast_reciprocal_NNs(d1, d2, subsample_or_initxy1=(xs_, ys_), pixel_tol=a.recip_tol, device=dev, dist="dot", block_size=2 ** 13)
                    if len(m0):
                        f1 = d1[m0[:, 1], m0[:, 0]]; f2 = d2[m1[:, 1], m1[:, 0]]; sim = (f1 * f2).sum(1).cpu().numpy()
                        cand.append((m0.astype(np.float64), m1.astype(np.float64), sim, 1))
                tim["recip"] += time.time() - t0
            if a.matcher in ("both", "guided"):
                t0 = time.time()
                yy_, xx_ = np.mgrid[0:h1:a.stride, 0:w1:a.stride]; p1c = np.stack([xx_.ravel(), yy_.ravel()], 1).astype(np.float64); p1f = tofull(p1c, "ext1")
                s1 = inreg(p1f, "ext1"); p1c, p1f = p1c[s1], p1f[s1]
                yy2, xx2 = np.mgrid[0:h2, 0:w2]; p2c = np.stack([xx2.ravel(), yy2.ravel()], 1).astype(np.float64); p2f = tofull(p2c, "ext2")
                s2 = inreg(p2f, "ext2"); p2c, p2f = p2c[s2], p2f[s2]
                if len(p1c) and len(p2c):
                    l2 = np.c_[p1f, np.ones(len(p1f))] @ F.T; nrm = np.hypot(l2[:, 0], l2[:, 1])
                    h2f = torch.from_numpy(np.c_[p2f, np.ones(len(p2f))]).to(dev).float(); l2t = torch.from_numpy(l2 / nrm[:, None]).to(dev).float()
                    f1 = d1[p1c[:, 1].astype(int), p1c[:, 0].astype(int)]; f2 = d2[p2c[:, 1].astype(int), p2c[:, 0].astype(int)]
                    best_j = np.full(len(p1c), -1); best_s = np.full(len(p1c), -np.inf, np.float32)
                    col_best_i = torch.full((len(p2c),), -1, device=dev, dtype=torch.long); col_best_s = torch.full((len(p2c),), -float("inf"), device=dev)
                    CH = max(1, int(2e8 // max(len(p2c), 1)))
                    for i0 in range(0, len(p1c), CH):
                        dist = (l2t[i0:i0 + CH] @ h2f.T).abs()
                        S = f1[i0:i0 + CH] @ f2.T; S = torch.where(dist <= a.epi, S, torch.full_like(S, -float("inf")))
                        v, j = S.max(1); best_j[i0:i0 + CH] = j.cpu().numpy(); best_s[i0:i0 + CH] = v.cpu().numpy()
                        vc, ic = S.max(0); upd = vc > col_best_s; col_best_s = torch.where(upd, vc, col_best_s); col_best_i = torch.where(upd, ic + i0, col_best_i)
                    col_best_i = col_best_i.cpu().numpy()
                    ok = np.isfinite(best_s) & (best_s >= a.sim) & (best_j >= 0)
                    ok &= col_best_i[np.clip(best_j, 0, len(p2c) - 1)] == np.arange(len(p1c))
                    if ok.any(): cand.append((p1c[ok], p2c[best_j[ok]], best_s[ok], 2))
                tim["guided"] += time.time() - t0
            n = 0; n_recip = 0
            if cand:
                p1c = np.vstack([c_[0] for c_ in cand]); p2c = np.vstack([c_[1] for c_ in cand]); sim = np.concatenate([c_[2] for c_ in cand]); src = np.concatenate([np.full(len(c_[0]), c_[3], np.int8) for c_ in cand])
                key1 = np.round(p1c).astype(int); order = np.lexsort((src, key1[:, 1], key1[:, 0]))
                p1c, p2c, sim, src = p1c[order], p2c[order], sim[order], src[order]
                keep = np.ones(len(p1c), bool)
                for i in range(1, len(p1c)):     # same view-1 pixel from both matchers, view-2 within 1 crop px -> one match, src 3
                    if keep[i - 1] and (np.round(p1c[i]) == np.round(p1c[i - 1])).all() and np.abs(p2c[i] - p2c[i - 1]).max() <= 1.0:
                        src[i - 1] = 3; keep[i] = False
                p1c, p2c, sim, src = p1c[keep], p2c[keep], sim[keep], src[keep]
                x1 = tofull(p1c, "ext1"); x2 = tofull(p2c, "ext2")
                ok = inreg(x1, "ext1") & inreg(x2, "ext2") & (sim >= a.sim)
                res = rmt.epi_dist(F, x1, x2); ok &= res <= a.epi
                if ok.any():
                    x1, x2, sim, src, res, p1c, p2c = x1[ok], x2[ok], sim[ok], src[ok], res[ok], p1c[ok], p2c[ok]
                    Xh = cv2.triangulatePoints(P["ext1"], P["ext2"], x1.T, x2.T); X = (Xh[:3] / Xh[3]).T
                    z1 = (E["ext1"][:3, :3] @ X.T + E["ext1"][:3, 3:]).T[:, 2]; z2 = (E["ext2"][:3, :3] @ X.T + E["ext2"][:3, 3:]).T[:, 2]
                    okz = (z1 > DEPTH[0]) & (z1 < DEPTH[1]) & (z2 > DEPTH[0]) & (z2 < DEPTH[1])
                    n = int(okz.sum())
                    if n:
                        q1 = sample_map(pm1, p1c[okz]); q2 = sample_map(pm2, p2c[okz]); sk = src[okz]; n_recip = int(((sk == 1) | (sk == 3)).sum())
                        out["frame"].append(np.full(n, t, np.int32)); out["x1"].append(x1[okz].astype(np.float32)); out["x2"].append(x2[okz].astype(np.float32))
                        out["X"].append(X[okz].astype(np.float32)); out["sim"].append(sim[okz].astype(np.float32)); out["epi"].append(res[okz].astype(np.float32))
                        out["conf"].append(np.c_[sample_map(c1[..., None], p1c[okz])[:, 0], sample_map(c2[..., None], p2c[okz])[:, 0]].astype(np.float32))
                        out["src"].append(sk); out["m3d"].append(np.linalg.norm(q1 - q2, axis=1).astype(np.float32))
                        out["zm1"].append(q1[:, 2].astype(np.float32)); out["zm2"].append(q2[:, 2].astype(np.float32)); out["z1"].append(z1[okz].astype(np.float32))
                        out["pconf"].append(np.c_[sample_map(pc1[..., None], p1c[okz])[:, 0], sample_map(pc2[..., None], p2c[okz])[:, 0]].astype(np.float32))
            s_ = float(np.median(out["z1"][-1] / np.maximum(out["zm1"][-1], 1e-6))) if (n >= 5) else float("nan")
            per_frame.append(dict(t=t, region="mask" if t == b else "hull", n=n, n_recip=n_recip, scale=s_))
            n_frames += 1; t += 1
        for c in caps: caps[c].release()
        shp = {"x1": (2,), "x2": (2,), "conf": (2,), "pconf": (2,), "X": (3,)}
        cat = {k: (np.concatenate(v) if v else np.zeros((0,) + shp.get(k, ()), np.int8 if k == "src" else np.float32)) for k, v in out.items()}
        np.savez_compressed(of + ".tmp.npz", **cat, meta=json.dumps(dict(episode=ep, birth=b, n_store=n_store, T=T_eff, K_src=ksrc, args=vars(a), per_frame=per_frame)))
        os.replace(of + ".tmp.npz", of); n_done += 1
        if n_done % 5 == 0:
            el = time.time() - t_start; print(f"{n_done} scenes, {n_frames} frames, {n_frames/el:.1f} fr/s; time split " + ", ".join(f"{k} {v/max(n_frames,1)*1000:.0f} ms" for k, v in tim.items()), flush=True)
    print(f"done shard {a.shard}: {n_done} scenes, {n_frames} frames in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
