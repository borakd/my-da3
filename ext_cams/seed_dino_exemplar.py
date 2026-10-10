#!/usr/bin/env python
"""seed_dino_exemplar (IDEA 3): gripper seed masks on the two STATIC exterior cameras of one DROID episode by DINOv2
EXEMPLAR RE-IDENTIFICATION against a bank of gripper crops taken from the shelf episode, verified across the two views.

METHOD (per frame t, both cameras in lockstep; pixel work at 640x360, geometry and outputs at native 1280x720):
  1. motion: per camera the seed_verified_motion.py background model (OpenCV MOG2 with the dark-achromatic shadow
     exception) and its candidate extractor (fg components with motion, moving fragments, dark-achromatic components,
     distal caps) -- reused unchanged by import; each candidate gives one square WINDOW = its tight bbox + 15% margin.
  2. search: while a track is alive (a verified pick within the last coast_frames), 27 extra windows per camera around
     the previous pick in that camera: shifts {-1/2, 0, +1/2} x side in x and y, scales {0.8, 1, 1.25} (causal: the
     previous pick only).
  3. re-identification: every window is cropped from the native frame, resized to 224x224 and embedded with DINOv2
     ViT-B/14 (descriptor = [L2(CLS), L2(mean patch token)] / sqrt 2, so cosine = mean of the two cosines); its score is
     the max cosine to the exemplar bank; windows with score > sim_thr (from the bank's own leave-one-view-out
     distribution, see build_bank) are kept, the top_k per camera go to verification.
  4. cross-view verification (declared calibration only): DLT triangulation of the two window centres, accepted only
     if the rays meet (per-view reprojection residual < ray_tol_m x f / depth, 0.04 m as in seed_verified_motion),
     |X| < reach (0.9 m) and X_z > zmin (-0.15 m), the point is >= min_cam_depth in front of both cameras and the
     window's object size (side / (1 + 2 margin) x depth / f) lies in [size_min, size_max] m.
  5. selection: pair score = s1 + s2; while a track is alive the verified pairs within jump_m x (frames since the last
     pick) of the previous 3D point are preferred (consistent set), else the best verified pair starts a new track.
  6. emission: mask = MOG2 foreground pixels inside the winning window (upsampled nearest to 1280x720); if that has
     fewer than min_mask_px pixels (a gripper standing still long enough to be absorbed by the background model) the
     dark-achromatic pixels inside the window (opened by dark_open_px) are used; still fewer -> empty mask. No SAM3
     (the GPU budget goes to the embedding; stated in seed_info.json). Frames without a verified pick get an EMPTY mask;
     there is no hold and no propagation.

CAUSALITY: frame t uses frames <= t only (MOG2 online, previous pick only); emitted masks / points are final.
GT-FREE: the store is opened only to count frames; kinematic GT is never read. No DROID-trained component (DINOv2 is a
generic pretrained model; the bank is a declared per-rig constant, below).
DECLARED CONSTANTS (written to seed_info.json as such): (a) the exemplar bank: 8 + 8 crops of the shelf episode
RAIL+80edfcb1+2023-07-14-14h-28m-45s (both exterior views, tight bounding boxes of the hand-prompted SAM3 shelf masks,
frames listed in bank/bank.json) and the similarity threshold derived from it; (b) PointWorld optimized_extrinsics +
factory intrinsics (as every seeder in this study).
COST: reported in seed_info.json (ms per frame for motion/candidates, embedding, verification) and in the study notes;
the motion/candidate stage alone is the ~150-240 ms/frame of seed_verified_motion (single CPU thread), so this idea is
OVER the 10-20 fps budget on CPU; the embedding adds ~20-60 ms/frame on one H100.

    # bank (login node, CPU is enough: 16 crops):
    XFORMERS_DISABLED=1 OMP_NUM_THREADS=1 TORCH_HOME=/gpfs/scratch/etur59/koc821022/torch_hub \
        python seed_dino_exemplar.py --build_bank [--bank_dir DIR]
    # episode (GPU, see seed_dino_exemplar.sbatch):
    python seed_dino_exemplar.py --episode EP --out_root .../ideas/dino_exemplar [--png_frames 60,200,400]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seed_verified_motion import (ViewMotion, extract_candidates, load_calib, triangulate, project, open_video, read_pair,  # noqa: E402
                                  NATIVE_W, NATIVE_H, RAW_ROOT, STORE_ROOT, CAMS)

METHOD = "dino_exemplar"
IDEAS_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas"
OUT_ROOT = f"{IDEAS_ROOT}/{METHOD}"
BANK_DIR = f"{OUT_ROOT}/bank"
SHELF_EP = "RAIL+80edfcb1+2023-07-14-14h-28m-45s"
SHELF_MASKS = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/gripper_sam3"
TORCH_HUB_REPO = "/gpfs/scratch/etur59/koc821022/torch_hub/hub/facebookresearch_dinov2_main"
DINO_ARCH = "dinov2_vitb14"
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
CROP_PX = 224


# ----------------------------------------------------------------------------------------------- DINOv2 embedder
class DinoEmbedder:
    """DINOv2 ViT-B/14 window descriptor: [L2(CLS), L2(mean patch)] / sqrt(2) (unit norm)."""

    def __init__(self, device=None, arch=DINO_ARCH):
        import torch
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        os.environ.setdefault("TORCH_HOME", "/gpfs/scratch/etur59/koc821022/torch_hub")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
        self.model = torch.hub.load(TORCH_HUB_REPO, arch, source="local", pretrained=True).eval().to(self.device)
        self.half = self.device.startswith("cuda")
        if self.half:
            self.model = self.model.half()
        self.mean = torch.tensor(IMNET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMNET_STD, device=self.device).view(1, 3, 1, 1)
        self.arch = arch
        self.dim = 2 * self.model.embed_dim

    def embed(self, crops_bgr):
        """crops_bgr: list of (224,224,3) uint8 BGR -> (N, 2*D) float32 unit vectors."""
        if not crops_bgr:
            return np.zeros((0, self.dim), np.float32)
        torch = self.torch
        x = np.stack(crops_bgr)[..., ::-1].astype(np.float32) / 255.0                  # RGB
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).to(self.device)
        x = (x - self.mean) / self.std
        if self.half:
            x = x.half()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 128):
                f = self.model.forward_features(x[i:i + 128])
                cls = torch.nn.functional.normalize(f["x_norm_clstoken"].float(), dim=1)
                mp = torch.nn.functional.normalize(f["x_norm_patchtokens"].float().mean(1), dim=1)
                out.append(torch.cat([cls, mp], 1) / np.sqrt(2.0))
        return torch.cat(out).cpu().numpy().astype(np.float32)


def crop_window(img, win, size=CROP_PX):
    """Square window (cx, cy, side) in native px -> (size, size, 3) crop; outside-image parts padded with the image mean."""
    cx, cy, side = win
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    H, W = img.shape[:2]
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        canvas = np.empty((y1 - y0, x1 - x0, 3), np.uint8)
        canvas[:] = img[sy0:sy1, sx0:sx1].reshape(-1, 3).mean(0).astype(np.uint8)
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    else:
        canvas = img[y0:y1, x0:x1]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def mask_window(mask_native, margin, min_side=48):
    ys, xs = np.nonzero(mask_native)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1), float(max(side, min_side)))


# ----------------------------------------------------------------------------------------------- bank (declared constant)
def build_bank(a):
    """8 + 8 crops of the shelf episode (both exterior views), evenly spaced over the frames whose hand-prompted SAM3 mask
    has >= half the median present area and does not touch the image border (a fully visible gripper); threshold = min over crops of the max cosine to the
    crops of the OTHER view (leave-one-view-out: every bank crop is re-identified from the other viewpoint)."""
    os.makedirs(f"{a.bank_dir}/crops", exist_ok=True)
    meta = json.load(open(f"{RAW_ROOT}/{SHELF_EP}/metadata_{SHELF_EP}.json"))
    serials = {c: str(meta[f"{c}_cam_serial"]) for c in CAMS}
    emb = DinoEmbedder(device=a.device)
    crops, items = [], []
    for c in CAMS:
        mp = f"{SHELF_MASKS}/{SHELF_EP}/{c}_{serials[c]}__gripper_boxclick_masks.npz"
        d = np.load(mp)
        u = np.unpackbits(d["union"], axis=2)[:, :, :NATIVE_W].astype(bool)
        area = u.reshape(len(u), -1).sum(1)
        med = np.median(area[area > 0])
        elig = []
        for t in np.where(area >= 0.5 * med)[0]:            # fully visible gripper: silhouette not cut by the image border
            ys, xs = np.nonzero(u[t])
            if xs.min() > 0 and ys.min() > 0 and xs.max() < NATIVE_W - 1 and ys.max() < NATIVE_H - 1:
                elig.append(int(t))
        elig = np.array(elig)
        pick = elig[np.unique(np.linspace(0, len(elig) - 1, a.bank_per_view).round().astype(int))]
        cap = open_video(f"{RAW_ROOT}/{SHELF_EP}/recordings/MP4/{serials[c]}.mp4")
        t = 0
        frames = {}
        while True:
            ok, f = cap.read()
            if not ok or t > pick.max():
                break
            if t in set(pick.tolist()):
                frames[t] = f
            t += 1
        cap.release()
        for t in pick:
            win = mask_window(u[t], a.margin)
            crop = crop_window(frames[int(t)], win)
            cv2.imwrite(f"{a.bank_dir}/crops/{c}_{serials[c]}_f{int(t):04d}.png", crop)
            crops.append(crop)
            items.append(dict(cam=c, serial=serials[c], frame=int(t), window=[round(float(v), 1) for v in win], mask_area_px=int(area[t]), mask_source=mp))
    E = emb.embed(crops)
    S = E @ E.T
    cams = np.array([it["cam"] for it in items])
    loo_same, loo_other = [], []
    for i in range(len(items)):
        same = (cams == cams[i]); same[i] = False
        loo_same.append(float(S[i, same].max()))
        loo_other.append(float(S[i, ~(cams == cams[i])].max()))
    thr = float(min(loo_other))
    # montage for the visual check
    tiles = [cv2.putText(cr.copy(), f"{it['cam']} f{it['frame']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) for cr, it in zip(crops, items)]
    rows = [np.hstack(tiles[i:i + a.bank_per_view]) for i in range(0, len(tiles), a.bank_per_view)]
    cv2.imwrite(f"{a.bank_dir}/bank_montage.png", np.vstack(rows))
    np.savez(f"{a.bank_dir}/bank.npz", emb=E, cam=cams, frame=np.array([it["frame"] for it in items]), sim_thr=thr,
             note="DECLARED PER-RIG CONSTANT: DINOv2 ViT-B/14 descriptors of gripper crops from the shelf episode only")
    info = dict(declared_constant=True, kind="exemplar bank of gripper crops", source_episode=SHELF_EP, source_masks="hand-prompted SAM3 box+click shelf masks (gripper_sam3/*gripper_boxclick_masks.npz)",
                model=dict(arch=emb.arch, repo=TORCH_HUB_REPO, descriptor="[L2(CLS), L2(mean patch)] / sqrt2", crop_px=CROP_PX, margin=a.margin),
                n_crops=len(items), per_view=a.bank_per_view, items=items,
                frame_rule="frames with mask area >= 0.5 x median present area whose silhouette does not touch the image border (fully visible gripper), evenly spaced (np.linspace) per view",
                threshold_rule="sim_thr = min over bank crops of the max cosine to the crops of the OTHER exterior view (leave-one-view-out)",
                sim_thr=thr, loo_other_view=dict(min=thr, median=float(np.median(loo_other)), max=float(max(loo_other)), values=[round(v, 4) for v in loo_other]),
                loo_same_view=dict(min=float(min(loo_same)), median=float(np.median(loo_same)), max=float(max(loo_same))),
                sim_matrix=[[round(float(v), 4) for v in r] for r in S], device=emb.device)
    json.dump(info, open(f"{a.bank_dir}/bank.json", "w"), indent=1)
    print(json.dumps({k: info[k] for k in ("n_crops", "sim_thr", "loo_other_view", "loo_same_view")}, indent=1))
    print("wrote", f"{a.bank_dir}/bank.npz", f"{a.bank_dir}/bank.json", f"{a.bank_dir}/bank_montage.png")


# ----------------------------------------------------------------------------------------------- windows
def dedupe_windows(wins, iou_thr=0.9):
    keep = []
    for w in wins:
        cx, cy, s = w
        dup = False
        for k in keep:
            kx, ky, ks = k
            ix = max(0, min(cx + s / 2, kx + ks / 2) - max(cx - s / 2, kx - ks / 2))
            iy = max(0, min(cy + s / 2, ky + ks / 2) - max(cy - s / 2, ky - ks / 2))
            inter = ix * iy
            if inter / (s * s + ks * ks - inter) >= iou_thr:
                dup = True
                break
        if not dup:
            keep.append(w)
    return keep


def candidate_windows(cands, s, margin, max_windows):
    """One square window per motion candidate (tight bbox of its proc-res mask, scaled to native, + margin)."""
    wins = []
    for cd in cands:
        ys, xs = np.nonzero(cd["mask"])
        x0, x1 = (xs.min() + 0.5) * s - 0.5, (xs.max() + 0.5) * s - 0.5
        y0, y1 = (ys.min() + 0.5) * s - 0.5, (ys.max() + 0.5) * s - 0.5
        side = max(x1 - x0, y1 - y0, 1.0) * (1 + 2 * margin)
        wins.append((0.5 * (x0 + x1), 0.5 * (y0 + y1), float(max(side, 48.0))))
    wins = dedupe_windows(wins)
    if len(wins) > max_windows:                       # keep the smaller windows (gripper-sized) first
        wins = sorted(wins, key=lambda w: w[2])[:max_windows]
    return wins


def search_windows(prev, shifts=(-0.5, 0.0, 0.5), scales=(0.8, 1.0, 1.25)):
    cx, cy, side = prev
    return [(cx + dx * side, cy + dy * side, side * sc) for sc in scales for dy in shifts for dx in shifts]


# ----------------------------------------------------------------------------------------------- per-episode run
def run_episode(a):
    ep = a.episode
    raw = os.path.join(RAW_ROOT, ep)
    meta = json.load(open(os.path.join(raw, f"metadata_{ep}.json")))
    cams = list(CAMS)
    serials = {c: str(meta[f"{c}_cam_serial"]) for c in cams}
    videos = {c: os.path.join(raw, "recordings", "MP4", f"{serials[c]}.mp4") for c in cams}
    for c, p in videos.items():
        if not os.path.exists(p):
            sys.exit(f"missing video for {c}: {p}")
    out_dir = os.path.join(a.out_root, ep)
    os.makedirs(out_dir, exist_ok=True)
    bank = np.load(f"{a.bank_dir}/bank.npz")
    bank_info = json.load(open(f"{a.bank_dir}/bank.json"))
    B = bank["emb"].astype(np.float32)
    sim_thr = float(bank["sim_thr"]) if a.sim_thr is None else a.sim_thr
    proc_w, proc_h = a.proc_w, a.proc_w * 9 // 16
    s = NATIVE_W / proc_w
    T_store = len([f for f in os.listdir(os.path.join(STORE_ROOT, ep, "dense", "cam")) if f.endswith(".npz")])
    cal = load_calib(ep, serials)
    px_per_m = {c: cal[c]["fx"] / s / max(cal[c]["plausible_depth"], a.min_depth) for c in cams}
    emb = DinoEmbedder(device=a.device)
    print(f"episode {ep}\nstore frames {T_store}\nout {out_dir}\nbank {a.bank_dir} n={len(B)} sim_thr={sim_thr:.4f} device={emb.device}", flush=True)

    t_all = time.time()
    views = {c: ViewMotion(a) for c in cams}
    caps = {c: open_video(videos[c]) for c in cams}
    packed = {c: np.zeros((T_store, NATIVE_H, NATIVE_W // 8), np.uint8) for c in cams}
    present = {c: np.zeros(T_store, bool) for c in cams}
    k_dark = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.dark_open_px, a.dark_open_px))
    prev_win = {c: None for c in cams}     # last picked window per camera (cx, cy, side) native
    X_prev, t_prev = None, None
    tracks = []
    logs = []
    modes = {}
    tm = dict(motion=0.0, embed=0.0, verify=0.0)
    n_win_total = 0
    png_frames = set(int(x) for x in a.png_frames.split(",") if x.strip())
    t = 0
    while t < T_store and (a.max_frames is None or t < a.max_frames):
        fr = read_pair(caps, proc_w, proc_h)
        if fr is None:
            break
        t0 = time.time()
        info = dict(t=t, mode="warmup", n_win={}, best_sim={}, n_above={}, n_verified=0)
        alive = X_prev is not None and (t - t_prev) <= a.coast_frames
        wins, st_ = {}, {}
        for c in cams:
            st = views[c].step(fr[c][1])
            if st is None:
                continue
            fg, moving, moving_dil, gray, chroma = st
            st_[c] = st
            _, cands = extract_candidates(fg, moving, moving_dil, gray, chroma, cal[c], a, s, px_per_m[c])
            w = candidate_windows(cands, s, a.margin, a.max_windows)
            n_cand = len(w)
            if alive and prev_win[c] is not None:
                w = dedupe_windows(w + search_windows(prev_win[c]))
            wins[c] = dict(wins=w, n_cand=n_cand)
        t1 = time.time(); tm["motion"] += t1 - t0
        if all(c in wins for c in cams):
            # --- embed all windows of both cameras in one batch
            crops, owner = [], []
            for c in cams:
                for w in wins[c]["wins"]:
                    cr = crop_window(fr[c][0], w)
                    if cr is not None:
                        crops.append(cr); owner.append(c)
            E = emb.embed(crops)
            sims = (E @ B.T).max(1) if len(E) else np.zeros(0, np.float32)
            n_win_total += len(crops)
            t2 = time.time(); tm["embed"] += t2 - t1
            scored = {c: [] for c in cams}
            for w, c, sm in zip([w for c in cams for w in wins[c]["wins"]], owner, sims):
                scored[c].append((float(sm), w))
            for c in cams:
                scored[c].sort(key=lambda z: -z[0])
                info["n_win"][c] = len(scored[c]); info["best_sim"][c] = round(scored[c][0][0], 4) if scored[c] else None
                info["n_above"][c] = int(sum(1 for sm, _ in scored[c] if sm > sim_thr))
            top = {c: [(sm, w) for sm, w in scored[c] if sm > sim_thr][:a.top_k] for c in cams}
            # --- cross-view verification of the top-k x top-k pairs
            verified = []
            c1, c2 = cams
            for i, (s1, w1) in enumerate(top[c1]):
                for j, (s2, w2) in enumerate(top[c2]):
                    X, r, rv = triangulate(cal[c1]["P"], cal[c2]["P"], (w1[0], w1[1]), (w2[0], w2[1]))
                    dist = float(np.linalg.norm(X))
                    ok = np.isfinite(r) and dist < a.reach and X[2] > a.zmin
                    depths, sizes = [], []
                    for w, c, rr in ((w1, c1, rv[0]), (w2, c2, rv[1])):
                        _, _, z = project(cal[c], X)
                        depths.append(round(float(z), 3))
                        if z < a.min_cam_depth or rr > a.ray_tol_m * cal[c]["fx"] / max(z, 1e-6):
                            ok = False
                        size_m = w[2] / (1 + 2 * a.margin) * z / cal[c]["fx"]
                        sizes.append(round(float(size_m), 3))
                        if not (a.size_min <= size_m <= a.size_max):
                            ok = False
                    if ok:
                        verified.append(dict(i=i, j=j, X=[round(float(x), 4) for x in X], r=round(float(r), 2), dist=round(dist, 3), depths=depths, sizes=sizes,
                                             sims=[round(s1, 4), round(s2, 4)], score=s1 + s2, wins={c1: w1, c2: w2}))
            info["n_verified"] = len(verified)
            best, mode = None, "none"
            if verified:
                if alive:
                    gap = t - t_prev
                    cons = [v for v in verified if np.linalg.norm(np.array(v["X"]) - X_prev) <= a.jump_m * gap]
                    best, mode = (max(cons, key=lambda v: v["score"]), "track") if cons else (max(verified, key=lambda v: v["score"]), "switch")
                else:
                    best, mode = max(verified, key=lambda v: v["score"]), "new"
                if mode != "track":
                    tracks.append(dict(track=len(tracks), frame=int(t), mode=mode, **{k: best[k] for k in ("X", "r", "dist", "depths", "sizes", "sims")}))
                X_prev, t_prev = np.array(best["X"]), t
            info["mode"] = mode
            if best is not None:
                X = np.array(best["X"])
                info.update(X=best["X"], r=best["r"], dist=best["dist"], sizes=best["sizes"], depths=best["depths"], sims=best["sims"], win={}, proj={}, area={}, mask_src={})
                for c in cams:
                    w = best["wins"][c]
                    prev_win[c] = w
                    fg, moving, moving_dil, gray, chroma = st_[c]
                    cx, cy, side = w
                    x0, x1 = int(max(0, np.floor(cx - side / 2))), int(min(NATIVE_W, np.ceil(cx + side / 2) + 1))
                    y0, y1 = int(max(0, np.floor(cy - side / 2))), int(min(NATIVE_H, np.ceil(cy + side / 2) + 1))
                    box = np.zeros((NATIVE_H, NATIVE_W), bool); box[y0:y1, x0:x1] = True
                    nat = cv2.resize(fg.astype(np.uint8), (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_NEAREST).astype(bool) & box
                    src = "fg"
                    if nat.sum() < a.min_mask_px:
                        dark = ((gray < a.dark_thresh) & (chroma < a.chroma_thresh)).astype(np.uint8)
                        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k_dark)
                        nat = cv2.resize(dark, (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_NEAREST).astype(bool) & box
                        src = "dark"
                    if nat.sum() < a.min_mask_px:
                        nat[:] = False; src = "empty"
                    packed[c][t] = np.packbits(nat, axis=1)
                    present[c][t] = bool(nat.any())
                    px_, py_, z_ = project(cal[c], X)
                    info["win"][c] = [round(float(v), 1) for v in w]; info["proj"][c] = [round(px_, 1), round(py_, 1), round(z_, 3)]
                    info["area"][c] = int(nat.sum()); info["mask_src"][c] = src
            tm["verify"] += time.time() - t2
            if t in png_frames:
                panels = []
                for c in cams:
                    img = fr[c][0].copy()
                    for sm, w in scored[c][:12]:
                        cx, cy, side = w
                        col = (0, 0, 255) if (best is not None and w == best["wins"][c]) else ((0, 255, 255) if sm > sim_thr else (160, 160, 160))
                        cv2.rectangle(img, (int(cx - side / 2), int(cy - side / 2)), (int(cx + side / 2), int(cy + side / 2)), col, 2)
                        cv2.putText(img, f"{sm:.2f}", (int(cx - side / 2), int(cy - side / 2) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                    if present[c][t]:
                        m = np.unpackbits(packed[c][t], axis=1)[:, :NATIVE_W].astype(bool)
                        ov = img.copy(); ov[m] = (0, 255, 0); img = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
                    hdr = f"{c}/{serials[c]} {METHOD} f{t} {mode} win={info['n_win'].get(c)} above={info['n_above'].get(c)} best={info['best_sim'].get(c)} thr={sim_thr:.2f}"
                    cv2.rectangle(img, (0, 0), (NATIVE_W, 26), (0, 0, 0), -1)
                    cv2.putText(img, hdr, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    panels.append(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
                cv2.imwrite(os.path.join(out_dir, f"{METHOD}_check_f{t:04d}.png"), np.concatenate(panels, axis=1))
        modes[info["mode"]] = modes.get(info["mode"], 0) + 1
        logs.append(info)
        if t % 100 == 0:
            print(f"  f{t}: mode={info['mode']} win={info['n_win']} best={info['best_sim']} above={info['n_above']} verified={info['n_verified']} X={info.get('X')} sims={info.get('sims')} ({time.time() - t_all:.0f}s)", flush=True)
        t += 1
    for cap in caps.values():
        cap.release()
    T_mp4 = t
    if T_mp4 < T_store:
        print(f"WARNING: mp4 pair yielded {T_mp4} frames < store {T_store}; remaining frames are empty", flush=True)
        for k in range(T_mp4, T_store):
            logs.append(dict(t=k, mode="no_frame", n_win={}, best_sim={}, n_above={}, n_verified=0))
    t_seed = time.time() - t_all

    products, stats = {}, {}
    for c in cams:
        path = os.path.join(out_dir, f"{c}_{serials[c]}__{METHOD}_masks.npz")
        np.savez_compressed(path, union=packed[c], shape=np.array([T_store, NATIVE_H, NATIVE_W]), frames_present=present[c])
        chk = np.load(path)
        u = np.unpackbits(chk["union"], axis=2)[:, :, :NATIVE_W].astype(bool)
        nonempty = u.reshape(T_store, -1).any(1)
        ok = tuple(int(v) for v in chk["shape"]) == (T_store, NATIVE_H, NATIVE_W) and u.shape == (T_store, NATIVE_H, NATIVE_W) \
            and bool((chk["frames_present"] == nonempty).all()) and int(nonempty.sum()) == int(present[c].sum())
        pr = present[c]
        products[c] = dict(npz=path, T=T_store, store_frames=T_store, mp4_frames_read=T_mp4, bytes=os.path.getsize(path), frames_present=int(pr.sum()), product_ok=bool(ok))
        areas = [lg.get("area", {}).get(c) for lg in logs if lg.get("area", {}).get(c)]
        srcs = {}
        for lg in logs:
            k = lg.get("mask_src", {}).get(c)
            if k:
                srcs[k] = srcs.get(k, 0) + 1
        stats[c] = dict(cam=c, serial=serials[c], n_frames=T_store, frames_present=int(pr.sum()), frac_present=float(pr.mean()) if T_store else 0.0,
                        first_present=int(np.argmax(pr)) if pr.any() else None, last_present=int(T_store - 1 - np.argmax(pr[::-1])) if pr.any() else None,
                        mean_area_frac_present=float(np.mean(areas) / (NATIVE_W * NATIVE_H)) if areas else None, mask_sources=srcs,
                        best_sim_median=float(np.median([lg["best_sim"][c] for lg in logs if lg.get("best_sim", {}).get(c) is not None])) if any(lg.get("best_sim", {}).get(c) is not None for lg in logs) else None,
                        calib=dict(fx=round(cal[c]["fx"], 1), fy=round(cal[c]["fy"], 1), base_px=cal[c]["base_px"], cam_to_base=round(cal[c]["cam_to_base"], 3),
                                   plausible_depth=round(cal[c]["plausible_depth"], 3)))
        print(f"[{c}] wrote {path} T={T_store} present={int(pr.sum())} ok={ok} sources={srcs}", flush=True)
    vp_xyz = np.full((T_store, 3), np.nan)
    vp_acc = np.zeros(T_store, bool)
    for lg in logs:
        if lg.get("X") is not None and 0 <= lg["t"] < T_store:
            vp_xyz[lg["t"]] = np.array(lg["X"], float); vp_acc[lg["t"]] = True
    vp_path = os.path.join(out_dir, "verified_points.npz")
    np.savez(vp_path, frames=np.arange(T_store, dtype=np.int32), xyz=vp_xyz, accepted=vp_acc,
             note="base-frame (m) DLT point of the two picked window centres per frame; NaN / accepted=False where nothing was verified; causal (frame t uses frames <= t); calibration only, no GT")
    chk = np.load(vp_path)
    vp_ok = (chk["xyz"].shape == (T_store, 3) and chk["accepted"].shape == (T_store,) and bool((np.isfinite(chk["xyz"]).all(1) == chk["accepted"]).all())
             and int(chk["accepted"].sum()) == sum(1 for lg in logs if lg.get("X") is not None))
    print(f"[verified_points] wrote {vp_path} T={T_store} accepted={int(vp_acc.sum())} ok={vp_ok}", flush=True)
    n_picked = int(vp_acc.sum())
    n_proc = max(1, sum(1 for lg in logs if lg["mode"] not in ("warmup", "no_frame")))
    gpu = None
    try:
        gpu = emb.torch.cuda.get_device_name(0) if emb.device.startswith("cuda") else "cpu"
    except Exception:  # noqa: BLE001
        pass
    info = dict(episode=ep, method=METHOD, serials=serials, params=vars(a), proc_res=[proc_w, proc_h], native_res=[NATIVE_W, NATIVE_H],
                store_frames=T_store, mp4_frames_read=T_mp4, causal=True, uses_gt=False, training_free=True,
                declared_constants=dict(exemplar_bank=dict(path=f"{a.bank_dir}/bank.npz", info=f"{a.bank_dir}/bank.json", source_episode=bank_info["source_episode"],
                                                            n_crops=bank_info["n_crops"], frames={f"{it['cam']}": [j["frame"] for j in bank_info["items"] if j["cam"] == it["cam"]] for it in bank_info["items"]},
                                                            sim_thr=sim_thr, threshold_rule=bank_info["threshold_rule"], model=bank_info["model"]),
                                        calibration="PointWorld optimized_extrinsics + factory intrinsics (cross-view triangulation, reach/height/size gates; plausible depth for the candidate cap lengths only)"),
                mask_rule="MOG2 foreground pixels inside the winning window; dark-achromatic pixels inside the window when fewer than min_mask_px; no SAM3",
                verification=dict(ray_tol_m=a.ray_tol_m, reach_m=a.reach, zmin_m=a.zmin, min_cam_depth=a.min_cam_depth, size_window_m=[a.size_min, a.size_max], top_k=a.top_k,
                                  jump_m_per_frame=a.jump_m, coast_frames=a.coast_frames),
                summary=dict(frames_picked=n_picked, frac_picked=n_picked / T_store if T_store else 0.0, modes=modes, n_tracks=len(tracks),
                             windows_embedded=n_win_total, windows_per_frame=n_win_total / n_proc, frames_with_any_verified=sum(1 for lg in logs if lg["n_verified"] > 0)),
                tracks=tracks, per_cam=stats, products=products, verified_points=dict(npz=vp_path, T=T_store, accepted=n_picked, product_ok=bool(vp_ok)),
                timing=dict(total_seconds=round(time.time() - t_all, 1), seeding_seconds=round(t_seed, 1), ms_per_frame_total=round(1000 * t_seed / max(T_mp4, 1), 1),
                            ms_per_frame_motion_candidates=round(1000 * tm["motion"] / max(T_mp4, 1), 1), ms_per_frame_embed=round(1000 * tm["embed"] / max(T_mp4, 1), 1),
                            ms_per_frame_verify_emit=round(1000 * tm["verify"] / max(T_mp4, 1), 1), device=emb.device, gpu=gpu, omp_threads=os.environ.get("OMP_NUM_THREADS")),
                per_frame=logs)
    json.dump(info, open(os.path.join(out_dir, "seed_info.json"), "w"))
    print(f"done {ep}: picked {n_picked}/{T_store} modes={modes} tracks={len(tracks)} timing={info['timing']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build_bank", action="store_true")
    ap.add_argument("--bank_dir", default=BANK_DIR)
    ap.add_argument("--bank_per_view", type=int, default=8)
    ap.add_argument("--episode", default=None)
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    ap.add_argument("--proc_w", type=int, default=640)
    # background model + candidates: seed_verified_motion.py defaults (reused code)
    ap.add_argument("--history", type=int, default=300)
    ap.add_argument("--var_thresh", type=float, default=25.0)
    ap.add_argument("--no_shadows", action="store_true")
    ap.add_argument("--shadow_tau", type=float, default=0.3)
    ap.add_argument("--dark_thresh", type=int, default=70)
    ap.add_argument("--chroma_thresh", type=int, default=15)
    ap.add_argument("--warmup_frames", type=int, default=5)
    ap.add_argument("--warmup_lr", type=float, default=0.3)
    ap.add_argument("--close_px", type=int, default=9)
    ap.add_argument("--open_px", type=int, default=5)
    ap.add_argument("--move_thresh", type=int, default=10)
    ap.add_argument("--move_dilate_px", type=int, default=7)
    ap.add_argument("--min_area_frac", type=float, default=0.0015)
    ap.add_argument("--min_area_mov_frac", type=float, default=0.0004)
    ap.add_argument("--cap_lengths", default="0.1,0.15,0.2,0.25")
    ap.add_argument("--geo_grid", type=int, default=2)
    ap.add_argument("--min_depth", type=float, default=0.35)
    ap.add_argument("--max_hole_px", type=int, default=400)
    ap.add_argument("--max_frag", type=int, default=8)
    ap.add_argument("--dark_open_px", type=int, default=5)
    # windows + re-identification
    ap.add_argument("--margin", type=float, default=0.15, help="window = tight bbox + this fraction of its longer side on each side (bank crops use the same)")
    ap.add_argument("--max_windows", type=int, default=64, help="candidate windows per camera per frame (smallest first)")
    ap.add_argument("--sim_thr", type=float, default=None, help="override the bank's leave-one-view-out threshold (diagnostics only)")
    ap.add_argument("--top_k", type=int, default=5)
    # cross-view verification (seed_verified_motion values; size window widened because it is applied to a window, not a silhouette)
    ap.add_argument("--ray_tol_m", type=float, default=0.04)
    ap.add_argument("--reach", type=float, default=0.9)
    ap.add_argument("--zmin", type=float, default=-0.15)
    ap.add_argument("--min_cam_depth", type=float, default=0.15)
    ap.add_argument("--size_min", type=float, default=0.05)
    ap.add_argument("--size_max", type=float, default=0.40)
    ap.add_argument("--jump_m", type=float, default=0.12)
    ap.add_argument("--coast_frames", type=int, default=15)
    ap.add_argument("--min_mask_px", type=int, default=300)
    ap.add_argument("--png_frames", default="")
    ap.add_argument("--max_frames", type=int, default=None, help="smoke test: stop after this many frames (products keep T = store frames, the rest empty)")
    a = ap.parse_args()
    a.cap_lengths = [float(x) for x in a.cap_lengths.split(",") if x.strip()]
    if a.build_bank:
        build_bank(a)
    elif a.episode:
        run_episode(a)
    else:
        ap.error("--build_bank or --episode EP")


if __name__ == "__main__":
    main()
