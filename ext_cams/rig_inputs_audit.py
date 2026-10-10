#!/usr/bin/env python
"""rig_inputs_audit: sanity audit of the exterior-rig INPUTS on every v2 test scene (2026-10-08). GT (FK) is used here only
as the reference, never by a method. Per (scene, camera):
  calibration  intrinsics source + values (rig_trackon_eval.calib), max |K_rig - K_labels| (the labels were projected with),
               camera distance to the robot base, gripper depth at birth, positions_v1 calibration-outlier flag, MP4 serial
               (Leonardo worklist) == metadata serial
  birth frame  items_v2 frame == npz meta birth == event-0 entry; v2 birth vs the positions_v1 b050 recomputation;
               store vs MP4 frame counts
  mask         RobotSeg birth mask vs the FK gripper silhouette (union of per-link convex hulls of the 116 projected label
               points, occlusion ignored): share of mask pixels within a strict (0.15 sqrt(A) + 3 px) / loose (0.5 sqrt(A))
               margin of the silhouette, share of the in-image silhouette covered by the mask, centroid offset / sqrt(A),
               connected components (> 2 % of the mask)
  seeds        share of Track-On seeds within the strict / loose margin
  tracks       rig window (birth .. FK exit): per frame the share of VISIBLE Track-On points within the margins of that
               frame's silhouette; lost frame = < 50 % within the loose margin (>= 3 visible points); fail frame = first of 3
               consecutive lost frames.
    python rig_inputs_audit.py run OUT [--workers N] | summary OUT | qa OUT CATEGORY [--n 24]
"""
import argparse, csv, glob, json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_track as rt          # noqa: E402
import rig_trackon_eval as rte  # noqa: E402

S = "/gpfs/scratch/etur59/koc821022"; DBF = f"{S}/outputs/droid_birth_frames"
MASKS = f"{DBF}/birth_masks_v2_seeding"; TRACKS = f"{DBF}/trackon_run_v2/tracks"; EVENTS = f"{DBF}/entries/entry_events_gap3.csv"
POS = f"{S}/gripper_points/positions_v1"; EPS = f"{S}/outputs/cut3r_eval/ext_cams/rig_trackon/eps_v2.txt"
SUBSET = f"{S}/outputs/cut3r_eval/ext_cams/rig_v0/eval_subset_masks_ok.txt"
W, H = 1280, 720


def silhouette(uv, ok, link, scale, shape):
    """Union of per-link convex hulls of the projected label points (uv at 320x180 x scale)."""
    m = np.zeros(shape, np.uint8)
    for name in np.unique(link):
        p = uv[(link == name) & ok] * scale
        if len(p) >= 3: cv2.fillConvexPoly(m, cv2.convexHull(np.round(p).astype(np.int32)), 1)
    return m


def dist_to(sil):
    return cv2.distanceTransform((sil == 0).astype(np.uint8), cv2.DIST_L2, 3)


def lookup(d, pts, scale):
    q = np.clip(np.round(pts * scale).astype(int), 0, [d.shape[1] - 1, d.shape[0] - 1]); return d[q[:, 1], q[:, 0]]


def audit_scene(ep, ctx):
    items, ev, wl, excl = ctx["items"], ctx["events"], ctx["worklist"], ctx["excluded"]
    out = []
    try:
        meta = json.load(open(f"{rt.RAW_ROOT}/{ep}/metadata_{ep}.json"))
        cams, K, E, P, ksrc = rte.calib(ep, f"{POS}/positions")
    except Exception as ex:
        return [dict(episode=ep, cam="", error=f"calib: {ex}")]
    zf = f"{POS}/positions/{ep}_gripper.npz"; z = np.load(zf) if os.path.isfile(zf) else None
    e0 = ev.get(ep)
    for cam, s in cams:
        r = dict(episode=ep, cam=cam, lab=ep.split("+")[0], serial=s, K_src=ksrc[cam], fx=K[cam][0, 0], fy=K[cam][1, 1], cx=K[cam][0, 2], cy=K[cam][1, 2],
                 calib_outlier=(ep, s) in excl, cam_dist_m=float(np.linalg.norm(-E[cam][:3, :3].T @ E[cam][:3, 3])))
        w_ = wl.get((ep, cam)); r["serial_mp4"] = os.path.basename(w_["mp4"]).split(".")[0] if w_ else ""
        r["serial_match"] = r["serial_mp4"] == s
        it = items.get((ep, cam))
        if it is None: r["error"] = "no v2 item"; out.append(r); continue
        b = int(it["frame"]); r.update(birth=b, pw_flag=str(it["projection_disagree_pw"]) == "True")
        f = f"{TRACKS}/{ep}/{cam}_f{b:05d}.npz"
        if not os.path.isfile(f): r["error"] = "track file missing"; out.append(r); continue
        tz = np.load(f); m_ = json.loads(str(tz["meta"])); tr = tz["tracks"]; vis = tz["visibility"]; q = tz["queries"]
        r.update(npz_birth=m_["birth_frame"], n_store=m_["n_frames_store"], n_mp4=m_["n_frames_mp4"], ev_entry=e0[0] if e0 else -1, ev_exit=e0[1] if e0 else -1)
        r["frames_consistent"] = (r["npz_birth"] == b) and (r["ev_entry"] == b)
        if z is None or s not in [str(x) for x in z["serials"]]:
            r["error"] = "no labels for this camera" + (" (calibration outlier)" if r["calib_outlier"] else ""); out.append(r); continue
        link = z["link"]; uv = z[f"{s}_uv"].astype(np.float64); zz = z[f"{s}_z"].astype(np.float64); front = z[f"{s}_front"] & z[f"{s}_in_frame"]
        Kl = np.diag([4, 4, 1.0]) @ z[f"{s}_K"]; r["K_vs_labels_px"] = float(np.abs(Kl - K[cam]).max())
        Tl = min(len(uv), tr.shape[0]); fr = front[b] if b < len(front) else np.zeros(len(link), bool)
        r["grip_depth_m"] = float(np.median(zz[b][fr])) if fr.any() else np.nan
        # ---- birth mask vs FK silhouette (720p)
        mk = cv2.imread(f"{MASKS}/{it['mask_relpath']}", 0)
        if mk is None or b >= len(uv): r["error"] = "mask missing" if mk is None else "birth beyond labels"; out.append(r); continue
        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST) > 0 if mk.shape != (H, W) else mk > 0
        sil = silhouette(uv[b], zz[b] > 0, link, 4.0, (H, W)); A = max(float(sil.sum()), 1.0); d = dist_to(sil)
        ms, ml = 0.15 * np.sqrt(A) + 3, 0.5 * np.sqrt(A)
        r.update(mask_area=int(mk.sum()), sil_area=int(A), mask_prec_strict=float((d[mk] <= ms).mean()) if mk.any() else np.nan,
                 mask_prec_loose=float((d[mk] <= ml).mean()) if mk.any() else np.nan,
                 mask_cov=float((cv2.dilate(mk.astype(np.uint8), np.ones((11, 11), np.uint8))[sil > 0] > 0).mean()))
        if mk.any():
            yx = np.argwhere(mk).mean(0); syx = np.argwhere(sil).mean(0); r["centroid_off_norm"] = float(np.linalg.norm(yx - syx) / np.sqrt(A))
            n, lab_, st, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8), 8); ar = st[1:, cv2.CC_STAT_AREA]
            r["n_cc"] = int((ar > 0.02 * mk.sum()).sum()); r["largest_cc_frac"] = float(ar.max() / mk.sum())
        qb = q[q[:, 0] == b, 1:]
        if len(qb):
            dq = lookup(d, qb, 1.0); r.update(n_seeds=len(qb), seed_in_strict=float((dq <= ms).mean()), seed_in_loose=float((dq <= ml).mean()))
        # ---- tracks over the rig window (320x180 silhouettes)
        t_end = min(r["ev_exit"] if r["ev_exit"] > 0 else Tl, Tl) - 1
        fs, fl, nv, lost = [], [], [], []
        for t in range(b, t_end + 1):
            ok = vis[t] & ~np.isnan(tr[t, :, 0])
            nv.append(int(ok.sum()))
            if ok.sum() < 3: fs.append(np.nan); fl.append(np.nan); lost.append(False); continue
            s320 = silhouette(uv[t], zz[t] > 0, link, 1.0, (180, 320)); a320 = max(float(s320.sum()), 1.0); d3 = dist_to(s320)
            dd = lookup(d3, tr[t][ok], 0.25)
            fs.append(float((dd <= 0.15 * np.sqrt(a320) + 0.75).mean())); fl.append(float((dd <= 0.5 * np.sqrt(a320)).mean())); lost.append(fl[-1] < 0.5)
        lost = np.array(lost); L = len(lost); fail = next((i for i in range(L - 2) if lost[i] and lost[i + 1] and lost[i + 2]), None)
        r.update(win_len=L, vis_pts_med=float(np.median(nv)) if nv else np.nan, trk_in_strict_med=float(np.nanmedian(fs)) if L else np.nan,
                 trk_in_loose_med=float(np.nanmedian(fl)) if L else np.nan, lost_frac=float(lost.mean()) if L else np.nan,
                 fail_rel=(fail / L) if fail is not None else np.nan, fail_frame=(b + fail) if fail is not None else -1,
                 trk_in_loose_last=float(np.nanmean(fl[-max(1, L // 10):])) if L else np.nan)
        out.append(r)
    return out


def ctx_load():
    items = {(r["episode"], r["cam"]): r for r in csv.DictReader(open(f"{MASKS}/items_v2.csv"))}
    events = {r["episode"]: (int(r["entry_frame"]), int(r["exit_frame"])) for r in csv.DictReader(open(EVENTS)) if r["event"] == "0"}
    wl = {(w["episode"], w["cam"]): w for w in json.load(open(f"{DBF}/trackon_run_v2/worklist.json"))}
    excl = {(r["ep"], r["serial"]) for r in csv.DictReader(open(f"{POS}/excluded.csv"))}
    return dict(items=items, events=events, worklist=wl, excluded=excl)


CTX = None
def _init():
    global CTX; CTX = ctx_load()
def _one(ep):
    try: return audit_scene(ep, CTX)
    except Exception as ex: return [dict(episode=ep, cam="", error=f"{type(ex).__name__}: {ex}")]


def run(out, workers):
    os.makedirs(out, exist_ok=True); eps = [l.strip() for l in open(EPS) if l.strip()]
    with Pool(workers, initializer=_init) as p: rows = [r for rs in p.imap_unordered(_one, eps, chunksize=4) for r in rs]
    pd.DataFrame(rows).to_csv(f"{out}/items.csv", index=False); print(len(rows), "rows")


FLAGS = {  # name: (description, rule)
    "serial_mismatch": ("MP4 serial != metadata serial", lambda d: d.serial_match == False),
    "frames_inconsistent": ("birth frame differs between mask / npz / event-0", lambda d: d.frames_consistent == False),
    "calib_outlier": ("positions_v1 calibration outlier (session jump)", lambda d: d.calib_outlier == True),
    "K_mismatch": ("rig K differs from the labels' K by > 2 px", lambda d: d.K_vs_labels_px > 2),
    "K_odd": ("intrinsics implausible (fx outside 450-800 or principal point > 80 px off centre)",
              lambda d: (d.fx < 450) | (d.fx > 800) | ((d.cx - 640).abs() > 80) | ((d.cy - 360).abs() > 80)),
    "cam_far": ("camera > 2.5 m or < 0.2 m from the robot base", lambda d: (d.cam_dist_m > 2.5) | (d.cam_dist_m < 0.2)),
    "mp4_store_len": ("MP4 and store frame counts differ", lambda d: d.n_store != d.n_mp4),
    "mask_off": ("< 50 % of mask pixels within the loose margin of the FK gripper", lambda d: d.mask_prec_loose < 0.5),
    "mask_spill": ("< 70 % of mask pixels within the strict margin (arm / background spill)", lambda d: d.mask_prec_strict < 0.7),
    "mask_small": ("mask covers < 30 % of the in-image FK gripper", lambda d: d.mask_cov < 0.3),
    "mask_fragmented": ("largest mask component < 70 % of the mask", lambda d: d.largest_cc_frac < 0.7),
    "seeds_off": ("< 80 % of seeds within the loose margin", lambda d: d.seed_in_loose < 0.8),
    "tracks_lost": ("tracks lose the gripper (3 consecutive frames < 50 % of visible points within the loose margin)", lambda d: d.fail_frame >= 0),
    "tracks_lost_early": ("... within the first half of the rig window", lambda d: (d.fail_frame >= 0) & (d.fail_rel < 0.5)),
}


def summary(out):
    d = pd.read_csv(f"{out}/items.csv"); sub = set(l.strip() for l in open(SUBSET) if l.strip())
    d["in_subset"] = d.episode.isin(sub); F = []
    def pr(*a): s = " ".join(str(x) for x in a); print(s); F.append(s)
    pr("# Exterior-rig input audit (v2 tracks, birth masks, PointWorld calibration), 2026-10-08\n")
    pr(f"items {len(d)} (scenes {d.episode.nunique()}); errors: {d.error.value_counts().to_dict() if 'error' in d else {}}")
    ok = d[d.get("error").isna()] if "error" in d else d
    pr(f"audited items {len(ok)}; in the 4002-scene eval subset: {int(ok.in_subset.sum())}\n")
    pr("| flag | rule | items | scenes | items in eval subset |\n|---|---|---|---|---|")
    for k, (desc, rule) in FLAGS.items():
        m = rule(ok).fillna(False).astype(bool); ok[f"F_{k}"] = m
        pr(f"| {k} | {desc} | {int(m.sum())} ({m.mean():.1%}) | {ok[m].episode.nunique()} | {int((m & ok.in_subset).sum())} |")
    pr("\n## distributions (audited items)\n")
    for c in ["fx", "cam_dist_m", "grip_depth_m", "K_vs_labels_px", "mask_prec_strict", "mask_prec_loose", "mask_cov", "centroid_off_norm",
              "seed_in_strict", "seed_in_loose", "trk_in_strict_med", "trk_in_loose_med", "lost_frac", "trk_in_loose_last", "vis_pts_med", "win_len"]:
        if c in ok: v = ok[c].astype(float); pr(f"- {c}: p1 {v.quantile(.01):.3f} p10 {v.quantile(.1):.3f} median {v.median():.3f} p90 {v.quantile(.9):.3f} p99 {v.quantile(.99):.3f}")
    pr(f"\nK sources: {ok.K_src.value_counts().to_dict()}")
    anyf = ok[[c for c in ok if c.startswith("F_")]].any(axis=1)
    pr(f"\nitems with any flag: {int(anyf.sum())} ({anyf.mean():.1%}); scenes with any flagged camera: {ok[anyf].episode.nunique()}; "
       f"eval-subset scenes with any flagged camera: {ok[anyf & ok.in_subset].episode.nunique()}")
    per = ok.assign(any=anyf).groupby("lab").agg(items=("cam", "size"), any_flag=("any", "mean"), mask_off=("F_mask_off", "mean"),
                                                 tracks_lost=("F_tracks_lost", "mean"), trk_in_loose=("trk_in_loose_med", "median"))
    pr("\n## per lab\n" + per.round(3).to_string())
    open(f"{out}/SUMMARY.md", "w").write("\n".join(F) + "\n"); ok.to_csv(f"{out}/items_flagged.csv", index=False)


def read_frame(ep, serial, t):
    cap = cv2.VideoCapture(f"{rt.RAW_ROOT}/{ep}/recordings/MP4/{serial}.mp4"); img = None
    for _ in range(t + 1):
        ok, img = cap.read()
        if not ok: img = None; break
    cap.release()
    return None if img is None else (cv2.resize(img, (W, H)) if img.shape[:2] != (H, W) else img)


def qa(out, cat, n):
    d = pd.read_csv(f"{out}/items_flagged.csv")
    if cat == "random": sel = d.sample(n=min(n, len(d)), random_state=0)
    else:
        sel = d[d[f"F_{cat}"]]
        key = {"mask_off": "mask_prec_loose", "mask_spill": "mask_prec_strict", "mask_small": "mask_cov", "seeds_off": "seed_in_loose",
               "tracks_lost": "trk_in_loose_med", "tracks_lost_early": "trk_in_loose_med"}.get(cat)
        sel = sel.sort_values(key).head(n) if key else sel.sample(n=min(n, len(sel)), random_state=0)
    items = {(r["episode"], r["cam"]): r for r in csv.DictReader(open(f"{MASKS}/items_v2.csv"))}
    tiles = []
    for _, r in sel.iterrows():
        ep, cam, s, b = r.episode, r.cam, str(int(r.serial)), int(r.birth)
        track_cat = cat.startswith("tracks") and r.fail_frame >= 0
        t = int(r.fail_frame) if track_cat else b
        img = read_frame(ep, s, t)
        if img is None: continue
        z = np.load(f"{POS}/positions/{ep}_gripper.npz"); uv = z[f"{s}_uv"].astype(np.float64); zz = z[f"{s}_z"].astype(np.float64)
        sil = silhouette(uv[t], zz[t] > 0, z["link"], 4.0, (H, W))
        if not track_cat:
            mk = cv2.imread(f"{MASKS}/{items[(ep, cam)]['mask_relpath']}", 0) > 0
            img[mk] = (0.5 * img[mk] + 0.5 * np.array([255, 80, 0])).astype(np.uint8)
        cnt, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(img, cnt, -1, (0, 255, 0), 2)
        tz = np.load(f"{TRACKS}/{ep}/{cam}_f{b:05d}.npz"); pts = tz["tracks"][t]; vv = tz["visibility"][t] & ~np.isnan(pts[:, 0])
        for x, y in pts[vv]: cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255) if track_cat else (0, 0, 255), -1)
        ys, xs = np.nonzero(sil) if sil.any() else (np.array([H // 2]), np.array([W // 2]))
        cy, cx = int(ys.mean()), int(xs.mean()); half = int(max(80, 1.6 * max(np.ptp(ys), np.ptp(xs))))
        crop = img[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
        tile = np.concatenate([cv2.resize(img, (480, 270)), cv2.resize(crop, (270, 270))], 1)
        txt = f"{ep[:34]} {cam} t={t}"; cv2.putText(tile, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        val = f"prec {r.mask_prec_strict:.2f}/{r.mask_prec_loose:.2f} cov {r.mask_cov:.2f} seeds {r.seed_in_loose:.2f} trk {r.trk_in_loose_med:.2f}"
        cv2.putText(tile, val, (5, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1); tiles.append(tile)
    if not tiles: print("nothing to draw"); return
    while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles), 2)]
    os.makedirs(f"{out}/qa", exist_ok=True); f = f"{out}/qa/{cat}.jpg"; cv2.imwrite(f, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 80]); print(f, len(tiles))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["run", "summary", "qa"]); ap.add_argument("out"); ap.add_argument("cat", nargs="?")
    ap.add_argument("--workers", type=int, default=80); ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    if a.mode == "run": run(a.out, a.workers)
    elif a.mode == "summary": summary(a.out)
    else: qa(a.out, a.cat, a.n)
