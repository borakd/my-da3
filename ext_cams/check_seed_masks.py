#!/usr/bin/env python
"""DIAGNOSTIC ONLY (uses kinematic GT for scoring, never in a method path): check seed masks of one method against the
projected ground-truth wrist-camera centre in each exterior camera.

Per episode and camera: frame count vs the store, product sanity (shape, frames_present == non-empty union), and per
frame the distance (native px) from the projected GT camera centre to the nearest mask pixel, whether the centre lies
inside the mask dilated by `--tol` px (the centre sits inside the camera housing on the hand), the mask area, and the
distance from the mask centroid to the centre. Prints one line per camera and writes <out>/<EP>.json.

    OMP_NUM_THREADS=1 python check_seed_masks.py --method motion_only --episodes smoke13_scenes.txt
"""
import argparse, glob, json, os, sys
import cv2, numpy as np

RAW_ROOT = "/gpfs/scratch/etur59/koc821022/vggt_cache/raw"
STORE_ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
AUDIT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/gt_audit_2026-09-08"
SEEDSTUDY = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy"


def check_episode(ep, method, root, tol):
    meta = json.load(open(f"{RAW_ROOT}/{ep}/metadata_{ep}.json"))
    intr = json.load(open(f"{AUDIT}/docs/hf_intrinsics.json"))[ep]
    cams = json.load(open(f"{AUDIT}/pointworld/droid/cameras/{ep}_cameras.json"))
    store = sorted(glob.glob(f"{STORE_ROOT}/{ep}/dense/cam/*.npz"))
    n_store = len(store)
    gt = np.array([np.load(f)["pose"] for f in store])  # c2w, base frame (GT: scoring only)
    res = dict(episode=ep, method=method, store_frames=n_store, cams={})
    for cam in ("ext1", "ext2"):
        s = meta[f"{cam}_cam_serial"]
        path = f"{root}/{ep}/{cam}_{s}__{method}_masks.npz"
        r = dict(serial=s, npz=path, exists=os.path.exists(path))
        if not r["exists"]:
            res["cams"][cam] = r
            print(f"  {ep} {cam}: MISSING {path}")
            continue
        m = np.load(path)
        T, H, W = [int(v) for v in m["shape"]]
        packed, present = m["union"], m["frames_present"].astype(bool)
        r.update(T=T, H=H, W=W, bytes=os.path.getsize(path), T_matches_store=(T == n_store), shape_ok=(H, W) == (720, 1280),
                 packed_shape_ok=tuple(packed.shape) == (T, H, (W + 7) // 8))
        fx, cx, fy, cy = intr[s]["cameraMatrix"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        E = np.array(cams[s]["optimized_extrinsics"])
        rows, nonempty = [], np.zeros(T, bool)
        for t in range(T):
            mask = np.unpackbits(packed[t], axis=1)[:, :W].astype(bool)  # per frame: no full-stack unpack
            ys, xs = np.nonzero(mask)
            nonempty[t] = len(xs) > 0
            if t >= n_store:
                rows.append(dict(t=t, in_view=False, present=bool(nonempty[t]))); continue
            pc = E @ gt[t][:, 3]
            if pc[2] <= 0:
                rows.append(dict(t=t, in_view=False, present=bool(nonempty[t]))); continue
            uv = K @ pc[:3]; x, y = uv[0] / uv[2], uv[1] / uv[2]
            in_view = 0 <= x < W and 0 <= y < H
            row = dict(t=t, in_view=bool(in_view), x=round(float(x), 1), y=round(float(y), 1), z=round(float(pc[2]), 3), present=bool(nonempty[t]))
            if nonempty[t] and in_view:
                d = float(np.min(np.hypot(xs - x, ys - y)))  # 0 when the centre is inside the mask
                row.update(dist_px=round(d, 1), inside_tol=bool(d <= tol), area_frac=round(len(xs) / (H * W), 5),
                           centroid_dist=round(float(np.hypot(xs.mean() - x, ys.mean() - y)), 1))
            rows.append(row)
        r.update(present_consistent=bool((nonempty == present).all()), frames_present=int(nonempty.sum()), frac_present=float(nonempty.mean()))
        iv = [x for x in rows if x["in_view"]]
        hit = [x for x in iv if x.get("present")]
        d = np.array([x["dist_px"] for x in hit]) if hit else np.zeros(0)
        r.update(gt_in_view=len(iv), present_when_in_view=len(hit), recall_in_view=round(len(hit) / max(len(iv), 1), 3),
                 inside_tol_rate=round(float(np.mean([x["inside_tol"] for x in hit])) if hit else 0.0, 3),
                 dist_px_median=round(float(np.median(d)), 1) if len(d) else None, dist_px_p90=round(float(np.percentile(d, 90)), 1) if len(d) else None,
                 far_frames_gt100px=int((d > 100).sum()), present_when_out_of_view=int(sum(1 for x in rows if not x["in_view"] and x.get("present"))),
                 mean_area_frac=round(float(np.mean([x["area_frac"] for x in hit])), 4) if hit else None, per_frame=rows)
        res["cams"][cam] = r
        print(f"  {ep[:38]:38s} {cam} T={T}/{n_store} shape_ok={r['shape_ok']} packed_ok={r['packed_shape_ok']} present_ok={r['present_consistent']} present={r['frames_present']}/{T} "
              f"gt_in_view={len(iv)} recall={r['recall_in_view']:.2f} inside{tol}px={r['inside_tol_rate']:.2f} dist med/p90={r['dist_px_median']}/{r['dist_px_p90']} far>100px={r['far_frames_gt100px']} area={r['mean_area_frac']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="motion_only")
    ap.add_argument("--root", default=None, help="default: SEEDSTUDY/<method>")
    ap.add_argument("--episodes", default="/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt")
    ap.add_argument("--episode", default=None)
    ap.add_argument("--tol", type=int, default=25)
    ap.add_argument("--out", default=None, help="default: <root>/check")
    a = ap.parse_args()
    root = a.root or f"{SEEDSTUDY}/{a.method}"
    out = a.out or f"{root}/check"
    os.makedirs(out, exist_ok=True)
    eps = [a.episode] if a.episode else [l.strip() for l in open(a.episodes) if l.strip()]
    for ep in eps:
        r = check_episode(ep, a.method, root, a.tol)
        json.dump(r, open(f"{out}/{ep}.json", "w"), indent=1)
    all_eps = [l.strip() for l in open(a.episodes) if l.strip()] if a.episode else eps
    summary = []
    for ep in all_eps:  # merge every per-episode json present (per-episode processes stay under the login-node CPU cap)
        f = f"{out}/{ep}.json"
        if os.path.exists(f):
            r = json.load(open(f))
            summary.append({k: v for k, v in r.items() if k != "cams"} | {c: {k: v for k, v in r["cams"][c].items() if k != "per_frame"} for c in r["cams"]})
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    ok = [all(r.get(c, {}).get("T_matches_store") and r[c].get("shape_ok") and r[c].get("packed_shape_ok") and r[c].get("present_consistent") for c in ("ext1", "ext2")) for r in summary]
    print(f"products ok: {sum(ok)}/{len(ok)} episodes checked; summary {out}/summary.json")


if __name__ == "__main__":
    main()
