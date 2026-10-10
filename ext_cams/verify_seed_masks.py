#!/usr/bin/env python
"""Product check for a seeding method's masks on a scene list: both npz files per scene exist, unpack to (T,720,1280) with
T = store frame count, frames_present == union.any; optional IoU against reference masks (RAIL validated box+click).

    OMP_NUM_THREADS=1 python verify_seed_masks.py --method motion_sam3 [--scenes smoke13_scenes.txt]
"""
import argparse, glob, json, os, sys
import numpy as np

STORE = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
RAW = "/gpfs/scratch/etur59/koc821022/vggt_cache/raw"
REF = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/gripper_sam3"


def unpack(path):
    z = np.load(path)
    T, H, W = [int(v) for v in z["shape"]]
    u = np.unpackbits(z["union"], axis=2)[:, :, :W].astype(bool)
    return u, z["frames_present"], (T, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="motion_sam3")
    ap.add_argument("--root", default="/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy")
    ap.add_argument("--scenes", default="/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt")
    a = ap.parse_args()
    eps = [l.strip() for l in open(a.scenes) if l.strip()]
    ok_all, rows = True, []
    for ep in eps:
        meta = json.load(open(f"{RAW}/{ep}/metadata_{ep}.json"))
        T_store = len(glob.glob(f"{STORE}/{ep}/dense/cam/*.npz"))
        d = f"{a.root}/{a.method}/{ep}"
        si = os.path.join(d, "seed_info.json")
        info = json.load(open(si)) if os.path.exists(si) else {}
        for cam in ("ext1", "ext2"):
            s = meta[f"{cam}_cam_serial"]
            p = f"{d}/{cam}_{s}__{a.method}_masks.npz"
            row = dict(ep=ep, cam=cam, path=p)
            if not os.path.exists(p):
                row.update(status="MISSING"); ok_all = False; rows.append(row); continue
            try:
                u, fp, shp = unpack(p)
                good = shp == (T_store, 720, 1280) and u.shape == shp and fp.shape == (shp[0],) and np.array_equal(fp, u.any(axis=(1, 2)))
                row.update(status="OK" if good else "BAD", T=shp[0], T_store=T_store, n_present=int(fp.sum()),
                           first=int(np.argmax(fp)) if fp.any() else None, last=int(shp[0] - 1 - np.argmax(fp[::-1])) if fp.any() else None,
                           mean_area=float(u[fp].mean()) if fp.any() else 0.0, size_kb=os.path.getsize(p) // 1024)
                ci = info.get("cams", {}).get(cam, {})
                row.update(prompt=ci.get("prompt_frame"), reprompts=ci.get("n_reprompts"), rejected=len(ci.get("rejected", [])),
                           drops=len(ci.get("drops", [])), t_track=ci.get("t_track"), fps=ci.get("fps_track"))
                ok_all &= good
                ref = f"{REF}/{ep}/{cam}_{s}__gripper_boxclick_masks.npz"
                if os.path.exists(ref):
                    r, rfp, _ = unpack(ref)
                    both = fp & rfp
                    iou = [float((u[t] & r[t]).sum() / max((u[t] | r[t]).sum(), 1)) for t in np.nonzero(both)[0]]
                    cov = [float((u[t] & r[t]).sum() / max(r[t].sum(), 1)) for t in np.nonzero(both)[0]]
                    prec = [float((u[t] & r[t]).sum() / max(u[t].sum(), 1)) for t in np.nonzero(both)[0]]
                    row.update(ref_frames=int(rfp.sum()), both=int(both.sum()), iou_med=float(np.median(iou)) if iou else None,
                               ref_cov_med=float(np.median(cov)) if cov else None, prec_med=float(np.median(prec)) if prec else None,
                               present_when_ref=float(both.sum() / max(rfp.sum(), 1)))
            except Exception as e:  # noqa: BLE001
                row.update(status=f"ERROR {e!r}"); ok_all = False
            rows.append(row)
        if info:
            rows[-1]["t_total"] = info.get("t_total")
    for r in rows:
        print(json.dumps(r))
    print("ALL_OK" if ok_all else "PROBLEMS", f"{sum(r.get('status') == 'OK' for r in rows)}/{len(rows)} files OK")
    json.dump(rows, open(f"{a.root}/{a.method}/verify_{a.method}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
