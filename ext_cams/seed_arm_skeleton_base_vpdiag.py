#!/usr/bin/env python
"""DIAGNOSTIC ONLY (kinematic GT): for the arm_skeleton_base seeder's verified_points.npz, the distance of the per-frame
3D pick to the GT wrist-lens position, per scene, for BOTH the raw end-effector leaf (xyz_leaf) and the declared lens-back
point (xyz, what the tracker's wrong-body gate consumes). Reproduces the round-3 judge's "verified points > 0.20 m from
the lens" ceiling. The method never sees these numbers.

    OMP_NUM_THREADS=1 python seed_arm_skeleton_base_vpdiag.py --root .../ideas/arm_skeleton_base [--out vp_diag.json]
"""
import argparse
import collections
import json
import os

import numpy as np

STORE = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
SMOKE13 = "/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--episodes", default=SMOKE13)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dist_m", type=float, default=0.20)
    a = ap.parse_args()
    rows, pool_x, pool_l = [], [], []
    for ep in [l.strip() for l in open(a.episodes) if l.strip()]:
        p = os.path.join(a.root, ep, "verified_points.npz")
        if not os.path.isfile(p):
            rows.append(dict(episode=ep, missing=True))
            print(f"{ep[:40]:40s} MISSING")
            continue
        vp = np.load(p)
        T = len(vp["frames"])
        acc = np.nonzero(vp["accepted"])[0]
        dx, dl, src = [], [], collections.Counter()
        for t in acc:
            g = np.load(f"{STORE}/{ep}/dense/cam/{int(t):06d}.npz")["pose"][:3, 3]
            dx.append(float(np.linalg.norm(vp["xyz"][t] - g)))
            dl.append(float(np.linalg.norm(vp["xyz_leaf"][t] - g)) if "xyz_leaf" in vp else np.nan)
            src[str(vp["source"][t]) if "source" in vp else "?"] += 1
        dx, dl = np.array(dx), np.array(dl)
        pool_x.extend(dx.tolist()); pool_l.extend(dl[np.isfinite(dl)].tolist())
        r = dict(episode=ep, T=T, accepted=int(len(acc)), frac_accepted=float(len(acc) / T),
                 xyz_median_m=float(np.median(dx)) if len(dx) else None, xyz_p90_m=float(np.percentile(dx, 90)) if len(dx) else None,
                 xyz_n_gt=int((dx > a.dist_m).sum()), xyz_frac_gt=float((dx > a.dist_m).mean()) if len(dx) else None,
                 leaf_median_m=float(np.nanmedian(dl)) if len(dl) else None, leaf_n_gt=int((dl > a.dist_m).sum()), leaf_frac_gt=float(np.nanmean(dl > a.dist_m)) if len(dl) else None,
                 back_ok=int(vp["back_ok"].sum()) if "back_ok" in vp else None, source_counts=dict(src))
        rows.append(r)
        print(f"{ep[:40]:40s} acc {len(acc):4d}/{T:4d}  xyz med {r['xyz_median_m']:.3f} p90 {r['xyz_p90_m']:.3f} >{a.dist_m}: {r['xyz_n_gt']:4d} ({100 * r['xyz_frac_gt']:.0f}%) | "
              f"leaf med {r['leaf_median_m']:.3f} >{a.dist_m}: {r['leaf_n_gt']:4d} ({100 * r['leaf_frac_gt']:.0f}%)  src {dict(src)}")
    px, pl = np.array(pool_x), np.array(pool_l)
    agg = dict(n_points=int(len(px)), xyz_median_m=float(np.median(px)) if len(px) else None, xyz_p90_m=float(np.percentile(px, 90)) if len(px) else None,
               xyz_n_gt=int((px > a.dist_m).sum()), xyz_frac_gt=float((px > a.dist_m).mean()) if len(px) else None,
               leaf_median_m=float(np.median(pl)) if len(pl) else None, leaf_n_gt=int((pl > a.dist_m).sum()), leaf_frac_gt=float((pl > a.dist_m).mean()) if len(pl) else None,
               dist_m=a.dist_m, note="GT diagnostic only; round-3 verified_motion ceiling was 868/3202 = 27.1% of verified points > 0.20 m from the lens")
    print("POOLED", json.dumps(agg))
    out = a.out or os.path.join(a.root, "vp_diag.json")
    json.dump(dict(root=a.root, aggregate=agg, per_scene=rows), open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
