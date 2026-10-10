#!/usr/bin/env python
"""DIAGNOSTIC (GT): for one idea root, per scene, the seeder's own per-frame verified 3D pick (verified_points.npz, base
frame) against the kinematic GT wrist-lens position (store cam/*.npz): n accepted, n > 0.20 m from the lens (the wrong-body
ceiling of the tracker gate, cf. seeding_study_round3.md appendix D), median point-to-lens distance; pooled totals; plus
the seeder's timing and pick statistics from seed_info.json. Writes <root>/vp_diag.json and prints a table.

    OMP_NUM_THREADS=1 python ideas_vp_diag.py --root .../ideas/dino_exemplar [--scenes smoke13_scenes.txt]
"""
import argparse, json, os
import numpy as np

STORE = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
SMOKE13 = "/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", default=SMOKE13)
    ap.add_argument("--dist_m", type=float, default=0.20)
    a = ap.parse_args()
    eps = [l.strip() for l in open(a.scenes) if l.strip()]
    rows, n_all, n_wrong = [], 0, 0
    for ep in eps:
        p = f"{a.root}/{ep}/verified_points.npz"
        row = dict(episode=ep)
        if not os.path.isfile(p):
            row.update(missing=True); rows.append(row); print(f"{ep[:40]:40s} MISSING verified_points.npz"); continue
        v = np.load(p)
        acc = v["accepted"].astype(bool); xyz = v["xyz"]
        d = []
        for t in np.where(acc)[0]:
            f = f"{STORE}/{ep}/dense/cam/{int(t):06d}.npz"
            if os.path.isfile(f):
                d.append(float(np.linalg.norm(xyz[t] - np.load(f)["pose"][:3, 3])))
        d = np.array(d)
        si_p = f"{a.root}/{ep}/seed_info.json"
        si = json.load(open(si_p)) if os.path.isfile(si_p) else {}
        row.update(T=int(len(acc)), n_accepted=int(acc.sum()), n_wrong=int((d > a.dist_m).sum()) if len(d) else 0,
                   frac_wrong=float((d > a.dist_m).mean()) if len(d) else None, median_m=float(np.median(d)) if len(d) else None,
                   p90_m=float(np.percentile(d, 90)) if len(d) else None,
                   timing=si.get("timing"), modes=si.get("summary", {}).get("modes"), n_tracks=si.get("summary", {}).get("n_tracks"),
                   windows_per_frame=si.get("summary", {}).get("windows_per_frame"),
                   best_sim_median={c: si["per_cam"][c].get("best_sim_median") for c in si.get("per_cam", {})},
                   mask_sources={c: si["per_cam"][c].get("mask_sources") for c in si.get("per_cam", {})},
                   frames_present={c: si["per_cam"][c].get("frames_present") for c in si.get("per_cam", {})})
        n_all += row["n_accepted"]; n_wrong += row["n_wrong"]
        rows.append(row)
        tm = row["timing"] or {}
        print(f"{ep[:40]:40s} T={row['T']:4d} picks={row['n_accepted']:4d} wrong={row['n_wrong']:4d} ({(row['frac_wrong'] or 0) * 100:5.1f}%) med={row['median_m']} "
              f"present={row['frames_present']} ms/frame={tm.get('ms_per_frame_total')} (motion {tm.get('ms_per_frame_motion_candidates')}, embed {tm.get('ms_per_frame_embed')}) "
              f"win/frame={row['windows_per_frame']} tracks={row['n_tracks']} src={row['mask_sources']}")
    agg = dict(n_scenes=len(rows), n_accepted_total=n_all, n_wrong_total=n_wrong, frac_wrong_pooled=(n_wrong / n_all) if n_all else None,
               dist_m=a.dist_m, mean_ms_per_frame=float(np.mean([r["timing"]["ms_per_frame_total"] for r in rows if r.get("timing")])) if any(r.get("timing") for r in rows) else None,
               mean_ms_embed=float(np.mean([r["timing"]["ms_per_frame_embed"] for r in rows if r.get("timing")])) if any(r.get("timing") for r in rows) else None,
               mean_ms_motion=float(np.mean([r["timing"]["ms_per_frame_motion_candidates"] for r in rows if r.get("timing")])) if any(r.get("timing") for r in rows) else None)
    json.dump(dict(note="DIAGNOSTIC (GT): seeder verified points vs kinematic GT lens", root=a.root, aggregate=agg, per_scene=rows), open(f"{a.root}/vp_diag.json", "w"), indent=1)
    print(json.dumps(agg, indent=1)); print("wrote", f"{a.root}/vp_diag.json")


if __name__ == "__main__":
    main()
