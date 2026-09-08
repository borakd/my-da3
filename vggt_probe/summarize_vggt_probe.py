#!/usr/bin/env python3
"""Collect VGGT-Omega probe results into one table (login-node safe: no torch).

For every <out_root>/<variant>/eval/<EP>/eval_depth_pose_metrics.csv, take the
'ALL,MEAN' row (absrel,a1,ate,rpe_trans,rpe_rot) and join the run stats from
<out_root>/<variant>/meta/<EP>.json (T, batch, s/frame, peak GiB, parity, skip note).

  python summarize_vggt_probe.py [--out_root R] [--csv out.csv] [--variants a b ...]
"""
import argparse
import csv
import glob
import json
import os

DEFAULT_OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe"
COLS = ["variant", "ep", "T", "absrel", "a1", "ate", "rpe_trans", "rpe_rot",
        "batch", "s_per_frame", "peak_gib", "parity_pose_maxdiff", "skipped"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--csv", default=None, help="write the table here (default: print only)")
    ap.add_argument("--variants", nargs="*", default=None)
    a = ap.parse_args()
    variants = a.variants or sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(a.out_root, "*"))
        if os.path.isdir(os.path.join(p, "meta")))
    rows = []
    for v in variants:
        for mp in sorted(glob.glob(os.path.join(a.out_root, v, "meta", "*.json"))):
            ep = os.path.basename(mp)[:-5]
            try:
                m = json.load(open(mp))
            except Exception:
                m = {}
            row = {"variant": v, "ep": ep, "T": m.get("T"), "batch": m.get("batch_used"),
                   "s_per_frame": m.get("fwd_wall_s_per_frame_mean"), "peak_gib": m.get("peak_mem_alloc_gib"),
                   "parity_pose_maxdiff": (m.get("batch_parity") or {}).get("pose_max_abs_diff"),
                   "skipped": (m.get("skipped") or "")[:40]}
            c = os.path.join(a.out_root, v, "eval", ep, "eval_depth_pose_metrics.csv")
            if os.path.isfile(c):
                for r in csv.DictReader(open(c)):
                    if r["camera_id"] == "ALL" and r["local_timestep"] == "MEAN":
                        row.update({k: float(r[k]) for k in ("absrel", "a1", "ate", "rpe_trans", "rpe_rot")})
            rows.append(row)
    fmt = lambda x: (f"{x:.4f}" if isinstance(x, float) else ("" if x is None else str(x)))
    print("  ".join(f"{c:>18s}" if c in ("variant",) else f"{c:>10s}" for c in COLS if c != "ep") + "  ep")
    for r in rows:
        print("  ".join(f"{fmt(r.get(c)):>18s}" if c == "variant" else f"{fmt(r.get(c)):>10s}"
                        for c in COLS if c != "ep") + f"  {r['ep']}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in COLS})
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
