#!/usr/bin/env python
"""Re-evaluate the kinematic consistency flag of every RobotSeg birth-frame mask with the PointWorld
`optimized_extrinsics` instead of the raw DROID metadata extrinsics.

Same rule as segment_birth_frames.py: project the 54-point gripper model at the mask's frame, pad the
bbox of the visible points by 60 px, spill = fraction of mask pixels outside that box,
projection_disagree = spill > 0.2. Uses the AUTOMATIC RobotSeg mask (`<stem>_auto.png` where the
box fallback was stored, else `<stem>.png`), exactly what the v1 flag was computed on.

Outputs (never touches the v1 files):
  $OUT_ROOT/recheck_pointworld/recheck.csv       one row per mask: spill/flag raw vs pointworld, centroid shift
  $OUT_ROOT/recheck_pointworld/summary.md        rates raw vs pointworld, overall and per lab
  $OUT_ROOT/recheck_pointworld/sheet_still_disagree_pw.jpg   40 random b050 cameras still flagged under pointworld
  $OUT_ROOT/recheck_pointworld/sheet_fixed_by_pw.jpg         40 random b050 cameras flagged raw, clean under pointworld
Usage: recheck_projection_pointworld.py [--workers 8] [--limit N]
"""
import argparse
import collections
import csv
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import birth_frames as bf  # noqa: E402

RUN = f"{bf.OUT_ROOT}/masks_run"
OUT = f"{bf.OUT_ROOT}/recheck_pointworld"
PAD, SPILL_MAX = 60, 0.2


def box_of(uv, vis, pad):
    if vis.sum() == 0:
        return None
    x0, y0 = np.maximum(uv[vis].min(0) - pad, 0).astype(int)
    x1, y1 = np.minimum(uv[vis].max(0) + pad, [bf.W - 1, bf.H - 1]).astype(int)
    return x0, y0, x1, y1


def spill(mask, uv, vis):
    b = box_of(uv, vis, PAD)
    if b is None or mask.sum() == 0:
        return float("nan")
    x0, y0, x1, y1 = b
    return float(1 - mask[y0:y1 + 1, x0:x1 + 1].sum() / mask.sum())


def one(row, P_cam):
    ep, cam, t = row["episode"], row["cam"], int(row["frame"])
    stem = f"{cam}_f{t:05d}"
    p_auto = f"{RUN}/masks/{ep}/{stem}_auto.png"
    mpath = p_auto if os.path.isfile(p_auto) else f"{RUN}/masks/{ep}/{stem}.png"
    m = cv2.imread(mpath, 0)
    out = dict(episode=ep, cam=cam, which=row["which"], frame=t, serial=row["serial"],
               flag_v1=row["projection_disagree"] == "True")
    if m is None:
        out["error"] = "mask_missing"
        return out
    m = m > 0
    meta = json.load(open(f"{bf.META_DIR}/{ep}.json"))
    K = bf.load_zed_K(row["serial"])
    with np.load(f"{bf.STORE}/{ep}/dense/cam/{t:06d}.npz") as z:
        pose = z["pose"]
    Pb = (pose[:3, :3] @ P_cam.T).T + pose[:3, 3]
    res = {}
    for name in ("raw", "pointworld"):
        try:
            T_bc = bf.load_T_bc(ep, meta, cam, name)
        except Exception as e:  # noqa: BLE001
            out[f"error_{name}"] = str(e)[:80]
            continue
        frac, uv, vis = bf.visible_fraction(T_bc, K, Pb)
        sp = spill(m, uv, vis)
        c = uv[vis].mean(0) if vis.sum() else np.array([np.nan, np.nan])
        res[name] = (uv, vis)
        out[f"frac_{name}"] = float(frac)
        out[f"spill_{name}"] = sp
        out[f"flag_{name}"] = bool(not np.isnan(sp) and sp > SPILL_MAX)
        out[f"cx_{name}"], out[f"cy_{name}"] = float(c[0]), float(c[1])
        b = box_of(uv, vis, 0)
        out[f"box_{name}"] = list(map(int, b)) if b else None
    ys, xs = np.where(m)
    out["mask_cx"], out["mask_cy"] = float(xs.mean()), float(ys.mean())
    if "raw" in res and "pointworld" in res:
        out["shift_px"] = float(np.hypot(out["cx_raw"] - out["cx_pointworld"], out["cy_raw"] - out["cy_pointworld"]))
    return out


def _w(args):
    try:
        return one(*args)
    except Exception as e:  # noqa: BLE001
        return dict(episode=args[0]["episode"], cam=args[0]["cam"], which=args[0]["which"], error=f"exc:{e}"[:120])


def sheet(rows, path, cols=4, tile=(480, 270), max_tiles=40):
    tiles = []
    for r in rows[:max_tiles]:
        ep, cam, t = r["episode"], r["cam"], int(r["frame"])
        stem = f"{cam}_f{t:05d}"
        im = cv2.imread(f"{RUN}/frames/{ep}/{stem}.jpg")
        p_auto = f"{RUN}/masks/{ep}/{stem}_auto.png"
        m = cv2.imread(p_auto if os.path.isfile(p_auto) else f"{RUN}/masks/{ep}/{stem}.png", 0)
        if im is None or m is None:
            continue
        m = m > 0
        im[m] = (0.45 * im[m] + 0.55 * np.array([255, 80, 0])).astype(np.uint8)
        for name, col in (("raw", (0, 0, 255)), ("pointworld", (0, 255, 0))):
            b = r.get(f"box_{name}")
            if b:
                x0, y0, x1, y1 = b
                cv2.rectangle(im, (max(x0 - PAD, 0), max(y0 - PAD, 0)), (min(x1 + PAD, bf.W - 1), min(y1 + PAD, bf.H - 1)), col, 3)
        cv2.putText(im, f"{ep[:34]} {cam} f{t}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(im, f"spill raw {r['spill_raw']:.2f} (red)  PW {r['spill_pointworld']:.2f} (green)  shift {r['shift_px']:.0f}px",
                    (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(cv2.resize(im, tile))
    if not tiles:
        return 0
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    return len(tiles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = [r for r in csv.DictReader(open(f"{bf.OUT_ROOT}/birth_masks.csv")) if r["status"] == "ok"]
    if args.limit:
        rows = rows[: args.limit]
    P_cam = bf.gripper_points_cam()
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(_w, [(r, P_cam) for r in rows], chunksize=16))
    keys = ["episode", "cam", "which", "frame", "serial", "flag_v1", "flag_raw", "flag_pointworld", "spill_raw", "spill_pointworld",
            "frac_raw", "frac_pointworld", "shift_px", "mask_cx", "mask_cy", "cx_raw", "cy_raw", "cx_pointworld", "cy_pointworld",
            "error", "error_raw", "error_pointworld"]
    with open(f"{OUT}/recheck.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in res:
            w.writerow({k: r.get(k, "") for k in keys})
    ok = [r for r in res if "error" not in r and "flag_raw" in r and "flag_pointworld" in r]
    L = ["# Kinematic projection vs RobotSeg mask: raw DROID extrinsics vs PointWorld optimized_extrinsics", ""]
    L.append(f"masks: {len(res)}; evaluated with both calibrations: {len(ok)}; errors: {len(res) - len(ok)} "
             f"({dict(collections.Counter(r.get('error') or r.get('error_pointworld') or r.get('error_raw') for r in res if r not in ok))})")
    rep = sum(r["flag_v1"] == r["flag_raw"] for r in ok)
    L.append(f"v1 flag reproduced with the raw extrinsics: {rep}/{len(ok)}")
    L.append("")
    L.append("rule: projection_disagree = > 20 % of the automatic mask's pixels outside the projected 54-point box padded by 60 px")
    L.append("")
    L.append("| set | masks | disagree raw | disagree pointworld | raw only | pointworld only | both |")
    L.append("|---|---|---|---|---|---|---|")
    for which in ("b050", "b100", "all"):
        s = [r for r in ok if which == "all" or r["which"] == which]
        a = sum(r["flag_raw"] for r in s); b = sum(r["flag_pointworld"] for r in s)
        ro = sum(r["flag_raw"] and not r["flag_pointworld"] for r in s)
        po = sum(r["flag_pointworld"] and not r["flag_raw"] for r in s)
        bo = sum(r["flag_pointworld"] and r["flag_raw"] for r in s)
        L.append(f"| {which} | {len(s)} | {a} ({100*a/max(len(s),1):.1f}%) | {b} ({100*b/max(len(s),1):.1f}%) | {ro} | {po} | {bo} |")
    L.append("")
    L.append("## Per lab (b050)")
    L.append("")
    L.append("| lab | masks | disagree raw | disagree pointworld |")
    L.append("|---|---|---|---|")
    bylab = collections.defaultdict(list)
    for r in ok:
        if r["which"] == "b050":
            bylab[r["episode"].split("+")[0]].append(r)
    for lab, s in sorted(bylab.items(), key=lambda kv: -len(kv[1])):
        a = sum(r["flag_raw"] for r in s); b = sum(r["flag_pointworld"] for r in s)
        L.append(f"| {lab} | {len(s)} | {a} ({100*a/len(s):.1f}%) | {b} ({100*b/len(s):.1f}%) |")
    L.append("")
    sh = [r["shift_px"] for r in ok if r["which"] == "b050" and "shift_px" in r and not np.isnan(r["shift_px"])]
    shf = [r["shift_px"] for r in ok if r["which"] == "b050" and r["flag_raw"] and "shift_px" in r and not np.isnan(r["shift_px"])]
    shu = [r["shift_px"] for r in ok if r["which"] == "b050" and not r["flag_raw"] and "shift_px" in r and not np.isnan(r["shift_px"])]
    L.append("## Projected gripper-centroid shift raw -> pointworld at the b050 frame (px)")
    L.append("")
    for name, xs in (("all", sh), ("flagged raw", shf), ("unflagged raw", shu)):
        if xs:
            L.append(f"- {name}: n={len(xs)} median {np.median(xs):.1f}, p10 {np.percentile(xs,10):.1f}, p90 {np.percentile(xs,90):.1f}, "
                     f"> 30 px: {100*np.mean(np.array(xs) > 30):.1f}%")
    L.append("")
    # distance mask centroid -> projected centroid, a flag-free view of the same thing
    for name in ("raw", "pointworld"):
        d = [np.hypot(r["mask_cx"] - r[f"cx_{name}"], r["mask_cy"] - r[f"cy_{name}"]) for r in ok
             if r["which"] == "b050" and not np.isnan(r[f"cx_{name}"])]
        L.append(f"- mask centroid to projected centroid, {name}: median {np.nanmedian(d):.1f} px, p90 {np.nanpercentile(d,90):.1f} px, "
                 f"> 100 px: {100*np.nanmean(np.array(d) > 100):.1f}%")
    L.append("")
    scenes_pw = sorted(set(r["episode"] for r in ok if r["flag_pointworld"]))
    scenes_raw = sorted(set(r["episode"] for r in ok if r["flag_raw"]))
    L.append(f"scenes with >= 1 flagged camera: raw {len(scenes_raw)}, pointworld {len(scenes_pw)} "
             f"(of {len(set(r['episode'] for r in ok))}); pointworld list in `scenes_still_disagree_pointworld.txt`")
    open(f"{OUT}/scenes_still_disagree_pointworld.txt", "w").write("\n".join(scenes_pw) + "\n")
    random.seed(0)
    still = [r for r in ok if r["which"] == "b050" and r["flag_pointworld"]]
    fixed = [r for r in ok if r["which"] == "b050" and r["flag_raw"] and not r["flag_pointworld"]]
    n1 = sheet(random.sample(still, min(40, len(still))), f"{OUT}/sheet_still_disagree_pw.jpg")
    n2 = sheet(random.sample(fixed, min(40, len(fixed))), f"{OUT}/sheet_fixed_by_pw.jpg")
    L.append("")
    L.append(f"contact sheets: `sheet_still_disagree_pw.jpg` ({n1} random b050 cameras still flagged under pointworld; red box = raw, "
             f"green = pointworld, orange = automatic mask), `sheet_fixed_by_pw.jpg` ({n2} flagged raw, clean under pointworld)")
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
