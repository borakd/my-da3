#!/usr/bin/env python3
"""Build CUT3R "pseudo-scenes" whose cam npz carry VGGT-Omega anchored wrist poses.

For every episode a store-shaped scene is produced under
    <out>/<EP>/dense/{rgb, depth, sky_mask, outlier_mask}   -> symlinks to the store
    <out>/<EP>/dense/cam/%06d.npz  'intrinsic' = store intrinsic (per frame),
                                   'pose'      = VGGT c2w with translation * s
so infer_and_eval_worker_ray.py --scenes_root <out> --gt_scenes_root <store> feeds
CUT3R's ray channel (gt / prev_gt conditioning) from VGGT poses while the
evaluator still scores against the real store GT.

Inputs
  --variant_preds <root>/<variant>/preds   per episode camera/%06d.npz written by
        extract_vggt_omega.py: 'pose' = c2w (4x4) in VGGT's normalised scale,
        'intrinsics' (evaluator spelling; ignored here — the store K is used).
        Frames are matched to the store POSITIONALLY (sorted index).
  <root>/<variant>/meta/<EP>.json          the t=0 forward on [wrist0, ext1, ext2]
        (needed for --scale calib / the s_calib column). extract_vggt_omega.py's
        layout is read directly: frames[0].{d_w0_e1,d_w0_e2,d_e1_e2} (VGGT-scale
        anchor distances), else 't0_pose_enc' [S0,9] + 't0_order' names. Other
        accepted layouts, first hit wins (searched recursively through the json):
          'centers'/'t0_centers'/'cam_centers'      [3,3] camera centres (world)
          'c2w'/'t0_c2w'/'poses_c2w'                [3,4,4] c2w (or 3 dicts by name)
          'extr'/'t0_extr'/'extrinsic'/'extrinsics' [3,3,4] or [3,4,4] world->cam
          'pose_enc'/'t0_pose_enc'                  [3,9] absT_quaR_FoV (VGGT)
        Camera order is [wrist0, ext1, ext2] unless a 'names'/'order' list says
        otherwise. Missing meta -> s_calib = NaN (calib scaling then fails loudly).
  --manifest smoke13_manifest.json  {'episodes': [{'ep', 'calib_dists': {wrist0_ext1,
        wrist0_ext2, ext1_ext2}, ...}]} (layout written 2026-09-07). calib_dists is
        used directly when present; else *_xyz / *_extrinsics_xyzrpy /
        *_cam_extrinsics positions in the record; else
        --raw_root/<EP>/metadata_<EP>.json ('{wrist,ext1,ext2}_cam_extrinsics' =
        [x,y,z,rx,ry,rz], robot base frame, metres).
        VERIFIED on RAIL: store cam/000000.npz pose translation == the metadata
        wrist_cam_extrinsics xyz, so the store GT world IS the robot base frame.

Scaling (s multiplies VGGT translations; rotations untouched)
  raw     s = 1 (VGGT normalised scale as-is)
  calib   s = median over the 3 camera pairs {wrist0-ext1, wrist0-ext2, ext1-ext2}
              of calibrated metric distance / VGGT t=0 distance   (calibration only)
  oracle  s = Umeyama Sim(3) scale between VGGT and store GT wrist camera centres
              *** GT-USING DIAGNOSTIC — not an honest arm; label it so ***
Both s_calib and s_oracle are computed for every episode (NaN when their inputs
are missing) and written to <out>/scale_report.csv with ratio = s_calib/s_oracle.

CPU only, seconds per episode. Safe to re-run (symlinks/npz are overwritten).
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

STORE_DEFAULT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
RAW_DEFAULT = "/gpfs/scratch/etur59/koc821022/vggt_cache/raw"
SYMLINK_DIRS = ("rgb", "depth", "sky_mask", "outlier_mask")
CAM_ORDER = ("wrist0", "ext1", "ext2")
PAIRS = ((0, 1), (0, 2), (1, 2))


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def quat_to_rot(q):
    """VGGT pose_enc quaternion order is (x, y, z, w) (pose_enc[:, 3:7])."""
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def umeyama_scale(src, dst):
    """Sim(3) scale s minimising ||dst - (s R src + t)|| (Umeyama 1991).
    src, dst: [N,3]. Returns NaN for degenerate inputs (N<2 or zero spread)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    n = len(src)
    if n < 2 or len(dst) != n:
        return float("nan")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = (xs ** 2).sum() / n
    if var_s <= 0:
        return float("nan")
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    return float(np.trace(np.diag(D) @ S) / var_s)


def _find_key(obj, names):
    """Depth-first search for the first dict key in `names` anywhere in obj."""
    if isinstance(obj, dict):
        for k in names:
            if k in obj:
                return obj[k]
        for v in obj.values():
            r = _find_key(v, names)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, names)
            if r is not None:
                return r
    return None


def t0_vggt_dists(meta):
    """[d(w0,e1), d(w0,e2), d(e1,e2)] in VGGT scale from the t=0 forward.
    extract_vggt_omega.py writes them directly as frames[0].d_w0_e1/d_w0_e2/d_e1_e2
    (anchor_stats); otherwise they are derived from the camera centres."""
    frames = meta.get("frames") if isinstance(meta, dict) else None
    f0 = None
    if isinstance(frames, list) and frames:
        f0 = next((f for f in frames if isinstance(f, dict) and f.get("t", 0) == 0), frames[0])
    if isinstance(f0, dict) and all(k in f0 for k in ("d_w0_e1", "d_w0_e2", "d_e1_e2")):
        return np.array([f0["d_w0_e1"], f0["d_w0_e2"], f0["d_e1_e2"]], np.float64)
    return pair_dists(t0_centers_from_meta(meta))


def t0_centers_from_meta(meta):
    """Camera centres [3,3] in VGGT world for [wrist0, ext1, ext2] (see docstring)."""
    def _reorder(arr):
        arr = np.asarray(arr, np.float64)
        names = _find_key(meta, ("t0_order", "names", "order", "cam_names"))
        if names and len(names) == len(arr):
            idx = []
            for want in CAM_ORDER:
                hit = [i for i, nm in enumerate(names) if want in str(nm).lower()
                       or (want == "wrist0" and "wrist" in str(nm).lower())]
                if not hit:
                    return arr[:3]
                idx.append(hit[0])
            arr = arr[idx]
        return arr[:3]

    c = _find_key(meta, ("centers", "t0_centers", "cam_centers", "camera_centers"))
    if c is not None:
        return _reorder(c)
    c2w = _find_key(meta, ("c2w", "t0_c2w", "poses_c2w", "cam2world", "pose_c2w"))
    if c2w is not None:
        if isinstance(c2w, dict):  # {'wrist0': 4x4, 'ext1': 4x4, 'ext2': 4x4}
            keys = {k.lower(): k for k in c2w}
            c2w = [c2w[keys[[k for k in keys if want in k or (want == "wrist0" and "wrist" in k)][0]]]
                   for want in CAM_ORDER]
        c2w = np.asarray(c2w, np.float64)
        return _reorder(c2w[:, :3, 3])
    extr = _find_key(meta, ("extr", "t0_extr", "extrinsic", "extrinsics", "w2c"))
    if extr is not None:
        extr = np.asarray(extr, np.float64)
        R, t = extr[:, :3, :3], extr[:, :3, 3]
        centers = -np.einsum("nji,nj->ni", R, t)  # -R^T t
        return _reorder(centers)
    pe = _find_key(meta, ("pose_enc", "t0_pose_enc", "pose_encoding"))
    if pe is not None:
        pe = np.asarray(pe, np.float64).reshape(-1, 9)
        centers = np.stack([-quat_to_rot(p[3:7]).T @ p[:3] for p in pe])
        return _reorder(centers)
    raise KeyError("no t=0 camera information found in meta json "
                   "(looked for centers/c2w/extr/pose_enc)")


def manifest_entry(manifest, ep):
    """Locate the per-episode record: {EP: {...}}, {'episodes': [{'ep': EP, ...}]},
    {'episodes': {EP: {...}}} or a bare list of records."""
    if isinstance(manifest, dict):
        if ep in manifest and isinstance(manifest[ep], dict):
            return manifest[ep]
        eps = manifest.get("episodes")
        if isinstance(eps, dict):
            return eps.get(ep)
        manifest = eps
    if isinstance(manifest, list):
        for e in manifest:
            if isinstance(e, dict) and ep in (e.get("ep"), e.get("episode"), e.get("name")):
                return e
    return None


def calib_dists_from_manifest(entry):
    """[d(w0,e1), d(w0,e2), d(e1,e2)] from a smoke13_manifest episode record
    ('calib_dists' dict, else *_xyz / *_extrinsics_xyzrpy positions), or None."""
    if entry is None:
        return None
    cd = _find_key(entry, ("calib_dists",))
    if isinstance(cd, dict) and all(k in cd for k in ("wrist0_ext1", "wrist0_ext2", "ext1_ext2")):
        return np.array([cd["wrist0_ext1"], cd["wrist0_ext2"], cd["ext1_ext2"]], np.float64)
    pos = []
    for cam in ("wrist", "ext1", "ext2"):
        v = _find_key(entry, (f"{cam}0_xyz", f"{cam}_xyz", f"{cam}_extrinsics_xyzrpy",
                              f"{cam}_cam_extrinsics", f"{cam}0_extrinsics_xyzrpy"))
        if v is None:
            return None
        pos.append(np.asarray(v, np.float64)[:3])
    return pair_dists(np.stack(pos))


def calib_positions(ep, manifest, raw_root):
    """Metric camera positions [3,3] for [wrist0, ext1, ext2] from the calibrated
    episode metadata (robot base frame). Read from raw metadata_<EP>.json (the
    manifest path is tried first in build_episode via calib_dists_from_manifest)."""
    entry = manifest_entry(manifest, ep)
    keys = ("wrist_cam_extrinsics", "ext1_cam_extrinsics", "ext2_cam_extrinsics")
    vals = [_find_key(entry, (k,)) if entry is not None else None for k in keys]
    src = "manifest"
    if any(v is None for v in vals):
        meta_path = os.path.join(raw_root, ep, f"metadata_{ep}.json")
        if not os.path.isfile(meta_path):
            return None, f"no calibrated extrinsics in manifest and no {meta_path}"
        md = json.load(open(meta_path))
        vals = [md.get(k) for k in keys]
        src = meta_path
        if any(v is None for v in vals):
            return None, f"{meta_path} lacks {keys}"
    pos = np.asarray([np.asarray(v, np.float64)[:3] for v in vals])
    return pos, src


def pair_dists(p):
    return np.array([np.linalg.norm(p[a] - p[b]) for a, b in PAIRS])


# ----------------------------------------------------------------------------
def build_episode(ep, args, manifest, report_rows):
    store = os.path.join(args.store_root, ep, "dense")
    pred_cam = os.path.join(args.variant_preds, ep, "camera")
    variant_root = os.path.dirname(os.path.abspath(args.variant_preds.rstrip("/")))
    meta_path = os.path.join(variant_root, "meta", f"{ep}.json")
    out_dense = os.path.join(args.out, ep, "dense")

    store_cams = sorted(glob.glob(os.path.join(store, "cam", "*.npz")))
    vggt_cams = sorted(glob.glob(os.path.join(pred_cam, "*.npz")))
    if not store_cams:
        raise FileNotFoundError(f"{ep}: no store cam npz under {store}/cam")
    if not vggt_cams:
        raise FileNotFoundError(f"{ep}: no VGGT camera npz under {pred_cam}")
    n = min(len(store_cams), len(vggt_cams))
    if len(store_cams) != len(vggt_cams):
        msg = (f"{ep}: frame count mismatch store={len(store_cams)} "
               f"vggt={len(vggt_cams)} (positional matching)")
        if not args.allow_partial:
            raise RuntimeError(msg + " — pass --allow_partial to truncate to the shorter")
        print("WARNING " + msg + f"; truncating to {n}", flush=True)

    vggt_c2w = np.stack([np.load(p)["pose"].astype(np.float64) for p in vggt_cams[:n]])
    gt_c2w = np.stack([np.load(p)["pose"].astype(np.float64) for p in store_cams[:n]])
    store_K = [np.load(p)["intrinsic"].astype(np.float32) for p in store_cams[:n]]
    assert vggt_c2w.shape == (n, 4, 4), vggt_c2w.shape

    # --- s_oracle: GT-USING (Umeyama scale VGGT centres -> GT centres) ----------
    s_oracle = umeyama_scale(vggt_c2w[:, :3, 3], gt_c2w[:, :3, 3])

    # --- s_calib: calibrated metric distances / VGGT t=0 distances --------------
    s_calib = float("nan")
    d_cal = d_vggt = np.full(3, np.nan)
    d_cal_m = calib_dists_from_manifest(manifest_entry(manifest, ep))
    if d_cal_m is not None:
        calib_src = "manifest:calib_dists"
    else:
        calib_pos, calib_src = calib_positions(ep, manifest, args.raw_root)
        if calib_pos is None:
            print(f"WARNING {ep}: {calib_src}", flush=True)
            calib_src = "-"
        else:
            d_cal_m = pair_dists(calib_pos)
    if d_cal_m is None:
        pass
    elif not os.path.isfile(meta_path):
        print(f"WARNING {ep}: no VGGT t=0 meta {meta_path} -> s_calib=NaN", flush=True)
    else:
        try:
            d_vggt = t0_vggt_dists(json.load(open(meta_path)))
            d_cal = d_cal_m
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = d_cal / d_vggt
            ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
            s_calib = float(np.median(ratios)) if len(ratios) else float("nan")
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"WARNING {ep}: cannot derive s_calib from {meta_path}: {e!r}", flush=True)

    s = {"raw": 1.0, "calib": s_calib, "oracle": s_oracle}[args.scale]
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError(f"{ep}: scale '{args.scale}' unavailable (s={s}) — "
                           "see WARNINGs above")

    # --- write the pseudo-scene ---------------------------------------------------
    os.makedirs(out_dense, exist_ok=True)
    for d in SYMLINK_DIRS:
        src = os.path.join(store, d)
        dst = os.path.join(out_dense, d)
        if not os.path.isdir(src):
            print(f"WARNING {ep}: store has no {src}; skipping symlink", flush=True)
            continue
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    cam_out = os.path.join(out_dense, "cam")
    os.makedirs(cam_out, exist_ok=True)
    for stale in glob.glob(os.path.join(cam_out, "*.npz")):
        os.remove(stale)
    for i in range(n):
        pose = vggt_c2w[i].copy()
        pose[:3, 3] *= s
        np.savez(os.path.join(cam_out, f"{i:06d}.npz"),
                 intrinsic=store_K[i], pose=pose.astype(np.float32))

    ratio = s_calib / s_oracle if (np.isfinite(s_calib) and np.isfinite(s_oracle) and s_oracle != 0) else float("nan")
    report_rows.append(dict(
        ep=ep, s_calib=s_calib, s_oracle=s_oracle, ratio=ratio,
        scale_mode=args.scale, s_applied=s, n_frames=n,
        d_calib_w0e1=d_cal[0], d_calib_w0e2=d_cal[1], d_calib_e1e2=d_cal[2],
        d_vggt_w0e1=d_vggt[0], d_vggt_w0e2=d_vggt[1], d_vggt_e1e2=d_vggt[2],
        calib_source=calib_src,
    ))
    print(f"{ep}: n={n} s_calib={s_calib:.4f} s_oracle={s_oracle:.4f} "
          f"ratio={ratio:.3f} applied[{args.scale}]={s:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant_preds", required=True, help="<root>/<variant>/preds (VGGT camera npz per episode)")
    ap.add_argument("--scale", required=True, choices=["raw", "calib", "oracle"])
    ap.add_argument("--manifest", default="/gpfs/scratch/etur59/koc821022/vggt_cache/smoke13_manifest.json")
    ap.add_argument("--out", required=True, help=".../pseudo_scenes/<variant>_<scale>/")
    ap.add_argument("--store_root", default=STORE_DEFAULT)
    ap.add_argument("--raw_root", default=RAW_DEFAULT,
                    help="fallback for calibrated extrinsics: <raw_root>/<EP>/metadata_<EP>.json")
    ap.add_argument("--scenes", default=None, help="txt of episode names (default: every episode dir under --variant_preds)")
    ap.add_argument("--allow_partial", action="store_true", help="truncate to the shorter of store/VGGT frame lists instead of failing")
    args = ap.parse_args()

    manifest = None
    if os.path.isfile(args.manifest):
        manifest = json.load(open(args.manifest))
    else:
        print(f"NOTE: manifest {args.manifest} not found; calibrated extrinsics come from "
              f"{args.raw_root}/<EP>/metadata_<EP>.json", flush=True)

    if args.scenes:
        eps = [ln.strip() for ln in open(args.scenes) if ln.strip()]
    else:
        eps = sorted(os.path.basename(p.rstrip("/")) for p in glob.glob(os.path.join(args.variant_preds, "*"))
                     if os.path.isdir(os.path.join(p, "camera")))
    if not eps:
        sys.exit(f"no episodes found under {args.variant_preds}")

    os.makedirs(args.out, exist_ok=True)
    rows, failures = [], []
    for ep in eps:
        try:
            build_episode(ep, args, manifest, rows)
        except Exception as e:  # noqa: BLE001
            failures.append((ep, repr(e)))
            print(f"FAIL {ep}: {e!r}", flush=True)

    report = os.path.join(args.out, "scale_report.csv")
    fields = ["ep", "s_calib", "s_oracle", "ratio", "scale_mode", "s_applied", "n_frames",
              "d_calib_w0e1", "d_calib_w0e2", "d_calib_e1e2",
              "d_vggt_w0e1", "d_vggt_w0e2", "d_vggt_e1e2", "calib_source"]
    with open(report, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {report} ({len(rows)} episodes, {len(failures)} failures)"
          + (" — scale=oracle is a GT-USING DIAGNOSTIC" if args.scale == "oracle" else ""), flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
