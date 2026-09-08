#!/usr/bin/env python3
"""Make the C1-control context dir: exterior-camera frame 0 (or, with --which last,
frame T_store-1 -> C1b) per episode, 320x180.

Writes <out>/<EP>/00_ext1.png and <out>/<EP>/01_ext2.png (sorted order = ext1,
ext2), decoded from the raw DROID MP4s (<raw_root>/<EP>/recordings/MP4/<serial>.mp4,
frame 0 == store frame 0 — verified) and resized 1280x720 -> 320x180 with
cv2.INTER_AREA (16:9 kept, so infer_and_eval_worker_ray.py --context_dir resizes
them exactly like the wrist frames).

Camera serials come from --raw_paths (smoke13_raw_paths.json: {EP: {ext1, ext2}}),
or from --manifest ({'episodes': [{'ep', 'serials': {...}, 'ext1_png', ...}]}), or
from the raw metadata_<EP>.json. Source frame priority: --t0_dir/<EP>/*ext1*.png,
then the manifest's ext1_png/ext2_png (native PNGs from extract_ext_frames.py),
then MP4 frame 0 (all three are the same decoded frame).

Tiny: one cv2 first-frame read per MP4 (~50 ms). Runs on a login node.
"""
import argparse
import glob
import json
import os
import sys

import cv2

CACHE = "/gpfs/scratch/etur59/koc821022/vggt_cache"


def _find_key(obj, names):
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


def manifest_entry(manifest, ep):
    """{EP: {...}}, {'episodes': [{'ep': EP, ...}]}, {'episodes': {EP: ...}} or a list."""
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


def serials_for(ep, raw_paths, manifest, raw_root):
    out = {}
    for cam in ("ext1", "ext2"):
        v = None
        if raw_paths and ep in raw_paths:
            v = raw_paths[ep].get(cam) or raw_paths[ep].get(f"{cam}_cam_serial")
        if v is None and manifest is not None:
            entry = manifest_entry(manifest, ep)
            if entry is not None:
                v = _find_key(entry, (cam, f"{cam}_cam_serial", f"{cam}_serial"))
        if v is None:
            mp = os.path.join(raw_root, ep, f"metadata_{ep}.json")
            if os.path.isfile(mp):
                v = json.load(open(mp)).get(f"{cam}_cam_serial")
        if v is None:
            return None
        out[cam] = str(v)
    return out


def first_frame(mp4):
    cap = cv2.VideoCapture(mp4)
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        raise RuntimeError(f"cannot decode frame 0 of {mp4}")
    return fr


def last_frame(mp4):
    """Frame n-1 via CAP_PROP_POS_FRAMES seek (CAP_PROP_FRAME_COUNT is exact for
    these MP4s — verified by extract_ext_frames.py; == store frame T_store-1)."""
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        raise RuntimeError(f"cannot decode last frame ({n - 1}) of {mp4}")
    return fr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke13_scenes.txt"))
    ap.add_argument("--raw_root", default=os.path.join(CACHE, "raw"))
    ap.add_argument("--raw_paths", default=os.path.join(CACHE, "smoke13_raw_paths.json"))
    ap.add_argument("--manifest", default=os.path.join(CACHE, "smoke13_manifest.json"))
    ap.add_argument("--t0_dir", default=os.path.join(CACHE, "t0"),
                    help="optional pre-decoded frame-0 PNGs: <t0_dir>/<EP>/*ext1*.png, *ext2*.png")
    ap.add_argument("--which", choices=("first", "last"), default="first",
                    help="first: ext frame 0 (t0). last: ext frame T_store-1 (end of episode; "
                         "source = manifest ext*_last_png, else MP4 last-frame seek). "
                         "Default --out becomes context_ext_last for --which last.")
    ap.add_argument("--out", default=None,
                    help="default: <cache>/context_ext (first) or <cache>/context_ext_last (last)")
    ap.add_argument("--size", default="320x180")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(CACHE, "context_ext" if args.which == "first" else "context_ext_last")
    W, H = [int(v) for v in args.size.lower().split("x")]

    eps = [ln.strip() for ln in open(args.scenes) if ln.strip()]
    raw_paths = json.load(open(args.raw_paths)) if os.path.isfile(args.raw_paths) else None
    manifest = json.load(open(args.manifest)) if os.path.isfile(args.manifest) else None
    if manifest is None:
        print(f"NOTE: manifest {args.manifest} absent (not required)", flush=True)

    n_ok = n_fail = 0
    for ep in eps:
        ser = serials_for(ep, raw_paths, manifest, args.raw_root)
        if ser is None:
            print(f"FAIL {ep}: no ext1/ext2 serials found", flush=True)
            n_fail += 1
            continue
        out_dir = os.path.join(args.out, ep)
        os.makedirs(out_dir, exist_ok=True)
        ok = True
        for i, cam in enumerate(("ext1", "ext2")):
            dst = os.path.join(out_dir, f"{i:02d}_{cam}.png")
            if os.path.isfile(dst) and not args.force:
                continue
            src_png = sorted(glob.glob(os.path.join(args.t0_dir, ep, f"*{cam}*.png"))) \
                if args.which == "first" else []
            entry = manifest_entry(manifest, ep) if manifest is not None else None
            pkey = f"{cam}_png" if args.which == "first" else f"{cam}_last_png"
            mpng = entry.get(pkey) if isinstance(entry, dict) else None
            if not src_png and mpng and os.path.isfile(mpng):
                src_png = [mpng]  # native frame-0 PNG decoded by extract_ext_frames.py
            try:
                if src_png:
                    fr = cv2.imread(src_png[0], cv2.IMREAD_COLOR)
                    if fr is None:
                        raise RuntimeError(f"cannot read {src_png[0]}")
                    src = src_png[0]
                else:
                    src = os.path.join(args.raw_root, ep, "recordings", "MP4", f"{ser[cam]}.mp4")
                    fr = first_frame(src) if args.which == "first" else last_frame(src)
                if (fr.shape[1], fr.shape[0]) != (W, H):
                    assert abs(fr.shape[1] / fr.shape[0] - W / H) < 1e-2, \
                        f"{src}: {fr.shape[1]}x{fr.shape[0]} is not {W}:{H} aspect"
                    fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
                tmp = dst + ".part.png"
                if not cv2.imwrite(tmp, fr):
                    raise RuntimeError(f"cannot write {tmp}")
                os.replace(tmp, dst)
                print(f"OK {dst} <- {src}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {ep} {cam}: {e!r}", flush=True)
                ok = False
        n_ok += ok
        n_fail += (not ok)
    print(f"context frames: {n_ok} episodes OK, {n_fail} failed -> {args.out}", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
