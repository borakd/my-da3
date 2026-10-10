#!/usr/bin/env python
"""Restore PointWorld-DROID depth_320x180 episode HDF5s from the downloaded parts and convert the EXTERIOR-camera depth
into exactly the format of the existing wrist depth store (COMPUTE NODE; see pwdepth_extract_convert.sbatch).

Verified 2026-10-02 on the first 6 episodes: the wrist store's dense/depth/NNNNNN.npy is bit-identical to this package's
wrist group, depth.astype(float32) / 1000 (metres; 0 = invalid, and the store's outlier_mask PNG is exactly depth == 0).
Exterior groups have the same frame count F (= store frames = trajectory_length - 1) and frame t lines up with MP4 frame t.

1. EXTRACT: cat the contiguous prefix of sha256-verified parts (<part>.ok from pwdepth_download.sh) | zstd -d | stream
   tar. Each member is written to <name>.tmp and renamed only when complete. Members already restored with the right
   size are skipped. A truncated tail (download still running) stops extraction cleanly. Every run re-reads the stream
   from part 0, because the package is a single zstd frame.
2. CONVERT (multiprocessing, one task per restored episode that is in our wrist store and has no status file yet):
     OUT/ext1/<EP>/dense/depth/NNNNNN.npy   float32 (180, 320) metres = uint16 mm / 1000, np.save default header
     OUT/ext2/<EP>/dense/depth/NNNNNN.npy   (ext1/ext2 = metadata_<EP>.json ext1_cam_serial / ext2_cam_serial)
   Each camera's depth dir is written as dense/.depth.tmp and renamed. An exterior stream with MORE frames than the store
   (e.g. 44bb9c36+2023-11-25-10h-40m-09s ext2: 1032 vs 1031, one extra LEADING frame, also in its MP4) is aligned to the
   store index by timestamps (offset minimising the median |t_ext - t_wrist|; recorded as <cam>_offset, and an MP4-based
   consumer must apply the same offset). The status file OUT/_status/<EP>.json records
   serials, F per group, the wrist-store frame count, a bit-exact check of the wrist group against the existing store,
   and invalid fractions. It is written last and marks the episode done.

    python pwdepth_extract_convert.py [--workers 20] [--no_extract] [--limit N]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from multiprocessing import Pool

import h5py
import numpy as np

PKG = "/gpfs/scratch/etur59/koc821022/pointworld_droid_depth_pkg"
PARTS = os.path.join(PKG, "droid", "depth_320x180")
RESTORED = os.path.join(PKG, "restored", "droid", "depth_320x180")
OUT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_ext_all/dl3dv_multi"
STATUS = os.path.join(OUT, "_status")
WRIST = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
RAW = "/gpfs/scratch/etur59/koc821022/vggt_cache/raw"
ZSTD = "/apps/GPP/MINICONDA/24.1.2/bin/zstd"


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def verified_prefix():
    parts = [l.split("\t")[0] for l in open(os.path.join(PKG, "manifest.tsv")) if l.strip()]
    pre = []
    for p in parts:
        if not os.path.isfile(os.path.join(PARTS, p + ".ok")):
            break
        pre.append(os.path.join(PARTS, p))
    return pre, len(parts)


def extract(on_new):
    pre, n = verified_prefix()
    log(f"EXTRACT: {len(pre)}/{n} parts verified in a contiguous prefix")
    if not pre:
        return 0, False
    os.makedirs(RESTORED, exist_ok=True)
    cat = subprocess.Popen(["cat"] + pre, stdout=subprocess.PIPE)
    zst = subprocess.Popen([ZSTD, "-q", "-d", "-c"], stdin=cat.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    cat.stdout.close()
    new = seen = 0
    complete = False
    try:
        with tarfile.open(fileobj=zst.stdout, mode="r|") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                seen += 1
                dst = os.path.join(RESTORED, os.path.basename(m.name))
                if os.path.isfile(dst) and os.path.getsize(dst) == m.size:
                    continue
                tmp = f"{dst}.tmp.{os.getpid()}"   # per-process: concurrent extract runs never share a temp file
                with tf.extractfile(m) as src, open(tmp, "wb") as f:
                    shutil.copyfileobj(src, f, 16 << 20)
                if os.path.getsize(tmp) != m.size:
                    os.remove(tmp)
                    raise EOFError(f"short member {m.name}")
                os.rename(tmp, dst)
                new += 1
                on_new(os.path.basename(m.name)[: -len("_depth.h5")])
        complete = len(pre) == n
    except (tarfile.ReadError, EOFError, OSError) as e:
        log(f"EXTRACT stopped at the end of the available stream ({type(e).__name__}: {str(e)[:120]})")
    finally:
        for p in (zst, cat):
            p.kill()
            p.wait()
        for f in os.listdir(RESTORED):
            if f.endswith(f".tmp.{os.getpid()}"):
                os.remove(os.path.join(RESTORED, f))
    log(f"EXTRACT: members seen {seen}, newly restored {new}, stream complete={complete}")
    return new, complete


def done(ep):
    p = os.path.join(STATUS, ep + ".json")
    try:
        st = json.load(open(p))
        return st.get("ok", False) or st.get("final", False)
    except (OSError, ValueError):
        return False


def convert(ep):
    st_path = os.path.join(STATUS, ep + ".json")
    if done(ep):
        return ep, "skip"
    try:
        meta = json.load(open(os.path.join(RAW, ep, f"metadata_{ep}.json")))
        ser = {c: str(meta[f"{c}_cam_serial"]) for c in ("ext1", "ext2")}
        wdir = os.path.join(WRIST, ep, "dense", "depth")
        n_store = len([f for f in os.listdir(wdir) if f.endswith(".npy")])
        st = dict(episode=ep, serials=ser, F_wrist_store=n_store)
        with h5py.File(os.path.join(RESTORED, ep + "_depth.h5"), "r") as h:
            wk = [k for k in h.keys() if k.endswith("+wrist")]
            st["groups"] = sorted(k for k in h.keys() if k != "metadata")
            if wk:
                W = h[wk[0]]["depth"]
                st["F_wrist_h5"] = int(W.shape[0])
                t = min(n_store, W.shape[0]) // 2
                st["wrist_store_bit_exact_mid"] = bool(np.array_equal(np.load(os.path.join(wdir, f"{t:06d}.npy")), W[t].astype(np.float32) / 1000.0))
            for cam, s in ser.items():
                g = f"{s}+ext"
                if g not in h:
                    st[f"{cam}_missing"] = True
                    continue
                D = h[g]["depth"][:]
                ts = h[g]["timestamps"][:]
                st[f"F_{cam}_raw"] = int(D.shape[0])
                off = 0
                if D.shape[0] > n_store and wk:   # extra exterior frames: align to the store index by timestamps
                    wts = h[wk[0]]["timestamps"][:n_store]
                    dts = [float(np.median(np.abs(ts[o:o + n_store] - wts))) for o in range(D.shape[0] - n_store + 1)]
                    off = int(np.argmin(dts))
                    st[f"{cam}_offset_median_dt_ms"] = dts[off]
                    D, ts = D[off:off + n_store], ts[off:off + n_store]
                st[f"{cam}_offset"] = off
                st[f"F_{cam}"] = int(D.shape[0])
                st[f"{cam}_invalid_frac"] = round(float((D == 0).mean()), 4)
                st[f"{cam}_ts_monotonic"] = bool((np.diff(ts) > 0).all())
                ddir = os.path.join(OUT, cam, ep, "dense")
                final, tmp = os.path.join(ddir, "depth"), os.path.join(ddir, f".depth.tmp.{os.getpid()}")
                if os.path.isdir(final):
                    shutil.rmtree(final)
                if os.path.isdir(tmp):
                    shutil.rmtree(tmp)
                os.makedirs(tmp)
                for i in range(D.shape[0]):
                    np.save(os.path.join(tmp, f"{i:06d}.npy"), D[i].astype(np.float32) / 1000.0)
                os.rename(tmp, final)
        st["ok"] = all(st.get(f"F_{c}") == n_store for c in ("ext1", "ext2")) and st.get("wrist_store_bit_exact_mid", False)
        # an exterior stream SHORTER than the store cannot be fixed by re-conversion: record it once, do not retry
        st["final"] = st["ok"] or any(st.get(f"F_{c}", n_store) < n_store for c in ("ext1", "ext2"))
        with open(st_path + ".tmp", "w") as f:
            json.dump(st, f)
        os.rename(st_path + ".tmp", st_path)
        return ep, "ok" if st["ok"] else "flag"
    except Exception as e:  # noqa: BLE001  (recorded, episode retried on the next run)
        return ep, f"error {type(e).__name__}: {str(e)[:200]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--no_extract", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="convert at most N episodes (smoke test)")
    args = ap.parse_args()
    os.makedirs(STATUS, exist_ok=True)
    ours = set(os.listdir(WRIST))
    counts = {}
    queued = set()
    pool = Pool(args.workers)
    results = []

    def submit(ep):
        if ep in ours and ep not in queued and not done(ep) \
                and (args.limit is None or len(queued) < args.limit):
            queued.add(ep)
            results.append(pool.apply_async(convert, (ep,)))

    for f in sorted(os.listdir(RESTORED)) if os.path.isdir(RESTORED) else []:   # restored earlier, not yet converted
        if f.endswith("_depth.h5"):
            submit(f[: -len("_depth.h5")])
    complete = False
    if not args.no_extract:
        _, complete = extract(submit)
    pool.close()
    for r in results:
        ep, s = r.get()
        k = s.split()[0]
        counts[k] = counts.get(k, 0) + 1
        if k not in ("ok", "skip"):
            log(f"CONVERT {ep}: {s}")
    pool.join()
    n_ok = sum(done(f[:-5]) for f in os.listdir(STATUS) if f.endswith(".json"))
    n_done = sum(done(f[:-5]) for f in os.listdir(STATUS) if f.endswith(".json"))
    log(f"CONVERT this run: {counts} | episodes ok: {n_ok}/{len(ours)} | episodes done: {n_done}/{len(ours)} (ok + unfixable short streams) | stream complete={complete}")


if __name__ == "__main__":
    sys.exit(main())
