#!/usr/bin/env python3
"""Build the two-hop symlink layout for the full DROID wrist dataset on MareNostrum5.

    <SPLITS>/{train,test}/dl3dv_multi/wrist/<scene>   (hop 2)
        -> <WRIST>/<scene>                            (hop 1)
            -> <VALAR>/<scene>                        (REAL DATA -- never touched)

SOURCE-DATA SAFETY (the hard requirement):
  This script NEVER moves, modifies, deletes, opens-for-write, or otherwise
  touches anything under VALAR. It only:
    - reads directory listings under VALAR (os.scandir / os.path.isdir)
    - creates symlinks that POINT at VALAR, in directories outside VALAR
  Every write path is asserted to live outside VALAR before anything happens,
  and the script refuses to delete or replace anything that is not a symlink.

Usage:
    python build_dataset_symlinks.py --check          # verify only, no writes
    python build_dataset_symlinks.py --build          # create the symlinks
    python build_dataset_symlinks.py --build --allow-incomplete   # (not advised)

Why --allow-incomplete is dangerous: a scene that has not finished copying yields
a DANGLING link, which dl3dv.py:214-216 SILENTLY SKIPS (os.path.isdir is False --
no error, no warning). The short scan is then written to the disk cache, whose
signature (dl3dv.py:69-92) records only the level-1 dir's mtime and entry count --
neither of which changes when the missing targets later appear. The truncated
dataset would therefore persist forever. Build only when the copy is complete.
"""
import argparse
import os
import os.path as osp
import sys
from concurrent.futures import ThreadPoolExecutor

VALAR = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_VALAR"
DATA = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi"
WRIST = osp.join(DATA, "wrist")
SPLITDEF = osp.join(DATA, "splits")
SPLITS = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits"
REPO = osp.dirname(osp.abspath(__file__))
THREADS = 32

# --------------------------------------------------------------------------
# safety
# --------------------------------------------------------------------------
def assert_outside_source(*paths):
    """Refuse to run if any directory we WRITE to lives under the source store."""
    src = osp.realpath(VALAR).rstrip("/") + "/"
    for p in paths:
        rp = osp.realpath(osp.abspath(p)).rstrip("/") + "/"
        if rp.startswith(src):
            sys.exit(f"REFUSING: write target {p} resolves inside the source store {VALAR}")


def read_splits():
    out = {}
    for name in ("train", "test"):
        f = osp.join(SPLITDEF, f"{name}.txt")
        with open(f, encoding="utf-8") as fh:
            out[name] = [ln.strip() for ln in fh if ln.strip()]
    return out


def link_one(args):
    """Create ONE symlink. Never removes anything that is not a symlink."""
    target, linkpath = args
    try:
        if osp.islink(linkpath):
            if os.readlink(linkpath) == target:
                return "ok"
            os.remove(linkpath)          # replacing a symlink only, never real data
        elif osp.exists(linkpath):
            return f"BLOCKED:{linkpath} exists and is NOT a symlink"
        os.symlink(target, linkpath)
        return "made"
    except FileExistsError:
        return "ok"                      # race with a peer thread
    except OSError as e:
        return f"ERROR:{linkpath}:{e}"


def build(pairs, desc):
    made = ok = 0
    problems = []
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        for r in ex.map(link_one, pairs, chunksize=256):
            if r == "made":
                made += 1
            elif r == "ok":
                ok += 1
            else:
                problems.append(r)
    print(f"  {desc}: {made} created, {ok} already correct, {len(problems)} problems")
    for p in problems[:10]:
        print("     " + p)
    return not problems


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def verify(splits, verbose=True):
    ok = True

    def check(label, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}: {got}" + ("" if good else f"  (expected {want})"))

    allscenes = sorted(set(splits["train"]) | set(splits["test"]))
    check("hop-1 links in wrist/", len(os.listdir(WRIST)) if osp.isdir(WRIST) else -1, len(allscenes))

    # dangling links are the silent killer -- count them explicitly
    def dangling(d):
        return sum(1 for e in os.scandir(d) if e.is_symlink() and not osp.isdir(e.path))

    check("hop-1 DANGLING links", dangling(WRIST) if osp.isdir(WRIST) else -1, 0)

    for name in ("train", "test"):
        w = osp.join(SPLITS, name, "dl3dv_multi", "wrist")
        check(f"hop-2 links in {name}/", len(os.listdir(w)) if osp.isdir(w) else -1, len(splits[name]))
        check(f"hop-2 DANGLING in {name}/", dangling(w) if osp.isdir(w) else -1, 0)

    tr, te = set(splits["train"]), set(splits["test"])
    check("train n test (must be disjoint)", len(tr & te), 0)
    check("train u test == all scenes", len(tr | te), len(allscenes))

    # CONTENT check. A resolving, non-dangling link is NOT enough: during the
    # staging copy all 42925 scene dirs appeared while 15 still had an EMPTY
    # dense/rgb. Those pass every structural check above and then blow up in
    # _build_raw_scan ("<dir> is empty", dl3dv.py:225) -- or worse, a partially
    # copied scene caches a short frame list silently. So walk the content.
    def scene_frames(args):
        root, s = args
        try:
            return len([f for f in os.listdir(osp.join(root, s, "dense", "rgb"))
                        if f.endswith(".png")])
        except OSError:
            return -1

    for name in ("train", "test"):
        w = osp.join(SPLITS, name, "dl3dv_multi", "wrist")
        if not osp.isdir(w):
            continue
        with ThreadPoolExecutor(max_workers=64) as ex:
            counts = list(ex.map(scene_frames, [(w, s) for s in splits[name]], chunksize=64))
        empty = sum(1 for c in counts if c <= 0)
        check(f"{name}: scenes with EMPTY/unreadable dense/rgb", empty, 0)
        if empty == 0:
            print(f"         ({sum(counts):,} rgb frames across {len(counts)} scenes, "
                  f"min {min(counts)}, max {max(counts)})")

    # MODALITY CONSISTENCY (sampled). _get_views opens 5 files per view and
    # derives depth/cam/sky_mask/outlier_mask names from the rgb filename
    # (dl3dv.py:299-307), so a scene whose counterparts lag behind rgb loads
    # rgb fine and then dies on np.load / returns None from cv2.imread. Sampled
    # rather than exhaustive to keep this usable as a pre-flight gate.
    import random
    rnd = random.Random(0)
    sample = rnd.sample(splits["train"], min(2000, len(splits["train"])))

    def modality_ok(s):
        d = osp.join(SPLITS, "train", "dl3dv_multi", "wrist", s, "dense")
        try:
            counts = {m: len(os.listdir(osp.join(d, m)))
                      for m in ("rgb", "depth", "cam", "sky_mask", "outlier_mask")}
        except OSError:
            return False
        return len(set(counts.values())) == 1 and counts["rgb"] > 0

    with ThreadPoolExecutor(max_workers=64) as ex:
        bad = sum(1 for good in ex.map(modality_ok, sample, chunksize=32) if not good)
    check(f"modality counts agree (sample of {len(sample)} train scenes)", bad, 0)

    # end-to-end: resolve one scene through BOTH hops and confirm real frames
    probe = splits["train"][0]
    p = osp.join(SPLITS, "train", "dl3dv_multi", "wrist", probe)
    n_png = len([f for f in os.listdir(osp.join(p, "dense", "rgb"))
                 if f.endswith(".png")]) if osp.isdir(p) else -1
    check(f"end-to-end resolve ({probe[:28]}... rgb frames)", n_png > 0, True)
    if verbose and n_png > 0:
        chain = p
        for _ in range(3):
            if not osp.islink(chain):
                break
            nxt = os.readlink(chain)
            print(f"      {chain}\n        -> {nxt}")
            chain = nxt
    return ok


def warm_cache(splits):
    """Populate the DL3DV disk cache from the LOGIN node.

    _build_raw_scan walks 42925 scenes x ~3 metadata ops through two symlink hops.
    Doing that inside the 64-node job would leave 256 GPUs idle for the duration,
    and all ranks would contend on the same lock. Do it once here instead.

    Also acts as the real acceptance test: the scan MUST report 38633 / 4292.
    """
    sys.path[:0] = [f"{R}" for R in (
        osp.join(REPO, "src", "CUT3R", "src"),
        osp.join(REPO, "src", "CUT3R"),
        osp.join(REPO, "src"),
        REPO,
    )]
    os.environ.setdefault(
        "DL3DV_CACHE_DIR", "/gpfs/scratch/etur59/koc821022/.dl3dv_cache_wrist_all")
    print(f"  DL3DV_CACHE_DIR = {os.environ['DL3DV_CACHE_DIR']}")
    import dust3r.heads  # noqa: F401  (import order: heads before utils.camera)
    from dust3r.datasets.dl3dv import DL3DV_Multi

    ok = True
    for name, nviews in (("train", 64), ("test", 4)):
        root = osp.join(SPLITS, name, "dl3dv_multi")
        print(f"  scanning {name} ({root}) ...", flush=True)
        ds = DL3DV_Multi(allow_repeat=True, split=name, ROOT=root, aug_crop=0,
                         resolution=[(320, 192)], num_views=nviews, n_corres=0,
                         force_consecutive_frame_sampling=True)
        got, want = len(ds.scenes), len(splits[name])
        good = got == want
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: scanned {got} scenes "
              f"(expected {want}); {len(ds)} sampling windows")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--warm-cache", action="store_true",
                    help="scan both split trees to populate the DL3DV disk cache")
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    if not (a.build or a.check or a.warm_cache):
        ap.error("pass --build, --check and/or --warm-cache")

    assert_outside_source(WRIST, SPLITS)
    splits = read_splits()
    allscenes = sorted(set(splits["train"]) | set(splits["test"]))
    print(f"canonical split definition: {len(splits['train'])} train + {len(splits['test'])} test "
          f"= {len(allscenes)} scenes")

    on_disk = {e.name for e in os.scandir(VALAR) if e.is_dir()}
    missing = [s for s in allscenes if s not in on_disk]
    print(f"source store: {len(on_disk)} scene dirs present, {len(missing)} of the canonical set missing")

    if a.build:
        if missing and not a.allow_incomplete:
            sys.exit(f"REFUSING to build: {len(missing)} scenes have not finished copying.\n"
                     f"  Building now would create dangling links that dl3dv.py SILENTLY SKIPS,\n"
                     f"  and the truncated scan would be cached permanently.\n"
                     f"  Wait for the copy, or pass --allow-incomplete if you truly mean it.")
        print("\nbuilding hop 1 (wrist/ -> VALAR) ...")
        os.makedirs(WRIST, exist_ok=True)
        b1 = build([(osp.join(VALAR, s), osp.join(WRIST, s)) for s in allscenes], "hop 1")

        print("building hop 2 (split trees -> wrist/) ...")
        b2 = True
        for name in ("train", "test"):
            w = osp.join(SPLITS, name, "dl3dv_multi", "wrist")
            os.makedirs(w, exist_ok=True)
            b2 &= build([(osp.join(WRIST, s), osp.join(w, s)) for s in splits[name]], f"hop 2 {name}")
        if not (b1 and b2):
            sys.exit("build reported problems -- see above")

    ok = True
    if a.build or a.check:
        print("\nverification:")
        ok = verify(splits)

    if a.warm_cache:
        print("\nwarming the DL3DV scan cache (this is the slow walk):")
        ok = warm_cache(splits) and ok

    print("\n" + ("ALL CHECKS PASSED" if ok else "CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
