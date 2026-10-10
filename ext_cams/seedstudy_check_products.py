#!/usr/bin/env python
"""Check the PRODUCTS of a seedstudy method (not exit codes): for every scene, both exterior-camera npz files exist,
unpack to (T, 720, 1280) with T == the store's cam/*.npz count, frames_present == (union non-empty per frame); plus
coverage and per-scene timing/prompt stats from seed_info.json when present.

    OMP_NUM_THREADS=1 python seedstudy_check_products.py --method text_arm_sam3 \
        --scenes /gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt
"""
import argparse
import json
import os

import numpy as np

SEEDSTUDY = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy"
STORE = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
RAW = "/gpfs/scratch/etur59/koc821022/vggt_cache/raw"


def check_file(path, T_store):
    if not os.path.isfile(path):
        return dict(ok=False, why="missing")
    try:
        z = np.load(path)
        shape = [int(v) for v in z["shape"]]
        union, fp = z["union"], z["frames_present"].astype(bool)
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, why=f"unreadable: {e!r}")
    T, H, W = shape
    m = np.unpackbits(union, axis=2)[:, :, :W].astype(bool)
    probs = []
    if m.shape != (T, H, W):
        probs.append(f"unpacked {m.shape} != shape {shape}")
    if (H, W) != (720, 1280):
        probs.append(f"resolution {H}x{W} != 720x1280")
    if T_store is not None and T != T_store:
        probs.append(f"T {T} != store {T_store}")
    present = m.reshape(T, -1).any(1)
    if fp.shape != (T,) or not np.array_equal(present, fp):
        probs.append("frames_present inconsistent with union")
    return dict(ok=not probs, why="; ".join(probs), T=T, present=int(present.sum()), frac_present=float(present.mean()),
                mean_area=float(m.mean()), bytes=os.path.getsize(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--root", default=None, help="default: <SEEDSTUDY>/<method>")
    ap.add_argument("--scenes", default="/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt")
    ap.add_argument("--suffix", default=None, help="file suffix after the serial; default __<method>_masks.npz")
    args = ap.parse_args()
    root = args.root or os.path.join(SEEDSTUDY, args.method)
    suffix = args.suffix or f"__{args.method}_masks.npz"
    eps = [l.strip() for l in open(args.scenes) if l.strip()]
    n_ok, rows = 0, []
    for ep in eps:
        cam_dir = os.path.join(STORE, ep, "dense", "cam")
        T_store = len([f for f in os.listdir(cam_dir) if f.endswith(".npz")]) if os.path.isdir(cam_dir) else None
        meta = json.load(open(os.path.join(RAW, ep, f"metadata_{ep}.json")))
        info_p = os.path.join(root, ep, "seed_info.json")
        info = json.load(open(info_p)) if os.path.isfile(info_p) else None
        ok_ep = True
        line = f"{ep} T={T_store}"
        for cam in ("ext1", "ext2"):
            ser = meta[f"{cam}_cam_serial"]
            r = check_file(os.path.join(root, ep, f"{cam}_{ser}{suffix}"), T_store)
            ok_ep &= r["ok"]
            if r["ok"]:
                extra = ""
                if info and cam in info.get("cams", {}):
                    ci = info["cams"][cam]
                    extra = f" prompts={ci.get('n_prompts')} modes={''.join(ci.get('modes', []))} seed=f{ci.get('first_prompt_frame')} {ci.get('seconds', 0):.0f}s"
                line += f" | {cam} OK present {r['present']}/{r['T']} area {r['mean_area']*100:.2f}%{extra}"
            else:
                line += f" | {cam} BAD {r['why']}"
        n_ok += ok_ep
        rows.append(line)
        print(("OK  " if ok_ep else "BAD ") + line)
    print(f"\n{n_ok}/{len(eps)} scenes with both products valid ({args.method})")


if __name__ == "__main__":
    main()
