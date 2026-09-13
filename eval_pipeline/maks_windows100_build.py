#!/usr/bin/env python3
"""Scale the window / pre-window blocking protocol to a scene subset.

1. Sample N scenes (seeded) from the 4292-scene test list that have a baseline
   per-frame CSV (label augfull_lr1e5).
2. Detect decline windows on each baseline CSV with find_decline_windows.py's
   detector (robust-z aggregate over the 5 metrics, multi-scale step score,
   NMS), keeping the K strongest windows per scene.
3. Emit arms by window rank so one worker invocation (model loaded once) covers
   all scenes for that arm:
     W<k>: block the k-th strongest window's frames          (window itself)
     P<k>: block the same-length span immediately BEFORE it  (pre-window)
   Each arm gets a control json {scene: {"block": [...]}} and a scene list
   holding only the scenes that have a k-th window.
4. Write a manifest with every (scene, window) pair, its span, pre-span, Z,
   ramp, and whether the window touches the scene end (terminal).

  python maks_windows100_build.py --n_scenes 100 --seed 2 --max_windows 4 --out_dir <dir>
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_decline_windows import load, badness, step_scores, nms, METRICS  # noqa: E402

OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--max_windows", type=int, default=4, help="K strongest windows per scene")
    ap.add_argument("--max_width", type=int, default=20)
    ap.add_argument("--zmin", type=float, default=3.0)
    ap.add_argument("--baseline_label", default="augfull_lr1e5")
    ap.add_argument("--out_dir", required=True, help="where scene lists / control jsons / manifest go")
    ap.add_argument("--arm_dir", required=True, help="eval_pipeline dir for maks_arm_<name>.json")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    full = [l.strip() for l in open(f"{OUT}/scene_list.txt") if l.strip()]
    rng = random.Random(args.seed)
    pool = full[:]
    rng.shuffle(pool)
    scenes = []
    for s in pool:
        if os.path.isfile(f"{OUT}/{args.baseline_label}/eval/{s}/eval_depth_pose_metrics.csv"):
            scenes.append(s)
        if len(scenes) == args.n_scenes:
            break
    assert len(scenes) == args.n_scenes, f"only {len(scenes)} scenes with a baseline CSV"

    manifest = {"scenes": scenes, "seed": args.seed, "max_windows": args.max_windows,
                "zmin": args.zmin, "max_width": args.max_width, "windows": []}
    arms = {}  # name -> {scene: spec}
    n_win_hist = {}
    for s in scenes:
        csv_path = f"{OUT}/{args.baseline_label}/eval/{s}/eval_depth_pose_metrics.csv"
        ts, raw = load(csv_path)
        n = len(ts)
        z = {m: badness(raw[m], m)[0] for m in METRICS}
        B = np.nanmean(np.vstack([z[m] for m in METRICS]), axis=0)
        Z, sigma = step_scores(B, args.max_width)
        dec = nms(Z, args.zmin, args.max_windows)
        n_win_hist[len(dec)] = n_win_hist.get(len(dec), 0) + 1
        for k, (t, w, score) in enumerate(dec, 1):
            a, b = int(ts[t]), int(ts[min(t + w, n) - 1])
            L = b - a + 1
            pre = list(range(max(1, a - L), a))
            win = list(range(a, b + 1))
            if not pre:            # window starts at frame <= 1: nothing to pre-block
                continue
            terminal = (b >= n - 1)
            manifest["windows"].append({"scene": s, "rank": k, "window": [a, b], "pre": [pre[0], pre[-1]],
                                        "len": L, "pre_len": len(pre), "z": round(float(score), 2),
                                        "n_frames": n, "terminal": terminal})
            arms.setdefault(f"w100_W{k}", {})[s] = {"block": win}
            arms.setdefault(f"w100_P{k}", {})[s] = {"block": pre}

    for name, spec in sorted(arms.items()):
        json.dump(spec, open(os.path.join(args.arm_dir, f"maks_arm_{name}.json"), "w"))
        open(os.path.join(args.out_dir, f"scenes_{name}.txt"), "w").write("\n".join(spec) + "\n")
        fr = sum(len(v["block"]) for v in spec.values())
        print(f"{name}: {len(spec)} scenes, {fr} blocked frames")
    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=1)
    open(os.path.join(args.out_dir, "scenes_all.txt"), "w").write("\n".join(scenes) + "\n")
    W = manifest["windows"]
    print(f"scenes={len(scenes)} windows={len(W)} terminal={sum(w['terminal'] for w in W)} "
          f"windows-per-scene histogram={dict(sorted(n_win_hist.items()))} "
          f"mean window len={np.mean([w['len'] for w in W]):.1f} frames, total frames to infer="
          f"{2 * sum(w['n_frames'] for w in W)}")


if __name__ == "__main__":
    main()
