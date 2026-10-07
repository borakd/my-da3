#!/usr/bin/env python
"""Build the Track-On work list (one item per (scene, exterior camera) with a b050 mask) from the
RobotSeg mask table. Reproduces trackon_run/worklist.json from birth_masks.csv for v1; for v2 pass
`--masks_csv birth_masks_v2.csv --out trackon_run_v2/worklist.json`.
Fields per item: episode, cam, frame (birth frame), n_frames, mp4_frames, mask (PNG path, 1280x720),
mp4 (raw exterior MP4), prompt_used, projection_disagree, auto_empty, area.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # vggt_probe
import birth_frames as bf  # noqa: E402

RAW_ROOT = "/leonardo_scratch/large/userexternal/bdursun0/robotseg_demo/raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks_csv", default=f"{bf.OUT_ROOT}/birth_masks.csv")
    ap.add_argument("--masks_run", default=f"{bf.OUT_ROOT}/masks_run", help="dir holding masks/<ep>/<cam>_f<t>.png")
    ap.add_argument("--out", default=f"{bf.OUT_ROOT}/trackon_run/worklist.json")
    args = ap.parse_args()
    items = []
    for r in csv.DictReader(open(args.masks_csv)):
        if r["status"] != "ok" or r["which"] != "b050":
            continue
        t = int(r["frame"])
        items.append(dict(
            episode=r["episode"], cam=r["cam"], frame=t, n_frames=int(r["n_frames"]),
            mp4_frames=int(r["mp4_frames"]) if r["mp4_frames"] else None,
            mask=f"{args.masks_run}/masks/{r['episode']}/{r['cam']}_f{t:05d}.png",
            mp4=f"{RAW_ROOT}/{r['episode']}/recordings/MP4/{r['serial']}.mp4",
            prompt_used=r["prompt_used"], projection_disagree=r["projection_disagree"] == "True",
            auto_empty=r["auto_empty"] == "True", area=float(r["area_final"])))
    items.sort(key=lambda w: (w["episode"], w["cam"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(items, open(args.out, "w"))
    print(f"{len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()
