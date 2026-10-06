#!/usr/bin/env python
"""Merge the per-shard results of segment_birth_frames.py into birth_masks.csv, summarise the
run (status counts, prompt fallbacks, consistency flags, MP4/store frame-count mismatches,
per-lab flag rates) and render QA contact sheets: a random sample of accepted masks and all
masks that stayed flagged after the box fallback."""
import csv
import glob
import json
import os
import random

import cv2
import numpy as np

OUT_ROOT = "/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames"
RUN = f"{OUT_ROOT}/masks_run"


def sheet(rows, path, cols=4, tile=(480, 270), max_tiles=40):
    tiles = []
    for r in rows[:max_tiles]:
        p = f"{RUN}/overlays/{r['episode']}/{r['cam']}_f{int(r['frame']):05d}.jpg"
        im = cv2.imread(p)
        if im is None:
            continue
        tiles.append(cv2.resize(im, tile))
    if not tiles:
        return 0
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    return len(tiles)


def main():
    src = f"{RUN}/results_refined.jsonl" if os.path.isfile(f"{RUN}/results_refined.jsonl") else None
    rows = []
    if src:
        rows = [json.loads(l) for l in open(src) if l.strip()]
    else:
        for f in sorted(glob.glob(f"{RUN}/results_shard*.jsonl")):
            for line in open(f):
                if line.strip():
                    rows.append(json.loads(line))
    rows.sort(key=lambda r: (r["episode"], r["which"], r["cam"]))
    keys = ["episode", "cam", "which", "frame", "n_frames", "mp4_frames", "serial", "status", "kin_frac",
            "prompt_used", "spill_auto", "area_auto", "auto_empty", "projection_disagree", "spill_box", "area_box",
            "spill_final", "area_final", "mask"]
    with open(f"{OUT_ROOT}/birth_masks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    ok = [r for r in rows if r.get("status") == "ok"]
    L = [f"# RobotSeg gripper masks on the birth frames, DROID wrist test split", ""]
    import collections
    L.append(f"rows: {len(rows)}  status: {dict(collections.Counter(r.get('status') for r in rows))}")
    L.append(f"scenes with masks: {len(set(r['episode'] for r in ok))}; masks: {len(ok)} "
             f"(b050: {sum(r['which']=='b050' for r in ok)}, b100: {sum(r['which']=='b100' for r in ok)})")
    mism = [r for r in ok if r.get("mp4_frames") is not None and r["mp4_frames"] != r["n_frames"]]
    L.append(f"MP4 frame count != store frame count: {len(mism)} masks in {len(set(r['episode'] for r in mism))} scenes "
             f"(frame index still taken as the store index)")
    L.append("")
    L.append("## Mask policy and the kinematic consistency flag")
    L.append("")
    L.append("Final mask = RobotSeg automatic prompt on the birth frame; the kinematic box prompt is used only when the")
    L.append("automatic mask is empty (< 0.05 % of the image). `projection_disagree` = the automatic mask does not sit")
    L.append("where the kinematic projection puts the gripper (> 20 % of mask pixels outside the projected box + 60 px).")
    L.append("QA showed that in these cases the MASK is usually right and the scene's exterior extrinsics are off, so the")
    L.append("flag is best read as 'extrinsics (and therefore birth frame) suspect for this camera'.")
    L.append("")
    L.append("| set | masks | automatic mask kept | box used (auto empty) | projection_disagree |")
    L.append("|---|---|---|---|---|")
    for which in ("b050", "b100", "all"):
        sub = [r for r in ok if which == "all" or r["which"] == which]
        auto = sum(r.get("prompt_used") == "auto" for r in sub)
        box = sum(str(r.get("prompt_used", "")).startswith("box") for r in sub)
        dis = sum(bool(r.get("projection_disagree")) for r in sub)
        L.append(f"| {which} | {len(sub)} | {auto} | {box} | {dis} ({100*dis/max(len(sub),1):.1f}%) |")
    L.append("")
    scenes_dis = sorted(set(r["episode"] for r in ok if r.get("projection_disagree")))
    percam = collections.defaultdict(set)
    for r in ok:
        if r["which"] == "b050" and r.get("projection_disagree"):
            percam[r["episode"]].add(r["cam"])
    both = sum(1 for v in percam.values() if v == {"ext1", "ext2"})
    L.append(f"scenes with projection_disagree in at least one camera: {len(scenes_dis)} of {len(set(r['episode'] for r in ok))} "
             f"(both cameras at b050: {both}); listed in `scenes_extrinsics_suspect.txt`")
    open(f"{OUT_ROOT}/scenes_extrinsics_suspect.txt", "w").write("\n".join(scenes_dis) + "\n")
    L.append("")
    L.append("## Per lab (b050, projection_disagree rate = extrinsics-suspect rate)")
    L.append("")
    L.append("| lab | masks | disagree | rate |")
    L.append("|---|---|---|---|")
    bylab = collections.defaultdict(list)
    for r in ok:
        if r["which"] == "b050":
            bylab[r["episode"].split("+")[0]].append(bool(r.get("projection_disagree")))
    for lab, fl in sorted(bylab.items(), key=lambda kv: -len(kv[1])):
        L.append(f"| {lab} | {len(fl)} | {sum(fl)} | {100*sum(fl)/len(fl):.1f}% |")
    L.append("")
    with open(f"{OUT_ROOT}/birth_masks_flagged.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in ok:
            if r.get("projection_disagree") or r.get("auto_empty"):
                w.writerow({k: r.get(k, "") for k in keys})
    # QA sheets
    random.seed(0)
    accepted = [r for r in ok if r["which"] == "b050" and not r.get("projection_disagree")]
    n1 = sheet(random.sample(accepted, min(40, len(accepted))), f"{OUT_ROOT}/qa_random_consistent_b050.jpg")
    dis = [r for r in ok if r.get("projection_disagree")]
    n2 = sheet(random.sample(dis, min(40, len(dis))) if len(dis) > 40 else dis, f"{OUT_ROOT}/qa_random_projection_disagree.jpg")
    empt = [r for r in ok if r.get("auto_empty")]
    n3 = sheet(empt, f"{OUT_ROOT}/qa_auto_empty_box_used.jpg")
    L.append("")
    L.append(f"QA sheets: `qa_random_consistent_b050.jpg` ({n1} random masks consistent with the projection), "
             f"`qa_random_projection_disagree.jpg` ({n2} random masks that disagree with it), "
             f"`qa_auto_empty_box_used.jpg` ({n3}: automatic mask empty, box mask used).")
    L.append("")
    L.append("Files: `masks_run/masks/<ep>/<cam>_f<t>.png` (binary mask), `masks_run/frames/...jpg` (the decoded birth frame),")
    L.append("`masks_run/overlays/...jpg`; `birth_masks.csv` has one row per mask with the frame index, prompt used and scores.")
    open(f"{OUT_ROOT}/birth_masks_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
