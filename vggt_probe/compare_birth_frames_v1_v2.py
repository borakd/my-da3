#!/usr/bin/env python
"""Compare the v1 (raw DROID extrinsics) and v2 (PointWorld optimized_extrinsics) birth frames.

Reports how many scenes' b050 / b100 birth frame changed and by how much, which of the v1
'never visible in both cameras' scenes are recovered, and writes
  birth_frames_v2/compare_v1_v2.md, compare_v1_v2.csv
  birth_frames_v2/scenes_to_resegment.txt   scenes whose b050 or b100 frame changed (RobotSeg v2 input)
  birth_frames_v2/items_to_retrack.txt      '<ep>:<cam>' whose b050 frame changed (Track-On v2 input)
"""
import argparse
import collections
import csv
import json
import os

import numpy as np

OUT_ROOT = "/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default=f"{OUT_ROOT}/birth_frames.json")
    ap.add_argument("--v2", default=f"{OUT_ROOT}/birth_frames_v2/birth_frames.json")
    ap.add_argument("--failed_md", default=f"{OUT_ROOT}/failed_scenes_never_visible_both.csv")
    args = ap.parse_args()
    out_dir = os.path.dirname(args.v2)
    v1 = {r["episode"]: r for r in json.load(open(args.v1))}
    v2 = {r["episode"]: r for r in json.load(open(args.v2))}
    verdict = {}
    if os.path.isfile(args.failed_md):
        for r in csv.DictReader(open(args.failed_md)):
            verdict[r["episode"]] = r.get("verdict") or r.get("judgement") or ""
    eps = sorted(set(v1) & set(v2))
    rows = []
    for ep in eps:
        a, b = v1[ep], v2[ep]
        if "error" in a or "error" in b:
            continue
        row = dict(episode=ep, n_frames=a["n_frames"], b050_v1=a["birth_f050"], b050_v2=b["birth_f050"],
                   b100_v1=a["birth_f100"], b100_v2=b["birth_f100"],
                   ext1_extrinsics=b.get("ext1_extrinsics", ""), ext2_extrinsics=b.get("ext2_extrinsics", ""))
        for k in ("b050", "b100"):
            x, y = row[f"{k}_v1"], row[f"{k}_v2"]
            row[f"{k}_change"] = ("same" if x == y else "recovered" if x < 0 <= y else "lost" if y < 0 <= x else "moved")
            row[f"{k}_delta"] = (y - x) if (x >= 0 and y >= 0) else ""
        rows.append(row)
    keys = list(rows[0].keys())
    with open(f"{out_dir}/compare_v1_v2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    L = ["# Birth frames: v1 (raw DROID extrinsics) vs v2 (PointWorld optimized_extrinsics)", ""]
    L.append(f"scenes compared: {len(rows)} (v1 {len(v1)}, v2 {len(v2)})")
    fb = collections.Counter(r["ext1_extrinsics"] for r in rows) + collections.Counter(r["ext2_extrinsics"] for r in rows)
    L.append(f"v2 extrinsics used per camera: {dict(fb)}")
    L.append("")
    for k in ("b050", "b100"):
        c = collections.Counter(r[f"{k}_change"] for r in rows)
        d = np.array([r[f"{k}_delta"] for r in rows if r[f"{k}_change"] == "moved"], float)
        L.append(f"## {k}")
        L.append("")
        L.append(f"- same {c['same']}, moved {c['moved']}, recovered (v1 -1 -> v2 frame) {c['recovered']}, lost (v1 frame -> v2 -1) {c['lost']}")
        if len(d):
            ad = np.abs(d)
            L.append(f"- moved: v2 earlier in {int((d < 0).sum())}, later in {int((d > 0).sum())}; |delta| frames median {np.median(ad):.0f}, "
                     f"p90 {np.percentile(ad, 90):.0f}, max {ad.max():.0f}; |delta| <= 2: {int((ad <= 2).sum())}, 3-10: {int(((ad > 2) & (ad <= 10)).sum())}, "
                     f"11-30: {int(((ad > 10) & (ad <= 30)).sum())}, > 30: {int((ad > 30).sum())}")
            L.append(f"- moved, with v2 frame 0 (gripper visible from the start under PointWorld): {sum(1 for r in rows if r[f'{k}_change']=='moved' and r[f'{k}_v2']==0)}")
        L.append("")
    L.append("## Per lab (b050 changed = moved or recovered or lost)")
    L.append("")
    L.append("| lab | scenes | changed | moved | recovered | lost | median \\|delta\\| (moved) |")
    L.append("|---|---|---|---|---|---|---|")
    bylab = collections.defaultdict(list)
    for r in rows:
        bylab[r["episode"].split("+")[0]].append(r)
    for lab, rs in sorted(bylab.items(), key=lambda kv: -len(kv[1])):
        c = collections.Counter(r["b050_change"] for r in rs)
        d = [abs(r["b050_delta"]) for r in rs if r["b050_change"] == "moved"]
        L.append(f"| {lab} | {len(rs)} | {len(rs)-c['same']} | {c['moved']} | {c['recovered']} | {c['lost']} | {np.median(d) if d else '-':} |")
    L.append("")
    L.append("## The 30 v1 scenes with no birth frame")
    L.append("")
    L.append("| scene | v1 verdict | v2 b050 | v2 b100 | ext1 max frac v2 | ext2 max frac v2 |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        if r["b050_v1"] < 0:
            b = v2[r["episode"]]
            L.append(f"| {r['episode']} | {verdict.get(r['episode'], '')} | {r['b050_v2']} | {r['b100_v2']} | {b['ext1_max_frac']:.2f} | {b['ext2_max_frac']:.2f} |")
    L.append("")
    reseg = sorted(r["episode"] for r in rows if r["b050_change"] != "same" or r["b100_change"] != "same")
    retrack = sorted(r["episode"] for r in rows if r["b050_change"] in ("moved", "recovered"))
    lost = sorted(r["episode"] for r in rows if r["b050_change"] == "lost")
    open(f"{out_dir}/scenes_to_resegment.txt", "w").write("\n".join(reseg) + "\n")
    open(f"{out_dir}/items_to_retrack.txt", "w").write("".join(f"{ep}:{cam}\n" for ep in retrack for cam in ("ext1", "ext2")))
    open(f"{out_dir}/scenes_lost_in_v2.txt", "w").write("\n".join(lost) + "\n")
    L.append(f"scenes to re-segment (b050 or b100 changed): {len(reseg)} -> `scenes_to_resegment.txt`; "
             f"(scene, camera) items to re-track (b050 moved or recovered): {2*len(retrack)} -> `items_to_retrack.txt`; "
             f"scenes with a v1 birth frame but none in v2: {len(lost)} -> `scenes_lost_in_v2.txt`")
    open(f"{out_dir}/compare_v1_v2.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
