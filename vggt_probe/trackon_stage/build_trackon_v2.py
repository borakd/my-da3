#!/usr/bin/env python
"""Assemble trackon_run_v2 (PointWorld-extrinsics birth frames) as a complete set.

Two phases, same script:
  --phase worklist   build trackon_run_v2/worklist.json = v1 items for scenes whose b050 frame did not
                     change (mask + frame from masks_run) + v2 items for the changed/recovered scenes
                     (from birth_masks_v2.csv, masks in masks_run_v2); also writes worklist_changed.json
                     (the items to track) and a list of scenes dropped in v2 (no birth frame any more).
  --phase assemble   after track_gripper_birth_v2.sbatch: symlink the unchanged v1 npz files into
                     trackon_run_v2/tracks/, copy their v1 log records into logs/track_shard_v1_reused.jsonl so
                     summarize_gripper_tracks.py --run trackon_run_v2 builds a complete tracks_index.csv.
Never writes into trackon_run/ (v1).
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # vggt_probe
import birth_frames as bf  # noqa: E402

RAW_ROOT = "/leonardo_scratch/large/userexternal/bdursun0/robotseg_demo/raw"
V1 = f"{bf.OUT_ROOT}/trackon_run"
V2 = f"{bf.OUT_ROOT}/trackon_run_v2"


def items_from_masks_csv(path, masks_run, scenes):
    items = []
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok" or r["which"] != "b050" or r["episode"] not in scenes:
            continue
        t = int(r["frame"])
        items.append(dict(
            episode=r["episode"], cam=r["cam"], frame=t, n_frames=int(r["n_frames"]),
            mp4_frames=int(r["mp4_frames"]) if r["mp4_frames"] else None,
            mask=f"{masks_run}/masks/{r['episode']}/{r['cam']}_f{t:05d}.png",
            mp4=f"{RAW_ROOT}/{r['episode']}/recordings/MP4/{r['serial']}.mp4",
            prompt_used=r["prompt_used"], projection_disagree=r["projection_disagree"] == "True",
            auto_empty=r["auto_empty"] == "True", area=float(r["area_final"])))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("worklist", "assemble"), required=True)
    ap.add_argument("--compare_csv", default=f"{bf.OUT_ROOT}/birth_frames_v2/compare_v1_v2.csv")
    ap.add_argument("--masks_v2_csv", default=f"{bf.OUT_ROOT}/birth_masks_v2.csv")
    args = ap.parse_args()
    cmp_rows = list(csv.DictReader(open(args.compare_csv)))
    changed = {r["episode"] for r in cmp_rows if r["b050_change"] in ("moved", "recovered")}
    lost = {r["episode"] for r in cmp_rows if r["b050_change"] == "lost"}
    v1_work = json.load(open(f"{V1}/worklist.json"))
    keep_v1 = [w for w in v1_work if w["episode"] not in changed and w["episode"] not in lost]
    os.makedirs(V2, exist_ok=True)
    if args.phase == "worklist":
        new = items_from_masks_csv(args.masks_v2_csv, f"{bf.OUT_ROOT}/masks_run_v2", changed)
        missing = sorted(changed - {w["episode"] for w in new})
        work = sorted(keep_v1 + new, key=lambda w: (w["episode"], w["cam"]))
        json.dump(work, open(f"{V2}/worklist.json", "w"))
        json.dump(sorted(new, key=lambda w: (w["episode"], w["cam"])), open(f"{V2}/worklist_changed.json", "w"))
        open(f"{V2}/scenes_dropped_in_v2.txt", "w").write("\n".join(sorted(lost)) + "\n")
        print(f"worklist: {len(work)} items = {len(keep_v1)} unchanged v1 + {len(new)} re-segmented; "
              f"changed scenes without a v2 mask: {len(missing)} {missing[:5]}; scenes dropped (no v2 birth frame): {len(lost)}")
        return
    # assemble
    n_link = n_have = 0
    for w in keep_v1:
        stem = f"{w['cam']}_f{w['frame']:05d}.npz"
        src = f"{V1}/tracks/{w['episode']}/{stem}"
        dst = f"{V2}/tracks/{w['episode']}/{stem}"
        if not os.path.isfile(src):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            n_have += 1
            continue
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
        n_link += 1
    keep = {(w["episode"], w["cam"]) for w in keep_v1}
    last = {}
    for f in sorted(glob.glob(f"{V1}/logs/track_shard*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if (r["episode"], r["cam"]) in keep:
                    last[(r["episode"], r["cam"])] = r
    os.makedirs(f"{V2}/logs", exist_ok=True)
    with open(f"{V2}/logs/track_shard_v1_reused.jsonl", "w") as fh:  # matches the summarizer's track_shard*.jsonl glob
        for k in sorted(last):
            r = dict(last[k]); r["reused_from_v1"] = True
            fh.write(json.dumps(r) + "\n")
    allnpz = glob.glob(f"{V2}/tracks/*/*.npz")
    n_new = sum(not os.path.islink(p) for p in allnpz)
    print(f"assemble: {n_link} v1 npz symlinked ({n_have} already present), {len(last)} v1 log records reused, "
          f"{n_new} newly tracked npz; total npz {len(allnpz)}")


if __name__ == "__main__":
    main()
