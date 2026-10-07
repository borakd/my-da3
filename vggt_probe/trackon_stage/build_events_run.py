#!/usr/bin/env python
"""v3 re-entries: Track-On work list from the re-entry masks, and the index over ALL gripper entry events.

  --phase worklist   masks_run_v3_reentries/results_refined.jsonl (status ok, which = e01, e02, ...) ->
                     trackon_run_v3_reentries/worklist.json, one item per (scene, camera, event), frame = entry
                     frame (track_gripper_birth.py then writes tracks/<ep>/<cam>_f<entry>.npz).
  --phase index      events_v3/events.csv x {v2 first entries (trackon_run_v2: worklist.json + tracks_index.csv),
                     v3 re-entries (trackon_run_v3_reentries: logs + mask results)} ->
                     trackon_run_v3_reentries/events_index.csv: episode, cam, event, entry_frame, exit_frame,
                     n_frames, run, npz (relative to OUT_ROOT), status, n_points, tracked_frames, vis_frac_mean,
                     vis_frac_last, mask_prompt, projection_disagree, auto_empty, mask_area; plus events_index_summary.md.
Never writes into trackon_run/ or trackon_run_v2/.
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # vggt_probe
import birth_frames as bf  # noqa: E402

RAW_ROOT = "/leonardo_scratch/large/userexternal/bdursun0/robotseg_demo/raw"
O = bf.OUT_ROOT
V2, V3, M3, EV = f"{O}/trackon_run_v2", f"{O}/trackon_run_v3_reentries", f"{O}/masks_run_v3_reentries", f"{O}/events_v3"


def mask_rows(run):
    src = f"{run}/results_refined.jsonl"
    files = [src] if os.path.isfile(src) else sorted(glob.glob(f"{run}/results_shard*.jsonl"))
    rows = []
    for f in files:
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("worklist", "index"), required=True)
    args = ap.parse_args()
    events = list(csv.DictReader(open(f"{EV}/events.csv")))
    ev_by = {(e["episode"], int(e["event"])): e for e in events}
    os.makedirs(V3, exist_ok=True)
    if args.phase == "worklist":
        items = []
        for r in mask_rows(M3):
            if r.get("status") != "ok" or not str(r["which"]).startswith("e"):
                continue
            k = int(r["which"][1:])
            e = ev_by[(r["episode"], k)]
            t = int(r["frame"])
            assert t == int(e["entry_frame"]), (r["episode"], k, t, e["entry_frame"])
            items.append(dict(episode=r["episode"], cam=r["cam"], frame=t, n_frames=int(r["n_frames"]),
                              mp4_frames=r.get("mp4_frames"), mask=f"{M3}/masks/{r['episode']}/{r['cam']}_f{t:05d}.png",
                              mp4=f"{RAW_ROOT}/{r['episode']}/recordings/MP4/{r['serial']}.mp4",
                              prompt_used=r.get("prompt_used"), projection_disagree=bool(r.get("projection_disagree")),
                              auto_empty=bool(r.get("auto_empty")), area=r.get("area_final"),
                              event=k, exit_frame=int(e["exit_frame"])))
        items.sort(key=lambda w: (w["episode"], w["cam"], w["frame"]))
        json.dump(items, open(f"{V3}/worklist.json", "w"))
        print(f"{len(items)} re-entry items -> {V3}/worklist.json")
        return
    # ---- index over all events
    v2_work = {(w["episode"], w["cam"]): w for w in json.load(open(f"{V2}/worklist.json"))}
    v2_idx = {(r["episode"], r["cam"]): r for r in csv.DictReader(open(f"{V2}/tracks_index.csv"))}
    v3_log = {}
    for f in sorted(glob.glob(f"{V3}/logs/track_shard*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                v3_log[(r["episode"], r["cam"], int(r["birth_frame"]))] = r
    v3_mask = {(r["episode"], r["cam"], r["which"]): r for r in mask_rows(M3)}
    keys = ["episode", "cam", "event", "entry_frame", "exit_frame", "n_frames", "run", "npz", "status", "n_points",
            "tracked_frames", "vis_frac_mean", "vis_frac_last", "mask_prompt", "projection_disagree", "auto_empty", "mask_area"]
    out, stat = [], collections.Counter()
    for e in events:
        ep, k, t = e["episode"], int(e["event"]), int(e["entry_frame"])
        for cam in ("ext1", "ext2"):
            row = dict(episode=ep, cam=cam, event=k, entry_frame=t, exit_frame=int(e["exit_frame"]), n_frames=int(e["n_frames"]))
            if k == 0:
                w, r = v2_work.get((ep, cam)), v2_idx.get((ep, cam))
                row["run"] = "trackon_run_v2"
                if r is None:
                    row["status"] = "no_v2_item"
                else:
                    assert int(r["birth_frame"]) == t, (ep, cam, r["birth_frame"], t)
                    row.update(npz=f"trackon_run_v2/{r['npz']}" if r["npz"] else "", status=r["status"], n_points=r["n_points"],
                               tracked_frames=r["tracked_frames"], vis_frac_mean=r["vis_frac_mean"], vis_frac_last=r["vis_frac_last"])
                if w:
                    row.update(mask_prompt=w["prompt_used"], projection_disagree=w["projection_disagree"], auto_empty=w["auto_empty"], mask_area=w["area"])
            else:
                m, r = v3_mask.get((ep, cam, f"e{k:02d}")), v3_log.get((ep, cam, t))
                row["run"] = "trackon_run_v3_reentries"
                if m:
                    row.update(mask_prompt=m.get("prompt_used"), projection_disagree=m.get("projection_disagree"),
                               auto_empty=m.get("auto_empty"), mask_area=m.get("area_final"))
                if r is None:
                    row["status"] = f"mask_{m.get('status')}" if m and m.get("status") != "ok" else ("not_tracked" if m else "no_mask")
                else:
                    row.update(status=r["status"], n_points=r.get("n_points"), tracked_frames=r.get("tracked_frames"),
                               vis_frac_mean=r.get("vis_frac_mean"), vis_frac_last=r.get("vis_frac_last"),
                               npz=f"trackon_run_v3_reentries/tracks/{ep}/{cam}_f{t:05d}.npz" if r["status"] == "ok" else "")
            stat[(k == 0, row["status"])] += 1
            out.append(row)
    with open(f"{V3}/events_index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for row in out:
            w.writerow({k: row.get(k, "") for k in keys})
    L = ["# Index over all gripper entry events (v2 first entries + v3 re-entries)", ""]
    L.append(f"events: {len(events)} in {len(set(e['episode'] for e in events))} scenes; rows (event x camera): {len(out)}")
    L.append(f"first entries (event 0, trackon_run_v2): {dict((s, n) for (f, s), n in stat.items() if f)}")
    L.append(f"re-entries (event >= 1, trackon_run_v3_reentries): {dict((s, n) for (f, s), n in stat.items() if not f)}")
    re_ = [r for r in out if r["event"] >= 1 and r.get("mask_prompt")]
    if re_:
        L.append(f"re-entry masks: {len(re_)}; projection_disagree {sum(bool(r['projection_disagree']) for r in re_)} "
                 f"({100*sum(bool(r['projection_disagree']) for r in re_)/len(re_):.1f} %); box used (auto empty) "
                 f"{sum(bool(r['auto_empty']) for r in re_)}")
    ok3 = [r for r in out if r["event"] >= 1 and r.get("status") == "ok"]
    if ok3:
        import numpy as np
        L.append(f"re-entry tracks ok: {len(ok3)}; points median {np.median([int(r['n_points']) for r in ok3]):.0f}; "
                 f"visible fraction median {np.median([float(r['vis_frac_mean']) for r in ok3]):.2f}; "
                 f"tracked frames total {sum(int(r['tracked_frames']) for r in ok3)}")
    L.append("")
    L.append("columns: entry_frame = first frame of the event (track start), exit_frame = first frame after the gripper drops "
             "below 50 % in either camera (n_frames if it stays), npz relative to the outputs root, mask flags from the RobotSeg "
             "stage (projection_disagree = mask inconsistent with the PointWorld projection).")
    open(f"{V3}/events_index_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
