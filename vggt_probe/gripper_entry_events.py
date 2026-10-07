#!/usr/bin/env python
"""Gripper entry events per scene: every frame where the gripper NEWLY enters the view of both exterior
cameras (v3 re-entries; the first entry per scene is the v2 birth frame).

Definition (agreed with MN5, 2026-10-07):
  both[t] = min(frac_ext1[t], frac_ext2[t]) from birth_frames.analyze_episode(ep, keep_curves=True) with the
  v2 settings (PointWorld extrinsics, raw fallback where the serial is absent); vis[t] = both[t] >= 0.5;
  runs of vis (start, end exclusive); consecutive runs are merged when next_start - prev_end < GAP (3);
  each merged run is one event with entry = run start, exit = run end (= n_frames if it lasts to the end).
  Event 0 is the first entry and must equal the v2 birth_f050. Scenes in --exclude are dropped.

Outputs in --out (default $OUT_ROOT/events_v3): curves.jsonl (per scene both[t], resumable), events.csv
(episode, event, entry_frame, exit_frame, n_frames, n_events), reentries.csv (events >= 1), summary.md.
"""
import argparse
import collections
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import birth_frames as bf  # noqa: E402

GAP = 3
THR = 0.5


def runs_of(vis):
    """[(start, end)] of True runs, end exclusive."""
    out, start = [], None
    for t, v in enumerate(vis):
        if v and start is None:
            start = t
        elif not v and start is not None:
            out.append((start, t)); start = None
    if start is not None:
        out.append((start, len(vis)))
    return out


def events_from_both(both, thr=THR, gap=GAP):
    rs = runs_of(np.asarray(both) >= thr)
    merged = []
    for s, e in rs:
        if merged and s - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged, rs


def _worker(ep):
    try:
        r = bf.analyze_episode(ep, keep_curves=True)
        if "error" in r:
            return {"episode": ep, "error": r["error"]}
        both = np.minimum(r["curves"]["ext1"], r["curves"]["ext2"])
        return {"episode": ep, "n_frames": r["n_frames"], "birth_f050": r["birth_f050"],
                "ext1_extrinsics": r.get("ext1_extrinsics"), "ext2_extrinsics": r.get("ext2_extrinsics"),
                "both": np.round(both, 3).tolist()}
    except Exception as e:  # noqa: BLE001
        return {"episode": ep, "error": f"exc:{e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{bf.OUT_ROOT}/events_v3")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extrinsics", choices=("raw", "pointworld"), default="pointworld")
    ap.add_argument("--v2_csv", default=f"{bf.OUT_ROOT}/birth_frames_v2/birth_frames.csv")
    ap.add_argument("--exclude", default=f"{bf.OUT_ROOT}/trackon_run_v2/scenes_to_exclude.tsv")
    ap.add_argument("--gap", type=int, default=GAP)
    args = ap.parse_args()
    bf.EXTRINSICS = args.extrinsics
    os.makedirs(args.out, exist_ok=True)
    eps = sorted(d for d in os.listdir(bf.STORE) if os.path.isdir(f"{bf.STORE}/{d}"))
    jsonl = f"{args.out}/curves.jsonl"
    done = {}
    if os.path.isfile(jsonl):
        for line in open(jsonl):
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done[r["episode"]] = r
    todo = [ep for ep in eps if ep not in done]
    print(f"{len(eps)} scenes, {len(done)} curves cached, {len(todo)} to compute", flush=True)
    rows = dict(done)
    with open(jsonl, "a") as fh, ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_worker, ep) for ep in todo]
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if "error" not in r:
                rows[r["episode"]] = r
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(todo)}", flush=True)
    excl = set()
    if os.path.isfile(args.exclude):
        excl = {l.split("\t")[0].strip() for l in open(args.exclude) if l.strip()}
    v2 = {r["episode"]: int(r["birth_f050"]) for r in csv.DictReader(open(args.v2_csv))} if os.path.isfile(args.v2_csv) else {}
    ev_rows, mism, n_nodeb = [], [], 0
    for ep in eps:
        r = rows.get(ep)
        if r is None:
            continue
        merged, raw_runs = events_from_both(r["both"], gap=args.gap)
        n_nodeb += len(raw_runs)
        first = merged[0][0] if merged else -1
        if ep in v2 and v2[ep] != first:
            mism.append((ep, v2[ep], first))
        if ep in excl:
            continue
        for k, (s, e) in enumerate(merged):
            ev_rows.append(dict(episode=ep, event=k, entry_frame=s, exit_frame=e, n_frames=r["n_frames"], n_events=len(merged)))
    keys = ["episode", "event", "entry_frame", "exit_frame", "n_frames", "n_events"]
    for name, sub in (("events.csv", ev_rows), ("reentries.csv", [x for x in ev_rows if x["event"] >= 1])):
        with open(f"{args.out}/{name}", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(sub)
    re_ = [x for x in ev_rows if x["event"] >= 1]
    scenes = sorted(set(x["episode"] for x in ev_rows))
    per = collections.Counter(x["episode"] for x in ev_rows)
    L = ["# Gripper entry events (both exterior cameras >= 50 %, PointWorld extrinsics)", ""]
    L.append(f"scenes analysed: {len(rows)} of {len(eps)}; excluded: {len(excl & set(rows))} ({sorted(excl & set(rows))})")
    L.append(f"event 0 == v2 birth_f050: {len(rows) - len(mism)}/{len(rows)}" + (f"; MISMATCHES: {mism[:10]}" if mism else ""))
    L.append(f"after exclusions: {len(ev_rows)} events in {len(scenes)} scenes; re-entries (event >= 1): {len(re_)} in "
             f"{len(set(x['episode'] for x in re_))} scenes; max events in one scene: {max(per.values()) if per else 0}")
    L.append(f"debounce (merge gaps < {args.gap} frames): {n_nodeb} raw runs over all analysed scenes -> "
             f"{sum(per.values()) + sum(len(events_from_both(rows[e]['both'], gap=args.gap)[0]) for e in excl & set(rows))} events")
    L.append(f"scenes with no event at all: {sum(1 for ep in scenes if per[ep] == 0)}; "
             f"scenes with no vis frame (dropped): {sum(1 for ep, r in rows.items() if ep not in excl and not events_from_both(r['both'], gap=args.gap)[0])}")
    hist = collections.Counter(per.values())
    L.append("events per scene: " + ", ".join(f"{k}: {hist[k]}" for k in sorted(hist)))
    ln = [x["exit_frame"] - x["entry_frame"] for x in re_]
    if ln:
        L.append(f"re-entry run length (frames): median {np.median(ln):.0f}, p10 {np.percentile(ln,10):.0f}, p90 {np.percentile(ln,90):.0f}, "
                 f"min {min(ln)}; re-entries lasting to the end of the episode: {sum(x['exit_frame']==x['n_frames'] for x in re_)}")
        rem = [x["n_frames"] - x["entry_frame"] for x in re_]
        L.append(f"frames from re-entry to episode end (tracking length): median {np.median(rem):.0f}, total {sum(rem)}")
    bylab = collections.defaultdict(lambda: [0, 0])
    for ep in scenes:
        bylab[ep.split("+")[0]][0] += 1
    for x in re_:
        bylab[x["episode"].split("+")[0]][1] += 1
    L.append("")
    L.append("| lab | scenes | re-entries |")
    L.append("|---|---|---|")
    for lab, (n, k) in sorted(bylab.items(), key=lambda kv: -kv[1][0]):
        L.append(f"| {lab} | {n} | {k} |")
    open(f"{args.out}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
