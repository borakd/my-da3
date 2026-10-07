"""Aggregate the per-shard logs of track_gripper_birth.py into tracks_index.csv and a summary.

Reports status counts, throughput, visibility statistics (mean visible fraction after birth, fraction
still visible at the last frame), points per item, per-lab breakdown, and lists failures. Also makes
a contact sheet of the LAST frame of a random sample of QA videos (tracks drawn) for a quick look.
"""
import collections
import csv
import glob
import json
import os
import random

OUT = "/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/trackon_run"


def q(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))] if xs else float("nan")


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=OUT, help="run dir (v2: .../trackon_run_v2)")
    OUT = ap.parse_args().run
    rows = []
    for f in sorted(glob.glob(f"{OUT}/logs/track_shard*.jsonl")):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    # keep the last record per (episode, cam) (resumed runs may append twice)
    last = {}
    for r in rows:
        last[(r["episode"], r["cam"])] = r
    rows = list(last.values())
    work = json.load(open(f"{OUT}/worklist.json"))
    keys = ["episode", "cam", "birth_frame", "status", "n_points", "grid_stride", "mask_area", "n_frames_store",
            "n_frames_mp4", "tracked_frames", "vis_frac_mean", "vis_frac_last", "seconds", "fps_tracking", "gpu_mem_gb", "decode_error"]
    with open(f"{OUT}/tracks_index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + ["npz"], extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["episode"], r["cam"])):
            d = {k: r.get(k, "") for k in keys}
            d["npz"] = f"tracks/{r['episode']}/{r['cam']}_f{int(r['birth_frame']):05d}.npz" if r.get("status") == "ok" else ""
            w.writerow(d)
    ok = [r for r in rows if r.get("status") == "ok"]
    L = ["# Track-On gripper tracks from the birth frame (DROID test split)", ""]
    L.append(f"work items: {len(work)}  logged: {len(rows)}  status: {dict(collections.Counter(r.get('status') for r in rows))}")
    L.append(f"scenes with >= 1 track file: {len(set(r['episode'] for r in ok))}; items with both cameras tracked: "
             f"{sum(1 for ep, c in collections.Counter(r['episode'] for r in ok).items() if c == 2)}")
    npz_on_disk = len(glob.glob(f"{OUT}/tracks/*/*.npz"))
    L.append(f"npz files on disk: {npz_on_disk}")
    fr = sum(r["tracked_frames"] for r in ok)
    sec = sum(r["seconds"] for r in ok)
    L.append(f"frames tracked: {fr}  GPU time: {sec/3600:.1f} h  mean throughput: {fr/max(sec,1):.1f} frames/s per GPU")
    L.append(f"GPU peak memory (max over items): {max((r.get('gpu_mem_gb') or 0) for r in ok):.1f} GB")
    L.append("")
    L.append("## Query points")
    npts = [r["n_points"] for r in ok]
    L.append(f"points per item: median {q(npts, .5)}  p10 {q(npts, .1)}  p90 {q(npts, .9)}  min {min(npts)}  (items with < 10 points: {sum(n < 10 for n in npts)})")
    L.append("")
    L.append("## Visibility (Track-On's own flag, threshold 0.8)")
    vm = [r["vis_frac_mean"] for r in ok]
    vl = [r["vis_frac_last"] for r in ok]
    L.append(f"mean visible fraction over the tracked span: median {q(vm, .5):.2f}  p10 {q(vm, .1):.2f}  p90 {q(vm, .9):.2f}")
    L.append(f"visible fraction at the LAST frame: median {q(vl, .5):.2f}  p10 {q(vl, .1):.2f};  items with < 25 % of points visible at the end: {sum(v < .25 for v in vl)} ({100*sum(v < .25 for v in vl)/len(vl):.1f} %)")
    L.append("")
    L.append("## Per lab")
    L.append("")
    L.append("| lab | items | median vis (mean) | median vis (last) | items with < 25 % visible at end |")
    L.append("|---|---|---|---|---|")
    bylab = collections.defaultdict(list)
    for r in ok:
        bylab[r["episode"].split("+")[0]].append(r)
    for lab, rs in sorted(bylab.items(), key=lambda kv: -len(kv[1])):
        L.append(f"| {lab} | {len(rs)} | {q([r['vis_frac_mean'] for r in rs], .5):.2f} | {q([r['vis_frac_last'] for r in rs], .5):.2f} | {sum(r['vis_frac_last'] < .25 for r in rs)} |")
    bad = [r for r in rows if r.get("status") != "ok"]
    L.append("")
    L.append(f"## Failures ({len(bad)})")
    L.append("")
    for r in bad[:40]:
        L.append(f"- {r['episode']} {r['cam']} f{r.get('birth_frame')}: {r.get('status')} {r.get('decode_error', '')}")
    # QA sheet: last frame of sampled QA videos
    try:
        import cv2
        import imageio
        import numpy as np
        vids = sorted(glob.glob(f"{OUT}/qa_videos/*.mp4"))
        random.seed(0)
        pick = random.sample(vids, min(24, len(vids)))
        tiles = []
        for v in pick:
            r = imageio.get_reader(v)
            n = r.count_frames()
            for t in (n // 2, n - 1):
                img = cv2.cvtColor(r.get_data(max(t, 0)), cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (480, 270))
                cv2.putText(img, os.path.basename(v)[:34] + f" f{t}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                tiles.append(img)
        if tiles:
            cols = 4
            while len(tiles) % cols:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(f"{OUT}/qa_tracks_sheet.jpg", np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
            L.append("")
            L.append(f"QA: `qa_videos/` ({len(vids)} rendered items), `qa_tracks_sheet.jpg` (mid and last frame of {len(pick)} of them).")
    except Exception as e:  # noqa: BLE001
        L.append(f"(QA sheet failed: {e})")
    L.append("")
    L.append("Files: `tracks/<ep>/<cam>_f<birth>.npz` with tracks (T,N,2) float32 in 1280x720 px (NaN before birth), visibility (T,N) bool,")
    L.append("queries (N,3) [t,x,y], meta (json: birth_frame, n_frames_store, n_frames_mp4, fps, store_scale=0.25 for the 320x180 ext store, support_grid, delta_v, ckpt).")
    L.append("`tracks_index.csv` is the per-item index.")
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
