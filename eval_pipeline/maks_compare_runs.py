#!/usr/bin/env python3
"""Compare a baseline per-scene eval CSV against one or more controlled runs
(same scene, same checkpoint, some frames' memory writes blocked).

For every control:
  - scene-level table: the five metrics for baseline and control, delta, better/worse
  - parity gate: frames BEFORE the first blocked frame must be byte-identical to
    the baseline predictions (proves the block took effect exactly at the boundary)
  - a figure: five panels, baseline vs control per frame, blocked span shaded,
    plus a panel of the per-frame difference in aggregate badness
Writes <out_dir>/compare_<name>.png and one <out_dir>/compare_summary.md.

  python maks_compare_runs.py --baseline_csv .. --baseline_preds .. --gt_frames N \
      --control NAME=CSV:PREDS:SKIPJSON [--control ...] --out_dir DIR
"""
import argparse
import csv
import json
import math
import os

import numpy as np

METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
LABEL = {"absrel": "AbsRel", "a1": "a1 (δ<1.25)", "ate": "ATE (m)",
         "rpe_trans": "RPE trans (m)", "rpe_rot": "RPE rot (°)"}
HIGHER = {"a1"}
BASE_C = "#2a78d6"; CTRL_C = "#eb6834"; BLOCK_C = "#52514e"
SURFACE = "#fcfcfb"; GRID = "#e6e5e1"; TEXT = "#0b0b0b"; TEXT2 = "#52514e"


def load(path):
    ts, cols, scene = [], {m: [] for m in METRICS}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            cam, lt = r["camera_id"].strip(), r["local_timestep"].strip()
            if lt == "MEAN":
                if cam in ("0", "ALL") and not scene:
                    scene = {m: _f(r[m]) for m in METRICS}
                continue
            if cam != "0":
                continue
            try:
                t = int(lt)
            except ValueError:
                continue
            ts.append(t)
            for m in METRICS:
                cols[m].append(_f(r[m]))
    o = np.argsort(ts)
    return np.asarray(ts)[o], {m: np.asarray(v, float)[o] for m, v in cols.items()}, scene


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def robust_z(x, m):
    v = 1.0 - x if m == "a1" else x.copy()
    ok = np.isfinite(v)
    med = np.median(v[ok]); mad = np.median(np.abs(v[ok] - med)) * 1.4826 or (np.std(v[ok]) or 1.0)
    z = np.full_like(v, np.nan); z[ok] = (v[ok] - med) / mad
    return z


def same(a, b):
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def fmt(m, v):
    return "n/a" if not np.isfinite(v) else (f"{v:.2f}" if m == "rpe_rot" else f"{v:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", required=True)
    ap.add_argument("--baseline_preds", required=True, help="<preds>/<scene> of the baseline")
    ap.add_argument("--control", action="append", required=True, help="NAME=CSV:PREDS:SKIPJSON")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ts, base, base_scene = load(args.baseline_csv)
    n = len(ts)
    zb = {m: robust_z(base[m], m) for m in METRICS}
    Bb = np.nanmean(np.vstack([zb[m] for m in METRICS]), axis=0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = [f"# Memory-write blocking controls: {args.title}", "",
         "Each control re-runs inference with ONE decline window's frames blocked from writing state/memory "
         "(frames still decoded + scored, all frames scored). Baseline = plain run. Scene values are the "
         "table convention (depth: mean over frames; pose: RMSE over frames, Sim3-aligned).", ""]
    L += ["| run | blocked frames | absrel | a1 | ate | rpe_trans | rpe_rot | wins / 5 | parity gate |",
          "|---|---|---|---|---|---|---|---|---|"]
    L.append("| baseline | – | " + " | ".join(fmt(m, base_scene[m]) for m in METRICS) + " | – | – |")

    for spec in args.control:
        name, rest = spec.split("=", 1)
        c_csv, c_preds, skip_json = rest.split(":")
        cts, ctrl, c_scene = load(c_csv)
        assert len(cts) == n, f"{name}: {len(cts)} rows vs baseline {n}"
        bj = json.load(open(skip_json))
        blocked = sorted(int(x) for x in (next(iter(bj.values())) if isinstance(bj, dict) else bj))
        b0, b1 = blocked[0], blocked[-1]
        runs, cur = [], [blocked[0], blocked[0]]
        for t in blocked[1:]:
            if t == cur[1] + 1:
                cur[1] = t
            else:
                runs.append(tuple(cur)); cur = [t, t]
        runs.append(tuple(cur))
        span_s = ", ".join(f"{a}–{b}" for a, b in runs)

        # parity gate: identical before the block, different after it starts
        i0 = int(np.searchsorted(ts, b0))
        pre_ok = all(same(os.path.join(args.baseline_preds, "camera", f"{int(t):06d}.npz"),
                          os.path.join(c_preds, "camera", f"{int(t):06d}.npz")) for t in ts[:i0 + 1])
        post_diff = any(not same(os.path.join(args.baseline_preds, "camera", f"{int(t):06d}.npz"),
                                 os.path.join(c_preds, "camera", f"{int(t):06d}.npz")) for t in ts[i0 + 1:])
        gate = "PASS" if (pre_ok and post_diff) else f"FAIL (pre_identical={pre_ok}, post_differs={post_diff})"

        wins = 0
        cells = []
        for m in METRICS:
            d = c_scene[m] - base_scene[m]
            better = (d > 0) if m in HIGHER else (d < 0)
            wins += int(better)
            cells.append(f"{fmt(m, c_scene[m])} ({'+' if d >= 0 else '−'}{fmt(m, abs(d))}, {'better' if better else 'worse'})")
        L.append(f"| {name} | {span_s} ({len(blocked)}) | " + " | ".join(cells) + f" | {wins} | {gate} |")

        # per-frame effect inside / after the block
        zc = {m: robust_z(ctrl[m], m) for m in METRICS}
        Bc = np.nanmean(np.vstack([zc[m] for m in METRICS]), axis=0)
        inb = np.array([int(t) in set(blocked) for t in ts])
        after = (ts > b0) & ~inb
        def mean_delta(m, mask):
            d = (ctrl[m] - base[m])[mask]
            return float(np.nanmean(d)) if np.isfinite(d).any() else math.nan
        L.append("")
        L.append(f"- **{name}** per-frame mean change (control − baseline), on blocked frames vs on unblocked frames after the first block: "
                 + "; ".join(f"{m} {fmt(m, mean_delta(m, inb))} / {fmt(m, mean_delta(m, after))}" for m in METRICS))
        L.append("")

        # figure
        fig, axes = plt.subplots(len(METRICS) + 1, 1, figsize=(12, 2.1 * (len(METRICS) + 1)), sharex=True)
        fig.patch.set_facecolor(SURFACE)
        for ax, m in zip(axes[:-1], METRICS):
            ax.set_facecolor(SURFACE)
            for sde in ("top", "right"): ax.spines[sde].set_visible(False)
            ax.grid(True, axis="y", color=GRID, lw=1); ax.tick_params(colors=TEXT2, labelsize=8, length=0)
            for a, b in runs:
                ax.axvspan(a - 0.5, b + 0.5, color=BLOCK_C, alpha=0.12, lw=0)
            ax.plot(ts, base[m], color=BASE_C, lw=2, label="baseline")
            ax.plot(ts, ctrl[m], color=CTRL_C, lw=2, label=f"{name}: block {span_s}")
            ax.set_ylabel(LABEL[m], fontsize=9, color=TEXT)
            if m == "a1": ax.set_ylim(0, 1.02)
            else: ax.set_ylim(bottom=0)
            ax.set_title(f"{LABEL[m]}: scene {fmt(m, base_scene[m])} → {fmt(m, c_scene[m])}", loc="left", fontsize=9, color=TEXT2)
        axes[0].legend(loc="upper left", fontsize=8, frameon=False)
        ax = axes[-1]
        ax.set_facecolor(SURFACE)
        for sde in ("top", "right"): ax.spines[sde].set_visible(False)
        ax.grid(True, axis="y", color=GRID, lw=1); ax.tick_params(colors=TEXT2, labelsize=8, length=0)
        for a, b in runs:
            ax.axvspan(a - 0.5, b + 0.5, color=BLOCK_C, alpha=0.12, lw=0)
        dB = Bc - Bb
        ax.fill_between(ts, 0, np.where(dB > 0, dB, 0), color="#e34948", alpha=0.6, lw=0)
        ax.fill_between(ts, np.where(dB < 0, dB, 0), 0, color="#1baf7a", alpha=0.6, lw=0)
        ax.axhline(0, color=TEXT2, lw=1)
        ax.set_ylabel("Δ aggregate badness\n(control − baseline)", fontsize=9, color=TEXT)
        ax.set_xlabel("frame #", fontsize=9, color=TEXT)
        ax.set_title("red = control worse than baseline on that frame, green = better; grey = blocked frames", loc="left", fontsize=9, color=TEXT2)
        fig.suptitle(f"{args.title} — {name}: memory write blocked on frames {span_s}", x=0.01, ha="left", fontsize=11, color=TEXT)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        png = os.path.join(args.out_dir, f"compare_{name}.png")
        fig.savefig(png, dpi=130, facecolor=SURFACE); plt.close(fig)
        L.append(f"Figure: `{png}`"); L.append("")

    md = os.path.join(args.out_dir, "compare_summary.md")
    open(md, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
