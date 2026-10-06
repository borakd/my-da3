#!/usr/bin/env python
"""Summarise birth_frames.json over the test split: distribution of birth frames, threshold
sensitivity, per-lab breakdown, and the scenes where the gripper is never seen by both cameras.
Writes summary.md and birth_frames_hist.png next to the input."""
import argparse
import collections
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames/birth_frames.json")
    args = ap.parse_args()
    rows = json.load(open(args.inp))
    out_dir = os.path.dirname(args.inp)
    ok = [r for r in rows if "error" not in r]
    err = [r for r in rows if "error" in r]
    lines = [f"# Gripper birth frames, DROID wrist test split ({len(rows)} scenes)", ""]
    lines.append(f"analysed: {len(ok)}   errors: {len(err)}  " +
                 (f"({collections.Counter(r['error'].split(':')[0] for r in err)})" if err else ""))
    nf = np.array([r["n_frames"] for r in ok])
    lines.append(f"frames per scene: median {np.median(nf):.0f}, min {nf.min()}, max {nf.max()}")
    lines.append("")
    lines.append("## Birth frame by visibility threshold (fraction of the 54 gripper points inside both images)")
    lines.append("")
    lines.append("| threshold | never (both) | birth = 0 | median birth | p90 | median birth / n_frames |")
    lines.append("|---|---|---|---|---|---|")
    for thr in ("025", "050", "075", "100"):
        b = np.array([r[f"birth_f{thr}"] for r in ok])
        never = (b < 0).sum()
        v = b[b >= 0]
        rel = (v / nf[b >= 0])
        lines.append(f"| {int(thr)}% | {never} ({100*never/len(b):.1f}%) | {(v==0).sum()} ({100*(v==0).sum()/len(b):.1f}%) | "
                     f"{np.median(v):.0f} | {np.percentile(v,90):.0f} | {np.median(rel):.2f} |")
    lines.append("")
    b50 = np.array([r["birth_f050"] for r in ok])
    e1 = np.array([r["ext1_first_f050"] for r in ok])
    e2 = np.array([r["ext2_first_f050"] for r in ok])
    lines.append("## Which camera is the bottleneck (50% threshold)")
    lines.append("")
    both = (e1 >= 0) & (e2 >= 0)
    lines.append(f"- ext1 sees the gripper first in {(e1[both] < e2[both]).sum()} scenes, ext2 first in {(e2[both] < e1[both]).sum()}, same frame in {(e1[both] == e2[both]).sum()}")
    lines.append(f"- never seen by ext1 at 50%: {(e1 < 0).sum()}; never by ext2: {(e2 < 0).sum()}; never by both: {((e1 < 0) & (e2 < 0)).sum()}")
    gap = np.abs(e1[both] - e2[both])
    lines.append(f"- |ext1 - ext2| entry gap: median {np.median(gap):.0f} frames, p90 {np.percentile(gap, 90):.0f}")
    lines.append("")
    lines.append("## Per lab (50% threshold)")
    lines.append("")
    lines.append("| lab | scenes | never | birth = 0 | median birth | p90 |")
    lines.append("|---|---|---|---|---|---|")
    bylab = collections.defaultdict(list)
    for r in ok:
        bylab[r["episode"].split("+")[0]].append(r["birth_f050"])
    for lab, bs in sorted(bylab.items(), key=lambda kv: -len(kv[1])):
        bs = np.array(bs)
        v = bs[bs >= 0]
        lines.append(f"| {lab} | {len(bs)} | {(bs < 0).sum()} | {(v == 0).sum()} | {np.median(v) if len(v) else float('nan'):.0f} | {np.percentile(v, 90) if len(v) else float('nan'):.0f} |")
    lines.append("")
    post = np.array([r["post_birth_both_visible_frac"] for r in ok])
    lines.append("## Stability after birth (50%)")
    lines.append("")
    lines.append(f"- fraction of post-birth frames where both cameras still see >= 50% of the gripper: median {np.median(post[b50>=0]):.2f}, "
                 f"10th percentile {np.percentile(post[b50>=0], 10):.2f}; scenes below 0.5: {(post[b50>=0] < 0.5).sum()}")
    lines.append("")
    never = [r["episode"] for r in ok if r["birth_f050"] < 0]
    lines.append(f"## Scenes never visible to both cameras at 50% ({len(never)})")
    lines.append("")
    for ep in never[:60]:
        r = next(x for x in ok if x["episode"] == ep)
        lines.append(f"- {ep}  ext1 max {r['ext1_max_frac']:.2f}  ext2 max {r['ext2_max_frac']:.2f}")
    if len(never) > 60:
        lines.append(f"- ... {len(never) - 60} more (see csv)")
    open(f"{out_dir}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))

    # histogram (small multiples by threshold), palette: categorical slot 1 light
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID, C1 = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), facecolor=SURF, sharey=True)
    for ax, thr in zip(axes, ("025", "050", "075", "100")):
        b = np.array([r[f"birth_f{thr}"] for r in ok])
        v = b[b >= 0]
        ax.set_facecolor(SURF)
        ax.hist(np.clip(v, 0, 200), bins=np.arange(0, 205, 5), color=C1, edgecolor=SURF, linewidth=0.5)
        ax.set_title(f"threshold {int(thr)}%   never: {(b < 0).sum()}   at 0: {(v == 0).sum()}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("birth frame (clipped at 200)", fontsize=8, color=INK2)
        ax.grid(color=GRID, lw=0.8, axis="y")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].set_ylabel("scenes", fontsize=8, color=INK2)
    fig.suptitle(f"Gripper birth frame (first frame visible in both exterior cameras), {len(ok)} DROID test scenes", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/birth_frames_hist.png", dpi=130, bbox_inches="tight", facecolor=SURF)
    print("wrote", f"{out_dir}/summary.md", f"{out_dir}/birth_frames_hist.png")


if __name__ == "__main__":
    main()
