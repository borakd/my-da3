#!/usr/bin/env python
"""Markdown table from check_seed_masks.py summaries: per-scene product checks, presence, and the GT diagnostic
(fraction of in-view frames whose projected GT wrist-camera centre lies within --tol px of the mask). GT is scoring only.

    python seed_summary_table.py --method motion_only [--method sam3_boxclick ...]  -> <root>/check/summary.md per method
"""
import argparse, json, os
import numpy as np

SEEDSTUDY = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy"


def table(method, root):
    summ = json.load(open(f"{root}/check/summary.json"))
    lines = [f"# {method}: seed-mask check on {len(summ)} scenes", "",
             "GT diagnostic (scoring only): `in` = fraction of frames with the GT wrist-camera centre in view whose centre lies within 25 px of the mask; "
             "`rec` = fraction of in-view frames with a non-empty mask; `far` = frames with the centre > 100 px from the mask; `area` = mean mask area (image fraction).", "",
             "| scene | T | prod | ext1 present | ext1 rec | ext1 in | ext1 far | ext1 area | ext2 present | ext2 rec | ext2 in | ext2 far | ext2 area |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    agg = dict(ext1=[], ext2=[])
    for r in summ:
        prod = all(r[c].get("T_matches_store") and r[c].get("shape_ok") and r[c].get("packed_shape_ok") and r[c].get("present_consistent") for c in ("ext1", "ext2"))
        cells = [r["episode"], str(r["store_frames"]), "ok" if prod else "FAIL"]
        for c in ("ext1", "ext2"):
            x = r[c]
            cells += [f"{x['frames_present']}/{x['T']}", f"{x['recall_in_view']:.2f}", f"{x['inside_tol_rate']:.2f}", str(x["far_frames_gt100px"]), f"{x['mean_area_frac']:.3f}" if x["mean_area_frac"] is not None else "-"]
            agg[c].append((x["gt_in_view"], x["present_when_in_view"], x["inside_tol_rate"] * x["present_when_in_view"], x["far_frames_gt100px"]))
        lines.append("| " + " | ".join(cells) + " |")
    tot = []
    for c in ("ext1", "ext2"):
        a = np.array(agg[c], float)
        tot.append(f"{c}: in-view frames {int(a[:,0].sum())}, present {a[:,1].sum()/a[:,0].sum():.2f}, centre within 25 px {a[:,2].sum()/max(a[:,1].sum(),1):.2f} (frame-weighted), "
                   f"scene-mean {np.mean([r[c]['inside_tol_rate'] for r in summ]):.2f}, far frames {int(a[:,3].sum())}")
    lines += ["", "Totals: " + "; ".join(tot), ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", default=None)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    for m in a.method or ["motion_only"]:
        root = a.root or f"{SEEDSTUDY}/{m}"
        md = table(m, root)
        open(f"{root}/check/summary.md", "w").write(md)
        print(md)


if __name__ == "__main__":
    main()
