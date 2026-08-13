#!/usr/bin/env python
"""Committed reduction behind the ckpt-10/30 gate logs (audit MINOR-4).

Bands (early 1-16, mid 17-47, late >=48) of e_head_{t,r} from p43 probe
JSONs; prints clean-vs-shuffle degradation and/or cross-checkpoint trends.

Usage: python eval_pipeline/ckpt_trend_report.py A.json [B.json ...]
  1 file : band table.  2 files: assumes (clean, shuffle) degradation table.
  3+     : trend table across checkpoints (first file = earliest).
"""
import json
import sys

import numpy as np


def bands(arm, key):
    s = np.array(arm["series"][key], dtype=float)
    v = np.array(arm["views"], dtype=int)
    return {b: float(np.nanmean(s[m])) for b, m in
            (("early", (v >= 1) & (v <= 16)), ("mid", (v >= 17) & (v <= 47)),
             ("late", v >= 48))}


def main():
    arms = [json.load(open(p))["arms"][0] for p in sys.argv[1:]]
    names = sys.argv[1:]
    for key in ("e_head_t", "e_head_r"):
        bs = [bands(a, key) for a in arms]
        for band in ("early", "mid", "late"):
            row = " -> ".join(f"{b[band]:.4f}" for b in bs)
            extra = ""
            if len(bs) == 2:
                extra = f"  ({100*(bs[1][band]-bs[0][band])/bs[0][band]:+.1f}%)"
            print(f"{key} {band}: {row}{extra}")
    if len(names) == 2:
        print("(2-file mode: second column interpreted as shuffle/late ckpt)")


if __name__ == "__main__":
    main()
