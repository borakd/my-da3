#!/usr/bin/env python
"""Harness-determinism re-score comparison.

Implements the reading PRE-REGISTERED in GRU_GAP_CLOSURE_DIRECTIVE.md
("PRE-REGISTERED: the harness-determinism re-score must run FIRST") before the
numbers existed. Written while the eval jobs were still pending, deliberately.

Compares the per-scene metric CSVs of two eval labels that scored the SAME
checkpoint. The question is NOT "are the averages close" -- averaging is what
hides the failure mode. It is "does each scene reproduce".

Pre-registered outcomes:
  EXACT      byte-identical on every common scene -> harness/sharding/claim-order
             noise is identically zero; 100% of the 0.00281 arm-to-arm scatter
             is training variance. Proceed to the seed replicate.
  DIFFERS    -> outranks every queued arm. All 12 table rows carry an unmeasured
             harness term and every lever attribution is confounded by it.
  AMBIGUOUS  reproduces on most scenes, differs on a few -> report the per-scene
             diff DISTRIBUTION, never a mean. A few large disagreements and a
             uniform small jitter are different failures with different causes.

A reproducing re-score bounds HARNESS noise only. It is not evidence that any
arm result is real; training variance stays unmeasured until the replicate runs.

Usage:
  python compare_rescore_determinism.py [LABEL_A] [LABEL_B]
"""
import csv
import os
import sys

OUT = os.environ.get(
    "OUT_ROOT", "/gpfs/scratch/etur59/koc821022/outputs"
) + "/cut3r_eval"

A = sys.argv[1] if len(sys.argv) > 1 else "gru_a4g3r8"
B = sys.argv[2] if len(sys.argv) > 2 else "gru_a4g3r8_rescore"
CSV_NAME = "eval_depth_pose_metrics.csv"


def scenes(label):
    d = os.path.join(OUT, label, "eval")
    return set(os.listdir(d)) if os.path.isdir(d) else set()


def rows(label, scene):
    p = os.path.join(OUT, label, "eval", scene, CSV_NAME)
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    sa, sb = scenes(A), scenes(B)
    common = sorted(sa & sb)
    print(f"label A: {A}  ({len(sa)} scenes)")
    print(f"label B: {B}  ({len(sb)} scenes)")
    print(f"common : {len(common)}   only-in-A: {len(sa - sb)}   only-in-B: {len(sb - sa)}")
    if not common:
        print("\nVERDICT: INCONCLUSIVE -- no common scenes to compare.")
        return 2

    identical, differing, unreadable = 0, [], []
    for s in common:
        pa = os.path.join(OUT, A, "eval", s, CSV_NAME)
        pb = os.path.join(OUT, B, "eval", s, CSV_NAME)
        try:
            if open(pa, "rb").read() == open(pb, "rb").read():
                identical += 1
                continue
        except OSError as e:
            unreadable.append((s, str(e)))
            continue
        # Not byte-identical: locate and size the numeric divergence.
        try:
            ra, rb = rows(A, s), rows(B, s)
        except OSError as e:
            unreadable.append((s, str(e)))
            continue
        worst_abs, worst_rel, worst_key = 0.0, 0.0, ""
        for x, y in zip(ra, rb):
            for k in x.keys() & y.keys():
                try:
                    fx, fy = float(x[k]), float(y[k])
                except (TypeError, ValueError):
                    continue
                d = abs(fx - fy)
                if d > worst_abs:
                    worst_abs, worst_key = d, k
                    worst_rel = d / max(abs(fx), abs(fy), 1e-12)
        differing.append((s, worst_abs, worst_rel, worst_key))

    n = len(common) - len(unreadable)
    print(f"\nBYTE-IDENTICAL: {identical}/{n}    DIFFERING: {len(differing)}"
          f"    UNREADABLE: {len(unreadable)}")

    if differing:
        differing.sort(key=lambda t: -t[1])
        mags = [d for _, d, _, _ in differing]
        print("\nper-scene divergence DISTRIBUTION (not a mean -- see module docstring):")
        mags_sorted = sorted(mags)
        for q, lbl in ((0, "min"), (len(mags) // 2, "p50"), (
                int(len(mags) * 0.95), "p95"), (len(mags) - 1, "max")):
            print(f"   {lbl:4} |diff| {mags_sorted[min(q, len(mags) - 1)]:.6g}")
        print("\n   worst 10 scenes:")
        for s, d, r, k in differing[:10]:
            print(f"     {s[:50]:50} {k:>14} |d|={d:.6g} rel={r:.3g}")

    for s, e in unreadable[:5]:
        print(f"   UNREADABLE {s}: {e}")

    print()
    if differing:
        frac = len(differing) / n
        if frac < 0.5:
            print(f"VERDICT: AMBIGUOUS -- {len(differing)}/{n} scenes ({frac:.1%}) differ.")
        else:
            print(f"VERDICT: DOES NOT REPRODUCE -- {len(differing)}/{n} scenes differ.")
        print("Per the pre-registration this OUTRANKS every queued arm: the")
        print("scoreboard carries an unmeasured harness term and every lever")
        print("attribution built on it is confounded. Do NOT spend a training")
        print("slot until the term is bounded.")
        return 1

    print(f"VERDICT: EXACT -- {identical}/{n} scenes byte-identical.")
    print("Harness/sharding/claim-order noise is identically zero, now including")
    print("the 32-GPU claim-queue path the earlier 32-scene check bypassed.")
    print("100% of the 0.00281 arm-to-arm scatter is therefore attributable to")
    print("TRAINING. The seed replicate is the only instrument that can")
    print("calibrate it. This bounds harness noise ONLY -- it is not evidence")
    print("that any arm result is real.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
