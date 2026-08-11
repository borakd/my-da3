#!/usr/bin/env python
"""patch_probe_verdict.py — re-evaluate a probe JSON's verdict after the gate changed.

The 2026-08-12 probe runs were scored against the superseded P3.4 targets (see
probe_gru_input_stats.py's VALIDATION REFERENCE) and their JSONs therefore carry
a stale "FAILED VALIDATION" verdict over perfectly good measurements. The
MEASUREMENTS are unaffected by the gate change; only the derived verdict fields
are. Re-running 800 scenes on a GPU purely to recompute a verdict that is a pure
function of three already-stored scalars would be waste, so this recomputes them
in place through probe_gru_input_stats.build_verdict — the same code path the
probe itself uses, so the two cannot drift.

Rewrites: validation, implied_head_scale, superseded_expected,
all_checks_passed, wiring_checks_passed, scale_checks_passed. Stamps
meta.verdict_recomputed so a reader can tell this JSON's verdict was scored
after the fact. Everything measured is left byte-identical.

    python patch_probe_verdict.py gru_input_stats.json [more.json ...]
"""
import json
import sys

import probe_gru_input_stats as P


def patch(path):
    with open(path) as f:
        d = json.load(f)
    dn = d.get("delta_t_norm")
    has_delta = dn is not None
    absT = d["absT_norm"]["mean_graded"]
    p50 = dn["p50"] if has_delta else float("nan")
    p95 = dn["p95"] if has_delta else float("nan")

    hs, validation = P.build_verdict(absT, p50, p95, has_delta)
    d["implied_head_scale"] = hs
    d["validation"] = validation
    d["superseded_expected"] = P.SUPERSEDED_EXPECTED
    d["all_checks_passed"] = all(c["passed"] for c in validation)
    d["wiring_checks_passed"] = all(
        c["passed"] for c in validation if c["kind"] == "wiring")
    d["scale_checks_passed"] = all(
        c["passed"] for c in validation if c["kind"] == "regression")
    d["meta"]["verdict_recomputed"] = (
        "2026-08-12: scored by patch_probe_verdict.py against the measured "
        "expectations; measurements unchanged from the original run")

    with open(path, "w") as f:
        json.dump(d, f, indent=2)

    print(f"{path}")
    print(f"  head scale: dt_p50 {hs['from_dt_p50']:.2f}x  dt_p95 "
          f"{hs['from_dt_p95']:.2f}x  absT {hs['from_absT_sup']:.2f}x")
    for c in validation:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['kind']:<10} "
              f"{c['name']:<34} = {c['value']:.5f}")
    print(f"  -> wiring {d['wiring_checks_passed']}, "
          f"regression {d['scale_checks_passed']}")
    return d["all_checks_passed"]


if __name__ == "__main__":
    ok = all([patch(p) for p in sys.argv[1:]])
    sys.exit(0 if ok else 1)
