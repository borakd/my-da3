#!/usr/bin/env python
"""Step-2 kill/pass gates from the regenerated p43 artifacts (GRU v4 campaign).

(c) CAUSAL: POSE_GRU_PROBE_SHUFFLE must degrade committed e_head vs the clean
    trained run at e10.  PASS > 10% (mid+late), KILL < 5%.
(d) PAIRED: scene-block bootstrap CI of dR2_t(trained) - dR2_t(placebo) at e5,
    mid band, must exclude zero.  Includes zero => KILL (refine demoted, 26h
    launch cancelled).

dR2 from scene_terms sums [Svv,Spp,Svp,Svw,Spw,Sww]: R2_vel = Svw^2/(Svv*Sww);
R2_joint from the 2x2 normal equations over (vel, probe); dR2 = joint - vel.
Same resample indices hit trained and placebo (paired draw).
"""
import json

import numpy as np

BANDS = ["early_t", "mid_t", "late_t", "early_r", "mid_r", "late_r"]


def r2s(terms):
    Svv, Spp, Svp, Svw, Spw, Sww = terms
    if Sww <= 0 or Svv <= 0:
        return 0.0, 0.0
    r2_vel = Svw * Svw / (Svv * Sww)
    det = Svv * Spp - Svp * Svp
    if abs(det) < 1e-18 or Spp <= 0:
        return r2_vel, r2_vel
    a = (Spp * Svw - Svp * Spw) / det
    b = (Svv * Spw - Svp * Svw) / det
    r2_joint = (a * Svw + b * Spw) / Sww
    return r2_vel, r2_joint


def band_dr2(scene_terms, band, idx=None):
    st = scene_terms[band]
    keys = sorted(st, key=int)
    rows = np.array([st[k] for k in keys])
    if idx is not None:
        rows = rows[idx]
    r2v, r2j = r2s(rows.sum(0))
    return r2j - r2v


def paired_boot(tr, pl, band, n=4000, seed=123):
    keys = sorted(tr[band], key=int)
    m = len(keys)
    rng = np.random.RandomState(seed)
    point = band_dr2(tr, band) - band_dr2(pl, band)
    d = [
        band_dr2(tr, band, i) - band_dr2(pl, band, i)
        for i in (rng.randint(0, m, m) for _ in range(n))
    ]
    return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def eh_bands(arm, key):
    s = np.array(arm["series"][key], dtype=float)
    v = np.array(arm["views"], dtype=int)
    return {
        "early": float(np.nanmean(s[(v >= 1) & (v <= 16)])),
        "mid": float(np.nanmean(s[(v >= 17) & (v <= 47)])),
        "late": float(np.nanmean(s[v >= 48])),
    }


def main():
    load = lambda p: json.load(open(p))["arms"][0]
    tr5, tr10 = load("p43_regen_trained_e5.json"), load("p43_regen_trained_e10.json")
    pl5, pl10 = load("p43_regen_placebo_e5.json"), load("p43_regen_placebo_e10.json")
    sh10 = load("p43_regen_shuffle_e10.json")

    print("== (c) CAUSAL: shuffle degradation of committed e_head at e10 ==")
    causal_ok = True
    for key in ("e_head_t", "e_head_r"):
        cb, sb = eh_bands(tr10, key), eh_bands(sh10, key)
        for band in ("early", "mid", "late"):
            pct = 100.0 * (sb[band] - cb[band]) / cb[band] if cb[band] else float("nan")
            print(f"  {key} {band}: clean {cb[band]:.4f} shuffle {sb[band]:.4f}  ({pct:+.1f}%)")
            if key == "e_head_t" and band in ("mid", "late") and pct < 10.0:
                causal_ok = False
    print(f"  CAUSAL GATE: {'PASS' if causal_ok else 'FAIL (<10% mid/late trans)'}")

    print("== (d) PAIRED trained-minus-placebo dR2 (scene-block bootstrap) ==")
    verdicts = {}
    for label, tr, pl in (
        ("e5", tr5["scene_terms"], pl5["scene_terms"]),
        ("e10", tr10["scene_terms"], pl10["scene_terms"]),
    ):
        for band in BANDS:
            pt, lo, hi = paired_boot(tr, pl, band)
            verdicts[f"{label}_{band}"] = (pt, lo, hi)
            print(f"  {label} {band}: ddR2 {pt:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]"
                  f"  {'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'}")
    pt, lo, hi = verdicts["e5_mid_t"]
    gate_d = lo > 0
    print(f"  PAIRED GATE (e5 mid_t CI excludes zero upward): {'PASS' if gate_d else 'KILL'}")

    json.dump(
        dict(causal_gate="PASS" if causal_ok else "FAIL",
             paired_gate="PASS" if gate_d else "KILL",
             paired={k: v for k, v in verdicts.items()}),
        open("eval_pipeline/evidence/p43_regen_gates.json", "w"), indent=1)


if __name__ == "__main__":
    main()
