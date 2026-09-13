#!/usr/bin/env python3
"""Build the control specs for the six memory-write explorations
(HARMFUL_FRAME_SKIP_PLAN.md follow-up) from a baseline run's decline_windows.json.

For every run it writes, under <scene_dir>/<exp>/masks/:
  <run>.control.json   {scene: spec}   consumed by infer_and_eval_worker.py --control_json
  <run>.affected.json  {scene: [frames]} frames whose commit differs from the baseline
                        (used for shading / parity gates in the plots, video and compare)
and <scene_dir>/explore_manifest.json listing {exp: [run, ...]}.

Explorations (windows = the baseline's decline windows D1..Dk, precise boundaries):
  exp1_preblock        block the L frames immediately BEFORE each window (L = window length)
  exp2_attenuate       soft commit alpha in {0.3, 0.5, 0.7} on all windows, and on D2 alone
  exp3_token_gate      per-state-token commit keyed on the frame's own confidence:
                       top-q tokens commit fully, the rest with gmin; on window frames / all frames
  exp4_reentry_ramp    hard block on windows, then alpha ramp 0.2..1.0 over the 5 frames after each
  exp5_reset           reset state+mem to init at each window's end (with and without the block),
                       or right before each window starts
  exp6_boundary_shift  the ALL block shifted by -3 / +3 frames
"""
import argparse
import json
import os

ALPHAS = [0.3, 0.5, 0.7]
RAMP = [0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decline_json", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n_frames", type=int, required=True)
    ap.add_argument("--scene_dir", required=True)
    args = ap.parse_args()
    n = args.n_frames
    S = args.scene
    wins = [tuple(r["window"]) for r in json.load(open(args.decline_json))["declines"]]
    names = [f"D{k}" for k in range(1, len(wins) + 1)]
    allw = sorted(set(t for a, b in wins for t in range(a, b + 1)))
    d2 = wins[1] if len(wins) > 1 else wins[0]

    runs = {}  # exp -> {run: (spec, affected)}

    def add(exp, run, spec, affected):
        runs.setdefault(exp, {})[run] = (spec, sorted(set(int(t) for t in affected if 0 <= t < n)))

    # exp1: pre-window blocks
    for nm, (a, b) in zip(names, wins):
        L = b - a + 1
        fr = [t for t in range(max(1, a - L), a)]
        add("exp1_preblock", f"pre_{nm}", {"block": fr}, fr)

    # exp2: attenuation
    for al in ALPHAS:
        tag = f"a{int(al*10):02d}"
        add("exp2_attenuate", f"att_{tag}_ALL", {"alpha": {str(t): al for t in allw}}, allw)
        fr = list(range(d2[0], d2[1] + 1))
        add("exp2_attenuate", f"att_{tag}_D2", {"alpha": {str(t): al for t in fr}}, fr)

    # exp3: per-token gating
    add("exp3_token_gate", "tok_q50_windows", {"token_gate": {"frames": allw, "q": 0.5, "gmin": 0.0}}, allw)
    add("exp3_token_gate", "tok_q50_all", {"token_gate": {"frames": "all", "q": 0.5, "gmin": 0.0}}, range(n))
    add("exp3_token_gate", "tok_q50g3_all", {"token_gate": {"frames": "all", "q": 0.5, "gmin": 0.3}}, range(n))
    add("exp3_token_gate", "tok_q25_windows", {"token_gate": {"frames": allw, "q": 0.25, "gmin": 0.0}}, allw)

    # exp4: hard block + re-entry ramp
    def ramp_spec(ws):
        block = sorted(set(t for a, b in ws for t in range(a, b + 1)))
        alpha = {}
        for a, b in ws:
            for j, al in enumerate(RAMP, 1):
                t = b + j
                if t < n and t not in block:
                    alpha[str(t)] = al
        return {"block": block, "alpha": alpha}, block + [int(t) for t in alpha]
    sp, af = ramp_spec(wins); add("exp4_reentry_ramp", "ramp5_ALL", sp, af)
    sp, af = ramp_spec([d2]); add("exp4_reentry_ramp", "ramp5_D2", sp, af)

    # exp5: reset
    ends = [b for a, b in wins if b < n - 1]
    starts = [a - 1 for a, b in wins if a - 1 >= 1]
    add("exp5_reset", "reset_end_block_ALL", {"block": allw, "reset": ends}, allw + ends)
    add("exp5_reset", "reset_end_only_ALL", {"reset": ends}, ends)
    add("exp5_reset", "reset_start_only_ALL", {"reset": starts}, starts)

    # exp6: boundary shift
    for sh in (-3, 3):
        fr = sorted(set(t for a, b in wins for t in range(max(1, a + sh), min(n - 1, b + sh) + 1)))
        add("exp6_boundary_shift", f"shift_{'m' if sh < 0 else 'p'}{abs(sh)}_ALL", {"block": fr}, fr)

    manifest = {}
    for exp, rr in runs.items():
        md = os.path.join(args.scene_dir, exp, "masks")
        os.makedirs(md, exist_ok=True)
        manifest[exp] = []
        for run, (spec, affected) in rr.items():
            json.dump({S: spec}, open(os.path.join(md, f"{run}.control.json"), "w"))
            json.dump({S: affected}, open(os.path.join(md, f"{run}.affected.json"), "w"))
            manifest[exp].append(run)
            keys = ",".join(f"{k}={len(v) if hasattr(v, '__len__') and not isinstance(v, str) else v}"
                            for k, v in spec.items())
            print(f"{exp:20s} {run:22s} affected={len(affected):3d} frames  spec: {keys}")
    json.dump(manifest, open(os.path.join(args.scene_dir, "explore_manifest.json"), "w"), indent=2)
    print(f"windows: {dict(zip(names, wins))}; {sum(len(v) for v in manifest.values())} runs")


if __name__ == "__main__":
    main()
