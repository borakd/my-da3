#!/usr/bin/env python
"""Per-frame attribution for the confidence-gated memory-write experiment.

For each gated arm (diag_cg_rel_a2, diag_cg_th_p20) vs the clean reference
(diag_cg_log430, byte-identical to ungated inference):

Q1  Classify every frame by gate context (accepted/accepted, re-entry,
    during-skip, forced write after MAX_SKIP=8, first-skip-of-run) and
    compute mean per-frame RPE deltas (gated - clean) per class.
Q2  Scene-level rpe_rot delta vs skip-run count / skipped-frame count /
    skip fraction (midrank Spearman).
Q3  Scene ATE delta (ALL/MEAN rows) vs clean-run mean conf_mean signal;
    tercile breakdown.

Caveat carried into the JSON: sim3 alignment is fit per-scene on the whole
trajectory, so per-frame ATE deltas between gated and clean runs are
confounded by the differing alignments; per-frame RPE deltas are much less
alignment-sensitive (relative pose between consecutive frames) but are not
perfectly alignment-free either (rpe_trans inherits the sim3 scale).

Frame index alignment (verified empirically): state_gate.json frames are in
view order; the eval CSV has one camera per scene with per-frame rows
local_timestep = 0..N-1 matching state_gate index, plus '0,MEAN' and
'ALL,MEAN' summary rows. RPE at timestep t is the relative-pose error of the
transition (t-1 -> t); timestep 0 has rpe = nan. A frame's RPE row is
attributed to the gate class of frame t (computed from g[t], g[t-1]).
"""

import csv
import json
import math
import os

import numpy as np

OUT_ROOT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
SCENE_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "noise_oracle_subset_430.txt")
CLEAN_LABEL = "diag_cg_log430"
GATED_LABELS = ["diag_cg_rel_a2", "diag_cg_th_p20"]
MAX_SKIP = 8
JSON_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "evidence", "cg_frame_attrib.json")


# ----- midrank Spearman (same as eval_pipeline/cg_calibrate.py) -----------

def midrank(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra, rb = midrank(a), midrank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


# ----- IO ------------------------------------------------------------------

def read_eval_csv(path):
    """Return (per_frame dict of np arrays keyed by metric, all_mean row dict).

    Per-frame rows: camera_id != 'ALL', local_timestep numeric. Verified
    single-camera; multi-camera scenes are rejected (return None)."""
    frames = {}
    all_mean = None
    cams = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            cid = row.get("camera_id")
            ts = row.get("local_timestep")
            if cid == "ALL" and ts == "MEAN":
                all_mean = {k: float(row[k]) for k in
                            ("absrel", "a1", "ate", "rpe_trans", "rpe_rot")}
                continue
            if ts == "MEAN":
                continue
            cams.add(cid)
            t = int(ts)
            frames[t] = {k: float(row[k]) for k in
                         ("absrel", "a1", "ate", "rpe_trans", "rpe_rot")}
    if len(cams) != 1:
        return None, None
    n = max(frames) + 1
    if set(frames) != set(range(n)):
        return None, None
    out = {k: np.array([frames[t][k] for t in range(n)]) for k in
           ("absrel", "a1", "ate", "rpe_trans", "rpe_rot")}
    return out, all_mean


def read_gate(path):
    with open(path) as f:
        d = json.load(f)
    keys = d["keys"]
    fr = np.array(d["frames"], dtype=float)
    return keys, fr, d.get("config", {})


# ----- classification ------------------------------------------------------

CLASSES = ["accepted_after_accepted", "reentry_natural", "forced_write",
           "skip_first", "skip_during"]


def classify(g):
    """g: array of gate values, view order. Returns list of class names,
    one per frame; frame 0 gets 'warmup' (always accepted, no predecessor,
    rpe undefined anyway)."""
    acc = g >= 0.5
    cls = ["warmup"]
    run = 0  # consecutive skips immediately before current frame
    for t in range(1, len(g)):
        prev_acc = acc[t - 1]
        if acc[t]:
            if prev_acc:
                cls.append("accepted_after_accepted")
            else:
                cls.append("forced_write" if run >= MAX_SKIP
                           else "reentry_natural")
        else:
            cls.append("skip_first" if prev_acc else "skip_during")
        run = 0 if acc[t] else run + 1
    return cls


# ----- main ----------------------------------------------------------------

def main():
    with open(SCENE_LIST) as f:
        scenes = [ln.strip() for ln in f if ln.strip()]

    # clean reference: per-frame metrics, ALL/MEAN row, mean conf_mean signal
    clean = {}
    for s in scenes:
        cp = os.path.join(OUT_ROOT, CLEAN_LABEL, "eval", s,
                          "eval_depth_pose_metrics.csv")
        gp = os.path.join(OUT_ROOT, CLEAN_LABEL, "preds", s,
                          "state_gate.json")
        if not (os.path.isfile(cp) and os.path.isfile(gp)):
            continue
        pf, am = read_eval_csv(cp)
        if pf is None:
            continue
        keys, fr, _ = read_gate(gp)
        i_conf = keys.index("conf_mean")
        clean[s] = {"pf": pf, "all_mean": am,
                    "conf_mean": float(np.nanmean(fr[:, i_conf])),
                    "n": len(pf["ate"])}

    results = {"caveats": {
        "ate_per_frame": ("sim3 alignment is fit per-scene on the whole "
                          "trajectory; gated and clean runs get different "
                          "alignments, so per-frame ATE deltas are "
                          "confounded and are NOT analyzed per-frame."),
        "rpe": ("per-frame RPE deltas are much less alignment-sensitive "
                "(consecutive-frame relative pose) but rpe_trans still "
                "inherits the per-scene sim3 scale; treat small deltas "
                "with care."),
        "forced_write_inference": ("forced writes are inferred as accepted "
                                   "frames following exactly >=8 consecutive "
                                   "skips (MAX_SKIP=8 in both arms); a "
                                   "natural re-entry coinciding at 8 skips "
                                   "is indistinguishable and counted as "
                                   "forced.")},
        "labels": {}}

    for label in GATED_LABELS:
        per_class = {c: {"rpe_rot": [], "rpe_trans": []} for c in CLASSES}
        clean_class_rpe = {c: {"rpe_rot": [], "rpe_trans": []}
                           for c in CLASSES}
        scene_rows = []
        n_scenes = 0
        skipped_scenes = []
        config = None
        for s in scenes:
            if s not in clean:
                skipped_scenes.append(s)
                continue
            ep = os.path.join(OUT_ROOT, label, "eval", s,
                              "eval_depth_pose_metrics.csv")
            gp = os.path.join(OUT_ROOT, label, "preds", s,
                              "state_gate.json")
            if not (os.path.isfile(ep) and os.path.isfile(gp)):
                skipped_scenes.append(s)
                continue
            pf, am = read_eval_csv(ep)
            if pf is None:
                skipped_scenes.append(s)
                continue
            keys, fr, cfg = read_gate(gp)
            config = config or cfg
            g = fr[:, keys.index("g")]
            n = len(g)
            if n != len(pf["ate"]) or n != clean[s]["n"]:
                skipped_scenes.append(s)
                continue
            n_scenes += 1
            cls = classify(g)

            d_rot = pf["rpe_rot"] - clean[s]["pf"]["rpe_rot"]
            d_trn = pf["rpe_trans"] - clean[s]["pf"]["rpe_trans"]
            for t in range(1, n):
                c = cls[t]
                if math.isfinite(d_rot[t]):
                    per_class[c]["rpe_rot"].append(d_rot[t])
                    clean_class_rpe[c]["rpe_rot"].append(
                        clean[s]["pf"]["rpe_rot"][t])
                if math.isfinite(d_trn[t]):
                    per_class[c]["rpe_trans"].append(d_trn[t])
                    clean_class_rpe[c]["rpe_trans"].append(
                        clean[s]["pf"]["rpe_trans"][t])

            acc = g >= 0.5
            n_skipped = int((~acc).sum())
            # skip runs = number of accepted->skipped transitions
            n_trans = sum(1 for t in range(1, n)
                          if (not acc[t]) and acc[t - 1])
            if not acc[0]:
                n_trans += 1  # cannot happen with warmup=1, but be safe
            scene_rows.append({
                "scene": s,
                "n_frames": n,
                "n_skipped": n_skipped,
                "n_transitions": n_trans,
                "skip_frac": n_skipped / n,
                "d_rpe_rot": am["rpe_rot"]
                             - clean[s]["all_mean"]["rpe_rot"],
                "d_rpe_trans": am["rpe_trans"]
                               - clean[s]["all_mean"]["rpe_trans"],
                "d_ate": am["ate"] - clean[s]["all_mean"]["ate"],
                "clean_conf_mean": clean[s]["conf_mean"],
            })

        # ---- Q1: per-class deltas ----
        q1 = {}
        for c in CLASSES:
            rr = np.array(per_class[c]["rpe_rot"])
            rt = np.array(per_class[c]["rpe_trans"])
            q1[c] = {
                "n_frames_rot": int(len(rr)),
                "mean_d_rpe_rot": float(np.mean(rr)) if len(rr) else None,
                "median_d_rpe_rot": float(np.median(rr)) if len(rr) else None,
                "mean_d_rpe_trans": float(np.mean(rt)) if len(rt) else None,
                "median_d_rpe_trans": (float(np.median(rt))
                                       if len(rt) else None),
                "clean_mean_rpe_rot": (float(np.mean(
                    clean_class_rpe[c]["rpe_rot"])) if len(rr) else None),
            }

        # ---- Q2: scene-level correlations ----
        d_rot = np.array([r["d_rpe_rot"] for r in scene_rows])
        n_tr = np.array([r["n_transitions"] for r in scene_rows], float)
        n_sk = np.array([r["n_skipped"] for r in scene_rows], float)
        s_fr = np.array([r["skip_frac"] for r in scene_rows])
        def partial_spearman(x, y, z):
            """Rank-based partial correlation of x,y controlling z."""
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            rx, ry, rz = (midrank(x[ok]), midrank(y[ok]), midrank(z[ok]))
            def resid(a, b):
                b = b - b.mean()
                a = a - a.mean()
                return a - b * (a @ b) / (b @ b)
            ex, ey = resid(rx, rz), resid(ry, rz)
            den = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
            return float((ex * ey).sum() / den) if den > 0 else float("nan")

        q2 = {
            "spearman_d_rpe_rot_vs_n_transitions": spearman(d_rot, n_tr),
            "spearman_d_rpe_rot_vs_n_skipped": spearman(d_rot, n_sk),
            "spearman_d_rpe_rot_vs_skip_frac": spearman(d_rot, s_fr),
            # partial-ish check: transitions and skipped are correlated;
            # report their mutual Spearman for context
            "spearman_n_transitions_vs_n_skipped": spearman(n_tr, n_sk),
            # transitions and skipped are near-degenerate (MAX_SKIP=8 caps
            # run length), so raw Spearmans cannot separate them; partial
            # rank correlations attempt to:
            "partial_spearman_d_rot_vs_n_trans_given_n_skipped":
                partial_spearman(d_rot, n_tr, n_sk),
            "partial_spearman_d_rot_vs_n_skipped_given_n_trans":
                partial_spearman(d_rot, n_sk, n_tr),
            # mean skip-run length as an orthogonal-ish axis
            "spearman_d_rot_vs_mean_run_len": spearman(
                d_rot, np.where(n_tr > 0, n_sk / np.maximum(n_tr, 1),
                                np.nan)),
            "mean_n_transitions": float(np.mean(n_tr)),
            "mean_n_skipped": float(np.mean(n_sk)),
            "mean_skip_frac": float(np.mean(s_fr)),
            "mean_scene_d_rpe_rot": float(np.mean(d_rot)),
        }

        # ---- Q3: ATE delta vs clean conf_mean terciles ----
        d_ate = np.array([r["d_ate"] for r in scene_rows])
        conf = np.array([r["clean_conf_mean"] for r in scene_rows])
        ok = np.isfinite(d_ate) & np.isfinite(conf)
        da, cf = d_ate[ok], conf[ok]
        qs = np.quantile(cf, [1 / 3, 2 / 3])
        terc = np.digitize(cf, qs)  # 0=low,1=mid,2=high conf
        q3 = {"spearman_d_ate_vs_clean_conf_mean": spearman(da, cf),
              "tercile_edges_clean_conf_mean": [float(x) for x in qs],
              "terciles": {}}
        for i, name in enumerate(["low_conf", "mid_conf", "high_conf"]):
            m = terc == i
            q3["terciles"][name] = {
                "n_scenes": int(m.sum()),
                "mean_d_ate": float(np.mean(da[m])),
                "median_d_ate": float(np.median(da[m])),
                "frac_scenes_improved": float(np.mean(da[m] < 0)),
                "mean_clean_conf_mean": float(np.mean(cf[m])),
            }
        q3["overall_mean_d_ate"] = float(np.mean(da))

        results["labels"][label] = {
            "config": config,
            "n_scenes": n_scenes,
            "n_scenes_missing_or_mismatched": len(skipped_scenes),
            "q1_per_class_rpe_deltas": q1,
            "q2_scene_correlations": q2,
            "q3_ate_vs_clean_conf": q3,
            "per_scene": scene_rows,
        }
        print(f"[{label}] scenes={n_scenes} "
              f"(skipped {len(skipped_scenes)})")
        for c in CLASSES:
            v = q1[c]
            print(f"  {c:26s} n={v['n_frames_rot']:6d} "
                  f"d_rot={v['mean_d_rpe_rot']!s:>12} "
                  f"d_trans={v['mean_d_rpe_trans']!s:>12}")
        print(f"  Q2 rho(d_rot, n_trans)={q2['spearman_d_rpe_rot_vs_n_transitions']:+.4f} "
              f"rho(d_rot, n_skip)={q2['spearman_d_rpe_rot_vs_n_skipped']:+.4f} "
              f"rho(d_rot, skip_frac)={q2['spearman_d_rpe_rot_vs_skip_frac']:+.4f}")
        for name, v in q3["terciles"].items():
            print(f"  Q3 {name}: n={v['n_scenes']} mean_dATE={v['mean_d_ate']:+.5f} "
                  f"improved={v['frac_scenes_improved']:.2f}")

    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    with open(JSON_OUT, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", JSON_OUT)


if __name__ == "__main__":
    main()
