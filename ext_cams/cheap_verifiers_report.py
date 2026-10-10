#!/usr/bin/env python
"""Idea-4 attribution report (DIAGNOSTIC: kinematic GT is read here for scoring only, never in a method path).

For every (variant root, method suffix, rig subdir) given, per scene: the seeder's picks (frames accepted, median distance of the
verified 3D point to the GT lens, fraction > 0.20 m, test counters, ms/frame) and the tracker record when present (status,
anchored frames, coverage, per-scene medians, rig / CUT3R ATE, max jump, wrong-body anchors = emitted anchors whose GT-free
centroid is > 0.20 m from the GT lens). Prints markdown tables and writes <out>.json.

    OMP_NUM_THREADS=1 python cheap_verifiers_report.py --variant round3=/…/seedstudy2/verified_motion_v2:verified_motion:rig2 \
        --variant cheap_verifiers=/…/ideas/cheap_verifiers:cheap_verifiers:rig2 [--variant …] [--episodes FILE] --out /…/attribution
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seedstudy2_summary import wrong_body_frac  # noqa: E402

STORE = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
SMOKE13 = "/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt"


def short(ep):
    lab, uid, ts = ep.split("+")
    return f"{lab[:4]}+{uid[:4]}+{ts[5:16]}"


def seeder_row(root, ep):
    p = f"{root}/{ep}/verified_points.npz"
    if not os.path.isfile(p):
        return None
    vp = np.load(p)
    T = len(vp["frames"])
    gt = np.array([np.load(f"{STORE}/{ep}/dense/cam/{t:06d}.npz")["pose"][:3, 3] for t in range(T)])
    acc = vp["accepted"]
    d = np.linalg.norm(vp["xyz"] - gt, axis=1)
    row = dict(T=T, accepted=int(acc.sum()), frac_accepted=float(acc.mean()) if T else None,
               lens_dist_median_m=float(np.median(d[acc])) if acc.any() else None, lens_dist_p90_m=float(np.percentile(d[acc], 90)) if acc.any() else None,
               n_far=int((d[acc] > 0.2).sum()), frac_far=float((d[acc] > 0.2).mean()) if acc.any() else None)
    ip = f"{root}/{ep}/seed_info.json"
    if os.path.isfile(ip):
        info = json.load(open(ip))
        sm, tm = info.get("summary", {}), info.get("timing", {})
        row.update(ms_per_frame=tm.get("ms_per_frame_seeding"), cpu_s=tm.get("cpu_seconds_process"), test_counts=sm.get("test_counts"),
                   rejected_best_fails=sm.get("rejected_best_fails"), frames_round3_base=sm.get("frames_with_any_round3_verified"),
                   tests_enabled=info.get("tests_enabled"), postfilter_removed=(info.get("postfilter") or {}).get("n_removed"))
    return row


def tracker_row(rig, ep):
    rp = f"{rig}/record.json"
    if not os.path.isfile(rp):
        return None
    rec = json.load(open(rp))
    row = {k: rec.get(k) for k in ("status", "failure_reason", "n_anchored", "coverage_frac", "n_segments", "pos_err_median_cm", "rot_err_median_deg",
                                   "centroid_err_median_cm", "rig_ate_centroid", "cut3r_ate_same_frames", "max_jump_cm", "segments_rejected_wrong_body",
                                   "segments_ended_wrong_body", "frames_dropped_wrong_body", "t_seed", "elapsed_s")}
    wb = rec.get("wrong_body_gate") or {}
    row["starts_unchecked"] = wb.get("segment_starts_unchecked_no_point")
    row["starts_refused"] = wb.get("segment_starts_refused_no_point")
    row["links_refused"] = wb.get("links_refused_no_point")
    row["wrong_body"] = wrong_body_frac(rig, ep) if rec.get("status") == "ok" else None
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", required=True, help="name=root:method_suffix:rig_subdir (rig_subdir may be empty)")
    ap.add_argument("--episodes", default=SMOKE13)
    ap.add_argument("--out", required=True, help="output stem; writes <out>.json and <out>.md")
    a = ap.parse_args()
    eps = [l.strip() for l in open(a.episodes) if l.strip()]
    variants = []
    for v in a.variant:
        name, spec = v.split("=", 1)
        root, method, rig = (spec.split(":") + ["", ""])[:3]
        variants.append(dict(name=name, root=root, method=method, rig=rig))
    res = {}
    for v in variants:
        res[v["name"]] = dict(spec=v, scenes={})
        for ep in eps:
            sr = seeder_row(v["root"], ep)
            tr = tracker_row(f"{v['root']}/{ep}/{v['rig']}", ep) if v["rig"] else None
            if sr is None and tr is None:
                continue
            res[v["name"]]["scenes"][ep] = dict(seeder=sr, tracker=tr)
        # pooled
        S = [s["seeder"] for s in res[v["name"]]["scenes"].values() if s["seeder"]]
        Tr = [s["tracker"] for s in res[v["name"]]["scenes"].values() if s["tracker"]]
        ok = [t for t in Tr if t["status"] == "ok"]
        wbs = [t["wrong_body"] for t in ok if t.get("wrong_body")]
        pooled = dict(n_scenes_seeded=len(S), picks=int(sum(s["accepted"] for s in S)), picks_far=int(sum(s["n_far"] for s in S)),
                      picks_far_frac=(sum(s["n_far"] for s in S) / max(1, sum(s["accepted"] for s in S))) if S else None,
                      lens_dist_median_of_scenes=float(np.median([s["lens_dist_median_m"] for s in S if s["lens_dist_median_m"] is not None])) if S else None,
                      ms_per_frame_mean=float(np.mean([s["ms_per_frame"] for s in S if s.get("ms_per_frame")])) if S else None,
                      n_scenes_tracked=len(Tr), scenes_ok=len(ok), anchored=int(sum(t["n_anchored"] or 0 for t in ok)),
                      mean_coverage=float(np.mean([t["coverage_frac"] or 0 for t in Tr])) if Tr else None,
                      wrong_body_anchors=int(sum(w["n_wrong"] for w in wbs)), wrong_body_anchored=int(sum(w["n"] for w in wbs)),
                      wrong_body_frac=(sum(w["n_wrong"] for w in wbs) / max(1, sum(w["n"] for w in wbs))) if wbs else None,
                      median_pos_err_cm=float(np.median([t["pos_err_median_cm"] for t in ok if t.get("pos_err_median_cm") is not None])) if ok else None,
                      median_rot_err_deg=float(np.median([t["rot_err_median_deg"] for t in ok if t.get("rot_err_median_deg") is not None])) if ok else None,
                      median_centroid_err_cm=float(np.median([t["centroid_err_median_cm"] for t in ok if t.get("centroid_err_median_cm") is not None])) if ok else None,
                      worst_max_jump_cm=max([t["max_jump_cm"] or 0 for t in ok]) if ok else None,
                      score=float(np.mean([(t["coverage_frac"] or 0) * max(0.0, 1 - t["rig_ate_centroid"] / t["cut3r_ate_same_frames"]) if (t["status"] == "ok" and t.get("rig_ate_centroid") is not None and t.get("cut3r_ate_same_frames")) else 0.0 for t in Tr])) if Tr else None,
                      starts_unchecked=int(sum(t.get("starts_unchecked") or 0 for t in Tr)), starts_refused=int(sum(t.get("starts_refused") or 0 for t in Tr)))
        res[v["name"]]["pooled"] = pooled
    lines = ["# Idea 4 (cheap_verifiers) attribution report (diagnostic, GT lens used for scoring only)", "",
             "## Pooled per variant", "",
             "| variant | scenes seeded | picks | picks > 0.20 m from lens | median-of-scenes lens dist (m) | ms/frame | tracked / ok | anchored | mean cov | wrong-body anchors | pos med cm | rot med deg | centroid med cm | worst jump cm | score | starts unchecked / refused |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, d=3: ("-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x)))
    for name, r in res.items():
        p = r["pooled"]
        lines.append(f"| {name} | {p['n_scenes_seeded']} | {p['picks']} | {p['picks_far']} ({f(p['picks_far_frac'], 3)}) | {f(p['lens_dist_median_of_scenes'])} | {f(p['ms_per_frame_mean'], 0)} | "
                     f"{p['n_scenes_tracked']} / {p['scenes_ok']} | {p['anchored']} | {f(p['mean_coverage'])} | {p['wrong_body_anchors']}/{p['wrong_body_anchored']} ({f(p['wrong_body_frac'], 4)}) | "
                     f"{f(p['median_pos_err_cm'], 2)} | {f(p['median_rot_err_deg'], 2)} | {f(p['median_centroid_err_cm'], 2)} | {f(p['worst_max_jump_cm'], 2)} | {f(p['score'], 4)} | {p['starts_unchecked']} / {p['starts_refused']} |")
    lines += ["", "## Per scene: seeder picks (accepted, > 0.20 m from the GT lens, median lens distance m)", ""]
    names = list(res)
    lines.append("| scene | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for ep in eps:
        cells = []
        for n in names:
            s = res[n]["scenes"].get(ep, {}).get("seeder")
            cells.append("-" if not s else f"{s['accepted']} / {s['n_far']} / {f(s['lens_dist_median_m'])}")
        lines.append(f"| {short(ep)} | " + " | ".join(cells) + " |")
    lines += ["", "## Per scene: tracker (anchored, wrong-body anchors, pos med cm, rot med deg, rig ATE centroid vs CUT3R)", ""]
    lines.append("| scene | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for ep in eps:
        cells = []
        for n in names:
            t = res[n]["scenes"].get(ep, {}).get("tracker")
            if not t:
                cells.append("-")
            elif t["status"] != "ok":
                cells.append(f"FAILED ({t.get('failure_reason', '')[:40]})")
            else:
                w = t.get("wrong_body") or {}
                cells.append(f"{t['n_anchored']} / wb {w.get('n_wrong')} / {f(t['pos_err_median_cm'], 2)} / {f(t['rot_err_median_deg'], 1)} / {f(t['rig_ate_centroid'], 4)} vs {f(t['cut3r_ate_same_frames'], 4)}")
        lines.append(f"| {short(ep)} | " + " | ".join(cells) + " |")
    lines += ["", "## Per scene: seeder test counters (primary variants)", ""]
    for n in names:
        for ep, sc in res[n]["scenes"].items():
            s = sc["seeder"]
            if s and s.get("test_counts"):
                lines.append(f"- {n} {short(ep)}: round-3 base frames {s.get('frames_round3_base')}, accepted {s['accepted']}, tests {s['test_counts']}, rejected-best fails {s.get('rejected_best_fails')}, postfilter removed {s.get('postfilter_removed')}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=1, default=str)
    open(a.out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", a.out + ".json", a.out + ".md")


if __name__ == "__main__":
    main()
