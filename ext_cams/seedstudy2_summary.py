#!/usr/bin/env python
"""Seed-study-2 scorer: seeder mask proxies + rig_track2 (--diag) records -> <root>/summary.json for one seeding METHOD.

DIAGNOSTIC ONLY: kinematic GT is used here (and inside rig_track2 --diag) for SCORING, never in a method path.

Inputs per episode EP under <root>: <root>/mask_accuracy/EP.json (seeder proxies from seedstudy_summary.proxies_episode;
recomputed here when missing), <root>/EP/<rig_subdir>/record.json, diag.json, per_frame.csv (rig_track2 --diag).
Aggregates give BOTH readings: (a) median over scenes of the per-scene medians and (b) pooled per-frame medians over all
anchored frames of all ok scenes (from per_frame.csv: pos_err_cm, rot_err_deg, centroid_err_cm).

    OMP_NUM_THREADS=1 python seedstudy2_summary.py --method verified_motion [--root ..] [--rig_subdir rig2] [--ref_method gtbox_sam3]
                                                   [--before_root ..]   # round 3: wrong_body_anchor_frac before (that root) / after (--root)

Round-3 addition: wrong_body_anchor_frac = fraction of anchored frames (anchors.npz) whose GT-free centroid lies > 0.20 m from
the kinematic GT lens position (DIAGNOSTIC, GT), reported per scene and pooled for --before_root (round-2 tracker, no gate)
and --root (round-3 tracker with the --verified_points gate); plus the tracker's own gate counters from record.json.
"""
import argparse, csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seedstudy_summary import proxies_episode, SEEDSTUDY, SMOKE13, STORE_ROOT, fmed, fq  # noqa: E402

SEEDSTUDY2 = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy2"
REC_KEYS = ["status", "failure_reason", "n_frames", "n_frames_used", "n_frames_store", "n_frames_mp4", "n_frames_cut3r", "mask_frames",
            "mask_frames_both", "window", "window_len", "window_gap_frames", "t_seed", "size_scale", "n_tracks", "seed_events",
            "track_len_median", "pairs_confirmed_total", "pairing", "mean_confirmation_delay_frames", "reproj_px", "live_pairs_per_frame",
            "n_anchored", "coverage_frac", "anchored_frac_mask_both", "anchored_frac_window", "n_segments", "n_linked", "n_frames_linked",
            "segment_end_events", "link_attempts", "link_fail_reasons", "model_hygiene", "unsolved_reasons", "segments", "inliers",
            "rigid_resid_mm", "max_jump_cm", "max_jump_deg", "max_anchor_gap_frames", "model_extent_cm", "diag_note",
            "pos_err_median_cm", "pos_err_p90_cm", "centroid_err_median_cm", "centroid_err_p90_cm", "rot_err_median_deg", "rot_err_p90_deg",
            "rig_ate_centroid", "rig_ate_lens", "rig_rpe_t", "rig_rpe_rot", "rig_rpe_t_lens", "sim3_scale", "sim3_scale_lens",
            "cut3r_ate_same_frames", "cut3r_rpe_t_same_frames", "cut3r_rpe_rot_same_frames", "cut3r_sim3_scale_same_frames",
            "per_segment_diag", "timing_s", "elapsed_s", "elapsed_total_s", "video", "video_s", "video_bytes",
            "t_seed", "seed_gate", "wrong_body_gate", "segments_rejected_wrong_body", "frames_dropped_wrong_body", "segments_ended_wrong_body",
            "n_anchored_after_last_mask_frame", "centroid_to_vp_m", "centroid_to_gt_lens_m", "wrong_body_anchor_frac_gt", "wrong_body_anchors_gt", "verified_points", "gt_opened", "tool"]
ROW_KEYS = ["rig_ate_centroid", "rig_ate_lens", "rig_rpe_t", "rig_rpe_rot", "rig_rpe_t_lens", "sim3_scale", "sim3_scale_lens",
            "pos_err_median_cm", "pos_err_p90_cm", "rot_err_median_deg", "rot_err_p90_deg", "centroid_err_median_cm", "centroid_err_p90_cm",
            "cut3r_ate_same_frames", "cut3r_rpe_t_same_frames", "cut3r_rpe_rot_same_frames", "cut3r_sim3_scale_same_frames",
            "window", "window_len", "n_segments", "n_linked", "n_frames_linked", "anchored_frac_mask_both", "anchored_frac_window",
            "max_jump_cm", "max_jump_deg", "max_anchor_gap_frames", "pairs_confirmed_total", "unsolved_reasons", "elapsed_s", "elapsed_total_s", "video",
            "segments_rejected_wrong_body", "frames_dropped_wrong_body", "segments_ended_wrong_body", "n_anchored_after_last_mask_frame",
            "wrong_body_anchor_frac_gt", "wrong_body_anchors_gt", "centroid_to_vp_m", "t_seed"]
WB_DIST_M = 0.20


def wrong_body_frac(rig, ep, dist_m=WB_DIST_M):
    """DIAGNOSTIC (GT): over the emitted anchors (anchors.npz), the distance of the GT-free centroid to the kinematic GT lens
    position; n_wrong / frac = anchors farther than dist_m (the Franka hand is < 0.20 m from the wrist lens)."""
    p = f"{rig}/anchors.npz"
    if not os.path.isfile(p):
        return None
    an = np.load(p)
    d = []
    for t, c in zip(an["frames"], an["centroid"]):
        f = f"{STORE_ROOT}/{ep}/dense/cam/{int(t):06d}.npz"
        if os.path.isfile(f):
            d.append(float(np.linalg.norm(np.asarray(c, float) - np.load(f)["pose"][:3, 3])))
    d = np.array(d)
    if not len(d):
        return dict(n=0, n_wrong=0, frac=None, median_m=None, p90_m=None)
    return dict(n=int(len(d)), n_wrong=int((d > dist_m).sum()), frac=float((d > dist_m).mean()), median_m=fmed(d), p90_m=fq(d, 90),
                segments_wrong=sorted(set(int(g) for g, dd in zip(an["segment_id"], d) if dd > dist_m)))


def read_per_frame(path, anchored_frames):
    """rows of per_frame.csv whose frame is an emitted anchor (anchors.npz frames) -> dict of float arrays (pos/rot/centroid err).
    (segment_id >= 0 alone is NOT the anchored set: solved-but-rejected frames (reason degenerate/jump) also carry a segment id.)"""
    out = dict(pos_err_cm=[], rot_err_deg=[], centroid_err_cm=[], frames=[])
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["frame"]) not in anchored_frames:
                continue
            out["frames"].append(int(r["frame"]))
            for k in ("pos_err_cm", "rot_err_deg", "centroid_err_cm"):
                v = r.get(k, "")
                out[k].append(float(v) if v not in ("", None) else np.nan)
    return {k: np.array(v, float) for k, v in out.items()}


def product_check(rig, rec):
    """anchors.npz must agree with record.json (frame count, R/centroid shapes); diag.json must exist under --diag."""
    msgs = []
    if rec.get("status") != "ok":
        return "n/a (failed)"
    p = f"{rig}/anchors.npz"
    if not os.path.isfile(p):
        return "MISSING anchors.npz"
    an = np.load(p)
    k = int(rec.get("n_anchored") or 0)
    if len(an["frames"]) != k or an["R"].shape != (k, 3, 3) or an["centroid"].shape != (k, 3) or an["trans"].shape != (k, 3):
        msgs.append(f"anchors shape mismatch frames={len(an['frames'])} R={an['R'].shape} rec={k}")
    if not np.isfinite(an["centroid"]).all():
        msgs.append("non-finite centroid")
    if rec.get("diag") and not os.path.isfile(f"{rig}/diag.json"):
        msgs.append("MISSING diag.json")
    if rec.get("video") and not (os.path.isfile(rec["video"]) and os.path.getsize(rec["video"]) > 0):
        msgs.append("MISSING/empty video")
    return "ok" if not msgs else "; ".join(msgs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True)
    ap.add_argument("--root", default=None, help="default SEEDSTUDY2/<method>")
    ap.add_argument("--ref_method", default="gtbox_sam3")
    ap.add_argument("--ref_root", default=None, help="default SEEDSTUDY/<ref_method> (phase-1 reference masks)")
    ap.add_argument("--episodes", default=SMOKE13)
    ap.add_argument("--rig_subdir", default="rig2")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--out", default=None, help="default <root>/summary.json")
    ap.add_argument("--before_root", default=None, help="round-2 root (same rig_subdir) for the wrong_body_anchor_frac BEFORE the gate; default SEEDSTUDY2/<method>")
    a = ap.parse_args()
    root = a.root or f"{SEEDSTUDY2}/{a.method}"
    before_root = a.before_root or f"{SEEDSTUDY2}/{a.method}"
    ref_root = a.ref_root or f"{SEEDSTUDY}/{a.ref_method}"
    cache_dir = f"{root}/mask_accuracy"
    os.makedirs(cache_dir, exist_ok=True)
    eps = [l.strip() for l in open(a.episodes) if l.strip()]
    rows, pooled = [], dict(pos_err_cm=[], rot_err_deg=[], centroid_err_cm=[])
    for ep in eps:
        cache = f"{cache_dir}/{ep}.json"
        if os.path.isfile(cache) and not a.recompute:
            px, src = json.load(open(cache)), "cached"
        else:
            t0 = time.time()
            px = proxies_episode(ep, a.method, root, a.ref_method, ref_root)
            json.dump(px, open(cache, "w"), indent=1)
            src = f"recomputed {time.time() - t0:.1f}s"
        rig = f"{root}/{ep}/{a.rig_subdir}"
        rp = f"{rig}/record.json"
        rec = json.load(open(rp)) if os.path.isfile(rp) else None
        row = dict(episode=ep, n_frames=px.get("n_frames_store"), mask_frames_both=px.get("mask_frames_both", 0),
                   lens_to_mask_px_median=px.get("lens_to_mask_px_median"), lens_to_mask_px_p90=px.get("lens_to_mask_px_p90"),
                   lens_to_mask_cm_median=px.get("lens_to_mask_cm_median"), inview_agreement=px.get("inview_agreement"),
                   iou_vs_ref_median=px.get("iou_vs_ref_median", -1), mask_products_ok=px.get("products_ok"),
                   per_cam_proxies={c: {k: px["cams"][c].get(k) for k in ("serial", "frames_present", "lens_to_mask_px_median", "inview_agreement", "recall_in_view", "far_frames_gt100px")}
                                    for c in px.get("cams", {})}, proxies_source=src, rig_dir=rig)
        if rec is None:
            row.update(failed=True, failure_reason="no record.json (tracker not run or crashed)", n_anchored=0, coverage_frac=0.0,
                       product_check="MISSING record.json", n_frames_pooled=0)
            rows.append(row); print(f"{ep[:40]:40s} NO RECORD"); continue
        failed = rec.get("status") != "ok"
        row.update(failed=failed, failure_reason=rec.get("failure_reason"), n_anchored=int(rec.get("n_anchored") or 0),
                   coverage_frac=float(rec.get("coverage_frac") or 0.0), n_frames=rec.get("n_frames", row["n_frames"]),
                   mask_frames_both=rec.get("mask_frames_both", row["mask_frames_both"]), segments=rec.get("n_segments"))
        for k in ROW_KEYS:
            if rec.get(k) is not None:
                row[k] = rec[k]
        row["product_check"] = product_check(rig, rec)
        row["wrong_body_after"] = wrong_body_frac(rig, ep) if not failed else None
        row["wrong_body_before"] = wrong_body_frac(f"{before_root}/{ep}/{a.rig_subdir}", ep) if os.path.isdir(f"{before_root}/{ep}/{a.rig_subdir}") else None
        pf = read_per_frame(f"{rig}/per_frame.csv", set(np.load(f"{rig}/anchors.npz")["frames"].tolist())) if (not failed and os.path.isfile(f"{rig}/anchors.npz")) else None
        if pf is not None and len(pf["frames"]):
            row["n_frames_pooled"] = int(len(pf["frames"]))
            row["per_frame_check"] = "ok" if len(pf["frames"]) == row["n_anchored"] else f"per_frame anchored rows {len(pf['frames'])} != n_anchored {row['n_anchored']}"
            for k in pooled:
                v = pf[k][np.isfinite(pf[k])]
                pooled[k].extend(v.tolist())
                row[f"{k}_pooled_median_check"] = fmed(v)  # should equal the record's median
        else:
            row["n_frames_pooled"] = 0
        row["record"] = {k: rec[k] for k in REC_KEYS if k in rec}
        rows.append(row)
        wbb, wba = row.get("wrong_body_before") or {}, row.get("wrong_body_after") or {}
        print(f"{ep[:40]:40s} {rec['status']:6s} anch={row['n_anchored']:4d} cov={row['coverage_frac']:.3f} pos={row.get('pos_err_median_cm')} rot={row.get('rot_err_median_deg')} "
              f"jump={row.get('max_jump_cm')} l2m={row['lens_to_mask_px_median']} agree={row['inview_agreement']} prod={row['product_check']} "
              f"wb_before={wbb.get('n_wrong')}/{wbb.get('n')} wb_after={wba.get('n_wrong')}/{wba.get('n')} rej={rec.get('segments_rejected_wrong_body')} "
              f"ended={rec.get('segments_ended_wrong_body')} dropped={rec.get('frames_dropped_wrong_body')} ({src})", flush=True)
    ok = [r for r in rows if not r["failed"]]
    scored = [r for r in ok if r.get("rig_ate_centroid") is not None]

    def vals(key, src):
        return [r[key] for r in src if r.get(key) is not None]

    def fmean(x):
        return float(np.mean(x)) if len(x) else None

    P = {k: np.array(v) for k, v in pooled.items()}
    agg = dict(n_scenes=len(rows), scenes_ok=len(ok), scenes_failed=[r["episode"] for r in rows if r["failed"]],
               failure_reasons={r["episode"]: r["failure_reason"] for r in rows if r["failed"]},
               scenes_scored=len(scored), scenes_ok_unscored=[r["episode"] for r in ok if r.get("rig_ate_centroid") is None],
               product_checks_not_ok={r["episode"]: r["product_check"] for r in rows if r.get("product_check") not in ("ok", "n/a (failed)")},
               mean_coverage=fmean([r["coverage_frac"] for r in rows]), mean_coverage_ok_only=fmean([r["coverage_frac"] for r in ok]),
               n_anchored_total=int(sum(r["n_anchored"] for r in rows)), n_frames_total=int(sum(r["n_frames"] or 0 for r in rows)),
               mean_anchored_frac_mask_both=fmean(vals("anchored_frac_mask_both", ok)),
               mean_rig_ate_centroid=fmean(vals("rig_ate_centroid", ok)), mean_rig_ate_lens=fmean(vals("rig_ate_lens", ok)),
               mean_rig_rpe_t=fmean(vals("rig_rpe_t", ok)), mean_rig_rpe_rot=fmean(vals("rig_rpe_rot", ok)),
               mean_cut3r_ate_same_frames=fmean(vals("cut3r_ate_same_frames", ok)),
               mean_cut3r_rpe_t_same_frames=fmean(vals("cut3r_rpe_t_same_frames", ok)), mean_cut3r_rpe_rot_same_frames=fmean(vals("cut3r_rpe_rot_same_frames", ok)),
               # (a) medians of the per-scene medians (scored scenes)
               median_pos_err_cm=fmed(vals("pos_err_median_cm", scored)), median_rot_err_deg=fmed(vals("rot_err_median_deg", scored)),
               median_centroid_err_cm=fmed(vals("centroid_err_median_cm", scored)),
               mean_pos_err_median_cm=fmean(vals("pos_err_median_cm", scored)), mean_rot_err_median_deg=fmean(vals("rot_err_median_deg", scored)),
               # (b) pooled per-frame medians over all anchored frames of all ok scenes
               pooled_n_frames=int(len(P["pos_err_cm"])), pooled_pos_err_median_cm=fmed(P["pos_err_cm"]), pooled_pos_err_p90_cm=fq(P["pos_err_cm"], 90),
               pooled_rot_err_median_deg=fmed(P["rot_err_deg"]), pooled_rot_err_p90_deg=fq(P["rot_err_deg"], 90),
               pooled_centroid_err_median_cm=fmed(P["centroid_err_cm"]), pooled_centroid_err_p90_cm=fq(P["centroid_err_cm"], 90),
               worst_max_jump_cm=max(vals("max_jump_cm", ok)) if vals("max_jump_cm", ok) else None,
               worst_max_jump_scene=(max(ok, key=lambda r: r.get("max_jump_cm") or -1)["episode"] if vals("max_jump_cm", ok) else None),
               scenes_max_jump_gt10cm=[r["episode"] for r in ok if (r.get("max_jump_cm") or 0) > 10],
               n_segments_total=int(sum(r.get("segments") or 0 for r in ok)), n_linked_total=int(sum(r.get("n_linked") or 0 for r in ok)),
               median_lens_to_mask_px=fmed([r["lens_to_mask_px_median"] for r in rows if r["lens_to_mask_px_median"] is not None]),
               mean_inview_agreement=fmean([r["inview_agreement"] for r in rows if r["inview_agreement"] is not None]),
               median_iou_vs_ref=fmed([r["iou_vs_ref_median"] for r in rows if r["iou_vs_ref_median"] not in (None, -1)]),
               scenes_lens_to_mask_le10px=int(sum(1 for r in rows if r["lens_to_mask_px_median"] is not None and r["lens_to_mask_px_median"] <= 10)),
               scenes_lens_to_mask_gt100px=int(sum(1 for r in rows if r["lens_to_mask_px_median"] is not None and r["lens_to_mask_px_median"] > 100)),
               mask_products_ok=int(sum(bool(r["mask_products_ok"]) for r in rows)),
               score_phase1_rule=fmean([r["coverage_frac"] * max(0.0, 1.0 - (r["rig_ate_centroid"] / r["cut3r_ate_same_frames"])) if (not r["failed"] and r.get("rig_ate_centroid") is not None and r.get("cut3r_ate_same_frames")) else 0.0 for r in rows]),
               gates_phase1=None)
    # round 3: wrong-body anchors (DIAGNOSTIC, GT lens) before (before_root) and after (root) the verified-point gate
    for tag in ("before", "after"):
        wbs = [r[f"wrong_body_{tag}"] for r in rows if r.get(f"wrong_body_{tag}")]
        n_all, n_wrong = int(sum(w["n"] for w in wbs)), int(sum(w["n_wrong"] for w in wbs))
        agg[f"wrong_body_anchor_frac_{tag}"] = (n_wrong / n_all) if n_all else None
        agg[f"wrong_body_anchors_{tag}"] = n_wrong
        agg[f"wrong_body_anchored_total_{tag}"] = n_all
        agg[f"wrong_body_anchor_frac_{tag}_mean_of_scenes"] = fmean([w["frac"] for w in wbs if w["frac"] is not None])
        agg[f"wrong_body_scenes_{tag}"] = {r["episode"]: r[f"wrong_body_{tag}"]["n_wrong"] for r in rows if r.get(f"wrong_body_{tag}") and r[f"wrong_body_{tag}"]["n_wrong"]}
    agg["wrong_body_gate_totals"] = dict(segments_rejected=int(sum(r.get("segments_rejected_wrong_body") or 0 for r in rows)),
                                         segments_ended=int(sum(r.get("segments_ended_wrong_body") or 0 for r in rows)),
                                         frames_dropped=int(sum(r.get("frames_dropped_wrong_body") or 0 for r in rows)),
                                         anchors_after_last_mask_frame=int(sum(r.get("n_anchored_after_last_mask_frame") or 0 for r in rows)))
    agg["before_root"] = before_root
    g = dict(scenes_ok_ge11=agg["scenes_ok"] >= 11, median_pos_le_2cm=(agg["median_pos_err_cm"] is not None and agg["median_pos_err_cm"] <= 2.0),
             median_rot_le_8deg=(agg["median_rot_err_deg"] is not None and agg["median_rot_err_deg"] <= 8.0),
             max_jump_le_10cm=(agg["worst_max_jump_cm"] is not None and agg["worst_max_jump_cm"] <= 10.0))
    g["passed"] = int(sum(g.values()))
    agg["gates_phase1"] = g
    summary = dict(method=a.method, tracker="rig_track2 v2 --diag", ref_method=a.ref_method, root=root, rig_subdir=a.rig_subdir, episodes_file=a.episodes,
                   generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                   definitions=dict(lens_to_mask_px="median over frames (both cams pooled) with mask non-empty AND GT lens in view of the px distance from the projected GT lens to the nearest mask pixel (0 = lens inside mask)",
                                    lens_to_mask_cm="same distance scaled by the lens depth: px * z / fx * 100",
                                    inview_agreement="fraction of frames (both cams pooled) where (mask non-empty) == (GT lens in view, z>0 and inside the 1280x720 image)",
                                    iou_vs_ref_median="median per-frame IoU vs the phase-1 reference (gtbox_sam3) masks over frames where either is non-empty; -1 when absent",
                                    coverage_frac="tracker: anchored frames / store frames", anchored_frac_mask_both="anchored frames / frames with both masks non-empty (can exceed 1: the rigid model keeps anchoring after a mask drops)",
                                    pos_err_cm="rig_track2 --diag lens transfer: GT lens pose at each segment's reference frame carried by the segment's rigid motion, vs kinematic GT (cm); rot_err_deg likewise",
                                    centroid_err_cm="GT-free body-fixed centroid vs the GT lens position after a per-scene constant offset (see rig_track2 --diag)",
                                    rig_ate_centroid="Sim3-aligned ATE of the GT-free centroid trajectory on the anchored frames (m); rig_ate_lens = same for the lens-transfer poses",
                                    cut3r_ate_same_frames="finetuned CUT3R (augfull_lr1e5) ATE on the anchored frames only, Sim3-aligned",
                                    median_pos_err_cm="(a) median over scored scenes of the per-scene median over anchored frames; pooled_* = (b) pooled per-frame medians over all anchored frames of all ok scenes",
                                    score_phase1_rule="mean over the 13 scenes of coverage_frac x max(0, 1 - rig_ate_centroid / cut3r_ate_same_frames); failed / unscored scenes contribute 0",
                                    wrong_body_anchor_frac="DIAGNOSTIC (GT): fraction of emitted anchors whose GT-free centroid is > 0.20 m from the kinematic GT lens; _before = --before_root (round-2 tracker, no gate), _after = --root (round-3 tracker, --verified_points gate); pooled over scenes, plus mean of per-scene fractions"),
                   aggregate=agg, per_scene=rows)
    out = a.out or f"{root}/summary.json"
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps(agg, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
