#!/usr/bin/env python
"""Pose-product scorer for IDEA 5 (seed_silhouette_model): per-scene score.json (written by --diag, GT used for SCORING only)
-> <root>/pose_summary.json with per-scene rows, medians of per-scene medians, pooled per-frame medians, the rig-style
score rule (mean over scenes of coverage x max(0, 1 - ATE_sim3 / CUT3R ATE on the same frames)), the wrong-body fraction
of the emitted lens positions (> 0.20 m from the GT lens), and the cost (ms per frame).

    OMP_NUM_THREADS=1 python seed_silhouette_model_summary.py --root .../ideas/silhouette_model [--episodes smoke13.txt]
"""
import argparse, json, os
import numpy as np

SMOKE13 = "/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt"
STORE = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"


def fmed(x):
    return float(np.median(x)) if len(x) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--episodes", default=SMOKE13); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    eps = [l.strip() for l in open(a.episodes) if l.strip()]
    rows, pooled_pos, pooled_rot, pooled_lens_d = [], [], [], []
    for ep in eps:
        d = f"{a.root}/{ep}"
        si_p, sc_p = f"{d}/seed_info.json", f"{d}/score.json"
        row = dict(episode=ep)
        if not os.path.isfile(si_p):
            row.update(failed=True, failure_reason="no seed_info.json (crash or timeout)"); rows.append(row); print(ep[:40], "NO PRODUCT"); continue
        si = json.load(open(si_p)); s = si["summary"]; tm = si["timing"]
        row.update(failed=False, n_frames=si["n_frames"], n_solved=s["n_solved"], coverage=s["coverage"], n_tracks=s["n_tracks"], init_frames=s["init_frames"],
                   n_rejected=s["n_rejected"], n_init_failed=s["n_init_failed"], sources=s["sources"], resid_px_median=s["resid_px_median"],
                   ms_per_frame_total=tm["ms_per_frame_total"], ms_per_frame_excl_init=tm["ms_per_frame_excl_init"], ms_per_track_call=tm["ms_per_track_call"],
                   ms_per_init_call=tm["ms_per_init_call"], n_init_calls=tm["n_init_calls"], anchors_used=si["inputs"]["anchors"] is not None)
        if s["n_solved"] == 0:
            row.update(failed=True, failure_reason="no frame solved (init never accepted)")
        if os.path.isfile(sc_p):
            sc = json.load(open(sc_p))
            row.update(pos_err_median_cm=sc["pos_err_median_cm"], pos_err_p90_cm=sc["pos_err_p90_cm"], rot_err_median_deg=sc["rot_err_median_deg"],
                       rot_err_p90_deg=sc["rot_err_p90_deg"], ate_metric_no_alignment=sc["ate_metric_no_alignment"], wrong_body_frac_gt=sc["wrong_body_frac_gt"],
                       ate_sim3=(sc["sim3"] or {}).get("ate"), rpe_t_sim3=(sc["sim3"] or {}).get("rpe_trans"), rpe_rot_sim3=(sc["sim3"] or {}).get("rpe_rot"),
                       sim3_scale=(sc["sim3"] or {}).get("sim3_scale"), cut3r_ate_same_frames=(sc["cut3r_same_frames"] or {}).get("ate"),
                       cut3r_rpe_rot_same_frames=(sc["cut3r_same_frames"] or {}).get("rpe_rot"), max_jump_cm=sc["max_jump_cm"], per_track=sc.get("per_track"))
            pf = [r for r in si["per_frame"] if r["status"] == "solved" and "pos_err_cm" in r]
            pooled_pos += [r["pos_err_cm"] for r in pf]; pooled_rot += [r["rot_err_deg"] for r in pf]
            # per-frame lens-to-GT distance for the wrong-body count
            an = np.load(f"{d}/anchors_silhouette.npz")
            for t, tt in zip(an["frames"], an["t"]):
                g = np.load(f"{STORE}/{ep}/dense/cam/{int(t):06d}.npz")["pose"][:3, 3]
                pooled_lens_d.append(float(np.linalg.norm(tt - g)))
        rows.append(row)
        print(f"{ep[:40]:40s} solved={row['n_solved']:4d}/{row['n_frames']} tracks={row['n_tracks']} pos={row.get('pos_err_median_cm')} rot={row.get('rot_err_median_deg')} "
              f"wb={row.get('wrong_body_frac_gt')} ate_sim3={row.get('ate_sim3')} cut3r={row.get('cut3r_ate_same_frames')} {row['ms_per_frame_total']:.0f} ms/f")
    ok = [r for r in rows if not r["failed"]]
    scored = [r for r in ok if r.get("ate_sim3") is not None]

    def vals(k, src):
        return [r[k] for r in src if r.get(k) is not None]

    def fmean(x):
        return float(np.mean(x)) if len(x) else None

    Ld = np.array(pooled_lens_d)
    agg = dict(n_scenes=len(rows), scenes_ok=len(ok), scenes_failed=[r["episode"] for r in rows if r["failed"]],
               failure_reasons={r["episode"]: r["failure_reason"] for r in rows if r["failed"]},
               n_solved_total=int(sum(r.get("n_solved", 0) for r in rows)), n_frames_total=int(sum(r.get("n_frames", 0) for r in rows)),
               mean_coverage=fmean([r.get("coverage", 0.0) for r in rows]), n_tracks_total=int(sum(r.get("n_tracks", 0) for r in ok)),
               median_pos_err_cm=fmed(vals("pos_err_median_cm", scored)), median_rot_err_deg=fmed(vals("rot_err_median_deg", scored)),
               mean_pos_err_median_cm=fmean(vals("pos_err_median_cm", scored)), mean_rot_err_median_deg=fmean(vals("rot_err_median_deg", scored)),
               pooled_n_frames=len(pooled_pos), pooled_pos_err_median_cm=fmed(pooled_pos), pooled_pos_err_p90_cm=float(np.percentile(pooled_pos, 90)) if pooled_pos else None,
               pooled_rot_err_median_deg=fmed(pooled_rot), pooled_rot_err_p90_deg=float(np.percentile(pooled_rot, 90)) if pooled_rot else None,
               pooled_frac_pos_le_2cm=float(np.mean(np.array(pooled_pos) <= 2.0)) if pooled_pos else None,
               pooled_frac_rot_le_8deg=float(np.mean(np.array(pooled_rot) <= 8.0)) if pooled_rot else None,
               wrong_body_anchor_frac=float((Ld > 0.20).mean()) if len(Ld) else None, wrong_body_anchors=int((Ld > 0.20).sum()) if len(Ld) else None,
               wrong_body_scenes={r["episode"]: r["wrong_body_frac_gt"] for r in scored if r.get("wrong_body_frac_gt")},
               mean_ate_sim3=fmean(vals("ate_sim3", scored)), mean_cut3r_ate_same_frames=fmean(vals("cut3r_ate_same_frames", scored)),
               mean_rpe_rot_sim3=fmean(vals("rpe_rot_sim3", scored)), mean_ate_metric_no_alignment=fmean(vals("ate_metric_no_alignment", scored)),
               worst_max_jump_cm=max(vals("max_jump_cm", scored)) if vals("max_jump_cm", scored) else None,
               score_rig_rule=fmean([(r["coverage"] * max(0.0, 1.0 - r["ate_sim3"] / r["cut3r_ate_same_frames"])) if (not r["failed"] and r.get("ate_sim3") is not None and r.get("cut3r_ate_same_frames")) else 0.0 for r in rows]),
               ms_per_frame_total_mean=fmean(vals("ms_per_frame_total", ok)), ms_per_frame_excl_init_mean=fmean(vals("ms_per_frame_excl_init", ok)),
               ms_per_track_call_mean=fmean(vals("ms_per_track_call", ok)), ms_per_init_call_mean=fmean(vals("ms_per_init_call", ok)),
               frames_per_second_tracking=(1000.0 / fmean(vals("ms_per_frame_excl_init", ok))) if vals("ms_per_frame_excl_init", ok) else None)
    out = a.out or f"{a.root}/pose_summary.json"
    json.dump(dict(method="silhouette_model", note="GT USED for scoring only (score.json per scene from seed_silhouette_model --diag)", root=a.root,
                   definitions=dict(pos_err_cm="|estimated wrist-camera position - kinematic GT lens| per solved frame (metric, no alignment)",
                                    rot_err_deg="angle(R_est^T R_gt) per solved frame", ate_sim3="rig_track2.traj_metrics: one Sim3 over all solved frames (tracks concatenated), RMSE",
                                    ate_metric_no_alignment="RMSE of the metric position error, no alignment",
                                    wrong_body_anchor_frac="fraction of emitted poses whose estimated lens is > 0.20 m from the GT lens (pooled)",
                                    score_rig_rule="mean over the 13 scenes of coverage x max(0, 1 - ate_sim3 / CUT3R ATE on the same frames); failed scenes 0"),
                   aggregate=agg, per_scene=rows), open(out, "w"), indent=1)
    print(json.dumps(agg, indent=1)); print("wrote", out)


if __name__ == "__main__":
    main()
