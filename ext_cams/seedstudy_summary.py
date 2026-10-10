#!/usr/bin/env python
"""Seed-study scorer: mask-accuracy proxies + rig-tracker records -> <root>/summary.json for one seeding METHOD.

DIAGNOSTIC ONLY: the kinematic GT wrist pose (store cam/*.npz) is used here for SCORING, never in a method path.

Per episode and exterior camera (native 1280x720; GT lens = store camera centre projected with the PointWorld
optimized_extrinsics (world->camera, used directly) and the factory intrinsics, in view = z > 0 and inside the image):
  lens_to_mask_px   distance (px) from the projected GT lens to the nearest mask pixel, on frames where the mask is
                    non-empty AND the lens is in view (0 when the lens falls inside the mask). Also in cm at the lens
                    depth (px * z / fx) so the expectation scales with depth (RAIL geometry: 0-40 px, ~0.4-0.5 m).
  inview_agreement  fraction of frames where (mask non-empty) == (lens in view)
  iou_vs_ref        per-frame IoU against the reference (privileged ceiling) masks over frames where either mask is
                    non-empty; also over frames where both are non-empty. -1 when the reference file is absent.
Scene-level numbers pool the frames of both cameras. Per-episode proxies are cached in <root>/mask_accuracy/<EP>.json.
Tracker records are read from <root>/<EP>/rig/record.json (written by rig_track.py); a missing record = failed.

    OMP_NUM_THREADS=1 python seedstudy_summary.py --method gtbox_sam3 --ref_method gtbox_sam3 [--episode EP] [--no_records]
"""
import argparse, glob, json, os, sys, time
import numpy as np

RAW_ROOT = "/gpfs/scratch/etur59/koc821022/vggt_cache/raw"
STORE_ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_wrist_all/dl3dv_multi/wrist"
AUDIT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/gt_audit_2026-09-08"
SEEDSTUDY = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy"
INTR_CACHE = f"{SEEDSTUDY}/intrinsics_cache"
SMOKE13 = "/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt"
W, H = 1280, 720
CAMS = ("ext1", "ext2")
RECORD_KEYS = ["status", "failure_reason", "n_frames", "n_frames_used", "mask_frames", "mask_frames_both", "window", "window_len",
               "window_gap_frames", "t_seed", "t_ref", "size_scale", "n_tracks", "pairs", "epipolar_px_median", "reproj_px",
               "tri_pairs_per_frame", "model_points_final", "n_anchored", "coverage_frac", "anchored_frac_window", "inliers",
               "rigid_resid_mm", "rigidity_bad_frac", "max_jump_cm", "max_jump_deg", "max_anchor_gap_frames",
               "pos_err_median_cm", "pos_err_p90_cm", "pos_err_max_cm", "rot_err_median_deg", "rot_err_p90_deg", "rot_err_max_deg",
               "rig_ate", "rig_rpe_t", "rig_rpe_rot", "sim3_scale", "cut3r_ate_same_frames", "cut3r_rpe_t_same_frames",
               "cut3r_rpe_rot_same_frames", "cut3r_sim3_scale_same_frames", "elapsed_s", "video", "video_bytes", "prepass_warning"]


def load_intrinsics(ep):
    cache = f"{INTR_CACHE}/{ep}.json"
    if os.path.isfile(cache):
        return json.load(open(cache))
    return json.load(open(f"{AUDIT}/docs/hf_intrinsics.json"))[ep]


def unpack(path):
    m = np.load(path)
    T, h, w = [int(v) for v in m["shape"]]
    u = np.unpackbits(m["union"], axis=2)[:, :, :w].astype(bool)
    return u, m["frames_present"].astype(bool), (T, h, w)


def fmed(x):
    return float(np.median(x)) if len(x) else None


def fq(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def proxies_episode(ep, method, root, ref_method, ref_root):
    meta = json.load(open(f"{RAW_ROOT}/{ep}/metadata_{ep}.json"))
    intr = load_intrinsics(ep)
    camj = json.load(open(f"{AUDIT}/pointworld/droid/cameras/{ep}_cameras.json"))
    store = sorted(glob.glob(f"{STORE_ROOT}/{ep}/dense/cam/*.npz"))
    n_store = len(store)
    gt = np.array([np.load(f)["pose"] for f in store])  # c2w, base frame (GT: scoring only)
    res = dict(episode=ep, method=method, ref_method=ref_method, n_frames_store=n_store, cams={}, products_ok=True)
    pooled = dict(dist_px=[], dist_cm=[], agree=[], iou_any=[], iou_both=[])
    nonempty_cams = {}
    for cam in CAMS:
        s = str(meta[f"{cam}_cam_serial"])
        path = f"{root}/{ep}/{cam}_{s}__{method}_masks.npz"
        r = dict(serial=s, npz=path, exists=os.path.isfile(path))
        if not r["exists"]:
            res["products_ok"] = False
            res["cams"][cam] = r
            continue
        u, present, (T, h, w) = unpack(path)
        nonempty = u.reshape(T, -1).any(1)
        r.update(T=T, H=h, W=w, bytes=os.path.getsize(path), T_matches_store=bool(T == n_store), shape_ok=bool((h, w) == (H, W)),
                 present_consistent=bool((nonempty == present).all()), frames_present=int(nonempty.sum()), frac_present=float(nonempty.mean()))
        if not (r["T_matches_store"] and r["shape_ok"] and r["present_consistent"]):
            res["products_ok"] = False
        nonempty_cams[cam] = nonempty
        fx, cx, fy, cy = intr[s]["cameraMatrix"]
        iw, ih = intr[s].get("width", W), intr[s].get("height", H)
        sx, sy = W / iw, H / ih
        fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy
        E = np.array(camj[s]["optimized_extrinsics"], np.float64)
        n = min(T, n_store)
        pc = (E[:3, :3] @ gt[:n, :3, 3].T + E[:3, 3:]).T  # (n, 3) lens in camera coords
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            px = fx * pc[:, 0] / z + cx
            py = fy * pc[:, 1] / z + cy
        in_view = (z > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        dist_px = np.full(n, np.nan)
        for t in np.where(in_view & nonempty[:n])[0]:
            ys, xs = np.nonzero(u[t])
            dist_px[t] = float(np.min(np.hypot(xs - px[t], ys - py[t])))
        # distance is 0 when the lens pixel itself is inside the mask; nearest-pixel distance is otherwise centre-to-centre
        both_ok = np.isfinite(dist_px)
        dist_cm = dist_px * z / fx * 100.0
        agree = (nonempty[:n] == in_view)
        r.update(gt_in_view=int(in_view.sum()), present_and_in_view=int(both_ok.sum()),
                 present_when_out_of_view=int((nonempty[:n] & ~in_view).sum()), in_view_without_mask=int((in_view & ~nonempty[:n]).sum()),
                 recall_in_view=float(both_ok.sum() / max(in_view.sum(), 1)),
                 lens_to_mask_px_median=fmed(dist_px[both_ok]), lens_to_mask_px_p90=fq(dist_px[both_ok], 90),
                 lens_to_mask_px_max=float(dist_px[both_ok].max()) if both_ok.any() else None,
                 lens_inside_mask_frac=float((dist_px[both_ok] == 0).mean()) if both_ok.any() else None,
                 lens_to_mask_cm_median=fmed(dist_cm[both_ok]), lens_depth_m_median=fmed(z[in_view]),
                 far_frames_gt100px=int((dist_px[both_ok] > 100).sum()), inview_agreement=float(agree.mean()),
                 mean_area_frac_present=float(u[:n][nonempty[:n]].reshape(-1, H * W).mean()) if nonempty[:n].any() else None)
        pooled["dist_px"].extend(dist_px[both_ok].tolist()); pooled["dist_cm"].extend(dist_cm[both_ok].tolist()); pooled["agree"].extend(agree.tolist())
        # IoU against the reference masks
        ref_path = f"{ref_root}/{ep}/{cam}_{s}__{ref_method}_masks.npz"
        r["ref_npz"] = ref_path
        if os.path.realpath(ref_path) == os.path.realpath(path):
            r.update(iou_is_self_reference=True, iou_vs_ref_median_any=1.0, iou_vs_ref_median_both=1.0, iou_frames_any=int(nonempty.sum()), iou_frames_both=int(nonempty.sum()))
            pooled["iou_any"].extend([1.0] * int(nonempty.sum())); pooled["iou_both"].extend([1.0] * int(nonempty.sum()))
        elif os.path.isfile(ref_path):
            v, _, (Tr, hr, wr) = unpack(ref_path)
            m = min(T, Tr)
            inter = (u[:m] & v[:m]).reshape(m, -1).sum(1).astype(float)
            union = (u[:m] | v[:m]).reshape(m, -1).sum(1).astype(float)
            any_ = union > 0
            both_ = nonempty[:m] & v[:m].reshape(m, -1).any(1)
            iou = np.where(any_, inter / np.maximum(union, 1), np.nan)
            r.update(iou_is_self_reference=False, ref_T=Tr, iou_vs_ref_median_any=fmed(iou[any_]), iou_vs_ref_median_both=fmed(iou[both_]),
                     iou_vs_ref_mean_any=float(iou[any_].mean()) if any_.any() else None, iou_frames_any=int(any_.sum()), iou_frames_both=int(both_.sum()),
                     ref_frames_present=int(v[:m].reshape(m, -1).any(1).sum()))
            pooled["iou_any"].extend(iou[any_].tolist()); pooled["iou_both"].extend(iou[both_].tolist())
        else:
            r.update(iou_is_self_reference=False, iou_vs_ref_median_any=-1, iou_vs_ref_median_both=-1, ref_missing=True)
        res["cams"][cam] = r
    if len(nonempty_cams) == 2:
        n = min(len(nonempty_cams["ext1"]), len(nonempty_cams["ext2"]))
        res["mask_frames_both"] = int((nonempty_cams["ext1"][:n] & nonempty_cams["ext2"][:n]).sum())
    else:
        res["mask_frames_both"] = 0
    d = np.array(pooled["dist_px"]); dc = np.array(pooled["dist_cm"])
    res.update(lens_to_mask_px_median=fmed(d), lens_to_mask_px_p90=fq(d, 90), lens_to_mask_cm_median=fmed(dc),
               lens_inside_mask_frac=float((d == 0).mean()) if len(d) else None,
               inview_agreement=float(np.mean(pooled["agree"])) if pooled["agree"] else None,
               iou_vs_ref_median=(fmed(np.array(pooled["iou_any"])) if pooled["iou_any"] else -1) if all(res["cams"][c].get("iou_vs_ref_median_any", -1) != -1 for c in res["cams"]) else -1,
               iou_vs_ref_median_both_nonempty=fmed(np.array(pooled["iou_both"])) if pooled["iou_both"] else -1,
               iou_is_self_reference=any(res["cams"][c].get("iou_is_self_reference") for c in res["cams"]),
               n_frames_scored=int(len(d)))
    return res


def per_scene_row(ep, px, rec):
    row = dict(episode=ep, n_frames=px.get("n_frames_store"), mask_frames_both=px.get("mask_frames_both", 0),
               lens_to_mask_px_median=px.get("lens_to_mask_px_median"), lens_to_mask_cm_median=px.get("lens_to_mask_cm_median"),
               inview_agreement=px.get("inview_agreement"), iou_vs_ref_median=px.get("iou_vs_ref_median", -1),
               mask_products_ok=px.get("products_ok"))
    if rec is None:
        row.update(failed=True, failure_reason="no record.json (tracker not run or crashed)", n_anchored=0, coverage_frac=0.0)
        return row
    failed = rec.get("status") != "ok"
    row.update(failed=failed, failure_reason=rec.get("failure_reason"), n_anchored=int(rec.get("n_anchored") or 0),
               coverage_frac=float(rec.get("coverage_frac") or 0.0))
    for k in ("rig_ate", "rig_rpe_t", "rig_rpe_rot", "sim3_scale", "pos_err_median_cm", "pos_err_p90_cm", "rot_err_median_deg", "rot_err_p90_deg",
              "cut3r_ate_same_frames", "cut3r_rpe_t_same_frames", "cut3r_rpe_rot_same_frames", "window", "window_len", "pairs", "model_points_final",
              "anchored_frac_window", "max_anchor_gap_frames", "rigidity_bad_frac", "elapsed_s", "video"):
        if k in rec and rec[k] is not None:
            row[k] = rec[k]
    row["record"] = {k: rec[k] for k in RECORD_KEYS if k in rec}
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True)
    ap.add_argument("--root", default=None, help="default SEEDSTUDY/<method>")
    ap.add_argument("--ref_method", default="gtbox_sam3")
    ap.add_argument("--ref_root", default=None, help="default SEEDSTUDY/<ref_method>")
    ap.add_argument("--episodes", default=SMOKE13)
    ap.add_argument("--episode", default=None, help="only compute (and cache) the proxies of this episode; no summary")
    ap.add_argument("--rig_subdir", default="rig", help="tracker out dir = <root>/<EP>/<rig_subdir>")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--no_records", action="store_true", help="proxies only")
    a = ap.parse_args()
    root = a.root or f"{SEEDSTUDY}/{a.method}"
    ref_root = a.ref_root or f"{SEEDSTUDY}/{a.ref_method}"
    cache_dir = f"{root}/mask_accuracy"
    os.makedirs(cache_dir, exist_ok=True)
    eps = [a.episode] if a.episode else [l.strip() for l in open(a.episodes) if l.strip()]
    rows, prox = [], {}
    for ep in eps:
        t0 = time.time()
        cache = f"{cache_dir}/{ep}.json"
        if os.path.isfile(cache) and not a.recompute:
            px = json.load(open(cache))
            src = "cached"
        else:
            px = proxies_episode(ep, a.method, root, a.ref_method, ref_root)
            json.dump(px, open(cache, "w"), indent=1)
            src = f"{time.time() - t0:.1f}s"
        prox[ep] = px
        print(f"proxies {ep[:40]:40s} ok={px['products_ok']} both={px['mask_frames_both']} lens2mask px={px['lens_to_mask_px_median']} "
              f"cm={px['lens_to_mask_cm_median']} agree={px['inview_agreement']} iou={px['iou_vs_ref_median']} ({src})", flush=True)
    if a.episode or a.no_records:
        return
    for ep in eps:
        rp = f"{root}/{ep}/{a.rig_subdir}/record.json"
        rec = json.load(open(rp)) if os.path.isfile(rp) else None
        rows.append(per_scene_row(ep, prox[ep], rec))
    ok = [r for r in rows if not r["failed"]]
    # a status-ok scene with < 2 anchored frames has no trajectory metrics (rig_ate etc. None): it counts as ok for
    # scenes_ok / coverage, and is left out of the metric means, which are taken over the scenes that hold the value
    scored = [r for r in ok if r.get("rig_ate") is not None]

    def vals(key, src):
        return [r[key] for r in src if r.get(key) is not None]

    def fmean(x):
        return float(np.mean(x)) if len(x) else None

    agg = dict(n_scenes=len(rows), scenes_ok=len(ok), scenes_failed=[r["episode"] for r in rows if r["failed"]],
               scenes_scored=len(scored), scenes_ok_unscored=[r["episode"] for r in ok if r.get("rig_ate") is None],
               mean_coverage=fmean([r["coverage_frac"] for r in rows]),
               mean_coverage_ok_only=fmean([r["coverage_frac"] for r in ok]),
               mean_rig_ate=fmean(vals("rig_ate", ok)),
               mean_rig_rpe_t=fmean(vals("rig_rpe_t", ok)),
               mean_rig_rpe_rot=fmean(vals("rig_rpe_rot", ok)),
               mean_cut3r_ate_same_frames=fmean(vals("cut3r_ate_same_frames", ok)),
               median_pos_err_cm=fmed(vals("pos_err_median_cm", scored)),
               median_rot_err_deg=fmed(vals("rot_err_median_deg", scored)),
               mean_pos_err_median_cm=fmean(vals("pos_err_median_cm", scored)),
               mean_rot_err_median_deg=fmean(vals("rot_err_median_deg", scored)),
               median_lens_to_mask_px=fmed([r["lens_to_mask_px_median"] for r in rows if r["lens_to_mask_px_median"] is not None]),
               mean_inview_agreement=float(np.mean([r["inview_agreement"] for r in rows if r["inview_agreement"] is not None])) if rows else None,
               median_iou_vs_ref=fmed([r["iou_vs_ref_median"] for r in rows if r["iou_vs_ref_median"] not in (None, -1)]) if rows else None,
               mask_products_ok=int(sum(bool(r["mask_products_ok"]) for r in rows)),
               n_frames_total=int(sum(r["n_frames"] or 0 for r in rows)), n_anchored_total=int(sum(r["n_anchored"] for r in rows)))
    summary = dict(method=a.method, ref_method=a.ref_method, root=root, episodes_file=a.episodes, generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                   definitions=dict(lens_to_mask_px="median over frames (both cams pooled) with mask non-empty AND GT lens in view of the px distance from the projected GT lens to the nearest mask pixel (0 = lens inside mask)",
                                    lens_to_mask_cm="same distance scaled by the lens depth: px * z / fx * 100",
                                    inview_agreement="fraction of frames (both cams pooled) where (mask non-empty) == (GT lens in view, z>0 and inside the 1280x720 image)",
                                    iou_vs_ref_median="median per-frame IoU vs the reference masks over frames where either is non-empty (both cams pooled); 1.0 by construction when the method IS the reference; -1 when the reference is absent",
                                    coverage_frac="tracker: anchored frames / store frames", cut3r_ate_same_frames="finetuned CUT3R (augfull_lr1e5) ATE on the anchored frames only, Sim3-aligned"),
                   aggregate=agg, per_scene=rows)
    out = f"{root}/summary.json"
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps(agg, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
