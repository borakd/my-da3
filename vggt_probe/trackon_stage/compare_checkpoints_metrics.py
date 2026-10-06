"""Quantitative comparison of Track-On checkpoints on the example scenes.

There is no ground truth, so three independent proxies are used, each gated so that frames where the
gripper has left the view (or the pseudo-GT is implausible) do not count as tracking failures.

Pseudo ground truth (evaluation only): RobotSeg automatic gripper masks on every 3rd frame after birth.
Valid eval frame := mask area >= 0.5 x the camera's median eval-mask area, and (where the camera's
extrinsics passed the consistency check) kinematic in-view fraction of the gripper model >= 0.5.
Evaluable points := query points inside the largest connected component of the birth mask and >= 5 px
from the image border (drops points RobotSeg put on background fragments at birth). Point-frames within
3 px of the border are excluded from mask metrics (border-clamped exits).

A. Agreement with RobotSeg masks, pooled over valid point-frames (reported for 0 px and 5 px dilation):
     in_mask_vis   P(point inside mask | tracker flags it visible)        -> precision of visible tracks
     in_mask_all   P(point inside mask) over all evaluable points         -> points still riding the gripper
     lost_frac     visible point-frames > 60 px from the mask              -> points that left the gripper
     boundary_frac visible point-frames outside but < 10 px from the mask -> boundary jitter, not loss
     survive_end   in_mask_all at the LAST valid eval frame
B. Visibility flag quality vs pseudo-labels:
     vis_precision P(inside mask | flagged visible); vis_recall P(flagged visible | inside mask) on valid frames
     false_vis_out fraction of points flagged visible on frames where the gripper is OUT of view
                   (kinematic in-view fraction < 0.25 or mask area < 0.25 x median)  -> lower is better
     flicker       visibility toggles per point per 100 frames                      -> lower is better
C. Kinematic motion agreement (cameras with valid extrinsics; frames with in-view fraction >= 0.9):
     drift_px      median |(c_track(t) - c_kin(t)) - (c_track(t0) - c_kin(t0))|, c_track = median over all
                   evaluable points; cancels the constant mask-vs-box offset            -> lower is better
     step_err_px   median |delta c_track - delta c_kin| between consecutive valid eval frames
D. Smoothness: jitter_px (median |second difference| of visible tracks), jump_frac (visible steps > 30 px).
E. Cross-checkpoint agreement: median distance where both visible; fraction < 8 px.
fps is reported as the median over items (first item of a job carries CUDA warm-up); both checkpoints
share one architecture, so speed is identical by construction.
Writes comparisons/metrics.json, metrics.md, in_mask_over_time.png.
"""
import collections
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # vggt_probe: birth_frames.py
import birth_frames as bf  # noqa: E402

O = bf.OUT_ROOT
CMP = f"{O}/comparisons"
DILATES = (0, 5)
BORDER = 3
LOST_PX = 60.0
BOUND_PX = 10.0
JUMP_PX = 30.0
W, H = 1280, 720


def nanmed(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.median(x)) if x else float("nan")


def load_tracks(ckpt, ep, cam, B):
    z = np.load(f"{CMP}/{ckpt}/tracks/{ep}/{cam}_f{B:05d}.npz")
    return z["tracks"], z["visibility"], z["queries"], json.loads(str(z["meta"]))


def eval_masks(ep, cam):
    out = {}
    for p in glob.glob(f"{CMP}/eval_masks/{ep}/{cam}_f*.png"):
        t = int(os.path.basename(p).split("_f")[1][:5])
        out[t] = cv2.imread(p, 0) > 0
    return out


def kin_info(ep, cam, frames):
    meta = json.load(open(f"{bf.META_DIR}/{ep}.json"))
    K = bf.load_zed_K(meta[f"{cam}_cam_serial"])
    T_bc = np.linalg.inv(bf.pose6_to_T(np.array(meta[f"{cam}_cam_extrinsics"])))
    P = bf.gripper_points_cam()
    out = {}
    for t in frames:
        with np.load(f"{bf.STORE}/{ep}/dense/cam/{t:06d}.npz") as z:
            pose = z["pose"]
        Pb = (pose[:3, :3] @ P.T).T + pose[:3, 3]
        frac, uv, vis = bf.visible_fraction(T_bc, K, Pb)
        out[t] = dict(frac=float(frac), centroid=(uv[vis].mean(0) if vis.sum() else None))
    return out


def evaluable_points(birth_mask, queries):
    n, lab = cv2.connectedComponents(birth_mask.astype(np.uint8))
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        big = sizes.argmax()
    xi = np.clip(queries[:, 1].round().astype(int), 0, W - 1)
    yi = np.clip(queries[:, 2].round().astype(int), 0, H - 1)
    keep = (lab[yi, xi] == big) if n > 1 else np.ones(len(queries), bool)
    keep &= (queries[:, 1] >= 5) & (queries[:, 1] <= W - 6) & (queries[:, 2] >= 5) & (queries[:, 2] <= H - 6)
    return keep


def main():
    items = open(f"{CMP}/items.txt").read().split()
    births = {r["episode"]: r for r in json.load(open(f"{O}/birth_frames.json"))}
    bm = {(r["episode"], r["cam"]): r for r in csv.DictReader(open(f"{O}/birth_masks.csv")) if r["which"] == "b050"}
    ckpts = sorted(d for d in os.listdir(CMP) if os.path.isdir(f"{CMP}/{d}/tracks"))
    timing = collections.defaultdict(list)
    for c in ckpts:
        for l in open(f"{CMP}/{c}/timing.jsonl"):
            r = json.loads(l)
            timing[c].append(r["fps"])
    results, curves, gating = [], {}, {}
    for item in items:
        ep, cam = item.split(":")
        B = births[ep]["birth_f050"]
        masks = eval_masks(ep, cam)
        frames = sorted(masks)
        areas = {t: masks[t].mean() for t in frames}
        med_area = float(np.median(list(areas.values())))
        kin_ok = bm[(ep, cam)]["projection_disagree"] != "True"
        kin = kin_info(ep, cam, frames) if kin_ok else {}
        valid = [t for t in frames if areas[t] >= 0.5 * med_area and (not kin_ok or kin[t]["frac"] >= 0.5)]
        out_frames = [t for t in frames if (kin_ok and kin[t]["frac"] < 0.25) or areas[t] < 0.25 * med_area]
        kin_strict = [t for t in valid if kin_ok and kin[t]["frac"] >= 0.9 and kin[t]["centroid"] is not None]
        gating[item] = dict(n_eval=len(frames), n_valid=len(valid), n_out=len(out_frames), kin_ok=kin_ok, n_kin_strict=len(kin_strict))
        birth_mask = cv2.imread(bm[(ep, cam)]["mask"].replace("masks/", f"{O}/masks_run/masks/") if not bm[(ep, cam)]["mask"].startswith("/") else bm[(ep, cam)]["mask"], 0)
        if birth_mask is None:
            birth_mask = cv2.imread(f"{O}/masks_run/{bm[(ep, cam)]['mask']}", 0)
        birth_mask = birth_mask > 0
        dil = {d: {t: (cv2.dilate(masks[t].astype(np.uint8), np.ones((2 * d + 1,) * 2, np.uint8)) > 0 if d else masks[t]) for t in frames} for d in DILATES}
        dist5 = {t: cv2.distanceTransform((~dil[5][t]).astype(np.uint8), cv2.DIST_L2, 5) for t in frames}
        per_ckpt = {}
        for c in ckpts:
            tr, vi, q, meta = load_tracks(c, ep, cam, B)
            keep = evaluable_points(birth_mask, q)
            tr_e, vi_e = tr[:, keep], vi[:, keep]
            T, N = tr_e.shape[:2]
            acc = {d: dict(vis_in=0, vis_n=0, all_in=0, all_n=0) for d in DILATES}
            lost = bound = strays_n = 0
            fv_out = []
            vis_rec_in = vis_rec_n = 0
            curve = []
            surv = float("nan")
            for t in valid:
                if t >= T or np.isnan(tr_e[t]).all():
                    continue
                pts = tr_e[t]
                xi = np.clip(pts[:, 0].round().astype(int), 0, W - 1)
                yi = np.clip(pts[:, 1].round().astype(int), 0, H - 1)
                inb = (pts[:, 0] >= BORDER) & (pts[:, 0] <= W - 1 - BORDER) & (pts[:, 1] >= BORDER) & (pts[:, 1] <= H - 1 - BORDER)
                v = vi_e[t] & inb
                for d in DILATES:
                    inside = dil[d][t][yi, xi]
                    acc[d]["all_in"] += int(inside[inb].sum()); acc[d]["all_n"] += int(inb.sum())
                    acc[d]["vis_in"] += int(inside[v].sum()); acc[d]["vis_n"] += int(v.sum())
                inside5 = dil[5][t][yi, xi]
                dd = dist5[t][yi[v & ~inside5], xi[v & ~inside5]]
                strays_n += len(dd); lost += int((dd > LOST_PX).sum()); bound += int((dd < BOUND_PX).sum())
                vis_rec_in += int((vi_e[t] & inside5 & inb).sum()); vis_rec_n += int((inside5 & inb).sum())
                curve.append((t, float(inside5[inb].mean()) if inb.any() else np.nan))
                surv = float(inside5[inb].mean()) if inb.any() else surv
            for t in out_frames:
                if t < T and not np.isnan(tr_e[t]).all():
                    fv_out.append(float(vi_e[t].mean()))
            # flicker
            vseg = vi_e[B:]
            toggles = np.abs(np.diff(vseg.astype(int), axis=0)).sum()
            flicker = float(toggles / max(N, 1) / max(len(vseg), 1) * 100)
            # kinematic motion agreement
            drift, steps = [], []
            if kin_strict:
                t0 = kin_strict[0]
                c0 = np.nanmedian(tr_e[t0], axis=0) - kin[t0]["centroid"]
                prev = None
                for t in kin_strict:
                    ct = np.nanmedian(tr_e[t], axis=0)
                    drift.append(float(np.linalg.norm((ct - kin[t]["centroid"]) - c0)))
                    if prev is not None:
                        steps.append(float(np.linalg.norm((ct - prev[0]) - (kin[t]["centroid"] - prev[1]))))
                    prev = (ct, kin[t]["centroid"])
            # smoothness
            seg = tr_e[B:]
            d1 = np.diff(seg, axis=0)
            step = np.linalg.norm(d1, axis=-1)
            both_vis = vseg[1:] & vseg[:-1]
            d2 = np.linalg.norm(np.diff(d1, axis=0), axis=-1)
            vis3 = vseg[2:] & vseg[1:-1] & vseg[:-2]
            res = dict(ckpt=c, episode=ep, cam=cam, birth=B, n_points=int(tr.shape[1]), n_points_eval=int(N),
                       n_eval_valid=len(valid), kin_ok=kin_ok)
            for d in DILATES:
                a = acc[d]
                res[f"in_mask_vis_d{d}"] = a["vis_in"] / a["vis_n"] if a["vis_n"] else float("nan")
                res[f"in_mask_all_d{d}"] = a["all_in"] / a["all_n"] if a["all_n"] else float("nan")
            res.update(lost_frac=lost / acc[5]["vis_n"] if acc[5]["vis_n"] else float("nan"),
                       boundary_frac=bound / acc[5]["vis_n"] if acc[5]["vis_n"] else float("nan"),
                       survive_end=surv,
                       vis_precision=res["in_mask_vis_d5"],
                       vis_recall=vis_rec_in / vis_rec_n if vis_rec_n else float("nan"),
                       false_vis_out=float(np.mean(fv_out)) if fv_out else float("nan"),
                       n_out_frames=len(fv_out), flicker=flicker,
                       drift_px=nanmed(drift), step_err_px=nanmed(steps),
                       jitter_px=float(np.median(d2[vis3])) if vis3.any() else float("nan"),
                       jump_frac=float((step[both_vis] > JUMP_PX).mean()) if both_vis.any() else float("nan"),
                       vis_mean=float(vseg.mean()), vis_last=float(vseg[-1].mean()),
                       fps_median=nanmed(timing[c]))
            results.append(res)
            per_ckpt[c] = (tr_e, vi_e)
            curves[(item, c)] = curve
        if len(ckpts) == 2:
            (ta, va), (tb_, vb) = per_ckpt[ckpts[0]], per_ckpt[ckpts[1]]
            T = min(len(ta), len(tb_))
            both = va[:T] & vb[:T]
            d = np.linalg.norm(ta[:T] - tb_[:T], axis=-1)[both]
            results.append(dict(ckpt=f"{ckpts[0]} vs {ckpts[1]}", episode=ep, cam=cam, birth=B,
                                agree_median_px=float(np.median(d)) if len(d) else float("nan"),
                                agree_lt8px=float((d < 8).mean()) if len(d) else float("nan"),
                                both_visible_frac=float(both[B:].mean())))
    json.dump(dict(results=results, gating=gating), open(f"{CMP}/metrics.json", "w"), indent=1)

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        return f"{v:.3f}" if isinstance(v, float) and abs(v) < 10 else f"{v:.1f}"

    L = ["# Track-On checkpoint comparison on the two example scenes", "",
         f"Checkpoints: {', '.join(ckpts)} (track_on_r = Track-On-R, trackon2 = Track-On2). Identical query points from the "
         "birth-frame RobotSeg mask; causal tracking from the birth frame to the end. No ground truth exists: the proxies "
         "below use RobotSeg masks on every 3rd frame (evaluation only) and, where the extrinsics are trustworthy, the kinematic "
         "gripper projection. Metric definitions and gating rules are in the script docstring "
         "(`vggt_probe/trackon_stage/compare_checkpoints_metrics.py`).", ""]
    L.append("## Frame and point gating")
    L.append("")
    L.append("| scene | cam | eval frames | valid | gripper-out frames | kinematics usable | strict-kin frames |")
    L.append("|---|---|---|---|---|---|---|")
    for item in items:
        g = gating[item]
        ep, cam = item.split(":")
        L.append(f"| {ep.split('+')[2][:16]} | {cam} | {g['n_eval']} | {g['n_valid']} | {g['n_out']} | {g['kin_ok']} | {g['n_kin_strict']} |")
    L.append("")
    main_cols = ["n_points_eval", "in_mask_vis_d0", "in_mask_vis_d5", "in_mask_all_d5", "lost_frac", "boundary_frac", "survive_end",
                 "vis_recall", "false_vis_out", "flicker", "drift_px", "step_err_px", "jitter_px", "jump_frac"]
    L.append("## Per camera-video")
    L.append("")
    L.append("| scene | cam | ckpt | " + " | ".join(main_cols) + " |")
    L.append("|" + "---|" * (3 + len(main_cols)))
    for r in results:
        if "agree_median_px" in r:
            continue
        L.append(f"| {r['episode'].split('+')[2][:16]} | {r['cam']} | {r['ckpt']} | " + " | ".join(fmt(r.get(k)) for k in main_cols) + " |")
    L.append("")
    L.append("## Mean over the 4 camera-videos (kinematic columns over the 3 with usable extrinsics)")
    L.append("")
    L.append("| ckpt | " + " | ".join(main_cols[1:]) + " | fps (median) |")
    L.append("|" + "---|" * (2 + len(main_cols[1:])))
    for c in ckpts:
        rs = [r for r in results if r["ckpt"] == c]
        vals = []
        for k in main_cols[1:]:
            xs = [r[k] for r in rs if not (isinstance(r.get(k), float) and np.isnan(r[k])) and r.get(k) is not None]
            vals.append(fmt(float(np.mean(xs))) if xs else "-")
        L.append(f"| {c} | " + " | ".join(vals) + f" | {fmt(rs[0]['fps_median'])} |")
    L.append("")
    L.append("Direction: higher is better for in_mask_*, survive_end, vis_recall; lower is better for lost_frac, boundary_frac, "
             "false_vis_out, flicker, drift_px, step_err_px, jitter_px, jump_frac. vis_mean/vis_last are descriptive only "
             "(the gripper leaves the view at the end of scene 2 ext1, so low visibility there is correct). "
             "Speed is identical by construction (same architecture); the median fps removes the first-item CUDA warm-up.")
    L.append("")
    L.append("## Cross-checkpoint agreement")
    L.append("")
    for r in results:
        if "agree_median_px" in r:
            L.append(f"- {r['episode'].split('+')[2][:16]} {r['cam']}: median distance {r['agree_median_px']:.1f} px where both visible, "
                     f"{100*r['agree_lt8px']:.0f}% within 8 px, both visible on {100*r['both_visible_frac']:.0f}% of point-frames")
    L.append("")
    L.append("Caveat: two episodes from one lab rig (four camera views) are enough to see whether the checkpoints behave "
             "differently on this data, not to rank them in general; differences below ~0.02 in the mask metrics are within "
             "the pseudo-GT noise (the 0 px vs 5 px dilation columns bound it).")
    open(f"{CMP}/metrics.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    COL = {ckpts[i]: c for i, c in enumerate(["#2a78d6", "#eb6834", "#1baf7a"][:len(ckpts)])}
    fig, axes = plt.subplots(1, len(items), figsize=(4.2 * len(items), 3.2), facecolor=SURF, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, item in zip(axes, items):
        ep, cam = item.split(":")
        ax.set_facecolor(SURF)
        for c in ckpts:
            cv = curves[(item, c)]
            if cv:
                ax.plot([x[0] for x in cv], [x[1] for x in cv], color=COL[c], lw=2, label=c)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{ep.split('+')[2][:16]} {cam}  (valid frames only)", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("frame", fontsize=8, color=INK2)
        ax.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].set_ylabel("fraction of evaluable points inside\nthe RobotSeg gripper mask (+5 px)", fontsize=8, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(ckpts), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(f"{CMP}/in_mask_over_time.png", dpi=130, bbox_inches="tight", facecolor=SURF)
    print("wrote", f"{CMP}/in_mask_over_time.png")


if __name__ == "__main__":
    main()
