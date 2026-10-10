#!/usr/bin/env python
"""Score + visualise the RobotSeg exterior-camera gripper masks written by robotseg_gripper_ext.py (CPU only).

Against the REFERENCE = SAM3 box+click gripper masks (human-prompted, approved 2026-09-28; not human-annotated GT):
  J, F           per frame, computed exactly like RobotSeg's tools/benchmark.py (vos-benchmark Evaluator: get_iou,
                 _seg2bmap, skimage disk of ceil(0.008 * diag)); frames enter the average from the first frame where
                 either mask is non-empty (cumulative object set), both-empty frames count 1 => "J&F (VRS protocol)".
  J_ref          mean IoU over the frames where the reference is non-empty (stricter; empty prediction = 0).
  presence       TP/FP/FN/TN frames (non-empty pred vs non-empty ref), first non-empty pred frame vs reference entry.
  pix_prec/rec   pooled pixel precision / recall vs the reference (over- vs under-segmentation).
  area_ratio     median pred/ref area on TP frames.
Against the KINEMATIC projection (independent of SAM3; seedstudy_score_gt.py): hit@25, present_inview, miss_far, d_med,
spill — on the 1280x720 contract masks (320x180 runs upsampled x4 nearest). The reference itself is scored as a row.
For arm/robot categories: ref_cov = fraction of reference gripper pixels inside the mask, and IoU vs the SAM3
"robotic arm" text masks (rough; that file also holds a tiny background-robot instance).
320x180 runs are scored at 320x180 against the reference area-downsampled (INTER_AREA >= 0.5).

Products under OUT_ROOT/<res>/<EP>/score/: summary.json, summary.md, per_frame.csv, timeline.png,
contact_<cam>_crop.png, contact_<cam>_full.png, overlay_<variant>.mp4 (ext1 | ext2; pred = magenta fill + bbox, ref = yellow outline).

    OMP_NUM_THREADS=1 python robotseg_score.py --res 1280x720 [--no_video]
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robotseg_gripper_ext import OUT_ROOT, REF_DIR, REF_SUFFIX, ROBOTSEG, VARIANTS, load_contract  # noqa: E402
from sam3_gripper_masks import FfmpegWriter, RAW_ROOT  # noqa: E402
import seedstudy_score_gt as ssg  # noqa: E402

ARM_SUFFIX = "__arm_masks.npz"   # SAM3 "robotic arm" text-prompt masks (gripper_sam3 dir)
# categorical slots 1..8 of the dataviz default palette, fixed order (never cycled)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRIPPER_VARIANTS = ["auto_f0_gripper", "auto_entry_gripper", "image_gripper", "bbox_entry_gripper", "auto_vis_gripper", "bbox_vis_gripper"]


def vos_bench():
    spec = importlib.util.spec_from_file_location("robotseg_benchmark", os.path.join(ROBOTSEG, "tools", "benchmark.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unpack(z, key):
    T, H, W = [int(v) for v in z["shape"]]
    return np.unpackbits(z[key], axis=2)[:, :, :W].astype(bool)


def jf_frame(pred, gt, disk, bm):
    J = bm.get_iou(int((pred & gt).sum()), int(pred.sum() + gt.sum()))
    pb, gb = bm._seg2bmap(pred), bm._seg2bmap(gt)
    pd = cv2.dilate(pb.astype(np.uint8), disk)
    gd = cv2.dilate(gb.astype(np.uint8), disk)
    n_fg, n_gt = pb.sum(), gb.sum()
    if n_fg == 0 and n_gt > 0:
        P, R = 1.0, 0.0
    elif n_fg > 0 and n_gt == 0:
        P, R = 0.0, 1.0
    elif n_fg == 0 and n_gt == 0:
        P, R = 1.0, 1.0
    else:
        P, R = float((pb * gd).sum()) / n_fg, float((gb * pd).sum()) / n_gt
    F = 0.0 if P + R == 0 else 2 * P * R / (P + R)
    return float(J), float(F)


def score_vs_ref(pred, ref, bm):
    T = len(ref)
    from skimage.morphology import disk as skdisk
    disk = skdisk(math.ceil(0.008 * np.linalg.norm(ref.shape[1:])))
    pp, rp = pred.reshape(T, -1).any(1), ref.reshape(T, -1).any(1)
    J = np.zeros(T)
    F = np.zeros(T)
    for t in range(T):
        J[t], F[t] = jf_frame(pred[t], ref[t], disk, bm)
    started = np.cumsum(pp | rp) > 0          # vos-benchmark: object set is cumulative over frames
    inter = (pred & ref).reshape(T, -1).sum(1)
    pa, ra = pred.reshape(T, -1).sum(1), ref.reshape(T, -1).sum(1)
    tp = pp & rp
    entry = int(np.argmax(rp))
    first_pred = int(np.argmax(pp)) if pp.any() else None
    cdist = np.full(T, np.nan)
    for t in np.nonzero(tp)[0]:
        py, px = np.nonzero(pred[t])
        ry, rx = np.nonzero(ref[t])
        cdist[t] = math.hypot(px.mean() - rx.mean(), py.mean() - ry.mean())
    s = dict(JF_vrs=float(((J + F) / 2)[started].mean()), J_vrs=float(J[started].mean()), F_vrs=float(F[started].mean()),
             J_ref=float(np.where(rp, J, np.nan)[rp].mean()), F_ref=float(F[rp].mean()),
             TP=int(tp.sum()), FP=int((pp & ~rp).sum()), FN=int((~pp & rp).sum()), TN=int((~pp & ~rp).sum()),
             FP_pre_entry=int(pp[:entry].sum()), ref_entry=entry, first_pred=first_pred,
             pix_prec=float(inter.sum() / max(pa.sum(), 1)), pix_rec=float(inter.sum() / max(ra.sum(), 1)),
             area_ratio_med=float(np.median(pa[tp] / ra[tp])) if tp.any() else None,
             cdist_med_px=float(np.nanmedian(cdist)) if tp.any() else None,
             frac_ref_frames_J50=float((J[rp] >= 0.5).mean()))
    return s, J, F, pa / pred[0].size, ra / ref[0].size


def bbox(m):
    ys, xs = np.nonzero(m)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None


def paint(img, pred, ref, thick=1, boxes=True):
    """pred = magenta fill + outline (+ bbox), ref = yellow outline (+ dashed-looking thin bbox). High contrast against this
    scene's blue background and green towel."""
    out = img.copy()
    if pred is not None and pred.any():
        ov = out.copy()
        ov[pred] = (255, 0, 255)
        out = cv2.addWeighted(ov, 0.5, out, 0.5, 0)
        cs, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, (255, 0, 255), thick)
        if boxes:
            x0, y0, x1, y1 = bbox(pred)
            cv2.rectangle(out, (int(x0) - 3 * thick, int(y0) - 3 * thick), (int(x1) + 3 * thick, int(y1) + 3 * thick), (255, 0, 255), thick)
    if ref is not None and ref.any():
        cs, _ = cv2.findContours(ref.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, (0, 255, 255), thick + 1)
    return out


def frame_roi(pred, ref, W, H, min_w):
    """Tight 16:9 crop around pred U ref on this frame (whole frame if both empty)."""
    bs = [b for b in (bbox(pred) if pred is not None else None, bbox(ref)) if b is not None]
    if not bs:
        return 0, 0, W, H
    x0, y0 = min(b[0] for b in bs), min(b[1] for b in bs)
    x1, y1 = max(b[2] for b in bs), max(b[3] for b in bs)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max(min_w, 1.5 * (x1 - x0), 1.5 * (y1 - y0) * 16 / 9)
    w = min(w, W)
    h = min(w * 9 / 16, H)
    xa, ya = int(np.clip(cx - w / 2, 0, W - w)), int(np.clip(cy - h / 2, 0, H - h))
    return xa, ya, int(w), int(h)


def label(img, text, scale=0.45):
    cv2.rectangle(img, (0, 0), (img.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(img, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def contact_sheet(frames, preds, ref, rows, cols, roi, cell_w, path, title, Js=None):
    """roi=None -> per-cell tight crop around pred U ref of that frame; otherwise a fixed (x0, y0, w, h)."""
    H, W = ref.shape[1:]
    cell_h = int(round(cell_w * 9 / 16))
    thick = max(1, W // 640)
    sheet = np.full((24 + len(rows) * (cell_h + 4), 150 + len(cols) * (cell_w + 4), 3), 255, np.uint8)
    cv2.putText(sheet, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for r, name in enumerate(rows):
        yy = 24 + r * (cell_h + 4)
        cv2.putText(sheet, name, (4, yy + cell_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        for c, t in enumerate(cols):
            pred = None if name == "reference" else preds[name][t]
            img = paint(frames[t], pred, ref[t], thick=thick)
            x0, y0, w, h = roi if roi is not None else frame_roi(pred, ref[t], W, H, min_w=W // 8)
            cell = cv2.resize(img[y0:y0 + h, x0:x0 + w], (cell_w, cell_h), interpolation=cv2.INTER_AREA if w > cell_w else cv2.INTER_NEAREST)
            jt = "" if (Js is None or name == "reference" or np.isnan(Js[name][t])) else f" J={Js[name][t]:.2f}"
            label(cell, f"f{t}{jt}" + ("" if roi is not None else f" [{x0},{y0} {w}px]"), scale=0.38)
            xx = 150 + c * (cell_w + 4)
            sheet[yy:yy + cell_h, xx:xx + cell_w] = cell
    cv2.imwrite(path, sheet)


def timeline(per, cams, entries, path, res, gvars):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, len(cams), figsize=(6.2 * len(cams), 8.4), sharex=True, constrained_layout=True)
    rows = [("J", "IoU vs reference (per frame)"), ("area", "mask area, % of image"), ("score", "object-score logit")]
    for ci, cam in enumerate(cams):
        for ri, (key, ylab) in enumerate(rows):
            ax = axes[ri, ci]
            if key == "area":
                ax.plot(per[cam]["ref_area"] * 100, color="#555555", lw=2, ls="--", label="reference (SAM3 box+click)")
            for k, v in enumerate(gvars):
                y = per[cam][v][key] * (100 if key == "area" else 1)
                ax.plot(np.arange(len(y)), y, color=SLOTS[k], lw=2 if v == "auto_f0_gripper" else 1.3, label=v)
            ax.axvline(entries[cam], color="#888888", lw=1, ls=":")
            if key == "score":
                ax.axhline(0, color="#bbbbbb", lw=0.8)
            ax.grid(alpha=0.25, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylabel(ylab)
            if ri == 0:
                ax.set_title(f"{cam}  ({res}; dotted = reference entry f{entries[cam]})", fontsize=10)
            if ri == 2:
                ax.set_xlabel("frame")
    axes[1, 0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="RAIL+80edfcb1+2023-07-14-14h-28m-45s")
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--raw", action="store_true", help="score the un-refined (pre guided-filter) masks instead")
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--variants", nargs="+", default=None, help="subset (default: every variant present on disk)")
    args = ap.parse_args()
    ep, res = args.episode, args.res
    W, H = [int(v) for v in res.split("x")]
    d = os.path.join(args.out_root, res, ep)
    sd = os.path.join(d, "score" + ("_raw" if args.raw else ""))
    os.makedirs(sd, exist_ok=True)
    meta = json.load(open(os.path.join(RAW_ROOT, ep, f"metadata_{ep}.json")))
    cams = {c: meta[f"{c}_cam_serial"] for c in ("ext1", "ext2")}
    bm = vos_bench()
    key = "raw" if args.raw else "refined"
    ser1 = cams["ext1"]
    variants = args.variants or [v for v in VARIANTS if os.path.isfile(os.path.join(d, f"ext1_{ser1}__robotseg_{v}_raw.npz"))]
    summary, per, frames_by_cam, preds_by_cam, refs_by_cam, entries = {}, {}, {}, {}, {}, {}
    csv_rows = []
    for cam, ser in cams.items():
        ref720 = load_contract(os.path.join(REF_DIR, ep, f"{cam}_{ser}{REF_SUFFIX}"))
        T = len(ref720)
        ref = ref720 if (W, H) == (1280, 720) else np.stack([cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) >= 0.5 for m in ref720])
        arm720 = load_contract(os.path.join(REF_DIR, ep, f"{cam}_{ser}{ARM_SUFFIX}"))
        arm = arm720 if (W, H) == (1280, 720) else np.stack([cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) >= 0.5 for m in arm720])
        uv, inview = ssg.gt_projections(ep, ser)
        frames = [cv2.imread(os.path.join(d, "frames", cam, f"{t:05d}.jpg")) for t in range(T)]
        frames_by_cam[cam], refs_by_cam[cam] = frames, ref
        entries[cam] = int(np.argmax(ref.reshape(T, -1).any(1)))
        summary[cam] = dict(serial=ser, reference_kin=ssg.score_masks(ref720, uv, inview))
        per[cam] = dict(ref_area=ref.reshape(T, -1).mean(1))
        preds_by_cam[cam] = {}
        for v in variants:
            z = np.load(os.path.join(d, f"{cam}_{ser}__robotseg_{v}_raw.npz"))
            pred = unpack(z, key)
            assert pred.shape == ref.shape, (pred.shape, ref.shape)
            preds_by_cam[cam][v] = pred
            s, J, F, pa, ra = score_vs_ref(pred, ref, bm)
            m720 = load_contract(os.path.join(d, f"{cam}_{ser}__robotseg_{v}_masks.npz"))
            if args.raw:   # contract file holds refined masks; rebuild from raw for the kinematic score
                m720 = pred if (W, H) == (1280, 720) else np.stack([cv2.resize(m.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST) > 0 for m in pred])
            s["kin"] = ssg.score_masks(m720, uv, inview)
            if not v.endswith("_gripper"):
                s["ref_cov"] = float((pred & ref).sum() / max(ref.sum(), 1))
                s["IoU_vs_sam3_arm_pooled"] = float((pred & arm).sum() / max((pred | arm).sum(), 1))
            s["obj_score_pos_frames"] = int(np.nansum(z["obj_score"] > 0))
            summary[cam][v] = s
            per[cam][v] = dict(J=np.where(ref.reshape(T, -1).any(1) | pred.reshape(T, -1).any(1), J, np.nan), area=pa, score=z["obj_score"].astype(float))
            for t in range(T):
                csv_rows.append([cam, v, t, round(J[t], 4), round(F[t], 4), int(pred[t].any()), int(ref[t].any()), round(pa[t], 6), round(ra[t], 6), round(float(z["obj_score"][t]), 3)])
            print(f"{res} {cam} {v:20s} JF_vrs {s['JF_vrs']:.3f} J_ref {s['J_ref']:.3f} TP {s['TP']} FP {s['FP']} FN {s['FN']} "
                  f"first_pred {s['first_pred']} entry {s['ref_entry']} prec {s['pix_prec']:.2f} rec {s['pix_rec']:.2f} "
                  f"area_ratio {s['area_ratio_med']} hit@25 {s['kin']['hit']} far {s['kin']['miss_far']} spill {s['kin']['spill']}"
                  + (f" ref_cov {s['ref_cov']:.2f} IoU_sam3arm {s['IoU_vs_sam3_arm_pooled']:.2f}" if 'ref_cov' in s else ""), flush=True)
        rk = summary[cam]["reference_kin"]
        print(f"{res} {cam} {'REFERENCE (SAM3)':20s} hit@25 {rk['hit']} far {rk['miss_far']} spill {rk['spill']} present_inview {rk['present_inview']}")

    with open(os.path.join(sd, "per_frame.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cam", "variant", "frame", "J", "F", "pred_present", "ref_present", "pred_area_frac", "ref_area_frac", "obj_score"])
        w.writerows(csv_rows)
    json.dump(dict(episode=ep, res=res, masks=key, summary=summary), open(os.path.join(sd, "summary.json"), "w"), indent=1)

    lines = [f"# RobotSeg out of the box, {ep}, {res}, {key} masks", "",
             "| cam | variant | J&F (VRS) | J (ref frames) | TP | FP | FN | first pred / ref entry | pix prec | pix rec | area ratio | hit@25 | miss_far | spill |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cam in cams:
        for v in variants:
            s = summary[cam][v]
            k = s["kin"]
            fmt = lambda x, n=2: "-" if x is None else f"{x:.{n}f}"
            lines.append(f"| {cam} | {v} | {s['JF_vrs']:.3f} | {s['J_ref']:.3f} | {s['TP']} | {s['FP']} | {s['FN']} | {s['first_pred']} / {s['ref_entry']} | "
                         f"{s['pix_prec']:.2f} | {s['pix_rec']:.2f} | {fmt(s['area_ratio_med'])} | {fmt(k['hit'])} | {fmt(k['miss_far'])} | {fmt(k['spill'])} |")
        rk = summary[cam]["reference_kin"]
        lines.append(f"| {cam} | reference (SAM3 box+click) | 1 | 1 | | | | - / {entries[cam]} | 1 | 1 | 1 | {rk['hit']:.2f} | {rk['miss_far']:.2f} | {rk['spill']:.2f} |")
    open(os.path.join(sd, "summary.md"), "w").write("\n".join(lines) + "\n")

    timeline(per, list(cams), entries, os.path.join(sd, "timeline.png"), res, [v for v in GRIPPER_VARIANTS if v in variants])
    for cam in cams:
        T = len(refs_by_cam[cam])
        e = entries[cam]
        cols = sorted({0, max(e - 8, 0), e, e + 4, e + 12, (e + T) // 2, int(0.85 * T), T - 1})
        rows = variants + ["reference"]
        Js = {v: per[cam][v]["J"] for v in variants}
        contact_sheet(frames_by_cam[cam], preds_by_cam[cam], refs_by_cam[cam], rows, cols, None, 220, os.path.join(sd, f"contact_{cam}_crop.png"),
                      f"{cam} {res} per-frame crop around pred U ref  magenta = RobotSeg ({key}), yellow outline = SAM3 reference", Js)
        contact_sheet(frames_by_cam[cam], preds_by_cam[cam], refs_by_cam[cam], rows, cols, (0, 0, W, H), 300, os.path.join(sd, f"contact_{cam}_full.png"),
                      f"{cam} {res} full frame  magenta = RobotSeg ({key}) with bbox, yellow outline = SAM3 reference", Js)
    if not args.no_video:
        pw, ph = 640, 360
        for v in variants:
            wr = FfmpegWriter(os.path.join(sd, f"overlay_{v}.mp4"), 2 * pw, ph, 15)
            T = len(refs_by_cam["ext1"])
            for t in range(T):
                panels = []
                for cam in cams:
                    img = paint(frames_by_cam[cam][t], preds_by_cam[cam][v][t], refs_by_cam[cam][t], thick=max(1, W // 640))
                    img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA if W > pw else cv2.INTER_LINEAR)
                    J = per[cam][v]["J"][t]
                    label(img, f"{cam} f{t} RobotSeg {v} ({res})  J={'-' if np.isnan(J) else f'{J:.2f}'}  magenta=pred yellow=SAM3 ref")
                    panels.append(img)
                wr.write(np.concatenate(panels, 1))
            wr.close()
    print(f"wrote {sd}")


if __name__ == "__main__":
    main()
