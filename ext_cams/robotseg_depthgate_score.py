#!/usr/bin/env python
"""Score the depth-gated RobotSeg variants (robotseg_depthgate.py) next to the earlier seed-study methods on the same
scenes, with the seed study's own proxy code (seedstudy_summary.proxies_episode; kinematic GT lens for SCORING only).

Per method (scene-level numbers pool both cameras, as in the seed study):
  lens_px_med      median over scenes of the scene-median distance (px, 1280x720) from the projected GT lens to the
                   nearest mask pixel (frames with mask AND lens in view)
  <=10px / >100px  scenes with that scene median
  agree            mean over scenes of in-view agreement: (mask non-empty) == (lens in view)
  recall_iv        pooled fraction of lens-in-view frames that have a mask
  fp_oov           pooled fraction of lens-out-of-view frames that still have a mask (absence errors; approximate,
                   the gripper can be partly visible while the lens is not)
  iou_med          median over scenes of the scene-median IoU vs the gtbox_sam3 reference (privileged ceiling masks)
Methods missing a scene are reported over the scenes they have (n column).

    OMP_NUM_THREADS=1 python robotseg_depthgate_score.py [--episodes FILE] [--res 320x180 1280x720]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seedstudy_summary as ss  # noqa: E402
from robotseg_depthgate import OUT_ROOT  # noqa: E402

VARIANTS = ["plain_auto", "plain_image", "plain_auto_dselF", "plain_auto_dselK", "plain_image_dselF", "plain_image_dselK",
            "blankF_auto", "blankF_image", "blankF_auto_dselF", "blankF_image_dselF",
            "blankK_auto", "blankK_image", "blankK_auto_dselK", "blankK_image_dselK"]
SEED = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams"
BASELINES = [("motion_only", f"{SEED}/seedstudy/motion_only"), ("motion_sam3", f"{SEED}/seedstudy/motion_sam3"),
             ("text_arm_sam3", f"{SEED}/seedstudy/text_arm_sam3"), ("verified_motion", f"{SEED}/seedstudy2/verified_motion_v2"),
             ("gtbox_sam3", f"{SEED}/seedstudy/gtbox_sam3")]
REF = ("gtbox_sam3", f"{SEED}/seedstudy/gtbox_sam3")


def proxies(ep, method, root, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    c = os.path.join(cache_dir, f"{ep}__{method}.json")
    if os.path.isfile(c):
        return json.load(open(c))
    r = ss.proxies_episode(ep, method, root, REF[0], REF[1])
    json.dump(r, open(c, "w"))
    return r


def aggregate(rows):
    rows = [r for r in rows if r and all(r["cams"].get(c, {}).get("exists") for c in ss.CAMS)]
    if not rows:
        return None
    lp = [r["lens_to_mask_px_median"] for r in rows if r["lens_to_mask_px_median"] is not None]
    cams = [r["cams"][c] for r in rows for c in ss.CAMS]
    iv = sum(c["gt_in_view"] for c in cams)
    oov = sum(c["T"] - c["gt_in_view"] for c in cams)
    return dict(n=len(rows), lens_px_med=float(np.median(lp)) if lp else None, le10=sum(x <= 10 for x in lp), gt100=sum(x > 100 for x in lp),
                no_mask_scenes=len(rows) - len(lp), agree=float(np.mean([r["inview_agreement"] for r in rows])),
                recall_iv=sum(c["present_and_in_view"] for c in cams) / max(iv, 1),
                fp_oov=sum(c["present_when_out_of_view"] for c in cams) / max(oov, 1),
                iou_med=float(np.median([r["iou_vs_ref_median"] for r in rows if r["iou_vs_ref_median"] not in (None, -1)] or [np.nan])))


def rail_vs_sam3(ep, resolutions):
    """Per (res, variant): J_ref / J&F / FP vs the approved SAM3 box+click masks, scored at 1280x720 on the contract masks."""
    import cv2  # noqa: F401  (robotseg_score imports cv2 too)
    from robotseg_score import score_vs_ref, vos_bench
    from robotseg_gripper_ext import REF_DIR, REF_SUFFIX, load_contract
    bm = vos_bench()
    meta = json.load(open(os.path.join(ss.RAW_ROOT, ep, f"metadata_{ep}.json")))
    refs = {c: load_contract(os.path.join(REF_DIR, ep, f"{c}_{meta[c + '_cam_serial']}{REF_SUFFIX}")) for c in ss.CAMS}
    out = {}
    for res in resolutions:
        for v in VARIANTS:
            r = {}
            for c in ss.CAMS:
                p = os.path.join(OUT_ROOT, res, ep, f"{c}_{meta[c + '_cam_serial']}__rsdg_{v}_masks.npz")
                if not os.path.isfile(p):
                    break
                sc = score_vs_ref(load_contract(p), refs[c], bm)[0]
                r[c] = {k: sc[k] for k in ("J_ref", "JF_vrs", "FP", "FN", "pix_prec", "pix_rec")}
            if len(r) == 2:
                out[(res, v)] = r
    return out


def fmt(a):
    if a is None:
        return "| - | - | - | - | - | - | - | - |"
    f = lambda x, n=1: "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"
    return (f"| {a['n']} | {f(a['lens_px_med'])} | {a['le10']} | {a['gt100']} | {f(a['agree'], 3)} | {f(a['recall_iv'], 3)} | "
            f"{f(a['fp_oov'], 3)} | {f(a['iou_med'], 3)} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=ss.SMOKE13)
    ap.add_argument("--res", nargs="+", default=["320x180", "1280x720"])
    args = ap.parse_args()
    eps = [l.strip() for l in open(args.episodes) if l.strip()]
    hdr = ("| method | n | lens_px_med | <=10px | >100px | agree | recall_iv | fp_oov | iou_med |\n"
           "|---|---|---|---|---|---|---|---|---|")
    lines = [f"# Depth-gated RobotSeg vs seed-study methods ({len(eps)} episodes listed in {os.path.basename(args.episodes)})", "",
             "Scored with seedstudy_summary.proxies_episode (kinematic GT lens for scoring only; IoU vs gtbox_sam3 = privileged "
             "reference). RobotSeg rows scored on their 1280x720 contract masks (320x180 runs upsampled x4 nearest).", ""]
    out = {}
    lines += ["## Earlier seed-study methods (1280x720)", "", hdr]
    for name, root in BASELINES:
        rows = [proxies(ep, name, root, os.path.join(OUT_ROOT, "_proxies_baselines")) for ep in eps if os.path.isdir(os.path.join(root, ep))]
        a = aggregate(rows)
        out[name] = a
        lines.append(f"| {name}{' (REFERENCE)' if name == REF[0] else ''} |" + fmt(a)[1:])
    for res in args.res:
        root = os.path.join(OUT_ROOT, res)
        lines += ["", f"## RobotSeg, gripper, processed at {res}", "", hdr]
        for v in VARIANTS:
            rows = [proxies(ep, f"rsdg_{v}", root, os.path.join(root, "_proxies")) for ep in eps if os.path.isfile(os.path.join(root, ep, "run_info.json"))]
            a = aggregate(rows)
            out[f"{res}/{v}"] = a
            lines.append(f"| {v} |" + fmt(a)[1:])
    rail = "RAIL+80edfcb1+2023-07-14-14h-28m-45s"
    if rail in eps:
        lines += ["", f"## {rail} vs the SAM3 box+click reference (human-prompted; J over reference-present frames, "
                  "J&F = RobotSeg benchmark protocol, FP = mask on a reference-empty frame; gripper enters ext1 f27 / ext2 f37)", "",
                  "| res | variant | J_ref ext1 | J_ref ext2 | J&F ext1 | J&F ext2 | FP frames ext1 | FP frames ext2 |", "|---|---|---|---|---|---|---|---|"]
        rows_rail = rail_vs_sam3(rail, args.res)
        out["rail_vs_sam3"] = rows_rail
        for (res, v), r in rows_rail.items():
            lines.append(f"| {res} | {v} | {r['ext1']['J_ref']:.3f} | {r['ext2']['J_ref']:.3f} | {r['ext1']['JF_vrs']:.3f} | {r['ext2']['JF_vrs']:.3f} | "
                         f"{r['ext1']['FP']} | {r['ext2']['FP']} |")
    md = "\n".join(lines) + "\n"
    open(os.path.join(OUT_ROOT, "depthgate_vs_seedstudy.md"), "w").write(md)
    out["rail_vs_sam3"] = {f"{k[0]}/{k[1]}": v for k, v in out.get("rail_vs_sam3", {}).items()}
    json.dump(out, open(os.path.join(OUT_ROOT, "depthgate_vs_seedstudy.json"), "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()
