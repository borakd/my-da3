#!/usr/bin/env python
"""Before/after table: rig_track v1 (seedstudy/<method>/EP/rig/record.json) vs rig_track2 (seedstudy2/<run>/EP/record.json)
on the smoke-13 scenes; also verifies the v2 products (record status, anchors.npz frame counts). Prints markdown + JSON.
    python seedstudy2_table.py [--v1_root ...] [--v2_root ...] [--json OUT]"""
import argparse, json, os, numpy as np
SM = "/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt"
ap = argparse.ArgumentParser()
ap.add_argument("--v1_root", default="/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy/gtbox_sam3")
ap.add_argument("--v2_root", default="/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/seedstudy2/tracker2_on_reference")
ap.add_argument("--json", default=None)
a = ap.parse_args()
def f(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
rows, out = [], []
hdr = "| scene | T | mask both | anchored v1 | anchored v2 | v2 / mask both | segs | links | pos err v2 cm (v1) | rot err v2 deg (v1) | centroid err cm | rig ATE centroid | rig ATE lens (v1) | CUT3R ATE same | max jump cm/deg | unsolved (no_mask/no_pairs/lt3/ransac/warmup/degenerate/jump) | s |"
print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
for ep in open(SM).read().split():
    r1 = json.load(open(f"{a.v1_root}/{ep}/rig/record.json"))
    p2 = f"{a.v2_root}/{ep}/record.json"
    if not os.path.isfile(p2):
        print(f"| {ep} | MISSING v2 record |"); continue
    r2 = json.load(open(p2))
    prod = "ok"
    if r2["status"] == "ok":
        an = np.load(f"{a.v2_root}/{ep}/anchors.npz")
        if len(an["frames"]) != r2["n_anchored"] or an["R"].shape != (r2["n_anchored"], 3, 3) or an["centroid"].shape != (r2["n_anchored"], 3):
            prod = "PRODUCT MISMATCH"
    u = r2.get("unsolved_reasons") or {}
    rec = dict(episode=ep, n_frames=r2.get("n_frames"), mask_both=r2.get("mask_frames_both"), anchored_v1=r1.get("n_anchored"), anchored_v2=r2.get("n_anchored"),
               frac_v2=r2.get("anchored_frac_mask_both"), segments_v2=r2.get("n_segments"), links=r2.get("n_linked"),
               pos_err_v2_cm=r2.get("pos_err_median_cm"), pos_err_v1_cm=r1.get("pos_err_median_cm"), rot_err_v2_deg=r2.get("rot_err_median_deg"), rot_err_v1_deg=r1.get("rot_err_median_deg"),
               centroid_err_cm=r2.get("centroid_err_median_cm"), rig_ate_centroid=r2.get("rig_ate_centroid"), rig_ate_lens=r2.get("rig_ate_lens"), rig_ate_v1=r1.get("rig_ate"),
               cut3r_ate=r2.get("cut3r_ate_same_frames"), max_jump_cm=r2.get("max_jump_cm"), max_jump_deg=r2.get("max_jump_deg"), unsolved=u, elapsed_s=r2.get("elapsed_s"),
               status=r2["status"], failure_reason=r2.get("failure_reason"), product_check=prod)
    out.append(rec)
    short = ep.replace("AUTOLab+", "AUTO+").replace("0d4edc83", "0d4e").replace("44bb9c36", "44bb").replace("80edfcb1", "80ed")[:26]
    print(f"| {short} | {rec['n_frames']} | {rec['mask_both']} | {rec['anchored_v1']} | {rec['anchored_v2']} | {f(rec['frac_v2'])} | {rec['segments_v2']} | {rec['links']} | "
          f"{f(rec['pos_err_v2_cm'])} ({f(rec['pos_err_v1_cm'])}) | {f(rec['rot_err_v2_deg'],1)} ({f(rec['rot_err_v1_deg'],1)}) | {f(rec['centroid_err_cm'])} | {f(rec['rig_ate_centroid'],4)} | "
          f"{f(rec['rig_ate_lens'],4)} ({f(rec['rig_ate_v1'],4)}) | {f(rec['cut3r_ate'],4)} | {f(rec['max_jump_cm'],1)}/{f(rec['max_jump_deg'],1)} | "
          f"{u.get('no_mask')}/{u.get('no_pairs')}/{u.get('lt3_points')}/{u.get('ransac_fail')}/{u.get('warmup')}/{u.get('degenerate')}/{u.get('jump')} | {f(rec['elapsed_s'],0)} {'' if prod == 'ok' else prod} |")
ok = [r for r in out if r["status"] == "ok"]
long5 = [r for r in ok if "44bb9c36" in r["episode"] and r["n_frames"] > 400]
print(f"\nscenes ok {len(ok)}/13; anchored total v1 {sum(r['anchored_v1'] for r in out)} -> v2 {sum(r['anchored_v2'] for r in ok)}; "
      f"median pos err v2 {np.median([r['pos_err_v2_cm'] for r in ok]):.2f} cm (v1 {np.median([r['pos_err_v1_cm'] for r in out]):.2f}); "
      f"median rot err v2 {np.median([r['rot_err_v2_deg'] for r in ok]):.2f} deg (v1 {np.median([r['rot_err_v1_deg'] for r in out]):.2f}); "
      f"five long 44bb scenes: anchored/mask_both v2 = {[round(r['frac_v2'],2) for r in long5]} (v1 {[round(r['anchored_v1']/r['mask_both'],3) for r in long5]})")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
