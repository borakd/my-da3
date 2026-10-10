#!/usr/bin/env python
"""Summarise rig_v0 runs (one row per TAG): coverage, per-frame position / rotation error (median of scene medians),
Sim3 ATE and RPE vs the store GT, CUT3R finetuned on the same frames, and the linking diagnostics.
A TAG may also be an absolute run directory (e.g. rig_trackon/v2_cut, rig_trackon/match_lkr15/eval) for comparison.
    python rig_v0_summary.py [--list SCENES.txt] TAG [TAG ...]"""
import glob, json, os, sys
import numpy as np, pandas as pd
V = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/rig_v0"
args = sys.argv[1:]; keep = None
if args and args[0] == "--list": keep = set(l.strip() for l in open(args[1]) if l.strip()); args = args[2:]
rows = []
for tag in args:
    root = tag if os.path.isabs(tag) else f"{V}/{tag}"
    recs = [json.load(open(f)) for f in glob.glob(f"{root}/*/record.json")]
    if keep is not None: recs = [r for r in recs if r["episode"] in keep]
    d = pd.DataFrame([dict(ep=r["episode"], fail=r.get("failure_reason", ""), **r["windows"][0], **{f"link_{k}": v for k, v in (r.get("link") or {}).items()
                                                                                                    if not isinstance(v, dict)}) for r in recs])
    if d.empty: continue
    ok = d[(d.fail == "") & (d.n_anchored >= 2)] if "n_anchored" in d else d.iloc[:0]
    med = lambda c: float(np.nanmedian(ok[c])) if c in ok and len(ok) else np.nan
    p90 = lambda c: float(np.nanpercentile(ok[c], 90)) if c in ok and len(ok) else np.nan
    mean = lambda c: float(np.nanmean(ok[c])) if c in ok and len(ok) else np.nan
    rows.append(dict(tag=tag if not os.path.isabs(tag) else "/".join(tag.rstrip("/").split("/")[-2:]), scenes=len(d), evaluated=len(ok), coverage=float(np.nanmedian(ok.n_anchored / ok.window_len)) if len(ok) else np.nan,
                     pos_cm=med("pos_err_median_cm"), pos_p90=p90("pos_err_median_cm"), rot_deg=med("rot_err_median_deg"), rot_p90=p90("rot_err_median_deg"),
                     ate=med("rig_ate"), rpe_rot_mean=mean("rig_rpe_rot"), cut3r_rpe_rot_mean=mean("cut3r_rpe_rot_same_frames"),
                     interval_cm=med("link_interval_cm_median"), alpha_true=med("link_alpha_true_median"), in_interval=med("link_true_in_interval_frac"),
                     depth_init_mm=med("link_depth_err_init_mm"), depth_end_mm=med("link_depth_err_end_mm"),
                     fails=d[d.fail != ""].fail.str.replace(r"\(\d+\)", "", regex=True).value_counts().to_dict()))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(rows).round(3).to_string(index=False))
