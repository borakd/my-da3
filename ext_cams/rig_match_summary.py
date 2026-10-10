#!/usr/bin/env python
"""Compare a rig_match_track + rig_trackon_eval --paired run (match_<TAG>) with the Track-On v2 rig (v2_cut) on the same scenes.
    python rig_match_summary.py TAG"""
import glob, json, os, sys
import numpy as np, pandas as pd
R = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/rig_trackon"; tag = sys.argv[1]
def load(run, want_match=False):
    rows = []
    for f in glob.glob(f"{run}/*/record.json"):
        r = json.load(open(f)); w = r["windows"][0]; ep = r["episode"]
        d = r.get("diag_drift_mm") or {}
        row = dict(ep=ep, fail=w.get("skipped", ""), pairs=r.get("pairs"), drift=d.get("median"), drift_gt5=d.get("gt5mm_frac"),
                   n_anchored=w.get("n_anchored"), window_len=w.get("window_len"), pos=w.get("pos_err_median_cm"), rot=w.get("rot_err_median_deg"),
                   pos_p90=w.get("pos_err_p90_cm"), pos_max=w.get("pos_err_max_cm"), ate=w.get("rig_ate"), cut_ate=w.get("cut3r_ate_same_frames"), inl=w.get("inliers_median"))
        if want_match:
            mf = f"{R}/match_{tag}/tracks/{ep}/match_record.json"; m = json.load(open(mf)) if os.path.isfile(mf) else {}
            row.update(n_pairs_birth=m.get("n_pairs_birth"), n_pairs_total=m.get("n_pairs_total"), pair_len=m.get("pair_len_median"), match_fail=m.get("failure_reason", ""))
        rows.append(row)
    return pd.DataFrame(rows).set_index("ep")
new, old = load(f"{R}/match_{tag}/eval", True), load(f"{R}/v2_cut")
F = []
def pr(*a): s = " ".join(str(x) for x in a); print(s); F.append(s)
pr(f"# fix 1 ({tag}: guided DISK matching at the birth frame + {tag} tracking + per-frame epipolar gate) vs Track-On v2 rig, same scenes\n")
pr(f"scenes with a record: new {len(new)}, v2 {len(old)}")
ok_n, ok_o = new[(new.fail == "") & (new.n_anchored >= 2)], old[(old.fail == "") & (old.n_anchored >= 2)]
pr(f"evaluated (>= 2 solved frames): new {len(ok_n)} ({len(ok_n)/len(new):.1%}), v2 {len(ok_o)} ({len(ok_o)/len(old):.1%})")
pr("new failures:", new[new.fail != ""].fail.str.replace(r"\(\d+\)", "", regex=True).str.strip().value_counts().to_dict())
pr(f"birth-frame gated pairs: median {new.n_pairs_birth.median()}, scenes with < 3: {(new.n_pairs_birth.fillna(0) < 3).sum()}")
q = lambda s, p: float(np.nanpercentile(s, p))
for name, d in (("new", ok_n), ("v2", ok_o)):
    pr(f"{name}: pairs median {d.pairs.median()}, drift mm median {d.drift.median():.1f} (pairs > 5 mm {d.drift_gt5.median():.2f}), solved frames/window median {(d.n_anchored/d.window_len).median():.2f}, "
       f"per-scene pos median {d.pos.median():.2f} cm (p90 {q(d.pos,90):.2f}), rot median {d.rot.median():.2f} deg (p90 {q(d.rot,90):.1f}), ATE median {d.ate.median():.4f}, scenes with max err > 5 cm {(d.pos_max > 5).mean():.1%}, rig ATE < CUT3R ft {(d.ate < d.cut_ate).mean():.1%}")
both = ok_n.join(ok_o, lsuffix="_new", rsuffix="_v2", how="inner")
pr(f"\n## head-to-head on the {len(both)} scenes both solved")
pr(f"| | new | v2 |\n|---|---|---|")
for k, lab in (("pos", "per-frame position err, median of scene medians (cm)"), ("rot", "rotation err (deg)"), ("ate", "window ATE (Sim3)"), ("drift", "point drift in gripper frame (mm)"), ("inl", "RANSAC inliers")):
    pr(f"| {lab} | {both[k+'_new'].median():.3f} | {both[k+'_v2'].median():.3f} |")
pr(f"| solved frames per window | {(both.n_anchored_new/both.window_len_new).median():.2f} | {(both.n_anchored_v2/both.window_len_v2).median():.2f} |")
pr(f"new better per scene: pos {(both.pos_new < both.pos_v2).mean():.1%}, rot {(both.rot_new < both.rot_v2).mean():.1%}, ATE {(both.ate_new < both.ate_v2).mean():.1%}; new solves more frames in {(both.n_anchored_new > both.n_anchored_v2).mean():.1%}")
v2fail = old[old.fail != ""].index; pr(f"\nscenes v2 failed ({len(v2fail)}): new solves {int(ok_n.index.isin(v2fail).sum())}")
per_lab = ok_n.assign(lab=[e.split("+")[0] for e in ok_n.index]).groupby("lab").agg(n=("pos", "size"), pos=("pos", "median"), rot=("rot", "median"), drift=("drift", "median"))
pr("\n## new, per lab (scenes, pos cm, rot deg, drift mm)\n" + per_lab.round(2).to_string())
open(f"{R}/match_{tag}/SUMMARY.md", "w").write("\n".join(F) + "\n"); new.join(old, lsuffix="_new", rsuffix="_v2", how="outer").to_csv(f"{R}/match_{tag}/per_scene_vs_v2.csv")
