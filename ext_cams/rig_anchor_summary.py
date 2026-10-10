#!/usr/bin/env python
"""Scorecard for rig arms on a common scene list. Arms: anchor dirs (rig_anchor.py) and rig_trackon_eval dirs (match_lkr15,
match_ct, v2_cut, ...); every dir holds <ep>/record.json and <ep>/per_frame.csv (+ anchors.npz).
    python rig_anchor_summary.py --eps LIST --arms name=DIR [name=DIR ...] [--out FILE.md]
Per arm: scenes evaluated, solved frames per window (median), median of scene-median pos / rot error, pooled per-frame rot
error by frames since the reference, consecutive-frame increment errors (rotation, deg), ATE median. Head-to-head on the
scenes every arm solved."""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_track as rt

ap = argparse.ArgumentParser(); ap.add_argument("--eps", required=True); ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--out")
ap.add_argument("--inc", action="store_true", help="recompute increment errors from anchors.npz + store GT for arms that lack them")
a = ap.parse_args(); eps = [l.strip() for l in open(a.eps) if l.strip()]
BINS = [0, 5, 20, 50, 100, 10000]; LAB = ["1-5", "6-20", "21-50", "51-100", ">100"]
out = []
def pr(*x): s = " ".join(str(v) for v in x); print(s); out.append(s)

def load(d):
    rows, pf = [], []
    for ep in eps:
        f = f"{d}/{ep}/record.json"
        if not os.path.isfile(f): rows.append(dict(ep=ep, fail="missing")); continue
        r = json.load(open(f)); w = r["windows"][0]
        row = dict(ep=ep, fail=w.get("skipped", "") or r.get("failure_reason", "") or "", n_anchored=w.get("n_anchored"), window_len=w.get("window_len"),
                   t_ref=w.get("t_ref"), pos=w.get("pos_err_median_cm"), rot=w.get("rot_err_median_deg"), ate=w.get("rig_ate"), rpe_rot=w.get("rig_rpe_rot"),
                   cut_ate=w.get("cut3r_ate_same_frames"), cut_rpe_rot=w.get("cut3r_rpe_rot_same_frames"), rpe_t=w.get("rig_rpe_t"), cut_rpe_t=w.get("cut3r_rpe_t_same_frames"), inc_rot=w.get("inc_rot_err_median_deg"), inc_rot_p90=w.get("inc_rot_err_p90_deg"))
        rows.append(row)
        p = f"{d}/{ep}/per_frame.csv"
        if os.path.isfile(p) and row["t_ref"] is not None:
            q = pd.read_csv(p)
            if "solved" in q: q = q[q.solved.astype(bool)]
            elif "n_inl" in q: q = q[q.n_inl >= 3]
            q = q[q.rot_err_deg.notna()]; q = q.assign(ep=ep, dt=q.frame - row["t_ref"]); pf.append(q[["ep", "frame", "dt", "pos_err_cm", "rot_err_deg"] + [c for c in ("inc_rot_err_deg",) if c in q]])
    return pd.DataFrame(rows).set_index("ep"), (pd.concat(pf) if pf else pd.DataFrame())

def inc_from_anchors(d, ep):
    f = f"{d}/{ep}/anchors.npz"
    if not os.path.isfile(f): return []
    z = np.load(f); fr = z["frames"]; Pp = z["poses"]; gt = rt.load_poses(f"{rt.STORE_ROOT}/{ep}/dense/cam", int(fr.max()) + 1, "gt"); o = []
    for i in range(1, len(fr)):
        if fr[i] == fr[i - 1] + 1 and fr[i] in gt and fr[i - 1] in gt:
            dp = np.linalg.inv(Pp[i - 1]) @ Pp[i]; dg = np.linalg.inv(gt[fr[i - 1]]) @ gt[fr[i]]; o.append((fr[i], rt.rot_angle_deg((np.linalg.inv(dg) @ dp)[:3, :3])))
    return o

res = {}
for spec in a.arms:
    name, d = spec.split("=", 1); df, pf = load(d)
    if (a.inc or df.inc_rot.isna().all()) and len(pf):
        inc = {ep: dict(inc_from_anchors(d, ep)) for ep in df.index[df.fail == ""]}
        df["inc_rot"] = [np.median(list(inc[e].values())) if inc.get(e) else np.nan for e in df.index]
        df["inc_rot_p90"] = [np.percentile(list(inc[e].values()), 90) if inc.get(e) else np.nan for e in df.index]
        pf["inc_rot_err_deg"] = [inc.get(e, {}).get(f, np.nan) for e, f in zip(pf.ep, pf.frame)]
    res[name] = (df, pf)

pr(f"# rig scorecard: {len(eps)} scenes ({a.eps})\n")
pr("| arm | evaluated | solved frames / window (median) | pos cm | rot deg | inc rot deg (median of scene medians) | inc rot p90 | ATE | ATE < CUT3R ft |")
pr("|---|---|---|---|---|---|---|---|---|")
for name, (df, pf) in res.items():
    ok = df[(df.fail == "") & (df.n_anchored >= 2)]
    pr(f"| {name} | {len(ok)} | {(ok.n_anchored / ok.window_len).median():.2f} | {ok.pos.median():.2f} | {ok.rot.median():.2f} | {ok.inc_rot.median():.2f} | {ok.inc_rot_p90.median():.1f} | {ok.ate.median():.4f} | {(ok.ate < ok.cut_ate).mean():.0%} |")
pr("\n## trajectory metrics on each arm's solved frames vs the CUT3R finetune (augfull_lr1e5) on the SAME frames (medians over scenes; win = rig better)")
pr("| arm | rig RPE_rot | CUT3R RPE_rot | win | rig RPE_t cm | CUT3R RPE_t cm | win | rig ATE | CUT3R ATE | win |"); pr("|---|---|---|---|---|---|---|---|---|---|")
for name, (df, pf) in res.items():
    ok = df[(df.fail == "") & (df.n_anchored >= 2)]
    pr(f"| {name} | {ok.rpe_rot.median():.2f} | {ok.cut_rpe_rot.median():.2f} | {(ok.rpe_rot < ok.cut_rpe_rot).mean():.0%} | {ok.rpe_t.median()*100:.2f} | {ok.cut_rpe_t.median()*100:.2f} | {(ok.rpe_t < ok.cut_rpe_t).mean():.0%} | {ok.ate.median():.4f} | {ok.cut_ate.median():.4f} | {(ok.ate < ok.cut_ate).mean():.0%} |")
pr("\n## pooled per-frame rotation error (deg, median) by frames since the reference; solved frames per scene-window in brackets")
pr("| arm | " + " | ".join(LAB) + " |"); pr("|---|" + "---|" * len(LAB))
for name, (df, pf) in res.items():
    if not len(pf): continue
    pf = pf.assign(b=pd.cut(pf.dt, BINS, labels=LAB)); g = pf.groupby("b", observed=False)
    nsc = df.index.size
    pr(f"| {name} | " + " | ".join(f"{g.rot_err_deg.median().get(l, np.nan):.2f} [{g.size().get(l, 0) / nsc:.1f}]" for l in LAB) + " |")
pr("\n## pooled increment rotation error (deg) by frames since the reference: median / p90")
pr("| arm | " + " | ".join(LAB) + " |"); pr("|---|" + "---|" * len(LAB))
for name, (df, pf) in res.items():
    if not len(pf) or "inc_rot_err_deg" not in pf: continue
    pf = pf[pf.inc_rot_err_deg.notna()].assign(b=lambda x: pd.cut(x.dt, BINS, labels=LAB)); g = pf.groupby("b", observed=False).inc_rot_err_deg
    pr(f"| {name} | " + " | ".join(f"{g.median().get(l, np.nan):.2f} / {g.quantile(.9).get(l, np.nan):.1f}" for l in LAB) + " |")
common = None
for name, (df, pf) in res.items():
    s = set(df[(df.fail == "") & (df.n_anchored >= 2)].index); common = s if common is None else common & s
pr(f"\n## head-to-head on the {len(common)} scenes every arm solved")
pr("| arm | solved/window | pos cm | rot deg | inc rot deg | ATE |"); pr("|---|---|---|---|---|---|")
for name, (df, pf) in res.items():
    c = df.loc[sorted(common)]
    pr(f"| {name} | {(c.n_anchored / c.window_len).median():.2f} | {c.pos.median():.2f} | {c.rot.median():.2f} | {c.inc_rot.median():.2f} | {c.ate.median():.4f} |")
if a.out: open(a.out, "w").write("\n".join(out) + "\n")
