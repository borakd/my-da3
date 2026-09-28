#!/usr/bin/env python3
"""Figure + table for the per-frame write-dose oracle (perframe_dose.sbatch output).

  python eval_pipeline/perframe_dose_plot.py --backbone zs|ft

Reads $OUT/perframe_dose/<bb>/search_*.json (schedules, ATE landscapes) and the worker-scored per-frame CSVs
of every schedule arm, plus the backbone's plain run and its best 4292 constant. Writes
$OUT/perframe_dose/<bb>/perframe_dose_<bb>.{png,md}.
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", choices=["zs", "ft"], required=True)
args = ap.parse_args()
OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
ROOT = f"{OUT}/perframe_dose/{args.backbone}"
scene = open(f"{ROOT}/scene.txt").read().strip()
REF = {"zs": [("plain", "CUT3R zero-shot (plain)", f"{OUT}/cut3r_zeroshot/eval"),
              ("c10", "constant a=.10 (4292 optimum)", f"{OUT}/zs4292/fgc10/eval")],
       "ft": [("plain", "CUT3R finetuned (plain)", f"{OUT}/augfull_lr1e5/eval"),
              ("c20", "constant a=.20 (4292 optimum)", f"{OUT}/wgate4292/fgc20/eval")]}[args.backbone]
COLS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]


def load_csv(d):
    p = f"{d}/{scene}/eval_depth_pose_metrics.csv"
    if not os.path.isfile(p):
        return None
    a = np.genfromtxt(p, delimiter=",", names=True)
    return {c: np.asarray(a[c], float) for c in COLS}


def scene_row(m):
    r = {c: float(np.nanmean(m[c])) for c in ("absrel", "a1", "rpe_trans", "rpe_rot")}
    r["ate"] = float(np.sqrt(np.nanmean(m["ate"] ** 2)))  # evaluator: pose_reduce=rmse
    return r


searches = {}
for f in sorted(glob.glob(f"{ROOT}/search_*.json")):
    s = json.load(open(f)); searches[(s["site"], s["tag"])] = s

runs = [(k, lab, load_csv(d)) for k, lab, d in REF]
for (site, tag), s in sorted(searches.items()):
    for p in s.get("passes", []):
        arm = f"pf_{site}_{tag}_p{p['pass']}"
        if tag == "placebo":
            lab = f"PLACEBO per-frame {site} (grid 1-k*1e-6), pass {p['pass']}"
        else:
            init = "plain" if tag == "init1" else f"scene-best const {s['init']:g}"
            lab = f"per-frame {site}, init {init}, pass {p['pass']}"
        runs.append((arm, lab, load_csv(f"{ROOT}/{arm}/eval")))
for site in ("state", "mem"):
    s = next((v for (st, tg), v in searches.items() if st == site and tg != "placebo"), None)
    if s:
        runs.append((f"pf_{site}_constbest", f"scene-best constant {site} {s['const_best']['w']:g}",
                     load_csv(f"{ROOT}/pf_{site}_constbest/eval")))
runs = [r for r in runs if r[2] is not None]

# ---------------------------------------------------------------- table
md = [f"# Per-frame write-dose oracle: {scene} ({'zero-shot' if args.backbone == 'zs' else 'finetuned'} CUT3R)", "",
      "Scene metrics from the standard worker + evaluator (ATE = RMSE over frames after one Sim(3) fit).", "",
      "| run | AbsRel | d<1.25 | ATE | RPE trans | RPE rot | mean weight |", "|---|---|---|---|---|---|---|"]
for k, lab, m in runs:
    r = scene_row(m)
    mw = ""
    for (site, tag), s in searches.items():
        for p in s.get("passes", []):
            if k == f"pf_{site}_{tag}_p{p['pass']}":
                mw = f"{np.mean(p['schedule'][1:-1]):.3f}"
    md.append(f"| {lab} | {r['absrel']:.4f} | {r['a1']:.4f} | {r['ate']:.4f} | {r['rpe_trans']:.4f} | {r['rpe_rot']:.3f} | {mw} |")
md += ["", "Search harness (batch-1 rollouts, bit-identical to the worker; the worker numbers above are authoritative):", "",
       "| site/init | plain ATE (harness / stored) | const sweep | pass: ATE after search, full re-roll, incumbent drift (0 expected), seconds |", "|---|---|---|---|"]
for (site, tag), s in sorted(searches.items()):
    sw = " ".join(f"{float(w):.7g}:{a:.4f}" for w, a in s["const_sweep"].items())
    ps = "; ".join(f"p{p['pass']}: {p['best_after_t'][-1]:.4f}, {p['final_ate_batch1']:.4f}, {p['drift_maxabs']:.1e}, {p['seconds']:.0f}"
                   for p in s.get("passes", []))
    md.append(f"| {site}/{tag} (init {s['init']:.7g}) | {s['plain_ate_harness']:.5f} / {s.get('plain_ate_stored', float('nan')):.5f} | {sw} | {ps} |")
open(f"{ROOT}/perframe_dose_{args.backbone}.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(18, 20))
gs = fig.add_gridspec(5, 2, hspace=0.38, wspace=0.18)
sty = {"plain": dict(color="k", lw=1.6), "c10": dict(color="0.5", lw=1.4, ls="--"), "c20": dict(color="0.5", lw=1.4, ls="--")}
pal = plt.cm.tab10.colors
show = [r for r in runs if r[0] in ("plain", "c10", "c20") or r[0].endswith("constbest") or r[0].endswith("_p2")]


def line(ax, k, lab, y, i):
    kw = sty.get(k, dict(color=pal[i % 10], lw=1.2))
    ax.plot(np.arange(len(y)), y, label=lab, **kw)


# (0,0)/(0,1) chosen weights per site
for j, site in enumerate(("state", "mem")):
    ax = fig.add_subplot(gs[0, j])
    for i, ((st, tag), s) in enumerate(sorted(searches.items())):
        if st != site or tag == "placebo":
            continue
        for p in s.get("passes", []):
            ax.step(np.arange(len(p["schedule"])), p["schedule"], where="post", lw=1.0 + 0.6 * (p["pass"] == 2),
                    alpha=0.55 + 0.45 * (p["pass"] == 2), label=f"init {s['init']:g}, pass {p['pass']}")
        ax.axhline(s["const_best"]["w"], color="r", ls=":", lw=1, label=f"scene-best constant {s['const_best']['w']:g}")
    ax.set_title(f"chosen per-frame weight, {site} site ({'a_t' if site == 'state' else 'b_t'})")
    ax.set_xlabel("frame t"); ax.set_ylim(-0.03, 1.03); ax.legend(fontsize=7)
# (1,0)/(1,1) landscape of the last pass, init1: ATE(w,t) - min_w ATE(.,t)
for j, site in enumerate(("state", "mem")):
    ax = fig.add_subplot(gs[1, j])
    s = searches.get((site, "init1"))
    if s and s.get("passes"):
        L = np.array(s["passes"][-1]["landscape"])  # (T-2, K)
        D = (L - L.min(1, keepdims=True)).T
        im = ax.imshow(D, aspect="auto", origin="lower", cmap="magma_r",
                       extent=[0.5, L.shape[0] + 0.5, -0.5, len(s["grid"]) - 0.5], vmax=np.percentile(D, 97))
        ax.set_yticks(range(len(s["grid"]))); ax.set_yticklabels([f"{g:g}" for g in s["grid"]])
        best = L.argmin(1)
        ax.plot(np.arange(1, L.shape[0] + 1), best, "c.", ms=3)
        plt.colorbar(im, ax=ax, fraction=0.03, label="scene ATE - best at this frame")
        ax.set_title(f"{site} site, init plain, pass {s['passes'][-1]['pass']}: ATE cost of each weight at each frame")
        ax.set_xlabel("frame t whose weight is varied"); ax.set_ylabel("weight")
# (2,0) search progress
ax = fig.add_subplot(gs[2, 0])
for (site, tag), s in sorted(searches.items()):
    xs = []; ys = []
    for p in s.get("passes", []):
        n = len(p["best_after_t"]); off = (p["pass"] - 1) * n
        xs += list(np.arange(1, n + 1) + off); ys += p["best_after_t"]
    ax.plot(xs, ys, label=f"{site}, " + ("PLACEBO" if tag == "placebo" else f"init {s['init']:g}"), ls="--" if tag == "placebo" else "-")
    if tag != "placebo":
        ax.axhline(s["const_best"]["ate"], ls=":", lw=0.8, color=ax.lines[-1].get_color())
ax.axhline(searches[next(iter(searches))]["plain_ate_harness"], color="k", lw=0.8, label="plain")
ax.set_title("scene ATE during the search (pass 1 then pass 2; dotted = scene-best constant)")
ax.set_xlabel("frames optimised so far"); ax.set_ylabel("scene ATE"); ax.legend(fontsize=7)
# (2,1) causal ATE so far for shown runs (from per-frame csv not possible: recompute not stored) -> per-frame ATE
panels = [("ate", "per-frame ATE (one Sim(3) fit over the scene)"), ("rpe_trans", "RPE trans (frame t-1 -> t)"),
          ("rpe_rot", "RPE rot, deg (frame t-1 -> t)"), ("absrel", "depth AbsRel"), ("a1", "depth d<1.25"), ]
slots = [gs[2, 1], gs[3, 0], gs[3, 1], gs[4, 0], gs[4, 1]]
for (c, title), sl in zip(panels, slots):
    ax = fig.add_subplot(sl)
    for i, (k, lab, m) in enumerate(show):
        y = m[c]
        if c.startswith("rpe"):
            y = np.convolve(np.nan_to_num(y), np.ones(5) / 5, mode="same")
            title_s = title + " (5-frame mean)"
        else:
            title_s = title
        r = scene_row(m)
        line(ax, k, f"{lab}  [{r[c]:.4f}]" if c != "rpe_rot" else f"{lab}  [{r[c]:.3f}]", y, i)
    ax.set_title(title_s); ax.set_xlabel("frame t"); ax.legend(fontsize=6.5)
fig.suptitle(f"Per-frame write-dose oracle, {scene}, {'zero-shot' if args.backbone == 'zs' else 'finetuned'} CUT3R "
             f"(weights chosen to minimise scene ATE; GT-driven, not causal)", fontsize=12)
fig.savefig(f"{ROOT}/perframe_dose_{args.backbone}.png", dpi=110, bbox_inches="tight")
print("wrote", f"{ROOT}/perframe_dose_{args.backbone}.png")
