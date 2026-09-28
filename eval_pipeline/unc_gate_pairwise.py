"""Paired arm-vs-arm comparison on all 4292 scenes for the uncgate campaign (heads vs the self-view aligned gate,
vs the shuffled-alignment logic captured by the frame gate, etc.). Writes $OUT/unc_full4292/table/pairwise.md."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maks_subset_compare import load_label, boot_ci
OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"; R = f"{OUT}/unc_full4292"
scenes = [l.strip() for l in open(f"{OUT}/scene_list.txt") if l.strip()]
arms = {"base": f"{OUT}/augfull_lr1e5/eval", "g7ema": f"{OUT}/augfull_cg_g7ema/eval", "posefix": f"{OUT}/causal_posefix_tc/eval",
        "self": f"{R}/tok_al_q50_g50/eval", "A": f"{R}/tok_uncA_q50_g50/eval", "A_ema": f"{R}/tok_uncA_q50_g50_ema7/eval",
        "B": f"{R}/tok_uncB_q50_g50/eval", "B_ema": f"{R}/tok_uncB_q50_g50_ema7/eval", "C": f"{R}/tok_uncC_q50_g50/eval",
        "C_ema": f"{R}/tok_uncC_q50_g50_ema7/eval"}
D = {k: load_label(v, scenes) for k, v in arms.items()}
common = [s for s in scenes if all(s in D[k] for k in arms)]
mets = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
lines = [f"Paired per scene on {len(common)} common scenes; 95% bootstrap CI over scenes; wins = scenes where the first arm is better.\n",
         "| pair | absrel | a1 | ate | rpe_trans | rpe_rot |", "|---|---|---|---|---|---|"]
for a, b in [("self", "base"), ("A", "self"), ("B", "self"), ("C", "self"), ("A_ema", "self"), ("B_ema", "self"), ("C_ema", "self"),
             ("A_ema", "A"), ("B_ema", "B"), ("B", "A"), ("C", "A"), ("self", "g7ema"), ("B", "g7ema"), ("B", "posefix"), ("self", "posefix")]:
    cells = []
    for m in mets:
        d = np.array([D[a][s][m] - D[b][s][m] for s in common]); d = d[np.isfinite(d)]; lo, hi = boot_ci(d, n=10000)
        star = "*" if (lo > 0 or hi < 0) else ""; wins = int((d > 0).sum()) if m == "a1" else int((d < 0).sum())
        cells.append(f"{d.mean():+.5f}{star} [{lo:+.5f},{hi:+.5f}] {wins}/{len(d)}")
    lines.append(f"| {a} minus {b} | " + " | ".join(cells) + " |")
open(f"{R}/table/pairwise.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
