import sys, os, numpy as np
sys.path.insert(0, "/gpfs/home/koc/koc821022/maks_idea/eval_pipeline")
from maks_subset_compare import load_label, boot_ci
OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"; R = f"{OUT}/unc_full4292"
scenes = [l.strip() for l in open(f"{OUT}/scene_list.txt") if l.strip()]
arms = {"base": f"{OUT}/augfull_lr1e5/eval", "self": f"{R}/tok_al_q50_g50/eval", "posefix": f"{OUT}/causal_posefix_tc/eval"}
for r in sys.argv[1:]:
    arms[r] = f"{R}/tok_cb{r}_q50_g50/eval"; arms[r+"_soft"] = f"{R}/tok_cb{r}_q50_g50_soft05/eval"
if os.path.isdir(f"{R}/tok_cbinit_q50_g50/eval"): arms["init"] = f"{R}/tok_cbinit_q50_g50/eval"
D = {k: load_label(v, scenes) for k, v in arms.items()}
common = [s for s in scenes if all(s in D[k] for k in arms)]
mets = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]
print(f"n common = {len(common)}")
print("| arm | " + " | ".join(mets) + " |")
for k in arms:
    print(f"| {k} | " + " | ".join(f"{np.nanmean([D[k][s][m] for s in common]):.4f}" for m in mets) + " |")
pairs = []
for r in sys.argv[1:]:
    pairs += [(r, "self"), (r+"_soft", "self"), (r, "base"), (r+"_soft", "base"), (r+"_soft", r), (r, "posefix")]
if "init" in arms: pairs += [("init", "self")]
print("\n| pair | " + " | ".join(mets) + " |")
for a, b in pairs:
    cells = []
    for m in mets:
        d = np.array([D[a][s][m] - D[b][s][m] for s in common]); d = d[np.isfinite(d)]; lo, hi = boot_ci(d, n=10000)
        star = "*" if (lo > 0 or hi < 0) else ""; wins = int((d > 0).sum()) if m == "a1" else int((d < 0).sum())
        cells.append(f"{d.mean():+.5f}{star} [{lo:+.5f},{hi:+.5f}] {wins}/{len(d)}")
    print(f"| {a} minus {b} | " + " | ".join(cells) + " |")
