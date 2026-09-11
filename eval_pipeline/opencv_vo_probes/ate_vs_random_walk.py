"""Paired per-scene test of every evaluated arm against a random-walk trajectory with ground-truth
step sizes (3 seeds per scene). Answers "is this arm doing better than knowing only how fast the
camera moved?" for ATE.

Result on 537 scenes: CUT3R zero-shot t=-0.3 and zero-shot+OpenCV t=-0.6 are INDISTINGUISHABLE from
a random walk; finetuned+OpenCV t=-10.5 and CUT3R finetuned t=-32.8 are genuinely better.

Usage: python eval_pipeline/opencv_vo_probes/ate_vs_random_walk.py
"""
import numpy as np, sys, json
from multiprocessing import Pool
sys.path.insert(0,"/gpfs/home/koc/koc821022/my-da3/eval_pipeline")
from pose_sim3_both import _load_poses_sorted, _sim3_pose_series, _agg
from aggregate_results import read_scene_summary
R="/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
OUT="/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
scenes=[l.strip() for l in open(f"{OUT}/scene_list.txt") if l.strip()][::8]
def one(s):
    try:
        gt=_load_poses_sorted(f"{R}/{s}/dense/cam"); n=len(gt)
        if n<5: return None
        P=np.stack([gt[i][:3,3] for i in range(n)]); stepnorm=np.linalg.norm(np.diff(P,axis=0),axis=1)
        vals=[]
        for seed in range(3):   # 3 independent random walks per scene
            rng=np.random.default_rng(seed*99991+abs(hash(s))%(2**20))
            d=rng.normal(size=(n-1,3)); d/=np.linalg.norm(d,axis=1,keepdims=True); d*=stepnorm[:,None]
            Q=np.cumsum(np.vstack([P[0],d]),axis=0)
            rw={}
            for i in range(n): rw[i]=np.eye(4); rw[i][:3,3]=Q[i]; rw[i][:3,:3]=gt[i][:3,:3]
            a,_,_=_sim3_pose_series(rw,gt); vals.append(_agg(a,"rmse"))
        out={"rw":float(np.mean(vals))}
        for lbl,k in [("cut3r_zeroshot","zs"),("vo_full_zs","zscv"),("vo_full_ft","ftcv"),("augfull_lr1e5","ft")]:
            r=read_scene_summary(f"{OUT}/{lbl}/eval/{s}/eval_depth_pose_metrics.csv")
            out[k]=r["ate"] if r else np.nan
        return out
    except Exception: return None
with Pool(48) as p: res=[r for r in p.imap_unordered(one,scenes,chunksize=8) if r]
A={k:np.array([r[k] for r in res]) for k in res[0]}
m=np.all([np.isfinite(A[k]) for k in A],axis=0)
A={k:v[m] for k,v in A.items()}; n=len(A["rw"])
print(f"paired per-scene ATE vs a RANDOM WALK with ground-truth step sizes, {n} scenes\n")
print(f"{'arm':22s} {'mean ATE':>9s} {'vs random walk':>15s} {'paired t':>9s} {'verdict':>28s}")
print(f"{'random walk (3 seeds)':22s} {A['rw'].mean():9.4f} {'--':>15s}")
for k,name in [("zs","CUT3R zero-shot"),("zscv","zero-shot + OpenCV"),("ftcv","finetuned + OpenCV"),("ft","CUT3R finetuned")]:
    dd=A[k]-A["rw"]; se=dd.std(ddof=1)/np.sqrt(n); t=dd.mean()/se
    v="INDISTINGUISHABLE from random" if abs(t)<2 else ("better than random" if t<0 else "worse than random")
    print(f"{name:22s} {A[k].mean():9.4f} {dd.mean():+15.4f} {t:9.1f} {v:>28s}")
