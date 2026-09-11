"""Control for "how can two different pose estimators score the same?": two INDEPENDENT
random-walk ensembles, sharing nothing but the ground-truth step sizes, are scored through the same
Sim(3)/RMSE path and their mean ATEs compared.

Result on 537 scenes: they differ by 0.67 mm, while their per-scene difference has a 27.3 mm std.
CUT3R zero-shot and zero-shot+OpenCV differ by 0.60 mm on the full 4292. Agreement at the
millimetre level between mean ATEs is what averaging thousands of scenes does to any two methods
in the same band - it is not evidence that the two methods are related.

Usage: python eval_pipeline/opencv_vo_probes/ate_two_random_walks.py
"""
import numpy as np, sys
from multiprocessing import Pool
sys.path.insert(0,"/gpfs/home/koc/koc821022/my-da3/eval_pipeline")
from pose_sim3_both import _load_poses_sorted, _sim3_pose_series, _agg
R="/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
OUT="/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
scenes=[l.strip() for l in open(f"{OUT}/scene_list.txt") if l.strip()][::8]
def one(s):
    try:
        gt=_load_poses_sorted(f"{R}/{s}/dense/cam"); n=len(gt)
        if n<5: return None
        P=np.stack([gt[i][:3,3] for i in range(n)]); step=np.linalg.norm(np.diff(P,axis=0),axis=1)
        out=[]
        for seed in (12345,67890):   # two INDEPENDENT random-walk ensembles
            rng=np.random.default_rng(seed+abs(hash(s))%(2**20))
            d=rng.normal(size=(n-1,3)); d/=np.linalg.norm(d,axis=1,keepdims=True); d*=step[:,None]
            Q=np.cumsum(np.vstack([P[0],d]),axis=0)
            pr={}
            for i in range(n): pr[i]=np.eye(4); pr[i][:3,3]=Q[i]; pr[i][:3,:3]=gt[i][:3,:3]
            a,_,_=_sim3_pose_series(pr,gt); out.append(_agg(a,"rmse"))
        return out
    except Exception: return None
with Pool(48) as p: res=[r for r in p.imap_unordered(one,scenes,chunksize=8) if r]
A=np.array(res); d=A[:,1]-A[:,0]
print(f"Two INDEPENDENT random-walk ensembles, {len(A)} scenes:")
print(f"  ensemble A mean ATE {A[:,0].mean():.4f}")
print(f"  ensemble B mean ATE {A[:,1].mean():.4f}")
print(f"  difference of the means: {1000*abs(d.mean()):.2f} mm     (per-scene std of the difference: {1000*d.std(ddof=1):.1f} mm)")
print(f"\n  for comparison, CUT3R zero-shot vs zero-shot+OpenCV differ by 0.60 mm on the full 4292.")
