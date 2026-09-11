"""ATE of deliberately-constructed control trajectories, scored by the same Sim(3)/RMSE path as
every table row. Establishes the achievable ATE band on the DROID wrist set: constant pose,
GT positions permuted in time, a random walk with GT step sizes and random directions, and
GT + isotropic noise at several fractions of the trajectory extent.

Key result: the random walk scores ~0.1206, which is where CUT3R zero-shot and the zero-shot-focal
OpenCV arm both sit. See OPENCV_VO_DESIGN.md, "DECISIVE CONTROL".

Usage: python eval_pipeline/opencv_vo_probes/ate_controls.py
"""
import numpy as np, sys, json
from multiprocessing import Pool
sys.path.insert(0,"/gpfs/home/koc/koc821022/my-da3/eval_pipeline")
from pose_sim3_both import _load_poses_sorted, _sim3_pose_series, _agg
R="/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
OUT="/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
scenes=[l.strip() for l in open(f"{OUT}/scene_list.txt") if l.strip()][::8]   # 537-scene sample
def ate(pred,gt): 
    a,_,_=_sim3_pose_series(pred,gt); return _agg(a,"rmse")
def one(s):
    try:
        gt=_load_poses_sorted(f"{R}/{s}/dense/cam"); n=len(gt)
        if n<5: return None
        rng=np.random.default_rng(abs(hash(s))%(2**31))
        P=np.stack([gt[i][:3,3] for i in range(n)])
        steps=np.diff(P,axis=0); stepnorm=np.linalg.norm(steps,axis=1)
        out={}
        # 1 constant pose
        out["const_pose"]=ate({i:np.eye(4) for i in range(n)},gt)
        # 2 random walk with GT step-size statistics, random directions
        d=rng.normal(size=(n-1,3)); d/=np.linalg.norm(d,axis=1,keepdims=True); d*=stepnorm[:,None]
        Q=np.cumsum(np.vstack([P[0],d]),axis=0)
        rw={i:np.eye(4) for i in range(n)}
        for i in range(n): rw[i]=np.eye(4); rw[i][:3,3]=Q[i]; rw[i][:3,:3]=gt[i][:3,:3]
        out["random_walk"]=ate(rw,gt)
        # 3 GT positions permuted in time (shape preserved, order destroyed)
        perm=rng.permutation(n); sh={}
        for i in range(n): sh[i]=np.eye(4); sh[i][:3,3]=P[perm[i]]; sh[i][:3,:3]=gt[i][:3,:3]
        out["gt_shuffled"]=ate(sh,gt)
        # 4 GT + isotropic noise at a few sigmas (as fraction of trajectory extent)
        ext=np.linalg.norm(P-P.mean(0),axis=1).mean()
        for f in (0.25,0.5,1.0):
            nz={}
            for i in range(n): nz[i]=np.eye(4); nz[i][:3,3]=P[i]+rng.normal(scale=f*ext,size=3); nz[i][:3,:3]=gt[i][:3,:3]
            out[f"gt_noise_{f}"]=ate(nz,gt)
        return out
    except Exception: return None
with Pool(48) as p: res=[r for r in p.imap_unordered(one,scenes,chunksize=8) if r]
keys=list(res[0])
print(f"ATE of deliberately-constructed trajectories, same scorer, {len(res)}-scene sample:")
for k in keys: print(f"  {k:16s} {np.nanmean([r[k] for r in res]):.4f}")
print("\n  for reference on the SAME sample indices, the real rows (full-set values):")
print("  finetuned 0.0759 | zero-shot 0.1197 | zs+OpenCV 0.1203 | ft+OpenCV 0.1043 | const-vel 0.1248")
