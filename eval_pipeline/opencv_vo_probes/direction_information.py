"""How much DIRECTIONAL information does each arm's trajectory carry?

For each arm: Sim(3)-align the predicted camera centres to GT (as the scorer does), then take the
median angle between the predicted and true displacement over gaps of 1, 8 and 32 frames.
90 deg = no information (two random directions in 3D); 0 deg = perfect.

Result (144-scene sample): random walk 85.8/80.9/65.0, CUT3R zero-shot 68.7/64.9/54.5,
zero-shot+OpenCV 66.5/59.7/50.3, CUT3R finetuned 40.6/32.1/24.1, finetuned+OpenCV 44.4/43.2/35.5.

NOTE: the alignment matters. Comparing world-frame directions WITHOUT aligning gives nonsense
(>90 deg for every arm), because a predicted trajectory's world frame is arbitrary.
Scenes where an arm produced a fully constant trajectory yield undefined angles and are skipped.

Usage: python eval_pipeline/opencv_vo_probes/direction_information.py
"""
import numpy as np, sys
from multiprocessing import Pool
sys.path.insert(0,"/gpfs/home/koc/koc821022/my-da3/eval_pipeline")
from pose_sim3_both import _load_poses_sorted, _umeyama
R="/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
OUT="/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
scenes=[l.strip() for l in open(f"{OUT}/scene_list.txt") if l.strip()][::30]
LAB=[("CUT3R zero-shot","cut3r_zeroshot"),("CUT3R finetuned","augfull_lr1e5"),
     ("zs + OpenCV","vo_full_zs"),("ft + OpenCV","vo_full_ft"),("random walk","__rw__")]
GAPS=[1,8,32]
def angs(u,v):
    nu,nv=np.linalg.norm(u,axis=1),np.linalg.norm(v,axis=1); ok=(nu>1e-9)&(nv>1e-9)
    c=np.sum(u*v,axis=1)[ok]/(nu[ok]*nv[ok]); return np.degrees(np.arccos(np.clip(c,-1,1)))
def one(s):
    gt=_load_poses_sorted(f"{R}/{s}/dense/cam")
    if len(gt)<40: return None
    n=len(gt); G=np.stack([gt[i][:3,3] for i in range(n)])
    rng=np.random.default_rng(abs(hash(s))%(2**31))
    step=np.linalg.norm(np.diff(G,axis=0),axis=1)
    out={}
    for name,l in LAB:
        if l=="__rw__":
            d=rng.normal(size=(n-1,3)); d/=np.linalg.norm(d,axis=1,keepdims=True); d*=step[:,None]
            P=np.cumsum(np.vstack([G[0],d]),axis=0)
        else:
            pr=_load_poses_sorted(f"{OUT}/{l}/preds/{s}/camera")
            ts=sorted(set(pr)&set(gt))
            if len(ts)<40: continue
            P=np.stack([pr[t][:3,3] for t in ts]); Gs=np.stack([gt[t][:3,3] for t in ts])
        Gs = G if l=="__rw__" else Gs
        # Sim(3)-align the predicted CENTRES to GT, exactly as the scorer does, then compare directions
        sc,Rt,tt=_umeyama(P,Gs,with_scale=True); A=(sc*(Rt@P.T)).T+tt
        for g in GAPS:
            if len(A)<=g: continue
            dg=Gs[g:]-Gs[:-g]; dp=A[g:]-A[:-g]
            keep=np.linalg.norm(dg,axis=1)>2e-3
            if keep.sum()<5: continue
            aa=angs(dp[keep],dg[keep])
            if aa.size: out.setdefault((name,g),[]).append(float(np.median(aa)))
    return out
with Pool(48) as p: res=[r for r in p.imap_unordered(one,scenes,chunksize=4) if r]
print(f"median angle between predicted and true displacement direction AFTER Sim(3) alignment, {len(res)} scenes")
print("(90 deg = no directional information; 0 deg = perfect)\n")
print(f"{'arm':18s} " + "".join(f"{'gap '+str(g):>12s}" for g in GAPS) + "   (deg / n_scenes)")
for name,_ in LAB:
    row=[]
    for g in GAPS:
        v=[x for r in res for x in r.get((name,g),[]) if np.isfinite(x)]
        row.append((f"{np.median(v):7.1f}" if v else "      -")+f"/{len(v):<4d}")
    print(f"{name:18s} " + "".join(row))
