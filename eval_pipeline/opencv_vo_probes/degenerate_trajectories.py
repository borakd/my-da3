"""Count scenes where an arm emitted a degenerate (constant) trajectory, and the fraction of
frame-to-frame steps that are exactly zero (the backbone holds the previous pose on a failed frame).

Result over all 4292 scenes: vo_full_ft 26 fully-constant scenes and a median 12.7% frozen steps;
vo_full_zs 15 and 10.1%; the CUT3R rows 0 and 0.0% (a network always emits a fresh pose).

Usage: python eval_pipeline/opencv_vo_probes/degenerate_trajectories.py
"""
import numpy as np, glob, json, sys
from multiprocessing import Pool
OUT="/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
scenes=[l.strip() for l in open(f"{OUT}/scene_list.txt") if l.strip()]
def one(a):
    l,s=a
    P=np.stack([np.load(f)["pose"][:3,3] for f in sorted(glob.glob(f"{OUT}/{l}/preds/{s}/camera/*.npz"))])
    d=np.linalg.norm(np.diff(P,axis=0),axis=1)
    return (l, s, len(P), float(np.ptp(P,axis=0).max()), float((d==0).mean()))
LAB=["vo_full_ft","vo_full_zs","augfull_lr1e5","cut3r_zeroshot"]
tasks=[(l,s) for l in LAB for s in scenes]
R={l:[] for l in LAB}
with Pool(48) as p:
    for l,s,n,ext,zfrac in p.imap_unordered(one,tasks,chunksize=32): R[l].append((s,n,ext,zfrac))
print(f"{'label':16s} {'scenes':>7s} {'fully-constant traj':>20s} {'median % frozen steps':>22s}")
for l in LAB:
    a=R[l]; const=sum(1 for _,_,e,_ in a if e<1e-9)
    print(f"{l:16s} {len(a):7d} {const:20d} {100*np.median([z for *_,z in a]):21.1f}%")
json.dump({l:[(s,n,e,z) for s,n,e,z in R[l]] for l in R}, open("/tmp/degen.json","w"))
