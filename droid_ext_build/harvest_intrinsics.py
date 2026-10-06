"""Extract per-episode exterior intrinsics from PointWorld flows h5 files in a directory tree."""
import h5py, json, os, sys, glob, numpy as np
src=sys.argv[1]; outdir=sys.argv[2]; os.makedirs(outdir,exist_ok=True); n=0
for f in glob.glob(f'{src}/**/*_flows.h5',recursive=True):
    u=os.path.basename(f)[:-len('_flows.h5')]
    try:
        with h5py.File(f,'r') as h:
            clips=[k for k in h if ':' in k]
            res={}
            for c in clips:
                for g in h[c]:
                    if g.startswith('camera_') and g.endswith('_ext'):
                        s=g.split('_')[1]; K=h[c][g]['intrinsic'][()].astype(float).tolist()
                        res.setdefault(s,K)
            if res: json.dump(res,open(f'{outdir}/{u}.json','w')); n+=1
    except Exception as e: print('ERR',u,e,file=sys.stderr)
print('harvested',n)
