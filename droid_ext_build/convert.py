"""Convert exterior cameras of DROID episodes to the dl3dv_multi/dense layout used by the wrist data.
Per frame: rgb PNG (PIL bilinear 320x180 from raw MP4 left-eye frame), depth .npy float32 m (PointWorld uint16 mm / 1000),
cam .npz {intrinsic (PointWorld flows, SDK-rectified @320x180), pose (raw DROID per-step camera_extrinsics -> c2w, as wrist),
pose_pointworld (inv of PointWorld optimized extrinsics)}, sky_mask (zeros), outlier_mask (255 where depth==0)."""
import os, sys, json, argparse, shutil, traceback, time
import numpy as np, h5py, cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R
from multiprocessing import Pool
B=os.environ['B']; OUT=os.environ['OUT']
def c2w_from_6d(v):
    T=np.eye(4,dtype=np.float64); T[:3,3]=v[:3]; T[:3,:3]=R.from_euler('xyz',v[3:6]).as_matrix(); return T.astype(np.float32)
def process(u):
    mark=f'{B}/state/converted/{u}'
    if os.path.exists(mark): return (u,'skip','')
    try: os.mkdir(f'{B}/state/claim_conv/{u}')
    except FileExistsError: return (u,'skip','')
    try:
        raw=f'{B}/dl/raw/{u}'; md=json.load(open(f'{raw}/metadata.json'))
        cams=json.load(open(f'{B}/meta/cameras/{u}_cameras.json'))
        K_all=json.load(open(f'{B}/intrinsics/{u}.json'))
        dh=h5py.File(f'{B}/restore/droid/depth_320x180/{u}_depth.h5','r')
        tr=h5py.File(f'{raw}/trajectory.h5','r')
        n_w=len(os.listdir(f'{os.environ["WRIST"]}/{u}/dense/rgb'))
        notes=[]
        for name in ['ext1','ext2']:
            s=str(md[f'{name}_cam_serial'])
            if s not in K_all or s not in cams: raise RuntimeError(f'serial {s} missing in intrinsics/cameras')
            D=dh[f'{s}+ext']['depth']; F=D.shape[0]
            K=np.array(K_all[s],dtype=np.float32)
            E=tr[f'observation/camera_extrinsics/{s}_left'][:] if f'observation/camera_extrinsics/{s}_left' in tr else None
            if E is None: E=np.tile(np.array(md[f'{name}_cam_extrinsics'],dtype=np.float64),(F,1)); notes.append(f'{s}:static_extr')
            Wc=np.linalg.inv(np.array(cams[s]['optimized_extrinsics'],dtype=np.float64)).astype(np.float32)
            cap=cv2.VideoCapture(f'{raw}/{s}.mp4')
            dst=f'{OUT}/{name}/{u}/dense'; tmp=f'{OUT}/{name}/.tmp_{u}/dense'
            shutil.rmtree(os.path.dirname(tmp),ignore_errors=True)
            for sub in ['rgb','depth','cam','sky_mask','outlier_mask']: os.makedirs(f'{tmp}/{sub}')
            sky=Image.fromarray(np.zeros((180,320),np.uint8))
            i=0
            while i<F:
                ok,fr=cap.read()
                if not ok: break
                rgb=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)).resize((320,180),Image.BILINEAR)
                rgb.save(f'{tmp}/rgb/{i:06d}.png')
                d=(D[i].astype(np.float32)/1000.0); np.save(f'{tmp}/depth/{i:06d}.npy',d)
                Image.fromarray(((d==0)*255).astype(np.uint8)).save(f'{tmp}/outlier_mask/{i:06d}.png')
                sky.save(f'{tmp}/sky_mask/{i:06d}.png')
                e=E[min(i,len(E)-1)]
                np.savez(f'{tmp}/cam/{i:06d}.npz',intrinsic=K,pose=c2w_from_6d(e),pose_pointworld=Wc)
                i+=1
            cap.release()
            if i!=F or i!=n_w: notes.append(f'{s}:frames mp4={i} depthF={F} wrist={n_w}')
            if i==0: raise RuntimeError(f'no frames decoded for {s}')
            if os.path.exists(dst): shutil.rmtree(os.path.dirname(dst))
            os.rename(os.path.dirname(tmp),os.path.dirname(dst))
        open(mark,'w').write(';'.join(notes)); return (u,'ok',';'.join(notes))
    except Exception as e:
        try: os.rmdir(f'{B}/state/claim_conv/{u}')
        except OSError: pass
        return (u,'fail',f'{type(e).__name__}: {e}')
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--chunk',type=int,default=0); ap.add_argument('--nchunks',type=int,default=1)
    ap.add_argument('--workers',type=int,default=8); ap.add_argument('--scenes',nargs='*'); a=ap.parse_args()
    for d in [f'{B}/state/converted',f'{B}/state/claim_conv',f'{OUT}/ext1',f'{OUT}/ext2']: os.makedirs(d,exist_ok=True)
    # sweep stale claims (claimed >10 min ago, never finished) left by killed jobs
    now=time.time(); swept=0
    for c in os.listdir(f'{B}/state/claim_conv'):
        cp=f'{B}/state/claim_conv/{c}'
        try:
            if not os.path.exists(f'{B}/state/converted/{c}') and now-os.stat(cp).st_mtime>600: os.rmdir(cp); swept+=1
        except OSError: pass
    print('swept stale claims:',swept,flush=True)
    allsc=sorted(json.load(open(f'{B}/meta/scenes.json')))
    # own chunk first, then everything else (claims prevent double work across concurrent jobs)
    own=allsc[a.chunk::a.nchunks]; rest=[u for i,u in enumerate(allsc) if i%a.nchunks!=a.chunk]
    sc=a.scenes or own+rest
    t0=time.time(); n=collections=0; fails=[]
    with Pool(a.workers) as p:
        for k,(u,st,msg) in enumerate(p.imap_unordered(process,sc,chunksize=4)):
            if st=='fail': fails.append(u); print('FAIL',u,msg,flush=True)
            elif msg: print('NOTE',u,msg,flush=True)
            if k%200==0: print(f'{k}/{len(sc)} {time.time()-t0:.0f}s',flush=True)
    with open(f'{B}/state/convert_failed_{a.chunk}.txt','w') as f: f.write('\n'.join(fails))
    print('done',len(sc),'fails',len(fails))
