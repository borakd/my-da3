import os, json, h5py
from multiprocessing import Pool
B=os.environ['B']; OUT=os.environ['OUT']; W=os.environ['WRIST']
sc=json.load(open(f'{B}/meta/scenes.json'))
MODS=['rgb','depth','cam','sky_mask','outlier_mask']
def chk(u):
    nw=sc[u]['n_wrist']; bad=[]; info=[]
    try:
        md=json.load(open(f'{B}/dl/raw/{u}/metadata.json'))
        h=h5py.File(f'{B}/restore/droid/depth_320x180/{u}_depth.h5','r')
    except Exception as e: return u,[f'source:{e}'],[]
    for cam in ['ext1','ext2']:
        s=str(md[f'{cam}_cam_serial']); d=f'{OUT}/{cam}/{u}/dense'
        if not os.path.isdir(d): bad.append(f'{cam}:missing'); continue
        try: F=h[f'{s}+ext']['depth'].shape[0]
        except Exception as e: bad.append(f'{cam}:no depth group {s}'); continue
        cnt={m:len(os.listdir(f'{d}/{m}')) for m in MODS}
        if len(set(cnt.values()))!=1 or list(cnt.values())[0]!=F: bad.append(f'{cam}:{cnt} expected={F}')
        elif F!=nw: info.append(f'{cam}:ext={F} wrist={nw}')
    return u,bad,info
if __name__=='__main__':
    with Pool(8) as p: res=p.map(chk,sorted(sc),chunksize=64)
    bad=[(u,b) for u,b,i in res if b]; info=[(u,i) for u,b,i in res if i and not b]
    rep=(f'scenes {len(sc)}  ok {len(sc)-len(bad)}  problems {len(bad)}  '
         f'ok-but-frame-count-differs-from-wrist {len(info)}\n\nPROBLEMS:\n'+'\n'.join(f'{u} {b}' for u,b in bad)+
         '\n\nFRAME COUNT DIFFERS FROM WRIST (source-limited, complete w.r.t. PointWorld):\n'+'\n'.join(f'{u} {i}' for u,i in info))
    open(f'{B}/REPORT.txt','w').write(rep); print(rep[:2500])
    import shutil; os.makedirs(f'{OUT}/splits',exist_ok=True)
    for f in os.listdir(f'{W}/../splits'): shutil.copy(f'{W}/../splits/{f}',f'{OUT}/splits/{f}')
    raise SystemExit(0 if not bad else 1)
