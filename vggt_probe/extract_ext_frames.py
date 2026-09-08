#!/usr/bin/env python
"""Per-episode (one process each, login-node safe): extract ext1/ext2 frame 0 and frame T_store-1
as native-resolution PNGs, count MP4 frames, and cross-check metadata vs store cam/000000.npz.
Writes <raw>/<EP>/frames/*.png and <raw>/<EP>/frames/ep_check.json.
"""
import sys, os, json, time
import numpy as np, cv2

CACHE = '/gpfs/scratch/etur59/koc821022/vggt_cache'
RAW = f'{CACHE}/raw'
STORE = '/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist'

def main(ep):
    t0 = time.time()
    paths = json.load(open(f'{CACHE}/smoke13_raw_paths.json'))[ep]
    meta = json.load(open(f'{RAW}/{ep}/metadata_{ep}.json'))
    fdir = f'{RAW}/{ep}/frames'; os.makedirs(fdir, exist_ok=True)
    store_dense = f'{STORE}/{ep}/dense'
    T_store = len([f for f in os.listdir(f'{store_dense}/rgb') if f.endswith('.png')])
    n_cam = len([f for f in os.listdir(f'{store_dense}/cam') if f.endswith('.npz')])
    cam0 = np.load(f'{store_dense}/cam/000000.npz')
    pose0 = cam0['pose']; K0 = cam0['intrinsic']
    wrist0_store = pose0[:3, 3].astype(float)
    out = dict(ep=ep, raw_path=paths['path'], T_store=T_store, n_cam_npz=n_cam,
               traj_len_meta=meta['trajectory_length'], serials=dict(wrist=meta['wrist_cam_serial'],
               ext1=meta['ext1_cam_serial'], ext2=meta['ext2_cam_serial']),
               serials_match_json=(meta['ext1_cam_serial'] == paths['ext1'] and meta['ext2_cam_serial'] == paths['ext2']
                                   and meta['wrist_cam_serial'] == paths['wrist']),
               success=meta.get('success'), task=meta.get('current_task'),
               wrist0_xyz_store=wrist0_store.tolist(), K0_store=K0.tolist(),
               wrist_extr=meta['wrist_cam_extrinsics'], ext1_extr=meta['ext1_cam_extrinsics'],
               ext2_extr=meta['ext2_cam_extrinsics'], mp4={}, notes=[])
    wm = np.array(meta['wrist_cam_extrinsics'][:3]); e1 = np.array(meta['ext1_cam_extrinsics'][:3]); e2 = np.array(meta['ext2_cam_extrinsics'][:3])
    out['wrist0_xyz_match_maxdiff'] = float(np.abs(wm - wrist0_store).max())
    out['calib_dists'] = dict(wrist0_ext1=float(np.linalg.norm(wm - e1)), wrist0_ext2=float(np.linalg.norm(wm - e2)),
                              ext1_ext2=float(np.linalg.norm(e1 - e2)))
    for tag in ('ext1', 'ext2'):
        s = meta[f'{tag}_cam_serial']
        mp4 = f'{RAW}/{ep}/recordings/MP4/{s}.mp4'
        cap = cv2.VideoCapture(mp4)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps = cap.get(cv2.CAP_PROP_FPS)
        info = dict(serial=s, path=mp4, frame_count=n, width=W, height=H, fps=fps)
        ok, f0 = cap.read()
        info['frame0_ok'] = bool(ok)
        if ok:
            p = f'{fdir}/{tag}_000000.png'; cv2.imwrite(p, f0); info['png0'] = p; info['frame0_shape'] = list(f0.shape)
            g = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
            info['frame0_mean'] = float(g.mean()); info['frame0_std'] = float(g.std())
            # stereo-split check: left/right halves should be similar if this is a SBS stereo frame
            L, R = g[:, :W//2].astype(float), g[:, W//2:].astype(float)
            info['half_absdiff_mean'] = float(np.abs(L - R).mean())
            info['half_corr'] = float(np.corrcoef(L.ravel(), R.ravel())[0, 1]) if L.std() > 0 and R.std() > 0 else None
        last = T_store - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, last)
        ok2, fl = cap.read(); info['last_index'] = last; info['last_ok'] = bool(ok2)
        if ok2:
            p = f'{fdir}/{tag}_last.png'; cv2.imwrite(p, fl); info['png_last'] = p
            gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY).astype(float)
            info['last_vs_first_absdiff_mean'] = float(np.abs(gl - g.astype(float)).mean()) if ok else None
        # verify CAP_PROP_FRAME_COUNT matches true decodable count only if cheap: try reading one past last
        cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1); okn, _ = cap.read(); info['frame_nminus1_readable'] = bool(okn)
        okn1, _ = cap.read(); info['frame_n_readable'] = bool(okn1)
        cap.release()
        out['mp4'][tag] = info
    out['mp4_count_match'] = all(v['frame_count'] == T_store for v in out['mp4'].values())
    if not out['mp4_count_match']:
        out['notes'].append(f"MP4 frame count != T_store: ext1={out['mp4']['ext1']['frame_count']} ext2={out['mp4']['ext2']['frame_count']} T_store={T_store}")
    if T_store != meta['trajectory_length'] - 1:
        out['notes'].append(f"T_store != trajectory_length-1 ({meta['trajectory_length']})")
    if n_cam != T_store:
        out['notes'].append(f"cam npz count {n_cam} != rgb count {T_store}")
    out['elapsed_s'] = time.time() - t0
    json.dump(out, open(f'{fdir}/ep_check.json', 'w'), indent=1)
    print(ep, 'T_store', T_store, 'mp4', [v['frame_count'] for v in out['mp4'].values()], 'wristdiff %.2e' % out['wrist0_xyz_match_maxdiff'], 'notes', out['notes'], '%.1fs' % out['elapsed_s'])

if __name__ == '__main__':
    main(sys.argv[1])
