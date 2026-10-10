#!/usr/bin/env python
"""VGGT-Omega t=0 three-camera pose accuracy on the 4292-scene DROID test split.

Per scene, ONE 3-image forward per reference ordering (the first image is
VGGT's reference camera), frames identical to vggt_features/vggt_probe/
t0_pose_forward.py so its 40-scene cache is a parity check:
  wrist_first : [wrist_0, ext1_0, ext2_0]
  ext1_first  : [ext1_0, ext2_0, wrist_0]
  ext2_first  : [ext2_0, ext1_0, wrist_0]
wrist_0 = store dense/rgb/000000.png (320x180); ext*_0 = frame 0 of the
exterior MP4 (1280x720; MP4 index == store index, verified pixel-exact),
decoded with cv2 to a node-local PNG exactly as extract_t0_ext_frames.py did.
Forward = the README recipe (VGGT-Omega-1B-512, image_resolution=512).

Scored with the standard evaluator (eval_bundle/bin/eval_depth_poses.py, all
default args: Sim3 Umeyama on camera centres, pose RMSE over frames), one
3-frame "trajectory" per scene in the CANONICAL order [wrist_0, ext1_0, ext2_0]
for every ordering, so the reference camera is the only thing that changes:
  GT   wrist_0 = store cam/000000.npz pose (kinematic, robot base frame);
       ext1/ext2 = inv(PointWorld optimized_extrinsics[serial]) (w2c -> c2w).
  pred c2w = inv(VGGT extrinsic) (camera-from-world -> world-from-camera).
Layout under --out_root:
  gt/<EP>/camera/00000{0,1,2}.npz
  <order>/preds/<EP>/camera/00000{0,1,2}.npz   (pose c2w, intrinsics, hw)
  <order>/raw/<EP>.npz                         (t0_pose_forward.py format)
  <order>/eval/<EP>/eval_depth_pose_metrics.csv
"""
import os, sys, json, time, argparse, subprocess, tempfile, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, cv2
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

CACHE = '/gpfs/scratch/etur59/koc821022/vggt_cache'
STORE = '/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist'
PW = '/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/gt_audit_2026-09-08/pointworld/droid/cameras'
CKPT = '/gpfs/home/koc/koc821022/vggt-omega/checkpoints/VGGT-Omega-1B-512/model.pt'
EVAL_PY = '/home/koc/koc821022/.conda/envs/cuteanything/bin/python'
EVAL_SCRIPT = '/gpfs/home/koc/koc821022/using_ext_cams/eval_bundle/bin/eval_depth_poses.py'
ORDERS = {'wrist_first': ['wrist0', 'ext1', 'ext2'],
          'ext1_first': ['ext1', 'ext2', 'wrist0'],
          'ext2_first': ['ext2', 'ext1', 'wrist0']}
CANON = ['wrist0', 'ext1', 'ext2']


def save_cams(d, poses, extra=None):
    os.makedirs(f'{d}/camera', exist_ok=True)
    for i, P in enumerate(poses):
        kw = {k: v[i] for k, v in (extra or {}).items()}
        np.savez(f'{d}/camera/{i:06d}.npz', pose=P.astype(np.float64), **kw)


def write_gt(ep, out_root, store=STORE, compact=False):
    meta = json.load(open(f'{CACHE}/raw/{ep}/metadata_{ep}.json'))
    mp4s = json.load(open(f'{CACHE}/raw/{ep}/ext_mp4s.json'))
    pw = json.load(open(f'{PW}/{ep}_cameras.json'))
    assert pw['optimization_success'], ep
    serial = {t: str(meta[f'{t}_cam_serial']) for t in ('ext1', 'ext2')}
    for t in ('ext1', 'ext2'):
        assert serial[t] == str(mp4s[t]), (ep, t, serial[t], mp4s[t])
    w0 = np.load(f'{store}/{ep}/dense/cam/000000.npz')['pose'].astype(np.float64)
    gt = {'wrist0': w0}
    for t in ('ext1', 'ext2'):
        gt[t] = np.linalg.inv(np.array(pw[serial[t]]['optimized_extrinsics'], dtype=np.float64))
    if not compact:
        save_cams(f'{out_root}/gt/{ep}', [gt[n] for n in CANON])
    return serial, np.stack([gt[n] for n in CANON])


def run_eval(pred_root, gt_root, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    r = subprocess.run([EVAL_PY, EVAL_SCRIPT, '--pred_root', pred_root, '--gt_root', gt_root,
                        '--output_csv', out_csv], capture_output=True, text=True,
                       env={**os.environ, 'OMP_NUM_THREADS': '1', 'PYTHONNOUSERSITE': '1'})
    if r.returncode != 0 or not os.path.getsize(out_csv):
        raise RuntimeError(f'eval failed {out_csv}: {r.stderr[-2000:]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_list', required=True)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--shard_id', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--eval_workers', type=int, default=12)
    ap.add_argument('--store', default=STORE, help='wrist store (test split by default)')
    ap.add_argument('--compact', action='store_true',
                    help='no evaluator: one <order>/compact/<EP>.npz with canonical pred_c2w, gt_c2w, conf_mean')
    ap.add_argument('--wrist_depth', action='store_true',
                    help='also save the wrist depth (pred depth/000000.npy, GT = store depth/000000.npy) '
                         'and write the CSVs to <order>/eval_depth/ so AbsRel/a1 are scored too')
    args = ap.parse_args()
    store = args.store
    eps = [l.strip() for l in open(args.scene_list) if l.strip()]
    eps = eps[args.shard_id::args.num_shards]
    if args.limit:
        eps = eps[:args.limit]
    model = VGGTOmega().cuda().eval()
    model.load_state_dict(torch.load(CKPT, map_location='cpu'))
    print(f'model loaded; shard {args.shard_id}/{args.num_shards}: {len(eps)} scenes', flush=True)
    tmp = tempfile.mkdtemp(prefix='t0v3_', dir=os.environ.get('TMPDIR', '/tmp'))
    pool = ThreadPoolExecutor(args.eval_workers)
    futs, n_done, n_skip, fails, t_start = [], 0, 0, [], time.time()
    for k, ep in enumerate(eps):
        ev = 'eval_depth' if args.wrist_depth else 'eval'
        csvs = {o: f'{args.out_root}/{o}/{ev}/{ep}/eval_depth_pose_metrics.csv' for o in ORDERS}
        if args.compact:
            csvs = {o: f'{args.out_root}/{o}/compact/{ep}.npz' for o in ORDERS}
        if all(os.path.exists(c) and os.path.getsize(c) for c in csvs.values()):
            n_skip += 1; continue
        try:
            serial, gt_c2w = write_gt(ep, args.out_root, store, args.compact)
            if args.wrist_depth:
                os.makedirs(f'{args.out_root}/gt/{ep}/depth', exist_ok=True)
                link = f'{args.out_root}/gt/{ep}/depth/000000.npy'
                if not os.path.lexists(link):
                    os.symlink(f'{store}/{ep}/dense/depth/000000.npy', link)
            paths = {'wrist0': f'{store}/{ep}/dense/rgb/000000.png'}
            for t in ('ext1', 'ext2'):
                cap = cv2.VideoCapture(f'{CACHE}/raw/{ep}/recordings/MP4/{serial[t]}.mp4')
                ok, fr = cap.read(); cap.release()
                assert ok, f'no frame 0 for {ep} {t}'
                paths[t] = f'{tmp}/{t}_000000.png'
                cv2.imwrite(paths[t], fr)
            for order, names in ORDERS.items():
                src = [paths[n] for n in names]
                images = load_and_preprocess_images(src, image_resolution=512).cuda()
                t0 = time.time()
                with torch.inference_mode():
                    pred = model(images)
                torch.cuda.synchronize()
                extr, intr = encoding_to_camera(pred['pose_enc'], pred['images'].shape[-2:])
                extr = extr[0].float().cpu().numpy(); intr = intr[0].float().cpu().numpy()
                conf = pred['depth_conf'][0].float().mean(dim=(1, 2)).cpu().numpy()
                hw = np.array(images.shape[-2:])
                c2w = {}
                for i, n in enumerate(names):
                    E = np.eye(4); E[:3] = extr[i].astype(np.float64)
                    c2w[n] = np.linalg.inv(E)
                idx = [names.index(n) for n in CANON]
                if args.compact:
                    os.makedirs(f'{args.out_root}/{order}/compact', exist_ok=True)
                    np.savez(csvs[order], pred_c2w=np.stack([c2w[n] for n in CANON]), gt_c2w=gt_c2w,
                             conf_mean=conf[idx], intr=intr[idx], hw=hw)
                    continue
                os.makedirs(f'{args.out_root}/{order}/raw', exist_ok=True)
                np.savez(f'{args.out_root}/{order}/raw/{ep}.npz', extr=extr, intr=intr,
                         pose_enc=pred['pose_enc'][0].float().cpu().numpy(), conf_mean=conf,
                         order=np.array(names), hw=hw, sources=np.array(src), forward_s=np.float32(time.time() - t0))
                pred_root = f'{args.out_root}/{order}/preds/{ep}'
                save_cams(pred_root, [c2w[n] for n in CANON],
                          {'intrinsics': intr[idx].astype(np.float64), 'hw': np.stack([hw] * 3), 'conf_mean': conf[idx]})
                if args.wrist_depth:
                    os.makedirs(f'{pred_root}/depth', exist_ok=True)
                    np.save(f'{pred_root}/depth/000000.npy',
                            pred['depth'][0, names.index('wrist0'), ..., 0].float().cpu().numpy())
                futs.append((ep, order, pool.submit(run_eval, pred_root, f'{args.out_root}/gt/{ep}', csvs[order])))
            n_done += 1
        except Exception:
            fails.append(ep); traceback.print_exc()
        if (k + 1) % 50 == 0:
            el = time.time() - t_start
            print(f'[{k + 1}/{len(eps)}] done {n_done} skip {n_skip} fail {len(fails)} {el:.0f}s', flush=True)
    for ep, order, f in futs:
        try:
            f.result()
        except Exception:
            fails.append(f'{ep}:{order}'); traceback.print_exc()
    pool.shutdown()
    print(f'DONE shard {args.shard_id}: done {n_done} skip {n_skip} fail {len(fails)} {fails[:20]} '
          f'{time.time() - t_start:.0f}s', flush=True)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
