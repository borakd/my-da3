#!/usr/bin/env python
"""Per-frame wrist-GT consistency check with VGGT-Omega.

One forward per scene over [ext1_0, ext2_0, wrist_t for 8 t evenly spaced over
the episode] (README recipe, VGGT-Omega-1B-512, res 512). The exterior cameras
are static, so they anchor every wrist frame. For each wrist frame t and each
exterior e, the wrist-frame correction D = R_gt(e->w_t)^T R_pred(e->w_t); a
wrong wrist hand-eye calibration is a constant per-episode error, so it shows as
the same D on all 8 frames (a 180 deg roll about the optical axis for the ZED
flip), while a VGGT mis-registration does not repeat across frames.
GT: wrist = store cam/<t>.npz pose; ext = inv(PointWorld optimized_extrinsics).
Writes <out>/<EP>.npz with D (8, 2, 3, 3), t_idx, and prints a summary line.
"""
import os, json, argparse, tempfile
import numpy as np, torch, cv2
from scipy.spatial.transform import Rotation as R
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

CACHE = '/gpfs/scratch/etur59/koc821022/vggt_cache'
STORE = '/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist'
PW = '/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/vggt_probe/gt_audit_2026-09-08/pointworld/droid/cameras'
CKPT = '/gpfs/home/koc/koc821022/vggt-omega/checkpoints/VGGT-Omega-1B-512/model.pt'
RZ = np.diag([-1., -1., 1.])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_list', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_wrist', type=int, default=8)
    ap.add_argument('--store', default=STORE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    store = args.store
    eps = [l.strip() for l in open(args.scene_list) if l.strip()]
    model = VGGTOmega().cuda().eval()
    model.load_state_dict(torch.load(CKPT, map_location='cpu'))
    tmp = tempfile.mkdtemp(dir=os.environ.get('TMPDIR', '/tmp'))
    for ep in eps:
        meta = json.load(open(f'{CACHE}/raw/{ep}/metadata_{ep}.json'))
        pw = json.load(open(f'{PW}/{ep}_cameras.json'))
        paths, gt = [], []
        for e in ('ext1', 'ext2'):
            ser = str(meta[f'{e}_cam_serial'])
            cap = cv2.VideoCapture(f'{CACHE}/raw/{ep}/recordings/MP4/{ser}.mp4'); ok, fr = cap.read(); cap.release()
            assert ok, (ep, e)
            paths.append(f'{tmp}/{e}.png'); cv2.imwrite(paths[-1], fr)
            gt.append(np.linalg.inv(np.array(pw[ser]['optimized_extrinsics'], dtype=np.float64)))
        n = len(os.listdir(f'{store}/{ep}/dense/cam'))
        ts = np.linspace(0, n - 1, args.n_wrist).round().astype(int)
        for t in ts:
            paths.append(f'{store}/{ep}/dense/rgb/{t:06d}.png')
            gt.append(np.load(f'{store}/{ep}/dense/cam/{t:06d}.npz')['pose'].astype(np.float64))
        images = load_and_preprocess_images(paths, image_resolution=512).cuda()
        with torch.inference_mode():
            pred = model(images)
        extr, _ = encoding_to_camera(pred['pose_enc'], pred['images'].shape[-2:])
        c2w = []
        for E3 in extr[0].float().cpu().numpy():
            E = np.eye(4); E[:3] = E3; c2w.append(np.linalg.inv(E))
        D = np.zeros((len(ts), 2, 3, 3))
        for i in range(len(ts)):
            for e in range(2):
                rg = (np.linalg.inv(gt[e]) @ gt[2 + i])[:3, :3]
                rp = (np.linalg.inv(c2w[e]) @ c2w[2 + i])[:3, :3]
                D[i, e] = rg.T @ rp
        e12 = np.degrees(R.from_matrix((np.linalg.inv(gt[0]) @ gt[1])[:3, :3].T @ (np.linalg.inv(c2w[0]) @ c2w[1])[:3, :3]).magnitude())
        np.savez(f'{args.out}/{ep}.npz', D=D, t_idx=ts, e12=e12)
        ang = np.degrees(R.from_matrix(D.reshape(-1, 3, 3)).magnitude())
        fix = np.degrees(R.from_matrix(np.einsum('ij,njk->nik', RZ.T, D.reshape(-1, 3, 3))).magnitude())
        print(f'{ep} e12 {e12:.1f} ang med {np.median(ang):.1f} [{ang.min():.1f},{ang.max():.1f}] '
              f'fixed med {np.median(fix):.1f} frac_flip {np.mean(fix < ang):.2f}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
