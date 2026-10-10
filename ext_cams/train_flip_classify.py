#!/usr/bin/env python
"""Flag faulty wrist GT on a split from t0_vggt3cam_forward.py --compact outputs.

Flip signature (as on the test split): for each of the 3 orderings and both
exterior cameras, the wrist-frame correction D = R_gt(e->w)^T R_pred(e->w);
flagged when median angle > 150, the mean rotation axis is the optical axis
(|z| > 0.9), the residual after undoing a 180 deg roll < 15, and the ext1-ext2
pair is registered (< 15 deg). 'ambiguous' = wrist error > 20 deg otherwise.
Type split for flagged scenes: gripper up/down template score on the temporal
mean of 8 wrist frames (template from the test split; < 0.3 needs a look).
"""
import os, csv, argparse
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ThreadPoolExecutor

ORDERS = ['wrist_first', 'ext1_first', 'ext2_first']
RZ = np.diag([-1., -1., 1.])


def score(path_fmt, s):
    Ds, e12 = [], []
    for o in ORDERS:
        z = np.load(path_fmt.format(o=o, s=s)); p, g = z['pred_c2w'], z['gt_c2w']
        for e in (1, 2):
            Ds.append((np.linalg.inv(g[e]) @ g[0])[:3, :3].T @ (np.linalg.inv(p[e]) @ p[0])[:3, :3])
        e12.append(np.degrees(R.from_matrix((np.linalg.inv(g[1]) @ g[2])[:3, :3].T @ (np.linalg.inv(p[1]) @ p[2])[:3, :3]).magnitude()))
    Rs = R.from_matrix(np.stack(Ds)); ang = np.degrees(Rs.magnitude()); m = Rs.mean().as_rotvec()
    fix = [np.degrees(R.from_matrix(RZ.T @ d).magnitude()) for d in Ds]
    return dict(scene=s, w_ang_med=float(np.median(ang)), w_axis_z=float(abs(m[2]) / max(np.linalg.norm(m), 1e-9)),
                w_ang_fixed_med=float(np.median(fix)), e12_med=float(np.median(e12)))


def updown(store, s, tmpl):
    fs = sorted(os.listdir(f'{store}/{s}/dense/rgb')); idx = np.linspace(0, len(fs) - 1, 8).astype(int)
    a = np.stack([np.asarray(Image.open(f'{store}/{s}/dense/rgb/{fs[i]}').convert('L').resize((80, 45)), np.float32) for i in idx])
    f = -(a.mean(0) / 255.) - (a.std(0) / 255.); f = (f - f.mean()) / (f.std() + 1e-6)
    return float(np.mean(f * tmpl) - np.mean(f * tmpl[::-1, ::-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compact_root', required=True)
    ap.add_argument('--scene_list', required=True)
    ap.add_argument('--store', required=True)
    ap.add_argument('--template', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--threads', type=int, default=32)
    args = ap.parse_args()
    scenes = [l.strip() for l in open(args.scene_list) if l.strip()]
    fmt = args.compact_root + '/{o}/compact/{s}.npz'
    have = [s for s in scenes if all(os.path.exists(fmt.format(o=o, s=s)) for o in ORDERS)]
    with ThreadPoolExecutor(args.threads) as ex:
        rows = list(ex.map(lambda s: score(fmt, s), have))
    tmpl = np.load(args.template)
    for r in rows:
        flip = r['w_ang_med'] > 150 and r['w_axis_z'] > 0.9 and r['w_ang_fixed_med'] < 15 and r['e12_med'] < 15
        r['status'] = 'flip' if flip else ('ext_pair_fail' if r['e12_med'] >= 15 else ('ambiguous' if r['w_ang_med'] > 20 else 'ok'))
    need = [r for r in rows if r['status'] in ('flip', 'ambiguous')]
    with ThreadPoolExecutor(args.threads) as ex:
        uds = list(ex.map(lambda r: updown(args.store, r['scene'], tmpl), need))
    for r, u in zip(need, uds):
        r['updown_score'] = u
    with open(args.out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['scene', 'status', 'w_ang_med', 'w_axis_z', 'w_ang_fixed_med', 'e12_med', 'updown_score'])
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    print('listed', len(scenes), 'scored', len(have), Counter(r['status'] for r in rows))
    f = [r for r in rows if r['status'] == 'flip']
    print('flip: updown<0', sum(r['updown_score'] < 0 for r in f), ' 0..0.3', sum(0 <= r['updown_score'] < 0.3 for r in f),
          ' >=0.3', sum(r['updown_score'] >= 0.3 for r in f))


if __name__ == '__main__':
    main()
