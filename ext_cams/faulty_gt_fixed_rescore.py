#!/usr/bin/env python
"""Re-score existing predictions on the faulty-wrist-GT scenes against CORRECTED GT.

Correction (droid_wrist_calib_check, validated 7/7 by projection): the pose
consistent with the RGB is pose_store @ D, D = [diag(-1,-1,1) | (0.063, 0, 0)],
an involution, so the same D fixes both rgb_upside_down and offset_flipped.
For each faulty scene a GT tree with cam/ only (poses; no depth, so depth is
untouched and taken from the original CSV) is written twice: gt_orig (store
poses, parity control) and gt_fixed (store @ D). eval_depth_poses.py runs with
default args on <label>/preds/<scene> against both.
"""
import os, csv, glob, argparse, subprocess
import numpy as np
from multiprocessing import Pool

OUT = '/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval'
STORE = '/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist'
EVAL_PY = '/home/koc/koc821022/.conda/envs/cuteanything/bin/python'
EVAL_SCRIPT = '/gpfs/home/koc/koc821022/using_ext_cams/eval_bundle/bin/eval_depth_poses.py'
D = np.eye(4); D[0, 0] = D[1, 1] = -1; D[0, 3] = 0.063
LABELS = ['cut3r_zeroshot', 'augfull_lr1e5', 'augfull_cg_g7ema_bwd', 'augfull_cg_fuse_g7', 'vo_full_zs', 'vo_full_ft',
          'vo_full_ftgt', 'causal_posefix', 'causal_posefix_ema', 'causal_posefix_tc', 'causal_posefix_tc_ema',
          'tf_cut3r_pretrained', 'tf_ttt3r_pretrained', 'tf_raymap3r_pretrained', 'scal3r_droid_full_320_final',
          'scal3r_droid_full_320_ftbb_final']


def write_gt(scene, root):
    for kind in ('gt_orig', 'gt_fixed'):
        d = f'{root}/{kind}/{scene}/cam'; os.makedirs(d, exist_ok=True)
        for f in sorted(glob.glob(f'{STORE}/{scene}/dense/cam/*.npz')):
            z = np.load(f); P = z['pose'].astype(np.float64)
            np.savez(f'{d}/{os.path.basename(f)}', pose=P if kind == 'gt_orig' else P @ D,
                     **{k: z[k] for k in z.files if k != 'pose'})


def run(job):
    label, scene, kind, root = job
    out_csv = f'{root}/eval_{kind}/{label}/{scene}/eval_depth_pose_metrics.csv'
    if os.path.exists(out_csv) and os.path.getsize(out_csv):
        return job, 0
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    r = subprocess.run([EVAL_PY, EVAL_SCRIPT, '--pred_root', f'{OUT}/{label}/preds/{scene}',
                        '--gt_root', f'{root}/{kind}/{scene}', '--output_csv', out_csv],
                       capture_output=True, text=True, env={**os.environ, 'OMP_NUM_THREADS': '1'})
    return job, r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--faulty_csv', required=True)
    ap.add_argument('--procs', type=int, default=36)
    args = ap.parse_args()
    scenes = [r['scene'] for r in csv.DictReader(open(args.faulty_csv)) if r['category'] in ('rgb_upside_down', 'offset_flipped')]
    with Pool(args.procs) as p:
        p.starmap(write_gt, [(s, args.root) for s in scenes])
        jobs = [(l, s, 'gt_fixed', args.root) for l in LABELS for s in scenes]
        jobs += [(l, s, 'gt_orig', args.root) for l in LABELS for s in scenes]
        res = p.map(run, jobs)
    bad = [j for j, rc in res if rc != 0]
    print(f'{len(scenes)} scenes, {len(jobs)} evals, {len(bad)} failed', bad[:10])


if __name__ == '__main__':
    main()
