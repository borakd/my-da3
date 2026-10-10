#!/usr/bin/env python
"""Summarise faulty_gt_fixed_rescore.py: parity of gt_orig vs the original eval
CSVs, then 4292-scene pose means with the 186 faulty scenes scored against
the store GT (as published) vs the corrected GT. Depth is unchanged."""
import csv, sys, numpy as np
sys.path.insert(0, '/gpfs/home/koc/koc821022/using_ext_cams/ext_cams')
from faulty_gt_rescore import scene_row, OUT, M
from faulty_gt_fixed_rescore import LABELS
W = f'{OUT}/vggt_t0_3cam/wristgt_check'
scenes = [l.strip() for l in open(f'{OUT}/scene_list.txt') if l.strip()]
cat = {r['scene']: r['category'] for r in csv.DictReader(open(f'{W}/faulty_wrist_gt.csv')) if r['category'] in ('rgb_upside_down', 'offset_flipped')}
P = ['ate', 'rpe_trans', 'rpe_rot']; J = [M.index(m) for m in P]
rows = []
for l in LABELS:
    X = np.load(f'{W}/rescore/per_scene_{l}.npz')['X'][:, J]
    Xf = X.copy(); par = 0.0
    for i, s in enumerate(scenes):
        if s in cat:
            o = np.array(scene_row(f'{W}/fixedgt/eval_gt_orig/{l}/{s}/eval_depth_pose_metrics.csv'))[J]
            f = np.array(scene_row(f'{W}/fixedgt/eval_gt_fixed/{l}/{s}/eval_depth_pose_metrics.csv'))[J]
            par = max(par, float(np.nanmax(np.abs(o - X[i]))))
            Xf[i] = f
    c = np.array([cat.get(s, 'clean') for s in scenes])
    r = {'label': l, 'parity_maxdiff': par}
    for k, m in (('all', c != 'x'), ('ud', c == 'rgb_upside_down'), ('off', c == 'offset_flipped')):
        for j, met in enumerate(P):
            r[f'{met}_{k}_orig'] = float(np.mean(X[m, j])); r[f'{met}_{k}_fixed'] = float(np.mean(Xf[m, j]))
    r['clean_ate'] = float(np.mean(X[c == 'clean', 0])); r['clean_rpe_rot'] = float(np.mean(X[c == 'clean', 2]))
    r['clean_rpe_trans'] = float(np.mean(X[c == 'clean', 1]))
    rows.append(r)
with open(f'{W}/rescore/fixedgt_summary.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('| label | parity | ATE pub | ATE fixed | RPEt pub | RPEt fixed | RPErot pub | RPErot fixed | ud RPErot pub->fixed | off RPErot pub->fixed | ud ATE pub->fixed | off ATE pub->fixed |')
for r in rows:
    print(f"| {r['label']} | {r['parity_maxdiff']:.0e} | {r['ate_all_orig']:.4f} | {r['ate_all_fixed']:.4f} | {r['rpe_trans_all_orig']:.4f} | {r['rpe_trans_all_fixed']:.4f} | "
          f"{r['rpe_rot_all_orig']:.3f} | {r['rpe_rot_all_fixed']:.3f} | {r['rpe_rot_ud_orig']:.2f}->{r['rpe_rot_ud_fixed']:.2f} | {r['rpe_rot_off_orig']:.2f}->{r['rpe_rot_off_fixed']:.2f} | "
          f"{r['ate_ud_orig']:.4f}->{r['ate_ud_fixed']:.4f} | {r['ate_off_orig']:.4f}->{r['ate_off_fixed']:.4f} |")
