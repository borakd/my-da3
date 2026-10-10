#!/usr/bin/env python
"""Re-score existing 4292 eval labels with and without the faulty-wrist-GT scenes.

Per label: per-scene ALL/MEAN rows of eval_depth_pose_metrics.csv (the
aggregate_results.py convention), unweighted mean over scenes for
  all     the 4292 split (cross-checked against master_4292.csv when present)
  clean   minus the 186 faulty-wrist-GT scenes (wristgt_check/faulty_wrist_gt.csv)
  faulty  the 186, and split into rgb_upside_down / offset_flipped
Writes <out>/rescore_per_label.csv and <out>/per_scene_<label>.npz.
"""
import os, csv, argparse
import numpy as np
from multiprocessing import Pool

OUT = '/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval'
M = ['absrel', 'a1', 'ate', 'rpe_trans', 'rpe_rot']
LABELS = {  # label -> eval dir (None = $OUT/<label>/eval)
    'cut3r_zeroshot': None, 'augfull_lr1e5': None, 'prevpred_lr1e5': None, 'cgtrain_plain': None,
    'cgtrain_g7ema': None, 'augfull_cg_g7ema': None, 'augfull_cg_g7ema85': None, 'augfull_cg_g7ema_bwd': None,
    'augfull_cg_fuse_g7': None, 'gtray_lr1e5': None, 'prevgt_lr1e5': None, 'vo_full_zs': None, 'vo_full_ft': None,
    'vo_full_ftgt': None, 'scal3r_blk120_xyz_amp': None, 'causal_posefix': None, 'causal_posefix_ema': None,
    'causal_posefix_tc': None, 'causal_posefix_tc_ema': None, 'vggtft_pool_frz': None, 'vggtft_pool_unf': None,
    'vggtft_raw_frz': None, 'vggtft_raw_unf': None,
    'tok_uncA_q50_g50': f'{OUT}/unc_full4292/tok_uncA_q50_g50/eval',
    'tok_al_q50_g50': f'{OUT}/unc_full4292/tok_al_q50_g50/eval',
    'tok_cbG1L_q50_g50': f'{OUT}/unc_full4292/tok_cbG1L_q50_g50/eval',
    'tf_cut3r_pretrained': None, 'tf_ttt3r_pretrained': None, 'tf_raymap3r_pretrained': None,
    'tf_recal3r_pretrained': None, 'cgA_ttt3r': None, 'cgA_raymap3r': None, 'cgA_recal3r': None,
    'cg1_token_final': f'{OUT}/wgate4292/cg1_token_final/eval',
    'scal3r_droid_full_320_final': None, 'scal3r_droid_full_320_ftbb_final': None,
}


def scene_row(path):
    try:
        rows = list(csv.DictReader(open(path)))
    except OSError:
        return None
    for want in (lambda r: r['camera_id'] == 'ALL' and r['local_timestep'] == 'MEAN',
                 lambda r: r['local_timestep'] == 'MEAN'):
        for r in rows:
            if want(r):
                return [float(r[m]) if r[m] not in ('', 'nan') else np.nan for m in M]
    return None


def load(args):
    label, d, scenes = args
    d = d or f'{OUT}/{label}/eval'
    X = np.full((len(scenes), len(M)), np.nan)
    for i, s in enumerate(scenes):
        r = scene_row(f'{d}/{s}/eval_depth_pose_metrics.csv')
        if r is not None:
            X[i] = r
    return label, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--faulty_csv', required=True)
    ap.add_argument('--procs', type=int, default=36)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scenes = [l.strip() for l in open(f'{OUT}/scene_list.txt') if l.strip()]
    cat = {r['scene']: r['category'] for r in csv.DictReader(open(args.faulty_csv))
           if r['category'] in ('rgb_upside_down', 'offset_flipped')}
    master = {r['setup']: r for r in csv.DictReader(open(f'{OUT}/master/master_4292.csv'))}
    with Pool(args.procs) as p:
        res = dict(p.map(load, [(l, d, scenes) for l, d in LABELS.items()]))
    c = np.array([cat.get(s, 'clean') for s in scenes])
    masks = {'all': np.ones(len(scenes), bool), 'clean': c == 'clean', 'faulty': c != 'clean',
             'upside_down': c == 'rgb_upside_down', 'offset_flipped': c == 'offset_flipped'}
    out = []
    for label, X in res.items():
        np.savez(f'{args.out}/per_scene_{label}.npz', X=X, scenes=np.array(scenes), metrics=np.array(M))
        row = {'label': label, 'n_missing': int(np.isnan(X[:, 2]).sum())}
        for k, m in masks.items():
            for j, met in enumerate(M):
                v = X[m, j]
                row[f'{met}_{k}'] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        if label in master:
            row['master_maxdiff'] = max(abs(row[f'{met}_all'] - float(master[label][met])) for met in M)
        out.append(row)
    with open(f'{args.out}/rescore_per_label.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()) + (['master_maxdiff'] if 'master_maxdiff' not in out[0] else []))
        w.writeheader(); w.writerows(out)
    print({k: int(v.sum()) for k, v in masks.items()})
    for r in out:
        print(f"{r['label']:34s} miss {r['n_missing']:4d} master_diff {r.get('master_maxdiff', float('nan')):.1e} "
              f"ATE all {r['ate_all']:.4f} clean {r['ate_clean']:.4f} faulty {r['ate_faulty']:.4f} | "
              f"RPErot all {r['rpe_rot_all']:.3f} clean {r['rpe_rot_clean']:.3f} faulty {r['rpe_rot_faulty']:.3f} | "
              f"AbsRel all {r['absrel_all']:.4f} clean {r['absrel_clean']:.4f} ud {r['absrel_upside_down']:.4f} off {r['absrel_offset_flipped']:.4f}")


if __name__ == '__main__':
    main()
