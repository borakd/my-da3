#!/usr/bin/env python
"""Aggregate t0_vggt3cam_forward.py outputs into the per-reference table.

Standard columns come straight from each scene's eval_depth_pose_metrics.csv
(eval_depth_poses.py defaults): the ALL/MEAN row (ATE, RPE_trans, RPE_rot =
RMSE over the 3 frames / 2 consecutive pairs of the canonical order
wrist->ext1->ext2), averaged unweighted over scenes (aggregate_results.py
convention). Per-camera ATE = that scene's per-frame Sim3-aligned centre error.
Paired scene-bootstrap 95% CIs of each ordering minus wrist_first
(maks_subset_compare.boot_ci, 10k resamples).

Supplementary alignment-free pairwise columns from the saved c2w poses, over
the 3 camera pairs of each scene: relative rotation error (RRE) and the angle
between predicted and GT relative camera-centre directions (RTE, no sign
folding), RRA/RTA@5 and @15 over pairs, and AUC@30 of max(RRE, RTE) per pair.
"""
import os, csv, argparse, json
import numpy as np

ORDERS = ['wrist_first', 'ext1_first', 'ext2_first']
CANON = ['wrist0', 'ext1', 'ext2']
PAIRS = [(0, 1), (0, 2), (1, 2)]
M = ['ate', 'rpe_trans', 'rpe_rot']
DM = ['absrel', 'a1']  # wrist frame only; present with --eval_dir eval_depth


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def read_scene(csv_path):
    if not (os.path.exists(csv_path) and os.path.getsize(csv_path)):
        return None
    rows = list(csv.DictReader(open(csv_path)))
    out = {}
    for r in rows:
        if r['camera_id'] == 'ALL' and r['local_timestep'] == 'MEAN':
            out.update({m: float(r[m]) for m in M + DM})
        elif r['camera_id'] == '0' and r['local_timestep'] in ('0', '1', '2'):
            out[f'ate_{CANON[int(r["local_timestep"])]}'] = float(r['ate'])
    return out if all(m in out for m in M) else None


def load_c2w(d):
    return [np.load(f'{d}/camera/{i:06d}.npz')['pose'] for i in range(3)]


def ang(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def dir_err(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))))


def pair_errors(pred, gt):
    out = []
    for i, j in PAIRS:
        rp, rg = np.linalg.inv(pred[i]) @ pred[j], np.linalg.inv(gt[i]) @ gt[j]
        out.append((ang(rp[:3, :3].T @ rg[:3, :3]), dir_err(rp[:3, 3], rg[:3, 3])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--scene_list', required=True)
    ap.add_argument('--out_md', required=True)
    ap.add_argument('--eval_dir', default='eval')
    args = ap.parse_args()
    scenes = [l.strip() for l in open(args.scene_list) if l.strip()]
    per = {o: {} for o in ORDERS}
    pairs = {o: {} for o in ORDERS}
    for o in ORDERS:
        for s in scenes:
            r = read_scene(f'{args.out_root}/{o}/{args.eval_dir}/{s}/eval_depth_pose_metrics.csv')
            if r is None:
                continue
            per[o][s] = r
            pairs[o][s] = pair_errors(load_c2w(f'{args.out_root}/{o}/preds/{s}'), load_c2w(f'{args.out_root}/gt/{s}'))
    common = [s for s in scenes if all(s in per[o] for o in ORDERS)]
    L = [f'# VGGT-Omega t=0 three-camera poses, DROID test split', '',
         f'scenes listed {len(scenes)}; scored by all three orderings {len(common)}; '
         + '; '.join(f'{o} missing {len(scenes) - len(per[o])}' for o in ORDERS), '',
         'Standard metrics (eval_depth_poses.py defaults, Sim3 on camera centres, per-scene RMSE over the 3 frames / '
         '2 consecutive pairs wrist->ext1->ext2, unweighted mean over scenes; metres / degrees):', '',
         '| reference | n | AbsRel (wrist) | d<1.25 (wrist) | ATE | RPE trans | RPE rot | ATE wrist | ATE ext1 | ATE ext2 | med ATE | med RPE rot |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    stats = {}
    for o in ORDERS:
        a = {k: np.array([per[o][s][k] for s in common]) for k in M + DM + [f'ate_{c}' for c in CANON]}
        stats[o] = a
        L.append(f'| {o} | {len(common)} | {np.nanmean(a["absrel"]):.4f} | {np.nanmean(a["a1"]):.4f} | {a["ate"].mean():.4f} | {a["rpe_trans"].mean():.4f} | {a["rpe_rot"].mean():.2f} | '
                 f'{a["ate_wrist0"].mean():.4f} | {a["ate_ext1"].mean():.4f} | {a["ate_ext2"].mean():.4f} | '
                 f'{np.median(a["ate"]):.4f} | {np.median(a["rpe_rot"]):.2f} |')
    L += ['', 'Paired difference vs wrist_first (mean, 95% scene-bootstrap CI, share of scenes where the ordering is better):', '',
          '| reference | dATE | dRPE trans | dRPE rot |', '|---|---|---|---|']
    for o in ORDERS[1:]:
        cells = []
        for m in M:
            d = stats[o][m] - stats['wrist_first'][m]
            lo, hi = boot_ci(d)
            f = '{:+.2f}' if m == 'rpe_rot' else '{:+.4f}'
            sig = '*' if (lo > 0 or hi < 0) else ''
            cells.append(f'{f.format(d.mean())}{sig} [{f.format(lo)}, {f.format(hi)}], better {np.mean(d < 0) * 100:.0f}%')
        L.append(f'| {o} | ' + ' | '.join(cells) + ' |')
    L += ['', 'Supplementary, alignment-free over the 3 camera pairs per scene (wrist-ext1, wrist-ext2, ext1-ext2):', '',
          '| reference | mean RRE | med RRE | mean RTE | med RTE | RRA@5 | RRA@15 | RTA@5 | RTA@15 | AUC@30 | scenes all pairs RRE<5 |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for o in ORDERS:
        E = np.array([pairs[o][s] for s in common])  # (n, 3, 2)
        rre, rte = E[..., 0].ravel(), E[..., 1].ravel()
        mx = np.maximum(rre, rte)
        auc = np.mean([np.mean(mx < t) for t in range(1, 31)])
        allok = np.mean((E[..., 0] < 5).all(axis=1))
        L.append(f'| {o} | {rre.mean():.2f} | {np.median(rre):.2f} | {rte.mean():.2f} | {np.median(rte):.2f} | '
                 f'{np.mean(rre < 5) * 100:.1f}% | {np.mean(rre < 15) * 100:.1f}% | {np.mean(rte < 5) * 100:.1f}% | '
                 f'{np.mean(rte < 15) * 100:.1f}% | {auc * 100:.1f} | {allok * 100:.1f}% |')
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    open(args.out_md, 'w').write('\n'.join(L) + '\n')
    with open(args.out_md[:-3] + '_per_scene.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['scene', 'order'] + M + [f'ate_{c}' for c in CANON] + [f'rre_{i}{j}' for i, j in PAIRS] + [f'rte_{i}{j}' for i, j in PAIRS])
        for s in common:
            for o in ORDERS:
                r = per[o][s]; E = pairs[o][s]
                w.writerow([s, o] + [r[m] for m in M] + [r[f'ate_{c}'] for c in CANON] + [e[0] for e in E] + [e[1] for e in E])
    print('\n'.join(L))


if __name__ == '__main__':
    main()
