import sys, os, csv, json, numpy as np, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import rig_track as rt, rig_trackon_eval as rte, rig_anchor as ra, rig_match_track as rmt
ep, out, D = sys.argv[1], sys.argv[2], int(sys.argv[3])
b = [int(r['entry_frame']) for r in csv.DictReader(open(ra.EVENTS)) if r['episode'] == ep and r['event'] == '0'][0]
cams, K, E, P, _ = rte.calib(ep, rte.POSITIONS); it = pd.read_csv(f'{ra.MASKS}/items_v2.csv'); it = it[(it.episode == ep) & (it.frame == b)]
col = {c: rt.read_frames(f'{rt.RAW_ROOT}/{ep}/recordings/MP4/{s}.mp4', b + 1, gray=False)[b] for c, s in cams}; gray = {c: cv2.cvtColor(col[c], cv2.COLOR_BGR2GRAY) for c in col}
reg = {}
for c in ('ext1', 'ext2'):
    mm = cv2.imread(f'{ra.MASKS}/{it[it.cam == c].mask_relpath.iloc[0]}', 0); mm = cv2.resize(mm, (rt.W, rt.H), interpolation=cv2.INTER_NEAREST); reg[c] = (mm > 0).astype(np.uint8) * 255
R1, t1, R2, t2 = E['ext1'][:3, :3], E['ext1'][:3, 3], E['ext2'][:3, :3], E['ext2'][:3, 3]; Rr = R2 @ R1.T; trel = t2 - Rr @ t1
F = np.linalg.inv(K['ext2']).T @ rt.skew(trel) @ Rr @ np.linalg.inv(K['ext1'])
dil = {c: cv2.dilate(reg[c], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * D + 1, 2 * D + 1))) for c in reg}
xa, xb, cf = rmt.match_disk_epi(gray['ext1'], gray['ext2'], dil['ext1'], dil['ext2'], F, 640, 2.5); sel, _ = rmt.gate(xa, xb, cf, dil['ext1'], dil['ext2'], F, P, E, 2.5, 0.3)
xa, xb = xa[sel], xb[sel]; inm = reg['ext1'][np.clip(np.round(xa[:, 1]).astype(int), 0, rt.H - 1), np.clip(np.round(xa[:, 0]).astype(int), 0, rt.W - 1)] > 0
print('ext candidates', len(xa), 'inside mask', int(inm.sum()), 'outside', int((~inm).sum()))
tiles = []
for c, x in (('ext1', xa), ('ext2', xb)):
    img = col[c].copy(); cnt, _ = cv2.findContours(reg[c], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(img, cnt, -1, (255, 200, 0), 1)
    cnt2, _ = cv2.findContours(dil[c], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(img, cnt2, -1, (0, 200, 255), 1)
    for (u, v), i_ in zip(x, inm): cv2.circle(img, (int(u), int(v)), 4, (0, 255, 0) if i_ else (0, 0, 255), -1)
    ys, xs = np.nonzero(dil[c]); x0, y0, x1, y1 = max(xs.min() - 10, 0), max(ys.min() - 10, 0), min(xs.max() + 10, rt.W), min(ys.max() + 10, rt.H)
    crop = img[y0:y1, x0:x1]; crop = cv2.resize(crop, (int(crop.shape[1] * 500 / crop.shape[0]), 500)); tiles.append(crop)
cv2.imwrite(out, np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 90])
