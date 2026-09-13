#!/usr/bin/env python3
"""Apply the causal pose post-process to CUT3R predictions and score with the OFFICIAL evaluator.

Two operating points, both strictly causal, both pure post-processing of the emitted pose stream:
  posefix      subtract a constant per-step translation bias, then damp step t by 1/(1+t/T)
  posefix_ema  the same, then a causal EMA on the increments (world-frame translation, rotation-vector)

Depth is symlinked from the source label: this method never touches depth, so absrel/a1 must come out
identical to the source run. That identity is a built-in correctness check on the whole sweep.

The bias is a single 3-vector calibrated offline on --calib_scenes; nothing else is fitted.
"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np
from multiprocessing import Pool

OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
EVAL = "/gpfs/home/koc/koc821022/maks_idea/eval_bundle/bin/eval_depth_poses.py"
T_GAIN = 200.0
EMA_T, EMA_R = 0.25, 0.40


def load_poses(d, n):
    P = np.full((n, 4, 4), np.nan)
    for t in range(n):
        p = f"{d}/{t:06d}.npz"
        if not os.path.isfile(p):
            continue
        z = np.load(p)
        M = np.asarray(z["pose"] if "pose" in z else z[list(z.keys())[0]], float)
        if M.shape == (3, 4):
            M = np.vstack([M, [0, 0, 0, 1]])
        P[t] = M
    return P


def umeyama_scale(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, s, Vt = np.linalg.svd(D.T @ S / len(src))
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    var = (S ** 2).sum() / len(src)
    return float(np.trace(np.diag(s) @ W) / var) if var > 0 else 1.0


def rotvec(R):
    a = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if a < 1e-9:
        return np.zeros(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    nn = np.linalg.norm(ax)
    return np.zeros(3) if nn < 1e-12 else ax / nn * a


def rotmat(v):
    a = np.linalg.norm(v)
    if a < 1e-9:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def n_frames(scene, root=ROOT):
    d = f"{root}/{scene}/dense/rgb"
    return len([x for x in os.listdir(d) if x.endswith((".png", ".jpg"))]) if os.path.isdir(d) else 0


def calib_one(args):
    scene, pred_base, root = args
    n = n_frames(scene, root)
    if n < 3:
        return None
    gt = load_poses(f"{root}/{scene}/dense/cam", n)
    pr = load_poses(f"{pred_base}/{scene}/camera", n)
    ok = [t for t in range(n) if np.all(np.isfinite(pr[t])) and np.all(np.isfinite(gt[t]))]
    if len(ok) < 3:
        return None
    s = umeyama_scale(pr[ok][:, :3, 3], gt[ok][:, :3, 3])
    if not np.isfinite(s) or s <= 0:
        return None
    d = []
    for a, b in zip(ok, ok[1:]):
        if b != a + 1:
            continue
        d.append((np.linalg.inv(pr[a]) @ pr[b])[:3, 3] - (np.linalg.inv(gt[a]) @ gt[b])[:3, 3] / s)
    return np.mean(d, 0) if d else None


def correct(pr, n, bias, ema):
    out = pr.copy()
    ew = er = None
    for t in range(1, n):
        if not (np.all(np.isfinite(out[t - 1])) and np.all(np.isfinite(pr[t]))):
            continue
        T = (np.linalg.inv(pr[t - 1]) @ pr[t]).copy()
        T[:3, 3] = (T[:3, 3] - bias) * (1.0 / (1.0 + t / T_GAIN))
        if ema:
            Rw = out[t - 1][:3, :3]
            w = Rw @ T[:3, 3]
            ew = w if ew is None else EMA_T * w + (1 - EMA_T) * ew
            T[:3, 3] = Rw.T @ ew
            rv = rotvec(T[:3, :3])
            er = rv if er is None else EMA_R * rv + (1 - EMA_R) * er
            T[:3, :3] = rotmat(er)
        out[t] = out[t - 1] @ T
    return out


def run_one(args):
    scene, pred_base, label_root, ema, bias = args
    n = n_frames(scene)
    ecsv = f"{label_root}/eval/{scene}/eval_depth_pose_metrics.csv"
    if os.path.isfile(ecsv) and os.path.getsize(ecsv) > 0:
        return scene, "cached"
    if n < 3:
        return scene, "skip"
    pr = load_poses(f"{pred_base}/{scene}/camera", n)
    if not np.isfinite(pr).any():
        return scene, "nopred"
    cor = correct(pr, n, bias, ema)
    cdir = f"{label_root}/preds/{scene}/camera"
    os.makedirs(cdir, exist_ok=True)
    for t in range(n):
        if not np.all(np.isfinite(cor[t])):
            continue
        src = f"{pred_base}/{scene}/camera/{t:06d}.npz"
        K = np.load(src)["intrinsics"] if "intrinsics" in np.load(src) else None
        if K is None:
            np.savez(f"{cdir}/{t:06d}.npz", pose=cor[t])
        else:
            np.savez(f"{cdir}/{t:06d}.npz", pose=cor[t], intrinsics=K)
    dlink = f"{label_root}/preds/{scene}/depth"
    if not os.path.exists(dlink):
        try:
            os.symlink(f"{pred_base}/{scene}/depth", dlink)
        except FileExistsError:
            pass
    os.makedirs(os.path.dirname(ecsv), exist_ok=True)
    r = subprocess.run([sys.executable, EVAL, "--pred_root", f"{label_root}/preds/{scene}",
                        "--gt_root", f"{ROOT}/{scene}/dense", "--output_csv", ecsv],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(ecsv):
        return scene, f"FAIL {r.stderr[-160:]}"
    return scene, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=f"{OUT}/scene_list.txt")
    ap.add_argument("--calib_scenes", default=f"{OUT}/maks_windows100/scenes_100.txt")
    ap.add_argument("--pred_base", default=f"{OUT}/augfull_lr1e5/preds")
    ap.add_argument("--calib_root", default=ROOT,
                    help="GT split root for the CALIBRATION scenes (default: the test root). "
                         "Point at the train split to keep test labels out of the fit.")
    ap.add_argument("--calib_pred_base", default=None,
                    help="prediction tree for the calibration scenes (default: --pred_base). "
                         "Required when --calib_root is a different split.")
    ap.add_argument("--label", required=True)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--procs", type=int, default=60)
    a = ap.parse_args()
    scenes = [l.strip() for l in open(a.scenes) if l.strip()]
    calib = [l.strip() for l in open(a.calib_scenes) if l.strip()]
    root = f"{OUT}/{a.label}"
    os.makedirs(root, exist_ok=True)
    t0 = time.time()
    bp = f"{root}/bias.json"
    if os.path.isfile(bp):
        bias = np.array(json.load(open(bp))["bias"])
    else:
        cpb = a.calib_pred_base or a.pred_base
        with Pool(min(a.procs, 16)) as p:
            bs = [x for x in p.map(calib_one, [(s, cpb, a.calib_root) for s in calib]) if x is not None]
        if len(bs) < 10:
            raise SystemExit(f"only {len(bs)} calibration scenes usable under {cpb}")
        bias = np.mean(bs, 0)
        json.dump({"bias": bias.tolist(), "n_calib_scenes": len(bs), "T_gain": T_GAIN,
                   "ema": bool(a.ema), "ema_trans": EMA_T, "ema_rot": EMA_R,
                   "calib_list": a.calib_scenes, "pred_base": a.pred_base,
                   "calib_root": a.calib_root, "calib_pred_base": cpb}, open(bp, "w"), indent=1)
    print(f"[{a.label}] bias={bias} norm={np.linalg.norm(bias):.6f}  ({time.time()-t0:.0f}s)", flush=True)
    with Pool(a.procs) as p:
        done = 0
        for scene, st in p.imap_unordered(run_one, [(s, a.pred_base, root, a.ema, bias) for s in scenes], chunksize=4):
            done += 1
            if st.startswith("FAIL"):
                print(f"  {scene}: {st}", flush=True)
            if done % 500 == 0:
                print(f"  {done}/{len(scenes)}  {time.time()-t0:.0f}s", flush=True)
    n = len([1 for s in scenes if os.path.isfile(f"{root}/eval/{s}/eval_depth_pose_metrics.csv")])
    print(f"[{a.label}] scored {n}/{len(scenes)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
