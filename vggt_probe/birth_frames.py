#!/usr/bin/env python
"""Gripper "birth frame" detection for DROID episodes via kinematics + projection.

Birth frame := first frame t at which the gripper is visible in BOTH exterior cameras at once.

Method
  1. Per-frame gripper pose. The Leonardo store (`<scene>/dense/cam/<t>.npz` "pose") holds the
     wrist-camera pose in the robot base frame for every frame. It is exactly the robot's own
     forward kinematics (trajectory.h5 `cartesian_position`) composed with a constant hand-eye
     offset (verified on RAIL+80edfcb1: residual 0.00 mm / 0.000 deg over all frames), so no
     trajectory download is needed. The ZED Mini is bolted to the Robotiq 2F-85, so a point
     model of the gripper expressed in the WRIST-CAMERA frame is rigid hardware geometry.
  2. Gripper point model: a box in the EE (flange) frame, x,y in +-HALF_W, z in [Z0, Z1] along
     the approach axis (fingertips ~0.16-0.18 m from the flange), mapped once into the wrist-
     camera frame with T_ee_cam calibrated from RAIL+80edfcb1 (`--calibrate`).
  3. Exterior cameras: `--extrinsics pointworld` (default) takes the PointWorld `optimized_extrinsics`
     (<ep>_cameras.json, keyed by serial, 4x4 base->camera used as-is); `--extrinsics raw` takes
     `ext{1,2}_cam_extrinsics` from the DROID raw metadata json (cam->base, xyz + scipy 'xyz' Euler;
     convention verified against the store; wrong for ~23 % of cameras, see recheck_pointworld/).
     Intrinsics: per-serial ZED factory calibration (calib.stereolabs.com/?SN=<serial>, [LEFT_CAM_HD],
     MP4s are 1280x720).
  4. A point is "seen" if it projects inside the image with positive depth (occlusion by the
     arm/objects is NOT modelled). Per frame we record the fraction of points seen by each
     camera; the birth frame at threshold f is the first t with both fractions >= f.

Usage
  birth_frames.py --calibrate                          # writes ee_cam_offset.json from RAIL h5
  birth_frames.py --episodes EP [EP ...] [--viz FRAMES] # per-episode report (+ overlay images)
  birth_frames.py --all --workers 16                   # all scenes in the test split -> csv/json
"""
import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial.transform import Rotation as R

STORE = "/leonardo_scratch/large/userexternal/bdursun0/pointworld_droid_splits/test/dl3dv_multi/wrist"
OUT_ROOT = "/leonardo_work/AIFAC_S07_110/bora/outputs/droid_birth_frames"
META_DIR = f"{OUT_ROOT}/metadata"
CALIB_DIR = f"{OUT_ROOT}/zed_calib"
CAMERAS_DIR = "/leonardo_scratch/large/userexternal/bdursun0/ext_build/meta/cameras"  # PointWorld <ep>_cameras.json
EXTRINSICS = "pointworld"  # default source of the exterior extrinsics ('raw' | 'pointworld'), see load_T_bc
HERE = os.path.dirname(os.path.abspath(__file__))
EE_CAM_JSON = os.path.join(HERE, "ee_cam_offset.json")
RAIL_EP = "RAIL+80edfcb1+2023-07-14-14h-28m-45s"
RAIL_RAW = f"/leonardo_scratch/large/userexternal/bdursun0/robotseg_demo/raw/{RAIL_EP}"
W, H = 1280, 720

# Robotiq 2F-85 + DROID coupling, in the EE (flange) frame: approach axis = +z.
HALF_W = 0.045     # half width/thickness of the body incl. open fingers (m)
Z0, Z1 = 0.02, 0.18  # from just below the flange to the fingertips (m)
GRID = (3, 3, 6)   # points along x, y, z -> 54 points
THRESHOLDS = (0.25, 0.5, 0.75, 1.0)


def pose6_to_T(p):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", p[3:6]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def load_T_bc(ep, meta, tag, extrinsics=None):
    """4x4 base->camera for exterior camera `tag` ('ext1'/'ext2') of episode `ep`.
    'raw': DROID metadata `<tag>_cam_extrinsics` (cam->base, inverted); 'pointworld': PointWorld
    `optimized_extrinsics` from CAMERAS_DIR (base->camera, as-is). KeyError if the serial is absent."""
    extrinsics = extrinsics or EXTRINSICS
    if extrinsics == "raw":
        return np.linalg.inv(pose6_to_T(np.array(meta[f"{tag}_cam_extrinsics"])))
    cams = json.load(open(f"{CAMERAS_DIR}/{ep}_cameras.json"))
    return np.array(cams[str(meta[f"{tag}_cam_serial"])]["optimized_extrinsics"], dtype=np.float64)


def gripper_points_ee():
    xs = np.linspace(-HALF_W, HALF_W, GRID[0])
    ys = np.linspace(-HALF_W, HALF_W, GRID[1])
    zs = np.linspace(Z0, Z1, GRID[2])
    P = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    return P


def load_ee_cam():
    return np.array(json.load(open(EE_CAM_JSON))["T_ee_cam"])


def gripper_points_cam():
    """Gripper model in the wrist-camera frame (rigid hardware offset)."""
    T_ee_cam = load_ee_cam()
    T_cam_ee = np.linalg.inv(T_ee_cam)
    P = gripper_points_ee()
    return (T_cam_ee[:3, :3] @ P.T).T + T_cam_ee[:3, 3]


def load_zed_K(sn):
    path = f"{CALIB_DIR}/SN{sn}.conf"
    txt = open(path).read()
    sec = txt.split("[LEFT_CAM_HD]")[1].split("[")[0]
    d = dict(re.findall(r"(\w+)=([-\d.eE+]+)", sec))
    return np.array([[float(d["fx"]), 0, float(d["cx"])], [0, float(d["fy"]), float(d["cy"])], [0, 0, 1]])


def load_store_poses(ep):
    cam_dir = f"{STORE}/{ep}/dense/cam"
    files = sorted(f for f in os.listdir(cam_dir) if f.endswith(".npz"))
    poses = []
    for f in files:
        with np.load(os.path.join(cam_dir, f)) as z:
            poses.append(z["pose"])
    return np.stack(poses)


def project(T_bc, K, P):
    Pc = (T_bc[:3, :3] @ P.T).T + T_bc[:3, 3]
    z = Pc[:, 2]
    uv = (K @ Pc.T).T
    uv = uv[:, :2] / np.where(np.abs(uv[:, 2:3]) < 1e-9, 1e-9, uv[:, 2:3])
    return uv, z


def visible_fraction(T_bc, K, P_base):
    uv, z = project(T_bc, K, P_base)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return ok.mean(), uv, ok


def analyze_episode(ep, P_cam=None, keep_curves=True):
    """Returns a dict with per-threshold birth frames and the per-frame visible fractions."""
    out = {"episode": ep}
    meta_path = f"{META_DIR}/{ep}.json"
    if not os.path.isfile(meta_path):
        out["error"] = "no_metadata"
        return out
    meta = json.load(open(meta_path))
    try:
        poses = load_store_poses(ep)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"store:{e}"
        return out
    if P_cam is None:
        P_cam = gripper_points_cam()
    T = len(poses)
    out["n_frames"] = T
    out["trajectory_length_meta"] = meta.get("trajectory_length")
    cams = {}
    for tag in ("ext1", "ext2"):
        sn = meta[f"{tag}_cam_serial"]
        try:
            K = load_zed_K(sn)
        except Exception as e:  # noqa: BLE001
            out["error"] = f"calib:{sn}:{e}"
            return out
        try:
            T_bc = load_T_bc(ep, meta, tag)
            out[f"{tag}_extrinsics"] = EXTRINSICS
        except KeyError:  # serial absent from the PointWorld cameras file: fall back to the raw metadata
            T_bc = load_T_bc(ep, meta, tag, "raw")
            out[f"{tag}_extrinsics"] = "raw_fallback"
        cams[tag] = (sn, K, T_bc)
        out[f"{tag}_serial"] = sn
    # gripper points in base frame for every frame: (T, N, 3)
    Pb = np.einsum("tij,nj->tni", poses[:, :3, :3], P_cam) + poses[:, None, :3, 3]
    fr = {}
    for tag, (sn, K, T_bc) in cams.items():
        f = np.array([visible_fraction(T_bc, K, Pb[t])[0] for t in range(T)])
        fr[tag] = f
        out[f"{tag}_max_frac"] = float(f.max())
    both = np.minimum(fr["ext1"], fr["ext2"])
    for thr in THRESHOLDS:
        idx = np.where(both >= thr)[0]
        out[f"birth_f{int(thr*100):03d}"] = int(idx[0]) if len(idx) else -1
        for tag in ("ext1", "ext2"):
            idx_c = np.where(fr[tag] >= thr)[0]
            out[f"{tag}_first_f{int(thr*100):03d}"] = int(idx_c[0]) if len(idx_c) else -1
    # how "stable" is visibility after birth at 0.5: fraction of later frames still >= 0.5 in both
    b = out["birth_f050"]
    out["post_birth_both_visible_frac"] = float((both[b:] >= 0.5).mean()) if b >= 0 else 0.0
    if keep_curves:
        out["curves"] = {k: np.round(v, 3).tolist() for k, v in fr.items()}
    return out


def visualize(ep, frames, raw_dir, out_dir):
    """Draw projected gripper points on the exterior MP4 frames (frames dir from the RobotSeg
    demo, or decode from the MP4 directly)."""
    import cv2
    meta = json.load(open(f"{META_DIR}/{ep}.json"))
    poses = load_store_poses(ep)
    P_cam = gripper_points_cam()
    os.makedirs(out_dir, exist_ok=True)
    for tag in ("ext1", "ext2"):
        sn = meta[f"{tag}_cam_serial"]
        K = load_zed_K(sn)
        T_bc = load_T_bc(ep, meta, tag)
        cap = cv2.VideoCapture(f"{raw_dir}/recordings/MP4/{sn}.mp4")
        for t in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, img = cap.read()
            if not ok:
                continue
            Pb = (poses[t][:3, :3] @ P_cam.T).T + poses[t][:3, 3]
            frac, uv, vis = visible_fraction(T_bc, K, Pb)
            for (u, v), ok_ in zip(uv, vis):
                if ok_:
                    cv2.circle(img, (int(u), int(v)), 4, (0, 255, 0), -1)
            cv2.putText(img, f"{ep} {tag} f{t} visible {frac*100:.0f}%", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imwrite(f"{out_dir}/{tag}_f{t:05d}.jpg", img)
        cap.release()


def calibrate():
    """T_ee_cam from the RAIL episode: store wrist-cam pose vs trajectory.h5 EE pose."""
    import h5py
    with h5py.File(f"{RAIL_RAW}/trajectory.h5") as f:
        cart = f["observation/robot_state/cartesian_position"][:]
    poses = load_store_poses(RAIL_EP)
    n = len(poses)
    Ts_ee = np.stack([pose6_to_T(c) for c in cart[:n]])
    T_ee_cam = np.linalg.inv(Ts_ee[0]) @ poses[0]
    pred = Ts_ee @ T_ee_cam
    dt = np.linalg.norm(pred[:, :3, 3] - poses[:, :3, 3], axis=1).max() * 1000
    dr = max(np.degrees(R.from_matrix(pred[i, :3, :3].T @ poses[i, :3, :3]).magnitude()) for i in range(n))
    json.dump({"source": RAIL_EP, "T_ee_cam": T_ee_cam.tolist(),
               "max_residual_mm": float(dt), "max_residual_deg": float(dr)},
              open(EE_CAM_JSON, "w"), indent=1)
    print(f"T_ee_cam written to {EE_CAM_JSON}; residual over {n} frames: {dt:.3f} mm, {dr:.4f} deg")


def _worker(ep):
    try:
        return analyze_episode(ep, keep_curves=False)
    except Exception as e:  # noqa: BLE001
        return {"episode": ep, "error": f"exc:{e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--episodes", nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help="every scene dir in the test split")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--viz", nargs="*", type=int, default=None, help="frames to draw (needs --raw_dir)")
    ap.add_argument("--raw_dir", default=None, help="<raw>/<EP> dir holding recordings/MP4/*.mp4")
    ap.add_argument("--viz_out", default=None)
    ap.add_argument("--out", default=f"{OUT_ROOT}/birth_frames")
    ap.add_argument("--extrinsics", choices=("raw", "pointworld"), default="pointworld",
                    help="exterior extrinsics: PointWorld optimized_extrinsics (default) or the raw DROID metadata")
    args = ap.parse_args()
    global EXTRINSICS
    EXTRINSICS = args.extrinsics  # inherited by the forked workers

    if args.calibrate:
        calibrate()
        return

    if args.episodes and not args.all:
        for ep in args.episodes:
            r = analyze_episode(ep)
            curves = r.pop("curves", None)
            print(json.dumps(r, indent=1))
            if curves:
                print("frame : ext1 ext2 visible fraction")
                for t, (a, b) in enumerate(zip(curves["ext1"], curves["ext2"])):
                    if a > 0 or b > 0:
                        print(f"{t:5d} : {a:.2f} {b:.2f}")
            if args.viz is not None:
                visualize(ep, args.viz, args.raw_dir, args.viz_out or f"{OUT_ROOT}/viz/{ep}")
        return

    if args.all:
        eps = sorted(d for d in os.listdir(STORE) if os.path.isdir(f"{STORE}/{d}"))
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        # resumable: one JSON object per line, appended as scenes finish; a restart skips them
        jsonl = args.out + ".jsonl"
        done = {}
        if os.path.isfile(jsonl):
            for line in open(jsonl):
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if "error" not in r:
                        done[r["episode"]] = r
        todo = [ep for ep in eps if ep not in done]
        print(f"{len(eps)} scenes, {len(done)} already done, {len(todo)} to do", flush=True)
        rows = list(done.values())
        with open(jsonl, "a") as fh, ProcessPoolExecutor(args.workers) as ex:
            futs = {ex.submit(_worker, ep): ep for ep in todo}
            for i, fut in enumerate(as_completed(futs)):
                r = fut.result()
                rows.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if (i + 1) % 250 == 0:
                    print(f"  {i+1}/{len(todo)}", flush=True)
        rows.sort(key=lambda r: r["episode"])
        json.dump(rows, open(args.out + ".json", "w"), indent=0)
        keys = ["episode", "n_frames", "ext1_serial", "ext2_serial", "error",
                "ext1_max_frac", "ext2_max_frac", "post_birth_both_visible_frac", "ext1_extrinsics", "ext2_extrinsics"]
        for thr in THRESHOLDS:
            k = f"f{int(thr*100):03d}"
            keys += [f"birth_{k}", f"ext1_first_{k}", f"ext2_first_{k}"]
        with open(args.out + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})
        ok = [r for r in rows if "error" not in r]
        print(f"done: {len(ok)} analysed, {len(rows)-len(ok)} errors -> {args.out}.csv/.json")


if __name__ == "__main__":
    main()
