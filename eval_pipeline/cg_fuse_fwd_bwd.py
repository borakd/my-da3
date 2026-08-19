#!/usr/bin/env python3
"""Forward/backward trajectory fusion for the CUT3R eval harness.

The same checkpoint is run twice per scene: once in normal frame order (the
forward label) and once with REVERSE=1 in infer_and_eval_worker_ray.py (the
backward label, whose depth/camera files are already saved under ORIGINAL
frame indices). The two passes see identical per-frame inputs but opposite
memory trajectories, so their errors are complementary: forward drifts late,
backward drifts early. This script fuses the two camera trajectories per
scene and scores the fused result with the standard eval script.

Per scene:
  1. read fwd + bwd preds (camera/NNNNNN.npz: pose=c2w 4x4, intrinsics=3x3;
     depth/NNNNNN.npy) and require identical frame-index sets;
  2. sim3-align bwd -> fwd with an Umeyama fit on camera centers (same math
     as eval_bundle/bin/eval_depth_poses.py:_estimate_umeyama — the eval's
     own sim3 gauge alignment, so fusion adds no new alignment convention);
  3. fuse per frame: translation = midpoint of the aligned centers, rotation
     = quaternion slerp at t=0.5 (sign-corrected), intrinsics = forward's;
  4. write fused camera npz under OUT/<fused_label>/preds/<scene>/camera and
     SYMLINK the forward depth dir as .../depth (forward depth is scored;
     the eval script reads .npy through directory symlinks fine);
  5. run eval_depth_poses.py (default args, like the inference worker) into
     OUT/<fused_label>/eval/<scene>/eval_depth_pose_metrics.csv.

Resumable (scenes with a non-empty eval CSV are skipped) and shardable
(--shard_id/--num_shards) so several invocations can split the list; simple
sequential within one invocation. CPU/numpy only — no GPU, no torch.

Usage:
  python eval_pipeline/cg_fuse_fwd_bwd.py \
      --fwd_label diag_cg_log430 --bwd_label diag_cg_bwd \
      --fused_label diag_cg_fuse \
      --scene_list eval_pipeline/noise_oracle_subset_430.txt
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

DEF_OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
DEF_SCENES = ("/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/"
              "test/dl3dv_multi/wrist")
DEF_EVAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "eval_bundle", "bin", "eval_depth_poses.py")


# --- sim3 / rotation math ----------------------------------------------------

def estimate_umeyama(src_xyz, dst_xyz, with_scale=True):
    """Least-squares sim3 (s,R,t) with dst ~= s*R@src + t. Same math as
    eval_bundle/bin/eval_depth_poses.py:_estimate_umeyama."""
    src = np.asarray(src_xyz, dtype=np.float64)
    dst = np.asarray(dst_xyz, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Invalid center shapes: src={src.shape}, dst={dst.shape}")
    n = src.shape[0]
    if n < 1:
        raise ValueError("Need at least 1 point for alignment")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / max(n, 1)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt

    if with_scale:
        var_src = np.mean(np.sum(src_c * src_c, axis=1))
        if var_src <= 1e-15:
            scale = 1.0
        else:
            scale = float(np.trace(np.diag(S) @ D) / var_src)
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def rotmat_to_quat(R):
    """3x3 rotation -> unit quaternion [w,x,y,z] (Shepperd's method)."""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s,
                      0.25 * s,
                      (m[0, 1] + m[1, 0]) / s,
                      (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s,
                      (m[0, 1] + m[1, 0]) / s,
                      0.25 * s,
                      (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s,
                      (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s,
                      0.25 * s])
    return q / np.linalg.norm(q)


def quat_to_rotmat(q):
    """Unit quaternion [w,x,y,z] -> 3x3 rotation."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0, q1, t=0.5):
    """Spherical interpolation with hemisphere (sign) correction."""
    q0 = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:  # q and -q are the same rotation; take the short arc
        q1 = -q1
        d = -d
    if d > 0.9995:  # nearly parallel: lerp avoids a 0/0
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    return (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


# --- per-scene fusion --------------------------------------------------------

def load_cams(cam_dir):
    """Return (names, poses Nx4x4, intrinsics Nx3x3) sorted by filename."""
    files = sorted(glob.glob(os.path.join(cam_dir, "*.npz")))
    names, poses, intr = [], [], []
    for f in files:
        with np.load(f) as z:
            poses.append(np.asarray(z["pose"], dtype=np.float64))
            intr.append(np.asarray(z["intrinsics"], dtype=np.float64))
        names.append(os.path.basename(f))
    if not names:
        raise RuntimeError(f"no camera npz files in {cam_dir}")
    return names, np.stack(poses), np.stack(intr)


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def fuse_scene(fwd_pred, bwd_pred, fused_pred,
               max_angle_deg=45.0, max_center_frac=0.25, max_resid_frac=1.0):
    """Fuse one scene's fwd+bwd camera trajectories into fused_pred/camera
    and symlink fwd depth as fused_pred/depth.

    v3 guards, ALL-OR-NOTHING per scene: v1's plain midpoint won rpe_rot on
    411/430 scenes but 19 outliers detonated the mean; v2's per-frame
    fallback was WORSE — mixing forward poses and midpoint poses inside one
    trajectory injects relative-pose jumps at every fallback boundary
    (exactly what RPE measures, evidence/cg_fusion2_430.json). So the
    decision is per scene: if the p95 per-frame rotation disagreement
    exceeds max_angle_deg, or the sim3 alignment residual exceeds
    max_resid_frac x scene extent, the WHOLE scene stays forward; otherwise
    EVERY frame is midpoint-fused. max_center_frac feeds the p95 center
    disagreement into the same scene-level decision. Returns (n_frames,
    n_frame_fallbacks(0|n), scene_fell_back)."""
    f_names, f_poses, f_intr = load_cams(os.path.join(fwd_pred, "camera"))
    b_names, b_poses, _ = load_cams(os.path.join(bwd_pred, "camera"))
    if f_names != b_names:
        raise RuntimeError(
            f"frame mismatch: fwd {len(f_names)} vs bwd {len(b_names)} "
            f"(first diff: {sorted(set(f_names) ^ set(b_names))[:3]})")

    # sim3-align backward onto forward using camera centers.
    s, R, t = estimate_umeyama(b_poses[:, :3, 3], f_poses[:, :3, 3],
                               with_scale=True)
    b_aligned = np.empty_like(b_poses)
    for i in range(len(b_poses)):
        T = np.eye(4)
        T[:3, :3] = R @ b_poses[i, :3, :3]
        T[:3, 3] = s * (R @ b_poses[i, :3, 3]) + t
        b_aligned[i] = T

    f_centers = f_poses[:, :3, 3]
    extent = float(np.sqrt(((f_centers - f_centers.mean(0)) ** 2)
                           .sum(1).mean())) + 1e-9
    resid = float(np.sqrt(((b_aligned[:, :3, 3] - f_centers) ** 2)
                          .sum(1).mean()))
    angles = np.array([rot_angle_deg(f_poses[i, :3, :3],
                                     b_aligned[i, :3, :3])
                       for i in range(len(f_names))])
    dcent = np.linalg.norm(f_centers - b_aligned[:, :3, 3], axis=1)
    scene_fallback = (
        float(np.percentile(angles, 95)) > max_angle_deg
        or float(np.percentile(dcent, 95)) > max_center_frac * extent * 4.0
        or resid > max_resid_frac * extent)

    cam_out = os.path.join(fused_pred, "camera")
    os.makedirs(cam_out, exist_ok=True)
    n_fallback = len(f_names) if scene_fallback else 0
    for i, name in enumerate(f_names):
        if scene_fallback:
            T = f_poses[i].copy()
        else:
            q = slerp(rotmat_to_quat(f_poses[i, :3, :3]),
                      rotmat_to_quat(b_aligned[i, :3, :3]), 0.5)
            T = np.eye(4)
            T[:3, :3] = quat_to_rotmat(q)
            T[:3, 3] = 0.5 * (f_poses[i, :3, 3] + b_aligned[i, :3, 3])
        np.savez(os.path.join(cam_out, name),
                 pose=T.astype(np.float32),
                 intrinsics=f_intr[i].astype(np.float32))

    # Depth = forward's, via a directory symlink (saves ~130 npy copies/scene;
    # eval_depth_poses.py globs pred_root/depth/*.npy, which follows dir
    # symlinks). An existing link/dir from a previous run is left in place.
    depth_link = os.path.join(fused_pred, "depth")
    depth_src = os.path.realpath(os.path.join(fwd_pred, "depth"))
    if not os.path.isdir(depth_src):
        raise RuntimeError(f"forward depth dir missing: {depth_src}")
    if os.path.islink(depth_link):
        if os.path.realpath(depth_link) != depth_src:
            os.unlink(depth_link)
            os.symlink(depth_src, depth_link)
    elif not os.path.exists(depth_link):
        os.symlink(depth_src, depth_link)
    return len(f_names), n_fallback, scene_fallback


def main():
    ap = argparse.ArgumentParser(
        description="Fuse forward+backward trajectories per scene and score.")
    ap.add_argument("--fwd_label", required=True,
                    help="label of the normal-order pass (e.g. diag_cg_log430)")
    ap.add_argument("--bwd_label", required=True,
                    help="label of the REVERSE=1 pass (e.g. diag_cg_bwd)")
    ap.add_argument("--fused_label", required=True,
                    help="output label (e.g. diag_cg_fuse)")
    ap.add_argument("--scene_list", required=True,
                    help="txt of scene names, one per line")
    ap.add_argument("--out_root", default=DEF_OUT,
                    help=f"cut3r_eval root holding <label>/{{preds,eval}} "
                         f"(default {DEF_OUT})")
    ap.add_argument("--scenes_root", default=DEF_SCENES,
                    help="GT scene root; GT per scene is <root>/<scene>/dense")
    ap.add_argument("--eval_script", default=DEF_EVAL)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1,
                    help="split the scene list across parallel invocations")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N scenes (smoke test)")
    ap.add_argument("--skip_eval", action="store_true",
                    help="only write fused preds, do not run the eval script")
    args = ap.parse_args()

    with open(args.scene_list) as f:
        scenes = [ln.strip() for ln in f if ln.strip()]
    scenes = [s for i, s in enumerate(scenes)
              if i % args.num_shards == args.shard_id]
    if args.limit > 0:
        scenes = scenes[: args.limit]

    fwd_base = os.path.join(args.out_root, args.fwd_label, "preds")
    bwd_base = os.path.join(args.out_root, args.bwd_label, "preds")
    fused_base = os.path.join(args.out_root, args.fused_label, "preds")
    eval_base = os.path.join(args.out_root, args.fused_label, "eval")
    os.makedirs(eval_base, exist_ok=True)
    fail_log = os.path.join(eval_base, f"_failures_shard{args.shard_id}.txt")

    tag = f"[fuse {args.fused_label} shard {args.shard_id}/{args.num_shards}]"
    print(f"{tag} {len(scenes)} scenes: {args.fwd_label} + {args.bwd_label} "
          f"-> {args.fused_label}", flush=True)

    done = skipped = failed = 0
    for idx, scene in enumerate(scenes):
        eval_dir = os.path.join(eval_base, scene)
        eval_csv = os.path.join(eval_dir, "eval_depth_pose_metrics.csv")
        if (not args.skip_eval and os.path.isfile(eval_csv)
                and os.path.getsize(eval_csv) > 0):
            skipped += 1
            continue
        try:
            nfr, nfb, sfb = fuse_scene(os.path.join(fwd_base, scene),
                                       os.path.join(bwd_base, scene),
                                       os.path.join(fused_base, scene))
            if sfb or nfb:
                print(f"{tag} {scene}: "
                      + ("SCENE fallback to forward"
                         if sfb else f"{nfb}/{nfr} frame fallbacks"),
                      flush=True)
            if not args.skip_eval:
                os.makedirs(eval_dir, exist_ok=True)
                cmd = [sys.executable, args.eval_script,
                       "--pred_root", os.path.join(fused_base, scene),
                       "--gt_root", os.path.join(args.scenes_root, scene, "dense"),
                       "--output_csv", eval_csv]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0 or not (os.path.isfile(eval_csv)
                                             and os.path.getsize(eval_csv) > 0):
                    raise RuntimeError(
                        f"eval failed rc={r.returncode}: {r.stderr[-500:]}")
            done += 1
            if done <= 3 or done % 50 == 0:
                print(f"{tag} [{idx+1}/{len(scenes)}] {scene} frames={nfr} "
                      f"done={done} skip={skipped} fail={failed}", flush=True)
        except Exception as e:
            failed += 1
            with open(fail_log, "a") as fh:
                fh.write(f"{scene}\t{repr(e)}\n")
            print(f"{tag} FAIL {scene}: {repr(e)[:200]}", flush=True)

    print(f"{tag} DONE done={done} skipped={skipped} failed={failed}",
          flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
