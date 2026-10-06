"""Second pass over segment_birth_frames.py results: revise the selection policy.

Finding from QA: when the automatic mask disagrees with the kinematic projection, it is usually
the PROJECTION that is wrong (stale exterior extrinsics in the DROID metadata), and the box
prompt then drags the mask onto background inside the wrong box. New policy:
  final mask = automatic mask, unless it is (near-)empty (area < 0.05 %) -> then the box mask.
  `projection_disagree` = automatic mask inconsistent with the projection (spill > 0.2):
      a per-camera flag that the scene's extrinsics (and hence its birth frame) are suspect.
For rows where the first pass stored the BOX mask, re-run the automatic prompt on the saved
frame, keep both masks (<stem>_auto.png, <stem>_box.png) and rewrite <stem>.png + overlay.
Writes results_refined.jsonl (all rows, updated fields).
"""
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # vggt_probe: birth_frames.py
import birth_frames as bf  # noqa: E402
from robotseg.build_robotseg import build_robotseg_video_predictor  # noqa: E402
import robotseg  # noqa: E402

# RobotSeg clone (editable install). The config name below is resolved by Hydra relative to the
# robotseg package, not the cwd; only the checkpoint is a filesystem path.
ROBOTSEG_ROOT = os.environ.get("ROBOTSEG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(robotseg.__file__)))

RUN = f"{bf.OUT_ROOT}/masks_run"
AREA_MIN = 0.0005
SPILL_MAX = 0.2
PAD = 60


def spill_area(mask, uv, vis):
    m = mask.astype(bool)
    area = float(m.mean())
    if vis.sum() and m.sum():
        x0, y0 = np.maximum(uv[vis].min(0) - PAD, 0).astype(int)
        x1, y1 = np.minimum(uv[vis].max(0) + PAD, [m.shape[1] - 1, m.shape[0] - 1]).astype(int)
        spill = float(1 - m[y0:y1 + 1, x0:x1 + 1].sum() / m.sum())
    else:
        spill = float("nan")
    return spill, area


def overlay(img, mask, text):
    ov = img.copy()
    ov[mask] = (0.45 * ov[mask] + 0.55 * np.array([255, 80, 0])).astype(np.uint8)
    cv2.putText(ov, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return ov


def main():
    rows = []
    for f in sorted(glob.glob(f"{RUN}/results_shard*.jsonl")):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    todo = [r for r in rows if r.get("status") == "ok" and r.get("prompt_used") == "box"]
    print(f"{len(rows)} rows, {len(todo)} box-overridden masks to restore", flush=True)
    torch.cuda.set_device(0)
    predictor = build_robotseg_video_predictor("../robotseg/configs/robotseg-infer", f"{ROBOTSEG_ROOT}/checkpoints/robotseg.pt")
    P_cam = bf.gripper_points_cam()
    tmpdir = f"{RUN}/tmp/refine"
    out = open(f"{RUN}/results_refined.jsonl", "w")
    n = 0
    for r in rows:
        if r.get("status") != "ok":
            out.write(json.dumps(r) + "\n")
            continue
        ep, cam, t = r["episode"], r["cam"], r["frame"]
        stem = f"{cam}_f{t:05d}"
        if r.get("prompt_used") == "box":
            img = cv2.imread(f"{RUN}/frames/{ep}/{stem}.jpg")
            box_mask_path = f"{RUN}/masks/{ep}/{stem}_box.png"
            if not os.path.isfile(box_mask_path):
                shutil.copy(f"{RUN}/masks/{ep}/{stem}.png", box_mask_path)
            meta = json.load(open(f"{bf.META_DIR}/{ep}.json"))
            K = bf.load_zed_K(meta[f"{cam}_cam_serial"])
            T_bc = np.linalg.inv(bf.pose6_to_T(np.array(meta[f"{cam}_cam_extrinsics"])))
            with np.load(f"{bf.STORE}/{ep}/dense/cam/{t:06d}.npz") as z:
                pose = z["pose"]
            Pb = (pose[:3, :3] @ P_cam.T).T + pose[:3, 3]
            _, uv, vis = bf.visible_fraction(T_bc, K, Pb)
            shutil.rmtree(tmpdir, ignore_errors=True)
            os.makedirs(tmpdir)
            cv2.imwrite(f"{tmpdir}/00000.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(video_path=tmpdir, async_loading_frames=False,
                                             offload_video_to_cpu=False, offload_state_to_cpu=False)
                _, _, logits = predictor.add_new_robot(inference_state=state, frame_idx=0, obj_id=0, robot="gripper")
                auto_mask = (logits[0, 0] > 0).cpu().numpy()
                predictor.reset_state(state)
            del state
            sp_a, ar_a = spill_area(auto_mask, uv, vis)
            cv2.imwrite(f"{RUN}/masks/{ep}/{stem}_auto.png", auto_mask.astype(np.uint8) * 255)
            box_mask = cv2.imread(box_mask_path, 0) > 0
            if ar_a >= AREA_MIN:
                final, used, sp_f, ar_f = auto_mask, "auto", sp_a, ar_a
            else:
                final, used, sp_f, ar_f = box_mask, "box(auto empty)", r.get("spill_box", float("nan")), r.get("area_box", 0.0)
            cv2.imwrite(f"{RUN}/masks/{ep}/{stem}.png", final.astype(np.uint8) * 255)
            r.update(spill_auto=sp_a, area_auto=ar_a, prompt_used=used, spill_final=sp_f, area_final=ar_f)
            r["projection_disagree"] = bool((not np.isnan(sp_a)) and sp_a > SPILL_MAX)
            r["auto_empty"] = bool(ar_a < AREA_MIN)
            cv2.imwrite(f"{RUN}/overlays/{ep}/{stem}.jpg",
                        overlay(img, final, f"{ep} {cam} {r['which']} f{t} {used} spill {sp_f:.2f} area {100*ar_f:.2f}%"
                                + ("  [PROJECTION DISAGREES]" if r["projection_disagree"] else "")),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            n += 1
            if n % 200 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
        else:
            sp_a = r.get("spill_auto", float("nan"))
            r["projection_disagree"] = bool((sp_a is not None) and (not np.isnan(sp_a)) and sp_a > SPILL_MAX)
            r["auto_empty"] = bool(r.get("area_auto", 0.0) < AREA_MIN)
        r["flagged_final"] = r["projection_disagree"]  # flag now means: extrinsics suspect for this camera
        out.write(json.dumps(r) + "\n")
    out.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("done", n, "restored")


if __name__ == "__main__":
    main()
