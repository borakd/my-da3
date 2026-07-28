#!/usr/bin/env python
"""
verify_prev_flags.py — end-to-end, fully-traced verification of the two
previous-step conditioning flags on the captain_ray branch:

  Flag 1  feed_prev_gt_ray_map : view x is conditioned on the GROUND-TRUTH
          camera of view x-1 (its entire ray map — pose AND intrinsics —
          shifted one step in the data loader).
  Flag 2  feed_prev_pred       : view x is conditioned on the pose the model
          itself PREDICTED at step x-1, converted to a ray map inside the
          decode loop and fed through the same pretrained ray-map encoder.

The script drives the real pretrained checkpoint on real wrist_test data, on
CPU (login-node safe: no GPU is touched), and prints EVERYTHING along the
pipeline: every GT pose, every ray map's provenance and fingerprint, every
mask, every encoder-token injection, every stash read/write of the predicted
pose, and every prediction. Each claim is also hard-asserted; the run ends
with a PASS/FAIL table and a non-zero exit code if anything failed.

Run (from anywhere; conda env `cuteanything`):

  cd /scratch/bdursun25/cuteanything/captain_gru_v2
  export PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:$PYTHONPATH"
  export DL3DV_CACHE_DIR=/scratch/bdursun25/cuteanything/.dl3dv_cache
  python verify_prev_flags.py
"""

import hashlib
import os
import sys

WORKTREE = "/scratch/bdursun25/cuteanything/captain_gru_v2"
for p in [
    os.path.join(WORKTREE, "src"),
    os.path.join(WORKTREE, "src/CUT3R"),
    os.path.join(WORKTREE, "src/CUT3R/src"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

import dust3r.heads  # noqa: F401  MUST precede dust3r.utils.camera (circular import)
from dust3r.datasets.base.base_multiview_dataset import get_ray_map
from dust3r.datasets.dl3dv import DL3DV_Multi
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.model import ARCroco3DStereo
from dust3r.utils.camera import (
    get_ray_map_torch,
    pose_encoding_to_camera,
    quaternion_to_matrix,
)
from torch.utils.data._utils.collate import default_collate

# ----------------------------------------------------------------------------- config
DATA_ROOT = (
    "/frozen/avg/bora_data/droid_datasets/training_data/"
    "pointworld_droid_wrist_test/dl3dv_multi"
)
CKPT = "/scratch/bdursun25/cuteanything/my-da3/src/CUT3R/src/cut3r_512_dpt_4_64.pth"
NV = 4  # views per sequence
RES = (320, 192)  # (W, H) -> landscape, no portrait swap path
IDXS = [5, 100]  # two dataset samples -> batch size 2
SEED = 777

torch.set_num_threads(min(16, os.cpu_count() or 16))
torch.manual_seed(0)
np.set_printoptions(precision=5, suppress=True, linewidth=200)

# ----------------------------------------------------------------------------- helpers
WIDTH = 96
CHECKS = []


def banner(t):
    print("\n" + "=" * WIDTH + f"\n {t}\n" + "=" * WIDTH, flush=True)


def sub(t):
    print("\n" + "-" * WIDTH + f"\n {t}\n" + "-" * WIDTH, flush=True)


def check(gid, desc, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {gid}: {desc}" + (f"\n         -> {detail}" if detail else ""), flush=True)
    CHECKS.append((gid, desc, bool(ok), detail))


def to_np(a):
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def fp(a):
    """12-hex-char content fingerprint (byte-exact identity witness)."""
    return hashlib.sha1(np.ascontiguousarray(to_np(a)).tobytes()).hexdigest()[:12]


def map_str(m):
    """m: (H,W,6) ray map -> origin (constant), direction at center, fingerprint."""
    m = to_np(m)
    h, w = m.shape[:2]
    return (
        f"origin={m[0, 0, :3]}  dir@center={m[h // 2, w // 2, 3:]}  fp={fp(m)}"
    )


def pose_str(enc):
    """enc: (7,) absT_quaR pose encoding."""
    e = to_np(enc)
    return f"t={e[:3]}  quat(wxyz)={e[3:7]}"


def maxdiff(a, b):
    return float((to_np(a).astype(np.float64) - to_np(b).astype(np.float64)).__abs__().max())


def show_code(path, marker, n_lines, title):
    """Print the on-disk code block being verified (finds `marker`, prints window)."""
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if marker in l)
    print(f"\n  --- code on disk: {title}  ({os.path.relpath(path, WORKTREE)}) ---")
    for i in range(start, min(start + n_lines, len(lines))):
        print(f"  {i + 1:5d} | {lines[i].rstrip()}")
    print("  --- end code ---")


def matrix_to_quaternion(R):
    """(B,3,3) rotation -> (B,4) quaternion, real part (w) first."""
    out = []
    for b in range(R.shape[0]):
        m = R[b]
        t = m[0, 0] + m[1, 1] + m[2, 2]
        if t > 0:
            s = torch.sqrt(t + 1.0) * 2
            q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        out.append(torch.stack(q))
    return torch.stack(out)


# =============================================================================
banner("STAGE 0 — setup: datasets + real pretrained checkpoint (CPU, fp32)")
# =============================================================================
print(f"  torch {torch.__version__} | data root: {DATA_ROOT}")
print(f"  checkpoint: {CKPT}")
print(f"  batch = samples {IDXS} | {NV} consecutive views each | resolution {RES} (W,H)")


def mk_ds(**flags):
    return DL3DV_Multi(
        split="test",
        ROOT=DATA_ROOT,
        resolution=[RES],
        transform=ImgNorm,
        num_views=NV,
        n_corres=0,
        aug_crop=0,
        allow_repeat=True,
        force_consecutive_frame_sampling=True,
        seed=SEED,
        **flags,
    )


ds_plain = mk_ds()  # no conditioning         (ray_mask all False)
ds_gt = mk_ds(feed_gt_ray_map=True)  # own-GT conditioning     (unshifted reference)
ds_prev = mk_ds(feed_prev_gt_ray_map=True)  # Flag 1: prev-GT conditioning (shifted)
print(f"  datasets built: {len(ds_plain)} scenes (identical seed -> identical frame windows)")

model = ARCroco3DStereo.from_pretrained(CKPT).to("cpu").eval()
n_ray_tensors = sum(
    1
    for k in model.state_dict()
    if k.startswith(("patch_embed_ray_map", "enc_blocks_ray_map", "enc_norm_ray_map"))
) + int("masked_ray_map_token" in model.state_dict())
print(f"  model loaded on CPU, eval mode | pretrained ray-map-encoder tensors present: {n_ray_tensors}")

H_IMG, W_IMG = RES[1], RES[0]

samples = {"plain": [ds_plain[i] for i in IDXS], "gt": [ds_gt[i] for i in IDXS]}
samples["prev"] = [ds_prev[i] for i in IDXS]

for name in ("plain", "gt", "prev"):
    for s, i in zip(samples[name], IDXS):
        assert len(s) == NV
labels_ok = all(
    samples["plain"][b][v]["label"] == samples["gt"][b][v]["label"] == samples["prev"][b][v]["label"]
    for b in range(len(IDXS))
    for v in range(NV)
)
check(
    "G0",
    "the three datasets (plain / feed_gt / feed_prev_gt) sample identical frame windows",
    labels_ok,
    "same seed -> same episodes/frames; every later comparison is apples-to-apples",
)

# =============================================================================
banner("STAGE 1 — Flag 1 (feed_prev_gt_ray_map): DATA PIPELINE — what the loader emits")
# =============================================================================
show_code(
    os.path.join(WORKTREE, "src/CUT3R/src/dust3r/datasets/dl3dv.py"),
    "GT-pose oracle: expose a ground-truth camera",
    12,
    "dl3dv.py ray_mask wiring",
)
show_code(
    os.path.join(WORKTREE, "src/CUT3R/src/dust3r/datasets/base/base_multiview_dataset.py"),
    "One-step-lagged GT-pose oracle",
    12,
    "base_multiview_dataset.py one-step shift",
)

b0 = 0  # print full detail for sample IDXS[0]; assertions cover both samples
sv_gt, sv_prev, sv_plain = samples["gt"][b0], samples["prev"][b0], samples["plain"][b0]

sub(f"sample idx={IDXS[b0]}: per-view ground truth (from the GT camera files)")
cam0 = sv_gt[0]["camera_pose"]
for v in range(NV):
    view = sv_gt[v]
    rel = np.linalg.inv(cam0) @ view["camera_pose"]
    print(f"\n  VIEW {v}: frame = {view['label']}")
    print(f"    absolute GT camera_pose (c2w):\n{np.array2string(view['camera_pose'], prefix='      ')}")
    print(f"    GT pose RELATIVE to view 0 (inv(cam0) @ cam_{v}) — the frame ray maps live in:")
    print(f"{np.array2string(rel, prefix='      ')}")
    print(f"    intrinsics K (post-crop):\n{np.array2string(view['camera_intrinsics'], prefix='      ')}")
    print(f"    true_shape (H,W) = {view['true_shape']}  (landscape -> no portrait swap)")
    assert view["true_shape"][1] > view["true_shape"][0]

sub("ray-map table: UNSHIFTED (feed_gt_ray_map) vs SHIFTED (feed_prev_gt_ray_map)")
print("  Each ray map is (H,W,6) = [origin | direction]; origin == translation of the relative pose.")
print("  Expectation: shifted view v carries EXACTLY the unshifted map of view v-1.\n")
for v in range(NV):
    print(f"  view {v}  (img_mask={sv_prev[v]['img_mask']}, ray_mask={sv_prev[v]['ray_mask']}):")
    print(f"    unshifted map (own GT camera)     : {map_str(sv_gt[v]['ray_map'])}")
    print(f"    SHIFTED map (what Flag 1 feeds)   : {map_str(sv_prev[v]['ray_map'])}")

ok = all(
    np.array_equal(samples["prev"][b][v]["ray_map"], samples["gt"][b][v - 1]["ray_map"])
    for b in range(len(IDXS))
    for v in range(1, NV)
)
check(
    "G1",
    "loader shift is byte-exact: prev-flag view v's ray_map == own-GT map of view v-1 (v=1..3, both samples)",
    ok,
    f"e.g. shifted view1 fp={fp(sv_prev[1]['ray_map'])} == unshifted view0 fp={fp(sv_gt[0]['ray_map'])}",
)

masks_ok = all(
    (samples["prev"][b][0]["ray_mask"] is False or samples["prev"][b][0]["ray_mask"] == False)  # noqa: E712
    and all(samples["prev"][b][v]["ray_mask"] for v in range(1, NV))
    and all(samples["prev"][b][v]["img_mask"] for v in range(NV))
    for b in range(len(IDXS))
)
check(
    "G2",
    "masks: view 0 ray_mask=False (no predecessor), views 1..3 ray_mask=True, img_mask always True",
    masks_ok,
    "image is ALWAYS fed; rays are fed alongside it for views >= 1 only",
)

ok = True
for b in range(len(IDXS)):
    c0 = samples["gt"][b][0]["camera_pose"]
    for v in range(1, NV):
        prev_view = samples["gt"][b][v - 1]
        recomputed = get_ray_map(
            c0, prev_view["camera_pose"], prev_view["camera_intrinsics"], H_IMG, W_IMG
        ).astype(np.float32)
        ok &= np.array_equal(recomputed, samples["prev"][b][v]["ray_map"])
check(
    "G3",
    "provenance: recomputing get_ray_map(cam0, GT pose_{v-1}, K_{v-1}) from raw GT reproduces the shifted map byte-exactly",
    ok,
    "so the map view v receives encodes pose x-1 AND intrinsics x-1 — nothing else",
)

sup_keys = ["camera_pose", "camera_intrinsics", "depthmap", "pts3d", "valid_mask"]
ok = all(
    np.array_equal(to_np(samples["prev"][b][v][k]), to_np(samples["plain"][b][v][k]))
    and np.array_equal(to_np(samples["gt"][b][v][k]), to_np(samples["plain"][b][v][k]))
    and torch.equal(samples["prev"][b][v]["img"], samples["plain"][b][v]["img"])
    for b in range(len(IDXS))
    for v in range(NV)
    for k in sup_keys
)
check(
    "G4",
    f"supervision untouched: {sup_keys} + img are byte-identical across plain/gt/prev datasets",
    ok,
    "the flags change ONLY what is fed to the ray encoder, never the loss targets",
)

ok = all(
    np.array_equal(samples["prev"][b][0]["ray_map"], samples["gt"][b][0]["ray_map"])
    for b in range(len(IDXS))
)
check(
    "G5",
    "view 0 keeps its own (identity-relative) map but ray_mask=False -> it is never encoded",
    ok,
    f"view0 map origin={to_np(sv_prev[0]['ray_map'])[0, 0, :3]} (identity pose), masked off",
)

os.environ["GT_RAY_MAP_SHUFFLE"] = "1"
s_fals = ds_prev[IDXS[b0]]
del os.environ["GT_RAY_MAP_SHUFFLE"]
shifted = [sv_prev[v]["ray_map"] for v in range(NV)]  # [m0, m0, m1, m2]
expect_fals = [shifted[-1]] + shifted[:-1]  # [m2, m0, m0, m1]
ok = all(np.array_equal(s_fals[v]["ray_map"], expect_fals[v]) for v in range(NV)) and all(
    np.array_equal(to_np(s_fals[v][k]), to_np(sv_prev[v][k])) for v in range(NV) for k in sup_keys
)
check(
    "G6",
    "falsifier GT_RAY_MAP_SHUFFLE=1 composes on top of the shift: maps cyclically rotated, supervision untouched",
    ok,
    "control experiment: same masks, wrong camera content",
)

# =============================================================================
banner("STAGE 2 — Flag 1: MODEL PIPELINE — exactly what gets added to which view's tokens")
# =============================================================================
show_code(
    os.path.join(WORKTREE, "src/CUT3R/src/dust3r/model.py"),
    "ray_out, ray_pos, _ = self._encode_ray_map(selected_ray_maps",
    6,
    "model.py _encode_views ray-token injection",
)


def make_batch(view_lists):
    return [default_collate([vl[v] for vl in view_lists]) for v in range(NV)]


batch_plain = make_batch(samples["plain"])
batch_prev = make_batch(samples["prev"])

SPY = {"maps": [], "shapes": []}
_orig_encode_ray_map = model._encode_ray_map


def spy_encode_ray_map(ray_map, true_shape):
    SPY["maps"].append(ray_map.detach().clone())
    SPY["shapes"].append(true_shape)
    return _orig_encode_ray_map(ray_map, true_shape)


model._encode_ray_map = spy_encode_ray_map
model.feed_prev_pred = False

sub("running _encode_views on the Flag-1 batch (spy on the ray encoder records every map it sees)")
SPY["maps"].clear()
with torch.no_grad():
    _, feat_ls_prev, _ = model._encode_views(batch_prev)
feats_prev = feat_ls_prev[-1]  # tuple over views, each (B, S, D)

n_calls, n_rows = len(SPY["maps"]), sum(m.shape[0] for m in SPY["maps"])
print(f"  ray encoder called {n_calls}x with {n_rows} total maps (expected 1 call, 6 maps = 3 ray views x batch 2)")
rows = torch.cat(SPY["maps"], dim=0)
ok = n_rows == (NV - 1) * len(IDXS)
print("\n  identity of every map that entered the pretrained ray encoder (view-major order):")
for r in range(rows.shape[0]):
    v, b = 1 + r // len(IDXS), r % len(IDXS)
    expected = torch.from_numpy(samples["prev"][b][v]["ray_map"]).permute(2, 0, 1)
    same = torch.equal(rows[r], expected)
    ok &= same
    print(
        f"    row {r}: view {v} / sample idx={IDXS[b]}  fp={fp(rows[r])}  "
        f"== loader's shifted map (GT camera of view {v - 1})? {same}"
    )
check(
    "G7a",
    "the ray encoder received EXACTLY the loader's shifted maps (bitwise), one per (view>=1, sample)",
    ok,
)

with torch.no_grad():
    SPY["maps"].clear()
    _, feat_ls_plain, _ = model._encode_views(batch_plain)
    feats_plain = feat_ls_plain[-1]
    print(
        f"\n  plain (no-ray) batch for comparison: ray encoder saw {len(SPY['maps'])} dummy call(s) "
        f"with all-zero map, multiplied by 0.0 (pretrained-code quirk; numerically inert): "
        f"max|dummy|={float(SPY['maps'][0].abs().max()) if SPY['maps'] else 0.0}"
    )

sub("token decomposition: tokens(view v) == image_tokens(view v) + ray_tokens(map of GT pose_{v-1})")
ok_add, ok_masked = True, True
with torch.no_grad():
    for v in range(NV):
        if v == 0:
            delta = feats_prev[0] - feats_plain[0]
            d = maxdiff(delta, model.masked_ray_map_token.expand_as(delta))
            ok_masked &= d < 1e-5
            print(
                f"  view 0: tokens == image_tokens + masked_ray_map_token (learned 'no camera given' token); "
                f"max|residual|={d:.2e}  ||masked_token||={float(model.masked_ray_map_token.norm()):.4f}"
            )
        else:
            rm = batch_prev[v]["ray_map"].permute(0, 3, 1, 2)
            ray_tok = model._encode_ray_map(rm, batch_prev[v]["true_shape"])[0][0]
            d = maxdiff(feats_prev[v], feats_plain[v] + ray_tok)
            ok_add &= d < 1e-4
            print(
                f"  view {v}: tokens == image_tokens + ray_encoder(shifted map of pose_{v - 1}); "
                f"max|residual|={d:.2e}  ||img_tok||={float(feats_plain[v].norm()):.1f} "
                f"||ray_tok||={float(ray_tok.norm()):.1f}"
            )
check("G7", "additive injection verified for every ray-fed view (residual ~ fp noise)", ok_add)
check("G8", "view 0 receives the pretrained masked_ray_map_token instead of a camera", ok_masked)

# =============================================================================
banner("STAGE 3 — Flag 2 (feed_prev_pred): CLOSED-LOOP TRACE through the decode loop")
# =============================================================================
show_code(
    os.path.join(WORKTREE, "src/CUT3R/src/dust3r/model.py"),
    "Condition view x on the pose the model itself predicted",
    52,
    "model.py _forward_decoder_group_step in-loop conditioning",
)
show_code(
    os.path.join(WORKTREE, "src/CUT3R/src/dust3r/model.py"),
    "Stash this step's predicted pose",
    8,
    "model.py stash write (detached)",
)

model.feed_prev_pred = True

sub("encoder pass (feed_prev_pred=True): dataset feeds NO rays; view 0 gets the masked token")
with torch.no_grad():
    _, feat_ls_pp, _ = model._encode_views(batch_plain)
    feats_pp = feat_ls_pp[-1]
    d0 = maxdiff(feats_pp[0] - feats_plain[0], model.masked_ray_map_token.expand_as(feats_pp[0]))
    dv = max(maxdiff(feats_pp[v], feats_plain[v]) for v in range(1, NV))
check(
    "G13",
    "encoder tokens under feed_prev_pred: view 0 += masked_ray_map_token; views>=1 UNCHANGED "
    "(their rays are injected later, in the decode loop)",
    d0 < 1e-6 and dv < 1e-6,
    f"view0 residual={d0:.2e}, max change views1..3={dv:.2e}",
)

with torch.no_grad():
    (feat, pos, shape), (isf, imem, sf0, sp, mem0) = model._forward_encoder(batch_plain)
print(f"  _forward_encoder done: {len(feat)} views, feat[v] shape={tuple(feat[0].shape)}; "
      f"recurrent state {tuple(sf0.shape)}, pose memory {tuple(mem0.shape)}")


def rollout(batch, tag, forced_stash=None, collect_spy=True):
    """Drive _forward_decoder_group_step exactly like loss_of_one_batch_tbptt does
    (sequential view_indices [0],[1],[2],[3] threading state) with full tracing.

    forced_stash: optional list [enc_for_step1, enc_for_step2, enc_for_step3]
                  written into the stash right before each step >=1 (Stage-4 GT forcing).
    """
    sub(f"ROLLOUT '{tag}': sequential decode, views 0..{NV - 1}")
    preds, traces = [], []
    sf, mem = sf0, mem0
    prev_pred_saved = None
    with torch.no_grad():
        for x in range(NV):
            if forced_stash is not None and x >= 1:
                model._prev_pred_pose_enc = forced_stash[x - 1]
                print(f"\n  [step x={x}] stash FORCED to GT encoding of view {x - 1} (Stage-4 equivalence probe)")
            stash_before = getattr(model, "_prev_pred_pose_enc", None)
            SPY["maps"].clear()
            res_group, (sf, mem) = model._forward_decoder_group_step(
                views=batch,
                view_indices=[x],
                feat_group=[feat[x] if batch is batch_plain else featA[x]],
                pos_group=[pos[x] if batch is batch_plain else posA[x]],
                shape_group=[shape[x] if batch is batch_plain else shapeA[x]],
                init_state_feat=isf if batch is batch_plain else isfA,
                init_mem=imem if batch is batch_plain else imemA,
                state_feat=sf,
                state_pos=sp if batch is batch_plain else spA,
                mem=mem,
            )
            res = res_group[0]
            pred_pose = res["camera_pose"].detach().clone()
            stash_after = getattr(model, "_prev_pred_pose_enc", None)
            trace = dict(
                x=x,
                stash_before=None if stash_before is None else stash_before.detach().clone(),
                injected_maps=[m.clone() for m in SPY["maps"]],
                pred_pose=pred_pose,
                stash_after=None if stash_after is None else stash_after.detach().clone(),
                prev_pred_saved=prev_pred_saved,
            )
            traces.append(trace)
            if collect_spy:
                print(f"\n  [step x={x}]")
                if x == 0:
                    print("    stash at entry : RESET to None (sequence start; no predecessor)")
                    print(f"    in-loop ray-encoder calls: {len(SPY['maps'])} (expected 0 — view 0 is conditioned on nothing)")
                else:
                    src = "GT-FORCED" if forced_stash is not None else f"model's own prediction from step {x - 1}"
                    print(f"    stash READ ({src}):")
                    for b in range(stash_before.shape[0]):
                        print(f"      sample idx={IDXS[b]}: {pose_str(stash_before[b])}")
                    print(f"    detached? requires_grad={stash_before.requires_grad}, grad_fn={stash_before.grad_fn}")
                    if SPY["maps"]:
                        inj = SPY["maps"][0].permute(0, 2, 3, 1)
                        print(f"    ray map built IN-LOOP from that pose (intrinsics of view {x - 1}) and injected into view {x}'s tokens:")
                        for b in range(inj.shape[0]):
                            print(f"      sample idx={IDXS[b]}: {map_str(inj[b])}")
                for b in range(pred_pose.shape[0]):
                    print(f"    PREDICTION at x={x}, sample idx={IDXS[b]}: {pose_str(pred_pose[b])}")
                if stash_after is not None:
                    print(
                        f"    stash WRITE: _prev_pred_pose_enc <- prediction at x={x} "
                        f"(fp={fp(stash_after)}), detached={not stash_after.requires_grad} "
                        f"-> will condition step x={x + 1}"
                    )
            preds.append({k: res[k].detach().clone() for k in ("camera_pose", "pts3d_in_self_view", "pts3d_in_other_view") if k in res})
            prev_pred_saved = pred_pose
    return preds, traces


# sentinel: prove step 0 resets a stale stash instead of consuming it (no cross-batch leakage)
model._prev_pred_pose_enc = torch.full((len(IDXS), 7), 123.0)
preds_C, traces_C = rollout(batch_plain, "Flag 2, TRUE closed loop (own predictions)")

ok_reset = traces_C[0]["injected_maps"] == [] and traces_C[0]["stash_before"] is not None
check(
    "G11",
    "step 0: a stale/garbage stash from a previous batch is RESET, no ray is encoded for view 0",
    len(traces_C[0]["injected_maps"]) == 0,
    "sentinel stash (all 123.0) was discarded at sequence start; 0 in-loop encoder calls at x=0",
)

ok_loop = all(
    torch.equal(traces_C[x]["stash_before"], traces_C[x]["prev_pred_saved"]) for x in range(1, NV)
)
check(
    "G9",
    "CLOSED LOOP: the pose read at step x is BITWISE the pose predicted at step x-1 (x=1..3)",
    ok_loop,
    "  ".join(f"x={x}: fp(read)={fp(traces_C[x]['stash_before'])}==fp(pred_{x - 1})={fp(traces_C[x]['prev_pred_saved'])}" for x in range(1, NV)),
)

ok_maps = True
for x in range(1, NV):
    with torch.no_grad():
        c2w = pose_encoding_to_camera(traces_C[x]["stash_before"].float())
        rmap_exp = get_ray_map_torch(c2w, batch_plain[x - 1]["camera_intrinsics"], H_IMG, W_IMG)
    got = traces_C[x]["injected_maps"]
    same = len(got) == 1 and torch.equal(got[0], rmap_exp.permute(0, 3, 1, 2))
    ok_maps &= same
    print(f"  step {x}: injected map == get_ray_map_torch(pose_encoding_to_camera(pred_{x - 1}), K_{x - 1})? {same}")
check(
    "G10",
    "the injected ray map is EXACTLY the one recomputed independently from the stashed prediction + view x-1 intrinsics (bitwise)",
    ok_maps,
    "pipeline: pred pose (7d absT_quaR) -> 4x4 c2w (relative to view 0) -> ray map -> frozen ray encoder -> += view x tokens",
)

ok_stash = all(
    torch.equal(traces_C[x]["stash_after"], traces_C[x]["pred_pose"])
    and not traces_C[x]["stash_after"].requires_grad
    and traces_C[x]["stash_after"].grad_fn is None
    for x in range(NV)
)
check(
    "G12",
    "stash write: after every step, _prev_pred_pose_enc == that step's predicted camera_pose, detached (grad_fn=None)",
    ok_stash,
)

sub("G14 — gradient safety at a simulated TBPTT chunk boundary (grad-enabled run)")
sfg, memg = sf0, mem0
grad_ok, err = True, ""
try:
    res0, (sfg, memg) = model._forward_decoder_group_step(
        views=batch_plain, view_indices=[0], feat_group=[feat[0]], pos_group=[pos[0]],
        shape_group=[shape[0]], init_state_feat=isf, init_mem=imem,
        state_feat=sfg, state_pos=sp, mem=memg,
    )
    loss0 = res0[0]["camera_pose"].sum() + res0[0]["pts3d_in_self_view"].mean()
    loss0.backward()  # frees the step-0 graph — exactly what per-chunk backward does
    stash_gradfn = model._prev_pred_pose_enc.grad_fn
    sfg, memg = sfg.detach(), memg.detach()  # chunk boundary detaches state (inference.py:142-144)
    res1, _ = model._forward_decoder_group_step(
        views=batch_plain, view_indices=[1], feat_group=[feat[1]], pos_group=[pos[1]],
        shape_group=[shape[1]], init_state_feat=isf, init_mem=imem,
        state_feat=sfg, state_pos=sp, mem=memg,
    )
    loss1 = res1[0]["camera_pose"].sum() + res1[0]["pts3d_in_self_view"].mean()
    loss1.backward()  # would raise 'backward through the graph a second time' if the fed-back pose carried step-0's graph
    model.zero_grad(set_to_none=True)
except RuntimeError as e:  # pragma: no cover
    grad_ok, err = False, str(e)
check(
    "G14",
    "grad-enabled double-backward across a simulated TBPTT chunk boundary succeeds; "
    "fed-back pose carries NO autograd graph even when produced under grad",
    grad_ok and stash_gradfn is None,
    err or f"stash.grad_fn under grad-enabled forward = {stash_gradfn} (detach() in the stash write)",
)
model._prev_pred_pose_enc = None

# =============================================================================
banner("STAGE 4 — CROSS-FLAG EQUIVALENCE + DIVERGENCE + FALSIFIER (the strongest guarantees)")
# =============================================================================
sub("Run A — Flag 1 forward (dataset-shifted GT maps), same batch content")
model.feed_prev_pred = False
with torch.no_grad():
    (featA, posA, shapeA), (isfA, imemA, sfA0, spA, memA0) = model._forward_encoder(batch_prev)
preds_A = []
sfA, memA = sfA0, memA0
with torch.no_grad():
    for x in range(NV):
        res_group, (sfA, memA) = model._forward_decoder_group_step(
            views=batch_prev, view_indices=[x], feat_group=[featA[x]], pos_group=[posA[x]],
            shape_group=[shapeA[x]], init_state_feat=isfA, init_mem=imemA,
            state_feat=sfA, state_pos=spA, mem=memA,
        )
        preds_A.append({k: v.detach().clone() for k, v in res_group[0].items() if isinstance(v, torch.Tensor)})
        print(f"  [Flag 1, step {x}] prediction sample idx={IDXS[0]}: {pose_str(res_group[0]['camera_pose'][0])}")

sub("GT pose encodings (what Flag 1's maps encode), built independently for the forcing probe")
gt_enc = []
for v in range(NV - 1):
    rel = torch.linalg.inv(batch_plain[0]["camera_pose"]) @ batch_plain[v]["camera_pose"]
    q = matrix_to_quaternion(rel[:, :3, :3].double()).float()
    enc = torch.cat([rel[:, :3, 3].float(), q], dim=1)
    gt_enc.append(enc)
    rt = maxdiff(quaternion_to_matrix(q.double()), rel[:, :3, :3].double())
    print(f"  GT enc of view {v}: sample idx={IDXS[0]}: {pose_str(enc[0])}  (quat roundtrip err={rt:.2e})")
check(
    "G15",
    "matrix->quaternion->matrix roundtrip of the GT relative poses is exact to fp precision",
    all(maxdiff(quaternion_to_matrix(matrix_to_quaternion((torch.linalg.inv(batch_plain[0]['camera_pose']) @ batch_plain[v]['camera_pose'])[:, :3, :3].double())), (torch.linalg.inv(batch_plain[0]['camera_pose']) @ batch_plain[v]['camera_pose'])[:, :3, :3].double()) < 1e-6 for v in range(NV - 1)),
)

model.feed_prev_pred = True
preds_B, traces_B = rollout(batch_plain, "Flag 2 machinery, stash FORCED to GT pose of x-1", forced_stash=gt_enc)

sub("EQUIVALENCE: Flag-2 machinery fed GT poses must reproduce the Flag-1 forward")
print("  (identical camera information through two different code paths: loader-shifted numpy maps vs in-loop torch maps)")
ok_eq, det = True, []
for v in range(NV):
    dp = maxdiff(preds_B[v]["camera_pose"], preds_A[v]["camera_pose"])
    dx = maxdiff(preds_B[v]["pts3d_in_self_view"], preds_A[v]["pts3d_in_self_view"])
    ok_eq &= dp < 1e-3 and dx < 1e-2
    det.append(f"view{v}: max|Δpose|={dp:.2e} max|Δpts3d|={dx:.2e}")
    print(f"  view {v}: max|Δcamera_pose|={dp:.2e}   max|Δpts3d_self|={dx:.2e}")
check(
    "G16",
    "Flag 2 == Flag 1 when fed the same (GT) pose: the two flags are ONE mechanism differing only in pose source",
    ok_eq,
    "; ".join(det),
)

sub("DIVERGENCE: true closed loop (own predictions) vs GT-conditioned — difference is exactly the pose source")
ok_div_v0 = maxdiff(preds_C[0]["camera_pose"], preds_A[0]["camera_pose"]) < 1e-5
any_div = False
for v in range(NV):
    d = maxdiff(preds_C[v]["camera_pose"], preds_A[v]["camera_pose"])
    if v > 0:
        gap = maxdiff(traces_C[v]["stash_before"], gt_enc[v - 1])
        any_div |= d > 1e-6
        print(f"  view {v}: |pred_pose(x-1) - GT_pose(x-1)| (input gap) = {gap:.2e}   ->   |Δ output pose| = {d:.2e}")
    else:
        print(f"  view 0: |Δ output pose| = {d:.2e} (no conditioning at x=0 -> must match)")
check(
    "G17",
    "view 0 identical across Flag-2/Flag-1 runs; views>=1 diverge exactly when predicted != GT pose",
    ok_div_v0 and any_div,
)

sub("FALSIFIER: PREV_PRED_RAY_SHUFFLE=1 — feed each sample the OTHER sample's predicted pose")
os.environ["PREV_PRED_RAY_SHUFFLE"] = "1"
preds_F, traces_F = rollout(batch_plain, "Flag 2 + falsifier (batch-rolled fed-back pose)", collect_spy=False)
del os.environ["PREV_PRED_RAY_SHUFFLE"]
ok_f_maps = True
for x in range(1, NV):
    with torch.no_grad():
        rolled = torch.roll(traces_F[x]["stash_before"], shifts=1, dims=0)
        rmap_exp = get_ray_map_torch(pose_encoding_to_camera(rolled.float()), batch_plain[x - 1]["camera_intrinsics"], H_IMG, W_IMG)
    ok_f_maps &= torch.equal(traces_F[x]["injected_maps"][0], rmap_exp.permute(0, 3, 1, 2))
v0_same = torch.equal(preds_F[0]["camera_pose"], preds_C[0]["camera_pose"])
v_diff = max(maxdiff(preds_F[v]["camera_pose"], preds_C[v]["camera_pose"]) for v in range(1, NV))
check(
    "G18",
    "falsifier verified: injected maps come from the batch-ROLLED pose; view 0 bit-identical; views>=1 change",
    ok_f_maps and v0_same and v_diff > 1e-6,
    f"view0 bitwise-equal={v0_same}; max pose change views1..3 = {v_diff:.2e}",
)
model.feed_prev_pred = False
model._encode_ray_map = _orig_encode_ray_map

# =============================================================================
banner("VERDICT")
# =============================================================================
print("""
  Verified information flow
  -------------------------
  Flag 1 (feed_prev_gt_ray_map):
    GT pose x-1 + K x-1 --get_ray_map--> ray_map --[loader shift: view x carries it]-->
    frozen ray encoder --+--> view x encoder tokens --recurrent decoder--> prediction x
    (view 0: masked_ray_map_token; supervision untouched)

  Flag 2 (feed_prev_pred):
    prediction x-1 (camera_pose, 7d, DETACHED stash) --pose_encoding_to_camera--> c2w
    --get_ray_map_torch(K of view x-1)--> ray_map --frozen ray encoder--> +feat(view x)
    --recurrent decoder--> prediction x --detach--> stash (conditions x+1)
    (view 0: stash reset + masked_ray_map_token; identical mechanism to Flag 1, only the
     pose source differs — proven by the GT-forcing equivalence run)
""")
n_fail = sum(1 for *_r, ok, _ in CHECKS if not ok)
for gid, desc, ok, _ in CHECKS:
    print(f"  {'PASS' if ok else 'FAIL'}  {gid:5s} {desc}")
print(f"\n  {len(CHECKS) - n_fail}/{len(CHECKS)} guarantees hold.")
if n_fail:
    print("  *** AT LEAST ONE GUARANTEE FAILED — DO NOT TRAIN UNTIL RESOLVED ***")
    sys.exit(1)
print("  All guarantees hold: the pipelines do exactly what was specified.")
