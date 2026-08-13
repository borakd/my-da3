#!/usr/bin/env python
"""probe_p43_premise.py — is a POSE-FREE readout of the current view a better
estimate of that view's pose than the fed-back trajectory is?

That single question is the entire premise of P4.3 (GRU_GAP_CLOSURE_DIRECTIVE.md
§4). P4.3 gives the corrector an observation O(x) obtained by decoding view x
with the pose conditioning removed, on the state the real pass is about to use.
The claim is that O(x) — unlike the fed-back P(x-1) — has actually LOOKED at
frame x, so it carries the accumulated head drift that the trajectory cannot
observe. If that is false the whole arm is pointless, and this script finds out
in minutes instead of 26 h x 8 nodes.

Nothing here needs the P4.3 code. It computes the probe decode inline from
public model methods, so it runs against checkpoints that exist TODAY and is
unaffected by the implementation landing alongside it.

=== WHAT IS MEASURED, PER VIEW x, IN EACH MODEL'S OWN ROLLOUT ===

    e_lag(x)   = err(P(x-1),      GT(x))   reuse the previous pose
    e_vel(x)   = err(P(x-1) o D,  GT(x))   constant-velocity extrapolation
    e_probe(x) = err(O(x),        GT(x))   the pose-free observation
    e_head(x)  = err(P(x),        GT(x))   what the arm actually commits

with D = pose_delta_encoding(P(x-2), P(x-1)), i.e. exactly the velocity the A4
lever already hands the cell. Rotation is the geodesic quaternion angle in
degrees (scale-free). Translation is SCALE-NORMALIZED: each side is divided by
its OWN per-sequence '?avg_dis' factor before differencing, exactly as
PoseGRULoss does (losses.py: gru_pose[:, :3]/factor_pr - target[:, :3]/factor_gt).

That normalization is load-bearing, not hygiene. The first version of this
script differenced them RAW, and since the head's poses live at ~4.83x GT scale
the result measured the SCALE MISMATCH rather than pose error: every estimator
scored ~0.92 against an absT magnitude of ~0.995, so all four looked equally bad
and none of them discriminated. Rotation was unaffected and was the only half of
that run worth reading. e_null (the error of predicting the ORIGIN) is reported
alongside as a saturation calibration: if every estimator sits near e_null, the
regime carries no signal and no ratio between estimators means anything.

=== MAGNITUDE IS THE WEAK TEST; DIRECTION IS THE DECISIVE ONE ===

e_lag contains the true inter-frame MOTION, not just drift, so anything that
models motion beats it -- including a trajectory extrapolator that never looks
at an image. So e_probe < e_lag proves little, and the first bar was e_vel.

But e_probe vs e_vel is still only a POINT-ESTIMATE comparison, and that is not
what a corrector needs. A signal with larger error can be highly informative if
its error is DECORRELATED from the trajectory's: the corrector does not adopt
the observation, it extracts the component the trajectory lacks. So the decisive
quantity is DIRECTION --

    u(x) = O(x)  - P(x-1)      what the source proposes to change
    w(x) = GT(x) - P(x-1)      what actually needs changing
    a*   = <u,w>/<u,u>         the optimal scalar gain
    residual after a*  =  |w|^2 (1 - cos^2)

so cos^2 IS, exactly, the best fractional error reduction any corrector could
extract from that signal. Reported per band for the probe AND for velocity
extrapolation, because velocity is a proposal the cell can already form from
inputs it is handed (P(x-1) is its residual anchor, delta is block two).

BUT cos2_probe > cos2_vel is NOT the test -- those are two MARGINAL fits and they
overlap, so that inequality can hold while the probe adds nothing delta already
carries. The P4.3 claim is INCREMENTAL, so the quantity is a partial R^2: the
joint 2-parameter fit min_(a,b) |w - a*u_vel - b*u_probe|^2, and
DELTA_R2 = R2_joint - cos2_vel. That needs the cross term <u_vel, u_probe>,
without which the joint fit is not identifiable.

And DELTA_R2 needs its own falsifier, because a bootstrap CI measures sampling
variance around whatever the estimator converges to -- not whether that value is
reachable by chance. So the same proposal is also rolled to a DIFFERENT SCENE
(same magnitudes, same geometry, same estimator, same degrees of freedom, only
the correspondence to THIS frame destroyed) and DELTA_R2 recomputed. Measured:
the honest value is 34-390x the shuffled null at mid/late views.

    cos2_probe ~ 0 at late views          -> P4.3 refuted, on the right quantity
    cos2_probe > cos2_vel materially      -> the probe IS informative and the
                                             magnitude test measured the wrong
                                             thing, since a corrector scaling by
                                             a* beats velocity even though the
                                             raw probe does not

=== HOW TO READ THE TWO ARMS, AND WHY THE BRACKET HAS ONLY ONE USABLE END ===

  GRU arm (r8) -- the probe here is OUT OF DISTRIBUTION. That model trained with
  a ray token on every view x>0; "rich accumulated state, no ray token"
  mid-sequence is a regime it never saw (view 0 is pose-free but its state is
  empty, which is not the same condition). A negative here is SUGGESTIVE, not a
  refutation. Note also that OOD severity GROWS with view index -- by view 60 the
  state carries 60 views of conditioning-dependent structure the probe was never
  trained to decode -- so "the probe's advantage decays with x" and "OOD severity
  grows with x" predict the same curve, and this arm cannot separate them.

  Arm A -- DEGENERATE AS A BRACKET, which is not what an earlier version of this
  file claimed. Arm A has no conditioning loop, so its readout and its trajectory
  are THE SAME TENSOR (e_probe_t and e_head_t are bit-identical for it here, and
  correctly differ for r8). Its comparison is therefore one estimator against an
  extrapolation of its own two previous outputs, which measures the temporal
  smoothness of the head's error -- guaranteed for any state-recurrent model --
  and says nothing about whether a pose-free readout observes drift. A model
  cannot observe its own error by asking itself. Arm A does NOT close the OOD
  hole, and cos^2 is not meaningful for it either (u is its own step, not an
  independent proposal).

  What arm A DOES establish, non-degenerately, is worth more than the bracket
  would have been: by late views its pose readout is nearly predictable from its
  own two previous outputs (probe/vel 0.984). The head has become a smooth
  extrapolator that barely reads the current image. That is architectural, and it
  is the real obstacle P4.3 runs into.

Usage:
    python probe_p43_premise.py                    # both arms, 100 scenes
    sbatch probe_p43_premise.sbatch

Env: P43_GRU_CKPT P43_A_CKPT P43_BATCHES(25) P43_BS(4) P43_NV(64)
     P43_WORKERS(16) P43_OUT DL3DV_TEST_ROOT
"""
import json
import os
import sys
import time

# Self-locating: derive the checkout from THIS file, never a hardcoded path.
# sys.path.insert(0, <nonexistent>) SILENTLY SUCCEEDS, so a stale hardcoded
# worktree would verify a DIFFERENT checkout with no warning. Assert.
WORKTREE = os.path.dirname(os.path.abspath(__file__))
assert os.path.isfile(
    os.path.join(WORKTREE, "src", "CUT3R", "src", "train_cut3r_baseline.py")
), f"not a my-da3 checkout: {WORKTREE}"
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
from dust3r.datasets.dl3dv import DL3DV_Multi
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.heads.postprocess import postprocess_pose
from dust3r.model import gt_pose_encoding, load_model, pose_delta_encoding
from dust3r.utils.camera import (
    camera_to_pose_encoding,
    matrix_to_quaternion,
    pose_encoding_to_camera,
    quaternion_to_matrix,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

DATA_ROOT = os.environ.get(
    "DL3DV_TEST_ROOT",
    "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi",
)
CKPT_ROOT = os.environ.get("CKPT_ROOT", "/gpfs/projects/etur59/koc821022/checkpoints")
GRU_CKPT = os.environ.get(
    "P43_GRU_CKPT",
    f"{CKPT_ROOT}/captain_cut3r_finetune_aug_full/captain_gru_v3_a4_g3_r8_finetune/checkpoint-final.pth",
)
A_CKPT = os.environ.get(
    "P43_A_CKPT",
    f"{CKPT_ROOT}/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth",
)
N_BATCHES = int(os.environ.get("P43_BATCHES", 25))
BS = int(os.environ.get("P43_BS", 4))
NV = int(os.environ.get("P43_NV", 64))
WORKERS = int(os.environ.get("P43_WORKERS", 16))
OUT_JSON = os.environ.get("P43_OUT", os.path.join(WORKTREE, "p43_premise.json"))
RES = (320, 192)
SEED = 777
DEVICE = "cuda"

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
np.set_printoptions(precision=5, suppress=True)


def guard_falsifier_env():
    """These silently change what the rollout produces; refuse rather than
    measure a contaminated trajectory. PREV_PRED_RAY_SHUFFLE would roll the
    fed-back pose (so e_lag would be another sample's), POSE_GRU_FORCE_ITERS
    would override the R lever, and the hidden/feature knobs change the
    trajectory the loop produces."""
    live = [
        k
        for k in (
            "PREV_PRED_RAY_SHUFFLE",
            "GT_RAY_MAP_SHUFFLE",
            "POSE_GRU_HIDDEN_SHUFFLE",
            "POSE_GRU_HIDDEN_ZERO",
            "POSE_GRU_IMG_FEAT_SHUFFLE",
            "POSE_GRU_IMG_FEAT_ZERO",
            "POSE_GRU_FORCE_ITERS",
            # P4.3. These were MISSING from this list until 2026-08-12, which is
            # how the shuffle falsifier run silently produced no banner: the
            # guard saw nothing live while POSE_GRU_PROBE_SHUFFLE was in fact
            # reaching _probe_falsify in model.py. The measurement was valid, but
            # a STRAY setting would have contaminated a clean run with no warning
            # — the precise failure this guard exists to prevent. Any new
            # falsifier env var must be added here at the same time it is added
            # to model.py; there is no registry that does it automatically.
            "POSE_GRU_PROBE_SHUFFLE",
            "POSE_GRU_PROBE_LAG",
            "POSE_GRU_FORCE_REFINE",
        )
        if os.environ.get(k)
    ]
    if live and os.environ.get("P43_ALLOW_CONTAMINATION") == "1":
        # DELIBERATE falsifier run. The guard exists to stop an inherited
        # --export=ALL from silently changing what is measured; a run that is
        # ABOUT the falsifier has to be able to opt in, loudly.
        print("=" * 78, flush=True)
        print(f"DELIBERATE FALSIFIER RUN — active: {live}", flush=True)
        print("Errors below are NOT the deployed ones. Compare e_head against an", flush=True)
        print("otherwise identical clean run; do not read them in isolation.", flush=True)
        print("=" * 78, flush=True)
        return
    assert not live, (
        f"falsifier env vars are set: {live}. They change the rollout, so the "
        "measured errors would not be the deployed ones. Unset them."
    )


class _SeqSet(Dataset):
    """One item = one scene = a list of NV view dicts (what DL3DV_Multi returns)."""

    def __init__(self, ds, idxs):
        self.ds, self.idxs = ds, idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        return self.ds[self.idxs[k]]


def _collate(seqs):
    """Per-VIEW collation: the rollout wants a list of NV batched view dicts."""
    return [default_collate([s[v] for s in seqs]) for v in range(len(seqs[0]))]


def to_device(batch):
    return [
        {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in view.items()}
        for view in batch
    ]


# ---------------------------------------------------------------------------
# error metrics — raw head units, applied identically to every quantity so the
# four series are comparable to each other within one model.
# ---------------------------------------------------------------------------
def trans_err(pred, gt):
    """(B,7) x (B,7) -> (B,) L2 translation error in HEAD units."""
    return torch.norm(pred[:, :3].float() - gt[:, :3].float(), dim=-1)


def rot_err_deg(pred, gt):
    """Geodesic angle between two quaternions, degrees. |dot| removes the q/-q
    double cover, so this is sign-safe (the chordal form is not)."""
    d = (pred[:, 3:7].float() * gt[:, 3:7].float()).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.arccos(d))


def compose(prev_enc, delta_enc):
    """Constant-velocity extrapolation P(x-1) o D, in pose-encoding space.

    D = pose_delta_encoding(P(x-2), P(x-1)) is inv(P(x-2)) @ P(x-1) expressed in
    P(x-2)'s frame, so composing it onto P(x-1) advances one more step at the
    same velocity -- the trajectory-only predictor the probe has to beat.
    """
    c2w_prev = pose_encoding_to_camera(prev_enc.float())
    c2w_delta = pose_encoding_to_camera(delta_enc.float())
    return camera_to_pose_encoding(c2w_prev @ c2w_delta)


@torch.no_grad()
def probe_decode(model, feat_v, pos_v, state_feat, state_pos, mem, init_state_feat):
    """The pose-free observation O(x): decode view x with the conditioning ray
    replaced by masked_ray_map_token, on the state the REAL pass is about to
    consume, and read only the pose.

    Mirrors the real pass's pose branch exactly except for the ray term: same
    state, same mem, same pose_retriever.inquire seed (NOT self.pose_token --
    that is view 0's seed, and using it would change what is being ablated from
    "the ray token" to "the ray token and the memory read").

    Reads the pose off the pose token directly: camera_pose comes from a
    768->3072->7 MLP on ONE token, computed BEFORE the three dense DPT adapters
    (which are ~96% of the head and run FP32), so the dense machinery is skipped
    entirely. Nothing here writes to state_feat or mem -- _decoder is
    x = x + f(x) throughout and update_mem is never called -- so the caller's
    tensors are untouched and no copy is needed.
    """
    feat_probe = feat_v + model.masked_ray_map_token.to(feat_v.dtype)
    global_img_feat = model._get_img_level_feat(feat_probe)
    pose_seed = model.pose_retriever.inquire(global_img_feat, mem)
    pose_pos = torch.zeros(
        feat_probe.shape[0], 1, 2, device=feat_probe.device, dtype=pos_v.dtype
    )
    _new_state, dec = model._recurrent_rollout(
        state_feat, state_pos, feat_probe, pos_v, pose_seed, pose_pos, init_state_feat
    )
    pose_token = dec[model.dec_depth][:, 0].float()
    return postprocess_pose(model.downstream_head.pose_head(pose_token), model.pose_mode)


def rotvec_between(q_from, q_to):
    """Tangent-space rotation from q_from to q_to as a 3-vector (axis * angle).

    Rotation has no meaningful vector difference, so the direction test needs
    the log map: log(R_from^T R_to). Goes through matrix_to_quaternion, which
    standardizes to w>=0, so the double cover cannot flip the sign of the
    resulting axis.
    """
    R1 = quaternion_to_matrix(q_from.float())
    R2 = quaternion_to_matrix(q_to.float())
    q = matrix_to_quaternion(R1.transpose(1, 2) @ R2)
    w = q[:, 0:1].clamp(-1.0, 1.0)
    v = q[:, 1:4]
    n = v.norm(dim=-1, keepdim=True)
    ang = 2.0 * torch.atan2(n, w)
    return v / n.clamp(min=1e-12) * ang


def inner_products(u, w):
    """(<u,w>, <u,u>, <w,w>) per sample — the three sums the cos^2 bound needs."""
    return (u * w).sum(-1), (u * u).sum(-1), (w * w).sum(-1)


def norm_factor(poses):
    """Per-sample '?avg_dis' scale factor over a whole sequence: the mean L2 of
    that side's translations across all views. (B,1).

    THIS IS NOT OPTIONAL, and getting it wrong is how the first version of this
    script produced a meaningless answer. The head's poses live at ~4.83x GT
    scale (probe_gru_input_stats.py, 2026-08-12), so ||P_head - P_gt|| computed
    RAW is dominated by the scale mismatch, not by pose error: every estimator
    then scores ~0.92 head units against an absT magnitude of ~0.995, i.e. all
    of them look equally bad and none of them discriminate. PoseGRULoss has
    always divided each side by its own factor for exactly this reason
    (losses.py: gru_pose[:, :3] / factor_pr - target[:, :3] / factor_gt, with
    factor_gt/factor_pr from get_norm_factor_poses under norm_mode '?avg_dis').

    Computed ONCE over the whole sequence and applied to every view -- never
    per-view, which would divide out the accumulating drift this test exists to
    measure.
    """
    return torch.stack([p[:, :3].norm(dim=-1) for p in poses], 0).mean(0, keepdim=True).T


@torch.no_grad()
def rollout(model, batch, do_probe):
    """One scene batch. Returns per-view dicts of (B,) error tensors.

    do_probe=True  -> conditioned arm: run the extra pose-free decode per view.
    do_probe=False -> arm A: every view is ALREADY pose-free, so its ordinary
                      forward IS the readout and no probe construction is used.
    """
    (feat, pos, shape), (isf, imem, sf, sp, mem) = model._forward_encoder(batch)
    heads, probes, gts, oks = [], [], [], []
    for v in range(len(batch)):
        probe = None
        if do_probe and v >= 1:
            probe = probe_decode(model, feat[v], pos[v], sf, sp, mem, isf)
        res_group, (sf, mem) = model._forward_decoder_group_step(
            views=batch, view_indices=[v],
            feat_group=[feat[v]], pos_group=[pos[v]], shape_group=[shape[v]],
            init_state_feat=isf, init_mem=imem,
            state_feat=sf, state_pos=sp, mem=mem,
        )
        heads.append(res_group[-1]["camera_pose"].detach().float())
        # On arm A every view is already pose-free, so the head's own output IS
        # the readout; on the conditioned arm it is the extra probe decode.
        probes.append(heads[-1] if not do_probe else (probe.detach().float() if probe is not None else None))
        gt, ok = gt_pose_encoding(batch, v)
        gts.append(gt)
        oks.append(ok.reshape(-1))

    # P43_PLACEBO=1 — THE MISSING CONTROL ON DELTA_R2 ITSELF.
    #
    # POSE_GRU_PROBE_SHUFFLE only reaches the MODEL's internal probe (the one
    # that conditions the ray build). It does NOT touch probe_decode() above,
    # which is what this script measures, so every DELTA_R2 reported so far has
    # been run without a placebo. That is a real hole: a positive DELTA_R2 could
    # in principle arise from any incidental correlation between a head-produced
    # pose and the needed correction, rather than from this frame's observation.
    #
    # The control: batch-roll O(x), so every sequence is scored against ANOTHER
    # sequence's observation of the same view index — identical shapes, identical
    # magnitudes, identical head, wrong content. Done here, before ANY scoring,
    # so every probe-derived number in the report becomes its placebo version.
    #
    #   DELTA_R2 collapses toward the ~3e-3 scene-block null  -> the signal is
    #       genuinely this frame's observation. The headline stands.
    #   DELTA_R2 survives near its real value                 -> the number is
    #       NOT coming from the observation and the positive result is an
    #       artifact. This is the outcome that matters more, and the one nothing
    #       measured until now.
    if os.environ.get("P43_PLACEBO") == "1" and len(batch) > 1:
        rolled = [None] + [
            torch.roll(p, shifts=1, dims=0) if p is not None else None for p in probes[1:]
        ]
        probes = rolled

    # Scale-normalize ONCE over the sequence, each side by its own factor. See
    # norm_factor(): raw head-vs-GT differences measure the ~4.8x scale
    # mismatch, not pose error.
    f_pr = norm_factor(heads).clip(min=1e-8)
    f_gt = norm_factor(gts).clip(min=1e-8)

    def npose(p, f):
        out = p.clone()
        out[:, :3] = out[:, :3] / f
        return out

    rows = []
    for v in range(1, len(batch)):
        gt_n = npose(gts[v], f_gt)
        head_n = npose(heads[v], f_pr)
        prev_n = npose(heads[v - 1], f_pr)
        readout_n = npose(probes[v], f_pr)
        row = dict(view=v, ok=oks[v].cpu())
        row["e_probe_t"] = trans_err(readout_n, gt_n).cpu()
        row["e_probe_r"] = rot_err_deg(readout_n, gt_n).cpu()
        row["e_lag_t"] = trans_err(prev_n, gt_n).cpu()
        row["e_lag_r"] = rot_err_deg(prev_n, gt_n).cpu()
        row["e_head_t"] = trans_err(head_n, gt_n).cpu()
        row["e_head_r"] = rot_err_deg(head_n, gt_n).cpu()
        # THE MECHANISM NUMBER. How far does the observation actually move away
        # from the pose it is supposed to be correcting? Both sides are
        # pred-side, so both take factor_pr.
        #   gap/e_probe ~ 1   -> an independent measurement
        #   gap/e_probe << 1  -> the readout and the trajectory share almost all
        #                        of their error, i.e. the drift is COMMON MODE
        #                        and removing the ray token did not remove it
        # NOTE for arm A this is a tautology (its readout IS its trajectory, so
        # gap is by construction one frame of motion) -- it only carries
        # information for the conditioned arm.
        row["gap_t"] = trans_err(readout_n, prev_n).cpu()
        row["gap_r"] = rot_err_deg(readout_n, prev_n).cpu()
        if v >= 2:
            prev2_n = npose(heads[v - 2], f_pr)
            row["dlag_t"] = trans_err(prev_n, prev2_n).cpu()
            row["dlag_r"] = rot_err_deg(prev_n, prev2_n).cpu()
            # Compose in RAW head space (the composition is a rigid transform,
            # not a linear op on the encoding), then normalize the result.
            vel = compose(heads[v - 1], pose_delta_encoding(heads[v - 2], heads[v - 1]))
            vel_n = npose(vel, f_pr)
            row["e_vel_t"] = trans_err(vel_n, gt_n).cpu()
            row["e_vel_r"] = rot_err_deg(vel_n, gt_n).cpu()
        # Saturation calibration: the error of predicting the ORIGIN. If every
        # estimator sits near this, the regime is saturated and NO ratio between
        # them carries information -- which is a fact about the test's power,
        # not about the probe.
        # ---- DIRECTION, not magnitude. The quantity that actually decides it.
        # e_probe > e_vel only says the probe is a worse POINT ESTIMATE. A
        # signal with larger error can still be highly informative if its error
        # is DECORRELATED from the trajectory's -- the corrector's job is not to
        # adopt the observation, it is to extract the component the trajectory
        # lacks. So compare what each source PROPOSES to change against what
        # actually needs changing:
        #     u = proposal - P(x-1)      w = GT(x) - P(x-1)
        # The optimal scalar gain is a* = <u,w>/<u,u>, the residual after
        # applying it is |w|^2 (1 - cos^2), so cos^2 is exactly the best
        # fractional error reduction ANY corrector could extract from this
        # signal. Logged as the three inner products so the pooled bound can be
        # formed per band afterwards.
        # The comparison that matters is cos2_probe vs cos2_vel: velocity
        # extrapolation is a proposal the cell can already form from inputs it
        # is handed (P(x-1) as the residual anchor, delta as block two), so the
        # probe only carries new information if it beats that.
        u_t = readout_n[:, :3] - prev_n[:, :3]
        w_t = gt_n[:, :3] - prev_n[:, :3]
        uw, uu, ww = inner_products(u_t, w_t)
        row["uw_probe_t"], row["uu_probe_t"], row["ww_t"] = uw.cpu(), uu.cpu(), ww.cpu()
        u_r = rotvec_between(prev_n[:, 3:7], readout_n[:, 3:7])
        w_r = rotvec_between(prev_n[:, 3:7], gt_n[:, 3:7])
        uw, uu, ww = inner_products(u_r, w_r)
        row["uw_probe_r"], row["uu_probe_r"], row["ww_r"] = uw.cpu(), uu.cpu(), ww.cpu()
        if v >= 2:
            uv_t = vel_n[:, :3] - prev_n[:, :3]
            uw, uu, _ = inner_products(uv_t, w_t)
            row["uw_vel_t"], row["uu_vel_t"] = uw.cpu(), uu.cpu()
            uv_r = rotvec_between(prev_n[:, 3:7], vel_n[:, 3:7])
            uw2, uu2, _ = inner_products(uv_r, w_r)
            row["uw_vel_r"], row["uu_vel_r"] = uw2.cpu(), uu2.cpu()
            # THE CROSS TERM, and the whole reason the joint fit is possible.
            # cos2_probe > cos2_vel compares two MARGINAL fits, and the probe and
            # the velocity proposal are correlated (both partly encode
            # inter-frame motion), so that inequality can hold while the probe
            # adds nothing delta does not already carry — the trajectory-only
            # ceiling restated in new units, which is the confusion this program
            # keeps walking into. The P4.3-specific claim is INCREMENTAL: does
            # the probe explain variance in w that velocity cannot? That is a
            # partial R^2, and without <u_vel, u_probe> the 2-parameter joint fit
            # is not identifiable and DELTA_R2 cannot be recovered post hoc from
            # anything else stored here.
            row["svp_t"] = (uv_t * u_t).sum(-1).cpu()
            row["svp_r"] = (uv_r * u_r).sum(-1).cpu()
            # PERMUTATION NULL for the DELTA_R2 statistic itself: the same probe
            # proposal, rolled to a DIFFERENT SCENE. Same magnitudes, same
            # geometry, same estimator, same fit degrees of freedom -- only the
            # correspondence to THIS frame is destroyed. This is the program's
            # own standard ("a metric that survives its relevant shuffle is not
            # using that channel") applied to the statistic rather than to the
            # arm, and it is the null a block bootstrap cannot provide: a
            # bootstrap CI measures sampling variance around whatever the
            # estimator converges to, not whether that value is reachable by
            # chance. If dR2_shuf ~ dR2_honest, the finding is dead.
            if u_t.shape[0] > 1:
                us_t = torch.roll(u_t, shifts=1, dims=0)
                us_r = torch.roll(u_r, shifts=1, dims=0)
                row["uu_shuf_t"] = (us_t * us_t).sum(-1).cpu()
                row["uw_shuf_t"] = (us_t * w_t).sum(-1).cpu()
                row["svs_t"] = (uv_t * us_t).sum(-1).cpu()
                row["uu_shuf_r"] = (us_r * us_r).sum(-1).cpu()
                row["uw_shuf_r"] = (us_r * w_r).sum(-1).cpu()
                row["svs_r"] = (uv_r * us_r).sum(-1).cpu()
        row["e_null_t"] = gt_n[:, :3].norm(dim=-1).cpu()
        row["e_null_r"] = rot_err_deg(
            torch.tensor([[0., 0., 0., 1., 0., 0., 0.]], device=gt_n.device).expand_as(gt_n),
            gt_n,
        ).cpu()
        rows.append(row)
    return rows


BANDS = (("early", 1, NV // 4), ("mid", NV // 4 + 1, 3 * NV // 4 - 1),
         ("late", 3 * NV // 4, NV - 1))
SIX = ("uu_vel_{}", "uu_probe_{}", "svp_{}", "uw_vel_{}", "uw_probe_{}", "ww_{}")


def accumulate_scenes(scene_store, rows, base_id):
    """Sum the six inner products per SCENE per band.

    The block unit has to be the scene, not the view: views inside one sequence
    share a camera, a trajectory and an accumulated state, so treating them as
    independent draws inflates the effective sample size by ~16x and makes any
    null far too tight. n_eff is much nearer the scene count than the view count.
    """
    for r in rows:
        v = int(r["view"])
        band = next((b for b, lo, hi in BANDS if lo <= v <= hi), None)
        if band is None:
            continue
        ok = r["ok"].numpy().astype(bool)
        for ax in ("t", "r"):
            keys = [k.format(ax) for k in SIX]
            if not all(k in r for k in keys):
                continue
            for j in np.nonzero(ok)[0]:
                d = scene_store.setdefault(base_id + int(j), {}).setdefault((band, ax), [0.0] * 6)
                for i, k in enumerate(keys):
                    d[i] += float(r[k].numpy()[j])


def bootstrap_dr2(scene_store, n_draws=1000, seed=17):
    """Block bootstrap over SCENES: resample whole sequences with replacement,
    re-pool the six scalars, recompute DELTA_R2. Returns the 95% CI per band."""
    rng = np.random.default_rng(seed)
    ids = sorted(scene_store)
    out = {}
    for band, _, _ in BANDS:
        for ax in ("t", "r"):
            rows = [scene_store[i].get((band, ax)) for i in ids]
            rows = [r for r in rows if r is not None]
            if len(rows) < 8:
                continue
            M = np.array(rows)  # (n_scenes, 6)
            draws = []
            for _ in range(n_draws):
                S = M[rng.integers(0, len(M), len(M))].sum(0)
                Svv, Spp, Svp, Svw, Spw, Sww = S
                det = Svv * Spp - Svp * Svp
                if det <= 0 or Sww <= 0 or Svv <= 0:
                    continue
                r2j = (Spp * Svw**2 - 2 * Svp * Svw * Spw + Svv * Spw**2) / (det * Sww)
                draws.append(r2j - Svw**2 / (Svv * Sww))
            if draws:
                out[f"{band}_{ax}"] = dict(
                    lo=float(np.percentile(draws, 2.5)),
                    hi=float(np.percentile(draws, 97.5)),
                    median=float(np.percentile(draws, 50)),
                    n_scenes=len(M),
                )
    return out


def accumulate(store, rows):
    for r in rows:
        v = int(r["view"])
        ok = r["ok"].numpy().astype(bool)
        if not ok.any():
            continue
        d = store.setdefault(v, {})
        for k, t in r.items():
            if k in ("view", "ok"):
                continue
            d.setdefault(k, []).append(t.numpy()[ok])


def summarize(store, label):
    """Per-view means and the two ratios that matter. Ratio-of-means, not
    mean-of-ratios: per-sample ratios blow up wherever e_lag ~ 0."""
    views = sorted(store)
    out = dict(label=label, views=views, n_rows={}, series={}, ratio={})
    for key in ("e_probe_t", "e_probe_r", "e_lag_t", "e_lag_r",
                "e_vel_t", "e_vel_r", "e_head_t", "e_head_r",
                "gap_t", "gap_r", "dlag_t", "dlag_r", "e_null_t", "e_null_r",
                "uw_probe_t", "uu_probe_t", "ww_t", "uw_probe_r", "uu_probe_r", "ww_r",
                "uw_vel_t", "uu_vel_t", "uw_vel_r", "uu_vel_r", "svp_t", "svp_r",
                "uu_shuf_t", "uw_shuf_t", "svs_t", "uu_shuf_r", "uw_shuf_r", "svs_r"):
        out["series"][key] = [
            float(np.concatenate(store[v][key]).mean()) if key in store[v] else float("nan")
            for v in views
        ]
    out["n_rows"] = {int(v): int(sum(len(a) for a in store[v]["e_lag_t"])) for v in views}
    # SUMS (not means) for the cos^2 bound: pooling over a band needs the summed
    # inner products, since the optimal single gain over the band is
    # a* = sum<u,w>/sum<u,u> and the achievable reduction is
    # (sum<u,w>)^2 / (sum<u,u> * sum|w|^2).
    out["sums"] = {}
    for key in ("uw_probe_t", "uu_probe_t", "ww_t", "uw_probe_r", "uu_probe_r", "ww_r",
                "uw_vel_t", "uu_vel_t", "uw_vel_r", "uu_vel_r", "svp_t", "svp_r",
                "uu_shuf_t", "uw_shuf_t", "svs_t", "uu_shuf_r", "uw_shuf_r", "svs_r"):
        out["sums"][key] = [
            float(np.concatenate(store[v][key]).sum()) if key in store[v] else float("nan")
            for v in views
        ]
    for comp in ("lag", "vel"):
        for ax in ("t", "r"):
            num = np.array(out["series"][f"e_probe_{ax}"], dtype=float)
            den = np.array(out["series"][f"e_{comp}_{ax}"], dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["ratio"][f"probe_over_{comp}_{ax}"] = (num / den).tolist()
    # gap / dlag: is the observation an independent measurement, or a restatement
    # of the pose it was meant to correct? (see rollout() for the reading)
    for ax in ("t", "r"):
        num = np.array(out["series"][f"gap_{ax}"], dtype=float)
        den = np.array(out["series"][f"dlag_{ax}"], dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["ratio"][f"gap_over_dlag_{ax}"] = (num / den).tolist()
            out["ratio"][f"gap_over_eprobe_{ax}"] = (
                num / np.array(out["series"][f"e_probe_{ax}"], dtype=float)).tolist()
            # saturation: how much better than predicting the origin is anyone?
            out["ratio"][f"probe_over_null_{ax}"] = (
                np.array(out["series"][f"e_probe_{ax}"], dtype=float)
                / np.array(out["series"][f"e_null_{ax}"], dtype=float)).tolist()
            out["ratio"][f"lag_over_null_{ax}"] = (
                np.array(out["series"][f"e_lag_{ax}"], dtype=float)
                / np.array(out["series"][f"e_null_{ax}"], dtype=float)).tolist()
    return out


def _band(series, views, lo, hi):
    """Mean of `series` over view indices in [lo, hi] — the slope handles."""
    sel = [s for s, v in zip(series, views) if lo <= v <= hi and np.isfinite(s)]
    return float(np.mean(sel)) if sel else float("nan")


def report(summary):
    views = summary["views"]
    label = summary["label"]
    print(f"\n{'='*78}\n{label}\n{'='*78}", flush=True)
    print(f"{'view':>5} {'e_lag_t':>10} {'e_vel_t':>10} {'e_probe_t':>10} "
          f"{'p/lag':>7} {'p/vel':>7} | {'e_lag_r':>8} {'e_vel_r':>8} {'e_probe_r':>9} "
          f"{'p/vel':>7}", flush=True)
    for i, v in enumerate(views):
        if v % 4 and v != views[-1]:
            continue  # every 4th view keeps the table readable; JSON has all
        s, r = summary["series"], summary["ratio"]
        print(f"{v:>5} {s['e_lag_t'][i]:>10.5f} {s['e_vel_t'][i]:>10.5f} "
              f"{s['e_probe_t'][i]:>10.5f} {r['probe_over_lag_t'][i]:>7.3f} "
              f"{r['probe_over_vel_t'][i]:>7.3f} | {s['e_lag_r'][i]:>8.3f} "
              f"{s['e_vel_r'][i]:>8.3f} {s['e_probe_r'][i]:>9.3f} "
              f"{r['probe_over_vel_r'][i]:>7.3f}", flush=True)

    lo_hi = [(1, NV // 4), (3 * NV // 4, NV - 1)]
    verdict = {}
    for name in ("probe_over_lag_t", "probe_over_vel_t", "probe_over_lag_r",
                 "probe_over_vel_r"):
        early = _band(summary["ratio"][name], views, *lo_hi[0])
        late = _band(summary["ratio"][name], views, *lo_hi[1])
        verdict[name] = dict(early=early, late=late,
                             slope=(late - early) if np.isfinite(early + late) else float("nan"))
    # drift itself: does the lag baseline degrade with view index at all?
    for name in ("e_lag_t", "e_vel_t", "e_probe_t", "e_head_t"):
        early = _band(summary["series"][name], views, *lo_hi[0])
        late = _band(summary["series"][name], views, *lo_hi[1])
        verdict[name] = dict(early=early, late=late,
                             growth=(late / early) if early else float("nan"))
    summary["verdict"] = verdict

    print(f"\n  early = views {lo_hi[0][0]}..{lo_hi[0][1]}, "
          f"late = views {lo_hi[1][0]}..{lo_hi[1][1]}", flush=True)
    for name in ("probe_over_vel_t", "probe_over_vel_r", "probe_over_lag_t"):
        d = verdict[name]
        print(f"  {name:<20} early {d['early']:.3f} -> late {d['late']:.3f} "
              f"(slope {d['slope']:+.3f})", flush=True)
    print(f"  drift check: e_lag_t grows {verdict['e_lag_t']['growth']:.2f}x "
          f"early->late; e_probe_t grows {verdict['e_probe_t']['growth']:.2f}x", flush=True)
    for ax, name in (("t", "translation"), ("r", "rotation")):
        g = _band(summary["ratio"][f"gap_over_eprobe_{ax}"], views, *lo_hi[1])
        summary["verdict"][f"gap_over_eprobe_{ax}"] = dict(late=g)
        print(f"  independence ({name}, late): gap / e_probe = {g:.3f}  -> "
              f"{1-g:.0%} of the observation's error is COMMON MODE with the fed-back pose",
              flush=True)
    for ax, name in (("t", "translation"), ("r", "rotation")):
        pn = _band(summary["ratio"][f"probe_over_null_{ax}"], views, *lo_hi[1])
        ln = _band(summary["ratio"][f"lag_over_null_{ax}"], views, *lo_hi[1])
        summary["verdict"][f"probe_over_null_{ax}"] = dict(late=pn)
        print(f"  SATURATION ({name}, late): e_probe/e_null = {pn:.3f}, "
              f"e_lag/e_null = {ln:.3f}"
              f"{'   <-- WARNING: near 1.0 means every estimator is at the trivial baseline and NO ratio above is informative' if min(pn, ln) > 0.9 else ''}",
              flush=True)
    # ---- THE DIRECTION TEST -------------------------------------------------
    print("\n  cos^2 = best fractional error reduction ANY corrector could extract")
    print("          from that signal (optimal gain a* = <u,w>/<u,u>).")
    for lo, hi, nm in ((1, NV // 4, "early"), (NV // 4 + 1, 3 * NV // 4 - 1, "mid"),
                       (3 * NV // 4, NV - 1, "late")):
        def pooled(src, ax):
            sel = [i for i, v in enumerate(views) if lo <= v <= hi]
            uw = sum(summary["sums"][f"uw_{src}_{ax}"][i] for i in sel
                     if np.isfinite(summary["sums"][f"uw_{src}_{ax}"][i]))
            uu = sum(summary["sums"][f"uu_{src}_{ax}"][i] for i in sel
                     if np.isfinite(summary["sums"][f"uu_{src}_{ax}"][i]))
            ww = sum(summary["sums"][f"ww_{ax}"][i] for i in sel
                     if np.isfinite(summary["sums"][f"ww_{ax}"][i]))
            return (uw * uw / (uu * ww)) if uu > 0 and ww > 0 else float("nan")
        row = {f"{s_}_{a_}": pooled(s_, a_) for s_ in ("probe", "vel") for a_ in ("t", "r")}
        # INCREMENTAL fit: how much of w does the probe explain that velocity
        # cannot? Joint 2-parameter least squares min_(a,b) |w - a*u_vel - b*u_probe|^2,
        #   R2_joint = [Svw Spw] G^-1 [Svw Spw]^T / Sww,  G = [[Svv, Svp],[Svp, Spp]]
        #   DELTA_R2 = R2_joint - cos2_vel      <- P4.3 lives or dies here
        # cos2_probe large with DELTA_R2 ~ 0 is the trap: it means the probe is
        # redundant with a proposal the cell can already form from delta.
        sel = [i for i, v in enumerate(views) if lo <= v <= hi]
        S = lambda k: sum(summary["sums"][k][i] for i in sel
                          if np.isfinite(summary["sums"][k][i]))
        for ax in ("t", "r"):
            Svv, Spp, Svp = S(f"uu_vel_{ax}"), S(f"uu_probe_{ax}"), S(f"svp_{ax}")
            Svw, Spw, Sww = S(f"uw_vel_{ax}"), S(f"uw_probe_{ax}"), S(f"ww_{ax}")
            corr = Svp / np.sqrt(Svv * Spp) if Svv > 0 and Spp > 0 else float("nan")
            det = Svv * Spp - Svp * Svp
            if not np.isfinite(corr) or abs(corr) > 0.99 or det <= 0:
                # Near-collinear proposals ARE the DELTA_R2 ~ 0 finding; say so
                # rather than inverting a singular G and reporting noise.
                row[f"dR2_{ax}"], row[f"r2joint_{ax}"] = float("nan"), float("nan")
            else:
                r2j = (Spp * Svw**2 - 2 * Svp * Svw * Spw + Svv * Spw**2) / (det * Sww)
                row[f"r2joint_{ax}"] = r2j
                row[f"dR2_{ax}"] = r2j - row[f"vel_{ax}"]
            row[f"corr_{ax}"] = corr
            # same arithmetic on the SHUFFLED proposal
            Sss, Sws, Svs = S(f"uu_shuf_{ax}"), S(f"uw_shuf_{ax}"), S(f"svs_{ax}")
            dets = Svv * Sss - Svs * Svs
            if dets > 0 and Sss > 0:
                r2s = (Sss * Svw**2 - 2 * Svs * Svw * Sws + Svv * Sws**2) / (dets * Sww)
                row[f"dR2shuf_{ax}"] = r2s - Svw**2 / (Svv * Sww)
            else:
                row[f"dR2shuf_{ax}"] = float("nan")
        summary["verdict"][f"cos2_{nm}"] = row
        print(f"  cos^2 {nm:5s}: probe t={row['probe_t']:.4f} r={row['probe_r']:.4f}   "
              f"|  vel t={row['vel_t']:.4f} r={row['vel_r']:.4f}", flush=True)
        print(f"        {'':5s}  DELTA_R2 (probe beyond velocity) t={row['dR2_t']:+.4f} "
              f"r={row['dR2_r']:+.4f}   |  corr(u_vel,u_probe) t={row['corr_t']:+.3f} "
              f"r={row['corr_r']:+.3f}", flush=True)
        print(f"        {'':5s}  SHUFFLED-PROBE NULL             t={row['dR2shuf_t']:+.4f} "
              f"r={row['dR2shuf_r']:+.4f}   <- must be ~0 or the finding is dead",
              flush=True)

    hl = verdict["probe_over_vel_t"]["late"]
    print(f"\n  HEADLINE (translation, late views): e_probe / e_vel = {hl:.3f}"
          f"  -> {'probe BEATS velocity extrapolation' if hl < 1.0 else 'probe does NOT beat velocity extrapolation'}",
          flush=True)
    return summary


def run_arm(ckpt, label, do_probe, loader):
    print(f"\n>>> {label}\n>>> {ckpt}", flush=True)
    model = load_model(ckpt, device=DEVICE, verbose=True).eval()
    model.views_per_step = 1
    # The conditioned arm must roll closed-loop; arm A must NOT (its forward is
    # already pose-free, and setting this would build ray maps it never saw).
    model.feed_prev_pred = bool(do_probe)
    assert getattr(model, "pose_head_flag", False), "no pose head — nothing to read"
    if do_probe:
        gru = getattr(model, "pose_gru", None)
        assert gru is not None, f"{label}: expected a pose_gru checkpoint"
        print(f"    pose_gru: mode={gru.mode} input={gru.input_mode} "
              f"iters={gru.iters} img_feat={gru.img_feat}", flush=True)
    else:
        assert getattr(model, "pose_gru", None) is None, (
            f"{label}: expected a NO-GRU baseline; got a pose_gru checkpoint"
        )
    store, scene_store, t0 = {}, {}, time.time()
    for bi, batch in enumerate(loader):
        rows = rollout(model, to_device(batch), do_probe)
        accumulate(store, rows)
        accumulate_scenes(scene_store, rows, bi * BS)
        if (bi + 1) % 5 == 0:
            print(f"    batch {bi+1}/{len(loader)}  ({time.time()-t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    summary = report(summarize(store, label))
    summary["bootstrap_dR2"] = bootstrap_dr2(scene_store)
    # Per-scene (Svv, Spp, Svp, Svw, Spw, Sww) per band. Dumped because a
    # trained/placebo pair shares the scene list and the rollout exactly, so the
    # correct test is a PAIRED bootstrap of the DIFFERENCE on common scenes --
    # far tighter than comparing two marginal CIs, and overlapping marginal CIs
    # do NOT establish no-difference. Without these the pairing is unrecoverable
    # from the artifact.
    summary["scene_terms"] = {
        f"{band}_{ax}": {str(i): scene_store[i][(band, ax)]
                         for i in sorted(scene_store) if (band, ax) in scene_store[i]}
        for band, _, _ in BANDS for ax in ("t", "r")
    }
    summary["scene_terms_order"] = ["Svv", "Spp", "Svp", "Svw", "Spw", "Sww"]
    print("\n  BLOCK BOOTSTRAP over SCENES (1000 draws, 95% CI on DELTA_R2).")
    print("  Views inside a sequence are NOT independent, so the block unit is the")
    print("  scene; a per-view null would be ~16x too tight.")
    for band, _, _ in BANDS:
        for ax, nm in (("t", "trans"), ("r", "rot  ")):
            ci = summary["bootstrap_dR2"].get(f"{band}_{ax}")
            if ci:
                excl = "EXCLUDES 0" if ci["lo"] > 0 else "includes 0"
                print(f"    {band:5s} {nm}: dR2 median {ci['median']:+.4f}  "
                      f"95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]  {excl}  "
                      f"(n={ci['n_scenes']} scenes)", flush=True)
    return summary


def main():
    guard_falsifier_env()
    assert torch.cuda.is_available(), "GPU required"
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    ds = DL3DV_Multi(
        split="test", ROOT=DATA_ROOT, resolution=[RES], transform=ImgNorm,
        num_views=NV, n_corres=0, aug_crop=0, allow_repeat=True,
        force_consecutive_frame_sampling=True, seed=SEED,
    )
    n_seq = N_BATCHES * BS
    # Spread over the split (scene order is lab-correlated) and permute, so a
    # partial run is not a lab-contiguous slice. Fixed seed => reproducible, and
    # BOTH arms see the identical scenes, which is what makes the two brackets
    # comparable at all.
    idxs = np.unique(np.linspace(0, len(ds) - 1, num=n_seq).astype(int))
    idxs = np.random.default_rng(SEED).permutation(idxs).tolist()
    loader = DataLoader(
        _SeqSet(ds, idxs), batch_size=BS, shuffle=False, num_workers=WORKERS,
        collate_fn=_collate, pin_memory=False, persistent_workers=bool(WORKERS),
    )
    print(f"data: {len(idxs)} scenes / {len(loader)} batches x {BS} x {NV} views",
          flush=True)

    results = []
    results.append(run_arm(GRU_CKPT, "GRU arm (r8) — probe is OUT OF DISTRIBUTION: "
                                     "positive => strong go, negative => suggestive only",
                           True, loader))
    results.append(run_arm(A_CKPT, "arm A (trained pose-free) — the CEILING: its "
                                   "ordinary forward IS the readout, no probe construction",
                           False, loader))

    out = dict(
        meta=dict(
                  # THE MANIPULATION, recorded in the artifact rather than the
                  # filename. Without this a trained/placebo pair is
                  # indistinguishable in meta and the control that defines every
                  # DIFF is unrecoverable.
                  placebo=os.environ.get("P43_PLACEBO") == "1",
                  placebo_manipulation=(
                      "O(x) batch-rolled by 1 along dim 0, per view index, applied "
                      "after the rollout and before ANY scoring. PRESERVES: the "
                      "marginal distribution of observations, the head that produced "
                      "them, the view index, the scale. BREAKS: the pairing with this "
                      "scene. NOT A PURE NULL -- a rolled pose still carries "
                      "SCENE-GENERIC structure (what a head-produced pose at view 60 "
                      "typically looks like, including the dataset's camera-motion "
                      "statistics), which is why the placebo floor GROWS with view "
                      "index. It is therefore CONSERVATIVE: trained-minus-placebo "
                      "UNDERSTATES the scene-specific content."
                  ) if os.environ.get("P43_PLACEBO") == "1" else None,
                  falsifier_env={k: os.environ[k] for k in (
                      "POSE_GRU_PROBE_SHUFFLE", "POSE_GRU_PROBE_LAG",
                      "POSE_GRU_FORCE_REFINE", "PREV_PRED_RAY_SHUFFLE",
                  ) if os.environ.get(k)},
                  gru_ckpt=GRU_CKPT, a_ckpt=A_CKPT, data_root=DATA_ROOT,
                  scenes=len(idxs), num_views=NV, batch_size=BS,
                  resolution=list(RES), seed=SEED,
                  units="translation = raw head units (L2); rotation = degrees",
                  caveat="RAW errors, NOT the per-batch-normalized quantities "
                         "PoseGRULoss reports as gru_trans_err_gtscale. Do not "
                         "cross-read the two."),
        arms=results,
    )
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_JSON}", flush=True)

    print(f"\n{'='*78}\nHOW TO READ THIS\n{'='*78}")
    print("  headline = e_probe / e_vel on the LATE views (translation).")
    print("  r8   < 1.0  -> STRONG GO: an untrained-for readout already beats")
    print("                 trajectory extrapolation; P4.3 trains for it.")
    print("  r8  >= 1.0  -> SUGGESTIVE ONLY, NOT A REFUTATION. The r8 probe is")
    print("                 out of distribution; read arm A before concluding.")
    print("  arm A < 1.0 -> a trained-for pose-free readout DOES observe drift,")
    print("                 so a negative r8 means 'must be trained for', which")
    print("                 is what P4.3 does — not 'the premise is false'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
