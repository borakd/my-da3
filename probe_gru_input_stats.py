#!/usr/bin/env python
"""probe_gru_input_stats.py — what SCALE does the PoseGRU actually see?

P3.4 of GRU_GAP_CLOSURE_DIRECTIVE.md, the measurement half. The rung-0 probe
showed the module has NO scale invariance: handed a numerically PERFECT pose in
the wrong scale it scored ATE 0.1165, worse than the 0.0888 it gets from its own
stale one. So before anything feeds the GRU a better pose, its input has to be
normalized — and normalizing needs the real per-dimension distribution of
`gru_in` (built at model.py: the `gru_in = (...)` cat inside
_forward_decoder_group_step), not an assumed one.

WHAT IT MEASURES. A trained checkpoint is rolled out closed-loop (feed_prev_pred,
no grad) over real DROID test sequences, and a forward-pre-hook on `pose_gru`
captures the exact tensor the cell consumes at every view. For A4 (input_mode
'pose_delta') that is 14 columns:

    [ absT(3) | absQ(4) | delta_t(3) | delta_q(4) ]      quaternion REAL PART FIRST

and the finding the directive is built on is that absT runs ~0.5-0.8 head-units
at the supervised views while delta_t — the only informative channel, the loop's
velocity — runs ~0.018-0.068, i.e. tens of times smaller. Reported per dimension:
std (the 1/std the gain is filled from), the |absT| and |delta_t| norm
percentiles, and the PER-SEQUENCE scale factor, which is what prices the dynamic
(scale-equivariant) version the dossier actually prescribes.

HEADLINE (2026-08-12, measured; supersedes the round-2 estimates). On the train
split the velocity block sits **12.86x** below the pose block by median norm
(16.9x on test) — NOT the ~70x the directive carried, which was a ratio of gate
pre-activation stds resting on the same superseded head-scale constant. The
practical consequence: the delta_t gain wants to be **20-30**, not the 50-70
the directive prescribed, so following the old figure would over-correct the
velocity columns by roughly 3-4x.

VALIDATION GATE — this is the point of the script, not decoration. Two tiers,
because they catch different failures (see the VALIDATION REFERENCE block for
the full argument):

    WIRING  (non-circular)  delta_t p95/p50 shape vs the GT-derived 3.778;
                            p50 and p95 must imply the SAME head scale.
    SCALE   (regression)    absolute values vs what THIS probe measured on the
                            train split on 2026-08-12, +/-50%.

Both tiers gate the exit code. A failure means the gain vector must not be
used — but read the tiers differently: a WIRING failure says the probe is
mis-wired, while a SCALE failure alone says something downstream of it moved
(new checkpoint, changed forward path, changed loader).

WHICH CHECKPOINT. A D-arm one — `captain_gru_v3_a4_g3_finetune`, the honest
A4xG3xF0xR1 arm at lr 1e-5 whose graded window is exactly views 48..63. Pass a
keep_freq/final snapshot, never checkpoint-last/best of a LIVE job (they are
rewritten every epoch and the read races the writer). R>1 checkpoints are handled
too: the hook fires once per cell iteration and only ITERATE 0 is kept, because
iterates 1..n-1 re-feed the GRU's own output rather than the fed-back head pose.

Usage:
    python probe_gru_input_stats.py [ckpt]
    sbatch probe_gru_input_stats.sbatch [ckpt]

Env: PROBE_CKPT PROBE_BATCHES(200) PROBE_BS(4) PROBE_NV(64) PROBE_WORKERS(8)
     PROBE_OUT(gru_input_stats.json) DL3DV_TEST_ROOT
"""
import json
import os
import sys
import time

# Self-locating: derive the checkout from THIS file, never a hardcoded path.
# sys.path.insert(0, <nonexistent>) SILENTLY SUCCEEDS, so a stale hardcoded
# worktree makes imports fall through to PYTHONPATH -- letting you verify a
# DIFFERENT checkout than the file you are editing, with no warning. Assert.
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
from dust3r.model import load_model
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

DATA_ROOT = os.environ.get(
    "DL3DV_TEST_ROOT",
    "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi",
)
# The gain is filled from the distribution the module is TRAINED on, and the
# round-2 audit's figures are train-side. Point DL3DV_TEST_ROOT at
# .../train/dl3dv_multi and set PROBE_AUG_CROP=16 to reproduce the training
# loader exactly (the train config is aug_crop=16 + SeqColorJitter; only the
# crop touches geometry, so that is the one worth mirroring).
AUG_CROP = int(os.environ.get("PROBE_AUG_CROP", 0))
DEFAULT_CKPT = os.environ.get(
    "PROBE_CKPT",
    os.path.join(
        os.environ.get("CKPT_ROOT", "/gpfs/projects/etur59/koc821022/checkpoints"),
        "captain_cut3r_finetune_aug_full",
        "captain_gru_v3_a4_g3_finetune",
        "checkpoint-final.pth",
    ),
)
N_BATCHES = int(os.environ.get("PROBE_BATCHES", 200))
BS = int(os.environ.get("PROBE_BS", 4))
NV = int(os.environ.get("PROBE_NV", 64))
WORKERS = int(os.environ.get("PROBE_WORKERS", 8))
# Stop sampling early enough to still PRINT the report and the validation gate.
# acc_debug caps at 2h; being SIGKILLed at the wall would leave only the last
# incremental JSON and no verdict, which is the one thing this script is for.
MAX_SECONDS = int(os.environ.get("PROBE_MAX_SECONDS", 90 * 60))
OUT_JSON = os.environ.get("PROBE_OUT", os.path.join(WORKTREE, "gru_input_stats.json"))
RES = (320, 192)
SEED = 777
DEVICE = "cuda"

# The TBPTT gradient window: loss_of_one_batch_tbptt grades only the last
# GRADED_CHUNKS=4 chunks of 4 views (inference.py, `chunk_id < num_chunks - 4`),
# so at num_views=64 the supervised views are 48..63 -- exactly the window the
# directive's absT figure is quoted over. Derived, not hardcoded, so a different
# PROBE_NV still reports the right slice.
GRADED_VIEWS = 16

# ---------------------------------------------------------------------------
# VALIDATION REFERENCE — read this before changing a band.
#
# The expectations below are a PRIMARY MEASUREMENT (this probe, 2026-08-12:
# 50,400 rows, 800 scenes per split, two splits that agree). They replace the
# original P3.4 targets, which were a SECONDARY INFERENCE: round-2 GT-space
# statistics scaled by an assumed head-scale constant. That chain is visible in
# captain_gru_autoresearch/evidence/round2_evidence.md:159 —
#
#     GT inter-frame |dt| p50 0.0045 / p95 0.017 GT-units ... Head scale ~4x GT
#     (prior audit) -> fed-back absT ~0.01-3.1 head-units (typ 0.5-0.8 at
#     supervised views); delta_t ~0.018-0.068 head-units
#
# — 0.0045*4 = 0.018 and 0.017*4 = 0.068, exactly. (The sibling constant
# HEAD_SCALE = 0.37 in verify_gru_oracle.py is the same estimate from the other
# direction.) Measured, the factor is 7.0x on test and 8.1-9.2x on train, i.e.
# the constant was low by ~2x. GRU_GAP_CLOSURE_DIRECTIVE.md was corrected
# accordingly on 2026-08-12; the superseded values are kept in the JSON under
# `superseded_expected` so this is an auditable revision, not a silent one.
#
# WHY THE SHAPE CHECK IS THE ONE THAT VALIDATES. Re-pointing an expectation at
# your own measurement is circular — a probe compared against its own output
# always agrees. So the checks are split by what they can actually catch:
#   * WIRING checks (gating, NON-circular). Scale-INVARIANT shape properties.
#     A mis-wired probe (wrong columns, wrong view window, wrong tensor) breaks
#     these; a head-scale disagreement cannot touch them. These passing on both
#     splits is what established that round 2 predicted the DISTRIBUTION
#     correctly and only its scale constant was wrong.
#   * SCALE checks (gating, REGRESSION-only). Absolute bands re-pointed at the
#     measured train-split values. They cannot validate the run that produced
#     them; their value is catching future drift — a different checkpoint, a
#     changed forward path, a loader change. Read them as "still what we
#     measured on 2026-08-12", never as independent confirmation.
# The tolerance is deliberately NOT widened: widening it around the old, wrong
# centre is what would have made the check circular AND toothless.
GT_DT_P50 = 0.0045               # round2_evidence.md:159, GT units
GT_DT_P95 = 0.017
GT_ABST_SUP_MEAN = 0.206         # GT view-0-relative |t|, mean over views 48-63
ASSUMED_HEAD_SCALE = 4.0         # the superseded "prior audit" factor
TOL = 1.5                        # multiplicative band, unchanged from before
# MEASURED on the TRAIN split (aug_crop=16, 800 scenes) — the distribution the
# gain acts on, because that is where gradients flow. gru_input_stats_train.json.
MEASURED_ABST_SUP = 0.99473
MEASURED_DT_P50 = 0.04125
MEASURED_DT_P95 = 0.13686
EXPECT_ABST_RANGE = (MEASURED_ABST_SUP / TOL, MEASURED_ABST_SUP * TOL)
EXPECT_DT_P50 = MEASURED_DT_P50
EXPECT_DT_P95 = MEASURED_DT_P95
# The superseded targets, carried into the JSON for provenance.
SUPERSEDED_EXPECTED = dict(
    source="round2_evidence.md:159 GT stats x assumed head scale "
           f"{ASSUMED_HEAD_SCALE:g}x (superseded 2026-08-12)",
    absT_sup_range=[0.5, 0.8],
    delta_t_p50=GT_DT_P50 * ASSUMED_HEAD_SCALE,
    delta_t_p95=GT_DT_P95 * ASSUMED_HEAD_SCALE,
    absT_over_delta_t_median=70.0,
)
# Head-scale-invariant: the SHAPE of the inter-frame motion distribution. Kept
# on the GT ratio deliberately — it is the one round-2 quantity that survived
# the head-scale error, so it stays an INDEPENDENT reference rather than
# becoming another copy of our own measurement.
EXPECT_DT_P95_OVER_P50 = GT_DT_P95 / GT_DT_P50   # 3.778
SHAPE_TOL = 1.25
# Two independent percentiles must imply the SAME head scale. This is the real
# wiring test: a probe reading the wrong columns has no reason to produce a
# consistent factor across p50 and p95.
CONSISTENCY_TOL = 1.20

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
np.set_printoptions(precision=5, suppress=True)


def guard_falsifier_env():
    """The falsifier env vars silently change what the GRU consumes.

    PREV_PRED_RAY_SHUFFLE rolls the fed-back pose across the batch (so absT
    would be another sample's), POSE_GRU_FORCE_ITERS overrides the R lever (so
    iterate 0 would not be the only fed-back-pose call), and the hidden/feature
    knobs change the trajectory the loop produces. Any of them would corrupt
    this measurement without a single warning -- refuse to run instead.
    """
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
        )
        if os.environ.get(k)
    ]
    assert not live, (
        f"falsifier env vars are set: {live}. They change the GRU's input, so "
        "the measured distribution would not be the training-time one. Unset them."
    )


def to_device(view):
    return {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in view.items()}


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


@torch.no_grad()
def rollout_capture(model, batch, captured):
    """Sequential per-view decode (mirrors loss_of_one_batch_tbptt, no grad).

    Returns a list of (view_index, gru_in) for ITERATE 0 of each view. View 0
    makes no GRU call at all -- it takes the masked_ray_map_token path and only
    resets the stashes -- so the first entry is view 1.
    """
    (feat, pos, shape), (isf, imem, sf, sp, mem) = model._forward_encoder(batch)
    out = []
    for v in range(len(batch)):
        captured.clear()
        _res, (sf, mem) = model._forward_decoder_group_step(
            views=batch, view_indices=[v],
            feat_group=[feat[v]], pos_group=[pos[v]], shape_group=[shape[v]],
            init_state_feat=isf, init_mem=imem,
            state_feat=sf, state_pos=sp, mem=mem,
        )
        if captured:
            # Iterate 0 only: under the R lever, iterates 1..n-1 are fed the
            # GRU's OWN output, not the fed-back head pose, so their scale is
            # the module's own and would contaminate the distribution.
            out.append((v, captured[0]))
    return out


def _round_sig(x, sig=1):
    if not np.isfinite(x) or x == 0.0:
        return 0.0
    return float(round(float(x), -int(np.floor(np.log10(abs(float(x))))) + (sig - 1)))


def build_verdict(absT_graded_mean, dt_p50, dt_p95, has_delta):
    """The validation verdict as a pure function of the three summary scalars.

    Split out so the gate can be re-evaluated on an ALREADY-MEASURED run
    (patch_probe_verdict.py) without re-running 800 scenes on a GPU, and
    without a second copy of the thresholds drifting away from this one. The
    verdict genuinely is a function of these three numbers and the module
    constants — nothing else in the summary feeds it.

    Returns (implied_head_scale, validation_rows).
    """
    hs = dict(
        from_dt_p50=(dt_p50 / GT_DT_P50) if has_delta else float("nan"),
        from_dt_p95=(dt_p95 / GT_DT_P95) if has_delta else float("nan"),
        from_absT_sup=absT_graded_mean / GT_ABST_SUP_MEAN,
        assumed_by_directive=ASSUMED_HEAD_SCALE,
    )

    wiring, scale = [], []
    if has_delta:
        ratio = dt_p95 / max(dt_p50, 1e-12)
        wiring.append(("delta_t p95/p50 shape", ratio,
                       EXPECT_DT_P95_OVER_P50 / SHAPE_TOL,
                       EXPECT_DT_P95_OVER_P50 * SHAPE_TOL,
                       f"GT-derived {EXPECT_DT_P95_OVER_P50:.3f}, head-scale-invariant"))
        agree = max(hs["from_dt_p50"], hs["from_dt_p95"]) / max(
            min(hs["from_dt_p50"], hs["from_dt_p95"]), 1e-12)
        wiring.append(("head scale p50-vs-p95 agreement", agree,
                       1.0, CONSISTENCY_TOL,
                       "two percentiles must imply the same factor"))
        scale += [
            ("delta_t p50", dt_p50, EXPECT_DT_P50 / TOL, EXPECT_DT_P50 * TOL,
             f"measured {EXPECT_DT_P50:g} +/-50%"),
            ("delta_t p95", dt_p95, EXPECT_DT_P95 / TOL, EXPECT_DT_P95 * TOL,
             f"measured {EXPECT_DT_P95:g} +/-50%"),
        ]
    scale.append(("absT mean @ supervised views", absT_graded_mean,
                  EXPECT_ABST_RANGE[0], EXPECT_ABST_RANGE[1],
                  f"measured {MEASURED_ABST_SUP:g} +/-50%"))

    def mk(rows, kind):
        return [dict(name=n, value=v, lo=lo, hi=hi, expected=e, kind=kind,
                     gating=True,
                     passed=bool(np.isfinite(v) and lo <= v <= hi))
                for n, v, lo, hi, e in rows]

    return hs, mk(wiring, "wiring") + mk(scale, "regression")


def dim_labels(input_mode, input_dim):
    base = ["absT_x", "absT_y", "absT_z", "absQ_w", "absQ_x", "absQ_y", "absQ_z"]
    if input_mode == "pose_delta":
        base += ["delta_t_x", "delta_t_y", "delta_t_z",
                 "delta_q_w", "delta_q_x", "delta_q_y", "delta_q_z"]
    # F arms append image-feature columns after the pose block; they are already
    # LayerNormed and the gain leaves them at 1.0, so one label covers them.
    base += [f"imgfeat_{i}" for i in range(input_dim - len(base))]
    return base[:input_dim]


def summarize(vals, views, seqs, input_mode, input_dim, nv, meta):
    """vals: (N, input_dim) float64. views/seqs: (N,) int. All GRU calls seen."""
    labels = dim_labels(input_mode, input_dim)
    graded_start = max(1, nv - GRADED_VIEWS)
    graded = views >= graded_start
    has_delta = input_mode == "pose_delta"

    # |absT| and |delta_t| as NORMS -- the head-unit scale the directive quotes.
    absT = np.linalg.norm(vals[:, 0:3], axis=1)
    # delta_t is a structural ZERO at view 1: pose_delta_encoding(None, prev)
    # returns the identity motion because P(x-2) does not exist yet. Including
    # it would drag every percentile down and break the p50 check.
    dt = np.linalg.norm(vals[:, 7:10], axis=1) if has_delta else None
    dt_ok = (views >= 2) if has_delta else None

    def pct(a, q):
        return float(np.percentile(a, q)) if a.size else float("nan")

    std_all = vals.std(axis=0)
    std_graded = vals[graded].std(axis=0) if graded.any() else std_all

    # Per-sequence scale factor: the dossier's prescription for the DYNAMIC
    # (scale-equivariant) version is to divide by the mean |t| of the fed-back
    # trajectory. Its spread across sequences is what prices that version --
    # a static buffer can only ever normalize the middle of this distribution.
    seq_ids = np.unique(seqs)
    seq_factor = np.array([absT[seqs == s].mean() for s in seq_ids])

    # The gain itself: 1/std on the TRANSLATION columns, 1.0 on the quaternion
    # ones (a unit quaternion is already O(1) and its real part is ~1 with a
    # tiny std -- 1/std there would blow the column up, not condition it).
    trans_dims = [0, 1, 2] + ([7, 8, 9] if has_delta else [])
    def gain_from(std):
        g = [1.0] * input_dim
        for d in trans_dims:
            g[d] = _round_sig(1.0 / max(float(std[d]), 1e-12), 1)
        return g

    # Isotropic alternative: ONE factor per block, from the norm median. Per-dim
    # 1/std is what the directive prescribes and is reported as `input_gain`,
    # but it rescales x/y/z unequally, which distorts the DIRECTION of the
    # translation the cell reads. This variant preserves direction; it is
    # offered so the choice can be made on evidence rather than by default.
    iso = [1.0] * input_dim
    med_absT = float(np.median(absT)) if absT.size else 1.0
    for d in (0, 1, 2):
        iso[d] = _round_sig(1.0 / max(med_absT, 1e-12), 1)
    if has_delta:
        med_dt = float(np.median(dt[dt_ok])) if dt_ok.any() else 1.0
        for d in (7, 8, 9):
            iso[d] = _round_sig(1.0 / max(med_dt, 1e-12), 1)

    absT_graded_mean = float(absT[graded].mean()) if graded.any() else float("nan")
    dt_p50 = pct(dt[dt_ok], 50) if has_delta else float("nan")
    dt_p95 = pct(dt[dt_ok], 95) if has_delta else float("nan")

    hs, validation = build_verdict(absT_graded_mean, dt_p50, dt_p95, has_delta)

    return dict(
        meta=dict(**meta, graded_start=int(graded_start), n_rows=int(vals.shape[0]),
                  n_sequences=int(seq_ids.size), dim_labels=labels),
        per_dim=dict(
            std_all=[float(x) for x in std_all],
            std_graded=[float(x) for x in std_graded],
            mean_all=[float(x) for x in vals.mean(axis=0)],
            absmean_all=[float(x) for x in np.abs(vals).mean(axis=0)],
        ),
        absT_norm=dict(
            mean_graded=absT_graded_mean,
            p50_graded=pct(absT[graded], 50), p95_graded=pct(absT[graded], 95),
            mean_all=float(absT.mean()), p50_all=pct(absT, 50),
            min_all=float(absT.min()), max_all=float(absT.max()),
        ),
        delta_t_norm=(dict(
            p50=dt_p50, p95=dt_p95,
            p5=pct(dt[dt_ok], 5), p25=pct(dt[dt_ok], 25),
            p50_graded=pct(dt[dt_ok & graded], 50), p95_graded=pct(dt[dt_ok & graded], 95),
            mean=float(dt[dt_ok].mean()), max=float(dt[dt_ok].max()),
            # allow_repeat=True lets a scene shorter than num_views repeat
            # frames, and a repeated frame has delta_t == 0 exactly. A large
            # zero mass would pull the MEDIAN down without changing the scale
            # of real motion at all -- which is one of the few things that
            # could reconcile a p50 disagreement with the round-2 audit. Make
            # it visible instead of guessing.
            frac_near_zero=float((dt[dt_ok] < 1e-5).mean()),
        ) if has_delta else None),
        # The headline of the whole proposal: how far below the nuisance pose
        # columns the informative velocity channel sits.
        absT_over_delta_t_median=(float(np.median(absT) / max(np.median(dt[dt_ok]), 1e-12))
                                  if has_delta else None),
        per_sequence_scale=dict(
            p5=float(np.percentile(seq_factor, 5)), p50=float(np.percentile(seq_factor, 50)),
            p95=float(np.percentile(seq_factor, 95)), p99=float(np.percentile(seq_factor, 99)),
            spread_p99_over_p5=float(np.percentile(seq_factor, 99)
                                     / max(np.percentile(seq_factor, 5), 1e-12)),
        ),
        implied_head_scale=hs,
        input_gain=gain_from(std_all),
        input_gain_graded=gain_from(std_graded),
        input_gain_isotropic=iso,
        validation=validation,
        # Provenance for the revision: what P3.4 originally expected, and why
        # it no longer does. Kept in every JSON so the change stays auditable.
        superseded_expected=SUPERSEDED_EXPECTED,
        all_checks_passed=all(c["passed"] for c in validation),
        wiring_checks_passed=all(
            c["passed"] for c in validation if c["kind"] == "wiring"
        ),
        scale_checks_passed=all(
            c["passed"] for c in validation if c["kind"] == "regression"
        ),
    )


def main():
    guard_falsifier_env()
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    assert torch.cuda.is_available(), "GPU required"
    print(f"device: {torch.cuda.get_device_name(0)} | ckpt: {ckpt_path}", flush=True)

    model = load_model(ckpt_path, device=DEVICE, verbose=True).eval()
    model.views_per_step = 1
    model.feed_prev_pred = True
    gru = getattr(model, "pose_gru", None)
    assert gru is not None, "checkpoint has no pose_gru — nothing to probe"
    print(
        f"pose_gru: mode={gru.mode}, input={gru.input_mode}, input_dim={gru.input_dim}, "
        f"hidden={gru.hidden_dim}, img_feat={gru.img_feat}, iters={gru.iters}, "
        f"input_gain={'off' if gru.input_gain is None else gru.input_gain.tolist()}",
        flush=True,
    )
    if gru.input_gain is not None:
        print(
            "WARNING: this checkpoint ALREADY carries an input_gain buffer, so the "
            "measured distribution is the POST-gain one. To fill a fresh gain vector, "
            "probe a checkpoint without it.",
            flush=True,
        )

    # Capture gru_in exactly as the cell receives it. A forward-PRE hook sees
    # the positional args of PoseGRU.forward(gru_in, hidden, img_feat=...), so
    # args[0] is the pose(+delta) block BEFORE the img_feat concat and before
    # the P3.4 gain -- which is the distribution the gain has to be built from.
    captured = []
    gru.register_forward_pre_hook(
        lambda _m, inp: captured.append(inp[0].detach().float().cpu())
    )

    ds = DL3DV_Multi(
        split="test", ROOT=DATA_ROOT, resolution=[RES], transform=ImgNorm,
        num_views=NV, n_corres=0, aug_crop=AUG_CROP, allow_repeat=True,
        force_consecutive_frame_sampling=True, seed=SEED,
    )
    n_seq = N_BATCHES * BS
    # Evenly spread over the split rather than the first n_seq scenes: scene
    # order in the split tree is lab-correlated, and the per-sequence scale
    # factor is exactly the quantity that would be biased by taking one lab.
    idxs = np.unique(np.linspace(0, len(ds) - 1, num=n_seq).astype(int))
    # ...and PERMUTE the traversal order. The spread above only makes the FULL
    # sample unbiased; walking it in ascending index order makes every partial
    # sample (the incremental JSON, and anything a wall-clock kill leaves
    # behind) a contiguous slice of the lab-ordered split. That is not a
    # cosmetic difference: the per-sequence |t| factor spans ~8x across
    # sequences, so a lab-contiguous prefix can read a scale well off the
    # split-wide one. Fixed seed, so the run stays reproducible.
    idxs = np.random.default_rng(SEED).permutation(idxs).tolist()
    dl = DataLoader(
        _SeqSet(ds, idxs), batch_size=BS, shuffle=False, num_workers=WORKERS,
        collate_fn=_collate, pin_memory=False,
        persistent_workers=bool(WORKERS),
    )
    print(
        f"data: {len(idxs)} scenes / {len(dl)} batches x {BS} x {NV} views "
        f"(workers={WORKERS}) | supervised window = views "
        f"{max(1, NV - GRADED_VIEWS)}..{NV - 1}",
        flush=True,
    )

    meta = dict(ckpt=str(ckpt_path), data_root=DATA_ROOT, aug_crop=AUG_CROP,
                num_views=NV,
                batch_size=BS, batches_requested=N_BATCHES, resolution=list(RES),
                pose_gru_mode=gru.mode, input_mode=gru.input_mode,
                input_dim=int(gru.input_dim), iters=int(gru.iters),
                img_feat=gru.img_feat, img_feat_src=gru.img_feat_src)

    v_rows, view_rows, seq_rows = [], [], []
    seq_base = 0
    t0 = time.time()
    batches_done = 0
    for bi, batch in enumerate(dl):
        if time.time() - t0 > MAX_SECONDS:
            print(
                f"stopping at batch {bi}/{len(dl)} — PROBE_MAX_SECONDS ({MAX_SECONDS}s) "
                "reached; reporting on what was sampled",
                flush=True,
            )
            break
        batches_done = bi + 1
        batch = [to_device(v) for v in batch]
        b = batch[0]["img"].shape[0]
        for view_idx, gin in rollout_capture(model, batch, captured):
            v_rows.append(gin.numpy().astype(np.float64))
            view_rows.append(np.full(b, view_idx, dtype=np.int64))
            seq_rows.append(seq_base + np.arange(b, dtype=np.int64))
        seq_base += b
        if (bi + 1) % 25 == 0 or bi + 1 == len(dl):
            # Write incrementally: a wall-clock kill still leaves usable output,
            # and the running validation says early whether the probe is sane.
            stats = summarize(np.concatenate(v_rows), np.concatenate(view_rows),
                              np.concatenate(seq_rows), gru.input_mode,
                              int(gru.input_dim), NV,
                              dict(meta, batches_done=bi + 1))
            with open(OUT_JSON, "w") as f:
                json.dump(stats, f, indent=2)
            dn = stats["delta_t_norm"]
            print(
                f"[{bi + 1}/{len(dl)}] absT@sup={stats['absT_norm']['mean_graded']:.4f}"
                + (f" delta_t p50={dn['p50']:.5f} p95={dn['p95']:.5f}" if dn else "")
                + f" | {'ok' if stats['all_checks_passed'] else 'OUT OF BAND'}",
                flush=True,
            )

    stats = summarize(np.concatenate(v_rows), np.concatenate(view_rows),
                      np.concatenate(seq_rows), gru.input_mode,
                      int(gru.input_dim), NV,
                      dict(meta, batches_done=batches_done,
                           elapsed_seconds=round(time.time() - t0, 1)))
    with open(OUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    labels = stats["meta"]["dim_labels"]
    print("\n" + "=" * 92)
    print(f"{'dim':>12} | {'std(all)':>12} | {'std(sup)':>12} | {'mean':>12} | {'gain=1/std':>12}")
    print("-" * 92)
    for i, name in enumerate(labels):
        if name.startswith("imgfeat_"):
            continue
        print(f"{name:>12} | {stats['per_dim']['std_all'][i]:>12.6f} | "
              f"{stats['per_dim']['std_graded'][i]:>12.6f} | "
              f"{stats['per_dim']['mean_all'][i]:>12.6f} | "
              f"{stats['input_gain'][i]:>12g}")
    a, d = stats["absT_norm"], stats["delta_t_norm"]
    print("-" * 92)
    print(f"|absT|   mean@sup {a['mean_graded']:.4f}  p50@sup {a['p50_graded']:.4f}  "
          f"range {a['min_all']:.4f}..{a['max_all']:.4f}")
    if d:
        print(f"|delta_t| p50 {d['p50']:.5f}  p95 {d['p95']:.5f}  max {d['max']:.5f}")
        print(f"velocity channel sits {stats['absT_over_delta_t_median']:.1f}x below "
              "the pose columns (median norms)")
    s = stats["per_sequence_scale"]
    print(f"per-sequence |t| factor: p5 {s['p5']:.4f}  p50 {s['p50']:.4f}  "
          f"p95 {s['p95']:.4f}  p99 {s['p99']:.4f}  -> spread {s['spread_p99_over_p5']:.1f}x")
    print("  (a STATIC buffer can only normalize the middle of that spread — this is the "
          "number that prices the per-sequence running scale)")

    h = stats["implied_head_scale"]
    print(f"\nIMPLIED HEAD SCALE (measured head-units / GT-unit statistic): "
          f"delta_t p50 {h['from_dt_p50']:.2f}x, p95 {h['from_dt_p95']:.2f}x, "
          f"absT@sup {h['from_absT_sup']:.2f}x  |  directive assumes "
          f"{h['assumed_by_directive']:.1f}x")

    print("\nVALIDATION — WIRING (head-scale-invariant; this is the tier that "
          "can actually falsify the probe)")
    for c in stats["validation"]:
        if c["kind"] != "wiring":
            continue
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']:<34} "
              f"= {c['value']:.4f}   expected {c['expected']} "
              f"(band {c['lo']:.4f}..{c['hi']:.4f})")
    print("\nVALIDATION — SCALE (regression vs the 2026-08-12 train-split "
          "measurement; circular for that run, drift-detecting for later ones)")
    for c in stats["validation"]:
        if c["kind"] != "regression":
            continue
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']:<34} "
              f"= {c['value']:.5f}   expected {c['expected']} "
              f"(band {c['lo']:.5f}..{c['hi']:.5f})")
    if stats["wiring_checks_passed"] and not stats["scale_checks_passed"]:
        print("  -> Wiring holds but the scale moved: the probe is fine and something "
              "downstream of it changed (checkpoint, forward path, loader).\n"
              "     Re-derive the gain; do NOT widen this band to make it pass.")
    s0 = stats["superseded_expected"]
    print(f"\n  (superseded P3.4 targets, kept in the JSON: absT "
          f"{s0['absT_sup_range']}, delta_t p50 {s0['delta_t_p50']:g} / p95 "
          f"{s0['delta_t_p95']:g} — {s0['source']})")
    print(f"\nwrote {OUT_JSON}")

    if not stats["all_checks_passed"]:
        print("\nPROBE FAILED ITS WIRING CHECKS — do NOT fill pose_gru_input_gain "
              "from this run.", flush=True)
        return 1
    print("\nSuggested config (per-dim 1/std, quaternion dims 1.0):")
    print(f"  pose_gru_input_gain: {stats['input_gain']}")
    print(f"  # direction-preserving alternative: {stats['input_gain_isotropic']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
