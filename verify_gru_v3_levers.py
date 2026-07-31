#!/usr/bin/env python
"""
verify_gru_v3_levers.py — GPU verification of the captain_gru_v3 levers:
F (image features into the PoseGRU: pooled pre-ray feats of views t and t-1)
and R (N back-to-back GRUCell iterations per view, RAFT-style), on the REAL
pretrained checkpoint and REAL wrist_test data.

Stages:
  PAR   flag-off byte-identity ACROSS WORKTREES: rollouts produced by the
        pre-change captain_gru_v2 code vs this worktree with default flags
        (plain prev_pred + A4 GRU) must match bitwise
  UNIT  PoseGRU v3 ctor: widths, zero-init img_proj, ctor asserts,
        forward init-equivalence with features attached
  INIT  full model: F1xR8 at init == plain prev_pred (residual zero-head
        identity survives both levers); iterate translations bit-stable
  ITER  R lever: N calls/view, running-estimate re-feed, static delta,
        gru_pose_iters (N,B,7) with final == gru_pose, FORCE_ITERS=1 ==
        an iters=1 module bitwise
  FEAT  F lever: recorded features == independently recomputed pre-ray means
        (current + previous view, view-0 masked-token subtraction);
        IMG_FEAT_ZERO / IMG_FEAT_SHUFFLE falsifiers wired
  LOSS  PoseGRULoss iter_gamma: value-gating bit-identity, hand-checked
        gamma weighting, gradient isolation
  LOAD  save -> load_model round-trip restores img_feat/frames/D/iters;
        legacy v2 sniff unchanged; size-mismatched pose_gru loads HARD-FAIL

Submit via:  sbatch verify_gru_v3_levers.sbatch   (1x L40S)

Subprocess mode (used internally by stage PAR):
  python verify_gru_v3_levers.py --dump-rollout OUT.pt --worktree WT --arm {plain,a4}
"""
import argparse
import os
import subprocess
import sys

WORKTREE = "/scratch/bdursun25/cuteanything/captain_gru_v3"
V2_WORKTREE = "/scratch/bdursun25/cuteanything/captain_gru_v2"
DATA_ROOT = (
    "/frozen/avg/bora_data/droid_datasets/training_data/"
    "pointworld_droid_wrist_test/dl3dv_multi"
)
CKPT = "/scratch/bdursun25/cuteanything/my-da3/src/CUT3R/src/cut3r_512_dpt_4_64.pth"
MODEL_STR = (
    "ARCroco3DStereo(ARCroco3DStereoConfig(freeze='encoder', state_size=768, "
    "state_pe='2d', pos_embed='RoPE100', rgb_head=True, pose_head=True, "
    "patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', "
    "output_mode='pts3d+pose', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), "
    "pose_mode=('exp', -inf, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, "
    "dec_embed_dim=768, dec_depth=12, dec_num_heads=12, landscape_only=False))"
)
NV = 8
RES = (320, 192)
IDXS = [5, 100]
SEED = 777
GRU_SEED = 123
DEVICE = "cuda"


def setup_paths(worktree):
    for p in [
        worktree,
        os.path.join(worktree, "src"),
        os.path.join(worktree, "src/CUT3R"),
        os.path.join(worktree, "src/CUT3R/src"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)


def build_batch():
    from dust3r.datasets.dl3dv import DL3DV_Multi
    from dust3r.datasets.utils.transforms import ImgNorm
    from torch.utils.data._utils.collate import default_collate
    import torch

    ds = DL3DV_Multi(
        split="test", ROOT=DATA_ROOT, resolution=[RES], transform=ImgNorm,
        num_views=NV, n_corres=0, aug_crop=0, allow_repeat=True,
        force_consecutive_frame_sampling=True, seed=SEED,
    )
    view_lists = [ds[i] for i in IDXS]

    def to_device(view):
        return {
            k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in view.items()
        }

    return [to_device(default_collate([vl[v] for vl in view_lists])) for v in range(NV)]


def build_model():
    import torch
    from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig  # noqa: F401

    inf = float("inf")  # noqa: F841  used by MODEL_STR
    model = eval(MODEL_STR)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    del ckpt
    model = model.to(DEVICE).eval()
    model.views_per_step = 1
    model.feed_prev_pred = True
    return model


def rollout(model, batch):
    """Plain sequential no-grad rollout (G flags are forward-invariant)."""
    import torch

    ress = []
    with torch.no_grad():
        (feat, pos, shape), (isf, imem, sf, sp, mem) = model._forward_encoder(batch)
        for v in range(len(batch)):
            res_group, (sf, mem) = model._forward_decoder_group_step(
                views=batch, view_indices=[v],
                feat_group=[feat[v]], pos_group=[pos[v]], shape_group=[shape[v]],
                init_state_feat=isf, init_mem=imem,
                state_feat=sf, state_pos=sp, mem=mem,
            )
            ress.append({k: t.detach().clone() for k, t in res_group[0].items()})
    return ress


def dump_rollout(out_path, worktree, arm):
    """Subprocess entrypoint: rollout under `worktree`'s code, save to out_path."""
    setup_paths(worktree)
    import torch
    import dust3r.heads  # noqa: F401  circular-import order

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    model = build_model()
    batch = build_batch()
    if arm == "a4":
        torch.manual_seed(GRU_SEED)  # identical fresh GRU init across worktrees
        model.enable_pose_gru(hidden_dim=128, mode="residual", input_mode="pose_delta")
        model.pose_gru.to(DEVICE)
    ress = rollout(model, batch)
    torch.save(
        {
            "arm": arm,
            "worktree": worktree,
            "res": [{k: v.cpu() for k, v in r.items()} for r in ress],
        },
        out_path,
    )
    print(f"dumped {arm} rollout from {worktree} -> {out_path}")


# =============================================================================
# main verification
# =============================================================================


def main():
    setup_paths(WORKTREE)
    import numpy as np
    import torch
    import dust3r.heads  # noqa: F401  MUST precede dust3r.utils.camera
    from dust3r.model import PoseGRU, load_model
    from dust3r.losses import PoseGRULoss
    from dust3r.utils.camera import pose_encoding_to_camera  # noqa: F401

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    np.set_printoptions(precision=5, suppress=True, linewidth=200)

    CHECKS = []

    def banner(t):
        print("\n" + "=" * 96 + f"\n {t}\n" + "=" * 96, flush=True)

    def check(gid, desc, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        print(
            f"  [{tag}] {gid}: {desc}" + (f"\n         -> {detail}" if detail else ""),
            flush=True,
        )
        CHECKS.append((gid, desc, bool(ok), detail))

    def maxdiff(a, b):
        return float((a.double() - b.double()).abs().max())

    scratch = os.path.join(WORKTREE, "logs")
    os.makedirs(scratch, exist_ok=True)

    # =========================================================================
    banner("STAGE PAR — flag-off byte-identity across worktrees (v2 code vs v3 code)")
    # =========================================================================
    dumps = {}
    for wt_name, wt in (("v2", V2_WORKTREE), ("v3", WORKTREE)):
        for arm in ("plain", "a4"):
            out = os.path.join(scratch, f"parity_{wt_name}_{arm}.pt")
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            r = subprocess.run(
                [sys.executable, os.path.abspath(__file__),
                 "--dump-rollout", out, "--worktree", wt, "--arm", arm],
                env=env, capture_output=True, text=True,
            )
            if r.returncode != 0:
                check("P0", f"dump {wt_name}/{arm} runs", False, r.stderr[-500:])
                continue
            dumps[(wt_name, arm)] = torch.load(out, map_location="cpu", weights_only=False)

    if len(dumps) == 4:
        for arm, gid in (("plain", "P1"), ("a4", "P2")):
            r2, r3 = dumps[("v2", arm)]["res"], dumps[("v3", arm)]["res"]
            keys = sorted(set(r2[0].keys()) & set(r3[0].keys()))
            same_keys = all(set(a.keys()) == set(b.keys()) for a, b in zip(r2, r3))
            worst = max(
                maxdiff(a[k], b[k]) for a, b in zip(r2, r3) for k in keys
            )
            bitwise = same_keys and all(
                torch.equal(a[k], b[k]) for a, b in zip(r2, r3) for k in keys
            )
            check(
                gid,
                f"{arm} rollout: v2-code vs v3-code flag-off BYTE-IDENTICAL",
                bitwise,
                f"res keys same={same_keys}, worst maxdiff {worst:.2e}",
            )
    else:
        check("P1", "parity dumps all produced", False, f"got {sorted(dumps)}")

    # =========================================================================
    banner("STAGE UNIT — PoseGRU v3 ctor and forward")
    # =========================================================================
    g_f1 = PoseGRU(
        input_mode="pose_delta", img_feat="input", img_feat_dim=32, img_feat_frames=2
    )
    check(
        "U1",
        "F1 widths: cell input 14+32=46, img_proj (32, 2048), zero-init projector",
        g_f1.cell.weight_ih.shape[1] == 46
        and tuple(g_f1.img_proj.weight.shape) == (32, 2048)
        and float(g_f1.img_proj.weight.abs().sum()) == 0.0
        and float(g_f1.img_proj.bias.abs().sum()) == 0.0
        and g_f1.iters == 1,
    )

    g_r8 = PoseGRU(input_mode="pose_delta", iters=8)
    check(
        "U2",
        "R8 without features: width stays 14, iters stored, no img modules",
        g_r8.cell.weight_ih.shape[1] == 14
        and g_r8.iters == 8
        and not hasattr(g_r8, "img_proj"),
    )

    raised = {}
    for name, kw in (
        ("bad_img_feat", dict(img_feat="bogus")),
        ("bad_iters", dict(iters=0)),
        ("bad_frames", dict(img_feat="input", img_feat_frames=3)),
    ):
        try:
            PoseGRU(**kw)
            raised[name] = False
        except AssertionError:
            raised[name] = True
    check("U3", "ctor asserts on bad img_feat / iters / frames", all(raised.values()),
          str(raised))

    torch.manual_seed(7)
    pose = torch.randn(4, 7)
    delta = torch.randn(4, 7)
    feats = torch.randn(4, 2048)
    exp = torch.cat([pose[:, :3], torch.nn.functional.normalize(pose[:, 3:7], dim=-1)], -1)
    out, _ = g_f1(torch.cat([pose, delta], -1), None, img_feat=feats)
    check(
        "U4",
        "zero-init residual identity holds WITH features attached (zero img_proj)",
        maxdiff(out, exp) < 1e-6,
        f"maxdiff {maxdiff(out, exp):.1e}",
    )

    # =========================================================================
    banner("STAGE INIT/ITER/FEAT — full model on real data")
    # =========================================================================
    assert torch.cuda.is_available(), "this verification requires a GPU"
    print(f"  device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}",
          flush=True)
    model = build_model()
    batch = build_batch()

    base = rollout(model, batch)  # plain prev_pred reference (no GRU)
    base2 = rollout(model, batch)
    noise = max(maxdiff(a["camera_pose"], b["camera_pose"]) for a, b in zip(base, base2))
    check("I0", "determinism floor (plain prev_pred twice)", noise == 0.0,
          f"floor {noise:.1e}")

    torch.manual_seed(GRU_SEED)
    model.enable_pose_gru(
        hidden_dim=128, mode="residual", input_mode="pose_delta",
        img_feat="input", img_feat_dim=32, img_feat_frames=2, iters=8,
    )
    model.pose_gru.to(DEVICE)
    f1r8_init = rollout(model, batch)
    dp = max(maxdiff(a["camera_pose"], b["camera_pose"]) for a, b in zip(f1r8_init, base))
    dx = max(
        maxdiff(a["pts3d_in_self_view"], b["pts3d_in_self_view"])
        for a, b in zip(f1r8_init, base)
    )
    check(
        "I1",
        "F1xR8 at init == plain prev_pred (residual zero-head identity, fp tolerance)",
        dp < 1e-3 and dx < 1e-2,
        f"pose {dp:.1e}, pts {dx:.1e}",
    )
    t_bit = all(
        torch.equal(f1r8_init[v]["gru_pose"][:, :3], f1r8_init[v - 1]["camera_pose"][:, :3])
        for v in range(1, NV)
    )
    check("I2", "init iterate translations bit-stable (t chain exact through 8 iterates)",
          t_bit)

    # ---- nudge head + projector so both channels are live for ITER/FEAT
    torch.manual_seed(1234)
    with torch.no_grad():
        model.pose_gru.head.weight.normal_(0, 1e-3)
        model.pose_gru.img_proj.weight.normal_(0, 1e-3)
        model.pose_gru.img_proj.bias.normal_(0, 1e-3)

    class Spy:
        def __init__(self, m):
            self.m, self.calls, self.orig = m, [], m.pose_gru.forward

        def __enter__(self):
            def spy(gru_in, hidden, img_feat=None):
                rec = dict(
                    gru_in=gru_in.detach().clone(),
                    img_feat=None if img_feat is None else img_feat.detach().clone(),
                )
                out, h = self.orig(gru_in, hidden, img_feat=img_feat)
                rec["out"] = out.detach().clone()
                self.calls.append(rec)
                return out, h

            self.m.pose_gru.forward = spy
            return self

        def __exit__(self, *a):
            self.m.pose_gru.forward = self.orig

    with Spy(model) as spy:
        f1r8 = rollout(model, batch)

    check("T1", f"R8: exactly {(NV - 1) * 8} GRU calls ({NV - 1} views x 8 iterates)",
          len(spy.calls) == (NV - 1) * 8, f"got {len(spy.calls)}")

    ok_est, ok_delta_static, ok_feat_static = True, True, True
    for v in range(1, NV):
        c0 = spy.calls[(v - 1) * 8]
        for k in range(1, 8):
            ck = spy.calls[(v - 1) * 8 + k]
            if not torch.equal(ck["gru_in"][:, :7], spy.calls[(v - 1) * 8 + k - 1]["out"]):
                ok_est = False
            if not torch.equal(ck["gru_in"][:, 7:14], c0["gru_in"][:, 7:14]):
                ok_delta_static = False
            if not torch.equal(ck["img_feat"], c0["img_feat"]):
                ok_feat_static = False
    check("T2", "iterate k>0 pose slice == previous iterate's output (running estimate)",
          ok_est)
    check("T3", "delta slice bit-constant across a view's iterates (static context)",
          ok_delta_static)
    check("T4", "image features bit-constant across a view's iterates (static context)",
          ok_feat_static)

    ok_iters_key = all(
        ("gru_pose_iters" in f1r8[v]) == (v >= 1)
        and (v == 0 or tuple(f1r8[v]["gru_pose_iters"].shape) == (8, 2, 7))
        for v in range(NV)
    )
    ok_final = all(
        torch.equal(f1r8[v]["gru_pose_iters"][-1], f1r8[v]["gru_pose"])
        for v in range(1, NV)
    )
    moved = max(
        maxdiff(f1r8[v]["gru_pose_iters"][-1], f1r8[v]["gru_pose_iters"][0])
        for v in range(1, NV)
    )
    check("T5", "gru_pose_iters: (8,B,7) on views>=1 only, final row == gru_pose",
          ok_iters_key and ok_final)
    check("T6", "iterates actually move with a nudged head (refinement is live)",
          moved > 0.0, f"max |iter7 - iter0| = {moved:.2e}")

    os.environ["POSE_GRU_FORCE_ITERS"] = "1"
    forced1 = rollout(model, batch)
    os.environ.pop("POSE_GRU_FORCE_ITERS")
    sd = {k: v.clone() for k, v in model.pose_gru.state_dict().items()}
    torch.manual_seed(GRU_SEED)
    model.enable_pose_gru(
        hidden_dim=128, mode="residual", input_mode="pose_delta",
        img_feat="input", img_feat_dim=32, img_feat_frames=2, iters=1,
    )
    model.pose_gru.to(DEVICE)
    model.pose_gru.load_state_dict(sd)
    n1 = rollout(model, batch)
    eq = all(
        torch.equal(a[k], b[k])
        for a, b in zip(forced1, n1)
        for k in ("camera_pose", "gru_pose", "pts3d_in_self_view")
        if k in a and k in b
    )
    no_iters_key = all("gru_pose_iters" not in r for r in n1)
    check("T7", "FORCE_ITERS=1 on the R8 module == an iters=1 module bitwise "
          "(anytime-N eval is exact)", eq)
    check("T8", "iters=1: gru_pose_iters key absent (flag-off res dict unchanged)",
          no_iters_key)

    # ---- FEAT checks (iters=1 module still loaded, features nudged-live)
    with Spy(model) as fspy:
        f1 = rollout(model, batch)
    with torch.no_grad():
        (feat, _, _), _ = model._forward_encoder(batch)
        ref = [None] * NV
        ref[0] = (
            (feat[0] - model.masked_ray_map_token.to(feat[0].dtype)).mean(dim=1).float()
        )
        for v in range(1, NV):
            ref[v] = feat[v].mean(dim=1).float()
    ok_cur = all(
        torch.equal(fspy.calls[v - 1]["img_feat"][:, :1024], ref[v]) for v in range(1, NV)
    )
    ok_prev = all(
        torch.equal(fspy.calls[v - 1]["img_feat"][:, 1024:], ref[v - 1])
        for v in range(1, NV)
    )
    check(
        "F1",
        "recorded features == independently recomputed PRE-ray means: "
        "current view first half, previous view second half (view-0 minus masked token)",
        ok_cur and ok_prev,
    )

    os.environ["POSE_GRU_IMG_FEAT_ZERO"] = "1"
    fz = rollout(model, batch)
    os.environ.pop("POSE_GRU_IMG_FEAT_ZERO")
    d0 = maxdiff(fz[0]["camera_pose"], f1[0]["camera_pose"])
    dl = max(maxdiff(a["camera_pose"], b["camera_pose"]) for a, b in zip(fz[1:], f1[1:]))
    check("F2", "IMG_FEAT_ZERO: view 0 untouched, later views perturbed "
          "(feature channel is live)", d0 == 0.0 and dl > 0.0,
          f"view0 {d0:.1e}, later {dl:.2e}")

    os.environ["POSE_GRU_IMG_FEAT_SHUFFLE"] = "1"
    with Spy(model) as sspy:
        rollout(model, batch)
    os.environ.pop("POSE_GRU_IMG_FEAT_SHUFFLE")
    ok_roll = all(
        torch.equal(
            sspy.calls[v - 1]["img_feat"],
            torch.roll(torch.cat([ref[v], ref[v - 1]], dim=-1), 1, 0),
        )
        for v in range(1, NV)
    )
    check("F3", "IMG_FEAT_SHUFFLE: recorded pair == coherent batch-roll of the honest "
          "pair (stash itself stays honest)", ok_roll)

    # =========================================================================
    banner("STAGE LOSS — PoseGRULoss iter_gamma")
    # =========================================================================
    from dust3r.utils.camera import pose_encoding_to_camera as p2c

    torch.manual_seed(5)
    B, NVL, NIT = 2, 4, 3

    def rand_enc():
        e = torch.randn(B, 7)
        e[:, 3:] = torch.nn.functional.normalize(e[:, 3:], dim=-1)
        return e

    gts = [
        dict(camera_pose=p2c(rand_enc()), is_metric=torch.zeros(B, dtype=torch.bool))
        for _ in range(NVL)
    ]
    preds = [dict(camera_pose=rand_enc()) for _ in range(NVL)]
    for i in range(1, NVL):
        iters = torch.stack([rand_enc() for _ in range(NIT)], dim=0).requires_grad_(True)
        preds[i]["gru_pose_iters"] = iters
        preds[i]["gru_pose"] = iters[-1]

    l_plain, d_plain = PoseGRULoss().compute_loss(gts, preds)
    l_g0, d_g0 = PoseGRULoss(iter_gamma=0.0).compute_loss(gts, preds)
    preds_no_iters = [
        {k: v for k, v in p.items() if k != "gru_pose_iters"} for p in preds
    ]
    l_bare, _ = PoseGRULoss().compute_loss(gts, preds_no_iters)
    check("S1", "iter_gamma=0.0 is value-gated: loss bit-identical with and without "
          "the iterates key", torch.equal(l_plain, l_g0) and torch.equal(l_plain, l_bare),
          f"plain {float(l_plain):.6f}")

    gamma = 0.8
    l_g, d_g = PoseGRULoss(iter_gamma=gamma).compute_loss(gts, preds)
    expected_extra = 0.0
    for k in range(NIT - 1):
        pk = [dict(p) for p in preds]
        for i in range(1, NVL):
            pk[i] = dict(pk[i])
            pk[i]["gru_pose"] = preds[i]["gru_pose_iters"][k]
            pk[i].pop("gru_pose_iters", None)
        lk, _ = PoseGRULoss().compute_loss(gts, pk)
        expected_extra += (gamma ** (NIT - 1 - k)) * float(lk)
    got_extra = float(l_g) - float(l_plain)
    check("S2", "gamma weighting == hand-computed sum over intermediate iterates "
          "(final term unchanged at weight 1.0)",
          abs(got_extra - expected_extra) < 1e-5,
          f"got {got_extra:.6f}, expected {expected_extra:.6f}")
    check("S3", "details: gru_trans/quat final-only unchanged, iters term reported",
          d_g["gru_trans_loss"] == d_plain["gru_trans_loss"]
          and d_g["gru_quat_loss"] == d_plain["gru_quat_loss"]
          and "gru_pose_loss_iters" in d_g
          and "gru_pose_loss_iters" not in d_plain)

    l_g.backward()
    grads_ok = all(
        preds[i]["gru_pose_iters"].grad is not None
        and float(preds[i]["gru_pose_iters"].grad.abs().sum()) > 0
        for i in range(1, NVL)
    )
    check("S4", "sequence loss backprops into every iterate", grads_ok)

    # =========================================================================
    banner("STAGE LOAD — ckpt round-trip, legacy sniff, hard-fail")
    # =========================================================================
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=f"{WORKTREE}/src/CUT3R/config", version_base=None):
        v3_cfg = compose(config_name="captain_gru_v3_a4_g1_f1_r8")
        v2_cfg = compose(config_name="captain_gru_v2_a4_g1")

    mini = {
        "model": {f"pose_gru.{k}": v.cpu() for k, v in model.pose_gru.state_dict().items()},
        "args": v3_cfg,
    }
    mini_path = os.path.join(scratch, "tmp_v3_mini_ckpt.pth")
    torch.save(mini, mini_path)
    net = load_model(mini_path, device="cpu", verbose=True)
    g = net.pose_gru
    check(
        "L1",
        "load_model round-trip: img_feat=input, frames=2, D=32, iters=8, "
        "input=pose_delta all restored",
        g is not None and g.img_feat == "input" and g.img_feat_frames == 2
        and g.img_feat_dim == 32 and g.iters == 8 and g.input_mode == "pose_delta"
        and g.cell.weight_ih.shape[1] == 46,
    )
    check("L2", "trained F1 GRU weights actually loaded",
          g is not None
          and torch.equal(g.cell.weight_ih, mini["model"]["pose_gru.cell.weight_ih"])
          and torch.equal(g.img_proj.weight, mini["model"]["pose_gru.img_proj.weight"]))
    del net

    torch.manual_seed(11)
    legacy_gru = PoseGRU(input_mode="pose_delta")
    legacy = {
        "model": {f"pose_gru.{k}": v.cpu() for k, v in legacy_gru.state_dict().items()},
        "args": v2_cfg,
    }
    legacy_path = os.path.join(scratch, "tmp_v2_legacy_ckpt.pth")
    torch.save(legacy, legacy_path)
    net = load_model(legacy_path, device="cpu", verbose=True)
    g = net.pose_gru
    check("L3", "legacy v2 ckpt sniffs exactly as before: pose_delta, no features, iters=1",
          g is not None and g.img_feat == "none" and g.iters == 1
          and g.input_mode == "pose_delta" and g.cell.weight_ih.shape[1] == 14)
    del net

    inf = float("inf")  # noqa: F841
    from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig  # noqa: F401

    clash = eval(MODEL_STR)
    clash.enable_pose_gru(hidden_dim=128, mode="residual", input_mode="pose_delta")
    try:
        clash.load_state_dict(mini["model"], strict=False)
        hard_fail = False
    except RuntimeError as e:
        hard_fail = "pose_gru" in str(e)
    check("L4", "loading an F1 GRU into a plain 14-wide module HARD-FAILS "
          "(no silent random-GRU eval)", bool(hard_fail))

    os.remove(mini_path)
    os.remove(legacy_path)

    # =========================================================================
    banner("SUMMARY")
    # =========================================================================
    n_fail = sum(1 for _, _, ok, _ in CHECKS if not ok)
    for gid, desc, ok, _ in CHECKS:
        print(f"  {'PASS' if ok else 'FAIL':4s}  {gid:4s}  {desc}")
    print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-rollout", default=None)
    ap.add_argument("--worktree", default=WORKTREE)
    ap.add_argument("--arm", default="plain", choices=["plain", "a4"])
    a = ap.parse_args()
    if a.dump_rollout:
        dump_rollout(a.dump_rollout, a.worktree, a.arm)
    else:
        main()
