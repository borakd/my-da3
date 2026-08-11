import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional
import dust3r.utils.path_to_croco  # noqa: F401
import torch
import torch.nn as nn
from dust3r.blocks import (  # noqa
    Attention,
    Block,
    CrossAttention,
    CustomDecoderBlock,
    DecoderBlock,
    DropPath,
    Mlp,
)
from dust3r.heads import head_factory
from dust3r.patch_embed import get_patch_embed
# Relative import on purpose: demo scripts call add_ckpt_path with the checkpoint's
# own repo, which prepends THAT repo's src/ to sys.path — a plain
# `dust3r.utils.camera` would then resolve to the other tree, which may lack
# get_ray_map_torch. `.utils.camera` always binds to the tree this file lives in.
from .utils.camera import (
    camera_to_pose_encoding,
    get_ray_map_torch,
    matrix_to_quaternion,
    pose_encoding_to_camera,
    quaternion_to_matrix,
)
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    transpose_to_landscape,
)
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig
from transformers import PreTrainedModel  # noqa: F401  re-exported: train*.py do `from dust3r.model import PreTrainedModel`
from transformers.file_utils import ModelOutput

from models.croco import CrocoConfig, CroCoNet  # noqa

inf = float("inf")
from accelerate.logging import get_logger

printer = get_logger(__name__, log_level="DEBUG")


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """

    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None


def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def _sniff_pose_gru_config(state, train_args, enc_embed_dim, where="checkpoint"):
    """Recover the enable_pose_gru kwargs for a checkpoint from its state dict
    (keys + tensor shapes) and its saved training args. Pure function — no
    model build, no I/O — so the sniff is unit-testable on synthetic state
    dicts (verify_gru_encoder_levers.py) without instantiating the net.

    Weight shapes stay authoritative wherever they can speak (hidden width,
    F on/off, projector presence, source width, block count, cell input
    width); ckpt args fill in only what is INVISIBLE in shapes (mode, iters)
    or ambiguous (source disambiguation), and every overlap is cross-checked
    with a hard fail — a mis-sniffed module either refuses to load (shape
    mismatch) or, worse, silently evaluates the wrong arm.

    Returns None when the state dict carries no pose_gru weights.
    """
    if not any(k.startswith(("pose_gru.", "module.pose_gru.")) for k in state):
        return None

    def g(suffix):
        return state.get("pose_gru." + suffix, state.get("module.pose_gru." + suffix))

    # residual / direct / split_anchor all share ONE head of identical
    # shape, so the mode is invisible in the weights and ckpt args are
    # the only source. A missing key used to fall back to "residual",
    # which builds the WRONG algebra with matching shapes and zero errors
    # — i.e. a silently mislabelled evaluation. Hard-fail instead.
    mode = getattr(train_args, "pose_gru_mode", None)
    if mode is None:
        raise RuntimeError(
            f"{where}: checkpoint carries pose_gru weights but its "
            "ckpt['args'] has no 'pose_gru_mode'. The mode cannot be "
            "recovered from weight shapes — refusing to guess."
        )
    hidden_dim = getattr(train_args, "pose_gru_hidden_dim", 128)
    input_mode = getattr(train_args, "pose_gru_input", "pose")
    # R lever: the iteration count is INVISIBLE in weight shapes (shared-
    # weight iteration) — ckpt args are the only source, like mode.
    iters = int(getattr(train_args, "pose_gru_iters", 1))
    w_hh = g("cell.weight_hh")
    if w_hh is not None:
        hidden_dim = w_hh.shape[1]
    w_ih = g("cell.weight_ih")
    # F lever: the weights stay authoritative in BOTH projector modes.
    #   img_norm.weight exists iff img_feat == "input" (it is built for
    #     the raw-feature arm too), so it — not img_proj — is the switch.
    #   img_proj.weight exists iff img_feat_proj is ALSO True; it then
    #     gives the appended width (rows) and block count (cols/src).
    # Sniffing "F is on" off img_proj alone would rebuild an F-OFF module
    # for an img_feat_proj=False checkpoint.
    w_proj = g("img_proj.weight")
    w_norm = g("img_norm.weight")
    img_feat = "input" if w_norm is not None else "none"
    img_feat_proj = w_proj is not None
    img_feat_dim = 32  # appended width; recomputed below when F is on
    img_feat_frames = 2
    img_feat_src = "pooled"
    if img_feat == "input":
        from dust3r.img_encoders import CORR_FEAT_DIM, ENCODER_DIMS

        src_dim = int(w_norm.shape[0])
        # Source lever: the frozen encoders leave unmistakable key
        # fingerprints (their weights ARE serialized); corr and pooled add
        # no keys and are told apart by the img_norm width instead. The
        # saved args cross-check every branch — on a disagreement nobody
        # should guess which side is mislabelled.
        args_src = getattr(train_args, "pose_gru_img_feat_src", None)

        def enc_key(fragment):
            # Fingerprint by suffix inside the img_encoder namespace: the
            # frozen net nests under FrozenImageEncoder's own attribute
            # (img_encoder.net.conv1.weight / img_encoder.net.cls_token),
            # so exact-path lookups would silently miss it.
            return any(
                k.startswith(("pose_gru.img_encoder.", "module.pose_gru.img_encoder."))
                and k.endswith(fragment)
                for k in state
            )

        if enc_key("conv1.weight"):
            img_feat_src = "resnet18"
        elif enc_key("cls_token"):
            img_feat_src = "dinov2_vits14"
        elif src_dim == CORR_FEAT_DIM:
            img_feat_src = "corr"
        elif src_dim == int(enc_embed_dim):
            img_feat_src = "pooled"
        else:
            raise RuntimeError(
                f"{where}: pose_gru.img_norm width {src_dim} matches no "
                f"known source (pooled={int(enc_embed_dim)}, "
                f"corr={CORR_FEAT_DIM}) and no encoder weights are present "
                "— cannot identify img_feat_src"
            )
        if args_src is not None and str(args_src) != img_feat_src:
            raise RuntimeError(
                f"{where}: weights identify img_feat_src={img_feat_src!r} "
                f"but ckpt['args'].pose_gru_img_feat_src={args_src!r} — "
                "mislabelled checkpoint, refusing to guess which side is wrong"
            )
        if img_feat_src in ENCODER_DIMS and src_dim != ENCODER_DIMS[img_feat_src]:
            raise RuntimeError(
                f"{where}: {img_feat_src} encoder weights present but "
                f"img_norm width is {src_dim}, expected "
                f"{ENCODER_DIMS[img_feat_src]}"
            )
        if img_feat_proj:
            img_feat_dim = w_proj.shape[0]
            assert w_proj.shape[1] % src_dim == 0, (
                f"pose_gru.img_proj input width {w_proj.shape[1]} is not a "
                f"multiple of the source width {src_dim}"
            )
            blocks = w_proj.shape[1] // src_dim
        else:
            # No projector tensor to read the block count off. Solve it
            # from the cell input width instead: base(7|14) + blocks*
            # src_dim. Even the narrowest source width (corr, 54) dwarfs
            # the 7-wide base gap, so the split is unique — assert that
            # rather than assume it.
            assert w_ih is not None, (
                "pose_gru checkpoint has img_norm but no img_proj and no "
                "cell.weight_ih — the block count cannot be recovered"
            )
            cand = [
                b
                for b in (1, 2)
                for base in (7, 14)
                if int(w_ih.shape[1]) == base + b * src_dim
            ]
            assert len(cand) == 1, (
                f"pose_gru cell input width {int(w_ih.shape[1])} does not "
                f"decompose uniquely as base(7|14) + blocks(1|2)*{src_dim} "
                f"(candidates: {cand})"
            )
            blocks = cand[0]
            img_feat_dim = blocks * src_dim
        if img_feat_src == "corr":
            # corr appends exactly ONE pair-statistic block; its (semantic)
            # frame count is fixed at 2 — it consumes the previous view.
            assert blocks == 1, (
                f"{where}: img_feat_src='corr' appends one "
                f"{CORR_FEAT_DIM}-wide block, but the weights show "
                f"{blocks} blocks"
            )
            img_feat_frames = 2
        else:
            assert blocks in (1, 2), (
                f"{where}: {blocks} appended blocks — expected 1 (F0) or 2 (F1)"
            )
            img_feat_frames = blocks
    # The weights are authoritative for the input width: base (7 = pose,
    # 14 = pose+delta) plus the APPENDED F width; the saved config only
    # breaks ties. Hard-fail on anything unexplained — a mis-sniffed
    # width would otherwise rebuild the wrong cell and (before the
    # load_state_dict hardening) silently evaluate a random GRU.
    if w_ih is not None:
        base_width = w_ih.shape[1] - (img_feat_dim if img_feat == "input" else 0)
        assert base_width in (7, 14), (
            f"pose_gru cell input width {w_ih.shape[1]} minus appended F width "
            f"{img_feat_dim if img_feat == 'input' else 0} = {base_width}; "
            f"expected 7 (pose) or 14 (pose_delta)"
        )
        input_mode = "pose_delta" if base_width == 14 else "pose"
    return dict(
        hidden_dim=hidden_dim,
        mode=mode,
        input_mode=input_mode,
        img_feat=img_feat,
        img_feat_dim=img_feat_dim,
        img_feat_frames=img_feat_frames,
        img_feat_proj=img_feat_proj,
        img_feat_src=img_feat_src,
        iters=iters,
    )


def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    # ckpt = torch.load(model_path, map_location="cpu")
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")

    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )  # ManyAR only for aspect ratio not consistent
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace("landscape_only=True", "landscape_only=False")
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    # The PoseGRU refiner is enabled by the trainer AFTER construction (it is
    # not part of the args.model string), so checkpoints that carry pose_gru
    # weights need the module materialized before load_state_dict (which
    # otherwise refuses them). mode/hidden_dim come from the training config
    # saved inside the checkpoint — the authoritative source (mode is not
    # inferable from the weights).
    gru_cfg = _sniff_pose_gru_config(
        ckpt["model"], ckpt["args"], net.enc_embed_dim, where=str(model_path)
    )
    has_pose_gru = gru_cfg is not None
    if has_pose_gru:
        # img_encoder_pretrained=False: for the encoder sources the ckpt's
        # own serialized img_encoder weights overwrite the random init in
        # load_state_dict below — eval never needs the pretrained file.
        net.enable_pose_gru(img_encoder_pretrained=False, **gru_cfg)
        if verbose:
            print(
                "... pose_gru enabled from ckpt: "
                + ", ".join(f"{k}={v}" for k, v in gru_cfg.items())
            )
    # Training-time RUN-MODE flags that live outside the weights (unlike the
    # pose_gru module config above, which IS restored). They are deliberately
    # NOT auto-enabled: conditioning needs caller-built inputs (ray maps,
    # intrinsics, and — for the oracle — real GT camera_pose in the views), so
    # flipping them here would break plain image-only callers. Instead they are
    # stashed on the net so every inference surface can see what the checkpoint
    # was trained as and warn instead of silently evaluating out of
    # distribution. demo_ray.py / the eval workers read these.
    run_args = ckpt.get("args")
    if bool(getattr(run_args, "feed_gt_ray_map", False)):
        net.trained_conditioning = "gt"
    elif bool(getattr(run_args, "feed_prev_gt_ray_map", False)):
        net.trained_conditioning = "prev_gt"
    elif bool(getattr(run_args, "feed_prev_pred", False)):
        net.trained_conditioning = "prev_pred_gru" if has_pose_gru else "prev_pred"
    else:
        net.trained_conditioning = "none"
    net.trained_pose_gru_oracle = str(getattr(run_args, "pose_gru_oracle", "off") or "off")
    if verbose and net.trained_conditioning != "none":
        print(
            f"NOTE: this checkpoint was trained with ray-map conditioning "
            f"('{net.trained_conditioning}'). Nothing is auto-enabled here — "
            f"run it through demo_ray.py / infer_and_eval_worker_ray.py with "
            f"--conditioning {net.trained_conditioning}, or an image-only run "
            f"is an out-of-distribution open-loop probe."
        )
    if verbose and net.trained_pose_gru_oracle != "off":
        print(
            "*** NOTE: this checkpoint was trained with pose_gru_oracle="
            f"'{net.trained_pose_gru_oracle}' — a GT-injection DIAGNOSTIC arm, "
            "not an honest one. The oracle is OFF at inference unless the "
            "caller enables it (model.pose_gru_oracle='gt', views carrying "
            "real GT camera_pose); either way, do not compare its metrics "
            "against honest arms. ***"
        )
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class ARCroco3DStereoConfig(PretrainedConfig):
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",  # or dpt
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        ray_enc_depth=2,
        state_size=324,
        local_mem_size=256,
        state_pe="2d",
        state_dec_num_heads=16,
        depth_head=False,
        rgb_head=False,
        pose_conf_head=False,
        pose_head=False,
        **croco_kwargs,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.ray_enc_depth = ray_enc_depth
        self.state_size = state_size
        self.state_pe = state_pe
        self.state_dec_num_heads = state_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.rgb_head = rgb_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.croco_kwargs = croco_kwargs


class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(torch.randn(1, 1, v_dim) * 0.2, requires_grad=True)
        self.mem = nn.Parameter(torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True)
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C]
        feat_v: [B, 1, C]
        """
        feat_k = self.proj_q(feat_k)  # [B, 1, C]
        feat = torch.cat([feat_k, feat_v], dim=-1)
        for blk in self.write_blocks:
            mem, _ = blk(mem, feat, None, None)
        return mem

    def inquire(self, query, mem):
        x = self.proj_q(query)  # [B, 1, C]
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1)
        for blk in self.read_blocks:
            x, _ = blk(x, mem, None, None)
        return x[..., -self.v_dim :]


def pose_delta_encoding(prev_enc, cur_enc):
    """Relative-motion (velocity) encoding between two absT_quaR poses.

    Returns the 7-d encoding of inv(P_prev) @ P_cur — the frame-to-frame
    transform expressed in P_prev's frame. Computed as a relative rigid
    transform (NOT a componentwise encoding difference): quaternion
    subtraction is meaningless under the double cover, whereas the relative
    transform is frame-consistent and sign-stable. When prev_enc is None
    (first step with no earlier pose) returns the identity-motion encoding
    [0,0,0, 1,0,0,0]. Inputs are detached stashes — no gradient flows here.
    """
    if prev_enc is None:
        out = cur_enc.new_zeros(cur_enc.shape[0], 7)
        out[:, 3] = 1.0  # identity quaternion, real part first
        return out
    prev_enc = prev_enc.float()
    cur_enc = cur_enc.float()
    # quaternion_to_matrix normalizes internally, so raw head quats are fine.
    R1 = quaternion_to_matrix(prev_enc[:, 3:7])
    R2 = quaternion_to_matrix(cur_enc[:, 3:7])
    R1t = R1.transpose(1, 2)
    rel_R = R1t @ R2
    rel_t = (R1t @ (cur_enc[:, :3] - prev_enc[:, :3]).unsqueeze(-1)).squeeze(-1)
    return torch.cat([rel_t, matrix_to_quaternion(rel_R)], dim=-1)


def gt_pose_encoding(views, idx, ref_idx=0):
    """View-ref-relative GT pose encoding for view `idx` — the ORACLE target.

    Built byte-for-byte the way PoseGRULoss builds its supervision target
    (losses.py: camera_to_pose_encoding(inv(cam1) @ gt["camera_pose"]) with
    cam1 = views[0]["camera_pose"]), so that feeding this into a zero-init
    residual PoseGRU makes the aux error identically zero. Any deviation from
    that construction — a different reference view, a missing inverse, a
    non-standardized quaternion — shows up as a nonzero oracle reading, which
    is exactly what makes the diagnostic informative.

    Returns (enc, ok):
      enc: (B, 7) absT_quaR, FLOAT32, detached.
      ok:  (B, 1) bool — False for samples whose reference pose is singular or
           whose encoding is not finite.

    fp32 is not incidental. The honest fed-back pose lives in the head's amp
    dtype (fp16 under torch.cuda.amp), and round-tripping a GT quaternion
    through fp16 leaves ~1e-4 of quaternion error — enough to make a perfectly
    wired oracle look broken. Callers must keep this fp32 and must call it
    with autocast disabled.

    torch.linalg.inv_ex (not inv): a view with no pose gets a SINGULAR
    ones((4,4)) placeholder from the dataset base class, and plain inv() raises
    for the whole batch. inv_ex returns an info code instead, so that one
    sample degrades to the honest closed loop while the rest proceed. Returns
    (None, None) for idx < 0 (no such view yet).
    """
    if idx < 0:
        return None, None
    ref = views[ref_idx]["camera_pose"].float()
    cur = views[idx]["camera_pose"].float()
    inv_ref, info = torch.linalg.inv_ex(ref)
    enc = camera_to_pose_encoding(inv_ref @ cur)
    ok = (info == 0)[:, None] & torch.isfinite(enc).all(dim=-1, keepdim=True)
    return enc.detach(), ok


class PoseGRU(nn.Module):
    """Recurrent filter for the feed_prev_pred pose loop.

    Each step consumes the (detached) 7-d absT_quaR pose encoding stashed at
    the previous step and emits a refined encoding for the current step, which
    is then rendered into the conditioning ray map. Trained by an auxiliary
    loss against the current step's GT pose; whether the main reconstruction
    loss ALSO reaches it is the model-level pose_gru_e2e switch (the copy that
    feeds the ray build is detached unless that flag is on).

    mode="residual": zero-init head adds a correction to the input pose, so an
    untrained module reproduces plain feed_prev_pred (init-equivalence check).
    mode="direct": the head regresses the pose from the hidden state alone.
    mode="split_anchor" (A5, "anchored A3"): translation is regressed DIRECTLY
    (as mode="direct") while rotation is ANCHORED on the fed-back quaternion
    (as mode="residual"). Motivated by the epoch-50 component split of the A3
    and A4 grid arms, each as a multiple of that arm's own lag baseline
    (= reuse P(x-1) unchanged; 1.00 means the module learned nothing):
        A3 direct   : translation 0.80-0.81   rotation 0.99-1.08
        A4 residual : translation 0.85-0.86   rotation 0.89-0.90
    i.e. the direct head WINS translation and is worthless on rotation, and the
    anchored head is the mirror image. Frame-to-frame rotation is near-identity,
    so q(x-1) is an excellent prior that a zero-init correction exploits and a
    from-scratch regression cannot match — and rotation is only 3-4% of the
    unweighted PoseGRULoss (t_loss + q_loss), so the aux gradient can never
    teach it. Anchoring rotation is therefore STRUCTURAL, not learned.
    Only the ROTATION rows (3:7) of the shared head are zero-init; the
    translation rows (0:3) keep the default nn.Linear init, exactly as in
    mode="direct". The parameter SET is byte-identical to residual/direct
    (one shared head), so an A5 checkpoint stays state_dict-compatible with
    A3/A4 and the mode — like residual-vs-direct already — is recoverable only
    from ckpt["args"], never from weight shapes.
    HONEST LIMIT: this anchors rotation only. Translation stays a free
    regression off the hidden state, so A5 inherits A3's translation tail —
    with the hidden zeroed, 98% of A3's degradation is translation. A5 is a
    fix for the measured ROTATION deficit, not a fallback mechanism.

    input_mode="pose": input is the 7-d fed-back pose P(x-1).
    input_mode="pose_delta": input is 14-d — P(x-1) concatenated with the
    relative motion pose_delta_encoding(P(x-2), P(x-1)), handing the module
    the loop's velocity explicitly (identity motion at the first step). In
    residual mode the correction is anchored on the POSE part (first 7 dims).

    img_feat="input" (F lever): the cell input is additionally conditioned on
    pooled PRE-ray image-encoder features of the current view (and, with
    img_feat_frames=2, the previous view): per-frame LayerNorm (shared
    affine) then an optional ZERO-INIT Linear(src_dim*frames -> img_feat_dim)
    appended AFTER the pose block — so gru_in keeps its pose-first 7/14-d
    layout and the residual anchor is untouched.

    img_feat_frames selects WHICH frames the lever contributes:
        1 ("F0")  -> the current view only            -> src_dim  ( 1024) raw
        2 ("F1")  -> current view + previous view     -> 2*src_dim( 2048) raw
    img_feat_proj selects WHETHER those pooled features are compressed:
        True  (default) -> zero-init Linear to img_feat_dim (32) columns
        False           -> the LayerNormed features go into the cell RAW,
                           i.e. the cell input widens by the FULL 1024/2048
    So the four F arms append, respectively, 32 / 32 / 1024 / 2048 columns.

    img_feat_src (source lever, suffixed onto the F arm name) selects WHAT
    produces the per-step feature — the frame/proj machinery above is shared
    by every source, only the block width changes:
        "pooled"        (default) -> mean over the CUT3R pre-ray tokens, the
                                     original arms. src_dim = enc width (1024).
        "resnet18"      ("r")     -> frozen ImageNet resnet18 avgpool feature
                                     of the RAW image. src_dim = 512.
        "dinov2_vits14" ("d")     -> frozen DINOv2 ViT-S/14 CLS token of the
                                     RAW image. src_dim = 384.
        "corr"          ("c")     -> hand-crafted correlation/flow statistics
                                     BETWEEN the current and previous views'
                                     CUT3R token sets (dust3r/img_encoders.py:
                                     corr_motion_stats). src_dim = 54. The
                                     only source that carries explicit
                                     relative-motion evidence — the per-frame
                                     sources are global summaries the cell
                                     must learn to compare.
    The encoder sources hold the frozen network as self.img_encoder — its
    (requires_grad=False, eval-locked) weights serialize into every ckpt, so
    EVAL rebuilds bit-exact from ckpt["model"] on offline nodes with no
    pretrained file; the pretrained path matters at TRAIN init only. "corr"
    is per-frame in NEITHER sense: frames must be 2 (it consumes the previous
    view) but the appended feature is ONE 54-wide block, so img_feat_blocks
    (which drives the LayerNorm reshape and projector width) is 1 there and
    == img_feat_frames everywhere else.

    self.img_feat_dim always reports the APPENDED WIDTH (what the cell
    actually receives), not the constructor's img_feat_dim, which is ignored
    when img_feat_proj=False.

    Init-equivalence to img_feat="none" holds in BOTH projector modes, but for
    different reasons, and this distinction matters:
      * img_feat_proj=True  — the projector is zero-init, so the appended
        columns are exactly zero and even the HIDDEN trajectory is identical
        to the F-off arm. The cell's default-init input columns keep the
        gradient to img_proj alive.
      * img_feat_proj=False — there is no zero gate. Image content reaches the
        hidden state from step one through the cell's default-init weight_ih
        columns. Only mode="residual"'s zero-init HEAD makes the OUTPUT
        identical to F-off at init; the hidden trajectory already differs.
        With mode="direct" there is no init-equivalence at all in this mode.
    Either way the head is the gradient gate: while it is zero, d(loss)/d(cell)
    is exactly zero, so the (much wider) input columns stay locked until the
    head unlocks — the same bootstrap the projector arm has.
    NOTE the parameter cost of img_feat_proj=False: weight_ih is
    3*hidden x input_dim, so at hidden=128 the cell grows from 17,664 params
    (46 wide) to 398,592 (1038 wide, F0) or 791,808 (2062 wide, F1), and there
    is no low-rank bottleneck forcing the module to summarise appearance.

    iters (R lever): number of back-to-back cell iterations per view. The
    CALL SITE owns the loop (this module stays a pure single-step cell — the
    probe/spy surface); the count lives here so it rides enable_pose_gru and
    the checkpoint sniff (it is invisible in weight shapes). In residual mode
    the zero-init head makes any number of untrained iterations an identity
    chain (up to repeated-quat-normalize fp noise; bit-exact at iters=1).
    """

    def __init__(
        self,
        pose_dim=7,
        hidden_dim=128,
        mode="residual",
        input_mode="pose",
        img_feat="none",
        img_feat_dim=32,
        img_feat_frames=2,
        img_feat_proj=True,
        img_feat_src_dim=1024,
        img_feat_src="pooled",
        img_encoder_pretrained=True,
        img_encoder_weights=None,
        iters=1,
    ):
        super().__init__()
        assert mode in ("residual", "direct", "split_anchor"), (
            f"unknown pose_gru mode {mode!r}"
        )
        assert input_mode in ("pose", "pose_delta"), f"unknown pose_gru input {input_mode!r}"
        assert img_feat in ("none", "input"), f"unknown pose_gru img_feat {img_feat!r}"
        assert int(iters) >= 1, f"pose_gru iters must be >= 1, got {iters!r}"
        self.mode = mode
        self.input_mode = input_mode
        self.hidden_dim = hidden_dim
        self.img_feat = img_feat
        self.img_feat_src = str(img_feat_src)
        self.img_feat_src_dim = int(img_feat_src_dim)
        self.img_encoder = None
        self.iters = int(iters)
        self.input_dim = pose_dim if input_mode == "pose" else 2 * pose_dim
        if img_feat == "input":
            from dust3r.img_encoders import (
                CORR_FEAT_DIM,
                ENCODER_DIMS,
                FrozenImageEncoder,
            )

            assert self.img_feat_src in ("pooled",) + tuple(ENCODER_DIMS) + ("corr",), (
                f"unknown pose_gru img_feat_src {img_feat_src!r}"
            )
            assert int(img_feat_frames) in (1, 2), (
                f"img_feat_frames must be 1 (current view) or 2 (current + previous), "
                f"got {img_feat_frames!r}"
            )
            self.img_feat_frames = int(img_feat_frames)
            self.img_feat_proj = bool(img_feat_proj)
            # Source lever: which network produces the appended feature.
            #   "pooled" keeps the constructor's img_feat_src_dim (the CUT3R
            #   encoder width); the other sources DICTATE their own width, so
            #   the constructor arg is overridden — self.img_feat_src_dim
            #   always reports the width of ONE block as the cell sees it.
            if self.img_feat_src == "corr":
                # The pair statistic needs the previous view by construction.
                assert self.img_feat_frames == 2, (
                    "img_feat_src='corr' is a two-frame statistic — "
                    "img_feat_frames must be 2 (arm F1c)"
                )
                self.img_feat_src_dim = CORR_FEAT_DIM
            elif self.img_feat_src in ENCODER_DIMS:
                self.img_feat_src_dim = ENCODER_DIMS[self.img_feat_src]
                self.img_encoder = FrozenImageEncoder(
                    self.img_feat_src,
                    pretrained=bool(img_encoder_pretrained),
                    weights_path=img_encoder_weights,
                )
            # Number of src_dim-wide blocks appended to the cell input. The
            # per-frame sources contribute one block per frame; corr compresses
            # the PAIR into a single block (frames stays 2 semantically — it
            # records that the previous view is consumed — but the LayerNorm
            # reshape and the projector width follow blocks, not frames).
            self.img_feat_blocks = 1 if self.img_feat_src == "corr" else self.img_feat_frames
            # LayerNorm is kept in BOTH modes: it is feature conditioning, not
            # the projection. Pooled encoder activations are not unit-scaled,
            # and feeding 1024/2048 raw columns into a GRUCell whose weight_ih
            # is initialised for O(1) inputs would swamp the 7/14 pose columns
            # that carry the residual anchor.
            self.img_norm = nn.LayerNorm(self.img_feat_src_dim)
            if self.img_feat_proj:
                assert int(img_feat_dim) > 0, (
                    "img_feat_dim must be positive when img_feat='input' and "
                    "img_feat_proj=True"
                )
                self.img_feat_dim = int(img_feat_dim)
                self.img_proj = nn.Linear(
                    self.img_feat_src_dim * self.img_feat_blocks, self.img_feat_dim
                )
                nn.init.zeros_(self.img_proj.weight)
                nn.init.zeros_(self.img_proj.bias)
            else:
                # No projector: the LayerNormed features are the cell input.
                # img_feat_dim (the constructor arg) is deliberately ignored —
                # the appended width is dictated by the source and the block
                # count, and self.img_feat_dim reports that actual width.
                self.img_feat_dim = self.img_feat_src_dim * self.img_feat_blocks
            self.input_dim += self.img_feat_dim
        else:
            # A source with the F lever OFF is a mis-paired config: the run
            # would train F-OFF while its ckpt args record a source arm, and
            # the sniff (which ignores args when img_feat='none') would later
            # relabel it 'pooled' — three names for one run. Refuse instead.
            assert self.img_feat_src == "pooled", (
                f"pose_gru_img_feat_src={img_feat_src!r} has no effect with "
                "img_feat='none' — refusing a config that would train F-OFF "
                "while recording a source arm in ckpt args"
            )
            self.img_feat_dim = 0
            self.img_feat_frames = 0
            self.img_feat_blocks = 0
            self.img_feat_proj = False
        self.cell = nn.GRUCell(self.input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, pose_dim)
        if mode == "residual":
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        elif mode == "split_anchor":
            # A5: zero ONLY the rotation rows, so q starts exactly at the
            # anchor q(x-1) (A4's winning path) while t keeps the default
            # init and is regressed from scratch (A3's winning path). Same
            # tensor, same shapes -- no new parameters, no new state_dict keys.
            with torch.no_grad():
                self.head.weight[3:pose_dim].zero_()
                self.head.bias[3:pose_dim].zero_()

    def extract_img_feat(self, img):
        """Frozen-encoder appearance of ONE view: (B, 3, H, W) in the loader's
        [-1, 1] normalization -> (B, src_dim) detached fp32. Only valid for the
        encoder sources ("resnet18"/"dinov2_vits14"); "pooled" and "corr" build
        their features from the CUT3R tokens at the call site instead."""
        assert self.img_encoder is not None, (
            f"extract_img_feat needs an encoder source, got "
            f"img_feat_src={self.img_feat_src!r}"
        )
        return self.img_encoder(img)

    def forward(self, gru_in, hidden, img_feat=None):
        """gru_in: (B, 7) or (B, 14) detached prev-step pose (+delta); hidden:
        (B, H) or None at sequence start; img_feat: (B, blocks*src_dim) image
        feature (per-frame sources: current view first, then previous; corr:
        one pair-statistic block) when img_feat="input", else None. Returns
        (refined_pose_enc, new_hidden)."""
        if hidden is None:
            hidden = gru_in.new_zeros(gru_in.shape[0], self.hidden_dim)
        cell_in = gru_in
        if self.img_feat == "input":
            assert img_feat is not None, "img_feat='input' needs the image features"
            f = img_feat.reshape(
                img_feat.shape[0], self.img_feat_blocks, self.img_feat_src_dim
            )
            f = self.img_norm(f).reshape(img_feat.shape[0], -1)
            # img_feat_proj=False sends the LayerNormed features in RAW: the
            # cell input is [pose(7|14) | features(src_dim*frames)].
            cell_in = torch.cat(
                [gru_in, self.img_proj(f) if self.img_feat_proj else f], dim=-1
            )
        hidden = self.cell(cell_in, hidden)
        raw = self.head(hidden)
        # Explicit per-mode branches with a terminal raise: the old catch-all
        # `else` silently executed the DIRECT algebra for ANY unrecognised
        # mode string (e.g. a post-hoc `gru.mode = ...` assignment or an
        # unpickled module), which would mislabel a run with no error.
        if self.mode == "residual":
            t = gru_in[:, :3] + raw[:, :3]
            q = gru_in[:, 3:7] + raw[:, 3:7]
        elif self.mode == "direct":
            t = raw[:, :3]
            q = raw[:, 3:7]
        elif self.mode == "split_anchor":
            # A5: direct translation, anchored rotation. See the class docstring.
            t = raw[:, :3]
            q = gru_in[:, 3:7] + raw[:, 3:7]
        else:
            raise ValueError(f"unknown pose_gru mode {self.mode!r}")
        q = torch.nn.functional.normalize(q, dim=-1)
        return torch.cat([t, q], dim=-1), hidden


class ARCroco3DStereo(CroCoNet):
    config_class = ARCroco3DStereoConfig
    base_model_prefix = "arcroco3dstereo"
    supports_gradient_checkpointing = True

    def __init__(self, config: ARCroco3DStereoConfig):
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        config.croco_kwargs = fill_default_args(config.croco_kwargs, CrocoConfig.__init__)
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        self.enc_blocks_ray_map = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,
                    16,
                    4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    rope=self.rope,
                )
                for _ in range(config.ray_enc_depth)
            ]
        )
        self.enc_norm_ray_map = nn.LayerNorm(self.enc_embed_dim, eps=1e-6)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            )
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,
            )
        self.register_tokens = nn.Embedding(config.state_size, self.enc_embed_dim)
        self.state_size = config.state_size
        self.state_pe = config.state_pe
        self.masked_img_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.masked_ray_map_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self._set_state_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.state_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )
        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.rgb_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )
        self.views_per_step = 1
        self.set_freeze(config.freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super().from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        self.patch_embed_ray_map = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=6
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_state = dec_depth
        self.dec_embed_dim_state = dec_embed_dim
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        if any(k.startswith("pose_gru.") for k in ckpt) and getattr(self, "pose_gru", None) is None:
            raise RuntimeError(
                "checkpoint contains pose_gru weights but the module is absent — "
                "call enable_pose_gru(hidden_dim, mode) from the config BEFORE load_state_dict"
            )
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks_state") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_state")] = value
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except:
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except:
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        elif key.startswith("pose_gru."):
                            # NEVER silently drop a trained GRU: a skipped
                            # pose_gru tensor means the module was rebuilt
                            # with the wrong hyperparameters and would
                            # evaluate as random init with no error.
                            raise RuntimeError(
                                f"pose_gru weight '{key}' size mismatch "
                                f"(ckpt: {new_ckpt[key].size()}, model: "
                                f"{self.state_dict()[key].size()}) — rebuild the "
                                f"module with the checkpoint's hyperparameters "
                                f"(enable_pose_gru / load_model sniff) or use the "
                                f"pose_gru_expand_ckpt.py surgery script"
                            )
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    elif key.startswith("pose_gru."):
                        raise RuntimeError(
                            f"pose_gru weight '{key}' not present in the model — "
                            f"call enable_pose_gru with the checkpoint's "
                            f"configuration (img_feat/iters included) before loading"
                        )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def enable_pose_gru(
        self,
        hidden_dim=128,
        mode="residual",
        input_mode="pose",
        img_feat="none",
        img_feat_dim=32,
        img_feat_frames=2,
        img_feat_proj=True,
        img_feat_src="pooled",
        img_encoder_pretrained=True,
        img_encoder_weights=None,
        iters=1,
    ):
        """Materialize the PoseGRU refiner for the feed_prev_pred loop.

        Call BEFORE load_state_dict when the checkpoint carries pose_gru
        weights, and BEFORE the optimizer is built (the module needs its own
        param group / lr; the frozen img_encoder of the "resnet18"/"dinov2_*"
        sources is requires_grad=False and skipped by get_parameter_groups).
        Created on CPU — move with .to(device) afterwards if the model already
        lives on GPU. New kwargs are defaulted so every legacy caller keeps
        constructing the exact v2 module. img_encoder_pretrained=False is the
        load_model path: the ckpt's own serialized encoder weights overwrite
        the random init, so eval never needs the pretrained file on disk.
        """
        if mode in ("direct", "split_anchor") and int(iters) > 1:
            what = "the pose" if mode == "direct" else "the TRANSLATION"
            printer.info(
                f"WARNING: pose_gru mode={mode!r} with iters>1 — nothing anchors "
                f"{what} of iterate k on iterate k-1, so the loop can collapse to "
                "a 1-step fixed point. The R lever is designed for residual mode."
            )
        self.pose_gru = PoseGRU(
            hidden_dim=hidden_dim,
            mode=mode,
            input_mode=input_mode,
            img_feat=img_feat,
            img_feat_dim=img_feat_dim,
            img_feat_frames=img_feat_frames,
            img_feat_proj=img_feat_proj,
            img_feat_src_dim=self.enc_embed_dim,
            img_feat_src=img_feat_src,
            img_encoder_pretrained=img_encoder_pretrained,
            img_encoder_weights=img_encoder_weights,
            iters=iters,
        )
        self._pose_gru_hidden = None
        self._prev_img_feat = None
        return self.pose_gru

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token] if hasattr(self, "mask_token") else [],
            "encoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
            ],
            "encoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.downstream_head,
            ],
            "encoder_and_decoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
            ],
            "decoder": [
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_rgb=bool(rgb_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(self.downstream_head, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm(x)
        return [x], pos, None

    def _encode_ray_map(self, ray_map, true_shape):
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks_ray_map:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm_ray_map(x)
        return [x], pos, None

    def _encode_state(self, image_tokens, image_pos):
        batch_size = image_tokens.shape[0]
        state_feat = self.register_tokens(torch.arange(self.state_size, device=image_pos.device))
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None

    def _encode_views(self, views, img_mask=None, ray_mask=None):
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]
        given = True
        if img_mask is None and ray_mask is None:
            given = False
        if not given:
            img_mask = torch.stack(
                [view["img_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
            ray_mask = torch.stack(
                [view["ray_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)
        ray_maps = torch.stack(
            [view["ray_map"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, H, W, C)
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(imgs.device)  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(-1, *imgs.shape[2:])  # Shape: (num_views * batch_size, C, H, W)
        ray_maps = ray_maps.view(
            -1, *ray_maps.shape[2:]
        )  # Shape: (num_views * batch_size, H, W, C)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        ray_masks_flat = ray_mask.view(-1)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device)
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i][img_masks_flat] += img_out[i]
            full_out[i][~img_masks_flat] += self.masked_img_token
        full_pos[img_masks_flat] += img_pos
        ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        selected_ray_maps = ray_maps[ray_masks_flat]
        selected_shapes_ray = shapes[ray_masks_flat]
        if selected_ray_maps.size(0) > 0:
            ray_out, ray_pos, _ = self._encode_ray_map(selected_ray_maps, selected_shapes_ray)
            assert len(ray_out) == len(full_out), f"{len(ray_out)}, {len(full_out)}"
            for i in range(len(ray_out)):
                full_out[i][ray_masks_flat] += ray_out[i]
                full_out[i][~ray_masks_flat] += self.masked_ray_map_token
            full_pos[ray_masks_flat] += (
                ray_pos * (~img_masks_flat[ray_masks_flat][:, None, None]).long()
            )
        else:
            raymaps = torch.zeros(
                1, 6, imgs[0].shape[-2], imgs[0].shape[-1], device=img_out[0].device
            )
            ray_mask_flat = torch.zeros_like(img_masks_flat)
            ray_mask_flat[:1] = True
            ray_out, ray_pos, _ = self._encode_ray_map(raymaps, shapes[ray_mask_flat])
            for i in range(len(ray_out)):
                full_out[i][ray_mask_flat] += ray_out[i] * 0.0
                full_out[i][~ray_mask_flat] += self.masked_ray_map_token * 0.0
        if getattr(self, "feed_prev_pred", False):
            # Previous-prediction conditioning: views >= 1 get ray tokens built
            # from the model's own pose prediction inside the decode loop (see
            # _forward_decoder_group_step), so the dataset must not feed rays.
            assert not ray_masks_flat.any(), (
                "feed_prev_pred is incompatible with dataset-side ray feeding "
                "(feed_gt_ray_map / feed_prev_gt_ray_map / ray_mask=True)"
            )
            # Mirror the GT-conditioned runs, where every ray-free view gets the
            # pretrained masked_ray_map_token: here only view 0 stays ray-free.
            # Rows are view-major after the flatten, so view 0 is [:batch_size].
            for i in range(len(full_out)):
                full_out[i][:batch_size] += self.masked_ray_map_token
        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
        )

    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose):
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        f_img = self.decoder_embed(f_img)
        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            f_img = torch.cat([f_pose, f_img], dim=1)
            pos_img = torch.cat([pos_pose, pos_img], dim=1)
        final_output.append((f_state, f_img))
        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                f_state, _ = checkpoint(
                    blk_state,
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    use_reentrant=not self.fixed_input_length,
                )
                f_img, _ = checkpoint(
                    blk_img,
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    use_reentrant=not self.fixed_input_length,
                )
            else:
                f_state, _ = blk_state(*final_output[-1][::+1], pos_state, pos_img)
                f_img, _ = blk_img(*final_output[-1][::-1], pos_img, pos_state)
            final_output.append((f_state, f_img))
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        # Decoder returns a list of length dec_depth+1 where each element is a tuple
        # (f_state, f_img). The outputs of the last decoder layer are replaced with
        # normalized state and image features (see lines above).
        # The zip return below transposes final_output into: (all states), (all images)
        return zip(*final_output)

    def _downstream_head(self, decout, img_shape, **kwargs):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head")
        return head(decout, img_shape, **kwargs)

    def _init_state(self, image_tokens, image_pos):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos)
        state_feat = self.decoder_embed_state(state_feat)
        return state_feat, state_pos

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        init_state_feat,
        img_mask=None,
        reset_mask=None,
        update=None,
    ):
        new_state_feat, dec = self._decoder(
            state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos
        )
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    def _forward_encoder(self, views):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        return (feat, pos, shape), (
            init_state_feat,
            init_mem,
            state_feat,
            state_pos,
            mem,
        )

    def _group_view_ranges(self, num_views):
        views_per_step = max(1, int(getattr(self, "views_per_step", 1)))
        return [
            (start, min(start + views_per_step, num_views))
            for start in range(0, num_views, views_per_step)
        ]

    def _concat_group_feat_pos(self, feat_group, pos_group):
        token_sizes = [f.shape[1] for f in feat_group]
        feat_cat = torch.cat(feat_group, dim=1)
        if len(pos_group) == 1:
            pos_cat = pos_group[0]
        else:
            max_w = max(int(p[..., 1].max().item()) for p in pos_group)
            x_stride = max_w + 1
            shifted_pos = []
            for local_idx, p in enumerate(pos_group):
                p_shift = p.clone()
                p_shift[..., 1] = p_shift[..., 1] + local_idx * x_stride
                shifted_pos.append(p_shift)
            pos_cat = torch.cat(shifted_pos, dim=1)
        offsets = []
        start = 0
        for size in token_sizes:
            offsets.append((start, start + size))
            start += size
        return feat_cat, pos_cat, offsets

    def _forward_decoder_group_step(
        self,
        views,
        view_indices,
        feat_group,
        pos_group,
        shape_group,
        init_state_feat,
        init_mem,
        state_feat,
        state_pos,
        mem,
    ):
        group_size = len(view_indices)
        feed_prev_pred = bool(getattr(self, "feed_prev_pred", False))
        gru_pose_pred = None
        gru_iterates = []  # all R-lever iterates of this view (last == gru_pose_pred)
        pose_gru_e2e = False  # set True inside the GRU block when lever G2 is live
        if feed_prev_pred:
            # Condition view x on the pose the model itself predicted at step
            # x-1, fed through the pretrained ray-map encoder — the closed-loop
            # counterpart of feed_prev_gt_ray_map. The predicted pose only
            # exists at rollout time, so the ray map is built here rather than
            # in the dataset. The pose is stashed on the module because this
            # step function is driven sequentially by both _forward_impl and
            # the TBPTT loop in loss_of_one_batch_tbptt (which bypasses
            # _forward_impl); the stash also carries it across TBPTT chunks.
            assert group_size == 1, "feed_prev_pred requires views_per_step=1"
            if view_indices[0] == 0:
                # Sequence start: nothing predicted yet. View 0 already carries
                # the masked_ray_map_token from _encode_views. Resetting here
                # also prevents leakage across batches/sequences.
                self._prev_pred_pose_enc = None
                self._prev_prev_pred_pose_enc = None
                self._pose_gru_hidden = None
                self._prev_img_feat = None
                pose_gru = getattr(self, "pose_gru", None)
                if pose_gru is not None and pose_gru.img_feat == "input":
                    # Stash view 0's appearance for the next step's (current,
                    # previous) feature pair, per the SOURCE lever. The token
                    # sources ("pooled"/"corr") see view 0's tokens carrying
                    # the constant, pose-free masked_ray_map_token added in
                    # _encode_views — subtract it so the stash is the same
                    # pure-image statistic every other view provides. The
                    # encoder sources read the RAW image, which never carries
                    # the ray token — nothing to correct.
                    src = getattr(pose_gru, "img_feat_src", "pooled")
                    if src == "pooled":
                        self._prev_img_feat = (
                            (feat_group[0] - self.masked_ray_map_token.to(feat_group[0].dtype))
                            .detach()
                            .mean(dim=1)
                            .float()
                        )
                    elif src == "corr":
                        # corr compares TOKEN SETS, so the stash keeps the
                        # full (B, N, C) tokens instead of their mean —
                        # detached fp32 data, ~1 MB/sample, crossing TBPTT
                        # chunks exactly like the pooled stash.
                        self._prev_img_feat = (
                            (feat_group[0] - self.masked_ray_map_token.to(feat_group[0].dtype))
                            .detach()
                            .float()
                        )
                    else:
                        self._prev_img_feat = pose_gru.extract_img_feat(
                            views[view_indices[0]]["img"]
                        )
            else:
                prev_pose_enc = getattr(self, "_prev_pred_pose_enc", None)
                prev_prev_pose_enc = getattr(self, "_prev_prev_pred_pose_enc", None)
                assert prev_pose_enc is not None, (
                    "feed_prev_pred: no stashed pose from the previous step — "
                    "views must be processed sequentially starting at view 0"
                )
                if os.environ.get("PREV_PRED_RAY_SHUFFLE") == "1" and prev_pose_enc.shape[0] > 1:
                    # Falsifier: batch-roll the fed-back pose so every sample is
                    # conditioned on another sample's prediction (same masks,
                    # wrong content). Supervision is untouched. BOTH stashes get
                    # the same shift so the (pose, delta) pair the GRU sees
                    # stays coherent per (wrong) source sample.
                    prev_pose_enc = torch.roll(prev_pose_enc, shifts=1, dims=0)
                    if prev_prev_pose_enc is not None:
                        prev_prev_pose_enc = torch.roll(prev_prev_pose_enc, shifts=1, dims=0)
                pose_gru = getattr(self, "pose_gru", None)
                if pose_gru is not None:
                    # Refine the fed-back pose with the recurrent filter before
                    # it is rendered into the conditioning ray map. Runs in fp32
                    # outside autocast (a pose correction is a handful of
                    # precision-sensitive scalars) and OUTSIDE no_grad — the aux
                    # criterion on res["gru_pose"] is its only training signal.
                    hidden = getattr(self, "_pose_gru_hidden", None)
                    if (
                        os.environ.get("POSE_GRU_HIDDEN_SHUFFLE") == "1"
                        and hidden is not None
                        and hidden.shape[0] > 1
                    ):
                        # Falsifier: every sample gets another sample's memory.
                        # If metrics don't degrade, the recurrence is unused.
                        hidden = torch.roll(hidden, shifts=1, dims=0)
                    if os.environ.get("POSE_GRU_HIDDEN_ZERO") == "1":
                        # Falsifier: stateless control — every step runs from a
                        # zero hidden, so any gap vs the normal arm is exactly
                        # what the recurrent memory contributes. Works at
                        # batch size 1 (unlike the batch-roll shuffle).
                        hidden = None
                    img_feat_vec = None
                    cur_img_feat = None
                    if pose_gru.img_feat == "input":
                        # F lever, per the SOURCE sub-lever. All sources are
                        # PRE-ray (the ray add below happens later) and
                        # structurally detached: the producing network is
                        # frozen and (under TBPTT) already detached — the
                        # feature is data, never a gradient path.
                        src = getattr(pose_gru, "img_feat_src", "pooled")
                        if src == "pooled":
                            cur_img_feat = feat_group[0].detach().mean(dim=1).float()
                        elif src == "corr":
                            # Full token set: the pair statistic below needs
                            # per-token correspondence, and this tensor also
                            # becomes the NEXT step's stash.
                            cur_img_feat = feat_group[0].detach().float()
                        else:
                            # Frozen-encoder appearance of the raw image
                            # (fp32 no-grad island inside extract_img_feat).
                            cur_img_feat = pose_gru.extract_img_feat(
                                views[view_indices[0]]["img"]
                            )
                        if src == "corr":
                            prev_img_feat = getattr(self, "_prev_img_feat", None)
                            assert prev_img_feat is not None, (
                                "pose_gru img_feat: no stashed view x-1 tokens — "
                                "views must be processed sequentially from view 0"
                            )
                            from dust3r.img_encoders import corr_motion_stats

                            h, w = views[view_indices[0]]["img"].shape[-2:]
                            ps = getattr(self.patch_embed, "patch_size", (16, 16))
                            if isinstance(ps, int):
                                ps = (ps, ps)
                            img_feat_vec = corr_motion_stats(
                                cur_img_feat, prev_img_feat, (h // ps[0], w // ps[1])
                            )
                        elif pose_gru.img_feat_frames == 2:
                            prev_img_feat = getattr(self, "_prev_img_feat", None)
                            assert prev_img_feat is not None, (
                                "pose_gru img_feat: no stashed view x-1 feature — "
                                "views must be processed sequentially from view 0"
                            )
                            img_feat_vec = torch.cat([cur_img_feat, prev_img_feat], dim=-1)
                        else:
                            img_feat_vec = cur_img_feat
                        if (
                            os.environ.get("POSE_GRU_IMG_FEAT_SHUFFLE") == "1"
                            and img_feat_vec.shape[0] > 1
                        ):
                            # Falsifier: every sample sees another sample's
                            # image pair (rolled coherently, mirroring
                            # PREV_PRED_RAY_SHUFFLE); pose feedback and
                            # supervision stay honest. If metrics don't
                            # degrade, the GRU ignores image evidence.
                            img_feat_vec = torch.roll(img_feat_vec, shifts=1, dims=0)
                        if os.environ.get("POSE_GRU_IMG_FEAT_ZERO") == "1":
                            # Falsifier: featureless control (works at B=1).
                            img_feat_vec = torch.zeros_like(img_feat_vec)
                    # R lever: n_iters back-to-back cell iterations, RAFT
                    # style. The pose slice of the input is the RUNNING
                    # estimate (detached between iterations per
                    # pose_gru_iter_detach — RAFT's coords1.detach()); the
                    # delta and image features are STATIC per-view context
                    # re-injected every iteration; the hidden is NEVER
                    # detached inside a view (only at the view boundary, per
                    # the G lever below). POSE_GRU_FORCE_ITERS=<n> overrides
                    # the count at eval — the anytime-inference probe.
                    n_iters = int(getattr(pose_gru, "iters", 1))
                    if os.environ.get("POSE_GRU_FORCE_ITERS"):
                        n_iters = max(1, int(os.environ["POSE_GRU_FORCE_ITERS"]))
                    iter_detach = bool(getattr(self, "pose_gru_iter_detach", True))
                    delta_static = None
                    if pose_gru.input_mode == "pose_delta":
                        # Hand the loop's velocity to the GRU explicitly: the
                        # relative motion between the last two RAW head poses
                        # (identity at view 1, where no P(x-2) exists yet).
                        delta = pose_delta_encoding(prev_prev_pose_enc, prev_pose_enc)
                        delta_static = delta.to(prev_pose_enc.dtype)
                    est = prev_pose_enc
                    # ---------------- ORACLE DIAGNOSTIC (pose_gru_oracle) ----
                    # THE ONLY CHANGE: the POSE handed to the GRU becomes the
                    # CURRENT view's GT pose instead of the previous step's
                    # predicted pose P(x-1). Nothing else. Specifically NOT
                    # changed: the delta half of the A4 input (still the honest
                    # velocity between the last two raw head poses), the image
                    # features, the hidden, the pose stashes, the ray build (it
                    # still consumes the GRU's own output), the e2e/bptt tapes,
                    # and res["gru_pose"] (still the GRU's OUTPUT — writing GT
                    # there would zero the loss by construction and test
                    # nothing). So GT enters at exactly one point and reaches
                    # the rest of the pipeline only THROUGH the GRU.
                    #
                    # The GT pose is built in the exact frame and encoding
                    # PoseGRULoss targets (gt_pose_encoding is a byte-for-byte
                    # copy of the loss's own target construction), so a GRU that
                    # simply returns its input scores zero — which a zero-init
                    # residual head does exactly. Any nonzero reading at init is
                    # a wiring, frame, dtype or masking bug.
                    #
                    # Substituted at iterate 0 only; with iters>1 the R loop
                    # re-feeds the GRU's own output as always.
                    # Placed AFTER the F-lever pooling above so image features
                    # can never become GT-contaminated.
                    oracle = str(getattr(self, "pose_gru_oracle", "off") or "off")
                    assert oracle in ("off", "gt"), (
                        f"unknown pose_gru_oracle {oracle!r} (expected off|gt)"
                    )
                    if oracle == "gt":
                        with torch.autocast(
                            device_type=feat_group[0].device.type, enabled=False
                        ):
                            gt_cur, ok_cur = gt_pose_encoding(views, view_indices[0])
                        # Stay in FLOAT32 — do NOT cast to prev_pose_enc.dtype.
                        # That stash is the head's camera_pose produced under
                        # torch.cuda.amp, i.e. fp16; round-tripping the GT quat
                        # through fp16 leaves ~1e-4 of error and a perfectly
                        # wired oracle would read nonzero. torch.where promotes
                        # the honest per-sample fallback up to fp32.
                        est = torch.where(ok_cur, gt_cur, prev_pose_enc.float())
                        if delta_static is not None:
                            # Widen the (unchanged) honest delta so the cat below
                            # is homogeneous. fp16 -> fp32 is exact: same value.
                            delta_static = delta_static.to(est.dtype)
                        if (
                            os.environ.get("PREV_PRED_RAY_SHUFFLE") == "1"
                            and est.shape[0] > 1
                        ):
                            # Re-apply the falsifier AFTER the substitution: the
                            # roll above hit the stash the oracle just
                            # overwrote and would otherwise be a silent no-op.
                            # The loss target is NOT rolled, so under the oracle
                            # this MUST destroy the zero — the positive control.
                            est = torch.roll(est, shifts=1, dims=0)
                    with torch.autocast(device_type=feat_group[0].device.type, enabled=False):
                        for _it in range(n_iters):
                            gru_in = (
                                est
                                if delta_static is None
                                else torch.cat([est, delta_static], dim=-1)
                            )
                            gru_pose_pred, hidden = pose_gru(
                                gru_in.float(), hidden, img_feat=img_feat_vec
                            )
                            gru_iterates.append(gru_pose_pred)
                            if _it + 1 < n_iters:
                                est = (
                                    gru_pose_pred.detach() if iter_detach else gru_pose_pred
                                )
                    new_hidden = hidden
                    if pose_gru.img_feat == "input":
                        # Stash the current view's (honest, unfalsified)
                        # pre-ray appearance for the next step's pair — the
                        # pooled/encoder vector or, for corr, the full token
                        # set. Pure detached data — crosses TBPTT chunks like
                        # the pose stashes, no boundary handling needed.
                        self._prev_img_feat = cur_img_feat
                    if bool(getattr(self, "pose_gru_bptt", False)):
                        # Within-chunk BPTT (lever G1): keep the tape so the aux
                        # loss at a later view reaches the GRU calls of earlier
                        # views IN THE SAME TBPTT CHUNK. The chunk-boundary
                        # detach lives in loss_of_one_batch_tbptt, next to the
                        # state/mem detaches — without it the next chunk's
                        # backward would cross this chunk's freed graph.
                        self._pose_gru_hidden = new_hidden
                    else:
                        # Per-step truncation (G0, the simple variant): the
                        # hidden crosses steps as data only.
                        self._pose_gru_hidden = new_hidden.detach()
                    # The refined pose replaces the raw prediction for the ray
                    # build below. Detached by default (aux-loss-only training);
                    # pose_gru_e2e (lever G2) keeps the tape so the main
                    # reconstruction loss also reaches the GRU through the
                    # pose -> ray map -> frozen ray encoder chain (frozen only
                    # pins the encoder's params; gradient passes through).
                    # NOT touched by pose_gru_oracle: the refined pose drives the
                    # ray build exactly as always, under 'off' and 'gt' alike.
                    pose_gru_e2e = bool(getattr(self, "pose_gru_e2e", False)) and (
                        torch.is_grad_enabled() and gru_pose_pred.requires_grad
                    )
                    prev_pose_enc = gru_pose_pred if pose_gru_e2e else gru_pose_pred.detach()
                prev_view = views[view_indices[0] - 1]
                assert "camera_intrinsics" in prev_view, (
                    "feed_prev_pred needs per-view camera_intrinsics "
                    "to build ray maps from predicted poses"
                )
                # The ray encoder is frozen (freeze='encoder') and, by default,
                # the pose entering here is detached — nothing needs gradients
                # (grad disabled, exactly the old no_grad). Under pose_gru_e2e
                # (lever G2) the pose carries the GRU's tape, so grad stays
                # enabled and the main loss reaches the GRU through this build
                # (the frozen encoder's params still get no gradient).
                with torch.set_grad_enabled(pose_gru_e2e):
                    # Build the map in fp32 even under an ambient autocast: the
                    # GT path gets loader-built fp32 maps, only the encoder runs
                    # autocast'd — mirror that split exactly.
                    with torch.autocast(device_type=feat_group[0].device.type, enabled=False):
                        # Predicted camera_pose is the postprocessed absT_quaR
                        # encoding, supervised relative to view 0 — the same
                        # frame the GT ray maps use (inv(cam0) @ cam_i).
                        # Intrinsics come from view x-1, matching
                        # feed_prev_gt_ray_map's shift of view x-1's entire map.
                        c2w = pose_encoding_to_camera(prev_pose_enc.float())
                        h, w = views[view_indices[0]]["img"].shape[-2:]
                        rmap = get_ray_map_torch(c2w, prev_view["camera_intrinsics"], h, w)
                    ray_out, _, _ = self._encode_ray_map(
                        rmap.permute(0, 3, 1, 2).to(feat_group[0].dtype),
                        shape_group[0],
                    )
                feat_group = [feat_group[0] + ray_out[-1].to(feat_group[0].dtype)]
        feat_cat, pos_cat, token_offsets = self._concat_group_feat_pos(feat_group, pos_group)
        debug_grouped = bool(getattr(self, "debug_grouped_updates", False))
        debug_once = bool(getattr(self, "debug_grouped_updates_once", True))
        debug_emitted = int(getattr(self, "_debug_grouped_updates_emitted", 0))
        should_debug_print = (
            debug_grouped and group_size > 1 and ((not debug_once) or (debug_emitted == 0))
        )
        if should_debug_print:
            for local_idx, view_idx in enumerate(view_indices):
                print(
                    f"[GroupedUpdate] received view {view_idx} ({local_idx + 1}/{group_size}); not committing state/memory yet"
                )
            print(
                f"[GroupedUpdate] concatenated group tokens: feat_cat={tuple(feat_cat.shape)}, pos_cat={tuple(pos_cat.shape)}"
            )
        if self.pose_head_flag:
            global_img_feat_group = torch.stack(
                [self._get_img_level_feat(f) for f in feat_group], dim=0
            ).mean(dim=0)
            if view_indices[0] == 0:
                pose_feat_group = self.pose_token.expand(feat_cat.shape[0], group_size, -1)
            else:
                pose_seed = self.pose_retriever.inquire(global_img_feat_group, mem)
                pose_feat_group = pose_seed.expand(-1, group_size, -1).contiguous()
            pose_pos_group = torch.zeros(
                feat_cat.shape[0],
                group_size,
                2,
                device=feat_cat.device,
                dtype=pos_cat.dtype,
            )
            pose_pos_group[..., 1] = torch.arange(
                group_size, device=feat_cat.device, dtype=pos_cat.dtype
            )[None]
        else:
            global_img_feat_group = None
            pose_feat_group = None
            pose_pos_group = None
        new_state_feat, dec = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_cat,
            pos_cat,
            pose_feat_group,
            pose_pos_group,
            init_state_feat,
        )
        if self.pose_head_flag:
            out_pose_feat_group = dec[-1][:, :group_size]
            pooled_pose_feat = out_pose_feat_group.mean(dim=1, keepdim=True)
            new_mem = self.pose_retriever.update_mem(mem, global_img_feat_group, pooled_pose_feat)
        else:
            new_mem = mem
        assert len(dec) == self.dec_depth + 1

        res_group = []
        for local_idx, view_idx in enumerate(view_indices):
            start, end = token_offsets[local_idx]
            if self.pose_head_flag:
                stage_q2 = dec[self.dec_depth * 2 // 4][:, group_size + start : group_size + end]
                stage_q3 = dec[self.dec_depth * 3 // 4][:, group_size + start : group_size + end]
                stage_last = torch.cat(
                    [
                        dec[self.dec_depth][:, local_idx : local_idx + 1],
                        dec[self.dec_depth][:, group_size + start : group_size + end],
                    ],
                    dim=1,
                )
            else:
                stage_q2 = dec[self.dec_depth * 2 // 4][:, start:end]
                stage_q3 = dec[self.dec_depth * 3 // 4][:, start:end]
                stage_last = dec[self.dec_depth][:, start:end]
            head_input = [
                dec[0][:, start:end].float(),
                stage_q2.float(),
                stage_q3.float(),
                stage_last.float(),
            ]
            res = self._downstream_head(
                head_input, shape_group[local_idx], pos=pos_group[local_idx]
            )
            res_group.append(res)

        if feed_prev_pred:
            # Stash this step's predicted pose for the next step's ray map.
            # detach(): the fed-back pose is an INPUT at step x+1, not a second
            # gradient path into step x's pose head — and under TBPTT the
            # previous chunk's graph is already freed, so backprop through a
            # non-detached pose would crash at the chunk boundary. The old
            # stash shifts to P(x-2) first — the pose_delta GRU input needs
            # the last TWO raw head poses to compute the loop's velocity.
            self._prev_prev_pred_pose_enc = getattr(self, "_prev_pred_pose_enc", None)
            self._prev_pred_pose_enc = res_group[-1]["camera_pose"].detach()
            if gru_pose_pred is not None:
                # Grad-carrying refined pose for the aux criterion. The loss
                # builds its own GT target from the views — same construction
                # as the main pose loss: camera_to_pose_encoding(inv(cam1)@gt).
                res_group[-1]["gru_pose"] = gru_pose_pred
                if len(gru_iterates) > 1:
                    # R lever: all iterates (final == gru_pose) for the
                    # gamma-weighted sequence loss. ONE stacked (N, B, 7)
                    # tensor, not a list — the TBPTT all_preds collection
                    # blanket-detaches tensor values and passes it through.
                    res_group[-1]["gru_pose_iters"] = torch.stack(gru_iterates, dim=0)

        img_mask_group = torch.stack([views[i]["img_mask"] for i in view_indices], dim=0).any(
            dim=0
        )
        updates = [views[i].get("update", None) for i in view_indices]
        if any(update is not None for update in updates):
            update_group = torch.stack(
                [
                    (
                        update
                        if update is not None
                        else torch.ones_like(img_mask_group, dtype=torch.bool)
                    )
                    for update in updates
                ],
                dim=0,
            ).any(dim=0)
            update_mask = img_mask_group & update_group
        else:
            update_mask = img_mask_group
        update_mask = update_mask[:, None, None].float()
        state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
        mem = new_mem * update_mask + mem * (1 - update_mask)
        reset_mask = torch.stack([views[i]["reset"] for i in view_indices], dim=0).any(dim=0)
        reset_mask = reset_mask[:, None, None].float()
        state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
        mem = init_mem * reset_mask + mem * (1 - reset_mask)
        if should_debug_print:
            print(f"[GroupedUpdate] committed single state/memory update for views {view_indices}")
            self._debug_grouped_updates_emitted = debug_emitted + 1
        return res_group, (state_feat, mem)

    def _forward_decoder_step(
        self,
        views,
        i,
        feat_i,
        pos_i,
        shape_i,
        init_state_feat,
        init_mem,
        state_feat,
        state_pos,
        mem,
    ):
        res_group, (state_feat, mem) = self._forward_decoder_group_step(
            views=views,
            view_indices=[i],
            feat_group=[feat_i],
            pos_group=[pos_i],
            shape_group=[shape_i],
            init_state_feat=init_state_feat,
            init_mem=init_mem,
            state_feat=state_feat,
            state_pos=state_pos,
            mem=mem,
        )
        return res_group[0], (state_feat, mem)

    def _forward_impl(self, views, ret_state=False):
        shape, feat_ls, pos = self._encode_views(views)  # monkey patched to use DA3 tokens

        # feat_ls is a single tuple of length 118. Each tensor has shape (1, 1024, 1024)
        feat = feat_ls[-1]
        # feat[0] are the encoded features of the first input view, this is what needs to be
        # substitude with DA3 transformer tokens.

        # State and memory are initialized
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]

        ress = []
        for start, end in self._group_view_ranges(len(views)):
            view_indices = list(range(start, end))
            feat_group = [feat[i] for i in view_indices]
            pos_group = [pos[i] for i in view_indices]
            shape_group = [shape[i] for i in view_indices]
            res_group, (state_feat, mem) = self._forward_decoder_group_step(
                views=views,
                view_indices=view_indices,
                feat_group=feat_group,
                pos_group=pos_group,
                shape_group=shape_group,
                init_state_feat=init_state_feat,
                init_mem=init_mem,
                state_feat=state_feat,
                state_pos=state_pos,
                mem=mem,
            )
            ress.extend(res_group)
            for _ in view_indices:
                all_state_args.append((state_feat, state_pos, init_state_feat, mem, init_mem))
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward(self, views, ret_state=False):
        if ret_state:
            ress, views, state_args = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            ress, views = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)

    ##################################################################################################
    ### DEPTH ANYTHING 3 #############################################################################
    ##################################################################################################
    def _da3_forward_impl(self, shape, feat_ls, pos, ret_state=False):
        feat = feat_ls
        # print(f"DEBUG feat.shape: {feat.shape}")

        # feat[0] are the encoded features of the first input view, this is what needs to be
        # substitude with DA3 transformer tokens.
        print(f"DEBUG feat[0].shape: {feat[0].shape}")
        print(f"DEBUG pos[0].shape: {pos[0].shape}")
        # State and memory are initialized
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]

        # Debug prints for state arguments
        print(f"DEBUG state_feat.shape: {state_feat.shape}")
        print(f"DEBUG state_pos.shape: {state_pos.shape}")
        print(f"DEBUG init_state_feat.shape: {init_state_feat.shape}")
        print(f"DEBUG mem.shape: {mem.shape}")
        print(f"DEBUG init_mem.shape: {init_mem.shape}")

        ress = []
        # for i in range(feat.shape[0]):
        print(f"DEBUG len(feat): {len(feat)}")
        for i in range(len(feat)):
            feat_i = feat[i]
            pos_i = pos[i]
            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                # pose_pos_i = -torch.ones(
                #     feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                # )
                pose_pos_i = torch.zeros(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None

            # Manually set image mask, reset mask, and update
            img_mask = torch.ones(feat_i.shape[0], dtype=torch.bool, device=feat_i.device)
            reset_mask = torch.zeros(feat_i.shape[0], dtype=torch.bool, device=feat_i.device)
            update = torch.ones(feat_i.shape[0], dtype=torch.bool, device=feat_i.device)
            new_state_feat, dec = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                # img_mask=views[i]["img_mask"],
                # reset_mask=views[i]["reset"],
                # update=views[i].get("update", None),
                img_mask=img_mask,
                reset_mask=reset_mask,
                update=update,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)
            img_mask = torch.ones(feat_i.shape[0], dtype=torch.bool, device=feat_i.device)
            update = torch.ones(feat_i.shape[0], dtype=torch.bool, device=feat_i.device)
            if update is not None:
                update_mask = img_mask & update  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            mem = new_mem * update_mask + mem * (1 - update_mask)  # then update local state
            # reset_mask = views[i]["reset"]
            reset_mask = torch.zeros(feat_i.shape[0], dtype=torch.bool, device=feat_i.device)
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append((state_feat, state_pos, init_state_feat, mem, init_mem))
        if ret_state:
            return ress, None, all_state_args
        return ress, None

    def da3_forward(self, shape, feat_ls, pos, ret_state=False):
        if ret_state:
            ress, views, state_args = self._da3_forward_impl(
                shape, feat_ls, pos, ret_state=ret_state
            )
            return ARCroco3DStereoOutput(ress, views), state_args
        else:
            print(f"Running DA3 forward WITHOUT returning state arguments.")
            ress, views = self._da3_forward_impl(shape, feat_ls, pos, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)

    ##################################################################################################
    ##################################################################################################
    ##################################################################################################

    def inference_step(self, view, state_feat, state_pos, init_state_feat, mem, init_mem):
        batch_size = view["img"].shape[0]
        raymaps = []
        shapes = []
        for j in range(batch_size):
            assert view["ray_mask"][j]
            raymap = view["ray_map"][[j]].permute(0, 3, 1, 2)
            raymaps.append(raymap)
            shapes.append(
                view.get(
                    "true_shape",
                    torch.tensor(view["ray_map"].shape[-2:])[None].repeat(
                        view["ray_map"].shape[0], 1
                    ),
                )[[j]]
            )

        raymaps = torch.cat(raymaps, dim=0)
        shape = torch.cat(shapes, dim=0).to(raymaps.device)
        feat_ls, pos, _ = self._encode_ray_map(raymaps, shapes)

        feat_i = feat_ls[-1]
        pos_i = pos
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            # pose_pos_i = -torch.ones(
            #     feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            # )
            pose_pos_i = torch.zeros(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
        )

        out_pose_feat_i = dec[-1][:, 0:1]
        self.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
        assert len(dec) == self.dec_depth + 1
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape, pos=pos_i)
        return res, view

    def forward_recurrent(self, views, device, ret_state=False):
        ress = []
        all_state_args = []
        for i, view in enumerate(views):
            device = view["img"].device
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(-1, batch_size)  # Shape: (1, batch_size)
            ray_mask = view["ray_mask"].reshape(-1, batch_size)  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            ray_maps = view["ray_map"].unsqueeze(0)  # Shape: (num_views, batch_size, H, W, C)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(-1, *imgs.shape[2:])  # Shape: (num_views * batch_size, C, H, W)
            ray_maps = ray_maps.view(
                -1, *ray_maps.shape[2:]
            )  # Shape: (num_views * batch_size, H, W, C)
            shapes = shapes.view(-1, 2).to(imgs.device)  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
            ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(selected_ray_maps, selected_shapes_ray)
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()
                all_state_args.append((state_feat, state_pos, init_state_feat, mem, init_mem))

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                # pose_pos_i = -torch.ones(
                #     feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                # )
                pose_pos_i = torch.zeros(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)
            ress.append(res)
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = img_mask & update  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            mem = new_mem * update_mask + mem * (1 - update_mask)  # then update local state
            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append((state_feat, state_pos, init_state_feat, mem, init_mem))
        if ret_state:
            return ress, views, all_state_args
        return ress, views


if __name__ == "__main__":
    print(ARCroco3DStereo.mro())

    cfg = ARCroco3DStereoConfig(
        state_size=256,
        pos_embed="RoPE100",
        rgb_head=True,
        pose_head=True,
        img_size=(224, 224),
        head_type="linear",
        output_mode="pts3d+pose",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        pose_mode=("exp", -inf, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
    )

    ARCroco3DStereo(cfg)
