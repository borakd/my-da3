#!/usr/bin/env python
"""Train the two write-gate confidence heads on features recorded by record_wgate_features.py
(WRITE_GATE_VARIANTS.md, "Recording + training", Component B).

Both heads are Kendall & Gal heteroscedastic heads on a FROZEN CUT3R, fitted offline on the recorded table
(frames t>0 only; frame 0 is always written in full):

  FrameConfHead (variant 1, in=1796 -> 256 -> GELU -> 1): s = log sigma^2 of the frame's own error
      r = res_pose_t + res_pose_R + res_pts_self,      loss = exp(-s) r^2 + s          (paper eq. 8 form)
  PoseSigmaHead (variant 2, frozen h = GELU(fc1(pose_feat)) from the pose decoder -> NEW Linear(3072, 2)):
      (s_t, s_R) = log sigma^2 of r_t = ||t_pred - t_gt|| and r_R = ||q_pred - q_gt||,
      loss = [exp(-s_t) r_t^2 + s_t] + [exp(-s_R) r_R^2 + s_R]   (only the new linear layer trains)

Residuals are standardised by their TRAIN median (the divisor is stored in the checkpoint; the sigma -> weight
map uses only sigma_ref / sigma so the scale cancels at inference). AdamW lr 1e-3, 20 epochs, batch 4096
frames, cosine decay. Reports held-out NLL (vs. a constant-sigma baseline), Spearman(sigma, residual) and a
5-bin calibration table per head, and writes exactly the contract's checkpoints:
    <out>/frame_conf_head.pth = {"state_dict", "log_ref", "wmin", "args", "metrics"} (+ "res_scale", "head_cfg")
    <out>/pose_sigma_head.pth = {"state_dict" (out layer only), "log_ref_t", "log_ref_R", "wmin", "args", "metrics"}
                                (+ "res_scale_t", "res_scale_R")
with log_ref = median log-variance over the TRAIN table and wmin = 0.5.

Target (`--target`, WRITE_GATE_VARIANTS.md "the literal residual is a clock"): `abs` (default, the above = the
finetune loss's pose residual relative to frame 0, which ACCUMULATES with t) or `rel` = the consecutive-frame
RELATIVE pose residual the recorder stores as res_rel_t / res_rel_R (frame t-1 -> t, same normalisation):
    FrameConfHead  r = res_rel_t + res_rel_R + res_pts_self;   PoseSigmaHead (r_t, r_R) = (res_rel_t, res_rel_R)
everything else identical. With --target rel the outputs are frame_conf_head_rel.pth, pose_sigma_head_rel.pth,
train_wgate_metrics_rel.json, train_wgate_log_rel.txt (abs keeps the names above); the checkpoint dict and the
metrics carry "target". Both targets also report the clock diagnostic Spearman(t, r) on the TRAIN table
(metrics "clock": per residual array, plus per head "spearman_t_r*" under "train").

  python train_wgate_heads.py                       # all npz under <root>/features/{train,heldout}
  python train_wgate_heads.py --target rel          # the relative-residual heads (npz need res_rel_*; --fill_rel)
  python train_wgate_heads.py --smoke               # 2 episodes, 1 epoch
  python train_wgate_heads.py --selftest            # CPU: synthetic npz table -> both heads, both targets, must beat chance
"""
import argparse, glob, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import dust3r.utils.path_to_croco  # noqa: F401
import dust3r.heads  # noqa: F401  (must precede dust3r.utils.camera: the two modules import each other)
from dust3r.utils.camera import PoseDecoder
from train_unc_gate import CKPT_DEFAULT
try:  # Component A's loss helper (the one the model package ships); the local fallback below matches it
    from dust3r.wgate.heads import gaussian_nll as _gaussian_nll_wgate
except ImportError:
    _gaussian_nll_wgate = None

WGATE_ROOT = "/gpfs/scratch/etur59/koc821022/checkpoints/wgate"
FRAME_FEAT_DIM, POSE_FEAT_DIM = 1796, 768
S_CLAMP = 30.0  # fallback only: clamp of s inside exp() against fp32 overflow (exp(30) ~ 1e13); s itself is unbounded
WMIN = 0.5
RES_KEYS = ("res_pose_t", "res_pose_R", "res_pts_self")
REL_KEYS = ("res_rel_t", "res_rel_R")          # recorder's consecutive-frame relative residuals (0 at t = 0)
ALL_RES_KEYS = RES_KEYS + REL_KEYS
TARGETS = {"abs": {"frame": ("res_pose_t", "res_pose_R", "res_pts_self"), "pose": ("res_pose_t", "res_pose_R"), "suffix": ""},
           "rel": {"frame": ("res_rel_t", "res_rel_R", "res_pts_self"), "pose": ("res_rel_t", "res_rel_R"), "suffix": "_rel"}}


def target_spec(target):
    """Residual keys of a training target: {"frame": keys summed for FrameConfHead, "pose": (r_t key, r_R key),
    "suffix": output-file suffix, "keys": every key the target needs}."""
    if target not in TARGETS:
        raise ValueError(f"--target must be one of {sorted(TARGETS)}, got {target!r}")
    d = dict(TARGETS[target]); d["keys"] = tuple(dict.fromkeys(d["frame"] + d["pose"])); d["name"] = target
    return d


# ------------------------------------------------------------------------------------------------ heads
def import_heads(allow_standin=False):
    """Component A's heads. The stand-ins below exist ONLY so --selftest can run before dust3r.wgate lands;
    they follow the contract's shapes (FrameConfHead(in_dim, hidden) -> s (B,), PoseSigmaHead(pose_decoder)
    -> (B,2) with self.out = Linear(3072, 2))."""
    try:
        from dust3r.wgate.heads import FrameConfHead, PoseSigmaHead
        return FrameConfHead, PoseSigmaHead, False
    except ImportError as e:
        if not allow_standin:
            raise ImportError("dust3r.wgate.heads is missing (Component A of WRITE_GATE_VARIANTS.md not landed?): "
                              f"{e}") from e
        print("[wgate] WARNING: dust3r.wgate.heads not importable; --selftest uses the contract stand-in heads", flush=True)
        return _StandInFrameConfHead, _StandInPoseSigmaHead, True


class _StandInFrameConfHead(nn.Module):
    """SELFTEST ONLY. LayerNorm(in) -> Linear(in, hidden) -> GELU -> Linear(hidden, 1); returns s (B,)."""
    def __init__(self, in_dim=FRAME_FEAT_DIM, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim); self.fc1 = nn.Linear(in_dim, hidden); self.act = nn.GELU(); self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))[:, 0]


class _StandInPoseSigmaHead(nn.Module):
    """SELFTEST ONLY. h = act(fc1(pose_feat)) from the frozen pose decoder, then self.out = Linear(3072, 2)."""
    def __init__(self, pose_decoder):
        super().__init__()
        self.fc1 = pose_decoder.mlp.fc1; self.act = pose_decoder.mlp.act
        for p in self.fc1.parameters():
            p.requires_grad_(False)
        self.out = nn.Linear(self.fc1.out_features, 2)

    def forward(self, pose_feat):
        with torch.no_grad():
            h = self.act(self.fc1(pose_feat))
        return self.out(h)


def load_pose_decoder(path, ckpt_fallback=CKPT_DEFAULT):
    """The frozen PoseDecoder (Mlp 768 -> 3072 -> 7). Prefers <root>/pose_decoder.pth written by the recorder;
    otherwise mmap-loads the full checkpoint and extracts `downstream_head.pose_head.*` (with or without the
    DDP `module.` prefix)."""
    if path and os.path.isfile(path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        sd = d["pose_decoder"] if isinstance(d, dict) and "pose_decoder" in d else d
        src = path
    else:
        if not os.path.isfile(ckpt_fallback):
            raise FileNotFoundError(f"neither pose decoder file {path} nor checkpoint {ckpt_fallback} exists")
        ck = torch.load(ckpt_fallback, map_location="cpu", mmap=True, weights_only=False)
        tag = "downstream_head.pose_head."
        sd = {k.split(tag, 1)[1]: v.clone() for k, v in ck["model"].items() if tag in k}
        src = ckpt_fallback
    assert "mlp.fc1.weight" in sd, f"pose decoder weights not found in {src}: keys {list(sd)[:5]}"
    pd = PoseDecoder(hidden_size=int(sd["mlp.fc1.weight"].shape[1]), mlp_ratio=int(sd["mlp.fc1.weight"].shape[0] // sd["mlp.fc1.weight"].shape[1]))
    pd.load_state_dict(sd)
    for p in pd.parameters():
        p.requires_grad_(False)
    return pd.eval(), src


# ------------------------------------------------------------------------------------------------ data
def load_split(split_dir, max_episodes=0, t_min=1, P=print, res_keys=RES_KEYS):
    """Concatenate all npz of one split (frames t >= t_min). Returns dict of numpy arrays:
    frame_feat (N,1796) f16, pose_feat (N,768) f16, res_* (N,) f32 (every ALL_RES_KEYS array; one absent from
    the npz, e.g. res_rel_* in a table recorded before them, is NaN-filled unless it is in `res_keys`), ep (N,)
    i64 (episode index), t (N,) i32, pts_valid_frac (N,) f32, and 'files'. Drops frames with a non-finite
    residual among `res_keys` (the training target's arrays) and episodes whose criterion pose_mask is False;
    reports both. A `res_keys` array missing from an npz is an error naming the fix (--fill_rel)."""
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    if max_episodes:
        files = files[:max_episodes]
    if not files:
        raise FileNotFoundError(f"no npz under {split_dir}")
    cols = {k: [] for k in ("frame_feat", "pose_feat", "t", "pts_valid_frac", "ep") + ALL_RES_KEYS}
    n_bad_ep = n_bad_fr = 0; n_missing = {k: 0 for k in ALL_RES_KEYS}
    for f in files:
        z = np.load(f)
        if "pose_mask" in z and not bool(z["pose_mask"]):
            n_bad_ep += 1; continue
        T = z["frame_feat"].shape[0]
        t = z["t"] if "t" in z else np.arange(T, dtype=np.int32)
        sel = t >= t_min
        fin = np.ones(T, dtype=bool)
        for k in res_keys:
            if k not in z:
                raise KeyError(f"{f} has no '{k}' array (recorded before the {k} target existed?); re-record it with "
                               f"record_wgate_features.py --fill_rel (EXTRA='--fill_rel' in record_wgate.sbatch)")
            fin &= np.isfinite(z[k])
        n_bad_fr += int((sel & ~fin).sum()); sel &= fin
        ep_idx = int(os.path.splitext(os.path.basename(f))[0]) if os.path.basename(f)[:-4].isdigit() else len(cols["ep"])
        cols["frame_feat"].append(z["frame_feat"][sel].astype(np.float16)); cols["pose_feat"].append(z["pose_feat"][sel].astype(np.float16))
        for k in ALL_RES_KEYS:
            if k in z:
                cols[k].append(z[k][sel].astype(np.float32))
            else:
                n_missing[k] += 1; cols[k].append(np.full(int(sel.sum()), np.nan, dtype=np.float32))
        cols["t"].append(t[sel].astype(np.int32)); cols["ep"].append(np.full(int(sel.sum()), ep_idx, dtype=np.int64))
        cols["pts_valid_frac"].append((z["pts_valid_frac"][sel] if "pts_valid_frac" in z else np.ones(int(sel.sum()))).astype(np.float32))
    out = {k: np.concatenate(v, 0) for k, v in cols.items()}
    out["files"] = files
    P(f"[wgate] {split_dir}: {len(files)} episodes -> {out['frame_feat'].shape[0]} frames (t>={t_min}); "
      f"dropped {n_bad_ep} episodes (pose_mask False), {n_bad_fr} frames (non-finite residual in {list(res_keys)})"
      + ("; arrays absent in some npz (NaN-filled, not a target): " + ", ".join(f"{k} x{n}" for k, n in n_missing.items() if n)
         if any(n_missing.values()) else ""))
    assert out["frame_feat"].shape[1] == FRAME_FEAT_DIM and out["pose_feat"].shape[1] == POSE_FEAT_DIM, \
        (out["frame_feat"].shape, out["pose_feat"].shape)
    return out


# ------------------------------------------------------------------------------------------------ maths
def gaussian_nll(s, r):
    """Per-element K&G Gaussian NLL exp(-s) r^2 + s.

    Uses `dust3r.wgate.heads.gaussian_nll` (Component A, |s| clamped at 30 inside exp only) when importable so
    the trainer optimises exactly the loss the package defines; the local fallback (for --selftest without
    Component A) clamps only the overflow side, `exp(-max(s, -30))`. A tight symmetric clamp (the earlier +-10)
    is a trap: below the clamp the loss degenerates to const + s with gradient +1 and nothing pulls s back, which a
    standardised residual under e^-5 medians would have triggered; with 30 that needs r < e^-15 medians."""
    if _gaussian_nll_wgate is not None:
        return _gaussian_nll_wgate(s, r)
    return torch.exp(-s.clamp(min=-S_CLAMP)) * r ** 2 + s


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1); b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).correlation)
    except Exception:
        ra, rb = _rankdata(a), _rankdata(b)
        return float(np.corrcoef(ra, rb)[0, 1])


def _rankdata(x):
    order = np.argsort(x, kind="mergesort"); ranks = np.empty(x.size, dtype=np.float64)
    xs = x[order]; i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return ranks


def calibration_table(s, r, bins=5):
    """Bin frames by predicted sigma quantiles; a calibrated Gaussian head has rms(r) ~ rms(sigma) per bin."""
    s = np.asarray(s, dtype=np.float64).reshape(-1); r = np.asarray(r, dtype=np.float64).reshape(-1)
    sigma = np.exp(0.5 * s)
    edges = np.quantile(sigma, np.linspace(0, 1, bins + 1)); rows = []
    for i in range(bins):
        m = (sigma >= edges[i]) & ((sigma <= edges[i + 1]) if i == bins - 1 else (sigma < edges[i + 1]))
        if m.sum() == 0:
            rows.append({"bin": i, "n": 0}); continue
        srms = float(np.sqrt((sigma[m] ** 2).mean())); rrms = float(np.sqrt((r[m] ** 2).mean()))
        rows.append({"bin": i, "n": int(m.sum()), "sigma_lo": float(edges[i]), "sigma_hi": float(edges[i + 1]),
                     "sigma_rms": srms, "r_rms": rrms, "r_med": float(np.median(np.abs(r[m]))),
                     "ratio_r_over_sigma": float(rrms / max(srms, 1e-12))})
    return rows


def sigma_weight(s, log_ref, wmin):
    """clip(sigma_ref / sigma, wmin, 1) = clip(exp((log_ref - s)/2), wmin, 1) (local copy of the contract's map)."""
    return np.clip(np.exp(0.5 * (log_ref - np.asarray(s, dtype=np.float64))), wmin, 1.0)


def head_metrics(s, r, names, s_const, log_refs, wmin):
    """s, r: (N,k) numpy (standardised residual units). s_const: (k,) constant-s baseline from the TRAIN split."""
    s = np.asarray(s, dtype=np.float64); r = np.asarray(r, dtype=np.float64)
    st = torch.as_tensor(s); rt = torch.as_tensor(r)
    m = {"n": int(s.shape[0]), "nll": float(gaussian_nll(st, rt).sum(-1).mean()),
         "nll_const": float(gaussian_nll(torch.as_tensor(np.ascontiguousarray(np.broadcast_to(s_const, s.shape))), rt).sum(-1).mean())}
    for j, name in enumerate(names):
        m[f"nll_{name}"] = float(gaussian_nll(st[:, j], rt[:, j]).mean())
        m[f"spearman_{name}"] = spearman(np.exp(0.5 * s[:, j]), r[:, j])
        m[f"calib_{name}"] = calibration_table(s[:, j], r[:, j])
        m[f"s_mean_{name}"] = float(s[:, j].mean()); m[f"s_std_{name}"] = float(s[:, j].std())
        m[f"weight_mean_{name}"] = float(sigma_weight(s[:, j], log_refs[j], wmin).mean())
    if len(names) == 2:  # variant 2: the combined map is monotone in s_t + s_R (refs are constants)
        m["spearman_comb"] = spearman(s.sum(-1), r.sum(-1))
        lc = 0.25 * ((s[:, 0] - log_refs[0]) + (s[:, 1] - log_refs[1]))  # log sigma_comb
        m["weight_mean_comb"] = float(np.clip(np.exp(-lc), wmin, 1.0).mean())
    return m


def fmt_calib(rows):
    out = ["  bin      n   sigma_rms     r_rms    r_med  r/sigma"]
    for c in rows:
        if c.get("n", 0) == 0:
            out.append(f"  {c['bin']:>3}      0"); continue
        out.append(f"  {c['bin']:>3} {c['n']:>6}   {c['sigma_rms']:9.4f} {c['r_rms']:9.4f} {c['r_med']:8.4f} {c['ratio_r_over_sigma']:8.3f}")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ training
def _call(head, x, k):
    s = head(x)
    s = s.reshape(x.shape[0], -1)
    assert s.shape[1] == k, f"head returned {tuple(s.shape)} for a {k}-channel target"
    return s


@torch.no_grad()
def predict(head, X, k, bs=8192):
    head.eval(); out = []
    for i in range(0, X.shape[0], bs):
        out.append(_call(head, X[i:i + bs].float(), k).float().cpu())
    return torch.cat(out, 0).numpy() if out else np.zeros((0, k))


def train_head(name, head, params, X, R, Xh, Rh, args, device, P, names, val=None):
    """Generic K&G fit: X (N,D) f16 on device, R (N,k) f32 standardised residuals on device; Xh/Rh held-out
    (reporting only). val = (Xv, Rv) is an optional slice of TRAIN episodes used only when --early_stop is set:
    the epoch with the lowest val NLL wins and its weights are restored (the heldout split never selects).
    Returns (s_train, s_heldout, s_const, per-epoch log)."""
    k = R.shape[1]; N = X.shape[0]
    Xv, Rv = val if val is not None else (None, None)
    best = (float("inf"), None, -1)
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, math.ceil(N / args.batch)); total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=args.lr * 0.01)
    g = torch.Generator().manual_seed(args.seed)
    s_const = torch.log((R ** 2).mean(0).clamp(min=1e-12))  # constant-s optimum on the train table
    nll_const_h = float(gaussian_nll(s_const[None].expand_as(Rh), Rh).sum(-1).mean()) if Rh.shape[0] else float("nan")
    P(f"[wgate] {name}: {N} train / {Xh.shape[0]} heldout frames, {n_params/1e3:.1f}k trainable params, "
      f"{total} steps ({steps_per_epoch}/epoch x {args.epochs}), const-s baseline heldout NLL {nll_const_h:.4f}")
    log = []; t0 = time.time(); step = 0
    for epoch in range(args.epochs):
        head.train(); perm = torch.randperm(N, generator=g).to(device); run = []
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            s = _call(head, X[idx].float(), k)
            loss = gaussian_nll(s, R[idx]).sum(-1).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step(); sched.step(); step += 1
            run.append(float(loss.detach()))
        sh = predict(head, Xh, k) if Xh.shape[0] else np.zeros((0, k))
        nll_h = float(gaussian_nll(torch.as_tensor(sh), Rh.cpu()).sum(-1).mean()) if Xh.shape[0] else float("nan")
        rec = {"epoch": epoch + 1, "train_nll": float(np.mean(run)), "heldout_nll": nll_h, "lr": float(sched.get_last_lr()[0])}
        if Xv is not None and Xv.shape[0]:
            rec["val_nll"] = float(gaussian_nll(torch.as_tensor(predict(head, Xv, k)), Rv.cpu()).sum(-1).mean())
            if args.early_stop and rec["val_nll"] < best[0]:
                best = (rec["val_nll"], [p.detach().clone() for p in params], epoch + 1)
        log.append(rec)
        P(f"[wgate] {name} epoch {epoch+1}/{args.epochs} train NLL {rec['train_nll']:.4f} heldout NLL {nll_h:.4f} "
          + (f"val NLL {rec['val_nll']:.4f} " if "val_nll" in rec else "") + f"lr {rec['lr']:.2e} {(time.time()-t0):.0f}s")
    if args.early_stop and best[1] is not None:
        with torch.no_grad():
            for p, b in zip(params, best[1]):
                p.copy_(b)
        P(f"[wgate] {name}: early stop -> restored epoch {best[2]} (val NLL {best[0]:.4f})")
        log.append({"early_stop_epoch": best[2], "val_nll": best[0]})
    return predict(head, X, k), predict(head, Xh, k), s_const.cpu().numpy(), log


def clock_diagnostic(tr, keys=ALL_RES_KEYS):
    """Spearman(t, r) over the TRAIN table for every residual array (NaN when absent): the 'clock' test --
    an ACCUMULATING residual (abs pose error relative to frame 0) correlates with the frame index, a
    consecutive-frame one should not."""
    out = {}
    for k in keys:
        if k in tr:
            m = np.isfinite(tr[k])
            out[f"spearman_t_{k}"] = spearman(tr["t"][m], tr[k][m]) if m.sum() >= 3 else float("nan")
    return out


def run(args, P=print):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tgt = target_spec(args.target); suffix = tgt["suffix"]
    FrameConfHead, PoseSigmaHead, standin = import_heads(allow_standin=args.allow_standin)
    os.makedirs(args.out, exist_ok=True)
    max_ep = 2 if args.smoke else args.max_episodes
    P(f"[wgate] target '{args.target}': FrameConfHead r = {' + '.join(tgt['frame'])}; PoseSigmaHead (r_t, r_R) = {tgt['pose']}; "
      f"output suffix '{suffix}'")
    tr = load_split(os.path.join(args.features, "train"), max_ep, P=P, res_keys=tgt["keys"])
    held_dir = os.path.join(args.features, "heldout"); heldout_is_train = False
    if glob.glob(os.path.join(held_dir, "*.npz")):
        he = load_split(held_dir, max_ep, P=P, res_keys=tgt["keys"])
    else:
        P(f"[wgate] WARNING: no heldout npz under {held_dir}; reporting 'heldout' metrics on the TRAIN files")
        he = tr; heldout_is_train = True
    if args.min_valid_frac > 0:
        for d in (tr, he):
            keep = d["pts_valid_frac"] >= args.min_valid_frac
            for k in list(d):
                if k != "files":
                    d[k] = d[k][keep]
        P(f"[wgate] min_valid_frac {args.min_valid_frac}: {tr['frame_feat'].shape[0]} train / {he['frame_feat'].shape[0]} heldout frames kept")
    va = None
    if args.val_frac > 0 and len(tr["files"]) >= 4:
        eps = np.unique(tr["ep"]); rng = np.random.default_rng(args.seed)
        val_eps = set(rng.choice(eps, size=max(1, int(round(len(eps) * args.val_frac))), replace=False).tolist())
        m = np.isin(tr["ep"], list(val_eps))
        va = {k: v[m] for k, v in tr.items() if k != "files"}; va["files"] = [f for f in tr["files"] if int(os.path.basename(f)[:-4]) in val_eps]
        tr = {**{k: v[~m] for k, v in tr.items() if k != "files"}, "files": [f for f in tr["files"] if int(os.path.basename(f)[:-4]) not in val_eps]}
        P(f"[wgate] val split: {len(val_eps)} train episodes ({va['frame_feat'].shape[0]} frames) held back for "
          f"{'early stopping' if args.early_stop else 'monitoring'}; {tr['frame_feat'].shape[0]} frames train")
    n_tr, n_he = tr["frame_feat"].shape[0], he["frame_feat"].shape[0]
    assert n_tr > 0, "empty train table"
    clock = clock_diagnostic(tr)
    P("[wgate] clock diagnostic, Spearman(t, r) on the train table: " + ", ".join(f"{k[11:]} {v:.3f}" for k, v in clock.items()))
    results = {"device": str(device), "standin_heads": standin, "heldout_is_train": heldout_is_train, "target": args.target,
               "target_frame_residual": " + ".join(tgt["frame"]), "target_pose_residual": list(tgt["pose"]), "clock": clock,
               "n_train_frames": int(n_tr), "n_heldout_frames": int(n_he), "n_train_episodes": len(tr["files"]),
               "n_heldout_episodes": len(he["files"]), "n_val_frames": int(va["frame_feat"].shape[0]) if va else 0, "args": vars(args)}
    kf = tgt["frame"]; kt, kR = tgt["pose"]

    def _std(name, r_tr, r_he):
        scale = float(np.median(r_tr)); scale = scale if scale > 0 else 1.0
        a, b = r_tr / scale, r_he / scale
        if args.r_clip > 0:
            a, b = np.minimum(a, args.r_clip), np.minimum(b, args.r_clip)
        P(f"[wgate] residual {name}: train median {scale:.5f} (divisor), mean {r_tr.mean():.5f}, p90 {np.quantile(r_tr, .9):.5f}, "
          f"max {r_tr.max():.4f}; standardised clip {args.r_clip or 'off'}")
        return scale, a.astype(np.float32), b.astype(np.float32)

    # ---------------- variant 1: FrameConfHead on r = r_t + r_R + r_pts (abs) or r_rel_t + r_rel_R + r_pts (rel)
    if args.heads in ("both", "frame"):
        r_tr = sum(tr[k] for k in kf); r_he = sum(he[k] for k in kf)
        scale, a, b = _std(f"frame({'+'.join(kf)})", r_tr, r_he)
        head = FrameConfHead(in_dim=FRAME_FEAT_DIM, hidden=args.hidden).to(device)
        params = [p for p in head.parameters() if p.requires_grad]
        X = torch.from_numpy(tr["frame_feat"]).to(device); Xh = torch.from_numpy(he["frame_feat"]).to(device)
        R = torch.from_numpy(a)[:, None].to(device); Rh = torch.from_numpy(b)[:, None].to(device)
        val = None
        if va is not None:
            rv = sum(va[k] for k in kf) / scale
            rv = np.minimum(rv, args.r_clip) if args.r_clip > 0 else rv
            val = (torch.from_numpy(va["frame_feat"]).to(device), torch.from_numpy(rv.astype(np.float32))[:, None].to(device))
        s_tr, s_he, s_const, log = train_head("FrameConfHead", head, params, X, R, Xh, Rh, args, device, P, ["frame"], val=val)
        log_ref = float(np.median(s_tr[:, 0]))
        met = {"train": head_metrics(s_tr, a[:, None], ["frame"], s_const, [log_ref], args.wmin),
               "heldout": head_metrics(s_he, b[:, None], ["frame"], s_const, [log_ref], args.wmin),
               "log_ref": log_ref, "res_scale": scale, "epochs": log, "target": args.target, "residual_keys": list(kf)}
        met["train"]["spearman_t_r"] = spearman(tr["t"], a)   # clock diagnostic of THIS head's residual
        met["heldout"]["spearman_t_r"] = spearman(he["t"], b)
        P(f"[wgate] FrameConfHead heldout: NLL {met['heldout']['nll']:.4f} (const {met['heldout']['nll_const']:.4f}) "
          f"Spearman(sigma, r) {met['heldout']['spearman_frame']:.3f} (train {met['train']['spearman_frame']:.3f}) "
          f"log_ref {log_ref:.3f} mean weight {met['heldout']['weight_mean_frame']:.3f} clock Spearman(t, r) train "
          f"{met['train']['spearman_t_r']:.3f}\n" + fmt_calib(met["heldout"]["calib_frame"]))
        ck = {"state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()}, "log_ref": log_ref, "wmin": args.wmin,
              "args": vars(args), "metrics": met, "res_scale": scale, "head_cfg": {"in_dim": FRAME_FEAT_DIM, "hidden": args.hidden},
              "residual": " + ".join(kf) + ", divided by res_scale (train median)", "standin": standin, "target": args.target}
        fpath = os.path.join(args.out, f"frame_conf_head{suffix}.pth")
        torch.save(ck, fpath); results["frame"] = met
        P(f"[wgate] wrote {fpath}")
        del X, Xh, R, Rh, head

    # ---------------- variant 2: PoseSigmaHead on (r_t, r_R)
    if args.heads in ("both", "pose"):
        pose_decoder, pd_src = load_pose_decoder(args.pose_decoder, args.ckpt); pose_decoder = pose_decoder.to(device)
        P(f"[wgate] pose decoder from {pd_src}: fc1 {tuple(pose_decoder.mlp.fc1.weight.shape)}")
        scale_t, at, bt = _std(f"pose r_t({kt})", tr[kt], he[kt])
        scale_R, aR, bR = _std(f"pose r_R({kR})", tr[kR], he[kR])
        head = PoseSigmaHead(pose_decoder).to(device)
        out_params = [p for n, p in head.named_parameters() if n.startswith("out.")]
        if not out_params:
            raise RuntimeError("PoseSigmaHead has no parameters named 'out.*' (contract: self.out = nn.Linear(3072, 2)); "
                               f"found {[n for n, _ in head.named_parameters()]}")
        for n, p in head.named_parameters():
            p.requires_grad_(n.startswith("out."))
        X = torch.from_numpy(tr["pose_feat"]).to(device); Xh = torch.from_numpy(he["pose_feat"]).to(device)
        R = torch.from_numpy(np.stack([at, aR], 1)).to(device); Rh = torch.from_numpy(np.stack([bt, bR], 1)).to(device)
        val = None
        if va is not None:
            rv = np.stack([va[kt] / scale_t, va[kR] / scale_R], 1)
            rv = np.minimum(rv, args.r_clip) if args.r_clip > 0 else rv
            val = (torch.from_numpy(va["pose_feat"]).to(device), torch.from_numpy(rv.astype(np.float32)).to(device))
        s_tr, s_he, s_const, log = train_head("PoseSigmaHead", head, out_params, X, R, Xh, Rh, args, device, P, ["t", "R"], val=val)
        log_ref_t, log_ref_R = float(np.median(s_tr[:, 0])), float(np.median(s_tr[:, 1]))
        met = {"train": head_metrics(s_tr, np.stack([at, aR], 1), ["t", "R"], s_const, [log_ref_t, log_ref_R], args.wmin),
               "heldout": head_metrics(s_he, np.stack([bt, bR], 1), ["t", "R"], s_const, [log_ref_t, log_ref_R], args.wmin),
               "log_ref_t": log_ref_t, "log_ref_R": log_ref_R, "res_scale_t": scale_t, "res_scale_R": scale_R, "epochs": log,
               "target": args.target, "residual_keys": [kt, kR]}
        met["train"]["spearman_t_r_t"] = spearman(tr["t"], at); met["train"]["spearman_t_r_R"] = spearman(tr["t"], aR)
        met["heldout"]["spearman_t_r_t"] = spearman(he["t"], bt); met["heldout"]["spearman_t_r_R"] = spearman(he["t"], bR)
        P(f"[wgate] PoseSigmaHead clock Spearman(t, r) train: t {met['train']['spearman_t_r_t']:.3f} R {met['train']['spearman_t_r_R']:.3f}")
        P(f"[wgate] PoseSigmaHead heldout: NLL {met['heldout']['nll']:.4f} (const {met['heldout']['nll_const']:.4f}) "
          f"Spearman t {met['heldout']['spearman_t']:.3f} R {met['heldout']['spearman_R']:.3f} comb {met['heldout']['spearman_comb']:.3f} "
          f"(train t {met['train']['spearman_t']:.3f} R {met['train']['spearman_R']:.3f}) log_ref_t {log_ref_t:.3f} log_ref_R {log_ref_R:.3f} "
          f"mean weight {met['heldout']['weight_mean_comb']:.3f}\n t:\n" + fmt_calib(met["heldout"]["calib_t"]) + "\n R:\n" + fmt_calib(met["heldout"]["calib_R"]))
        sd_out = {k: v.detach().cpu() for k, v in head.state_dict().items() if k.startswith("out.")}
        ck = {"state_dict": sd_out, "log_ref_t": log_ref_t, "log_ref_R": log_ref_R, "wmin": args.wmin, "args": vars(args),
              "metrics": met, "res_scale_t": scale_t, "res_scale_R": scale_R, "pose_decoder_src": pd_src,
              "residual": f"r_t = {kt} / res_scale_t, r_R = {kR} / res_scale_R (train medians)", "standin": standin,
              "target": args.target}
        ppath = os.path.join(args.out, f"pose_sigma_head{suffix}.pth")
        torch.save(ck, ppath); results["pose"] = met
        P(f"[wgate] wrote {ppath}")
    with open(os.path.join(args.out, f"train_wgate_metrics{suffix}.json"), "w") as f:
        json.dump(results, f, indent=1)
    return results


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=os.path.join(WGATE_ROOT, "features"), help="dir with train/ and heldout/ npz")
    ap.add_argument("--out", default=WGATE_ROOT, help="where frame_conf_head.pth / pose_sigma_head.pth go")
    ap.add_argument("--pose_decoder", default=os.path.join(WGATE_ROOT, "pose_decoder.pth"),
                    help="pose decoder state dict written by the recorder; falls back to --ckpt")
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--heads", choices=["both", "frame", "pose"], default="both")
    ap.add_argument("--target", choices=sorted(TARGETS), default="abs",
                    help="abs = finetune-loss pose residual relative to frame 0 (default, accumulates with t); "
                         "rel = consecutive-frame relative residual (res_rel_*); rel outputs get the suffix _rel")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--wmin", type=float, default=WMIN)
    ap.add_argument("--r_clip", type=float, default=0.0, help="clip the standardised residual at this many medians (0 = off)")
    ap.add_argument("--min_valid_frac", type=float, default=0.0, help="drop frames with fewer valid pixels than this fraction")
    ap.add_argument("--max_episodes", type=int, default=0, help="use only the first N npz of each split (0 = all)")
    ap.add_argument("--val_frac", type=float, default=0.0, help="fraction of TRAIN episodes held back (val NLL per epoch)")
    ap.add_argument("--early_stop", action="store_true", help="with --val_frac: keep the epoch with the lowest val NLL")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="auto")
    ap.add_argument("--allow_standin", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--smoke", action="store_true", help="2 episodes, 1 epoch")
    ap.add_argument("--selftest", action="store_true", help="CPU test on a synthetic feature table")
    return ap


def main():
    args = build_parser().parse_args()
    if args.selftest:
        return selftest()
    if args.smoke:
        args.epochs = 1
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, f"train_wgate_log{target_spec(args.target)['suffix']}.txt"), "a")
    def P(*a):
        msg = " ".join(str(x) for x in a); print(msg, flush=True); log.write(msg + "\n"); log.flush()
    P(f"[wgate] {time.strftime('%F %T')} args={vars(args)}")
    run(args, P)
    P("[wgate] done")


# ------------------------------------------------------------------------------------------------ selftest
def make_synthetic_features(root, n_train=8, n_held=2, T=64, seed=0, pose_decoder=None):
    """Synthetic npz table in the recorder's format. Features live on a low-dimensional manifold (a 32-d
    latent through a fixed random projection plus small noise, like real encoder/state features) and the
    residuals are heteroscedastic functions of that latent: log-scale linear in 4 latent dims for r_pts,
    linear in 32 dims of h = GELU(fc1(pose_feat)) for r_t / r_R, times |N(0,1)| noise, so a K&G head can
    rank them but never predict them exactly. The RELATIVE residuals (res_rel_t / res_rel_R, keyed to h dims
    64:96 / 96:128, 0 at t = 0) are generated the same way. No residual depends on t here (iid frames), so the
    clock diagnostic must come out ~0 for both targets on this table (the real abs table gives ~.5-.6)."""
    from record_wgate_features import assemble_episode, save_episode
    rng = np.random.default_rng(seed); lat = 32
    A_f = rng.standard_normal((lat, FRAME_FEAT_DIM)) / np.sqrt(lat); A_p = rng.standard_normal((lat, POSE_FEAT_DIM)) / np.sqrt(lat)
    w_f = rng.standard_normal(4) * 0.8; w_t = rng.standard_normal(32) * 0.8; w_R = rng.standard_normal(32) * 0.8
    rng_rel = np.random.default_rng(seed + 1)   # own stream: the abs arrays stay bit-identical to the pre-rel table
    w_rt = rng_rel.standard_normal(32) * 0.8; w_rR = rng_rel.standard_normal(32) * 0.8
    for split, n, off in (("train", n_train, 0), ("heldout", n_held, 10_000)):
        os.makedirs(os.path.join(root, split), exist_ok=True)
        for e in range(n):
            L_f = rng.standard_normal((T, lat)); L_p = rng.standard_normal((T, lat))
            ff = (L_f @ A_f + 0.05 * rng.standard_normal((T, FRAME_FEAT_DIM))).astype(np.float32)
            pf = (L_p @ A_p + 0.05 * rng.standard_normal((T, POSE_FEAT_DIM))).astype(np.float32)
            with torch.no_grad():
                h = pose_decoder.mlp.act(pose_decoder.mlp.fc1(torch.from_numpy(pf))).numpy()
            z_f = L_f[:, :4] @ w_f; z_t = h[:, :32] @ w_t; z_R = h[:, 32:64] @ w_R
            z_rt = h[:, 64:96] @ w_rt; z_rR = h[:, 96:128] @ w_rR
            r_pts = 0.10 * np.exp(0.6 * z_f) * np.abs(rng.standard_normal(T))
            r_t = 0.05 * np.exp(0.6 * z_t) * np.abs(rng.standard_normal(T))
            r_R = 0.02 * np.exp(0.6 * z_R) * np.abs(rng.standard_normal(T))
            r_rt = 0.03 * np.exp(0.6 * z_rt) * np.abs(rng_rel.standard_normal(T))
            r_rR = 0.01 * np.exp(0.6 * z_rR) * np.abs(rng_rel.standard_normal(T))
            r_pts[0] = r_t[0] = r_R[0] = r_rt[0] = r_rR[0] = 0.0
            res = {"res_pose_t": r_t[None].astype(np.float32), "res_pose_R": r_R[None].astype(np.float32),
                   "res_pts_self": r_pts[None].astype(np.float32), "pts_valid_frac": np.full((1, T), 0.9, np.float32),
                   "res_rel_t": r_rt[None].astype(np.float32), "res_rel_R": r_rR[None].astype(np.float32),
                   "pose_mask": np.array([True])}
            cam = rng.standard_normal((T, 7)).astype(np.float32)
            ep = assemble_episode(ff.astype(np.float16), pf.astype(np.float16), cam, res,
                                  {"scene": f"syn{e}", "first_frame": "0.png", "last_frame": f"{T-1}.png", "dataset_idx": e}, split)
            save_episode(os.path.join(root, split, f"{off + e:06d}.npz"), ep)


def selftest():
    import tempfile
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        pd = PoseDecoder(hidden_size=POSE_FEAT_DIM)
        torch.save({"pose_decoder": pd.state_dict(), "hidden_size": POSE_FEAT_DIM}, os.path.join(d, "pose_decoder.pth"))
        make_synthetic_features(os.path.join(d, "features"), n_train=64, n_held=4, T=64, seed=0, pose_decoder=pd)
        # loader: frames t>0 only, shapes/dtypes
        tr = load_split(os.path.join(d, "features", "train"))
        assert tr["frame_feat"].shape == (64 * 63, FRAME_FEAT_DIM) and tr["frame_feat"].dtype == np.float16 and tr["t"].min() == 1
        assert tr["ep"].min() == 0 and tr["ep"].max() == 63
        assert tr["res_rel_t"].shape == (64 * 63,) and np.isfinite(tr["res_rel_t"]).all() and tr["res_rel_t"].min() > 0
        # clock diagnostic: exact on a handmade table (accumulating residual -> 1, shuffled -> ~0, absent -> no key,
        # NaN frames ignored); ~0 for every array of the iid synthetic table
        rng_c = np.random.default_rng(0); tt = np.arange(400)
        clk0 = clock_diagnostic({"t": tt, "res_pose_t": (tt / 400.0) ** 2, "res_rel_t": rng_c.permutation(400).astype(float),
                                 "res_pose_R": np.where(tt % 7 == 0, np.nan, tt.astype(float))})
        assert abs(clk0["spearman_t_res_pose_t"] - 1.0) < 1e-9 and abs(clk0["spearman_t_res_rel_t"]) < 0.15 and \
            abs(clk0["spearman_t_res_pose_R"] - 1.0) < 1e-9 and set(clk0) == {"spearman_t_res_pose_t", "spearman_t_res_rel_t", "spearman_t_res_pose_R"}, clk0
        clk = clock_diagnostic(tr)
        assert set(clk) == {f"spearman_t_{k}" for k in ALL_RES_KEYS} and all(np.isfinite(v) and abs(v) < 0.1 for v in clk.values()), clk
        # a pre-rel table (npz without res_rel_*): abs target loads (NaN-filled extras), rel target names the fix
        old_dir = os.path.join(d, "features_old", "train"); os.makedirs(old_dir)
        for f in tr["files"][:2]:
            z = dict(np.load(f)); [z.pop(k) for k in REL_KEYS]; np.savez(os.path.join(old_dir, os.path.basename(f)), **z)
        old = load_split(old_dir, P=lambda *a: None)
        assert old["frame_feat"].shape[0] == 2 * 63 and np.isnan(old["res_rel_t"]).all() and np.isfinite(old["res_pose_t"]).all()
        try:
            load_split(old_dir, P=lambda *a: None, res_keys=target_spec("rel")["keys"]); raise AssertionError("expected KeyError")
        except KeyError as e:
            assert "--fill_rel" in str(e), e
        assert target_spec("abs")["suffix"] == "" and target_spec("rel")["suffix"] == "_rel" and target_spec("rel")["keys"] == REL_KEYS + ("res_pts_self",)
        # a non-finite residual frame and a pose_mask=False episode are dropped
        z = dict(np.load(os.path.join(d, "features", "train", "000001.npz"))); z["res_pose_t"][5] = np.nan
        np.savez(os.path.join(d, "features", "train", "000001.npz"), **z)
        z = dict(np.load(os.path.join(d, "features", "train", "000002.npz"))); z["pose_mask"] = np.array(False)
        np.savez(os.path.join(d, "features", "train", "000002.npz"), **z)
        tr2 = load_split(os.path.join(d, "features", "train"))
        assert tr2["frame_feat"].shape[0] == 63 * 63 - 1, tr2["frame_feat"].shape
        # pose decoder loading from the small file
        pd2, src = load_pose_decoder(os.path.join(d, "pose_decoder.pth"), ckpt_fallback="/nonexistent")
        assert torch.equal(pd2.mlp.fc1.weight, pd.mlp.fc1.weight) and not pd2.mlp.fc1.weight.requires_grad
        # maths
        s = torch.full((1000, 1), math.log(4.0)); r = torch.randn(1000, 1) * 2
        assert abs(float(gaussian_nll(s, r).mean()) - (1 + math.log(4.0))) < 0.15
        assert abs(spearman(np.arange(50), np.arange(50) ** 2) - 1.0) < 1e-9 and abs(spearman(np.arange(50), -np.arange(50)) + 1.0) < 1e-9
        rows = calibration_table(np.log(np.linspace(0.5, 2.0, 500) ** 2), np.linspace(0.5, 2.0, 500))
        assert len(rows) == 5 and all(abs(c["ratio_r_over_sigma"] - 1.0) < 1e-6 for c in rows), rows
        assert np.allclose(sigma_weight(np.array([0.0, -2.0, 1.0, 2.0]), 0.0, 0.5), [1.0, 1.0, np.exp(-0.5), 0.5])  # clip at wmin
        # full run on the synthetic table (small batches so the heads get enough steps on CPU)
        args = build_parser().parse_args(["--features", os.path.join(d, "features"), "--out", os.path.join(d, "out"),
                                          "--pose_decoder", os.path.join(d, "pose_decoder.pth"), "--epochs", "20",
                                          "--batch", "256", "--lr", "1e-3", "--device", "cpu", "--allow_standin"])
        res = run(args, P=lambda *a: None)
        fh = res["frame"]["heldout"]; ph = res["pose"]["heldout"]
        print(f"selftest frame head: heldout NLL {fh['nll']:.3f} vs const {fh['nll_const']:.3f}, Spearman {fh['spearman_frame']:.3f}")
        print(f"selftest pose head : heldout NLL {ph['nll']:.3f} vs const {ph['nll_const']:.3f}, Spearman t {ph['spearman_t']:.3f} "
              f"R {ph['spearman_R']:.3f} comb {ph['spearman_comb']:.3f}")
        assert fh["nll"] < fh["nll_const"] and fh["spearman_frame"] > 0.2, fh
        assert ph["nll"] < ph["nll_const"] and ph["spearman_t"] > 0.2 and ph["spearman_R"] > 0.2, ph
        assert res["target"] == "abs" and res["frame"]["target"] == "abs" and set(res["clock"]) == set(clk)
        for k in ("spearman_t_r_t", "spearman_t_r_R"):   # per-head clock diagnostics present, finite, ~0 on iid frames
            assert np.isfinite(res["pose"]["train"][k]) and abs(res["pose"]["train"][k]) < 0.1, (k, res["pose"]["train"][k])
        assert np.isfinite(res["frame"]["train"]["spearman_t_r"]) and abs(res["frame"]["train"]["spearman_t_r"]) < 0.1
        # --target rel: same pipeline on res_rel_*, suffixed outputs, abs files untouched
        args_r = build_parser().parse_args(["--features", os.path.join(d, "features"), "--out", os.path.join(d, "out"),
                                            "--pose_decoder", os.path.join(d, "pose_decoder.pth"), "--epochs", "20",
                                            "--batch", "256", "--lr", "1e-3", "--device", "cpu", "--allow_standin", "--target", "rel"])
        res_r = run(args_r, P=lambda *a: None)
        fr = res_r["frame"]["heldout"]; pr = res_r["pose"]["heldout"]
        print(f"selftest REL frame head: heldout NLL {fr['nll']:.3f} vs const {fr['nll_const']:.3f}, Spearman {fr['spearman_frame']:.3f}")
        print(f"selftest REL pose head : heldout NLL {pr['nll']:.3f} vs const {pr['nll_const']:.3f}, Spearman t {pr['spearman_t']:.3f} "
              f"R {pr['spearman_R']:.3f} comb {pr['spearman_comb']:.3f}; clock Spearman(t, r_t) train rel {res_r['pose']['train']['spearman_t_r_t']:.3f} "
              f"abs {res['pose']['train']['spearman_t_r_t']:.3f} (iid synthetic frames: both ~0)")
        assert fr["nll"] < fr["nll_const"] and fr["spearman_frame"] > 0.2, fr
        assert pr["nll"] < pr["nll_const"] and pr["spearman_t"] > 0.2 and pr["spearman_R"] > 0.2, pr
        assert res_r["target"] == "rel" and abs(res_r["pose"]["train"]["spearman_t_r_t"]) < 0.1 and res_r["pose"]["residual_keys"] == ["res_rel_t", "res_rel_R"]
        assert res_r["frame"]["residual_keys"] == ["res_rel_t", "res_rel_R", "res_pts_self"]
        for fn in ("frame_conf_head_rel.pth", "pose_sigma_head_rel.pth", "train_wgate_metrics_rel.json"):
            assert os.path.isfile(os.path.join(d, "out", fn)), fn
        ckfr = torch.load(os.path.join(d, "out", "frame_conf_head_rel.pth"), map_location="cpu", weights_only=False)
        ckpr = torch.load(os.path.join(d, "out", "pose_sigma_head_rel.pth"), map_location="cpu", weights_only=False)
        assert ckfr["target"] == "rel" and ckpr["target"] == "rel" and ckfr["metrics"]["target"] == "rel"
        assert set(ckpr["state_dict"]) == {"out.weight", "out.bias"} and abs(ckfr["res_scale"] - float(np.median(tr2["res_rel_t"] + tr2["res_rel_R"] + tr2["res_pts_self"]))) < 1e-5
        assert json.load(open(os.path.join(d, "out", "train_wgate_metrics_rel.json")))["target"] == "rel"
        assert json.load(open(os.path.join(d, "out", "train_wgate_metrics.json")))["target"] == "abs"   # abs run's file untouched
        # checkpoints: exactly the contract's keys, out-layer-only pose state dict, weights recomputable
        ckf = torch.load(os.path.join(d, "out", "frame_conf_head.pth"), map_location="cpu", weights_only=False)
        ckp = torch.load(os.path.join(d, "out", "pose_sigma_head.pth"), map_location="cpu", weights_only=False)
        for k in ("state_dict", "log_ref", "wmin", "args", "metrics", "res_scale", "target"):
            assert k in ckf, k
        for k in ("state_dict", "log_ref_t", "log_ref_R", "wmin", "args", "metrics", "res_scale_t", "res_scale_R", "target"):
            assert k in ckp, k
        assert ckf["target"] == "abs" and ckp["target"] == "abs" and ckf["residual"].startswith("res_pose_t + res_pose_R + res_pts_self")
        assert set(ckp["state_dict"]) == {"out.weight", "out.bias"} and ckp["state_dict"]["out.weight"].shape == (2, 3072), list(ckp["state_dict"])
        assert ckf["wmin"] == 0.5 and ckp["wmin"] == 0.5 and ckf["res_scale"] > 0
        FrameConfHead, PoseSigmaHead, _ = import_heads(allow_standin=True)
        head = FrameConfHead(in_dim=FRAME_FEAT_DIM, hidden=256); head.load_state_dict(ckf["state_dict"]); head.eval()
        he = load_split(os.path.join(d, "features", "heldout"))
        s = predict(head, torch.from_numpy(he["frame_feat"]), 1)[:, 0]
        assert abs(float(np.median(predict(head, torch.from_numpy(tr2["frame_feat"]), 1))) - ckf["log_ref"]) < 1e-4
        w = sigma_weight(s, ckf["log_ref"], ckf["wmin"]); assert w.min() >= 0.5 and w.max() <= 1.0 and 0.5 < w.mean() < 1.0
        assert os.path.isfile(os.path.join(d, "out", "train_wgate_metrics.json"))
        # --val_frac/--early_stop path: val split carved from TRAIN by episode, best epoch restored
        args2 = build_parser().parse_args(["--features", os.path.join(d, "features"), "--out", os.path.join(d, "out2"),
                                           "--pose_decoder", os.path.join(d, "pose_decoder.pth"), "--epochs", "6", "--batch", "256",
                                           "--device", "cpu", "--heads", "frame", "--val_frac", "0.25", "--early_stop"])
        res2 = run(args2, P=lambda *a: None)
        # 63 usable episodes (one dropped), 16 in val; the NaN frame sits in whichever side got episode 1
        assert res2["n_val_frames"] in (16 * 63, 16 * 63 - 1) and res2["n_val_frames"] + res2["n_train_frames"] == 63 * 63 - 1, \
            (res2["n_val_frames"], res2["n_train_frames"])
        ep_log = res2["frame"]["epochs"]; assert "early_stop_epoch" in ep_log[-1] and all("val_nll" in e for e in ep_log[:-1]), ep_log
    print("train_wgate_heads selftest OK")


if __name__ == "__main__":
    main()
