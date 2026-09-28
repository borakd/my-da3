#!/usr/bin/env python
"""Record per-frame write-gate features and GT residuals from the FROZEN CUT3R (WRITE_GATE_VARIANTS.md,
"Recording + training", Component B).

For every episode (64 consecutive frames of one scene, `train_unc_gate.build_loader`'s DL3DV_Multi dataset) the
frozen `augfull_lr1e5` model is rolled out causally with NO gates (plain model, no grad) while the model's
recording hook (`model.wgate_record = []`, Component A) appends one dict per decoder step:
    {t, frame_feat (1796,), pose_feat (768,), camera_pose (7,)}
`frame_feat` is `dust3r.wgate.heads.frame_features(new_state_feat, state_feat, global_img_feat)` = the input of
the variant-1 FrameConfHead; `pose_feat` is the pose token after the last decoder block = the input of the pose
decoder (variant-2 PoseSigmaHead recomputes h = GELU(fc1(pose_feat)) from it).

Precision: the rollout runs in fp32 by default (autocast OFF), which is exactly what the eval worker's
`inference()` path does (`dust3r/inference.py`: `autocast(enabled=False)`), so the heads are trained on the
same feature distribution they are served at eval. `--bf16` records under bf16 autocast instead (the contract's
original wording; ~1.5-2x faster, but a small train/serve shift).

After the rollout the finetune criterion's own normalisation is applied,
    criterion = Regr3DPoseBatchList(L21, norm_mode='?avg_dis');  criterion.get_all_pts3d(views, preds, camera1)
and per frame t we derive
    res_pose_t[t]   = || t_pred - t_gt ||           (absT part of the normalised absT_quaR encodings, GT in camera 1)
    res_pose_R[t]   = || q_pred - q_gt ||           (quaternion part)
    res_pts_self[t] = mean_{valid px} || pr_pts_self - gt_pts_self ||   (L21 on the normalised self-view point maps)
exactly as `compute_pose_loss` / the self-view point term of the finetune loss define them (before their mean over
frames), and, as the alternative NON-accumulating target (WRITE_GATE_VARIANTS.md, "the literal residual is a clock"),
    res_rel_t[t]    = || t_rel_pred - t_rel_gt ||   (t >= 1; 0 at t = 0)
    res_rel_R[t]    = || q_rel_pred - q_rel_gt ||
where (t_rel, q_rel) = dust3r.losses.relative_pose_absT_quatR(pose[t-1], pose[t]) is the CONSECUTIVE-frame relative
encoding (frame t-1 -> t, exactly the function compute_relative_pose_loss uses) of the SAME normalised gt_poses /
pr_poses lists, so the translation scale is the loss's. One `.npz` per episode is written to
    <root>/features/<split>/<episode_idx>.npz
with arrays (T = 64):
    frame_feat (T,1796) f16, pose_feat (T,768) f16, camera_pose (T,7) f32,
    res_pose_t (T,) f32, res_pose_R (T,) f32, res_pts_self (T,) f32, res_rel_t (T,) f32, res_rel_R (T,) f32,
  + extras: t (T,) i32, pts_valid_frac (T,) f32 (fraction of valid pixels; res_pts_self is 0 where it is 0),
    pose_mask () bool (criterion's per-sequence pose validity), scene (str), first_frame / last_frame (str),
    dataset_idx () i64 (sampler index), split (str), precision (str: 'fp32' | 'bf16').

Episode selection (`select_episodes`): the dataset exposes every frame with >= 64 frames after it as a window
start (8.79M starts for TRAIN_ROOT), so a plain "first N of the seeded permutation" gives windows that partly
overlap within a scene. The recorder therefore walks the seeded order (`CustomRandomSampler.set_epoch(seed)`,
the very permutation the training loader would use) and keeps the first N windows whose frame range does not
intersect an already kept window of the same scene (`--allow_overlap` disables the check and reproduces the
plain order). The kept list is written to `<root>/features/<split>/episodes.json` BEFORE recording, and
`<episode_idx>` is the position in that list. Ranks (accelerate processes, or `--shard i/n`) record the
episodes with `episode_idx % n_proc == rank`, each through its own fixed-index DataLoader; no
`accelerator.prepare(loader)`, so accelerate's DataLoaderShard (whose `__iter__` re-applies
`set_epoch(self.iteration)` and would silently override `--seed`) is never in the loop. Episodes whose npz
already exists are not even loaded on a re-run (`--overwrite` to redo them; `--fill_rel` re-records ONLY the
episodes whose npz lacks the res_rel_* arrays -- a table recorded before they existed -- and rewrites the whole
npz, every array recomputed from the fresh rollout; the rel arrays cannot be back-filled without the rollout
because the normalised GT/pred encodings are not stored). 64 held-out episodes come from the
test-split loader (`TEST_ROOT`) under `<split>=heldout` (disjoint scenes) for calibration reporting.

Also writes <root>/pose_decoder.pth (the checkpoint's `downstream_head.pose_head` state dict, ~10 MB) on rank 0
so the head trainer does not need the 3 GB checkpoint.

  accelerate launch --multi_gpu --num_processes 4 record_wgate_features.py --episodes 4000 --heldout 64
  python record_wgate_features.py --shard 0/4 ...   # same episode numbering, one process per shard
  python record_wgate_features.py --smoke          # 2 train + 2 heldout episodes
  python record_wgate_features.py --selftest       # CPU: residual derivation, npz assembly, episode selection
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import dust3r.utils.path_to_croco  # noqa: F401
from dust3r.losses import Regr3DPoseBatchList, L21, relative_pose_absT_quatR
from train_unc_gate import build_loader, to_device, CKPT_DEFAULT, TRAIN_ROOT, TEST_ROOT

WGATE_ROOT = "/gpfs/scratch/etur59/koc821022/checkpoints/wgate"
FRAME_FEAT_DIM, POSE_FEAT_DIM, POSE_DIM = 1796, 768, 7
PRED_KEYS = ("pts3d_in_self_view", "pts3d_in_other_view", "conf_self", "conf", "camera_pose")
RES_KEYS = ("res_pose_t", "res_pose_R", "res_pts_self", "pts_valid_frac")
REL_KEYS = ("res_rel_t", "res_rel_R")   # consecutive-frame (t-1 -> t) relative pose residuals, 0 at t = 0


def require_wgate():
    """Component A's package must be importable: the recording hook lives in model.py and its feature
    function in dust3r.wgate.heads. Fail with a message that says which side is missing."""
    try:
        from dust3r.wgate import heads  # noqa: F401
    except ImportError as e:
        raise ImportError("dust3r.wgate.heads is missing (Component A of WRITE_GATE_VARIANTS.md not landed?): "
                          f"{e}") from e
    return heads


def make_criterion():
    """The finetune loss's geometry/normalisation (`?avg_dis`: predictions normalised by their own factor on
    non-metric data); only get_all_pts3d is used here, so the L21 reduction mode is irrelevant."""
    return Regr3DPoseBatchList(L21, norm_mode="?avg_dis")


@torch.no_grad()
def rollout(model, batch):
    """Causal 64-frame rollout of the frozen model (train_conf_gate.run_batch without grad/chunking).
    Returns the per-frame head outputs needed by the criterion, detached and in fp32."""
    (feat, pos, shape), (init_state_feat, init_mem, state_feat, state_pos, mem) = model._forward_encoder(batch)
    preds = []
    for t in range(len(batch)):
        res_group, (state_feat, mem) = model._forward_decoder_group_step(
            views=batch, view_indices=[t], feat_group=[feat[t]], pos_group=[pos[t]], shape_group=[shape[t]],
            init_state_feat=init_state_feat, init_mem=init_mem, state_feat=state_feat, state_pos=state_pos, mem=mem)
        res = res_group[0]
        preds.append({k: res[k].detach().float() for k in PRED_KEYS})
    return preds


@torch.no_grad()
def residuals(criterion, views, preds, camera1):
    """Per-frame GT residuals as the finetune loss defines them, from criterion.get_all_pts3d.

    Returns a dict of (B,T) float32 CPU tensors: res_pose_t, res_pose_R, res_pts_self, pts_valid_frac,
    res_rel_t, res_rel_R, plus pose_mask (B,) bool. res_pts_self is 0 (and pts_valid_frac 0) on frames with no
    valid pixel. res_rel_* are the norms of the differences between the GT and predicted consecutive-frame
    RELATIVE encodings (relative_pose_absT_quatR on the same normalised gt_poses / pr_poses lists, so the
    translation scale is the loss's); 0 at t = 0 (no previous frame). A constant absolute offset of every
    predicted pose cancels in the relative encoding, and so does a global quaternion sign flip.
    Runs with autocast OFF: the norm factors and sub-unit residuals must not be rounded in bf16."""
    dev_type = "cuda" if views[0]["camera_pose"].is_cuda else "cpu"
    with torch.autocast(device_type=dev_type, enabled=False):
        (gt_pts_self, _gt_pts_cross, pr_pts_self, _pr_pts_cross, gt_poses, pr_poses, valids, _skys, pose_masks,
         _mon) = criterion.get_all_pts3d(views, preds, camera1=camera1)
        rt, rR, rp, vf = [], [], [], []
        for t in range(len(views)):
            gt_t, gt_q = gt_poses[t]; pr_t, pr_q = pr_poses[t]
            rt.append(torch.norm(pr_t.float() - gt_t.float(), dim=-1))
            rR.append(torch.norm(pr_q.float() - gt_q.float(), dim=-1))
            valid = valids[t].bool()  # (B,H,W)
            dist = torch.norm(pr_pts_self[t].float() - gt_pts_self[t].float(), dim=-1)  # (B,H,W), the L21 distance
            n = valid.flatten(1).sum(1).float()
            s = torch.where(valid, dist, torch.zeros_like(dist)).flatten(1).sum(1)
            rp.append(torch.where(n > 0, s / n.clamp(min=1), torch.zeros_like(s)))
            vf.append(n / float(valid[0].numel()))
        rrt, rrR = [torch.zeros_like(rt[0])], [torch.zeros_like(rR[0])]   # t = 0: no previous frame
        for t in range(1, len(views)):
            g0_t, g0_q = gt_poses[t - 1]; g1_t, g1_q = gt_poses[t]
            p0_t, p0_q = pr_poses[t - 1]; p1_t, p1_q = pr_poses[t]
            g_rel_t, g_rel_q = relative_pose_absT_quatR(g0_t.float(), g0_q.float(), g1_t.float(), g1_q.float())
            p_rel_t, p_rel_q = relative_pose_absT_quatR(p0_t.float(), p0_q.float(), p1_t.float(), p1_q.float())
            rrt.append(torch.norm(p_rel_t - g_rel_t, dim=-1))
            rrR.append(torch.norm(p_rel_q - g_rel_q, dim=-1))
    out = {"res_pose_t": torch.stack(rt, 1), "res_pose_R": torch.stack(rR, 1), "res_pts_self": torch.stack(rp, 1),
           "pts_valid_frac": torch.stack(vf, 1), "res_rel_t": torch.stack(rrt, 1), "res_rel_R": torch.stack(rrR, 1)}
    out = {k: v.float().cpu() for k, v in out.items()}
    out["pose_mask"] = pose_masks.reshape(-1).bool().cpu()
    return out


def _vec(x, n, name):
    a = torch.as_tensor(x).detach().float().cpu().numpy().reshape(-1)
    assert a.size == n, f"wgate_record[{name}] has {a.size} values, expected {n}"
    return a


def collect_record(record, T):
    """Validate and stack the model's wgate_record list for one episode (batch size 1).
    Returns frame_feat (T,1796) f16, pose_feat (T,768) f16, rec_pose (T,7) f32 ordered by t."""
    if not isinstance(record, list) or len(record) == 0:
        raise RuntimeError("model.wgate_record stayed empty: the recording hook in model.py (Component A) is not "
                           "active. Set model.wgate_record = [] before the rollout and make sure model.py appends.")
    assert len(record) == T, f"wgate_record has {len(record)} entries for a {T}-frame episode"
    ts = [int(r["t"]) for r in record]
    assert sorted(ts) == list(range(T)), f"wgate_record t values are {ts}, expected 0..{T-1}"
    order = np.argsort(ts)
    ff = np.stack([_vec(record[i]["frame_feat"], FRAME_FEAT_DIM, "frame_feat") for i in order]).astype(np.float16)
    pf = np.stack([_vec(record[i]["pose_feat"], POSE_FEAT_DIM, "pose_feat") for i in order]).astype(np.float16)
    cp = np.stack([_vec(record[i]["camera_pose"], POSE_DIM, "camera_pose") for i in order]).astype(np.float32)
    assert np.isfinite(ff).all() and np.isfinite(pf).all(), "non-finite values in the recorded features"
    return ff, pf, cp


def episode_meta(batch):
    """Scene name and frame labels of the first batch element (DL3DV_Multi label = '<scene>_<rgb file>')."""
    def _lab(v):
        lab = v.get("label", [""])
        return str(lab[0] if isinstance(lab, (list, tuple)) else lab)
    first, last = _lab(batch[0]), _lab(batch[-1])
    scene = first.rsplit("_", 1)[0] if "_" in first else first
    idx = batch[0].get("idx", None)
    try:
        dataset_idx = int(idx[0][0]) if isinstance(idx, (list, tuple)) else int(idx)
    except Exception:
        dataset_idx = -1
    return {"scene": scene, "first_frame": first.rsplit("_", 1)[-1], "last_frame": last.rsplit("_", 1)[-1],
            "dataset_idx": dataset_idx}


def assemble_episode(frame_feat, pose_feat, camera_pose, res, meta, split, b=0, precision="fp32"):
    """Arrays for one npz: features (f16), the model's camera_pose (f32) and residuals (f32) for batch item b."""
    T = frame_feat.shape[0]
    ep = {"frame_feat": frame_feat.astype(np.float16), "pose_feat": pose_feat.astype(np.float16),
          "camera_pose": camera_pose.astype(np.float32), "t": np.arange(T, dtype=np.int32),
          "split": np.array(split), "pose_mask": np.array(bool(res["pose_mask"][b])), "precision": np.array(precision)}
    for k in RES_KEYS + REL_KEYS:   # a re-recorded npz is rewritten whole, so existing arrays are overwritten
        a = np.asarray(res[k][b], dtype=np.float32).reshape(-1)
        assert a.shape == (T,), (k, a.shape, T)
        ep[k] = a
    assert float(ep["res_rel_t"][0]) == 0.0 and float(ep["res_rel_R"][0]) == 0.0, "rel residual must be 0 at t = 0"
    ep["scene"] = np.array(meta.get("scene", "")); ep["first_frame"] = np.array(meta.get("first_frame", ""))
    ep["last_frame"] = np.array(meta.get("last_frame", "")); ep["dataset_idx"] = np.array(int(meta.get("dataset_idx", -1)))
    return ep


def save_episode(path, ep):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **ep); os.replace(tmp, path)


def npz_lacks(path, keys=REL_KEYS):
    """True when the npz at `path` is unreadable or misses any of `keys` (a table recorded before res_rel_*)."""
    try:
        with np.load(path) as z:
            return any(k not in z for k in keys)
    except Exception:
        return True


def save_pose_decoder(model, path):
    """Store the frozen pose decoder (fc1/act/fc2 of downstream_head.pose_head) for the head trainer."""
    pd = model.downstream_head.pose_head
    sd = {k: v.detach().cpu() for k, v in pd.state_dict().items()}
    torch.save({"pose_decoder": sd, "hidden_size": int(pd.mlp.fc1.in_features),
                "mlp_hidden": int(pd.mlp.fc1.out_features), "src": "record_wgate_features.py"}, path)


# ----------------------------------------------------------------------------------- episode selection
def window_of(ds, k, num_views):
    """(scene index, first position, last position) inside the scene of dataset index k under
    force_consecutive_frame_sampling (window = the num_views frames starting at start_img_ids[k])."""
    start_id = int(ds.start_img_ids[k]); sc = int(ds.sceneids[start_id])
    pos = start_id - int(ds.scene_img_list[sc][0])
    assert pos + num_views <= len(ds.scene_img_list[sc]), (k, sc, pos, num_views, len(ds.scene_img_list[sc]))
    return sc, pos, pos + num_views - 1


def select_episodes(ds, sampler, n, num_views, seed, distinct=True):
    """The first n episodes of the seeded sampler order, optionally de-overlapped per scene.

    `sampler` is the dataset's CustomRandomSampler (`loader.batch_sampler.sampler`); `set_epoch(seed)` makes it
    draw the permutation for rng seed `seed + 788`, i.e. exactly the order the training loader would iterate at
    that epoch, on every rank alike. With `distinct=True` a window whose [first, last] frame range intersects an
    already kept window of the same scene is skipped, so no frame is recorded twice. Returns
    (episodes, stats): episodes = list of dicts {g (= episode_idx), dataset_idx, feat_idx, nview, scene,
    start_pos, first_frame, last_frame} in order, stats = selection counters for the manifest."""
    sampler.set_epoch(seed)
    kept, taken, n_cand, n_skip = [], {}, 0, 0
    for tup in sampler:
        n_cand += 1
        k, feat_idx, nview = (int(x) for x in tup)
        assert nview == num_views, f"sampler drew nview {nview}, expected the fixed length {num_views}"
        sc, lo, hi = window_of(ds, k, num_views)
        if distinct and any(lo <= b and a <= hi for a, b in taken.get(sc, ())):
            n_skip += 1
            continue
        taken.setdefault(sc, []).append((lo, hi))
        start_id = int(ds.start_img_ids[k])
        kept.append({"g": len(kept), "dataset_idx": k, "feat_idx": feat_idx, "nview": nview,
                     "scene": str(ds.scenes[sc]), "start_pos": lo, "first_frame": str(ds.images[start_id]),
                     "last_frame": str(ds.images[start_id + num_views - 1])})
        if len(kept) >= n:
            break
    stats = {"requested": int(n), "kept": len(kept), "candidates": n_cand, "overlap_skipped": n_skip,
             "seed": int(seed), "sampler_epoch": sampler.epoch, "distinct_scenes": bool(distinct),
             "dataset_len": len(ds), "n_scenes": len(taken), "num_views": int(num_views)}
    return kept, stats


class FixedBatchSampler:
    """torch DataLoader batch_sampler that yields one-episode batches [(dataset_idx, feat_idx, nview)] in a
    fixed order (the dataset's __getitem__ takes exactly this tuple, as CustomRandomSampler yields it)."""
    def __init__(self, tuples):
        self.tuples = [tuple(int(x) for x in t) for t in tuples]

    def __iter__(self):
        for t in self.tuples:
            yield [t]

    def __len__(self):
        return len(self.tuples)


def make_fixed_loader(ds, episodes, num_workers):
    tuples = [(e["dataset_idx"], e["feat_idx"], e["nview"]) for e in episodes]
    return torch.utils.data.DataLoader(ds, batch_sampler=FixedBatchSampler(tuples), num_workers=num_workers,
                                       pin_memory=torch.cuda.is_available())


def my_episodes(episodes, rank, n_proc):
    """Episode g goes to rank g % n_proc (the same round-robin layout accelerate's BatchSamplerShard uses)."""
    return [e for e in episodes if e["g"] % n_proc == rank]


def record_split(model, criterion, ds, sampler, split, n_episodes, rank, n_proc, device, out_dir, args, P):
    """Select the seeded episode list (identically on every rank), write it, then record this rank's share
    through a fixed-index DataLoader. Returns the number of npz written by this rank."""
    os.makedirs(out_dir, exist_ok=True)
    episodes, stats = select_episodes(ds, sampler, n_episodes, args.num_views, args.seed, distinct=not args.allow_overlap)
    if stats["kept"] < n_episodes:
        P(f"[record] WARNING {split}: only {stats['kept']} non-overlapping episodes available (requested {n_episodes})")
    if rank == 0:
        with open(os.path.join(out_dir, "episodes.json"), "w") as f:
            json.dump({"stats": stats, "episodes": episodes}, f, indent=0)
    mine = my_episodes(episodes, rank, n_proc)
    def _todo(e):
        path = os.path.join(out_dir, f"{e['g']:06d}.npz")
        return args.overwrite or not os.path.isfile(path) or (args.fill_rel and npz_lacks(path, REL_KEYS))
    todo = [e for e in mine if _todo(e)]
    n_skip = len(mine) - len(todo)
    n_fill = sum(1 for e in todo if os.path.isfile(os.path.join(out_dir, f"{e['g']:06d}.npz"))) if args.fill_rel else 0
    P(f"[record] {split} rank {rank}/{n_proc}: {stats['kept']} episodes selected from {stats['candidates']} candidates "
      f"({stats['overlap_skipped']} overlapping skipped, {stats['n_scenes']} scenes, seed {args.seed}); "
      f"{len(mine)} on this rank, {n_skip} already on disk, {len(todo)} to record"
      + (f" ({n_fill} re-recorded for the missing res_rel_* arrays)" if args.fill_rel else ""))
    precision = "bf16" if (args.bf16 and device.type == "cuda") else "fp32"
    loader = make_fixed_loader(ds, todo, args.num_workers)
    n_done = 0; t0 = time.time(); warned_pose = False
    for e, batch in zip(todo, loader):
        g = e["g"]; path = os.path.join(out_dir, f"{g:06d}.npz")
        batch = to_device(batch, device)
        assert batch[0]["img"].shape[0] == 1, "record with batch_size 1 (one npz per episode)"
        meta = episode_meta(batch)
        assert meta["scene"] == e["scene"] and meta["first_frame"] == e["first_frame"], \
            f"episode list / loader mismatch at g={g}: list {e['scene']}/{e['first_frame']} vs batch {meta}"
        model.wgate_record = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(precision == "bf16")):
            preds = rollout(model, batch)
        record = model.wgate_record; model.wgate_record = None
        ff, pf, rec_pose = collect_record(record, len(batch))
        res = residuals(criterion, batch, preds, camera1=batch[0]["camera_pose"])
        cam = torch.stack([p["camera_pose"][0] for p in preds]).cpu().numpy().astype(np.float32)
        if not warned_pose and not np.allclose(cam, rec_pose, atol=2e-2, rtol=2e-2):
            P(f"[record] WARNING ep {g}: wgate_record camera_pose differs from the head output "
              f"(max |d| {np.abs(cam - rec_pose).max():.4f}); the npz stores the head output")
            warned_pose = True
        ep = assemble_episode(ff, pf, cam, res, meta, split, precision=precision)
        save_episode(path, ep); n_done += 1
        if n_done % 20 == 0 or n_done <= 2:
            P(f"[record] {split} rank {rank}: {n_done}/{len(todo)} written (+{n_skip} skipped) last ep {g} '{ep['scene']}' "
              f"r_t {ep['res_pose_t'][1:].mean():.4f} r_R {ep['res_pose_R'][1:].mean():.4f} "
              f"r_pts {ep['res_pts_self'][1:].mean():.4f} rel_t {ep['res_rel_t'][1:].mean():.4f} "
              f"rel_R {ep['res_rel_R'][1:].mean():.4f} {(time.time()-t0)/60:.1f} min")
    P(f"[record] {split} rank {rank} done: {n_done} written, {n_skip} skipped, {(time.time()-t0)/60:.1f} min ({precision})")
    return n_done, stats


def parse_shard(s):
    """'i/n' -> (i, n) with 0 <= i < n."""
    i, n = (int(x) for x in s.split("/"))
    assert 0 <= i < n, f"--shard {s}: need 0 <= i < n"
    return i, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--root", default=WGATE_ROOT, help="<root>/features/<split>/<idx>.npz and <root>/pose_decoder.pth")
    ap.add_argument("--episodes", type=int, default=4000, help="first N (non-overlapping) episodes of the seeded train order")
    ap.add_argument("--heldout", type=int, default=64, help="first N episodes of the seeded test order -> split 'heldout'")
    ap.add_argument("--num_views", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="sampler epoch (permutation seed) and torch seed")
    ap.add_argument("--shard", default=None, help="'i/n': record episodes g with g %% n == i (instead of accelerate's ranks)")
    ap.add_argument("--allow_overlap", action="store_true", help="do not skip windows overlapping a kept window of the same scene")
    ap.add_argument("--overwrite", action="store_true", help="re-record episodes whose npz exists")
    ap.add_argument("--fill_rel", action="store_true",
                    help="also re-record episodes whose npz exists but lacks the res_rel_t / res_rel_R arrays (whole npz rewritten)")
    ap.add_argument("--bf16", action="store_true", help="rollout under bf16 autocast (default fp32 = the eval worker's inference() path)")
    ap.add_argument("--fp32", action="store_true", help="(default; kept for compatibility) rollout in fp32, autocast off")
    ap.add_argument("--skip_train", action="store_true"); ap.add_argument("--skip_heldout", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 train + 2 heldout episodes")
    ap.add_argument("--selftest", action="store_true", help="CPU unit test of the residual/npz/selection code on synthetic data")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.bf16 and args.fp32:
        ap.error("--bf16 and --fp32 are mutually exclusive")
    if args.smoke:
        args.episodes, args.heldout = 2, 2
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs
    from datetime import timedelta
    from dust3r.model import ARCroco3DStereo
    require_wgate()
    torch.manual_seed(args.seed)
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=3))])
    device = accelerator.device
    if args.shard is not None:
        assert accelerator.num_processes == 1, "--shard i/n is for single-process launches; under accelerate the ranks shard"
        rank, n_proc = parse_shard(args.shard)
    else:
        rank, n_proc = accelerator.process_index, accelerator.num_processes
    is_main = rank == 0
    os.makedirs(args.root, exist_ok=True)
    log = open(os.path.join(args.root, "record_log.txt"), "a") if is_main else None
    def P(*a):
        msg = " ".join(str(x) for x in a); print(f"[rank {rank}] {msg}", flush=True)
        if log is not None:
            log.write(msg + "\n"); log.flush()
    P(f"[record] {time.strftime('%F %T')} rank {rank}/{n_proc} args={vars(args)}")
    model = ARCroco3DStereo.from_pretrained(args.ckpt).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if getattr(model, "pose_gru", None) is not None:
        model.pose_gru = None
    if not hasattr(model, "_wgate_record_step"):
        raise RuntimeError("model.py has no _wgate_record_step hook (Component A of WRITE_GATE_VARIANTS.md not landed?)")
    if is_main:
        save_pose_decoder(model, os.path.join(args.root, "pose_decoder.pth"))
    criterion = make_criterion().to(device)
    splits = []
    if not args.skip_train:
        splits.append(("train", TRAIN_ROOT, args.episodes))
    if not args.skip_heldout:
        splits.append(("heldout", TEST_ROOT, args.heldout))
    all_stats = {}
    for split, root, n in splits:
        # build_loader only to get the dataset + its CustomRandomSampler (the seeded order); it is never iterated,
        # the episodes are read through make_fixed_loader instead.
        base_loader = build_loader(root, args.num_views, 1, 0, accelerator)
        ds, sampler = base_loader.dataset, getattr(base_loader.batch_sampler, "sampler", None)
        assert sampler is not None and hasattr(sampler, "set_epoch"), type(base_loader.batch_sampler)
        for attr in ("start_img_ids", "sceneids", "scene_img_list", "scenes", "images"):
            assert hasattr(ds, attr), f"dataset {type(ds).__name__} has no '{attr}' (episode selection needs DL3DV_Multi's index)"
        P(f"[record] split {split}: {n} episodes from {root} ({len(ds)} window starts, {len(ds.scenes)} scenes)")
        _, all_stats[split] = record_split(model, criterion, ds, sampler, split, n, rank, n_proc, device,
                                           os.path.join(args.root, "features", split), args, P)
        accelerator.wait_for_everyone()
    if is_main:
        for split, _, n in splits:
            d = os.path.join(args.root, "features", split)
            files = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
            P(f"[record] {split}: {len(files)} npz on disk (requested {n})")
            json.dump({"split": split, "requested": n, "n_files": len(files), "seed": args.seed, "ckpt": args.ckpt,
                       "precision": "bf16" if (args.bf16 and device.type == "cuda") else "fp32",
                       "selection": all_stats[split], "time": time.strftime("%F %T")},
                      open(os.path.join(d, "manifest.json"), "w"), indent=1)
    P("[record] done")


# ----------------------------------------------------------------------------------------------- selftest
def _synthetic_views_preds(B=1, T=4, H=24, W=32, seed=0, perturb=0.0, device="cpu", perturb_frames=None):
    """GT views with random camera poses and a random point map; preds that reproduce the GT exactly
    (residual 0) or with the pose translation perturbed by `perturb` units on every frame (a constant
    absolute offset) or only on the frames listed in `perturb_frames`."""
    from dust3r.utils.geometry import geotrf, inv
    from dust3r.utils.camera import camera_to_pose_encoding, quaternion_to_matrix
    g = torch.Generator().manual_seed(seed)
    views, cams = [], []
    for t in range(T):
        q = torch.randn(B, 4, generator=g); q = q / q.norm(dim=-1, keepdim=True)
        R = quaternion_to_matrix(q); tr = torch.randn(B, 3, generator=g) * 0.3
        cam = torch.eye(4).repeat(B, 1, 1); cam[:, :3, :3] = R; cam[:, :3, 3] = tr
        cams.append(cam)
    for t in range(T):
        pts_cam = torch.rand(B, H, W, 3, generator=g) * 2 - 1; pts_cam[..., 2] = pts_cam[..., 2].abs() + 1.0
        pts_world = geotrf(cams[t], pts_cam)
        valid = torch.rand(B, H, W, generator=g) > 0.1
        views.append({"camera_pose": cams[t], "pts3d": pts_world, "valid_mask": valid,
                      "sky_mask": torch.zeros(B, H, W, dtype=torch.bool),
                      "camera_only": torch.zeros(B, dtype=torch.bool), "is_metric": torch.zeros(B, dtype=torch.bool),
                      "img": torch.zeros(B, 3, H, W), "label": [f"scene{seed}_{t:06d}.png"],
                      "idx": [torch.tensor([7 + seed]), torch.tensor([0]), torch.tensor([t])]})
    in_cam1 = inv(cams[0]); preds = []
    for t in range(T):
        enc = camera_to_pose_encoding(in_cam1 @ cams[t]).clone()
        if perturb_frames is None or t in perturb_frames:
            enc[:, :3] += perturb
        preds.append({"pts3d_in_self_view": geotrf(inv(cams[t]), views[t]["pts3d"]),
                      "pts3d_in_other_view": geotrf(in_cam1, views[t]["pts3d"]),
                      "conf_self": torch.full((B, H, W), float(np.e)), "conf": torch.full((B, H, W), float(np.e)),
                      "camera_pose": enc})
    return views, preds


class _IndexDataset:
    """Selftest stand-in for DL3DV_Multi's index: scenes of the given lengths, every frame with >= num_views
    frames after it is a window start (as _derive_index does with allow_repeat=False)."""
    def __init__(self, lengths, num_views):
        self.num_views = num_views; self.scenes = [f"scene{i}" for i in range(len(lengths))]
        self.sceneids, self.images, self.scene_img_list, self.start_img_ids = [], [], [], []
        off = 0
        for j, n in enumerate(lengths):
            ids = list(range(off, off + n)); self.scene_img_list.append(ids); self.sceneids += [j] * n
            self.images += [f"{i:06d}.png" for i in range(n)]; self.start_img_ids += ids[: max(0, n - num_views + 1)]
            off += n

    def __len__(self):
        return len(self.start_img_ids)

    def __getitem__(self, idx):  # what the fixed loader hands back: the tuple it was given, plus the window
        k, ar, nv = idx
        sc, lo, hi = window_of(self, k, nv)
        return {"k": torch.tensor(k), "scene": self.scenes[sc], "lo": torch.tensor(lo), "hi": torch.tensor(hi)}


def selftest():
    import tempfile
    from dust3r.datasets.base.batched_sampler import CustomRandomSampler
    torch.manual_seed(0)
    criterion = make_criterion()
    B, T = 1, 4
    # 1. exact predictions -> zero residuals
    views, preds = _synthetic_views_preds(B, T)
    res = residuals(criterion, views, preds, camera1=views[0]["camera_pose"])
    for k in ("res_pose_t", "res_pose_R", "res_pts_self", "res_rel_t", "res_rel_R"):
        assert res[k].shape == (B, T), (k, res[k].shape)
        assert float(res[k].abs().max()) < 1e-4, (k, res[k])   # exact predictions: abs AND rel residuals are 0
    assert float(res["res_rel_t"][0, 0]) == 0.0 and float(res["res_rel_R"][0, 0]) == 0.0
    assert bool(res["pose_mask"].all()) and float(res["pts_valid_frac"].min()) > 0.8
    # 2. perturbed translation -> positive, equal translation residuals on every frame, rotation still 0
    views, preds = _synthetic_views_preds(B, T, perturb=0.05)
    res2 = residuals(criterion, views, preds, camera1=views[0]["camera_pose"])
    assert float(res2["res_pose_t"].min()) > 0 and float(res2["res_pose_R"].abs().max()) < 1e-4, res2
    # translation residual = 0.05*sqrt(3) / pred norm factor (pred points are unchanged, so the factor is the GT one)
    assert float((res2["res_pose_t"] / res2["res_pose_t"][0, 0]).std()) < 1e-4, res2["res_pose_t"]
    # 2b. the SAME constant absolute offset cancels in the consecutive-frame RELATIVE encoding: rel residual 0
    assert float(res2["res_rel_t"].abs().max()) < 1e-4 and float(res2["res_rel_R"].abs().max()) < 1e-4, \
        (res2["res_rel_t"], res2["res_rel_R"])
    # 2c. an offset on frame 2 only: abs residual on frame 2 only; rel residual on frames 2 (1->2) and 3 (2->3) only
    views_l, preds_l = _synthetic_views_preds(B, T, perturb=0.05, perturb_frames=(2,))
    res_l = residuals(criterion, views_l, preds_l, camera1=views_l[0]["camera_pose"])
    nz_abs = [t for t in range(T) if float(res_l["res_pose_t"][0, t]) > 1e-4]
    nz_rel = [t for t in range(T) if float(res_l["res_rel_t"][0, t]) > 1e-4]
    assert nz_abs == [2] and nz_rel == [2, 3], (nz_abs, nz_rel, res_l["res_pose_t"], res_l["res_rel_t"])
    assert float(res_l["res_rel_R"].abs().max()) < 1e-4
    # the rel translation residual is the loss-normalised offset rotated into the previous camera: same norm on 2 and 3
    assert abs(float(res_l["res_rel_t"][0, 2]) - float(res_l["res_rel_t"][0, 3])) < 1e-4 and \
        abs(float(res_l["res_rel_t"][0, 2]) - float(res_l["res_pose_t"][0, 2])) < 1e-4, res_l["res_rel_t"]
    # 3. a frame with no valid pixels -> res_pts_self 0, pts_valid_frac 0, others unaffected
    views[2]["valid_mask"][:] = False
    res3 = residuals(criterion, views, preds, camera1=views[0]["camera_pose"])
    assert float(res3["res_pts_self"][0, 2]) == 0.0 and float(res3["pts_valid_frac"][0, 2]) == 0.0
    # 4. record collection + npz assembly round trip (record given out of order, batch dim present on some)
    record = [{"t": t, "frame_feat": torch.randn(1, FRAME_FEAT_DIM), "pose_feat": torch.randn(POSE_FEAT_DIM),
               "camera_pose": preds[t]["camera_pose"][0]} for t in reversed(range(T))]
    ff, pf, cp = collect_record(record, T)
    assert ff.shape == (T, FRAME_FEAT_DIM) and ff.dtype == np.float16 and pf.shape == (T, POSE_FEAT_DIM)
    assert np.allclose(cp[1], preds[1]["camera_pose"][0].numpy()) and np.allclose(ff[0], record[-1]["frame_feat"][0].numpy(), atol=1e-2)
    cam = torch.stack([p["camera_pose"][0] for p in preds]).numpy()
    ep = assemble_episode(ff, pf, cam, res3, episode_meta(views), "train")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "000003.npz"); save_episode(path, ep)
        z = np.load(path)
        assert z["frame_feat"].dtype == np.float16 and z["res_pose_t"].dtype == np.float32 and z["camera_pose"].shape == (T, 7)
        assert z["res_rel_t"].dtype == np.float32 and z["res_rel_t"].shape == (T,) and z["res_rel_R"].shape == (T,)
        assert float(z["res_rel_t"][0]) == 0.0 and float(z["res_rel_R"][0]) == 0.0
        assert not npz_lacks(path) and npz_lacks(path, ("no_such_key",)) and npz_lacks(path + ".missing")
        # a pre-rel npz (arrays dropped) is detected by --fill_rel's guard
        old = {k: z[k] for k in z.files if k not in REL_KEYS}; np.savez(path, **old)
        assert npz_lacks(path), "npz without res_rel_* must be flagged for re-recording"
        assert str(z["scene"]) == "scene0" and str(z["first_frame"]) == "000000.png" and int(z["dataset_idx"]) == 7
        assert str(z["split"]) == "train" and z["t"].tolist() == list(range(T)) and str(z["precision"]) == "fp32"
        assert not os.path.exists(path + ".tmp.npz")
    # 5. wrong record length / missing hook -> clear errors
    for bad in ([], record[:-1]):
        try:
            collect_record(bad, T); raise AssertionError("expected an error")
        except (RuntimeError, AssertionError) as e:
            assert "wgate_record" in str(e), e
    # 6. episode selection on a synthetic index: 3 scenes of 200 / 100 / 64 frames, num_views 64
    #    -> 137 + 37 + 1 = 175 window starts; non-overlapping capacity is 3 + 1 + 1 = 5 windows
    ds = _IndexDataset([200, 100, 64], 64); assert len(ds) == 175
    sampler = CustomRandomSampler(ds, 1, 1, 64, 64, 1)
    eps, st = select_episodes(ds, sampler, 10, 64, seed=5, distinct=True)
    assert st["kept"] == len(eps) <= 5 and st["candidates"] == 175 and st["overlap_skipped"] == 175 - st["kept"], st
    assert st["sampler_epoch"] == 5 and [e["g"] for e in eps] == list(range(len(eps)))
    per_scene = {}
    for e in eps:
        sc, lo, hi = window_of(ds, e["dataset_idx"], 64)
        assert (e["scene"], e["start_pos"]) == (ds.scenes[sc], lo) and e["first_frame"] == f"{lo:06d}.png" and e["last_frame"] == f"{hi:06d}.png"
        assert all(hi < a or lo > b for a, b in per_scene.get(sc, [])), ("overlap", e)
        per_scene.setdefault(sc, []).append((lo, hi))
    assert st["kept"] >= 3, st  # the 64-frame scene and the 100-frame scene each fit one, the 200-frame scene >= 1
    # the plain order (allow_overlap) is the sampler's permutation, prefix-consistent with the distinct one
    sampler.set_epoch(5); order = [int(t[0]) for t in list(sampler)]
    eps_all, st_all = select_episodes(ds, sampler, 10, 64, seed=5, distinct=False)
    assert [e["dataset_idx"] for e in eps_all] == order[:10] and st_all["overlap_skipped"] == 0 and st_all["candidates"] == 10
    assert eps[0]["dataset_idx"] == order[0]
    eps_s0, _ = select_episodes(ds, sampler, 10, 64, seed=0, distinct=False)
    assert [e["dataset_idx"] for e in eps_s0] != [e["dataset_idx"] for e in eps_all], "seed must change the order"
    # 7. sharding + fixed loader: rank r gets g = r (mod n), batches come back in list order with the right window
    for r in range(4):
        assert [e["g"] % 4 for e in my_episodes(eps_all, r, 4)] == [r] * len(my_episodes(eps_all, r, 4))
    assert sum(len(my_episodes(eps_all, r, 4)) for r in range(4)) == len(eps_all)
    mine = my_episodes(eps_all, 1, 4)
    loader = make_fixed_loader(ds, mine, 0)
    got = list(loader); assert len(got) == len(mine) == len(loader)
    for e, b in zip(mine, got):
        assert int(b["k"][0]) == e["dataset_idx"] and b["scene"][0] == e["scene"] and int(b["lo"][0]) == e["start_pos"], (e, b)
    assert parse_shard("2/4") == (2, 4)
    print("record_wgate_features selftest OK "
          f"(r_t perturbed {float(res2['res_pose_t'][0,0]):.4f} with rel_t {float(res2['res_rel_t'].abs().max()):.1e}, "
          f"single-frame offset: abs on {nz_abs} rel on {nz_rel}; exact max |r| "
          f"{float(max(res[k].abs().max() for k in ('res_pose_t','res_pose_R','res_pts_self'))):.2e}; "
          f"selection: {st['kept']} distinct of 10 requested from 175 starts, seed-5 head {order[:3]})")


if __name__ == "__main__":
    main()
