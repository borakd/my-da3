#!/usr/bin/env python3
"""Control arms for the write-gate variants (WRITE_GATE_VARIANTS.md): dose-matched constants
read from the head arms' trace dumps, and GT-residual oracles.

Inputs (a finished 430- or 4292-scene pilot of the head arms, e.g. $OUT/wgate430):
  <pilot_dir>/<fg_arm>/eval/<scene>/wgate_trace.json   {"trace": [[t, a_t], ...]}  (variant 1)
  <pilot_dir>/<mg_arm>/eval/<scene>/wgate_trace.json   {"trace": [[t, b_t], ...]}  (variant 2)
  <scenes_root>/<scene>/dense/cam/*.npz  (key 'pose', c2w)      GT cameras
  <base_preds>/<scene>/camera/*.npz      (key 'pose', c2w)      the plain augfull_lr1e5 run

Outputs (control jsons consumed by infer_and_eval_worker.py --control_json, written to --out_dir):
  maks_arm_fg_const.json   {"*": {"frame_gate": {"const": c}}}              c = mean a_t over all frames t>0
                           of all scenes; the explicit "alpha": {"0": 1.0} keeps frame 0 at full write
                           (the worker applies alpha_all to EVERY view, frame 0 included, and the per-frame
                           "alpha" entry overrides it; the model's update_alpha path has no t == 0
                           exemption, whereas the frame-gate head hard-codes a_0 = 1 -- frame 0 is the
                           largest write of the sequence, so a constant applied there is not dose-matched)
  maks_arm_mg_const.json   {"*": {"mem_gate": {"const": mean b_t over all frames t>0 of all scenes}}}
                           (the model itself writes b_0 = 1 for every mem_gate key, so no frame-0 entry)
  maks_arm_fg_oracle.json  {scene: {"frame_gate": {"alpha": {str(t): a_t}}}}  per-scene per-frame (V1 oracle)
  maks_arm_mg_oracle.json  {scene: {"mem_gate": {"alpha": {str(t): b_t}}}}  per-scene per-frame (V2 oracle)
  maks_arm_{fg,mg}_oracle.skipped.txt   scenes of the list with no oracle entry (no GT / pred cameras):
                           the worker runs those PLAIN under the oracle arm; wgate_table.py drops them
  <pilot_dir>/wgate_controls_summary.json   the constants, medians, counts (for the run log)

--fg_const_name / --mg_const_name / --fg_oracle_name / --mg_oracle_name NAME (default: the
  suffix scheme below): write that control to maks_arm_<NAME>.json instead, so an arm family whose
  name is not fg_*/mg_* gets control arms with matching names. Variant C (the consequence-trained
  gate, WRITE_GATE_VARIANTS.md) uses exactly this:
      wgate_make_controls.py --only const --fg_arm cg1_head --mg_arm cg2_head \
          --fg_const_name cg1_const --mg_const_name cg2_const --suffix _cg
  -> maks_arm_cg1_const.json (state gate, {"frame_gate": {"const": c}}) and maks_arm_cg2_const.json
  ({"mem_gate": {"const": c}}), the dose-matched constants of the two variant-C head arms, with the
  run summary at <pilot_dir>/wgate_controls_summary_cg.json. The explicit names win over --suffix
  (which still names the summary file and the default head arms); the cg heads are ordinary
  frame_gate / mem_gate arms, so the traces are read exactly as for fg_head / mg_head.

--suffix S (default ""): every output name gets S before its extension (maks_arm_fg_const_rel.json, ...,
  wgate_controls_summary_rel.json) and the head arms default to fg_head<S> / mg_head<S>, so the controls of
  the RELATIVE-residual heads (train_wgate_heads.py --target rel, arms fg_head_rel / mg_head_rel) are
  generated with `--suffix _rel` without touching the abs files. --oracle_residual {auto,abs,rel} picks the
  GT residual of the oracles (auto = rel iff the suffix ends in "_rel"): abs = the frame-0-relative error
  below; rel = the CONSECUTIVE-frame error from the same GT/pred cameras (gt_pose_errors_rel):
  e_trans_rel[t] = ||(C[t] - C[t-1]) - (G[t] - G[t-1])|| on the Sim(3)-aligned predicted centres C and the GT
  centres G (metres, world = frame-0 frame; scale from the same umeyama fit), e_rot_rel[t] = geodesic angle
  between the GT and predicted frame-to-frame rotations R_gt[t-1]^T R_gt[t] vs R_pr[t-1]^T R_pr[t] (this is
  the evaluator's per-pair rpe_rot); both 0 at t = 0. The oracle map is unchanged: each term normalised by
  its pooled median over the subset, V1 a_t = clip(median(r)/r_t, wmin, 1), V2 b_t = clip(1/sqrt(sig_t sig_R)).

Oracle weights use the head's own sigma -> weight map with sigma replaced by the frame's GT pose
error normalised by its median over the subset (all frames t>0 of all listed scenes), term-wise so
the residual is unit-free (metres and radians are NOT added raw: at the subset medians, .068 m vs
.30 rad, a raw sum ranks frames almost entirely by rotation):
  V1  r_t = e_trans / median(e_trans) + e_rot / median(e_rot),   a_t = clip( median(r) / r_t, wmin, 1 )
  V2  sig_t = e_trans / median(e_trans), sig_R = e_rot / median(e_rot),
      b_t = clip( 1 / sqrt(sig_t * sig_R), wmin, 1 )        (= combine_pose_sigma with ref = median)
  frame 0 -> 1.0; a missing or exactly-zero residual -> 1.0 (no information = full write).
GT pose error per frame: c2w poses of GT and of the plain run made relative to frame 0 (both then
live in the frame-0 camera frame), Sim(3) umeyama fit of the predicted centres onto GT over the whole
scene (the evaluator's --align sim3; e_trans equals the evaluator's per-frame ATE to 1e-9),
e_trans = ||t_aligned - t_gt|| (GT metres), e_rot = geodesic angle of R_gt^T R_pred in radians WITHOUT
the fit's rotation: the umeyama rotation is fitted on centres only and is undetermined about the axis
of a near-linear wrist path (verified: applying it adds a constant 15-33 deg offset to every frame),
whereas the frame-0-relative rotation needs no alignment (e_rot(1) ~ 0.003 rad, growing with drift).

    python eval_pipeline/wgate_make_controls.py [--pilot_dir $OUT/wgate430] [--scene_list eval_pipeline/maks_subset430.txt]
"""
import argparse
import json
import os
import sys

import numpy as np

OUT = "/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval"
SCENES_ROOT = "/gpfs/scratch/etur59/koc821022/pointworld_droid_splits/test/dl3dv_multi/wrist"
EP = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- geometry
def umeyama(src, dst):
    """Sim(3) fit dst ~ s R src + t (copied from maks_render_conf_gate_video.py)."""
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, sv, Vt = np.linalg.svd(D.T @ S / len(src))
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    var = (S ** 2).sum() / len(src)
    s = float(np.trace(np.diag(sv) @ W) / var) if var > 0 else 1.0
    return s, R, md - s * (R @ ms)


def load_pose_dir(d, key="pose"):
    """All <digits>.npz of a camera dir, sorted by name, as (n,4,4) c2w; NaN rows for unreadable files.
    Positional order = the evaluator's join order (it renumbers files positionally per stream)."""
    files = sorted(f for f in os.listdir(d) if f.endswith(".npz") and f[:-4].isdigit())
    P = np.full((len(files), 4, 4), np.nan)
    for i, f in enumerate(files):
        try:
            P[i] = np.asarray(np.load(os.path.join(d, f))[key], float)
        except Exception:
            pass
    return P


def _aligned(gt_cam_dir, pred_cam_dir):
    """Shared front end of both error functions: (G, P, C, ok, n) with G / P the frame-0-relative GT / predicted
    c2w poses (n,4,4), C the predicted centres Sim(3)-aligned onto the GT centres (umeyama over all valid
    frames) and ok the per-frame validity; None when fewer than 3 valid frames or frame 0 is missing."""
    G, P = load_pose_dir(gt_cam_dir), load_pose_dir(pred_cam_dir)
    n = min(len(G), len(P))
    if len(G) != len(P):
        print(f"  WARNING: {gt_cam_dir}: {len(G)} GT vs {len(P)} pred frames; using the first {n}", file=sys.stderr)
    G, P = G[:n], P[:n]
    ok = np.isfinite(G[:, 0, 0]) & np.isfinite(P[:, 0, 0])
    if n < 3 or ok.sum() < 3 or not (ok[0]):
        return None, None, None, ok, n
    G = np.linalg.inv(G[0])[None] @ np.where(ok[:, None, None], G, np.eye(4))
    P = np.linalg.inv(P[0])[None] @ np.where(ok[:, None, None], P, np.eye(4))
    s, R, t = umeyama(P[ok][:, :3, 3], G[ok][:, :3, 3])
    C = (s * (R @ P[:, :3, 3].T)).T + t
    return G, P, C, ok, n


def _geodesic(E):
    tr = np.trace(E, axis1=1, axis2=2)
    return np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))


def gt_pose_errors(gt_cam_dir, pred_cam_dir):
    """Per-frame pose error of a run vs GT: (e_trans, e_rot_rad) arrays of length n, NaN where a frame
    is missing. Both trajectories are expressed relative to their frame 0, the predicted centres are
    Sim(3)-aligned onto GT over all valid frames (umeyama), then e_trans = ||C_aligned - C_gt|| (this is
    the evaluator's per-frame ATE) and e_rot = arccos((tr(R_gt^T R_pred) - 1) / 2) on the frame-0-relative
    rotations, deliberately without the umeyama rotation (see the module docstring)."""
    G, P, C, ok, n = _aligned(gt_cam_dir, pred_cam_dir)
    e_t = np.full(n, np.nan)
    e_R = np.full(n, np.nan)
    if G is None:
        return e_t, e_R
    e_t[ok] = np.linalg.norm(C - G[:, :3, 3], axis=1)[ok]
    E = np.transpose(G[:, :3, :3], (0, 2, 1)) @ P[:, :3, :3]   # no R_fit: both are frame-0-relative
    e_R[ok] = _geodesic(E)[ok]
    return e_t, e_R


def gt_pose_errors_rel(gt_cam_dir, pred_cam_dir):
    """CONSECUTIVE-frame pose error of a run vs GT (the RELATIVE-target oracle): (e_trans_rel, e_rot_rel)
    of length n, 0 at frame 0, NaN where frame t or t-1 is missing. Same cameras and the same Sim(3)
    alignment as gt_pose_errors; e_trans_rel[t] = ||(C[t]-C[t-1]) - (G[t]-G[t-1])|| (aligned predicted step
    vs GT step, metres, frame-0 world frame; a constant offset or a wrong global scale of the prediction
    cancels), e_rot_rel[t] = geodesic angle between R_gt[t-1]^T R_gt[t] and R_pr[t-1]^T R_pr[t] (= the
    evaluator's per-pair rpe_rot, no alignment needed). Non-accumulating, like the recorder's res_rel_*."""
    G, P, C, ok, n = _aligned(gt_cam_dir, pred_cam_dir)
    e_t = np.full(n, np.nan)
    e_R = np.full(n, np.nan)
    if G is None:
        return e_t, e_R
    pair = np.zeros(n, dtype=bool); pair[1:] = ok[1:] & ok[:-1]
    dC = C[1:] - C[:-1]; dG = G[1:, :3, 3] - G[:-1, :3, 3]
    e_t[1:][pair[1:]] = np.linalg.norm(dC - dG, axis=1)[pair[1:]]
    Rg = np.transpose(G[:-1, :3, :3], (0, 2, 1)) @ G[1:, :3, :3]
    Rp = np.transpose(P[:-1, :3, :3], (0, 2, 1)) @ P[1:, :3, :3]
    e_R[1:][pair[1:]] = _geodesic(np.transpose(Rg, (0, 2, 1)) @ Rp)[pair[1:]]
    if ok[0]:
        e_t[0] = 0.0; e_R[0] = 0.0
    return e_t, e_R


# ----------------------------------------------------------------------------- traces
def read_trace(path, gate=None):
    """<eval_dir>/<scene>/wgate_trace.json -> {t: weight}. gate = "frame" | "mem" | None: prefer the
    model's per-gate list ("trace_frame" / "trace_mem", unambiguous when both gates ran) and fall back
    to "trace" (rows [t, weight(, ...)], the first weight column). A weight recorded as a list (batch
    axis) is averaged. Returns {} when the file is missing or empty."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    rows = d.get(f"trace_{gate}") if gate else None
    if not rows:
        rows = d.get("trace", [])
    out = {}
    for r in rows:
        if not isinstance(r, (list, tuple)) or len(r) < 2:
            continue
        t, w = int(r[0]), r[1]
        w = float(np.mean(w)) if isinstance(w, (list, tuple)) else float(w)
        if np.isfinite(w):
            out[t] = w
    return out


def collect_traces(pilot_dir, arm, scenes, gate=None):
    """Per-scene {t: w} traces of one head arm; scenes without a trace file are reported, not fatal."""
    traces, missing = {}, []
    for s in scenes:
        tr = read_trace(os.path.join(pilot_dir, arm, "eval", s, "wgate_trace.json"), gate)
        if tr:
            traces[s] = tr
        else:
            missing.append(s)
    return traces, missing


def frame_weight(spec, t):
    """The state-update weight the worker gives frame t under a "*" control spec. Two paths:
    frame_gate.const (state-only, model exempts t == 0) and the legacy alpha_all / alpha pair (joint
    state+mem: alpha_all is set on every view first, then a per-frame "alpha" entry overrides it)."""
    fg = spec.get("frame_gate") or {}
    if fg.get("const") is not None:
        return 1.0 if t == 0 else float(fg["const"])
    w = 1.0 if spec.get("alpha_all") is None else float(spec["alpha_all"])
    a = spec.get("alpha") or {}
    if str(t) in a:
        w = float(a[str(t)])
    return w


def v1_const_spec(c):
    """State-only constant for the frame-gate control (model key frame_gate_const via the worker's
    frame_gate.const; the model writes frame 0 in full and leaves the retriever-mem commit untouched, exactly
    like the frame-gate head arm). NOT alpha_all: that is the joint state+mem gate of the conf-gate campaign."""
    spec = {"*": {"frame_gate": {"const": float(c)}}}
    assert frame_weight(spec["*"], 0) == 1.0 and abs(frame_weight(spec["*"], 1) - float(c)) < 1e-12
    return spec


def pooled_mean(traces):
    """Mean weight over all frames t>0 of all scenes (pooled over frames), plus the mean of per-scene means."""
    per_frame = [w for tr in traces.values() for t, w in tr.items() if t > 0]
    per_scene = [np.mean([w for t, w in tr.items() if t > 0]) for tr in traces.values() if any(t > 0 for t in tr)]
    if not per_frame:
        return None, None, 0
    return float(np.mean(per_frame)), float(np.mean(per_scene)), len(per_frame)


# ----------------------------------------------------------------------------- oracles
def oracle_weights(err, scenes, wmin):
    """err: {scene: (e_t, e_R)} -> (v1 {scene: {t: a}}, v2 {scene: {t: b}}, stats).
    Medians are pooled over all frames t>0 of all scenes (the 'subset' normalisation). Both oracles
    are invariant to a rescaling of either error term (metres vs mm, radians vs degrees)."""
    et_all = np.concatenate([e[0][1:] for e in err.values()]) if err else np.array([])
    er_all = np.concatenate([e[1][1:] for e in err.values()]) if err else np.array([])
    med_t, med_R = (float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan for x in (et_all, er_all))
    with np.errstate(divide="ignore", invalid="ignore"):
        r_all = et_all / med_t + er_all / med_R   # unit-free: each term normalised by its own pooled median
    med_r = float(np.nanmedian(r_all)) if np.isfinite(r_all).any() else np.nan
    v1, v2 = {}, {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for s in scenes:
            if s not in err:
                continue
            e_t, e_R = err[s]
            r = e_t / med_t + e_R / med_R
            a = np.clip(med_r / r, wmin, 1.0)                                 # V1: clip(ref / sigma), sigma = r / median(r)
            b = np.clip(1.0 / np.sqrt((e_t / med_t) * (e_R / med_R)), wmin, 1.0)  # V2: clip(1 / sigma_comb)
            for arr in (a, b):
                arr[~np.isfinite(arr)] = 1.0   # missing / zero residual -> full write
                arr[0] = 1.0                   # frame 0 always written in full
            v1[s] = {str(t): round(float(a[t]), 4) for t in range(len(a))}
            v2[s] = {str(t): round(float(b[t]), 4) for t in range(len(b))}
    stats = {"median_r_norm": med_r, "median_e_trans": med_t, "median_e_rot_rad": med_R,
             "v1_residual": "e_trans/median(e_trans) + e_rot/median(e_rot)",
             "n_frames": int(np.isfinite(r_all).sum()), "n_scenes": len(v1),
             "mean_a_oracle_t>0": float(np.mean([float(v) for d in v1.values() for t, v in d.items() if int(t) > 0])) if v1 else None,
             "mean_b_oracle_t>0": float(np.mean([float(v) for d in v2.values() for t, v in d.items() if int(t) > 0])) if v2 else None}
    return v1, v2, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot_dir", default=f"{OUT}/wgate430", help="pilot root holding <arm>/eval/<scene>/wgate_trace.json")
    ap.add_argument("--scene_list", default=os.path.join(EP, "maks_subset430.txt"))
    ap.add_argument("--suffix", default="", help="output-name suffix, e.g. _rel: maks_arm_fg_const_rel.json ...; also the "
                                                 "default head arms become fg_head<suffix> / mg_head<suffix>")
    ap.add_argument("--fg_arm", default=None, help="variant-1 head arm name under pilot_dir (default fg_head<suffix>)")
    ap.add_argument("--mg_arm", default=None, help="variant-2 head arm name under pilot_dir (default mg_head<suffix>)")
    for _k, _w in (("fg_const", "V1 dose-matched constant"), ("mg_const", "V2 dose-matched constant"),
                   ("fg_oracle", "V1 oracle"), ("mg_oracle", "V2 oracle")):
        ap.add_argument(f"--{_k}_name", default=None,
                        help=f"arm name of the {_w} output: maks_arm_<NAME>.json (default {_k}<suffix>)")
    ap.add_argument("--oracle_residual", choices=["auto", "abs", "rel"], default="auto",
                    help="GT residual of the oracles: abs = frame-0-relative pose error (gt_pose_errors), rel = consecutive-frame "
                         "error (gt_pose_errors_rel); auto = rel iff --suffix ends in _rel")
    ap.add_argument("--scenes_root", default=SCENES_ROOT)
    ap.add_argument("--base_preds", default=f"{OUT}/augfull_lr1e5/preds", help="plain run's preds (<scene>/camera/*.npz)")
    ap.add_argument("--wmin", type=float, default=0.5, help="floor of the sigma -> weight map (contract: 0.5)")
    ap.add_argument("--out_dir", default=EP, help="where the maks_arm_*.json go (default: eval_pipeline/)")
    ap.add_argument("--only", choices=["all", "const", "oracle"], default="all")
    ap.add_argument("--min_frac", type=float, default=0.95,
                    help="refuse to write a constant when fewer than this fraction of scenes have a trace")
    args = ap.parse_args()
    sfx = args.suffix
    if args.fg_arm is None:
        args.fg_arm = f"fg_head{sfx}"
    if args.mg_arm is None:
        args.mg_arm = f"mg_head{sfx}"
    if args.oracle_residual == "auto":
        args.oracle_residual = "rel" if sfx.endswith("_rel") else "abs"
    # output arm names: explicit --*_name wins, else the suffix scheme (fg_const<suffix>, ...)
    names = {k: (getattr(args, f"{k}_name") or f"{k}{sfx}")
             for k in ("fg_const", "mg_const", "fg_oracle", "mg_oracle")}
    err_fn = gt_pose_errors_rel if args.oracle_residual == "rel" else gt_pose_errors

    scenes = [l.strip() for l in open(args.scene_list) if l.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {"pilot_dir": args.pilot_dir, "scene_list": args.scene_list, "n_scenes": len(scenes), "wmin": args.wmin,
               "suffix": sfx, "fg_arm": args.fg_arm, "mg_arm": args.mg_arm, "oracle_residual": args.oracle_residual,
               "out_names": names}
    written = []
    print(f"suffix '{sfx}': head arms {args.fg_arm} / {args.mg_arm}, oracle residual {args.oracle_residual}; "
          f"outputs " + ", ".join(f"maks_arm_{v}.json" for v in names.values()))

    # ---- dose-matched constants from the head arms' traces
    if args.only in ("all", "const"):
        for arm, gate, fname, mk in ((args.fg_arm, "frame", f"maks_arm_{names['fg_const']}.json", v1_const_spec),
                                     (args.mg_arm, "mem", f"maks_arm_{names['mg_const']}.json",
                                      lambda c: {"*": {"mem_gate": {"const": c}}})):
            traces, missing = collect_traces(args.pilot_dir, arm, scenes, gate)
            pooled, scene_mean, nfr = pooled_mean(traces)
            key = "V1" if arm == args.fg_arm else "V2"
            summary[f"{key}_const"] = {"arm": arm, "n_scenes_with_trace": len(traces), "n_missing": len(missing),
                                       "n_frames_t>0": nfr, "mean_pooled": pooled, "mean_of_scene_means": scene_mean}
            if pooled is None:
                print(f"{key} constant: no traces under {args.pilot_dir}/{arm}/eval -- skipped")
                continue
            frac = len(traces) / max(len(scenes), 1)
            if frac < args.min_frac:
                print(f"{key} constant: only {len(traces)}/{len(scenes)} scenes have a trace (< {args.min_frac}) -- skipped")
                continue
            c = round(pooled, 4)
            p = os.path.join(args.out_dir, fname)
            json.dump(mk(c), open(p, "w"), indent=1)
            written.append(p)
            print(f"{key} constant = {c} (pooled over {nfr} frames t>0 of {len(traces)} scenes; "
                  f"mean of scene means {scene_mean:.4f}; {len(missing)} scenes without trace) -> {p}")

    # ---- oracles from the GT residual of the plain run
    if args.only in ("all", "oracle"):
        err, bad = {}, []
        for s in scenes:
            g = os.path.join(args.scenes_root, s, "dense", "cam")
            if not os.path.isdir(g):
                g = os.path.join(args.scenes_root, s, "dense", "camera")
            p = os.path.join(args.base_preds, s, "camera")
            if not (os.path.isdir(g) and os.path.isdir(p)):
                bad.append(s)
                continue
            e_t, e_R = err_fn(g, p)
            if np.isfinite(e_t).sum() >= 3:
                err[s] = (e_t, e_R)
            else:
                bad.append(s)
        v1, v2, stats = oracle_weights(err, scenes, args.wmin)
        if args.oracle_residual == "rel":
            stats["v1_residual"] = "e_trans_rel/median(e_trans_rel) + e_rot_rel/median(e_rot_rel) (consecutive-frame)"
        summary["oracle"] = dict(stats, n_scenes_skipped=len(bad), skipped=bad[:20], residual=args.oracle_residual)
        for fname, d in ((f"maks_arm_{names['fg_oracle']}.json", {s: {"frame_gate": {"alpha": w}} for s, w in v1.items()}),
                         (f"maks_arm_{names['mg_oracle']}.json", {s: {"mem_gate": {"alpha": w}} for s, w in v2.items()})):
            p = os.path.join(args.out_dir, fname)
            json.dump(d, open(p, "w"))
            written.append(p)
            # scenes with no entry run PLAIN under the oracle arm (the worker warns); list them for the table
            sk = p[:-len(".json")] + ".skipped.txt"
            open(sk, "w").write("".join(f"{s}\n" for s in bad))
            written.append(sk)
        if bad:
            print(f"  WARNING: {len(bad)} listed scenes have no oracle entry (no GT/pred cameras) and would run "
                  f"plain under the oracle arms; listed in the .skipped.txt sidecars: {bad[:5]}{' ...' if len(bad) > 5 else ''}")
        print(f"oracles ({args.oracle_residual} residual): {len(v1)} scenes ({len(bad)} skipped), {stats['n_frames']} frames t>0; "
              f"median r_norm={stats['median_r_norm']:.4f} (e_trans {stats['median_e_trans']:.4f} m, e_rot {stats['median_e_rot_rad']:.4f} rad); "
              f"mean oracle a={stats['mean_a_oracle_t>0']}, b={stats['mean_b_oracle_t>0']}")

    sp = os.path.join(args.pilot_dir, f"wgate_controls_summary{sfx}.json")
    try:
        os.makedirs(args.pilot_dir, exist_ok=True)
        json.dump(summary, open(sp, "w"), indent=1)
        written.append(sp)
    except OSError as e:
        print(f"summary not written ({e})")
    print("written:", *written, sep="\n  ")


if __name__ == "__main__":
    main()
