#!/usr/bin/env python3
"""CPU self-test of the write-gate eval plumbing (WRITE_GATE_VARIANTS.md, Component C), on a
fabricated pilot in a temp dir -- no checkpoint, no GPU, ~10 s:

  1. wgate_make_controls.py: the V1 dose-matched constant keeps frame 0 at weight 1.0 under the
     worker's key precedence (alpha_all first, per-frame alpha overrides), its value is the mean over
     frames t>0 only; the V1/V2 oracles are unit-free (invariant to rescaling either error term), lie
     in [wmin, 1] with a_0 = b_0 = 1; scenes without cameras land in the .skipped.txt sidecars.
     --suffix _rel: reads the fg_head_rel / mg_head_rel traces, writes maks_arm_*_rel.json + a _rel
     summary, oracles from the CONSECUTIVE-frame error (gt_pose_errors_rel: zero for a globally
     rescaled prediction, localised to frames k and k+1 for a single perturbed frame k).
  2. infer_and_eval_worker.apply_wgate_controls on the emitted mem-gate constant / oracle jsons, and
     a source check that the worker's alpha_all block precedes the per-frame alpha block.
  3. wgate_table.py: one common scene set across every complete row (baseline included), oracle
     scenes with no control entry dropped, incomplete/missing arms recorded in the json "skipped"
     list and the md/tex footnote; --allow_partial rows carry the dagger and their own n.
  4. Variant C (WRITE_GATE_VARIANTS.md, the CONSEQUENCE-trained gate): the four maks_arm_cg*.json
     arms; wgate_make_controls.py with --fg_const_name / --mg_const_name (the supervisor's C430
     command) writing maks_arm_{cg1,cg2}_const.json; the table's C rows + group heading; and the
     worker's weight-mode control path (attach_wgate_head on a FABRICATED {"mode": "weight", ...}
     checkpoint against stub models with and without a `train` kwarg, plus the sigma-mode
     regression and the error message when model.py cannot take a weight-mode checkpoint).

    OMP_NUM_THREADS=1 timeout 280 python eval_pipeline/wgate_selftest.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

EP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EP)
import wgate_make_controls as mc  # noqa: E402
import wgate_table as wt  # noqa: E402

METRICS = ["absrel", "a1", "ate", "rpe_trans", "rpe_rot"]


def rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def write_cams(d, poses):
    os.makedirs(d, exist_ok=True)
    for i, P in enumerate(poses):
        np.savez(os.path.join(d, f"{i:05d}.npz"), pose=P, intrinsics=np.eye(3))


def fake_trajectory(rng, n, noise):
    P = np.tile(np.eye(4), (n, 1, 1))
    for i in range(1, n):
        P[i, :3, :3] = rot(rng.normal(size=3), 0.02 * i) @ rot([0, 0, 1], noise * rng.normal())
        P[i, :3, 3] = np.array([0.01 * i, 0.002 * i * i, 0.0]) + noise * rng.normal(size=3)
    return P


def write_eval_csv(eval_dir, scene, vals):
    d = os.path.join(eval_dir, scene)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "eval_depth_pose_metrics.csv"), "w") as f:
        f.write("camera_id,local_timestep," + ",".join(METRICS) + "\n")
        f.write("ALL,MEAN," + ",".join(f"{vals[m]:.6f}" for m in METRICS) + "\n")


def test_generator(tmp):
    rng = np.random.default_rng(0)
    scenes = [f"scene{i:02d}" for i in range(6)]
    pilot, gt_root, preds, out = (os.path.join(tmp, k) for k in ("pilot", "gt", "preds", "out"))
    n = 12
    for i, s in enumerate(scenes):
        # traces: a_0 = 1 always; a_t = .6 for the first 3 scenes, .8 for the rest -> pooled .7
        for arm, key, w in (("fg_head", "trace_frame", 0.6 if i < 3 else 0.8), ("mg_head", "trace_mem", 0.9),
                            ("fg_head_rel", "trace_frame", 0.75), ("mg_head_rel", "trace_mem", 0.85),
                            ("cg1_head", "trace_frame", 0.65), ("cg2_head", "trace_mem", 0.88)):
            d = os.path.join(pilot, arm, "eval", s)
            os.makedirs(d, exist_ok=True)
            json.dump({"scene": s, "gates": {}, "trace": [[0, 1.0]] + [[t, w] for t in range(1, n)],
                       key: [[0, 1.0]] + [[t, w] for t in range(1, n)]}, open(os.path.join(d, "wgate_trace.json"), "w"))
        G = fake_trajectory(rng, n, 0.0)
        write_cams(os.path.join(gt_root, s, "dense", "cam"), G)
        if i == 5:
            continue  # no predictions for the last scene -> oracle must skip it
        write_cams(os.path.join(preds, s, "camera"), fake_trajectory(rng, n, 0.01))
    lst = os.path.join(tmp, "list.txt")
    open(lst, "w").write("\n".join(scenes) + "\n")
    cmd = [sys.executable, os.path.join(EP, "wgate_make_controls.py"), "--pilot_dir", pilot, "--scene_list", lst,
           "--scenes_root", gt_root, "--base_preds", preds, "--out_dir", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    fg = json.load(open(os.path.join(out, "maks_arm_fg_const.json")))
    assert set(fg) == {"*"} and abs(fg["*"]["frame_gate"]["const"] - 0.7) < 1e-9, fg   # state-only path
    assert "alpha_all" not in fg["*"], fg
    assert mc.frame_weight(fg["*"], 0) == 1.0 and abs(mc.frame_weight(fg["*"], 5) - 0.7) < 1e-9
    assert wt.const_of(os.path.join(out, "x"), "y") is None
    mg = json.load(open(os.path.join(out, "maks_arm_mg_const.json")))
    assert mg == {"*": {"mem_gate": {"const": 0.9}}}, mg
    o1 = json.load(open(os.path.join(out, "maks_arm_fg_oracle.json")))
    o2 = json.load(open(os.path.join(out, "maks_arm_mg_oracle.json")))
    assert set(o1) == set(o2) == set(scenes[:5]), (sorted(o1), sorted(o2))
    for s in o1:
        a, b = o1[s]["frame_gate"]["alpha"], o2[s]["mem_gate"]["alpha"]
        assert a["0"] == 1.0 and b["0"] == 1.0
        assert all(0.5 <= v <= 1.0 for v in a.values()) and all(0.5 <= v <= 1.0 for v in b.values())
        assert min(a.values()) < 1.0 and min(b.values()) < 1.0, "oracle never below 1?"
    for k in ("fg", "mg"):
        sk = open(os.path.join(out, f"maks_arm_{k}_oracle.skipped.txt")).read().split()
        assert sk == [scenes[5]], sk
    summ = json.load(open(os.path.join(pilot, "wgate_controls_summary.json")))
    assert summ["oracle"]["n_scenes"] == 5 and summ["oracle"]["n_scenes_skipped"] == 1
    assert "v1_residual" in summ["oracle"] and "median_r_norm" in summ["oracle"]
    # unit-free: rescaling either term changes nothing (metres -> mm, radians -> degrees)
    err = {s: (np.abs(rng.normal(size=n)) * 0.07, np.abs(rng.normal(size=n)) * 0.3) for s in scenes[:4]}
    v1, v2, st = mc.oracle_weights(err, scenes, 0.5)
    v1b, v2b, _ = mc.oracle_weights({s: (e[0] * 1000.0, e[1] * 180 / np.pi) for s, e in err.items()}, scenes, 0.5)
    assert v1 == v1b and v2 == v2b, "oracle not unit-free"
    # the raw-sum reading would have ranked by rotation: check both terms move a_t here
    e_t, e_R = err[scenes[0]]
    a = np.array([v1[scenes[0]][str(t)] for t in range(n)])
    rr = e_t / st["median_e_trans"] + e_R / st["median_e_rot_rad"]
    ref = np.clip(st["median_r_norm"] / rr, 0.5, 1.0); ref[0] = 1.0
    assert np.allclose(a, np.round(ref, 4)), (a, ref)
    print("  generator OK:", r.stdout.strip().splitlines()[0])
    # ---- --suffix _rel: the rel head arms' traces, _rel file names, consecutive-frame oracle; abs files untouched
    before = {f: open(os.path.join(out, f)).read() for f in os.listdir(out)}
    r = subprocess.run(cmd + ["--suffix", "_rel"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert all(open(os.path.join(out, f)).read() == txt for f, txt in before.items()), "abs control files changed by the _rel run"
    fgr = json.load(open(os.path.join(out, "maks_arm_fg_const_rel.json")))
    assert fgr == {"*": {"frame_gate": {"const": 0.75}}}, fgr
    assert json.load(open(os.path.join(out, "maks_arm_mg_const_rel.json"))) == {"*": {"mem_gate": {"const": 0.85}}}
    o1r = json.load(open(os.path.join(out, "maks_arm_fg_oracle_rel.json")))
    o2r = json.load(open(os.path.join(out, "maks_arm_mg_oracle_rel.json")))
    assert set(o1r) == set(o2r) == set(scenes[:5]) and o1r != o1 and o2r != o2, "rel oracle must differ from the abs one"
    for sc in o1r:
        a, b = o1r[sc]["frame_gate"]["alpha"], o2r[sc]["mem_gate"]["alpha"]
        assert a["0"] == 1.0 and b["0"] == 1.0 and all(0.5 <= v <= 1.0 for v in a.values()) and all(0.5 <= v <= 1.0 for v in b.values())
    for k in ("fg", "mg"):
        assert open(os.path.join(out, f"maks_arm_{k}_oracle_rel.skipped.txt")).read().split() == [scenes[5]]
    summr = json.load(open(os.path.join(pilot, "wgate_controls_summary_rel.json")))
    assert summr["suffix"] == "_rel" and summr["fg_arm"] == "fg_head_rel" and summr["oracle"]["residual"] == "rel" \
        and "consecutive" in summr["oracle"]["v1_residual"] and summr["oracle"]["n_scenes"] == 5, summr
    assert json.load(open(os.path.join(pilot, "wgate_controls_summary.json")))["oracle"]["residual"] == "abs"
    # gt_pose_errors_rel: a globally rescaled prediction has zero consecutive-frame error (the Sim(3) fit absorbs
    # the scale); a translation offset on frame k alone shows up on frames k and k+1 only (rotation error 0)
    G = fake_trajectory(rng, n, 0.0)
    P2 = G.copy(); P2[:, :3, 3] *= 2.0
    d_g, d_p = os.path.join(tmp, "relcams", "gt"), os.path.join(tmp, "relcams", "p2")
    write_cams(d_g, G); write_cams(d_p, P2)
    et, eR = mc.gt_pose_errors_rel(d_g, d_p)
    assert et.shape == (n,) and et[0] == 0.0 and eR[0] == 0.0 and np.nanmax(et) < 1e-9 and np.nanmax(eR) < 1e-6, (et, eR)   # arccos noise ~1e-8 on identical rotations
    et_abs, _ = mc.gt_pose_errors(d_g, d_p)
    assert np.nanmax(et_abs) < 1e-9   # the abs error is scale-free too (same fit)
    k = 6
    P3 = G.copy(); P3[k, :3, 3] += np.array([0.05, -0.02, 0.03])
    d_p3 = os.path.join(tmp, "relcams", "p3"); write_cams(d_p3, P3)
    et3, eR3 = mc.gt_pose_errors_rel(d_g, d_p3)
    top2 = set(np.argsort(np.nan_to_num(et3, nan=-1.0))[-2:].tolist())
    assert top2 == {k, k + 1} and np.nanmax(eR3) < 1e-6 and et3[0] == 0.0, (et3, eR3)
    assert et3[k] > 10 * np.max(np.delete(et3, [k, k + 1])), et3   # the umeyama refit leaks only a little elsewhere
    et3a, _ = mc.gt_pose_errors(d_g, d_p3)
    assert int(np.nanargmax(et3a)) == k, et3a
    print("  generator --suffix _rel OK (rel oracle: scaled pred -> 0, single-frame offset -> frames", sorted(top2), ")")
    # ---- variant C: the supervisor's C430 command line (explicit out names win over --suffix)
    before = {f: open(os.path.join(out, f)).read() for f in os.listdir(out)}
    r = subprocess.run(cmd + ["--only", "const", "--suffix", "_cg", "--fg_arm", "cg1_head", "--mg_arm", "cg2_head",
                              "--fg_const_name", "cg1_const", "--mg_const_name", "cg2_const"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert all(open(os.path.join(out, f)).read() == txt for f, txt in before.items()), "V1/V2 control files changed by the _cg run"
    assert json.load(open(os.path.join(out, "maks_arm_cg1_const.json"))) == {"*": {"frame_gate": {"const": 0.65}}}
    assert json.load(open(os.path.join(out, "maks_arm_cg2_const.json"))) == {"*": {"mem_gate": {"const": 0.88}}}
    assert not os.path.exists(os.path.join(out, "maks_arm_fg_const_cg.json")), "the --*_name override did not win over --suffix"
    sc = json.load(open(os.path.join(pilot, "wgate_controls_summary_cg.json")))
    assert sc["out_names"]["fg_const"] == "cg1_const" and sc["out_names"]["mg_const"] == "cg2_const" \
        and sc["V1_const"]["arm"] == "cg1_head" and abs(sc["V2_const"]["mean_pooled"] - 0.88) < 1e-9, sc
    assert "oracle" not in sc, "--only const must not compute oracles"
    print("  generator variant-C constants OK (cg1_const .65 frame_gate / cg2_const .88 mem_gate)")
    return out


def test_worker(out):
    import torch  # noqa: F401  (the worker imports torch at module level)
    import infer_and_eval_worker as w
    views = [{} for _ in range(4)]
    act = w.apply_wgate_controls(views, json.load(open(os.path.join(out, "maks_arm_mg_const.json")))["*"])
    assert act == {"mem_gate": {"const": 0.9}} and all(abs(float(v["mem_gate_const"]) - 0.9) < 1e-6 for v in views)
    o2 = json.load(open(os.path.join(out, "maks_arm_mg_oracle.json")))
    s0 = sorted(o2)[0]
    views = [{} for _ in range(4)]
    act = w.apply_wgate_controls(views, o2[s0])
    assert act["mem_gate"]["alpha_frames"] == 4 and float(views[0]["mem_gate_const"]) == 1.0
    vfc = [{} for _ in range(3)]
    act = w.apply_wgate_controls(vfc, json.load(open(os.path.join(out, "maks_arm_fg_const.json")))["*"])
    assert abs(act["frame_gate"]["const"] - 0.7) < 1e-9 and all(abs(float(v["frame_gate_const"]) - 0.7) < 1e-6 for v in vfc), act
    assert "frame_gate_on" not in vfc[0] and "update_alpha" not in vfc[0]   # state-only constant, no head, not the joint path
    src = open(os.path.join(EP, "infer_and_eval_worker.py")).read()
    i_all = src.index('if ctl.get("alpha_all") is not None:')
    i_per = src.index('for fr_, a_ in (ctl.get("alpha") or {}).items():')
    assert i_all < i_per, "worker: per-frame alpha must be applied AFTER alpha_all (frame_weight precedence)"
    assert "has no control entry in" in src and "only the 'block' lists" in src
    print("  worker OK")


def test_variant_c(tmp):
    """The variant-C arm jsons + the worker's weight-mode attach dispatch (fabricated checkpoints)."""
    import torch
    import torch.nn as nn
    import infer_and_eval_worker as w

    # ---- the four arm jsons of the campaign
    # the head path is the TRAINER's real output: --out defaults to <root>/consequence and the
    # sbatch's documented NAME is the arm name itself, so <root>/consequence/<arm>/gate_head_best.pth.
    want = {"cg1_head": ("frame_gate", "consequence/cg1_head"),
            "cg1_head_lr5": ("frame_gate", "consequence/cg1_head_lr5"),
            "cg2_head": ("mem_gate", "consequence/cg2_head"),
            "cg2_head_lr5": ("mem_gate", "consequence/cg2_head_lr5")}
    for arm, (gate, ckdir) in want.items():
        c = json.load(open(os.path.join(EP, f"maks_arm_{arm}.json")))
        assert set(c) == {"*"} and set(c["*"]) == {gate}, (arm, c)
        assert c["*"][gate]["wmin"] == 0.5
        assert c["*"][gate]["head"] == f"{w.WGATE_HEAD_ROOT}/{ckdir}/gate_head_best.pth", (arm, c)
        assert arm in wt.NAMES, f"{arm} missing from wgate_table.ROWS"
    # ... and that path is the one the TRAINER actually writes: <trainer OUT_ROOT>/<NAME> with
    # NAME = the arm name (the sbatch's documented NAME=), so an arm is scorable standalone with
    # `ARM=cg1_head sbatch eval_pipeline/maks_sweep430.sbatch`, no symlink needed.
    import re
    root = os.path.dirname(EP)
    tr = os.path.join(root, "src", "CUT3R", "src", "train_wgate_consequence.py")
    sb = os.path.join(EP, "train_wgate_consequence.sbatch")
    if os.path.exists(tr) and os.path.exists(sb):
        m = re.search(r'^OUT_ROOT\s*=\s*"([^"]+)"', open(tr).read(), re.M)
        assert m, "train_wgate_consequence.py: no OUT_ROOT"
        out_root = m.group(1).rstrip("/")
        m2 = re.search(r'^WGATE_OUT=\$\{WGATE_OUT:-([^}]+)\}', open(sb).read(), re.M)
        assert m2, "train_wgate_consequence.sbatch: no WGATE_OUT default"
        assert m2.group(1).rstrip("/") == out_root, (m2.group(1), out_root)
        for arm, (gate, _) in want.items():
            c = json.load(open(os.path.join(EP, f"maks_arm_{arm}.json")))
            assert c["*"][gate]["head"] == f"{out_root}/{arm}/gate_head_best.pth", \
                (arm, c["*"][gate]["head"], out_root)
        print(f"  variant-C arm heads agree with the trainer output root {out_root}/<NAME>")
    for a in ("cg1_const", "cg2_const"):
        assert a in wt.NAMES, f"{a} missing from wgate_table.ROWS"
    cg_rows = [r for r in wt.ROWS if r[0].startswith("cg")]
    assert {r[3] for r in cg_rows} == {"C"} and wt.GROUP_HEADINGS.get("C") == "Variant C (consequence-trained)"
    assert [r[0] for r in cg_rows] == ["cg1_head", "cg2_head", "cg1_head_lr5", "cg2_head_lr5", "cg1_const", "cg2_const"]

    # ---- fabricated checkpoints: variant C (weight mode) and variants 1/2 (sigma mode)
    cw = {"mode": "weight", "state_dict": {"norm.weight": torch.ones(4), "out.weight": torch.zeros(1, 4),
                                           "out.bias": torch.full((1,), 2.0)},
          "wmin": 0.5, "init_logit": 2.0, "args": {"lr": 1e-4}, "metrics": {}}
    cs = {"state_dict": {"norm.weight": torch.ones(4)}, "log_ref": 1.17, "wmin": 0.5}
    cm = {"state_dict": {"out.weight": torch.zeros(2, 8), "out.bias": torch.zeros(2)},
          "log_ref_t": 1.05, "log_ref_R": 0.83, "wmin": 0.5}
    pw, ps, pm = (os.path.join(tmp, f) for f in ("cg_weight.pth", "fg_sigma.pth", "mg_sigma.pth"))
    for d, f in ((cw, pw), (cs, ps), (cm, pm)):
        torch.save(d, f)
    assert w.wgate_mode(w.load_wgate_head(pw)) == "weight"
    assert w.wgate_mode(w.load_wgate_head(ps)) == "sigma" and w.wgate_mode({"mode": None, "state_dict": {}}) == "sigma"

    class Stub:
        """Records the attach call. train_kw=False mimics a model.py without the variant-C `train` kwarg;
        sigma_only=True mimics one whose attach asserts on log_ref (i.e. no weight-mode dispatch)."""

        def __init__(self, train_kw=True, sigma_only=False):
            self.calls = []
            self.sigma_only = sigma_only
            self.frame_gate = self.mem_gate = None
            if train_kw:
                self.attach_frame_gate = lambda h, log_ref=None, wmin=None, train=False: self._att(
                    "frame", h, {"log_ref": log_ref, "wmin": wmin, "train": train})
                self.attach_mem_gate = lambda h, log_ref_t=None, log_ref_R=None, wmin=None, train=False: self._att(
                    "mem", h, {"log_ref_t": log_ref_t, "log_ref_R": log_ref_R, "wmin": wmin, "train": train})
            else:
                self.attach_frame_gate = lambda h, log_ref=None, wmin=None: self._att(
                    "frame", h, {"log_ref": log_ref, "wmin": wmin})
                self.attach_mem_gate = lambda h, log_ref_t=None, log_ref_R=None, wmin=None: self._att(
                    "mem", h, {"log_ref_t": log_ref_t, "log_ref_R": log_ref_R, "wmin": wmin})

        def _att(self, kind, h, kw):
            if self.sigma_only:
                assert kw.get("log_ref") is not None or kw.get("log_ref_t") is not None, \
                    "attach needs log_ref (sigma-only model.py)"
            self.calls.append((kind, h, kw))
            head = nn.Linear(4, 1)          # returned in TRAIN mode with grads on: the worker must fix both
            head.train()
            for p_ in head.parameters():
                p_.requires_grad_(True)
            setattr(self, "frame_gate" if kind == "frame" else "mem_gate", head)
            return head

    # weight mode: the WHOLE dict goes through, no log_ref, train=False when the signature has it
    m = Stub()
    head, mode = w.attach_wgate_head(m, "frame", w.load_wgate_head(pw))
    kind, h, kw = m.calls[-1]
    assert mode == "weight" and kind == "frame" and isinstance(h, dict) and h.get("mode") == "weight"
    assert kw["log_ref"] is None and kw["wmin"] == 0.5 and kw["train"] is False, kw
    assert not head.training and all(not p_.requires_grad for p_ in head.parameters()), "head must be attached in EVAL mode"
    head, mode = w.attach_wgate_head(m, "mem", w.load_wgate_head(pw), wmin=0.7)
    kind, h, kw = m.calls[-1]
    assert kind == "mem" and kw["log_ref_t"] is None and kw["log_ref_R"] is None and kw["wmin"] == 0.7 and kw["train"] is False

    # a model.py without the `train` kwarg must not be passed one
    m2 = Stub(train_kw=False)
    w.attach_wgate_head(m2, "frame", w.load_wgate_head(pw))
    assert "train" not in m2.calls[-1][2], m2.calls[-1][2]

    # sigma mode is untouched: state_dict + the trainer's refs, positionally as before
    m3 = Stub()
    _, mode = w.attach_wgate_head(m3, "frame", w.load_wgate_head(ps))
    kind, h, kw = m3.calls[-1]
    assert mode == "sigma" and "norm.weight" in h and "state_dict" not in h and abs(kw["log_ref"] - 1.17) < 1e-9, (h.keys(), kw)
    _, mode = w.attach_wgate_head(m3, "mem", w.load_wgate_head(pm))
    kind, h, kw = m3.calls[-1]
    assert mode == "sigma" and "out.weight" in h and abs(kw["log_ref_t"] - 1.05) < 1e-9 and abs(kw["log_ref_R"] - 0.83) < 1e-9

    # a model.py that cannot take a weight-mode checkpoint fails with an actionable message
    m4 = Stub(sigma_only=True)
    try:
        w.attach_wgate_head(m4, "frame", w.load_wgate_head(pw))
        raise SystemExit("weight-mode attach on a sigma-only model.py must raise")
    except RuntimeError as e:
        assert "weight-mode checkpoint" in str(e) and "model.py" in str(e), e
    # no attach_* at all
    try:
        w.attach_wgate_head(object(), "mem", w.load_wgate_head(pw))
        raise SystemExit("missing attach_mem_gate must raise")
    except RuntimeError as e:
        assert "attach_mem_gate" in str(e), e

    # ---- end-to-end control path: an arm json pointing at the weight-mode checkpoint, worker-style callback
    m5 = Stub()
    seen = {}

    def attach_fg(path):
        ck = w.load_wgate_head(path)
        w.attach_wgate_head(m5, "frame", ck)
        seen["fg"] = path
        return ck

    ctl = {"frame_gate": {"head": pw}}                      # no "wmin" in the json -> falls back to the checkpoint's
    views = [{} for _ in range(5)]
    act = w.apply_wgate_controls(views, ctl, attach_fg, None)
    assert act == {"frame_gate": {"head": pw, "wmin": 0.5}} and seen["fg"] == pw, act
    assert all(float(v["frame_gate_on"]) == 1.0 and abs(float(v["frame_gate_wmin"]) - 0.5) < 1e-6 for v in views)
    assert "frame_gate_const" not in views[0] and "mem_gate_on" not in views[0]
    views = [{} for _ in range(3)]
    act = w.apply_wgate_controls(views, {"mem_gate": {"head": pw, "wmin": 0.6}},
                                 None, lambda path: w.attach_wgate_head(m5, "mem", w.load_wgate_head(path))[0] and None)
    assert act == {"mem_gate": {"head": pw, "wmin": 0.6}} and abs(float(views[0]["mem_gate_wmin"]) - 0.6) < 1e-6
    # ---- the table really renders the C rows under their heading, with the constant read from control.json
    cbase, cpilot, ctdir = (os.path.join(tmp, "c_" + k) for k in ("base", "pilot", "table"))
    cscenes = [f"c{i:02d}" for i in range(6)]
    for s in cscenes:
        write_eval_csv(cbase, s, {"absrel": 0.13, "a1": 0.9, "ate": 0.07, "rpe_trans": 0.006, "rpe_rot": 0.8})
        for arm in ("cg1_head", "cg2_head", "cg1_const"):
            write_eval_csv(os.path.join(cpilot, arm, "eval"), s,
                           {"absrel": 0.128, "a1": 0.902, "ate": 0.068, "rpe_trans": 0.0059, "rpe_rot": 0.79})
    json.dump({"*": {"frame_gate": {"const": 0.72}}}, open(os.path.join(cpilot, "cg1_const", "control.json"), "w"))
    clist = os.path.join(tmp, "clist.txt")
    open(clist, "w").write("\n".join(cscenes) + "\n")
    r = subprocess.run([sys.executable, os.path.join(EP, "wgate_table.py"), "--scene_list", clist, "--pilot_dir", cpilot,
                        "--base_eval", cbase, "--table_dir", ctdir, "--subset", "cg6", "--no_pdf"],
                       capture_output=True, text=True, timeout=200)
    assert r.returncode == 0, r.stdout + r.stderr
    md = open(os.path.join(ctdir, "wgate_cg6.md")).read()
    tex = open(os.path.join(ctdir, "wgate_cg6.tex")).read()
    assert "| **Variant C (consequence-trained)** |" in md, md
    assert r"\textit{Variant C (consequence-trained)}" in tex, tex
    assert "C1 state gate, consequence-trained (lr 1e-4)" in md and "C1 dose-matched constant a = 0.720" in md, md
    jt = json.load(open(os.path.join(ctdir, "wgate_cg6.json")))
    assert [row["arm"] for row in jt["rows"]] == ["augfull_lr1e5", "cg1_head", "cg2_head", "cg1_const"], jt["rows"]
    assert all(row["group"] == "C" for row in jt["rows"][1:]) and jt["common_n"] == 6
    print("  variant C OK (4 arm jsons, table rows + heading, weight/sigma attach dispatch, eval mode, control path)")


def test_table(tmp):
    base_eval, pilot, tdir = (os.path.join(tmp, k) for k in ("base", "tpilot", "table"))
    scenes = [f"s{i:02d}" for i in range(20)]
    rng = np.random.default_rng(1)

    def vals(off):
        return {"absrel": 0.13 + off, "a1": 0.9 - off, "ate": 0.07 + off, "rpe_trans": 0.006 + off, "rpe_rot": 0.8 + off}
    for s in scenes:
        write_eval_csv(base_eval, s, vals(rng.normal() * 1e-3))
    arms = {"fg_head": scenes, "mg_head": scenes[:19], "mg_const0": scenes[5:6], "fg_oracle": scenes}
    for arm, sc in arms.items():
        for s in sc:
            write_eval_csv(os.path.join(pilot, arm, "eval"), s, vals(-0.002 + rng.normal() * 1e-3))
    # oracle arm: per-scene control.json lacking two scenes (they ran plain) + one in the sidecar
    os.makedirs(os.path.join(pilot, "fg_oracle"), exist_ok=True)
    json.dump({s: {"frame_gate": {"alpha": {"1": 1.0}}} for s in scenes[2:]}, open(os.path.join(pilot, "fg_oracle", "control.json"), "w"))
    sk = os.path.join(EP, "maks_arm_fg_oracle.skipped.txt")
    had = os.path.isfile(sk)
    if not had:
        open(sk, "w").write(scenes[18] + "\n")
    lst = os.path.join(tmp, "tlist.txt")
    open(lst, "w").write("\n".join(scenes) + "\n")
    cmd = [sys.executable, os.path.join(EP, "wgate_table.py"), "--scene_list", lst, "--pilot_dir", pilot,
           "--base_eval", base_eval, "--table_dir", tdir, "--subset", "t20", "--no_pdf", "--min_frac", "0.8"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=200)
        assert r.returncode == 0, r.stdout + r.stderr
        j = json.load(open(os.path.join(tdir, "wgate_t20.json")))
        # min_frac .8 -> 16/20 admits mg_head (19) and fg_oracle (17 after the 3 plain scenes), not mg_const0 (1);
        # common = 20 - s19 (mg_head) - s00,s01 (oracle: no control entry) - s18 (oracle sidecar, when ours)
        exp = 17 if had else 16
        assert j["common_n"] == exp and len(j["dropped_scenes"]) == 20 - exp, (j["common_n"], j["dropped_scenes"])
        shown = {row["arm"]: row for row in j["rows"]}
        assert set(shown) == {"augfull_lr1e5", "fg_head", "mg_head", "fg_oracle"}, sorted(shown)
        assert all(row["n"] == exp and not row["partial"] for row in j["rows"]), [(r_["arm"], r_["n"]) for r_ in j["rows"]]
        sk_arms = {d["arm"]: d["reason"] for d in j["skipped"]}
        assert sk_arms["mg_const0"] == "1/20 scored" and sk_arms["fg_const"] == "missing", sk_arms
        # every contract row except the four fabricated arms is reported: mg_const0 as incomplete, the rest missing
        scored = {"fg_head", "mg_head", "fg_oracle", "mg_const0"}
        expect_missing = {a for a in wt.NAMES if a not in scored}
        assert set(sk_arms) == expect_missing | {"mg_const0"}, (sorted(sk_arms), sorted(expect_missing))
        assert all(sk_arms[a] == "missing" for a in expect_missing), sk_arms
        assert {"fg_head_rel", "mg_oracle_rel", "cg1_head", "cg2_head_lr5", "cg1_const", "cg2_const"} <= expect_missing
        assert [a for a in wt.NAMES if a.endswith("_rel")] == ["fg_head_rel", "fg_const_rel", "fg_oracle_rel", "mg_head_rel", "mg_const_rel", "mg_oracle_rel"]
        md = open(os.path.join(tdir, "wgate_t20.md")).read()
        tex = open(os.path.join(tdir, "wgate_t20.tex")).read()
        assert f"Common scene set: {exp}/20" in md and "Rows not shown: " in md and "mg_const0 (1/20 scored)" in md
        assert f"common scene set {exp}/20" in tex and "mg\\_const0 (1/20 scored)" in tex
        # --allow_partial: mg_const0 shown with the dagger on its own n (1), common set unchanged
        r = subprocess.run(cmd + ["--allow_partial"], capture_output=True, text=True, timeout=200)
        assert r.returncode == 0, r.stdout + r.stderr
        j = json.load(open(os.path.join(tdir, "wgate_t20.json")))
        assert j["common_n"] == exp
        row = {row["arm"]: row for row in j["rows"]}["mg_const0"]
        assert row["partial"] and row["n"] == 1, row
        assert "(dagger)" in open(os.path.join(tdir, "wgate_t20.md")).read()
        # the footnoted tex must still build (pdflatex + gs, when available on this node)
        import shutil
        env_path = wt.LATEX_BIN + os.pathsep + os.environ.get("PATH", "")
        if shutil.which("pdflatex", path=env_path) and shutil.which("gs", path=env_path):
            r = subprocess.run([c for c in cmd if c != "--no_pdf"], capture_output=True, text=True, timeout=250)
            assert r.returncode == 0 and os.path.isfile(os.path.join(tdir, "wgate_t20.png")), r.stdout + r.stderr
            print("  table tex -> pdf -> png OK")
        else:
            print("  (pdflatex/gs not on this node: tex build not checked)")
    finally:
        if not had:
            os.remove(sk)
    print("  table OK (common set", exp, "/ 20)")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="wgate_selftest_") as tmp:
        out = test_generator(tmp)
        test_worker(out)
        test_variant_c(tmp)
        test_table(tmp)
    print("wgate_selftest: ALL OK")
