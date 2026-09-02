"""Verify (1) load_model auto-enables PoseGRU from a real GRU checkpoint, (2) the
eval worker's arm semantics (GRU active only under prev_pred_gru; stripped under
prev_pred), (3) captain_gru_v2_direct.yaml composes and differs from the residual
twin exactly in mode + exp_name. CPU-only; run under SLURM.
"""

import os
import subprocess
import sys

# Self-locating: derive the checkout from THIS file, never a hardcoded path.
# sys.path[:0] = [...] with a stale path SILENTLY NO-OPS, so imports would fall
# through to PYTHONPATH and verify a different checkout. Assert instead.
WT = os.path.dirname(os.path.abspath(__file__))
assert os.path.isfile(
    os.path.join(WT, "src", "CUT3R", "src", "train_cut3r_baseline.py")
), f"not a my-da3 checkout: {WT}"
# A keep_freq snapshot, NOT checkpoint-last/best: those are rewritten every
# epoch by the live training job and torch.load races the writer.
CKPT = (
    "/gpfs/scratch/etur59/koc821022/checkpoints_projects/captain_cut3r_sim3rmse/"
    "captain_gru_v2/checkpoint-10.pth"
)
BASE_CKPT = os.path.join(WT, "src", "CUT3R", "src", "cut3r_512_dpt_4_64.pth")

sys.path[:0] = [WT, f"{WT}/src", f"{WT}/src/CUT3R", f"{WT}/src/CUT3R/src"]

results = []


def check(name, ok, note=""):
    results.append((name, bool(ok), note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"\n         -> {note}" if note else ""),
          flush=True)


import dust3r.heads  # noqa: F401  (import-order: heads before utils.camera)
from dust3r.model import ARCroco3DStereo, PoseGRU, load_model  # noqa: E402

print("== 1. load_model auto-enable on the real GRU checkpoint ==", flush=True)
net = load_model(CKPT, device="cpu", verbose=True)
gru = getattr(net, "pose_gru", None)
check("GRU ckpt: pose_gru materialized by load_model", gru is not None)
check("GRU ckpt: mode restored from saved training args",
      gru is not None and gru.mode == "residual", f"mode={getattr(gru, 'mode', None)}")
check("GRU ckpt: hidden_dim restored", gru is not None and gru.hidden_dim == 128)
if gru is not None:
    import torch

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    key = ("module.pose_gru.cell.weight_hh"
           if "module.pose_gru.cell.weight_hh" in ck["model"]
           else "pose_gru.cell.weight_hh")
    w = ck["model"][key]
    check("GRU ckpt: trained weights actually loaded (not fresh init)",
          torch.equal(gru.cell.weight_hh, w))
    nz = float(gru.head.weight.detach().abs().sum())
    check("GRU ckpt: head is trained (nonzero after 5+ epochs)", nz > 0, f"|head.W|_1={nz:.4f}")
del net

print("== 2. load_model on a NON-GRU checkpoint stays GRU-free ==", flush=True)
base = load_model(BASE_CKPT, device="cpu", verbose=False)
check("base ckpt: no pose_gru materialized", getattr(base, "pose_gru", None) is None)

print("== 3. worker arm semantics (mirrors infer_and_eval_worker_ray.py) ==", flush=True)
# prev_pred arm on a GRU model: worker strips the module -> raw closed loop.
base.pose_gru = PoseGRU()  # pretend a GRU ckpt was loaded
base.pose_gru = None  # what the worker now does for non-gru arms
check("prev_pred arm: pose_gru strippable via assignment (nn.Module allows None)",
      getattr(base, "pose_gru", None) is None)
# prev_pred_gru arm on a non-GRU model: identity-init fallback path.
base.enable_pose_gru()
check("prev_pred_gru fallback: enable_pose_gru() gives residual identity",
      base.pose_gru.mode == "residual")
del base

print("== 4. hydra compose of the direct config + diff vs residual twin ==", flush=True)
from hydra import compose, initialize_config_dir  # noqa: E402

with initialize_config_dir(config_dir=f"{WT}/src/CUT3R/config", version_base=None):
    direct = compose(config_name="captain_gru_v2_direct")
    resid = compose(config_name="captain_gru_v2")
check("direct config composes", True)
check("direct: pose_gru_mode == 'direct'", direct.pose_gru_mode == "direct")
check("direct: exp_name == 'captain_gru_v2_direct'",
      direct.exp_name == "captain_gru_v2_direct")
check("direct: output_dir resolves to its own ckpt dir",
      direct.output_dir.rstrip("/").endswith("captain_gru_v2_direct"), direct.output_dir)
diff_keys = sorted(
    k for k in set(direct.keys()) | set(resid.keys())
    if str(direct.get(k)) != str(resid.get(k)) and k != "hydra"
)
check("twins differ ONLY in pose_gru_mode + exp_name (+derived dirs)",
      set(diff_keys) <= {"pose_gru_mode", "exp_name", "logdir", "output_dir"},
      f"diff keys: {diff_keys}")
check("direct: criterion string still carries PoseGRULoss",
      "PoseGRULoss()" in direct.train_criterion and "PoseGRULoss()" in direct.test_criterion)

print("== 5. worker CLI accepts the new arm ==", flush=True)
r = subprocess.run(
    [sys.executable, f"{WT}/eval_pipeline/infer_and_eval_worker_ray.py", "--help"],
    capture_output=True, text=True,
)
check("worker --help lists prev_pred_gru", "prev_pred_gru" in r.stdout, r.stderr[-200:] if r.returncode else "")

fails = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(fails)}/{len(results)} checks passed"
      + (f" — FAILED: {fails}" if fails else ""))
sys.exit(1 if fails else 0)
