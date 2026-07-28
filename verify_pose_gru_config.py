"""Step-5 verification: captain_gru_v2.yaml composes, resolves, and drives the
trainer wiring as intended. CPU-only, no model build, no data touch.

Run from the worktree root:
    PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src" \
        conda run -n cuteanything python verify_pose_gru_config.py
"""
import os

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = os.path.abspath("src/CUT3R/config")
PASS = []


def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name)
    print(f"  PASS  {name}")


def load(name, overrides=()):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=name, overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg


print("== 1. composes + resolves ==")
cfg = load("captain_gru_v2")
check("hydra compose + resolve OK", True)

print("== 2. trainer-visible keys (same getattr reads train_cut3r_baseline does) ==")
check("pose_gru True", bool(cfg.pose_gru) is True)
check("feed_prev_pred True (assert in trainer passes)", bool(cfg.feed_prev_pred) is True)
check("mode residual", str(cfg.pose_gru_mode) == "residual")
check("hidden_dim 128", int(cfg.pose_gru_hidden_dim) == 128)
check("lr_scale 100", float(cfg.pose_gru_lr_scale) == 100.0)
check("dataset ray flags off", cfg.feed_gt_ray_map is False and cfg.feed_prev_gt_ray_map is False)
check("gru term in train_criterion", "1.0*PoseGRULoss()" in cfg.train_criterion)
check("gru term in test_criterion", "1.0*PoseGRULoss()" in cfg.test_criterion)
check("no gru term in extra 'orig' diagnostic", "PoseGRULoss" not in cfg.extra_test_criteria["orig"])

print("== 3. criterion strings eval in the trainer's namespace ==")
from accelerate import PartialState

PartialState()
import dust3r.heads  # noqa: F401  import-order gotcha
from dust3r.losses import *  # noqa: F401,F403

train_crit = eval(cfg.train_criterion)
test_crit = eval(cfg.test_criterion)
check("train criterion builds", "PoseGRULoss" in repr(train_crit))
check("test criterion builds", "PoseGRULoss" in repr(test_crit))

print("== 4. CLI overrides ==")
cfg_w = load("captain_gru_v2", ["pose_gru_loss_weight=0.25"])
check("weight override reaches both strings",
      "0.25*PoseGRULoss()" in cfg_w.train_criterion and "0.25*PoseGRULoss()" in cfg_w.test_criterion)
crit_w = eval(cfg_w.train_criterion)
check("weighted criterion builds (alpha in repr)", "0.25*PoseGRULoss" in repr(crit_w))
cfg_off = load("captain_gru_v2", ["pose_gru=False", "pose_gru_loss_weight=0.0"])
check("pose_gru off override", bool(cfg_off.pose_gru) is False)
check("loss term zeroed", "0.0*PoseGRULoss()" in cfg_off.train_criterion)
cfg_dir = load("captain_gru_v2", ["pose_gru_mode=direct", "pose_gru_hidden_dim=256"])
check("mode/hidden sweep overrides", cfg_dir.pose_gru_mode == "direct" and cfg_dir.pose_gru_hidden_dim == 256)

print("== 5. diff vs captain_ray_prev_pred: only intended keys change ==")
base = load("captain_ray_prev_pred")
new_keys = set(cfg.keys()) - set(base.keys())
check(
    "added keys are exactly the pose_gru block",
    new_keys == {"pose_gru", "pose_gru_mode", "pose_gru_hidden_dim", "pose_gru_lr_scale", "pose_gru_loss_weight"},
)
changed = {
    k for k in base.keys()
    if OmegaConf.to_container(OmegaConf.create({"v": base[k]})) != OmegaConf.to_container(OmegaConf.create({"v": cfg[k]}))
}
check(
    "changed keys are exactly criteria + exp identity",
    changed == {"train_criterion", "test_criterion", "exp_name", "logdir", "output_dir"},
)
check("output dir separated from prev_pred run", "captain_gru_v2" in cfg.output_dir and cfg.output_dir != base.output_dir)
check("same pretrained ckpt as base", cfg.pretrained == base.pretrained)
check("same data roots as base", cfg.train_data == base.train_data and cfg.test_data == base.test_data)

print(f"\nALL {len(PASS)} CHECKS PASSED")
