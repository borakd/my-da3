#!/usr/bin/env python
"""Graft a trained PoseGRU (donor ckpt) onto a trunk checkpoint (phase B1).

Copies every module.pose_gru.* tensor and every pose_gru*/refine arg from the
donor into the target, so load_model's sniff enables the GRU on the grafted
checkpoint and the harness can run --conditioning prev_pred_gru. Also sets
feed_prev_pred=True in args (the GRU's deployment mode). Zero training.

Usage: python graft_pose_gru_ckpt.py <trunk.pth> <donor.pth> <out.pth>
"""
import sys

import torch


def main():
    trunk_p, donor_p, out_p = sys.argv[1:4]
    trunk = torch.load(trunk_p, map_location="cpu", weights_only=False)
    donor = torch.load(donor_p, map_location="cpu", weights_only=False)

    keys = [k for k in donor["model"] if ".pose_gru." in k]
    assert keys, "donor has no pose_gru tensors"
    for k in keys:
        trunk["model"][k] = donor["model"][k].clone()

    # args are OmegaConf DictConfigs: enumerate via keys(), assign via
    # open_dict so struct mode cannot silently drop new keys.
    from omegaconf import OmegaConf, open_dict
    ta, da = trunk["args"], donor["args"]
    moved = []
    with open_dict(ta):
        for name in da.keys():
            if str(name).startswith("pose_gru"):
                ta[name] = da[name]
                moved.append(str(name))
        ta["feed_prev_pred"] = True
        ta["feed_gt_ray_map"] = False
        ta["feed_prev_gt_ray_map"] = False
    assert "pose_gru_refine_passes" in moved and "pose_gru" in moved, (
        f"graft moved {len(moved)} args but the load-bearing keys are missing")

    torch.save(trunk, out_p)
    print(f"grafted {len(keys)} tensors, args: {sorted(moved)}")
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
