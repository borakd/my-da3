"""Save the UNTRAINED conf branch (identity to the plain head) so the parity arm can be scored."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dust3r.utils.path_to_croco  # noqa
from dust3r.model import ARCroco3DStereo
from train_unc_gate import CKPT_DEFAULT
out = sys.argv[1] if len(sys.argv) > 1 else "/gpfs/scratch/etur59/koc821022/checkpoints/unc_gate/cb_init/conf_branch.pth"
model = ARCroco3DStereo.from_pretrained(CKPT_DEFAULT)
br = model.attach_conf_branch(None, train=False)
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save({"conf": br.state_dict(), "args": {"note": "untrained, identity to the plain self-view head"}, "step": 0, "seen": 0}, out)
print("saved", out, sum(p.numel() for p in br.parameters()) / 1e6, "M params")
