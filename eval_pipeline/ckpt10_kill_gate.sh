#!/bin/bash
# Step-8 kill gate, one shot. Run from repo root when checkpoint-10.pth is
# settled. Submits the paired clean+shuffle probe jobs on checkpoint-10 and
# prints the weight-norm guard immediately.
set -eo pipefail
# Pre-2026-08-14 run, so checkpoints_projects/ (the old /gpfs/projects root is
# gone). A run started after that date lives under .../checkpoints/ instead.
RUN=${RUN:-/gpfs/scratch/etur59/koc821022/checkpoints_projects/captain_cut3r_finetune_aug_full/captain_gru_v3_a4_g3_r8_refine_finetune}
CK=$RUN/checkpoint-10.pth
[ -f "$CK" ] || { echo "no checkpoint-10 yet"; exit 1; }
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
python - "$CK" <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["model"]
k = [x for x in sd if "pose_gru" in x and "weight_ih" in x][0]
n = sd[k][:, 14:21].norm().item()
print(f"ckpt10 probe-block norm: {n:.4f}  GATE(i): {'PASS' if n > 0.01 else 'FAIL'}")
PY
sbatch --qos=acc_ehpc --job-name=p43_ck10_clean \
  --export="ALL,P43_GRU_CKPT=$CK,P43_OUT=$PWD/p43_ck10_clean.json" probe_p43_premise.sbatch
sbatch --qos=acc_ehpc --job-name=p43_ck10_shuf \
  --export="ALL,P43_GRU_CKPT=$CK,P43_OUT=$PWD/p43_ck10_shuffle.json,P43_ALLOW_CONTAMINATION=1,POSE_GRU_PROBE_SHUFFLE=1" probe_p43_premise.sbatch
echo "submitted clean+shuffle probe pair on checkpoint-10"
