#!/bin/bash
# bf16-vs-fp32 recording check (WRITE_GATE_VARIANTS.md, Component B fix pass): the eval worker rolls the model out
# in fp32 (dust3r/inference.py, autocast off) while the contract's recorder used bf16 autocast. Records the SAME
# 2 train + 2 heldout smoke episodes twice (--fp32 and --bf16) into $PREC_ROOT/{fp32,bf16} and reports, per episode,
# the feature differences (frame_feat / pose_feat, relative to their frame-to-frame spread) and the head-output
# differences (log-variances s_t / a_t / b_t of the smoke heads or the trained heads) between the two recordings.
# Read-out: if the bf16-vs-fp32 head-output gap is a small fraction of the across-frame spread, either precision
# would do; the recorder's default is fp32 regardless (exact train/serve match).
#   sbatch -A etur59 -p acc -q acc_debug --gres=gpu:1 -c 20 --time=00:40:00 eval_pipeline/record_wgate_precision_check.sbatch
#   HEADS_DIR=/gpfs/scratch/etur59/koc821022/checkpoints/wgate sbatch ...   # use the trained heads instead of the smoke ones
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:40:00
#SBATCH --job-name=wg_prec
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_wg_prec_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_wg_prec_%j.err
set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMEXPR_MAX_THREADS=16
PREC_ROOT=${PREC_ROOT:-/gpfs/scratch/etur59/koc821022/checkpoints/wgate/features_prec}
HEADS_DIR=${HEADS_DIR:-/gpfs/scratch/etur59/koc821022/checkpoints/wgate/smoke}
CKPT=${CKPT:-$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth}
mn5_require f "$CKPT" f "$CUT3R_DIR/src/record_wgate_features.py" f "$CUT3R_DIR/src/dust3r/wgate/heads.py"
mkdir -p "$PREC_ROOT"
echo "Node $(hostname) root=$PREC_ROOT heads=$HEADS_DIR start $(date)"
for PREC in fp32 bf16; do
  echo "== RECORD $PREC"; T0=$(date +%s)
  python "$CUT3R_DIR/src/record_wgate_features.py" --ckpt "$CKPT" --root "$PREC_ROOT/$PREC" --smoke --num_workers 8 --overwrite --$PREC \
    || { echo "RECORD $PREC FAILED rc=$?"; exit 1; }
  echo "$PREC done in $(( $(date +%s) - T0 )) s"
done
export CUT3R_DIR; echo "== COMPARE"
python - "$PREC_ROOT" "$HEADS_DIR" <<'PY' || { echo "COMPARE FAILED rc=$?"; exit 1; }
import sys, os, glob, json, numpy as np, torch
root, heads_dir = sys.argv[1:3]
sys.path.insert(0, os.path.join(os.environ["CUT3R_DIR"], "src"))
from dust3r.wgate.heads import FrameConfHead, PoseSigmaHead, sigma_to_weight, pose_sigma_to_weight
from train_wgate_heads import load_pose_decoder
fg = torch.load(os.path.join(heads_dir, "frame_conf_head.pth"), map_location="cpu", weights_only=False)
mg = torch.load(os.path.join(heads_dir, "pose_sigma_head.pth"), map_location="cpu", weights_only=False)
fh = FrameConfHead(**fg.get("head_cfg", {"in_dim": 1796, "hidden": 256})); fh.load_state_dict(fg["state_dict"]); fh.eval()
pd, _ = load_pose_decoder(os.path.join(root, "fp32", "pose_decoder.pth"))
ph = PoseSigmaHead(pd).load_head_state_dict(mg["state_dict"]); ph.eval()
def rel(a, b):  # max |a-b| over frames t>0, relative to the across-frame std of a (per feature, then median)
    d = np.abs(a[1:] - b[1:]).max(0); s = a[1:].std(0) + 1e-8
    return float(np.median(d / s)), float((d / s).max())
summary = {}
for split in ("train", "heldout"):
    for f32 in sorted(glob.glob(os.path.join(root, "fp32", "features", split, "*.npz"))):
        f16 = f32.replace("/fp32/", "/bf16/"); assert os.path.isfile(f16), f16
        a, b = np.load(f32, allow_pickle=True), np.load(f16, allow_pickle=True)
        assert str(a["scene"]) == str(b["scene"]) and str(a["first_frame"]) == str(b["first_frame"]), (f32, f16)
        assert str(a["precision"]) == "fp32" and str(b["precision"]) == "bf16"
        row = {"scene": str(a["scene"]), "first_frame": str(a["first_frame"])}
        row["frame_feat_rel(med,max)"] = rel(a["frame_feat"].astype(np.float32), b["frame_feat"].astype(np.float32))
        row["pose_feat_rel(med,max)"] = rel(a["pose_feat"].astype(np.float32), b["pose_feat"].astype(np.float32))
        for k in ("res_pose_t", "res_pose_R", "res_pts_self"):
            row[f"{k}(max|d|, spread)"] = (float(np.abs(a[k][1:] - b[k][1:]).max()), float(a[k][1:].std()))
        with torch.no_grad():
            s32 = fh(torch.from_numpy(a["frame_feat"].astype(np.float32))).numpy(); s16 = fh(torch.from_numpy(b["frame_feat"].astype(np.float32))).numpy()
            p32 = ph(torch.from_numpy(a["pose_feat"].astype(np.float32))); p16 = ph(torch.from_numpy(b["pose_feat"].astype(np.float32)))
            a32 = sigma_to_weight(torch.from_numpy(s32), fg["log_ref"], fg["wmin"]).numpy(); a16 = sigma_to_weight(torch.from_numpy(s16), fg["log_ref"], fg["wmin"]).numpy()
            lr = torch.tensor([mg["log_ref_t"], mg["log_ref_R"]])
            b32 = pose_sigma_to_weight(p32, lr, mg["wmin"]).numpy(); b16 = pose_sigma_to_weight(p16, lr, mg["wmin"]).numpy()
        row["s_frame(max|d|, spread)"] = (float(np.abs(s32[1:] - s16[1:]).max()), float(s32[1:].std()))
        row["s_pose_t(max|d|, spread)"] = (float((p32[1:, 0] - p16[1:, 0]).abs().max()), float(p32[1:, 0].std()))
        row["s_pose_R(max|d|, spread)"] = (float((p32[1:, 1] - p16[1:, 1]).abs().max()), float(p32[1:, 1].std()))
        row["a_t(max|d|, spread)"] = (float(np.abs(a32[1:] - a16[1:]).max()), float(a32[1:].std()))
        row["b_t(max|d|, spread)"] = (float(np.abs(b32[1:] - b16[1:]).max()), float(b32[1:].std()))
        summary[f"{split}/{os.path.basename(f32)}"] = row
        print(f"[prec] {split}/{os.path.basename(f32)} {row['scene']}@{row['first_frame']}: " + " ".join(f"{k}={tuple(round(x, 5) for x in v)}" for k, v in row.items() if isinstance(v, tuple)))
json.dump(summary, open(os.path.join(root, "precision_compare.json"), "w"), indent=1)
print("[prec] wrote", os.path.join(root, "precision_compare.json"))
PY
echo "done $(date)"
