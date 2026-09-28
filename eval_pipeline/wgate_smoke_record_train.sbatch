#!/bin/bash
# INTEGRATION SMOKE, job 1 (WRITE_GATE_VARIANTS.md): record 2 train + 2 heldout episodes with
# record_wgate_features.py --smoke into $REC_ROOT (npz land at $REC_ROOT/features/{train,heldout}/,
# the recorder's fixed layout), then in the SAME job train both heads with train_wgate_heads.py --smoke
# on that table, writing $TRAIN_OUT/{frame_conf_head,pose_sigma_head}.pth, then verify the npz arrays
# and checkpoint keys against the contract.
#   sbatch -A etur59 -p acc -q acc_debug --gres=gpu:1 -c 20 --time=00:30:00 eval_pipeline/wgate_smoke_record_train.sbatch
#SBATCH --account=etur59
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:30:00
#SBATCH --job-name=wg_smoke1
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_wg_smoke1_%j.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/logs/slurm_wg_smoke1_%j.err
set -o pipefail
WT="${WT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate cuteanything
source "$WT/eval_pipeline/mn5_paths.sh"
export PYTHONPATH="$WT/src:$CUT3R_DIR:$CUT3R_DIR/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMEXPR_MAX_THREADS=16
REC_ROOT=${REC_ROOT:-/gpfs/scratch/etur59/koc821022/checkpoints/wgate/features_smoke}
TRAIN_OUT=${TRAIN_OUT:-/gpfs/scratch/etur59/koc821022/checkpoints/wgate/smoke}
CKPT=${CKPT:-$CKPT_ROOT/cut3r_finetune_baselines/cut3r_finetune_aug_full_32gpu_lr1e5/checkpoint-final.pth}
mn5_require f "$CKPT" f "$CUT3R_DIR/src/record_wgate_features.py" f "$CUT3R_DIR/src/train_wgate_heads.py" f "$CUT3R_DIR/src/dust3r/wgate/heads.py"
mkdir -p "$REC_ROOT" "$TRAIN_OUT"
echo "Node $(hostname) gpus=$(nvidia-smi -L | wc -l) rec_root=$REC_ROOT train_out=$TRAIN_OUT start $(date)"
echo "== RECORD"; T0=$(date +%s)
python "$CUT3R_DIR/src/record_wgate_features.py" --ckpt "$CKPT" --root "$REC_ROOT" --smoke --num_workers 8 --overwrite \
  || { echo "RECORD FAILED rc=$?"; exit 1; }
echo "record done in $(( $(date +%s) - T0 )) s; train npz: $(ls "$REC_ROOT/features/train" | grep -c npz) heldout npz: $(ls "$REC_ROOT/features/heldout" | grep -c npz)"
echo "== TRAIN"; T0=$(date +%s)
python "$CUT3R_DIR/src/train_wgate_heads.py" --features "$REC_ROOT/features" --out "$TRAIN_OUT" \
  --pose_decoder "$REC_ROOT/pose_decoder.pth" --ckpt "$CKPT" --heads both --smoke \
  || { echo "TRAIN FAILED rc=$?"; exit 1; }
echo "train done in $(( $(date +%s) - T0 )) s"
echo "== VERIFY (contract shapes/dtypes/keys)"
python - "$REC_ROOT" "$TRAIN_OUT" <<'PY' || { echo "VERIFY FAILED rc=$?"; exit 1; }
import sys, os, glob, numpy as np, torch
rec, out = sys.argv[1:3]
want = {"frame_feat": ((64, 1796), np.float16), "pose_feat": ((64, 768), np.float16),
        "camera_pose": ((64, 7), np.float32), "res_pose_t": ((64,), np.float32),
        "res_pose_R": ((64,), np.float32), "res_pts_self": ((64,), np.float32)}
for split in ("train", "heldout"):
    files = sorted(glob.glob(os.path.join(rec, "features", split, "*.npz")))
    assert len(files) == 2, (split, files)
    for f in files:
        z = np.load(f, allow_pickle=True)
        for k, (shp, dt) in want.items():
            assert k in z.files, (f, k, z.files)
            assert z[k].shape == shp and z[k].dtype == dt, (f, k, z[k].shape, z[k].dtype)
            assert np.isfinite(z[k].astype(np.float64)).all(), (f, k, "non-finite")
        print(f"[verify] {split}/{os.path.basename(f)} OK: " + ", ".join(f"{k}{z[k].shape}:{z[k].dtype}" for k in want)
              + f"; extra={sorted(set(z.files) - set(want))}; res medians t>0: t={np.median(z['res_pose_t'][1:]):.4f} "
              f"R={np.median(z['res_pose_R'][1:]):.4f} pts={np.median(z['res_pts_self'][1:]):.4f}; delta_norm[0]={float(z['frame_feat'][0,768]):.2f} [63]={float(z['frame_feat'][63,768]):.2f}")
fg = torch.load(os.path.join(out, "frame_conf_head.pth"), map_location="cpu", weights_only=False)
mg = torch.load(os.path.join(out, "pose_sigma_head.pth"), map_location="cpu", weights_only=False)
for need in ("state_dict", "log_ref", "wmin", "args", "metrics"):
    assert need in fg, ("frame_conf_head.pth missing", need, sorted(fg))
for need in ("state_dict", "log_ref_t", "log_ref_R", "wmin", "args", "metrics"):
    assert need in mg, ("pose_sigma_head.pth missing", need, sorted(mg))
assert set(mg["state_dict"]) == {"out.weight", "out.bias"}, sorted(mg["state_dict"])
assert tuple(mg["state_dict"]["out.weight"].shape) == (2, 3072), mg["state_dict"]["out.weight"].shape
print("[verify] frame_conf_head.pth keys:", sorted(fg), "| sd:", {k: tuple(v.shape) for k, v in fg["state_dict"].items()},
      "| log_ref=%.4f wmin=%s standin=%s" % (float(fg["log_ref"]), fg["wmin"], fg.get("standin")))
print("[verify] pose_sigma_head.pth keys:", sorted(mg), "| sd:", {k: tuple(v.shape) for k, v in mg["state_dict"].items()},
      "| log_ref_t=%.4f log_ref_R=%.4f wmin=%s standin=%s" % (float(mg["log_ref_t"]), float(mg["log_ref_R"]), mg["wmin"], mg.get("standin")))
from dust3r.wgate.heads import FrameConfHead, PoseSigmaHead
h = FrameConfHead.from_state_dict(fg["state_dict"]) if hasattr(FrameConfHead, "from_state_dict") else None
print("[verify] FrameConfHead.from_state_dict OK" if h is not None else "[verify] (no from_state_dict; skipped)")
print("[verify] ALL OK")
PY
echo "done $(date)"
