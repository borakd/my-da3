# Track-On on Leonardo (CINECA)

Everything cluster-specific lives in this folder (`vggt_probe/trackon_stage/` of the my-da3 repo,
branch `vggt_features`; folded here from `~/track_on/leonardo/` on 2026-10-06). The upstream Track-On
clone at `$TRACKON_ROOT` (`~/track_on`) is untouched and is only used for its `model/` and `utils/` packages.

## One-time setup (already done)
- conda env `track_on_r`: Python 3.12, torch 2.4.1+cu121, mmcv 2.2.0 (prebuilt cu121/torch2.4
  wheel, works on A100 sm80), numpy<2, repo `requirements.txt`, plus matplotlib / opencv / pyyaml.
- Checkpoints in `/leonardo_work/AIFAC_S07_110/bora/checkpoints/track_on/`:
  - `track_on_r.pt` — **Track-On-R** (Kubric + real-world fine-tuning). Default (`$TRACKON_CKPT`).
    Best on RoboTAP (82.6 vs 80.5 for Track-On2), the benchmark closest to DROID exterior views.
  - `trackon2_dinov3.pt` — Track-On2 (Kubric only), `$TRACKON2_CKPT`. Same architecture; use
    `--ckpt $TRACKON2_CKPT` for an A/B on your own clips.

## Still required once: DINOv3 backbone (gated on Hugging Face)
The checkpoint does not contain the DINOv3 weights. On a **login node**:
```bash
source ~/vggt_features/vggt_probe/trackon_stage/env.sh
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
hf auth login            # token from https://huggingface.co/settings/tokens
# request access first at https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov3-vits16plus-pretrain-lvd1689m')"
```
This caches the weights under `$HF_HOME` (= `$WORK/bora/hf_cache`), which compute nodes read offline.

## Run
```bash
# causal tracking, 10x10 grid at frame 0, outputs tracks.npz + tracks.mp4
sbatch $TRACKON_STAGE/track.sbatch --video /path/to/video.mp4 --grid 10 --output-dir outputs/run1

# specific points: "x,y" (frame 0) or "x,y,t"
sbatch $TRACKON_STAGE/track.sbatch --video v.mp4 --points "190,190;300,120,15" --output-dir outputs/run2

# authors' demo on media/sample.mp4 (wrapper kept in ~/track_on/leonardo/, not in this repo)
sbatch ~/track_on/leonardo/demo.sbatch
```
Interactive: `srun -p boost_usr_prod -A AIFAC_S07_110 --gres=gpu:1 --time=0:30:00 --pty bash`,
then `source $TRACKON_STAGE/env.sh && python $TRACKON_STAGE/causal_track.py ...`
(`$TRACKON_STAGE` = this folder; the sbatch wrappers `cd` to `$TRACKON_ROOT`, so relative `--output-dir` paths land there).

From Python:
```python
import os, sys; sys.path.insert(0, os.environ["TRACKON_STAGE"])   # after `source .../env.sh`
from causal_track import CausalTracker
tr = CausalTracker()              # uses $TRACKON_CKPT, cuda
tr.reset()
for frame in frames:              # (H, W, 3) uint8 RGB, any resolution
    pts, vis = tr.step(frame, new_points=[(x, y), ...] or None)   # (N,2), (N,) bool
```

## Sanity check without HF access
`sbatch ~/track_on/leonardo/smoke.sbatch` (wrapper kept in the clone, not in this repo) runs the whole pipeline on the GPU with a randomly initialised
DINOv3 of the right shape (tracks are meaningless; it only validates the install).

## DROID gripper tracks from the birth frame
`track_gripper_birth.py` / `track_gripper_birth.sbatch` (32-shard array) track grid points sampled inside the
RobotSeg birth-frame mask to the end of each episode; `summarize_gripper_tracks.py` builds `tracks_index.csv`.
`compare_checkpoints_track.py` + `compare_checkpoints_metrics.py` are the Track-On-R vs Track-On2 comparison,
`make_pipeline_video.py` / `make_comparison_video.py` the showcase videos. Birth frames and masks come from
`../birth_frames.py` and `../robotseg_stage/`; results live in `$WORK/bora/outputs/droid_birth_frames/`.
