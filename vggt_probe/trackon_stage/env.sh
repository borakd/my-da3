# Source this on Leonardo before using Track-On:   source ~/vggt_features/vggt_probe/trackon_stage/env.sh
source ~/miniforge3/etc/profile.d/conda.sh
conda activate track_on_r

export TRACKON_ROOT=~/track_on   # upstream clone, used only for its model/ and utils/ packages
export TRACKON_STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this folder: causal_track.py + DROID gripper scripts
export TRACKON_CKPT=/leonardo_work/AIFAC_S07_110/bora/checkpoints/track_on/track_on_r.pt   # Track-On-R (default, best on RoboTAP)
export TRACKON2_CKPT=/leonardo_work/AIFAC_S07_110/bora/checkpoints/track_on/trackon2_dinov3.pt  # Track-On2, Kubric-only, for A/B

# Hugging Face cache on project storage (DINOv3 backbone is fetched here on the login node)
export HF_HOME=/leonardo_work/AIFAC_S07_110/bora/hf_cache
# Compute nodes have no internet: always resolve DINOv3 from the cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
