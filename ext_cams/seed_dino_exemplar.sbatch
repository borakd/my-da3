#!/bin/bash
#SBATCH --job-name=dino_exemplar --account=etur59 --partition=acc --qos=acc_ehpc --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --time=01:30:00
#SBATCH --array=0-12
#SBATCH --output=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas/dino_exemplar/logs/%A_%a.out
#SBATCH --error=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas/dino_exemplar/logs/%A_%a.err
# IDEA 3 (dino_exemplar) on the 13 smoke scenes, one array task per scene:
#   1. seed_dino_exemplar.py (DINOv2 on the GPU)          -> ideas/dino_exemplar/EP/{ext1,ext2}_*__dino_exemplar_masks.npz, verified_points.npz, seed_info.json
#   2. rig_track2.py --diag --verified_points (CPU)        -> ideas/dino_exemplar/EP/rig2/
#   3. rig_track2_dino.py --diag --verified_points (GPU)   -> ideas/dino_exemplar/EP/rig2_dino/   (segment linking with DINOv2 patch features)
# usage: mkdir -p .../ideas/dino_exemplar/logs; sbatch seed_dino_exemplar.sbatch
#        SCENES=/path/to/list.txt sbatch --array=0-N seed_dino_exemplar.sbatch
# Check PRODUCTS afterwards (seedstudy_check_products.py --root ... --suffix __dino_exemplar_masks.npz; record.json per rig dir).
source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
conda activate /home/koc/koc821022/.conda/envs/cuteanything
export TORCH_HOME=/gpfs/scratch/etur59/koc821022/torch_hub XFORMERS_DISABLED=1 OMP_NUM_THREADS=4 OPENCV_FFMPEG_CAPTURE_OPTIONS="threads;1"
cd /gpfs/home/koc/koc821022/using_ext_cams/ext_cams
ROOT=/gpfs/scratch/etur59/koc821022/outputs/cut3r_eval/ext_cams/ideas/dino_exemplar
SCENES=${SCENES:-/gpfs/home/koc/koc821022/vggt_features/vggt_probe/smoke13_scenes.txt}
EP=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SCENES")
[ -n "$EP" ] || { echo "no scene at index $SLURM_ARRAY_TASK_ID in $SCENES"; exit 1; }
nvidia-smi -L
echo "task $SLURM_ARRAY_TASK_ID episode $EP start $(date)"
python seed_dino_exemplar.py --episode "$EP" --out_root "$ROOT" --png_frames 60,200,400 "$@"
echo "seeder end $(date) rc=$?"
META=/gpfs/scratch/etur59/koc821022/vggt_cache/raw/$EP/metadata_$EP.json
S1=$(python -c "import json;print(json.load(open('$META'))['ext1_cam_serial'])")
S2=$(python -c "import json;print(json.load(open('$META'))['ext2_cam_serial'])")
M1=$ROOT/$EP/ext1_${S1}__dino_exemplar_masks.npz; M2=$ROOT/$EP/ext2_${S2}__dino_exemplar_masks.npz; VP=$ROOT/$EP/verified_points.npz
ls -la "$M1" "$M2" "$VP" || { echo "seeder products missing"; exit 2; }
mkdir -p "$ROOT/$EP/rig2" "$ROOT/$EP/rig2_dino"
OMP_NUM_THREADS=1 python rig_track2.py --episode "$EP" --mask_npz_ext1 "$M1" --mask_npz_ext2 "$M2" --out "$ROOT/$EP/rig2" --diag --verified_points "$VP" > "$ROOT/$EP/rig2/rig_track2.log" 2>&1
echo "rig2 end $(date) rc=$? $(tail -n 1 $ROOT/$EP/rig2/rig_track2.log | cut -c1-300)"
OMP_NUM_THREADS=4 python rig_track2_dino.py --episode "$EP" --mask_npz_ext1 "$M1" --mask_npz_ext2 "$M2" --out "$ROOT/$EP/rig2_dino" --diag --verified_points "$VP" > "$ROOT/$EP/rig2_dino/rig_track2_dino.log" 2>&1
echo "rig2_dino end $(date) rc=$? $(tail -n 1 $ROOT/$EP/rig2_dino/rig_track2_dino.log | cut -c1-300)"
echo "end $(date)"
