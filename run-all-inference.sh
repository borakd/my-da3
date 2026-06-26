#!/usr/bin/env bash
# Run inference on ours with and without MVU and on CUT3R overfit as baseline (in that order)

# `conda activate` needs conda shell functions in non-interactive scripts
eval "$(conda shell.bash hook)"
conda activate cuteanything
cd /scratch/bdursun25/cuteanything/my-da3

# can
PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:$PYTHONPATH" python with_cut3r_v3.py \
--input_path /scratch/bdursun25/cuteanything/scenes/robomimic_can_only/dl3dv_multi/can/demo_0/dense/rgb \
--output_path /scratch/bdursun25/cuteanything/outputs/multiview_update/ours_overfit_can/final \
--cut3r_model /scratch/bdursun25/cuteanything/checkpoints/our_model_mvu/ours_overfit_can/checkpoint-final.pth
PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:$PYTHONPATH" python with_cut3r_v3.py \
--input_path /scratch/bdursun25/cuteanything/scenes/robomimic_can_only/dl3dv_multi/can/demo_0/dense/rgb \
--output_path /scratch/bdursun25/cuteanything/outputs/multiview_update/ours_overfit_can_mvu/final \
--cut3r_model /scratch/bdursun25/cuteanything/checkpoints/our_model_mvu/ours_overfit_can_mvu/checkpoint-final.pth
PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:$PYTHONPATH" python src/CUT3R/demo.py \
--model_path /scratch/bdursun25/cuteanything/checkpoints/cut3r_baselines/cut3r_overfit_can/checkpoint-final.pth \
--seq_path /scratch/bdursun25/cuteanything/scenes/robomimic_can_only/dl3dv_multi/can/demo_0/dense/rgb \
--output_dir /scratch/bdursun25/cuteanything/outputs/multiview_update/cut3r_overfit_can/final \
--disable_viewer

# lift
PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:$PYTHONPATH" python with_cut3r_v3.py \
--input_path /scratch/bdursun25/cuteanything/scenes/robomimic_lift_only/dl3dv_multi/lift/demo_0/dense/rgb \
--output_path /scratch/bdursun25/cuteanything/outputs/multiview_update/ours_overfit_lift/final \
--cut3r_model /scratch/bdursun25/cuteanything/checkpoints/our_model_mvu/ours_overfit_lift/checkpoint-final.pth
PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:$PYTHONPATH" python with_cut3r_v3.py \
--input_path /scratch/bdursun25/cuteanything/scenes/robomimic_lift_only/dl3dv_multi/lift/demo_0/dense/rgb \
--output_path /scratch/bdursun25/cuteanything/outputs/multiview_update/ours_overfit_lift_mvu/final \
--cut3r_model /scratch/bdursun25/cuteanything/checkpoints/our_model_mvu/ours_overfit_lift_mvu/checkpoint-final.pth
PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:$PYTHONPATH" python src/CUT3R/demo.py \
--model_path /scratch/bdursun25/cuteanything/checkpoints/cut3r_baselines/cut3r_overfit_lift/checkpoint-final.pth \
--seq_path /scratch/bdursun25/cuteanything/scenes/robomimic_lift_only/dl3dv_multi/lift/demo_0/dense/rgb \
--output_dir /scratch/bdursun25/cuteanything/outputs/multiview_update/cut3r_overfit_lift/final \
--disable_viewer

# square
PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:$PYTHONPATH" python with_cut3r_v3.py \
--input_path /scratch/bdursun25/cuteanything/scenes/robomimic_square_only/dl3dv_multi/square/demo_0/dense/rgb \
--output_path /scratch/bdursun25/cuteanything/outputs/multiview_update/ours_overfit_square/final \
--cut3r_model /scratch/bdursun25/cuteanything/checkpoints/our_model_mvu/ours_overfit_square/checkpoint-final.pth
PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:$PYTHONPATH" python with_cut3r_v3.py \
--input_path /scratch/bdursun25/cuteanything/scenes/robomimic_square_only/dl3dv_multi/square/demo_0/dense/rgb \
--output_path /scratch/bdursun25/cuteanything/outputs/multiview_update/ours_overfit_square_mvu/final \
--cut3r_model /scratch/bdursun25/cuteanything/checkpoints/our_model_mvu/ours_overfit_square_mvu/checkpoint-final.pth
PYTHONPATH="$PWD/src:$PWD/src/CUT3R:$PWD/src/CUT3R/src:$PYTHONPATH" python src/CUT3R/demo.py \
--model_path /scratch/bdursun25/cuteanything/checkpoints/cut3r_baselines/cut3r_overfit_square/checkpoint-final.pth \
--seq_path /scratch/bdursun25/cuteanything/scenes/robomimic_square_only/dl3dv_multi/square/demo_0/dense/rgb \
--output_dir /scratch/bdursun25/cuteanything/outputs/multiview_update/cut3r_overfit_square/final \
--disable_viewer

# tool hang
# TODO

# transport
# TODO