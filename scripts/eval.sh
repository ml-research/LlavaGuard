#!/bin/bash

# LlavaGuard model output directories
MODEL="AIML-TUDA/LlavaGuard-v1.2-0.5B-OV"

# Data paths
data_pth2="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/lhelff/ds/LlavaGuard/data/LlavaGuard-DS/v24/test.json"

# Device settings
DEVICE="0,1,2,3"

# Evaluation command
CUDA_VISIBLE_DEVICES=$DEVICE python3 ~/llavaguard/evaluation/evaluation.py \
 --model_dir $MODEL \
 --data_path $data_pth2 \
 --device $DEVICE \
 --engine "sglang"