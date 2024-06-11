#!/bin/bash

TEMPLATE_VERSION16="json-v16" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
DS="smid_and_crawled_v2_with_augmented_policies"

# llavaguard
MODEL_OUTPUT_DIR1="/common-repos/LlavaGuard/models/LlavaGuard-v1.1-7b-full/${DS}/${TEMPLATE_VERSION16}"
MODEL_OUTPUT_DIR2="/common-repos/LlavaGuard/models/LlavaGuard-v1.1-13b-full/${DS}/${TEMPLATE_VERSION16}"
MODEL_OUTPUT_DIR3="/common-repos/LlavaGuard/models/LlavaGuard-v1.2-34b-full/${DS}/${TEMPLATE_VERSION16}"

# llava base model
MODEL_VERSION1="liuhaotian/llava-v1.5-7b" # the model version to use for training
MODEL_VERSION2="liuhaotian/llava-v1.5-13b" # the model version to use for training
MODEL_VERSION3="liuhaotian/llava-v1.6-34b" # the model version to use for training



data_pth1="/common-repos/LlavaGuard/data/${DS}/${TEMPLATE_VERSION16}"

## LlavaGuard
python3 /workspace/llavaguard/sglang/evaluation_wrapper.py \
  --model_dir $MODEL_OUTPUT_DIR1 \
  --data_path "$data_pth1" \
  --device 6

python3 /workspace/llavaguard/sglang/evaluation_wrapper.py \
  --model_dir $MODEL_OUTPUT_DIR2 \
  --data_path "$data_pth1" \
  --device 7

python3 /workspace/llavaguard/sglang/evaluation_wrapper.py \
  --model_dir $MODEL_OUTPUT_DIR3 \
  --data_path "$data_pth1" \
  --device 7
# Llava
python3 /workspace/llavaguard/sglang/evaluation_wrapper.py \
--model_dir $MODEL_VERSION1 \
--data_path $data_pth1 \
--device 0 \
--infer_train_data

python3 /workspace/llavaguard/sglang/evaluation_wrapper.py \
--model_dir $MODEL_VERSION2 \
--data_path $data_pth1 \
--device 7

python3 /workspace/llavaguard/sglang/evaluation_wrapper.py \
--model_dir $MODEL_VERSION3 \
--data_path $data_pth1 \
--device 7

