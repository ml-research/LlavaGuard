#!/bin/bash

# set visible GPU id to 6
GPU_ID="2"

# dataset settings
TEMPLATE_VERSION="json-v6" # (json, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
DS_VERSION1="smid_and_crawled_policy"
DS_VERSION="smid_and_crawled_with_augmented_policies"

# model settings
MODEL_15_7="liuhaotian/llava-v1.5-7b" # the model version to use for training
MODEL_15_13="liuhaotian/llava-v1.5-13b" # the model version to use for training
MODEL_16_13="liuhaotian/llava-v1.6-vicuna-13b" # the model name to use for training
MODEL_16_34="liuhaotian/llava-v1.6-34b" # the model version to use for training

# choose trained LORA adapter to evaluate
LlavaGuard-v1-7b="/common-repos/LlavaGuard/models/llava-v1.5-7b/LORA/${DS_VERSION1}/${TEMPLATE_VERSION}"
LlavaGuard-v1-13b="/common-repos/LlavaGuard/models/llava-v1.5-13b/LORA/${DS_VERSION1}/${TEMPLATE_VERSION}"
LlavaGuard-v1.1-7b="/common-repos/LlavaGuard/models/llava-v1.5-7b/LORA/${DS_VERSION2}/${TEMPLATE_VERSION}"
LlavaGuard-v1.1-13b="/common-repos/LlavaGuard/models/llava-v1.5-13b/LORA/${DS_VERSION2}/${TEMPLATE_VERSION}"
#LORA="/common-repos/LlavaGuard/models/llava-v1.5-7b/LORA/smid_and_crawled_policy/json-v4_oversampled/checkpoint-150"

# updating paths for training and evaluation (do not change)
data_path_eval="/common-repos/LlavaGuard/data/${DS_VERSION1}/${TEMPLATE_VERSION}/eval.json"
data_path_eval_policy_augmentation="/common-repos/LlavaGuard/data/${DS_VERSION2}/${TEMPLATE_VERSION}/eval.json"
data_path_train="/common-repos/LlavaGuard/data/${DS_VERSION}/${TEMPLATE_VERSION}/train.json"
data_path_all_data="/common-repos/LlavaGuard/data/${DS_VERSION}/${TEMPLATE_VERSION}/all_data.json"
data_path_eval_v2="/common-repos/LlavaGuard/data/${DS_VERSION}/${TEMPLATE_VERSION}/eval_no_edge_cases.json"
data_path_train_v2="/common-repos/LlavaGuard/data/${DS_VERSION}/${TEMPLATE_VERSION}/train_no_edge_cases.json"
data_path_no_train="None" # disable evaluation on train data (optional)


# evaluate the fully fine-tuned LlavaGuard
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path_eval $data_path_eval \
#    --data_path_train $data_path_no_train \
#    --model_base $MODEL_OUTPUT_DIR1 \
#    --lora_dir $data_path_no_train

#CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path_eval $data_path_eval \
#    --data_path_train $data_path_no_train \
#    --model_base $MODEL_OUTPUT_DIR2 \
#    --lora_dir $data_path_no_train



