#!/bin/bash

# set visible GPU id to 6
GPU_ID="6"

# dataset settings
TEMPLATE_VERSION="json-v9" # (json, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
DS_VERSION1="smid_and_crawled_v2_policy"
DS_VERSION2="smid_and_crawled_v2_with_augmented_policies"

# model settings
MODEL_15_7="liuhaotian/llava-v1.5-7b" # the model version to use for training
MODEL_15_13="liuhaotian/llava-v1.5-13b" # the model version to use for training
MODEL_16_13="liuhaotian/llava-v1.6-vicuna-13b" # the model name to use for training
MODEL_16_34="liuhaotian/llava-v1.6-34b" # the model version to use for training
NO_LORA="None" # disable LORA (optional)

# updating paths for training and evaluation (do not change)
data_path="/common-repos/LlavaGuard/data/${DS_VERSION1}/${TEMPLATE_VERSION}"
data_path_policy_augmentation="/common-repos/LlavaGuard/data/${DS_VERSION2}/${TEMPLATE_VERSION}"

#data_path_train="/common-repos/LlavaGuard/data/${DS_VERSION}/${TEMPLATE_VERSION}/train.json"
#data_path_all_data="/common-repos/LlavaGuard/data/${DS_VERSION}/${TEMPLATE_VERSION}/all_data.json"
#data_path_eval_v2="/common-repos/LlavaGuard/data/${DS_VERSION}/${TEMPLATE_VERSION}/eval_no_edge_cases.json"
#data_path_train_v2="/common-repos/LlavaGuard/data/${DS_VERSION}/${TEMPLATE_VERSION}/train_no_edge_cases.json"

################ default policy ################
# evaluate the foundation models on the evaluation dataset
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path $data_path \
#    --model_base $MODEL_15_7 \
#    --lora_dir $NO_LORA
#
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path $data_path \
#    --model_base $MODEL_15_13 \
#    --lora_dir $NO_LORA

#CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path $data_path \
#    --model_base $MODEL_16_34 \
#    --lora_dir $NO_LORA

#CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path_eval $data_path \
#    --model_base $MODEL_16_13 \
#    --lora_dir $NO_LORA

################ augmented policies ################
# evaluate the foundation models on the evaluation dataset
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path $data_path_policy_augmentation \
#    --model_base $MODEL_15_7

CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
    --data_path $data_path_policy_augmentation \
    --model_base $MODEL_15_13

#CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path $data_path_policy_augmentation \
#    --model_base $MODEL_16_34
#
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path $data_path_policy_augmentation \
#    --model_base $MODEL_16_13