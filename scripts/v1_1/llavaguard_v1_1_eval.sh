#!/bin/bash

# set visible GPU id to 6
GPU_ID="5,6,7"

# dataset settings
TEMPLATE_VERSION="json-v9" # (json, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
DS_VERSION1="smid_and_crawled_v2_policy"
DS_VERSION2="smid_and_crawled_v2_with_augmented_policies"

# model settings
MODEL_15_7="liuhaotian/llava-v1.5-7b" # the model version to use for training
MODEL_15_13="liuhaotian/llava-v1.5-13b" # the model version to use for training
MODEL_16_13="liuhaotian/llava-v1.6-vicuna-13b" # the model name to use for training
MODEL_16_34="liuhaotian/llava-v1.6-34b" # the model version to use for training

# choose trained LORA adapter to evaluate
LlavaGuard_v11_7b="/common-repos/LlavaGuard/models/LlavaGuard-v1.1-7b/${DS_VERSION2}/${TEMPLATE_VERSION}"
LlavaGuard_v10_13b="/common-repos/LlavaGuard/models/LlavaGuard-v1.1-13b/${DS_VERSION1}/${TEMPLATE_VERSION}"
LlavaGuard_v11_13b="/common-repos/LlavaGuard/models/LlavaGuard-v1.1-13b/${DS_VERSION2}/${TEMPLATE_VERSION}"

NO_LORA="None" # disable LORA (optional)

# updating paths for training and evaluation (do not change)
data_path_eval="/common-repos/LlavaGuard/data/${DS_VERSION1}/${TEMPLATE_VERSION}/eval.json"
data_path_eval_policy_augmentation="/common-repos/LlavaGuard/data/${DS_VERSION2}/${TEMPLATE_VERSION}"
data_path_no_train="None" # disable evaluation on train data (optional)



#################################### LlavaGuard-v1.1 evaluation####################################

############### default policy ################
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path_eval $data_path_eval \
#    --data_path_train $data_path_no_train \
#    --model_base $MODEL_15_7 \
#    --lora_dir $LlavaGuard_v11_7b

#CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path_eval $data_path_eval \
#    --data_path_train $data_path_no_train \
#    --model_base $MODEL_15_13 \
#    --lora_dir $LlavaGuard_v11_13b


############### augmented policies ################
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path_eval $data_path_eval_policy_augmentation \
#    --data_path_train $data_path_no_train \
#    --model_base $MODEL_15_13 \
#    --lora_dir $LlavaGuard_v11_7b


CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
    --data_path $data_path_eval_policy_augmentation \
    --model_base $MODEL_15_13 \
    --lora_dir $LlavaGuard_v11_13b

CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
    --data_path $data_path_eval_policy_augmentation \
    --model_base $MODEL_15_13 \
    --lora_dir $LlavaGuard_v10_13b


