#!/bin/bash

# set visible GPUs
GPU_ID="1,2,3,4"

# dataset settings
TEMPLATE_VERSION="json-v9" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
DS_VERSION1="smid_and_crawled_v2_policy"
DS_VERSION2="smid_and_crawled_v2_with_augmented_policies"
PROMPT_VERSION=v1

# model settings
MODEL_VERSION1="liuhaotian/llava-v1.5-13b" # the model version to use for training
MODEL_VERSION2="liuhaotian/llava-v1.5-7b" # the model version to use for training

MODEL_OUTPUT_DIR1_1="/common-repos/LlavaGuard/models/LlavaGuard-v1.1-13b-full/${DS_VERSION1}/${TEMPLATE_VERSION}"
MODEL_OUTPUT_DIR1_2="/common-repos/LlavaGuard/models/LlavaGuard-v1.1-13b-full/${DS_VERSION2}/${TEMPLATE_VERSION}"
MODEL_OUTPUT_DIR2_1="/common-repos/LlavaGuard/models/LlavaGuard-v1.1-7b-full/${DS_VERSION1}/${TEMPLATE_VERSION}"
MODEL_OUTPUT_DIR2_2="/common-repos/LlavaGuard/models/LlavaGuard-v1.1-7b-full/${DS_VERSION2}/${TEMPLATE_VERSION}"

data_path1="/common-repos/LlavaGuard/data/${DS_VERSION1}/${TEMPLATE_VERSION}"
data_path2="/common-repos/LlavaGuard/data/${DS_VERSION2}/${TEMPLATE_VERSION}"
#data_path_train_oversampled="${data_path}/train_oversampled.json"
#data_path_eval="${data_path}/eval.json"
#data_path_no_train="None" # disable evaluation on train data (optional)

# remove previous runs if they exist otherwise it will skip the training for existing runs
#rm -rf $MODEL_OUTPUT_DIR

zero="/LLaVA/scripts/zero3.json"
zero_quant="llavaguard/zero/zero3_quant.json"
zero_offload="/LLaVA/scripts/zero3_offload.json"


# LlavaGuard-v1.1 13b model training
deepspeed --include="localhost:${GPU_ID}" \
    train.py \
    --deepspeed $zero_offload \
    --model_name_or_path $MODEL_VERSION1 \
    --version $PROMPT_VERSION \
    --data_path "${data_path2}/train_oversampled.json" \
    --data_path_eval "${data_path2}/eval.json" \
    --image_folder /common-repos \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $MODEL_OUTPUT_DIR1_2 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 3 \
    --evaluation_strategy "no" \
    --eval_steps 50 \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

#CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path $data_path1 \
#    --model_base $MODEL_OUTPUT_DIR1_1

CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
    --data_path $data_path2 \
    --model_base $MODEL_OUTPUT_DIR1_2


# LlavaGuard-v1.1 7b model training
#deepspeed --include="localhost:${GPU_ID}" \
#    train.py \
#    --deepspeed /LLaVA/scripts/zero3.json \
#    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#    --model_name_or_path $MODEL_VERSION2 \
#    --version $PROMPT_VERSION \
#    --data_path "${data_path1}/train_oversampled.json" \
#    --data_path_eval "${data_path1}/eval.json" \
#    --image_folder /common-repos \
#    --vision_tower openai/clip-vit-large-patch14-336 \
#    --mm_projector_type mlp2x_gelu \
#    --mm_vision_select_layer -2 \
#    --mm_use_im_start_end False \
#    --mm_use_im_patch_token False \
#    --image_aspect_ratio pad \
#    --group_by_modality_length True \
#    --bf16 True \
#    --output_dir $MODEL_OUTPUT_DIR2_1 \
#    --num_train_epochs 2 \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 4 \
#    --gradient_accumulation_steps 1 \
#    --evaluation_strategy "steps" \
#    --eval_steps 50 \
#    --save_strategy "steps" \
#    --save_steps 50 \
#    --save_total_limit 5 \
#    --learning_rate 2e-5 \
#    --weight_decay 0. \
#    --warmup_ratio 0.05 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --tf32 True \
#    --model_max_length 4096 \
#    --gradient_checkpointing True \
#    --dataloader_num_workers 4 \
#    --lazy_preprocess True \
#    --report_to wandb
#
#  CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path $data_path1 \
#    --model_base $MODEL_OUTPUT_DIR2_1
#  CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path $data_path1 \
#    --model_base $MODEL_OUTPUT_DIR2_1