#!/bin/bash

# set visible GPUs
GPU_ID="0"

# dataset settings
TEMPLATE_VERSION="json-v6" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
DS_VERSION2="smid_and_crawled_policy"
DS_VERSION="smid_and_crawled_with_augmented_policies"
PROMPT_VERSION=v1

# model settings
MODEL_VERSION1="liuhaotian/llava-v1.5-13b" # the model version to use for training
MODEL_VERSION2="liuhaotian/llava-v1.5-7b" # the model version to use for training

MODEL_OUTPUT_DIR1="/common-repos/LlavaGuard/models/llava-v1.5-13b/LORA/${DS_VERSION}/${TEMPLATE_VERSION}"
MODEL_OUTPUT_DIR2="/common-repos/LlavaGuard/models/llava-v1.5-7b/LORA/${DS_VERSION}/${TEMPLATE_VERSION}"

data_path_eval="/common-repos/LlavaGuard/data/${DS_VERSION}/${TEMPLATE_VERSION}/eval.json"
data_path_train="/common-repos/LlavaGuard/data/${DS_VERSION}/${TEMPLATE_VERSION}/train.json"
data_path_train_oversampled="/common-repos/LlavaGuard/data/${DS_VERSION}/${TEMPLATE_VERSION}/train_oversampled.json"
data_path_no_train="None" # disable evaluation on train data (optional)

# remove previous runs if they exist otherwise it will skip the training for existing runs
#rm -rf $MODEL_OUTPUT_DIR



deepspeed --include="localhost:${GPU_ID}" \
    train.py \
    --deepspeed /LLaVA/scripts/zero3.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path $MODEL_VERSION1 \
    --version $PROMPT_VERSION \
    --data_path $data_path_train_oversampled \
    --data_path_eval $data_path_eval \
    --image_folder /common-repos \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $MODEL_OUTPUT_DIR1 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
    --data_path_eval $data_path_eval \
    --data_path_train $data_path_no_train \
    --model_base $MODEL_VERSION1 \
    --lora_dir $MODEL_OUTPUT_DIR1


#deepspeed --include="localhost:${GPU_ID}" \
#    train.py \
#    --deepspeed /LLaVA/scripts/zero3.json \
#    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#    --model_name_or_path $MODEL_VERSION2 \
#    --version $PROMPT_VERSION \
#    --data_path $data_path_train_oversampled \
#    --data_path_eval $data_path_eval \
#    --image_folder /common-repos \
#    --vision_tower openai/clip-vit-large-patch14-336 \
#    --mm_projector_type mlp2x_gelu \
#    --mm_vision_select_layer -2 \
#    --mm_use_im_start_end False \
#    --mm_use_im_patch_token False \
#    --image_aspect_ratio pad \
#    --group_by_modality_length True \
#    --bf16 True \
#    --output_dir $MODEL_OUTPUT_DIR2 \
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
#    --model_max_length 2048 \
#    --gradient_checkpointing True \
#    --dataloader_num_workers 4 \
#    --lazy_preprocess True \
#    --report_to wandb

#  CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
#    --data_path_eval $data_path_eval \
#    --data_path_train $data_path_no_train \
#    --model_base $MODEL_VERSION2 \
#    --lora_dir $LORA2