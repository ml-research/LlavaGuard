#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

 PROMPT_VERSION="llava_v1"
 MODEL_VERSION="liuhaotian/llava-v1.6-34b"

python3 /workspace/data_helper.py

deepspeed train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /LLaVA/scripts/zero3.json \
    --model_name_or_path $MODEL_VERSION \
    --version v1 \
    --data_path /common-repos/LlavaGuard/data/smid_and_real_images_human_feedback/train.json \
    --data_path_eval /common-repos/LlavaGuard/data/smid_and_real_images_human_feedback/eval.json \
    --image_folder /common-repos \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir /common-repos/LlavaGuard/models/naive_SMID_CRAWLED \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 15 \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
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
