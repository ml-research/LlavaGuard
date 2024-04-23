#!/bin/bash

# set visible GPU id to 6
GPU_ID="0,1"

# dataset settings
TEMPLATE_VERSION="json-v9" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
DS_VERSION1="smid_and_crawled_v2_policy"
DS_VERSION2="smid_and_crawled_v2_with_augmented_policies"
PROMPT_VERSION="chatml_direct"

# model settings
MODEL_VERSION="liuhaotian/llava-v1.6-34b" # the model version to use for training
MODEL_OUTPUT_DIR1="/common-repos/LlavaGuard/models/LlavaGuard-v1.1-34b-full/${DS_VERSION1}/${TEMPLATE_VERSION}"
MODEL_OUTPUT_DIR2="/common-repos/LlavaGuard/models/LlavaGuard-v1.1-34b-full/${DS_VERSION2}/${TEMPLATE_VERSION}"


data_path1="/common-repos/LlavaGuard/data/${DS_VERSION1}/${TEMPLATE_VERSION}"
data_path2="/common-repos/LlavaGuard/data/${DS_VERSION2}/${TEMPLATE_VERSION}"

zero="/LLaVA/scripts/zero3.json"
zero_quant="llavaguard/zero/zero3_quant.json"
zero_offload="/LLaVA/scripts/zero3_offload.json"

# run training
deepspeed --include="localhost:${GPU_ID}" \
    train.py \
    --deepspeed $zero_offload \
    --model_name_or_path "${MODEL_OUTPUT_DIR2}-run1-3ep" \
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
    --output_dir $MODEL_OUTPUT_DIR2 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 3 \
    --evaluation_strategy "epoch" \
    --eval_steps 1 \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

CUDA_VISIBLE_DEVICES=$GPU_ID python3 /workspace/eval_llavaguard.py \
    --data_path $data_path2 \
    --model_base $MODEL_OUTPUT_DIR2