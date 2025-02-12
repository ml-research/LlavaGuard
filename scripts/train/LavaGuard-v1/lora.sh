#!/bin/bash

#!/bin/bash

# set visible GPUs
GPU_ID="0,1,2,3,4,5,6"
export HF_HOME="/HF_TMP"

# dataset settings
TEMPLATE_VERSION="json-v16" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
DS_VERSION2="smid_and_crawled_v2_with_augmented_policies"
PROMPT_VERSION=v1

# llavaguard-v1.1 7b model training
MODEL_VERSION1="liuhaotian/llava-v1.5-7b" # the model version to use for training
MODEL_OUTPUT_DIR1=f"{local_data_dir}/models/lora/LlavaGuard-v1.1-7b/${DS_VERSION2}/${TEMPLATE_VERSION}"

# llavaguard-v1.1 13b model training
MODEL_VERSION2="liuhaotian/llava-v1.5-13b" # the model version to use for training
MODEL_OUTPUT_DIR2=f"{local_data_dir}/models/lora/LlavaGuard-v1.1-13b/${DS_VERSION2}/${TEMPLATE_VERSION}"

# llavaguard-v1.2 34b model training
MODEL_VERSION3="liuhaotian/llava-v1.6-34b" # the model version to use for training
MODEL_OUTPUT_DIR3=f"{local_data_dir}/models/lora/LlavaGuard-v1.2-34b/${DS_VERSION2}/${TEMPLATE_VERSION}"

data_path=f"{local_data_dir}/data/${DS_VERSION2}/${TEMPLATE_VERSION}"

# remove previous runs if they exist otherwise it will skip the training for existing runs
#rm -rf $MODEL_OUTPUT_DIR1

zero="/LLaVA/scripts/zero3.json"
zero_offload="/LLaVA/scripts/zero3_offload.json"


# LlavaGuard-v1.1 7b model training
deepspeed --include="localhost:${GPU_ID}" \
    train_llava.py \
    --deepspeed $zero_offload \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path $MODEL_VERSION1 \
    --version $PROMPT_VERSION \
    --data_path "${data_path}/train_oversampled.json" \
    --data_path_eval "${data_path}/eval.json" \
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
    --num_train_epochs 3 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 1 \
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

#python3 /workspace/llavaguard/sglang/evaluation.py \
#  --model_dir $MODEL_OUTPUT_DIR1 \
#  --data_path $data_path \
#  --device $GPU_ID


# LlavaGuard-v1.1 13b model training
deepspeed --include="localhost:${GPU_ID}" \
    train_llava.py \
    --deepspeed $zero_offload \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path $MODEL_VERSION2 \
    --version $PROMPT_VERSION \
    --data_path "${data_path}/train_oversampled.json" \
    --data_path_eval "${data_path}/eval.json" \
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
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 3 \
    --evaluation_strategy "steps" \
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

#python3 /workspace/llavaguard/sglang/evaluation.py \
#  --model_dir $MODEL_OUTPUT_DIR2 \
#  --data_path $data_path \
#  --device $GPU_ID


PROMPT_VERSION="chatml_direct"
# model settings

# run training
deepspeed --include="localhost:${GPU_ID}" \
    train_llava.py \
    --deepspeed $zero_offload \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path $MODEL_VERSION3 \
    --version $PROMPT_VERSION \
    --data_path "${data_path}/train_oversampled.json" \
    --data_path_eval "${data_path}/eval.json" \
    --image_folder /common-repos \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_patch_merge_type spatial_unpad \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $MODEL_OUTPUT_DIR3 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 1 \
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



#python3 /workspace/llavaguard/sglang/evaluation.py \
#  --model_dir $MODEL_OUTPUT_DIR3 \
#  --data_path $data_path \
#  --device $GPU_ID