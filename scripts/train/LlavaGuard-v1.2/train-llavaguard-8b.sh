export OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=2
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=ERROR

# set visible GPUs
GPU_ID="2"
export HF_HOME="/HF_TMP"

# dataset settings
MODEL_NAME="LlavaGuard-8b"
TEMPLATE_VERSION="v22" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
DS_VERSION="LlavaGuard-DS"

zero="/LLaVA/scripts/zero3.json"
zero_offload="/LLaVA/scripts/zero3_offload.json"
zero="/workspace/scripts/train/zero3.json"
zero_offload="/workspace/scripts/train/zero3_offload.json"


LLM_VERSION="meta-llama/Meta-Llama-3-8B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
VLM_VERSION="lmms-lab/llama3-llava-next-8b"


PROMPT_VERSION=plain
############### Pretrain ################

BASE_RUN_NAME="${MODEL_NAME}-${LLM_VERSION_CLEAN}-${VISION_MODEL_VERSION_CLEAN}-${DS_VERSION}-${TEMPLATE_VERSION}"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

PROMPT_VERSION="llava_llama_3"
MID_RUN_NAME="${MODEL_NAME}-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-${DS_VERSION}-${TEMPLATE_VERSION}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"


local_data_dir="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/lhelff/ds/LlavaGuard"

# llavaguard-8b model training
MODEL_OUTPUT_DIR="${local_data_dir}/models/${MODEL_NAME}/${TEMPLATE_VERSION}"
PROJECTOR_DIR="${local_data_dir}/models/${MODEL_NAME}/projectors/${BASE_RUN_NAME}"
data_path="${local_data_dir}/data/${DS_VERSION}/${TEMPLATE_VERSION}"

#torchrun --include="localhost:${GPU_ID}" \
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
#    --pretrain_mm_mlp_adapter="${PROJECTOR_DIR}/mm_projector.bin" \

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="1" --nnodes="1"\
    train_llava.py \
    --deepspeed ${zero} \
    --model_name_or_path ${VLM_VERSION} \
    --version $PROMPT_VERSION \
    --data_path "${data_path}/train_oversampled.json" \
    --data_path_eval "${data_path}/eval.json" \
    --image_folder /common-repos \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir $MODEL_OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True

python3 llavaguard/evaluation/evaluation.py \
  --model_dir $MODEL_OUTPUT_DIR \
  --data_path "${data_path}/test.json" \
  --device $GPU_ID