export OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=2
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=ERROR

# set visible GPUs
GPU_ID="0,1,2,3,4,5,6,7"
nproc_per_node=$(echo $GPU_ID | awk -F',' '{print NF}')
# export HF_HOME="/HF_TMP"
export WANDB_PROJECT="LlavaGuard"

# dataset settings
MODEL_NAME="LlavaGuard-v1.2-72b-ov"
TEMPLATE_VERSION="v24" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
DS_VERSION="LlavaGuard-DS"

#### set Paths
train_dir="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/lhelff/LlavaGuard"
local_data_dir="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/lhelff/ds/LlavaGuard"


# zero="LLaVA-NeXT/scripts/zero3.json"
# zero_offload="LLaVA-NeXT/scripts/zero3_offload.json"
zero="${train_dir}/scripts/train/zero3.json"
zero_offload="${train_dir}/scripts/train/zero3_offload.json"

VLM_VERSION="lmms-lab/llava-onevision-qwen2-72b-ov"
LLM_VERSION="Qwen/Qwen2-72B-Instruct"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VLM_VERSION_CLEAN="${VLM_VERSION//\//_}"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"


PROMPT_VERSION="qwen_1_5"
############### Pretrain ################
RUN_NAME="${MODEL_NAME}-${VLM_VERSION_CLEAN}-${DS_VERSION}-${TEMPLATE_VERSION}"
echo "RUN_NAME: ${RUN_NAME}"

# llavaGuard-OV model training
MODEL_OUTPUT_DIR="${local_data_dir}/models/${MODEL_NAME}/${TEMPLATE_VERSION}"
PROJECTOR_DIR="${local_data_dir}/models/${MODEL_NAME}/projectors/${RUN_NAME}"
data_path="${local_data_dir}/data/${DS_VERSION}/${TEMPLATE_VERSION}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${nproc_per_node}" --nnodes="1"\
    train_llava_ov.py \
    --deepspeed ${zero_offload} \
    --model_name_or_path ${VLM_VERSION} \
    --version $PROMPT_VERSION \
    --data_path "${data_path}/train_oversampled.json" \
    --data_path_eval "${data_path}/eval.json" \
    --image_folder /common-repos \
    --mm_tunable_parts="mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir $MODEL_OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --eval_steps 1 \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True

    # --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \


python3 llavaguard/evaluation/evaluation.py \
  --model_dir "${MODEL_OUTPUT_DIR}" \
  --data_path "${data_path}/test.json" \
  --device '7'
  --disable_sglang