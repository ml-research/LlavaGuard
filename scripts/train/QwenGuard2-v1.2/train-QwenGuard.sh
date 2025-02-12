GPU_ID="0,1,2,3,4,5,6,7"
NPROC_PER_NODE=$(echo $GPU_ID | awk -F',' '{print NF}')
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=2222

MODEL_NAME="QwenGuard-v1.2-7b-run1"
BASE_MODEL="Qwen/Qwen2-VL-7B-Instruct"

#### set Paths
train_dir="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/lhelff/LlavaGuard"
local_data_dir="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/lhelff/ds/LlavaGuard"

### set model output path
DS_CONFIG_PATH="${train_dir}/scripts/train/zero3.json"
data_path="${local_data_dir}/data/LlavaGuard-DS/v24/dataset_info.json"
OUTPUT_PATH="${local_data_dir}/models/${MODEL_NAME}/v24"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    "

torchrun $DISTRIBUTED_ARGS llamafactory_train_llava.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --model_name_or_path $BASE_MODEL \
    --dataset LlavaGuard-DS-v24 \
    --template qwen2_vl \
    --finetuning_type full \
    --output_dir $OUTPUT_PATH \
    --freeze_vision_tower \
    --freeze_multi_modal_projector \
    --train_mm_proj_only false \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --report_to wandb \
    --run_name $MODEL_NAME \
    --plot_loss \
    --num_train_epochs 3 \
    --bf16