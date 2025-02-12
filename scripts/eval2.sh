#!/bin/bash
# dataset settings
#TEMPLATE_VERSION11="json-v11-neurips" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
#TEMPLATE_VERSION11="json-v10-neurips" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data

local_data_dir="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/lhelff/ds/LlavaGuard"

# template version
TEMPLATE_VERSION20="json-v20"
TEMPLATE_VERSION22="json-v22"
TEMPLATE_VERSION17="json-v17" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
TEMPLATE_VERSION16="json-v16" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
TEMPLATE_VERSION15="json-v15" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
TEMPLATE_VERSION14="json-v14" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
TEMPLATE_VERSION13="json-v13" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
TEMPLATE_VERSION12="json-v12" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
TEMPLATE_VERSION11="json-v11" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
TEMPLATE_VERSION10="json-v10" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
TEMPLATE_VERSION24="v24"
TEMPLATE_VERSION25="v25"

# llavaguard
MODEL_OUTPUT_DIR1="${local_data_dir}/models/LlavaGuard-v1.2-0.5B-OV-local/${TEMPLATE_VERSION24}"
MODEL_OUTPUT_DIR2="${local_data_dir}/models/LlavaGuard-v1.2-7B-OV-local/${TEMPLATE_VERSION24}"
# MODEL_OUTPUT_DIR3=f"${local_data_dir}/models/LlavaGuard-v1.2-34b/${TEMPLATE_VERSION24}"
# MODEL_OUTPUT_DIR4=f"${local_data_dir}/models/LlavaGuard-v1.2-7b-ov/${TEMPLATE_VERSION24}"
MODEL_OUTPUT_DIR3="AIML-TUDA/LlavaGuard-v1.2-0.5B-OV"
MODEL_OUTPUT_DIR4="AIML-TUDA/LlavaGuard-v1.2-7B-OV"
MODEL_OUTPUT_DIR5="${local_data_dir}/models/LlavaGuard-8b/${TEMPLATE_VERSION24}"
# MODEL_OUTPUT_DIR6="${local_data_dir}/models/QwenGuard-v1.2-7b-run1/${TEMPLATE_VERSION24}"
# MODEL_OUTPUT_DIR6="${local_data_dir}/models/QwenGuard-v1.2-2b-run1/${TEMPLATE_VERSION24}"
MODEL_OUTPUT_DIR6="${local_data_dir}/models/QwenGuard-v1.2-2b-stage0-run1/${TEMPLATE_VERSION25}"
MODEL_OUTPUT_DIR7="${local_data_dir}/models/QwenGuard-v1.2-2b-mixed-run0/${TEMPLATE_VERSION24}"
MODEL_OUTPUT_DIR8="${local_data_dir}/models/QwenGuard-v1.2-3b-run1/${TEMPLATE_VERSION24}"
# MODEL_OUTPUT_DIR8="${local_data_dir}/models/QwenGuard2.5-v1.2-7b-run1/${TEMPLATE_VERSION24}"



# old model
MODEL_VERSION1="liuhaotian/llava-v1.5-7b" # the model version to use for training
MODEL_VERSION2="liuhaotian/llava-v1.5-13b" # the model version to use for training
MODEL_VERSION3="liuhaotian/llava-v1.6-34b" # the model version to use for training
#MODEL_VERSION3="/common-repos/LLaVA/llava-v1.6-34b"
MODEL_VERSION4="liuhaotian/llava-v1.6-vicuna-7b" # the model version to use for training
MODEL_VERSION5="liuhaotian/llava-v1.6-vicuna-13b" # the model name to use for training

# new models

MODEL_VERSION6="lmms-lab/llama3-llava-next-8b"
MODEL_VERSION7="lmms-lab/llava-onevision-qwen2-0.5b-ov"
MODEL_VERSION8="lmms-lab/llava-onevision-qwen2-7b-ov" # the model version to use for training
# MODEL_VERSION9="lmms-lab/llava-onevision-qwen2-7b-ov-chat" # the model version to use for training
MODEL_VERSION10="lmms-lab/llava-onevision-qwen2-72b-ov" # the model version to use for training
# MODEL_VERSION11="lmms-lab/llava-onevision-qwen2-72b-ov-chat" # the model version to use for training
MODEL_VERSION12="Qwen/Qwen2-VL-2B-Instruct"
MODEL_VERSION13="Qwen/Qwen2-VL-7B-Instruct"
MODEL_VERSION14="Qwen/Qwen2-VL-72B-Instruct"
MODEL_VERSION15="meta-llama/Llama-3.2-11B-Vision-Instruct"
# MODEL_VERSION16="/common-repos/LlamaGuard/Llama-Guard-3-11B-Vision"
MODEL_VERSION16="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/HF_HOME/models/Llama-Guard-3-11B-Vision"

MODEL_VERSION17="OpenGVLab/InternVL2_5-1B-MPO"
MODEL_VERSION18="OpenGVLab/InternVL2_5-2B-MPO"
MODEL_VERSION19="OpenGVLab/InternVL2_5-4B-MPO"
MODEL_VERSION20="OpenGVLab/InternVL2_5-8B-MPO"
MODEL_VERSION21="OpenGVLab/InternVL2_5-26B-MPO"
MODEL_VERSION22="OpenGVLab/InternVL2_5-38B-MPO"
MODEL_VERSION23="OpenGVLab/InternVL2_5-78B-MPO"
MODEL_VERSION24="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/HF_HOME/models/Llama-3.2-90B-Vision-Instruct"
MODEL_VERSION25="omni-moderation-latest"
MODEL_VERSION26="gpt-4o"
MODEL_VERSION27="o1-mini"

MODEL_VERSION28="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_VERSION29="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_VERSION30="Qwen/Qwen2.5-VL-72B-Instruct"



# data_pth1="data/LlavaGuard-DS/${TEMPLATE_VERSION24}/all_data.json"
data_pth2="data/LlavaGuard-DS/${TEMPLATE_VERSION24}/test.json"
data_pth25="data/LlavaGuard-DS/${TEMPLATE_VERSION25}"


DEVICE="0,1,2,3"
## LlavaGuard
# python3 ~/llavaguard/evaluation/evaluation.py \
#  --model_dir $MODEL_OUTPUT_DIR1 \
#  --data_path $data_pth2 \
#  --device $DEVICE

# python3 ~/llavaguard/evaluation/evaluation.py \
#   --model_dir $MODEL_OUTPUT_DIR2 \
#   --data_path $data_pth2 \
#   --device $DEVICE

#   python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
#   --model_dir "$MODEL_OUTPUT_DIR2" \
#   --data_path "$data_pth2" \
#   --device $DEVICE

# python3 ~/llavaguard/llavaguard/evaluation/evaluation.py \
#  --model_dir $MODEL_OUTPUT_DIR3 \
#  --data_path $data_pth2 \
#  --device $DEVICE

# python3 ~/llavaguard/evaluation/evaluation.py \
#  --model_dir $MODEL_OUTPUT_DIR4 \
#  --data_path $data_pth2 \
#  --device $DEVICE

# python3 ~/llavaguard/evaluation/evaluation.py \
#  --model_dir $MODEL_OUTPUT_DIR6 \
#  --data_path $data_pth2 \
#  --device $DEVICE
#  --engine "transformers"

# python3 ~/llavaguard/evaluation/evaluation.py \
#  --model_dir $MODEL_OUTPUT_DIR6 \
#  --data_path $data_pth2 \
#  --device $DEVICE

# CUDA_VISIBLE_DEVICES="4,5,6,7" python3 ~/llavaguard/evaluation/evaluation.py \
#  --model_dir "AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf" \
#  --data_path $data_pth2 \
#  --device "4,5,6,7" \
#  --engine "vllm"

CUDA_VISIBLE_DEVICES=$DEVICE python3 ~/llavaguard/evaluation/evaluation.py \
 --model_dir $MODEL_OUTPUT_DIR8 \
 --data_path $data_pth2 \
 --device $DEVICE \
 --engine "vllm"

#docker exec -it llavaguard_llavatrainer_1 \

# # Llava old models
# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION1 \
# --data_path $data_pth2 \
# --device $DEVICE

# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION2 \
# --data_path $data_pth2 \
# --device $DEVICE

# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION3 \
# --data_path $data_pth2 \
# --device $DEVICE

# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION4 \
# --data_path $data_pth2 \
# --device $DEVICE

# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION5 \
# --data_path $data_pth2 \
# --device $DEVICE






############## new models

# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION6 \
# --data_path $data_pth2 \
# --device $DEVICE


# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION7 \
# --data_path $data_pth2 \
# --device $DEVICE

# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION8 \
# --data_path $data_pth2 \
# --device $DEVICE


# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION10 \
# --data_path $data_pth2 \
# --device $DEVICE


# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION12 \
# --data_path $data_pth2 \
# --device $DEVICE


# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION13 \
# --data_path $data_pth2 \
# --device $DEVICE

# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION10 \
# --data_path $data_pth2 \
# --engine "sglang" \
# --batching "off" \
# --device $DEVICE


# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION16 \
# --data_path $data_pth2 \
# --device $DEVICE \
# --engine "transformers"

# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION24 \
# --data_path $data_pth2 \
# --device $DEVICE 


# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION17 \
# --data_path $data_pth2 \
# --device $DEVICE 

# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION18 \
# --data_path $data_pth2 \
# --device $DEVICE 


# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION19 \
# --data_path $data_pth2 \
# --device $DEVICE 


# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION20 \
# --data_path $data_pth2 \
# --device $DEVICE 

# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION21 \
# --data_path $data_pth2 \
# --device $DEVICE 

# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION22 \
# --data_path $data_pth2 \
# --device $DEVICE 

# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION23 \
# --data_path $data_pth2 \
# --device $DEVICE 

# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION25 \
# --data_path $data_pth2 \
# --device $DEVICE 


# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION26 \
# --data_path $data_pth2 \
# --device $DEVICE 

######## old models


# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION12 \
# --data_path $data_pth2 \
# --device $DEVICE \
# --output_dir /common-repos/LlavaGuard/eval/Llama-Guard-3-11B-Vision \
# --disable_sglang


#####################
#python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
#--model_dir $MODEL_VERSION4 \
#--data_path $data_pth2 \
#--device 0


#python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
#--model_dir $MODEL_VERSION5 \
#--data_path $data_pth2 \
#--device 1





# python3 ~/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION28 \
# --data_path $data_pth2 \
# --device $DEVICE \
# --engine "transformers"


# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION29 \
# --data_path $data_pth2 \
# --device $DEVICE

# python3 ~/LlavaGuard/llavaguard/evaluation/evaluation.py \
# --model_dir $MODEL_VERSION30 \
# --data_path $data_pth2 \
# --device $DEVICE