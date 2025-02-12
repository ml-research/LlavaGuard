# prepare a dataset for training and evaluation
MODEL_VERSION1="liuhaotian/llava-v1.5-7b"  # The model version to use for training
MODEL_VERSION2="liuhaotian/llava-v1.5-13b"  # The model version to use for training
MODEL_VERSION3="liuhaotian/llava-v1.6-34b"  # The model version to use for training
M3_LOCAL='/common-repos/LLaVA/llava-v1.6-34b'
TEMPLATE_VERSION="json-v20"
OUTPUT_DIR=f'{local_data_dir}/data/rationales/llava-v1.6-34b-json-v16-v5'
DEVICE='1'

python3 /workspace/llavaguard/data/generate_rationales.py \
     --template_version ${TEMPLATE_VERSION} \
     --output_dir ${OUTPUT_DIR} \
     --replace_existing "False" \
     --model_name_or_path ${MODEL_VERSION3} \
     --device ${DEVICE}