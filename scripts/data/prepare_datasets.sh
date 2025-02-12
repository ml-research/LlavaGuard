# prepare a dataset for training and evaluation
TEMPLATE_VERSION="24" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data

python3 /workspace/llavaguard/data/build_dataset.py \
     --version ${TEMPLATE_VERSION}