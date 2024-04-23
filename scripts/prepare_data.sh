# prepare a dataset for training and evaluation
TEMPLATE_VERSION="json-v10" # (json-v0, json-v1, json-v2, json-v3, json-v4, or nl) the version of the template used to generate the data
AUGMENTATION="False" # (True or False) whether to augment the data with additional examples

python3 /workspace/prepare_data.py \
     --template_version ${TEMPLATE_VERSION} \
     --augmentation ${AUGMENTATION}