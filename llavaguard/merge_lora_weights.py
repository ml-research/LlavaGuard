import os
import torch
from llava.model.builder import load_pretrained_model


def merge_lora_into_model(model_path, model_base, model_name, out_dir='output/merged_model'):
    os.makedirs(out_dir, exist_ok=True)
    # Load the model
    print(f'Loading model from {model_path}')
    tokenizer, merged_model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                                  # load_8bit=False, load_4bit=True,
                                                                                  torch_dtype=torch.bfloat16,
                                                                                  device='cpu')
    merged_model.config.torch_dtype = torch.bfloat16
    merged_model.to(torch.bfloat16)
    # remove previous merged model
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    # save merged model
    print(f'saving merged model to {out_dir}...')
    merged_model.save_pretrained(save_directory=out_dir, safe_serialization=False, save_peft_format=False,
                                 torch_dtype=torch.bfloat16)
    # load model config.json and change the model name
    import json
    with open(os.path.join(out_dir, 'config.json'), 'r') as f:
        config = json.load(f)
        config['model_type'] = 'llava'
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)


# model_path = f'{base_path}/naive_SMID_CRAWLED'
# model_base = "liuhaotian/llava-v1.5-13b"

model_path = '/common-repos/LlavaGuard/models/LlavaGuard-v1.2-34b-lora/smid_and_crawled_v2_with_augmented_policies/json-v10'
model_base = "liuhaotian/llava-v1.6-34b"
model_name = model_base.split('/')[1] + 'lora'
# model_path = '/storage-02/common-repos/LlavaGuard/models/LlavaGuard-v1.1-13b-full/smid_and_crawled_v2_with_augmented_policies/json-v9/llava'
# model_name = "liuhaotian/llava-v1.5-13b"

# model_name = model_base.split('/')[1] + '_lora'
out_path_merged_model = f'{model_path}/llava'

merge_lora_into_model(model_path, model_base, model_name, out_path_merged_model)

###########################
# did not work, using transformer implementation for now
