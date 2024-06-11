import argparse
import glob
import json
import os
import sys
import warnings
from transformers import set_seed
import torch
if '/workspace' not in sys.path:
    sys.path.append('/workspace')
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llavaguard.eval_utils import get_model_dir, load_data
from llavaguard.evaluation_metrics_calculator import EvaluationMetricsCalculator, parse_json
from llavaguard.inference import run_llava_batched, run_llava, run_llava_not_batched


def evaluation(lora_dir=None, model_base='liuhaotian/llava-v1.5-13b',
               data_path='smid_and_crawled_policy/json-v4', infer_train_data=False,
               batched_forward=True, copy_images=False, replace_existing_output=False):
    print(f'Loading model {model_base} with attached LORA: {lora_dir}')

    print(f'Dataset: {data_path}')
    # model_name = get_model_name_from_path(model_base)
    root = '/common-repos/LlavaGuard' if os.path.exists('/common-repos/LlavaGuard') else 'output'
    data_paths, data = load_data(data_path)
    if not infer_train_data:
        data.pop('train', None)
        data_paths.pop('train', None)
    # check available memory on GPU
    gb_per_image = {
        7: 15,
        13: 15,
        34: 18,
    }
    model_size = 7 if '-7b' in model_base else 13 if '-13b' in model_base else 34 if '-34b' in model_base else 13
    mem = torch.cuda.get_device_properties(0).total_memory - model_size * 1024 ** 3
    ims_per_device = int(mem / 1024 ** 3 / gb_per_image[model_size])
    batch_size = ims_per_device * torch.cuda.device_count()
    # if batched_forward and 'augmented' not in data_path and '34b' not in model_base:
    if batched_forward and '34b' not in model_base:
        print(f'Selected devices: {torch.cuda.device_count()}, Mem per device (GB): {mem / 1024 ** 3}, '
              f'Batching turned On, Total batch size: {batch_size} (per device: {ims_per_device})')
    else:
        batch_size, batched_forward = 1, False
        print(f'Selected devices: {torch.cuda.device_count()}, Mem per device (GB): {mem / 1024 ** 3}')
        print(f'34b model and augmented data do not support batching: Batching turned Off!!')
    # set seed
    set_seed(48)
    if lora_dir is not None and lora_dir != 'None':
        # load lora models
        model_path = get_model_dir(lora_dir)
        run_name = model_path.split("models/")[1]
        eval_output_dir = f'{root}/eval/{run_name}'
        # model_base = "liuhaotian/llava-v1.5-13b"
        model_name = f'{get_model_name_from_path(model_base)}_lora'
        load_4bit = False
    elif get_model_dir(model_base) is not None:
        # load fine-tuned models
        model_path = get_model_dir(model_base)
        model_base = None
        run_name = model_path.split("models/")[1]
        model_name = run_name.split("/")[0]
        eval_output_dir = f'{root}/eval/{run_name}'
        # disable_torch_init()
        load_4bit = True
    elif model_base is not None:
        # load foundation models
        model_name = get_model_name_from_path(model_base)
        model_path = model_base
        eval_output_dir = f"{root}/eval/{get_model_name_from_path(model_base)}/foundation_model"
        model_base = None
        disable_torch_init()
        load_4bit = True
    else:
        raise ValueError('Please provide a model_save_dir or model_base to load the model.')

    eval_output_dir += f"/{data_paths['eval'].split('/')[-3]}-{data_paths['eval'].split('/')[-2]}"
    # set the output directory
    # template_version = data_path_eval.split('smid_and_crawled_policy/')[-1].split('/')[0]
    # eval_output_dir += f'/{template_version}'

    print(f'Model path: {model_path}, Model base: {model_base}, Model name: {model_name}, with 4bit: {load_4bit}')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                               load_4bit=load_4bit,
                                                                               )
    for warning in w:
        if "vision" not in str(warning.message).lower():
            print(warning.message)
    model.config.tokenizer_model_max_length = 2048 * 2

    os.makedirs(f'{eval_output_dir}/model_output', exist_ok=True)
    if copy_images:
        os.makedirs(f'{eval_output_dir}/eval_ims', exist_ok=True)

    if "llava-v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "llava-v1.5" in model_name.lower() or 'LlavaGuard' in model_name:
        conv_mode = "v1"
    elif "llava-v1.6" in model_name.lower():
        conv_mode = "v1"
    else:
        raise ValueError(f'Unknown model: {model_name}')
    conv = conv_templates[conv_mode].copy()
    for d_name, d_json in data.items():
        print(f'Evaluating {d_name} dataset')
        emc = EvaluationMetricsCalculator(pred_dir=f'{eval_output_dir}/model_output', debug=True)
        # d_json = d_json[:300] if len(d_json) > 300 else d_json
        prompts, gts, ids, im_paths = [], [], [], []
        save_prompt = 0
        e = 0
        for eval_item in d_json:
            sample_id = eval_item['id']
            gt = eval_item['conversations'][1]["value"]
            prompt = eval_item['conversations'][0]["value"]
            if save_prompt < 1:
                with open(f'{eval_output_dir}/{d_name}_prompt_{save_prompt}.txt', 'w+') as f:
                    f.write(prompt)
                save_prompt += 1
            path = glob.glob(f'{eval_output_dir}/model_output/{sample_id}.*')
            try:
                if len(path) > 0 and not replace_existing_output:
                    out = json.load(open(path[0]))
                    out = json.dumps(out['LlavaGuard'], indent=4) if 'LlavaGuard' in out else json.dumps(
                        out['prediction'], indent=4)
                    emc.add_sample(sample_id, out, gt)
                    e += 1
                    # print(f'Output for {sample_id} already exists. Skipping...')
                else:
                    raise FileNotFoundError
            except:
                prompts.append(prompt)
                gts.append(gt)
                ids.append(sample_id)
                im_paths.append(eval_item['image'])
        print(
            f'Existing predictions {e}/{len(d_json)} samples. Running LlavaGuard for {len(prompts)} remaining samples')
        # safe example prompt
        if batched_forward:
            run_llava_batched(model, tokenizer, emc, image_processor, prompts, gts, ids, im_paths, conv, batch_size)
        else:
            run_llava_not_batched(model, tokenizer, emc, image_processor, prompts, gts, ids, im_paths, conv)
        metrics_name = f'{eval_output_dir}/{d_name}_metrics.json' if 'no_edge_cases' not in data_path else f'{eval_output_dir}/{d_name}_metrics_no_edge_cases.json'
        out_name = f'{eval_output_dir}/{d_name}_results.txt' if 'no_edge_cases' not in data_path else f'{eval_output_dir}/{d_name}_results_no_edge_cases.txt'
        emc.compute_stats(print_output=True, save_metric_path=metrics_name, save_txt_path=out_name)
        print('#' * 20 + 'Evaluation Done ' + '#' * 20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLaVA Guard Evaluation')
    parser.add_argument('--lora_dir', type=str,
                        default=None,
                        help='Model save directory absolute path or relative to /common-repos/LlavaGuard/models/')
    parser.add_argument('--model_base', type=str, default='liuhaotian/llava-v1.5-13b', help='Model base')
    parser.add_argument('--data_path', type=str, default='smid_and_crawled_policy/json-v9',
                        help='dataset path either directory or json file')
    parser.add_argument('--infer_train_data', action='store_true',
                        help='Infer on training data, only possible if data_path is a directory')
    parser.add_argument('--copy_images', action='store_true', help='Copy images to eval_ims folder')
    parser.add_argument('--replace_existing_output', action='store_true', help='Replace existing predictions')
    args = parser.parse_args()
    lora_dir = args.lora_dir if args.lora_dir is not None and args.lora_dir != 'None' else None
    data_path = args.data_path
    infer_train_data = args.infer_train_data
    # string to bool conversion if needed
    if isinstance(args.copy_images, str):
        args.copy_images = args.copy_images.lower() in ['true', '1']
    if isinstance(args.replace_existing_output, str):
        args.replace_existing_output = args.replace_existing_output.lower() in ['true', '1']

    # # @todo: fix batched forward for batches with different sized inputs
    evaluation(lora_dir=lora_dir, model_base=args.model_base, data_path=data_path, infer_train_data=infer_train_data,
               batched_forward=True, copy_images=args.copy_images,
               replace_existing_output=args.replace_existing_output)

