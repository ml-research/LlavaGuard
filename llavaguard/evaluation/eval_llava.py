import argparse
import copy
import glob
import json
import os
import warnings
from transformers import set_seed
import torch

from llavaguard.inference_llava import run_llava_batched, run_llava_not_batched

from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llavaguard.evaluation.eval_utils import get_base_model_name_from_path, get_conv, get_model_name_from_path, load_data, model_dir_to_output_dir
from llavaguard.evaluation.metrics_calculator import EvaluationMetricsCalculator
from huggingface_hub import HfApi


def check_repo_exists(repo_id):
    api = HfApi()
    try:
        # Check if the repository exists
        api.repo_info(repo_id)
        return True
    except Exception as e:
        return False


def evaluation(model_dir = None, data_path='smid_and_crawled_policy/json-v4', output_dir=None,
               batched_forward=True, copy_images=False, replace_existing_output=False,
               device=0):
    print(f'Evaluating model: {model_dir} on data: {data_path}')
    print(f'Output directory: {output_dir}')
    # set cuda_visible_devices
    
    
    # model_name = get_model_name_from_path(model_base)
    data_paths, data = load_data(data_path)
    eval_output_dir = output_dir if output_dir is not None else model_dir_to_output_dir(args.model_dir, args.data_path)
    # check available memory on GPU
    gb_per_image = {
        7: 15,
        8: 15,
        13: 15,
        34: 18,
    }
    model_size = 7 if '-7b' in model_dir else 13 if '-13b' in model_dir else 34 if '-34b' in model_dir else 13
    mem = torch.cuda.get_device_properties(0).total_memory - model_size * 1024 ** 3
    ims_per_device = int(mem / 1024 ** 3 / gb_per_image[model_size])
    batch_size = ims_per_device * torch.cuda.device_count()

    # set seed
    set_seed(48)
    base_model_name = get_base_model_name_from_path(model_dir)
    model_name = get_model_name_from_path(model_dir)
    if os.path.exists(model_dir):
        # if there is a subfolder with name llava and model weights inside then update model_dir to that
        if len(glob.glob(f'{model_dir}/llava/*.safetensors')) > 0:
            model_dir = f'{model_dir}/llava'
        model_path = model_dir

        if 'lora' in model_dir.lower():
            # load lora models
            # run_name = model_path.split("models/")[1]
            base_model_name += "-lora"
            load_4bit = False
        else:
            # load fine-tuned models
            model_base = None
            # disable_torch_init()
            load_4bit = True
    elif check_repo_exists(model_dir):
        # load foundation models
        model_path = model_dir
        model_base = None
        disable_torch_init()
        load_4bit = True
    else:
        raise ValueError('Please provide a model_save_dir or model_base to load the model.')

    # eval_output_dir += f"/{data_paths['eval'].split('/')[-3]}-{data_paths['eval'].split('/')[-2]}"
    # model_name = 'llama3-llava-next-8b' if model_name == 'LlavaGuard-8b' else model_name
    # model_name = 'llama3-llava-next-8b' if model_name == 'LlavaGuard-v1.2-7b-ov' else model_name
    print(f'Model path: {model_path}, Model base: {model_base}, Model name: {model_name}, with 4bit: {load_4bit}')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, base_model_name,
                                                                            #    multimodal=True,
                                                                               device_map='auto',
                                                                               # load_4bit=load_4bit,
                                                                               )
    for warning in w:
        if "vision" not in str(warning.message).lower():
            print(warning.message)
    # if batched_forward and 'augmented' not in data_path and '34b' not in model_base:
    if batched_forward and '34b' not in model_dir.lower() and 'ov' not in model_dir.lower():
        print(f'Selected devices: {torch.cuda.device_count()}, Mem per device (GB): {mem / 1024 ** 3}, '
              f'Batching turned On, Total batch size: {batch_size} (per device: {ims_per_device})')
    else:
        batch_size, batched_forward = 1, False
        print(f'Selected devices: {torch.cuda.device_count()}, Mem per device (GB): {mem / 1024 ** 3}')
        print(f'Batching turned Off')

    # model.config.tokenizer_model_max_length = 2048 * 2

    os.makedirs(f'{eval_output_dir}/model_output', exist_ok=True)
    if copy_images:
        os.makedirs(f'{eval_output_dir}/eval_ims', exist_ok=True)

    conv = get_conv(model_name)
    model.eval()
    if 'ov' in model_name:
        model.tie_weights()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LlavaGuard Evaluation')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', default=0)
    args = parser.parse_args()
    d = args.device if isinstance(args.device, int) else int(args.device[0])
    output_dir = args.output_dir if args.output_dir is not None else model_dir_to_output_dir(args.model_dir, args.data_path)
    print(f'Running evaluation (SGLang turned {"off" if args.disable_sglang else "on"}) for model: {args.model_dir} on data: {args.data_path}')
    print(f'Output directory: {output_dir}')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(d)

    output_dir = f'{output_dir}-base'
    evaluation(model_dir=args.model_dir, data_path=args.data_path, output_dir=output_dir,
            batched_forward=fALSE, copy_images=False, replace_existing_output=False)

