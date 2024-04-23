import glob
import json
import os

import rtpt
from tqdm import tqdm

import llava.mm_utils
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llavaguard.evaluation_metrics_calculator import EvaluationMetricsCalculator
from llavaguard.inference import run_llava


def get_model_dir(run_name):
    if os.path.exists(run_name):
        return run_name
    if os.path.exists(f'/common-repos/LlavaGuard/models/{run_name}'):
        return f'/common-repos/LlavaGuard/models/{run_name}'
    elif os.path.exists(f'output/models/{run_name}'):
        return f'output/models/{run_name}'
    else:
        return None


def load_data(data_path, infer_train_data=False):
    dd = {}
    paths = {}
    if data_path.endswith('.json'):
        dd = {data_path.split('/')[-1].split('.')[0]: json.load(open(data_path))}
        paths = {data_path.split('/')[-1].split('.')[0]: data_path}
        return paths, dd

    for p, type in [(data_path, 'test'), (data_path, 'eval'), (data_path, 'train')]:
        if type == 'train' and not infer_train_data:
            continue
        if not p.endswith('/'):
            p += '/'
        p += f'{type}.json'
        if os.path.exists(p):
            dd[type] = json.load(open(p))
        elif os.path.exists(f'/common-repos/LlavaGuard/data/{p}'):
            dd[type] = json.load(open(f'/common-repos/LlavaGuard/data/{p}'))
        elif os.path.exists(f'output/data/{p}'):
            dd[type] = json.load(open(f'output/data/{p}'))
        else:
            raise FileNotFoundError(f'No data found for {p}')
        paths[type] = p

    return paths, dd


def eval_loop(lora_dir=None, model_base='liuhaotian/llava-v1.5-13b',
              data_path_eval='smid_and_crawled_policy/json-v4', data_path_train='smid_and_crawled_policy/json-v4',
              copy_images=False, replace_existing_output=False):
    print(f'Loading model {model_base} with attached LORA: {lora_dir}')
    print(f'Evaluation dataset: {data_path_eval}')
    print(f'Training dataset: {data_path_train}')
    model_name = llava.mm_utils.get_model_name_from_path(model_base)
    root = '/common-repos/LlavaGuard' if os.path.exists('/common-repos/LlavaGuard') else '/output'
    paths, data = load_data(data_path_train, data_path_eval)

    if lora_dir is not None and lora_dir != 'None':
        # load lora models
        model_path = get_model_dir(lora_dir)
        eval_output_dir = f'{root}/eval/{model_name}/lora/{model_path.split("/")[-1]}'
        # model_base = "liuhaotian/llava-v1.5-13b"
        model_name += '_lora'
    elif get_model_dir(model_base) is not None:
        # load fine-tuned models
        model_path = get_model_dir(model_base)
        model_base = None
        model_name = model_path.split("/")[-4]
        eval_output_dir = f'{root}/eval/{model_name}/fine-tuned/{model_path.split("/")[-1]}'
        disable_torch_init()
    elif model_base is not None:
        # load foundation models
        model_path = model_base
        eval_output_dir = f'{root}/eval/{model_name}/foundation_model'
        model_base = None
        disable_torch_init()
    else:
        raise ValueError('Please provide a model_save_dir or model_base to load the model.')

    # set the output directory
    template_version = data_path_eval.split('smid_and_crawled_policy/')[-1].split('/')[0]
    eval_output_dir += f'/{template_version}'
    print(f'Output directory: {eval_output_dir}')

    print(f'Model path: {model_path}, Model base: {model_base}, Model name: {model_name}')
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

    os.makedirs(f'{eval_output_dir}/model_output', exist_ok=True)
    if copy_images:
        os.makedirs(f'{eval_output_dir}/eval_ims', exist_ok=True)

    if "llava-v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "llava-v1.5" in model_name.lower():
        conv_mode = "v1"
    elif "llava-v1.6" in model_name.lower():
        conv_mode = "v1"
    else:
        raise ValueError(f'Unknown model: {model_name}')
    conv_ = conv_templates[conv_mode].copy()

    for d_name, d_json in data.items():
        # only evaluate on max 100 samples
        # if len(d_json) > 300:
        #     d_json = d_json[:300]
        emc = EvaluationMetricsCalculator(pred_dir=f'{eval_output_dir}/model_output')
        pbar = tqdm(d_json)
        rt = rtpt.RTPT(name_initials='LH', experiment_name=f'LlavaGuard-Eval', max_iterations=len(d_json))
        rt.start()
        for eval_item in pbar:
            metrics = emc.get_metrics()
            pbar.set_description(
                f'Evaluating ({d_name} DS), TP: {metrics["TP"]}, FP: {metrics["FP"]}, TN: {metrics["TN"]}, FN: {metrics["FN"]}, Invalid: {metrics["invalid_assessments"]}')
            im_path = eval_item['image']
            prompt = eval_item['conversations'][0]["value"]
            gt = eval_item['conversations'][1]["value"]
            gt = json.loads(gt)
            sample_id = eval_item['id']
            if pbar.n == 0:
                # save prompt
                with open(f'{eval_output_dir}/prompt.txt', 'w+') as f:
                    f.write(prompt)

            # image_.resize((512, 512)).show()
            path = glob.glob(f'{eval_output_dir}/model_output/{sample_id}.*')
            if len(path) > 0 and not replace_existing_output:
                out = json.load(open(path[0]))
                out = json.dumps(out['LlavaGuard'], indent=4) if 'LlavaGuard' in out else json.dumps(
                    out['prediction'], indent=4)
                emc.add_sample(sample_id, out, gt)

            # if os.path.exists(f'{eval_output_dir}/model_output/{sample_id}.txt') and not replace_existing_output:
            #     out = json.load(open(f'{eval_output_dir}/model_output/{sample_id}.txt'))['LlavaGuard']
            #     out = json.dumps(out, indent=4)
            #     emc.add_sample(sample_id, out, gt)
            #     print(f'Output for {sample_id} already exists. Skipping...')
            else:
                out = run_llava(model, tokenizer, image_processor, prompt, im_path, conv_)
                emc.add_sample(sample_id, out, gt, save_output=True)
            rt.step()

        metrics_name = f'{eval_output_dir}/{d_name}_metrics.json' if 'no_edge_cases' not in data_path_eval else f'{eval_output_dir}/{d_name}_metrics_no_edge_cases.json'
        out_name = f'{eval_output_dir}/{d_name}_results.txt' if 'no_edge_cases' not in data_path_eval else f'{eval_output_dir}/{d_name}_results_no_edge_cases.txt'
        emc.compute_stats(print_output=True, save_metric_path=metrics_name, save_txt_path=out_name)
