from llavaguard.server.server import  evaluate, set_up_server
from llavaguard_config import local_data_dir
from llavaguard.evaluation.metrics_calculator import EvaluationMetricsCalculator
from llavaguard.evaluation.eval_utils import get_model_name_from_path, load_data, model_dir_to_output_dir

import importlib.metadata
import argparse

from random import sample, seed
import rtpt
import os
from tqdm import tqdm
import json
import glob

def prepare_data(data: list[dict], emc: EvaluationMetricsCalculator, output_dir:str, replace_existing_output: bool =False):
    inputs = []
    d_name =  output_dir.split('/')[-1]
    eval_im_dir =   f'{local_data_dir}/eval/eval_ims/{d_name}'
    print('eval_im_dir', eval_im_dir)
    print(f'Evaluation output: {output_dir}')

    os.makedirs(f'{output_dir}/model_output', exist_ok=True)
    os.makedirs(eval_im_dir, exist_ok=True)

    print(f'Evaluating {d_name} dataset')
    save_prompt = 0
    e = 0
    for eval_item in data:
        sample_id = eval_item['id']
        gt = eval_item['conversations'][1]["value"]
        prompt = eval_item['conversations'][0]["value"]
        if save_prompt < 1:
            with open(f'{output_dir}/{d_name}_prompt_{save_prompt}.txt', 'w+') as f:
                f.write(prompt)
            save_prompt += 1
        if not os.path.exists(f'{eval_im_dir}/{sample_id}.png') and 'eval' in output_dir:
            im_p = eval_item['image'].replace(" ", "\\ ")
            os.system(f'cp {im_p} {eval_im_dir}/{sample_id}.png')
        path = glob.glob(f'{output_dir}/model_output/{sample_id}.*')
        try:
            if len(path) > 0 and not replace_existing_output:
                out = json.load(open(path[0]))
                out = json.dumps(out['LlavaGuard'], indent=4) if 'LlavaGuard' in out else json.dumps(
                    out['prediction'], indent=4)
                emc.add_sample(sample_id, out, gt)
                e += 1
            else:
                raise FileNotFoundError
        except:
            inputs.append({
                'prompt': prompt,
                'gt': gt,
                'id': sample_id,
                'image': eval_item['image']
            })
    return inputs


       
def evaluate(model_dir:str, data_path='smid_and_crawled_policy/json-v4',
                    output_dir: str = None, device=0, engine=True, batch_size=1000,
                    replace_existing_output=False):
    if 'eval' not in output_dir:
        raise ValueError(f'Output directory must be de defined for local model evaluation: {model_dir}')
    _, data = load_data(data_path)
    print_text = ''
    server = set_up_server(engine, model_dir, device)
    for d_name, d_json in data.items():
        if d_name == 'train':
            print(f'Train daataset Eval is cut too 800 sampels')
            seed(42)
            d_json = sample(d_json, 800)
        emc = EvaluationMetricsCalculator(pred_dir=f'{output_dir}/model_output',)
        inputs = prepare_data(d_json, emc, output_dir, replace_existing_output)
        print('#' * 20 + f'Running Evaluation on {d_name} dataset' + '#' * 20)
        print(f'Existing predictions {len(d_json)-len(inputs)}/{len(d_json)} samples. Running eval for {len(inputs)} remaining samples')
        if len(inputs) == 0:
            _, metrics_txt = emc.compute_stats(print_output=False)
            print_text += metrics_txt + '\n'
            continue
        emc.log_time()
        rt = rtpt.RTPT(name_initials='LH', experiment_name=f'LlavaGuard-Eval', max_iterations=len(inputs) + 1)
        rt.start()
        progress_bar = tqdm(total=len(inputs), desc=f"Evaluating model {get_model_name_from_path(model_dir)}")
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            outputs = server.request(batch, tqdm_bar=progress_bar)
            for idx in range(len(batch)):
                batch[idx]['prediction'] = outputs[idx]
            emc.add_batch(batch, save_output=True)
            rt.step()
        metrics_name = f'{output_dir}/{d_name}_metrics.json' if 'no_edge_cases' not in data_path else f'{output_dir}/{d_name}_metrics_no_edge_cases.json'
        out_name = f'{output_dir}/{d_name}_results.txt' if 'no_edge_cases' not in data_path else f'{output_dir}/{d_name}_results_no_edge_cases.txt'
        _, metrics_txt = emc.compute_stats(print_output=False, save_metric_path=metrics_name, save_txt_path=out_name)
        print_text += metrics_txt + '\n'
        print('#' * 20 + 'Evaluation Done' + '#' * 20)
        print()
    server.tearDownClass()
    print(print_text)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LlavaGuard Evaluation')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--engine', type=str, default='auto')
    parser.add_argument('--batching', type=str, default='auto', choices=['auto', 'off', 'continues batching'])
    parser.add_argument('--device', default=0)
    args = parser.parse_args()
    d = args.device if isinstance(args.device, int) else int(args.device[0])
    # we disable multi gpu for all but sglang evaluation
    data_path = local_data_dir + '/'  + args.data_path
    output_dir = args.output_dir if args.output_dir is not None else model_dir_to_output_dir(args.model_dir, data_path)

    if args.engine != 'auto':
        output_dir += f'-{args.engine.replace("_", "-")}'

    if args.batching == 'continues batching' or args.batching == 'auto':
        batch_size = 1000
    elif args.batching == 'off':
        batch_size = 1
        output_dir += '-no-batching'
    else:
        raise ValueError(f'Batching option {args.batching} not supported')
    
    # raise('Stop')
    print(f'Running evaluation (engine: {args.engine}) for model: {args.model_dir}')
    print(f'Data path: {data_path}')
    print(f'Output path: {output_dir}')

    installed = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}

    if args.engine == 'llava':
        print(f'Transformer evaluation loop')
        if 'llava' in args.model_dir.lower():
            from llavaguard.evaluation import eval_llava
            eval_llava.evaluation(model_dir=args.model_dir, data_path=data_path, output_dir=f'{output_dir}-llava',
                batched_forward=True, copy_images=False, replace_existing_output=False, device=d)
    elif args.engine == 'transformers':
        from llavaguard.evaluation import eval_transformers
        eval_transformers.evaluation(model_dir=args.model_dir, data_path=data_path, output_dir=output_dir,
            batched_forward=True, copy_images=False, replace_existing_output=False, device=d)
    elif args.engine == 'sglang_old':
            # old sglang version does not support openai-api
            print(f'SGLang evaluation using old API with sglang version {installed["sglang"]}')
            from llavaguard.server.sglang_old_api import launch_server_and_run_funct
            from llavaguard.evaluation.eval_sglang_old import evaluate_sglang
            d_path = data_path if data_path.endswith('.json') else f'{data_path}/test.json'
            launch_server_and_run_funct(model_dir=args.model_dir, device=d, function=evaluate_sglang,
                                    function_kwargs={'data_path': data_path,'output_dir': output_dir})
    else:
        evaluate(model_dir=args.model_dir, data_path=data_path, output_dir=output_dir, replace_existing_output=False, device=args.device, engine=args.engine,
                 batch_size=batch_size)

