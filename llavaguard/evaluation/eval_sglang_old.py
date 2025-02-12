import argparse
import glob

from sglang import RuntimeEndpoint
# from sglang.lang.chat_template import get_chat_template, chat_template_registry, ChatTemplate

import rtpt
import sglang as sgl
import os

from sglang.lang.chat_template import get_chat_template, chat_template_registry, register_chat_template, ChatTemplate
from sglang.srt.conversation import SeparatorStyle, register_conv_template, Conversation, chat_template_exists
from tqdm import tqdm
import json

from transformers import set_seed

from llavaguard.evaluation.eval_utils import load_data, model_dir_to_output_dir, set_up_dynamic_regex
from llavaguard.evaluation.metrics_calculator import EvaluationMetricsCalculator
from llavaguard.server.sglang_old_api import launch_server_and_run_funct
from llavaguard_config import local_image_dirs, local_data_dir


@sgl.function
def guard_gen(s, image_path, prompt, tmpl_version, rx=None):
    # if int(tmpl_version.split('v')[1]) >= 20:
    #     s += sgl.system(prompt)
    #     s += sgl.user(sgl.image(image_path))
    # else:
    s += sgl.user(sgl.image(image_path) + prompt)

    hyperparameters = {
        'temperature': 0.2,
        'top_p': 0.95,
        'top_k': 50,
        'max_tokens': 500,
        # 'stop': "}",
    }
    if rx is None:
        s += sgl.assistant(sgl.gen("json_output", **hyperparameters))
    else:
        s += sgl.assistant(sgl.gen("json_output", **hyperparameters, regex=rx))


def run_sglang(emc, prompts, gts, ids, im_paths, tmpl_version, rx=None):
    print('Running sglang inference')
    # update prompt with conversation template
    # run batches of size 200
    b_size = 400
    for i in range(0, len(prompts), b_size):
        print(f'Running chunk {i + 1}/{1 + len(prompts) // b_size}')
        b_size = min(b_size, len(prompts) - i)
        prompts_b, gts_b, ids_b, im_paths_b = prompts[i:i + b_size], gts[i:i + b_size], ids[i:i + b_size], im_paths[
                                                                                                           i:i + b_size]
        inputs = [{'prompt': p.replace('<image>', ''), 'image_path': im_path, 'rx': rx,
                   'tmpl_version': tmpl_version} for p, im_path in zip(prompts_b, im_paths_b)]
        out = guard_gen.run_batch(inputs, progress_bar=True)
        for sample_id, out, gt, prompt in zip(ids_b, out, gts_b, prompts_b):
            emc.add_sample(sample_id, out['json_output'], gt, prompt, save_output=True)


def run_sglang_single(emc, prompts, gts, ids, im_paths, tmpl_version, rx=None):
    # single forward
    rt = rtpt.RTPT(name_initials='LH', experiment_name=f'LlavaGuard-Eval', max_iterations=len(prompts) + 1)
    rt.start()
    pbar = tqdm(zip(prompts, gts, ids, im_paths), total=len(prompts))
    for prompt, gt, sample_id, im_path in pbar:
        prompt = prompt.replace('<image>', '')

        metrics = emc.get_metrics()
        pbar.set_description(
            f'Evaluating TP: {metrics["TP"]}, FP: {metrics["FP"]}, TN: {metrics["TN"]}, FN: {metrics["FN"]}, '
            f'Invalid: {metrics["invalid_assessments"]}')
        out = guard_gen.run(
            image_path=im_path,
            prompt=prompt,
            rx=rx,
            tmpl_version=tmpl_version)
        emc.add_sample(sample_id, out['json_output'], gt, prompt, save_output=True)
        rt.step()


def chat_template_llava_llama_3():
    ct = ChatTemplate(
        name="llava_llama_3",
        default_system_prompt="You are a helpful language and vision assistant. " "You are able to understand the visual content that the user provides, " "and assist the user with a variety of tasks using natural language.",
        role_prefix_and_suffix={
            "system": (
                "<|start_header_id|>system<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
            "user": (
                "<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
            "assistant": (
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
        },
        stop_str=("<|eot_id|>",),
        image_token=" <image>\n",
    )
    if not chat_template_exists("llava_llama_3"):
        register_conv_template(Conversation(
            name="llava_llama_3",
            system_message="You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
            system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
            roles=("user", "assistant"),
            sep_style=SeparatorStyle.LLAMA3,
            sep="",
            stop_str=["<|end_of_text|>", "<|eot_id|>"],
        ))
        print("Registered Conversation: llava_llama_3")
    if "llava_llama_3" not in chat_template_registry.keys():
        register_chat_template(ct)
        print("Registered Chat Template: llama-3-instruct")
    return ct


def get_template(model_base):
    if '34b' in model_base:
        return get_chat_template("chatml-llava")
    elif '8b' in model_base:
        return chat_template_llava_llama_3()
    else:
        return get_chat_template('vicuna_v1.1')


def evaluate_sglang(data_path='smid_and_crawled_policy/json-v4',
                    output_dir: str = None,
                    replace_existing_output=False, 
                    port=10000):
    # set up backend
    backend = RuntimeEndpoint(f"http://localhost:{port}")
    model_base = backend.get_model_name()
    
    if '34b' in model_base:
        backend.chat_template = get_template(model_base)
    # sglang.srt.server.launch_server()
    # ServerArgs.add_cli_args(parser)
    sgl.set_default_backend(backend)
    # root = f'{local_data_dir}' if os.path.exists(f'{local_data_dir}') else 'output'
    data_paths, data = load_data(data_path)
    templ_version = data_path.split('/')[-1]
    use_regex = False
    batch_infer = True
    save_eval_images = True
    
    os.makedirs(output_dir, exist_ok=True)
    # set seed
    set_seed(48)
    conv = None
    # d_path = f"{data_paths['eval'].split('/')[-3]}-{data_paths['eval'].split('/')[-2]}"
    # eval_output_dir += f"/sglang{'-bi' if batch_infer else ''}{'-rx' if use_regex else ''}-{d_path}"
    for d_name, d_json in data.items():
        p = data_paths[d_name]
        # eval_output_dir = f"{output_dir}/{p.split('/')[-3]}-{p.split('/')[-2]}-{d_name}"
        base, d_name = output_dir.split('/eval/')[0], '/'.join(output_dir.split('/eval/')[1].split('/')[1:])
        eval_im_dir =  base + '/eval/eval_ims/' + d_name
        print('#' * 30 + ' Evaluation Start ' + '#' * 30)
        print(f'Evaluating {d_name} dataset: {p}')
        print(f'Model base: {model_base}')
        print(f'Evaluation output: {output_dir}')
        print(f'BATCH INFER: {batch_infer}, USE REGEX: {use_regex}')
        print(f'Chat template: {backend.get_chat_template()}')

        os.makedirs(f'{output_dir}/model_output', exist_ok=True)
        os.makedirs(eval_im_dir, exist_ok=True)


        emc = EvaluationMetricsCalculator(pred_dir=f'{output_dir}/model_output', debug=True)
        prompts, gts, ids, im_paths = [], [], [], []
        save_prompt = 0
        e = 0
        for eval_item in d_json:
            sample_id = eval_item['id']
            gt = eval_item['conversations'][1]["value"]
            prompt = eval_item['conversations'][0]["value"]
            if save_prompt < 1:
                with open(f'{output_dir}/{d_name}_prompt_{save_prompt}.txt', 'w+') as f:
                    f.write(prompt)
                save_prompt += 1
            if save_eval_images and not os.path.exists(f'{eval_im_dir}/{sample_id}.png'):
                im_p = eval_item['image'].replace(" ", "\\ ")
                os.system(f'cp {im_p} {eval_im_dir}/{sample_id}.png')
            path = glob.glob(f'{output_dir}/model_output/{sample_id}.*')
            try:
                if len(path) > 0 and not replace_existing_output:
                    out = json.load(open(path[0]))
                    out = json.dumps(out['LlavaGuard'], indent=4) if 'LlavaGuard' in out else json.dumps(
                        out['prediction'], indent=4)
                    eval, metrics = emc.add_sample(sample_id, out, gt)
                    e += 1
                    # if isinstance(eval['prediction'], dict):
                    #     e += 1
                    # else:
                    #     raise ValueError
                else:
                    raise FileNotFoundError
            except:
                prompts.append(prompt)
                gts.append(gt)
                ids.append(sample_id)
                im_paths.append(eval_item['image'])
       
        if len(prompts) == 0:
            print(f'Existing predictions {e}/{len(d_json)} samples. No samples to evaluate')
        else:
            print(f'Existing predictions {e}/{len(d_json)} samples. Running LlavaGuard for {len(prompts)} remaining samples')
            # safe example prompt
            rx = set_up_dynamic_regex(templ_version) if use_regex else None
            if batch_infer:
                run_sglang(emc, prompts, gts, ids, im_paths, templ_version, rx=rx)
            else:
                run_sglang_single(emc, prompts, gts, ids, im_paths, templ_version, rx=rx)

        metrics_name = f'{output_dir}/{d_name}_metrics.json' if 'no_edge_cases' not in data_path else f'{output_dir}/{d_name}_metrics_no_edge_cases.json'
        out_name = f'{output_dir}/{d_name}_results.txt' if 'no_edge_cases' not in data_path else f'{output_dir}/{d_name}_results_no_edge_cases.txt'
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
    print(f'Running evaluation (SGLang turned on") for model: {args.model_dir} on data: {args.data_path}')
    print(f'Output directory: {output_dir}')
    launch_server_and_run_funct(model_dir=args.model_dir, device=d, function=evaluate_sglang,
                            function_kwargs={'data_path': args.data_path,'output_dir': output_dir})
