import argparse
import glob
import sys

from sglang import RuntimeEndpoint
from sglang.lang.chat_template import get_chat_template

from llavaguard.taxonomy.assessment import categories2, categories_v3

if '/workspace' not in sys.path:
    sys.path.append('/workspace')

import rtpt
import sglang as sgl
import numpy as np
import os

from tqdm import tqdm
import json

from transformers import set_seed
from llavaguard.eval_utils import load_data
from llavaguard.evaluation_metrics_calculator import EvaluationMetricsCalculator
from llavaguard.taxonomy.policies import safety_assessment_json_templates


def set_up_static_regex():
    return (
            r"""\{\n"""
            + r"""    "image-category": "(O1: Hate, Humiliation, Harassment|O2: Violence, Harm or Cruelty|O3: Sexual Content|O4: Nudity Content|O5: Criminal Planning|O6: Illegal Weapons or Substance Abuse|O7: Self-Harm|O8: Animal Cruelty|O9: Disasters or Emergencies|10: None applying)",\n"""
            + r"""    "assessment": "[\w\d ]{1,250}",\n"""
            + r"""    "decision": "(Review Needed|Compliant)",\n"""
            + r"""\}"""
    )


def set_up_dynamic_regex(template='json-v10'):
    cats = categories2 if template in ['json-v10', 'json-v11'] else categories_v3
    cats_txt = '|'.join(cats)
    if template not in safety_assessment_json_templates:
        raise ValueError(f'Unknown template: {template}')
    j_templ = repr(safety_assessment_json_templates[template])
    j_templ = j_templ.split('{')[1].split('}')[0]
    # j_templ.replace("'", '')
    j_templ = j_templ.replace('str<"Review Needed"|"Compliant">', r""" "(Review Needed|Compliant)" """)
    j_templ = j_templ.replace(
        'str<"O1: Hate, Humiliation, Harassment"|"O2: Violence, Harm or Cruelty"|"O3: Sexual Content"|"O4: Nudity Content"|"O5: Criminal Planning"|"O6: Illegal Weapons or Substance Abuse"|"O7: Self-Harm"|"O8: Animal Cruelty"|"O9: Disasters or Emergencies"|"10: None applying">',
        f'"({cats_txt})"')
    j_templ = j_templ.replace('str', r""" "[\w\d ]{1,250}" """)
    j_templ = '\{' + j_templ + '\}'
    # to raw string
    return j_templ


@sgl.function
def guard_gen(s, image_path, prompt, rx=None):
    # s += sgl.system(prompt)
    # s += sgl.user(sgl.image(image_path))
    s += sgl.user(sgl.image(image_path) + prompt)
    hyperparameters = {
        'temperature': 0.2,
        'top_p': 0.95,
        'top_k': 50,
        'max_tokens': 500,
        # 'stop': "}",
    }
    if rx is None:
        s += sgl.assistant(
            sgl.gen("json_output", **hyperparameters))
    else:
        s += sgl.assistant(sgl.gen("json_output", **hyperparameters, regex=rx))


def chunks(df, n):
    """Yield n chunks from df."""
    for split in np.array_split(df, n):
        yield split


def run_sglang(emc, prompts, gts, ids, im_paths, conv, rx=None):
    print('Running sglang inference')
    # update prompt with conversation template
    # run batches of size 200
    b_size = 400
    for i in range(0, len(prompts), b_size):
        print(f'Running chunk {i + 1}/{1 + len(prompts) // b_size}\n')
        b_size = min(b_size, len(prompts) - i)
        prompts_b, gts_b, ids_b, im_paths_b = prompts[i:i + b_size], gts[i:i + b_size], ids[i:i + b_size], im_paths[
                                                                                                           i:i + b_size]
        inputs = [{'prompt': p.replace('<image>', ''), 'image_path': im_path, 'rx': rx} for p, im_path in
                  zip(prompts_b, im_paths_b)]
        out = guard_gen.run_batch(inputs, progress_bar=True)
        for sample_id, out, gt, prompt in zip(ids_b, out, gts_b, prompts_b):
            emc.add_sample(sample_id, out['json_output'], gt, prompt, save_output=True)
    # inputs = [{'prompt': p.replace('<image>', ''), 'image_path': im_path, 'rx': rx} for p, im_path in
    #           zip(prompts, im_paths)]
    #
    # out = guard_gen.run_batch(inputs, progress_bar=True)
    # for sample_id, out, gt, prompt in zip(ids, out, gts, prompts):
    #     emc.add_sample(sample_id, out['json_output'], gt, prompt, save_output=True)


def run_sglang_single(emc, prompts, gts, ids, im_paths, conv, rx=None):
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
            rx=rx
        )
        emc.add_sample(sample_id, out['json_output'], gt, prompt, save_output=True)
        rt.step()


def evaluate_sglang(data_path='smid_and_crawled_policy/json-v4', infer_train_data: bool = False,
                    replace_existing_output=False,
                    port=10000):
    # set up backend
    backend = RuntimeEndpoint(f"http://localhost:{port}")
    sgl.set_default_backend(backend)
    if '34b' in backend.get_model_name():
        backend.chat_template = get_chat_template("chatml-llava")
    else:
        backend.chat_template = get_chat_template('vicuna_v1.1')
    chat_template = backend.get_chat_template()
    # sglang.srt.server.launch_server()
    # ServerArgs.add_cli_args(parser)
    model_base = backend.get_model_name()
    root = '/common-repos/LlavaGuard' if os.path.exists('/common-repos/LlavaGuard') else 'output'
    split = 'eval' if not infer_train_data else ['all_data']
    data_paths, data = load_data(data_path, split)
    templ_version = data_path.split('/')[-1]
    use_regex = False
    batch_infer = True
    save_eval_images = True

    # set seed
    set_seed(48)
    if 'llava' in model_base[-len('llava'):]:
        # load fine-tuned models
        model_base = model_base[:model_base.rfind('/')]
        run_name = model_base.split("models/")[1]
        model_name = run_name.split("/")[0]
        eval_output_dir = f'{root}/eval/{run_name}'
    elif model_base is not None:
        # load foundation models
        model_name = model_base.split('/')[-1]
        eval_output_dir = f"{root}/eval/{model_name}/foundation_model"
    else:
        raise ValueError('Please provide a model_save_dir or model_base to load the model.')

    d_path = f"{data_paths['eval'].split('/')[-3]}-{data_paths['eval'].split('/')[-2]}"
    eval_output_dir += f"/sglang{'-bi' if batch_infer else ''}{'-rx' if use_regex else ''}-{d_path}"
    eval_im_dir = f'{root}/eval/eval_ims/{templ_version}'
    print(f'Chat template: {chat_template}')
    print(f'BATCH INFER: {batch_infer}, USE REGEX: {use_regex}')
    print(f'Model base: {model_base}')
    print(f'Dataset: {data_path}')
    print(f'Evaluation output: {eval_output_dir}')

    os.makedirs(f'{eval_output_dir}/model_output', exist_ok=True)
    os.makedirs(eval_im_dir, exist_ok=True)

    # if "34b" in model_base.lower():
    #     conv_mode = "chatml_direct"
    # else:
    #     conv_mode = "v1"
    # conv = conv_templates[conv_mode].copy()
    conv = None
    for d_name, d_json in data.items():
        print(f'Evaluating {d_name} dataset')
        emc = EvaluationMetricsCalculator(pred_dir=f'{eval_output_dir}/model_output', debug=True)
        prompts, gts, ids, im_paths = [], [], [], []
        save_prompt = 0
        e = 0
        # d_json = d_json[:800] if len(d_json) > 800 else d_json
        for eval_item in d_json:
            sample_id = eval_item['id']
            gt = eval_item['conversations'][1]["value"]
            prompt = eval_item['conversations'][0]["value"]
            if save_prompt < 1:
                with open(f'{eval_output_dir}/{d_name}_prompt_{save_prompt}.txt', 'w+') as f:
                    f.write(prompt)
                save_prompt += 1
            if save_eval_images and not os.path.exists(f'{eval_im_dir}/{sample_id}.png'):
                im_p = eval_item['image'].replace(" ", "\\ ")
                os.system(f'cp {im_p} {eval_im_dir}/{sample_id}.png')
            path = glob.glob(f'{eval_output_dir}/model_output/{sample_id}.*')
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
        print(
            f'Existing predictions {e}/{len(d_json)} samples. Running LlavaGuard for {len(prompts)} remaining samples')
        # safe example prompt
        rx = set_up_dynamic_regex(templ_version) if use_regex else None
        if batch_infer:
            run_sglang(emc, prompts, gts, ids, im_paths, conv, rx=rx)
        else:
            run_sglang_single(emc, prompts, gts, ids, im_paths, conv, rx=rx)

        metrics_name = f'{eval_output_dir}/{d_name}_metrics.json' if 'no_edge_cases' not in data_path else f'{eval_output_dir}/{d_name}_metrics_no_edge_cases.json'
        out_name = f'{eval_output_dir}/{d_name}_results.txt' if 'no_edge_cases' not in data_path else f'{eval_output_dir}/{d_name}_results_no_edge_cases.txt'
        emc.compute_stats(print_output=True, save_metric_path=metrics_name, save_txt_path=out_name)
        print('#' * 20 + 'Evaluation Done ' + '#' * 20)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLaVA Guard Evaluation')
    parser.add_argument('--data_path', type=str, default='smid_and_crawled_policy/json-v9',
                        help='dataset path either directory or json file')
    parser.add_argument('--infer_train_data', action='store_true',
                        help='Infer on training data, only possible if data_path is a directory')
    parser.add_argument('--replace_existing_output', action='store_true', help='Replace existing predictions')
    args = parser.parse_args()
    data_path = args.data_path
    infer_train_data = args.infer_train_data
    # string to bool conversion if needed
    if isinstance(args.replace_existing_output, str):
        args.replace_existing_output = args.replace_existing_output.lower() in ['true', '1']

    evaluate_sglang(data_path=data_path, infer_train_data=infer_train_data,
                    replace_existing_output=args.replace_existing_output)
