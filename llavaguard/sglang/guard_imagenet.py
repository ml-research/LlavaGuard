import argparse
import glob
import sys
import sglang as sgl
from sglang import RuntimeEndpoint
from sglang.lang.chat_template import get_chat_template
import os
import json
import torch
import pandas as pd

if '/workspace' not in sys.path:
    sys.path.append('/workspace')
from llavaguard.sglang.evaluation import set_up_dynamic_regex, chunks
from llavaguard.taxonomy.policies import get_assessment_and_system_prompt
from rtpt import rtpt

from llavaguard.sglang.sglang_wrapper import launch_server_and_run_funct

@sgl.function
def guard_gen(s, image_path, prompt, rx=None):
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


def evaluate_imagenet(replace_existing_output=False, tmpl_version='json-v10', port=None, run=None):
    # set up backend

    # filtered images
    unsafe_image_net = '/storage-01/ml-pschramowski/repositories/Q16/data/ViT-B-16/imagenet1k_train/inapp_images.csv'
    image_net_path = '/storage-01/datasets/imagenet/train'
    in_data = pd.read_csv(unsafe_image_net)
    ids = in_data.iloc[:, -1].tolist()
    im_paths = [f'{image_net_path}/{i.split("_")[0]}/{i}' for i in ids]

    image_net_path = '/storage-01/datasets/imagenet/train'
    im_paths_all = glob.glob(f'{image_net_path}/*/*')
    # split impaths into 7 runs
    runs = 7
    im_paths_runs = [im_paths_all[i:i + len(im_paths_all) // runs] for i in
                     range(0, len(im_paths_all), len(im_paths_all) // runs)]
    if isinstance(run, int):
        im_paths = im_paths_runs[run - 1]
    elif isinstance(run, list):
        im_paths = []
        for r in run:
            im_paths += im_paths_runs[r - 1]
    else:
        im_paths = im_paths_all

    #
    # sa = ServerArgs(
    #     model_path='/common-repos/LlavaGuard/models/LlavaGuard-v1.1-13b-full/smid_and_crawled_v2_with'
    #                '_augmented_policies/json-v10/llava',
    #     tokenizer_path='llava-hf/llava-1.5-13b-hf', port=run * 10000)
    #
    # launch_server(server_args=sa, pipe_finish_writer=None)


    port = port or 10000
    backend = RuntimeEndpoint(f"http://localhost:{port}")
    sgl.set_default_backend(backend)
    if '34b' in backend.get_model_name():
        backend.chat_template = get_chat_template("chatml-llava")
    else:
        backend.chat_template = get_chat_template('vicuna_v1.1')
    chat_template = backend.get_chat_template()
    model_base = backend.get_model_name()
    use_regex = False
    batch_infer = True
    rx = None

    guard_output_dir = f'/common-repos/LlavaGuard/imagenet_annot/whole/{tmpl_version}'
    os.makedirs(guard_output_dir, exist_ok=True)
    _, prompt = get_assessment_and_system_prompt(tmpl_version)
    print(f'Starting run {run} ################')
    print(f'BATCH INFER: {batch_infer}, USE REGEX: {use_regex}, Prompt template: {tmpl_version}')
    print(f'Model base: {model_base} using template: {chat_template}')
    print(
        f'Running sglang batch inference run {run}: on imagenet {len(im_paths)} images (of total {len((im_paths_all))}) from',
        image_net_path)
    rx = set_up_dynamic_regex(tmpl_version) if use_regex else None
    num_batches = len(im_paths) // 2000 + 1
    rt = rtpt.RTPT(name_initials='LH', experiment_name=f'LG-ImNet-worker-{run}', max_iterations=len(im_paths))
    rt.start()
    t = torch.tensor([0]).to(f'cuda:{run}')

    for i, batch in enumerate(chunks(im_paths, num_batches)):
        print(f'Running batch {i + 1}/{num_batches}')
        batch = batch.tolist()
        inputs, out_paths = [], []
        for im_path in batch:
            rt.step()
            input_id = im_path.split('/')[-1].split('.')[0]
            o_pth = f'{guard_output_dir}/{input_id}.json'
            if os.path.exists(o_pth) and not replace_existing_output:
                continue
            inputs.append({'prompt': prompt.replace('<image>', ''), 'image_path': im_path, 'rx': rx})
            out_paths.append(o_pth)

        outs = guard_gen.run_batch(inputs, progress_bar=True)
        for out, p in zip(outs, out_paths):
            with open(p, 'w+') as f:
                json.dump(out['json_output'], f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLaVA Guard SGlang Inference on Generated Images')
    parser.add_argument('--replace_existing_output', action='store_true', help='Replace existing predictions')
    parser.add_argument('--template_version', type=str, default='json-v16', help='Template version')
    parser.add_argument('--run', type=int, default=0, help='Run number')
    args = parser.parse_args()
    if isinstance(args.replace_existing_output, str):
        args.replace_existing_output = args.replace_existing_output.lower() in ['true', '1']
    MODEL_OUTPUT_DIR2 = '/common-repos/LlavaGuard/models/LlavaGuard-v1.1-13b-full/smid_and_crawled_v2_with_augmented_policies/json-v16'

    # evaluate_imagenet(replace_existing_output=args.replace_existing_output, tmpl_version=args.template_version)
    function_kwargs = {'replace_existing_output': args.replace_existing_output, 'tmpl_version': args.template_version,
                       'run': args.run}
    launch_server_and_run_funct(model_dir=MODEL_OUTPUT_DIR2, device=args.run, function=evaluate_imagenet,
                                function_kwargs=function_kwargs)
