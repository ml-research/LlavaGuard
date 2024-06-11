import argparse
import glob
import sys
import os
import json
import sglang as sgl
from sglang.lang.chat_template import get_chat_template


if '/workspace' not in sys.path:
    sys.path.append('/workspace')
from llavaguard.sglang.evaluation import set_up_dynamic_regex, chunks
from llavaguard.sglang.runtime_endpoint import RuntimeEndpoint
from llavaguard.taxonomy.policies import get_assessment_and_system_prompt
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


def guard_genai(replace_existing_output=False, tmpl_version='json-v10', port=None):
    # set up backend
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
    gen_ims = '/common-repos/LlavaGuard/generated_images/LlavaGuard/SD_1-5'
    guard_output_dir = f'/common-repos/LlavaGuard/generated_images/LlavaGuard/annot-{tmpl_version}'
    os.makedirs(guard_output_dir, exist_ok=True)

    _, prompt = get_assessment_and_system_prompt(tmpl_version)

    print(f'BATCH INFER: {batch_infer}, USE REGEX: {use_regex}, Chat template: {tmpl_version}')
    print(f'Model base: {model_base} using template: {chat_template}')
    print('Running sglang inference on generated images at:', gen_ims)
    im_paths = glob.glob(f'{gen_ims}/*.png')
    ids = [f.split('/')[-1].split('.')[0] for f in im_paths]
    im_paths = [f'{gen_ims}/{i}.png' for i in ids if
                not os.path.exists(f'{guard_output_dir}/{i}_lg.json') or replace_existing_output]
    rx = set_up_dynamic_regex(tmpl_version) if use_regex else None
    inputs = [{'prompt': prompt.replace('<image>', ''), 'image_path': im_path, 'rx': rx} for im_path in im_paths]
    num_batches = len(inputs) // 5000 + 1
    for i, batch in enumerate(chunks(inputs, num_batches)):
        print(f'Running batch {i + 1}/{num_batches}')
        batch = batch.tolist()
        out = guard_gen.run_batch(batch, progress_bar=True)
        out_ids = [i['image_path'].split('/')[-1].split('.')[0] for i in batch]
        for sample_id, out in zip(out_ids, out):
            if not os.path.exists(f'{guard_output_dir}/{sample_id}_lg.json') or replace_existing_output:
                with open(f'{guard_output_dir}/{sample_id}_lg.json', 'w') as f:
                    json.dump(out['json_output'], f, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLaVA Guard SGlang Inference on Generated Images')
    parser.add_argument('--replace_existing_output', action='store_true', help='Replace existing predictions')
    parser.add_argument('--template_version', type=str, default='json-v16', help='Template version')
    args = parser.parse_args()
    if isinstance(args.replace_existing_output, str):
        args.replace_existing_output = args.replace_existing_output.lower() in ['true', '1']
    MODEL_OUTPUT_DIR1 = '/common-repos/LlavaGuard/models/LlavaGuard-v1.1-7b-full/smid_and_crawled_v2_with_augmented_policies/json-v16'
    MODEL_OUTPUT_DIR2 = '/common-repos/LlavaGuard/models/LlavaGuard-v1.1-13b-full/smid_and_crawled_v2_with_augmented_policies/json-v16'
    MODEL_OUTPUT_DIR3 = '/common-repos/LlavaGuard/models/LlavaGuard-v1.2-34b-full/smid_and_crawled_v2_with_augmented_policies/json-v16'
    # guard_genai(replace_existing_output=args.replace_existing_output, tmpl_version=args.template_version)
    function_kwargs = {'replace_existing_output': args.replace_existing_output, 'tmpl_version': args.template_version}
    launch_server_and_run_funct(model_dir=MODEL_OUTPUT_DIR2, device=6, function=guard_genai,
                                function_kwargs=function_kwargs)
