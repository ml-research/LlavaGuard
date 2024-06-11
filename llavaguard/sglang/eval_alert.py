import sys

from datasets import load_dataset
from sglang import RuntimeEndpoint
from sglang.lang.chat_template import get_chat_template

if '/workspace' not in sys.path:
    sys.path.append('/workspace')

import rtpt
import sglang as sgl
import numpy as np
import os

import json

from transformers import set_seed
from llava.mm_utils import get_model_name_from_path
from llavaguard.taxonomy.policies import get_assessment_and_system_prompt


@sgl.function
def guard_gen(s, prompt, rx=None):
    _, policy = get_assessment_and_system_prompt('json-v10')
    policy = policy.replace('<image>', '')
    p1 = 'User content:' + prompt.replace('### Instruction:', '').replace('### Response:', '').replace('\n', '')
    s += sgl.user(prompt)
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


# set up backend
backend = RuntimeEndpoint("http://localhost:10000")
sgl.set_default_backend(backend)
if '34b' in backend.get_model_name():
    backend.chat_template = get_chat_template("chatml-llava")
else:
    backend.chat_template = get_chat_template('vicuna_v1.1')
chat_template = backend.get_chat_template()
model_base = backend.get_model_name()
root = '/common-repos/LlavaGuard' if os.path.exists('/common-repos/LlavaGuard') else 'output'
use_regex = False
batch_infer = True
hf_alert_ds = load_dataset('Babelscape/ALERT', 'alert', split='test')

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
    model_name = get_model_name_from_path(model_base)
    eval_output_dir = f"{root}/eval/{model_name}/foundation_model"
else:
    raise ValueError('Please provide a model_save_dir or model_base to load the model.')

eval_output_dir += f"/sglang{'-bi' if batch_infer else ''}{'-rx' if use_regex else ''}-ALERT-v2-llava-1.5-13b"

print(f'Chat template: {chat_template}')
print(f'BATCH INFER: {batch_infer}, USE REGEX: {use_regex}')
print(f'Model base: {model_base}')
print(f'Dataset: Babelscape/ALERT')
print(f'Evaluation output: {eval_output_dir}')

os.makedirs(f'{eval_output_dir}/model_output', exist_ok=True)

num_batches = len(hf_alert_ds) // 2000 + 1
rt = rtpt.RTPT(name_initials='LH', experiment_name=f'LlavaGuard-ImageNet', max_iterations=num_batches)
rt.start()
for i, batch in enumerate(chunks(hf_alert_ds, num_batches)):
    print(f'Running batch {i + 1}/{num_batches}')
    batch = batch.tolist()
    inputs = [{'prompt': sample['prompt']} for sample in batch]
    outs = guard_gen.run_batch(inputs, progress_bar=True)
    for out, sample in zip(outs, batch):
        sample['Llavaguard_output'] = out['json_output']
        with open(f'{eval_output_dir}/model_output/{sample["id"]}.json', 'w+') as f:
            json.dump(sample, f, indent=4)
    rt.step()
