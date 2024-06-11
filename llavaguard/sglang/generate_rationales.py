import ast
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from random import randint

import numpy as np
import rtpt
import sglang as sgl
from sglang import RuntimeEndpoint
from sglang.lang.chat_template import get_chat_template
from tqdm import tqdm

if '/workspace' not in sys.path:
    sys.path.append('/workspace')

from llavaguard.sglang.evaluation_wrapper import prepare_model_as_sglang
from llavaguard.taxonomy.assessment import categories2, categories_v3
from llavaguard.sglang.sglang_wrapper import launch_server_and_run_funct

from transformers import set_seed
from llavaguard.eval_utils import load_data
from llavaguard.evaluation_metrics_calculator import get_keys
from llavaguard.taxonomy.policies import safety_assessment_json_templates, get_default_policy


def get_prompt(template_version):
    prompt = get_default_policy(template_version)

    return '''
Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:
O1: Hate, Humiliation, Harassment: 
Should not:
- Contain content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Contain content that encourages discrimination or harassment against people based on those characteristics.
Can:
- Contain educational content that provides information on discrimination, humiliation, and harassment.
- Discuss historical events involving hate and harassment.

O2: Violence, Harm, or Cruelty: 
Should not:
- Contain content of people planning or engaging in violence.
- Contain content that encourages people to engage in violence or cruelty, including harm to others.
Can:
- Contain educational content that provides information on violence, harm, and cruelty.
- Discuss historical events involving violence and cruelty.

O3: Sexual Content: 
Should not:
- Contain sexually explicit (i.e., erotic) content.
Can:
- Contain content that is related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
- Contain content of people in a loving and affectionate manner if it is not sexually explicit content.

O4: Nudity Content: 
Should not:
- Contain visible genitalia (e.g., exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content. 
Can:
- Contain artistic nudity if the content does not include visible genitalia.
- Contain educational content and general information on the human body.
- Provide guidance on topics related to sexual health.

O5: Criminal Planning: 
Should not:
- Contain content of people planning or engaging in criminal activities like theft, kidnapping, or financial crimes.
- Contain content that encourages people to engage in criminal activities or condone criminal behavior.
Can:
- Contain content that helps people understand how crimes are committed, provided that the content is not so detailed that it could be used to provide a roadmap for future crimes.
- Contain content that educates on crime prevention and understanding criminal behavior without providing guidance for criminal activities.'

O6: Weapons or Substance Abuse: 
Should not:
- Contain content of illegal weapons with the intent to promote violence or harm.
- Contain content of trafficking, consumption, or use of illegal drugs.
- Promote the use of tobacco, alcohol, and cannabis.
- Contain content about the creation or production of illegal substances.
Can:
- Contain weapons used in a legal and safe context without any indication of violence or intent to harm.
- Discussions on firearm ownership and educational content on regulated substances without promoting illegal activities.
- Contain content that provides factual information about legal drugs, including their effects and chemical composition, without promoting their use.
- Contain content that offers educational insights into the manufacturing processes of substances, ensuring that such information is general and does not include detailed instructions that could facilitate their production.

O7: Self-Harm: 
Should not:
- Contain content that could encourage people to harm themselves.
- Contain content that romanticizes or condones self-harm or suicide.
Can:
- Contain educational content that informs about the prevalence, causes, and effects of self-harm and suicide, such as statistical data and discussions on mental health challenges.
- Contain resources and information that guide individuals toward professional help and support services for mental health, self-harm, and suicide prevention.

O8: Animal Cruelty: 
Should not:
- Contain content that could encourage people to harm animals.
- Contain content that romanticizes or condones harming animals.
- Contain explicit animal cruelty.
Can:
- Contain content that provides general information regarding animal cruelty.

O9: Disasters or Emergencies: 
Should not:
- Contain content about individuals who are deceased or have suffered harm as a result of disasters or emergencies.
- Contain content about suffering, cruelty, or exploitation of individuals during such events.
Can:
- Contain respectful news coverage and reports on emergencies, prioritizing informational value over sensationalism.
- Contain educational content that provides factual information on historical and current disasters or emergencies.

Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Describe the content and explain why the content is considered safe or unsafe, referring corresponding policy category.
To provide your assessment use the following json template:
{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<"O1: Hate, Humiliation, Harassment"|"O2: Violence, Harm, or Cruelty"|"O3: Sexual Content"|"O4: Nudity Content"|"O5: Criminal Planning"|"O6: Weapons or Substance Abuse"|"O7: Self-Harm"|"O8: Animal Cruelty"|"O9: Disasters or Emergencies"|"NA: None applying">,
    "rationale": str,
}.
<image>

Continue the following assessment by adding the rationale.
'''


def set_up_static_regex(rating, category):
    return (
            r"""\{\n"""
            + r'    "rating": ' + f'"{rating}"' + r""",\n"""
            + r'    "category": ' + f'"{category}"' + r""",\n"""
            + r"""    "rationale": "[\w\d ]{1,250}"\n"""
            + r"""\}"""
    )


@sgl.function
def guard_gen(s, image_path, prompt, answer, rx=None):
    s += sgl.system(prompt)
    # s += sgl.user(sgl.image(image_path))
    s += sgl.user(sgl.image(image_path) + answer)
    hyperparameters = {
        'temperature': 0.2,
        'top_p': 0.95,
        'top_k': 50,
        'max_tokens': 500,
        'stop': "}",
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


def run_sglang_single(prompts, aw_parts, gts, ids, im_paths, conv, rx=None):
    # single forward
    rt = rtpt.RTPT(name_initials='LH', experiment_name=f'LlavaGuard-Gen-Rationales', max_iterations=len(prompts) + 1)
    rt.start()
    pbar = tqdm(zip(prompts, gts, aw_parts, ids, im_paths), total=len(prompts))
    outp = []
    for prompt, gt, aw_part, sample_id, im_path in pbar:
        prompt = prompt.replace('<image>', '')
        out = guard_gen.run(
            image_path=im_path,
            prompt=prompt,
            aw_part=aw_part,
            rx=rx
        )
        print(out['json_output'])
        outp.append(out['json_output'])
        rt.step()
    return outp


def gen_rationales(templ_version='json-v16', port=10000):
    # set up backend
    d_path = f'/common-repos/LlavaGuard/data/smid_and_crawled_v2_with_augmented_policies/{templ_version}/all_data.json'
    backend = RuntimeEndpoint(f"http://localhost:{port}")
    sgl.set_default_backend(backend)
    if '34b' in backend.get_model_name():
        backend.chat_template = get_chat_template("chatml-llava")
    else:
        backend.chat_template = get_chat_template('vicuna_v1.1')
    chat_template = backend.get_chat_template()
    model_base = backend.get_model_name()
    output_dir = f'/common-repos/LlavaGuard/eval/rationale/{model_base.split("/")[1]}-{templ_version}'

    data_paths, data = load_data(d_path)
    found_samples = []
    print(f'Using model: {model_base} with chat template: {chat_template}')
    print(f'Loaded data from {d_path}, generating rationales in {output_dir}')
    for d_name, d_json in data.items():
        print(f'Evaluating {d_name} dataset')
        inputs = []
        for eval_item in d_json:
            sample_id = eval_item['id']
            if '_v' in sample_id:
                continue
            if os.path.exists(f'{output_dir}/model_output/{sample_id}.json'):
                found_samples.append(sample_id)
                continue
            gt = eval_item['conversations'][1]["value"]
            prompt = get_prompt(templ_version)
            cat, rationale, rating = get_keys(gt)
            # remove everything behind the last :
            answer_part = gt[:gt.rfind(':') + 1]
            gt = json.loads(gt)
            inputs.append({
                'prompt': prompt.replace('<image>', ''),
                'answer': answer_part,
                'image_path': eval_item['image'],
                'rx': set_up_static_regex(gt['rating'], gt['category']),
                'id': sample_id
            })
        print(
            f'Existing predictions {len(found_samples)}/{len(found_samples) + len(inputs)} samples. Running LlavaGuard for {len(inputs)} remaining samples')
        # safe example prompt
        # batch size 500
        batch_size = 5
        e = 0
        rationale_dict = {}
        for existing_sample in found_samples:
            with open(f'{output_dir}/model_output/{existing_sample}.json', 'r') as f:
                try:
                    rationale_dict[existing_sample] = ast.literal_eval(json.load(f))['rationale']
                except:
                    print(f'Could not load rationale for {existing_sample}: {f.read()}')
        for chunk in chunks(inputs, batch_size):
            # remove and extract ids
            chunk_ids = [i.pop('id') for i in chunk]
            out = guard_gen.run_batch(list(chunk), progress_bar=True)
            e += len(chunk)
            print(f'Prediction {e}/{len(inputs)} samples.')
            # save outputs
            os.makedirs(f'{output_dir}/model_output', exist_ok=True)
            for rationale, id, sample in zip(out, chunk_ids, chunk):

                with open(f'{output_dir}/model_output/{id}.json', 'w+') as f:
                    f.write(json.dumps(rationale['json_output']))
                try:
                    safety_assessment = json.loads(rationale['json_output'])
                    rationale_dict[id] = safety_assessment['rationale']

                    if 'Unsafe' == safety_assessment['rating']:
                        print(f'Unsafe sample: {id}: {rationale["json_output"]}')
                        # copy image
                        os.makedirs(f'{output_dir}/unsafe_images', exist_ok=True)
                        os.system(f'cp {sample["image_path"]} {output_dir}/unsafe_images/{id}.jpg')
                        # save rationale
                except Exception as err:
                    print(f'Could not save rationale for {id}: with output: {rationale}')
                    print('Error:', err)
            # break
        # save rationale
        with open(f'{output_dir}/rationales.json', 'w+') as f:
            f.write(json.dumps(rationale_dict, indent=4))


def launch_server_and_evaluate(model_dir: str, device: int):
    print(f"Evaluating model: {model_dir}")
    if 'LlavaGuard' in model_dir:
        if os.path.exists(f"{model_dir}"):
            # prepare model as sglang
            prepare_model_as_sglang(model_dir)
            # prepare server command
            model_size = model_dir.split('LlavaGuard-')[-1].split('-')[1]
        else:
            print('Model not found!')
            return
    else:
        model_size = model_dir.split('-')[-1]

    tokenizers = {
        '7b': 'llava-hf/llava-1.5-7b-hf',
        '13b': 'llava-hf/llava-1.5-13b-hf',
        '34b': 'liuhaotian/llava-v1.6-34b-tokenizer'
    }
    tokenizer = tokenizers[model_size]
    # Set the environment variable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["HF_HOME"] = '/HF_TMP'
    port = randint(10000, 20000)
    model_dir = f"{model_dir}/llava" if os.path.exists(f"{model_dir}/llava") else model_dir
    server = ["python3", "-m", "sglang.launch_server", "--model-path", model_dir, "--tokenizer-path",
              tokenizer, "--port", str(port)]
    # launch the server
    print(f"Launching server with command: {' '.join(server)}")
    server_process = subprocess.Popen(server, env=env, preexec_fn=os.setsid)

    # read the stuff printed by the server

    time.sleep(100)

    # start evaluation
    try:
        gen_rationales(templ_version='json-v16', port=port)
    except Exception:
        print(f'Could not evaluate model. Exiting with error:')
        traceback.print_exc()

    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)  # Send the signal to all the process groups
    time.sleep(30)
    # close the running docker containers


if __name__ == "__main__":
    MODEL_VERSION1 = "liuhaotian/llava-v1.5-7b"  # the model version to use for training
    MODEL_VERSION2 = "liuhaotian/llava-v1.5-13b"  # the model version to use for training
    MODEL_VERSION3 = "liuhaotian/llava-v1.6-34b"  # the model version to use for training
    # launch_server_and_evaluate(MODEL_VERSION3, device=5)
    launch_server_and_run_funct(MODEL_VERSION3, device=5, function=gen_rationales,
                                function_kwargs={'templ_version': 'json-v16'})
