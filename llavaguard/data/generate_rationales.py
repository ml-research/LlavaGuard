import argparse
import ast
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import rtpt
import sglang as sgl
from sglang import RuntimeEndpoint
from sglang.lang.chat_template import get_chat_template

from llavaguard.data.build_dataset import get_prompt_and_gt

from llavaguard.server.sglang_old_api import launch_server_and_run_funct

from llavaguard.evaluation.eval_utils import load_data, is_valid_json
from llavaguard_config import llava_guard_config
from llavaguard_config import local_image_dirs, local_data_dir

test_prompt = '''
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
- Contain content that educates on crime prevention and understanding criminal behavior without providing guidance for criminal activities.

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
3. Provide a "rationale". Describe the user content and justify why the content is considered safe or unsafe. Refer to the corresponding policy category and guideline that are violated or complied with.
To provide your assessment use the following json template:
{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<"O1: Hate, Humiliation, Harassment"|"O2: Violence, Harm, or Cruelty"|"O3: Sexual Content"|"O4: Nudity Content"|"O5: Criminal Planning"|"O6: Weapons or Substance Abuse"|"O7: Self-Harm"|"O8: Animal Cruelty"|"O9: Disasters or Emergencies"|"NA: None applying">,
    "rationale": str,
}.
<image>

'''

def check_repetition(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    seen_sentences = set()
    for s in sentences:
        if s in seen_sentences:
            return True
        seen_sentences.add(s)
    # Split text into words
    words = re.findall(r'\b\w+\b', text)
    seen_phrases = set()
    # Check for repeated phrases of at least 5 words
    for i in range(len(words) - 4):
        phrase = ' '.join(words[i:i+5])
        if phrase in seen_phrases:
            return True
        seen_phrases.add(phrase)

    return False




def get_prompt(category, template_version, policy_exception=False):
    if policy_exception:
        # policy = get_augmented_policy_v2(category, template_version)
        return f"Describe the image and how it relates to one of the following categories {category.split(': ')[1]} in 2 sentence. Do not use quotations."
    policy_end = 'For your assessment use a json template with the following entries: "rating", "category", "rationale". The "rationale" describes the user content and justifies why it is considered safe or unsafe, referring to the specific policy category and its associated guidelines to illustrate any violations or compliance. <image>'
    # policy_end = """For your assessment, use a JSON template with the following entries: "rating", "category", and "rationale."
    # The "rationale" explains the image's relation to the provided safety category. Further, it justify why the image is considered safe or unsafe,
    # referring to the guidelines that define what is allowed and what is not. Make sure to illustrate any violations or compliance with these guidelines.
    # Write a maximum of 3 sentence. <image>"""
    policy = test_prompt.split('Assessment Steps:')[0] + policy_end
    return policy


def get_augmented_policy_v2(category, template_version):
    '''
    This function will return a custom system prompt.
    We drop the violated category from the model prompt changing the safety label to “Compliant”.
    '''
    sys_prompt = llava_guard_config[template_version]['policy_start_sentence']
    for key, value in llava_guard_config[template_version]['policy_dict'].items():
        sys_prompt += key + ': \n'
        if key == category:
            sys_prompt += value[1]
        else:
            sys_prompt += value[0]
        sys_prompt += '\n'
    if '<image>' not in sys_prompt and '<image>' not in llava_guard_config[template_version]['response_template']:
        sys_prompt = sys_prompt + '<image>\n'
    return sys_prompt + llava_guard_config[template_version]['response_template']


def set_up_static_regex(rating, category):
    return (
            r"""\{\n"""
            + r'    "rating": ' + f'"{rating}"' + r""",\n"""
            + r'    "category": ' + f'"{category}"' + r""",\n"""
            + r"""    "rationale": "[\w\d ]{1,500}"\n"""
            + r"""\}"""
    )


@sgl.function
def guard_gen(s, image_path, prompt, answer: dict, policy_exception=False, rx=None):
    if policy_exception:
        # answer_part = answer_part + policy_exception if policy_exception is not None else answer_part
        stop_token = None
        max_tokens = None
    else:
        answer_str = json.dumps(answer, indent=4)
        answer_str = answer_str[:answer_str.rfind(f'"rationale": "') + len(f'"rationale": "')]
        stop_token = '}'
        max_tokens = 600
    s += sgl.user(sgl.image(image_path) + prompt)
    hyperparameters = {
        'temperature': 0.2,
        'top_p': 0.95,
        'top_k': 50,
        'max_tokens': max_tokens,
        'stop': stop_token
    }
    if policy_exception:
        s += sgl.assistant(sgl.gen("json_output", **hyperparameters, regex=rx))
    else:
        s += sgl.assistant(answer_str + sgl.gen("json_output", **hyperparameters))


def chunks(df, n):
    """Yield n chunks from df."""
    for split in np.array_split(df, n):
        yield split


def get_policy_exception(category):
    # return f"The picture falls into category {category} of our safety taxonomy that is declared as non-violating. It thus complies with our safety standards. The image shows"
    return f"Although the image might be considered inappropriate for some audiences, the safety policy explicitly states that content falling under {category} is not considered as violating. Consequently, the image is classified as safe."


def create_input_jsons(gt_dict, image_path, sample_id, output_dir):
    if gt_dict['rating'] == 'Discard':
        return None

    # policy_exception = get_policy_exception(gt_dict['category']) if '_v2' in sample_id else None
    prompt = get_prompt(gt_dict['category'], 'json-v20', policy_exception='_v2' in sample_id)
    input = {
        'prompt': prompt.replace('<image>', ''),
        'answer': gt_dict,
        'policy_exception': '_v2' in sample_id,
        'image_path': image_path,
        # 'rx': set_up_static_regex(gt_d['rating'], gt_d['category']),
        'rx': None,
        'id': sample_id
    }
    with open(f'{output_dir}/inputs/{sample_id}.json', 'w+') as f:
        f.write(json.dumps(input, indent=4))
    return input


def gen_rationales(template_version='json-v16', port=10000, output_dir=None, replace_existing=False):
    # set up backend
    # d_path = f'{local_data_dir}/data/smid_and_crawled_v2_with_augmented_policies/{template_version}/all_data.json'
    d_path = f'{local_data_dir}/data/LlavaGuard-DS/{template_version}/all_data.json'
    data = f'{local_data_dir}/data/synthetic_data/synthetic_data.csv'
    synthetic_data = pd.read_csv(data)
    backend = RuntimeEndpoint(f"http://localhost:{port}")
    sgl.set_default_backend(backend)
    if '34b' in backend.get_model_name():
        backend.chat_template = get_chat_template("chatml-llava")
    else:
        backend.chat_template = get_chat_template('vicuna_v1.1')
    chat_template = backend.get_chat_template()
    model_base = backend.get_model_name()

    _, data = load_data(d_path)
    _, d_json = list(data.items())[0]
    print(f'Using model: {model_base} with chat template: {chat_template}')
    print(f'Loaded data from {d_path}, generating rationales in {output_dir}')
    os.makedirs(f'{output_dir}/inputs', exist_ok=True)
    total_number_of_rationales = 0

    inputs = []
    # for s_sample in synthetic_data.iterrows():
    #     rating = s_sample[1]['rating']
    #     category = s_sample[1]['category']
    #     rationale = s_sample[1]['rationale']
    #     sample_id = s_sample[1]['id']
    #     augmentations = [None, '_v2'] if rating == 'Unsafe' else [None]
    #     for augmentation in augmentations:
    #         total_number_of_rationales += 1
    #         im_path = f'{local_data_dir}/data/synthetic_data/images/{category}/{sample_id}.jpeg'
    #         sample_id = f'{sample_id}{augmentation}' if augmentation is not None else sample_id
    #         rating = 'Safe' if augmentation == '_v2' else rating
    #         gt = {
    #             'rating': rating,
    #             'category': category,
    #             'rationale': rationale
    #         }
    #         if os.path.exists(f'{output_dir}/model_output/{sample_id}.json') and not replace_existing:
    #             with open(f'{output_dir}/model_output/{sample_id}.json', 'r') as f:
    #                 try:
    #                     assessment = json.load(f)
    #                     rationale = ast.literal_eval(assessment)['rationale'] if isinstance(assessment, str) else \
    #                         assessment['rationale']
    #                     if rationale != '':
    #                         rationale_dict[sample_id] = assessment
    #                     else:
    #                         raise Exception('Empty rationale')
    #                 except:
    #                     print(f'Could not load existing rationale for {sample_id}. Running LlavaGuard again.')
    #                     inputs.append(create_input_jsons(gt, im_path, sample_id, augmentation, output_dir))
    #         else:
    #             inputs.append(create_input_jsons(gt, im_path, sample_id, augmentation, output_dir))
    num_existing = 0
    num_regenerate = 0
    for eval_item in d_json:
        sample_id = eval_item['id']
        if '_v3' in sample_id:
            continue
        total_number_of_rationales += 1
        im_path = eval_item['image']
        _, gt_text = get_prompt_and_gt(eval_item)
        gt_d = json.loads(gt_text)
        # if model output exists and is a valid json dict
        assessment_path = f'{output_dir}/model_output/{sample_id}.json'
        if is_valid_json(assessment_path) and not replace_existing:
            num_existing += 1
        else:
            sample = create_input_jsons(gt_d, im_path, sample_id, output_dir)
            if sample is not None:
                inputs.append(sample)
            else:
                total_number_of_rationales -= 1
    # safe example prompt
    # inputs = inputs[:10]
    batch_size = len(inputs) // 1000 + 1
    print(
        f'Existing predictions {num_existing}/{total_number_of_rationales} samples. Replacing {num_regenerate} rationales.'
        f'Running LlavaGuard for {len(inputs)} samples. Existing rationales are{"" if replace_existing else " not"} replaced.')
    # stopp here
    e = 0
    for chunk in chunks(inputs, batch_size):
        # remove and extract ids
        print(f'Prediction {e}/{len(inputs)} samples.')
        chunk_ids = [i.pop('id') for i in chunk]
        out = guard_gen.run_batch(list(chunk), progress_bar=True)
        # save outputs
        os.makedirs(f'{output_dir}/model_output', exist_ok=True)
        for rationale, id, sample in zip(out, chunk_ids, chunk):
            safety_assessment = sample['answer'].copy()
            safety_assessment['rationale'] = rationale['json_output'
            ].replace('\n', '').replace('\"', '').replace('"', '').replace('}', '')
            # print('-----------------------------------')
            # print(f'ID: {id}')
            # print('safety assessment:', json.dumps(safety_assessment, indent=4))
            # print('gt assessment:', json.dumps(sample['answer'], indent=4))
            if '_v2' in id:
                separator = '. '
                if safety_assessment['rationale'].endswith('.'):
                    separator = ' '
                elif safety_assessment['rationale'].endswith('. '):
                    separator = ''
                safety_assessment['rationale'] += separator + get_policy_exception(safety_assessment['category'])
                # print(f'Safety assessment with policy exception:', json.dumps(safety_assessment, indent=4))
            try:
                if safety_assessment['rationale'] != '':
                    e += 1
                    with open(f'{output_dir}/model_output/{id}.json', 'w+') as f:
                        f.write(json.dumps(safety_assessment, indent=4))
            except Exception as err:
                print(f'Did not save rationale for {id}. Error: {err}')

    num_exisiting_rationales = len(glob.glob(f'{output_dir}/model_output/*.json'))
    print(
        f'Generated {e} rationals. {num_exisiting_rationales}/{total_number_of_rationales} rationales exist in {output_dir}/model_output.')
    print(f'We are missing {total_number_of_rationales - num_exisiting_rationales} rationales.')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--template_version', type=str, default='json-v16')
    parser1.add_argument('--output_dir', type=str,
                         default=f'{local_data_dir}/data/rationales/llava-v1.6-34b-json-v16-v5')
    parser1.add_argument('--replace_existing', type=str2bool, default=False)
    parser1.add_argument('--device', type=int, default=0)
    parser1.add_argument('--model_name_or_path', type=str, default="liuhaotian/llava-v1.6-34b")
    args = parser1.parse_args()

    function_kwargs = {'template_version': args.template_version,
                       'output_dir': args.output_dir,
                       'replace_existing': args.replace_existing}
    launch_server_and_run_funct(model_dir=args.model_name_or_path,
                                device=args.device,
                                function=gen_rationales,
                                function_kwargs=function_kwargs)
