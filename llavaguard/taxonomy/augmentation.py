import ast
import glob
import json
import os.path
import random
import re

from llavaguard.taxonomy.assessment import get_mapping
from llavaguard.taxonomy.augmentation_funct import policy_augmentation_functions
from llavaguard.taxonomy.policy import get_assessment_and_system_prompt
from llavaguard.evaluation.metrics_calculator import parse_json


def get_explanation(pp, score):
    explanation = None
    try:
        p = json.load(open(pp))
        # check if prediction is a valid dictionary
        if isinstance(p, str):
            p = parse_json(p)
            p = ast.literal_eval(p)
        if not isinstance(p, dict):
            raise ValueError(f'Invalid prediction format. Expected dict, got {type(p)}')
    except FileNotFoundError:
        raise ValueError(f'Missing prediction: {pp}')
    except json.JSONDecodeError:
        raise ValueError(f'Failed loading JSON. Invalid prediction')
    if 'explanation' in p.keys():
        if p['score'] == score:
            explanation = p['explanation']
        else:
            raise ValueError('Predicted score does not match the ground truth score.')
    elif 'rationale' in p.keys():
        explanation = p['rationale']
    elif 'prediction' in p.keys():
        prediction = p['prediction']
        try:
            if 'decision' in prediction.keys() and 'assessment' in prediction.keys() and prediction['decision'] == \
                    p['GT']['decision']:
                explanation = prediction['assessment']
            else:
                raise ValueError('Predicted score does not match the ground truth score.')
        except Exception as e:
            raise ValueError('Invalid prediction')
    else:
        raise ValueError('Invalid Json missing dict keys')
    explanation = explanation.replace('  ', ' ').replace('}', '')
    # check if explanation contains repetitive words
    sentences = re.split(r'(?<=[.!?]) +', explanation)
    seen_sentences = set()
    for s in sentences:
        if s in seen_sentences:
            raise ValueError('Regenerate Rationales. Repetitive sentences in explanation: ' + explanation)
        seen_sentences.add(s)

    return explanation


def create_sample(data, image_folder, pred_path, system_prompt, assessment: callable, unique_id_suffix=None,
                  counter=[0, 0]):
    sample = {}
    # remove last 2 characters from json name
    sample['id'] = image_folder.split('/')[-1].replace(' ', '_') + '_' if 'real_images' in image_folder else ''
    if 'json' in data.keys():
        sample['id'] += data['json'].split(".")[0][:-2]
    else:
        sample['id'] += data['id']

    # augmented prompts are not of high quality we rather opt for the default prompt and append pol exception
    if unique_id_suffix == 'v2a' and os.path.exists(f'{pred_path}/{sample["id"]}_v2.json'):
        pred_file = f'{pred_path}/{sample["id"]}_v2.json'
    elif unique_id_suffix is not None and os.path.exists(f'{pred_path}/{sample["id"]}_{unique_id_suffix}.json'):
        pred_file = f'{pred_path}/{sample["id"]}_{unique_id_suffix}.json'
    elif os.path.exists(f'{pred_path}/{sample["id"]}.json'):
        pred_file = f'{pred_path}/{sample["id"]}.json'
    elif os.path.exists(f'{pred_path}/{data["json"]}'):
        pred_file = f'{pred_path}/{data["json"]}'
    else:
        pred_file = None

    # pred_file = f"{pred_path}/{sample['id']}.json" if 'llava-v1.6-34b' in pred_path else f"{pred_path}/{data['json']}"
    try:
        if os.path.isfile(image_folder):
            sample['image'] = image_folder
        else:
            sample['image'] = glob.glob(f'{image_folder}/{data["json"].split(".")[0][:-2]}.*')[0]
        # print(f'{image_folder}/{sample["id"]}.*')
        # print(im_paths)
    except FileNotFoundError:
        raise FileNotFoundError(f'Missing image: {sample["image"]}')
    try:
        explanation = get_explanation(pred_file, data['score'])
    except ValueError as e:
        print(f'Prediction dropped for {sample["id"]} due to Error: {e}')
        explanation = None
    if explanation is None:
        counter[1] += 1
    else:
        counter[0] += 1
    sample['id'] += f'_{unique_id_suffix}' if unique_id_suffix is not None else ''
    sample['final-assessment'] = 'Compliant' if 'Acceptable' in data['score'] else 'Review Needed'
    sample['score'] = data['score']
    sample['category'] = data['category']
    if '<image>' not in system_prompt:
        raise ValueError('Invalid system prompt. Missing <image> tag')
    sample['conversations'] = [
        {
            "from": "human",
            "value": system_prompt
        },
        {
            "from": "gpt",
            "value": assessment(data['score'], data['category'], explanation)
        }
    ]
    return sample


def create_samples_with_augmented_policies(data, image_folder, pred_path, template_version, augmentation_strategy=None,
                                           counter=[0, 0]):
    if 'Discard Sample' in data['score'] or 'Discard' in data['score']:
        print('Sample is marked for discarding. Skipping sample')
        return []
    counter_start = counter.copy()
    # category = 
    mapping = get_mapping(template_version)
    policy_category = mapping[str(data['category'])]
    # update policy category
    data['category'] = policy_category
    
    if policy_category == '10: None applying' or policy_category == 'NA: None applying' and 'Inappropriate' in data[
        'score']:
        print(
            f'Skipping sample: Inconsistent label! Category: {policy_category} with safety label: {data["score"]}.')
        return []
    t_version = int(template_version.split('-v')[-1].split('-')[0])
    assessment, system_prompt = get_augmented_assessment_and_system_prompt(template_version, data['score'],
                                                                           policy_category, None)
    samples = [create_sample(data, image_folder, pred_path, system_prompt, assessment, counter=counter)]
    if augmentation_strategy is None or augmentation_strategy == []:
        return samples

    if 'Inappropriate' in data['score'] and (t_version == 14 or t_version == 15):
        samples.append(create_sample(data, image_folder, pred_path, system_prompt, assessment,
                                     unique_id_suffix='v0', counter=counter))

    if 'Acceptable' in data['score']:
        augmentation_strategy = []
        # handle exception cases for v14 and v15
        if t_version == 14 or t_version == 15:
            if policy_category != '10: None applying' and policy_category != 'NA: None applying':
                augmentation_strategy = ['v4']
                # at 10 percent of the time we will add v3 augmentation
            if random.randint(0, 100) < 10:
                augmentation_strategy.append('v3')
            if random.randint(0, 100) < 25:
                augmentation_strategy.append('v5')

    for aug in augmentation_strategy:
        assessment, system_prompt = get_augmented_assessment_and_system_prompt(template_version, data['score'],
                                                                               policy_category, aug.split('_')[0])
        samples.append(create_sample(data, image_folder, pred_path, system_prompt, assessment, unique_id_suffix=aug,
                                     counter=counter))
    counter_dif = sum([counter[0] - counter_start[0], counter[1] - counter_start[1]])
    if len(samples) != counter_dif:
        print(f'Inconsistent counter increase. Expected {counter_dif} samples, got {len(samples)} samples')
    return samples


def get_augmented_assessment_and_system_prompt(template_version: str, pre_aug_score: str, policy_category: str, augmentation):
    '''
    This function will return the assessment and system prompt based on the augmentation type.
    '''

    def get_assessment_for_exception(sco, cat, explanation=None):
        ex = '' if explanation is None else explanation
        return get_assessment_json(exception, cat, ex)


    get_assessment_json, default_policy = get_assessment_and_system_prompt(template_version)

    if augmentation is None:
        return get_assessment_json, default_policy
    if policy_category == '10: None applying' or policy_category == 'NA: None applying':
        raise ValueError(f'Can not augment sys prompt if category is {policy_category}')
    if augmentation in policy_augmentation_functions.keys():
        augmented_policy = policy_augmentation_functions[augmentation](policy_category, template_version)
    else:
        raise ValueError(f'Invalid augmentation type: {augmentation}')

    if augmentation == 'v2' or augmentation == 'v2a':
        exception = 'Non-Violating Exception'
        return get_assessment_for_exception, augmented_policy
    elif augmentation == 'v4':
        exception = 'Sensitive Exception'
        return get_assessment_for_exception, augmented_policy
    return get_assessment_json, augmented_policy
