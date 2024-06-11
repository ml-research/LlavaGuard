import ast
import glob
import json
import random
import warnings

from llavaguard.taxonomy.assessment import get_mapping
from llavaguard.taxonomy.policies import safety_assessment_json_templates, get_assessment_and_system_prompt, \
    policy_start_sentence, custom_policy_dict
from llavaguard.evaluation_metrics_calculator import parse_json


def get_augmented_assessment_and_system_prompt(template_version: str, score: str, category: str, augmentation):
    '''
    This function will return the assessment and system prompt based on the augmentation type.
    '''

    def get_assessment_for_exception(sco, cat, explanation=None):
        ex = '' if explanation is None else explanation + ' '
        return get_assessment_json(exception, cat, ex)

    mapping = get_mapping(template_version)
    category = str(category)
    get_assessment_json, default_policy = get_assessment_and_system_prompt(template_version)
    if augmentation is None:
        system_prompt = default_policy
    elif mapping[category] == '10: None applying':
        raise ValueError(f'Can not augment sys prompt if category is {category}')
    elif augmentation == 'v1':
        system_prompt = get_augmented_policy_v1(category, template_version)
    elif augmentation == 'v2':
        system_prompt = get_augmented_policy_v2(category, template_version)
        exception = 'Non-Violating Exception'
        return get_assessment_for_exception, system_prompt
    elif augmentation == 'v3':
        system_prompt = get_augmented_policy_v3(category, template_version)
    elif augmentation == 'v4':
        system_prompt = get_augmented_policy_v4(category, template_version)
        exception = 'Sensitive Exception'
        return get_assessment_for_exception, system_prompt
    elif augmentation == 'v5':
        system_prompt = get_augmented_policy_v5(category, template_version)

    else:
        raise ValueError(f'Invalid augmentation type: {augmentation}')
    return get_assessment_json, system_prompt


def get_augmented_policy_v1(category, template_version):
    '''
    This function will return a custom system prompt. We drop/allow a random number of categories from the
    taxonomy that are not relevant to the image. The number of categories dropped is randomly selected between 1 and 8.
    '''
    sys_prompt = policy_start_sentence[template_version]
    mapping = get_mapping(template_version)
    for key, value in custom_policy_dict[template_version].items():
        sys_prompt += key + ': \n'
        if key == mapping[category]:
            sys_prompt += value[0]
        else:
            sys_prompt += value[1] if random.choice([True, False]) else value[0]
        sys_prompt += '\n'
    if '<image>' not in sys_prompt and '<image>' not in safety_assessment_json_templates[template_version]:
        sys_prompt = sys_prompt + '<image>\n'
    return sys_prompt + safety_assessment_json_templates[template_version]


def get_augmented_policy_v2(category, template_version):
    '''
    This function will return a custom system prompt.
    We drop the violated category from the model prompt changing the safety label to “Compliant”.
    '''
    sys_prompt = policy_start_sentence[template_version]
    mapping = get_mapping(template_version)
    for key, value in custom_policy_dict[template_version].items():
        sys_prompt += key + ': \n'
        if key == mapping[category]:
            sys_prompt += value[1]
        else:
            sys_prompt += value[0]
        sys_prompt += '\n'
    if '<image>' not in sys_prompt and '<image>' not in safety_assessment_json_templates[template_version]:
        sys_prompt = sys_prompt + '<image>\n'
    return sys_prompt + safety_assessment_json_templates[template_version]


def get_augmented_policy_v3(category, template_version):
    '''
    This function will return a custom system prompt. We declare some categories from the taxonomy as non-violating.
     The image category is not dropped from the model prompt.
     The number of categories dropped is randomly selected between 1 and 3.
    '''
    sys_prompt = policy_start_sentence[template_version]
    cat = get_mapping(template_version)[category]
    keys = list(custom_policy_dict[template_version].keys())
    if cat != 'NA: None applying':
        keys.remove(cat)
    augmentations = random.sample(keys, random.randint(1, 3))
    for key, value in custom_policy_dict[template_version].items():
        sys_prompt += key + ': \n'
        if key == cat:
            sys_prompt += value[0]
        elif key in augmentations:
            sys_prompt += value[1]
        else:
            sys_prompt += value[0]
        sys_prompt += '\n'
    if '<image>' not in sys_prompt and '<image>' not in safety_assessment_json_templates[template_version]:
        sys_prompt = sys_prompt + '<image>\n'
    return sys_prompt + safety_assessment_json_templates[template_version]


def get_augmented_policy_v4(category, template_version):
    '''
    This function will return a custom system prompt.
    We drop the violated category from the model prompt changing the safety label to “review needed”.
    '''
    if int(template_version.split('v')[-1]) < 12:
        raise ValueError('v4 augmentation is only available for template versions 12 and above')
    sys_prompt = policy_start_sentence[template_version]
    mapping = get_mapping(template_version)
    for key, value in custom_policy_dict[template_version].items():
        sys_prompt += key + ': \n'
        if key == mapping[category]:
            sys_prompt += value[2]
        else:
            sys_prompt += value[0]
        sys_prompt += '\n'
    if '<image>' not in sys_prompt and '<image>' not in safety_assessment_json_templates[template_version]:
        sys_prompt = sys_prompt + '<image>\n'
    return sys_prompt + safety_assessment_json_templates[template_version]


def get_augmented_policy_v5(category, template_version):
    '''
    This function will return a custom system prompt. We declare some categories from the taxonomy as sensitive.
     The image category is not dropped from the model prompt.
     The number of categories dropped is randomly selected between 1 and 3.
    '''
    if int(template_version.split('v')[-1]) < 12:
        raise ValueError('v5 augmentation is only available for template versions 12 and above')
    sys_prompt = policy_start_sentence[template_version]
    cat = get_mapping(template_version)[category]
    keys = list(custom_policy_dict[template_version].keys())
    if cat != 'NA: None applying':
        keys.remove(cat)
    augmentations = random.sample(keys, random.randint(1, 3))
    for key, value in custom_policy_dict[template_version].items():
        sys_prompt += key + ': \n'
        if key == cat:
            sys_prompt += value[0]
        elif key in augmentations:
            sys_prompt += value[2]
        else:
            sys_prompt += value[0]
        sys_prompt += '\n'
    if '<image>' not in sys_prompt and '<image>' not in safety_assessment_json_templates[template_version]:
        sys_prompt = sys_prompt + '<image>\n'
    return sys_prompt + safety_assessment_json_templates[template_version]


def get_explanation(pp, score):
    try:
        p = json.load(open(pp))
    except FileNotFoundError:
        warnings.warn(f'Missing prediction: {pp}')
        return None
    if 'explanation' in p.keys():
        return p['explanation'] if p['score'] == score else None
    elif 'prediction' in p:
        prediction = p['prediction']
        try:
            if isinstance(prediction, str):
                prediction = parse_json(prediction)
                # prediction = json.loads(prediction)
                prediction = ast.literal_eval(prediction)
            if 'decision' in prediction.keys() and 'assessment' in prediction.keys() and prediction['decision'] == \
                    p['GT']['decision']:
                return prediction['assessment']
            else:
                return None
        except Exception as e:
            # print(prediction)
            raise ValueError('Invalid prediction')
    else:
        raise ValueError('Invalid prediction format')


def create_sample(data, image_folder, pred_path, system_prompt, assessment: callable, unique_id_suffix=None,
                  counter=[0, 0]):
    sample = {}
    # remove last 2 characters from json name
    sample['id'] = image_folder.split('/')[-1].replace(' ', '_') + '_' if 'real_images' in image_folder else ''
    sample['id'] += data['json'].split(".")[0][:-2]

    pred_file = f"{pred_path}/{sample['id']}.json" if 'llava-v1.6-34b' in pred_path else f"{pred_path}/{data['json']}"
    try:
        im_paths = glob.glob(f'{image_folder}/{data["json"].split(".")[0][:-2]}.*')
        # print(f'{image_folder}/{sample["id"]}.*')
        # print(im_paths)
        sample['image'] = im_paths[0]
    except FileNotFoundError:
        raise FileNotFoundError(f'Missing image: {sample["image"]}')
    try:
        explanation = get_explanation(pred_file, data['score'])
    except ValueError as e:
        print(f'Prediction dropped for {sample["id"]}. {e}')
        explanation = None
    if explanation is not None:
        counter[0] += 1
    else:
        counter[1] += 1
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


def create_samples_with_augmented_policies(data, image_folder, pred_path, template_version, augmentation=True,
                                           counter=[0, 0]):
    if 'Discard Sample' in data['score']:
        print('Sample discarded')
        return []
    counter_start = counter.copy()
    category = str(data['category'])
    mapping = get_mapping(template_version)
    if mapping[category] == '10: None applying' or mapping[category] == 'NA: None applying' and 'Inappropriate' in data[
        'score']:
        print(
            f'Skipping sample: Inconsistent label! Category: {category} with safety label: {data["score"]}.')
        return []
    t_version = int(template_version.split('v')[-1])
    assessment, system_prompt = get_augmented_assessment_and_system_prompt(template_version, data['score'],
                                                                           category, None)
    samples = [create_sample(data, image_folder, pred_path, system_prompt, assessment, counter=counter)]
    if not augmentation:
        return samples

    if t_version < 9:
        augmentations = ['v1_a', 'v1_b', 'v1_c', 'v2'] if 'Inappropriate' in data['score'] else []
    elif t_version < 14:
        augmentations = ['v3_a', 'v3_b', 'v3_c', 'v2'] if 'Inappropriate' in data['score'] else []
    elif template_version == 'json-v14' or template_version == 'json-v15':
        if 'Inappropriate' in data['score']:
            samples.append(create_sample(data, image_folder, pred_path, system_prompt, assessment,
                                         unique_id_suffix='v0', counter=counter))
            augmentations = ['v2', 'v3_a', 'v3_b', 'v3_c', 'v5']
        elif 'Acceptable' in data['score']:
            augmentations = []
            if mapping[category] != '10: None applying' and mapping[category] != 'NA: None applying':
                augmentations = ['v4']
                # at 10 percent of the time we will add v3 augmentation
            if random.randint(0, 100) < 10:
                augmentations.append('v3')
            if random.randint(0, 100) < 25:
                augmentations.append('v5')
        else:
            raise ValueError(f'Invalid score {data["score"]}.')
    elif template_version == 'json-v16' or template_version == 'json-v17':
        augmentations = ['v3_a', 'v3_b', 'v3_c', 'v2'] if 'Inappropriate' in data['score'] else []
        # if 'Inappropriate' in data['score']:
        #     augmentations = ['v2', 'v3_a', 'v3_b', 'v3_c']
        #     # if random.randint(0, 100) < 50:
        #     #     augmentations.append('v5')
        # elif 'Acceptable' in data['score']:
        #     augmentations = []
        #     if mapping[category] != '10: None applying' and mapping[category] != 'NA: None applying':
        #         if random.randint(0, 100) < 40:
        #             augmentations = ['v4']
        #         # at 10 percent of the time we will add v3 augmentation
        #     if random.randint(0, 100) < 10:
        #         augmentations.append('v3')
        #     if random.randint(0, 100) < 15:
        #         augmentations.append('v5')
        # else:
        #     raise ValueError(f'Invalid score {data["score"]}.')
    else:
        raise ValueError(f'Invalid template version {template_version}.')

    for aug in augmentations:
        assessment, system_prompt = get_augmented_assessment_and_system_prompt(template_version, data['score'],
                                                                               category, aug.split('_')[0])
        samples.append(create_sample(data, image_folder, pred_path, system_prompt, assessment, unique_id_suffix=aug,
                                     counter=counter))
    counter_dif = sum([counter[0] - counter_start[0], counter[1] - counter_start[1]])
    if len(samples) != counter_dif:
        print(f'Inconsistent counter increase. Expected {counter_dif} samples, got {len(samples)} samples')
    return samples
