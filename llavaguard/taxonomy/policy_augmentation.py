import ast
import glob
import json
import random
import warnings
from json import JSONDecodeError

from llavaguard.taxonomy.custom_policy import custom_policy_dict
from llavaguard.taxonomy.policies import mapping, json_templates, get_assessment_and_system_prompt
from llavaguard.evaluation_metrics_calculator import parse_json


def get_augmented_assessment_and_system_prompt(template_version: str, score: str, category: str, augmentation):
    '''
    This function will return the assessment and system prompt based on the augmentation type.
    '''
    category = str(category)
    get_assessment_json, _ = get_assessment_and_system_prompt(template_version)
    system_prompt = get_sys_prompt(category, augmentation) + json_templates[template_version]
    if augmentation == 'v2' and 'Inappropriate' in score:
        def get_augmented_assessment_json(score, category, explanation=None):
            ex = '' if explanation is None else explanation + ' '

            return get_assessment_json('Policy Exception', category, ex)

        return get_augmented_assessment_json, system_prompt
    return get_assessment_json, system_prompt


def get_sys_prompt(category, augmentation):
    '''
    This function will return the system prompt based on the category and augmentation type.
    '''
    if augmentation is None:
        return get_sys_prompt_default(category)
    elif mapping[category] == '10: None applying':
        raise ValueError(f'Can not augment sys prompt if category is {category}')
    elif augmentation == 'v1':
        return get_sys_prompt_augmentation_v1(category)
    elif augmentation == 'v2':
        return get_sys_prompt_augmentation_v2(category)
    elif augmentation == 'v3':
        return get_sys_prompt_augmentation_v3(category)
    else:
        raise ValueError(f'Invalid augmentation type: {augmentation}')


def get_sys_prompt_augmentation_v1(category):
    '''
    This function will return a custom system prompt. We drop/allow a random number of categories from the
    taxonomy that are not relevant to the image. The number of categories dropped is randomly selected between 1 and 8.
    '''
    sys_prompt = 'Check if there is unsafe content in the image according to our safety policy with the categories as defined below.\n'
    for key, value in custom_policy_dict.items():
        sys_prompt += key + ': \n'
        if key == mapping[category]:
            sys_prompt += value[0]
        else:
            sys_prompt += value[1] if random.choice([True, False]) else value[0]
        sys_prompt += '\n'
    sys_prompt += '<image>\n'

    return sys_prompt


def get_sys_prompt_augmentation_v2(category):
    '''
    This function will return a custom system prompt.
    We drop the violation category from the model prompt changing the safety label to “Compliant”.
    '''
    sys_prompt = 'Check if there is unsafe content in the image according to our safety policy with the categories as defined below.\n'
    for key, value in custom_policy_dict.items():
        sys_prompt += key + ': \n'
        if key == mapping[category]:
            sys_prompt += value[1]
        else:
            sys_prompt += value[0]
        sys_prompt += '\n'
    sys_prompt += '<image>\n'

    return sys_prompt


def get_sys_prompt_augmentation_v3(category):
    '''
    This function will return a custom system prompt. We declare some categories from the taxonomy as non-violating.
     The violating category is not dropped from the model prompt.
     The number of categories dropped is randomly selected between 1 and 3.
    '''
    sys_prompt = 'Check if there is unsafe content in the image according to our safety policy with the categories as defined below.\n'
    keys = list(custom_policy_dict.keys())
    keys.remove(mapping[category])
    augmentations = random.sample(keys, random.randint(1, 3))
    for key, value in custom_policy_dict.items():
        sys_prompt += key + ': \n'
        if key == mapping[category]:
            sys_prompt += value[0]
        elif key in augmentations:
            sys_prompt += value[1]
        else:
            sys_prompt += value[0]
        sys_prompt += '\n'
    sys_prompt += '<image>\n'

    return sys_prompt


def get_sys_prompt_default(category):
    '''
    This function will return a custom system prompt.
    We drop the violation category from the model prompt changing the safety label to “Compliant”.
    '''
    sys_prompt = 'Check if there is unsafe content in the image according to our safety policy with the categories as defined below.\n'
    for key, value in custom_policy_dict.items():
        sys_prompt += key + ': \n'
        sys_prompt += value[0] + '\n'
    sys_prompt += '<image>\n'
    return sys_prompt


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
            print(prediction)
            print('Invalid prediction: Exception: ', e)
            return None
    else:
        raise ValueError('Invalid prediction format')


def create_sample(data, image_folder, pred_path, system_prompt, assessment: callable, unique_id_suffix=None,
                  counter=[0,0]):
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
    # try:
    #     prediction = json.load(open(f"{prediction_folder}/{data['json']}"))
    #     explanation = prediction['explanation'] if prediction['score'] == data['score'] else None
    # except FileNotFoundError:
    #     explanation = None
    #     warnings.warn(f'Missing prediction: {data["json"]}')
    explanation = get_explanation(pred_file, data['score'])
    if explanation is not None:
        counter[0] += 1
    else:
        counter[1] += 1
    sample['id'] += f'_{unique_id_suffix}' if unique_id_suffix is not None else ''
    sample['final-assessment'] = 'Compliant' if 'Acceptable' in data['score'] else 'Review Needed'
    sample['score'] = data['score']
    sample['category'] = data['category']
    # print(system_prompt)
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
                                           counter=[0,0]):
    category = str(data['category'])
    assessment, system_prompt = get_augmented_assessment_and_system_prompt(template_version, data['score'],
                                                                           category, None)
    sample_original = create_sample(data, image_folder, pred_path, system_prompt, assessment, counter=counter)
    if 'Inappropriate' in data['score'] and augmentation:
        augmentations = ['v3_a', 'v3_b', 'v3_c', 'v2'] if int(template_version.split('v')[-1]) > 9 else ['v1_a', 'v1_b',
                                                                                                         'v1_c', 'v2']
        samples = []
        if mapping[category] == '10: None applying':
            warnings.warn(f'DS inconsistency. Can not augment sys prompt if violated category is not provided. '
                          f'DS category is: {category}. Skipping sample....')
            return []
        for aug in augmentations:
            assessment, system_prompt = get_augmented_assessment_and_system_prompt(template_version, data['score'],
                                                                                   category, aug.split('_')[0])
            sample_v1 = create_sample(data, image_folder, pred_path, system_prompt, assessment, unique_id_suffix=aug,
                                      counter=counter)
            samples.append(sample_v1)
        return [sample_original] + samples
    return [sample_original]
