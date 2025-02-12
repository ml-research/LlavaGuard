import random

from llavaguard_config import llava_guard_config


def get_augmented_policy_v1(category, template_version):
    '''
    This function will return a custom system prompt. We drop/allow a random number of categories from the
    taxonomy that are not relevant to the image. The number of categories dropped is randomly selected between 1 and 8.
    '''
    sys_prompt = llava_guard_config[template_version]['policy_start_sentence']
    for key, value in llava_guard_config[template_version]['policy_dict'].items():
        sys_prompt += key + ': \n'
        if key == category:
            sys_prompt += value[0]
        else:
            sys_prompt += value[1] if random.choice([True, False]) else value[0]
        sys_prompt += '\n'
    if '<image>' not in sys_prompt and '<image>' not in llava_guard_config[template_version]['response_template']:
        sys_prompt = sys_prompt + '<image>\n'
    return sys_prompt + llava_guard_config[template_version]['response_template']


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


def get_augmented_policy_v2a(category, template_version):
    '''
    This function will return a custom system prompt. We declare some categories from the taxonomy as non-violating.
     The image category is not dropped from the model prompt.
     The number of categories dropped is randomly selected between 1 and 3.
    '''
    sys_prompt = llava_guard_config[template_version]['policy_start_sentence']
    keys = list(llava_guard_config[template_version]['policy_dict'].keys())
    if category != 'NA: None applying':
        keys.remove(category)
    augmentations = random.sample(keys, random.randint(0, 2))
    for key, value in llava_guard_config[template_version]['policy_dict'].items():
        sys_prompt += key + ': \n'
        if key == category:
            sys_prompt += value[1]
        elif key in augmentations:
            sys_prompt += value[1]
        else:
            sys_prompt += value[0]
        sys_prompt += '\n'
    if '<image>' not in sys_prompt and '<image>' not in llava_guard_config[template_version]['response_template']:
        sys_prompt = sys_prompt + '<image>\n'
    return sys_prompt + llava_guard_config[template_version]['response_template']


def get_augmented_policy_v3(category, template_version):
    '''
    This function will return a custom system prompt. We declare some categories from the taxonomy as non-violating.
     The image category is not dropped from the model prompt.
     The number of categories dropped is randomly selected between 1 and 3.
    '''
    sys_prompt = llava_guard_config[template_version]['policy_start_sentence']
    keys = list(llava_guard_config[template_version]['policy_dict'].keys())
    if category != 'NA: None applying':
        keys.remove(category)
    augmentations = random.sample(keys, random.randint(1, 3))
    for key, value in llava_guard_config[template_version]['policy_dict'].items():
        sys_prompt += key + ': \n'
        if key == category:
            sys_prompt += value[0]
        elif key in augmentations:
            sys_prompt += value[1]
        else:
            sys_prompt += value[0]
        sys_prompt += '\n'
    if '<image>' not in sys_prompt and '<image>' not in llava_guard_config[template_version]['response_template']:
        sys_prompt = sys_prompt + '<image>\n'
    return sys_prompt + llava_guard_config[template_version]['response_template']


def get_augmented_policy_v4(category, template_version):
    '''
    This function will return a custom system prompt.
    We drop the violated category from the model prompt changing the safety label to “review needed”.
    '''
    if int(template_version.split('-v')[-1].split('-')[0]) < 12:
        raise ValueError('v4 augmentation is only available for template versions 12 and above')
    sys_prompt = llava_guard_config[template_version]['policy_start_sentence']
    for key, value in llava_guard_config[template_version]['policy_dict'].items():
        sys_prompt += key + ': \n'
        if key == category:
            sys_prompt += value[2]
        else:
            sys_prompt += value[0]
        sys_prompt += '\n'
    if '<image>' not in sys_prompt and '<image>' not in llava_guard_config[template_version]['response_template']:
        sys_prompt = sys_prompt + '<image>\n'
    return sys_prompt + llava_guard_config[template_version]['response_template']


def get_augmented_policy_v5(category, template_version):
    '''
    This function will return a custom system prompt. We declare some categories from the taxonomy as sensitive.
     The image category is not dropped from the model prompt.
     The number of categories dropped is randomly selected between 1 and 3.
    '''
    if int(template_version.split('-v')[-1].split('-')[0]) < 12:
        raise ValueError('v5 augmentation is only available for template versions 12 and above')
    sys_prompt = llava_guard_config[template_version]['policy_start_sentence']
    keys = list(llava_guard_config[template_version]['policy_dict'].keys())
    if category != 'NA: None applying':
        keys.remove(category)
    augmentations = random.sample(keys, random.randint(1, 3))
    for key, value in llava_guard_config[template_version]['policy_dict'].items():
        sys_prompt += key + ': \n'
        if key == category:
            sys_prompt += value[0]
        elif key in augmentations:
            sys_prompt += value[2]
        else:
            sys_prompt += value[0]
        sys_prompt += '\n'
    if '<image>' not in sys_prompt and '<image>' not in llava_guard_config[template_version]['response_template']:
        sys_prompt = sys_prompt + '<image>\n'
    return sys_prompt + llava_guard_config[template_version]['response_template']


policy_augmentation_functions = {
    'v1': get_augmented_policy_v1,
    'v2': get_augmented_policy_v2,
    'v2a': get_augmented_policy_v2a,
    'v3': get_augmented_policy_v3,
    'v4': get_augmented_policy_v4,
    'v5': get_augmented_policy_v5
}
