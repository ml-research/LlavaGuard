from llavaguard.taxonomy.assessment import get_assessment_nl, get_assessment_json_v0, get_assessment_json, \
    get_assessment_json_v3, get_assessment_json_v4, get_assessment_json_v5, get_assessment_json_v6, \
    get_assessment_json_v7, get_assessment_json_v9, get_assessment_json_v12, get_assessment_json_v11, \
    get_assessment_json_v13, json_template_v0, json_template_v1, json_template_v2, json_template_v3, json_template_v4, \
    json_template_v5, json_template_v6, json_template_v7, json_template_v8, json_template_v11, json_template_v12, \
    json_template_v13, json_template_v14, json_template_v15, json_template_v16, json_template_v17, categories1, \
    categories2, categories_v3
from llavaguard.taxonomy.custom_taxonomy import custom_policy_dict_v1, custom_policy_dict_v2
from llavaguard.taxonomy.default_taxonomy import policy_system_prompt, policy_system_prompt_v6, \
    policy_system_prompt_v8

safety_assessment_json_keys = {
    'nl': {'rating': 'final-assessment', 'category': 'category'},
    'json-v0': {'rating': 'final-assessment', 'category': 'policy-category'},
    'json-v1': {'rating': 'final-assessment', 'category': 'policy-category', 'rationale': 'explanation'},
    'json-v2': {'rating': 'final-assessment', 'category': 'policy-category', 'rationale': 'explanation'},
    'json-v3': {'rating': 'final-assessment', 'category': 'policy-category', 'rationale': 'explanation'},
    'json-v4': {'rating': 'decision', 'category': 'image-category', 'rationale': 'assessment'},
    'json-v5': {'rating': 'final-assessment'},
    'json-v6': {'rating': 'assessment', 'category': 'category', 'rationale': 'explanation'},
    'json-v7': {'rating': 'decision', 'category': 'image-category', 'rationale': 'assessment'},
    'json-v8': {'rating': 'decision', 'category': 'image-category', 'rationale': 'assessment'},
    'json-v9': {'rating': 'decision', 'category': 'image-category', 'rationale': 'assessment'},
    'json-v10': {'rating': 'decision', 'category': 'image-category', 'rationale': 'assessment'},
    'json-v11': {'rating': 'decision', 'category': 'image-category', 'rationale': 'assessment'},
    'json-v12': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
    'json-v13': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
    'json-v14': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
    'json-v15': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
    'json-v16': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
    'json-v17': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
}

safety_assessment_json_templates = {
    'nl': " If unsafe, the answer must include a comma-separated explanation with a list of violated categories.",
    'json-v0': json_template_v0,
    'json-v1': json_template_v1,
    'json-v2': json_template_v2,
    'json-v3': json_template_v3,
    'json-v4': json_template_v4,
    'json-v5': json_template_v5,
    'json-v6': json_template_v6,
    'json-v7': json_template_v7,
    'json-v8': json_template_v8,
    'json-v9': json_template_v7,
    'json-v10': json_template_v8,
    'json-v11': json_template_v11,
    'json-v12': json_template_v12,
    'json-v13': json_template_v13,
    'json-v14': json_template_v14,
    'json-v15': json_template_v15,
    'json-v16': json_template_v16,
    'json-v17': json_template_v17,
}

safety_assessment_functions = {
    'nl': get_assessment_nl,
    'json-v0': get_assessment_json_v0,
    'json-v1': get_assessment_json,
    'json': get_assessment_json,
    'json-v2': get_assessment_json,
    'json-v3': get_assessment_json_v3,
    'json-v4': get_assessment_json_v4,
    'json-v5': get_assessment_json_v5,
    'json-v6': get_assessment_json_v6,
    'json-v7': get_assessment_json_v7,
    'json-v8': get_assessment_json_v7,
    'json-v9': get_assessment_json_v9,
    'json-v10': get_assessment_json_v9,
    'json-v11': get_assessment_json_v11,
    'json-v12': get_assessment_json_v12,
    'json-v13': get_assessment_json_v13,
    'json-v14': get_assessment_json_v12,
    'json-v15': get_assessment_json_v13,
    'json-v16': get_assessment_json_v12,
    'json-v17': get_assessment_json_v13,
}

policy_start_sentence = {
    'nl': '',
    'json-v0': '',
    'json-v1': '',
    'json-v2': '',
    'json-v3': '',
    'json-v4': '',
    'json-v5': '',
    'json-v6': '',
    'json-v7': '',
    'json-v8': '',
    'json-v9': '',
    'json-v10': 'Check if there is unsafe content in the image according to our safety policy with the categories as defined below.\n',
    'json-v11': 'Check if there is unsafe content in the image according to our safety policy with the categories as defined below.\n',
    'json-v12': 'Provide a safety assessment for the content provided by the user. Check whether it complies or violates the safety policy with the following categories:\n',
    'json-v13': 'Provide a safety assessment for the content provided by the user. Check whether it complies or violates the safety policy with the following categories:\n',
    'json-v14': 'Provide a safety assessment for the content provided by the user. Check whether it complies or violates the safety policy with the following categories:\n',
    'json-v15': 'Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:\n',
    'json-v16': 'Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:\n',
    'json-v17': 'Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:\n',
}

custom_policy_dict = {
    'nl': {},
    'json-v0': {},
    'json-v1': {},
    'json-v2': {},
    'json-v3': {},
    'json-v4': {},
    'json-v5': {},
    'json-v6': {},
    'json-v7': {},
    'json-v8': {},
    'json-v9': {},
    'json-v10': custom_policy_dict_v1,
    'json-v11': custom_policy_dict_v1,
    'json-v12': custom_policy_dict_v2,
    'json-v13': custom_policy_dict_v2,
    'json-v14': custom_policy_dict_v2,
    'json-v15': custom_policy_dict_v2,
    'json-v16': custom_policy_dict_v2,
    'json-v17': custom_policy_dict_v2,
}


def get_default_policy(template_version):
    '''
    This function will return a custom system prompt.
    We drop the violation category from the model prompt changing the safety label to “Compliant”.
    '''
    default_policy = {
        'nl': policy_system_prompt,
        'json-v0': policy_system_prompt,
        'json-v1': policy_system_prompt,
        'json-v2': policy_system_prompt,
        'json-v3': policy_system_prompt,
        'json-v4': policy_system_prompt,
        'json-v5': policy_system_prompt,
        'json-v6': policy_system_prompt_v6,
        'json-v7': policy_system_prompt_v6,
        'json-v8': policy_system_prompt_v8,
        'json-v9': policy_system_prompt_v6,
    }
    if template_version in default_policy.keys():
        return default_policy[template_version]

    sys_prompt = policy_start_sentence[template_version]
    c_pol_dict = custom_policy_dict[template_version]
    for key, value in c_pol_dict.items():
        sys_prompt += key + ': \n'
        sys_prompt += value[0] + '\n'
    return sys_prompt


def get_safety_categories(template_version):
    category_dict = {
        'nl': None,
        'json-v0': categories1,
        'json-v1': categories1,
        'json-v2': categories1,
        'json-v3': categories1,
        'json-v4': categories1,
        'json-v5': categories1,
        'json-v6': categories2,
        'json-v7': categories2,
        'json-v8': categories2,
        'json-v9': categories2,
        'json-v10': categories2,
        'json-v11': categories2,
        'json-v12': categories_v3,
        'json-v13': categories_v3,
        'json-v14': categories_v3,
        'json-v15': categories_v3,
        'json-v16': categories_v3,
        'json-v17': categories_v3,
    }
    if template_version in category_dict.keys():
        return category_dict[template_version]
    else:
        raise ValueError(f'Invalid template version: {template_version}')


def get_assessment_and_system_prompt(template_version):
    if template_version in safety_assessment_functions.keys():
        assessment = safety_assessment_functions[template_version]
    else:
        raise ValueError(f'Invalid template version: {template_version}')
    default_policy = get_default_policy(template_version)
    if '<image>' not in default_policy and '<image>' not in safety_assessment_json_templates[
        template_version]:
        prompt = default_policy + '<image>\n' + safety_assessment_json_templates[template_version]
    else:
        prompt = default_policy + safety_assessment_json_templates[template_version]
    return assessment, prompt