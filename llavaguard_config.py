import os
from llavaguard.taxonomy.assessment import get_assessment_nl, get_assessment_json_v0, get_assessment_json, \
    get_assessment_json_v3, get_assessment_json_v4, get_assessment_json_v5, get_assessment_json_v6, \
    get_assessment_json_v7, get_assessment_json_v9, get_assessment_json_v12, get_assessment_json_v11, \
    get_assessment_json_v13, get_assessment_rating_only, json_template_v0, json_template_v1, json_template_v2, json_template_v3, json_template_v4, \
    json_template_v5, json_template_v6, json_template_v7, json_template_v8, json_template_v11, json_template_v12, \
    json_template_v13, json_template_v14, json_template_v15, json_template_v16, json_template_v17, categories1, \
    categories2, categories_v3, get_assessment_json_v20, json_template_v20, json_template_v20b, \
    get_assessment_json_v20b, json_template_rating_only
from llavaguard.taxonomy.taxonomies import custom_policy_dict_v1, custom_policy_dict_v2, policy_start_sentence_v10, \
    policy_start_sentence_v12, policy_start_sentence_v16, policy_system_prompt, policy_system_prompt_v6, \
    policy_system_prompt_v8

llava_guard_config = {
    'nl': {
        'response_keys': {'rating': 'final-assessment', 'category': 'category'},
        'safety_categories': None,
        'response_template': " If unsafe, the answer must include a comma-separated explanation with a list of violated categories.",
        'assessment_function': get_assessment_nl,
        'policy_start_sentence': policy_system_prompt,
        'policy_dict': {},
        'rationale_version': 'v1',
        'augmentation_strategy': None
    },
    'json-v0': {
        'response_keys': {'rating': 'final-assessment', 'category': 'policy-category'},
        'safety_categories': categories1,
        'response_template': json_template_v0,
        'assessment_function': get_assessment_json_v0,
        'policy_start_sentence': policy_system_prompt,
        'policy_dict': {},
        'rationale_version': 'v1',
        'augmentation_strategy': None
    },
    'json-v1': {
        'response_keys': {'rating': 'final-assessment', 'category': 'policy-category', 'rationale': 'explanation'},
        'safety_categories': categories1,
        'response_template': json_template_v1,
        'assessment_function': get_assessment_json,
        'policy_start_sentence': policy_system_prompt,
        'policy_dict': {},
        'rationale_version': 'v1',
        'augmentation_strategy': None
    },
    'json-v2': {
        'response_keys': {'rating': 'final-assessment', 'category': 'policy-category', 'rationale': 'explanation'},
        'safety_categories': categories1,
        'response_template': json_template_v2,
        'assessment_function': get_assessment_json,
        'policy_start_sentence': policy_system_prompt,
        'policy_dict': {},
        'rationale_version': 'v1',
        'augmentation_strategy': None
    },
    'json-v3': {
        'response_keys': {'rating': 'final-assessment', 'category': 'policy-category', 'rationale': 'explanation'},
        'safety_categories': categories1,
        'response_template': json_template_v3,
        'assessment_function': get_assessment_json_v3,
        'policy_start_sentence': policy_system_prompt,
        'policy_dict': {},
        'rationale_version': 'v1',
        'augmentation_strategy': None
    },
    'json-v4': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'decision', 'category': 'image-category', 'rationale': 'assessment'},
        'safety_categories': categories1,
        'response_template': json_template_v4,
        'assessment_function': get_assessment_json_v4,
        'policy_start_sentence': policy_system_prompt,
        'policy_dict': {},
        'rationale_version': 'v1',
        'augmentation_strategy': None
    },
    'json-v5': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'final-assessment'},
        'safety_categories': categories1,
        'response_template': json_template_v5,
        'assessment_function': get_assessment_json_v5,
        'policy_start_sentence': policy_system_prompt,
        'policy_dict': {},
        'rationale_version': 'v1',
        'augmentation_strategy': ['v1_a', 'v1_b', 'v1_c', 'v2']
    },
    'json-v6': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'assessment', 'category': 'category', 'rationale': 'explanation'},
        'safety_categories': categories1,
        'response_template': json_template_v6,
        'assessment_function': get_assessment_json_v6,
        'policy_start_sentence': policy_system_prompt_v6,
        'policy_dict': {},
        'rationale_version': 'v1',
        'augmentation_strategy': ['v1_a', 'v1_b', 'v1_c', 'v2']
    },
    'json-v7': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'decision', 'category': 'image-category', 'rationale': 'assessment'},
        'safety_categories': categories1,
        'response_template': json_template_v7,
        'assessment_function': get_assessment_json_v7,
        'policy_start_sentence': policy_system_prompt_v6,
        'policy_dict': {},
        'rationale_version': 'v1',
        'augmentation_strategy': ['v1_a', 'v1_b', 'v1_c', 'v2']
    },
    'json-v8': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'decision', 'category': 'image-category', 'rationale': 'assessment'},
        'safety_categories': categories1,
        'response_template': json_template_v8,
        'assessment_function': get_assessment_json_v7,
        'policy_start_sentence': policy_system_prompt_v8,
        'policy_dict': {},
        'rationale_version': 'v1',
        'augmentation_strategy': ['v1_a', 'v1_b', 'v1_c', 'v2']
    },
    'json-v9': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'decision', 'category': 'image-category', 'rationale': 'assessment'},
        'safety_categories': categories1,
        'response_template': json_template_v7,
        'assessment_function': get_assessment_json_v9,
        'policy_start_sentence': policy_system_prompt_v6,
        'policy_dict': {},
        'rationale_version': 'v1',
        'augmentation_strategy': ['v3_a', 'v3_b', 'v3_c', 'v2']
    },
    'json-v10': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'decision', 'category': 'image-category', 'rationale': 'assessment'},
        'safety_categories': categories2,
        'response_template': json_template_v8,
        'assessment_function': get_assessment_json_v9,
        'policy_start_sentence': policy_start_sentence_v10,
        'policy_dict': custom_policy_dict_v1,
        'rationale_version': 'v2',
        'augmentation_strategy': ['v3_a', 'v3_b', 'v3_c', 'v2']
    },
    'json-v11': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'decision', 'category': 'image-category', 'rationale': 'assessment'},
        'safety_categories': categories2,
        'response_template': json_template_v11,
        'assessment_function': get_assessment_json_v11,
        'policy_start_sentence': policy_start_sentence_v10,
        'policy_dict': custom_policy_dict_v1,
        'rationale_version': 'v2',
        'augmentation_strategy': ['v3_a', 'v3_b', 'v3_c', 'v2']
    },
    'json-v12': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
        'safety_categories': categories_v3,
        'response_template': json_template_v12,
        'assessment_function': get_assessment_json_v12,
        'policy_start_sentence': policy_start_sentence_v12,
        'policy_dict': custom_policy_dict_v2,
        'rationale_version': 'v2',
        'augmentation_strategy': ['v3_a', 'v3_b', 'v3_c', 'v2']
    },
    'json-v13': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
        'safety_categories': categories_v3,
        'response_template': json_template_v13,
        'assessment_function': get_assessment_json_v13,
        'policy_start_sentence': policy_start_sentence_v12,
        'policy_dict': custom_policy_dict_v2,
        'rationale_version': 'v2',
        'augmentation_strategy': ['v3_a', 'v3_b', 'v3_c', 'v2']
    },
    'json-v14': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
        'safety_categories': categories_v3,
        'response_template': json_template_v14,
        'assessment_function': get_assessment_json_v12,
        'policy_start_sentence': policy_start_sentence_v12,
        'policy_dict': custom_policy_dict_v2,
        'rationale_version': 'v2',
        'augmentation_strategy': ['v2', 'v3_a', 'v3_b', 'v3_c', 'v5']
    },
    'json-v15': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
        'safety_categories': categories_v3,
        'response_template': json_template_v15,
        'assessment_function': get_assessment_json_v13,
        'policy_start_sentence': policy_start_sentence_v16,
        'policy_dict': custom_policy_dict_v2,
        'rationale_version': 'v2',
        'augmentation_strategy': ['v2', 'v3_a', 'v3_b', 'v3_c', 'v5']
    },
    'json-v16': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
        'safety_categories': categories_v3,
        'response_template': json_template_v16,
        'assessment_function': get_assessment_json_v12,
        'policy_start_sentence': policy_start_sentence_v16,
        'policy_dict': custom_policy_dict_v2,
        'rationale_version': 'v2',
        'augmentation_strategy': ['v3_a', 'v3_b', 'v3_c', 'v2']
    },
    'json-v17': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'category': 'category', 'rationale': 'rationale', 'rating': 'rating'},
        'safety_categories': categories_v3,
        'response_template': json_template_v17,
        'assessment_function': get_assessment_json_v13,
        'policy_start_sentence': policy_start_sentence_v16,
        'policy_dict': custom_policy_dict_v2,
        'rationale_version': 'v2',
        'augmentation_strategy': ['v3_a', 'v3_b', 'v3_c', 'v2']
    },
    'json-v20': {
        'datasets': ['smid', 'real_images', 'synthetic'],
        'response_keys': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
        'safety_categories': categories_v3,
        'response_template': json_template_v20,
        'assessment_function': get_assessment_json_v20,
        'policy_start_sentence': policy_start_sentence_v16,
        'policy_dict': custom_policy_dict_v2,
        'rationale_version': 'v3',
        'augmentation_strategy': ['v3_a', 'v3_b', 'v3_c', 'v2']
    },
    'json-v21': {
        'datasets': ['smid', 'real_images', 'synthetic'],
        'response_keys': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
        'safety_categories': categories_v3,
        'response_template': json_template_v20,
        'assessment_function': get_assessment_json_v20,
        'policy_start_sentence': policy_start_sentence_v16,
        'policy_dict': custom_policy_dict_v2,
        'rationale_version': 'v3',
        'augmentation_strategy': ['v3_a', 'v3_b', 'v3_c', 'v2']
    },
    'json-v22': {
        'datasets': ['smid', 'real_images', 'synthetic'],
        'response_keys': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
        'safety_categories': categories_v3,
        'response_template': json_template_v20,
        'assessment_function': get_assessment_json_v20,
        'policy_start_sentence': policy_start_sentence_v16,
        'policy_dict': custom_policy_dict_v2,
        'rationale_version': 'v4',
        'augmentation_strategy': ['v3_a', 'v3_b', 'v2a']
    },
    'json-v22-b': {
        'datasets': ['smid', 'real_images', 'synthetic'],
        'response_keys': {'rationale': 'rationale', 'rating': 'rating', 'category': 'category'},
        'safety_categories': categories_v3,
        'response_template': json_template_v20b,
        'assessment_function': get_assessment_json_v20b,
        'policy_start_sentence': policy_start_sentence_v16,
        'policy_dict': custom_policy_dict_v2,
        'rationale_version': 'v4',
        'augmentation_strategy': ['v3_a', 'v3_b', 'v2a']
    },
    'json-v23': {
        'datasets': ['smid', 'real_images', 'synthetic'],
        'response_keys': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
        'safety_categories': categories_v3,
        'response_template': json_template_v20,
        'assessment_function': get_assessment_json_v20,
        'policy_start_sentence': policy_start_sentence_v16,
        'policy_dict': custom_policy_dict_v2,
        'rationale_version': 'v4',
        'augmentation_strategy': ['v3_a', 'v3_b', 'v3_c', 'v2']
    },
    'json-v24': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'rating', 'category': 'category', 'rationale': 'rationale'},
        'safety_categories': categories_v3,
        'response_template': json_template_v20,
        'assessment_function': get_assessment_json_v20,
        'policy_start_sentence': policy_start_sentence_v16,
        'policy_dict': custom_policy_dict_v2,
        'rationale_version': 'v4',
        'augmentation_strategy': ['v3_a', 'v3_b', 'v3_c', 'v2']
    },
    'json-v25': {
        'datasets': ['smid', 'real_images'],
        'response_keys': {'rating': 'rating'},
        'safety_categories': categories_v3,
        'response_template': json_template_rating_only,
        'assessment_function': get_assessment_rating_only,
        'policy_start_sentence': policy_start_sentence_v16,
        'policy_dict': custom_policy_dict_v2,
        'rationale_version': 'v4',
        'augmentation_strategy': ['v3_a', 'v3_b', 'v3_c', 'v2']
    },
}

local_data_dir = f'/common-repos/LlavaGuard'
if not os.path.exists(local_data_dir):
    local_data_dir = '/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/lhelff/ds/LlavaGuard'
if not os.path.exists(local_data_dir):
    raise ValueError(f'could not find local data dir. Please correct path {local_data_dir}')




generated_rationales_dirs = {
    'v1': {
        'smid': f'{local_data_dir}/data/rationales/v1/smid',
        'real_images': f'{local_data_dir}/data/rationales/v1/real_images',
    },
    'v2': f'{local_data_dir}/data/rationales/llava-v1.6-34b-json-v8/model_output',
    'v3': f'{local_data_dir}/data/rationales/llava-v1.6-34b-json-v16/model_output',
    'v4': f'{local_data_dir}/data/rationales/llava-v1.6-34b-json-v16-v4/model_output',
    'v5': f'{local_data_dir}/data/rationales/llava-v1.6-34b-json-v16-v5/model_output'
}

local_image_dirs = {
    'smid': f'{local_data_dir}/data/images/smid',
    'real_images': f'{local_data_dir}/data/images/real_images',
    'synthetic': f'{local_data_dir}/data/images/synthetic_data',
    "crawled": f'{local_data_dir}/data/images/crawled_data'
}

human_feedback_dirs = {
    'smid': f'{local_data_dir}/data/annotations/smid',
    'real_images': f'{local_data_dir}/data/annotations/real_images',
    'synthetic': f'{local_data_dir}/data/annotations/synthetic_data',
}


